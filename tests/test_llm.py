"""Tests for repofinder.llm (and the prompt text in repofinder.prompts).

Everything runs against a fake Anthropic client; no API key or network is
needed. One integration test at the bottom talks to the real API and is
skipped unless ANTHROPIC_API_KEY is set.
"""

from __future__ import annotations

import copy
import os
import shutil
import time
from types import SimpleNamespace
from typing import Any

import anthropic
import pytest

from repofinder import backends as backends_mod
from repofinder import llm as llm_mod
from repofinder import prompts
from repofinder.llm import (
    DEFAULT_FAST_MODEL,
    DEFAULT_SMART_MODEL,
    LLM,
    Answer,
    Citation,
    ContextChunk,
    BackendUnavailable,
    ClaudeCodeBackend,
    LLMConfigError,
    LLMError,
    Usage,
    parse_citations,
)
from repofinder.models import FileRecord, RepoIndex, Symbol

try:  # anthropic >= 1.0 is built on httpx2, older releases on httpx
    import httpx2 as _httpx
except ImportError:  # pragma: no cover
    import httpx as _httpx  # type: ignore[no-redef]


# `url_for` from Flask's src/flask/helpers.py, lines 200-251 (Flask is
# BSD-3-Clause licensed, Copyright 2010 Pallets). Used as a realistic chunk.
URL_FOR_PATH = "src/flask/helpers.py"
URL_FOR_START, URL_FOR_END = 200, 251
URL_FOR_SOURCE = r'''def url_for(
    endpoint: str,
    *,
    _anchor: str | None = None,
    _method: str | None = None,
    _scheme: str | None = None,
    _external: bool | None = None,
    **values: t.Any,
) -> str:
    """Generate a URL to the given endpoint with the given values.

    This requires an active request or application context, and calls
    :meth:`current_app.url_for() <flask.Flask.url_for>`. See that method
    for full documentation.

    :param endpoint: The endpoint name associated with the URL to
        generate. If this starts with a ``.``, the current blueprint
        name (if any) will be used.
    :param _anchor: If given, append this as ``#anchor`` to the URL.
    :param _method: If given, generate the URL associated with this
        method for the endpoint.
    :param _scheme: If given, the URL will have this scheme if it is
        external.
    :param _external: If given, prefer the URL to be internal (False) or
        require it to be external (True). External URLs include the
        scheme and domain. When not in an active request, URLs are
        external by default.
    :param values: Values to use for the variable parts of the URL rule.
        Unknown keys are appended as query string arguments, like
        ``?a=b&c=d``.

    .. versionchanged:: 2.2
        Calls ``current_app.url_for``, allowing an app to override the
        behavior.

    .. versionchanged:: 0.10
       The ``_scheme`` parameter was added.

    .. versionchanged:: 0.9
       The ``_anchor`` and ``_method`` parameters were added.

    .. versionchanged:: 0.9
       Calls ``app.handle_url_build_error`` on build errors.
    """
    return current_app.url_for(
        endpoint,
        _anchor=_anchor,
        _method=_method,
        _scheme=_scheme,
        _external=_external,
        **values,
    )
'''

REDIRECT_SOURCE = '''def redirect(
    location: str, code: int = 303, Response: type[BaseResponse] | None = None
) -> BaseResponse:
    """Create a redirect response object.
'''

REPO_SUMMARY = (
    "Flask is organised around the `Flask` application object in `src/flask/app.py`; "
    "helpers such as `url_for` in `src/flask/helpers.py` proxy to the current app."
)


def url_for_chunk() -> ContextChunk:
    return ContextChunk(URL_FOR_PATH, URL_FOR_START, URL_FOR_END, URL_FOR_SOURCE, label="url_for")


def redirect_chunk() -> ContextChunk:
    return ContextChunk(URL_FOR_PATH, 254, 257, REDIRECT_SOURCE, label="redirect")


# --------------------------------------------------------------------------- #
# Fake client
# --------------------------------------------------------------------------- #

def make_response(text: str = "ok", *, input_tokens: int = 100, output_tokens: int = 20,
                  cache_read: Any = 0, stop_reason: str = "end_turn",
                  model: str = "fake-model", stop_details: Any = None) -> SimpleNamespace:
    """Shaped like anthropic.types.Message: a (empty) thinking block first, then text."""
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="", signature="sig"),
            SimpleNamespace(type="text", text=text),
        ],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                              cache_read_input_tokens=cache_read,
                              cache_creation_input_tokens=0),
        stop_reason=stop_reason,
        stop_details=stop_details,
        model=model,
    )


class FakeMessages:
    def __init__(self, outcomes: list[Any]):
        self.outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(copy.deepcopy(kwargs))
        outcome = self.outcomes.pop(0) if self.outcomes else make_response()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, *outcomes: Any):
        self.messages = FakeMessages(list(outcomes))

    @property
    def requests(self) -> list[dict[str, Any]]:
        return self.messages.requests


class SleepRecorder:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def make_llm(*outcomes: Any, **kwargs: Any) -> tuple[LLM, FakeClient, SleepRecorder]:
    client = FakeClient(*outcomes)
    sleep = SleepRecorder()
    return LLM(client=client, sleep=sleep, **kwargs), client, sleep


_REQUEST = _httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def status_error(cls: type, status: int, headers: dict[str, str] | None = None,
                 message: str | None = None) -> Exception:
    response = _httpx.Response(status, request=_REQUEST, headers=headers or {})
    body = {"type": "error", "error": {"type": "x", "message": message or f"status {status}"}}
    return cls(message or f"status {status}", response=response, body=body)


def user_text(request: dict[str, Any]) -> str:
    """All text of the (single) user message of a recorded request."""
    (message,) = request["messages"]
    assert message["role"] == "user"
    return "\n".join(block["text"] for block in message["content"])


def sym(path: str, qualname: str, kind: str, start: int, end: int, signature: str,
        docstring: str | None = None) -> Symbol:
    parent = qualname.rsplit(".", 1)[0] if "." in qualname else None
    return Symbol(id=f"{path}::{qualname}", path=path, name=qualname.rsplit(".", 1)[-1],
                  qualname=qualname, kind=kind, start_line=start, end_line=end,  # type: ignore[arg-type]
                  signature=signature, docstring=docstring, parent=parent)


@pytest.fixture
def no_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """An environment with no backend at all: the SDK finds no credentials of
    any kind, there is no .env in the working directory and no `claude` CLI."""
    for name in list(os.environ):
        if name.startswith("ANTHROPIC_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("REPOFINDER_BACKEND", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(backends_mod.shutil, "which", lambda name, *a, **k: None)
    # Point the SDK's profile discovery at an empty directory.
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))


# --------------------------------------------------------------------------- #
# Client construction and configuration
# --------------------------------------------------------------------------- #

class _RecordingAnthropic:
    instances: list["_RecordingAnthropic"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.api_key = kwargs.get("api_key")
        self.auth_token = None
        self.credentials = None
        self.messages = FakeMessages([])
        _RecordingAnthropic.instances.append(self)


def test_client_is_built_lazily_on_first_call(monkeypatch: pytest.MonkeyPatch) -> None:
    _RecordingAnthropic.instances = []
    monkeypatch.setattr(backends_mod.anthropic, "Anthropic", _RecordingAnthropic)

    model = LLM(api_key="sk-test")
    assert _RecordingAnthropic.instances == []  # constructing LLM builds nothing

    model.summarize_file("a.py", "x = 1\n", [])
    assert len(_RecordingAnthropic.instances) == 1
    created = _RecordingAnthropic.instances[0]
    assert created.kwargs["api_key"] == "sk-test"
    assert created.kwargs["max_retries"] == 0  # retries are ours, not the SDK's

    model.summarize_file("b.py", "y = 2\n", [])
    assert len(_RecordingAnthropic.instances) == 1  # reused
    assert len(created.messages.requests) == 2


def test_missing_key_raises_llm_error_on_first_call_only(no_credentials: None) -> None:
    model = LLM()  # must not raise without a key
    with pytest.raises(BackendUnavailable, match="ANTHROPIC_API_KEY") as info:
        model.summarize_file("a.py", "x = 1\n", [])
    assert isinstance(info.value, LLMConfigError) and isinstance(info.value, LLMError)
    assert "claude auth login" in str(info.value)  # both setup options are explained
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        model.answer("where?", [url_for_chunk()], "flask")
    assert model.usage == Usage()


def test_missing_key_aborts_summarize_files(no_credentials: None) -> None:
    progress: list[tuple[int, int, str]] = []
    with pytest.raises(LLMConfigError):
        LLM().summarize_files([("a.py", "x = 1\n", []), ("b.py", "y = 2\n", [])],
                              on_progress=lambda *a: progress.append(a))
    assert progress == []


def test_forced_anthropic_backend_without_key_fails_on_first_call(
        no_credentials: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPOFINDER_BACKEND", "anthropic")
    model = LLM()
    assert model.backend_name == "anthropic"
    with pytest.raises(BackendUnavailable, match="No Anthropic API key found"):
        model.summarize_file("a.py", "x = 1\n", [])


def test_backend_precedence_in_llm(no_credentials: None, monkeypatch: pytest.MonkeyPatch) -> None:
    class StubBackend:
        name = "stub"

        def complete(self, **kwargs: Any) -> backends_mod.Completion:
            return backends_mod.Completion("Stub summary.", kwargs["model"], 1, 2, 3, "end_turn")

    stub = StubBackend()
    explicit = LLM(backend=stub, client=FakeClient(), api_key="sk-x")
    assert explicit.backend is stub
    assert explicit.summarize_file("a.py", "x = 1\n", []) == "Stub summary."
    assert explicit.usage == Usage(calls=1, input_tokens=1, output_tokens=2, cache_read_tokens=3)

    assert isinstance(LLM(client=FakeClient()).backend, backends_mod.AnthropicBackend)
    assert isinstance(LLM(api_key="sk-x").backend, backends_mod.AnthropicBackend)

    monkeypatch.setenv("REPOFINDER_BACKEND", "claude-code")
    monkeypatch.setattr(backends_mod.shutil, "which", lambda name, *a, **k: "/usr/bin/claude")
    detected = LLM()
    assert detected._backend is None  # nothing is detected at construction
    assert isinstance(detected.backend, ClaudeCodeBackend)
    assert detected.backend_name == "claude-code"


def test_backend_exceptions_are_wrapped() -> None:
    class BrokenBackend:
        name = "broken"

        def complete(self, **kwargs: Any) -> Any:
            raise KeyError("oops")

    with pytest.raises(backends_mod.BackendError, match="broken: KeyError"):
        LLM(backend=BrokenBackend()).summarize_file("a.py", "x = 1\n", [])
    result = LLM(backend=BrokenBackend()).summarize_files([("a.py", "x = 1\n", [])])
    assert result["a.py"].startswith("(summary failed: broken: KeyError")


def test_sdk_authentication_type_error_is_wrapped() -> None:
    err = TypeError('"Could not resolve authentication method. Expected one of api_key, ..."')
    model, _, _ = make_llm(err)
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        model.summarize_file("a.py", "x = 1\n", [])


def test_model_defaults_env_and_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REPOFINDER_MODEL_FAST", raising=False)
    monkeypatch.delenv("REPOFINDER_MODEL_SMART", raising=False)
    plain = LLM(client=FakeClient())
    assert (plain.fast_model, plain.smart_model) == (DEFAULT_FAST_MODEL, DEFAULT_SMART_MODEL)
    assert (DEFAULT_FAST_MODEL, DEFAULT_SMART_MODEL) == ("claude-sonnet-5-5", "claude-opus-5-5")

    monkeypatch.setenv("REPOFINDER_MODEL_FAST", "env-fast")
    monkeypatch.setenv("REPOFINDER_MODEL_SMART", "env-smart")
    from_env = LLM(client=FakeClient())
    assert (from_env.fast_model, from_env.smart_model) == ("env-fast", "env-smart")

    explicit = LLM(fast_model="arg-fast", smart_model="arg-smart", client=FakeClient())
    assert (explicit.fast_model, explicit.smart_model) == ("arg-fast", "arg-smart")


# --------------------------------------------------------------------------- #
# summarize_file / summarize_files
# --------------------------------------------------------------------------- #

def _big_source() -> str:
    lines = ["import os", "from .tax import apply_tax", "HEAD_SENTINEL = 1"]
    lines += [f"value_{i} = {i}  # MIDDLE_SENTINEL padding line" for i in range(2000)]
    lines += ["TAIL_SENTINEL = 2"]
    return "\n".join(lines) + "\n"


def test_summarize_file_truncates_source_and_lists_symbols() -> None:
    model, client, _ = make_llm(make_response("## Summary\nDefines Invoice.\nUsed by views."))
    source = _big_source()
    symbols = [
        sym("billing/invoices.py", "Invoice.total", "method", 20, 30, "def total(self) -> int"),
        sym("billing/invoices.py", "Invoice", "class", 10, 40, "class Invoice(Base)",
            docstring="An invoice for one customer."),
        sym("billing/invoices.py", "render_pdf", "function", 50, 60, "def render_pdf(inv: Invoice) -> bytes"),
        sym("billing/invoices.py", "render_pdf.<locals>.helper", "function", 52, 53, "def helper()"),
    ]

    summary = model.summarize_file("billing/invoices.py", source, symbols)

    assert summary == "Defines Invoice. Used by views."  # heading dropped, one paragraph
    (request,) = client.requests
    assert request["model"] == DEFAULT_FAST_MODEL
    assert request["output_config"] == {"effort": "low"}
    assert request["system"] == [{"type": "text", "text": prompts.FILE_SUMMARY_SYSTEM,
                                  "cache_control": {"type": "ephemeral"}}]
    text = user_text(request)
    assert '<file path="billing/invoices.py">' in text
    # head and tail kept, middle cut, marker present, size bounded
    assert "HEAD_SENTINEL" in text and "TAIL_SENTINEL" in text
    assert "lines (" in text and "omitted ...]" in text
    assert text.count("MIDDLE_SENTINEL") < 400
    source_part = text.split("<source>", 1)[1]
    assert len(source_part) < llm_mod.FILE_SOURCE_CHAR_LIMIT + 300
    # symbols in line order, methods indented under their class, docstring head shown
    assert "- class Invoice(Base)  [lines 10-40]  # An invoice for one customer." in text
    assert "\n  - def total(self) -> int  [lines 20-30]" in text
    assert "- def render_pdf(inv: Invoice) -> bytes  [lines 50-60]" in text
    assert text.index("class Invoice(Base)") < text.index("def total(self)") < text.index("def render_pdf")
    assert "def helper()" not in text  # nested <locals> functions are noise


def test_summarize_file_small_source_is_sent_whole() -> None:
    model, client, _ = make_llm(make_response("Defines x."))
    model.summarize_file("pkg/x.py", "x = 1\ny = 2\n", [])
    text = user_text(client.requests[0])
    assert "```python\nx = 1\ny = 2\n```" in text
    assert "omitted" not in text
    assert prompts.NO_SYMBOLS in text


def test_truncate_middle_cuts_on_line_boundaries() -> None:
    text = "".join(f"line {i:04d}\n" for i in range(1000))  # 10 chars per line
    out = llm_mod.truncate_middle(text, 1200)
    head, marker, tail = out.partition("[...")
    assert head.startswith("line 0000\n") and head.endswith("\n")
    assert tail.rstrip().endswith("line 0999")
    kept = head.count("\n") + tail.split("\n", 1)[1].count("\n")
    assert f"{1000 - kept} lines" in marker + tail
    assert len(out) <= 1200 + 80
    assert llm_mod.truncate_middle("short", 1200) == "short"


def test_summarize_files_reports_progress_and_continues_after_failure() -> None:
    model, client, sleep = make_llm(
        make_response("Summary A."),
        status_error(anthropic.BadRequestError, 400, message="prompt is too long"),
        make_response("Summary C."),
    )
    progress: list[tuple[int, int, str]] = []
    items = [("a.py", "a = 1\n", []), ("b.py", "b = 1\n", []), ("c.py", "c = 1\n", [])]

    result = model.summarize_files(items, on_progress=lambda *args: progress.append(args))

    assert progress == [(1, 3, "a.py"), (2, 3, "b.py"), (3, 3, "c.py")]
    assert result["a.py"] == "Summary A."
    assert result["c.py"] == "Summary C."
    assert result["b.py"].startswith("(summary failed: ") and result["b.py"].endswith(")")
    assert "prompt is too long" in result["b.py"]
    assert len(client.requests) == 3 and sleep.calls == []


def test_summarize_files_continues_after_retries_are_exhausted() -> None:
    limited = [status_error(anthropic.RateLimitError, 429) for _ in range(4)]
    model, client, sleep = make_llm(*limited, make_response("Summary B."))
    result = model.summarize_files([("a.py", "a = 1\n", []), ("b.py", "b = 1\n", [])])
    assert result["a.py"].startswith("(summary failed: Anthropic API error 429 after 4 attempts")
    assert result["b.py"] == "Summary B."
    assert sleep.calls == [0.5, 1.0, 2.0]
    assert len(client.requests) == 5


def test_summarize_files_aborts_on_rejected_credentials() -> None:
    model, client, _ = make_llm(status_error(anthropic.AuthenticationError, 401),
                                make_response("never used"))
    with pytest.raises(LLMConfigError, match="401"):
        model.summarize_files([("a.py", "a = 1\n", []), ("b.py", "b = 1\n", [])])
    assert len(client.requests) == 1


# --------------------------------------------------------------------------- #
# answer(): prompt structure
# --------------------------------------------------------------------------- #

def test_answer_prompt_structure() -> None:
    model, client, _ = make_llm(make_response(f"See `{URL_FOR_PATH}:244-251`."))
    question = "Where is url_for implemented?"

    model.answer(question, [url_for_chunk(), redirect_chunk()], "flask", repo_summary=REPO_SUMMARY)

    (request,) = client.requests
    assert request["model"] == DEFAULT_SMART_MODEL
    assert request["max_tokens"] == llm_mod.ANSWER_MAX_TOKENS
    assert request["output_config"] == {"effort": "medium"}
    (system,) = request["system"]
    assert system["cache_control"] == {"type": "ephemeral"}
    assert '"flask"' in system["text"]
    assert "only from the excerpts" in system["text"]
    assert "`path:start-end`" in system["text"]
    assert "same language as the question" in system["text"]
    assert "do not contain the answer" in system["text"]

    (message,) = request["messages"]
    overview, main = message["content"]
    assert overview["cache_control"] == {"type": "ephemeral"}
    assert REPO_SUMMARY in overview["text"]
    assert "cache_control" not in main

    text = main["text"]
    assert f"### {URL_FOR_PATH}:200-251 (url_for)\n```python\n200 | def url_for(\n201 |     endpoint: str," in text
    assert "244 |     return current_app.url_for(" in text
    assert "251 |     )\n```" in text
    assert f"### {URL_FOR_PATH}:254-257 (redirect)" in text
    assert "257 |     \"\"\"Create a redirect response object." in text
    # question comes last, after every excerpt
    assert text.rstrip().endswith(f"<question>\n{question}\n</question>")
    assert text.index("(redirect)") < text.index(question)


def test_answer_without_summary_or_label() -> None:
    model, client, _ = make_llm(make_response("Not in the excerpts."))
    chunk = ContextChunk("pkg/m.py", 9, 10, "a = 1\nb = 2\n")
    model.answer("Nerede?", [chunk], "demo")
    (message,) = client.requests[0]["messages"]
    (block,) = message["content"]
    assert "### pkg/m.py:9-10\n```python\n 9 | a = 1\n10 | b = 2\n```" in block["text"]
    assert "<repository_overview>" not in block["text"]


def test_answer_system_prompt_is_stable_across_calls() -> None:
    model, client, _ = make_llm(make_response("a"), make_response("b"))
    model.answer("Where is url_for?", [url_for_chunk()], "flask", REPO_SUMMARY)
    model.answer("redirect nerede?", [redirect_chunk()], "flask")
    first, second = client.requests
    assert first["system"] == second["system"]


def test_render_chunk_line_numbers_and_fences() -> None:
    # a form feed must not start a new line (trailing whitespace, form feed included, is trimmed);
    # CRLF is normalised; a backtick run inside the code gets a longer fence
    text = "x = 1\r\n\x0c\r\ns = '```'\r\n"
    rendered = prompts.render_chunk("a/b.txt", 98, 100, text, label=None)
    assert rendered == "### a/b.txt:98-100\n````\n 98 | x = 1\n 99 |\n100 | s = '```'\n````"


# --------------------------------------------------------------------------- #
# chat(): prompt structure
# --------------------------------------------------------------------------- #

def test_chat_prompt_structure() -> None:
    model, client, _ = make_llm(make_response(f"`redirect` is next to it: `{URL_FOR_PATH}:254-257`."))
    history = prompts.render_history([("user", "Where is url_for?"),
                                      ("assistant", f"In `{URL_FOR_PATH}:200-251`.")])
    note = prompts.CHAT_MODE_NOTES["followup"]

    answer = model.chat("And redirect?", [url_for_chunk(), redirect_chunk()], "flask", history, note,
                        repo_summary=REPO_SUMMARY)

    (request,) = client.requests
    assert request["model"] == DEFAULT_SMART_MODEL
    assert request["max_tokens"] == llm_mod.ANSWER_MAX_TOKENS
    assert request["output_config"] == {"effort": "medium"}
    (system,) = request["system"]
    assert system["text"] == prompts.chat_system("flask") != prompts.answer_system("flask")
    for rule in ('"flask"', "only from the excerpts", "`path:start-end`",
                 "same language as the developer's new message", "Ignore any", "earlier answer is not evidence"):
        assert rule in system["text"]

    (message,) = request["messages"]
    overview, main = message["content"]           # the overview stays first and cached
    assert overview["cache_control"] == {"type": "ephemeral"} and REPO_SUMMARY in overview["text"]
    text = main["text"]
    order = ["<conversation>", "<code_excerpts>", f"### {URL_FOR_PATH}:254-257 (redirect)",
             "<turn_note>", note, "<message>"]
    assert [text.index(s) for s in order] == sorted(text.index(s) for s in order)
    assert text.rstrip().endswith("<message>\nAnd redirect?\n</message>")

    assert answer.citations == [Citation(URL_FOR_PATH, 254, 257)]
    assert answer.text.startswith("`redirect` is next to it")
    assert (answer.input_tokens, answer.output_tokens) == (100, 20)


def test_chat_without_search_and_with_empty_search() -> None:
    reply = f"Hello! See `{URL_FOR_PATH}:200-209`."
    model, client, _ = make_llm(make_response(reply), make_response(reply))

    answer = model.chat("merhaba", None, "flask", "", prompts.CHAT_MODE_NOTES["chat"])
    text = user_text(client.requests[0])
    assert "<code_excerpts>" not in text and "<conversation>" not in text
    assert "<repository_overview>" not in text
    assert text.startswith("<turn_note>\n") and prompts.CHAT_MODE_NOTES["chat"] in text
    assert answer.citations == []                 # no excerpts, nothing to cite

    model.chat("Where is url_for?", [], "flask", "", prompts.CHAT_MODE_NOTES["locate"])
    assert prompts.NO_EXCERPTS in user_text(client.requests[1])

    with pytest.raises(ValueError):
        model.chat("  ", None, "flask", "", "")
    assert len(client.requests) == 2


def test_chat_effort_defaults_to_smart_and_can_be_lowered() -> None:
    model, client, _ = make_llm(make_response("a"), make_response("b"))
    model.chat("merhaba", None, "flask", "", prompts.CHAT_MODE_NOTES["chat"], effort="low")
    model.chat("Where is url_for?", [url_for_chunk()], "flask", "", prompts.CHAT_MODE_NOTES["locate"])
    assert [r["output_config"] for r in client.requests] == [{"effort": "low"}, {"effort": "medium"}]


# --------------------------------------------------------------------------- #
# Citations
# --------------------------------------------------------------------------- #

def test_citations_valid_clamped_unknown_and_duplicates() -> None:
    reply = (
        f"`url_for` is defined in `{URL_FOR_PATH}:200-209` and delegates to the app at "
        f"{URL_FOR_PATH}:244-251. Again: `{URL_FOR_PATH}:200-209`.\n"
        f"Whole function: {URL_FOR_PATH}:190-260.\n"              # clamped to 200-251
        f"Single line: {URL_FOR_PATH}:244.\n"                      # 244-244
        f"Unknown file: src/flask/app.py:1102-1110.\n"             # not in context: dropped
        f"Outside every chunk: {URL_FOR_PATH}:300-310.\n"          # no overlap: dropped
        f"Prefixed path: vendor/{URL_FOR_PATH}:200-201.\n"         # different path: dropped
        f"Variants: {URL_FOR_PATH}:L246–L248 and ./{URL_FOR_PATH}:251-249.\n"
    )
    model, _, _ = make_llm(make_response(reply))
    result = model.answer("Where is url_for?", [url_for_chunk(), redirect_chunk()], "flask")

    assert result.text == reply.strip()
    assert result.citations == [
        Citation(URL_FOR_PATH, 200, 209),
        Citation(URL_FOR_PATH, 244, 251),
        Citation(URL_FOR_PATH, 200, 251),
        Citation(URL_FOR_PATH, 244, 244),
        Citation(URL_FOR_PATH, 246, 248),
        Citation(URL_FOR_PATH, 249, 251),
    ]


def test_citation_spanning_two_chunks_is_clamped_to_the_larger_overlap() -> None:
    chunks = [url_for_chunk(), redirect_chunk()]
    assert parse_citations(f"{URL_FOR_PATH}:250-256", chunks) == [Citation(URL_FOR_PATH, 254, 256)]
    assert parse_citations(f"{URL_FOR_PATH}:240-255", chunks) == [Citation(URL_FOR_PATH, 240, 251)]
    assert parse_citations("nothing cited here", chunks) == []
    assert parse_citations(f"{URL_FOR_PATH}:200-201", []) == []


# --------------------------------------------------------------------------- #
# Retries and error wrapping
# --------------------------------------------------------------------------- #

def test_retry_on_rate_limit_then_success() -> None:
    model, client, sleep = make_llm(
        status_error(anthropic.RateLimitError, 429),
        status_error(anthropic.RateLimitError, 429),
        make_response("Defines x."),
    )
    assert model.summarize_file("x.py", "x = 1\n", []) == "Defines x."
    assert sleep.calls == [0.5, 1.0]
    assert len(client.requests) == 3
    assert client.requests[0] == client.requests[2]  # identical request re-sent
    assert model.usage.calls == 1


def test_retry_honours_retry_after_header() -> None:
    model, _, sleep = make_llm(
        status_error(anthropic.RateLimitError, 429, headers={"retry-after": "7"}),
        make_response("ok"),
    )
    model.summarize_file("x.py", "x = 1\n", [])
    assert sleep.calls == [7.0]


def test_retry_on_server_and_connection_errors() -> None:
    model, client, sleep = make_llm(
        status_error(anthropic.InternalServerError, 500),
        status_error(anthropic.APIStatusError, 529),  # overloaded
        anthropic.APIConnectionError(request=_REQUEST),
        make_response("ok"),
    )
    model.summarize_file("x.py", "x = 1\n", [])
    assert sleep.calls == [0.5, 1.0, 2.0]
    assert len(client.requests) == 4


def test_retries_exhausted_raise_llm_error() -> None:
    model, client, sleep = make_llm(*[status_error(anthropic.RateLimitError, 429) for _ in range(5)],
                                    max_retries=2)
    with pytest.raises(LLMError, match="429 after 3 attempts") as info:
        model.answer("q", [url_for_chunk()], "flask")
    assert isinstance(info.value.__cause__, anthropic.RateLimitError)
    assert sleep.calls == [0.5, 1.0]
    assert len(client.requests) == 3


@pytest.mark.parametrize("error", [
    status_error(anthropic.BadRequestError, 400, message="bad request"),
    status_error(anthropic.UnprocessableEntityError, 422),
    RuntimeError("boom"),
])
def test_non_retriable_error_is_wrapped(error: Exception) -> None:
    model, client, sleep = make_llm(error, make_response("never used"))
    with pytest.raises(LLMError) as info:
        model.summarize_file("x.py", "x = 1\n", [])
    assert not isinstance(info.value, LLMConfigError)
    assert info.value.__cause__ is error
    assert len(client.requests) == 1 and sleep.calls == []


def test_unknown_model_is_a_config_error() -> None:
    model, _, _ = make_llm(status_error(anthropic.NotFoundError, 404, message="model: nope"),
                           smart_model="nope")
    with pytest.raises(LLMConfigError, match="'nope'"):
        model.answer("q", [url_for_chunk()], "flask")


def test_refusal_and_empty_responses_raise() -> None:
    refusal = make_response("", stop_reason="refusal",
                            stop_details=SimpleNamespace(type="refusal", category="cyber"))
    model, _, _ = make_llm(refusal, make_response("   ", stop_reason="max_tokens"))
    with pytest.raises(LLMError, match="declined.*cyber"):
        model.summarize_file("x.py", "x = 1\n", [])
    with pytest.raises(LLMError, match="no text"):
        model.summarize_file("x.py", "x = 1\n", [])
    assert model.usage.calls == 2  # billed calls are still counted


def test_truncated_answer_is_marked() -> None:
    model, _, _ = make_llm(make_response(f"Partly in {URL_FOR_PATH}:244", stop_reason="max_tokens"))
    result = model.answer("q", [url_for_chunk()], "flask")
    assert result.text.endswith(prompts.TRUNCATED_ANSWER_NOTE.strip())
    assert result.citations == [Citation(URL_FOR_PATH, 244, 244)]


# --------------------------------------------------------------------------- #
# Usage accounting
# --------------------------------------------------------------------------- #

def test_usage_accumulates_and_answer_reports_per_call_numbers() -> None:
    model, _, _ = make_llm(
        make_response("Defines x.", input_tokens=100, output_tokens=20, cache_read=0),
        make_response(f"In {URL_FOR_PATH}:200-251.", input_tokens=50, output_tokens=10,
                      cache_read=30, model="claude-opus-5-5"),
        make_response("Defines y.", input_tokens=7, output_tokens=3, cache_read=None),
    )
    model.summarize_file("x.py", "x = 1\n", [])
    result = model.answer("Where is url_for?", [url_for_chunk()], "flask")
    model.summarize_file("y.py", "y = 1\n", [])

    assert isinstance(result, Answer)
    assert (result.model, result.input_tokens, result.output_tokens, result.cache_read_tokens) == (
        "claude-opus-5-5", 50, 10, 30)
    assert model.usage == Usage(calls=3, input_tokens=157, output_tokens=33, cache_read_tokens=30)


# --------------------------------------------------------------------------- #
# summarize_repo
# --------------------------------------------------------------------------- #

def _index(paths: list[str]) -> RepoIndex:
    return RepoIndex(name="demo", root="/tmp/demo", source="https://example.com/demo",
                     files=[FileRecord(p, "python", 10, 1, True) for p in paths])


def test_summarize_repo_orders_summaries_by_map_and_respects_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    map_markdown = "## Layers\n- core: `pkg/core.py`\n- web: pkg.web.views\n"
    summaries = {
        "pkg/zz_unmapped.py": "Unmapped helper.",
        "pkg/web/views.py": "HTTP views.",
        "pkg/core.py": "Core engine.",
        "pkg/broken.py": "(summary failed: Anthropic API error 500)",
    }
    model, client, _ = make_llm(make_response("The core engine sits in `pkg/core.py`."))
    out = model.summarize_repo(_index(["pkg/core.py", "pkg/web/views.py", "pkg/zz_unmapped.py"]),
                               summaries, map_markdown)

    assert out == "The core engine sits in `pkg/core.py`."
    (request,) = client.requests
    assert request["model"] == DEFAULT_SMART_MODEL
    assert request["system"][0]["text"] == prompts.REPO_SUMMARY_SYSTEM
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    text = user_text(request)
    assert '"demo" (3 indexed files, 0 classes and functions)' in text
    assert map_markdown.strip() in text
    assert "summary failed" not in text
    core, web, unmapped = (text.index(f"- {p}:") for p in
                           ("pkg/core.py", "pkg/web/views.py", "pkg/zz_unmapped.py"))
    assert core < web < unmapped

    # With a tight budget the files that appear later in the map are dropped first.
    monkeypatch.setattr(llm_mod, "SUMMARIES_CHAR_BUDGET", 60)
    model2, client2, _ = make_llm(make_response("ok"))
    model2.summarize_repo(_index(list(summaries)), summaries, map_markdown)
    text2 = user_text(client2.requests[0])
    assert "- pkg/core.py: Core engine." in text2
    assert "pkg/zz_unmapped.py" not in text2
    assert "(summaries of 1 further files omitted for length)" in text2


def test_summarize_repo_truncates_a_huge_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_mod, "MAP_CHAR_BUDGET", 100)
    huge_map = "".join(f"- module_{i}\n" for i in range(100))
    model, client, _ = make_llm(make_response("ok"))
    model.summarize_repo(_index([]), {}, huge_map)
    text = user_text(client.requests[0])
    assert "- module_0\n" in text and "- module_99\n" not in text
    assert "architecture map truncated" in text
    assert prompts.NO_SUMMARIES in text


# --------------------------------------------------------------------------- #
# Real API (run manually once a key is available)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY is not set")
def test_real_api_answer_cites_url_for() -> None:
    model = LLM()
    result = model.answer(
        "Where is the url_for helper implemented, and what does it delegate to?",
        [url_for_chunk()],
        repo_name="flask",
    )
    print(f"\n--- answer ({result.model}, in={result.input_tokens}, out={result.output_tokens}, "
          f"cache_read={result.cache_read_tokens}) ---\n{result.text}\n--- citations: {result.citations}")
    assert result.text.strip()
    assert any(c.path == URL_FOR_PATH for c in result.citations)
    assert all(URL_FOR_START <= c.start_line <= c.end_line <= URL_FOR_END for c in result.citations)
    assert model.usage.calls == 1 and model.usage.output_tokens > 0


@pytest.mark.skipif(
    shutil.which("claude") is None or bool(os.environ.get("REPOFINDER_SKIP_LIVE")),
    reason="the claude CLI is not on PATH (or REPOFINDER_SKIP_LIVE is set)",
)
def test_real_claude_code_answer_cites_url_for() -> None:
    model = LLM(backend=ClaudeCodeBackend(), smart_model="claude-sonnet-5-5")
    started = time.perf_counter()
    result = model.answer(
        "Where is the url_for helper implemented, and what does it delegate to?",
        [url_for_chunk()],
        repo_name="flask",
    )
    elapsed = time.perf_counter() - started
    print(f"\n--- answer via claude-code ({result.model}, {elapsed:.1f}s, in={result.input_tokens}, "
          f"out={result.output_tokens}, cache_read={result.cache_read_tokens}) ---\n{result.text}\n"
          f"--- citations: {result.citations}")
    assert result.text.strip()
    assert any(c.path == URL_FOR_PATH for c in result.citations)
    assert all(URL_FOR_START <= c.start_line <= c.end_line <= URL_FOR_END for c in result.citations)
    assert model.usage.calls == 1 and model.usage.output_tokens > 0

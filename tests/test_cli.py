"""Tests for repofinder.cli via typer's CliRunner. The LLM runs on a fake backend."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import pytest
from typer.testing import CliRunner

from repofinder import cli, pipeline, prompts
from repofinder.backends import BackendUnavailable, Completion
from repofinder.llm import LLM

SAMPLE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"
_FILE_PATH_RE = re.compile(r'<file path="([^"]+)">')

runner = CliRunner()


class FakeBackend:
    name = "fake"

    def __init__(self, text: str = "ok") -> None:
        self.text = text
        self.calls: list[dict[str, str]] = []

    def complete(self, *, model: str, system: str, user: str, max_tokens: int,
                 effort: Optional[str] = None) -> Completion:
        self.calls.append({"model": model, "system": system, "user": user})
        if system == prompts.FILE_SUMMARY_SYSTEM:
            text = f"Summary of {_FILE_PATH_RE.search(user).group(1)}."
        elif system == prompts.REPO_SUMMARY_SYSTEM:
            text = "Overview."
        else:
            text = self.text
        return Completion(text=text, model=model, input_tokens=1234, output_tokens=56,
                          cache_read_tokens=7, stop_reason="end_turn")


class NoBackendLLM(LLM):
    @property
    def backend_name(self) -> str:  # type: ignore[override]
        raise BackendUnavailable("RepoFinder needs access to Claude. Choose one of these options: ...")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv(pipeline.ENV_HOME, raising=False)
    return tmp_path / "home"


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    backend = FakeBackend("The engine runs in `run_engine` (`mypkg/core.py:13-30`).")
    monkeypatch.setattr(cli, "_make_llm", lambda: LLM(backend=backend))
    return backend


def _run(*args: str):
    return runner.invoke(cli.app, list(args))


def test_help_and_version():
    result = _run("--help")
    assert result.exit_code == 0
    for command in ("index", "map", "ask", "summarize", "status", "search"):
        assert command in result.stdout
    result = _run("--version")
    assert result.exit_code == 0 and result.stdout.startswith("repofinder ")


def test_index_then_reuse(home: Path):
    result = _run("index", str(SAMPLE_REPO), "--data-dir", str(home))
    assert result.exit_code == 0, result.output
    assert "Indexed sample_repo" in result.stdout
    assert "5 files" in result.stdout and "1 parse failure" in result.stdout
    assert (home / "repos" / "sample_repo" / "index.db").is_file()

    result = _run("index", str(SAMPLE_REPO), "--data-dir", str(home))
    assert result.exit_code == 0
    assert "Loaded the existing index of sample_repo" in result.stdout

    result = _run("index", str(SAMPLE_REPO), "--data-dir", str(home), "--refresh")
    assert result.exit_code == 0
    assert "Indexed sample_repo" in result.stdout


def test_index_bad_source(home: Path, tmp_path: Path):
    result = _run("index", str(tmp_path / "missing"), "--data-dir", str(home))
    assert result.exit_code == 1
    assert "neither a GitHub repository URL nor an existing directory" in result.stderr


def test_status(home: Path):
    result = _run("status", str(SAMPLE_REPO), "--data-dir", str(home))
    assert result.exit_code == 0, result.output
    out = result.stdout
    assert re.search(r"^Repository\s+sample_repo$", out, re.M)
    assert "5 files · 21 symbols" in out
    assert re.search(r"^Summaries\s+0 of 5 files$", out, re.M)
    assert re.search(r"^Overview\s+no ", out, re.M)
    assert re.search(r"^Backend\s+\S", out, re.M)
    assert str(home / "repos" / "sample_repo") in out


def test_search(home: Path):
    result = _run("search", str(SAMPLE_REPO), "where is run_engine", "-k", "3", "--data-dir", str(home))
    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert re.match(r"^1\.\s+\d+\.\d\d\s+symbol\s+mypkg/core\.py:13-30\s+run_engine$", lines[0])
    assert "name match: run_engine" in lines[1]
    assert sum(1 for line in lines if re.match(r"^\d\.", line)) == 3


def test_map_stdout_and_file(home: Path, tmp_path: Path):
    result = _run("map", str(SAMPLE_REPO), "--data-dir", str(home), "--max-modules", "3")
    assert result.exit_code == 0, result.output
    assert result.stdout.startswith("# sample_repo — architecture map")
    out_file = tmp_path / "maps" / "map.md"
    result = _run("map", str(SAMPLE_REPO), "--data-dir", str(home), "--out", str(out_file))
    assert result.exit_code == 0
    assert out_file.read_text(encoding="utf-8").startswith("# sample_repo — architecture map")
    assert result.stdout == ""


def test_ask_json(home: Path, fake: FakeBackend):
    result = _run("ask", str(SAMPLE_REPO), "How does the engine run?", "--json", "-k", "4",
                  "--data-dir", str(home))
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert {"answer", "citations", "hits", "model", "usage", "elapsed_s"} <= set(payload)
    assert payload["answer"].startswith("The engine runs in")
    assert payload["citations"] == [{"path": "mypkg/core.py", "start_line": 13, "end_line": 30}]
    assert 0 < len(payload["hits"]) <= 4
    assert {"path", "symbol_id", "start_line", "end_line", "score", "kind", "why"} <= set(payload["hits"][0])
    assert payload["usage"] == {"calls": 1, "input_tokens": 1234, "output_tokens": 56, "cache_read_tokens": 7}
    assert payload["backend"] == "fake"
    assert payload["model"] == LLM().smart_model
    assert payload["chunks"] and "text" not in payload["chunks"][0]
    assert len(fake.calls) == 1


def test_ask_text_output(home: Path, fake: FakeBackend):
    result = _run("ask", str(SAMPLE_REPO), "How does the engine run?", "--show-context",
                  "--data-dir", str(home))
    assert result.exit_code == 0, result.output
    out = result.stdout
    assert "Backend: fake" in out
    assert "Context:" in out and "mypkg/core.py:13-30" in out
    assert "The engine runs in" in out
    sources = out.split("Sources", 1)[1]
    assert "mypkg/core.py:13-30" in sources
    assert "1 call" in out and "1,234 input + 56 output tokens" in out


def test_ask_without_backend_exits_2(home: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli, "_make_llm", lambda: NoBackendLLM())
    result = _run("ask", str(SAMPLE_REPO), "anything", "--data-dir", str(home))
    assert result.exit_code == 2
    assert "RepoFinder needs access to Claude" in result.stderr


def test_summarize(home: Path, fake: FakeBackend):
    result = _run("summarize", str(SAMPLE_REPO), "--limit", "2", "--data-dir", str(home))
    assert result.exit_code == 0, result.output
    assert "Summarized 2 files (0 failed" in result.stdout
    assert "Repository overview written." in result.stdout
    assert "Backend: fake" in result.stdout
    result = _run("status", str(SAMPLE_REPO), "--data-dir", str(home))
    assert re.search(r"^Summaries\s+2 of 5 files$", result.stdout, re.M)
    assert re.search(r"^Overview\s+yes$", result.stdout, re.M)


def test_summarize_without_backend_exits_2(home: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli, "_make_llm", lambda: NoBackendLLM())
    result = _run("summarize", str(SAMPLE_REPO), "--data-dir", str(home))
    assert result.exit_code == 2


def test_source_ranges_drop_nested_citations():
    from repofinder.llm import Citation

    p = "src/pkg/mod.py"
    cits = [Citation(p, 10, 40), Citation(p, 12, 12), Citation(p, 30, 40), Citation(p, 50, 60),
            Citation("b.py", 1, 5), Citation("b.py", 3, 9), Citation(p, 10, 40)]
    assert cli.source_ranges(cits) == [(p, 10, 40), (p, 50, 60), ("b.py", 1, 5), ("b.py", 3, 9)]


def test_ask_long_markdown_lines_are_wrapped_not_cropped(home: Path, monkeypatch: pytest.MonkeyPatch):
    words = " ".join(f"word{i}" for i in range(60))
    backend = FakeBackend(f"- {words} `mypkg/core.py:13-30`\n- second item")
    monkeypatch.setattr(cli, "_make_llm", lambda: LLM(backend=backend))
    result = _run("ask", str(SAMPLE_REPO), "engine", "--data-dir", str(home))
    assert result.exit_code == 0, result.output
    flat = " ".join(result.stdout.split())
    assert words in flat and "mypkg/core.py:13-30" in flat


# --------------------------------------------------------------------------- #
# chat
# --------------------------------------------------------------------------- #

import importlib  # noqa: E402
import sys  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from repofinder.backends import BackendError  # noqa: E402


class _ChatDecisions:
    """Stands in for ``repofinder.decisions`` (the module and its backend) in the
    chat command tests: routes are scripted, rerank keeps the search order and
    the previous excerpts always suffice."""

    ROUTE_LOCATE, ROUTE_EXPLAIN, ROUTE_FOLLOWUP = "locate", "explain", "followup"
    ROUTE_CHAT, ROUTE_OUTSIDE = "chat", "outside"
    ROUTES = ("locate", "explain", "followup", "chat", "outside")

    class DecisionsError(RuntimeError):
        pass

    class DecisionsConfigError(DecisionsError):
        pass

    def __init__(self, *routes: str, name: str = "rules") -> None:
        self.name = name
        self.routes = list(routes)
        self.preferred: list[Optional[str]] = []

    def route_message(self, decisions, message, history):
        route = self.routes.pop(0) if self.routes else "locate"
        runner_up = "locate" if route == "chat" else "chat"
        return SimpleNamespace(route=route, probabilities={route: 0.75, runner_up: 0.25}, cost_usd=0.0)

    def rerank_cards(self, decisions, question, cards, allow_none=True):
        return SimpleNamespace(order=list(range(len(cards))), none_selected=False, probabilities={},
                               cost_usd=0.0)

    def context_suffices(self, decisions, message, previous_question, cards):
        return SimpleNamespace(probability=0.8, cost_usd=0.0, input_tokens=0)


def _decisions_module(monkeypatch: pytest.MonkeyPatch, stub: _ChatDecisions):
    """``repofinder.decisions``, or ``stub`` registered under that name while the
    real module does not exist yet."""
    try:
        return importlib.import_module("repofinder.decisions")
    except ModuleNotFoundError as exc:
        if exc.name != "repofinder.decisions":
            raise
        monkeypatch.setitem(sys.modules, "repofinder.decisions", stub)
        return stub


@pytest.fixture
def chat_stub(monkeypatch: pytest.MonkeyPatch):
    def install(*routes: str, name: str = "rules") -> _ChatDecisions:
        stub = _ChatDecisions(*routes, name=name)
        _decisions_module(monkeypatch, stub)
        chat_mod = importlib.import_module("repofinder.chat")
        monkeypatch.setattr(chat_mod, "_decisions", stub)

        def make_decisions(preferred: Optional[str]) -> _ChatDecisions:
            stub.preferred.append(preferred)
            return stub

        monkeypatch.setattr(cli, "_make_decisions", make_decisions)
        return stub

    return install


def _chat(home: Path, stdin: str, *args: str):
    return runner.invoke(cli.app, ["chat", str(SAMPLE_REPO), "--data-dir", str(home), *args], input=stdin)


def test_help_lists_chat():
    result = _run("--help")
    assert result.exit_code == 0 and "chat" in result.stdout
    result = _run("chat", "--help")
    assert result.exit_code == 0
    for option in ("--decisions", "--json", "--top-k", "--data-dir"):
        assert option in result.stdout


def test_chat_text_session(home: Path, fake: FakeBackend, chat_stub, monkeypatch: pytest.MonkeyPatch):
    stub = chat_stub("locate", "chat")
    monkeypatch.setattr(cli, "_stdin_is_tty", lambda: True)   # a person typing: prompts are shown
    stdin = "How does the engine run?\n\nteşekkürler!\n/context\n/route\n/nope\n/quit\nnever sent\n"
    result = _chat(home, stdin, "--decisions", "rules")
    assert result.exit_code == 0, result.output
    out = result.stdout

    assert stub.preferred == ["rules"]
    assert f"Backend: fake · model {LLM().smart_model} · decisions: rules" in out
    assert out.count("you> ") == 7                  # the empty line and the commands re-prompt
    assert len(fake.calls) == 2                     # two turns; nothing after /quit
    assert "The engine runs in" in out
    sources = out.split("Sources", 1)[1].split("route=locate", 1)[0]
    assert "mypkg/core.py:13-30" in sources
    assert out.count("Sources") == 1                # the chat turn has no excerpts to list
    (locate_line,) = [line for line in out.splitlines() if line.startswith("route=locate")]
    assert "tokens 1,234 in / 56 out" in locate_line and locate_line.endswith("decisions: rules (0 calls)")
    after_chat = out.split("route=chat ·", 1)[1]
    assert "Excerpts of the last search (1) for: How does the engine run?" in after_chat
    assert "mypkg/core.py:1-63" in after_chat       # /context
    assert "route=chat (decisions: rules)" in after_chat and "chat 0.75 · locate 0.25" in after_chat
    assert "Unknown command /nope" in result.stderr
    assert "2 calls" in out.splitlines()[-1]        # session usage on the way out


def test_chat_json_lines(home: Path, fake: FakeBackend, chat_stub):
    chat_stub("locate")
    result = _chat(home, "How does the engine run?\n/route\n/context\n", "--json", "-k", "4")
    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert len(lines) == 3
    turn, route, context = (json.loads(line) for line in lines)

    assert turn["route"] == "locate" and turn["route_probabilities"] == {"locate": 0.75, "chat": 0.25}
    assert turn["answer"].startswith("The engine runs in")
    assert turn["citations"] == [{"path": "mypkg/core.py", "start_line": 13, "end_line": 30}]
    assert turn["reused_context"] is False and turn["widened"] is False
    assert "context_probability" in turn and turn["context_probability"] is None   # no follow-up check
    assert turn["usage"] == {"input_tokens": 1234, "output_tokens": 56, "cache_read_tokens": 7}
    assert turn["decisions"] == {"name": "rules", "calls": 0, "cost_usd": 0.0, "errors": []}
    assert turn["chunks"] and set(turn["chunks"][0]) == {"path", "start_line", "end_line", "label"}

    assert route["command"] == "route" and route["route"] == "locate"
    assert context["command"] == "context" and context["question"] == "How does the engine run?"
    assert "you> " not in result.stdout and "you> " not in result.stderr   # piped: no prompt


def test_chat_json_followup_reports_context_probability(home: Path, fake: FakeBackend, chat_stub):
    chat_stub("locate", "followup")
    result = _chat(home, "How does the engine run?\nand how often does it retry?\n", "--json")
    assert result.exit_code == 0, result.output
    first, second = (json.loads(line) for line in result.stdout.splitlines())
    assert first["context_probability"] is None
    assert second["route"] == "followup" and second["reused_context"] is True
    assert second["context_probability"] == 0.8


@pytest.mark.parametrize("tty", [False, True])
def test_chat_json_stdout_is_one_json_object_per_line(home: Path, fake: FakeBackend, chat_stub,
                                                      monkeypatch: pytest.MonkeyPatch, tty: bool):
    chat_stub("locate", "chat")
    monkeypatch.setattr(cli, "_stdin_is_tty", lambda: tty)
    stdin = "How does the engine run?\n\n/help\n/nope\nthanks\n/context\n/route\n/quit\n"
    result = _chat(home, stdin, "--json")
    assert result.exit_code == 0, result.output

    lines = result.stdout.splitlines()
    payloads = [json.loads(line) for line in lines]          # every stdout line is one JSON object
    assert all(isinstance(p, dict) for p in payloads)
    assert [p.get("route") for p in payloads[:2]] == ["locate", "chat"]
    assert [p.get("command") for p in payloads[2:]] == ["context", "route"]
    assert "you> " not in result.stdout
    assert ("you> " in result.stderr) is tty                 # the prompt only for a person, on stderr
    assert "/context" in result.stderr                       # /help goes to stderr too
    assert "Unknown command /nope" in result.stderr


def test_chat_piped_text_mode_has_no_prompt(home: Path, fake: FakeBackend, chat_stub):
    chat_stub("chat")
    result = _chat(home, "hello\n")
    assert result.exit_code == 0, result.output
    assert "you> " not in result.stdout and "you> " not in result.stderr
    assert "route=chat" in result.stdout


def test_chat_ends_at_end_of_input(home: Path, fake: FakeBackend, chat_stub):
    chat_stub("chat")
    result = _chat(home, "hello")
    assert result.exit_code == 0, result.output
    assert len(fake.calls) == 1 and "route=chat" in result.stdout


def test_chat_failed_turn_does_not_end_the_session(home: Path, monkeypatch: pytest.MonkeyPatch, chat_stub):
    class FlakyBackend(FakeBackend):
        def complete(self, **kwargs):
            if not self.calls:
                self.calls.append(kwargs)
                raise BackendError("rate limited")
            return super().complete(**kwargs)

    backend = FlakyBackend("Second try.")
    monkeypatch.setattr(cli, "_make_llm", lambda: LLM(backend=backend))
    chat_stub("chat", "chat")
    result = _chat(home, "hi\nhi again\n")
    assert result.exit_code == 0, result.output
    assert "LLM call failed: rate limited" in result.stderr
    assert "Second try." in result.stdout


def test_chat_without_backend_exits_2(home: Path, monkeypatch: pytest.MonkeyPatch, chat_stub):
    chat_stub()
    monkeypatch.setattr(cli, "_make_llm", lambda: NoBackendLLM())
    result = _chat(home, "hello\n")
    assert result.exit_code == 2
    assert "RepoFinder needs access to Claude" in result.stderr


def test_chat_decisions_config_error_exits_2(home: Path, monkeypatch: pytest.MonkeyPatch,
                                             fake: FakeBackend, chat_stub):
    stub = chat_stub()
    errors = _decisions_module(monkeypatch, stub)

    def no_decisions(preferred: Optional[str]):
        raise errors.DecisionsConfigError(f"unknown decision layer: {preferred}")

    monkeypatch.setattr(cli, "_make_decisions", no_decisions)
    result = _chat(home, "hello\n", "--decisions", "bogus")
    assert result.exit_code == 2
    assert "unknown decision layer: bogus" in result.stderr
    assert fake.calls == []

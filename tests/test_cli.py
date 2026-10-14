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
from repofinder.decisions import ENV_DECISIONS, OPENROUTER_KEY_ENV, TYPESAFE_KEY_ENV
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


@pytest.fixture(autouse=True)
def no_jev_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No Jev key, no REPOFINDER_DECISIONS and no .env in the working directory:
    the commands detect the local rules unless a test sets a key or replaces
    ``cli._make_decisions``. Setting before deleting makes monkeypatch also
    remove what a .env load adds."""
    for name in (ENV_DECISIONS, OPENROUTER_KEY_ENV, TYPESAFE_KEY_ENV):
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


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

    def route_message(self, decisions, message, history, repo=None):
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


# --------------------------------------------------------------------------- #
# ask with a decision layer
# --------------------------------------------------------------------------- #

from repofinder.decisions import NONE_LABEL, Choice, DecisionsConfigError, FakeDecisions  # noqa: E402

RERANK_Q = "engine user model"   # search: User, User.Meta, models.py, Engine, ...


@pytest.fixture
def ask_decisions(monkeypatch: pytest.MonkeyPatch):
    """Replace ``cli._make_decisions`` with one returning ``decisions``;
    the ``--decisions`` values it was called with are collected."""
    def install(decisions) -> list[Optional[str]]:
        preferred: list[Optional[str]] = []

        def make(name: Optional[str]):
            preferred.append(name)
            return decisions

        monkeypatch.setattr(cli, "_make_decisions", make)
        return preferred

    return install


def test_ask_help_lists_the_decision_options():
    result = _run("ask", "--help")
    assert result.exit_code == 0
    for option in ("--decisions", "--candidates", "--top-k", "--json"):
        assert option in result.stdout


def test_ask_json_shows_the_decisions_block(home: Path, fake: FakeBackend, ask_decisions):
    probs = {"A": 0.15, "B": 0.0, "C": 0.05, "D": 0.8, NONE_LABEL: 0.0}
    decisions = FakeDecisions(choices=[Choice("D", probs, 0.8, 0.00031, 400)])
    preferred = ask_decisions(decisions)
    result = _run("ask", str(SAMPLE_REPO), RERANK_Q, "--json", "-k", "2", "--decisions", "jev",
                  "--candidates", "4", "--data-dir", str(home))
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert preferred == ["jev"]
    assert payload["decisions"] == {"name": "fake", "calls": 1, "cost_usd": pytest.approx(0.00031),
                                    "reranked": True, "off_target": False, "error": None}
    assert payload["candidates"] == 4
    assert len(payload["hits"]) == 4 and payload["hits"][0]["symbol_id"] == "mypkg/core.py::Engine"
    assert payload["chunks"][0]["path"] == "mypkg/core.py"
    assert payload["probabilities"]["mypkg/core.py:33-63"] == pytest.approx(0.8)
    assert len(payload["probabilities"]) == 4
    assert all(re.fullmatch(r".+:\d+-\d+", key) for key in payload["probabilities"])
    assert len(decisions.calls) == 1 and len(fake.calls) == 1


@pytest.mark.parametrize("pick, flag", [("D", "· reranked"), (NONE_LABEL, "· off-target")])
def test_ask_text_reports_the_decisions(home: Path, fake: FakeBackend, ask_decisions, pick: str, flag: str):
    ask_decisions(FakeDecisions(choices=[pick]))
    result = _run("ask", str(SAMPLE_REPO), RERANK_Q, "-k", "2", "--candidates", "4", "--show-context",
                  "--data-dir", str(home))
    assert result.exit_code == 0, result.output
    out = result.stdout
    assert f"Backend: fake · model {LLM().smart_model} · decisions: fake" in out
    assert "Context:" in out
    last = out.strip().splitlines()[-1]
    assert last.endswith(f"· decisions: fake (1 call, $0.000000) {flag}")
    assert "1 call · 1,234 input + 56 output tokens" in last


def test_ask_text_warns_when_the_layer_fails(home: Path, fake: FakeBackend, ask_decisions):
    ask_decisions(FakeDecisions())          # no scripted answer: the search order is kept
    result = _run("ask", str(SAMPLE_REPO), RERANK_Q, "--data-dir", str(home))
    assert result.exit_code == 0, result.output
    assert "Decision layer failed, the search order was kept: DecisionsError" in result.stderr
    last = result.stdout.strip().splitlines()[-1]
    assert last.endswith("· decisions: fake (0 calls, $0.000000)")
    assert len(fake.calls) == 1


def test_ask_decisions_config_errors_exit_2(home: Path, fake: FakeBackend, monkeypatch: pytest.MonkeyPatch,
                                            ask_decisions):
    def no_decisions(preferred: Optional[str]):
        raise DecisionsConfigError(f"unknown decision layer: {preferred}")

    monkeypatch.setattr(cli, "_make_decisions", no_decisions)
    result = _run("ask", str(SAMPLE_REPO), "engine", "--decisions", "bogus", "--data-dir", str(home))
    assert result.exit_code == 2
    assert "unknown decision layer: bogus" in result.stderr

    class Rejected(FakeDecisions):
        name = "jev"

        def choose(self, state, instructions, options):
            raise DecisionsConfigError("Jev rejected the API key (401)")

    ask_decisions(Rejected())
    result = _run("ask", str(SAMPLE_REPO), "engine", "--json", "--data-dir", str(home))
    assert result.exit_code == 2
    assert "Jev rejected the API key (401)" in result.stderr
    assert fake.calls == []


def test_ask_without_decisions_behaves_as_before(home: Path, fake: FakeBackend, tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_DECISIONS, raising=False)
    monkeypatch.chdir(tmp_path)              # no .env file to pick a layer from
    calls: list[dict] = []
    real = pipeline.ask

    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append(kwargs)
        return result

    monkeypatch.setattr(pipeline, "ask", spy)
    result = _run("ask", str(SAMPLE_REPO), RERANK_Q, "--json", "-k", "3", "--data-dir", str(home))
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["decisions"] == {"name": "rules", "calls": 0, "cost_usd": 0.0, "reranked": False,
                                    "off_target": False, "error": None}
    assert payload["candidates"] == pipeline.DEFAULT_CANDIDATES and payload["probabilities"] == {}
    with pipeline.open_workspace(str(SAMPLE_REPO), data_dir=home) as ws:
        chunks, hits = pipeline.retrieve(ws, RERANK_Q, k=3)
    assert [h["symbol_id"] for h in payload["hits"]] == [h.symbol_id for h in hits]
    assert [(c["path"], c["start_line"], c["end_line"]) for c in payload["chunks"]] == \
        [(c.path, c.start_line, c.end_line) for c in chunks]
    assert calls[0]["decisions"].name == "rules"

    result = _run("ask", str(SAMPLE_REPO), RERANK_Q, "--data-dir", str(home))
    assert result.exit_code == 0, result.output
    assert f"Backend: fake · model {LLM().smart_model} · decisions: rules" in result.stdout
    last = result.stdout.strip().splitlines()[-1]
    assert "decisions" not in last and "reranked" not in last and "off-target" not in last
    assert last.endswith("7 cache read")


# --------------------------------------------------------------------------- #
# search and status with a decision layer
# --------------------------------------------------------------------------- #

_RANK_LINE = re.compile(r"^\s*\d+\.\s+-?\d+\.\d\d\s+\w+\s+(\S+:\d+-\d+)\s+(\S+)(?:\s+p=(\d\.\d\d))?$")


def _ranked(stdout: str) -> list[tuple[str, str, Optional[str]]]:
    """``(path:start-end, name, p)`` of each ranked hit line printed by ``search``."""
    return [m.groups() for m in map(_RANK_LINE.match, stdout.splitlines()) if m]


def _index_order(home: Path, query: str, k: int) -> list[str]:
    """``path:start-end`` of the raw index search hits, in search order."""
    with pipeline.open_workspace(str(SAMPLE_REPO), data_dir=home) as ws:
        return [f"{h.path}:{h.start_line}-{h.end_line}" for h in ws.search_index().search(query, k=k)]


class _FakeJev(FakeDecisions):
    """Scripted answers under Jev's name."""
    name = "jev"


def test_search_help_lists_the_decision_options():
    result = _run("search", "--help")
    assert result.exit_code == 0
    for option in ("--decisions", "--candidates", "--top-k", "--data-dir"):
        assert option in result.stdout


def test_search_rules_keeps_the_index_order(home: Path):
    result = _run("search", str(SAMPLE_REPO), RERANK_Q, "-k", "5", "--decisions", "rules", "--data-dir", str(home))
    assert result.exit_code == 0, result.output
    ranked = _ranked(result.stdout)
    assert [loc for loc, _name, _p in ranked] == _index_order(home, RERANK_Q, 5)
    assert all(p is None for _loc, _name, p in ranked)
    assert len(result.stdout.splitlines()) == 2 * 5               # two lines per hit, no decisions line
    default = _run("search", str(SAMPLE_REPO), RERANK_Q, "-k", "5", "--data-dir", str(home))   # no key: rules
    assert default.exit_code == 0, default.output
    assert default.stdout == result.stdout


def test_search_prints_the_reranked_order_and_the_decisions_line(home: Path, ask_decisions):
    probs = {"A": 0.15, "B": 0.0, "C": 0.05, "D": 0.8, NONE_LABEL: 0.0}
    decisions = _FakeJev(choices=[Choice("D", probs, 0.8, 0.00031, 400)])
    preferred = ask_decisions(decisions)
    result = _run("search", str(SAMPLE_REPO), RERANK_Q, "-k", "3", "--candidates", "4", "--decisions", "jev",
                  "--data-dir", str(home))
    assert result.exit_code == 0, result.output
    assert preferred == ["jev"]
    assert len(decisions.calls) == 1 and len(decisions.calls[0].options) == 5   # four cards and NONE
    searched = _index_order(home, RERANK_Q, 4)
    ranked = _ranked(result.stdout)
    assert [loc for loc, _name, _p in ranked] == [searched[i] for i in (3, 0, 2)]  # the pick, then by probability
    assert ranked[0][1] == "Engine"
    assert [p for _loc, _name, p in ranked] == ["0.80", "0.15", "0.05"]
    assert result.stdout.splitlines()[-1] == "decisions: jev (1 call, $0.000310) · reranked"
    assert result.stderr.count("Decision layer failed") == 0


@pytest.mark.parametrize("script, note", [
    ([NONE_LABEL], "decisions: jev (1 call, $0.000000) · off-target"),
    ([], "decisions: jev (0 calls, $0.000000)"),                   # nothing scripted: the call fails
])
def test_search_keeps_the_index_order_when_jev_does_not_rerank(home: Path, ask_decisions, script: list[str],
                                                               note: str):
    ask_decisions(_FakeJev(choices=script))
    result = _run("search", str(SAMPLE_REPO), RERANK_Q, "-k", "3", "--candidates", "4", "--data-dir", str(home))
    assert result.exit_code == 0, result.output
    assert [loc for loc, _name, _p in _ranked(result.stdout)] == _index_order(home, RERANK_Q, 4)[:3]
    assert result.stdout.splitlines()[-1] == note
    assert ("Decision layer failed, the search order was kept: DecisionsError" in result.stderr) is (not script)


def test_search_decisions_config_errors_exit_2(home: Path, ask_decisions):
    class Rejected(_FakeJev):
        def choose(self, state, instructions, options):
            raise DecisionsConfigError("Jev rejected the API key (401)")

    ask_decisions(Rejected())
    result = _run("search", str(SAMPLE_REPO), RERANK_Q, "--data-dir", str(home))
    assert result.exit_code == 2
    assert "Jev rejected the API key (401)" in result.stderr and result.stdout == ""


@pytest.mark.parametrize("env, expected", [
    ({}, "rules (no OPENROUTER_API_KEY / TYPESAFE_API_KEY; set one in .env to enable Jev)"),
    ({OPENROUTER_KEY_ENV: "sk-or-v1-test"}, "jev (OPENROUTER_API_KEY set)"),
    ({TYPESAFE_KEY_ENV: "ts-test"}, "jev (TYPESAFE_API_KEY set)"),
    ({OPENROUTER_KEY_ENV: "sk-or-v1-test", TYPESAFE_KEY_ENV: "ts-test"}, "jev (OPENROUTER_API_KEY set)"),
    ({OPENROUTER_KEY_ENV: "sk-or-v1-test", ENV_DECISIONS: "rules"}, "rules (REPOFINDER_DECISIONS=rules)"),
    ({ENV_DECISIONS: "bogus"}, "none (Unknown decisions backend 'bogus': use 'auto', 'jev', 'rules' or 'off'.)"),
])
def test_status_reports_the_decision_layer(home: Path, monkeypatch: pytest.MonkeyPatch, env: dict[str, str],
                                           expected: str):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    result = _run("status", str(SAMPLE_REPO), "--data-dir", str(home))
    assert result.exit_code == 0, result.output
    assert re.search(rf"^Decisions\s+{re.escape(expected)}$", result.stdout, re.M), result.stdout


def test_status_reads_the_key_from_a_dotenv_file(home: Path, tmp_path: Path):
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-or-v1-dotenv\n", encoding="utf-8")   # cwd is tmp_path
    result = _run("status", str(SAMPLE_REPO), "--data-dir", str(home))
    assert result.exit_code == 0, result.output
    assert re.search(r"^Decisions\s+jev \(OPENROUTER_API_KEY set\)$", result.stdout, re.M), result.stdout


def test_status_with_a_key_but_a_broken_sdk(home: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(OPENROUTER_KEY_ENV, "sk-or-v1-test")
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)          # makes `import typesafe_sdk` fail
    result = _run("status", str(SAMPLE_REPO), "--data-dir", str(home))
    assert result.exit_code == 0, result.output
    assert re.search(r"^Decisions\s+rules \(OPENROUTER_API_KEY is set, but typesafe-sdk could not be imported;"
                     r" reinstall RepoFinder\)$", result.stdout, re.M), result.stdout

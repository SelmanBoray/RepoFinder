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

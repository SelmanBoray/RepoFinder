"""Tests for repofinder.pipeline: workspaces, retrieval, ask and summarize.

No network and no real model: the LLM runs on a fake backend.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Optional

import pytest

from repofinder import pipeline, prompts
from repofinder.backends import BackendError, Completion, LLMConfigError
from repofinder.chunker import split_lines
from repofinder.llm import FAILED_SUMMARY_PREFIX, LLM, REPO_SUMMARY_META_KEY
from repofinder.pipeline import (
    AskResult,
    SummarizeResult,
    Workspace,
    ask,
    open_workspace,
    retrieve,
    summarize,
    summary_order,
)

SAMPLE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"
_FILE_PATH_RE = re.compile(r'<file path="([^"]+)">')


class FakeBackend:
    """Records every call; ``handler(model, system, user)`` returns the text or raises."""

    name = "fake"

    def __init__(self, handler: Optional[Callable[[str, str, str], str]] = None) -> None:
        self.calls: list[dict[str, str]] = []
        self.handler = handler

    def complete(self, *, model: str, system: str, user: str, max_tokens: int,
                 effort: Optional[str] = None) -> Completion:
        self.calls.append({"model": model, "system": system, "user": user})
        text = self.handler(model, system, user) if self.handler else "ok"
        return Completion(text=text, model=model, input_tokens=100, output_tokens=20,
                          cache_read_tokens=0, stop_reason="end_turn")

    def file_calls(self) -> list[str]:
        return [_FILE_PATH_RE.search(c["user"]).group(1) for c in self.calls
                if c["system"] == prompts.FILE_SUMMARY_SYSTEM]

    def repo_calls(self) -> int:
        return sum(1 for c in self.calls if c["system"] == prompts.REPO_SUMMARY_SYSTEM)


def summary_handler(model: str, system: str, user: str) -> str:
    if system == prompts.FILE_SUMMARY_SYSTEM:
        return f"Summary of {_FILE_PATH_RE.search(user).group(1)}."
    if system == prompts.REPO_SUMMARY_SYSTEM:
        return "Overview of the repository."
    return "ok"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv(pipeline.ENV_HOME, raising=False)
    return tmp_path / "home"


@pytest.fixture
def ws(home: Path):
    with open_workspace(str(SAMPLE_REPO), data_dir=home) as workspace:
        yield workspace


def _tree(root: Path) -> dict[str, int]:
    return {p.relative_to(root).as_posix(): p.stat().st_size for p in root.rglob("*") if p.is_file()}


# --------------------------------------------------------------------------- #
# open_workspace
# --------------------------------------------------------------------------- #

def test_open_workspace_creates_store_under_data_dir(home: Path):
    before = _tree(SAMPLE_REPO)
    with open_workspace(str(SAMPLE_REPO), data_dir=home) as ws:
        assert isinstance(ws, Workspace)
        assert ws.name == "sample_repo"
        assert ws.data_dir == home / "repos" / "sample_repo"
        assert (ws.data_dir / pipeline.DB_FILENAME).is_file()
        assert ws.root == SAMPLE_REPO.resolve()           # local path: used in place
        assert ws.source == str(SAMPLE_REPO.resolve())
        assert ws.built is True
        assert ws.store.has_index()
        assert ws.store.get_meta(pipeline.META_BUILT_AT)
        assert {f.path for f in ws.index.files} >= {"mypkg/core.py", "mypkg/sub/models.py"}
        assert not (ws.data_dir / pipeline.CHECKOUT_DIRNAME).exists()
    assert _tree(SAMPLE_REPO) == before                   # nothing written into the checkout


def test_second_open_reuses_the_stored_index(home: Path, monkeypatch: pytest.MonkeyPatch):
    with open_workspace(str(SAMPLE_REPO), data_dir=home) as first:
        files, symbols, imports = first.index.files, first.index.symbols, first.index.imports
        built_at = first.store.get_meta(pipeline.META_BUILT_AT)

    def no_build(*args, **kwargs):
        raise AssertionError("the index must not be rebuilt")

    monkeypatch.setattr(pipeline, "build_index", no_build)
    with open_workspace(str(SAMPLE_REPO), data_dir=home) as second:
        assert second.built is False
        assert (second.index.files, second.index.symbols, second.index.imports) == (files, symbols, imports)
        assert second.index.root == str(SAMPLE_REPO.resolve())
        assert second.store.get_meta(pipeline.META_BUILT_AT) == built_at


def test_refresh_rebuilds(home: Path, monkeypatch: pytest.MonkeyPatch):
    open_workspace(str(SAMPLE_REPO), data_dir=home).close()
    calls = []
    real = pipeline.build_index
    monkeypatch.setattr(pipeline, "build_index", lambda *a, **k: calls.append(a) or real(*a, **k))
    with open_workspace(str(SAMPLE_REPO), data_dir=home, refresh=True) as ws:
        assert ws.built is True
    assert len(calls) == 1


def test_refresh_keeps_summaries_of_the_same_source(home: Path):
    with open_workspace(str(SAMPLE_REPO), data_dir=home) as ws:
        ws.store.save_summary("mypkg/core.py", "Core engine.")
    # A different spelling of the same directory is the same source.
    relative = os.path.relpath(SAMPLE_REPO, Path.cwd()) if SAMPLE_REPO.drive == Path.cwd().drive else str(SAMPLE_REPO)
    with open_workspace(relative, data_dir=home, refresh=True) as ws:
        assert ws.summaries() == {"mypkg/core.py": "Core engine."}


def test_data_dir_from_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(pipeline.ENV_HOME, str(tmp_path / "envhome"))
    with open_workspace(str(SAMPLE_REPO)) as ws:
        assert ws.data_dir == tmp_path / "envhome" / "repos" / "sample_repo"
    assert pipeline.resolve_data_dir(tmp_path / "explicit") == tmp_path / "explicit"


def test_other_source_with_the_same_name_replaces_the_index(home: Path, tmp_path: Path):
    open_workspace(str(SAMPLE_REPO), data_dir=home).close()
    twin = tmp_path / "twin" / "sample_repo"
    twin.mkdir(parents=True)
    (twin / "only.py").write_text("def lonely():\n    return 1\n", encoding="utf-8")
    with open_workspace(str(twin), data_dir=home) as ws:
        assert ws.built is True
        assert [f.path for f in ws.index.files] == ["only.py"]
        assert ws.source == str(twin.resolve())


def test_github_source_clones_into_the_checkout_dir(home: Path, monkeypatch: pytest.MonkeyPatch):
    seen = []

    def fake_fetch(source: str, dest_dir: Path) -> Path:
        seen.append((source, Path(dest_dir)))
        return SAMPLE_REPO.resolve()

    monkeypatch.setattr(pipeline, "fetch_repo", fake_fetch)
    with open_workspace("https://github.com/Pallets/Flask.git", data_dir=home) as ws:
        assert ws.name == "Flask"
        assert ws.source == "https://github.com/pallets/flask"
        assert ws.index.source == "https://github.com/pallets/flask"
    assert seen == [("https://github.com/Pallets/Flask.git", home / "repos" / "Flask" / "checkout")]
    # Another spelling of the same repository reuses the index.
    with open_workspace("github.com/pallets/flask", data_dir=home) as ws:
        assert ws.built is False


# --------------------------------------------------------------------------- #
# retrieve / ask / map
# --------------------------------------------------------------------------- #

def test_retrieve_is_deterministic_and_exact(ws: Workspace):
    chunks, hits = retrieve(ws, "where is run engine implemented", k=5)
    again = retrieve(ws, "where is run engine implemented", k=5)
    assert (chunks, hits) == again
    assert hits and len(hits) <= 5
    assert hits[0].symbol_id == "mypkg/core.py::run_engine"
    assert chunks
    for c in chunks:
        lines = split_lines((ws.root / c.path).read_text(encoding="utf-8"))
        assert c.text == "\n".join(lines[c.start_line - 1 : c.end_line])
    assert sum(len(c.text) for c in chunks) <= pipeline.DEFAULT_MAX_CHARS


def test_retrieve_respects_max_chars(ws: Workspace):
    chunks, _ = retrieve(ws, "engine user model", k=10, max_chars=300)
    assert chunks and sum(len(c.text) for c in chunks) <= 300


def test_ask_with_fake_llm(ws: Workspace):
    def handler(model, system, user):
        assert "How does the engine run?" in user
        assert "mypkg/core.py" in user
        return "It is `run_engine` in `mypkg/core.py:13-30`, see also `mypkg/nowhere.py:1-2`."

    backend = FakeBackend(handler)
    llm = LLM(backend=backend)
    result = ask(ws, "How does the engine run?", llm=llm, k=4)
    assert isinstance(result, AskResult)
    assert result.answer.text.startswith("It is `run_engine`")
    assert [(c.path, c.start_line, c.end_line) for c in result.answer.citations] == [("mypkg/core.py", 13, 30)]
    assert result.hits and result.chunks
    assert result.elapsed_s >= 0
    assert llm.usage.calls == 1
    assert backend.calls[0]["model"] == llm.smart_model
    assert prompts.OVERVIEW_START not in backend.calls[0]["user"]  # no overview stored yet


def test_ask_passes_the_repo_summary(ws: Workspace):
    ws.store.set_meta(REPO_SUMMARY_META_KEY, "A tiny sample package.")
    backend = FakeBackend(lambda m, s, u: "Answer.")
    ask(ws, "engine", llm=LLM(backend=backend))
    assert "A tiny sample package." in backend.calls[0]["user"]


def test_ask_rejects_an_empty_question(ws: Workspace):
    with pytest.raises(ValueError):
        ask(ws, "  ", llm=LLM(backend=FakeBackend()))


def test_architecture_map_uses_summaries(ws: Workspace):
    ws.store.save_summary("mypkg/core.py", "Runs the engine. More text.")
    ws.store.save_summary("mypkg/sub/models.py", f"{FAILED_SUMMARY_PREFIX}boom)")
    text = pipeline.architecture_map(ws, max_modules=10)
    assert text.startswith("# sample_repo")
    assert "Runs the engine." in text
    assert "boom" not in text


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #

@pytest.fixture
def layered(make_repo, home: Path):
    root = make_repo({
        "pkg/__init__.py": "from .base import Base\n",
        "pkg/base.py": "class Base:\n    pass\n",
        "pkg/a.py": "from .base import Base\nfrom .util import helper\n\nclass A(Base):\n    pass\n",
        "pkg/b.py": "from .base import Base\nfrom .util import helper\n\ndef b():\n    return helper()\n",
        "pkg/util.py": "def helper():\n    return 1\n",
        "pkg/empty.py": "",
        "tests/test_a.py": "from pkg.a import A\n\ndef test_a():\n    assert A\n",
    })
    with open_workspace(str(root), data_dir=home) as workspace:
        yield workspace


def test_summary_order_puts_core_modules_first_and_tests_last(layered: Workspace):
    order = summary_order(layered)
    assert order[:2] == ["pkg/base.py", "pkg/util.py"]       # most imported first
    assert order[-1] == "tests/test_a.py"
    assert sorted(order) == sorted(f.path for f in layered.index.files)


def test_summarize_writes_batches_and_the_overview(layered: Workspace):
    backend = FakeBackend(summary_handler)
    llm = LLM(backend=backend)
    saved_at_progress = []

    def on_progress(done: int, total: int, path: str) -> None:
        saved_at_progress.append((done, total, len(layered.store.all_summaries())))

    result = summarize(layered, llm, batch_size=2, on_progress=on_progress)
    assert isinstance(result, SummarizeResult)
    expected = [p for p in summary_order(layered) if p != "pkg/empty.py"]
    assert backend.file_calls() == expected
    assert (result.done, result.failed, result.skipped) == (len(expected), 0, 1)
    assert result.remaining == 0 and not result.stopped_early
    # Saved every 2 files: before file n is processed, (n - 1) // 2 * 2 are stored.
    assert [s for _d, _t, s in saved_at_progress] == [(d - 1) // 2 * 2 for d, _t, _s in saved_at_progress]
    assert all(t == len(expected) for _d, t, _s in saved_at_progress)
    assert layered.summaries() == {p: f"Summary of {p}." for p in expected}
    assert result.repo_summary_written is True
    assert backend.repo_calls() == 1
    assert layered.repo_summary() == "Overview of the repository."
    # The new summaries reach the search index.
    assert any("summary" in h.why for h in layered.search_index().search("summary", k=3))


def test_summarize_skips_existing_and_all_redoes_them(layered: Workspace):
    summarize(layered, LLM(backend=FakeBackend(summary_handler)))
    again = FakeBackend(summary_handler)
    result = summarize(layered, LLM(backend=again))
    assert (result.done, result.failed) == (0, 0)
    assert result.skipped == len(layered.index.files)
    assert again.calls == []                                  # overview already exists
    assert result.repo_summary_written is False

    redo = FakeBackend(summary_handler)
    result = summarize(layered, LLM(backend=redo), only_missing=False)
    assert result.done == len(layered.index.files) - 1        # all but the empty file
    assert redo.repo_calls() == 1


def test_summarize_limit(layered: Workspace):
    backend = FakeBackend(summary_handler)
    result = summarize(layered, LLM(backend=backend), limit=2)
    assert backend.file_calls() == summary_order(layered)[:2]
    assert result.done == 2
    assert result.remaining == len(layered.index.files) - 1 - 2
    assert result.repo_summary_written is True                # written from the summaries available


def test_summarize_stops_after_three_consecutive_failures(layered: Workspace):
    def failing(model, system, user):
        raise BackendError("rate limited")

    backend = FakeBackend(failing)
    result = summarize(layered, LLM(backend=backend), batch_size=10)
    assert (result.done, result.failed) == (0, 3)
    assert result.stopped_early is True
    assert "rate limited" in (result.error or "")
    assert result.repo_summary_written is False
    assert len(backend.calls) == 3
    failed = layered.failed_summaries()
    assert len(failed) == 3 and all(s.startswith(FAILED_SUMMARY_PREFIX) for s in failed.values())
    assert layered.summaries() == {}


def test_failures_in_between_do_not_stop_the_run(layered: Workspace):
    count = {"n": 0}

    def flaky(model, system, user):
        if system == prompts.FILE_SUMMARY_SYSTEM:
            count["n"] += 1
            if count["n"] % 2 == 0:
                raise BackendError("flaky")
        return summary_handler(model, system, user)

    result = summarize(layered, LLM(backend=FakeBackend(flaky)))
    assert result.stopped_early is False
    assert result.failed == 3 and result.done == 3
    assert result.repo_summary_written is False               # not after failures


def test_a_failure_never_overwrites_a_good_summary(layered: Workspace):
    layered.store.save_summary("pkg/base.py", "Good summary.")

    def failing(model, system, user):
        raise BackendError("down")

    summarize(layered, LLM(backend=FakeBackend(failing)), only_missing=False, limit=1)
    assert layered.store.get_summary("pkg/base.py") == "Good summary."


def test_config_error_propagates_after_saving_finished_work(layered: Workspace):
    count = {"n": 0}

    def handler(model, system, user):
        count["n"] += 1
        if count["n"] == 3:
            raise LLMConfigError("no credentials")
        return summary_handler(model, system, user)

    with pytest.raises(LLMConfigError):
        summarize(layered, LLM(backend=FakeBackend(handler)), batch_size=10)
    assert len(layered.summaries()) == 2                      # saved although the batch was not full


def test_empty_repo_summarize_does_nothing(make_repo, home: Path):
    root = make_repo({"empty.py": ""})
    with open_workspace(str(root), data_dir=home) as ws:
        backend = FakeBackend(summary_handler)
        result = summarize(ws, LLM(backend=backend))
    assert (result.done, result.failed, result.skipped, result.repo_summary_written) == (0, 0, 1, False)
    assert backend.calls == []


def test_stale_files(make_repo, home: Path):
    root = make_repo({"m.py": "def f():\n    return 1\n", "n.py": "x = 1\n"})
    with open_workspace(str(root), data_dir=home) as ws:
        assert pipeline.stale_files(ws) == []
        (root / "m.py").write_text("def f():\n    return 12345\n", encoding="utf-8")
        (root / "n.py").unlink()
        assert pipeline.stale_files(ws) == ["m.py", "n.py"]

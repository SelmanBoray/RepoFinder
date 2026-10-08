"""Tests for repofinder.store: lossless round-trip, summaries, meta, lifecycle."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from repofinder.models import FileRecord, ImportEdge, RepoIndex, Symbol
from repofinder.store import Store, StoreError


def test_round_trip_is_lossless(sample_index: RepoIndex, tmp_path: Path) -> None:
    db = tmp_path / "nested" / "dirs" / "index.db"  # parent dirs are created
    with Store(db) as store:
        assert not store.has_index()
        store.save_index(sample_index)
        assert store.has_index()
        loaded = store.load_index()
    assert loaded == sample_index
    # sanity: the sample exercises the interesting values
    assert any(s.docstring is None for s in loaded.symbols)
    assert any(s.parent is None for s in loaded.symbols)
    assert any("<locals>" in s.qualname for s in loaded.symbols)
    assert any(e.names == [] for e in loaded.imports)
    assert any(e.names == ["*"] for e in loaded.imports)
    assert any(e.dst_path is None for e in loaded.imports)
    assert any(not f.parse_ok for f in loaded.files)
    assert all(isinstance(f.parse_ok, bool) for f in loaded.files)


def test_round_trip_survives_reopen(sample_index: RepoIndex, tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    with Store(db) as store:
        store.save_index(sample_index)
    with Store(db) as store:
        assert store.load_index() == sample_index


def test_round_trip_with_duplicate_ids_and_odd_text(tmp_path: Path) -> None:
    index = RepoIndex(name="odd", root="C:\\repos\\odd", source="https://github.com/o/odd")
    index.files = [FileRecord("a.py", "python", 10, 2, True), FileRecord("ü/b.py", "python", 0, 0, False)]
    dup = dict(path="a.py", name="p", qualname="C.p", kind="method", parent="C")
    index.symbols = [
        Symbol(id="a.py::C.p", start_line=1, end_line=2, signature="def p(self)", docstring=None, **dup),
        Symbol(id="a.py::C.p", start_line=4, end_line=5, signature="def p(self, v)",
               docstring="Setter 'quoted' \"double\" \n ünïcödé", **dup),
    ]
    index.imports = [
        ImportEdge("a.py", "os", None, [], 1),
        ImportEdge("a.py", "sys", None, [], 1),  # same line: source order kept
        ImportEdge("a.py", "x.y", "x/y.py", ["a", "b", "*"], 3),
    ]
    with Store(tmp_path / "odd.db") as store:
        store.save_index(index)
        assert store.load_index() == index


def test_save_index_replaces_previous_content(sample_index: RepoIndex, tmp_path: Path) -> None:
    smaller = copy.deepcopy(sample_index)
    smaller.files = smaller.files[:1]
    smaller.symbols = [s for s in smaller.symbols if s.path == smaller.files[0].path]
    smaller.imports = [e for e in smaller.imports if e.src_path == smaller.files[0].path]
    with Store(tmp_path / "index.db") as store:
        store.save_index(sample_index)
        store.save_index(smaller)
        assert store.load_index() == smaller


def test_load_index_on_empty_store_raises(tmp_path: Path) -> None:
    with Store(tmp_path / "empty.db") as store:
        assert store.has_index() is False
        with pytest.raises(StoreError):
            store.load_index()


def test_summaries(sample_index: RepoIndex, tmp_path: Path) -> None:
    with Store(tmp_path / "index.db") as store:
        store.save_index(sample_index)
        assert store.get_summary("mypkg/core.py") is None
        store.save_summary("mypkg/core.py", "Engine and runner.")
        store.save_summary("mypkg/sub/models.py", "Data models.")
        store.save_summary("mypkg/core.py", "Engine, runner, helpers.")  # overwrite
        assert store.get_summary("mypkg/core.py") == "Engine, runner, helpers."
        assert store.all_summaries() == {
            "mypkg/core.py": "Engine, runner, helpers.",
            "mypkg/sub/models.py": "Data models.",
        }

        # Re-indexing the same source keeps summaries of files that still exist.
        trimmed = copy.deepcopy(sample_index)
        trimmed.files = [f for f in trimmed.files if f.path != "mypkg/sub/models.py"]
        store.save_index(trimmed)
        assert store.all_summaries() == {"mypkg/core.py": "Engine, runner, helpers."}

        # A different source drops every summary.
        other = copy.deepcopy(sample_index)
        other.source = "https://github.com/someone/else"
        store.save_index(other)
        assert store.all_summaries() == {}


def test_meta(sample_index: RepoIndex, tmp_path: Path) -> None:
    with Store(tmp_path / "index.db") as store:
        assert store.get_meta("missing") is None
        store.set_meta("embedding_model", "none")
        store.set_meta("embedding_model", "v2")
        assert store.get_meta("embedding_model") == "v2"
        store.save_index(sample_index)
        assert store.get_meta("embedding_model") == "v2"  # caller meta survives re-index
        assert store.get_meta("index.source") == sample_index.source


def test_closed_store_raises(tmp_path: Path) -> None:
    store = Store(tmp_path / "index.db")
    with store:
        pass
    with pytest.raises(StoreError):
        store.has_index()
    store.close()  # idempotent

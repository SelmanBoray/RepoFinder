"""Search treats Rust unit tests inline in source files as tests.

Items inside a ``#[cfg(test)]`` module and functions carrying ``#[test]`` (or a
``*::test`` attribute) take the test penalty unless the query mentions tests;
product code of the same file does not. The flag is read from what the store
keeps (the signature and the range), so it survives a save/load round trip.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

pytest.importorskip("tree_sitter_rust")

from repofinder.graph import inline_test_scopes, is_test_symbol  # noqa: E402
from repofinder.ingest import build_index  # noqa: E402
from repofinder.models import FileRecord, RepoIndex, Symbol  # noqa: E402
from repofinder.search import SearchIndex  # noqa: E402
from repofinder.store import Store  # noqa: E402

LIB_RS = """
/// Counts the matching lines of a haystack.
pub fn count_matches(haystack: &str) -> usize {
    haystack.lines().count()
}

#[cfg(feature = "simd")]
mod simd {
    /// Counts the matching lines of a haystack with SIMD.
    pub fn count_matches_fast(haystack: &str) -> usize {
        super::count_matches(haystack)
    }
}

#[test]
fn count_matches_empty() {
    // counts the matching lines of an empty haystack
    assert_eq!(super::count_matches(""), 0);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Counts the matching lines of a haystack in a test.
    fn count_matches_helper(haystack: &str) -> usize {
        count_matches(haystack)
    }

    mod nested {
        /// Counts the matching lines of a haystack, nested.
        fn count_matches_nested() {}
    }

    #[tokio::test]
    async fn count_matches_lines() {
        // counts the matching lines of a haystack
        assert_eq!(super::count_matches_helper("a\\nb"), 2);
    }
}
"""


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")


@pytest.fixture()
def rust_index(tmp_path: Path) -> RepoIndex:
    _write(tmp_path, "src/lib.rs", LIB_RS)
    return build_index(tmp_path, "t", str(tmp_path))


def _flags(index: RepoIndex) -> dict[str, bool]:
    scopes = inline_test_scopes(index.symbols)
    return {s.qualname: is_test_symbol(s, scopes) for s in index.symbols}


def test_inline_tests_are_flagged_and_product_code_is_not(rust_index: RepoIndex) -> None:
    assert _flags(rust_index) == {
        "count_matches": False,
        "simd": False,
        "simd.count_matches_fast": False,
        "count_matches_empty": True,
        "tests": True,
        "tests.count_matches_helper": True,
        "tests.nested": True,
        "tests.nested.count_matches_nested": True,
        "tests.count_matches_lines": True,
    }


def test_flags_survive_a_store_round_trip(rust_index: RepoIndex) -> None:
    with Store(Path(":memory:")) as store:
        store.save_index(rust_index)
        loaded = store.load_index()
    assert _flags(loaded) == _flags(rust_index)


def test_inline_rust_tests_rank_below_product_code(rust_index: RepoIndex) -> None:
    search = SearchIndex(rust_index)
    hits = [h for h in search.search("count matches haystack lines", k=20) if h.kind == "symbol"]
    by_id = {h.symbol_id: h for h in hits}
    product = by_id["src/lib.rs::count_matches"]
    assert hits[0] is product
    assert "test file" not in product.why
    assert "test file" not in by_id["src/lib.rs::simd.count_matches_fast"].why
    for test_id in (
        "src/lib.rs::count_matches_empty",
        "src/lib.rs::tests.count_matches_helper",
        "src/lib.rs::tests.nested.count_matches_nested",
        "src/lib.rs::tests.count_matches_lines",
    ):
        assert "test file" in by_id[test_id].why
        assert by_id[test_id].score < product.score
    # The file holds product code: its file-level record is not a test.
    files = [h for h in search.search("lib", k=20) if h.kind == "file"]
    assert files and all("test file" not in h.why for h in files)


def test_the_penalty_lifts_when_the_query_mentions_tests(rust_index: RepoIndex) -> None:
    search = SearchIndex(rust_index)
    plain = {h.symbol_id: h for h in search.search("count matches haystack lines", k=20)}
    tests = {h.symbol_id: h for h in search.search("tests for count matches haystack lines", k=20)}
    assert all("test file" not in h.why for h in tests.values())
    helper = "src/lib.rs::tests.count_matches_helper"
    assert tests[helper].score > plain[helper].score


def test_a_hand_built_index_without_test_markers_stays_product() -> None:
    # Signatures without a leading test attribute (and non-Rust files) are product code.
    symbols = [
        Symbol("src/a.rs::tests", "src/a.rs", "tests", "tests", "namespace", 1, 9, "mod tests", None, None),
        Symbol("src/a.rs::tests.f", "src/a.rs", "f", "tests.f", "function", 2, 3, "fn f()", None, "tests"),
        Symbol("pkg/a.py::f", "pkg/a.py", "f", "f", "function", 1, 2, "#[test] def f()", None, None),
    ]
    index = RepoIndex("t", "does-not-exist", "t",
                      [FileRecord("src/a.rs", "rust", 1, 9, True), FileRecord("pkg/a.py", "python", 1, 2, True)],
                      symbols)
    assert _flags(index) == {"tests": False, "tests.f": False, "f": False}

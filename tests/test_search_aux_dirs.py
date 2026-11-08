"""Tests for the auxiliary-directory penalty of repofinder.search: examples,
docs, benchmarks and scripts at the repository root or beside a package
manifest rank below product code, unless the query asks for them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

import repofinder.search as search
from repofinder.models import FileRecord, RepoIndex, Symbol
from repofinder.search import SearchIndex, _AuxDirs, _shipped_paths


def sym(path: str, qualname: str, doc: Optional[str] = None) -> Symbol:
    name = qualname.rsplit(".", 1)[-1]
    return Symbol(
        id=f"{path}::{qualname}", path=path, name=name, qualname=qualname, kind="function",
        start_line=1, end_line=2, signature=f"def {name}()", docstring=doc, parent=None,
    )


def make_index(root: Path | str, symbols: list[Symbol]) -> RepoIndex:
    paths = sorted({s.path for s in symbols})
    files = [FileRecord(p, "python", 100, 50, True) for p in paths]
    return RepoIndex(name="t", root=str(root), source="t", files=files, symbols=symbols)


def write(root: Path, rel: str, text: str = "") -> None:
    full = root / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(text, encoding="utf-8")


def order(si: SearchIndex, query: str) -> list[str]:
    return [h.path for h in si.search(query, k=10) if h.kind == "symbol"]


def why_of(si: SearchIndex, query: str, path: str) -> str:
    return next(h.why for h in si.search(query, k=10) if h.path == path and h.kind == "symbol")


def boost_of(si: SearchIndex, query: str, path: str) -> float:
    q = si._analyze(query)
    doc = next(d for d in si._docs if d.kind == "symbol" and d.path == path)
    return si._boost(doc, q, None)


# Two copies of the same function: one in product code, one elsewhere. On a
# tie the path decides, so without the penalty "examples/..." (< "src/...")
# would come first.
PRODUCT = "src/pkg/loader.py"


def pair(root: Path | str, other: str) -> SearchIndex:
    return SearchIndex(make_index(root, [sym(PRODUCT, "load_config"), sym(other, "load_config")]))


# --------------------------------------------------------------------------- #
# which directories count
# --------------------------------------------------------------------------- #


def test_root_examples_dir_is_penalised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write(tmp_path, "pyproject.toml", "[project]\nname = 'pkg'\n")
    # The example matches a little better (its docstring repeats the words) ...
    si = SearchIndex(make_index(tmp_path, [
        sym(PRODUCT, "load_config"), sym("examples/app/loader.py", "load_config", doc="Load the config."),
    ]))
    monkeypatch.setattr(search, "AUX_PENALTY", 1.0)
    assert order(si, "load config") == ["examples/app/loader.py", PRODUCT]
    monkeypatch.undo()
    # ... but product code comes first.
    assert order(si, "load config") == [PRODUCT, "examples/app/loader.py"]
    assert f"auxiliary dir x{search.AUX_PENALTY:g}" in why_of(si, "load config", "examples/app/loader.py")
    assert "auxiliary" not in why_of(si, "load config", PRODUCT)


def test_root_rule_needs_no_checkout() -> None:
    si = pair("does-not-exist", "scripts/loader.py")
    assert order(si, "load config") == [PRODUCT, "scripts/loader.py"]


@pytest.mark.parametrize("name", ["examples", "example", "doc", "docs", "benchmark", "benchmarks", "benches", "scripts", "experiments"])
def test_every_aux_dir_name_counts_at_the_root(name: str) -> None:
    aux = _AuxDirs("does-not-exist")
    assert aux.is_aux(f"{name}/a.py")
    assert not aux.is_aux(f"{name}.py")  # a file, not a directory
    assert not aux.is_aux("src/a.py")


def test_example_dir_beside_a_manifest_is_penalised(tmp_path: Path) -> None:
    write(tmp_path, "packages/x/pubspec.yaml", "name: x\n")
    other = "packages/x/example/lib/loader.py"
    si = pair(tmp_path, other)
    assert order(si, "load config") == [PRODUCT, other]
    aux = _AuxDirs(str(tmp_path))
    for manifest in ("package.json", "Cargo.toml", "go.mod", "pom.xml", "build.gradle", "pyproject.toml", "setup.py"):
        write(tmp_path, f"packages/{manifest}-pkg/{manifest}", "")
        assert aux.is_aux(f"packages/{manifest}-pkg/examples/a.py"), manifest


def test_example_dir_without_a_manifest_beside_it_is_product_code(tmp_path: Path) -> None:
    write(tmp_path, "packages/x/pubspec.yaml", "name: x\n")
    aux = _AuxDirs(str(tmp_path))
    # packages/y has no manifest; packages/x/lib/example is deeper than x's root.
    assert not aux.is_aux("packages/y/example/a.py")
    assert not aux.is_aux("packages/x/lib/example/a.py")


def test_deep_scripts_dir_in_product_source_is_not_penalised(tmp_path: Path) -> None:
    write(tmp_path, "package.json", json.dumps({"name": "pkg"}))
    other = "src/a/scripts/loader.py"
    si = pair(tmp_path, other)
    assert "auxiliary" not in why_of(si, "load config", other)
    assert boost_of(si, "load config", other) == pytest.approx(boost_of(si, "load config", PRODUCT))


# --------------------------------------------------------------------------- #
# shipped files are exempt
# --------------------------------------------------------------------------- #


def test_file_a_manifest_declares_as_shipped_is_exempt(tmp_path: Path) -> None:
    write(tmp_path, "package.json", json.dumps({
        "name": "pkg",
        "bin": {"pkg": "./scripts/cli.mjs"},
        "files": ["dist", "scripts/hook.cjs", "scripts/*.mjs"],
    }))
    aux = _AuxDirs(str(tmp_path))
    assert not aux.is_aux("scripts/cli.mjs")      # bin
    assert not aux.is_aux("scripts/hook.cjs")     # a plain file entry of "files"
    assert aux.is_aux("scripts/build.mjs")        # only a glob covers it: not a declaration of this file
    si = pair(tmp_path, "scripts/cli.mjs")
    assert "auxiliary" not in why_of(si, "load config", "scripts/cli.mjs")


def test_shipped_declaration_applies_beside_a_nested_manifest(tmp_path: Path) -> None:
    write(tmp_path, "packages/x/package.json", json.dumps({"main": "scripts/index.js"}))
    aux = _AuxDirs(str(tmp_path))
    assert not aux.is_aux("packages/x/scripts/index.js")
    assert aux.is_aux("packages/x/scripts/release.js")


def test_shipped_paths_of_python_and_cargo_manifests(tmp_path: Path) -> None:
    write(tmp_path, "pyproject.toml",
          "[project.scripts]\ntool = 'scripts.tool:main'\n[tool.setuptools]\nscript-files = ['scripts/run.sh']\n")
    shipped = _shipped_paths(tmp_path / "pyproject.toml")
    assert {"scripts/tool.py", "scripts/tool/__init__.py", "src/scripts/tool.py", "scripts/run.sh"} <= shipped
    write(tmp_path, "setup.py", "setup(scripts=['scripts/a.py'], entry_points={'console_scripts': ['b = scripts.b:main']})")
    assert {"scripts/a.py", "scripts/b.py"} <= _shipped_paths(tmp_path / "setup.py")
    write(tmp_path, "Cargo.toml",
          "[[bin]]\nname = 'x'\npath = 'examples/x.rs'\n[[example]]\nname = 'y'\npath = 'examples/y.rs'\n")
    assert _shipped_paths(tmp_path / "Cargo.toml") == {"examples/x.rs"}  # [[example]] does not ship
    write(tmp_path, "broken/package.json", "{not json")
    assert _shipped_paths(tmp_path / "broken/package.json") == set()


# --------------------------------------------------------------------------- #
# the query lifts the penalty
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("query", [
    "load config example",
    "load config in the examples",
    "tutorial load config",
    "load config script",
    "örnekler load config",
    "örneğindeki load config nerede",
    "ornekte load config nerede",
])
def test_query_asking_for_examples_lifts_the_penalty(query: str) -> None:
    si = pair("does-not-exist", "examples/app/loader.py")
    assert si._analyze(query).wants_aux
    assert boost_of(si, query, "examples/app/loader.py") == pytest.approx(boost_of(si, query, PRODUCT))


@pytest.mark.parametrize("query", ["load config, for example on startup", "örneğin load config nerede", "load config"])
def test_for_example_does_not_ask_for_examples(query: str) -> None:
    si = pair("does-not-exist", "examples/app/loader.py")
    assert not si._analyze(query).wants_aux


# --------------------------------------------------------------------------- #
# tests and auxiliary directories together
# --------------------------------------------------------------------------- #


def test_a_test_in_an_aux_dir_gets_the_stronger_penalty_not_both(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both penalties say "not product code": the stronger one applies, they
    are not multiplied."""
    aux_test = "examples/app/tests/test_loader.py"
    si = SearchIndex(make_index("does-not-exist", [
        sym(PRODUCT, "load_config"),
        sym(aux_test, "load_config"),
        sym("examples/app/loader.py", "load_config"),
        sym("tests/test_loader.py", "load_config"),
    ]))
    base = boost_of(si, "load config", PRODUCT)
    for penalty in (0.3, 0.5, 0.7):
        monkeypatch.setattr(search, "AUX_PENALTY", penalty)
        strongest = min(penalty, search.TEST_PENALTY)
        assert boost_of(si, "load config", aux_test) == pytest.approx(base * strongest)
        assert boost_of(si, "load config", "examples/app/loader.py") == pytest.approx(base * penalty)
        assert boost_of(si, "load config", "tests/test_loader.py") == pytest.approx(base * search.TEST_PENALTY)
        # Asking for tests lifts only the test penalty, asking for examples
        # only the other (then it ranks as any test file does).
        assert boost_of(si, "load config tests", aux_test) == pytest.approx(
            boost_of(si, "load config tests", PRODUCT) * penalty)
        assert boost_of(si, "load config example", aux_test) == pytest.approx(
            boost_of(si, "load config example", "tests/test_loader.py"))

"""Tests for the auxiliary-directory penalty of repofinder.search: examples,
docs, benchmarks and scripts at the repository root or beside a package
manifest rank below product code, unless the query asks for them."""

from __future__ import annotations

import json
import os
import subprocess
import sys
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


# --------------------------------------------------------------------------- #
# hostile manifests never break indexing
# --------------------------------------------------------------------------- #

DEEP = 5000  # far past the default recursion limit


@pytest.mark.parametrize("rel, content", [
    ("package.json", '{"main": ' + "[" * DEEP + "]" * DEEP + "}"),
    ("package.json", '{"exports": ' + '{"a": ' * DEEP + '"./scripts/x.js"' + "}" * DEEP + "}"),
    ("Cargo.toml", "[lib]\nx = " + "[" * DEEP + "]" * DEEP + "\n"),
    ("Cargo.toml", "[lib]\nx = " + "{a = " * DEEP + "1" + "}" * DEEP + "\n"),
    ("pyproject.toml", "[project]\nx = " + "[" * DEEP + "]" * DEEP + "\n"),
    ("package.json", b"\xff\xfe\x00{\x00\x80 not json"),
    ("Cargo.toml", b"[lib]\npath = \"\xff\xfe\"\n\x00\x00"),
    ("pyproject.toml", "project = 'not a table'\ntool = 3\n"),
    ("package.json", '{"bin": 5, "main": null, "files": "scripts/a.py", "exports": [1, null, {"b": 2}]}'),
    ("package.json", "[1, 2, 3]"),
    ("setup.py", "setup(scripts=[" + "'x', " * 20000),
    ("setup.py", "scripts=[" * 20000),
], ids=[
    "json-deep-array", "json-deep-object", "cargo-deep-array", "cargo-deep-inline-table",
    "pyproject-deep-array", "json-binary", "cargo-binary", "pyproject-wrong-shapes",
    "json-wrong-shapes", "json-not-an-object", "setup-unclosed-list", "setup-repeated-opening",
])
def test_a_hostile_manifest_gives_no_exemption_and_never_breaks_indexing(
    tmp_path: Path, rel: str, content: str | bytes
) -> None:
    full = tmp_path / rel
    if isinstance(content, bytes):
        full.write_bytes(content)
    else:
        full.write_text(content, encoding="utf-8")
    si = pair(tmp_path, "scripts/a.py")  # builds: the manifest beside scripts/ is read
    assert order(si, "load config") == [PRODUCT, "scripts/a.py"]
    assert _AuxDirs(str(tmp_path)).is_aux("scripts/a.py")


def test_an_oversized_manifest_is_not_read(tmp_path: Path) -> None:
    padding = " " * (search._MANIFEST_MAX_BYTES + 1)
    write(tmp_path, "package.json", json.dumps({"main": "scripts/a.py"}) + padding)
    assert _shipped_paths(tmp_path / "package.json") == set()
    assert _AuxDirs(str(tmp_path)).is_aux("scripts/a.py")


def test_a_manifest_that_is_a_directory_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "packages" / "x" / "package.json").mkdir(parents=True)
    aux = _AuxDirs(str(tmp_path))
    assert not aux.is_aux("packages/x/examples/a.py")  # no manifest file: not a package root
    assert aux.is_aux("examples/a.py")


# --------------------------------------------------------------------------- #
# TOML manifests without tomllib (Python 3.10)
# --------------------------------------------------------------------------- #

CARGO = """\
[package]
name = "x"
description = '''
[[bin]]
path = "not/a/target.rs"
'''

[lib]
path = "examples/lib.rs"   # a comment with "quotes"

[[bin]]
name = "tool"
path = 'scripts/tool.rs'

[[bin]]
name = "other"
path = "benches/other.rs"

[[example]]
name = "demo"
path = "examples/demo.rs"

[[bench]]
name = "speed"
path = "benches/speed.rs"
"""

PYPROJECT = """\
[project]
name = "pkg"
gui-scripts = { gui = "scripts.gui:main" }
dependencies = [
    "requests>=2",  # not a script
]

[project.scripts]
tool = "scripts.tool:main"
"quoted-name" = 'scripts.quoted:run'

[tool.poetry.scripts]
poet = "scripts.poet:main"

[tool.setuptools]
script-files = [
    "scripts/run.sh",
    'scripts/other.sh',
]
"""

CARGO_SHIPPED = {"examples/lib.rs", "scripts/tool.rs", "benches/other.rs"}
PYPROJECT_SHIPPED = {
    f"{root}scripts/{name}{suffix}"
    for root in ("", "src/")
    for name in ("gui", "tool", "quoted", "poet")
    for suffix in (".py", "/__init__.py")
} | {"scripts/run.sh", "scripts/other.sh"}


def test_toml_manifests_are_read_by_line(tmp_path: Path) -> None:
    write(tmp_path, "Cargo.toml", CARGO)
    write(tmp_path, "pyproject.toml", PYPROJECT)
    assert _shipped_paths(tmp_path / "Cargo.toml") == CARGO_SHIPPED
    assert _shipped_paths(tmp_path / "pyproject.toml") == PYPROJECT_SHIPPED


def test_the_toml_samples_mean_what_tomllib_says() -> None:
    tomllib = pytest.importorskip("tomllib")
    cargo = tomllib.loads(CARGO)
    assert {cargo["lib"]["path"]} | {b["path"] for b in cargo["bin"]} == CARGO_SHIPPED
    py = tomllib.loads(PYPROJECT)
    modules = [*py["project"]["scripts"].values(), *py["project"]["gui-scripts"].values(),
               *py["tool"]["poetry"]["scripts"].values()]
    assert {m.split(":")[0] for m in modules} == {f"scripts.{n}" for n in ("gui", "tool", "quoted", "poet")}
    assert set(py["tool"]["setuptools"]["script-files"]) == {"scripts/run.sh", "scripts/other.sh"}


def test_toml_exemptions_hold_without_tomllib(tmp_path: Path) -> None:
    """Python 3.10 has no tomllib: the same files are exempt."""
    write(tmp_path, "Cargo.toml", CARGO)
    write(tmp_path, "py/pyproject.toml", PYPROJECT)
    code = (
        "import json, sys\n"
        "sys.modules['tomllib'] = None  # import tomllib -> ImportError\n"
        "from pathlib import Path\n"
        "import repofinder.search as s\n"
        "root = Path(sys.argv[1])\n"
        "aux = s._AuxDirs(str(root))\n"
        "print(json.dumps([sorted(s._shipped_paths(root / 'Cargo.toml')),\n"
        "                  sorted(s._shipped_paths(root / 'py' / 'pyproject.toml')),\n"
        "                  aux.is_aux('scripts/tool.rs'), aux.is_aux('scripts/build.rs')]))\n"
    )
    env = dict(os.environ, PYTHONPATH=str(Path(search.__file__).resolve().parents[1]))
    done = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True,
                          env=env, timeout=120)
    assert done.returncode == 0, done.stderr
    cargo, py, tool_aux, build_aux = json.loads(done.stdout)
    assert set(cargo) == CARGO_SHIPPED
    assert set(py) == PYPROJECT_SHIPPED
    assert (tool_aux, build_aux) == (False, True)

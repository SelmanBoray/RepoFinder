"""Tests for repofinder.graph: fan-in/out, dedup, packages, test detection, entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from repofinder.graph import (
    ModuleGraph,
    build_graph,
    detect_entry_points,
    entry_point_reasons,
    is_test_path,
    module_names,
    path_module_name,
    split_source_extension,
)
from repofinder.ingest import build_index
from repofinder.models import FileRecord, ImportEdge, RepoIndex, Symbol

from conftest import FLASK_ROOT, needs_flask  # noqa: E402

MakeRepo = Callable[[dict[str, "str | bytes"]], Path]


def tiny_index() -> RepoIndex:
    """src-layout package with a hub module, a test and an external import."""
    files = [
        "src/pkg/__init__.py",
        "src/pkg/core.py",
        "src/pkg/util.py",
        "src/pkg/extra.py",
        "tests/test_core.py",
        "tools/build.py",
    ]
    symbols = [
        Symbol("src/pkg/core.py::Engine", "src/pkg/core.py", "Engine", "Engine", "class", 1, 9, "class Engine", None, None),
        Symbol("src/pkg/core.py::Engine.run", "src/pkg/core.py", "run", "Engine.run", "method", 2, 3, "def run(self)", None, "Engine"),
        Symbol("src/pkg/util.py::helper", "src/pkg/util.py", "helper", "helper", "function", 1, 2, "def helper()", None, None),
    ]
    imports = [
        ImportEdge("src/pkg/__init__.py", "pkg.core", "src/pkg/core.py", ["Engine"], 1),
        ImportEdge("src/pkg/__init__.py", "pkg", "src/pkg/__init__.py", ["x"], 2),  # self-import: dropped
        ImportEdge("src/pkg/core.py", "pkg.util", "src/pkg/util.py", ["helper"], 1),
        ImportEdge("src/pkg/core.py", "pkg.util", "src/pkg/util.py", [], 2),       # duplicate pair: one edge
        ImportEdge("src/pkg/core.py", "os", None, [], 3),                          # external: ignored
        ImportEdge("src/pkg/extra.py", "pkg.util", "src/pkg/util.py", [], 1),
        ImportEdge("tests/test_core.py", "pkg", "src/pkg/__init__.py", ["Engine"], 1),
        ImportEdge("tests/test_core.py", "pkg.core", "src/pkg/core.py", [], 2),
        ImportEdge("tools/build.py", "pkg.util", "src/pkg/util.py", [], 1),
    ]
    return RepoIndex(
        name="tiny",
        root="Z:/does/not/exist",
        source="tiny",
        files=[FileRecord(p, "python", 10, 7, True) for p in files],
        symbols=symbols,
        imports=imports,
    )


def test_edges_are_deduplicated_sorted_and_in_repo_only() -> None:
    graph = build_graph(tiny_index())
    assert graph.edges == [
        ("src/pkg/__init__.py", "src/pkg/core.py"),
        ("src/pkg/core.py", "src/pkg/util.py"),
        ("src/pkg/extra.py", "src/pkg/util.py"),
        ("tests/test_core.py", "src/pkg/__init__.py"),
        ("tests/test_core.py", "src/pkg/core.py"),
        ("tools/build.py", "src/pkg/util.py"),
    ]


def test_nodes_fan_in_fan_out_and_metadata() -> None:
    graph = build_graph(tiny_index())
    assert set(graph.nodes) == {f.path for f in tiny_index().files}
    util, core, init = graph.nodes["src/pkg/util.py"], graph.nodes["src/pkg/core.py"], graph.nodes["src/pkg/__init__.py"]
    assert (util.fan_in, util.fan_out) == (3, 0)
    assert (core.fan_in, core.fan_out) == (2, 1)
    assert (init.fan_in, init.fan_out) == (1, 1)  # the self-import does not count
    assert core.n_symbols == 2 and util.n_symbols == 1 and init.n_symbols == 0
    assert core.n_lines == 7
    assert core.module == "pkg.core"   # shortest name wins over "src.pkg.core"
    assert init.module == "pkg"
    assert graph.nodes["tests/test_core.py"].is_test
    assert not core.is_test


def test_neighbour_queries() -> None:
    graph = build_graph(tiny_index())
    assert graph.importers_of("src/pkg/util.py") == ["src/pkg/core.py", "src/pkg/extra.py", "tools/build.py"]
    assert graph.imports_of("tests/test_core.py") == ["src/pkg/__init__.py", "src/pkg/core.py"]
    assert graph.importers_of("tools/build.py") == []
    assert graph.imports_of("missing.py") == []
    # The returned lists are copies.
    graph.importers_of("src/pkg/util.py").clear()
    assert len(graph.importers_of("src/pkg/util.py")) == 3


def test_top_by_fan_in() -> None:
    graph = build_graph(tiny_index())
    top = graph.top_by_fan_in(3)
    assert [n.path for n in top] == ["src/pkg/util.py", "src/pkg/core.py", "src/pkg/__init__.py"]
    everything = graph.top_by_fan_in(100, include_tests=True)
    assert len(everything) == 6 and "tests/test_core.py" in [n.path for n in everything]
    assert "tests/test_core.py" not in [n.path for n in graph.top_by_fan_in(100)]
    # Ties on fan-in (0): more symbols first, then path.
    assert [n.path for n in graph.top_by_fan_in(100)][-2:] == ["src/pkg/extra.py", "tools/build.py"]
    assert graph.top_by_fan_in(0) == []


def test_packages() -> None:
    graph = build_graph(tiny_index())
    assert graph.packages() == {
        "src/pkg": ["src/pkg/__init__.py", "src/pkg/core.py", "src/pkg/extra.py", "src/pkg/util.py"],
        "tests": ["tests/test_core.py"],
        "tools": ["tools/build.py"],
    }


def test_packages_for_root_modules_and_unresolvable_paths() -> None:
    files = ["setup.py", "my-scripts/run.py", "lib/pkg/__init__.py", "lib/pkg/a.py"]
    index = RepoIndex("r", "Z:/nope", "r", files=[FileRecord(p, "python", 1, 1, True) for p in files])
    packages = build_graph(index).packages()
    assert packages == {
        ".": ["setup.py"],
        "lib/pkg": ["lib/pkg/__init__.py", "lib/pkg/a.py"],
        "my-scripts": ["my-scripts/run.py"],
    }


def test_module_names_prefers_the_importable_name() -> None:
    names = module_names(["src/flask/app.py", "src/flask/__init__.py", "tests/test_x.py", "bad-dir/x.py"])
    assert names == {"src/flask/app.py": "flask.app", "src/flask/__init__.py": "flask", "tests/test_x.py": "tests.test_x"}


@pytest.mark.parametrize(
    "path, expected",
    [
        ("tests/test_app.py", True),
        ("tests/helpers.py", True),
        ("pkg/test/data.py", True),
        ("examples/tutorial/tests/conftest.py", True),
        ("pkg/test_utils.py", True),
        ("pkg/utils_test.py", True),
        ("conftest.py", True),
        ("src/flask/testing.py", False),
        ("pkg/contest.py", False),
        ("pkg/latest.py", False),
        ("testsuite/x.py", False),
    ],
)
def test_is_test_path(path: str, expected: bool) -> None:
    assert is_test_path(path) is expected


# --------------------------------------------------------------------------- #
# entry points (on-disk repos)
# --------------------------------------------------------------------------- #

PYPROJECT = """\
[project]
name = "demo"

[project.scripts]
demo = "demo.cli:main"
missing = "demo.nowhere:main"

[project.entry-points."pytest11"]
demo_plugin = "demo.plugin"
demo_plugin2 = "demo.plugin:hook"
"""


def test_entry_points_all_heuristics(make_repo: MakeRepo) -> None:
    root = make_repo(
        {
            "pyproject.toml": PYPROJECT,
            "src/demo/__init__.py": "",
            "src/demo/__main__.py": "from .cli import main\nmain()\n",
            "src/demo/cli.py": "def main():\n    pass\n\nif __name__ == '__main__':\n    main()\n",
            "src/demo/plugin.py": "def hook():\n    pass\n",
            "src/demo/server.py": "x = 1\n",
            "src/demo/core.py": "def f():\n    if __name__ == '__main__':  # indented: not a module guard\n        pass\n",
            "scripts/tool.py": "import sys\n\nif __name__ == \"__main__\":\n    sys.exit(0)\n",
            "tests/test_cli.py": "if __name__ == '__main__':\n    pass\n",
            "tests/app.py": "",
        }
    )
    index = build_index(root, "demo", str(root))
    reasons = entry_point_reasons(index)
    assert set(reasons) == {
        "src/demo/__main__.py",
        "src/demo/cli.py",
        "src/demo/plugin.py",
        "src/demo/server.py",
        "scripts/tool.py",
    }
    assert reasons["src/demo/cli.py"] == (
        'script `demo` = `demo.cli:main` (pyproject.toml [project.scripts]); '
        'has `if __name__ == "__main__"`; conventional name `cli.py`'
    )
    assert reasons["src/demo/__main__.py"] == "`__main__.py` (run with `python -m`)"
    assert reasons["src/demo/plugin.py"] == (
        "entry point `demo_plugin` = `demo.plugin` (pyproject.toml [project.entry-points.pytest11]); "
        "entry point `demo_plugin2` = `demo.plugin:hook` (pyproject.toml [project.entry-points.pytest11])"
    )
    assert reasons["scripts/tool.py"] == 'has `if __name__ == "__main__"`'
    assert reasons["src/demo/server.py"] == "conventional name `server.py`"
    assert detect_entry_points(index) == set(reasons)

    graph = build_graph(index)
    assert [n.path for n in graph.entry_points()] == [
        "src/demo/cli.py",       # declared script
        "src/demo/plugin.py",    # declared entry point
        "src/demo/__main__.py",
        "scripts/tool.py",       # main guard
        "src/demo/server.py",    # file name only
    ]
    assert graph.nodes["src/demo/cli.py"].is_entry
    assert graph.nodes["src/demo/cli.py"].entry_reason == reasons["src/demo/cli.py"]
    assert not graph.nodes["src/demo/core.py"].is_entry
    assert graph.nodes["src/demo/core.py"].entry_reason is None


def test_entry_points_from_setup_py_and_setup_cfg_in_nested_projects(make_repo: MakeRepo) -> None:
    root = make_repo(
        {
            "packages/alpha/setup.py": (
                "from setuptools import setup\n"
                "setup(name='alpha', entry_points={'console_scripts': ['alpha = alpha.tool:run [extra]']})\n"
            ),
            "packages/alpha/alpha/__init__.py": "",
            "packages/alpha/alpha/tool.py": "def run():\n    pass\n",
            "packages/beta/setup.cfg": (
                "[metadata]\nname = beta\n\n"
                "[options.entry_points]\nconsole_scripts =\n    beta-cli = beta.go:main\n\n"
                "[other]\nnot_this = beta.nope:main\n"
            ),
            "packages/beta/src/beta/__init__.py": "",
            "packages/beta/src/beta/go.py": "def main():\n    pass\n",
        }
    )
    reasons = entry_point_reasons(build_index(root, "mono", str(root)))
    assert reasons == {
        "packages/alpha/alpha/tool.py": "entry point `alpha` = `alpha.tool:run` (packages/alpha/setup.py)",
        "packages/beta/src/beta/go.py": (
            "entry point `beta-cli` = `beta.go:main` (packages/beta/setup.cfg [options.entry_points])"
        ),
    }


def test_entry_points_poetry_and_broken_pyproject(make_repo: MakeRepo) -> None:
    poetry = '[tool.poetry.scripts]\nrun-it = "app_pkg.main:start"\n'
    broken = (
        '[project.scripts]\nbroken = "app_pkg.cli:main"\n'
        '[project.entry-points.group]\nplug = "app_pkg.cli"\n[[[ not toml\n'
    )
    root = make_repo(
        {
            "pyproject.toml": poetry,
            "app_pkg/__init__.py": "",
            "app_pkg/main.py": "def start():\n    pass\n",
            "sub/pyproject.toml": broken,
            "sub/app_pkg/__init__.py": "",
            "sub/app_pkg/cli.py": "def main():\n    pass\n",
        }
    )
    reasons = entry_point_reasons(build_index(root, "p", str(root)))
    assert reasons["app_pkg/main.py"].startswith(
        "script `run-it` = `app_pkg.main:start` (pyproject.toml [tool.poetry.scripts])"
    )
    # Invalid TOML falls back to the line-based reader.
    assert reasons["sub/app_pkg/cli.py"] == (
        "script `broken` = `app_pkg.cli:main` (sub/pyproject.toml [project.scripts]); "
        "entry point `plug` = `app_pkg.cli` (sub/pyproject.toml [project.entry-points.group]); "
        "conventional name `cli.py`"
    )


def test_entry_points_without_a_checkout_use_file_names_only() -> None:
    index = tiny_index()
    index.files.append(FileRecord("src/pkg/__main__.py", "python", 1, 1, True))
    index.files.append(FileRecord("src/pkg/main.py", "python", 1, 1, True))
    assert entry_point_reasons(index) == {
        "src/pkg/__main__.py": "`__main__.py` (run with `python -m`)",
        "src/pkg/main.py": "conventional name `main.py`",
    }


def test_build_graph_on_the_sample_repo(sample_index: RepoIndex) -> None:
    graph = build_graph(sample_index)
    assert isinstance(graph, ModuleGraph)
    assert set(graph.nodes) == {f.path for f in sample_index.files}
    resolved = {(e.src_path, e.dst_path) for e in sample_index.imports if e.dst_path and e.dst_path != e.src_path}
    assert set(graph.edges) == resolved
    assert graph.edges == sorted(graph.edges)
    assert graph.nodes["mypkg/sub/models.py"].fan_in >= 1


# --------------------------------------------------------------------------- #
# Flask integration
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def flask_graph() -> ModuleGraph:
    return build_graph(build_index(FLASK_ROOT, "flask", "https://github.com/pallets/flask"))


@needs_flask
def test_flask_fan_in(flask_graph: ModuleGraph) -> None:
    top = [n.path for n in flask_graph.top_by_fan_in(10)]
    # Tests and examples import the package facade, so __init__.py is the hub.
    assert top[0] == "src/flask/__init__.py"
    # app.py is imported directly by only six modules (everyone else goes through
    # flask/__init__.py), which puts it in the top 10 but not the top 5.
    assert "src/flask/app.py" in top
    assert flask_graph.importers_of("src/flask/app.py") == [
        "src/flask/__init__.py",
        "src/flask/cli.py",
        "src/flask/ctx.py",
        "src/flask/globals.py",
        "src/flask/sessions.py",
        "src/flask/testing.py",
    ]
    assert flask_graph.nodes["src/flask/app.py"].module == "flask.app"


@needs_flask
def test_flask_entry_points(flask_graph: ModuleGraph) -> None:
    entries = {n.path: n.entry_reason or "" for n in flask_graph.entry_points()}
    assert "src/flask/__main__.py" in entries
    # pyproject.toml: [project.scripts] flask = "flask.cli:main"
    assert entries["src/flask/cli.py"].startswith(
        "script `flask` = `flask.cli:main` (pyproject.toml [project.scripts])"
    )
    assert flask_graph.entry_points()[0].path == "src/flask/cli.py"
    assert not any(flask_graph.nodes[p].is_test for p in entries)


@needs_flask
def test_flask_packages(flask_graph: ModuleGraph) -> None:
    packages = flask_graph.packages()
    assert {"src/flask", "tests", "examples", "docs"} <= set(packages)
    assert "src/flask/json/tag.py" in packages["src/flask"]


# --------------------------------------------------------------------------- #
# TypeScript: test paths, module names, packages, entry points
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path, expected",
    [
        ("src/foo.test.ts", True),
        ("src/foo.spec.ts", True),
        ("src/View.test.tsx", True),
        ("src/View.spec.tsx", True),
        ("src/lib.test.mts", True),
        ("src/ghidra/GhidraProvider.behavior.test.ts", True),
        ("src/__tests__/helpers.ts", True),
        ("packages/a/__tests__/deep/x.tsx", True),
        ("src/latest.ts", False),
        ("src/contest.ts", False),
        ("src/testing.ts", False),
        ("src/spec.ts", False),
        ("src/test-utils.ts", False),
        ("src/types.d.ts", False),
    ],
)
def test_is_test_path_typescript(path: str, expected: bool) -> None:
    assert is_test_path(path) is expected


@pytest.mark.parametrize(
    "path, expected",
    [
        ("pkg/mod.py", ("pkg/mod", ".py")),
        ("src/a.ts", ("src/a", ".ts")),
        ("src/View.tsx", ("src/View", ".tsx")),
        ("src/types.d.ts", ("src/types", ".d.ts")),
        ("scripts/lib/tool.d.mts", ("scripts/lib/tool", ".d.mts")),
        ("src/a.test.ts", ("src/a.test", ".ts")),
        ("README.md", ("README.md", "")),
        (".eslintrc", (".eslintrc", "")),
    ],
)
def test_split_source_extension(path: str, expected: tuple[str, str]) -> None:
    assert split_source_extension(path) == expected


def test_module_names_for_typescript_paths() -> None:
    names = module_names(
        [
            "src/util/strings.ts",
            "src/util/index.ts",
            "src/App.tsx",
            "src/types.d.ts",
            "scripts/lib/tool.d.mts",
            "index.ts",
            "src/flask/app.py",
            "src/flask/__init__.py",
            "README.md",
        ]
    )
    assert names == {
        "src/util/strings.ts": "src/util/strings",
        "src/util/index.ts": "src/util",          # an index file names its directory
        "src/App.tsx": "src/App",
        "src/types.d.ts": "src/types",
        "scripts/lib/tool.d.mts": "scripts/lib/tool",
        "index.ts": "index",                      # at the root there is no directory to name
        "src/flask/app.py": "flask.app",          # Python is unchanged
        "src/flask/__init__.py": "flask",
    }
    assert path_module_name("src/flask/app.py") is None


def test_packages_for_typescript_files_are_their_directories() -> None:
    files = [
        "index.ts",
        "src/cli.ts",
        "src/domain/index.ts",
        "src/domain/model.ts",
        "src/domain/rules/a.ts",
        "src/pkg/__init__.py",
        "src/pkg/core.py",
    ]
    index = RepoIndex(
        "r", "Z:/nope", "r",
        files=[FileRecord(p, "python" if p.endswith(".py") else "typescript", 1, 1, True) for p in files],
    )
    graph = build_graph(index)
    assert graph.packages() == {
        ".": ["index.ts"],
        "src": ["src/cli.ts"],
        "src/domain": ["src/domain/index.ts", "src/domain/model.ts"],
        "src/domain/rules": ["src/domain/rules/a.ts"],
        "src/pkg": ["src/pkg/__init__.py", "src/pkg/core.py"],
    }
    assert graph.nodes["src/domain/index.ts"].module == "src/domain"
    assert graph.nodes["src/pkg/core.py"].module == "pkg.core"


def test_entry_points_typescript_package_json_and_file_names(make_repo: MakeRepo) -> None:
    root_package = {"name": "@acme/tool", "bin": "./dist/cli.js", "main": "dist/index.js"}
    app_package = {
        "name": "app",
        "bin": {"app-cli": "bin/run.ts", "gone": "bin/missing.js", "bad": 3},
        "main": "lib/server.mjs",
    }
    indexed = [
        "src/cli.ts",
        "src/index.ts",
        "src/main.ts",
        "src/util.ts",
        "src/__tests__/main.ts",
        "src/app.test.ts",
        "packages/app/bin/run.ts",
        "packages/app/src/server.mts",
        "packages/broken/index.mts",
        "packages/list/app.ts",
    ]
    files: dict[str, str | bytes] = {p: "export {};\n" for p in indexed}
    files.update(
        {
            "package.json": json.dumps(root_package),
            "packages/app/package.json": json.dumps(app_package),
            "packages/broken/package.json": "{ not json",
            "packages/list/package.json": "[1, 2]",
        }
    )
    root = make_repo(files)
    index = RepoIndex("ts", str(root), "ts", files=[FileRecord(p, "typescript", 10, 1, True) for p in sorted(indexed)])
    reasons = entry_point_reasons(index)
    assert reasons == {
        # package.json "bin"/"main" at compiled output resolve to the indexed source
        "src/cli.ts": "script `tool` = `./dist/cli.js` (package.json [bin]); conventional name `cli.ts`",
        "src/index.ts": "main `dist/index.js` (package.json); conventional name `index.ts`",
        "packages/app/bin/run.ts": (
            "script `app-cli` = `bin/run.ts` (packages/app/package.json [bin]); conventional name `bin/run.ts`"
        ),
        "packages/app/src/server.mts": "main `lib/server.mjs` (packages/app/package.json)",
        # file names only (a malformed package.json is ignored)
        "src/main.ts": "conventional name `main.ts`",
        "packages/broken/index.mts": "conventional name `index.mts`",
        "packages/list/app.ts": "conventional name `app.ts`",
    }
    graph = build_graph(index)
    assert [n.path for n in graph.entry_points()] == [
        "packages/app/bin/run.ts",       # declared in a package.json
        "packages/app/src/server.mts",
        "src/cli.ts",
        "src/index.ts",
        "packages/broken/index.mts",     # file name only
        "packages/list/app.ts",
        "src/main.ts",
    ]
    assert not graph.nodes["src/__tests__/main.ts"].is_entry and graph.nodes["src/__tests__/main.ts"].is_test
    assert graph.nodes["src/app.test.ts"].is_test


def test_package_json_without_a_checkout_is_ignored() -> None:
    index = RepoIndex("ts", "Z:/does/not/exist", "ts", files=[FileRecord("src/server.ts", "typescript", 1, 1, True)])
    assert entry_point_reasons(index) == {"src/server.ts": "conventional name `server.ts`"}

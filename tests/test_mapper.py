"""Tests for repofinder.mapper: section order and content, truncation, determinism."""

from __future__ import annotations

import re
import socket
from pathlib import Path

import pytest

from repofinder.graph import build_graph
from repofinder.ingest import build_index
from repofinder.mapper import render_map
from repofinder.models import FileRecord, ImportEdge, RepoIndex, Symbol

from conftest import FLASK_ROOT, needs_flask  # noqa: E402

SECTIONS = [
    "## Overview",
    "## Entry points",
    "## Core modules",
    "## Package layout",
    "## Dependency hot spots",
    "## Where to start",
]


def _sym(path: str, qualname: str, kind: str, start: int, end: int) -> Symbol:
    name = qualname.rsplit(".", 1)[-1]
    parent = qualname.rsplit(".", 1)[0] if "." in qualname else None
    return Symbol(f"{path}::{qualname}", path, name, qualname, kind, start, end, f"def {name}()", None, parent)


def demo_index() -> RepoIndex:
    files = {
        "src/demo/__init__.py": 5,
        "src/demo/__main__.py": 3,
        "src/demo/core.py": 120,
        "src/demo/util.py": 40,
        "src/demo/io/__init__.py": 1,
        "src/demo/io/readers.py": 60,
        "tests/test_core.py": 30,
        "tests/unit/test_util.py": 20,
        "setup.py": 10,
    }
    symbols = [
        _sym("src/demo/core.py", "Engine", "class", 10, 100),
        _sym("src/demo/core.py", "Engine.run", "method", 20, 30),
        _sym("src/demo/core.py", "_private_helper", "function", 101, 119),
        _sym("src/demo/core.py", "build", "function", 1, 5),
        _sym("src/demo/core.py", "tiny", "function", 6, 7),
        _sym("src/demo/util.py", "slugify", "function", 1, 10),
        _sym("src/demo/io/readers.py", "read_csv", "function", 1, 20),
        _sym("tests/test_core.py", "test_engine", "function", 1, 5),
    ]
    imports = [
        ImportEdge("src/demo/__init__.py", "demo.core", "src/demo/core.py", ["Engine"], 1),
        ImportEdge("src/demo/__main__.py", "demo.core", "src/demo/core.py", ["build"], 1),
        ImportEdge("src/demo/core.py", "demo.util", "src/demo/util.py", ["slugify"], 1),
        ImportEdge("src/demo/io/readers.py", "demo.util", "src/demo/util.py", [], 1),
        ImportEdge("src/demo/io/readers.py", "demo.core", "src/demo/core.py", [], 2),
        ImportEdge("tests/test_core.py", "demo", "src/demo/__init__.py", ["Engine"], 1),
        ImportEdge("tests/unit/test_util.py", "demo.util", "src/demo/util.py", [], 1),
        ImportEdge("src/demo/core.py", "json", None, [], 2),
    ]
    return RepoIndex(
        name="demo",
        root="Z:/no/such/checkout",
        source="https://github.com/o/demo",
        files=[FileRecord(p, "python", n * 10, n, p != "src/demo/util.py") for p, n in files.items()],
        symbols=symbols,
        imports=imports,
    )


def section(markdown: str, title: str) -> str:
    start = markdown.index(title + "\n")
    rest = markdown[start + len(title) + 1:]
    nxt = rest.find("\n## ")
    return rest if nxt < 0 else rest[:nxt]


@pytest.fixture
def demo() -> tuple[RepoIndex, str]:
    index = demo_index()
    return index, render_map(index, build_graph(index))


def test_sections_appear_once_and_in_order(demo: tuple[RepoIndex, str]) -> None:
    _index, md = demo
    assert md.startswith("# demo — architecture map\n\n")
    headings = re.findall(r"^#{1,2} .*$", md, flags=re.MULTILINE)
    assert headings == ["# demo — architecture map", *SECTIONS]
    assert md.endswith("\n") and not md.endswith("\n\n")


def test_stats_line(demo: tuple[RepoIndex, str]) -> None:
    _index, md = demo
    stats = md.splitlines()[2]
    assert stats == (
        "9 files · 289 lines · 8 symbols (1 class, 6 functions, 1 method) · 1 parse failure"
    )


def test_overview_default_sentence_and_verbatim_summary() -> None:
    index = demo_index()
    graph = build_graph(index)
    overview = section(render_map(index, graph), "## Overview")
    assert overview.strip() == (
        "`demo` has 9 source files in 3 top-level packages and directories: `src/demo` (6 files), "
        "`tests` (2 files) and the repository root (1 file). 2 files are tests."
    )
    custom = "Demo is a *tiny* engine.\n\nSecond paragraph."
    md = render_map(index, graph, repo_summary=custom)
    assert section(md, "## Overview").strip() == custom


def test_entry_points_table(demo: tuple[RepoIndex, str]) -> None:
    _index, md = demo
    table = section(md, "## Entry points").strip().splitlines()
    assert table == [
        "| path | reason |",
        "|---|---|",
        "| `src/demo/__main__.py` | `__main__.py` (run with `python -m`) |",
    ]


def test_entry_points_empty() -> None:
    index = demo_index()
    index.files = [f for f in index.files if f.path != "src/demo/__main__.py"]
    md = render_map(index, build_graph(index))
    assert section(md, "## Entry points").strip() == "_No entry points detected._"


def test_core_modules_table_with_and_without_summaries() -> None:
    index = demo_index()
    graph = build_graph(index)
    long_summary = "Reads CSV | TSV files into rows with a configurable dialect and many many options. Second."
    summaries = {"src/demo/core.py": "Runs the engine.  It is the heart of demo.", "src/demo/io/readers.py": long_summary}
    rows = section(render_map(index, graph, summaries=summaries), "## Core modules").strip().splitlines()
    assert rows[0] == "Top 7 non-test files by fan-in (distinct in-repo files importing them)."
    assert rows[2] == "| path | module | fan-in | fan-out | symbols | summary |"
    body = rows[4:]
    # fan-in desc, then symbol count desc, then path
    assert [r.split(" | ")[0] for r in body] == [
        "| `src/demo/core.py`",
        "| `src/demo/util.py`",
        "| `src/demo/__init__.py`",
        "| `src/demo/io/readers.py`",
        "| `setup.py`",
        "| `src/demo/__main__.py`",
        "| `src/demo/io/__init__.py`",
    ]
    assert body[0] == "| `src/demo/core.py` | `demo.core` | 3 | 1 | 5 | Runs the engine. |"
    # No summary: public module-level names first, larger first.
    assert body[1] == "| `src/demo/util.py` | `demo.util` | 3 | 0 | 1 | `slugify` |"
    assert body[2].endswith("| (no classes or functions) |")
    readers = body[3]
    cell = readers.split(" | ")[-1].rstrip(" |")
    assert cell.endswith("…") and len(cell.replace("\\|", "|")) <= 80
    assert "\\|" in readers  # the pipe inside the summary is escaped
    assert readers.count(" | ") == 5

    no_summary = section(render_map(index, graph), "## Core modules")
    assert "| `src/demo/core.py` | `demo.core` | 3 | 1 | 5 | `Engine`, `build`, `tiny` |" in no_summary


def test_core_modules_respects_max_modules() -> None:
    index = demo_index()
    md = render_map(index, build_graph(index), max_modules=2)
    rows = [r for r in section(md, "## Core modules").splitlines() if r.startswith("| `")]
    assert len(rows) == 2


def test_package_layout_collapses_tests(demo: tuple[RepoIndex, str]) -> None:
    _index, md = demo
    assert section(md, "## Package layout").strip().splitlines() == [
        "- `./` — 1 file",
        "- `src/demo/` — 4 files, 6 in subtree",
        "  - `io/` — 2 files",
        "- `tests/` — 2 test files (collapsed)",
    ]


def test_hot_spots(demo: tuple[RepoIndex, str]) -> None:
    _index, md = demo
    lines = [l for l in section(md, "## Dependency hot spots").splitlines() if l.startswith("- ")]
    # Ranked by fan_in[src] + fan_in[dst], ties by (src, dst); test files excluded.
    assert lines == [
        "- `src/demo/core.py` → `src/demo/util.py` (fan-in 3 + 3)",
        "- `src/demo/__init__.py` → `src/demo/core.py` (fan-in 1 + 3)",
        "- `src/demo/__main__.py` → `src/demo/core.py` (fan-in 0 + 3)",
        "- `src/demo/io/readers.py` → `src/demo/core.py` (fan-in 0 + 3)",
        "- `src/demo/io/readers.py` → `src/demo/util.py` (fan-in 0 + 3)",
    ]
    assert "tests/" not in section(md, "## Dependency hot spots")


def test_where_to_start(demo: tuple[RepoIndex, str]) -> None:
    _index, md = demo
    assert section(md, "## Where to start").strip().splitlines() == [
        "- `src/demo/__main__.py` — entry point: `__main__.py` (run with `python -m`).",
        "- `src/demo/core.py` — imported by 3 files; defines `Engine`, `build`, `tiny`.",
        "- `src/demo/util.py` — imported by 3 files; defines `slugify`.",
        "- `src/demo/__init__.py` — imported by 1 file.",
        "- `tests/` — test suite, 2 files.",
    ]


def test_empty_repository() -> None:
    index = RepoIndex(name="empty", root="Z:/nope", source="empty")
    md = render_map(index, build_graph(index))
    headings = re.findall(r"^#{1,2} .*$", md, flags=re.MULTILINE)
    assert headings == ["# empty — architecture map", *SECTIONS]
    assert "0 files · 0 lines · 0 symbols · 0 parse failures" in md


def test_rendering_is_deterministic_and_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("render_map must not open sockets")

    index = demo_index()
    graph = build_graph(index)
    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    first = render_map(index, graph, summaries={"src/demo/util.py": "Text helpers."})
    second = render_map(index, build_graph(demo_index()), summaries={"src/demo/util.py": "Text helpers."})
    assert first == second


def test_sample_repo_map(sample_index: RepoIndex) -> None:
    md = render_map(sample_index, build_graph(sample_index))
    assert md.startswith("# sample_repo — architecture map")
    assert "1 parse failure" in md
    assert "`mypkg/sub/models.py`" in section(md, "## Core modules")


@needs_flask
def test_flask_map() -> None:
    index = build_index(FLASK_ROOT, "flask", "https://github.com/pallets/flask")
    md = render_map(index, build_graph(index))
    headings = re.findall(r"^#{1,2} .*$", md, flags=re.MULTILINE)
    assert headings == ["# flask — architecture map", *SECTIONS]
    entries = section(md, "## Entry points")
    assert "| `src/flask/__main__.py` |" in entries
    assert "| `src/flask/cli.py` | script `flask` = `flask.cli:main`" in entries
    core_rows = [r for r in section(md, "## Core modules").splitlines() if r.startswith("| `")]
    assert core_rows[0].startswith("| `src/flask/__init__.py` | `flask` |")
    assert all(not r.startswith("| `tests/") for r in core_rows)
    assert "- `tests/` — 41 test files (collapsed)" in section(md, "## Package layout")


# --------------------------------------------------------------------------- #
# TypeScript
# --------------------------------------------------------------------------- #


def ts_index() -> RepoIndex:
    files = {
        "src/cli.ts": 30,
        "src/domain/index.ts": 5,
        "src/domain/model.ts": 80,
        "src/domain/__tests__/model.test.ts": 40,
        "src/util.spec.ts": 10,
    }

    def ts_sym(path: str, name: str, kind: str, start: int, end: int) -> Symbol:
        return Symbol(f"{path}::{name}", path, name, name, kind, start, end, f"export {kind} {name}", None, None)

    symbols = [
        ts_sym("src/cli.ts", "main", "function", 1, 20),
        ts_sym("src/domain/model.ts", "Model", "interface", 1, 20),
        ts_sym("src/domain/model.ts", "Kind", "enum", 21, 30),
        ts_sym("src/domain/model.ts", "Id", "type", 31, 31),
    ]
    imports = [
        ImportEdge("src/cli.ts", "./domain", "src/domain/index.ts", ["Model"], 1),
        ImportEdge("src/domain/index.ts", "./model", "src/domain/model.ts", ["Model"], 1),
        ImportEdge("src/domain/__tests__/model.test.ts", "../model", "src/domain/model.ts", ["Kind"], 1),
    ]
    return RepoIndex(
        name="tsdemo",
        root="Z:/no/such/checkout",
        source="https://github.com/o/tsdemo",
        files=[FileRecord(p, "typescript", n * 10, n, True) for p, n in files.items()],
        symbols=symbols,
        imports=imports,
    )


def test_typescript_map() -> None:
    index = ts_index()
    md = render_map(index, build_graph(index))
    assert "4 symbols (1 function, 1 enum, 1 interface, 1 type)" in md
    # TypeScript groups modules by directory: the overview counts top-level directories.
    assert section(md, "## Overview").strip() == (
        "`tsdemo` has 5 source files, all TypeScript, in 1 top-level directory: `src` (5 files). "
        "2 files are tests."
    )
    assert "| `src/cli.ts` | conventional name `cli.ts` |" in section(md, "## Entry points")
    core = section(md, "## Core modules")
    assert "| `src/domain/model.ts` | `src/domain/model` | 2 | 0 | 3 | `Model`, `Kind`, `Id` |" in core
    assert "| `src/domain/index.ts` | `src/domain` | 1 | 1 | 0 |" in core
    assert "src/util.spec.ts" not in core and "__tests__" not in core
    assert "`__tests__/` — 1 test file (collapsed)" in section(md, "## Package layout")


# --------------------------------------------------------------------------- #
# Entry ranking, the row cap, unindexed bins and the language sentence
# --------------------------------------------------------------------------- #


def test_mixed_repo_entry_table_and_language_sentence(make_repo) -> None:
    root = make_repo(
        {
            "package.json": '{"name": "tool", "bin": {"tool": "scripts/run.mjs"}}',
            "scripts/run.mjs": "#!/usr/bin/env node\n",
            "src/cli.ts": "export function main(): void {}\n" * 17,
            "experiments/x.py": "if __name__ == '__main__':\n    pass\n",
            "tests/x.test.ts": "export {};\n",
        }
    )
    index = build_index(root, "mixed", str(root))
    md = render_map(index, build_graph(index))
    # 18 TypeScript lines, 2 Python lines.
    assert section(md, "## Overview").strip() == (
        "`mixed` has 3 source files, mostly TypeScript (90%) with some Python, in 3 top-level directories: "
        "`experiments` (1 file), `src` (1 file) and `tests` (1 file). 1 file is tests."
    )
    assert section(md, "## Entry points").strip().splitlines() == [
        "| path | reason |",
        "|---|---|",
        "| `scripts/run.mjs` | `package.json` bin (not indexed: JavaScript) |",
        "| `src/cli.ts` | conventional name `cli.ts` |",
        '| `experiments/x.py` | has `if __name__ == "__main__"` |',
    ]
    assert section(md, "## Where to start").strip().splitlines()[:3] == [
        "- `scripts/run.mjs` — entry point: `package.json` bin (not indexed: JavaScript).",
        "- `src/cli.ts` — entry point: conventional name `cli.ts`.",
        '- `experiments/x.py` — entry point: has `if __name__ == "__main__"`.',
    ]


def _named_files_index(n_nested: int) -> RepoIndex:
    """``src/cli.ts`` (score 80) plus ``n_nested`` ``pkgNN/main.py`` files (score 60); no checkout."""
    files = [FileRecord("src/cli.ts", "typescript", 1, 1, True)]
    files += [FileRecord(f"pkg{i:02d}/main.py", "python", 1, 1, True) for i in range(n_nested)]
    return RepoIndex(name="many", root="Z:/no/such/checkout", source="many", files=files)


def _entry_section(index: RepoIndex) -> str:
    return section(render_map(index, build_graph(index)), "## Entry points").strip()


def test_entry_table_is_capped_at_twelve_rows() -> None:
    index = _named_files_index(15)
    assert len(build_graph(index).all_entry_points()) == 16   # the data model keeps every entry point
    lines = _entry_section(index).splitlines()
    rows = [line for line in lines if line.startswith("| `")]
    assert [row.split(" | ")[0] for row in rows] == ["| `src/cli.ts`"] + [f"| `pkg{i:02d}/main.py`" for i in range(11)]
    assert lines[-2:] == ["", "… and 4 more (lower-confidence) entry points."]

    assert _entry_section(_named_files_index(12)).endswith("|\n\n… and 1 more (lower-confidence) entry point.")
    exact = _entry_section(_named_files_index(11))
    assert "more (lower-confidence)" not in exact and exact.count("| `") == 12


@pytest.mark.parametrize(
    "languages, expected",
    [
        # Python-led: grouped by package; the languages are named only when there are several.
        ([("python", 75), ("typescript", 25)],
         "`r` has 2 source files, mostly Python (75%) with some TypeScript, in 2 top-level packages and directories"),
        ([("python", 40), ("typescript", 35), ("tsx", 5), ("javascript", 20)],
         "`r` has 4 source files, a mix of Python (40%), TypeScript (40%) and JavaScript (20%), "
         "in 4 top-level packages and directories"),
        ([("python", 1)], "`r` has 1 source file in 1 top-level package or directory"),
        # TypeScript-led: grouped by top-level directory.
        ([("typescript", 999), ("python", 1)],
         "`r` has 2 source files, mostly TypeScript (99%) with some Python, in 2 top-level directories"),
    ],
)
def test_overview_names_the_languages(languages: list[tuple[str, int]], expected: str) -> None:
    files = [
        FileRecord(f"d{i}/m{i}.{'py' if lang == 'python' else 'ts'}", lang, 1, n, True)
        for i, (lang, n) in enumerate(languages)
    ]
    index = RepoIndex(name="r", root="Z:/no/such/checkout", source="r", files=files)
    overview = section(render_map(index, build_graph(index)), "## Overview").strip()
    assert overview.startswith(expected + ": "), overview


def test_overview_counts_the_repository_root_apart_from_directories() -> None:
    files = [FileRecord(p, "typescript", 1, 1, True) for p in ["index.ts", "src/a.ts", "lib/b.ts"]]
    index = RepoIndex(name="r", root="Z:/no/such/checkout", source="r", files=files)
    assert section(render_map(index, build_graph(index)), "## Overview").strip().startswith(
        "`r` has 3 source files, all TypeScript, in 2 top-level directories and the repository root: "
    )
    only_root = RepoIndex(name="r", root="Z:/no/such/checkout", source="r", files=files[:1])
    assert section(render_map(only_root, build_graph(only_root)), "## Overview").strip() == (
        "`r` has 1 source file, all TypeScript, in the repository root: the repository root (1 file). "
        "0 files are tests."
    )

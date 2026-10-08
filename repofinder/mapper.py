"""Deterministic Markdown architecture map of a repository.

:func:`render_map` combines the index, the import graph and (optionally) per-file
summaries written by the LLM layer into one Markdown document. Rendering never
touches the network or the checkout and always yields the same text for the
same inputs.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from .graph import TEST_DIR_NAMES, ModuleGraph
from .models import RepoIndex

SUMMARY_MAX_CHARS = 80
HOT_SPOTS = 10
MAX_OVERVIEW_PACKAGES = 12
MAX_START_ENTRIES = 5
START_CORE_MODULES = 3
LAYOUT_MAX_DEPTH = 6

_KIND_ORDER = ("class", "function", "method")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def render_map(
    index: RepoIndex,
    graph: ModuleGraph,
    summaries: Optional[dict[str, str]] = None,
    repo_summary: Optional[str] = None,
    max_modules: int = 40,
) -> str:
    """Render the architecture map of ``index`` as Markdown (see module docstring)."""
    summaries = summaries or {}
    sections = [
        _header(index),
        _overview(index, graph, repo_summary),
        _entry_points(graph),
        _core_modules(index, graph, summaries, max_modules),
        _package_layout(index),
        _hot_spots(graph),
        _where_to_start(index, graph),
    ]
    return "\n\n".join(section.rstrip("\n") for section in sections) + "\n"


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


def _header(index: RepoIndex) -> str:
    kinds = Counter(s.kind for s in index.symbols)
    ordered = [k for k in _KIND_ORDER if kinds.get(k)] + sorted(k for k in kinds if k not in _KIND_ORDER)
    by_kind = ", ".join(_plural(kinds[k], k, "classes" if k == "class" else None) for k in ordered)
    lines = sum(f.n_lines for f in index.files)
    failures = sum(1 for f in index.files if not f.parse_ok)
    symbols = _plural(len(index.symbols), "symbol") + (f" ({by_kind})" if by_kind else "")
    stats = (
        f"{_plural(len(index.files), 'file')} · {_plural(lines, 'line')} · {symbols}"
        f" · {_plural(failures, 'parse failure')}"
    )
    return f"# {index.name} — architecture map\n\n{stats}"


def _overview(index: RepoIndex, graph: ModuleGraph, repo_summary: Optional[str]) -> str:
    if repo_summary is not None:
        return f"## Overview\n\n{repo_summary.strip()}"
    packages = graph.packages()
    if not packages:
        return f"## Overview\n\n`{index.name}` has no indexed source files."
    ranked = sorted(packages.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    shown = ranked[:MAX_OVERVIEW_PACKAGES]
    parts = [
        f"{'the repository root' if key == '.' else f'`{key}`'} ({_plural(len(files), 'file')})"
        for key, files in shown
    ]
    if len(ranked) > len(shown):
        rest = sum(len(files) for _key, files in ranked[len(shown):])
        parts.append(f"{len(ranked) - len(shown)} more ({_plural(rest, 'file')})")
    n_tests = sum(1 for node in graph.nodes.values() if node.is_test)
    sentence = (
        f"`{index.name}` has {_plural(len(graph.nodes), 'source file')} in "
        f"{_plural(len(packages), 'top-level package or directory', 'top-level packages and directories')}: "
        f"{_join(parts)}. "
        f"{_plural(n_tests, 'file')} {'is' if n_tests == 1 else 'are'} tests."
    )
    return f"## Overview\n\n{sentence}"


def _entry_points(graph: ModuleGraph) -> str:
    entries = graph.entry_points()
    if not entries:
        return "## Entry points\n\n_No entry points detected._"
    rows = [f"| `{node.path}` | {_cell(node.entry_reason or '')} |" for node in entries]
    return "## Entry points\n\n| path | reason |\n|---|---|\n" + "\n".join(rows)


def _core_modules(
    index: RepoIndex, graph: ModuleGraph, summaries: dict[str, str], max_modules: int
) -> str:
    nodes = graph.top_by_fan_in(max_modules, include_tests=False)
    if not nodes:
        return "## Core modules\n\n_No non-test source files._"
    top_names = _top_symbol_names(index)
    rows = []
    for node in nodes:
        summary = summaries.get(node.path)
        if summary and summary.strip():
            text = _first_sentence(summary)
            text = _truncate(text, SUMMARY_MAX_CHARS)
        else:
            text = _code_list(top_names.get(node.path, []), SUMMARY_MAX_CHARS) or "(no classes or functions)"
        module = f"`{node.module}`" if node.module else ""
        rows.append(
            f"| `{node.path}` | {module} | {node.fan_in} | {node.fan_out} | {node.n_symbols} "
            f"| {_cell(text)} |"
        )
    intro = (
        f"Top {len(nodes)} non-test files by fan-in "
        "(distinct in-repo files importing them)."
    )
    header = "| path | module | fan-in | fan-out | symbols | summary |\n|---|---|---:|---:|---:|---|"
    return f"## Core modules\n\n{intro}\n\n{header}\n" + "\n".join(rows)


def _package_layout(index: RepoIndex) -> str:
    tree = _DirNode("")
    for record in index.files:
        parts = record.path.split("/")[:-1]
        node = tree
        for part in parts:
            if node.collapsed:
                break
            node = node.child(part)
        node.files += 1
    lines: list[str] = []
    if tree.files:
        lines.append(f"- `./` — {_plural(tree.files, 'file')}")
    for child in tree.sorted_children():
        _render_dir(child, "", 0, lines)
    if not lines:
        return "## Package layout\n\n_No source files._"
    return "## Package layout\n\n" + "\n".join(lines)


def _hot_spots(graph: ModuleGraph) -> str:
    nodes = graph.nodes
    edges = [
        (src, dst)
        for src, dst in graph.edges
        if not nodes[src].is_test and not nodes[dst].is_test
    ]
    if not edges:
        return "## Dependency hot spots\n\n_No imports between non-test files._"
    edges.sort(key=lambda e: (-(nodes[e[0]].fan_in + nodes[e[1]].fan_in), e[0], e[1]))
    lines = [
        f"- `{src}` → `{dst}` (fan-in {nodes[src].fan_in} + {nodes[dst].fan_in})"
        for src, dst in edges[:HOT_SPOTS]
    ]
    intro = "Imports between non-test files, ranked by the combined fan-in of both ends."
    return f"## Dependency hot spots\n\n{intro}\n\n" + "\n".join(lines)


def _where_to_start(index: RepoIndex, graph: ModuleGraph) -> str:
    lines: list[str] = []
    listed: set[str] = set()
    top_names = _top_symbol_names(index)

    entries = graph.entry_points()
    for node in entries[:MAX_START_ENTRIES]:
        reason = (node.entry_reason or "").split("; ", 1)[0]
        lines.append(f"- `{node.path}` — entry point: {reason}.")
        listed.add(node.path)
    if len(entries) > MAX_START_ENTRIES:
        lines.append(f"- … and {len(entries) - MAX_START_ENTRIES} more entry points (see above).")

    core = [n for n in graph.top_by_fan_in(len(graph.nodes)) if n.path not in listed and n.fan_in > 0]
    for node in core[:START_CORE_MODULES]:
        defines = _code_list(top_names.get(node.path, []), SUMMARY_MAX_CHARS)
        tail = f"; defines {defines}" if defines else ""
        lines.append(f"- `{node.path}` — imported by {_plural(node.fan_in, 'file')}{tail}.")
        listed.add(node.path)

    test_dir = _main_test_dir(graph)
    if test_dir is not None:
        directory, count = test_dir
        lines.append(f"- `{directory}/` — test suite, {_plural(count, 'file')}.")

    if not lines:
        return "## Where to start\n\n_Nothing to suggest: no entry points, imports or tests found._"
    return "## Where to start\n\n" + "\n".join(lines)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _DirNode:
    """A directory in the package-layout tree. Test directories absorb their subtree."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.files = 0
        self.children: dict[str, _DirNode] = {}
        self.collapsed = name in TEST_DIR_NAMES

    def child(self, name: str) -> "_DirNode":
        node = self.children.get(name)
        if node is None:
            node = self.children[name] = _DirNode(name)
        return node

    def sorted_children(self) -> list["_DirNode"]:
        return [self.children[name] for name in sorted(self.children)]

    def total(self) -> int:
        return self.files + sum(child.total() for child in self.children.values())

    def n_dirs(self) -> int:
        return len(self.children) + sum(child.n_dirs() for child in self.children.values())


def _render_dir(node: _DirNode, indent: str, depth: int, out: list[str]) -> None:
    label = node.name
    # Fold "src/" -> "src/flask/" style chains of file-less single-child directories.
    while not node.collapsed and node.files == 0 and len(node.children) == 1:
        (only,) = node.children.values()
        label = f"{label}/{only.name}"
        node = only
    total = node.total()
    if node.collapsed:
        out.append(f"{indent}- `{label}/` — {_plural(total, 'test file')} (collapsed)")
        return
    if node.files and total != node.files:
        count = f"{_plural(node.files, 'file')}, {total} in subtree"
    elif node.files:
        count = _plural(node.files, "file")
    else:
        count = f"{total} files in subtree"
    if depth + 1 >= LAYOUT_MAX_DEPTH and node.children:
        out.append(f"{indent}- `{label}/` — {count} ({_plural(node.n_dirs(), 'subdirectory', 'subdirectories')} not shown)")
        return
    out.append(f"{indent}- `{label}/` — {count}")
    for child in node.sorted_children():
        _render_dir(child, indent + "  ", depth + 1, out)


def _main_test_dir(graph: ModuleGraph) -> Optional[tuple[str, int]]:
    """The outermost ``tests``/``test`` directory with the most files (ties: path)."""
    counts: Counter[str] = Counter()
    for path in graph.nodes:
        parts = path.split("/")[:-1]
        for i, part in enumerate(parts):
            if part in TEST_DIR_NAMES:
                counts["/".join(parts[: i + 1])] += 1
                break
    if not counts:
        return None
    directory = min(counts, key=lambda d: (-counts[d], d.count("/"), d))
    return directory, counts[directory]


def _top_symbol_names(index: RepoIndex) -> dict[str, list[str]]:
    """Up to three module-level class/function names per file: public first, then by size."""
    candidates: dict[str, list[tuple[bool, int, int, str]]] = {}
    for sym in index.symbols:
        if sym.parent is None and sym.kind in ("class", "function"):
            candidates.setdefault(sym.path, []).append(
                (sym.name.startswith("_"), -(sym.end_line - sym.start_line), sym.start_line, sym.name)
            )
    out: dict[str, list[str]] = {}
    for path, items in candidates.items():
        names: list[str] = []
        for item in sorted(items):
            if item[3] not in names:
                names.append(item[3])
            if len(names) == 3:
                break
        out[path] = names
    return out


def _first_sentence(text: str) -> str:
    flat = " ".join(text.split())
    return _SENTENCE_END_RE.split(flat, maxsplit=1)[0]


def _truncate(text: str, limit: int) -> str:
    """Collapse whitespace and cut to ``limit`` characters, closing an open code span."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rstrip()
    if cut.count("`") % 2:
        cut += "`"
    return cut + "…"


def _code_list(names: list[str], limit: int) -> str:
    """``a``, ``b``, ... as code spans, keeping only whole names within ``limit`` characters."""
    out = ""
    for name in names:
        piece = f"`{name}`"
        candidate = f"{out}, {piece}" if out else piece
        if len(candidate) > limit:
            return out + ", …" if out else _truncate(piece, limit)
        out = candidate
    return out


def _cell(text: str) -> str:
    """Make ``text`` safe inside a Markdown table cell."""
    return " ".join(text.split()).replace("|", "\\|")


def _plural(n: int, word: str, plural: Optional[str] = None) -> str:
    return f"{n:,} {word if n == 1 else (plural or word + 's')}"


def _join(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


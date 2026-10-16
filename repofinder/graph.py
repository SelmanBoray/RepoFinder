"""File-level import graph and entry-point detection.

:func:`build_graph` turns a :class:`RepoIndex` into a :class:`ModuleGraph`: one
:class:`ModuleNode` per indexed file plus the deduplicated in-repo import edges
between files. Fan-in/fan-out count *distinct* files, so ``import a`` and
``from a import b`` in the same file are one edge.

Entry points are found with cheap heuristics and ranked by a confidence score
(see :func:`find_entry_points`). Only a few small files are read from
``index.root`` (packaging metadata and candidate sources); nothing from the
checkout is executed or imported.

Module names follow the language: Python files get their dotted import name
(``parser.build_module_map``); TypeScript and JavaScript files are named by
their path without the extension, and an ``index`` file names its directory
(see :func:`path_module_name`).
"""

from __future__ import annotations

import json
import posixpath
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from . import parser
from .models import FileRecord, RepoIndex
from .parser import build_module_map

# File names that conventionally start an application or a command. A file in
# a ``bin/`` directory counts too (see :func:`entry_point_reasons`).
ENTRY_FILENAMES = (
    "cli.py", "main.py", "app.py", "server.py", "wsgi.py", "asgi.py", "manage.py",
    "index.ts", "main.ts", "cli.ts", "server.ts", "app.ts", "index.mts",
)
BIN_DIR_NAME = "bin"
TEST_DIR_NAMES = frozenset({"tests", "test", "__tests__"})

PATH_MODULE_EXTENSIONS = (".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs")
"""Extensions of the languages whose modules are named by path (TypeScript, JavaScript)."""

KNOWN_EXTENSIONS = (".py", *PATH_MODULE_EXTENSIONS)
"""Source extensions known without asking the parser; :func:`source_extensions`
adds whatever ``parser.LANGUAGES`` registers on top."""

_DECLARATION_MARK = ".d"  # "types.d.ts" declares the module "types"
_INDEX_BASENAME = "index"  # "pkg/index.ts" is the module "pkg", like "pkg/__init__.py"

# "foo.test.ts", "foo.spec.tsx", "foo.behavior.test.mts" (JavaScript too).
_JS_TEST_FILE_RE = re.compile(r"\.(?:test|spec)\.(?:ts|tsx|mts|cts|js|jsx|mjs|cjs)$")
# package.json "main"/"bin" pointing at compiled output: source extensions to try instead.
_COMPILED_TO_SOURCE = {".js": (".ts", ".tsx"), ".jsx": (".tsx",), ".mjs": (".mts",), ".cjs": (".cts",)}
_BUILD_OUTPUT_DIRS = ("dist", "build", "out", "lib")  # tried as "src/" when the path itself is not indexed

MAX_READ_BYTES = 1024 * 1024  # never read more than this from a checkout file

# Reason strings. They are Markdown-friendly: code is wrapped in backticks.
REASON_MAIN_FILE = "`__main__.py` (run with `python -m`)"
REASON_MAIN_GUARD = 'has `if __name__ == "__main__"`'
REASON_FILENAME_PREFIX = "conventional name "
REASON_UNINDEXED_BIN = "`{source}` bin (not indexed{language})"  # language: ": JavaScript" or ""

# Entry-point confidence scores (see :func:`find_entry_points`). A file scores its
# strongest reason, minus PERIPHERAL_PENALTY once if it lives in a peripheral directory.
SCORE_DECLARED = 100            # packaging metadata: package.json "bin"/"main", pyproject/setup scripts
SCORE_NAME_AT_SOURCE_ROOT = 80  # conventional name at the repository root or directly in a "src/" dir
SCORE_MAIN_FILE = 70            # __main__.py
SCORE_NAME_NESTED = 60          # conventional name in any other directory
SCORE_MAIN_GUARD = 40           # module-level if __name__ == "__main__"
PERIPHERAL_PENALTY = 30
PERIPHERAL_DIRS = ("experiments", "examples", "scripts/fixtures", "benchmarks", "docs", "website")
"""Directories (matched at any depth) whose entry points are demos, fixtures or tooling."""
SOURCE_DIR_NAME = "src"
_PERIPHERAL_PARTS = tuple(tuple(d.split("/")) for d in PERIPHERAL_DIRS)

# Display names of FileRecord.language values (TSX is TypeScript).
LANGUAGE_NAMES = {"python": "Python", "typescript": "TypeScript", "tsx": "TypeScript", "javascript": "JavaScript"}
DIRECTORY_PACKAGE_LANGUAGES = frozenset({"TypeScript", "JavaScript"})
"""Languages (display names) whose modules are grouped by directory rather than importable package."""

# Language of a package.json "bin" target that is not indexed, by extension or shebang interpreter.
_UNINDEXED_LANGUAGES = {
    **{ext: "JavaScript" for ext in (".js", ".mjs", ".cjs", ".jsx")},
    **{ext: "TypeScript" for ext in (".ts", ".mts", ".cts", ".tsx")},
    ".py": "Python",
    ".sh": "shell",
}
_SHEBANG_LANGUAGES = (("node", "JavaScript"), ("deno", "JavaScript"), ("bun", "JavaScript"),
                      ("python", "Python"), ("bash", "shell"), ("sh", "shell"))

_MAIN_GUARD_RE = re.compile(
    rb"^if\s+(?:__name__\s*==\s*['\"]__main__['\"]|['\"]__main__['\"]\s*==\s*__name__)\s*:",
    re.MULTILINE,
)
# setup.cfg line "name = pkg.mod:func [extra]" (":func" optional).
_ENTRY_SPEC_RE = re.compile(
    r"^\s*([\w.\-]+)\s*=\s*([A-Za-z_][\w.]*)(?:\s*:\s*([A-Za-z_][\w.]*))?\s*(?:\[.*\])?\s*$"
)
_SETUP_PY_SPEC_RE = re.compile(r"""["']\s*([\w.\-]+)\s*=\s*([A-Za-z_][\w.]*)\s*:\s*([A-Za-z_][\w.]*)[^"'\n]*["']""")
_TOML_SECTION_RE = re.compile(r"^\s*\[([^\[\]]+)\]\s*(?:#.*)?$")
_TOML_ENTRY_RE = re.compile(
    r"""^\s*["']?([\w.\-]+)["']?\s*=\s*["']\s*([A-Za-z_][\w.]*)(?:\s*:\s*([A-Za-z_][\w.]*))?[^"']*["']"""
)


def source_extensions() -> tuple[str, ...]:
    """:data:`KNOWN_EXTENSIONS` plus every extension registered in ``parser.LANGUAGES``."""
    exts = list(KNOWN_EXTENSIONS)
    for spec in parser.LANGUAGES.values():
        exts.extend(e for e in spec.extensions if e not in exts)
    return tuple(exts)


def split_source_extension(path: str) -> tuple[str, str]:
    """``(path without extension, extension)`` for a known source extension.

    ``pkg/mod.py`` -> ``("pkg/mod", ".py")``; ``src/types.d.ts`` ->
    ``("src/types", ".d.ts")`` (a declaration file declares the module it is
    named after); an unknown extension gives ``(path, "")``.
    """
    base = path.rsplit("/", 1)[-1]
    dot = base.rfind(".")
    if dot <= 0:
        return path, ""
    ext = base[dot:]
    if ext not in source_extensions():
        return path, ""
    stem = path[: len(path) - len(ext)]
    if ext in PATH_MODULE_EXTENSIONS and base[:dot].endswith(_DECLARATION_MARK) and dot > len(_DECLARATION_MARK):
        return stem[: -len(_DECLARATION_MARK)], _DECLARATION_MARK + ext
    return stem, ext


def path_module_name(path: str) -> Optional[str]:
    """Module name of a TypeScript/JavaScript file: its path without the extension.

    ``src/util/strings.ts`` -> ``src/util/strings``; ``src/util/index.ts`` ->
    ``src/util`` (an ``index`` file names its directory, as ``__init__.py``
    does); ``index.ts`` at the repository root stays ``index``. None for files
    of other languages.
    """
    stem, ext = split_source_extension(path)
    if not ext.endswith(PATH_MODULE_EXTENSIONS):
        return None
    head, _, base = stem.rpartition("/")
    if base == _INDEX_BASENAME and head:
        return head
    return stem


def is_test_path(path: str) -> bool:
    """True for test code: under a ``tests/``, ``test/`` or ``__tests__/`` dir,
    ``test_*.py``, ``*_test.py``, ``conftest.py``, and TypeScript/JavaScript
    ``*.test.ts``/``*.spec.ts`` (any of their extensions: ``.tsx``, ``.mts``, ``.js``, ...)."""
    parts = path.split("/")
    base = parts[-1]
    if any(part in TEST_DIR_NAMES for part in parts[:-1]):
        return True
    return (
        base.startswith("test_")
        or base.endswith("_test.py")
        or base == "conftest.py"
        or _JS_TEST_FILE_RE.search(base) is not None
    )


@dataclass(frozen=True)
class EntryPoint:
    path: str
    reason: str           # every reason, "; "-joined, in heuristic order (see find_entry_points)
    score: int            # confidence (SCORE_* minus PERIPHERAL_PENALTY); higher is stronger
    indexed: bool = True  # False: a package.json "bin" file that exists but is not in the index


@dataclass
class ModuleNode:
    path: str
    module: Optional[str]      # Python: dotted name via parser.build_module_map; TS/JS: path_module_name; else None
    n_symbols: int
    n_lines: int
    fan_in: int                # distinct in-repo files importing this file
    fan_out: int               # distinct in-repo files this file imports
    is_entry: bool
    is_test: bool
    entry_reason: Optional[str] = None   # why the file counts as an entry point ("; "-joined)
    entry_score: int = 0                 # EntryPoint.score when is_entry


@dataclass
class ModuleGraph:
    nodes: dict[str, ModuleNode]              # keyed by path
    edges: list[tuple[str, str]]              # (src_path, dst_path), deduplicated, sorted
    unindexed_entries: list[EntryPoint] = field(default_factory=list)  # declared, on disk, not indexed
    _importers: dict[str, list[str]] = field(init=False, repr=False, compare=False)
    _imports: dict[str, list[str]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        importers: dict[str, list[str]] = {}
        imports: dict[str, list[str]] = {}
        for src, dst in self.edges:
            importers.setdefault(dst, []).append(src)
            imports.setdefault(src, []).append(dst)
        self._importers = {k: sorted(set(v)) for k, v in importers.items()}
        self._imports = {k: sorted(set(v)) for k, v in imports.items()}

    def importers_of(self, path: str) -> list[str]:
        """Files that import ``path`` (sorted)."""
        return list(self._importers.get(path, []))

    def imports_of(self, path: str) -> list[str]:
        """In-repo files imported by ``path`` (sorted)."""
        return list(self._imports.get(path, []))

    def top_by_fan_in(self, n: int = 10, include_tests: bool = False) -> list[ModuleNode]:
        """The ``n`` most imported files; ties broken by symbol count, then path."""
        nodes = [node for node in self.nodes.values() if include_tests or not node.is_test]
        nodes.sort(key=lambda node: (-node.fan_in, -node.n_symbols, node.path))
        return nodes[: max(n, 0)]

    def entry_points(self) -> list[ModuleNode]:
        """Indexed entry-point files, highest score first, then by path."""
        nodes = [node for node in self.nodes.values() if node.is_entry]
        nodes.sort(key=lambda node: (-node.entry_score, node.path))
        return nodes

    def all_entry_points(self) -> list[EntryPoint]:
        """Every entry point, indexed or not (:attr:`unindexed_entries`), highest score first, then by path."""
        entries = [
            EntryPoint(node.path, node.entry_reason or "", node.entry_score)
            for node in self.nodes.values()
            if node.is_entry
        ]
        entries.extend(self.unindexed_entries)
        entries.sort(key=lambda entry: (-entry.score, entry.path))
        return entries

    def top_level_directories(self) -> dict[str, list[str]]:
        """First path component -> sorted file paths (``"."`` for files at the repository root)."""
        groups: dict[str, list[str]] = {}
        for path in self.nodes:
            head, sep, _rest = path.partition("/")
            groups.setdefault(head if sep else ".", []).append(path)
        return {key: sorted(groups[key]) for key in sorted(groups)}

    def packages(self) -> dict[str, list[str]]:
        """Top-level package directory -> sorted file paths.

        The key is the directory of the outermost package as imported (``src/flask``
        for ``src/flask/json/tag.py``, ``tests`` for ``tests/test_x.py``). Files
        that are top-level modules of their import root map to that root
        (``"."`` for the repository root). A TypeScript/JavaScript file maps to
        its own directory. Unresolvable files fall back to their first path
        component.
        """
        groups: dict[str, list[str]] = {}
        for node in self.nodes.values():
            groups.setdefault(package_key(node.path, node.module), []).append(node.path)
        return {key: sorted(groups[key]) for key in sorted(groups)}


def package_key(path: str, module: Optional[str]) -> str:
    """Top-level package directory of ``path`` (see :meth:`ModuleGraph.packages`)."""
    if module and path_module_name(path) is not None:
        directory = path.rpartition("/")[0]   # a TS/JS module's package is its directory
        return directory or "."
    if module:
        parts = module.split(".")
        stem = path[:-3] if path.endswith(".py") else path
        is_init = stem == "__init__" or stem.endswith("/__init__")
        if is_init:
            stem = stem[: -len("__init__")].rstrip("/")
        rel = "/".join(parts)
        if stem == rel or stem.endswith("/" + rel):
            prefix = stem[: len(stem) - len(rel)]  # import root, "" or "src/" etc.
            if len(parts) > 1 or is_init:
                return prefix + parts[0]
            return prefix.rstrip("/") or "."
    pieces = path.split("/")
    return pieces[0] if len(pieces) > 1 else "."


def module_names(paths: Iterable[str]) -> dict[str, str]:
    """Path -> preferred module name.

    Python: ``parser.build_module_map`` can give one file several names
    (``src.flask.app`` from the repo root and ``flask.app`` from ``src/``); the
    shortest one is how the code is actually imported, so it wins (ties:
    alphabetical). TypeScript/JavaScript: :func:`path_module_name`.
    """
    ordered = sorted(paths)
    best: dict[str, str] = {}
    for name, path in build_module_map(ordered).items():
        current = best.get(path)
        if current is None or (name.count("."), name) < (current.count("."), current):
            best[path] = name
    for path in ordered:
        if path not in best:
            name = path_module_name(path)
            if name is not None:
                best[path] = name
    return best


def language_mix(files: Iterable[FileRecord]) -> list[tuple[str, int]]:
    """(language display name, indexed lines) over ``files``, largest first (ties: name).

    Languages are named by :data:`LANGUAGE_NAMES` (``tsx`` counts as TypeScript).
    File counts are used instead when no file has any lines.
    """
    records = list(files)
    use_lines = any(record.n_lines > 0 for record in records)
    weights: Counter[str] = Counter()
    for record in records:
        weights[LANGUAGE_NAMES.get(record.language, record.language)] += record.n_lines if use_lines else 1
    return sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))


def build_graph(index: RepoIndex) -> ModuleGraph:
    """Build the file-level import graph of ``index``. Self-imports are dropped."""
    paths = sorted(f.path for f in index.files)
    path_set = set(paths)
    modules = module_names(paths)
    n_symbols = Counter(s.path for s in index.symbols)
    edges = sorted(
        {
            (e.src_path, e.dst_path)
            for e in index.imports
            if e.dst_path is not None
            and e.dst_path != e.src_path
            and e.src_path in path_set
            and e.dst_path in path_set
        }
    )
    fan_in = Counter(dst for _src, dst in edges)
    fan_out = Counter(src for src, _dst in edges)
    found = find_entry_points(index)
    entries = {entry.path: entry for entry in found if entry.indexed}
    by_path = {f.path: f for f in index.files}

    nodes: dict[str, ModuleNode] = {}
    for path in paths:
        record = by_path[path]
        entry = entries.get(path)
        nodes[path] = ModuleNode(
            path=path,
            module=modules.get(path),
            n_symbols=n_symbols.get(path, 0),
            n_lines=record.n_lines,
            fan_in=fan_in.get(path, 0),
            fan_out=fan_out.get(path, 0),
            is_entry=entry is not None,
            is_test=is_test_path(path),
            entry_reason=entry.reason if entry else None,
            entry_score=entry.score if entry else 0,
        )
    return ModuleGraph(nodes=nodes, edges=edges, unindexed_entries=[e for e in found if not e.indexed])


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def detect_entry_points(index: RepoIndex) -> set[str]:
    """Paths of the indexed files that look like entry points (see :func:`find_entry_points`)."""
    return set(entry_point_reasons(index))


def entry_point_reasons(index: RepoIndex) -> dict[str, str]:
    """Indexed entry-point path -> its reasons, ``"; "``-joined (see :func:`find_entry_points`)."""
    return {entry.path: entry.reason for entry in sorted(find_entry_points(index), key=lambda e: e.path)
            if entry.indexed}


def find_entry_points(index: RepoIndex) -> list[EntryPoint]:
    """Every entry point of ``index``, highest score first, then by path.

    Heuristics, with their scores (a file scores its strongest reason; its
    reasons are kept in this order):

    1. ``SCORE_DECLARED``: targets of ``[project.scripts]``,
       ``[project.gui-scripts]``, ``[project.entry-points.*]`` and
       ``[tool.poetry.scripts]`` in any ``pyproject.toml``, of ``entry_points=``
       in ``setup.py`` and of ``[options.entry_points]`` in ``setup.cfg``
       (``pkg.mod:func`` is resolved to a file relative to the project
       directory that declares it), and the ``"bin"`` and ``"main"`` files of
       any ``package.json`` (compiled output such as ``dist/cli.js`` is traced
       back to its indexed source, ``dist/cli.ts`` or ``src/cli.ts``). A
       ``"bin"`` file that exists on disk but is not indexed (``scripts/run.mjs``
       while JavaScript is not parsed) is kept with ``indexed=False``;
    2. ``SCORE_MAIN_FILE``: ``__main__.py``;
    3. ``SCORE_MAIN_GUARD``: a module-level ``if __name__ == "__main__":`` (Python files);
    4. a conventional file name (:data:`ENTRY_FILENAMES`) or a file directly in
       a ``bin/`` directory: ``SCORE_NAME_AT_SOURCE_ROOT`` when the file (or
       its ``bin/``) sits at the repository root or directly in a ``src/``
       directory, ``SCORE_NAME_NESTED`` deeper.

    Files under a :data:`PERIPHERAL_DIRS` directory lose ``PERIPHERAL_PENALTY``.
    Test files are never entry points. Files that cannot be read are skipped silently.
    """
    paths = sorted(f.path for f in index.files)
    root = Path(index.root) if index.root else None
    reasons: dict[str, list[str]] = {}
    scores: dict[str, int] = {}
    unindexed: set[str] = set()

    def add(path: str, reason: str, score: int, indexed: bool = True) -> None:
        if is_test_path(path):
            return
        bucket = reasons.setdefault(path, [])
        if reason not in bucket:
            bucket.append(reason)
        scores[path] = max(scores.get(path, score), score)
        if not indexed:
            unindexed.add(path)

    if root is not None:
        for target_path, reason, indexed in _declared_entry_points(root, paths):
            add(target_path, reason, SCORE_DECLARED, indexed)

    candidates = [p for p in paths if not is_test_path(p)]
    py_candidates = [p for p in candidates if p.endswith(".py")]
    for path in py_candidates:
        if path.rsplit("/", 1)[-1] == "__main__.py":
            add(path, REASON_MAIN_FILE, SCORE_MAIN_FILE)
    if root is not None:
        for path in py_candidates:
            data = _read_bytes(root, path)
            if data and b"__main__" in data and _MAIN_GUARD_RE.search(data):
                add(path, REASON_MAIN_GUARD, SCORE_MAIN_GUARD)
    for path in candidates:
        parts = path.split("/")
        if parts[-1] in ENTRY_FILENAMES:
            add(path, f"{REASON_FILENAME_PREFIX}`{parts[-1]}`", _name_score(parts[:-1]))
        elif len(parts) > 1 and parts[-2] == BIN_DIR_NAME:
            add(path, f"{REASON_FILENAME_PREFIX}`{BIN_DIR_NAME}/{parts[-1]}`", _name_score(parts[:-2]))

    entries = [
        EntryPoint(path, "; ".join(reasons[path]), scores[path] - _peripheral_penalty(path), path not in unindexed)
        for path in reasons
    ]
    entries.sort(key=lambda entry: (-entry.score, entry.path))
    return entries


def _name_score(directory: list[str]) -> int:
    """Score of a conventional file name found in ``directory`` (path components)."""
    at_source_root = not directory or directory[-1] == SOURCE_DIR_NAME
    return SCORE_NAME_AT_SOURCE_ROOT if at_source_root else SCORE_NAME_NESTED


def _peripheral_penalty(path: str) -> int:
    """PERIPHERAL_PENALTY when a directory of ``path`` matches :data:`PERIPHERAL_DIRS`, else 0."""
    directory = path.split("/")[:-1]
    for i in range(len(directory)):
        for marker in _PERIPHERAL_PARTS:
            if tuple(directory[i:i + len(marker)]) == marker:
                return PERIPHERAL_PENALTY
    return 0


def _read_bytes(root: Path, rel: str, limit: int = MAX_READ_BYTES) -> Optional[bytes]:
    try:
        with open(root / rel, "rb") as handle:
            return handle.read(limit)
    except OSError:
        return None


def _read_text(root: Path, rel: str) -> Optional[str]:
    data = _read_bytes(root, rel)
    return None if data is None else data.decode("utf-8", errors="replace")


def _project_dirs(root: Path, paths: list[str]) -> list[tuple[str, list[str]]]:
    """(project dir, packaging files present) for the root and every indexed directory."""
    path_set = set(paths)
    dirs = {""}
    for path in paths:
        parts = path.split("/")[:-1]
        for n in range(1, len(parts) + 1):
            dirs.add("/".join(parts[:n]))
    found: list[tuple[str, list[str]]] = []
    for directory in sorted(dirs):
        prefix = f"{directory}/" if directory else ""
        present = []
        for name in ("pyproject.toml", "setup.cfg"):
            if (root / (prefix + name)).is_file():
                present.append(name)
        if prefix + "setup.py" in path_set:
            present.append("setup.py")
        if (root / (prefix + "package.json")).is_file():
            present.append("package.json")
        if present:
            found.append((directory, present))
    return found


def _declared_entry_points(root: Path, paths: list[str]) -> list[tuple[str, str, bool]]:
    """(target file, reason, indexed) for every packaging-declared entry point that resolves
    in the repo; only package.json "bin" files can be unindexed (they must exist on disk)."""
    out: list[tuple[str, str, bool]] = []
    path_set = set(paths)
    for directory, present in _project_dirs(root, paths):
        prefix = f"{directory}/" if directory else ""
        resolver = _ModuleResolver(prefix, paths)
        for filename in present:
            text = _read_text(root, prefix + filename)
            if text is None:
                continue
            if filename == "package.json":
                out.extend(_package_json_entries(text, prefix, path_set, root))
                continue
            if filename == "pyproject.toml":
                entries = _pyproject_entries(text)
            elif filename == "setup.cfg":
                entries = _setup_cfg_entries(text)
            else:
                entries = _setup_py_entries(text)
            for label, name, spec, where in entries:
                target = resolver.resolve(spec.split(":", 1)[0])
                if target is not None:
                    source = prefix + filename + (f" [{where}]" if where else "")
                    out.append((target, f"{label} `{name}` = `{spec}` ({source})", True))
    return out


class _ModuleResolver:
    """Resolve dotted modules of the project rooted at ``prefix`` to repo paths."""

    def __init__(self, prefix: str, paths: list[str]) -> None:
        self.prefix = prefix
        self._paths = paths
        self._map: Optional[dict[str, str]] = None

    def resolve(self, module: str) -> Optional[str]:
        if self._map is None:
            n = len(self.prefix)
            rel = [p[n:] for p in self._paths if p.startswith(self.prefix)]
            self._map = {name: self.prefix + path for name, path in build_module_map(rel).items()}
        return self._map.get(module)


def _package_json_entries(
    text: str, prefix: str, path_set: set[str], root: Path
) -> list[tuple[str, str, bool]]:
    """(target file, reason, indexed) for the ``"bin"`` and ``"main"`` entries of
    a package.json in directory ``prefix`` whose file is indexed, plus the
    ``"bin"`` files that exist under ``root`` but are not indexed. Anything
    malformed is ignored."""
    try:
        data = json.loads(text)
    except ValueError:
        return []
    if not isinstance(data, dict):
        return []
    source = prefix + "package.json"
    declared: list[tuple[str, str, bool]] = []   # (reason, declared path, is "bin")
    bin_field = data.get("bin")
    if isinstance(bin_field, str):
        package = data.get("name")
        command = package.rsplit("/", 1)[-1] if isinstance(package, str) and package else "bin"
        declared.append((f"script `{command}` = `{bin_field}` ({source} [bin])", bin_field, True))
    elif isinstance(bin_field, dict):
        for command in sorted(bin_field, key=str):
            value = bin_field[command]
            if isinstance(value, str):
                declared.append((f"script `{command}` = `{value}` ({source} [bin])", value, True))
    main = data.get("main")
    if isinstance(main, str):
        declared.append((f"main `{main}` ({source})", main, False))
    out: list[tuple[str, str, bool]] = []
    for reason, value, is_bin in declared:
        target = _package_target(prefix, value, path_set)
        if target is not None:
            out.append((target, reason, True))
            continue
        path = _package_path(prefix, value) if is_bin else None
        if path is not None and (root / path).is_file():
            language = _unindexed_language(root, path)
            out.append((path, REASON_UNINDEXED_BIN.format(
                source=source, language=f": {language}" if language else ""), False))
    return out


def _package_path(prefix: str, value: str) -> Optional[str]:
    """Repository path of a package.json path value, or None if it is empty or leaves the repository."""
    value = value.strip().replace("\\", "/")
    if not value or value.startswith("/") or ":" in value:
        return None
    path = posixpath.normpath(prefix + value)
    if path in (".", "..") or path.startswith("../"):
        return None
    return path


def _unindexed_language(root: Path, path: str) -> Optional[str]:
    """Language of a file that is not indexed: by extension, else by its ``#!`` interpreter."""
    language = _UNINDEXED_LANGUAGES.get(posixpath.splitext(path)[1].lower())
    if language is not None:
        return language
    head = _read_bytes(root, path, 256) or b""
    if not head.startswith(b"#!"):
        return None
    words = [posixpath.basename(w) for w in head[2:].split(b"\n", 1)[0].decode("utf-8", "replace").split()]
    if words and words[0] == "env":
        words = [w for w in words[1:] if not w.startswith("-")]
    if not words:
        return None
    return next((lang for name, lang in _SHEBANG_LANGUAGES if words[0].startswith(name)), None)


def _package_target(prefix: str, value: str, path_set: set[str]) -> Optional[str]:
    """The indexed file a package.json path names: the path itself, else its
    TypeScript source next to it (``cli.js`` -> ``cli.ts``), else the same
    under ``src/`` when the path is in a build-output directory
    (``dist/cli.js`` -> ``src/cli.ts``). None if none is indexed or the path
    leaves the repository."""
    path = _package_path(prefix, value)
    if path is None:
        return None
    stem, ext = posixpath.splitext(path)
    exts = (ext, *_COMPILED_TO_SOURCE.get(ext.lower(), ()))
    candidates = [stem + e for e in exts]
    rel = stem[len(prefix):] if stem.startswith(prefix) else None
    if rel is not None:
        first, _, rest = rel.partition("/")
        if rest and first in _BUILD_OUTPUT_DIRS:
            candidates += [f"{prefix}src/{rest}{e}" for e in exts]
    return next((c for c in candidates if c in path_set), None)


def _pyproject_entries(text: str) -> list[tuple[str, str, str, str]]:
    """(label, name, "module:attr", section) from a pyproject.toml; regex fallback if TOML fails."""
    try:
        import tomllib  # Python 3.11+
    except ImportError:  # pragma: no cover - Python 3.10
        return _pyproject_entries_regex(text)
    try:
        data = tomllib.loads(text)
    except (tomllib.TOMLDecodeError, ValueError):
        return _pyproject_entries_regex(text)

    out: list[tuple[str, str, str, str]] = []

    def collect(table: object, label: str, section: str) -> None:
        if not isinstance(table, dict):
            return
        for name in sorted(table, key=str):
            value = table[name]
            if isinstance(value, dict):  # poetry: {callable = "pkg.mod:func", ...}
                value = value.get("callable") or value.get("reference")
            if isinstance(value, str):
                spec = value.split("[", 1)[0]  # drop "[extras]"
                module, _, attr = (part.strip() for part in spec.partition(":"))
                if module and all(p.isidentifier() for p in module.split(".")):
                    out.append((label, str(name), f"{module}:{attr}" if attr else module, section))

    project = data.get("project")
    if isinstance(project, dict):
        collect(project.get("scripts"), "script", "project.scripts")
        collect(project.get("gui-scripts"), "gui script", "project.gui-scripts")
        groups = project.get("entry-points")
        if isinstance(groups, dict):
            for group in sorted(groups, key=str):
                collect(groups[group], "entry point", f"project.entry-points.{group}")
    tool = data.get("tool")
    poetry = tool.get("poetry") if isinstance(tool, dict) else None
    if isinstance(poetry, dict):
        collect(poetry.get("scripts"), "script", "tool.poetry.scripts")
    return out


_TOML_SECTIONS = {
    "project.scripts": "script",
    "project.gui-scripts": "gui script",
    "tool.poetry.scripts": "script",
}


def _pyproject_entries_regex(text: str) -> list[tuple[str, str, str, str]]:
    out: list[tuple[str, str, str, str]] = []
    section: Optional[str] = None
    for line in text.splitlines():
        header = _TOML_SECTION_RE.match(line)
        if header:
            section = header.group(1).strip().replace('"', "").replace("'", "")
            continue
        if section is None:
            continue
        if section in _TOML_SECTIONS:
            label = _TOML_SECTIONS[section]
        elif section.startswith("project.entry-points."):
            label = "entry point"
        else:
            continue
        match = _TOML_ENTRY_RE.match(line)
        if match:
            out.append((label, match.group(1), _spec(match.group(2), match.group(3)), section))
    return out


def _setup_cfg_entries(text: str) -> list[tuple[str, str, str, str]]:
    out: list[tuple[str, str, str, str]] = []
    in_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped[1:-1].strip() == "options.entry_points"
            continue
        if in_section:
            match = _ENTRY_SPEC_RE.match(line)
            if match:
                spec = _spec(match.group(2), match.group(3))
                out.append(("entry point", match.group(1), spec, "options.entry_points"))
    return out


def _spec(module: str, attr: Optional[str]) -> str:
    return f"{module}:{attr}" if attr else module


def _setup_py_entries(text: str) -> list[tuple[str, str, str, str]]:
    # String literals are only trusted with an explicit ":attr" (fewer false positives).
    if "entry_points" not in text and "console_scripts" not in text:
        return []
    return [
        ("entry point", m.group(1), f"{m.group(2)}:{m.group(3)}", "")
        for m in _SETUP_PY_SPEC_RE.finditer(text)
    ]

"""File-level import graph and entry-point detection.

:func:`build_graph` turns a :class:`RepoIndex` into a :class:`ModuleGraph`: one
:class:`ModuleNode` per indexed file plus the deduplicated in-repo import edges
between files. Fan-in/fan-out count *distinct* files, so ``import a`` and
``from a import b`` in the same file are one edge.

Entry points are found with cheap heuristics (see :func:`entry_point_reasons`).
Only a few small files are read from ``index.root`` (packaging metadata and
candidate sources); nothing from the checkout is executed or imported.

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
from .models import RepoIndex
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

# Reason strings, from the strongest to the weakest signal. ``entry_rank`` relies on them.
# They are Markdown-friendly: code is wrapped in backticks.
REASON_MAIN_FILE = "`__main__.py` (run with `python -m`)"
REASON_MAIN_GUARD = 'has `if __name__ == "__main__"`'
REASON_FILENAME_PREFIX = "conventional name "

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


@dataclass
class ModuleGraph:
    nodes: dict[str, ModuleNode]              # keyed by path
    edges: list[tuple[str, str]]              # (src_path, dst_path), deduplicated, sorted
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
        """Entry-point files, strongest signal first, then by path."""
        nodes = [node for node in self.nodes.values() if node.is_entry]
        nodes.sort(key=lambda node: (entry_rank(node.entry_reason), node.path))
        return nodes

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
    entries = entry_point_reasons(index)
    by_path = {f.path: f for f in index.files}

    nodes: dict[str, ModuleNode] = {}
    for path in paths:
        record = by_path[path]
        reason = entries.get(path)
        nodes[path] = ModuleNode(
            path=path,
            module=modules.get(path),
            n_symbols=n_symbols.get(path, 0),
            n_lines=record.n_lines,
            fan_in=fan_in.get(path, 0),
            fan_out=fan_out.get(path, 0),
            is_entry=reason is not None,
            is_test=is_test_path(path),
            entry_reason=reason,
        )
    return ModuleGraph(nodes=nodes, edges=edges)


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def detect_entry_points(index: RepoIndex) -> set[str]:
    """Paths of the files that look like entry points (see :func:`entry_point_reasons`)."""
    return set(entry_point_reasons(index))


def entry_rank(reason: Optional[str]) -> int:
    """Strength of an entry reason: 0 declared (packaging metadata), 1 ``__main__.py``,
    2 main guard, 3 file name or ``bin/`` directory."""
    if not reason:
        return 4
    first = reason.split("; ", 1)[0]
    if first == REASON_MAIN_FILE:
        return 1
    if first == REASON_MAIN_GUARD:
        return 2
    if first.startswith(REASON_FILENAME_PREFIX):
        return 3
    return 0


def entry_point_reasons(index: RepoIndex) -> dict[str, str]:
    """Entry-point path -> human-readable reasons, strongest first, ``"; "``-joined.

    Heuristics, in order:

    1. targets of ``[project.scripts]``, ``[project.gui-scripts]``,
       ``[project.entry-points.*]`` and ``[tool.poetry.scripts]`` in any
       ``pyproject.toml``, of ``entry_points=`` in ``setup.py`` and of
       ``[options.entry_points]`` in ``setup.cfg`` (``pkg.mod:func`` is resolved
       to a file relative to the project directory that declares it), and the
       ``"bin"`` and ``"main"`` files of any ``package.json`` that are indexed
       (compiled output such as ``dist/cli.js`` is traced back to its indexed
       source, ``dist/cli.ts`` or ``src/cli.ts``);
    2. ``__main__.py``;
    3. a module-level ``if __name__ == "__main__":`` (Python files);
    4. a conventional file name (:data:`ENTRY_FILENAMES`) or a file directly in
       a ``bin/`` directory.

    Test files only count through heuristic 1. Files that cannot be read are
    skipped silently.
    """
    paths = sorted(f.path for f in index.files)
    root = Path(index.root) if index.root else None
    reasons: dict[str, list[str]] = {}

    def add(path: str, reason: str) -> None:
        bucket = reasons.setdefault(path, [])
        if reason not in bucket:
            bucket.append(reason)

    if root is not None:
        for target_path, reason in _declared_entry_points(root, paths):
            add(target_path, reason)

    candidates = [p for p in paths if not is_test_path(p)]
    py_candidates = [p for p in candidates if p.endswith(".py")]
    for path in py_candidates:
        if path.rsplit("/", 1)[-1] == "__main__.py":
            add(path, REASON_MAIN_FILE)
    if root is not None:
        for path in py_candidates:
            data = _read_bytes(root, path)
            if data and b"__main__" in data and _MAIN_GUARD_RE.search(data):
                add(path, REASON_MAIN_GUARD)
    for path in candidates:
        parts = path.split("/")
        if parts[-1] in ENTRY_FILENAMES:
            add(path, f"{REASON_FILENAME_PREFIX}`{parts[-1]}`")
        elif len(parts) > 1 and parts[-2] == BIN_DIR_NAME:
            add(path, f"{REASON_FILENAME_PREFIX}`{BIN_DIR_NAME}/{parts[-1]}`")

    return {path: "; ".join(reasons[path]) for path in sorted(reasons)}


def _read_bytes(root: Path, rel: str) -> Optional[bytes]:
    try:
        with open(root / rel, "rb") as handle:
            return handle.read(MAX_READ_BYTES)
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


def _declared_entry_points(root: Path, paths: list[str]) -> list[tuple[str, str]]:
    """(target file, reason) for every packaging-declared entry point that resolves in the repo."""
    out: list[tuple[str, str]] = []
    path_set = set(paths)
    for directory, present in _project_dirs(root, paths):
        prefix = f"{directory}/" if directory else ""
        resolver = _ModuleResolver(prefix, paths)
        for filename in present:
            text = _read_text(root, prefix + filename)
            if text is None:
                continue
            if filename == "package.json":
                out.extend(_package_json_entries(text, prefix, path_set))
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
                    out.append((target, f"{label} `{name}` = `{spec}` ({source})"))
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


def _package_json_entries(text: str, prefix: str, path_set: set[str]) -> list[tuple[str, str]]:
    """(target file, reason) for the ``"bin"`` and ``"main"`` entries of a
    package.json in directory ``prefix`` whose file is indexed. Anything
    malformed is ignored."""
    try:
        data = json.loads(text)
    except ValueError:
        return []
    if not isinstance(data, dict):
        return []
    source = prefix + "package.json"
    declared: list[tuple[str, str]] = []   # (reason head, declared path)
    bin_field = data.get("bin")
    if isinstance(bin_field, str):
        package = data.get("name")
        command = package.rsplit("/", 1)[-1] if isinstance(package, str) and package else "bin"
        declared.append((f"script `{command}` = `{bin_field}` ({source} [bin])", bin_field))
    elif isinstance(bin_field, dict):
        for command in sorted(bin_field, key=str):
            value = bin_field[command]
            if isinstance(value, str):
                declared.append((f"script `{command}` = `{value}` ({source} [bin])", value))
    main = data.get("main")
    if isinstance(main, str):
        declared.append((f"main `{main}` ({source})", main))
    out: list[tuple[str, str]] = []
    for reason, value in declared:
        target = _package_target(prefix, value, path_set)
        if target is not None:
            out.append((target, reason))
    return out


def _package_target(prefix: str, value: str, path_set: set[str]) -> Optional[str]:
    """The indexed file a package.json path names: the path itself, else its
    TypeScript source next to it (``cli.js`` -> ``cli.ts``), else the same
    under ``src/`` when the path is in a build-output directory
    (``dist/cli.js`` -> ``src/cli.ts``). None if none is indexed or the path
    leaves the repository."""
    value = value.strip().replace("\\", "/")
    if not value or value.startswith("/") or ":" in value:
        return None
    path = posixpath.normpath(prefix + value)
    if path == ".." or path.startswith("../"):
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

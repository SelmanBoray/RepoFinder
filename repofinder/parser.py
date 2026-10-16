"""Tree-sitter based source parsing.

The module is organised around a small language registry (``LANGUAGES``) so more
tree-sitter grammars can be added later. Each :class:`LanguageSpec` declares the
file extensions it handles, how to load its grammar, and an extractor that turns
a syntax tree into :class:`Symbol` and :class:`ImportEdge` records, and a
resolver that maps its import edges to repository files. Python (``.py``) and
TypeScript (``typescript``: ``.ts``/``.mts``/``.cts``; ``tsx``: ``.tsx``) are
implemented.

Public contract:

* :func:`parse_file` parses one file and returns its record, symbols and imports.
* :func:`resolve_imports` fills ``ImportEdge.dst_path`` for edges that point
  inside the repository, using the resolver of the importing file's language.
"""

from __future__ import annotations

import functools
import inspect
import posixpath
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator, Optional

import tree_sitter_python
import tree_sitter_typescript
from tree_sitter import Language, Node, Parser, Query, QueryCursor, Tree

from .models import FileRecord, ImportEdge, RepoIndex, Symbol

DOCSTRING_MAX_CHARS = 400
_BOM = chr(0xFEFF)


# --------------------------------------------------------------------------- #
# Language registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SourceFile:
    """A file prepared for extraction (decoded, re-encoded as clean UTF-8)."""

    root: Path       # absolute, resolved repository root
    abs_path: Path   # absolute, resolved file path
    rel_path: str    # repo-relative POSIX path
    data: bytes      # UTF-8 bytes that were handed to tree-sitter


Extractor = Callable[[SourceFile, Tree], tuple[list[Symbol], list[ImportEdge]]]
# Resolves one edge to a repo-relative path (or None).
Resolver = Callable[[ImportEdge], Optional[str]]
# Builds a Resolver from every file of the index (all languages): languages
# pick the files they can import from.
ResolverFactory = Callable[[list[FileRecord]], Resolver]


@dataclass(frozen=True)
class LanguageSpec:
    name: str
    extensions: tuple[str, ...]
    grammar: Callable[[], object]   # returns the tree-sitter language pointer
    extractor: Extractor
    # Fills ImportEdge.dst_path in :func:`resolve_imports`. Languages that share
    # a factory (typescript and tsx) share one resolver per index.
    resolver: Optional[ResolverFactory] = None


LANGUAGES: dict[str, LanguageSpec] = {}


def register_language(spec: LanguageSpec) -> None:
    """Register a language. Later languages plug in here."""
    LANGUAGES[spec.name] = spec


def extensions_for(languages: tuple[str, ...]) -> tuple[str, ...]:
    """Return the file extensions handled by ``languages``.

    Raises ``ValueError`` for a language that has no registered parser.
    """
    exts: list[str] = []
    for lang in languages:
        spec = LANGUAGES.get(lang)
        if spec is None:
            known = ", ".join(sorted(LANGUAGES))
            raise ValueError(f"unsupported language {lang!r} (supported: {known})")
        exts.extend(spec.extensions)
    return tuple(exts)


def language_for_path(path: str | Path) -> Optional[LanguageSpec]:
    """Return the registered language that handles ``path``, if any."""
    suffix = PurePosixPath(str(path).replace("\\", "/")).suffix
    for spec in LANGUAGES.values():
        if suffix in spec.extensions:
            return spec
    return None


@functools.cache
def _get_language(language: str) -> Language:
    return Language(LANGUAGES[language].grammar())


@functools.cache
def _get_parser(language: str) -> Parser:
    return Parser(_get_language(language))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def parse_file(root: Path, path: Path) -> tuple[FileRecord, list[Symbol], list[ImportEdge]]:
    """Parse one source file.

    ``path`` may be absolute, relative to ``root``, or (when ``root`` itself is a
    relative path) relative to the working directory and starting with ``root``,
    as produced by ``root / "file.py"``. Syntax errors do not raise:
    tree-sitter always produces a tree, ``FileRecord.parse_ok`` is set to False
    when it contains errors, and whatever symbols were recovered are returned.
    I/O errors and unsupported file types do raise.
    """
    root = Path(root)
    path = Path(path)
    root_abs = root.resolve()
    if path.is_absolute():
        abs_path = path.resolve()
    elif not root.is_absolute() and path.is_relative_to(root) and root.parts:
        abs_path = path.resolve()  # e.g. root="repo", path="repo/pkg/mod.py" (cwd-relative)
    else:
        abs_path = (root_abs / path).resolve()
    rel_path = abs_path.relative_to(root_abs).as_posix()

    spec = language_for_path(rel_path)
    if spec is None:
        raise ValueError(f"no parser registered for {rel_path}")

    raw = abs_path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    if text.startswith(_BOM):  # a UTF-8 byte order mark is not part of the code
        text = text[len(_BOM):]
    data = text.encode("utf-8")

    tree = _get_parser(spec.name).parse(data)
    source = SourceFile(root=root_abs, abs_path=abs_path, rel_path=rel_path, data=data)
    symbols, imports = spec.extractor(source, tree)

    record = FileRecord(
        path=rel_path,
        language=spec.name,
        size=len(raw),
        n_lines=count_lines(raw),
        parse_ok=not tree.root_node.has_error,
    )
    return record, symbols, imports


def count_lines(raw: bytes) -> int:
    """Number of lines, counting a final line without a trailing newline."""
    if not raw:
        return 0
    return raw.count(b"\n") + (0 if raw.endswith(b"\n") else 1)


# --------------------------------------------------------------------------- #
# Python extraction
# --------------------------------------------------------------------------- #

# Statement containers that may hold definitions or imports. Expressions never
# contain statements, so the walk does not descend into them; ERROR nodes are
# included because tree-sitter wraps recoverable code in them on syntax errors.
_PY_CONTAINERS = frozenset(
    {
        "module",
        "block",
        "if_statement",
        "elif_clause",
        "else_clause",
        "for_statement",
        "while_statement",
        "try_statement",
        "except_clause",
        "except_group_clause",
        "finally_clause",
        "with_statement",
        "match_statement",
        "case_clause",
        "ERROR",
    }
)

_STRING_PREFIX_RE = re.compile(r"^([A-Za-z]*)")


@dataclass(frozen=True)
class _Scope:
    """Where a definition lives: module level, a class body or a function body."""

    kind: Optional[str]        # None (module), "class" or "function"
    qualname: Optional[str]    # qualname of the enclosing class/function
    prefix: Optional[str]      # prefix for the qualnames of definitions inside


_MODULE_SCOPE = _Scope(kind=None, qualname=None, prefix=None)


def _text(node: Node) -> str:
    raw = node.text
    return raw.decode("utf-8", errors="replace") if raw is not None else ""


# NOTE: read tree-sitter Points by index (point[0], point[1]), never through
# ``point.row`` / ``point.column``. With py-tree-sitter 0.26.0 the attribute
# getters corrupt the heap for values above 256 (outside CPython's small-int
# cache, so most likely a reference-count bug): the process later dies with an
# access violation in unrelated code. Index access is unaffected, and so is
# py-tree-sitter 0.25.2. See tests/test_parser.py::test_large_file_line_numbers.


def _line(node: Node) -> int:
    return node.start_point[0] + 1


def _end_line(node: Node) -> int:
    end_row, end_col = node.end_point
    # A node that ends at column 0 ends right after the previous line's newline.
    if end_col == 0 and end_row > node.start_point[0]:
        return end_row
    return end_row + 1


def _extract_python(source: SourceFile, tree: Tree) -> tuple[list[Symbol], list[ImportEdge]]:
    symbols: list[Symbol] = []
    imports: list[ImportEdge] = []
    package = _python_package_parts(source.root, source.rel_path)

    # Iterative pre-order walk: (node, scope, start-line override for decorated defs).
    stack: list[tuple[Node, _Scope, Optional[int]]] = [(tree.root_node, _MODULE_SCOPE, None)]
    while stack:
        node, scope, start_override = stack.pop()
        ntype = node.type
        children: list[tuple[Node, _Scope, Optional[int]]] = []

        if ntype == "decorated_definition":
            definition = node.child_by_field_name("definition")
            if definition is not None:
                children.append((definition, scope, start_override or _line(node)))
        elif ntype in ("function_definition", "class_definition"):
            child_scope = _python_definition(node, scope, start_override, source, symbols)
            body = node.child_by_field_name("body")
            if body is not None:
                children.append((body, child_scope, None))
        elif ntype in ("import_statement", "import_from_statement", "future_import_statement"):
            imports.extend(_python_imports(node, source.rel_path, package))
        elif ntype in _PY_CONTAINERS:
            children.extend((child, scope, None) for child in node.children)

        stack.extend(reversed(children))

    return symbols, imports


def _python_definition(
    node: Node,
    scope: _Scope,
    start_override: Optional[int],
    source: SourceFile,
    out: list[Symbol],
) -> _Scope:
    """Record a function/class symbol and return the scope for its body."""
    is_class = node.type == "class_definition"
    name_node = node.child_by_field_name("name")
    name = _text(name_node) if name_node is not None and not name_node.is_missing else ""

    if not name:
        # Broken header (syntax error): no symbol, but keep walking the body in
        # the current scope so nested definitions are not lost.
        return scope

    qualname = f"{scope.prefix}.{name}" if scope.prefix else name
    if is_class:
        kind = "class"
        child_scope = _Scope(kind="class", qualname=qualname, prefix=qualname)
    else:
        kind = "method" if scope.kind == "class" else "function"
        child_scope = _Scope(kind="function", qualname=qualname, prefix=f"{qualname}.<locals>")

    out.append(
        Symbol(
            id=f"{source.rel_path}::{qualname}",
            path=source.rel_path,
            name=name,
            qualname=qualname,
            kind=kind,
            start_line=start_override or _line(node),
            end_line=_end_line(node),
            signature=_python_signature(node, source.data),
            docstring=_python_docstring(node),
            parent=scope.qualname,
        )
    )
    return child_scope


def _python_signature(node: Node, data: bytes) -> str:
    """Header text from ``def``/``class`` (``async`` included) up to the body colon."""
    body = node.child_by_field_name("body")
    colon: Optional[Node] = None
    for child in node.children:
        if body is not None and child.start_byte >= body.start_byte:
            break
        if child.type == ":":
            colon = child
    if colon is not None:
        end = colon.start_byte
    elif body is not None:
        end = body.start_byte
    else:
        end = node.end_byte
    start = node.start_byte
    header = data[start:end]

    if b"#" in header:  # drop comments inside multi-line parameter lists
        pieces: list[bytes] = []
        cursor = start
        for comment in _iter_comments(node, start, end):
            pieces.append(data[cursor:comment.start_byte])
            cursor = comment.end_byte
        pieces.append(data[cursor:end])
        header = b"".join(pieces)

    text = " ".join(header.decode("utf-8", errors="replace").split())
    # Cosmetic for multi-line headers: "def f( a, b, )" -> "def f(a, b,)".
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]])", r"\1", text)
    return text


def _iter_comments(node: Node, start: int, end: int) -> Iterator[Node]:
    """Comment nodes under ``node`` that lie inside the byte range [start, end)."""
    stack = [node]
    found: list[Node] = []
    while stack:
        current = stack.pop()
        if current.end_byte <= start or current.start_byte >= end:
            continue
        if current.type == "comment":
            found.append(current)
            continue
        stack.extend(current.children)
    yield from sorted(found, key=lambda n: n.start_byte)


def _python_docstring(node: Node) -> Optional[str]:
    body = node.child_by_field_name("body")
    if body is None:
        return None
    first = next((c for c in body.named_children if c.type != "comment"), None)
    if first is None or first.type != "expression_statement" or first.named_child_count != 1:
        return None
    expr = first.named_children[0]
    if expr.type == "string":
        parts = [expr]
    elif expr.type == "concatenated_string":
        parts = [c for c in expr.named_children if c.type == "string"]
    else:
        return None

    chunks: list[str] = []
    for part in parts:
        content = _string_literal_content(part)
        if content is None:
            return None
        chunks.append(content)
    doc = "".join(chunks).replace("\r\n", "\n").replace("\r", "\n")
    doc = inspect.cleandoc(doc).strip()
    if not doc:
        return None
    return doc[:DOCSTRING_MAX_CHARS].rstrip()


def _string_literal_content(node: Node) -> Optional[str]:
    """Text between the quotes of a string literal; None for bytes/f-/t-strings."""
    start = next((c for c in node.children if c.type == "string_start"), None)
    end = next((c for c in node.children if c.type == "string_end"), None)
    if start is None or end is None:
        return None
    prefix = _STRING_PREFIX_RE.match(_text(start)).group(1).lower()
    if "b" in prefix or "f" in prefix or "t" in prefix:
        return None  # Python never treats these as docstrings
    data = node.text or b""
    base = node.start_byte
    return data[start.end_byte - base:end.start_byte - base].decode("utf-8", errors="replace")


def _dotted(node: Node) -> str:
    """Normalise a dotted_name node (``a . b`` -> ``a.b``)."""
    if node.type == "dotted_name":
        return ".".join(_text(c) for c in node.named_children if c.type == "identifier")
    return "".join(_text(node).split())


def _imported_name(node: Node) -> str:
    """Original (un-aliased) name of an import target."""
    if node.type == "aliased_import":
        inner = node.child_by_field_name("name")
        return _dotted(inner) if inner is not None else ""
    return _dotted(node)


def _python_imports(node: Node, rel_path: str, package: list[str]) -> list[ImportEdge]:
    line = _line(node)
    if node.type == "import_statement":
        edges = []
        for target in node.children_by_field_name("name"):
            module = _imported_name(target)
            if module:
                edges.append(ImportEdge(rel_path, module, None, [], line))
        return edges

    if node.type == "future_import_statement":
        module = "__future__"
    else:
        module_node = node.child_by_field_name("module_name")
        if module_node is None:
            return []
        if module_node.type == "relative_import":
            module = _resolve_relative(module_node, package)
        else:
            module = _dotted(module_node)
    if not module:
        return []

    if any(c.type == "wildcard_import" for c in node.children):
        names = ["*"]
    else:
        names = [n for n in (_imported_name(t) for t in node.children_by_field_name("name")) if n]
    return [ImportEdge(rel_path, module, None, names, line)]


def _resolve_relative(node: Node, package: list[str]) -> str:
    """Turn ``from ..pkg import x`` into an absolute dotted module name.

    Falls back to the text as written (``"..pkg"``) when the import climbs above
    the top-level package or nothing would be left to name.
    """
    prefix = next((c for c in node.children if c.type == "import_prefix"), None)
    level = _text(prefix).count(".") if prefix is not None else 0
    tail_node = next((c for c in node.children if c.type == "dotted_name"), None)
    tail = _dotted(tail_node) if tail_node is not None else ""
    written = "." * level + tail

    if level - 1 > len(package):
        return written
    base = package[: len(package) - (level - 1)]
    parts = base + ([tail] if tail else [])
    return ".".join(parts) if parts else written


@functools.lru_cache(maxsize=4096)
def _is_package_dir(directory: Path) -> bool:
    return (directory / "__init__.py").is_file()


def _python_package_parts(root: Path, rel_path: str) -> list[str]:
    """Dotted package of a file, e.g. ``src/flask/app.py`` -> ``["flask"]``.

    The package starts at the top-most directory of the unbroken ``__init__.py``
    chain above the file. A file outside any package falls back to its directory
    names relative to the repo root, with a leading ``src`` component dropped.
    """
    dirs = list(PurePosixPath(rel_path).parent.parts)
    top = len(dirs)
    while top > 0 and _is_package_dir(root.joinpath(*dirs[:top])):
        top -= 1
    if top < len(dirs):
        return dirs[top:]
    if dirs and dirs[0] == "src":
        return dirs[1:]
    return dirs


def _python_resolver(files: list[FileRecord]) -> Resolver:
    """Python module names resolved against the Python files of the index."""
    return _ImportResolver([f.path for f in files if f.language == "python"]).resolve


register_language(
    LanguageSpec(
        name="python",
        extensions=(".py",),
        grammar=tree_sitter_python.language,
        extractor=_extract_python,
        resolver=_python_resolver,
    )
)


# --------------------------------------------------------------------------- #
# Import resolution
# --------------------------------------------------------------------------- #


def _module_name(rel: str) -> Optional[str]:
    """``pkg/mod.py`` -> ``pkg.mod``; ``pkg/__init__.py`` -> ``pkg``."""
    if not rel.endswith(".py"):
        return None
    parts = rel[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts or not all(p.isidentifier() for p in parts):
        return None
    return ".".join(parts)


def module_roots(paths: list[str]) -> list[str]:
    """Global import roots in priority order, as POSIX prefixes (``""`` is the repo root).

    Global roots apply to every importer in the repository:

    * the repo root and ``src/``;
    * any other first-level directory that is not itself a package but directly
      contains one (``lib/pkg/__init__.py`` makes ``lib/`` a root), generalising
      the ``src`` layout. A first-level directory that *is* a package is already
      importable from the repo root and is not a root: otherwise its submodules
      would shadow stdlib names (``pkg/json.py`` would capture ``import json``).

    Nested roots (see :func:`nested_module_roots`) come on top of these but are
    *scoped*: the parent directory of every top-most ``__init__.py`` chain that
    is not already a global root (``examples/tutorial/`` for
    ``examples/tutorial/flaskr/__init__.py``) is a root only for importers located
    under its scope. The scope is the root itself, or its parent when the root is
    a ``src/`` directory (``examples/celery/src/`` serves ``examples/celery/``).
    Through a nested root, a name whose first component is a stdlib module never
    resolves, except for importers inside a package of that very name.

    Resolving an importer against the root of its own package chain is also how
    relative imports are resolved by path: ``from .db import x`` in
    ``a/b/c.py`` is stored as an absolute name relative to that root, so it
    lands on ``a/b/db.py`` or ``a/b/db/__init__.py`` (and on ``a/b/pkg/name.py``
    for ``from .pkg import name``) wherever the package sits in the repository.

    This function returns the global roots only; :func:`build_module_map` uses
    exactly these.
    """
    path_set = set(paths)
    extra: set[str] = set()
    for p in paths:
        parts = p.split("/")
        if len(parts) == 3 and parts[2] == "__init__.py":
            first = parts[0]
            if first != "src" and f"{first}/__init__.py" not in path_set:
                extra.add(f"{first}/")
    return ["", "src/", *sorted(extra)]


def _package_dirs(paths: list[str]) -> set[str]:
    """Directories (POSIX, no trailing slash) that contain an ``__init__.py``."""
    return {p[: -len("/__init__.py")] for p in paths if p.endswith("/__init__.py")}


def _package_chain_root(rel_path: str, package_dirs: set[str]) -> Optional[tuple[str, str]]:
    """``(root, top_package)`` for a file inside a package chain, else None.

    ``root`` is the parent directory of the top-most package directory above the
    file, as a POSIX prefix (``""`` or ``"examples/tutorial/"``), and
    ``top_package`` is that top-most package's name.
    """
    dirs = rel_path.split("/")[:-1]
    top = len(dirs)
    while top > 0 and "/".join(dirs[:top]) in package_dirs:
        top -= 1
    if top == len(dirs):
        return None
    return "".join(f"{d}/" for d in dirs[:top]), dirs[top]


def nested_module_roots(paths: list[str]) -> dict[str, str]:
    """Scoped nested import roots, mapped to their scope prefix.

    See :func:`module_roots` for the rule. Example: ``examples/celery/src/`` ->
    ``examples/celery/``; ``tests/test_apps/`` -> ``tests/test_apps/``.
    """
    package_dirs = _package_dirs(paths)
    global_roots = set(module_roots(paths))
    nested: dict[str, str] = {}
    for p in paths:
        found = _package_chain_root(p, package_dirs)
        if found is None or found[0] in global_roots:
            continue
        root = found[0]
        parts = root.rstrip("/").split("/")
        scope = "".join(f"{d}/" for d in parts[:-1]) if parts[-1] == "src" else root
        nested[root] = scope
    return dict(sorted(nested.items()))


def _modules_under(root: str, ordered_paths: list[str]) -> dict[str, str]:
    modules: dict[str, str] = {}
    for p in ordered_paths:
        if p.startswith(root):
            name = _module_name(p[len(root):])
            if name and name not in modules:
                modules[name] = p
    return modules


def build_module_map(paths: list[str]) -> dict[str, str]:
    """Map dotted module names importable from the global roots to file paths.

    When two roots yield the same module name, the earlier root wins. Nested,
    scoped roots are not included (see :func:`resolve_imports`).
    """
    modules: dict[str, str] = {}
    ordered = sorted(paths)
    for root in module_roots(ordered):
        for name, path in _modules_under(root, ordered).items():
            modules.setdefault(name, path)
    return modules


def _candidate_modules(edge: ImportEdge) -> Iterator[str]:
    """Dotted names to look up for ``edge``, best (longest) match first."""
    module = edge.dst_module
    # "from pkg import name" where pkg/name.py (or pkg/name/__init__.py) exists.
    if edge.names and edge.names != ["*"]:
        for name in edge.names:
            yield f"{module}.{name}"
    # The module itself, then its longest importable parent package.
    parts = module.split(".")
    for n in range(len(parts), 0, -1):
        yield ".".join(parts[:n])


def _resolve_edge(edge: ImportEdge, modules: dict[str, str]) -> Optional[str]:
    """Resolve ``edge`` against a single module map (no nested roots)."""
    if not edge.dst_module or edge.dst_module.startswith("."):
        return None
    for candidate in _candidate_modules(edge):
        hit = modules.get(candidate)
        if hit is not None:
            return hit
    return None


_STDLIB_MODULES = frozenset(sys.stdlib_module_names)


class _ImportResolver:
    """Resolves edges against the global module map plus scoped nested roots."""

    def __init__(self, paths: list[str]) -> None:
        ordered = sorted(paths)
        self.global_map = build_module_map(ordered)
        self.package_dirs = _package_dirs(ordered)
        self.nested = nested_module_roots(ordered)
        self.nested_maps = {root: _modules_under(root, ordered) for root in self.nested}
        self._roots_cache: dict[str, list[dict[str, str]]] = {}

    def _maps_for(self, src_path: str) -> list[dict[str, str]]:
        """Module maps of the nested roots whose scope contains ``src_path``, deepest first."""
        maps = self._roots_cache.get(src_path)
        if maps is None:
            roots = [r for r, scope in self.nested.items() if src_path.startswith(scope)]
            roots.sort(key=lambda r: (-r.count("/"), r))
            maps = [self.nested_maps[r] for r in roots]
            self._roots_cache[src_path] = maps
        return maps

    def resolve(self, edge: ImportEdge) -> Optional[str]:
        module = edge.dst_module
        if not module or module.startswith("."):
            return None
        maps = self._maps_for(edge.src_path)
        own = _package_chain_root(edge.src_path, self.package_dirs)
        own_top = own[1] if own is not None else None
        for candidate in _candidate_modules(edge):
            first = candidate.split(".", 1)[0]
            if maps and (first not in _STDLIB_MODULES or first == own_top):
                for modules in maps:
                    hit = modules.get(candidate)
                    if hit is not None:
                        return hit
            hit = self.global_map.get(candidate)
            if hit is not None:
                return hit
        return None


def resolve_imports(index: RepoIndex) -> None:
    """Fill ``ImportEdge.dst_path`` in place for edges that resolve inside the repo.

    Each edge is resolved by the language of its importing file (its
    ``FileRecord``, else its extension; Python when neither is known):

    * Python: module names are derived from the Python files of ``index.files``:
      the global roots of :func:`module_roots` for every importer, plus the
      scoped nested roots whose scope contains the importer (checked first for
      each candidate name). Stdlib and third-party modules keep ``dst_path=None``.
    * TypeScript/TSX: relative specifiers are resolved by path against every
      file of the index (:func:`resolve_ts_specifier`); bare package names keep
      ``dst_path=None``.
    """
    language_of = {f.path: f.language for f in index.files}
    resolvers: dict[ResolverFactory, Resolver] = {}
    for edge in index.imports:
        language = language_of.get(edge.src_path)
        spec = LANGUAGES.get(language) if language else language_for_path(edge.src_path)
        if spec is None:
            spec = LANGUAGES["python"]
        if spec.resolver is None:
            edge.dst_path = None
            continue
        resolve = resolvers.get(spec.resolver)
        if resolve is None:
            resolve = resolvers[spec.resolver] = spec.resolver(index.files)
        edge.dst_path = resolve(edge)


# --------------------------------------------------------------------------- #
# TypeScript extraction (the ``typescript`` and ``tsx`` grammars share it)
# --------------------------------------------------------------------------- #

SIGNATURE_MAX_CHARS = 200

# Statement containers walked for definitions and imports. As for Python, the
# walk does not descend into expressions (callbacks, object literals, JSX);
# ``expression_statement`` is listed only because the grammar wraps
# ``namespace X {}`` in one. ERROR nodes hold recoverable code on syntax errors.
_TS_CONTAINERS = frozenset(
    {
        "program",
        "statement_block",
        "if_statement",
        "else_clause",
        "for_statement",
        "for_in_statement",
        "while_statement",
        "do_statement",
        "try_statement",
        "catch_clause",
        "finally_clause",
        "switch_statement",
        "switch_body",
        "switch_case",
        "switch_default",
        "labeled_statement",
        "expression_statement",
        "ERROR",
    }
)
_TS_FUNCTION_DECLARATIONS = frozenset({"function_declaration", "generator_function_declaration"})
_TS_CLASS_DECLARATIONS = frozenset({"class_declaration", "abstract_class_declaration"})
_TS_TYPE_DECLARATIONS = {
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
    "enum_declaration": "enum",
}
_TS_NAMESPACES = frozenset({"internal_module", "module"})
# Declarations without a body: overloads, ``declare``/``.d.ts`` declarations and
# abstract members. Dropped when the same qualname has an implementation.
_TS_SIGNATURE_ONLY = frozenset({"function_signature", "method_signature", "abstract_method_signature"})
# Initialisers that make ``const x = ...`` or a class property ``p = ...`` a function.
_TS_FUNCTION_VALUES = frozenset({"arrow_function", "function_expression", "function", "generator_function"})

# require("x") and import("x") can sit anywhere in an expression, so they are
# found with a query over the whole tree instead of the statement walk.
_TS_CALL_IMPORTS = """
(call_expression
  function: (identifier) @fn
  arguments: (arguments . [(string) (template_string)] @src)
  (#eq? @fn "require"))
(call_expression
  function: (import)
  arguments: (arguments . [(string) (template_string)] @src))
"""
_TS_CALL_IMPORT_RE = re.compile(rb"\b(?:require|import)\s*\(")


@functools.cache
def _ts_call_import_query(language: str) -> Query:
    return Query(_get_language(language), _TS_CALL_IMPORTS)


# (node, scope, anchor): the anchor is the outermost node belonging to a
# definition (see _TsExtractor), None when that is the node itself.
_TsItem = tuple[Node, _Scope, Optional[Node]]


def _extract_typescript(source: SourceFile, tree: Tree) -> tuple[list[Symbol], list[ImportEdge]]:
    return _TsExtractor(source).run(tree)


class _TsExtractor:
    """Symbols and imports of one TypeScript/TSX file.

    Qualnames follow the Python conventions: ``Class.method``, ``Namespace.fn``
    and ``outer.<locals>.inner`` for definitions inside function bodies. The
    *anchor* of a definition is the outermost node that belongs to it: the
    ``export`` statement around it, a ``declare`` wrapper, the declaration of
    the first declarator in ``const a = () => 1, b = ...``, or the first
    decorator in front of a class member. ``start_line`` and the JSDoc lookup
    use the anchor (decorators and ``export`` are included, as decorators are
    for Python); the signature starts at the definition itself.
    """

    def __init__(self, source: SourceFile) -> None:
        self.source = source
        self.data = source.data
        self.found: list[tuple[Symbol, bool]] = []  # (symbol, signature-only)
        self.imports: list[ImportEdge] = []

    def run(self, tree: Tree) -> tuple[list[Symbol], list[ImportEdge]]:
        stack: list[_TsItem] = [(tree.root_node, _MODULE_SCOPE, None)]
        while stack:
            node, scope, anchor = stack.pop()
            stack.extend(reversed(self._visit(node, scope, anchor)))
        self._call_imports(tree)
        self.imports.sort(key=lambda e: e.line)  # stable: source order within a line
        return self._symbols(), self.imports

    # -- walk ---------------------------------------------------------------- #

    def _visit(self, node: Node, scope: _Scope, anchor: Optional[Node]) -> list[_TsItem]:
        ntype = node.type

        if ntype in _TS_CONTAINERS:
            return [(child, scope, None) for child in node.named_children]

        if ntype in ("lexical_declaration", "variable_declaration"):
            declarators = [c for c in node.named_children if c.type == "variable_declarator"]
            return [(d, scope, (anchor or node) if i == 0 else None) for i, d in enumerate(declarators)]

        if ntype == "variable_declarator":
            return self._declarator(node, scope, anchor)

        if ntype in _TS_FUNCTION_DECLARATIONS:
            body = node.child_by_field_name("body")
            signature = _ts_signature(self.data, node, _ts_header_start(node), _ts_body_start(node, body))
            return self._function(node, _ts_name(node.child_by_field_name("name")), scope, anchor, signature, body)

        if ntype in _TS_CLASS_DECLARATIONS:
            body = node.child_by_field_name("body")
            signature = _ts_signature(self.data, node, _ts_header_start(node), _ts_body_start(node, body))
            return self._class(node, _ts_name(node.child_by_field_name("name")), scope, anchor, signature, body)

        if ntype == "class_body":
            if scope.kind != "class":
                return []
            members: list[_TsItem] = []
            decorator: Optional[Node] = None  # first decorator in front of the next member
            for child in node.named_children:
                if child.type == "decorator":
                    decorator = decorator or child
                elif child.type != "comment":
                    members.append((child, scope, decorator))
                    decorator = None
            return members

        if ntype == "method_definition":
            if scope.kind != "class":
                return []
            body = node.child_by_field_name("body")
            signature = _ts_signature(self.data, node, _ts_header_start(node), _ts_body_start(node, body))
            return self._function(node, _ts_name(node.child_by_field_name("name")), scope, anchor, signature, body)

        if ntype == "public_field_definition":
            value = node.child_by_field_name("value")
            if scope.kind != "class" or value is None or value.type not in _TS_FUNCTION_VALUES:
                return []
            end = _ts_function_header_end(value)
            signature = _ts_signature(self.data, node, _ts_header_start(node), end)
            name = _ts_name(node.child_by_field_name("name"))
            return self._function(node, name, scope, anchor, signature, value.child_by_field_name("body"))

        if ntype in _TS_SIGNATURE_ONLY:
            if ntype != "function_signature" and scope.kind != "class":
                return []  # interface / object type members are not symbols
            signature = _ts_signature(self.data, node, _ts_header_start(node), node.end_byte)
            name = _ts_name(node.child_by_field_name("name"))
            return self._function(node, name, scope, anchor, signature, None, signature_only=True)

        kind = _TS_TYPE_DECLARATIONS.get(ntype)
        if kind is not None:
            name = _ts_name(node.child_by_field_name("name"))
            if name is not None:
                body = node.child_by_field_name("body")
                end = node.end_byte if ntype == "type_alias_declaration" else _ts_body_start(node, body)
                signature = _ts_signature(self.data, node, _ts_header_start(node), end)
                self._add(node, name, _ts_qualname(scope, name), kind, scope.qualname, anchor, signature)
            return []

        if ntype in _TS_NAMESPACES:
            return self._namespace(node, scope, anchor)

        if ntype == "export_statement":
            items: list[_TsItem] = []
            declaration = node.child_by_field_name("declaration")
            if declaration is not None:
                items.append((declaration, scope, anchor or node))
            else:
                value = node.child_by_field_name("value")
                if value is not None:
                    items.extend(self._default_export(node, value, scope))
            edge = _ts_reexport_edge(node, self.source.rel_path)
            if edge is not None:
                self.imports.append(edge)
            return items

        if ntype == "ambient_declaration":  # declare ... / declare global { ... }
            return [(child, scope, anchor or node) for child in node.named_children]

        if ntype == "import_statement":
            edge = _ts_import_edge(node, self.source.rel_path)
            if edge is not None:
                self.imports.append(edge)
            return []

        return []

    # -- definitions --------------------------------------------------------- #

    def _add(
        self,
        node: Node,
        name: str,
        qualname: str,
        kind: str,
        parent: Optional[str],
        anchor: Optional[Node],
        signature: str,
        signature_only: bool = False,
    ) -> None:
        outer = anchor or node
        symbol = Symbol(
            id=f"{self.source.rel_path}::{qualname}",
            path=self.source.rel_path,
            name=name,
            qualname=qualname,
            kind=kind,  # type: ignore[arg-type]
            start_line=_line(outer),
            end_line=_end_line(node),
            signature=signature,
            docstring=_ts_jsdoc(outer),
            parent=parent,
        )
        self.found.append((symbol, signature_only))

    def _function(
        self,
        node: Node,
        name: Optional[str],
        scope: _Scope,
        anchor: Optional[Node],
        signature: str,
        body: Optional[Node],
        signature_only: bool = False,
    ) -> list[_TsItem]:
        """Record a function/method and return its body to walk (if a block)."""
        has_block = body is not None and body.type == "statement_block"
        if name is None:
            # Broken header or computed member name ([Symbol.iterator]): no
            # symbol. Like Python, keep walking the body in the current scope,
            # except in a class body, where that would invent methods.
            return [(body, scope, None)] if has_block and scope.kind != "class" else []
        qualname = _ts_qualname(scope, name)
        kind = "method" if scope.kind == "class" else "function"
        self._add(node, name, qualname, kind, scope.qualname, anchor, signature, signature_only)
        if not has_block:
            return []
        return [(body, _Scope(kind="function", qualname=qualname, prefix=f"{qualname}.<locals>"), None)]

    def _class(
        self,
        node: Node,
        name: Optional[str],
        scope: _Scope,
        anchor: Optional[Node],
        signature: str,
        body: Optional[Node],
    ) -> list[_TsItem]:
        if name is None:
            return []  # members of a nameless (broken) class have no qualname
        qualname = _ts_qualname(scope, name)
        self._add(node, name, qualname, "class", scope.qualname, anchor, signature)
        if body is None:
            return []
        return [(body, _Scope(kind="class", qualname=qualname, prefix=qualname), None)]

    def _namespace(self, node: Node, scope: _Scope, anchor: Optional[Node]) -> list[_TsItem]:
        """``namespace A.B {}`` / ``module M {}``: one symbol, members nest under it."""
        name_node = node.child_by_field_name("name")
        body = node.child_by_field_name("body")
        parts: list[str] = []
        if name_node is not None and not name_node.is_missing and name_node.type != "string":
            parts = [p for p in "".join(_text(name_node).split()).split(".") if p]
        if not parts:
            # declare module "pkg" { ... } describes another module: no symbol,
            # its declarations are recorded in the enclosing scope.
            return [(body, scope, None)] if body is not None else []
        qualname = _ts_qualname(scope, ".".join(parts))
        parent = scope.qualname if len(parts) == 1 else qualname.rsplit(".", 1)[0]
        signature = _ts_signature(self.data, node, _ts_header_start(node), _ts_body_start(node, body))
        self._add(node, parts[-1], qualname, "namespace", parent, anchor, signature)
        if body is None:
            return []
        return [(body, _Scope(kind="namespace", qualname=qualname, prefix=qualname), None)]

    def _declarator(self, node: Node, scope: _Scope, anchor: Optional[Node]) -> list[_TsItem]:
        """``const f = () => ...`` / ``function () {}`` is a function; ``= class {}`` a class."""
        name_node = node.child_by_field_name("name")
        value = node.child_by_field_name("value")
        if name_node is None or value is None or name_node.type != "identifier":
            return []
        name = _text(name_node)
        keyword = _ts_declaration_keyword(node)
        if value.type in _TS_FUNCTION_VALUES:
            end = _ts_function_header_end(value)
            signature = _ts_signature(self.data, node, node.start_byte, end, prefix=keyword)
            return self._function(node, name, scope, anchor, signature, value.child_by_field_name("body"))
        if value.type == "class":
            body = value.child_by_field_name("body")
            signature = _ts_signature(self.data, node, node.start_byte, _ts_body_start(value, body), prefix=keyword)
            return self._class(node, name, scope, anchor, signature, body)
        return []

    def _default_export(self, export: Node, value: Node, scope: _Scope) -> list[_TsItem]:
        """``export default function () {}`` / ``() => ...`` / ``class {}``: named ``default``."""
        start = _ts_header_start(export)
        if value.type in _TS_FUNCTION_VALUES:
            name = _ts_name(value.child_by_field_name("name")) or "default"
            signature = _ts_signature(self.data, export, start, _ts_function_header_end(value))
            return self._function(value, name, scope, export, signature, value.child_by_field_name("body"))
        if value.type == "class":
            name = _ts_name(value.child_by_field_name("name")) or "default"
            body = value.child_by_field_name("body")
            signature = _ts_signature(self.data, export, start, _ts_body_start(value, body))
            return self._class(value, name, scope, export, signature, body)
        return []

    def _symbols(self) -> list[Symbol]:
        """Drop signature-only entries shadowed by an implementation (overloads)
        and keep the first of repeated signature-only entries."""
        implemented = {s.qualname for s, signature_only in self.found if not signature_only}
        seen: set[str] = set()
        out: list[Symbol] = []
        for symbol, signature_only in self.found:
            if signature_only:
                if symbol.qualname in implemented or symbol.qualname in seen:
                    continue
                seen.add(symbol.qualname)
            out.append(symbol)
        return out

    # -- require() / import() ----------------------------------------------- #

    def _call_imports(self, tree: Tree) -> None:
        if not _TS_CALL_IMPORT_RE.search(self.data):
            return
        spec = language_for_path(self.source.rel_path)
        if spec is None:
            return
        query = _ts_call_import_query(spec.name)
        for _pattern, captures in QueryCursor(query).matches(tree.root_node):
            for src in captures.get("src", ()):
                module = _ts_string_value(src)
                call = src.parent.parent if src.parent is not None else None
                if not module or call is None:
                    continue
                self.imports.append(
                    ImportEdge(self.source.rel_path, module, None, _ts_destructured_names(call), _line(call))
                )


def _ts_qualname(scope: _Scope, name: str) -> str:
    return f"{scope.prefix}.{name}" if scope.prefix else name


def _ts_name(node: Optional[Node]) -> Optional[str]:
    """Declared name from a ``name`` field; None when absent or not a literal."""
    if node is None or node.is_missing:
        return None
    ntype = node.type
    if ntype == "computed_property_name":  # ['literal']() has a name, [expr]() not
        inner = node.named_children[0] if node.named_child_count == 1 else None
        return _ts_string_value(inner) if inner is not None else None
    if ntype in ("string", "template_string"):
        return _ts_string_value(node)
    return _text(node) or None


def _ts_string_value(node: Optional[Node]) -> Optional[str]:
    """Content of a string literal (or a template string without ``${}``)."""
    if node is None:
        return None
    if node.type == "template_string":
        if any(c.type == "template_substitution" for c in node.named_children):
            return None
    elif node.type != "string":
        return None
    text = _text(node)
    return text[1:-1] if len(text) >= 2 else None


def _ts_header_start(node: Node) -> int:
    """Byte offset where the header starts, after leading decorators."""
    for child in node.children:
        if child.type not in ("decorator", "comment"):
            return child.start_byte
    return node.start_byte


def _ts_body_start(node: Node, body: Optional[Node]) -> int:
    return body.start_byte if body is not None else node.end_byte


def _ts_function_header_end(value: Node) -> int:
    """End of a function value's header: after ``=>`` for arrows, else the body."""
    if value.type == "arrow_function":
        for child in value.children:
            if child.type == "=>":
                return child.end_byte
    return _ts_body_start(value, value.child_by_field_name("body"))


def _ts_declaration_keyword(declarator: Node) -> str:
    """``const`` / ``let`` / ``var`` of the declaration around ``declarator``."""
    parent = declarator.parent
    if parent is None or parent.type not in ("lexical_declaration", "variable_declaration"):
        return ""
    first = parent.child(0)
    return _text(first) if first is not None and not first.is_named else ""


def _ts_signature(data: bytes, owner: Node, start: int, end: int, prefix: str = "") -> str:
    """One-line header text in [start, end) without comments, clipped."""
    header = data[start:end]
    if b"//" in header or b"/*" in header:
        pieces: list[bytes] = []
        cursor = start
        for comment in _iter_comments(owner, start, end):
            pieces.append(data[cursor:comment.start_byte])
            cursor = comment.end_byte
        pieces.append(data[cursor:end])
        header = b"".join(pieces)
    text = " ".join(header.decode("utf-8", errors="replace").split())
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]])", r"\1", text)
    text = text.rstrip(";,").rstrip()
    if prefix:
        text = f"{prefix} {text}"
    if len(text) > SIGNATURE_MAX_CHARS:
        text = text[:SIGNATURE_MAX_CHARS].rstrip()
    return text


def _ts_jsdoc(node: Node) -> Optional[str]:
    """First paragraph of the ``/** ... */`` comment right before ``node``.

    Line comments between the two (``// eslint-disable-next-line``) are
    skipped; one blank line is allowed between the comment and what it documents.
    """
    below = node
    prev = node.prev_sibling
    while prev is not None and prev.type == "comment":
        if below.start_point[0] - prev.end_point[0] > 2:
            return None
        text = _text(prev)
        if text.startswith("//"):
            below, prev = prev, prev.prev_sibling
            continue
        if text.startswith("/**") and not text.startswith("/**/") and text.endswith("*/"):
            return _jsdoc_summary(text[3:-2])
        return None
    return None


def _jsdoc_summary(body: str) -> Optional[str]:
    lines = []
    for raw in body.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if line.startswith("*"):
            line = line[1:].strip()
        lines.append(line)
    paragraph: list[str] = []
    for line in lines:
        if line.startswith("@"):  # block tags end the description
            break
        if not line:
            if paragraph:
                break
            continue
        paragraph.append(line)
    doc = " ".join(paragraph)
    return doc[:DOCSTRING_MAX_CHARS].rstrip() or None


# -- imports ------------------------------------------------------------------ #


def _ts_import_edge(node: Node, rel_path: str) -> Optional[ImportEdge]:
    """``import ... from "m"``, ``import "m"`` and ``import x = require("m")``.

    ``names`` holds the exported names as written in the source module:
    ``default`` for a default import, ``*`` for ``* as ns``, the original name
    for ``{a as b}``; empty for side-effect imports and ``import x = require()``.
    """
    source = node.child_by_field_name("source")
    names: list[str] = []
    for child in node.named_children:
        if child.type == "import_clause":
            for part in child.named_children:
                if part.type == "identifier":
                    names.append("default")
                elif part.type == "namespace_import":
                    names.append("*")
                elif part.type == "named_imports":
                    for spec in part.named_children:
                        if spec.type == "import_specifier":
                            name = _ts_name(spec.child_by_field_name("name"))
                            if name:
                                names.append(name)
        elif child.type == "import_require_clause":
            source = child.child_by_field_name("source")
    module = _ts_string_value(source)
    if not module:
        return None
    return ImportEdge(rel_path, module, None, names, _line(node))


def _ts_reexport_edge(node: Node, rel_path: str) -> Optional[ImportEdge]:
    """``export {a, b as c} from "m"`` (names a, b), ``export * [as ns] from "m"`` (``*``)."""
    module = _ts_string_value(node.child_by_field_name("source"))
    if not module:
        return None
    names: list[str] = []
    for child in node.children:
        if child.type == "export_clause":
            for spec in child.named_children:
                if spec.type == "export_specifier":
                    name = _ts_name(spec.child_by_field_name("name"))
                    if name:
                        names.append(name)
        elif child.type in ("namespace_export", "*"):
            names = ["*"]
            break
    return ImportEdge(rel_path, module, None, names, _line(node))


def _ts_destructured_names(call: Node) -> list[str]:
    """Names taken from ``const {a, b: c} = require("m")`` (or ``await import()``)."""
    parent = call.parent
    if parent is not None and parent.type == "await_expression":
        parent = parent.parent
    if parent is None or parent.type != "variable_declarator":
        return []
    pattern = parent.child_by_field_name("name")
    if pattern is None or pattern.type != "object_pattern":
        return []
    names: list[str] = []
    for child in pattern.named_children:
        if child.type == "shorthand_property_identifier_pattern":
            names.append(_text(child))
        elif child.type == "pair_pattern":
            name = _ts_name(child.child_by_field_name("key"))
            if name:
                names.append(name)
        elif child.type == "object_assignment_pattern":
            left = child.child_by_field_name("left")
            if left is not None:
                names.append(_text(left))
    return names


# -- resolution --------------------------------------------------------------- #

_TS_EXTENSIONS = (".ts", ".tsx", ".mts", ".cts", ".d.ts")
# TS sources import the emitted file name: "./a.js" means ./a.ts (or .tsx, .d.ts).
_TS_EMITTED_TO_SOURCE = {
    ".js": (".ts", ".tsx", ".d.ts"),
    ".jsx": (".tsx",),
    ".mjs": (".mts", ".d.mts"),
    ".cjs": (".cts", ".d.cts"),
}
_TS_INDEX_FILES = ("index.ts", "index.tsx", "index.d.ts")


def resolve_ts_specifier(src_path: str, specifier: str, known: set[str] | frozenset[str]) -> Optional[str]:
    """Repo-relative file a relative TS import specifier points at, or None.

    Candidates, in order: the specifier as written; with ``.ts``, ``.tsx``,
    ``.mts``, ``.cts``, ``.d.ts`` appended; the emitted-name rewrites ``.js`` ->
    ``.ts``/``.tsx``/``.d.ts``, ``.jsx`` -> ``.tsx``, ``.mjs`` -> ``.mts``, ``.cjs``
    -> ``.cts``; then ``<dir>/index.ts``, ``index.tsx``, ``index.d.ts``. The first
    candidate in ``known`` wins. Bare (package) specifiers, absolute paths and
    paths that leave the repository return None; tsconfig ``paths`` aliases are
    not consulted.
    """
    if not (specifier in (".", "..") or specifier.startswith(("./", "../"))):
        return None
    base = posixpath.normpath(posixpath.join(posixpath.dirname(src_path), specifier))
    if base == ".." or base.startswith("../"):
        return None
    candidates: list[str] = []
    if base != ".":
        candidates.append(base)
        candidates.extend(base + ext for ext in _TS_EXTENSIONS)
        stem, suffix = posixpath.splitext(base)
        candidates.extend(stem + ext for ext in _TS_EMITTED_TO_SOURCE.get(suffix, ()))
    directory = "" if base == "." else f"{base}/"
    candidates.extend(directory + index for index in _TS_INDEX_FILES)
    return next((c for c in candidates if c in known), None)


def _typescript_resolver(files: list[FileRecord]) -> Resolver:
    """Relative specifiers resolved by path against every file of the index."""
    known = frozenset(f.path for f in files)
    return lambda edge: resolve_ts_specifier(edge.src_path, edge.dst_module, known)


register_language(
    LanguageSpec(
        name="typescript",
        extensions=(".ts", ".mts", ".cts"),
        grammar=tree_sitter_typescript.language_typescript,
        extractor=_extract_typescript,
        resolver=_typescript_resolver,
    )
)
register_language(
    LanguageSpec(
        name="tsx",
        extensions=(".tsx",),
        grammar=tree_sitter_typescript.language_tsx,
        extractor=_extract_typescript,
        resolver=_typescript_resolver,
    )
)

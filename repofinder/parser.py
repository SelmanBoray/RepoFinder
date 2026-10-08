"""Tree-sitter based source parsing.

The module is organised around a small language registry (``LANGUAGES``) so more
tree-sitter grammars can be added later. Each :class:`LanguageSpec` declares the
file extensions it handles, how to load its grammar, and an extractor that turns
a syntax tree into :class:`Symbol` and :class:`ImportEdge` records. Only Python
is implemented for now.

Public contract:

* :func:`parse_file` parses one file and returns its record, symbols and imports.
* :func:`resolve_imports` fills ``ImportEdge.dst_path`` for edges that point
  inside the repository.
"""

from __future__ import annotations

import functools
import inspect
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator, Optional

import tree_sitter_python
from tree_sitter import Language, Node, Parser, Tree

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


@dataclass(frozen=True)
class LanguageSpec:
    name: str
    extensions: tuple[str, ...]
    grammar: Callable[[], object]   # returns the tree-sitter language pointer
    extractor: Extractor


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
def _get_parser(language: str) -> Parser:
    spec = LANGUAGES[language]
    return Parser(Language(spec.grammar()))


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


register_language(
    LanguageSpec(
        name="python",
        extensions=(".py",),
        grammar=tree_sitter_python.language,
        extractor=_extract_python,
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

    Module names are derived from ``index.files``: the global roots of
    :func:`module_roots` for every importer, plus the scoped nested roots whose
    scope contains the importer (checked first for each candidate name). Stdlib
    and third-party modules keep ``dst_path=None``.
    """
    paths = [f.path for f in index.files if f.language == "python"]
    resolver = _ImportResolver(paths)
    for edge in index.imports:
        edge.dst_path = resolver.resolve(edge)

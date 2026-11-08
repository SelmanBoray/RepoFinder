"""Rust support for :mod:`repofinder.parser`.

The grammar comes from ``tree-sitter-rust``. This module imports nothing from
:mod:`repofinder.parser`, which imports it.

Symbol mapping: free functions (any qualifiers, also the ``fn`` declarations of an
``extern "C" { }`` block) are ``"function"``; ``struct`` and ``union`` are
``"class"``, ``enum`` is ``"enum"``, ``trait`` is ``"interface"``, ``type X = ...;``
is ``"type"`` and an inline ``mod m { ... }`` is ``"namespace"``. Functions inside
``impl T``, ``impl Trait for T`` and trait bodies are ``"method"`` named ``T.name``
with parent ``T``. ``T`` is the base name of the implemented type: last path segment,
generics, lifetimes, references and pointers stripped (``impl<'a> fmt::Display for
&'a Foo<'a>`` gives ``Foo``, ``impl dyn Trait`` gives ``Trait``); a slice, array or
tuple keeps its text (``[T]``). The impl block itself is not a symbol. Items of an
inline module carry its name as a prefix: ``m.helper``, ``m.Type``,
``m.Type.method`` (parent ``m.Type``), ``a.b.f``. Not symbols: fields, enum
variants, associated types and consts, consts, statics, ``macro_rules!``
definitions, items inside function bodies, closures, and whatever a macro expands
to. Declarations with the same qualname all stay, with the same id: ``fmt`` of
``impl Display for X`` and of ``impl Debug for X`` are two symbols ``X.fmt``.

Ranges start at the first outer attribute (``#[...]``) attached to the item, else
at the item; doc comments are never part of a range. The signature is the item's
text up to the ``{`` of its body, or the whole item without its ``;``, with
comments removed and whitespace collapsed. The docstring is the nearest run of
``///`` lines, or the nearest ``/** */`` block, above the item (attributes and
plain comments may sit in between), markers and one following space removed, up to
the first blank doc line. ``#[doc = "..."]`` attributes and ``//!`` inner docs are
not read.

Imports: every ``use`` leaf is one edge, ``dst_module`` the path without its last
segment as written (``crate``/``self``/``super`` kept), ``names`` the last segment:
``use a::{B, c::D as E, f::*}`` gives ``a`` [B], ``a::c`` [D], ``a::f`` [*];
``use a::{self}`` and ``use a;`` give ``a`` []. A leading ``::`` stays in
``dst_module`` (``use ::std::io`` gives ``::std`` [io]). ``mod m;`` gives
``self::m`` [] and ``extern crate x;`` gives ``x`` []. ``use`` declarations inside
function bodies and inline modules count; those inside macro invocations are not seen.

Inside inline ``mod`` blocks a leading ``self``/``super`` is rewritten so that
``dst_module`` is relative to the file's module, as rustc reads it (the model
documents ``dst_module`` as "as written/resolved"). In ``mod tests { }``:
``use super::foo::X`` gives ``self::foo`` [X], ``use super::*`` gives ``self`` [*]
(the file itself), ``use self::y::Z`` gives ``self::tests::y`` [Z] and
``use super::super::x::Y`` gives ``super::x`` [Y]; ``mod m;`` inside ``mod a { }``
gives ``self::a::m``. Every other path, and every path outside inline modules, stays
as written.

Resolution (:func:`rust_resolver`) follows Rust's module tree by file paths:

* Crate roots: ``lib.rs`` and ``main.rs``, files directly in a package's
  ``src/bin/``, ``tests/``, ``examples/`` and ``benches/``, a package's
  ``build.rs``, and the ``path`` of ``[lib]``, ``[[bin]]``, ``[[test]]``,
  ``[[example]]`` and ``[[bench]]`` in ``Cargo.toml``. Below a crate root's
  directory ``a.rs`` or ``a/mod.rs`` is module ``a``, ``a/b.rs`` is ``a::b``.
* ``mod m;`` (``self::m``) is ``m.rs`` or ``m/mod.rs`` next to a crate root or
  ``mod.rs`` file, ``x/m.rs`` or ``x/m/mod.rs`` for any other ``x.rs``.
* ``crate::``, ``self::`` and ``super::`` are resolved from the importing file's
  module. A path that starts with the name of a crate of the repository resolves
  into that crate's library root. ``std``, ``core``, ``alloc`` and crates that are
  not in the repository give None.
* The target is the most specific module file among ``dst_module::name`` (for a
  single name other than ``*``) and ``dst_module``; shorter prefixes are never
  tried, so ``use crate::a::Enum::Variant`` resolves to nothing.

Packages and crate names come from the ``Cargo.toml`` files under the ``root``
given to :func:`rust_resolver`: those in a directory that holds indexed ``.rs``
files or lies above one, ``[package] name`` (``-`` read as ``_``) and ``[lib]
name``. A manifest without ``[package]`` (a virtual workspace) names no crate.
Manifests are read as text with a few regular expressions, never executed, only
inside ``root`` (a link that leads out is skipped), and cached by path, size and
mtime. Without a root, or without manifests, packages are guessed from the paths
(the repository root and the parent of every ``src/`` holding ``lib.rs`` or
``main.rs``), so ``crate``/``self``/``super`` and ``mod`` still resolve, but no
path resolves into another crate.

Limits: ``#[cfg]`` is ignored, every item counts. ``#[path = "..."]`` is ignored.
Re-exports are not followed: ``use crate::Foo`` resolves to the crate root that
re-exports ``Foo``, not to the file that defines it. Inline modules are not module
files: ``use crate::a::inner::X`` with an inline ``inner``, and ``use self::X``
inside ``mod tests { }`` (``self::tests`` [X]), resolve to nothing. A 2018 uniform
path to a local module (``use util::X`` after ``mod util;``) is read as a crate
name. Dependency renames (``foo = { package = "bar" }``) and 2015-edition
crate-relative ``use`` paths are not understood. A directory with both ``lib.rs``
and ``main.rs`` belongs to the library for files below it. Files outside every
package and crate directory resolve only ``self``, ``super`` and ``mod``.
"""

from __future__ import annotations

import posixpath
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

import tree_sitter_rust
from tree_sitter import Language, Node, Query, QueryCursor, Tree

from .models import FileRecord, ImportEdge, Symbol

if TYPE_CHECKING:
    from .parser import Resolver, SourceFile

# The limits of repofinder.parser, which cannot be imported from here.
_DOCSTRING_MAX_CHARS = 400
_SIGNATURE_MAX_CHARS = 200


def rust_grammar() -> object:
    """Grammar pointer for ``LanguageSpec.grammar``."""
    return tree_sitter_rust.language()


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

_TYPE_KINDS = {
    "struct_item": "class",
    "union_item": "class",
    "enum_item": "enum",
    "trait_item": "interface",
    "type_item": "type",
}
_FUNCTIONS = ("function_item", "function_signature_item")
_COMMENTS = ("line_comment", "block_comment")
# Outer attributes and comments are siblings in front of an item, never children.
_PREAMBLE = ("attribute_item", *_COMMENTS)
# Bodies that open with "{": the signature stops there. A tuple struct's
# ``ordered_field_declaration_list`` is part of its signature.
_BRACE_BODIES = ("block", "declaration_list", "field_declaration_list", "enum_variant_list")
_PATH_KEYWORDS = ("identifier", "self", "super", "crate", "metavariable")

_IMPORT_QUERY = "(use_declaration) @use (extern_crate_declaration) @extern (mod_item) @mod"
_query_lock = threading.Lock()
_import_query: Optional[Query] = None


def extract_rust(source: SourceFile, tree: Tree) -> tuple[list[Symbol], list[ImportEdge]]:
    return _RustExtractor(source).run(tree)


class _RustExtractor:
    """Symbols and imports of one Rust file."""

    def __init__(self, source: SourceFile) -> None:
        self.data = source.data
        self.path = source.rel_path
        self.symbols: list[Symbol] = []
        self.imports: list[ImportEdge] = []

    def run(self, tree: Tree) -> tuple[list[Symbol], list[ImportEdge]]:
        self._items(tree.root_node, None)
        self._collect_imports(tree.root_node)
        return self.symbols, self.imports

    # -- declarations -------------------------------------------------------- #

    # Items get their siblings and index from the loop over ``children``: tree-sitter
    # answers ``Node.prev_sibling`` and ``Node.parent`` by scanning from the first
    # child (of the parent, of the root), so calling them per item is quadratic.

    def _items(self, container: Node, prefix: Optional[str]) -> None:
        """Items of a file or an inline module body; ``prefix`` is the module qualname."""
        children = container.children
        for index, node in enumerate(children):
            ntype = node.type
            if ntype == "ERROR":  # recoverable code of a file with syntax errors
                self._items(node, prefix)
            elif ntype in _FUNCTIONS:
                self._add(children, index, "function", prefix)
            elif ntype in _TYPE_KINDS:
                qualname = self._add(children, index, _TYPE_KINDS[ntype], prefix)
                if ntype == "trait_item" and qualname is not None:
                    self._methods(node.child_by_field_name("body"), qualname)
            elif ntype == "impl_item":
                owner = _type_base_name(node.child_by_field_name("type"))
                if owner is not None:
                    self._methods(node.child_by_field_name("body"), _qualify(prefix, owner))
            elif ntype == "mod_item":
                body = node.child_by_field_name("body")
                if body is not None:  # ``mod m;`` is an import, not a symbol
                    qualname = self._add(children, index, "namespace", prefix)
                    if qualname is not None:
                        self._items(body, qualname)
            elif ntype == "foreign_mod_item":  # extern "C" { fn f(); }
                body = node.child_by_field_name("body")
                if body is not None:
                    self._items(body, prefix)

    def _methods(self, body: Optional[Node], owner: str) -> None:
        if body is None:
            return
        children = body.children
        for index, node in enumerate(children):
            if node.type == "ERROR":
                self._methods(node, owner)
            elif node.type in _FUNCTIONS:
                self._add(children, index, "method", owner)

    def _add(self, siblings: list[Node], index: int, kind: str, parent: Optional[str]) -> Optional[str]:
        """Symbol for the item ``siblings[index]``; its qualname, None if it has no name."""
        node = siblings[index]
        name = _name(node.child_by_field_name("name"))
        if name is None:
            return None
        qualname = _qualify(parent, name)
        preamble = _preamble(siblings, index)
        start_line = min([_line(node), *(_line(p) for p in preamble if p.type == "attribute_item")])
        self.symbols.append(
            Symbol(
                id=f"{self.path}::{qualname}",
                path=self.path,
                name=name,
                qualname=qualname,
                kind=kind,  # type: ignore[arg-type]
                start_line=start_line,
                end_line=_end_line(node),
                signature=self._signature(node),
                docstring=_docstring(preamble),
                parent=parent,
            )
        )
        return qualname

    def _signature(self, node: Node) -> str:
        body = node.child_by_field_name("body")
        if body is not None and body.type in _BRACE_BODIES:
            text = self._text_without_comments(node, node.start_byte, body.start_byte)
        else:
            text = self._text_without_comments(node, node.start_byte, node.end_byte)
            if text.endswith(";"):
                text = text[:-1].rstrip()
        return text[:_SIGNATURE_MAX_CHARS].rstrip()

    def _text_without_comments(self, owner: Node, start: int, end: int) -> str:
        """Text in [start, end) without comments, whitespace collapsed."""
        header = self.data[start:end]
        if b"//" in header or b"/*" in header:
            pieces: list[bytes] = []
            cursor = start
            for comment in _iter_comments(owner, start, end):
                pieces.append(self.data[cursor:comment.start_byte])
                cursor = comment.end_byte
            pieces.append(self.data[cursor:end])
            header = b" ".join(pieces)
        return " ".join(header.decode("utf-8", errors="replace").split())

    # -- imports ------------------------------------------------------------- #

    def _collect_imports(self, root: Node) -> None:
        captures = QueryCursor(_imports_query()).captures(root)
        nodes = sorted((n for found in captures.values() for n in found), key=lambda n: n.start_byte)
        # Inline modules around the current node, outermost first, as (end_byte, name).
        # Captures are sorted by start and syntax nodes nest, so this sweep finds
        # them without walking up ``Node.parent``.
        open_modules: list[tuple[int, str]] = []
        for node in nodes:
            while open_modules and open_modules[-1][0] <= node.start_byte:
                open_modules.pop()
            scope = tuple(name for _, name in open_modules)
            line = _line(node)
            if node.type == "use_declaration":
                argument = node.child_by_field_name("argument")
                if argument is None:
                    continue
                for module, names in _use_leaves(argument):
                    dst_module = "::".join(_relative_to_file(module, scope))
                    self.imports.append(ImportEdge(self.path, dst_module, None, names, line))
            elif node.type == "extern_crate_declaration":
                name = _name(node.child_by_field_name("name"))
                if name is not None:
                    self.imports.append(ImportEdge(self.path, name, None, [], line))
            elif node.child_by_field_name("body") is None:  # mod m;
                name = _name(node.child_by_field_name("name"))
                if name is not None:
                    dst_module = "::".join(["self", *scope, name])
                    self.imports.append(ImportEdge(self.path, dst_module, None, [], line))
            else:  # mod m { ... }: what follows up to its end is inside it
                open_modules.append((node.end_byte, _name(node.child_by_field_name("name")) or "_"))


def _imports_query() -> Query:
    global _import_query
    with _query_lock:
        if _import_query is None:
            _import_query = Query(Language(tree_sitter_rust.language()), _IMPORT_QUERY)
        return _import_query


def _relative_to_file(segments: list[str], scope: tuple[str, ...]) -> list[str]:
    """A path written inside the inline modules ``scope``, relative to the file's module.

    Only a leading ``self``/``super`` depends on where the path is written; any
    other path is returned as it is.
    """
    if not scope or not segments or segments[0] not in ("self", "super"):
        return segments
    module = list(scope)  # where the path is written, from the file's module down
    index = 1 if segments[0] == "self" else 0
    above = 0  # ``super`` steps that leave the file's module
    while index < len(segments) and segments[index] == "super":
        if module:
            module.pop()
        else:
            above += 1
        index += 1
    rest = segments[index:]
    if above:
        return ["super"] * above + rest
    return ["self", *module, *rest]


def _use_leaves(argument: Node) -> list[tuple[list[str], list[str]]]:
    """(module path, names) for every leaf of a ``use`` tree, in source order."""
    leaves: list[tuple[list[str], list[str]]] = []
    _use_tree(argument, [], leaves)
    return leaves


def _use_tree(node: Node, prefix: list[str], out: list[tuple[list[str], list[str]]]) -> None:
    ntype = node.type
    if ntype in _COMMENTS:
        return
    if ntype == "use_list":
        for child in node.named_children:
            _use_tree(child, prefix, out)
    elif ntype == "scoped_use_list":
        path = node.child_by_field_name("path")
        head = _path_segments(path) if path is not None else _leading_root(node)
        listing = node.child_by_field_name("list")
        if head is not None and listing is not None:
            _use_tree(listing, prefix + head, out)
    elif ntype == "use_wildcard":
        path = next((c for c in node.children if c.type not in ("::", "*", *_COMMENTS)), None)
        head = _path_segments(path) if path is not None else _leading_root(node)
        if head is not None and prefix + head:
            out.append((prefix + head, ["*"]))
    elif ntype == "use_as_clause":  # the alias does not matter, the path does
        path = node.child_by_field_name("path")
        if path is not None:
            _use_tree(path, prefix, out)
    elif ntype == "self" and prefix:  # a::{self}
        out.append((prefix, []))
    else:
        segments = _path_segments(node)
        if segments:
            segments = prefix + segments
            if len(segments) == 1 or (len(segments) == 2 and segments[0] == ""):  # use a; use ::a;
                out.append((segments, []))
            else:
                out.append((segments[:-1], [segments[-1]]))


def _path_segments(node: Node) -> Optional[list[str]]:
    """``a::b::c`` as ["a", "b", "c"]; a leading ``::`` is an empty first segment."""
    if node.type == "scoped_identifier":
        path = node.child_by_field_name("path")
        name = _name(node.child_by_field_name("name"))
        head = _path_segments(path) if path is not None else _leading_root(node)
        if head is None or name is None:
            return None
        return head + [name]
    if node.type in _PATH_KEYWORDS:
        name = _name(node)
        return [name] if name is not None else None
    text = "".join(_text(node).split())  # anything else: take it as written
    return text.split("::") if text and not node.has_error else None


def _leading_root(node: Node) -> list[str]:
    """[""] when a path without a ``path`` field starts with ``::``, else []."""
    first = node.children[0] if node.children else None
    return [""] if first is not None and first.type == "::" else []


def _type_base_name(node: Optional[Node]) -> Optional[str]:
    """``Foo`` of ``&'a mut crate::m::Foo<'a, T>``; the text of slices, arrays, tuples."""
    while node is not None:
        ntype = node.type
        if ntype in ("type_identifier", "primitive_type", "identifier"):
            return _name(node)
        if ntype == "scoped_type_identifier":
            node = node.child_by_field_name("name")
        elif ntype in ("generic_type", "reference_type", "pointer_type"):
            node = node.child_by_field_name("type")
        elif ntype in ("dynamic_type", "abstract_type"):
            node = node.child_by_field_name("trait")
        else:
            text = " ".join(_text(node).split())
            return text or None
    return None


def _qualify(prefix: Optional[str], name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _name(node: Optional[Node]) -> Optional[str]:
    if node is None or node.is_missing:
        return None
    return _text(node) or None


def _text(node: Node) -> str:
    raw = node.text
    return raw.decode("utf-8", errors="replace") if raw is not None else ""


# NOTE: read tree-sitter Points by index, see the note in repofinder/parser.py.
def _line(node: Node) -> int:
    return node.start_point[0] + 1


def _end_line(node: Node) -> int:
    end_row, end_col = node.end_point
    # A node that ends at column 0 ends right after the previous line's newline.
    if end_col == 0 and end_row > node.start_point[0]:
        return end_row
    return end_row + 1


def _preamble(siblings: list[Node], index: int) -> list[Node]:
    """Attributes and comments in front of the item ``siblings[index]``, nearest first."""
    found: list[Node] = []
    position = index - 1
    while position >= 0 and siblings[position].type in _PREAMBLE:
        found.append(siblings[position])
        position -= 1
    return found


def _is_outer_doc(node: Node) -> bool:
    # ``///`` and ``/**`` have an ``outer`` marker; ``////``, ``/***`` and ``/**/`` do not.
    return node.type in _COMMENTS and node.child_by_field_name("outer") is not None


def _docstring(preamble: list[Node]) -> Optional[str]:
    """The nearest ``///`` run or ``/** */`` block of a preamble, first paragraph."""
    run: list[Node] = []
    for node in preamble:
        if _is_outer_doc(node):
            if run and not (
                node.type == "line_comment"
                and run[-1].type == "line_comment"
                and node.start_point[0] + 1 == run[-1].start_point[0]
            ):
                break
            run.append(node)
            if node.type == "block_comment":
                break
        elif run:  # an attribute or a plain comment ends a run that has started
            break
    if not run:
        return None
    run.reverse()
    if run[0].type == "block_comment":
        lines = _block_doc_lines(_text(run[0]))
    else:
        lines = [_line_doc(_text(n)) for n in run]
    paragraph: list[str] = []
    for line in lines:
        if not line.strip():
            if paragraph:
                break
            continue
        paragraph.append(line)
    doc = "\n".join(paragraph)[:_DOCSTRING_MAX_CHARS].rstrip()
    return doc or None


def _line_doc(text: str) -> str:
    body = text[3:]
    if body.startswith(" "):
        body = body[1:]
    return body.rstrip()


def _block_doc_lines(text: str) -> list[str]:
    body = text[3:-2] if text.endswith("*/") else text[3:]
    lines: list[str] = []
    for index, raw in enumerate(body.splitlines()):
        line = raw if index == 0 else raw.lstrip()
        if index > 0 and line.startswith("*"):
            line = line[1:]
        if line.startswith(" "):
            line = line[1:]
        lines.append(line.rstrip())
    return lines


def _iter_comments(node: Node, start: int, end: int) -> Iterator[Node]:
    """Comment nodes under ``node`` that lie inside the byte range [start, end)."""
    stack = [node]
    found: list[Node] = []
    while stack:
        current = stack.pop()
        if current.end_byte <= start or current.start_byte >= end:
            continue
        if current.type in _COMMENTS:
            found.append(current)
            continue
        stack.extend(current.children)
    yield from sorted(found, key=lambda n: n.start_byte)


# --------------------------------------------------------------------------- #
# Cargo manifests
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Manifest:
    is_package: bool                 # has a [package] table (a virtual workspace has not)
    package_name: Optional[str]
    lib_name: Optional[str]
    lib_path: Optional[str]
    target_paths: tuple[str, ...]    # path of every [[bin]], [[test]], [[example]], [[bench]]


_TABLE_RE = re.compile(r"""^\s*\[\[?\s*([^\[\]]+?)\s*\]\]?\s*(?:#.*)?$""")
_KEY_RE = re.compile(r"""^\s*(name|path)\s*=\s*(?:"((?:[^"\\\n]|\\.)*)"|'([^'\n]*)')""")
_TARGET_TABLES = frozenset({"bin", "test", "example", "bench"})
_MANIFEST_MAX_BYTES = 1 << 20
_MANIFEST_CACHE_MAX = 4096

_manifest_lock = threading.Lock()
_manifest_cache: dict[str, tuple[int, int, _Manifest]] = {}


def _parse_manifest(text: str) -> _Manifest:
    """The few ``Cargo.toml`` keys resolution needs, by line; not a TOML parser."""
    table = ""
    is_package = False
    package_name = lib_name = lib_path = None
    targets: list[str] = []
    open_string: Optional[str] = None  # delimiter of a multi-line string being skipped
    for line in text.splitlines():
        if open_string is not None:
            if line.count(open_string) % 2:
                open_string = None
            continue
        opened = next((d for d in ('"""', "'''") if line.count(d) % 2), None)
        if opened is not None:  # description = """ ... : what follows is no key
            open_string = opened
            continue
        header = _TABLE_RE.match(line)
        if header is not None:
            table = header.group(1).strip().strip("\"'")
            is_package = is_package or table == "package"
            continue
        key = _KEY_RE.match(line)
        if key is None:
            continue
        value = key.group(2) if key.group(2) is not None else key.group(3)
        if table == "package" and key.group(1) == "name":
            package_name = value
        elif table == "lib":
            if key.group(1) == "name":
                lib_name = value
            else:
                lib_path = value
        elif table in _TARGET_TABLES and key.group(1) == "path":
            targets.append(value)
    return _Manifest(is_package, package_name, lib_name, lib_path, tuple(targets))


def _read_manifest(path: Path) -> Optional[_Manifest]:
    try:
        stat = path.stat()
    except OSError:
        return None
    key = str(path)
    with _manifest_lock:
        cached = _manifest_cache.get(key)
    if cached is not None and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
        return cached[2]
    try:
        with open(path, "rb") as file:
            raw = file.read(_MANIFEST_MAX_BYTES)
    except OSError:
        return None
    manifest = _parse_manifest(raw.decode("utf-8", errors="replace"))
    with _manifest_lock:
        if len(_manifest_cache) >= _MANIFEST_CACHE_MAX:
            _manifest_cache.clear()
        _manifest_cache[key] = (stat.st_mtime_ns, stat.st_size, manifest)
    return manifest


def _join_relative(directory: str, path: str) -> Optional[str]:
    """``directory/path`` normalised, None if it leaves the repository."""
    path = path.replace("\\", "/")
    if path.startswith("/") or re.match(r"^[A-Za-z]:", path):
        return None
    joined = posixpath.normpath(posixpath.join(directory, path))
    if joined == ".." or joined.startswith("../") or joined == ".":
        return None
    return joined


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #

_BUILTIN_CRATES = frozenset({"std", "core", "alloc"})
_CRATE_ROOT_NAMES = ("lib.rs", "main.rs")
# Directories of a package whose files are crate roots of their own.
_TARGET_DIRS = ("src/bin", "tests", "examples", "benches")


@dataclass(frozen=True)
class _Layout:
    packages: frozenset[str]        # package directories, "" for the repository root
    target_roots: frozenset[str]    # crate roots named by manifests
    lib_roots: frozenset[str]       # library roots of named crates
    crates: dict[str, tuple[str, ...]]  # crate name -> library roots


@dataclass(frozen=True)
class _Position:
    """A module: the directory of its child module files and, if known, its own file."""

    directory: str
    file: Optional[str]   # known for the importing file's own module and for crate roots
    root: bool            # a crate root has no parent


class _RustModules:
    """The indexed ``.rs`` files and the packages of the repository at ``root``."""

    def __init__(self, files: list[FileRecord], root: Optional[str]) -> None:
        self.files = frozenset(f.path for f in files if f.path.endswith(".rs"))
        self.root = Path(root) if root else None
        dirs: dict[str, set[str]] = {}
        for path in self.files:
            directory, _, base = path.rpartition("/")
            dirs.setdefault(directory, set()).add(base)
        self.dirs = dirs
        ancestors: set[str] = set()
        for directory in dirs:
            while directory not in ancestors:
                ancestors.add(directory)
                if not directory:
                    break
                directory = posixpath.dirname(directory)
        self.ancestors = ancestors
        # Without manifests: the repository root and the parent of a crate's src/.
        guessed = {""}
        for directory, names in dirs.items():
            if posixpath.basename(directory) == "src" and any(n in names for n in _CRATE_ROOT_NAMES):
                guessed.add(posixpath.dirname(directory))
        self.guessed_packages = frozenset(guessed)
        self._modules: Optional[_ModuleMap] = None

    def resolve(self, edge: ImportEdge) -> Optional[str]:
        if self._modules is None:  # manifests are read on the first Rust edge only
            self._modules = _ModuleMap(self, self._layout())
        return self._modules.resolve(edge)

    def _layout(self) -> _Layout:
        packages = set(self.guessed_packages)
        target_roots: set[str] = set()
        lib_roots: set[str] = set()
        crates: dict[str, list[str]] = {}
        root = self._resolved_root()
        if root is not None:
            for directory in sorted(self.ancestors):
                manifest = _package_manifest(root, directory)
                if manifest is None or not manifest.is_package:
                    continue
                packages.add(directory)
                lib = _join_relative(directory, manifest.lib_path or "src/lib.rs")
                if lib is not None and lib in self.files:
                    target_roots.add(lib)
                    lib_roots.add(lib)
                    names = {n.replace("-", "_") for n in (manifest.package_name, manifest.lib_name) if n}
                    for name in sorted(names):
                        crates.setdefault(name, []).append(lib)
                for path in manifest.target_paths:
                    target = _join_relative(directory, path)
                    if target is not None and target in self.files:
                        target_roots.add(target)
        return _Layout(
            frozenset(packages),
            frozenset(target_roots),
            frozenset(lib_roots),
            {name: tuple(libs) for name, libs in crates.items()},
        )

    def _resolved_root(self) -> Optional[Path]:
        if self.root is None:
            return None
        try:
            root = self.root.resolve()
        except OSError:
            return None
        return root if root.is_dir() else None


def _package_manifest(root: Path, directory: str) -> Optional[_Manifest]:
    """The ``Cargo.toml`` of ``directory`` (repo-relative) under the resolved ``root``."""
    path = root.joinpath(*directory.split("/"), "Cargo.toml") if directory else root / "Cargo.toml"
    if not path.is_file():
        return None
    try:  # a link that leads out of the repository is not its manifest
        path.resolve().relative_to(root)
    except (OSError, ValueError):
        return None
    return _read_manifest(path)


class _ModuleMap:
    """Rust's module tree over the indexed files, for one layout."""

    def __init__(self, modules: _RustModules, layout: _Layout) -> None:
        self.files = modules.files
        self.dirs = modules.dirs
        self.layout = layout
        self.target_dirs = frozenset(posixpath.dirname(t) for t in layout.target_roots)
        self._package_of: dict[str, Optional[str]] = {}
        self._crate_dir: dict[str, Optional[str]] = {}

    # -- module tree --------------------------------------------------------- #

    def package_of(self, directory: str) -> Optional[str]:
        if directory in self._package_of:
            return self._package_of[directory]
        current = directory
        found: Optional[str] = None
        while True:
            if current in self.layout.packages:
                found = current
                break
            if not current:
                break
            current = posixpath.dirname(current)
        self._package_of[directory] = found
        return found

    def is_crate_root(self, path: str) -> bool:
        directory, _, base = path.rpartition("/")
        if path in self.layout.target_roots or base in _CRATE_ROOT_NAMES:
            return True
        if base == "mod.rs":
            return False
        package = self.package_of(directory)
        if package is None:
            return False
        if base == "build.rs" and directory == package:
            return True
        return directory in (posixpath.join(package, d) for d in _TARGET_DIRS)

    def crate_dir(self, path: str) -> Optional[str]:
        """Directory of the crate root the file's module hangs under, if any."""
        if path in self._crate_dir:
            return self._crate_dir[path]
        directory = posixpath.dirname(path)
        found: Optional[str] = None
        if self.is_crate_root(path):
            found = directory
        else:
            package = self.package_of(directory)
            stop = package if package is not None else ""
            special = (
                {posixpath.join(package, d) for d in ("src", *_TARGET_DIRS)} if package is not None else set()
            )
            while True:
                names = self.dirs.get(directory, ())
                if (
                    any(n in names for n in _CRATE_ROOT_NAMES)
                    or directory in self.target_dirs
                    or directory in special
                ):
                    found = directory
                    break
                if directory == stop or not directory:
                    break
                directory = posixpath.dirname(directory)
        self._crate_dir[path] = found
        return found

    def root_file(self, crate_dir: str, importer: str) -> Optional[str]:
        """The crate root file of ``crate_dir`` as seen from ``importer``."""
        if posixpath.dirname(importer) == crate_dir and self.is_crate_root(importer):
            return importer
        libs = sorted(t for t in self.layout.lib_roots if posixpath.dirname(t) == crate_dir)
        if libs:
            return libs[0]
        for name in _CRATE_ROOT_NAMES:
            candidate = posixpath.join(crate_dir, name)
            if candidate in self.files:
                return candidate
        others = [t for t in self.layout.target_roots if posixpath.dirname(t) == crate_dir]
        return others[0] if len(others) == 1 else None

    def own_position(self, path: str) -> _Position:
        """The module of the file ``path`` itself."""
        root = self.is_crate_root(path)
        directory = posixpath.dirname(path) if root or path.endswith("/mod.rs") or path == "mod.rs" else path[:-3]
        return _Position(directory, path, root)

    def parent(self, position: _Position, importer: str) -> Optional[_Position]:
        if position.root or not position.directory:
            return None
        directory = posixpath.dirname(position.directory)
        crate_dir = self.crate_dir(importer)
        if crate_dir is not None and directory == crate_dir:
            return _Position(crate_dir, self.root_file(crate_dir, importer), True)
        return _Position(directory, None, False)

    def module_file(self, position: _Position) -> Optional[str]:
        if position.file is not None:
            return position.file
        if not position.directory:
            return None
        for candidate in (position.directory + ".rs", position.directory + "/mod.rs"):
            if candidate in self.files:
                return candidate
        return None

    # -- edges --------------------------------------------------------------- #

    def resolve(self, edge: ImportEdge) -> Optional[str]:
        segments = edge.dst_module.split("::")
        importer = edge.src_path
        first = segments[0]
        if first == "":  # ::name, always another crate
            return self._in_crate(segments[1:], edge.names, importer) if len(segments) > 1 else None
        if first in _BUILTIN_CRATES:
            return None
        if first in ("crate", "$crate"):
            crate_dir = self.crate_dir(importer)
            if crate_dir is None:
                return None
            position: Optional[_Position] = _Position(crate_dir, self.root_file(crate_dir, importer), True)
            rest = segments[1:]
        elif first in ("self", "super"):
            position = self.own_position(importer)
            rest = segments[1:] if first == "self" else segments
            while rest and rest[0] == "super":
                position = self.parent(position, importer)
                if position is None:
                    return None
                rest = rest[1:]
        else:
            return self._in_crate(segments, edge.names, importer)
        return self._most_specific(position, rest, edge.names)

    def _in_crate(self, segments: list[str], names: list[str], importer: str) -> Optional[str]:
        if not segments:
            return None
        libs = self.layout.crates.get(segments[0])
        if not libs:
            return None
        lib = min(libs, key=lambda lib: (-_shared_prefix(lib, importer), lib))
        position = _Position(posixpath.dirname(lib), lib, True)
        return self._most_specific(position, segments[1:], names)

    def _most_specific(self, position: _Position, rest: list[str], names: list[str]) -> Optional[str]:
        for name in rest:
            if not name or name in ("self", "super", "crate"):
                return None
            position = _Position(posixpath.join(position.directory, name), None, False)
        if len(names) == 1 and names[0] and names[0] != "*":
            child = _Position(posixpath.join(position.directory, names[0]), None, False)
            hit = self.module_file(child)
            if hit is not None:
                return hit
        return self.module_file(position)


def _shared_prefix(a: str, b: str) -> int:
    count = 0
    for x, y in zip(a.split("/"), b.split("/")):
        if x != y:
            break
        count += 1
    return count


def rust_resolver(files: list[FileRecord], root: Optional[str] = None) -> Resolver:
    """Resolver for edges of :func:`extract_rust`, over the ``.rs`` files of ``files``.

    ``root`` is the absolute local path of the checkout (``RepoIndex.root``); its
    ``Cargo.toml`` files name the packages and crates. Without it no edge resolves
    into another crate.
    """
    return _RustModules(files, root).resolve

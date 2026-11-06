"""Dart support for :mod:`repofinder.parser`.

The grammar comes from ``tree-sitter-language-pack``, an optional dependency
(``repofinder[dart]``). Without the package ``dart`` is not registered and
``.dart`` files are skipped like any other unknown extension. With the package
the grammar is fetched on first use, not on import: it is a native library the
pack downloads from its GitHub release once and caches on disk.

This module imports nothing from :mod:`repofinder.parser`, which imports it.

Symbol mapping: class, mixin, extension and extension type are ``"class"``, enum
is ``"enum"``, typedef is ``"type"``; top-level functions, getters and setters
are ``"function"``; methods, getters, setters, operators and constructors are
``"method"``. Fields, enum values and local functions are not symbols.

Member names: ``Foo.Foo``, ``Foo.named``, ``Foo.value=``, ``Foo.operator ==``. The
body of a primary constructor (``this;``) takes the name in the class header, so
``class Foo.raw(int a) { this; }`` gives ``Foo.raw``. Declarations with the same
qualname stay separate symbols: the first keeps the plain qualname and id, later
ones get ``#2``, ``#3`` in source order (``C.operator -`` and ``C.operator -#2``).
``name`` stays the plain name, so a search for it finds all of them.
"""

from __future__ import annotations

import posixpath
import re
import sys
import threading
from typing import TYPE_CHECKING, Iterator, Optional

from tree_sitter import Node, Tree

from .models import FileRecord, ImportEdge, Symbol

if TYPE_CHECKING:
    from .parser import Resolver, SourceFile

# The limits of repofinder.parser, which cannot be imported from here.
_DOCSTRING_MAX_CHARS = 400
_SIGNATURE_MAX_CHARS = 200


# --------------------------------------------------------------------------- #
# Grammar
# --------------------------------------------------------------------------- #


class DartGrammarUnavailable(RuntimeError):
    """The package is installed but the grammar cannot be loaded (no network on first use)."""


_load_lock = threading.Lock()
_language: object = None    # keeps the pack's Language object alive
_load_error: Optional[str] = None   # remembered so that a dead network is tried once per process

# PyCapsule keeps a pointer to its name instead of copying it.
_CAPSULE_NAME = b"tree_sitter.Language"


def dart_grammar_available() -> bool:
    """Whether the language pack can be imported. Never touches the network."""
    try:
        import tree_sitter_language_pack  # noqa: F401
    except Exception:  # noqa: BLE001 - ImportError, or OSError for a broken native module
        return False
    return True


def dart_grammar() -> object:
    """Grammar pointer for ``LanguageSpec.grammar``; the first call may download it.

    Raises :class:`DartGrammarUnavailable`. The first failure prints one warning
    to stderr; later calls in the process raise at once without retrying.
    """
    global _language, _load_error
    with _load_lock:
        if _load_error is not None:
            raise DartGrammarUnavailable(_load_error)
        try:
            import tree_sitter_language_pack as pack

            language = pack.get_language("dart")
            capsule = _language_capsule(language)
        except Exception as exc:  # noqa: BLE001 - every pack error means "no grammar"
            _load_error = f"cannot load the Dart grammar: {exc}"
            print(f"repofinder: {_load_error}; .dart files get no symbols", file=sys.stderr)
            raise DartGrammarUnavailable(_load_error) from exc
        _language = language
    return capsule


def _language_capsule(language: object) -> object:
    """Capsule around the pointer of a ``tree_sitter.Language``.

    The pack hands out ``Language`` objects, but ``Language()`` in the parser
    registry wants a capsule, like the ones from ``tree_sitter_python.language()``.
    An int is no option: it overflows C ``long`` on Windows. The pointer is only
    exposed as the hash; a wrong value would crash the process, hence the check
    against the id in the repr. The pack's grammars stay loaded for the whole
    process, so the capsule needs no destructor.
    """
    import ctypes

    pointer = hash(language)
    if f"id={pointer}," not in repr(language):
        raise TypeError(f"unexpected tree_sitter.Language layout: {language!r}")
    new_capsule = ctypes.PYFUNCTYPE(ctypes.py_object, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)(
        ("PyCapsule_New", ctypes.pythonapi)
    )
    return new_capsule(pointer, _CAPSULE_NAME, None)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

_CLASS_LIKE = {
    "class_definition": "class",
    "mixin_declaration": "class",
    "extension_declaration": "class",
    "extension_type_declaration": "class",
    "enum_declaration": "enum",
}
_BODIES = ("class_body", "enum_body", "extension_body")
_TOP_LEVEL_SIGNATURES = ("function_signature", "getter_signature", "setter_signature")
_CONSTRUCTOR_SIGNATURES = (
    "constructor_signature",
    "constant_constructor_signature",
    "factory_constructor_signature",
    "redirecting_factory_constructor_signature",
)
# ``this : super(1);`` / ``this { ... }``: the body of the class's primary constructor.
_PRIMARY_CONSTRUCTOR_BODY = "primary_constructor_body_signature"
# The ``.raw(int a)`` of ``class const Foo.raw(int a)``; extension types have their own node.
_PRIMARY_CONSTRUCTOR_HEADERS = ("primary_constructor", "representation_declaration")
_MEMBER_SIGNATURES = (
    *_TOP_LEVEL_SIGNATURES,
    "operator_signature",
    *_CONSTRUCTOR_SIGNATURES,
    _PRIMARY_CONSTRUCTOR_BODY,
)
_COMMENTS = ("comment", "documentation_comment")
# Comments and annotations are not attached consistently: in front of functions
# and members they are previous siblings, class-like nodes and typedefs swallow
# the annotations (and whatever follows them) as their first children.
_PREAMBLE = ("annotation", *_COMMENTS)

_STRING_LITERAL_RE = re.compile(r"""^[rR]?('''|\"\"\"|'|")(.*)\1$""", re.DOTALL)
_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def extract_dart(source: SourceFile, tree: Tree) -> tuple[list[Symbol], list[ImportEdge]]:
    return _DartExtractor(source).run(tree)


class _DartExtractor:
    """Symbols and imports of one Dart file.

    Dart declarations nest only one level: members inside a class-like body.
    ``qualname`` is unique per file. Class-likes of the same name share the first
    one's symbol, so two ``extension on String`` blocks are one symbol with the
    members of both; other declarations of a taken qualname get a ``#2`` suffix.
    """

    def __init__(self, source: SourceFile) -> None:
        self.data = source.data
        self.path = source.rel_path
        self.symbols: dict[str, Symbol] = {}
        self.declarations: dict[str, int] = {}  # qualname -> how many were declared
        self.imports: list[ImportEdge] = []

    def run(self, tree: Tree) -> tuple[list[Symbol], list[ImportEdge]]:
        self._top_level(tree.root_node)
        return list(self.symbols.values()), self.imports

    def _top_level(self, parent: Node) -> None:
        for node in parent.children:
            ntype = node.type
            if ntype == "ERROR":  # recoverable code of a file with syntax errors
                self._top_level(node)
            elif ntype in ("import_or_export", "part_directive"):
                self._directive(node)
            elif ntype in _CLASS_LIKE:
                self._class_like(node)
            elif ntype == "type_alias":
                self._typedef(node)
            elif ntype in _TOP_LEVEL_SIGNATURES:
                self._top_level_function(node)

    # -- declarations -------------------------------------------------------- #

    def _class_like(self, node: Node) -> None:
        name = self._class_name(node)
        if name is None:
            return  # members of a nameless (broken) declaration have no qualname
        body = next((c for c in node.children if c.type in _BODIES), None)
        end = body.start_byte if body is not None else node.end_byte
        signature = self._signature(node, _header(node).start_byte, end)
        self._add(node, name, name, _CLASS_LIKE[node.type], None, signature, node)
        if body is not None:
            self._members(body, name)

    def _class_name(self, node: Node) -> Optional[str]:
        named = _name(node.child_by_field_name("name"))
        if named is not None:
            return named
        if node.type == "mixin_declaration":
            return _name(next((c for c in node.children if c.type == "identifier"), None))
        if node.type == "class_definition":  # class A = B with C;
            for child in node.children:
                if child.type == "mixin_application_class":
                    return _name(next((c for c in child.children if c.type == "identifier"), None))
            return None
        if node.type == "extension_declaration":  # the type of ``extension on X {``
            on = next((c for c in node.children if c.type == "on"), None)
            body = node.child_by_field_name("body")
            if on is None or body is None:
                return None
            target = self._signature(node, on.end_byte, body.start_byte)
            return f"on {target}" if target else None
        return None

    def _typedef(self, node: Node) -> None:
        # typedef A = int;  typedef A<T> = List<T>;  typedef int A(int x);
        name_node: Optional[Node] = None
        for child in node.children:
            if child.type in ("=", "formal_parameter_list"):
                break
            if child.type == "type_identifier":
                name_node = child
        name = _name(name_node)
        if name is not None:
            signature = self._signature(node, _header(node).start_byte, node.end_byte)
            self._add(node, name, name, "type", None, signature, node)

    def _top_level_function(self, node: Node) -> None:
        name = _member_name(node, "", self.data)
        if name is None:
            return
        previous = node.prev_sibling
        outer = previous if previous is not None and previous.type == "external" else node
        signature = self._signature(node, outer.start_byte, node.end_byte)
        self._add(outer, name, name, "function", None, signature, _end_node(node))

    def _members(self, parent: Node, owner: str) -> None:
        for node in parent.children:
            if node.type == "ERROR":
                self._members(node, owner)
            elif node.type in ("method_signature", "declaration"):
                # A declaration without a signature is a field.
                signature_node = next((c for c in node.children if c.type in _MEMBER_SIGNATURES), None)
                if signature_node is not None:
                    self._member(node, signature_node, owner)

    def _member(self, node: Node, signature_node: Node, owner: str) -> None:
        name = _member_name(signature_node, owner, self.data)
        if name is None:
            return
        if signature_node.type == _PRIMARY_CONSTRUCTOR_BODY:
            signature = self._primary_constructor_signature(node, signature_node)
        else:  # initializer lists are siblings of the signature: they stay out
            signature = self._signature(node, node.start_byte, signature_node.end_byte)
        self._add(node, name, f"{owner}.{name}", "method", owner, signature, _end_node(node))

    def _primary_constructor_signature(self, node: Node, signature_node: Node) -> str:
        """``this`` says nothing: the header of the class tells the constructor."""
        cls = _enclosing_class_like(signature_node)
        header = _primary_constructor_header(cls) if cls is not None else None
        if cls is None or header is None:  # ``this;`` without a primary constructor: broken code
            return self._signature(node, node.start_byte, signature_node.children[0].end_byte)
        const = next((c for c in cls.children if c.type == "const_builtin"), None)
        name = next((c for c in cls.children if c.type == "identifier"), header)
        start = const.start_byte if const is not None else name.start_byte
        return self._signature(cls, start, header.end_byte)

    def _add(
        self,
        outer: Node,
        name: str,
        qualname: str,
        kind: str,
        parent: Optional[str],
        signature: str,
        end_node: Node,
    ) -> None:
        count = self.declarations[qualname] = self.declarations.get(qualname, 0) + 1
        if count > 1:
            if kind in ("class", "enum"):
                return
            qualname = f"{qualname}#{count}"
        preamble = _preamble(outer)
        start_line = min([_line(outer), *(_line(p) for p in preamble if p.type == "annotation")])
        self.symbols[qualname] = Symbol(
            id=f"{self.path}::{qualname}",
            path=self.path,
            name=name,
            qualname=qualname,
            kind=kind,  # type: ignore[arg-type]
            start_line=start_line,
            end_line=_end_line(end_node),
            signature=signature,
            docstring=_docstring(preamble, _header(outer).start_point[0]),
            parent=parent,
        )

    # -- imports ------------------------------------------------------------- #

    def _directive(self, node: Node) -> None:
        """``import``, ``export`` and ``part``. A conditional directive gets one edge per URI."""
        uris: list[str] = []
        names: list[str] = []
        stack = list(reversed(node.children))
        while stack:
            child = stack.pop()
            if child.type == "uri":
                value = _string_value(child)
                if value:
                    uris.append(value)
            elif child.type == "combinator":
                if child.children and child.children[0].type == "show":
                    names.extend(_text(c) for c in child.children if c.type == "identifier")
            elif child.type in ("library_import", "library_export", "import_specification",
                                "configurable_uri", "configuration_uri"):
                stack.extend(reversed(child.children))
        for uri in uris:
            self.imports.append(ImportEdge(self.path, uri, None, list(names), _line(node)))

    # -- text ---------------------------------------------------------------- #

    def _signature(self, owner: Node, start: int, end: int) -> str:
        """One-line header text in [start, end) without comments, clipped."""
        header = self.data[start:end]
        if b"//" in header or b"/*" in header:
            pieces: list[bytes] = []
            cursor = start
            for comment in _iter_comments(owner, start, end):
                pieces.append(self.data[cursor:comment.start_byte])
                cursor = comment.end_byte
            pieces.append(self.data[cursor:end])
            header = b"".join(pieces)
        text = " ".join(header.decode("utf-8", errors="replace").split())
        # Multi-line parameter lists: "f({ a, b, })" -> "f({a, b})".
        text = re.sub(r"([(\[{])\s+", r"\1", text)
        text = re.sub(r",?\s*([)\]}])", r"\1", text)
        text = text.rstrip(";,").rstrip()
        return text[:_SIGNATURE_MAX_CHARS].rstrip()


def _member_name(signature: Node, owner: str, data: bytes) -> Optional[str]:
    """Symbol name of a signature node; ``owner`` is the enclosing class-like name."""
    stype = signature.type
    if stype in ("function_signature", "getter_signature"):
        return _name(signature.child_by_field_name("name"))
    if stype == "setter_signature":  # distinct from a getter of the same name
        name = _name(signature.child_by_field_name("name"))
        return f"{name}=" if name is not None else None
    if stype == "operator_signature":
        token = next((c for c in signature.children if c.type == "operator"), None)
        params = next((c for c in signature.children if c.type == "formal_parameter_list"), None)
        if token is None or params is None:
            return None
        symbol = "".join(data[token.end_byte:params.start_byte].decode("utf-8", errors="replace").split())
        return f"operator {symbol}" if symbol else None
    if stype in _CONSTRUCTOR_SIGNATURES:
        # Foo(), factory Foo(): "Foo"; Foo.named(), factory Foo.named(): "named".
        last: Optional[Node] = None
        for child in signature.children:
            if child.type == "formal_parameter_list":
                break
            if child.type == "identifier":
                last = child
        return _name(last) if last is not None else owner
    if stype == _PRIMARY_CONSTRUCTOR_BODY:
        cls = _enclosing_class_like(signature)
        header = _primary_constructor_header(cls) if cls is not None else None
        named = next((c for c in header.children if c.type == "identifier"), None) if header else None
        # ``class Foo.raw(int a)``: the identifier follows a dot; ``(int value)`` of
        # ``extension type Foo(int value)`` is a parameter, not a constructor name.
        if named is not None and named.prev_sibling is not None and named.prev_sibling.type == ".":
            return _name(named) or owner
        return owner
    return None


def _enclosing_class_like(node: Node) -> Optional[Node]:
    current = node.parent
    while current is not None and current.type not in _CLASS_LIKE:
        current = current.parent
    return current


def _primary_constructor_header(cls: Node) -> Optional[Node]:
    return next((c for c in cls.children if c.type in _PRIMARY_CONSTRUCTOR_HEADERS), None)


def _name(node: Optional[Node]) -> Optional[str]:
    if node is None or node.is_missing:
        return None
    return _text(node) or None


def _text(node: Node) -> str:
    raw = node.text
    return raw.decode("utf-8", errors="replace") if raw is not None else ""


def _string_value(uri: Node) -> Optional[str]:
    """Content of the string literal of a ``uri`` node."""
    match = _STRING_LITERAL_RE.match(_text(uri).strip())
    return match.group(2) if match else None


# NOTE: read tree-sitter Points by index, see the note in repofinder/parser.py.
def _line(node: Node) -> int:
    return node.start_point[0] + 1


def _end_line(node: Node) -> int:
    end_row, end_col = node.end_point
    # A node that ends at column 0 ends right after the previous line's newline.
    if end_col == 0 and end_row > node.start_point[0]:
        return end_row
    return end_row + 1


def _header(node: Node) -> Node:
    """First child after leading annotations and comments: where the header starts."""
    return next((c for c in node.children if c.type not in _PREAMBLE), node)


def _end_node(node: Node) -> Node:
    """The ``function_body`` or ``;`` that ends a declaration, else the node itself."""
    following = node.next_sibling
    while following is not None and following.type in _COMMENTS:
        following = following.next_sibling
    if following is not None and following.type in ("function_body", ";"):
        return following
    return node


def _preamble(node: Node) -> list[Node]:
    """Annotations and comments in front of a declaration, nearest first."""
    inner: list[Node] = []
    for child in node.children:
        if child.type not in _PREAMBLE:
            break
        inner.append(child)
    inner.reverse()
    previous = node.prev_sibling
    while previous is not None and previous.type in _PREAMBLE:
        inner.append(previous)
        previous = previous.prev_sibling
    return inner


def _docstring(preamble: list[Node], header_row: int) -> Optional[str]:
    """The ``///`` lines or ``/** */`` block in front of a declaration, unmarked.

    Annotations and ``//`` comments may sit in between, and one blank line may
    separate the comment from what it documents. Blocks of ``///`` lines
    separated by a blank line are not joined: only the nearest one counts.
    """
    lines: list[str] = []
    below = header_row
    for node in preamble:
        if node.type == "documentation_comment":
            text = _text(node)
            if not text.startswith("////"):  # //// is a plain comment for the analyzer
                if below - (_end_line(node) - 1) > (1 if lines else 2):
                    break
                if text.startswith("/**"):
                    if not lines:
                        lines = _block_comment_lines(text)
                    break
                lines.insert(0, text[3:].strip())
        below = node.start_point[0]
    doc = "\n".join(lines).strip()
    return doc[:_DOCSTRING_MAX_CHARS].rstrip() or None


def _block_comment_lines(text: str) -> list[str]:
    body = text[3:-2] if text.endswith("*/") else text[3:]
    lines: list[str] = []
    for raw in body.splitlines():
        line = raw.strip()
        lines.append(line[1:].strip() if line.startswith("*") else line)
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
# Resolution
# --------------------------------------------------------------------------- #


# Packages of the Flutter SDK: outside the repository, except in Flutter's own,
# where a directory of that name holds them. Without this ``package:flutter/widgets.dart``
# would match the ``lib/widgets.dart`` that many packages have.
_SDK_PACKAGES = frozenset(
    {
        "flutter",
        "flutter_driver",
        "flutter_localizations",
        "flutter_test",
        "flutter_web_plugins",
        "integration_test",
    }
)


class _DartPackages:
    """Which package each file belongs to, as far as the file paths tell.

    A package root is a directory with a ``lib/`` that holds Dart files; only the
    first ``lib`` of a path counts. ``pubspec.yaml`` is not among the indexed files
    and a resolver factory gets no repository root, so a package's declared name
    is unknown here. Instead ``package:NAME/PATH`` needs ``lib/PATH`` to exist in
    the first package this evidence allows:

    1. A directory named NAME, at any depth: the importer's own, then
       ``packages/NAME``, then ``NAME``, then the nearest one to the importer. A tie
       resolves to nothing. Only this step can resolve ``_SDK_PACKAGES``.
    2. The importer's own package, for a package whose directory is named otherwise.
       Not for ``package:NAME/NAME.dart`` from inside its ``lib/``: that is how a
       dependency is imported, and a package rarely imports its own library file
       from inside.
    3. The package at the repository root, from another package, when it looks like
       a library named NAME: it has ``lib/NAME.dart`` and no ``lib/main.dart``.

    Steps 2 and 3 are guesses. A dependency whose import path matches a local file
    resolves wrongly (``package:other/game.dart`` from a package with ``lib/game.dart``),
    and a package importing its own library file from ``lib/`` does not resolve at all.
    """

    def __init__(self, known: set[str] | frozenset[str]) -> None:
        self.known = known
        self.roots: set[str] = set()
        self.by_directory: dict[str, list[str]] = {}
        for path in known:
            root = _package_root(path)
            if root is not None and root not in self.roots:
                self.roots.add(root)
                self.by_directory.setdefault(posixpath.basename(root), []).append(root)

    def resolve(self, src_path: str, uri: str) -> Optional[str]:
        if uri.startswith("package:"):
            package, _, path = uri[len("package:"):].partition("/")
            if not package or not path:
                return None
            path = posixpath.normpath(path)
            if path == ".." or path.startswith("../"):
                return None
            return self._resolve_package(src_path, package, path)
        if _URI_SCHEME_RE.match(uri) or uri.startswith("/"):
            return None
        target = posixpath.normpath(posixpath.join(posixpath.dirname(src_path), uri))
        if target == ".." or target.startswith("../"):
            return None
        return target if target in self.known else None

    def _resolve_package(self, src_path: str, package: str, path: str) -> Optional[str]:
        owner = self._owner(src_path)
        named = [r for r in self.by_directory.get(package, ()) if _lib_file(r, path) in self.known]
        if named:
            root = _pick_named(named, package, owner, src_path)
            return _lib_file(root, path) if root is not None else None
        if package in _SDK_PACKAGES:
            return None
        library_file = f"{package}.dart"
        if owner is not None and _lib_file(owner, path) in self.known:
            from_inside = src_path.startswith(_lib_file(owner, ""))
            if not (path == library_file and from_inside):
                return _lib_file(owner, path)
        root_library = _lib_file("", library_file) in self.known and _lib_file("", "main.dart") not in self.known
        if owner != "" and root_library and _lib_file("", path) in self.known:
            return _lib_file("", path)
        return None

    def _owner(self, src_path: str) -> Optional[str]:
        """Package root of a file: in front of its ``lib``, else the nearest one above it."""
        root = _package_root(src_path)
        if root is not None:
            return root
        directory = posixpath.dirname(src_path)
        while directory not in self.roots:
            if not directory:
                return None
            directory = posixpath.dirname(directory)
        return directory


def _package_root(path: str) -> Optional[str]:
    """Directory in front of the first ``lib`` of a Dart file's path, "" for ``lib/x.dart``."""
    if not path.endswith(".dart"):
        return None
    parts = path.split("/")[:-1]
    return "/".join(parts[:parts.index("lib")]) if "lib" in parts else None


def _lib_file(root: str, path: str) -> str:
    return f"{root}/lib/{path}" if root else f"lib/{path}"


def _pick_named(roots: list[str], package: str, owner: Optional[str], src_path: str) -> Optional[str]:
    """Which of several directories named like the package is meant, None for a tie."""
    for root in (owner, f"packages/{package}", package):
        if root in roots:
            return root
    src_parts = src_path.split("/")

    def shared(root: str) -> int:
        count = 0
        for a, b in zip(root.split("/"), src_parts):
            if a != b:
                break
            count += 1
        return count

    ranked = sorted(roots, key=shared, reverse=True)
    if len(ranked) > 1 and shared(ranked[0]) == shared(ranked[1]):
        return None
    return ranked[0]


def resolve_dart_uri(src_path: str, uri: str, known: set[str] | frozenset[str]) -> Optional[str]:
    """Repo-relative file a Dart URI points at, or None.

    ``package:`` URIs follow the evidence described in :class:`_DartPackages`.
    Relative URIs are resolved against the importing file. Other schemes
    (``dart:``) and paths that leave the repository return None.
    """
    return _DartPackages(known).resolve(src_path, uri)


def dart_resolver(files: list[FileRecord]) -> Resolver:
    packages = _DartPackages(frozenset(f.path for f in files))
    return lambda edge: packages.resolve(edge.src_path, edge.dst_module)

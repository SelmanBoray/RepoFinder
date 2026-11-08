"""Java support for :mod:`repofinder.parser`.

The grammar comes from ``tree-sitter-java``. It is imported when the grammar is
first needed, not when this module is imported, so the rest of RepoFinder keeps
working without the package; parsing a ``.java`` file then raises ImportError.

This module imports nothing from :mod:`repofinder.parser`, which imports it.

Symbol mapping: ``class`` and ``record`` are ``"class"``, ``interface`` and
``@interface`` (annotation types) are ``"interface"``, ``enum`` is ``"enum"``.
Methods, constructors and compact record constructors are ``"method"``; a
constructor is named after its type (``Gson.Gson``). Nested types and their
members take the dotted path of the enclosing types (``Outer.Inner.m``, parent
``Outer.Inner``). Methods at the top level of a compact source file (Java 25,
``void main() {}`` outside any class) are ``"function"`` with no parent. Fields,
enum constants and their bodies, initializer blocks, annotation type elements,
anonymous classes, local classes and records in method bodies and lambdas are
not symbols. Overloads share one qualname and one id, and all of them are kept.

Ranges start at the first annotation of a declaration and end at its closing
``}`` (or ``;``); the Javadoc above is not part of them. Signatures are the
header without annotations and comments, up to the body's ``{`` or the ``;`` of
a bodiless method. Docstrings come from the ``/** */`` comment in front of the
declaration, the way ``javac`` attaches it: ``//`` and ``/* */`` comments in
between are skipped. A Javadoc placed after the annotations, which ``javac``
ignores, is used only when there is none in front. The docstring is the main
description: it stops at the first blank line or block tag (``@param``).

Imports: ``import a.b.C;`` is ``("a.b", ["C"])``, ``import a.b.*;`` is
``("a.b", ["*"])``, ``import static a.b.C.m;`` is ``("a.b.C", ["m"])`` and
``import static a.b.C.*;`` is ``("a.b.C", ["*"])``. Package declarations,
module directives (``module-info.java``), module imports (``import module
java.base;``) and imports with a syntax error are not edges.

Resolution works on paths alone, since source roots (``src/main/java``) vary:
a class ``a.b.C`` is any file whose path ends with ``a/b/C.java``. See
:func:`java_resolver`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterator, Optional

from tree_sitter import Node, Tree

from .models import FileRecord, ImportEdge, Symbol

if TYPE_CHECKING:
    from .parser import Resolver, SourceFile

# The limits of repofinder.parser, which cannot be imported from here.
_DOCSTRING_MAX_CHARS = 400
_SIGNATURE_MAX_CHARS = 200


def java_grammar() -> object:
    """Grammar pointer for ``LanguageSpec.grammar``."""
    import tree_sitter_java

    return tree_sitter_java.language()


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

_TYPE_KINDS = {
    "class_declaration": "class",
    "record_declaration": "class",
    "interface_declaration": "interface",
    "annotation_type_declaration": "interface",
    "enum_declaration": "enum",
}
_METHODS = ("method_declaration", "constructor_declaration", "compact_constructor_declaration")
_CONSTRUCTORS = ("constructor_declaration", "compact_constructor_declaration")
_COMMENTS = ("line_comment", "block_comment")
_ANNOTATIONS = ("annotation", "marker_annotation")
# ``import module java.base;`` (Java 25), not ``import module.foo.Bar;`` of a package named module.
_MODULE_IMPORT_RE = re.compile(rb"import\s+module\s+[^\s.]")


def extract_java(source: SourceFile, tree: Tree) -> tuple[list[Symbol], list[ImportEdge]]:
    return _JavaExtractor(source).run(tree)


class _JavaExtractor:
    """Symbols and imports of one Java file.

    The walk visits type bodies only: method bodies, field initializers and enum
    constants are never entered, which keeps local and anonymous classes out.
    ERROR nodes are entered in the scope they appear in, so a file with syntax
    errors keeps whatever declarations tree-sitter recovered.
    """

    def __init__(self, source: SourceFile) -> None:
        self.data = source.data
        self.path = source.rel_path
        self.symbols: list[Symbol] = []
        self.imports: list[ImportEdge] = []

    def run(self, tree: Tree) -> tuple[list[Symbol], list[ImportEdge]]:
        self._scope(tree.root_node, None, top_level=True)
        return self.symbols, self.imports

    def _scope(self, parent: Node, owner: Optional[str], top_level: bool = False) -> None:
        """Declarations among the children of ``parent``; ``owner`` is the enclosing type.

        The Javadoc of a declaration is tracked on the way, not looked up from the
        declaration backwards: ``Node.prev_sibling`` scans from the parent's first
        child, which made a long run of comments quadratic.
        """
        javadoc: Optional[Node] = None  # the nearest Javadoc since the last non-comment
        for node in parent.children:
            ntype = node.type
            if ntype in _COMMENTS:
                if _is_javadoc(node):
                    javadoc = node
                continue  # ``//`` and ``/* */`` between Javadoc and declaration are skipped
            if ntype == "ERROR":
                # Not top_level: a method in a broken top-level region most likely
                # belongs to a class whose header is broken, not to a compact file.
                self._scope(node, owner)
            elif ntype in _TYPE_KINDS:
                self._type(node, owner, javadoc)
            elif ntype in _METHODS:
                if owner is not None:
                    self._method(node, owner, javadoc)
                elif top_level and ntype == "method_declaration":
                    self._function(node, javadoc)
            elif ntype == "enum_body_declarations":  # the members after the constants
                self._scope(node, owner)
            elif ntype == "import_declaration" and owner is None:
                self._import(node)
            javadoc = None

    # -- declarations -------------------------------------------------------- #

    def _type(self, node: Node, owner: Optional[str], javadoc: Optional[Node]) -> None:
        name = _name(node.child_by_field_name("name"))
        if name is None:
            return  # members of a nameless (broken) declaration have no qualname
        qualname = f"{owner}.{name}" if owner is not None else name
        body = node.child_by_field_name("body")
        end = body.start_byte if body is not None else node.end_byte
        self._add(node, name, qualname, _TYPE_KINDS[node.type], owner, end, javadoc)
        if body is not None:
            self._scope(body, qualname)

    def _method(self, node: Node, owner: str, javadoc: Optional[Node]) -> None:
        name = _name(node.child_by_field_name("name"))
        if name is None:
            if node.type not in _CONSTRUCTORS:
                return
            name = owner.rsplit(".", 1)[-1]
        self._add(node, name, f"{owner}.{name}", "method", owner, _header_end(node), javadoc)

    def _function(self, node: Node, javadoc: Optional[Node]) -> None:
        name = _name(node.child_by_field_name("name"))
        if name is not None:
            self._add(node, name, name, "function", None, _header_end(node), javadoc)

    def _add(
        self,
        node: Node,
        name: str,
        qualname: str,
        kind: str,
        parent: Optional[str],
        header_end: int,
        javadoc: Optional[Node],
    ) -> None:
        self.symbols.append(
            Symbol(
                id=f"{self.path}::{qualname}",
                path=self.path,
                name=name,
                qualname=qualname,
                kind=kind,  # type: ignore[arg-type]
                start_line=node.start_point[0] + 1,  # the modifiers, annotations included
                end_line=_end_line(node),
                signature=self._signature(node, header_end),
                docstring=_docstring(node, javadoc),
                parent=parent,
            )
        )

    # -- imports ------------------------------------------------------------- #

    def _import(self, node: Node) -> None:
        # A broken import gives no edge: grammar 0.23.5 reads the Java 25 module
        # import ``import module java.base;`` as ``module.<ERROR java>.base``.
        if node.has_error or _MODULE_IMPORT_RE.match(self.data, node.start_byte, node.end_byte):
            return
        target = next((c for c in node.children if c.type in ("scoped_identifier", "identifier")), None)
        if target is None:
            return
        parts = list(_identifiers(target))
        if not parts or any(not p for p in parts):
            return  # an empty identifier
        if any(c.type == "asterisk" for c in node.children):
            module, names = ".".join(parts), ["*"]
        else:
            module, names = ".".join(parts[:-1]), [parts[-1]]
        self.imports.append(ImportEdge(self.path, module, None, names, node.start_point[0] + 1))

    # -- text ---------------------------------------------------------------- #

    def _signature(self, node: Node, end: int) -> str:
        """The header in front of ``end`` without annotations and comments, on one line."""
        keywords: list[str] = []
        rest_start: Optional[int] = None
        for child in node.children:
            if child.start_byte >= end:
                break
            if child.type == "modifiers":
                keywords = [_text(c) for c in child.children if c.type not in _ANNOTATIONS + _COMMENTS]
            elif child.type not in _COMMENTS:
                rest_start = child.start_byte
                break
        rest = ""
        if rest_start is not None and rest_start < end:
            pieces: list[bytes] = []
            cursor = rest_start
            for comment in _iter_comments(node, rest_start, end):
                pieces.append(self.data[cursor:comment.start_byte])
                cursor = comment.end_byte
            pieces.append(self.data[cursor:end])
            rest = b" ".join(pieces).decode("utf-8", errors="replace")
        text = " ".join(" ".join([*keywords, rest]).split())
        # Cosmetic for multi-line parameter lists: "Gson( Excluder e, int x )" -> "Gson(Excluder e, int x)".
        text = re.sub(r"([(\[])\s+", r"\1", text)
        text = re.sub(r"\s+([)\]])", r"\1", text)
        return text[:_SIGNATURE_MAX_CHARS].rstrip()


def _header_end(node: Node) -> int:
    """Where a method's header ends: its body's ``{``, else its ``;``, else its end."""
    body = node.child_by_field_name("body")
    if body is not None:
        return body.start_byte
    semicolon = next((c for c in reversed(node.children) if c.type == ";"), None)
    return semicolon.start_byte if semicolon is not None else node.end_byte


def _identifiers(node: Node) -> Iterator[str]:
    """The identifiers of a (scoped) name in source order; "" for a missing one."""
    if node.type == "identifier":
        yield "" if node.is_missing else _text(node)
        return
    for child in node.children:
        if child.type in ("identifier", "scoped_identifier"):
            yield from _identifiers(child)


def _name(node: Optional[Node]) -> Optional[str]:
    if node is None or node.is_missing:
        return None
    return _text(node) or None


def _text(node: Node) -> str:
    raw = node.text
    return raw.decode("utf-8", errors="replace") if raw is not None else ""


# NOTE: read tree-sitter Points by index, see the note in repofinder/parser.py.
def _end_line(node: Node) -> int:
    end_row, end_col = node.end_point
    # A node that ends at column 0 ends right after the previous line's newline.
    if end_col == 0 and end_row > node.start_point[0]:
        return end_row
    return end_row + 1


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


def _is_javadoc(node: Node) -> bool:
    if node.type != "block_comment":
        return False
    text = _text(node)
    return text.startswith("/**") and not text.startswith("/**/")


def _docstring(node: Node, javadoc: Optional[Node]) -> Optional[str]:
    """The cleaned Javadoc of a declaration, None without one.

    ``javadoc`` is the one in front of the declaration, found by the caller with
    comments in between skipped: what ``javac`` attaches. Failing that, a Javadoc
    among the modifiers, after annotations.
    """
    if javadoc is not None:
        return _clean_javadoc(_text(javadoc))
    modifiers = node.children[0] if node.children and node.children[0].type == "modifiers" else None
    if modifiers is not None:
        for child in reversed(modifiers.children):
            if _is_javadoc(child):
                return _clean_javadoc(_text(child))
    return None


def _clean_javadoc(text: str) -> Optional[str]:
    """Main description of a ``/** */`` comment: up to a blank line or a block tag."""
    body = text[3:-2] if text.endswith("*/") else text[3:]
    if body.endswith("*"):  # "/** foo **/"
        body = body.rstrip("*")
    lines: list[str] = []
    for raw in body.splitlines():
        line = raw.strip().lstrip("*").strip()
        if not line:
            if lines:
                break
            continue  # the empty first line of "/**\n * text"
        if line.startswith("@"):
            break
        lines.append(line)
    doc = "\n".join(lines)
    return doc[:_DOCSTRING_MAX_CHARS].rstrip() or None


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def _rank(path: str) -> tuple[bool, bool, int, str]:
    """Sort key among several matches: main sources, a real class file, shortest, alphabetical."""
    parts = path.split("/")
    under_test = any(part in ("test", "tests") for part in parts[:-1])
    info_file = parts[-1] in ("package-info.java", "module-info.java")
    return under_test, info_file, len(path), path


class _JavaFiles:
    """The ``.java`` files of an index, looked up by the end of their path.

    Java source roots differ between build layouts and modules (``src/``,
    ``src/main/java/``, ``gson/src/test/java/``), so ``a.b.C`` is any file whose
    path ends with ``a/b/C.java`` and the package ``a.b`` any directory whose path
    ends with ``a/b``. Where several match, :func:`_rank` picks one.
    """

    def __init__(self, files: list[FileRecord]) -> None:
        self.by_filename: dict[str, list[str]] = {}
        self.by_dirname: dict[str, dict[str, list[str]]] = {}  # last component -> directory -> files
        for path in sorted({f.path for f in files if f.path.endswith(".java")}):
            directory, _, filename = path.rpartition("/")
            self.by_filename.setdefault(filename, []).append(path)
            if directory:
                last = directory.rpartition("/")[2]
                self.by_dirname.setdefault(last, {}).setdefault(directory, []).append(path)
        self.cache: dict[tuple[str, str], Optional[str]] = {}

    def resolve(self, edge: ImportEdge) -> Optional[str]:
        module = edge.dst_module
        if not module or not edge.names:
            return None
        if module in ("java", "javax") or module.startswith(("java.", "javax.")):
            return None
        key = (module, edge.names[0])
        if key not in self.cache:
            self.cache[key] = self._resolve(module.replace(".", "/"), edge.names[0])
        return self.cache[key]

    def _resolve(self, module_path: str, name: str) -> Optional[str]:
        if name == "*":
            # import static a.b.C.*  /  import a.b.*
            return self._class_file(module_path) or self._package_file(module_path)
        # import a.b.C  /  import a.b.Outer.Inner (dst "a.b.Outer", class file Outer.java)
        return self._class_file(f"{module_path}/{name}") or self._class_file(module_path)

    def _class_file(self, class_path: str) -> Optional[str]:
        """The file whose path is ``<class_path>.java`` or ends with ``/<class_path>.java``."""
        target = f"{class_path}.java"
        filename = target.rpartition("/")[2]
        matches = [p for p in self.by_filename.get(filename, ()) if p == target or p.endswith(f"/{target}")]
        return min(matches, key=_rank) if matches else None

    def _package_file(self, package_path: str) -> Optional[str]:
        """A file directly in a directory whose path is ``package_path`` or ends with it."""
        last = package_path.rpartition("/")[2]
        matches = [
            path
            for directory, paths in self.by_dirname.get(last, {}).items()
            if directory == package_path or directory.endswith(f"/{package_path}")
            for path in paths
        ]
        return min(matches, key=_rank) if matches else None


def java_resolver(files: list[FileRecord], root: Optional[str] = None) -> Resolver:
    """Resolver for Java imports, from the paths of the ``.java`` files among ``files``.

    ``root`` (the checkout, passed to every resolver factory) is not used: Java
    resolution needs the paths alone.

    With ``M`` the imported module as a path (``a/b``) and ``N`` the first name:

    * ``N`` a class: the file ``M/N.java`` (a path equal to it or ending with
      ``/M/N.java``), else ``M.java``: ``M`` ends with an outer class when a nested
      class is imported (``import a.b.Outer.Inner``) and with the class of a static
      import (``import static a.b.C.m``).
    * ``N`` is ``*``: ``M.java`` (a static wildcard), else a file directly in the
      package directory ``M`` (a package wildcard).

    Of several matches, the first by :func:`_rank`: a file outside test
    directories (a directory component ``test`` or ``tests``), then the shortest
    path, then the alphabetically first. ``java.*`` and ``javax.*`` are never
    resolved, other packages only when the repository holds them.
    """
    return _JavaFiles(files).resolve

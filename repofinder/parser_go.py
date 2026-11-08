"""Go support for :mod:`repofinder.parser`.

The grammar is ``tree-sitter-go``; it is imported when the grammar is first
loaded, not when this module is imported. This module imports nothing from
:mod:`repofinder.parser`, which imports it.

Symbol mapping (top-level declarations only)::

    func Name(...)                     "function"   Name
    func (r *Recv[T]) Name(...)        "method"     Recv.Name    parent "Recv"
    type X struct {...}                "class"      X
    type X interface {...}             "interface"  X
      Method(...) inside its body      "method"     X.Method     parent "X"
    type X int / func(...) / A = B     "type"       X

A receiver's type name loses ``*``, parentheses and type parameters. Type specs of
a grouped ``type ( ... )`` count like single declarations. A single declaration's
range starts at its ``type`` keyword, a grouped spec's at the spec. Declarations
with the same qualname (several ``func init()``) are all kept, with the same id.

Not symbols: struct fields, embedded types and type elements of interfaces,
consts, vars, function literals, and anything declared inside a function body
(local types). An alias always is ``"type"``, also ``type A = struct{...}``, and
its interface methods, if any, are no symbols.

Signatures are the header up to the body's ``{`` (the block of a function, the
field list of a struct, the ``{`` of an interface), else the whole declaration,
with comments removed and whitespace collapsed to single spaces. Grouped specs get
the prefix ``type ``. Docstrings follow ``go/ast``: the comment group that ends on
the line right above the declaration and does not trail code, with ``//`` (and one
following space) or ``/* */`` removed and directive lines (``//go:generate``,
``//line``, ``//export``) dropped, cut at the first blank line.

Import edges: one per import spec, ``dst_module`` the unquoted path (escapes are
not decoded), ``names`` ``["*"]`` for a dot import and ``[]`` otherwise,
``dst_path`` None. :func:`go_resolver` resolves an edge to a file of the imported
package when the path belongs to the module of the nearest ``go.mod`` above the
importing file, which it reads from the repository on disk.

Limits: build constraints are ignored, so ``//go:build`` variants of one package
all give their symbols (gin has ``Binding`` in two files). Only ``go.mod`` files
inside the repository root count, links that lead out of it are not followed, and
without a root nothing resolves. Imports of
other modules stay unresolved even when they live in the same repository
(``replace`` directives, ``go.work``) or in ``vendor/``.
"""

from __future__ import annotations

import os
import posixpath
import re
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

from tree_sitter import Node, Tree

from .models import FileRecord, ImportEdge, Symbol

if TYPE_CHECKING:
    from .parser import Resolver, SourceFile

# The limits of repofinder.parser, which cannot be imported from here.
_DOCSTRING_MAX_CHARS = 400
_SIGNATURE_MAX_CHARS = 200

# go.mod files bigger than this are not read; a real one is a few KiB.
_GO_MOD_MAX_BYTES = 1 << 20


def go_grammar() -> object:
    """Grammar pointer for ``LanguageSpec.grammar``."""
    import tree_sitter_go

    return tree_sitter_go.language()


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

_TYPE_SPECS = ("type_spec", "type_alias")
_INTERFACE_METHODS = ("method_elem", "method_spec")  # method_spec: grammars before 0.21
_RECEIVER_PARAMETERS = ("parameter_declaration", "variadic_parameter_declaration")

# go/ast's isDirective: "//line ", "//extern ", "//export " and "//[a-z0-9]+:[a-z0-9]".
_DIRECTIVE_RE = re.compile(r"(?:line |extern |export )|[a-z0-9]+:[a-z0-9]")


def extract_go(source: SourceFile, tree: Tree) -> tuple[list[Symbol], list[ImportEdge]]:
    """Symbols and import edges of one Go file; edges get ``dst_path=None``."""
    extractor = _GoExtractor(source)
    extractor.walk(tree.root_node)
    return extractor.symbols, extractor.imports


class _GoExtractor:
    def __init__(self, source: SourceFile) -> None:
        self.data = source.data
        self.path = source.rel_path
        self.symbols: list[Symbol] = []
        self.imports: list[ImportEdge] = []

    # A declaration is passed with its parent's children and its index there: the
    # doc comment is found by looking back in that list. ``Node.prev_sibling`` would
    # do, but costs time linear in the number of siblings before the node.

    def walk(self, parent: Node) -> None:
        siblings = parent.children
        for index, node in enumerate(siblings):
            ntype = node.type
            if ntype == "ERROR":  # recoverable code of a file with syntax errors
                self.walk(node)
            elif ntype == "function_declaration":
                self._function(node, siblings, index)
            elif ntype == "method_declaration":
                self._method(node, siblings, index)
            elif ntype == "type_declaration":
                self._type_declaration(node, siblings, index)
            elif ntype in ("import_declaration", "import_spec_list"):
                self.walk(node)
            elif ntype == "import_spec":
                self._import_spec(node)

    # -- declarations -------------------------------------------------------- #

    def _function(self, node: Node, siblings: list[Node], index: int) -> None:
        name = _name(node.child_by_field_name("name"))
        if name is None:
            return
        signature = self._signature(node, node.start_byte, _body_start(node))
        self._add(node, name, name, "function", None, signature, self._docstring(siblings, index))

    def _method(self, node: Node, siblings: list[Node], index: int) -> None:
        receiver = _receiver_type_name(node.child_by_field_name("receiver"))
        name = _name(node.child_by_field_name("name"))
        if receiver is None or name is None:
            return
        signature = self._signature(node, node.start_byte, _body_start(node))
        docstring = self._docstring(siblings, index)
        self._add(node, name, f"{receiver}.{name}", "method", receiver, signature, docstring)

    def _type_declaration(self, node: Node, siblings: list[Node], index: int) -> None:
        children = node.children
        grouped = any(child.type == "(" for child in children)
        for spec, spec_siblings, spec_index in _type_specs(children):
            if grouped:
                self._type_spec(spec, spec, spec_siblings, spec_index, "type ")
            else:
                self._type_spec(spec, node, siblings, index, "")

    def _type_spec(self, spec: Node, outer: Node, siblings: list[Node], index: int, prefix: str) -> None:
        """``outer`` is the spec in a group, else the declaration: it gives range and doc."""
        name = _name(spec.child_by_field_name("name"))
        if name is None:
            return
        type_node = spec.child_by_field_name("type")
        kind = "type"
        body_start = spec.end_byte
        if spec.type == "type_spec" and type_node is not None:
            if type_node.type == "struct_type":
                kind = "class"
                body = next((c for c in type_node.children if c.type == "field_declaration_list"), None)
                body_start = body.start_byte if body is not None else type_node.end_byte
            elif type_node.type == "interface_type":
                kind = "interface"
                body = next((c for c in type_node.children if c.type == "{"), None)
                body_start = body.start_byte if body is not None else type_node.end_byte
        signature = self._signature(outer, outer.start_byte, body_start, prefix)
        self._add(outer, name, name, kind, None, signature, self._docstring(siblings, index))
        if kind == "interface" and type_node is not None:
            self._interface_methods(type_node, name)

    def _interface_methods(self, parent: Node, owner: str) -> None:
        siblings = parent.children
        for index, node in enumerate(siblings):
            if node.type == "ERROR":
                self._interface_methods(node, owner)
            elif node.type in _INTERFACE_METHODS:
                name = _name(node.child_by_field_name("name"))
                if name is not None:
                    signature = self._signature(node, node.start_byte, node.end_byte)
                    docstring = self._docstring(siblings, index)
                    self._add(node, name, f"{owner}.{name}", "method", owner, signature, docstring)

    def _add(
        self,
        node: Node,
        name: str,
        qualname: str,
        kind: str,
        parent: Optional[str],
        signature: str,
        docstring: Optional[str],
    ) -> None:
        self.symbols.append(
            Symbol(
                id=f"{self.path}::{qualname}",
                path=self.path,
                name=name,
                qualname=qualname,
                kind=kind,  # type: ignore[arg-type]
                start_line=_line(node),
                end_line=_end_line(node),
                signature=signature,
                docstring=docstring,
                parent=parent,
            )
        )

    # -- imports ------------------------------------------------------------- #

    def _import_spec(self, spec: Node) -> None:
        path_node = spec.child_by_field_name("path")
        if path_node is None or path_node.is_missing:
            return
        path = _unquote(_text(path_node))
        if not path:
            return
        name_node = spec.child_by_field_name("name")
        names = ["*"] if name_node is not None and name_node.type == "dot" else []
        self.imports.append(ImportEdge(self.path, path, None, names, _line(spec)))

    # -- text ---------------------------------------------------------------- #

    def _signature(self, owner: Node, start: int, end: int, prefix: str = "") -> str:
        """Header text in [start, end) without comments, on one line, clipped."""
        header = self.data[start:end]
        if b"//" in header or b"/*" in header:
            pieces: list[bytes] = []
            cursor = start
            for comment in _iter_comments(owner, start, end):
                pieces.append(self.data[cursor:comment.start_byte])
                cursor = comment.end_byte
            pieces.append(self.data[cursor:end])
            header = b" ".join(pieces)
        text = prefix + " ".join(header.decode("utf-8", errors="replace").split())
        return text[:_SIGNATURE_MAX_CHARS].rstrip()

    def _docstring(self, siblings: list[Node], index: int) -> Optional[str]:
        """The comment group right above ``siblings[index]``, as ``go/ast`` finds a doc comment.

        Comments belong to the group while each starts on the line after the one
        above it ends; the group must end on the line right above the declaration.
        A comment that trails code on its line is no part of it.
        """
        group: list[Node] = []
        below = siblings[index]
        position = index - 1
        while position >= 0 and siblings[position].type == "comment":
            previous = siblings[position]
            gap = self.data[previous.end_byte:below.start_byte]
            newlines = gap.count(b"\n")
            if gap.strip() or newlines > 1 or (not group and newlines != 1):
                break
            if not self._starts_line(previous):
                break
            group.append(previous)
            below = previous
            position -= 1
        if not group:
            return None
        lines: list[str] = []
        for comment in reversed(group):
            lines.extend(_comment_lines(_text(comment)))
        paragraph: list[str] = []
        for line in lines:
            if line:
                paragraph.append(line)
            elif paragraph:
                break
        doc = "\n".join(paragraph).strip()
        return doc[:_DOCSTRING_MAX_CHARS].rstrip() or None

    def _starts_line(self, node: Node) -> bool:
        line_start = self.data.rfind(b"\n", 0, node.start_byte) + 1
        return not self.data[line_start:node.start_byte].strip()


def _type_specs(children: list[Node]) -> Iterator[tuple[Node, list[Node], int]]:
    """Type specs among ``children``, each with its parent's children and its index."""
    for index, child in enumerate(children):
        if child.type in _TYPE_SPECS:
            yield child, children, index
        elif child.type == "ERROR":
            yield from _type_specs(child.children)


def _receiver_type_name(receiver: Optional[Node]) -> Optional[str]:
    """``Recv`` of ``(r *Recv)``, ``(Recv)``, ``(l *List[T])`` or ``((*Recv))``."""
    if receiver is None:
        return None
    parameter = next((c for c in receiver.named_children if c.type in _RECEIVER_PARAMETERS), None)
    if parameter is None:
        return None
    # The first type identifier in source order is the base type: pointer and
    # parenthesized types wrap it, type arguments follow it.
    stack = [parameter.child_by_field_name("type") or parameter]
    while stack:
        current = stack.pop()
        if current.type == "type_identifier":
            return _name(current)
        stack.extend(reversed(current.children))
    return None


def _body_start(node: Node) -> int:
    body = node.child_by_field_name("body")
    return body.start_byte if body is not None else node.end_byte


def _comment_lines(text: str) -> list[str]:
    """Lines of one comment without markers; directive lines are left out."""
    if text.startswith("//"):
        body = text[2:]
        if _DIRECTIVE_RE.match(body):
            return []
        if body.startswith(" "):
            body = body[1:]
        return [body.rstrip()]
    body = text[2:-2] if text.endswith("*/") and len(text) >= 4 else text[2:]
    lines: list[str] = []
    for raw in body.splitlines():
        line = raw.strip()
        if line.startswith("*"):  # " * text" style continuation lines
            line = line[1:]
            if line.startswith(" "):
                line = line[1:]
        lines.append(line.rstrip())
    return lines


def _unquote(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"`":
        return text[1:-1]
    return ""


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


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def go_resolver(files: list[FileRecord], root: Optional[str] = None) -> Resolver:
    """Resolver for Go import edges; ``root`` is the repository's local path.

    An import path resolves to a file only when it names a package of the module
    of the importing file: the nearest ``go.mod`` at or above the file's directory,
    looked for inside ``root`` only, with a ``module`` line. A path names a package
    of that module when it is the module path or starts with it plus ``/``; the
    rest of the path is the package directory below the ``go.mod``. A directory
    that has its own ``go.mod``, or lies below one between it and the module's,
    belongs to another module. The standard library, other modules (nested ones
    included) and files outside any module resolve to None, and so does every
    edge when ``root`` is None, empty or not a directory.

    The target is a file of ``files`` in the package directory itself, not in a
    subdirectory, that the go command would compile: ending in ``.go``, not in
    ``_test.go``, not starting with ``.`` or ``_``. Of those it is ``<name>.go``,
    where ``<name>`` is the last element of the import path without a major
    version (``/v2`` or ``.v2``), compared exactly and then ignoring case;
    otherwise the first in path order. A package without such a file resolves
    to None.

    ``go.mod`` files are read from disk when an edge first needs them and are
    cached for the life of the returned resolver. Only files that stay inside the
    repository once links are resolved are read, see :func:`_read_go_mod`.
    """
    repository = _repository_root(root)
    if repository is None:
        return lambda edge: None
    modules = _GoModules(repository)
    packages: dict[str, list[str]] = {}
    for record in files:
        path = record.path
        base = posixpath.basename(path)
        if path.endswith(".go") and not path.endswith("_test.go") and base[:1] not in (".", "_"):
            packages.setdefault(posixpath.dirname(path), []).append(path)
    for paths in packages.values():
        paths.sort()

    def resolve(edge: ImportEdge) -> Optional[str]:
        directory = modules.package_dir(edge.src_path, edge.dst_module)
        if directory is None:
            return None
        candidates = packages.get(directory)
        if not candidates:
            return None
        return _pick_file(candidates, edge.dst_module)

    return resolve


def _repository_root(root: Optional[str]) -> Optional[Path]:
    """The resolved repository directory; None for no root, ``""`` or no directory."""
    if not root:  # Path("") would be the working directory
        return None
    try:
        resolved = Path(root).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved if resolved.is_dir() else None


class _GoModules:
    """The ``go.mod`` files of one repository, read on first use and then cached."""

    def __init__(self, root: Path) -> None:
        self.root = root  # resolved
        self.go_mods: dict[str, tuple[bool, Optional[str]]] = {}  # directory -> has go.mod, module path
        self.modules: dict[str, Optional[tuple[str, Optional[str]]]] = {}  # importer directory -> module

    def package_dir(self, src_path: str, import_path: str) -> Optional[str]:
        """Repository directory of the package ``import_path`` names, seen from ``src_path``."""
        module = self._module_of(posixpath.dirname(src_path))
        if module is None or module[1] is None:
            return None
        module_dir, module_path = module
        if import_path == module_path:
            rest: list[str] = []
        elif import_path.startswith(module_path + "/"):
            rest = import_path[len(module_path) + 1:].split("/")
        else:
            return None
        # Not valid in an import path (empty, "..", leading dot), or not a plain
        # directory name: the path must stay below the go.mod.
        if not all(_plain_element(element) for element in rest):
            return None
        parts = module_dir.split("/") if module_dir else []
        for element in rest:
            parts.append(element)
            if self._go_mod(parts)[0]:
                return None
        return "/".join(parts)

    def _module_of(self, directory: str) -> Optional[tuple[str, Optional[str]]]:
        """``(module directory, module path)`` of the nearest ``go.mod`` at or above ``directory``.

        The module path is None for a ``go.mod`` without a readable ``module`` line:
        it still ends the search, as it would for the go command.
        """
        if directory in self.modules:
            return self.modules[directory]
        parts = directory.split("/") if directory else []
        found: Optional[tuple[str, Optional[str]]] = None
        if all(_plain_element(element) for element in parts):
            for n in range(len(parts), -1, -1):
                has_go_mod, module_path = self._go_mod(parts[:n])
                if has_go_mod:
                    found = "/".join(parts[:n]), module_path
                    break
        self.modules[directory] = found
        return found

    def _go_mod(self, parts: list[str]) -> tuple[bool, Optional[str]]:
        key = "/".join(parts)
        cached = self.go_mods.get(key)
        if cached is None:
            cached = self.go_mods[key] = _read_go_mod(self.root, self.root.joinpath(*parts, "go.mod"))
        return cached


def _plain_element(element: str) -> bool:
    return bool(element) and element[0] != "." and "\\" not in element and ":" not in element


_MODULE_LINE_RE = re.compile(r"""^module\s+(?:"([^"]*)"|`([^`]*)`|(\S+))$""")
_MODULE_BLOCK_RE = re.compile(r"^module\s*\($")


def _read_go_mod(root: Path, path: Path) -> tuple[bool, Optional[str]]:
    """Whether ``path`` is a regular file, and the module path it declares.

    ``root`` is the resolved repository root. The file is read only when its
    resolved path lies inside it: a ``go.mod`` that is, or is reached through, a
    link leading out of the repository counts as one whose module is unknown. It
    still ends the module above it, and nothing resolves against it. Accepted
    limit: a link swapped in between this check and the read is followed.
    """
    try:
        info = os.stat(path)
    except (OSError, ValueError):
        return False, None
    if not stat.S_ISREG(info.st_mode):
        return False, None
    if info.st_size > _GO_MOD_MAX_BYTES:
        return True, None
    try:
        real = path.resolve()
        real.relative_to(root)
    except (OSError, RuntimeError, ValueError):  # ValueError: outside the repository
        return True, None
    try:
        with open(real, "rb") as file:
            data = file.read(_GO_MOD_MAX_BYTES)
    except OSError:
        return True, None
    text = data.decode("utf-8", errors="replace").removeprefix("\ufeff")
    in_block = False
    for raw in text.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        if in_block:  # module ( path )
            match = _MODULE_LINE_RE.match(f"module {line}")
            return True, (next((g for g in match.groups() if g), None) if match else None)
        if _MODULE_BLOCK_RE.match(line):
            in_block = True
            continue
        match = _MODULE_LINE_RE.match(line)
        if match:
            return True, next((g for g in match.groups() if g), None)
    return True, None


_MAJOR_VERSION_RE = re.compile(r"v[0-9]+")
_GOPKG_VERSION_RE = re.compile(r"\.v[0-9]+$")


def _pick_file(candidates: list[str], import_path: str) -> str:
    elements = import_path.split("/")
    name = elements[-1]
    if len(elements) > 1 and _MAJOR_VERSION_RE.fullmatch(name):
        name = elements[-2]
    wanted = _GOPKG_VERSION_RE.sub("", name) + ".go"
    for path in candidates:
        if posixpath.basename(path) == wanted:
            return path
    folded = wanted.casefold()
    for path in candidates:
        if posixpath.basename(path).casefold() == folded:
            return path
    return candidates[0]

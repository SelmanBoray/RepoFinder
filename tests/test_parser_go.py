"""Tests for repofinder/parser_go.py (Go support), written from the contract.

The module is exercised directly (not through parser.LANGUAGES).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Optional

import tree_sitter

from repofinder.models import FileRecord, RepoIndex
from repofinder.parser import parse_file, resolve_imports
from repofinder.parser import SourceFile
from repofinder.parser_go import extract_go, go_grammar, go_resolver


def _extract(src: str, rel_path: str = "pkg/x.go", root: Optional[Path] = None):
    data = textwrap.dedent(src).lstrip("\n").encode("utf-8")
    tree = tree_sitter.Parser(tree_sitter.Language(go_grammar())).parse(data)
    root = root or Path("/repo")
    sf = SourceFile(root=root, abs_path=root / rel_path, rel_path=rel_path, data=data)
    return extract_go(sf, tree)


def _syms(src: str, rel_path: str = "pkg/x.go"):
    return _extract(src, rel_path)[0]


def _by_q(syms, qualname):
    found = [s for s in syms if s.qualname == qualname]
    assert len(found) == 1, f"{qualname}: {[s.qualname for s in syms]}"
    return found[0]


# --------------------------------------------------------------------------- symbols


def test_function_and_id():
    syms = _syms(
        """\
        package p

        func New(addr string) *Server {
            return nil
        }
        """
    )
    s = _by_q(syms, "New")
    assert s.kind == "function"
    assert s.name == "New"
    assert s.parent is None
    assert s.id == "pkg/x.go::New"
    assert s.path == "pkg/x.go"
    assert (s.start_line, s.end_line) == (3, 5)


def test_generic_function():
    s = _by_q(_syms("package p\n\nfunc Map[T any](xs []T) []T {\n\treturn xs\n}\n"), "Map")
    assert s.kind == "function"
    assert s.parent is None


def test_pointer_receiver_method():
    syms = _syms(
        """\
        package p

        type Server struct{}

        func (s *Server) Run() error { return nil }
        """
    )
    s = _by_q(syms, "Server.Run")
    assert s.kind == "method"
    assert s.name == "Run"
    assert s.parent == "Server"
    assert s.id == "pkg/x.go::Server.Run"
    assert s.signature == "func (s *Server) Run() error"
    assert (s.start_line, s.end_line) == (5, 5)


def test_generic_receiver_method():
    syms = _syms(
        """\
        package p

        type List[T any] struct{}

        func (l *List[T]) Len() int {
            return 0
        }
        """
    )
    s = _by_q(syms, "List.Len")
    assert s.kind == "method"
    assert s.parent == "List"
    _by_q(syms, "List")


def test_value_receiver_method():
    syms = _syms('package p\n\nfunc (s Server) Name() string {\n\treturn ""\n}\n')
    s = _by_q(syms, "Server.Name")
    assert s.kind == "method"
    assert s.parent == "Server"


def test_type_kinds():
    syms = _syms(
        """\
        package p

        type Server struct {
            Addr string
        }

        type Store interface {
            Close() error
        }

        type Handler func(*Server)

        type ID int

        type Alias = Server

        type Pair[K comparable, V any] struct {
            Key K
            Val V
        }
        """
    )
    assert _by_q(syms, "Server").kind == "class"
    assert _by_q(syms, "Store").kind == "interface"
    assert _by_q(syms, "Handler").kind == "type"
    assert _by_q(syms, "ID").kind == "type"
    assert _by_q(syms, "Alias").kind == "type"
    assert _by_q(syms, "Pair").kind == "class"
    for q in ("Server", "Store", "Handler", "ID", "Alias", "Pair"):
        assert _by_q(syms, q).parent is None


def test_struct_range_and_signature():
    syms = _syms(
        """\
        package p

        // Server serves requests.
        type Server struct {
            Addr string
            Port int
        }
        """
    )
    s = _by_q(syms, "Server")
    assert s.start_line == 4  # the doc comment is not part of the range
    assert s.end_line == 7
    assert s.signature == "type Server struct"
    assert s.docstring == "Server serves requests."


def test_grouped_types():
    syms = _syms(
        """\
        package p

        type (
            // A is a struct.
            A struct {
                X int
            }

            B interface {
                M()
            }
            C int
        )
        """
    )
    a = _by_q(syms, "A")
    b = _by_q(syms, "B")
    c = _by_q(syms, "C")
    assert (a.kind, b.kind, c.kind) == ("class", "interface", "type")
    assert (a.start_line, a.end_line) == (5, 7)
    assert (b.start_line, b.end_line) == (9, 11)
    assert (c.start_line, c.end_line) == (12, 12)
    for s in (a, b, c):
        assert s.signature.startswith("type ")
        assert s.parent is None
    assert a.docstring == "A is a struct."
    assert b.docstring is None
    assert c.docstring is None


def test_interface_methods():
    syms = _syms(
        """\
        package p

        type Store interface {
            Get(key string) (string, error)
            Put(key, val string) error
        }
        """
    )
    g = _by_q(syms, "Store.Get")
    assert g.kind == "method"
    assert g.name == "Get"
    assert g.parent == "Store"
    assert g.id == "pkg/x.go::Store.Get"
    assert g.start_line == 4
    assert _by_q(syms, "Store.Put").parent == "Store"
    assert _by_q(syms, "Store").kind == "interface"


def test_non_symbols():
    syms = _syms(
        """\
        package p

        import "fmt"

        const Pi = 3.14

        const (
            A = 1
            B = 2
        )

        var Version = "1"

        var handler = func(x int) int { return x }

        type Base struct{}

        type Derived struct {
            Base
            *Other
            Field int
        }

        func Real() {
            f := func() {}
            _ = f
        }
        """
    )
    assert sorted(s.qualname for s in syms) == ["Base", "Derived", "Real"]


def test_duplicate_init_both_kept():
    syms = _syms(
        """\
        package p

        func init() {
        }

        func init() {
        }
        """
    )
    inits = [s for s in syms if s.qualname == "init"]
    assert len(inits) == 2
    assert {s.id for s in inits} == {"pkg/x.go::init"}
    assert sorted(s.start_line for s in inits) == [3, 6]


def test_docstring_variants():
    syms = _syms(
        """\
        package p

        // One line doc.
        func A() {}

        // First line.
        // Second line.
        func B() {}

        // Para one.
        //
        // Para two.
        func C() {}

        func D() {}

        // Orphan comment.

        func E() {}
        """
    )
    assert _by_q(syms, "A").docstring == "One line doc."
    assert _by_q(syms, "B").docstring == "First line.\nSecond line."
    assert _by_q(syms, "C").docstring == "Para one."
    assert _by_q(syms, "D").docstring is None
    assert _by_q(syms, "E").docstring is None


def test_go_directive_not_in_docstring():
    syms = _syms(
        """\
        package p

        // Run does things.
        //go:noinline
        func Run() {}

        //go:generate stringer -type=K
        func Other() {}
        """
    )
    run = _by_q(syms, "Run")
    assert run.docstring is None or "go:" not in run.docstring
    other = _by_q(syms, "Other")
    assert other.docstring is None or "go:" not in other.docstring


def test_build_directive_file_header():
    syms = _syms(
        """\
        //go:build linux

        package p

        func F() {}
        """
    )
    f = _by_q(syms, "F")
    assert f.docstring is None
    assert f.start_line == 5


def test_signature_collapsed_and_bounded():
    params = ", ".join(f"arg{i} int" for i in range(60))
    src = (
        "package p\n\nfunc Long(\n\t" + params + ",\n) (\n\tint,\n\terror,\n) {\n"
        "\treturn 0, nil\n}\n"
    )
    s = _by_q(_syms(src), "Long")
    assert "\n" not in s.signature and "\t" not in s.signature
    assert "  " not in s.signature
    assert len(s.signature) <= 200
    assert s.signature.startswith("func Long(")
    assert "{" not in s.signature


def test_signature_excludes_body():
    s = _by_q(_syms("package p\n\nfunc F(a int) int {\n\treturn a\n}\n"), "F")
    assert s.signature.startswith("func F(a int) int")
    assert "return" not in s.signature
    assert "{" not in s.signature


# --------------------------------------------------------------------------- imports


def test_import_group():
    _, imports = _extract(
        """\
        package p

        import (
            "fmt"
            jsoniter "github.com/json-iterator/go"
            . "example.com/m/dot"
            _ "example.com/m/side"
        )
        """,
        rel_path="a/main.go",
    )
    by = {e.dst_module: e for e in imports}
    assert len(imports) == 4
    assert set(by) == {
        "fmt",
        "github.com/json-iterator/go",
        "example.com/m/dot",
        "example.com/m/side",
    }
    assert by["fmt"].names == []
    assert by["github.com/json-iterator/go"].names == []
    assert by["example.com/m/dot"].names == ["*"]
    assert by["example.com/m/side"].names == []
    assert by["fmt"].line == 4
    assert by["github.com/json-iterator/go"].line == 5
    assert by["example.com/m/dot"].line == 6
    assert by["example.com/m/side"].line == 7
    assert all(e.src_path == "a/main.go" for e in imports)


def test_single_import():
    _, imports = _extract('package p\n\nimport "os"\n')
    assert len(imports) == 1
    e = imports[0]
    assert e.dst_module == "os"
    assert e.names == []
    assert e.line == 3
    assert e.src_path == "pkg/x.go"


def test_no_imports():
    assert _extract("package p\n\nfunc F() {}\n")[1] == []


# --------------------------------------------------------------------------- resolution

MAIN_SRC = """\
package main

import (
    "example.com/m/sub"
    "fmt"
    "github.com/other/x"
)
"""


def _write(root: Path, rel: str, text: str = "package x\n") -> FileRecord:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    data = p.read_bytes()
    return FileRecord(rel, "go", len(data), text.count("\n"), True)


def _edges(root: Path, rel: str, src: str):
    return _extract(src, rel_path=rel, root=root)[1]


def _basic_repo(tmp_path: Path):
    (tmp_path / "go.mod").write_text("module example.com/m\n\ngo 1.21\n", encoding="utf-8")
    return [
        _write(tmp_path, "main.go", MAIN_SRC),
        _write(tmp_path, "sub/a.go"),
        _write(tmp_path, "sub/b.go"),
        _write(tmp_path, "sub/a_test.go"),
    ]


def _sub_edge(tmp_path):
    return next(
        e for e in _edges(tmp_path, "main.go", MAIN_SRC) if e.dst_module == "example.com/m/sub"
    )


def test_resolve_module_import(tmp_path):
    recs = _basic_repo(tmp_path)
    resolver = go_resolver(recs, str(tmp_path))
    res = {e.dst_module: resolver(e) for e in _edges(tmp_path, "main.go", MAIN_SRC)}
    sub = res["example.com/m/sub"]
    assert sub is not None
    assert sub.startswith("sub/") and sub.endswith(".go")
    assert not sub.endswith("_test.go")
    assert res["fmt"] is None
    assert res["github.com/other/x"] is None


def test_resolution_deterministic(tmp_path):
    recs = _basic_repo(tmp_path)
    edge = _sub_edge(tmp_path)
    first = go_resolver(recs, str(tmp_path))(edge)
    resolver = go_resolver(recs, str(tmp_path))
    assert all(resolver(edge) == first for _ in range(5))
    assert go_resolver(list(reversed(recs)), str(tmp_path))(edge) == first


def test_file_not_in_files_never_returned(tmp_path):
    recs = _basic_repo(tmp_path)
    edge = _sub_edge(tmp_path)
    only_b = [r for r in recs if r.path in ("main.go", "sub/b.go")]
    assert go_resolver(only_b, str(tmp_path))(edge) == "sub/b.go"  # sub/a.go exists on disk, not indexed
    without_b = [r for r in recs if r.path != "sub/b.go"]
    out = go_resolver(without_b, str(tmp_path))(edge)
    assert out is None or out in {r.path for r in without_b}
    no_sub = [r for r in recs if not r.path.startswith("sub/")]
    assert go_resolver(no_sub, str(tmp_path))(edge) is None


def test_nested_module(tmp_path):
    (tmp_path / "go.mod").write_text("module example.com/m\n", encoding="utf-8")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "go.mod").write_text("module example.com/tools\n", encoding="utf-8")
    src = 'package main\n\nimport "example.com/tools/lib"\n'
    recs = [
        _write(tmp_path, "tools/cmd/main.go", src),
        _write(tmp_path, "tools/lib/l.go"),
    ]
    edges = _edges(tmp_path, "tools/cmd/main.go", src)
    assert len(edges) == 1
    assert go_resolver(recs, str(tmp_path))(edges[0]) == "tools/lib/l.go"


def test_no_go_mod(tmp_path):
    recs = [
        _write(tmp_path, "main.go", MAIN_SRC),
        _write(tmp_path, "sub/a.go"),
        _write(tmp_path, "sub/b.go"),
    ]
    edges = _edges(tmp_path, "main.go", MAIN_SRC)
    assert len(edges) == 3
    resolver = go_resolver(recs, str(tmp_path))
    assert all(resolver(e) is None for e in edges)


def test_no_root_resolves_nothing(tmp_path):
    recs = _basic_repo(tmp_path)
    edge = _sub_edge(tmp_path)
    assert go_resolver(recs)(edge) is None
    assert go_resolver(recs, None)(edge) is None


def test_end_to_end_parse_file_and_resolve_imports(tmp_path):
    recs = _basic_repo(tmp_path)
    index = RepoIndex(name="m", root=str(tmp_path), source=str(tmp_path))
    for rec in recs:
        record, symbols, imports = parse_file(tmp_path, tmp_path / rec.path)
        assert record.language == "go"
        assert record.path == rec.path
        index.files.append(record)
        index.symbols.extend(symbols)
        index.imports.extend(imports)
    resolve_imports(index)
    by = {e.dst_module: e for e in index.imports if e.src_path == "main.go"}
    sub = by["example.com/m/sub"].dst_path
    assert sub is not None and sub.startswith("sub/") and sub.endswith(".go")
    assert not sub.endswith("_test.go")
    assert by["fmt"].dst_path is None
    assert by["github.com/other/x"].dst_path is None


def test_empty_root_behaves_like_none_even_with_cwd_go_mod(tmp_path, monkeypatch):
    recs = _basic_repo(tmp_path)
    edges = _edges(tmp_path, "main.go", MAIN_SRC)
    monkeypatch.chdir(tmp_path)
    resolver = go_resolver(recs, "")
    assert len(edges) == 3
    assert all(resolver(e) is None for e in edges)
    # sanity: the same layout does resolve with an explicit root
    sub = next(e for e in edges if e.dst_module == "example.com/m/sub")
    assert go_resolver(recs, str(tmp_path))(sub) is not None

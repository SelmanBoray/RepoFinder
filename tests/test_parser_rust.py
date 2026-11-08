"""Tests for repofinder/parser_rust.py (Rust symbols, imports, resolution).

Written from the contract, not from the implementation. The module is exercised
directly (it need not be registered in parser.LANGUAGES).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

tree_sitter = pytest.importorskip("tree_sitter")
pytest.importorskip("tree_sitter_rust")

from repofinder.models import FileRecord, ImportEdge, Symbol  # noqa: E402
from repofinder.parser import SourceFile  # noqa: E402
from repofinder.parser_rust import extract_rust, rust_grammar, rust_resolver  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _src(text: str) -> bytes:
    return textwrap.dedent(text).lstrip("\n").encode("utf-8")


def _extract(
    text: str, rel_path: str = "src/x.rs", root: Path | None = None
) -> tuple[list[Symbol], list[ImportEdge]]:
    data = _src(text)
    root = root if root is not None else Path("/nonexistent-repo-root")
    tree = tree_sitter.Parser(tree_sitter.Language(rust_grammar())).parse(data)
    sf = SourceFile(root=root, abs_path=root / rel_path, rel_path=rel_path, data=data)
    return extract_rust(sf, tree)


def _symbols(text: str, rel_path: str = "src/x.rs") -> dict[str, Symbol]:
    syms, _ = _extract(text, rel_path)
    return {s.qualname: s for s in syms}


def _imports(text: str) -> list[ImportEdge]:
    return _extract(text)[1]


def _pairs(text: str) -> list[tuple[str, list[str]]]:
    return [(e.dst_module, e.names) for e in _imports(text)]


# --------------------------------------------------------------------------- #
# symbol kinds and qualnames
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "decl",
    [
        "fn name() {}",
        "pub fn name() {}",
        "pub(crate) fn name() {}",
        "async fn name() {}",
        "const fn name() {}",
        "unsafe fn name() {}",
        'extern "C" fn name() {}',
        "pub async fn name() {}",
    ],
)
def test_free_function_variants(decl):
    syms, _ = _extract(decl + "\n")
    assert len(syms) == 1
    s = syms[0]
    assert s.kind == "function"
    assert s.qualname == "name"
    assert s.name == "name"
    assert s.parent is None
    assert s.path == "src/x.rs"
    assert s.id == "src/x.rs::name"


def test_struct_union_enum_trait_type_kinds():
    syms = _symbols(
        """
        struct S { a: u32 }
        union U { a: u32, b: f32 }
        enum E { A, B(u8), C { x: i32 } }
        trait T { fn a(&self); }
        type R = Result<(), E>;
        """
    )
    assert syms["S"].kind == "class"
    assert syms["U"].kind == "class"
    assert syms["E"].kind == "enum"
    assert syms["T"].kind == "interface"
    assert syms["R"].kind == "type"
    for q in ("S", "U", "E", "T", "R"):
        assert syms[q].parent is None
        assert syms[q].id == f"src/x.rs::{q}"


def test_generics_not_in_qualname():
    syms = _symbols(
        """
        struct Pair<K, V> { k: K, v: V }
        enum Opt<T> { No, Yes(T) }
        trait Tr<T> { fn go(&self); }
        fn id<T>(t: T) -> T { t }
        """
    )
    assert "Pair" in syms
    assert "Opt" in syms
    assert "Tr" in syms and "Tr.go" in syms
    assert "id" in syms
    assert not any("<" in q for q in syms)


def test_unit_and_tuple_struct():
    syms = _symbols("pub struct Unit;\nstruct Tup(u8, u8);\n")
    assert syms["Unit"].kind == "class"
    assert syms["Tup"].kind == "class"


def test_impl_methods_inherent():
    syms = _symbols(
        """
        struct S;
        impl S {
            pub fn new() -> S { S }
            fn helper(&self) {}
        }
        """
    )
    assert syms["S.new"].kind == "method"
    assert syms["S.new"].parent == "S"
    assert syms["S.new"].name == "new"
    assert syms["S.helper"].parent == "S"
    assert syms["S.helper"].id == "src/x.rs::S.helper"
    # the impl block itself is not a symbol
    assert set(syms) == {"S", "S.new", "S.helper"}


@pytest.mark.parametrize(
    "header,base",
    [
        ("impl<'a> Foo<'a>", "Foo"),
        ("impl fmt::Display for Bar", "Bar"),
        ("impl<T> From<T> for Wrapper<T>", "Wrapper"),
        ("impl Trait for &Foo", "Foo"),
        ("impl Plain", "Plain"),
        ("impl Trait for Baz", "Baz"),
    ],
)
def test_impl_base_name_rules(header, base):
    syms, _ = _extract(f"{header} {{\n    fn m(&self) {{}}\n}}\n")
    assert [s.qualname for s in syms] == [f"{base}.m"]
    assert syms[0].kind == "method"
    assert syms[0].parent == base


def test_two_trait_impls_same_qualname_same_id():
    syms, _ = _extract(
        """
        struct X;
        impl fmt::Display for X {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { Ok(()) }
        }
        impl fmt::Debug for X {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { Ok(()) }
        }
        """
    )
    fmts = [s for s in syms if s.qualname == "X.fmt"]
    assert len(fmts) == 2
    assert fmts[0].id == fmts[1].id == "src/x.rs::X.fmt"
    assert all(s.kind == "method" and s.parent == "X" for s in fmts)
    assert fmts[0].start_line != fmts[1].start_line


def test_trait_methods():
    syms = _symbols(
        """
        trait T {
            fn a(&self);
            fn b(&self) {}
        }
        """
    )
    assert syms["T"].kind == "interface"
    assert syms["T.a"].kind == "method" and syms["T.a"].parent == "T"
    assert syms["T.b"].kind == "method" and syms["T.b"].parent == "T"
    assert syms["T.a"].id == "src/x.rs::T.a"


def test_inline_mod_nesting():
    syms = _symbols(
        """
        mod tests {
            fn helper() {}
            struct S;
            impl S {
                fn m(&self) {}
            }
        }
        """
    )
    assert syms["tests"].kind == "namespace"
    assert syms["tests"].parent is None
    assert syms["tests.helper"].parent == "tests"
    assert syms["tests.helper"].kind == "function"
    assert syms["tests.S"].parent == "tests"
    assert syms["tests.S"].kind == "class"
    assert syms["tests.S.m"].parent == "tests.S"
    assert syms["tests.S.m"].kind == "method"


def test_nested_mods():
    syms = _symbols("mod a {\n    mod b {\n        fn f() {}\n    }\n}\n")
    assert "a" in syms and "a.b" in syms
    assert syms["a.b.f"].kind == "function"
    assert syms["a.b.f"].parent == "a.b"
    assert syms["a.b"].parent == "a"


def test_mod_declaration_is_not_a_symbol():
    syms, _ = _extract("mod m;\npub mod n;\n")
    assert syms == []


def test_non_symbols():
    syms = _symbols(
        """
        struct S { field: u32 }
        enum E { Variant, Other(u8) }
        const C: u32 = 1;
        static ST: u32 = 2;
        macro_rules! mac { () => {}; }
        fn outer() {
            fn inner() {}
            struct Local;
            impl Local { fn lm(&self) {} }
            let c = |x: u32| x + 1;
        }
        """
    )
    assert set(syms) == {"S", "E", "outer"}


# --------------------------------------------------------------------------- #
# ranges
# --------------------------------------------------------------------------- #


def test_range_includes_attributes_excludes_doc():
    src = """
        /// doc
        #[derive(Debug)]
        #[cfg(test)]
        struct S {
            a: u32,
        }
        """
    s = _symbols(src)["S"]
    assert s.start_line == 2
    assert s.end_line == 6


def test_range_attribute_only_starts_at_first_attribute():
    src = """
        fn before() {}

        #[derive(Debug)]
        #[cfg(test)]
        fn after() {
        }
        """
    s = _symbols(src)["after"]
    assert s.start_line == 3
    assert s.end_line == 6
    b = _symbols(src)["before"]
    assert (b.start_line, b.end_line) == (1, 1)


def test_range_semicolon_end_and_single_line():
    syms = _symbols("struct U;\ntype R = u8;\n\ntrait T {\n    fn a(&self);\n}\n")
    assert (syms["U"].start_line, syms["U"].end_line) == (1, 1)
    assert (syms["R"].start_line, syms["R"].end_line) == (2, 2)
    assert (syms["T"].start_line, syms["T"].end_line) == (4, 6)
    assert (syms["T.a"].start_line, syms["T.a"].end_line) == (5, 5)


def test_method_range_inside_impl():
    src = """
        impl S {
            #[inline]
            pub fn run(&self) -> u32 {
                0
            }
        }
        """
    s = _symbols(src)["S.run"]
    assert s.start_line == 2
    assert s.end_line == 5


# --------------------------------------------------------------------------- #
# signature
# --------------------------------------------------------------------------- #


def test_signature_method():
    src = """
        impl S {
            pub fn run(&self) -> u32 { 0 }
        }
        """
    assert _symbols(src)["S.run"].signature == "pub fn run(&self) -> u32"


def test_signature_struct():
    src = """
        pub struct Config {
            a: u32,
        }
        """
    assert _symbols(src)["Config"].signature == "pub struct Config"


def test_signature_has_no_attributes():
    src = """
        #[derive(Debug)]
        #[cfg(test)]
        pub struct Config {
            a: u32,
        }

        #[inline]
        fn f() {}
        """
    syms = _symbols(src)
    assert "#[" not in syms["Config"].signature
    assert "derive" not in syms["Config"].signature
    assert "#[" not in syms["f"].signature
    assert "fn f" in syms["f"].signature


# --------------------------------------------------------------------------- #
# docstrings
# --------------------------------------------------------------------------- #


def test_doc_single_line():
    src = """
        /// Runs it.
        fn run() {}
        """
    assert _symbols(src)["run"].docstring == "Runs it."


def test_doc_two_lines_joined():
    src = """
        /// First line.
        /// Second line.
        fn run() {}
        """
    assert _symbols(src)["run"].docstring == "First line.\nSecond line."


def test_doc_blank_doc_line_ends_it():
    # contract: a blank `///` line ends the docstring (the part nearest the item
    # or the part before the blank is kept; either way the text after the blank
    # must not be glued to the part before it).
    src = """
        /// Summary.
        ///
        /// Details here.
        fn run() {}
        """
    doc = _symbols(src)["run"].docstring
    assert doc is not None
    assert not ("Summary." in doc and "Details here." in doc)


def test_doc_before_attribute():
    src = """
        /// doc
        #[derive(Debug)]
        struct S;
        """
    assert _symbols(src)["S"].docstring == "doc"


def test_doc_on_method_and_trait_item():
    src = """
        trait T {
            /// Does a.
            fn a(&self);
        }
        impl S {
            /// Does b.
            fn b(&self) {}
        }
        """
    syms = _symbols(src)
    assert syms["T.a"].docstring == "Does a."
    assert syms["S.b"].docstring == "Does b."


def test_inner_doc_is_not_item_docstring():
    src = """
        //! Crate docs.
        //! More.
        fn run() {}
        """
    assert _symbols(src)["run"].docstring is None


def test_no_comment_docstring_none():
    assert _symbols("fn run() {}\n")["run"].docstring is None


def test_plain_comment_is_not_docstring():
    src = """
        // just a comment
        fn run() {}
        """
    assert _symbols(src)["run"].docstring is None


# --------------------------------------------------------------------------- #
# imports
# --------------------------------------------------------------------------- #


def test_import_simple_path():
    edges = _imports("use crate::a::b::C;\n")
    assert len(edges) == 1
    e = edges[0]
    assert e.src_path == "src/x.rs"
    assert e.dst_module == "crate::a::b"
    assert e.names == ["C"]
    assert e.line == 1


def test_import_braced_with_nested_path():
    assert _pairs("use std::collections::{HashMap, hash_map::Entry};\n") == [
        ("std::collections", ["HashMap"]),
        ("std::collections::hash_map", ["Entry"]),
    ]


def test_import_glob():
    assert _pairs("use super::util::*;\n") == [("super::util", ["*"])]


def test_import_alias_uses_original_name():
    assert _pairs("use a::B as C;\n") == [("a", ["B"])]


def test_import_bare_crate():
    assert _pairs("use regex;\n") == [("regex", [])]


def test_import_self_in_braces():
    assert _pairs("use a::{self, B};\n") == [("a", []), ("a", ["B"])]


def test_import_pub_use():
    assert _pairs("pub use crate::x::Y;\n") == [("crate::x", ["Y"])]


def test_import_mod_declaration():
    assert _pairs("mod m;\n") == [("self::m", [])]


def test_import_extern_crate():
    assert _pairs("extern crate libc;\n") == [("libc", [])]


def test_import_lines_and_src_path():
    text = """
        // header
        use a::B;

        mod m;
        use c::{
            D,
            E,
        };
        """
    syms, edges = _extract(text, rel_path="crates/z/src/lib.rs")
    assert all(e.src_path == "crates/z/src/lib.rs" for e in edges)
    by_mod = {(e.dst_module, tuple(e.names)): e.line for e in edges}
    assert by_mod[("a", ("B",))] == 2
    assert by_mod[("self::m", ())] == 4
    assert by_mod[("c", ("D",))] == 5
    assert by_mod[("c", ("E",))] == 5
    assert len(edges) == 4


def test_import_dst_path_unset_by_extractor():
    assert all(e.dst_path is None for e in _imports("use a::B;\nmod m;\n"))


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #


def _write(root: Path, rel: str, text: str = "") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text), encoding="utf-8")


def _records(root: Path) -> list[FileRecord]:
    recs = []
    for p in sorted(root.rglob("*.rs")):
        rel = p.relative_to(root).as_posix()
        data = p.read_bytes()
        recs.append(FileRecord(rel, "rust", len(data), data.count(b"\n") + 1, True))
    return recs


def _edges(root: Path, rel: str) -> list[ImportEdge]:
    p = root / rel
    data = p.read_bytes()
    tree = tree_sitter.Parser(tree_sitter.Language(rust_grammar())).parse(data)
    sf = SourceFile(root=root.resolve(), abs_path=p.resolve(), rel_path=rel, data=data)
    return extract_rust(sf, tree)[1]


def _resolve(
    root: Path, rel: str, files: list[FileRecord] | None = None, *, with_root: bool = True
) -> dict[tuple, str | None]:
    files = files if files is not None else _records(root)
    resolve = rust_resolver(files, str(root.resolve()) if with_root else None)
    return {(e.dst_module, tuple(e.names)): resolve(e) for e in _edges(root, rel)}


@pytest.fixture
def single_crate(tmp_path: Path) -> Path:
    _write(tmp_path, "Cargo.toml", '[package]\nname = "demo-app"\n')
    _write(
        tmp_path,
        "src/lib.rs",
        """
        mod printer;
        mod util;
        use crate::printer::Standard;
        use std::fmt::Display;
        """,
    )
    _write(tmp_path, "src/printer/mod.rs", "mod json;\npub struct Standard;\n")
    _write(tmp_path, "src/printer/json.rs", "pub struct Json;\n")
    _write(
        tmp_path,
        "src/util.rs",
        """
        use super::printer::json::Json;
        use crate::util2::X;
        """,
    )
    return tmp_path


def test_resolve_lib_rs(single_crate):
    r = _resolve(single_crate, "src/lib.rs")
    assert r[("self::printer", ())] == "src/printer/mod.rs"
    assert r[("self::util", ())] == "src/util.rs"
    assert r[("crate::printer", ("Standard",))] == "src/printer/mod.rs"
    assert r[("std::fmt", ("Display",))] is None


def test_resolve_mod_rs_child(single_crate):
    r = _resolve(single_crate, "src/printer/mod.rs")
    assert r[("self::json", ())] == "src/printer/json.rs"


def test_resolve_super_and_missing_module(single_crate):
    r = _resolve(single_crate, "src/util.rs")
    assert r[("super::printer::json", ("Json",))] == "src/printer/json.rs"
    # crate::util2 does not exist: unresolved, no fallback to the crate root
    assert r[("crate::util2", ("X",))] is None


def test_resolve_non_mod_file_child(tmp_path):
    _write(tmp_path, "Cargo.toml", '[package]\nname = "demo"\n')
    _write(tmp_path, "src/lib.rs", "mod a;\n")
    _write(tmp_path, "src/a.rs", "mod b;\n")
    _write(tmp_path, "src/a/b.rs", "pub fn f() {}\n")
    r = _resolve(tmp_path, "src/a.rs")
    assert r[("self::b", ())] == "src/a/b.rs"
    assert _resolve(tmp_path, "src/lib.rs")[("self::a", ())] == "src/a.rs"


def test_resolve_workspace_crate(tmp_path):
    _write(tmp_path, "crates/searcher/Cargo.toml", '[package]\nname = "grep-searcher"\n')
    _write(tmp_path, "crates/searcher/src/lib.rs", "pub struct Searcher;\n")
    _write(tmp_path, "crates/core/Cargo.toml", '[package]\nname = "ripgrep"\n')
    _write(
        tmp_path,
        "crates/core/src/main.rs",
        """
        use grep_searcher::Searcher;
        use serde::Deserialize;
        """,
    )
    r = _resolve(tmp_path, "crates/core/src/main.rs")
    assert r[("grep_searcher", ("Searcher",))] == "crates/searcher/src/lib.rs"
    assert r[("serde", ("Deserialize",))] is None


def test_resolve_never_returns_file_outside_files(single_crate):
    full = _records(single_crate)
    files = [f for f in full if f.path != "src/printer/json.rs"]
    r = _resolve(single_crate, "src/printer/mod.rs", files)
    assert r[("self::json", ())] is None
    r = _resolve(single_crate, "src/util.rs", files)
    assert r[("super::printer::json", ("Json",))] is None
    # unrelated edges still resolve
    r = _resolve(single_crate, "src/lib.rs", files)
    assert r[("self::util", ())] == "src/util.rs"

    # and with only the importing file known, nothing resolves
    only = [f for f in full if f.path == "src/lib.rs"]
    r = _resolve(single_crate, "src/lib.rs", only)
    assert all(v is None for v in r.values())


def test_cross_crate_needs_root_but_crate_edges_do_not(tmp_path):
    _write(tmp_path, "crates/searcher/Cargo.toml", '[package]\nname = "grep-searcher"\n')
    _write(tmp_path, "crates/searcher/src/lib.rs", "pub struct Searcher;\n")
    _write(tmp_path, "crates/core/Cargo.toml", '[package]\nname = "ripgrep"\n')
    _write(tmp_path, "crates/core/src/util.rs", "pub struct U;\n")
    _write(
        tmp_path,
        "crates/core/src/main.rs",
        "use grep_searcher::Searcher;\nuse crate::util::U;\n",
    )
    r = _resolve(tmp_path, "crates/core/src/main.rs", with_root=False)
    assert r[("grep_searcher", ("Searcher",))] is None
    assert r[("crate::util", ("U",))] == "crates/core/src/util.rs"
    r = _resolve(tmp_path, "crates/core/src/main.rs")
    assert r[("grep_searcher", ("Searcher",))] == "crates/searcher/src/lib.rs"


# --------------------------------------------------------------------------- #
# self/super normalization inside inline modules
# --------------------------------------------------------------------------- #


def _edges_of(text: str, rel: str = "src/printer.rs") -> list[tuple[str, list[str]]]:
    return [(e.dst_module, e.names) for e in _extract(text, rel)[1]]


def test_inline_mod_super_relative_to_file_module():
    assert _edges_of("mod tests {\n    use super::util::X;\n}\n") == [("self::util", ["X"])]


def test_inline_mod_super_glob():
    assert _edges_of("mod tests {\n    use super::*;\n}\n") == [("self", ["*"])]


def test_inline_mod_self_prefix_gains_module_path():
    assert _edges_of("mod a {\n    use self::b::Y;\n}\n") == [("self::a::b", ["Y"])]


def test_inline_nested_mod_double_super():
    text = "mod a {\n    mod b {\n        use super::super::z::W;\n    }\n}\n"
    assert _edges_of(text) == [("self::z", ["W"])]


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #


def test_end_to_end_build_index(tmp_path):
    from repofinder.ingest import build_index

    _write(tmp_path, "Cargo.toml", '[package]\nname = "demo"\n')
    _write(tmp_path, "src/lib.rs", "mod util;\npub fn run() {}\n")
    _write(tmp_path, "src/util.rs", "pub fn helper() {}\n")
    index = build_index(tmp_path, "demo", str(tmp_path))
    recs = {f.path: f for f in index.files}
    assert recs["src/lib.rs"].language == "rust"
    assert recs["src/util.rs"].language == "rust"
    edges = [e for e in index.imports if e.src_path == "src/lib.rs" and e.dst_module == "self::util"]
    assert len(edges) == 1
    assert edges[0].dst_path == "src/util.rs"
    assert any(s.qualname == "run" and s.kind == "function" for s in index.symbols)

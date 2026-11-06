"""Tests for the Dart side of repofinder.parser (repofinder/parser_dart.py).

Contract: language "dart", extension ".dart", no new SymbolKind. class / mixin /
extension / extension type -> "class", enum -> "enum", typedef -> "type",
top-level function/getter/setter -> "function", members (methods, getters,
setters, operators, constructors) -> "method".
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip(
    "tree_sitter_language_pack",
    reason="tree_sitter_language_pack is not installed; Dart support is disabled",
)

from repofinder import parser_dart  # noqa: E402
from repofinder.models import FileRecord, ImportEdge, RepoIndex, Symbol  # noqa: E402
from repofinder.parser import (  # noqa: E402
    _get_parser,
    extensions_for,
    language_for_path,
    parse_file,
    resolve_imports,
)
from repofinder.search import SearchIndex  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# Tests that run in separate processes do not need the grammar in this process.
_NO_GRAMMAR_NEEDED = {
    "test_without_ctypes_and_language_pack_python_and_typescript_work",
    "test_without_ctypes_the_grammar_is_refused_and_python_works",
    "test_without_language_pack_dart_is_skipped_and_python_works",
    "test_import_parser_does_not_touch_the_network",
    "test_grammar_download_failure_gives_empty_record_and_one_warning",
}


# The pack's words for "the release could not be reached" (download.rs, v1.23.1). Its other
# download errors, a changed bundle among them, mean something is wrong, not that we are offline.
_NETWORK_FAILURES = (
    "Failed to fetch manifest from",
    "Failed to read manifest body",
    "Failed to download ",
    "Failed to read download body",
)


@pytest.fixture(autouse=True)
def _require_grammar(request: pytest.FixtureRequest) -> None:
    if request.node.name in _NO_GRAMMAR_NEEDED or request.node.name.startswith("test_pin_"):
        return
    if not parser_dart.dart_grammar_available():
        pytest.skip("Dart grammar unavailable: tree_sitter_language_pack cannot be imported")
    try:
        parser_dart.dart_grammar()  # like any first use, on a clean machine this downloads the verified grammar
    except parser_dart.DartGrammarUnavailable as exc:
        _skip_or_fail_for_missing_grammar(exc)


def _skip_or_fail_for_missing_grammar(error: parser_dart.DartGrammarUnavailable) -> None:
    """Only a grammar that cannot be had may skip; a refused one must not look green."""
    import tree_sitter_language_pack as pack

    if pack.__version__ != parser_dart._PACK_VERSION:
        pytest.fail(f"unexpected tree-sitter-language-pack version: {error}")
    pin = parser_dart._GRAMMAR_PINS.get(parser_dart._platform_key())
    if pin is None:
        pytest.skip(f"Dart grammar unavailable (no pinned digest for this platform): {error}")
    if (Path(pack.cache_dir()) / pin[0]).exists():
        pytest.fail(f"the cached Dart grammar did not load: {error}")
    if any(failure in str(error) for failure in _NETWORK_FAILURES):
        pytest.skip(f"Dart grammar unavailable (download failed, no cache): {error}")
    pytest.fail(f"the Dart grammar was not downloaded, and not for lack of a network: {error}")


def _write(tmp_path: Path, source: str, name: str = "lib/a.dart") -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(textwrap.dedent(source).lstrip("\n").encode("utf-8"))  # LF on every OS
    return path


def _parse(tmp_path: Path, source: str, name: str = "lib/a.dart"):
    path = _write(tmp_path, source, name)
    return parse_file(tmp_path, path)


_SIGNATURES = {
    "function_signature",
    "getter_signature",
    "setter_signature",
    "operator_signature",
    "constructor_signature",
    "constant_constructor_signature",
    "factory_constructor_signature",
    "redirecting_factory_constructor_signature",
    "primary_constructor_body_signature",
}
_TOP_LEVEL_SIGNATURES = {"function_signature", "getter_signature", "setter_signature"}
_BODIES = {"class_body", "enum_body", "extension_body"}


def _count_declarations(data: bytes) -> int:
    """Functions and members in the syntax tree, counted without the extractor."""
    count = 0
    for top in _get_parser("dart").parse(data).root_node.children:
        if top.type in _TOP_LEVEL_SIGNATURES:
            count += 1
        for body in (c for c in top.children if c.type in _BODIES):
            for member in body.children:
                if member.type in ("method_signature", "declaration"):
                    count += any(c.type in _SIGNATURES for c in member.children)
    return count


def _syms(tmp_path: Path, source: str) -> dict[str, Symbol]:
    path = _write(tmp_path, source)
    record, symbols, _ = parse_file(tmp_path, path)
    assert record.parse_ok, record
    callables = [s for s in symbols if s.kind in ("function", "method")]
    assert len(callables) == _count_declarations(path.read_bytes()), sorted(s.id for s in callables)
    ids = [s.id for s in symbols]
    assert len(ids) == len(set(ids)), sorted(ids)
    return {s.qualname: s for s in symbols}


def _imports(tmp_path: Path, source: str) -> list[ImportEdge]:
    record, _, edges = _parse(tmp_path, source)
    assert record.parse_ok, record
    return edges


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_dart_is_registered() -> None:
    assert extensions_for(("dart",)) == (".dart",)
    assert language_for_path("lib/a.dart").name == "dart"
    assert language_for_path("lib/a.py").name == "python"
    assert parser_dart.dart_grammar_available() is True


# --------------------------------------------------------------------------- #
# Kind mapping
# --------------------------------------------------------------------------- #

# (source, {qualname: (name, kind, parent)}, qualnames that must NOT exist)
MAPPING_CASES = {
    "class": (
        "class Foo {}\n",
        {"Foo": ("Foo", "class", None)},
        [],
    ),
    "abstract_class": (
        "abstract class Foo {\n  void run();\n}\n",
        {"Foo": ("Foo", "class", None), "Foo.run": ("run", "method", "Foo")},
        [],
    ),
    "mixin": (
        "mixin Walker {\n  void walk() {}\n}\n",
        {"Walker": ("Walker", "class", None), "Walker.walk": ("walk", "method", "Walker")},
        [],
    ),
    "extension": (
        "extension StringX on String {\n  int get shout => 1;\n}\n",
        {
            "StringX": ("StringX", "class", None),
            "StringX.shout": ("shout", "method", "StringX"),
        },
        [],
    ),
    "unnamed_extension": (
        "extension on String {\n  int twice() => 2;\n}\n",
        {
            "on String": ("on String", "class", None),
            "on String.twice": ("twice", "method", "on String"),
        },
        [],
    ),
    "extension_type": (
        "extension type Meters(int value) {\n  int double() => value * 2;\n}\n",
        {
            "Meters": ("Meters", "class", None),
            "Meters.double": ("double", "method", "Meters"),
        },
        [],
    ),
    "enum": (
        "enum Color {\n  red,\n  green;\n\n  String label() => 'x';\n}\n",
        {"Color": ("Color", "enum", None), "Color.label": ("label", "method", "Color")},
        ["Color.red", "Color.green"],
    ),
    "typedef": (
        "typedef Callback = void Function(int value);\n",
        {"Callback": ("Callback", "type", None)},
        [],
    ),
    "old_typedef": (
        "typedef int Op(int a, int b);\n",
        {"Op": ("Op", "type", None)},
        [],
    ),
    "top_level_function": (
        "int add(int a, int b) => a + b;\n",
        {"add": ("add", "function", None)},
        [],
    ),
    "top_level_getter": (
        "int get answer => 42;\n",
        {"answer": ("answer", "function", None)},
        [],
    ),
    "top_level_setter": (
        "int _x = 0;\nset answer(int v) {\n  _x = v;\n}\n",
        {"answer=": ("answer=", "function", None)},
        [],
    ),
    "method": (
        "class A {\n  void run(int n) {}\n}\n",
        {"A": ("A", "class", None), "A.run": ("run", "method", "A")},
        [],
    ),
    "operator": (
        "class A {\n  bool operator ==(Object other) => true;\n}\n",
        {"A.operator ==": ("operator ==", "method", "A")},
        [],
    ),
    "constructor": (
        "class Foo {\n  Foo();\n}\n",
        {"Foo.Foo": ("Foo", "method", "Foo")},
        [],
    ),
    "named_constructor": (
        "class Foo {\n  Foo.named();\n}\n",
        {"Foo.named": ("named", "method", "Foo")},
        [],
    ),
    "factory_constructor": (
        "class Foo {\n  Foo._();\n  factory Foo() => Foo._();\n}\n",
        {"Foo.Foo": ("Foo", "method", "Foo"), "Foo._": ("_", "method", "Foo")},
        [],
    ),
    "named_factory_constructor": (
        "class Foo {\n  Foo();\n  factory Foo.make() => Foo();\n}\n",
        {"Foo.make": ("make", "method", "Foo"), "Foo.Foo": ("Foo", "method", "Foo")},
        [],
    ),
}


@pytest.mark.parametrize("case", sorted(MAPPING_CASES))
def test_kind_mapping(tmp_path: Path, case: str) -> None:
    source, expected, absent = MAPPING_CASES[case]
    syms = _syms(tmp_path, source)
    for qualname, (name, kind, parent) in expected.items():
        assert qualname in syms, (qualname, sorted(syms))
        s = syms[qualname]
        assert (s.name, s.kind, s.parent) == (name, kind, parent), s
        assert s.id == f"lib/a.dart::{qualname}"
        assert s.path == "lib/a.dart"
    for qualname in absent:
        assert qualname not in syms


def test_getter_and_setter_with_same_name_do_not_collide(tmp_path: Path) -> None:
    syms = _syms(
        tmp_path,
        """
        class Box {
          int _v = 0;
          int get value => _v;
          set value(int v) {
            _v = v;
          }
        }
        """,
    )
    assert syms["Box.value"].kind == "method"
    assert syms["Box.value="].kind == "method"
    assert syms["Box.value="].name == "value="
    assert syms["Box.value"].id != syms["Box.value="].id
    assert syms["Box.value"].parent == "Box" == syms["Box.value="].parent


def test_fields_are_not_symbols(tmp_path: Path) -> None:
    syms = _syms(tmp_path, "class A {\n  int x = 1;\n  final String y = 'a';\n}\n")
    assert sorted(syms) == ["A"]


def test_no_member_is_lost_in_a_busy_file(tmp_path: Path) -> None:
    syms = _syms(
        tmp_path,
        """
        class Foo {
          Foo();
          Foo.named();
          factory Foo.make() => Foo();
          Foo.v();
          int get v => 1;
          set v(int x) {}
          bool operator ==(Object o) => true;
          int operator [](int i) => i;
          Foo operator -() => this;
          Foo operator -(Foo o) => this;
        }
        """,
    )
    wanted = {
        "Foo.Foo",
        "Foo.named",
        "Foo.make",
        "Foo.v",
        "Foo.v#2",
        "Foo.v=",
        "Foo.operator ==",
        "Foo.operator []",
        "Foo.operator -",
        "Foo.operator -#2",
    }
    assert wanted <= set(syms), sorted(syms)
    assert len(syms) == len(wanted) + 1  # the class


def test_same_name_members_are_separate_symbols(tmp_path: Path) -> None:
    syms = _syms(
        tmp_path,
        """
        class C {
          C operator -() => this;
          C operator -(C o) => this;
          C.sensor({required int a});
          int get sensor => 1;
          void sensor() {}
        }
        """,
    )
    assert syms["C.operator -"].signature == "C operator -()"
    assert syms["C.operator -"].id == "lib/a.dart::C.operator -"
    assert syms["C.operator -#2"].signature == "C operator -(C o)"
    assert syms["C.operator -#2"].id == "lib/a.dart::C.operator -#2"
    assert syms["C.operator -#2"].name == "operator -"
    assert syms["C.operator -#2"].parent == "C"
    assert (syms["C.operator -"].start_line, syms["C.operator -#2"].start_line) == (2, 3)
    assert syms["C.sensor"].signature == "C.sensor({required int a})"
    assert syms["C.sensor#2"].signature == "int get sensor"
    assert syms["C.sensor#3"].signature == "void sensor()"
    assert {syms[q].name for q in ("C.sensor", "C.sensor#2", "C.sensor#3")} == {"sensor"}
    assert [syms[q].start_line for q in ("C.sensor", "C.sensor#2", "C.sensor#3")] == [4, 5, 6]


def test_same_name_members_of_two_unnamed_extensions(tmp_path: Path) -> None:
    syms = _syms(
        tmp_path,
        """
        extension on String {
          int size() => 1;
        }

        extension on String {
          int size() => 2;
        }
        """,
    )
    assert sorted(syms) == ["on String", "on String.size", "on String.size#2"]
    assert (syms["on String.size"].start_line, syms["on String.size#2"].start_line) == (2, 6)


def test_same_name_members_are_found_by_search(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
        class Contact {
          Contact.sensor({required int a});
          int get sensor => 1;
        }
        """,
    )
    record, symbols, _ = parse_file(tmp_path, path)
    index = RepoIndex("r", str(tmp_path), "r", files=[record], symbols=symbols)
    ids = {hit.symbol_id for hit in SearchIndex(index).search("sensor", k=10)}
    assert {"lib/a.dart::Contact.sensor", "lib/a.dart::Contact.sensor#2"} <= ids


# --------------------------------------------------------------------------- #
# New syntax (real Flame examples)
# --------------------------------------------------------------------------- #


def test_class_const_primary_constructor_syntax(tmp_path: Path) -> None:
    record, symbols, _ = _parse(
        tmp_path,
        """
        class const AnchorPicker({required final String anchor, super.key}) extends StatelessWidget {
          @override
          Widget build(BuildContext context) => x;
        }
        """,
    )
    assert record.parse_ok is True
    syms = {s.qualname: s for s in symbols}
    assert "AnchorPicker" in syms and "const" not in {s.name for s in symbols}
    cls = syms["AnchorPicker"]
    assert (cls.name, cls.kind, cls.parent) == ("AnchorPicker", "class", None)
    assert "\n" not in cls.signature
    assert "required final String anchor" in cls.signature
    assert "super.key" in cls.signature
    assert "build" not in cls.signature
    build = syms["AnchorPicker.build"]
    assert (build.name, build.kind, build.parent) == ("build", "method", "AnchorPicker")
    assert "Widget build(BuildContext context)" in build.signature
    assert "=> x" not in build.signature


def test_sealed_class_with_primary_constructor(tmp_path: Path) -> None:
    record, symbols, _ = _parse(tmp_path, "sealed class _Asset<T>(var T value);\n")
    assert record.parse_ok is True
    syms = {s.qualname: s for s in symbols}
    assert syms["_Asset"].kind == "class"
    assert "var T value" in syms["_Asset"].signature
    assert "\n" not in syms["_Asset"].signature


def test_enum_with_primary_constructor(tmp_path: Path) -> None:
    record, symbols, _ = _parse(tmp_path, "enum LayerFill() { a, b }\n")
    assert record.parse_ok is True
    syms = {s.qualname: s for s in symbols}
    assert syms["LayerFill"].kind == "enum"
    assert sorted(syms) == ["LayerFill"]


# --------------------------------------------------------------------------- #
# Signatures
# --------------------------------------------------------------------------- #


def test_signatures_are_single_line_headers_without_body(tmp_path: Path) -> None:
    syms = _syms(
        tmp_path,
        """
        int add(
          int a,
          int b,
        ) {
          return a + b;
        }

        class Foo extends Bar {
          Foo(int x) : super() {
            print(x);
          }
        }
        """,
    )
    for s in syms.values():
        assert "\n" not in s.signature
        assert "return a + b" not in s.signature
        assert "print(x)" not in s.signature
    assert syms["add"].signature.startswith("int add(")
    assert "int a" in syms["add"].signature and "int b" in syms["add"].signature
    assert "extends Bar" in syms["Foo"].signature


# --------------------------------------------------------------------------- #
# Doc comments
# --------------------------------------------------------------------------- #


def test_multiline_doc_comment(tmp_path: Path) -> None:
    syms = _syms(
        tmp_path,
        """
        /// First line.
        /// Second line.
        class Foo {}
        """,
    )
    doc = syms["Foo"].docstring
    assert doc is not None
    assert doc.splitlines()[0] == "First line."
    assert "Second line." in doc
    assert "///" not in doc
    assert doc == doc.strip()


def test_doc_comment_before_annotation(tmp_path: Path) -> None:
    syms = _syms(
        tmp_path,
        """
        class Foo {
          /// Builds the widget.
          @override
          Widget build(BuildContext context) => x;
        }
        """,
    )
    assert syms["Foo.build"].docstring == "Builds the widget."


def test_no_doc_comment_is_none(tmp_path: Path) -> None:
    syms = _syms(tmp_path, "class Foo {\n  void run() {}\n}\n")
    assert syms["Foo"].docstring is None
    assert syms["Foo.run"].docstring is None


def test_plain_comment_is_not_a_docstring(tmp_path: Path) -> None:
    syms = _syms(
        tmp_path,
        """
        // just a note
        class Foo {}

        /* block */
        class Bar {}
        """,
    )
    assert syms["Foo"].docstring is None
    assert syms["Bar"].docstring is None


def test_doc_comment_is_not_inherited_by_the_next_declaration(tmp_path: Path) -> None:
    syms = _syms(
        tmp_path,
        """
        /// Documented.
        class Foo {}

        class Bar {}
        """,
    )
    assert syms["Foo"].docstring == "Documented."
    assert syms["Bar"].docstring is None


# --------------------------------------------------------------------------- #
# Lines
# --------------------------------------------------------------------------- #


def test_lines_are_one_based_and_inclusive(tmp_path: Path) -> None:
    syms = _syms(
        tmp_path,
        """
        // header
        class Foo {
          void a() {
            print(1);
          }

          int b() => 2;
        }

        int top() => 1;
        """,
    )
    assert (syms["Foo"].start_line, syms["Foo"].end_line) == (2, 8)
    assert (syms["Foo.a"].start_line, syms["Foo.a"].end_line) == (3, 5)
    assert (syms["Foo.b"].start_line, syms["Foo.b"].end_line) == (7, 7)
    assert (syms["top"].start_line, syms["top"].end_line) == (10, 10)


def test_file_record(tmp_path: Path) -> None:
    record, _, _ = _parse(tmp_path, "class Foo {}\n")
    assert record == FileRecord(
        path="lib/a.dart", language="dart", size=len(b"class Foo {}\n"), n_lines=1, parse_ok=True
    )


# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #


def test_import_show_list(tmp_path: Path) -> None:
    edges = _imports(tmp_path, "import 'package:a/b.dart' show X, Y;\nimport 'dart:async';\n")
    assert [(e.dst_module, e.names, e.line) for e in edges] == [
        ("package:a/b.dart", ["X", "Y"], 1),
        ("dart:async", [], 2),
    ]
    assert all(e.src_path == "lib/a.dart" and e.dst_path is None for e in edges)


def test_import_as_and_hide_have_no_names(tmp_path: Path) -> None:
    edges = _imports(
        tmp_path,
        "import 'b.dart' as b;\nimport 'c.dart' hide Z;\nimport 'd.dart' deferred as d;\n",
    )
    assert [(e.dst_module, e.names) for e in edges] == [
        ("b.dart", []),
        ("c.dart", []),
        ("d.dart", []),
    ]


def test_export_and_part(tmp_path: Path) -> None:
    edges = _imports(
        tmp_path,
        "export 'src/x.dart' show Foo;\nexport 'src/y.dart';\npart 'a.g.dart';\n",
    )
    assert [(e.dst_module, e.names, e.line) for e in edges] == [
        ("src/x.dart", ["Foo"], 1),
        ("src/y.dart", [], 2),
        ("a.g.dart", [], 3),
    ]


def test_part_of_makes_no_edge(tmp_path: Path) -> None:
    assert _imports(tmp_path, "part of 'main.dart';\n\nclass A {}\n") == []
    assert _imports(tmp_path, "part of my.library;\n\nclass A {}\n") == []


def test_double_quoted_uri_is_unquoted(tmp_path: Path) -> None:
    edges = _imports(tmp_path, 'import "package:a/b.dart";\n')
    assert [e.dst_module for e in edges] == ["package:a/b.dart"]


def _resolve(files: list[str], src: str, uri: str) -> str | None:
    index = RepoIndex(
        name="r",
        root="/r",
        source="/r",
        files=[
            FileRecord(path=p, language="dart", size=1, n_lines=1, parse_ok=True)
            for p in sorted({*files, src})
        ],
        imports=[ImportEdge(src_path=src, dst_module=uri, dst_path=None, names=[], line=1)],
    )
    resolve_imports(index)
    return index.imports[0].dst_path


def test_resolve_relative() -> None:
    files = ["lib/src/a.dart", "lib/src/c.dart", "lib/b.dart"]
    assert _resolve(files, "lib/src/a.dart", "c.dart") == "lib/src/c.dart"
    assert _resolve(files, "lib/src/a.dart", "../b.dart") == "lib/b.dart"
    assert _resolve(files, "lib/src/a.dart", "./c.dart") == "lib/src/c.dart"
    assert _resolve(files, "lib/src/a.dart", "missing.dart") is None


def test_resolve_part_relative() -> None:
    files = ["lib/a.dart", "lib/a.g.dart"]
    assert _resolve(files, "lib/a.dart", "a.g.dart") == "lib/a.g.dart"


def test_resolve_monorepo_packages_lib() -> None:
    files = ["packages/flame/lib/src/x.dart", "packages/app/lib/main.dart"]
    assert (
        _resolve(files, "packages/app/lib/main.dart", "package:flame/src/x.dart")
        == "packages/flame/lib/src/x.dart"
    )


def test_resolve_package_dir_lib() -> None:
    files = ["flame/lib/src/x.dart", "app/lib/main.dart"]
    assert (
        _resolve(files, "app/lib/main.dart", "package:flame/src/x.dart") == "flame/lib/src/x.dart"
    )


def test_resolve_single_package_lib() -> None:
    files = ["lib/src/x.dart", "lib/main.dart"]
    assert _resolve(files, "lib/main.dart", "package:myapp/src/x.dart") == "lib/src/x.dart"


def test_resolve_priority_order() -> None:
    uri = "package:flame/src/x.dart"
    all_three = ["packages/flame/lib/src/x.dart", "flame/lib/src/x.dart", "lib/src/x.dart"]
    assert _resolve(all_three, "lib/main.dart", uri) == "packages/flame/lib/src/x.dart"
    assert _resolve(all_three[1:], "lib/main.dart", uri) == "flame/lib/src/x.dart"
    assert _resolve(all_three[2:], "lib/main.dart", uri) == "lib/src/x.dart"


def test_resolve_dart_core_and_outside_repo_are_none() -> None:
    files = ["lib/main.dart", "lib/src/x.dart"]
    assert _resolve(files, "lib/main.dart", "dart:async") is None
    assert _resolve(files, "lib/main.dart", "dart:io") is None
    assert _resolve(files, "lib/main.dart", "package:http/http.dart") is None


def test_resolve_dart_and_python_side_by_side(tmp_path: Path) -> None:
    index = RepoIndex(
        name="r",
        root=str(tmp_path),
        source=str(tmp_path),
        files=[
            FileRecord(path="lib/a.dart", language="dart", size=1, n_lines=1, parse_ok=True),
            FileRecord(path="lib/b.dart", language="dart", size=1, n_lines=1, parse_ok=True),
            FileRecord(path="pkg/__init__.py", language="python", size=0, n_lines=0, parse_ok=True),
            FileRecord(path="pkg/m.py", language="python", size=1, n_lines=1, parse_ok=True),
            FileRecord(path="main.py", language="python", size=1, n_lines=1, parse_ok=True),
        ],
        imports=[
            ImportEdge("lib/a.dart", "b.dart", None, [], 1),
            ImportEdge("main.py", "pkg.m", None, [], 1),
        ],
    )
    resolve_imports(index)
    assert [e.dst_path for e in index.imports] == ["lib/b.dart", "pkg/m.py"]


# --------------------------------------------------------------------------- #
# Broken input
# --------------------------------------------------------------------------- #


def test_broken_dart_does_not_crash(tmp_path: Path) -> None:
    record, symbols, edges = _parse(
        tmp_path, "class Foo {\n  void ( ;\n}}}\n  int get => ;\n(((\n"
    )
    assert record.parse_ok is False
    assert record.language == "dart"
    assert isinstance(symbols, list) and isinstance(edges, list)


def test_empty_and_comment_only_files(tmp_path: Path) -> None:
    record, symbols, edges = _parse(tmp_path, "", "lib/empty.dart")
    assert (record.parse_ok, symbols, edges) == (True, [], [])
    record, symbols, edges = _parse(tmp_path, "// nothing\n", "lib/c.dart")
    assert (record.parse_ok, symbols, edges) == (True, [], [])


# --------------------------------------------------------------------------- #
# Environments without the grammar / without network (separate processes)
# --------------------------------------------------------------------------- #


def _run(code: str, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_without_language_pack_dart_is_skipped_and_python_works(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "m.ts").write_text("export function g(): number { return 1; }\n", encoding="utf-8")
    proc = _run(
        """
        import sys
        sys.modules["tree_sitter_language_pack"] = None
        from pathlib import Path
        import repofinder.parser as p
        from repofinder import parser_dart
        assert parser_dart.dart_grammar_available() is False
        assert "dart" not in p.LANGUAGES, sorted(p.LANGUAGES)
        assert p.language_for_path("a.dart") is None
        assert p.language_for_path("a.py").name == "python"
        assert p.language_for_path("a.ts").name == "typescript"
        root = Path(sys.argv[1])
        rec, syms, _ = p.parse_file(root, root / "m.py")
        assert rec.parse_ok and [s.qualname for s in syms] == ["f"]
        rec, syms, _ = p.parse_file(root, root / "m.ts")
        assert rec.parse_ok and [s.qualname for s in syms] == ["g"]
        try:
            p.extensions_for(("dart",))
        except ValueError:
            pass
        else:
            raise SystemExit("dart must not be registered")
        print("ok")
        """,
        str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_without_ctypes_and_language_pack_python_and_typescript_work(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "m.ts").write_text("export function g(): number { return 1; }\n", encoding="utf-8")
    proc = _run(
        """
        import sys
        sys.modules["ctypes"] = None  # a Python built without it
        sys.modules["tree_sitter_language_pack"] = None
        from pathlib import Path
        import repofinder.parser as p
        assert "dart" not in p.LANGUAGES, sorted(p.LANGUAGES)
        root = Path(sys.argv[1])
        rec, syms, _ = p.parse_file(root, root / "m.py")
        assert rec.parse_ok and [s.qualname for s in syms] == ["f"]
        rec, syms, _ = p.parse_file(root, root / "m.ts")
        assert rec.parse_ok and [s.qualname for s in syms] == ["g"]
        print("ok")
        """,
        str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_without_ctypes_the_grammar_is_refused_and_python_works(tmp_path: Path) -> None:
    """With the pack installed: Dart stays closed, with one warning and without a crash."""
    (tmp_path / "a.dart").write_bytes(b"class A {}\n")
    (tmp_path / "m.py").write_bytes(b"def f():\n    return 1\n")
    proc = _run(
        """
        import sys
        import tree_sitter_language_pack  # imported first: only this module must do without ctypes
        sys.modules["ctypes"] = None
        from pathlib import Path
        import repofinder.parser as p
        root = Path(sys.argv[1])
        rec, syms, edges = p.parse_file(root, root / "a.dart")
        assert (rec.parse_ok, syms, edges) == (False, [], []), rec
        rec, syms, _ = p.parse_file(root, root / "m.py")
        assert rec.parse_ok and [s.qualname for s in syms] == ["f"]
        print("ok")
        """,
        str(tmp_path),
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
    warnings = [line for line in proc.stderr.splitlines() if line.strip()]
    assert len(warnings) == 1, proc.stderr
    assert "Dart" in warnings[0]


def test_import_parser_does_not_touch_the_network() -> None:
    proc = _run(
        """
        import socket

        def _blocked(*a, **k):
            raise AssertionError("network access during import")

        class _NoSocket(socket.socket):
            def __init__(self, *a, **k):
                _blocked()

        socket.socket = _NoSocket
        socket.create_connection = _blocked
        socket.getaddrinfo = _blocked
        import repofinder.parser as p
        import repofinder.parser_dart
        assert p.language_for_path("a.py").name == "python"
        print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_grammar_download_failure_gives_empty_record_and_one_warning(tmp_path: Path) -> None:
    (tmp_path / "a.dart").write_bytes(b"import 'b.dart';\nclass A {}\n")
    (tmp_path / "b.dart").write_bytes(b"class B {}\n")
    (tmp_path / "m.py").write_bytes(b"def f():\n    return 1\n")
    proc = _run(
        """
        import sys
        from pathlib import Path
        import tree_sitter_language_pack as pack

        def _fail(*a, **k):
            raise RuntimeError("simulated: no network")

        pack.get_language = _fail          # the loader fails; nothing reaches the network
        for name in ("download", "init", "configure"):
            if hasattr(pack, name):
                setattr(pack, name, _fail)
        from repofinder.models import FileRecord
        import repofinder.parser as p
        root = Path(sys.argv[1])
        for name in ("a.dart", "b.dart"):
            rec, syms, edges = p.parse_file(root, root / name)
            assert rec == FileRecord(
                path=name, language="dart", size=rec.size, n_lines=rec.n_lines, parse_ok=False
            ), rec
            assert syms == [] and edges == [], (syms, edges)
        rec, syms, _ = p.parse_file(root, root / "m.py")
        assert rec.parse_ok and [s.qualname for s in syms] == ["f"]
        print("ok")
        """,
        str(tmp_path),
        env={**os.environ, "TREE_SITTER_LANGUAGE_PACK_CACHE_DIR": str(tmp_path / "empty_cache")},
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
    warnings = [line for line in proc.stderr.splitlines() if line.strip()]
    assert len(warnings) == 1, proc.stderr
    assert "Dart" in warnings[0]


# --------------------------------------------------------------------------- #
# Decided behaviours: docstring joining/truncation, conditional imports,
# primary constructor bodies (146 of them in Flame's lib)
# --------------------------------------------------------------------------- #


def test_docstring_lines_are_joined_with_newline(tmp_path: Path) -> None:
    syms = _syms(tmp_path, "/// One.\n///\n/// Three.\nclass Foo {}\n")
    assert syms["Foo"].docstring == "One.\n\nThree."


def test_docstring_is_cut_at_400_characters(tmp_path: Path) -> None:
    from repofinder.parser import DOCSTRING_MAX_CHARS

    assert DOCSTRING_MAX_CHARS == 400
    lines = "".join(f"/// {'x' * 50}\n" for _ in range(20))
    syms = _syms(tmp_path, lines + "class Long {}\n")
    doc = syms["Long"].docstring
    assert doc is not None and len(doc) == 400
    assert doc.startswith("x" * 50 + "\n" + "x" * 50)
    short = _syms(tmp_path, "/// " + "y" * 399 + "\nclass Fits {}\n")["Fits"].docstring
    assert short == "y" * 399


def test_conditional_import_gives_one_edge_per_uri(tmp_path: Path) -> None:
    edges = _imports(
        tmp_path,
        "import 'a.dart'\n"
        "    if (dart.library.io) 'a_io.dart'\n"
        "    if (dart.library.html) 'a_html.dart';\n"
        "export 'e.dart' if (dart.library.io) 'e_io.dart';\n",
    )
    assert [(e.dst_module, e.line) for e in edges] == [
        ("a.dart", 1),
        ("a_io.dart", 1),
        ("a_html.dart", 1),
        ("e.dart", 4),
        ("e_io.dart", 4),
    ]


@pytest.mark.parametrize(
    "body",
    ["this : super(1);", "this : assert(x > 0) {}", "this {\n    print(x);\n  }"],
)
def test_primary_constructor_body_is_constructor_method(tmp_path: Path, body: str) -> None:
    syms = _syms(tmp_path, f"class Foo(int x) {{\n  {body}\n  void run() {{}}\n}}\n")
    ctor = syms["Foo.Foo"]
    assert (ctor.name, ctor.kind, ctor.parent) == ("Foo", "method", "Foo")
    assert syms["Foo"].kind == "class"
    assert syms["Foo.run"].kind == "method"
    assert sorted(syms) == ["Foo", "Foo.Foo", "Foo.run"]


def test_named_primary_constructor_does_not_hide_the_factory(tmp_path: Path) -> None:
    # packages/flame/lib/src/sprite_warp/warp_grid.dart
    syms = _syms(
        tmp_path,
        """
        class const WarpGrid.raw(final int columns, final int rows) {
          /// The raw one.
          @internal
          this;

          /// The checked one.
          factory WarpGrid({required int columns}) {
            return WarpGrid.raw(columns, 1);
          }
        }
        """,
    )
    assert sorted(syms) == ["WarpGrid", "WarpGrid.WarpGrid", "WarpGrid.raw"]
    raw = syms["WarpGrid.raw"]
    assert (raw.name, raw.kind, raw.parent) == ("raw", "method", "WarpGrid")
    assert (raw.start_line, raw.end_line) == (3, 4)
    assert raw.docstring == "The raw one."
    factory = syms["WarpGrid.WarpGrid"]
    assert (factory.name, factory.kind, factory.parent) == ("WarpGrid", "method", "WarpGrid")
    assert (factory.start_line, factory.end_line) == (7, 9)
    assert factory.docstring == "The checked one."
    assert factory.signature == "factory WarpGrid({required int columns})"


@pytest.mark.parametrize(
    ("source", "qualname", "signature"),
    [
        ("class Foo(int x) {\n  this : super(1);\n}\n", "Foo.Foo", "Foo(int x)"),
        ("class Foo.zero() extends B {\n  this : super.zero();\n}\n", "Foo.zero", "Foo.zero()"),
        ("class const Foo.raw(int a, {int b = 1}) {\n  this;\n}\n", "Foo.raw", "const Foo.raw(int a, {int b = 1})"),
        ("class Foo<T>.raw(T a) {\n  this {\n    print(a);\n  }\n}\n", "Foo.raw", "Foo<T>.raw(T a)"),
        ("class Foo(\n  // the x\n  int x,\n  int y,\n) {\n  this;\n}\n", "Foo.Foo", "Foo(int x, int y)"),
        ("enum E.n(int v) {\n  a(1);\n\n  this : assert(v > 0);\n}\n", "E.n", "E.n(int v)"),
        ("extension type const M.n(int v) {\n  this;\n}\n", "M.n", "const M.n(int v)"),
        ("extension type M(int value) {\n  this;\n}\n", "M.M", "M(int value)"),
    ],
)
def test_primary_constructor_body_names_and_introduces_the_constructor(
    tmp_path: Path, source: str, qualname: str, signature: str
) -> None:
    syms = _syms(tmp_path, source)
    assert syms[qualname].signature == signature
    assert "\n" not in syms[qualname].signature


def test_primary_constructor_signature_is_clipped_like_the_others(tmp_path: Path) -> None:
    params = ", ".join(f"int parameter{i}" for i in range(30))
    syms = _syms(tmp_path, f"class Foo({params}) {{\n  this;\n}}\n")
    assert len(syms["Foo.Foo"].signature) <= 200
    assert syms["Foo.Foo"].signature.startswith("Foo(int parameter0, ")


def test_body_without_a_primary_constructor_does_not_crash(tmp_path: Path) -> None:
    record, symbols, _ = _parse(tmp_path, "class P {\n  this;\n}\n")
    assert record.language == "dart"
    assert {s.qualname: s.signature for s in symbols}["P.P"] == "this"


# Layouts of Flame and mobile_scanner, package names from their pubspec.yaml.


def test_resolve_package_in_a_directory_named_like_it_at_any_depth() -> None:
    files = ["packages/flame_jenny/jenny/lib/jenny.dart", "packages/flame_jenny/lib/src/a.dart"]
    uri = "package:jenny/jenny.dart"
    assert _resolve(files, "packages/flame_jenny/lib/src/a.dart", uri) == "packages/flame_jenny/jenny/lib/jenny.dart"
    assert _resolve(files, "packages/flame_jenny/jenny/test/a_test.dart", uri) == "packages/flame_jenny/jenny/lib/jenny.dart"


@pytest.mark.parametrize(
    ("src", "uri", "expected"),
    [
        ("doc/flame/examples/lib/a.dart", "package:doc_flame_examples/flower.dart", "doc/flame/examples/lib/flower.dart"),
        ("doc/flame/examples/test/a_test.dart", "package:doc_flame_examples/flower.dart", "doc/flame/examples/lib/flower.dart"),
        ("example/lib/main.dart", "package:mobile_scanner_example/screens/s.dart", "example/lib/screens/s.dart"),
        ("packages/a/test/own_test.dart", "package:a_core/a_core.dart", "packages/a/lib/a_core.dart"),
    ],
)
def test_resolve_own_package_in_a_directory_named_otherwise(src: str, uri: str, expected: str) -> None:
    assert _resolve([expected], src, uri) == expected


def test_resolve_lib_inside_lib_is_not_a_package() -> None:
    files = ["packages/a/lib/src/lib/b.dart", "packages/a/lib/c.dart"]
    assert _resolve(files, "packages/a/lib/src/lib/d.dart", "package:a_pkg/c.dart") == "packages/a/lib/c.dart"


def test_resolve_root_library_from_a_nested_package_and_from_tests() -> None:
    files = ["lib/mobile_scanner.dart", "lib/src/x.dart", "example/lib/main.dart"]
    barrel = "package:mobile_scanner/mobile_scanner.dart"
    assert _resolve(files, "example/lib/main.dart", barrel) == "lib/mobile_scanner.dart"
    assert _resolve(files, "example/lib/main.dart", "package:mobile_scanner/src/x.dart") == "lib/src/x.dart"
    assert _resolve(files, "test/a_test.dart", barrel) == "lib/mobile_scanner.dart"


def test_resolve_several_directories_named_like_the_package() -> None:
    files = ["a/foo/lib/x.dart", "b/foo/lib/x.dart", "packages/foo/lib/y.dart", "b/foo/lib/y.dart"]
    assert _resolve(files, "a/foo/lib/z.dart", "package:foo/x.dart") == "a/foo/lib/x.dart"  # its own
    assert _resolve(files, "b/c/lib/z.dart", "package:foo/x.dart") == "b/foo/lib/x.dart"  # nearest
    assert _resolve(files, "c/lib/z.dart", "package:foo/x.dart") is None  # a tie: cannot tell
    assert _resolve(files, "b/c/lib/z.dart", "package:foo/y.dart") == "packages/foo/lib/y.dart"


@pytest.mark.parametrize(
    ("files", "src", "uri"),
    [
        # A dependency must not bind to a local file that happens to share its path.
        (["lib/x.dart", "packages/b/lib/y.dart"], "packages/b/lib/y.dart", "package:unknown/x.dart"),
        (["lib/http.dart", "lib/main.dart"], "lib/main.dart", "package:http/http.dart"),
        (["lib/http.dart", "lib/client.dart"], "lib/client.dart", "package:http/http.dart"),
        (["lib/http.dart", "lib/main.dart", "packages/a/lib/x.dart"], "packages/a/lib/x.dart", "package:http/http.dart"),
        (["packages/a/lib/http.dart", "packages/a/lib/x.dart"], "packages/a/lib/x.dart", "package:http/http.dart"),
        (["lib/widgets.dart", "lib/main.dart"], "lib/main.dart", "package:flutter/widgets.dart"),
        (["packages/a/lib/widgets.dart"], "packages/a/lib/src/b.dart", "package:flutter/widgets.dart"),
        (["lib/flutter_test.dart"], "test/a_test.dart", "package:flutter_test/flutter_test.dart"),
    ],
)
def test_resolve_dependency_is_not_bound_to_a_local_file(files: list[str], src: str, uri: str) -> None:
    assert _resolve(files, src, uri) is None


def test_resolve_importer_outside_the_known_files_still_has_its_package() -> None:
    known = {"lib/x.dart", "packages/a/lib/x.dart"}
    assert parser_dart.resolve_dart_uri("packages/b/lib/y.dart", "package:unknown/x.dart", known) is None


def test_resolve_sdk_package_in_its_own_repository() -> None:
    files = ["packages/flutter/lib/widgets.dart", "packages/flutter/test/a_test.dart"]
    uri = "package:flutter/widgets.dart"
    assert _resolve(files, "packages/flutter/test/a_test.dart", uri) == "packages/flutter/lib/widgets.dart"


@pytest.mark.parametrize(
    ("files", "src", "uri", "got"),
    [
        # pubspec.yaml is out of reach, so these keep the wrong answer; if a resolver
        # gets to read it, they should flip. Real case: flame_3d with lib/game.dart
        # importing package:flame/game.dart, when only that package is indexed.
        (["lib/game.dart", "lib/main.dart"], "lib/main.dart", "package:flame/game.dart", "lib/game.dart"),
        (["lib/http.dart"], "test/a_test.dart", "package:http/http.dart", "lib/http.dart"),
        (["lib/http.dart", "packages/a/lib/x.dart"], "packages/a/lib/x.dart", "package:http/http.dart", "lib/http.dart"),
        # And the opposite: a package importing its own library file from lib/.
        (["lib/mobile_scanner.dart", "lib/src/a.dart"], "lib/src/a.dart", "package:mobile_scanner/mobile_scanner.dart", None),
    ],
)
def test_resolve_known_limits_without_pubspec(files: list[str], src: str, uri: str, got: str | None) -> None:
    assert _resolve(files, src, uri) == got


# --------------------------------------------------------------------------- #
# Pinned grammar: only the exact expected bytes become native code (separate
# processes, each with its own temporary cache; the real cache is only read)
# --------------------------------------------------------------------------- #

_PIN_PROBE = """
    import ctypes, json, os, platform, shutil, sys
    from pathlib import Path
    import tree_sitter_language_pack as pack

    root, libs, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    library = libs / name
    good = Path(sys.argv[4]) if sys.argv[4] else None

    pack_loaders = []  # calling any of these would load without the check
    for loader in ("get_language", "get_parser", "prefetch"):
        setattr(pack, loader, lambda *a, _n=loader, **k: pack_loaders.append(_n))
    downloads = []

    opened = []
    real_cdll = ctypes.CDLL
    def spy_cdll(path, *args, **kwargs):
        opened.append({"path": str(path), "kwargs": kwargs})
        before_load(path)
        return real_cdll(path, *args, **kwargs)
    ctypes.CDLL = spy_cdll
    def before_load(path):
        pass

    # SCENARIO

    from repofinder import parser as p, parser_dart

    try:
        parser_dart.dart_grammar()
        error = None
    except parser_dart.DartGrammarUnavailable as exc:
        error = str(exc)
    rec, syms, _ = p.parse_file(root, root / "a.dart")
    py_rec, py_syms, _ = p.parse_file(root, root / "m.py")

    in_process = module_file = None
    if os.name == "nt":  # what really happened, whatever the code above believes
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetModuleHandleW.restype = ctypes.c_void_p
        kernel32.GetModuleHandleW.argtypes = (ctypes.c_wchar_p,)
        handle = kernel32.GetModuleHandleW(name)
        in_process = bool(handle)
        if handle:
            buffer = ctypes.create_unicode_buffer(32768)
            kernel32.GetModuleFileNameW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_ulong)
            kernel32.GetModuleFileNameW(handle, buffer, len(buffer))
            module_file = buffer.value
    print(json.dumps({
        "error": error,
        "opened": opened,
        "pack_loaders": pack_loaders,
        "downloads": downloads,
        "in_process": in_process,
        "module_file": module_file,
        "parse_ok": rec.parse_ok,
        "symbols": [s.qualname for s in syms],
        "python_symbols": [s.qualname for s in py_syms],
    }))
"""


def _pinned_grammar_file() -> Path:
    """The real cached library, read-only, on a platform that has a pin."""
    import tree_sitter_language_pack as pack

    pin = parser_dart._GRAMMAR_PINS.get(parser_dart._platform_key())
    if pin is None:
        pytest.skip(f"no pin for {parser_dart._platform_key()}")
    path = Path(pack.cache_dir()) / pin[0]
    if not path.exists():
        pytest.skip(f"{path} is not cached")
    return path


def _pin_probe(
    tmp_path: Path,
    content: bytes | None,
    scenario: str = "",
    good: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[dict, list[str]]:
    """Run the grammar in a new process on a cache holding ``content``; result, stderr lines."""
    real = _pinned_grammar_file()
    base = tmp_path / "cache"
    libs = base / "tree-sitter-language-pack" / "v1.23.1" / "libs"
    libs.mkdir(parents=True, exist_ok=True)
    if content is not None:
        (libs / real.name).write_bytes(content)
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.dart").write_bytes(b"class A {}\n")
    (root / "m.py").write_bytes(b"def f():\n    return 1\n")
    code = textwrap.dedent(_PIN_PROBE).replace("# SCENARIO", textwrap.dedent(scenario))
    proc = subprocess.run(
        [sys.executable, "-c", code, str(root), str(libs), real.name, str(good) if good else ""],
        cwd=REPO_ROOT,
        env={**os.environ, "TREE_SITTER_LANGUAGE_PACK_CACHE_DIR": str(base), **(env or {})},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout), [line for line in proc.stderr.splitlines() if line.strip()]


def _assert_not_loaded(result: dict, warnings: list[str]) -> None:
    assert result["error"] is not None
    assert result["opened"] == []
    assert result["pack_loaders"] == []
    assert result["in_process"] in (False, None)
    # Indexing goes on: the file stays indexed, without symbols; other languages work.
    assert (result["parse_ok"], result["symbols"]) == (False, [])
    assert result["python_symbols"] == ["f"]
    assert len(warnings) == 1, warnings
    assert "Dart" in warnings[0]


def _assert_loaded(result: dict, warnings: list[str]) -> None:
    assert result["error"] is None
    assert len(result["opened"]) == 1
    assert result["pack_loaders"] == []
    assert result["in_process"] in (True, None)
    assert (result["parse_ok"], result["symbols"]) == (True, ["A"])
    assert result["python_symbols"] == ["f"]
    assert warnings == []


def test_pin_the_real_cached_file_has_the_pinned_digest() -> None:
    path = _pinned_grammar_file()
    expected = parser_dart._GRAMMAR_PINS[parser_dart._platform_key()][1]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_pin_table_is_well_formed() -> None:
    names = {"linux": "libtree_sitter_dart.so", "macos": "libtree_sitter_dart.dylib", "windows": "tree_sitter_dart.dll"}
    assert parser_dart._PACK_VERSION == "1.23.1"
    for key, (filename, digest) in parser_dart._GRAMMAR_PINS.items():
        assert filename == names[key.partition("-")[0]]
        assert re.fullmatch(r"[0-9a-f]{64}", digest), key
    digests = [digest for _, digest in parser_dart._GRAMMAR_PINS.values()]
    assert len(set(digests)) == len(digests)


def test_pin_exact_file_is_loaded_and_downloads_nothing(tmp_path: Path) -> None:
    good = _pinned_grammar_file().read_bytes()
    result, warnings = _pin_probe(
        tmp_path, good, "pack.download = lambda names: downloads.append(names)"
    )
    _assert_loaded(result, warnings)
    assert result["downloads"] == []
    libs = tmp_path / "cache" / "tree-sitter-language-pack" / "v1.23.1" / "libs"
    assert Path(result["opened"][0]["path"]).samefile(libs / _pinned_grammar_file().name)


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda b: b[:1000] + bytes([b[1000] ^ 1]) + b[1001:], id="one_bit_flipped"),
        pytest.param(lambda b: b + b"\0", id="one_byte_appended"),
        pytest.param(lambda b: b[:-1], id="one_byte_removed"),
        pytest.param(lambda b: b"", id="empty"),
    ],
)
def test_pin_changed_file_is_not_loaded_and_not_replaced(tmp_path: Path, change) -> None:
    changed = change(_pinned_grammar_file().read_bytes())
    # pack.download would load the file; it must not even be called.
    result, warnings = _pin_probe(
        tmp_path, changed, "pack.download = lambda names: downloads.append(names)"
    )
    _assert_not_loaded(result, warnings)
    assert result["downloads"] == []
    libs = tmp_path / "cache" / "tree-sitter-language-pack" / "v1.23.1" / "libs"
    file = libs / _pinned_grammar_file().name
    assert file.read_bytes() == changed
    assert hashlib.sha256(changed).hexdigest() in warnings[0]
    assert parser_dart._GRAMMAR_PINS[parser_dart._platform_key()][1] in warnings[0]


def test_pin_missing_file_and_failing_download_is_not_loaded(tmp_path: Path) -> None:
    result, warnings = _pin_probe(
        tmp_path,
        None,
        """
        def _fail(names):
            downloads.append(names)
            raise RuntimeError("simulated:\\n  no network")
        pack.download = _fail
        """,
    )
    _assert_not_loaded(result, warnings)
    assert result["downloads"] == [["dart"]]
    assert "simulated: no network" in warnings[0]


def test_pin_download_that_leaves_no_file_is_not_loaded(tmp_path: Path) -> None:
    result, warnings = _pin_probe(
        tmp_path, None, "pack.download = lambda names: downloads.append(names)"
    )
    _assert_not_loaded(result, warnings)
    assert "missing" in warnings[0]


def test_pin_downloaded_file_is_checked_before_loading(tmp_path: Path) -> None:
    (tmp_path / "bad").write_bytes(_pinned_grammar_file().read_bytes() + b"\0")
    result, warnings = _pin_probe(
        tmp_path,
        None,
        """
        def _download(names):
            downloads.append(names)
            shutil.copyfile(good, library)
        pack.download = _download
        """,
        good=tmp_path / "bad",
    )
    _assert_not_loaded(result, warnings)
    assert result["downloads"] == [["dart"]]


def test_pin_good_downloaded_file_is_loaded(tmp_path: Path) -> None:
    result, warnings = _pin_probe(
        tmp_path,
        None,
        """
        def _download(names):
            downloads.append(names)
            shutil.copyfile(good, library)
        pack.download = _download
        """,
        good=_pinned_grammar_file(),
    )
    _assert_loaded(result, warnings)
    assert result["downloads"] == [["dart"]]


@pytest.mark.skipif(sys.platform != "win32", reason="reads the module's path from Windows")
def test_pin_real_download_does_not_load_from_the_libs_dir_override(tmp_path: Path) -> None:
    """The pack's own download, offline from a copy of its cached bundle.

    It tries to load the grammar first, also from this variable's directory: without
    protection a changed file there runs, even if dart_grammar then reports a failure.
    """
    real = _pinned_grammar_file()
    version_dir = real.parent.parent
    bundles = list((version_dir / "bundles").glob("*.tar.zst"))
    if not bundles or not (version_dir / "manifest.json").exists():
        pytest.skip("the pack's bundle is not cached")
    cached = tmp_path / "cache" / "tree-sitter-language-pack" / "v1.23.1"
    (cached / "bundles").mkdir(parents=True)
    shutil.copyfile(version_dir / "manifest.json", cached / "manifest.json")
    shutil.copyfile(bundles[0], cached / "bundles" / bundles[0].name)
    override = tmp_path / "override"
    override.mkdir()
    (override / real.name).write_bytes(real.read_bytes() + b"\0")

    result, warnings = _pin_probe(tmp_path, None, env={"TREE_SITTER_LANGUAGE_PACK_LIBS_DIR": str(override)})

    _assert_loaded(result, warnings)
    assert Path(result["module_file"]).samefile(cached / "libs" / real.name)


@pytest.mark.skipif(sys.platform != "win32", reason="reads the module list from Windows")
def test_pin_known_limit_file_appearing_just_before_the_download(tmp_path: Path) -> None:
    """The documented limit: the pack's download loads what it finds. If this flips, drop the note."""
    changed = tmp_path / "changed"
    changed.write_bytes(_pinned_grammar_file().read_bytes() + b"\0")
    result, warnings = _pin_probe(
        tmp_path,
        None,
        """
        real_download = pack.download
        def _download(names):
            shutil.copyfile(good, library)  # the instant between "missing" and the pack's own look
            return real_download(names)
        pack.download = _download
        """,
        good=changed,
    )
    assert result["error"] is not None and len(warnings) == 1  # dart_grammar refuses the file
    assert result["opened"] == []  # and does not load it itself
    assert result["in_process"] is True  # but the pack already has


def test_pin_platform_without_a_pin_is_not_loaded(tmp_path: Path) -> None:
    # The right bytes in the right place: only the platform has no digest.
    result, warnings = _pin_probe(
        tmp_path,
        _pinned_grammar_file().read_bytes(),
        """
        import sysconfig
        platform.machine = lambda: "riscv64"
        sysconfig.get_platform = lambda: "win-riscv64"
        pack.download = lambda names: downloads.append(names)
        """,
    )
    _assert_not_loaded(result, warnings)
    assert result["downloads"] == []
    assert "riscv64" in warnings[0]


@pytest.mark.parametrize("version", ["1.23.2", "1.23.0", None])
def test_pin_other_package_version_is_not_loaded(tmp_path: Path, version: str | None) -> None:
    result, warnings = _pin_probe(
        tmp_path,
        _pinned_grammar_file().read_bytes(),
        f"""
        pack.__version__ = {version!r}
        pack.download = lambda names: downloads.append(names)
        """,
    )
    _assert_not_loaded(result, warnings)
    assert result["downloads"] == []


def test_pin_failure_warns_once_for_many_files_and_calls(tmp_path: Path) -> None:
    changed = _pinned_grammar_file().read_bytes() + b"\0"
    result, warnings = _pin_probe(
        tmp_path,
        changed,
        """
        pack.download = lambda names: downloads.append(names)
        for _ in range(3):
            try:
                __import__("repofinder.parser_dart").parser_dart.dart_grammar()
            except RuntimeError:
                pass
        """,
    )
    _assert_not_loaded(result, warnings)


@pytest.mark.parametrize(
    ("system", "build", "machine", "pointer_bits", "key"),
    [
        # Windows: the build of Python counts; platform.machine() is the machine.
        ("win32", "win-amd64", "AMD64", 64, "windows-x86_64"),
        ("win32", "win-arm64", "ARM64", 64, "windows-aarch64"),
        ("win32", "win-amd64", "ARM64", 64, "windows-x86_64"),  # x64 Python on Windows ARM64
        # Linux and macOS: uname follows the process, so an emulated one reports its own.
        ("linux", None, "x86_64", 64, "linux-x86_64"),
        ("linux", None, "aarch64", 64, "linux-aarch64"),
        ("darwin", None, "x86_64", 64, "macos-x86_64"),  # also x64 Python under Rosetta
        ("darwin", None, "arm64", 64, "macos-arm64"),
    ],
)
def test_pin_platform_key_is_the_one_the_pack_downloads_for(
    monkeypatch: pytest.MonkeyPatch, system: str, build: str | None, machine: str, pointer_bits: int, key: str
) -> None:
    monkeypatch.setattr(parser_dart.platform, "machine", lambda: machine)
    monkeypatch.setattr(parser_dart.sysconfig, "get_platform", lambda: build)
    monkeypatch.setattr(parser_dart.sys, "maxsize", 2 ** (pointer_bits - 1) - 1)
    monkeypatch.setattr(parser_dart.sys, "platform", system)
    assert parser_dart._platform_key() == key
    assert key in parser_dart._GRAMMAR_PINS


@pytest.mark.parametrize(
    ("system", "build", "machine", "pointer_bits"),
    [
        ("win32", "win32", "AMD64", 32),  # 32-bit Python on 64-bit Windows
        ("win32", "win-arm32", "ARM64", 32),
        ("linux", None, "x86_64", 32),  # 32-bit Python on a 64-bit kernel
        ("linux", None, "aarch64", 32),
        ("linux", None, "riscv64", 64),
        ("freebsd14", None, "amd64", 64),
    ],
)
def test_pin_interpreter_the_pack_has_no_build_for_has_no_pin(
    monkeypatch: pytest.MonkeyPatch, system: str, build: str | None, machine: str, pointer_bits: int
) -> None:
    monkeypatch.setattr(parser_dart.platform, "machine", lambda: machine)
    monkeypatch.setattr(parser_dart.sysconfig, "get_platform", lambda: build)
    monkeypatch.setattr(parser_dart.sys, "maxsize", 2 ** (pointer_bits - 1) - 1)
    monkeypatch.setattr(parser_dart.sys, "platform", system)
    assert parser_dart._platform_key() not in parser_dart._GRAMMAR_PINS


@pytest.mark.skipif(sys.platform != "win32", reason="the Windows branch of sysconfig")
@pytest.mark.parametrize(
    ("version", "key"),
    [
        ("3.12.10 (tags/v3.12.10, Apr  8 2025) [MSC v.1943 64 bit (AMD64)]", "windows-x86_64"),
        ("3.12.10 (tags/v3.12.10, Apr  8 2025) [MSC v.1943 64 bit (ARM64)]", "windows-aarch64"),
    ],
)
def test_pin_real_sysconfig_reads_the_build_of_python(
    monkeypatch: pytest.MonkeyPatch, version: str, key: str
) -> None:
    monkeypatch.setattr(parser_dart.platform, "machine", lambda: "ARM64")  # must not matter
    monkeypatch.setattr(parser_dart.sys, "version", version)
    assert parser_dart._platform_key() == key


@pytest.mark.skipif(sys.platform != "win32", reason="Windows share modes")
def test_pin_file_cannot_change_between_check_and_load(tmp_path: Path) -> None:
    """The real race: something tries to swap the file at the last moment before loading."""
    good = _pinned_grammar_file().read_bytes()
    (tmp_path / "evil").write_bytes(b"MZ evil")
    result, warnings = _pin_probe(
        tmp_path,
        good,
        """
        attempts = {}
        def before_load(path):
            for label, action in (
                ("replace", lambda: os.replace(good, library)),
                ("remove", lambda: os.remove(library)),
                ("append", lambda: open(library, "ab").close()),
                ("rename", lambda: os.rename(library, library.with_suffix(".away"))),
                ("rename_directory", lambda: os.rename(libs, libs.with_name("moved"))),
            ):
                try:
                    action()
                    attempts[label] = "done"
                except PermissionError:
                    attempts[label] = "blocked"
            print(json.dumps(attempts), file=sys.stderr)
        """,
        good=tmp_path / "evil",
    )
    # The probe's stderr carries the attempts instead of a warning.
    assert json.loads(warnings[0]) == dict.fromkeys(
        ("replace", "remove", "append", "rename", "rename_directory"), "blocked"
    )
    _assert_loaded(result, warnings[1:])
    libs = tmp_path / "cache" / "tree-sitter-language-pack" / "v1.23.1" / "libs"
    assert (libs / _pinned_grammar_file().name).read_bytes() == good


@pytest.mark.skipif(sys.platform != "win32", reason="Windows search paths")
def test_pin_library_imports_are_not_searched_next_to_it(tmp_path: Path) -> None:
    good = _pinned_grammar_file().read_bytes()
    result, warnings = _pin_probe(tmp_path, good)
    _assert_loaded(result, warnings)
    assert result["opened"][0]["kwargs"] == {"winmode": 0x800}  # LOAD_LIBRARY_SEARCH_SYSTEM32


@pytest.mark.skipif(sys.platform != "win32", reason="needs junctions")
def test_pin_load_path_does_not_go_through_a_junction(tmp_path: Path) -> None:
    """A junction is not protected by the lock, so the path to load must not use it."""
    good = _pinned_grammar_file().read_bytes()
    target = tmp_path / "target"
    target.mkdir()
    (target / "lib.dll").write_bytes(good)
    junction = tmp_path / "junction"
    subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(target)], check=True, capture_output=True)
    with parser_dart._hold_unchanged(str(junction / "lib.dll")) as (data, load_path):
        assert data == good
        assert "junction" not in load_path
        assert Path(load_path).samefile(target / "lib.dll")


def _inner_pytest(
    tmp_path: Path, plugin: str = "", cache: Path | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """One grammar-dependent test of this file in a new pytest, with something broken."""
    (tmp_path / "broken_setup.py").write_text(plugin, encoding="utf-8")
    env = {
        **os.environ,
        "PYTHONPATH": str(tmp_path),
        "TREE_SITTER_LANGUAGE_PACK_CACHE_DIR": str(cache or tmp_path / "c"),
        **(env or {}),
    }
    return subprocess.run(
        [
            sys.executable, "-m", "pytest", "tests/test_parser_dart.py", "-k", "test_file_record",
            "-q", "-rs", "-p", "no:cacheprovider", "-p", "broken_setup",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",  # the OS words an error in its own code page
        timeout=120,
    )


def test_pin_suite_fails_on_a_changed_cached_grammar(tmp_path: Path) -> None:
    real = _pinned_grammar_file()
    cache = tmp_path / "cache"
    libs = cache / "tree-sitter-language-pack" / "v1.23.1" / "libs"
    libs.mkdir(parents=True)
    (libs / real.name).write_bytes(real.read_bytes() + b"\0")
    proc = _inner_pytest(tmp_path, cache=cache)
    assert proc.returncode == 1, proc.stdout
    assert "1 error" in proc.stdout.strip().splitlines()[-1]  # pytest.fail in the fixture
    assert "did not load" in proc.stdout


def test_pin_suite_fails_on_an_unexpected_package_version(tmp_path: Path) -> None:
    proc = _inner_pytest(tmp_path, "import tree_sitter_language_pack as p\np.__version__ = '1.23.2'\n")
    assert proc.returncode == 1, proc.stdout
    assert "1 error" in proc.stdout.strip().splitlines()[-1]  # pytest.fail in the fixture
    assert "version" in proc.stdout


def test_pin_suite_skips_when_the_network_is_down(tmp_path: Path) -> None:
    """The pack's own request, refused by a port nothing listens on."""
    proc = _inner_pytest(tmp_path, env={"TREE_SITTER_LANGUAGE_PACK_MANIFEST_URL": "http://127.0.0.1:9/parsers.json"})
    assert proc.returncode == 0, proc.stdout
    assert proc.stdout.strip().splitlines()[-1].startswith("1 skipped,")
    assert "download failed, no cache" in proc.stdout


def test_pin_suite_fails_when_the_release_bundle_has_changed(tmp_path: Path) -> None:
    """The pack's own digest check: a bundle that does not match the manifest, offline through file://."""
    real = _pinned_grammar_file()
    manifest_file = real.parent.parent / "manifest.json"
    if not manifest_file.exists():
        pytest.skip("the pack's manifest is not cached")
    bundle = tmp_path / "changed.tar.zst"
    bundle.write_bytes(b"not the bundle")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["platforms"][parser_dart._platform_key()] = {"url": f"file://{bundle}", "sha256": "0" * 64, "size": 14}
    cache = tmp_path / "cache"
    version = cache / "tree-sitter-language-pack" / "v1.23.1"
    version.mkdir(parents=True)
    (version / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    proc = _inner_pytest(tmp_path, cache=cache)
    assert proc.returncode == 1, proc.stdout
    assert "1 error" in proc.stdout.strip().splitlines()[-1]
    assert "Checksum mismatch" in proc.stdout


@pytest.mark.parametrize(
    ("fake_download", "words"),
    [
        pytest.param("raise p.ChecksumMismatchError('simulated')", "simulated", id="checksum_mismatch"),
        pytest.param("pass", "missing after the download", id="succeeds_but_leaves_no_file"),
    ],
)
def test_pin_suite_fails_when_a_download_goes_wrong_without_a_network_failure(
    tmp_path: Path, fake_download: str, words: str
) -> None:
    plugin = f"import tree_sitter_language_pack as p\ndef _fake(names):\n    {fake_download}\np.download = _fake\n"
    proc = _inner_pytest(tmp_path, plugin)
    assert proc.returncode == 1, proc.stdout
    assert "1 error" in proc.stdout.strip().splitlines()[-1]
    assert words in proc.stdout


def test_pin_suite_skips_when_the_platform_has_no_pin(tmp_path: Path) -> None:
    plugin = (
        "import platform, sysconfig\n"
        "platform.machine = lambda: 'riscv64'\n"
        "sysconfig.get_platform = lambda: 'win-riscv64'\n"
    )
    proc = _inner_pytest(tmp_path, plugin)
    assert proc.returncode == 0, proc.stdout
    assert proc.stdout.strip().splitlines()[-1].startswith("1 skipped,")
    assert "no pinned digest" in proc.stdout


def test_pin_suite_skips_when_the_package_is_not_installed(tmp_path: Path) -> None:
    proc = _inner_pytest(tmp_path, "import sys\nsys.modules['tree_sitter_language_pack'] = None\n")
    assert proc.returncode in (0, 5), proc.stdout  # 5: with -k, nothing left to select after the module skip
    assert proc.stdout.strip().splitlines()[-1].startswith("1 skipped")
    assert "not installed" in proc.stdout

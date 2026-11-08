"""Tests for repofinder/parser_java.py (Java language module), written from the contract.

Not registered in parser.LANGUAGES yet, so the module functions are used directly.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import tree_sitter

from repofinder.models import FileRecord, ImportEdge
from repofinder.parser import SourceFile
from repofinder.parser_java import extract_java, java_grammar, java_resolver

REL = "src/main/java/a/b/X.java"


def parse(src: str, rel: str = REL):
    data = textwrap.dedent(src).lstrip("\n").encode("utf-8")
    tree = tree_sitter.Parser(tree_sitter.Language(java_grammar())).parse(data)
    root = Path("/repo")
    sf = SourceFile(root=root, abs_path=root / rel, rel_path=rel, data=data)
    return extract_java(sf, tree)


def syms(src: str):
    return parse(src)[0]


def by_q(symbols, qualname):
    return [s for s in symbols if s.qualname == qualname]


def one(symbols, qualname):
    found = by_q(symbols, qualname)
    assert len(found) == 1, f"{qualname}: {[s.qualname for s in symbols]}"
    return found[0]


def quals(symbols):
    return [s.qualname for s in symbols]


# --------------------------------------------------------------------------- #
# Kinds, names, nesting
# --------------------------------------------------------------------------- #


def test_class_kind_and_id():
    s = one(syms("public final class Gson {\n}\n"), "Gson")
    assert s.kind == "class"
    assert s.name == "Gson"
    assert s.parent is None
    assert s.path == REL
    assert s.id == f"{REL}::Gson"


def test_record_interface_annotation_enum_kinds():
    src = """
    record Point(int x, int y) {}
    interface Shape { int area(); }
    @interface Since { int value(); }
    enum Color { RED, GREEN }
    """
    ss = syms(src)
    assert one(ss, "Point").kind == "class"
    assert one(ss, "Shape").kind == "interface"
    assert one(ss, "Since").kind == "interface"
    assert one(ss, "Color").kind == "enum"


def test_nested_classes_and_methods():
    src = """
    class Outer {
        static class Inner {
            void m() {}
        }
    }
    """
    ss = syms(src)
    inner = one(ss, "Outer.Inner")
    assert inner.kind == "class" and inner.parent == "Outer"
    m = one(ss, "Outer.Inner.m")
    assert m.kind == "method" and m.parent == "Outer.Inner" and m.name == "m"


def test_three_level_nesting():
    src = """
    class A {
        class B {
            class C {}
        }
    }
    """
    ss = syms(src)
    c = one(ss, "A.B.C")
    assert c.parent == "A.B"
    assert c.name == "C"
    assert one(ss, "A.B").parent == "A"


def test_methods_in_class_interface_and_enum():
    src = """
    class T { void run() {} }
    interface Shape { int area(); }
    enum Color {
        RED, GREEN;
        Color next() { return GREEN; }
    }
    """
    ss = syms(src)
    run = one(ss, "T.run")
    assert run.kind == "method" and run.parent == "T"
    area = one(ss, "Shape.area")
    assert area.kind == "method" and area.parent == "Shape"
    nxt = one(ss, "Color.next")
    assert nxt.kind == "method" and nxt.parent == "Color"


def test_constructors():
    src = """
    public class Gson {
        public Gson() {}
    }
    record R(int x) { R { } }
    """
    ss = syms(src)
    ctor = one(ss, "Gson.Gson")
    assert ctor.kind == "method" and ctor.parent == "Gson"
    assert one(ss, "R.R").kind == "method"


def test_overloads_share_qualname_and_id():
    src = """
    class C {
        void add(int a) {}
        void add(String s) {}
    }
    """
    adds = by_q(syms(src), "C.add")
    assert len(adds) == 2
    assert {a.id for a in adds} == {f"{REL}::C.add"}
    assert all(a.kind == "method" for a in adds)
    assert adds[0].start_line != adds[1].start_line


def test_excluded_constructs_are_not_symbols():
    src = """
    import java.util.function.Supplier;

    class Host {
        int field = 1;
        static int counter;
        static { counter = 2; }
        { field = 3; }
        void run() {
            class Local { void hidden() {} }
            Runnable r = new Runnable() {
                public void run() {}
            };
            Supplier<Integer> s = () -> 1;
        }
        Object anon = new Object() { void inField() {} };
    }
    enum Color {
        RED { void special() {} },
        GREEN;
        void real() {}
    }
    @interface Since {
        String value();
    }
    """
    ss = syms(src)
    assert sorted(quals(ss)) == sorted(
        ["Host", "Host.run", "Color", "Color.real", "Since"]
    )
    qs = quals(ss)
    assert "Host.field" not in qs
    assert not any("Local" in q or "hidden" in q for q in qs)
    assert not any("RED" in q or "special" in q for q in qs)
    assert "Since.value" not in qs


# --------------------------------------------------------------------------- #
# Ranges
# --------------------------------------------------------------------------- #


def test_range_includes_annotation_not_javadoc():
    src = """\
class C {
    /** Doc here. */
    @Override
    public String toString() {
        return "";
    }
}
"""
    m = one(syms(src), "C.toString")
    assert (m.start_line, m.end_line) == (3, 6)


def test_range_two_annotations_starts_at_first():
    src = """\
class C {
    @Deprecated
    @SuppressWarnings("x")
    void old() {
    }
}
"""
    m = one(syms(src), "C.old")
    assert (m.start_line, m.end_line) == (2, 5)


def test_range_class_and_bodiless_method():
    src = """\
interface Shape {
    int area();
}
"""
    ss = syms(src)
    a = one(ss, "Shape.area")
    assert (a.start_line, a.end_line) == (2, 2)
    s = one(ss, "Shape")
    assert (s.start_line, s.end_line) == (1, 3)


# --------------------------------------------------------------------------- #
# Signatures
# --------------------------------------------------------------------------- #


def test_signature_generic_method():
    src = """\
class Gson {
    @Override
    public <T> T fromJson(String json, Class<T> classOfT) {
        return null;
    }
}
"""
    m = one(syms(src), "Gson.fromJson")
    assert m.signature == "public <T> T fromJson(String json, Class<T> classOfT)"


def test_signature_class_ctor_and_bodiless():
    src = """\
public final class Gson {
    public Gson() {}
}
interface Shape {
    int area();
}
"""
    ss = syms(src)
    assert one(ss, "Gson").signature == "public final class Gson"
    assert one(ss, "Gson.Gson").signature == "public Gson()"
    assert one(ss, "Shape.area").signature == "int area()"


def test_signature_whitespace_collapsed_multiline():
    src = """\
class C {
    public   void   go(int a,
                       int b)
        throws Exception {
    }
}
"""
    sig = one(syms(src), "C.go").signature
    assert "\n" not in sig and "  " not in sig
    assert sig.startswith("public void go(int a,")
    assert sig.endswith("throws Exception")
    assert "{" not in sig


# --------------------------------------------------------------------------- #
# Docstrings
# --------------------------------------------------------------------------- #


def test_docstring_single_line():
    src = """\
class C {
    /** Converts JSON. */
    void a() {}
}
"""
    assert one(syms(src), "C.a").docstring == "Converts JSON."


def test_docstring_multiline_strips_stars():
    src = """\
class C {
    /**
     * First line.
     * Second line.
     */
    void a() {}
}
"""
    assert one(syms(src), "C.a").docstring == "First line.\nSecond line."


def test_docstring_stops_at_blank_line():
    src = """\
class C {
    /**
     * Summary.
     *
     * More detail after blank.
     */
    void a() {}
}
"""
    assert one(syms(src), "C.a").docstring == "Summary."


def test_docstring_stops_at_block_tag():
    src = """\
class C {
    /**
     * Does a thing.
     * @param x the x
     * @return nothing
     */
    void a(int x) {}
}
"""
    assert one(syms(src), "C.a").docstring == "Does a thing."


def test_docstring_above_annotated_method():
    src = """\
class C {
    /** Annotated doc. */
    @Override
    public String toString() { return ""; }
}
"""
    assert one(syms(src), "C.toString").docstring == "Annotated doc."


def test_docstring_none_for_plain_comments():
    src = """\
class C {
    /* not doc */
    void a() {}

    // x
    void b() {}

    void c() {}
}
"""
    ss = syms(src)
    assert one(ss, "C.a").docstring is None
    assert one(ss, "C.b").docstring is None
    assert one(ss, "C.c").docstring is None


def test_docstring_keeps_inline_tags():
    src = """\
class C {
    /** Returns {@code x} value. */
    int a() { return 1; }
}
"""
    assert one(syms(src), "C.a").docstring == "Returns {@code x} value."


def test_docstring_on_class():
    src = """\
/** The facade. */
public class Gson {
}
"""
    assert one(syms(src), "Gson").docstring == "The facade."


# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #


def test_imports_forms_and_lines():
    src = """\
package a.b;

import com.google.gson.internal.bind.TypeAdapters;
import java.util.*;
import static org.junit.Assert.assertEquals;
import static a.b.C.*;

class X {}
"""
    _, imps = parse(src)
    assert [(e.dst_module, e.names, e.line) for e in imps] == [
        ("com.google.gson.internal.bind", ["TypeAdapters"], 3),
        ("java.util", ["*"], 4),
        ("org.junit.Assert", ["assertEquals"], 5),
        ("a.b.C", ["*"], 6),
    ]
    for e in imps:
        assert e.src_path == REL
        assert e.dst_path is None


def test_package_declaration_makes_no_edge():
    _, imps = parse("package a.b;\n\nclass X {}\n")
    assert imps == []


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #

MAIN = "gson/src/main/java/com/google/gson/"
BIND = MAIN + "internal/bind/"
FILES = [
    MAIN + "Gson.java",
    BIND + "TypeAdapters.java",
    BIND + "JsonTreeReader.java",
    "gson/src/test/java/com/google/gson/Gson.java",
    "extras/src/main/java/com/google/gson/extras/Foo.java",
]


def resolver():
    recs = [FileRecord(p, "java", 10, 1, True) for p in FILES]
    return java_resolver(recs)


def edge(module, names):
    return ImportEdge(src_path=REL, dst_module=module, dst_path=None, names=names, line=1)


def test_resolve_class_import():
    r = resolver()
    assert r(edge("com.google.gson.internal.bind", ["TypeAdapters"])) == BIND + "TypeAdapters.java"


def test_resolve_prefers_main_over_test_sources():
    r = resolver()
    assert r(edge("com.google.gson", ["Gson"])) == MAIN + "Gson.java"


def test_resolve_nested_class_import_to_outer_file():
    r = resolver()
    assert r(edge("com.google.gson.Gson", ["FutureTypeAdapter"])) == MAIN + "Gson.java"


def test_resolve_package_wildcard_is_deterministic():
    r = resolver()
    first = r(edge("com.google.gson.internal.bind", ["*"]))
    assert first in (BIND + "TypeAdapters.java", BIND + "JsonTreeReader.java")
    for _ in range(5):
        assert r(edge("com.google.gson.internal.bind", ["*"])) == first
    # A freshly built resolver agrees as well.
    assert resolver()(edge("com.google.gson.internal.bind", ["*"])) == first


def test_resolve_static_wildcard():
    r = resolver()
    assert r(edge("com.google.gson.internal.bind.TypeAdapters", ["*"])) == BIND + "TypeAdapters.java"


def test_resolve_external_and_missing_is_none():
    r = resolver()
    assert r(edge("java.util", ["List"])) is None
    assert r(edge("org.example", ["Missing"])) is None


def test_module_import_makes_no_edge():
    src = """\
import module java.base;
import a.b.C;

class A {}
"""
    _, imps = parse(src)
    assert [(e.dst_module, e.names) for e in imps] == [("a.b", ["C"])]

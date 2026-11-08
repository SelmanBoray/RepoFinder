"""Tests for repofinder.chunker: search hits -> exact-line context chunks."""

from __future__ import annotations

from pathlib import Path

import pytest

from repofinder import chunker
from repofinder.chunker import build_chunks, class_header_end, split_lines, split_windows
from repofinder.ingest import build_index
from repofinder.models import FileRecord, RepoIndex, Symbol
from repofinder.search import Hit

MOD = '''"""Module doc."""
import os


def small(a, b):
    """Add."""
    return a + b


class Widget:
    """A widget.

    More text.
    """

    size = 1

    def grow(self, n):
        return self.size + n

    def shrink(self, n):
        return self.size - n


class Verbose:
{long_doc}
    def act(self):
        return 1


def outer():
    x = 1

    def inner():
        return x

    return inner
'''

LONG_DOC = '    """Verbose class.\n' + "".join(f"    line {i}\n" for i in range(20)) + '    """\n'


def _long_function(name: str, n_body: int) -> str:
    body = []
    for i in range(n_body):
        if i % 10 == 9:
            body.append("")
        else:
            body.append(f"    v{i} = {i}")
    return f"def {name}():\n" + "\n".join(body) + "\n    return 0\n"


def _long_class(name: str, n_methods: int) -> str:
    parts = [f"class {name}:", '    """Doc."""', ""]
    for i in range(n_methods):
        parts += [f"    @staticmethod", f"    def m{i}():", f"        a = {i}", f"        b = a + 1",
                  f"        return b", ""]
    return "\n".join(parts) + "\n"


@pytest.fixture
def repo(make_repo) -> tuple[Path, RepoIndex]:
    root = make_repo({
        "pkg/__init__.py": "",
        "pkg/mod.py": MOD.replace("{long_doc}", LONG_DOC),
        "pkg/longf.py": _long_function("long_func", 420) + "\n\n" + _long_function("mid_func", 200),
        "pkg/longc.py": _long_class("Big", 40),
        "pkg/head.py": "".join(f"# comment {i}\n" for i in range(55))
        + "def straddle():\n" + "".join(f"    s{i} = {i}\n" for i in range(10)) + "    return 1\n",
    })
    return root, build_index(root, "repo", str(root))


def _sym(index: RepoIndex, qualname: str, path: str | None = None) -> Symbol:
    found = [s for s in index.symbols if s.qualname == qualname and (path is None or s.path == path)]
    assert len(found) == 1, (qualname, found)
    return found[0]


def _hit(sym: Symbol) -> Hit:
    return Hit(path=sym.path, symbol_id=sym.id, start_line=sym.start_line, end_line=sym.end_line,
               score=1.0, kind="symbol", why="test")


def _file_hit(index: RepoIndex, path: str) -> Hit:
    record = next(f for f in index.files if f.path == path)
    return Hit(path=path, symbol_id=None, start_line=1, end_line=record.n_lines, score=1.0, kind="file", why="test")


def _file_lines(root: Path, path: str) -> list[str]:
    return split_lines((root / path).read_bytes().decode("utf-8"))


def _assert_exact(root: Path, chunk) -> None:
    lines = _file_lines(root, chunk.path)
    assert chunk.text == "\n".join(lines[chunk.start_line - 1 : chunk.end_line])
    assert len(chunk.text.split("\n")) == chunk.end_line - chunk.start_line + 1


def test_whole_symbol(repo):
    root, index = repo
    small = _sym(index, "small")
    chunks = build_chunks(root, index, [_hit(small)], 10_000)
    assert len(chunks) == 1
    c = chunks[0]
    assert (c.path, c.start_line, c.end_line, c.label) == ("pkg/mod.py", small.start_line, small.end_line, "small")
    assert c.text.startswith("def small(a, b):") and c.text.endswith("return a + b")
    _assert_exact(root, c)


def test_text_keeps_lines_exactly_without_terminators(make_repo):
    content = "\ufeffdef f():\r\n\tx = 1   \r\n\treturn x\r\n"
    root = make_repo({"m.py": content.encode("utf-8")})
    index = build_index(root, "r", str(root))
    f = _sym(index, "f")
    [chunk] = build_chunks(root, index, [_hit(f)], 10_000)
    assert chunk.text == "def f():\n\tx = 1   \n\treturn x"
    assert (chunk.start_line, chunk.end_line) == (1, 3)


def test_invalid_utf8_is_replaced(make_repo):
    root = make_repo({"m.py": b"def f():\n    return '\xff'\n"})
    index = build_index(root, "r", str(root))
    [chunk] = build_chunks(root, index, [_hit(_sym(index, "f"))], 10_000)
    assert chunk.text == "def f():\n    return '\ufffd'"


def test_long_function_is_split_into_windows(repo):
    root, index = repo
    long_func = _sym(index, "long_func")
    n = long_func.end_line - long_func.start_line + 1
    assert n > 400
    chunks = build_chunks(root, index, [_hit(long_func)], 100_000)
    total_windows = len(split_windows(_file_lines(root, long_func.path), long_func.start_line, long_func.end_line))
    assert len(chunks) == chunker.MAX_FUNCTION_WINDOWS < total_windows
    assert chunks[0].start_line == long_func.start_line
    assert chunks[0].text.startswith("def long_func():")
    for i, c in enumerate(chunks, start=1):
        assert chunker.WINDOW_MIN_LINES <= c.end_line - c.start_line + 1 <= chunker.WINDOW_MAX_LINES
        assert c.label == f"long_func (part {i}/{total_windows})"
        _assert_exact(root, c)
    for a, b in zip(chunks, chunks[1:]):
        assert b.start_line == a.end_line + 1  # contiguous and not merged


def test_windows_cover_a_long_function_completely(repo):
    root, index = repo
    mid = _sym(index, "mid_func")
    chunks = build_chunks(root, index, [_hit(mid)], 100_000)
    assert len(chunks) == 2
    assert chunks[0].start_line == mid.start_line and chunks[-1].end_line == mid.end_line
    assert chunks[0].end_line + 1 == chunks[1].start_line
    # The cut lands after a blank line, so the next window starts at a statement.
    lines = _file_lines(root, mid.path)
    assert lines[chunks[0].end_line - 1] == ""


def test_split_windows_sizes_and_boundaries(repo):
    root, index = repo
    big = _sym(index, "Big")
    lines = _file_lines(root, big.path)
    windows = split_windows(lines, big.start_line, big.end_line)
    assert windows[0][0] == big.start_line and windows[-1][1] == big.end_line
    for (a, b), (c, _d) in zip(windows, windows[1:]):
        assert c == b + 1
        assert chunker.WINDOW_MIN_LINES <= b - a + 1 <= chunker.WINDOW_MAX_LINES
        # Never split a decorator from its def; start at a method boundary.
        assert lines[c - 1].strip() == "@staticmethod"


def test_long_class_keeps_only_its_first_window(repo):
    root, index = repo
    big = _sym(index, "Big")
    chunks = build_chunks(root, index, [_hit(big)], 100_000)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.start_line == big.start_line and c.text.startswith("class Big:")
    assert c.end_line - c.start_line + 1 <= chunker.WINDOW_MAX_LINES
    n = big.end_line - big.start_line + 1
    assert c.label == f"Big (first {c.end_line - c.start_line + 1} of {n} lines)"


def test_class_header_is_prepended_once_per_class(repo):
    root, index = repo
    grow, shrink, widget = _sym(index, "Widget.grow"), _sym(index, "Widget.shrink"), _sym(index, "Widget")
    chunks = build_chunks(root, index, [_hit(grow), _hit(shrink)], 10_000)
    assert [c.label for c in chunks] == ["Widget (class header)", "Widget.grow", "Widget.shrink"]
    header = chunks[0]
    assert header.start_line == widget.start_line
    assert header.text == 'class Widget:\n    """A widget.\n\n    More text.\n    """'
    for c in chunks:
        _assert_exact(root, c)


def test_class_header_skipped_when_docstring_is_long(repo):
    root, index = repo
    act = _sym(index, "Verbose.act")
    chunks = build_chunks(root, index, [_hit(act)], 10_000)
    assert [c.label for c in chunks] == ["Verbose.act"]


def test_class_header_not_repeated_when_class_is_included(repo):
    root, index = repo
    widget, grow = _sym(index, "Widget"), _sym(index, "Widget.grow")
    chunks = build_chunks(root, index, [_hit(widget), _hit(grow)], 10_000)
    assert len(chunks) == 1
    assert (chunks[0].start_line, chunks[0].end_line, chunks[0].label) == (widget.start_line, widget.end_line, "Widget")


def test_class_header_end_handles_multiline_bases_and_decorators():
    lines = split_lines(
        "@decorate(\n    1)\nclass A(\n    Base,\n):\n    # comment\n\n    r'''Doc\n    more.\n    '''\n    x = 1\n"
    )
    sym = Symbol(id="m.py::A", path="m.py", name="A", qualname="A", kind="class", start_line=1,
                 end_line=11, signature="class A(Base)", docstring="Doc", parent=None)
    assert class_header_end(lines, sym) == 10
    no_doc = split_lines("class B:\n    x = 1\n")
    sym_b = Symbol(id="m.py::B", path="m.py", name="B", qualname="B", kind="class", start_line=1,
                   end_line=2, signature="class B", docstring=None, parent=None)
    assert class_header_end(no_doc, sym_b) == 1


def test_nested_function_uses_the_enclosing_function(repo):
    root, index = repo
    inner, outer = _sym(index, "outer.<locals>.inner"), _sym(index, "outer")
    chunks = build_chunks(root, index, [_hit(inner)], 10_000)
    assert [(c.start_line, c.end_line, c.label) for c in chunks] == [(outer.start_line, outer.end_line, "outer")]


def test_file_hit_gives_module_head(repo):
    root, index = repo
    chunks = build_chunks(root, index, [_file_hit(index, "pkg/longf.py")], 10_000)
    assert len(chunks) == 1
    c = chunks[0]
    assert (c.start_line, c.end_line, c.label) == (1, chunker.MODULE_HEAD_LINES, "module head")
    _assert_exact(root, c)
    short = build_chunks(root, index, [_file_hit(index, "pkg/mod.py")], 100_000)[0]
    assert short.end_line == len(_file_lines(root, "pkg/mod.py")) < chunker.MODULE_HEAD_LINES


def test_overlapping_ranges_are_merged(repo):
    root, index = repo
    straddle = _sym(index, "straddle")
    assert straddle.start_line <= chunker.MODULE_HEAD_LINES < straddle.end_line
    chunks = build_chunks(root, index, [_file_hit(index, "pkg/head.py"), _hit(straddle)], 10_000)
    assert len(chunks) == 1
    c = chunks[0]
    assert (c.start_line, c.end_line) == (1, straddle.end_line)
    assert c.label == "module head + straddle"
    _assert_exact(root, c)


def test_contained_ranges_are_dropped(repo):
    root, index = repo
    outer, small = _sym(index, "outer"), _sym(index, "small")
    chunks = build_chunks(root, index, [_file_hit(index, "pkg/mod.py"), _hit(small), _hit(outer)], 100_000)
    assert [(c.path, c.label) for c in chunks] == [("pkg/mod.py", "module head")]


def test_max_chars_cutoff(repo):
    root, index = repo
    hits = [_hit(_sym(index, q)) for q in ("small", "Widget", "outer", "mid_func")]
    full = build_chunks(root, index, hits, 1_000_000)
    sizes = [len(c.text) for c in full]
    budget = sizes[0] + sizes[1] + 5
    cut = build_chunks(root, index, hits, budget)
    assert sum(len(c.text) for c in cut) <= budget
    assert [c.label for c in cut][:2] == [c.label for c in full][:2]
    assert len(cut) < len(full)
    assert build_chunks(root, index, hits, 0) == []


def test_smaller_lower_ranked_chunk_still_fits(repo):
    root, index = repo
    small, mid, outer = _sym(index, "small"), _sym(index, "mid_func"), _sym(index, "outer")
    sizes = {c.label: len(c.text) for c in build_chunks(root, index, [_hit(small), _hit(outer)], 100_000)}
    budget = sizes["small"] + sizes["outer"] + 1
    labels = [c.label for c in build_chunks(root, index, [_hit(small), _hit(mid), _hit(outer)], budget)]
    assert labels == ["small", "outer"]


def test_first_chunk_is_trimmed_when_alone_too_large(repo):
    root, index = repo
    widget = _sym(index, "Widget")
    [chunk] = build_chunks(root, index, [_hit(widget)], 40)
    assert chunk.start_line == widget.start_line and chunk.end_line < widget.end_line
    assert len(chunk.text) <= 40
    _assert_exact(root, chunk)


def test_unknown_or_missing_files_are_skipped(repo):
    root, index = repo
    ghost = Hit(path="pkg/ghost.py", symbol_id=None, start_line=1, end_line=10, score=1.0, kind="file", why="")
    stale = Hit(path="pkg/mod.py", symbol_id="pkg/mod.py::gone", start_line=5, end_line=7, score=1.0,
                kind="symbol", why="")
    escape = Hit(path="../outside.py", symbol_id=None, start_line=1, end_line=1, score=1.0, kind="file", why="")
    chunks = build_chunks(root, index, [ghost, stale, escape], 10_000)
    assert [(c.path, c.start_line, c.end_line, c.label) for c in chunks] == [("pkg/mod.py", 5, 7, "gone")]


def test_deterministic(repo):
    root, index = repo
    hits = [_hit(s) for s in index.symbols] + [_file_hit(index, f.path) for f in index.files]
    assert build_chunks(root, index, hits, 30_000) == build_chunks(root, index, hits, 30_000)


# --------------------------------------------------------------------------- #
# TypeScript
# --------------------------------------------------------------------------- #

TS_MOD = """import { Base } from "./base";

/** A widget. */
@sealed
export class Widget extends Base {
  #size = {
    n: 1 };

  grow(n: number): number {
    return this.#size.n + n;
  }
}

export function helper(a: string): string {
  return `x${a}`;
}
"""


def _ts_index(root: Path) -> tuple[RepoIndex, Symbol, Symbol, Symbol]:
    path = "src/widget.ts"
    widget = Symbol(f"{path}::Widget", path, "Widget", "Widget", "class", 4, 12,
                    "export class Widget extends Base", "A widget.", None)
    grow = Symbol(f"{path}::Widget.grow", path, "grow", "Widget.grow", "method", 9, 11,
                  "grow(n: number): number", None, "Widget")
    helper = Symbol(f"{path}::helper", path, "helper", "helper", "function", 14, 16,
                    "export function helper(a: string): string", None, None)
    index = RepoIndex("r", str(root), "r", files=[FileRecord(path, "typescript", len(TS_MOD), 16, True)],
                      symbols=[widget, grow, helper])
    return index, widget, grow, helper


def test_typescript_class_header_ends_at_the_opening_brace() -> None:
    lines = split_lines(TS_MOD)
    _index, widget, _grow, _helper = _ts_index(Path("."))
    assert class_header_end(lines, widget) == 5
    multiline = split_lines("// class in a comment {\nexport class A\n  extends B\n  implements C {\n  x = 1;\n}\n")
    a = Symbol("m.ts::A", "m.ts", "A", "A", "class", 1, 6, "export class A", None, None)
    assert class_header_end(multiline, a) == 4
    no_body = Symbol("m.ts::A", "m.ts", "A", "A", "class", 1, 1, "class A", None, None)
    assert class_header_end(["export declare class A"], no_body) is None


def test_typescript_chunks_are_sliced_by_line_range(make_repo) -> None:
    root = make_repo({"src/widget.ts": TS_MOD})
    index, _widget, grow, helper = _ts_index(root)
    chunks = build_chunks(root, index, [_hit(grow), _hit(helper)], 10_000)
    assert [(c.start_line, c.end_line, c.label) for c in chunks] == [
        (4, 5, "Widget (class header)"),
        (9, 11, "Widget.grow"),
        (14, 16, "helper"),
    ]
    assert chunks[0].text == "@sealed\nexport class Widget extends Base {"
    assert chunks[2].text.splitlines()[0] == "export function helper(a: string): string {"


DART_MOD = """/// Assets.
class _Asset<T extends Object>
    extends Base
    implements Other {
  final int value;
  int get size => value;
}

mixin Walker on Animal {
  void walk() {}
}

extension StringX on String {
  int get n => length;
}

enum Color { red, green }
"""


def _dart(qualname: str, kind: str, start: int, end: int, parent: str | None = None) -> Symbol:
    name = qualname.rsplit(".", 1)[-1]
    return Symbol(f"a.dart::{qualname}", "a.dart", name, qualname, kind, start, end, qualname, None, parent)


def test_dart_class_header_ends_at_the_opening_brace() -> None:
    lines = split_lines(DART_MOD)
    assert class_header_end(lines, _dart("_Asset", "class", 2, 7)) == 4
    assert class_header_end(lines, _dart("Walker", "class", 9, 11)) == 9
    assert class_header_end(lines, _dart("on String", "class", 13, 15)) == 13
    assert class_header_end(lines, _dart("Color", "enum", 17, 17)) == 17
    # not a Python file: "mixin" only counts in Dart
    ts = Symbol("a.ts::Walker", "a.ts", "Walker", "Walker", "class", 9, 11, "", None, None)
    assert class_header_end(lines, ts) is None


def test_dart_method_chunk_gets_class_header(make_repo) -> None:
    root = make_repo({"a.dart": DART_MOD})
    cls = _dart("Walker", "class", 9, 11)
    walk = _dart("Walker.walk", "method", 10, 10, "Walker")
    ops = _dart("Walker.operator ==", "method", 10, 10, "Walker")
    index = RepoIndex("r", str(root), "r", files=[FileRecord("a.dart", "dart", len(DART_MOD), 18, True)],
                      symbols=[cls, walk, ops])
    chunks = build_chunks(root, index, [_hit(walk), _hit(ops)], 100_000)
    assert chunks and "mixin Walker on Animal {" in chunks[0].text
    assert "Walker (class header)" in chunks[0].label


def test_dart_header_skips_braces_inside_parentheses_strings_and_comments() -> None:
    def end(src: str, kind: str = "class") -> int | None:
        lines = split_lines(src)
        return class_header_end(lines, _dart("A", kind, 1, len(lines)))

    flame = (
        "class const AnchorPicker({\n"
        "  required final String anchor,\n"
        "  required final ValueChanged<String> onSelected,\n"
        "  super.key,\n"
        "}) extends StatelessWidget {\n"
        "  final int x = 1;\n"
        "}\n"
    )
    assert end(flame) == 5
    assert end("sealed class _Asset<T>(var T value);\n") == 1
    assert end("sealed class _Asset<T>(\n  var T value,\n);\n") == 3
    assert end("extension type Foo({int x}) {\n  int get y => x;\n}\n", "class") == 1
    assert end("extension type const Foo(int x) implements int {\n}\n") == 1
    assert end("enum LayerFill() {\n  a;\n}\n", "enum") == 1
    assert end("class A({String s = '{', Map m = const {}}) // {\n    extends B /* { */ {\n}\n") == 2
    assert end("@Foo('class X {')\nclass A\n    extends B {\n}\n") == 3
    assert end("class A extends B\n") is None


DART_ENUM = """\
/// Kinds.
enum Kind {
  a(0),
  b(1);

  const Kind(this.v);
  final int v;

  factory Kind.of(int v) => a;
}
"""


def test_dart_enum_member_chunk_gets_enum_header(make_repo) -> None:
    root = make_repo({"a.dart": DART_ENUM})
    enum = _dart("Kind", "enum", 2, 11)
    of = _dart("Kind.of", "method", 10, 10, "Kind")
    index = RepoIndex("r", str(root), "r", files=[FileRecord("a.dart", "dart", len(DART_ENUM), 11, True)],
                      symbols=[enum, of])
    chunks = build_chunks(root, index, [_hit(of)], 100_000)
    assert [c.label for c in chunks] == ["Kind (enum header)", "Kind.of"]
    assert chunks[0].start_line == chunks[0].end_line == 2  # the header ends at the brace, values are not pulled in


def test_non_dart_enum_parent_gets_no_header(make_repo) -> None:
    src = "enum Kind {\n  A,\n  B,\n}\nfunction f() {}\n"
    root = make_repo({"a.ts": src})
    enum = Symbol("a.ts::Kind", "a.ts", "Kind", "Kind", "enum", 1, 4, "", None, None)
    m = Symbol("a.ts::Kind.f", "a.ts", "f", "Kind.f", "method", 2, 2, "", None, "Kind")
    index = RepoIndex("r", str(root), "r", files=[FileRecord("a.ts", "typescript", len(src), 5, True)],
                      symbols=[enum, m])
    chunks = build_chunks(root, index, [_hit(m)], 100_000)
    assert [c.label for c in chunks] == ["Kind.f"]


# --- Java / Go / Rust owner headers -------------------------------------------------------------------------------


def _owner_chunks(make_repo, path: str, lang: str, src: str, syms: list[Symbol], target: Symbol):
    root = make_repo({path: src})
    n = len(src.split("\n"))
    index = RepoIndex("r", str(root), "r", files=[FileRecord(path, lang, len(src), n, True)], symbols=syms)
    return build_chunks(root, index, [_hit(target)], 100_000)


def _msym(path: str, qualname: str, kind: str, start: int, end: int, parent: str | None = None) -> Symbol:
    return Symbol(f"{path}::{qualname}", path, qualname.rsplit(".", 1)[-1], qualname, kind, start, end,
                  qualname, None, parent)


def _java_case(make_repo, src: str, type_q: str, type_kind: str, t_range: tuple[int, int],
               method_q: str, m_range: tuple[int, int], parent: str):
    p = "src/A.java"
    t = _msym(p, type_q, type_kind, *t_range, parent=None)
    m = _msym(p, method_q, "method", *m_range, parent=parent)
    return _owner_chunks(make_repo, p, "java", src, [t, m], m)


def test_java_class_method_gets_class_header(make_repo) -> None:
    src = "package a;\n\n@Deprecated\npublic final class Gson {\n  int f;\n  public void toJson() {\n  }\n}\n"
    chunks = _java_case(make_repo, src, "Gson", "class", (3, 8), "Gson.toJson", (6, 7), "Gson")
    assert chunks and "public final class Gson {" in chunks[0].text
    assert "@Deprecated" in chunks[0].text
    assert chunks[0].start_line == 3 and chunks[0].end_line == 4
    assert "header" in chunks[0].label
    assert chunks[-1].label == "Gson.toJson"


def test_java_interface_method_gets_header(make_repo) -> None:
    src = "package a;\n\npublic interface JsonSerializer<T> {\n  void serialize(T t);\n}\n"
    chunks = _java_case(make_repo, src, "JsonSerializer", "interface", (3, 5), "JsonSerializer.serialize", (4, 4),
                        "JsonSerializer")
    assert chunks and "public interface JsonSerializer<T> {" in chunks[0].text
    assert chunks[0].end_line == 3


def test_java_enum_method_gets_header_without_constants(make_repo) -> None:
    src = ("public enum FieldNamingPolicy implements FieldNamingStrategy {\n  IDENTITY,\n  UPPER;\n\n"
           "  public String translate() {\n    return null;\n  }\n}\n")
    chunks = _java_case(make_repo, src, "FieldNamingPolicy", "enum", (1, 8), "FieldNamingPolicy.translate", (5, 7),
                        "FieldNamingPolicy")
    assert chunks and "public enum FieldNamingPolicy implements FieldNamingStrategy {" in chunks[0].text
    assert "IDENTITY" not in chunks[0].text
    assert chunks[0].end_line == 1


def test_java_record_method_gets_header(make_repo) -> None:
    src = "public record Point(int x, int y) {\n  int sum() {\n    return x + y;\n  }\n}\n"
    chunks = _java_case(make_repo, src, "Point", "class", (1, 5), "Point.sum", (2, 4), "Point")
    assert chunks and "public record Point(int x, int y) {" in chunks[0].text


def test_java_multiline_header_includes_both_lines(make_repo) -> None:
    src = ("public final class Excluder\n    implements TypeAdapterFactory, Cloneable {\n  int f;\n"
           "  public Excluder clone() {\n    return this;\n  }\n}\n")
    chunks = _java_case(make_repo, src, "Excluder", "class", (1, 7), "Excluder.clone", (4, 6), "Excluder")
    assert chunks
    assert "public final class Excluder\n    implements TypeAdapterFactory, Cloneable {" in chunks[0].text
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 2)


def test_java_nested_method_gets_inner_header_not_outer(make_repo) -> None:
    src = "public class Outer {\n  int a;\n  static class Inner {\n    void m() {\n    }\n  }\n}\n"
    p = "src/A.java"
    outer = _msym(p, "Outer", "class", 1, 7)
    inner = _msym(p, "Outer.Inner", "class", 3, 6, "Outer")
    m = _msym(p, "Outer.Inner.m", "method", 4, 5, "Outer.Inner")
    chunks = _owner_chunks(make_repo, p, "java", src, [outer, inner, m], m)
    assert chunks and "static class Inner {" in chunks[0].text
    assert "public class Outer {" not in chunks[0].text
    assert chunks[0].start_line == 3


def test_go_method_gets_struct_header(make_repo) -> None:
    src = "package s\n\ntype Server struct {\n\tAddr string\n}\n\nfunc (s *Server) Run() error {\n\treturn nil\n}\n"
    p = "s.go"
    t = _msym(p, "Server", "class", 3, 5)
    m = _msym(p, "Server.Run", "method", 7, 9, "Server")
    chunks = _owner_chunks(make_repo, p, "go", src, [t, m], m)
    assert chunks and "type Server struct {" in chunks[0].text
    assert chunks[-1].label == "Server.Run"
    assert "func (s *Server) Run" in chunks[-1].text


def test_go_method_on_named_type_gets_type_line(make_repo) -> None:
    src = 'package s\n\ntype ID int\n\nfunc (i ID) String() string {\n\treturn ""\n}\n'
    p = "s.go"
    t = _msym(p, "ID", "type", 3, 3)
    m = _msym(p, "ID.String", "method", 5, 7, "ID")
    chunks = _owner_chunks(make_repo, p, "go", src, [t, m], m)
    assert chunks and "type ID int" in chunks[0].text
    assert chunks[-1].label == "ID.String"


def test_go_receiver_declared_elsewhere_gets_no_header(make_repo) -> None:
    src = "package s\n\nfunc (s *Server) Run() error {\n\treturn nil\n}\n"
    p = "s2.go"
    m = _msym(p, "Server.Run", "method", 3, 5, "Server")
    chunks = _owner_chunks(make_repo, p, "go", src, [m], m)
    assert [c.label for c in chunks] == ["Server.Run"]


def test_rust_impl_method_gets_impl_line(make_repo) -> None:
    src = ("pub struct StandardSink<'s, M> {\n    m: M,\n}\n\n"
           "impl<'s, M: Matcher> Sink for StandardSink<'s, M> {\n    fn matched(&mut self) -> bool {\n"
           "        true\n    }\n}\n")
    p = "src/sink.rs"
    t = _msym(p, "StandardSink", "class", 1, 3)
    m = _msym(p, "StandardSink.matched", "method", 6, 8, "StandardSink")
    chunks = _owner_chunks(make_repo, p, "rust", src, [t, m], m)
    assert chunks and "impl<'s, M: Matcher> Sink for StandardSink<'s, M> {" in chunks[0].text
    assert chunks[-1].label == "StandardSink.matched"


def test_rust_impl_header_includes_cfg_test_attribute(make_repo) -> None:
    src = ("pub struct S {\n}\n\n#[cfg(test)]\nimpl<'s, M: Matcher> Sink for S<'s, M> {\n"
           "    fn matched(&mut self) -> bool {\n        true\n    }\n}\n")
    p = "src/sink.rs"
    t = _msym(p, "S", "class", 1, 2)
    m = _msym(p, "S.matched", "method", 6, 8, "S")
    chunks = _owner_chunks(make_repo, p, "rust", src, [t, m], m)
    assert chunks
    assert "#[cfg(test)]" in chunks[0].text
    assert "impl<'s, M: Matcher> Sink for S<'s, M> {" in chunks[0].text


def test_rust_trait_method_gets_trait_line(make_repo) -> None:
    src = "pub trait Sink {\n    fn matched(&mut self) -> bool;\n}\n"
    p = "src/sink.rs"
    t = _msym(p, "Sink", "interface", 1, 3)
    m = _msym(p, "Sink.matched", "method", 2, 2, "Sink")
    chunks = _owner_chunks(make_repo, p, "rust", src, [t, m], m)
    assert chunks and "pub trait Sink {" in chunks[0].text
    assert chunks[-1].label == "Sink.matched"


def test_typescript_interface_member_still_gets_no_header(make_repo) -> None:
    src = "export interface Opts {\n  run(): void;\n}\n"
    p = "a.ts"
    t = _msym(p, "Opts", "interface", 1, 3)
    m = _msym(p, "Opts.run", "method", 2, 2, "Opts")
    chunks = _owner_chunks(make_repo, p, "typescript", src, [t, m], m)
    assert [c.label for c in chunks] == ["Opts.run"]


def test_python_method_still_gets_class_header_unchanged(make_repo) -> None:
    src = "class A:\n    def f(self):\n        return 1\n"
    p = "a.py"
    t = _msym(p, "A", "class", 1, 3)
    m = _msym(p, "A.f", "method", 2, 3, "A")
    chunks = _owner_chunks(make_repo, p, "python", src, [t, m], m)
    assert [c.label for c in chunks] == ["A (class header)", "A.f"]
    assert chunks[0].text == "class A:"

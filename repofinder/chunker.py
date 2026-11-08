"""Turn search hits into :class:`~repofinder.llm.ContextChunk` excerpts.

Every chunk is a contiguous run of source lines copied exactly from the file
(only the line terminator is removed), so the LLM layer can number the lines
from ``start_line`` and the model can cite them.

Rules, applied to the hits in rank order:

* A symbol hit becomes the symbol's whole line range, labelled with its
  qualname. A nested function or class (``outer.<locals>.inner``) is replaced
  by its enclosing top-level function or method when that is at most
  :data:`LONG_SYMBOL_LINES` long: a three-line closure means little without
  the function around it.
* A symbol longer than :data:`LONG_SYMBOL_LINES` is split into windows of
  :data:`WINDOW_MIN_LINES`-:data:`WINDOW_MAX_LINES` lines (the last one may be
  shorter). The first window starts at the ``def``/``class`` header (decorators
  included); cuts prefer statement boundaries at shallow indentation. A
  function keeps at most :data:`MAX_FUNCTION_WINDOWS` windows; a long class
  only its first window (its methods are symbols of their own and can be hit
  directly).
* For a method, the enclosing class header plus its docstring is prepended as a
  separate chunk, once per class, when it is shorter than
  :data:`CLASS_HEADER_MAX_LINES` lines. Outside Python (TypeScript, ...) the
  header runs from the class's first line to the line that opens its body
  (``{``). A Java method gets the header of its class, interface, enum or
  record; a Go method that of its receiver type when the type is declared in
  the same file; a Rust method that of its trait, or else the ``impl ... {``
  lines (with the attributes above them) of the block it sits in. In Java, Go
  and Rust no header is added when it shares a line with the method.
* A file hit becomes the first :data:`MODULE_HEAD_LINES` lines of the file
  (``label="module head"``).
* Overlapping ranges of the same file are merged into one chunk (adjacent ones
  are not, so the windows of a long symbol stay separate); a range already
  covered by an earlier chunk is dropped.
* The total text length stays within ``max_chars``: a chunk that does not fit
  is skipped and smaller, lower-ranked chunks may still be added. Only the
  first chunk is ever shortened (to its leading lines) to fit.

Files are read as UTF-8 with ``errors="replace"``; nothing is executed. The
output is deterministic for a given index, set of hits and checkout.
"""

from __future__ import annotations

import io
import logging
import re
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from .llm import ContextChunk
from .models import RepoIndex, Symbol
from .search import Hit

log = logging.getLogger(__name__)

LONG_SYMBOL_LINES = 120        # symbols longer than this are split into windows
WINDOW_MIN_LINES = 80
WINDOW_TARGET_LINES = 100
WINDOW_MAX_LINES = 120
MAX_FUNCTION_WINDOWS = 3       # windows kept per long function or method
MAX_CLASS_WINDOWS = 1          # windows kept per long class (its header part)
MODULE_HEAD_LINES = 60         # lines shown for a file-level hit
CLASS_HEADER_MAX_LINES = 15    # class header + docstring must be shorter than this
HEADER_SCAN_LINES = 40         # lines tokenized to find a header and docstring
MAX_LABELS = 3                 # labels named on a merged chunk

MODULE_HEAD_LABEL = "module head"
_BOM = "\ufeff"
_LOCALS = ".<locals>."
_PYTHON_SUFFIXES = (".py", ".pyi")
_CLASS_KEYWORD_RE = re.compile(r"\bclass\b")
# Dart's class-like declarations: ``mixin M {``, ``extension E on X {`` and ``enum`` headers have no
# ``class`` keyword (``mixin class`` and ``extension type`` are matched by their first word).
_DART_SUFFIXES = (".dart",)
_DART_KEYWORDS = frozenset({"class", "mixin", "extension", "enum"})
_COMMENT_PREFIXES = ("//", "/*", "*")
# Java, Go and Rust are read by one small tokenizer (:func:`_c_tokens`).
_JAVA_SUFFIXES = (".java",)
_GO_SUFFIXES = (".go",)
_RUST_SUFFIXES = (".rs",)
_JAVA_TYPE_KEYWORDS = frozenset({"class", "interface", "enum", "record"})  # `@interface` too
_RUST_TYPE_KEYWORDS = frozenset({"struct", "enum", "union", "trait"})
_CLASS_KINDS = frozenset({"class"})
_ENUM_KINDS = frozenset({"enum"})
_TYPE_OWNER_KINDS = frozenset({"class", "interface", "enum"})  # Java types and Rust traits
_GO_RECEIVER_KINDS = frozenset({"class", "interface", "type"})
_DECORATOR_PREFIXES = ("@",)
_RUST_DECORATOR_PREFIXES = ("@", "#[")  # outer attributes count as decorators (``#![...]`` does not)
_C_TOKEN_RE = re.compile(r"(?P<space>\s+)|(?P<line_comment>//[^\n]*)|(?P<block_comment>/\*)|(?P<word>[\w$]+)|.", re.S)


@dataclass
class _Range:
    path: str
    start: int
    end: int
    label: str


@dataclass
class _Piece:
    path: str
    start: int
    end: int
    labels: list[str] = field(default_factory=list)
    chars: int = 0


def build_chunks(ws_root: Path, index: RepoIndex, hits: list[Hit], max_chars: int) -> list[ContextChunk]:
    """Context chunks for ``hits`` (see the module docstring for the rules)."""
    if max_chars <= 0 or not hits:
        return []
    files = _FileCache(Path(ws_root))
    symbols = _SymbolTable(index)
    builder = _Builder(files, max_chars)
    headers_done: set[tuple[str, int]] = set()
    impl_owners: dict[str, list[Optional[tuple[int, int]]]] = {}  # per Rust file, see _rust_impl_owners

    for hit in hits:
        lines = files.lines(hit.path)
        if not lines:
            continue
        if hit.kind != "symbol" or not hit.symbol_id:
            builder.add_group(None, [_Range(hit.path, 1, min(MODULE_HEAD_LINES, len(lines)), MODULE_HEAD_LABEL)])
            continue

        sym = symbols.get(hit.symbol_id, hit.start_line)
        if sym is None:  # the index and the hit disagree; use the hit's own range
            label = hit.symbol_id.split("::", 1)[-1]
            builder.add_group(None, [_Range(hit.path, hit.start_line, hit.end_line, label)])
            continue

        target = symbols.context_symbol(sym)
        header: Optional[_Range] = None
        if target.kind == "method":
            header = _owner_header(symbols, target, lines, impl_owners)
            if header is not None and (header.path, header.start) in headers_done:
                header = None
        added_header = builder.add_group(header, symbol_ranges(target, lines))
        if added_header and header is not None:
            headers_done.add((header.path, header.start))

    return [
        ContextChunk(
            path=p.path,
            start_line=p.start,
            end_line=p.end,
            text=files.text(p.path, p.start, p.end),
            label=_join_labels(p.labels),
        )
        for p in builder.pieces
    ]


def _owner_header(
    symbols: _SymbolTable,
    method: Symbol,
    lines: list[str],
    impl_owners: dict[str, list[Optional[tuple[int, int]]]],
) -> Optional[_Range]:
    """The header range that names ``method``'s owner, or None: the enclosing
    class's header (see :func:`class_header_end`) or, in Rust, the ``impl ... {``
    lines of the block the method sits in. ``impl_owners`` caches
    :func:`_rust_impl_owners` per file."""
    lang = _c_lang(method.path)
    cls = symbols.enclosing_class(method)
    if cls is not None:
        end = class_header_end(lines, cls)
        if end is None or end - cls.start_line + 1 >= CLASS_HEADER_MAX_LINES:
            return None
        what = cls.kind if cls.kind in ("enum", "interface", "type") else "class"
        header = _Range(cls.path, cls.start_line, end, f"{cls.qualname} ({what} header)")
    elif lang == "rust":
        if method.path not in impl_owners:
            impl_owners[method.path] = _rust_impl_owners(lines)
        owners = impl_owners[method.path]
        span = owners[method.start_line] if 0 < method.start_line < len(owners) else None
        # The block is found by a whole-file pass, so HEADER_SCAN_LINES does not bound it; the
        # stricter CLASS_HEADER_MAX_LINES rule does.
        if span is None or span[1] - span[0] + 1 >= CLASS_HEADER_MAX_LINES:
            return None
        header = _Range(method.path, span[0], span[1], f"{method.parent or 'impl'} (impl header)")
    else:
        return None
    if lang is not None and header.start <= method.end_line and method.start_line <= header.end:
        return None  # a one-line type or impl: the method's own lines already show the owner
    return header


# --------------------------------------------------------------------------- #
# Ranges for one symbol
# --------------------------------------------------------------------------- #

def symbol_ranges(sym: Symbol, lines: list[str]) -> list[_Range]:
    """The line ranges that represent ``sym``: the whole symbol, or windows."""
    start = max(1, sym.start_line)
    end = min(sym.end_line, len(lines))
    if start > end:
        return []
    n = end - start + 1
    if n <= LONG_SYMBOL_LINES:
        return [_Range(sym.path, start, end, sym.qualname)]
    decorators = _RUST_DECORATOR_PREFIXES if sym.path.endswith(_RUST_SUFFIXES) else _DECORATOR_PREFIXES
    windows = split_windows(lines, start, end, decorators)
    if sym.kind == "class":
        a, b = windows[0]
        label = f"{sym.qualname} (first {b - a + 1} of {n} lines)"
        return [_Range(sym.path, a, b, label)][:MAX_CLASS_WINDOWS]
    total = len(windows)
    return [
        _Range(sym.path, a, b, f"{sym.qualname} (part {i}/{total})")
        for i, (a, b) in enumerate(windows[:MAX_FUNCTION_WINDOWS], start=1)
    ]


def split_windows(
    lines: list[str], start: int, end: int, decorators: tuple[str, ...] = _DECORATOR_PREFIXES
) -> list[tuple[int, int]]:
    """Split ``start..end`` (1-based, inclusive) into consecutive windows of
    ``WINDOW_MIN_LINES``-``WINDOW_MAX_LINES`` lines; only the last may be shorter.
    ``decorators`` are the prefixes of decorator lines (see :func:`_best_cut`)."""
    windows: list[tuple[int, int]] = []
    a = start
    while end - a + 1 > WINDOW_MAX_LINES:
        b = _best_cut(lines, a, end, decorators)
        windows.append((a, b))
        a = b + 1
    windows.append((a, end))
    return windows


def _best_cut(lines: list[str], a: int, end: int, decorators: tuple[str, ...] = _DECORATOR_PREFIXES) -> int:
    """Last line of the window starting at ``a``: the next window should start at
    a statement with the shallowest indentation, not inside a decorator stack
    (lines starting with one of ``decorators``: ``@``, and in Rust ``#[``), as
    close to ``WINDOW_TARGET_LINES`` as possible."""
    lo = a + WINDOW_MIN_LINES - 1
    hi = min(a + WINDOW_MAX_LINES - 1, end - 1)
    target = a + WINDOW_TARGET_LINES - 1
    best: Optional[tuple[tuple[int, int, int, int], int]] = None
    for b in range(lo, hi + 1):
        last = lines[b - 1]
        nxt = lines[b] if b < len(lines) else ""
        indent = _indent(nxt) if nxt.strip() else 10_000  # never start a window on a blank line
        cost = (
            1 if last.lstrip().startswith(decorators) else 0,  # do not split a decorator stack
            indent,
            0 if not last.strip() else 1,                   # prefer ending on a blank line
            abs(b - target),
        )
        if best is None or cost < best[0]:
            best = (cost, b)
    return best[1] if best is not None else min(a + WINDOW_TARGET_LINES - 1, end)


def _indent(line: str) -> int:
    return len(line.expandtabs(8)) - len(line.expandtabs(8).lstrip())


def class_header_end(lines: list[str], cls: Symbol) -> Optional[int]:
    """Last line of the class header (``class X(...):``) and of its docstring,
    if any. ``None`` when it cannot be found within ``HEADER_SCAN_LINES`` lines
    (e.g. a docstring longer than that).

    A class outside a Python file (TypeScript, Dart, Java, Go, Rust, ...) is read
    by :func:`_brace_header_end` instead: Python's tokenizer means nothing there."""
    if not cls.path.endswith(_PYTHON_SUFFIXES):
        return _brace_header_end(lines, cls)
    start = cls.start_line
    stop = min(cls.end_line, start + HEADER_SCAN_LINES - 1, len(lines))
    if start > stop:
        return None
    text = "\n".join(lines[start - 1 : stop]) + "\n"
    depth = 0
    seen_keyword = False
    header_end: Optional[int] = None
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if header_end is None:
                if tok.type == tokenize.OP and tok.string in "([{":
                    depth += 1
                elif tok.type == tokenize.OP and tok.string in ")]}":
                    depth = max(0, depth - 1)
                elif depth == 0 and tok.type == tokenize.NAME and tok.string == "class":
                    seen_keyword = True
                elif seen_keyword and depth == 0 and tok.type == tokenize.OP and tok.string == ":":
                    header_end = tok.end[0]
                continue
            if tok.type in (tokenize.NEWLINE, tokenize.NL, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT):
                continue
            if tok.type == tokenize.STRING:
                return start + tok.end[0] - 1
            return start + header_end - 1
    except (tokenize.TokenError, SyntaxError):
        return None
    return start + header_end - 1 if header_end is not None else None


def _brace_header_end(lines: list[str], cls: Symbol) -> Optional[int]:
    """The line that opens the body (``{``) of a class in a brace language:
    the first ``{`` from the ``class`` keyword on (decorators and comment lines
    before it are skipped). ``None`` when there is none within
    ``HEADER_SCAN_LINES`` lines. Dart, Java, Go and Rust have readers of their own."""
    if cls.path.endswith(_DART_SUFFIXES):
        return _dart_header_end(lines, cls)
    lang = _c_lang(cls.path)
    if lang is not None:
        return _c_header_end(lines, cls, lang)
    start = cls.start_line
    stop = min(cls.end_line, start + HEADER_SCAN_LINES - 1, len(lines))
    seen_keyword = False
    keyword_re = _CLASS_KEYWORD_RE
    for number in range(start, stop + 1):
        line = lines[number - 1]
        if not seen_keyword:
            if line.lstrip().startswith(_COMMENT_PREFIXES):
                continue
            keyword = keyword_re.search(line)
            if keyword is None:
                continue
            seen_keyword = True
            line = line[keyword.end():]
        if "{" in line:
            return number
    return None


def _dart_header_end(lines: list[str], cls: Symbol) -> Optional[int]:
    """The line that ends a Dart class-like header: the ``{`` that opens the body, or the ``;`` of a
    bodiless declaration (``sealed class A<T>(var T v);``). Braces inside ``(...)`` (named parameters,
    default values), strings and comments do not count, so primary constructors such as
    ``class const A({required this.x}) extends B {`` end on the real body brace."""
    start = cls.start_line
    stop = min(cls.end_line, start + HEADER_SCAN_LINES - 1, len(lines))
    text = "\n".join(lines[start - 1 : stop])
    n = len(text)
    i = 0
    line = start
    depth = 0
    seen_keyword = False
    while i < n:
        c = text[i]
        if c == "\n":
            line += 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            end = n if j < 0 else j + 2
            line += text.count("\n", i, end)
            i = end
            continue
        elif c in "'\"":
            raw = i > 0 and text[i - 1] == "r"
            quote = text[i:i + 3] if text.startswith(c * 3, i) else c
            i += len(quote)
            while i < n and not text.startswith(quote, i):
                if text[i] == "\\" and not raw:
                    i += 1
                if i < n and text[i] == "\n":
                    line += 1
                i += 1
            i += len(quote)
            continue
        elif c.isalpha() or c == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] in "_$"):
                j += 1
            if depth == 0 and text[i:j] in _DART_KEYWORDS:
                seen_keyword = True
            i = j
            continue
        elif c in "([":
            depth += 1
        elif c in ")]":
            depth = max(0, depth - 1)
        elif seen_keyword and depth == 0 and c in "{;":
            return line
        i += 1
    return None


# --------------------------------------------------------------------------- #
# Java, Go and Rust
# --------------------------------------------------------------------------- #

def _c_lang(path: str) -> Optional[str]:
    """``"java"``, ``"go"`` or ``"rust"`` for a file read by :func:`_c_tokens`, else None."""
    if path.endswith(_JAVA_SUFFIXES):
        return "java"
    if path.endswith(_GO_SUFFIXES):
        return "go"
    if path.endswith(_RUST_SUFFIXES):
        return "rust"
    return None


def _c_header_end(lines: list[str], cls: Symbol, lang: str) -> Optional[int]:
    """The line that opens the body (``{``) of a Java, Go or Rust type.

    Java and Rust: the first ``{`` after the type keyword (``class``,
    ``interface``, ``@interface``, ``enum``, ``record``; ``struct``, ``enum``,
    ``union``, ``trait``), so annotations and ``#[...]`` attributes before it are
    skipped; a bodiless Rust item (``struct Unit;``) ends at its ``;``. Go: the
    first ``{`` from the type's start line (a spec inside ``type ( ... )`` has no
    keyword of its own); a type without one (``type ID int``) ends where the
    symbol ends. Braces inside ``(...)``, ``[...]``, strings and comments do not
    count. ``None`` when there is none within ``HEADER_SCAN_LINES`` lines."""
    start = cls.start_line
    stop = min(cls.end_line, start + HEADER_SCAN_LINES - 1, len(lines))
    if start > stop:
        return None
    keywords = _JAVA_TYPE_KEYWORDS if lang == "java" else _RUST_TYPE_KEYWORDS if lang == "rust" else None
    seen_keyword = keywords is None
    depth = 0
    prev = ""
    for tok, line in _c_tokens("\n".join(lines[start - 1 : stop]), start, lang):
        if tok in ("(", "["):
            depth += 1
        elif tok in (")", "]"):
            depth = max(0, depth - 1)
        elif depth == 0:
            if not seen_keyword:
                # `@Foo` is an annotation, `@interface` an annotation type
                seen_keyword = tok in keywords and (prev != "@" or tok == "interface")
            elif tok == "{" or (tok == ";" and lang == "rust"):
                return line
        prev = tok
    if lang == "go" and stop == cls.end_line:
        return cls.end_line
    return None


def _rust_impl_owners(lines: list[str]) -> list[Optional[tuple[int, int]]]:
    """For each line number (index 0 is unused): ``(first header line, line of
    the opening brace)`` of the ``impl`` block whose body directly encloses the
    start of that line, else None.

    One pass over the file. An ``impl`` counts when it starts an item (after
    ``{``, ``}``, ``;``, outer attributes, ``unsafe`` or ``default``), so
    ``-> impl Trait`` and ``x: impl Fn()`` do not; its header ends at the first
    ``{`` outside ``(...)``, ``[...]`` and ``<...>``. Outer attributes directly
    above the ``impl`` belong to the header."""
    owners: list[Optional[tuple[int, int]]] = [None] * (len(lines) + 2)
    stack: list[Optional[tuple[int, int]]] = []  # one entry per open brace; impl bodies carry their header
    filled = 0                                   # owners[1..filled] are set
    item_start = True                            # the next token may begin an item
    attr_line: Optional[int] = None              # first outer-attribute line of the coming item
    modifier_line: Optional[int] = None          # line of a leading `unsafe` / `default`
    attr_depth = 0                               # bracket depth inside #[...] / #![...]
    attr_resume = True                           # item_start to restore after the attribute
    hash_state = 0                               # 1 after `#`, 2 after `#!`
    hash_line = 0
    impl_start: Optional[int] = None             # first header line, between `impl` and its body
    nest = angle = 0
    prev = ""
    for tok, line in _c_tokens("\n".join(lines), 1, "rust"):
        while filled < line:
            filled += 1
            owners[filled] = stack[-1] if stack else None
        if attr_depth:
            if tok == "[":
                attr_depth += 1
            elif tok == "]":
                attr_depth -= 1
                if not attr_depth:
                    item_start = attr_resume
            continue
        if tok == "#":
            hash_state, hash_line, attr_resume = 1, line, item_start
            continue
        if tok == "!" and hash_state == 1:
            hash_state = 2
            continue
        if tok == "[" and hash_state:
            if hash_state == 1 and attr_resume and impl_start is None and attr_line is None:
                attr_line = hash_line
            attr_depth, hash_state = 1, 0
            continue
        hash_state = 0
        if impl_start is not None:
            if tok in ("(", "["):
                nest += 1
            elif tok in (")", "]"):
                nest = max(0, nest - 1)
            elif tok == "<":
                angle += 1
            elif tok == ">" and prev not in ("-", "="):  # not `->` / `=>`
                angle = max(0, angle - 1)
            elif tok == "{":
                if nest or angle:
                    stack.append(None)  # a const-generic block, not the body
                else:
                    stack.append((impl_start, line))
                    impl_start = None
                    item_start = True
            elif tok == "}":
                if stack:
                    stack.pop()
            elif tok == ";" and not nest:
                impl_start = None  # not an impl block after all
                item_start = True
            prev = tok
            continue
        if tok in ("{", "}", ";"):
            if tok == "{":
                stack.append(None)
            elif tok == "}" and stack:
                stack.pop()
            item_start, attr_line, modifier_line = True, None, None
        elif item_start and tok == "impl":
            impl_start = attr_line or modifier_line or line
            nest = angle = 0
            attr_line = modifier_line = None
        elif item_start and tok in ("unsafe", "default"):
            modifier_line = modifier_line or line
        else:
            item_start, attr_line, modifier_line = False, None, None
        prev = tok
    while filled < len(lines):
        filled += 1
        owners[filled] = stack[-1] if stack else None
    return owners


def _c_tokens(text: str, line: int, lang: str) -> Iterator[tuple[str, int]]:
    """Tokens of Java, Go or Rust source with their line numbers, counted from
    ``line``: words (identifiers, keywords, numbers) and single punctuation
    characters. Comments are dropped (Rust block comments nest) and each string
    or character literal becomes one ``"`` token: Java text blocks, Go raw
    strings, Rust raw strings (``r#"..."#``). A Rust lifetime (``'a``) is a ``'``
    token followed by its name."""
    rust = lang == "rust"
    n = len(text)
    i = 0
    while i < n:
        m = _C_TOKEN_RE.match(text, i)
        assert m is not None  # the last alternative matches any character
        kind, tok, j = m.lastgroup, m.group(), m.end()
        if kind == "space":
            line += tok.count("\n")
        elif kind == "line_comment":
            pass
        elif kind == "block_comment":
            j = _block_comment_end(text, i, nested=rust)
            line += text.count("\n", i, j)
        elif kind == "word":
            if rust and tok in ("r", "br", "cr") and j < n and text[j] in '#"':
                k = j
                while k < n and text[k] == "#":
                    k += 1
                if k < n and text[k] == '"':
                    close = '"' + "#" * (k - j)
                    e = text.find(close, k + 1)
                    j = n if e < 0 else e + len(close)
                    yield '"', line
                    line += text.count("\n", i, j)
                    i = j
                    continue
            yield tok, line
        elif tok == '"' or (tok == "`" and lang == "go") or (tok == "'" and not (rust and _is_lifetime(text, i))):
            quote = '"""' if lang == "java" and text.startswith('"""', i) else tok
            j = _quoted_end(text, i, quote, escapes=tok != "`")
            yield '"', line
            line += text.count("\n", i, j)
        else:
            yield tok, line
        i = j


def _block_comment_end(text: str, i: int, nested: bool) -> int:
    """Index just past the block comment that starts at ``i``."""
    n = len(text)
    depth = 0
    j = i
    while j < n:
        if text.startswith("/*", j) and (nested or depth == 0):
            depth += 1
            j += 2
        elif text.startswith("*/", j):
            depth -= 1
            j += 2
            if depth == 0:
                return j
        else:
            j += 1
    return n


def _quoted_end(text: str, i: int, quote: str, escapes: bool) -> int:
    """Index just past the literal that opens with ``quote`` at ``i``."""
    n = len(text)
    j = i + len(quote)
    while j < n and not text.startswith(quote, j):
        if escapes and text[j] == "\\":
            j += 1
        j += 1
    return min(n, j + len(quote))


def _is_lifetime(text: str, i: int) -> bool:
    """True if the ``'`` at ``i`` starts a Rust lifetime or label (``'a``), not a
    character literal (``'a'``, ``'\\n'``)."""
    return not (text.startswith("\\", i + 1) or text.startswith("'", i + 2))


# --------------------------------------------------------------------------- #
# Symbol lookups
# --------------------------------------------------------------------------- #

class _SymbolTable:
    def __init__(self, index: RepoIndex) -> None:
        self._by_key: dict[tuple[str, int], Symbol] = {}
        self._by_qualname: dict[tuple[str, str], list[Symbol]] = {}
        for sym in index.symbols:
            self._by_key.setdefault((sym.id, sym.start_line), sym)
            self._by_qualname.setdefault((sym.path, sym.qualname), []).append(sym)

    def get(self, symbol_id: str, start_line: int) -> Optional[Symbol]:
        return self._by_key.get((symbol_id, start_line))

    def _containing(
        self, path: str, qualname: str, inner: Symbol, kinds: Optional[frozenset[str]] = None
    ) -> Optional[Symbol]:
        found = [
            s for s in self._by_qualname.get((path, qualname), [])
            if s.start_line <= inner.start_line and inner.end_line <= s.end_line
            and (kinds is None or s.kind in kinds)
        ]
        if not found:
            return None
        return min(found, key=lambda s: (s.end_line - s.start_line, s.start_line))

    def context_symbol(self, sym: Symbol) -> Symbol:
        """The symbol to show for ``sym``: its outermost non-nested enclosing
        function or method when ``sym`` is nested and that one is short."""
        if _LOCALS not in sym.qualname:
            return sym
        outer_name = sym.qualname.split(_LOCALS, 1)[0]
        outer = self._containing(sym.path, outer_name, sym)
        if outer is None or outer.end_line - outer.start_line + 1 > LONG_SYMBOL_LINES:
            return sym
        return outer

    def enclosing_class(self, method: Symbol) -> Optional[Symbol]:
        """The type whose header names ``method``'s owner: the class (Dart: or
        enum) of that qualname containing it; in Java any type, in Rust a trait
        (an ``impl`` block is not a symbol, see :func:`_rust_impl_owners`); in Go
        the receiver type declared anywhere in the same file."""
        if not method.parent:
            return None
        path = method.path
        if path.endswith(_GO_SUFFIXES):  # methods are declared outside their receiver type
            found = [s for s in self._by_qualname.get((path, method.parent), []) if s.kind in _GO_RECEIVER_KINDS]
            return min(found, key=lambda s: s.start_line) if found else None
        if path.endswith(_JAVA_SUFFIXES + _RUST_SUFFIXES):
            return self._containing(path, method.parent, method, _TYPE_OWNER_KINDS)
        found = self._containing(path, method.parent, method, _CLASS_KINDS)
        if found is None and path.endswith(_DART_SUFFIXES):  # Dart members of an enum
            found = self._containing(path, method.parent, method, _ENUM_KINDS)
        return found


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #

class _FileCache:
    """Reads checkout files once, as lists of lines without terminators."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._lines: dict[str, Optional[list[str]]] = {}

    def lines(self, path: str) -> Optional[list[str]]:
        if path not in self._lines:
            self._lines[path] = self._read(path)
        return self._lines[path]

    def _read(self, path: str) -> Optional[list[str]]:
        full = (self.root / path).resolve()
        if not full.is_relative_to(self.root):
            log.warning("ignoring %s: outside the checkout", path)
            return None
        try:
            raw = full.read_bytes()
        except OSError as exc:
            log.warning("cannot read %s: %s", full, exc)
            return None
        return split_lines(raw.decode("utf-8", errors="replace"))

    def text(self, path: str, start: int, end: int) -> str:
        lines = self.lines(path) or []
        return "\n".join(lines[start - 1 : end])

    def chars(self, path: str, start: int, end: int) -> int:
        return len(self.text(path, start, end))


def split_lines(text: str) -> list[str]:
    """Lines as the indexer counts them: split on ``\\n`` only, one trailing
    ``\\r`` removed per line, a leading byte order mark dropped, and no empty
    last line for a file that ends with a newline."""
    if text.startswith(_BOM):
        text = text[len(_BOM):]
    if not text:
        return []
    parts = text.split("\n")
    if text.endswith("\n"):
        parts.pop()
    return [p[:-1] if p.endswith("\r") else p for p in parts]


# --------------------------------------------------------------------------- #
# Merging and the character budget
# --------------------------------------------------------------------------- #

class _Builder:
    def __init__(self, files: _FileCache, max_chars: int) -> None:
        self.files = files
        self.max_chars = max_chars
        self.pieces: list[_Piece] = []
        self.used = 0

    def add_group(self, header: Optional[_Range], parts: list[_Range]) -> bool:
        """Add a symbol's ranges: the optional header goes in only together with
        the first part. Parts are added in order until one does not fit.
        Returns True if the header was added (or was already covered)."""
        if not parts:
            return False
        header_added = False
        if header is not None:
            saved = ([_Piece(p.path, p.start, p.end, list(p.labels), p.chars) for p in self.pieces], self.used)
            if self._add(header) and self._add(parts[0]):
                header_added = True
                parts = parts[1:]
            else:
                self.pieces, self.used = saved
        for part in parts:
            if not self._add(part):
                break
        return header_added

    def _add(self, rng: _Range) -> bool:
        """Add one range (merging overlaps). True if it is now covered."""
        lines = self.files.lines(rng.path)
        if not lines:
            return False
        start, end = max(1, rng.start), min(rng.end, len(lines))
        if start > end:
            return False
        overlapping = [
            i for i, p in enumerate(self.pieces)
            if p.path == rng.path and p.start <= end and start <= p.end
        ]
        if any(self.pieces[i].start <= start and end <= self.pieces[i].end for i in overlapping):
            return True
        if overlapping:
            olds = [self.pieces[i] for i in overlapping]
            new_start = min([start] + [p.start for p in olds])
            new_end = max([end] + [p.end for p in olds])
            size = self.files.chars(rng.path, new_start, new_end)
            old = sum(p.chars for p in olds)
            if self.used - old + size > self.max_chars:
                return False
            labels = [label for p in olds for label in p.labels] + [rng.label]
            merged = _Piece(rng.path, new_start, new_end, list(dict.fromkeys(labels)), size)
            drop = set(overlapping)
            kept = [p for i, p in enumerate(self.pieces) if i not in drop]
            first = overlapping[0] - sum(1 for i in overlapping if i < overlapping[0])
            kept.insert(first, merged)
            self.pieces = kept
            self.used += size - old
            return True

        size = self.files.chars(rng.path, start, end)
        if self.used + size > self.max_chars:
            if self.pieces:
                return False
            # The very first chunk alone is too large: keep its leading lines.
            while end > start and self.files.chars(rng.path, start, end) > self.max_chars:
                end -= 1
            size = self.files.chars(rng.path, start, end)
            if size > self.max_chars:
                return False
        self.pieces.append(_Piece(rng.path, start, end, [rng.label], size))
        self.used += size
        return True


def _join_labels(labels: list[str]) -> Optional[str]:
    labels = [label for label in labels if label]
    if not labels:
        return None
    if len(labels) > MAX_LABELS:
        return " + ".join(labels[:MAX_LABELS]) + f" + {len(labels) - MAX_LABELS} more"
    return " + ".join(labels)

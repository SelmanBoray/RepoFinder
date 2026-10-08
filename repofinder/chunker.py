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
  :data:`CLASS_HEADER_MAX_LINES` lines.
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
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
            cls = symbols.enclosing_class(target)
            if cls is not None and (cls.path, cls.start_line) not in headers_done:
                end = class_header_end(lines, cls)
                if end is not None and end - cls.start_line + 1 < CLASS_HEADER_MAX_LINES:
                    header = _Range(cls.path, cls.start_line, end, f"{cls.qualname} (class header)")
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
    windows = split_windows(lines, start, end)
    if sym.kind == "class":
        a, b = windows[0]
        label = f"{sym.qualname} (first {b - a + 1} of {n} lines)"
        return [_Range(sym.path, a, b, label)][:MAX_CLASS_WINDOWS]
    total = len(windows)
    return [
        _Range(sym.path, a, b, f"{sym.qualname} (part {i}/{total})")
        for i, (a, b) in enumerate(windows[:MAX_FUNCTION_WINDOWS], start=1)
    ]


def split_windows(lines: list[str], start: int, end: int) -> list[tuple[int, int]]:
    """Split ``start..end`` (1-based, inclusive) into consecutive windows of
    ``WINDOW_MIN_LINES``-``WINDOW_MAX_LINES`` lines; only the last may be shorter."""
    windows: list[tuple[int, int]] = []
    a = start
    while end - a + 1 > WINDOW_MAX_LINES:
        b = _best_cut(lines, a, end)
        windows.append((a, b))
        a = b + 1
    windows.append((a, end))
    return windows


def _best_cut(lines: list[str], a: int, end: int) -> int:
    """Last line of the window starting at ``a``: the next window should start at
    a statement with the shallowest indentation, not inside a decorator stack,
    as close to ``WINDOW_TARGET_LINES`` as possible."""
    lo = a + WINDOW_MIN_LINES - 1
    hi = min(a + WINDOW_MAX_LINES - 1, end - 1)
    target = a + WINDOW_TARGET_LINES - 1
    best: Optional[tuple[tuple[int, int, int, int], int]] = None
    for b in range(lo, hi + 1):
        last = lines[b - 1]
        nxt = lines[b] if b < len(lines) else ""
        indent = _indent(nxt) if nxt.strip() else 10_000  # never start a window on a blank line
        cost = (
            1 if last.lstrip().startswith("@") else 0,      # do not split a decorator stack
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
    (e.g. a docstring longer than that)."""
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

    def _containing(self, path: str, qualname: str, inner: Symbol, kind: Optional[str] = None) -> Optional[Symbol]:
        found = [
            s for s in self._by_qualname.get((path, qualname), [])
            if s.start_line <= inner.start_line and inner.end_line <= s.end_line
            and (kind is None or s.kind == kind)
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
        if not method.parent:
            return None
        return self._containing(method.path, method.parent, method, kind="class")


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

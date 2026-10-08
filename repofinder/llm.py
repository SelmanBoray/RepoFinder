"""LLM layer: per-file summaries, a repository overview, and cited answers.

:class:`LLM` builds the prompts (wording in :mod:`repofinder.prompts`) and
sends them through a :class:`~repofinder.backends.Backend`: the Anthropic API
(``AnthropicBackend``) or the local Claude Code CLI (``ClaudeCodeBackend``).
See :mod:`repofinder.backends` for how the backend is chosen and set up.

Design notes
------------
* Nothing is contacted when ``LLM()`` is constructed; the backend is detected
  (and credentials checked) on first use, and failures there raise
  :class:`LLMConfigError` / :class:`BackendUnavailable`.
* Thinking cannot be disabled on the default models and counts toward
  ``max_tokens``, so the token limits below leave room for it; the visible
  length of each output is controlled by the prompts, and the amount of
  thinking by ``effort``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

from . import prompts
from .backends import (  # noqa: F401  (re-exported: part of this module's API)
    AnthropicBackend,
    Backend,
    BackendError,
    BackendUnavailable,
    ClaudeCodeBackend,
    Completion,
    ENV_BACKEND,
    LLMConfigError,
    LLMError,
    detect_backend,
)
from .models import RepoIndex, Symbol

DEFAULT_FAST_MODEL = "claude-sonnet-5-5"
DEFAULT_SMART_MODEL = "claude-opus-5-5"

ENV_MODEL_FAST = "REPOFINDER_MODEL_FAST"
ENV_MODEL_SMART = "REPOFINDER_MODEL_SMART"

# Store meta key under which callers keep the repository overview.
REPO_SUMMARY_META_KEY = "summary.repo"

# Input budgets (characters).
FILE_SOURCE_CHAR_LIMIT = 12_000      # source text sent per file summary (head + tail)
FILE_SYMBOL_LIMIT = 150              # symbols listed per file summary
FILE_SYMBOL_CHAR_LIMIT = 8_000       # characters of symbol list per file summary
SUMMARIES_CHAR_BUDGET = 40_000       # file summaries sent to summarize_repo
MAP_CHAR_BUDGET = 40_000             # architecture map sent to summarize_repo

# Output limits (tokens; include the model's thinking) and effort levels.
FILE_SUMMARY_MAX_TOKENS = 2_000
REPO_SUMMARY_MAX_TOKENS = 8_000
ANSWER_MAX_TOKENS = 8_000
FILE_SUMMARY_EFFORT = "low"
SMART_EFFORT = "medium"

FAILED_SUMMARY_PREFIX = "(summary failed: "


# --------------------------------------------------------------------------- #
# Public data types
# --------------------------------------------------------------------------- #

@dataclass
class ContextChunk:
    path: str
    start_line: int
    end_line: int
    text: str                   # the source lines, exactly as in the file
    label: Optional[str] = None # e.g. symbol qualname or "file summary"


@dataclass
class Citation:
    path: str
    start_line: int
    end_line: int


@dataclass
class Answer:
    text: str                   # final answer in Markdown, citations inline as `path:start-end`
    citations: list[Citation]   # parsed from the model output, deduplicated, only ones inside the provided context
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int


@dataclass
class Usage:
    """Running totals over every model call made by one :class:`LLM`."""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #

class LLM:
    """RepoFinder's model operations on top of a pluggable backend.

    Backend precedence: ``backend`` > ``client`` (an Anthropic-style client,
    wrapped in :class:`AnthropicBackend`) > ``api_key`` (likewise) >
    :func:`detect_backend` with ``REPOFINDER_BACKEND``, on first use.
    """

    def __init__(
        self,
        fast_model: Optional[str] = None,
        smart_model: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[object] = None,
        max_retries: int = 3,
        backend: Optional[Backend] = None,
        *,
        sleep: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.fast_model = fast_model or os.environ.get(ENV_MODEL_FAST) or DEFAULT_FAST_MODEL
        self.smart_model = smart_model or os.environ.get(ENV_MODEL_SMART) or DEFAULT_SMART_MODEL
        self.max_retries = max(0, int(max_retries))
        self.usage = Usage()
        self._api_key = api_key
        self._sleep = sleep
        if backend is None and (client is not None or api_key is not None):
            backend = AnthropicBackend(api_key=api_key, client=client,
                                       max_retries=self.max_retries, sleep=sleep)
        self._backend: Optional[Backend] = backend

    @property
    def backend(self) -> Backend:
        """The backend in use; detected on first access (may raise
        :class:`BackendUnavailable` with setup instructions)."""
        if self._backend is None:
            self._backend = detect_backend(os.environ.get(ENV_BACKEND), api_key=self._api_key,
                                           max_retries=self.max_retries, sleep=self._sleep)
        return self._backend

    @property
    def backend_name(self) -> str:
        return self.backend.name

    # ----------------------------------------------------------------- API --

    def summarize_file(self, path: str, source: str, symbols: list[Symbol]) -> str:
        """2-4 sentence plain-text summary of one source file (fast model)."""
        user = prompts.file_summary_user(
            path=path,
            source=truncate_middle(source, FILE_SOURCE_CHAR_LIMIT),
            symbols=_format_symbols(symbols),
        )
        result = self._complete(model=self.fast_model, system=prompts.FILE_SUMMARY_SYSTEM,
                                user=user, max_tokens=FILE_SUMMARY_MAX_TOKENS,
                                effort=FILE_SUMMARY_EFFORT)
        return _plain_text(result.text)

    def summarize_files(
        self,
        items: list[tuple[str, str, list[Symbol]]],
        on_progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> dict[str, str]:
        """Summarize files one by one.

        A file whose call fails stores ``"(summary failed: <reason>)"`` and the
        loop continues. An :class:`LLMConfigError` (e.g. no backend available)
        is raised instead, since every remaining file would fail the same way.
        ``on_progress(done, total, path)`` runs after each file.
        """
        results: dict[str, str] = {}
        total = len(items)
        for done, (path, source, symbols) in enumerate(items, start=1):
            try:
                results[path] = self.summarize_file(path, source, symbols)
            except LLMConfigError:
                raise
            except LLMError as exc:
                results[path] = f"{FAILED_SUMMARY_PREFIX}{exc})"
            if on_progress is not None:
                on_progress(done, total, path)
        return results

    def summarize_repo(self, index: RepoIndex, summaries: dict[str, str], map_markdown: str) -> str:
        """1-3 paragraph architecture narrative for a newcomer (smart model)."""
        user = prompts.repo_summary_user(
            repo_name=index.name,
            n_files=len(index.files),
            n_symbols=len(index.symbols),
            map_markdown=_truncate_head(map_markdown or "", MAP_CHAR_BUDGET),
            summaries=_select_summaries(summaries, map_markdown or "", index, SUMMARIES_CHAR_BUDGET),
        )
        result = self._complete(model=self.smart_model, system=prompts.REPO_SUMMARY_SYSTEM,
                                user=user, max_tokens=REPO_SUMMARY_MAX_TOKENS, effort=SMART_EFFORT)
        return result.text.strip()

    def answer(
        self,
        question: str,
        chunks: list[ContextChunk],
        repo_name: str,
        repo_summary: Optional[str] = None,
    ) -> Answer:
        """Answer a "where is X implemented" question from ``chunks`` only."""
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        rendered = [
            prompts.render_chunk(c.path, c.start_line, c.end_line, c.text, c.label) for c in chunks
        ]
        result = self._complete(
            model=self.smart_model,
            system=prompts.answer_system(repo_name),
            user=prompts.answer_user(repo_summary, rendered, question),
            max_tokens=ANSWER_MAX_TOKENS,
            effort=SMART_EFFORT,
        )
        text = result.text.strip()
        if result.stop_reason == "max_tokens":
            text += prompts.TRUNCATED_ANSWER_NOTE
        return Answer(
            text=text,
            citations=parse_citations(result.text, chunks),
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cache_read_tokens=result.cache_read_tokens,
        )

    # ----------------------------------------------------------- internals --

    def _complete(self, *, model: str, system: str, user: str, max_tokens: int,
                  effort: Optional[str]) -> Completion:
        backend = self.backend
        try:
            result = backend.complete(model=model, system=system, user=user,
                                      max_tokens=max_tokens, effort=effort)
        except LLMError:
            raise
        except Exception as exc:  # a third-party backend that raises its own errors
            raise BackendError(f"{getattr(backend, 'name', 'backend')}: "
                               f"{type(exc).__name__}: {exc}") from exc
        self.usage.calls += 1
        self.usage.input_tokens += result.input_tokens
        self.usage.output_tokens += result.output_tokens
        self.usage.cache_read_tokens += result.cache_read_tokens
        if result.stop_reason == "refusal":
            raise LLMError(f"the model declined the request (refusal, category={result.stop_detail})")
        if not result.text.strip():
            raise LLMError(f"the model returned no text (stop_reason={result.stop_reason})")
        return result


# --------------------------------------------------------------------------- #
# Citations
# --------------------------------------------------------------------------- #

_RANGE = r":L?(\d+)(?:\s*[-–—]\s*L?(\d+))?"


def parse_citations(text: str, chunks: list[ContextChunk]) -> list[Citation]:
    """Extract ``path:start-end`` / ``path:line`` citations from ``text``.

    Only paths of the given chunks are recognised. A citation is kept when its
    range overlaps a chunk of that path; it is clamped to that chunk (the one
    with the largest overlap when several qualify). Results are deduplicated
    and keep first-appearance order.
    """
    by_path: dict[str, list[ContextChunk]] = {}
    for chunk in chunks:
        if chunk.start_line <= chunk.end_line:
            by_path.setdefault(chunk.path, []).append(chunk)
    if not by_path or not text:
        return []
    alternation = "|".join(re.escape(p) for p in sorted(by_path, key=len, reverse=True))
    pattern = re.compile(r"(?<![\w./\\-])(?:\./)?(" + alternation + r")" + _RANGE)

    seen: set[tuple[str, int, int]] = set()
    citations: list[Citation] = []
    for match in pattern.finditer(text):
        path = match.group(1)
        start = int(match.group(2))
        end = int(match.group(3)) if match.group(3) else start
        if start > end:
            start, end = end, start
        best: Optional[ContextChunk] = None
        best_overlap = 0
        for chunk in by_path[path]:
            overlap = min(end, chunk.end_line) - max(start, chunk.start_line) + 1
            if overlap > best_overlap:
                best, best_overlap = chunk, overlap
        if best is None:
            continue
        key = (path, max(start, best.start_line), min(end, best.end_line))
        if key not in seen:
            seen.add(key)
            citations.append(Citation(*key))
    return citations


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def truncate_middle(text: str, limit: int, head_fraction: float = 2 / 3) -> str:
    """Keep the head and tail of ``text`` (cut at line boundaries) within about
    ``limit`` characters, with a marker saying how much was left out."""
    if len(text) <= limit:
        return text
    head = text[: int(limit * head_fraction)]
    cut = head.rfind("\n")
    if cut > 0:
        head = head[: cut + 1]
    tail_len = max(0, limit - int(limit * head_fraction))
    tail = text[len(text) - tail_len:] if tail_len else ""
    cut = tail.find("\n")
    if cut >= 0:
        tail = tail[cut + 1:]
    omitted = text[len(head): len(text) - len(tail)]
    marker = prompts.OMITTED_MARKER.format(n_lines=omitted.count("\n"), n_chars=len(omitted))
    if not head.endswith("\n"):
        head += "\n"
    return f"{head}{marker}\n{tail}"


def _truncate_head(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = text[:limit]
    cut = head.rfind("\n")
    if cut > 0:
        head = head[: cut + 1]
    return head + prompts.MAP_TRUNCATED.format(n_chars=len(text) - len(head))


def _format_symbols(symbols: list[Symbol]) -> str:
    lines: list[str] = []
    used = 0
    listed = [s for s in sorted(symbols, key=lambda s: (s.start_line, s.end_line))
              if "<locals>" not in s.qualname]
    for i, sym in enumerate(listed):
        if i >= FILE_SYMBOL_LIMIT:
            break
        depth = len(sym.qualname.split(".")) - 1
        line = prompts.format_symbol(
            sym.signature or f"{sym.kind} {sym.qualname}",
            sym.start_line, sym.end_line, sym.docstring, depth,
        )
        if used + len(line) > FILE_SYMBOL_CHAR_LIMIT:
            break
        lines.append(line)
        used += len(line) + 1
    if len(lines) < len(listed):
        lines.append(prompts.MORE_SYMBOLS.format(n=len(listed) - len(lines)))
    return "\n".join(lines)


def _select_summaries(summaries: dict[str, str], map_markdown: str,
                      index: RepoIndex, budget: int) -> str:
    """Summary lines in map order (files the map mentions first), within budget."""
    usable = {p: s.strip() for p, s in summaries.items()
              if s and s.strip() and not s.startswith(FAILED_SUMMARY_PREFIX)}
    file_order = {f.path: i for i, f in enumerate(index.files)}

    def rank(path: str) -> tuple[int, int, int, str]:
        pos = _map_position(path, map_markdown)
        if pos is not None:
            return (0, pos, 0, path)
        return (1, 0, file_order.get(path, len(file_order)), path)

    lines: list[str] = []
    used = 0
    ordered = sorted(usable, key=rank)
    for path in ordered:
        line = prompts.SUMMARY_LINE.format(path=path, summary=" ".join(usable[path].split()))
        if used + len(line) + 1 > budget:
            break
        lines.append(line)
        used += len(line) + 1
    if len(lines) < len(ordered):
        lines.append(prompts.SUMMARIES_OMITTED.format(n=len(ordered) - len(lines)))
    return "\n".join(lines)


def _map_position(path: str, map_markdown: str) -> Optional[int]:
    """Earliest offset at which the map mentions ``path`` (as a path or a
    dotted module name), or None."""
    candidates = [path]
    stem = path[:-3] if path.endswith(".py") else None
    if stem:
        if stem.endswith("/__init__"):
            stem = stem[: -len("/__init__")]
        dotted = stem.replace("/", ".")
        candidates.append(dotted)
        if dotted.startswith("src."):
            candidates.append(dotted[4:])
    positions = []
    for cand in candidates:
        if not cand:
            continue
        m = re.search(r"(?<![\w.])" + re.escape(cand) + r"(?![\w])", map_markdown)
        if m:
            positions.append(m.start())
    return min(positions) if positions else None


def _plain_text(text: str) -> str:
    """Collapse a summary to one plain paragraph without Markdown headings."""
    kept = [ln for ln in text.strip().splitlines()
            if not ln.lstrip().startswith("#") and not ln.lstrip().startswith("```")]
    return " ".join(" ".join(kept).split())

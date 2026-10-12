"""End-to-end pipeline: open a repository, retrieve context, ask, map, summarize.

This module is the API that the CLI, the evaluation runner and later front
ends call. Its function signatures are a stable contract.

Data layout (``DATA_DIR`` is ``~/.repofinder`` unless ``REPOFINDER_HOME`` or an
explicit ``data_dir`` says otherwise)::

    DATA_DIR/repos/<name>/index.db             the SQLite store (index, summaries, meta)
    DATA_DIR/repos/<name>/checkout/<name>/     shallow clone, for GitHub sources only

A local directory is indexed in place and never copied or modified.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from . import prompts
from .chunker import MODULE_HEAD_LABEL, build_chunks, split_lines
from .graph import ModuleGraph, build_graph
from .ingest import _parse_github, build_index, fetch_repo, repo_name_from_source
from .llm import (
    FAILED_SUMMARY_PREFIX,
    LLM,
    REPO_SUMMARY_META_KEY,
    Answer,
    ContextChunk,
    LLMConfigError,
    LLMError,
)
from .mapper import render_map
from .models import RepoIndex
from .search import Hit, SearchIndex
from .store import META_INDEX_SOURCE, Store

if TYPE_CHECKING:  # the decision layer is imported lazily, only when one is used
    from .decisions import Decisions

log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path.home() / ".repofinder"
ENV_HOME = "REPOFINDER_HOME"

DB_FILENAME = "index.db"
CHECKOUT_DIRNAME = "checkout"
META_BUILT_AT = "index.built_at"

DEFAULT_K = 8
DEFAULT_MAX_CHARS = 36_000
DEFAULT_CANDIDATES = 16        # search hits shown to the decision layer (at most decisions.MAX_CARDS)
CARD_LINES = 40                # source lines per candidate card
MAX_CONSECUTIVE_FAILURES = 3

# Decision backends that run locally and never reorder search hits.
LOCAL_DECISIONS = frozenset({"rules", "off"})

ProgressCallback = Callable[[int, int, str], None]


def resolve_data_dir(data_dir: Optional[Path] = None) -> Path:
    """``data_dir`` if given, else ``$REPOFINDER_HOME``, else :data:`DEFAULT_DATA_DIR`."""
    if data_dir is not None:
        return Path(data_dir).expanduser()
    env = os.environ.get(ENV_HOME, "").strip()
    return Path(env).expanduser() if env else DEFAULT_DATA_DIR


def canonical_source(source: str) -> str:
    """A stable spelling of ``source``: ``https://github.com/<owner>/<name>``
    (lowercase) for GitHub, the resolved absolute path for a local directory."""
    gh = _parse_github(source)
    if gh is not None:
        owner, name, _url = gh
        return f"https://github.com/{owner.lower()}/{name.lower()}"
    path = Path(source).expanduser()
    return str(path.resolve()) if path.exists() else source.strip()


def is_usable_summary(summary: Optional[str]) -> bool:
    return bool(summary and summary.strip() and not summary.startswith(FAILED_SUMMARY_PREFIX))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Workspace
# --------------------------------------------------------------------------- #

@dataclass
class Workspace:
    """One opened repository: its checkout, store and index. Close it (or use
    it as a context manager) to release the SQLite connection."""

    name: str
    source: str           # canonical source (see canonical_source)
    root: Path            # checkout
    data_dir: Path        # <DATA_DIR>/repos/<name>/
    store: Store
    index: RepoIndex
    built: bool = False   # True if this open (re)built the index
    _search: Optional[SearchIndex] = field(default=None, init=False, repr=False, compare=False)
    _graph: Optional[ModuleGraph] = field(default=None, init=False, repr=False, compare=False)

    def summaries(self) -> dict[str, str]:
        """Usable per-file summaries (failed attempts are left out)."""
        return {p: s for p, s in self.store.all_summaries().items() if is_usable_summary(s)}

    def failed_summaries(self) -> dict[str, str]:
        """Stored ``(summary failed: ...)`` markers, by path."""
        return {p: s for p, s in self.store.all_summaries().items() if not is_usable_summary(s)}

    def repo_summary(self) -> Optional[str]:
        text = self.store.get_meta(REPO_SUMMARY_META_KEY)
        return text if text and text.strip() else None

    def search_index(self) -> SearchIndex:
        """Search index over the symbols, files and file summaries (cached)."""
        if self._search is None:
            self._search = SearchIndex(self.index, self.summaries())
        return self._search

    def graph(self) -> ModuleGraph:
        if self._graph is None:
            self._graph = build_graph(self.index)
        return self._graph

    def invalidate_caches(self) -> None:
        """Forget the cached search index and graph (e.g. after new summaries)."""
        self._search = None
        self._graph = None

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_workspace(source: str, data_dir: Optional[Path] = None, refresh: bool = False) -> Workspace:
    """Open ``source`` (GitHub URL or local directory), indexing it if needed.

    The stored index is reused unless ``refresh`` is set or it was built from a
    different source with the same name (then it is replaced). Raises
    :class:`~repofinder.ingest.IngestError` when the source cannot be fetched.
    """
    home = resolve_data_dir(data_dir)
    name = repo_name_from_source(source)
    canonical = canonical_source(source)
    ws_dir = home / "repos" / name
    root = fetch_repo(source, ws_dir / CHECKOUT_DIRNAME)

    store = Store(ws_dir / DB_FILENAME)
    try:
        built = False
        stored_source = store.get_meta(META_INDEX_SOURCE) if store.has_index() else None
        if stored_source is not None and stored_source != canonical:
            log.warning("replacing the index of %s stored in %s with %s", stored_source, ws_dir, canonical)
        if refresh or stored_source != canonical:
            index = build_index(root, name, canonical)
            store.save_index(index)
            store.set_meta(META_BUILT_AT, _now())
            built = True
        else:
            index = store.load_index()
            index.root = str(root)  # the checkout may have moved with the data dir
    except BaseException:
        store.close()
        raise
    return Workspace(name=name, source=canonical, root=root, data_dir=ws_dir,
                     store=store, index=index, built=built)


# --------------------------------------------------------------------------- #
# Retrieval and answers
# --------------------------------------------------------------------------- #

def retrieve(ws: Workspace, question: str, k: int = DEFAULT_K,
             max_chars: int = DEFAULT_MAX_CHARS) -> tuple[list[ContextChunk], list[Hit]]:
    """Top-``k`` search hits for ``question`` and the context chunks built from
    them (exact source lines, at most ``max_chars`` characters). Deterministic."""
    hits = ws.search_index().search(question, k=k)
    chunks = build_chunks(ws.root, ws.index, hits, max_chars)
    return chunks, hits


@dataclass
class Retrieval:
    """What :func:`rerank_retrieve` found."""

    chunks: list[ContextChunk]
    hits: list[Hit]                 # all searched hits, in the final order (reranked candidates first,
                                    # then the rest in search order)
    reranked: bool = False          # the decision layer's order was applied (its pick may be the first hit)
    off_target: bool = False        # it selected NONE: the search order was kept
    decision_calls: int = 0
    decision_cost_usd: float = 0.0
    decision_error: Optional[str] = None   # "ExcType: message" when the call failed and search order was kept
    probabilities: dict[str, float] = field(default_factory=dict)  # "path:start-end" -> p, shown candidates


def rerank_retrieve(ws: Workspace, question: str, k: int = DEFAULT_K, max_chars: int = DEFAULT_MAX_CHARS,
                    decisions: Optional["Decisions"] = None,
                    candidates: int = DEFAULT_CANDIDATES) -> Retrieval:
    """:func:`retrieve`, with the search hits reordered by a decision layer.

    Without ``decisions``, or with a local one (``rules``/``off``), this is
    exactly :func:`retrieve` (top-``k`` hits, no decision call). Otherwise
    ``max(k, candidates)`` hits are searched, the first ``candidates`` (at most
    ``decisions.MAX_CARDS``) are shown to the layer as cards (see
    :func:`candidate_cards`) in one ``rerank_cards`` call, and the chunks are
    built from the first ``k`` hits of the resulting order. When the layer
    selects none of them, or its call fails (``decision_error``), the search
    order is kept. :class:`~repofinder.decisions.DecisionsConfigError`
    propagates. Nothing is searched a second time.
    """
    if decisions is None or getattr(decisions, "name", "") in LOCAL_DECISIONS:
        chunks, hits = retrieve(ws, question, k=k, max_chars=max_chars)
        return Retrieval(chunks=chunks, hits=hits)

    from . import decisions as _decisions

    candidates = max(1, int(candidates))
    hits = ws.search_index().search(question, k=max(k, candidates))
    shown = hits[: min(candidates, _decisions.MAX_CARDS)]
    found = Retrieval(chunks=[], hits=list(hits))
    if shown:
        try:
            result = _decisions.rerank_cards(decisions, question, candidate_cards(ws, shown), allow_none=True)
        except _decisions.DecisionsConfigError:
            raise
        except _decisions.DecisionsError as exc:
            log.warning("rerank failed, keeping the search order: %s", exc)
            found.decision_error = f"{type(exc).__name__}: {exc}"
        else:
            found.decision_calls = 1
            found.decision_cost_usd = float(result.cost_usd or 0.0)
            found.probabilities = _hit_probabilities(shown, result.probabilities, _decisions.CARD_LABELS)
            if result.none_selected:
                found.off_target = True
            else:
                found.hits = _apply_order(hits, result.order, len(shown))
                found.reranked = True
    found.chunks = build_chunks(ws.root, ws.index, found.hits[:k], max_chars)
    return found


def candidate_cards(ws: Workspace, hits: list[Hit]) -> list[str]:
    """One card per hit for the decision layer (:func:`prompts.render_card`):
    ``path:start-end | label`` and the first :data:`CARD_LINES` of the hit's own
    source lines. The label is ``<kind> <qualname>`` for a symbol (the qualname
    alone when the index does not know it) and ``module head`` for a file hit,
    as the chunker labels it. A file that cannot be read gives an empty body."""
    wanted = {h.symbol_id for h in hits if h.kind == "symbol" and h.symbol_id}
    symbols: dict[Any, Any] = {}
    for sym in ws.index.symbols:
        if sym.id in wanted:
            symbols.setdefault((sym.id, sym.start_line), sym)
            symbols.setdefault(sym.id, sym)
    sources: dict[str, Optional[list[str]]] = {}
    cards: list[str] = []
    for hit in hits:
        if hit.path not in sources:
            sources[hit.path] = _source_lines(ws.root, hit.path)
        lines = sources[hit.path]
        text = "\n".join(lines[max(hit.start_line, 1) - 1: hit.end_line]) if lines else ""
        if hit.kind != "symbol" or not hit.symbol_id:
            label = MODULE_HEAD_LABEL
        else:
            sym = symbols.get((hit.symbol_id, hit.start_line)) or symbols.get(hit.symbol_id)
            label = f"{sym.kind} {sym.qualname}" if sym is not None else hit.symbol_id.split("::", 1)[-1]
        cards.append(prompts.render_card(hit.path, hit.start_line, hit.end_line, text, label,
                                         max_lines=CARD_LINES))
    return cards


def _source_lines(root: Path, path: str) -> Optional[list[str]]:
    """The lines of ``root / path`` (as the indexer counts them), or None when
    it cannot be read or lies outside the checkout."""
    base = Path(root).resolve()
    full = (base / path).resolve()
    if not full.is_relative_to(base):
        return None
    try:
        return split_lines(full.read_bytes().decode("utf-8", errors="replace"))
    except OSError:
        return None


def _hit_probabilities(hits: list[Hit], probabilities: dict[str, float],
                       labels: tuple[str, ...]) -> dict[str, float]:
    """The layer's per-label probabilities keyed by the shown hits'
    ``path:start-end`` (the larger one when two hits share a range)."""
    out: dict[str, float] = {}
    for label, hit in zip(labels, hits):
        if label in probabilities:
            key = f"{hit.path}:{hit.start_line}-{hit.end_line}"
            out[key] = max(float(probabilities[label]), out.get(key, 0.0))
    return out


def _apply_order(items: list[Any], order: list[int], n_shown: int) -> list[Any]:
    """``items`` with the first ``n_shown`` reordered by ``order`` (0-based
    indices; invalid and repeated ones are ignored). Shown items missing from
    ``order`` follow in their old order, then the items that were not shown.
    (A copy of ``chat.apply_order``.)"""
    seen: set[int] = set()
    out: list[Any] = []
    for i in order or []:
        if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < min(n_shown, len(items)) \
                and i not in seen:
            seen.add(i)
            out.append(items[i])
    out.extend(item for i, item in enumerate(items) if i not in seen)
    return out


@dataclass
class AskResult:
    answer: Answer
    hits: list[Hit]
    chunks: list[ContextChunk]
    elapsed_s: float      # retrieval (with the decision call) plus the model call
    reranked: bool = False
    off_target: bool = False
    decision_calls: int = 0
    decision_cost_usd: float = 0.0
    decision_error: Optional[str] = None
    probabilities: dict[str, float] = field(default_factory=dict)


def ask(ws: Workspace, question: str, llm: Optional[LLM] = None, k: int = DEFAULT_K,
        decisions: Optional["Decisions"] = None, candidates: int = DEFAULT_CANDIDATES) -> AskResult:
    """Answer ``question`` about the repository with ``path:start-end`` citations.

    With a (non-local) ``decisions`` layer the context comes from
    :func:`rerank_retrieve`; without one, from :func:`retrieve` as before."""
    if not question or not question.strip():
        raise ValueError("question must not be empty")
    started = time.perf_counter()
    found = rerank_retrieve(ws, question, k=k, decisions=decisions, candidates=candidates)
    llm = llm if llm is not None else LLM()
    answer = llm.answer(question, found.chunks, ws.name, repo_summary=ws.repo_summary())
    return AskResult(answer=answer, hits=found.hits, chunks=found.chunks,
                     elapsed_s=time.perf_counter() - started, reranked=found.reranked,
                     off_target=found.off_target, decision_calls=found.decision_calls,
                     decision_cost_usd=found.decision_cost_usd, decision_error=found.decision_error,
                     probabilities=dict(found.probabilities))


def architecture_map(ws: Workspace, max_modules: int = 40) -> str:
    """The Markdown architecture map, with summaries and the overview if stored."""
    return render_map(ws.index, ws.graph(), summaries=ws.summaries(),
                      repo_summary=ws.repo_summary(), max_modules=max_modules)


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #

@dataclass
class SummarizeResult:
    done: int                       # files summarized in this run
    failed: int                     # files whose summary failed in this run
    skipped: int                    # files left alone: already summarized, or empty
    repo_summary_written: bool
    remaining: int = 0              # non-empty files still without a usable summary
    stopped_early: bool = False     # True after MAX_CONSECUTIVE_FAILURES failures in a row
    error: Optional[str] = None     # last failure message, if any


def summary_order(ws: Workspace) -> list[str]:
    """File paths in summarizing order: non-test modules by fan-in (most
    imported first; ties by symbol count, then path), then test files."""
    nodes = ws.graph().nodes.values()
    key = lambda n: (-n.fan_in, -n.n_symbols, n.path)  # noqa: E731
    core = sorted((n for n in nodes if not n.is_test), key=key)
    tests = sorted((n for n in nodes if n.is_test), key=key)
    ordered = [n.path for n in core + tests]
    seen = set(ordered)
    ordered += [f.path for f in ws.index.files if f.path not in seen]
    return ordered


def summarize(
    ws: Workspace,
    llm: LLM,
    limit: Optional[int] = None,
    only_missing: bool = True,
    batch_size: int = 10,
    on_progress: Optional[ProgressCallback] = None,
) -> SummarizeResult:
    """Write per-file summaries (fast model), then the repository overview
    (smart model) into meta ``summary.repo``.

    * ``only_missing``: skip files that already have a usable summary; files
      whose earlier attempt failed are retried. Empty files are always skipped.
    * ``limit``: at most this many files are sent to the model in this run.
    * Results are saved every ``batch_size`` files (and on any exit), so an
      interrupted run keeps its work. A failed attempt never overwrites a
      usable summary.
    * After :data:`MAX_CONSECUTIVE_FAILURES` failures in a row the run stops
      (usually a quota or rate limit). :class:`LLMConfigError` is raised.
    * The overview is (re)written when this run had no failures, did not stop
      early, and either summarized something or no overview exists yet.
    * ``on_progress(done, total, path)`` runs after each file.
    """
    batch_size = max(1, int(batch_size))
    files = {f.path: f for f in ws.index.files}
    existing = ws.store.all_summaries()
    symbols_by_path: dict[str, list] = {}
    for sym in ws.index.symbols:
        symbols_by_path.setdefault(sym.path, []).append(sym)

    todo: list[str] = []
    skipped = 0
    for path in summary_order(ws):
        record = files.get(path)
        if record is None:
            continue
        if record.size == 0 or (only_missing and is_usable_summary(existing.get(path))):
            skipped += 1
            continue
        todo.append(path)
    if limit is not None:
        todo = todo[: max(0, int(limit))]

    total = len(todo)
    done = failed = consecutive = 0
    stopped_early = False
    error: Optional[str] = None
    pending: dict[str, str] = {}

    def flush() -> None:
        for path, text in pending.items():
            if not is_usable_summary(text) and is_usable_summary(existing.get(path)):
                continue  # keep the earlier good summary
            ws.store.save_summary(path, text)
        pending.clear()

    try:
        for position, path in enumerate(todo, start=1):
            try:
                source = _read_source(ws.root, path)
                pending[path] = llm.summarize_file(path, source, symbols_by_path.get(path, []))
                done += 1
                consecutive = 0
            except LLMConfigError:
                raise
            except (LLMError, OSError) as exc:
                failed += 1
                consecutive += 1
                error = str(exc)
                pending[path] = f"{FAILED_SUMMARY_PREFIX}{exc})"
            if on_progress is not None:
                on_progress(position, total, path)
            if len(pending) >= batch_size:
                flush()
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                stopped_early = True
                break
    finally:
        flush()
        ws.invalidate_caches()

    remaining = sum(
        1 for f in ws.index.files if f.size > 0 and not is_usable_summary(ws.store.get_summary(f.path))
    )

    written = False
    summaries = ws.summaries()
    if failed == 0 and not stopped_early and summaries and (done > 0 or ws.repo_summary() is None):
        map_markdown = render_map(ws.index, ws.graph(), summaries=summaries, repo_summary=None)
        try:
            overview = llm.summarize_repo(ws.index, summaries, map_markdown)
        except LLMConfigError:
            raise
        except LLMError as exc:
            error = f"repository overview failed: {exc}"
        else:
            ws.store.set_meta(REPO_SUMMARY_META_KEY, overview)
            written = True

    return SummarizeResult(done=done, failed=failed, skipped=skipped, repo_summary_written=written,
                           remaining=remaining, stopped_early=stopped_early, error=error)


def _read_source(root: Path, path: str) -> str:
    raw = (Path(root) / path).read_bytes()
    return "\n".join(split_lines(raw.decode("utf-8", errors="replace"))) + "\n"


# --------------------------------------------------------------------------- #
# Status helpers
# --------------------------------------------------------------------------- #

def stale_files(ws: Workspace) -> list[str]:
    """Indexed files whose size on disk changed or that disappeared since
    indexing (a cheap staleness check; edits that keep the size are missed)."""
    changed: list[str] = []
    for record in ws.index.files:
        try:
            size = (ws.root / record.path).stat().st_size
        except OSError:
            changed.append(record.path)
            continue
        if size != record.size:
            changed.append(record.path)
    return changed

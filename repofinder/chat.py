"""Conversation mode: "the decision layer decides, the LLM talks".

A :class:`ChatSession` answers one message at a time about an opened
:class:`~repofinder.pipeline.Workspace`. Every turn the decision layer
(:mod:`repofinder.decisions`: Jev, or the local rules) makes the choices and the
LLM (:meth:`repofinder.llm.LLM.chat`) writes the reply:

1. ``route_message`` sorts the message into ``locate`` / ``explain`` /
   ``followup`` / ``chat`` / ``outside``.
2. ``locate`` and ``explain``: search (``retrieve``), show the excerpts to
   ``rerank_cards`` as candidate cards and reorder them by its choice. When it
   selects none, search once more with ``k * 3`` hits (at most 16 cards); when
   it still selects none, the excerpts are given anyway with a note that the
   search may have missed the right place.
3. ``followup``: when the previous excerpts exist and ``context_suffices`` says
   they answer the follow-up (probability >= 0.5), they are reused without a
   search; otherwise the previous question and the message are searched
   together as in step 2.
4. ``chat`` and ``outside``: no search and no excerpts; the LLM answers with
   :data:`LIGHT_EFFORT`.

The conversation so far (the last ``history_turns`` messages, each shortened)
goes to both the router and the LLM. A failing decision call (other than a
configuration error) does not end the turn: the step falls back to the
search-only behaviour and the failure is listed in
:attr:`TurnResult.decision_errors`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from . import decisions as _decisions
from . import prompts
from .llm import LLM, Answer, Citation, ContextChunk
from .pipeline import DEFAULT_K, DEFAULT_MAX_CHARS, Workspace, retrieve
from .search import Hit

log = logging.getLogger(__name__)

DEFAULT_HISTORY_TURNS = 6
HISTORY_TEXT_CHARS = 600        # each earlier message is shortened to about this length
MAX_CARDS = 16                  # candidate cards shown to the decision layer
CARD_LINES = 40                 # source lines per card
WIDEN_FACTOR = 3                # the second search asks for k * WIDEN_FACTOR hits
SUFFICES_THRESHOLD = 0.5        # context_suffices probability needed to reuse excerpts
MAX_QUERY_CHARS = 1_000         # a follow-up search query keeps at most this many characters
TRUNCATION_MARK = " […]"
LIGHT_EFFORT = "low"            # LLM effort for turns without excerpts (chat, outside)

# Decision backends that run locally; their calls are not counted as model calls.
LOCAL_DECISIONS = frozenset({"rules", "off"})


@dataclass
class Turn:
    """One message of the conversation."""
    role: str                                   # "user" | "assistant"
    text: str
    route: Optional[str] = None
    citations: list[Citation] = field(default_factory=list)


@dataclass
class TurnResult:
    route: str                  # the decision layer's route for the message
    answer: Answer
    hits: list[Hit]             # search hits of this turn, search order ([] when no search ran)
    chunks: list[ContextChunk]  # excerpts the LLM saw, in the order shown
    reused_context: bool        # follow-up answered from the previous excerpts
    widened: bool               # a second, wider search was needed
    elapsed_s: float
    decision_cost_usd: float
    decision_calls: int         # model calls made by the decision layer (0 for a local backend)
    route_probabilities: dict[str, float] = field(default_factory=dict)
    off_target: bool = False    # no excerpt was selected, even after widening
    context_probability: Optional[float] = None   # context_suffices answer, for a follow-up
    decision_errors: list[str] = field(default_factory=list)  # decision steps that fell back


@dataclass
class _Ledger:
    """Decision cost, model calls and fallbacks of one turn."""
    counts_calls: bool
    cost_usd: float = 0.0
    calls: int = 0
    errors: list[str] = field(default_factory=list)

    def add(self, cost_usd: Any) -> None:
        self.cost_usd += float(cost_usd or 0.0)
        if self.counts_calls:
            self.calls += 1

    def failed(self, step: str, exc: BaseException) -> None:
        log.warning("decision step %s failed, falling back: %s", step, exc)
        self.errors.append(f"{step}: {exc}")


class ChatSession:
    """A conversation about one repository. Not thread-safe."""

    def __init__(
        self,
        ws: Workspace,
        llm: LLM,
        decisions: "_decisions.Decisions",
        k: int = DEFAULT_K,
        max_chars: int = DEFAULT_MAX_CHARS,
        history_turns: int = DEFAULT_HISTORY_TURNS,
    ) -> None:
        self.ws = ws
        self.llm = llm
        self.decisions = decisions
        self.k = max(1, int(k))
        self.max_chars = int(max_chars)
        self.history_turns = max(0, int(history_turns))
        self.history: list[Turn] = []
        self.last_chunks: list[ContextChunk] = []     # excerpts of the last search
        self.last_question: Optional[str] = None      # the query that found them
        self.last_result: Optional[TurnResult] = None
        self._repo_summary = ws.repo_summary()

    @property
    def decisions_name(self) -> str:
        return str(getattr(self.decisions, "name", "unknown"))

    # ----------------------------------------------------------------- API --

    def turn(self, message: str) -> TurnResult:
        """Answer ``message``; see the module docstring for the flow.

        Raises :class:`ValueError` for an empty message; LLM errors and
        decision configuration errors propagate and leave the session as it was.
        """
        if not message or not message.strip():
            raise ValueError("message must not be empty")
        message = message.strip()
        started = time.perf_counter()
        ledger = _Ledger(counts_calls=self.decisions_name not in LOCAL_DECISIONS)
        history = self.history_window()
        route, route_probs = self._route(message, history, ledger)

        hits: list[Hit] = []
        chunks: list[ContextChunk] = []
        llm_chunks: Optional[list[ContextChunk]] = None   # None: no search ran
        query: Optional[str] = None                       # set when a search ran
        reused = widened = off_target = False
        context_p: Optional[float] = None
        notes: list[str] = []

        light = route in (_decisions.ROUTE_CHAT, _decisions.ROUTE_OUTSIDE)
        if light:
            notes.append(route)
        elif route == _decisions.ROUTE_FOLLOWUP and self.last_chunks:
            context_p = self._context_probability(message, ledger)
            if context_p is not None and context_p >= SUFFICES_THRESHOLD:
                reused = True
                chunks = list(self.last_chunks)
                llm_chunks = chunks
                notes.append("followup")
            else:
                query = join_query(self.last_question, message)
                chunks, hits, widened, off_target = self._search(query, ledger)
                llm_chunks = chunks
                notes.append("followup_search")
        else:  # locate, explain, or a follow-up with nothing to follow
            query = message
            chunks, hits, widened, off_target = self._search(query, ledger)
            llm_chunks = chunks
            notes.append(route if route == _decisions.ROUTE_EXPLAIN else _decisions.ROUTE_LOCATE)
        if off_target:
            notes.append("widened")

        answer = self.llm.chat(
            message,
            llm_chunks,
            self.ws.name,
            prompts.render_history(history),
            "\n\n".join(prompts.CHAT_MODE_NOTES[n] for n in notes),
            repo_summary=self._repo_summary,
            effort=LIGHT_EFFORT if light else None,
        )

        if query is not None:
            self.last_chunks = list(chunks)
            self.last_question = query
        self.history.append(Turn("user", message, route))
        self.history.append(Turn("assistant", answer.text, route, list(answer.citations)))
        result = TurnResult(
            route=route,
            answer=answer,
            hits=hits,
            chunks=chunks,
            reused_context=reused,
            widened=widened,
            elapsed_s=time.perf_counter() - started,
            decision_cost_usd=ledger.cost_usd,
            decision_calls=ledger.calls,
            route_probabilities=route_probs,
            off_target=off_target,
            context_probability=context_p,
            decision_errors=list(ledger.errors),
        )
        self.last_result = result
        return result

    def history_window(self) -> list[tuple[str, str]]:
        """The last ``history_turns`` messages as ``(role, shortened text)``."""
        if self.history_turns <= 0:
            return []
        return [(t.role, clip_text(t.text, HISTORY_TEXT_CHARS)) for t in self.history[-self.history_turns:]]

    # ------------------------------------------------------- decision steps --

    def _route(self, message: str, history: list[tuple[str, str]],
               ledger: _Ledger) -> tuple[str, dict[str, float]]:
        try:
            result = _decisions.route_message(self.decisions, message, history)
        except _decisions.DecisionsConfigError:
            raise
        except _decisions.DecisionsError as exc:
            ledger.failed("route", exc)
            return _decisions.ROUTE_LOCATE, {}
        ledger.add(result.cost_usd)
        route = result.route if result.route in _decisions.ROUTES else _decisions.ROUTE_LOCATE
        return route, dict(result.probabilities or {})

    def _rerank(self, query: str, chunks: list[ContextChunk],
                ledger: _Ledger) -> tuple[list[ContextChunk], bool]:
        """``chunks`` in the decision layer's order and whether it selected none.
        Chunks beyond :data:`MAX_CARDS` keep their order after the shown ones."""
        if not chunks:
            return [], False
        shown = chunks[:MAX_CARDS]
        cards = [render_card(c) for c in shown]
        try:
            result = _decisions.rerank_cards(self.decisions, query, cards, allow_none=True)
        except _decisions.DecisionsConfigError:
            raise
        except _decisions.DecisionsError as exc:
            ledger.failed("rerank", exc)
            return list(chunks), False
        ledger.add(result.cost_usd)
        if result.none_selected:
            return list(chunks), True
        return apply_order(chunks, result.order, len(shown)), False

    def _context_probability(self, message: str, ledger: _Ledger) -> Optional[float]:
        cards = [render_card(c) for c in self.last_chunks[:MAX_CARDS]]
        try:
            result = _decisions.context_suffices(self.decisions, message, self.last_question or "", cards)
        except _decisions.DecisionsConfigError:
            raise
        except _decisions.DecisionsError as exc:
            ledger.failed("context", exc)
            return None
        ledger.add(result.cost_usd)
        return float(result.probability)

    def _search(self, query: str,
                ledger: _Ledger) -> tuple[list[ContextChunk], list[Hit], bool, bool]:
        """Search, rerank, and widen once if nothing was selected.
        Returns ``(chunks, hits, widened, off_target)``."""
        chunks, hits = retrieve(self.ws, query, k=self.k, max_chars=self.max_chars)
        ordered, none_selected = self._rerank(query, chunks, ledger)
        if not none_selected:
            return ordered, hits, False, False
        wide_chunks, wide_hits = retrieve(self.ws, query, k=self.k * WIDEN_FACTOR, max_chars=self.max_chars)
        if len(wide_hits) <= len(hits):  # nothing new to show
            return ordered, hits, True, True
        ordered, none_selected = self._rerank(query, wide_chunks, ledger)
        return ordered, wide_hits, True, none_selected


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def render_card(chunk: ContextChunk) -> str:
    return prompts.render_card(chunk.path, chunk.start_line, chunk.end_line, chunk.text,
                               chunk.label, max_lines=CARD_LINES)


def apply_order(items: list[Any], order: list[int], n_shown: int) -> list[Any]:
    """``items`` with the first ``n_shown`` reordered by ``order`` (0-based
    indices; invalid and repeated ones are ignored). Shown items missing from
    ``order`` follow in their old order, then the items that were not shown."""
    seen: set[int] = set()
    out: list[Any] = []
    for i in order or []:
        if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < min(n_shown, len(items)) \
                and i not in seen:
            seen.add(i)
            out.append(items[i])
    out.extend(item for i, item in enumerate(items) if i not in seen)
    return out


def clip_text(text: str, limit: int) -> str:
    """``text`` shortened to about ``limit`` characters, cut at a word boundary
    when one is near, with a marker."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = max(cut.rfind(" "), cut.rfind("\n"))
    if space > limit * 0.6:
        cut = cut[:space]
    return cut.rstrip() + TRUNCATION_MARK


def join_query(previous: Optional[str], message: str) -> str:
    """The search query for a follow-up: the previous question, then the
    message; long chains keep their last :data:`MAX_QUERY_CHARS` characters."""
    if not previous:
        return message
    room = MAX_QUERY_CHARS - len(message) - 1
    if room <= 0:
        return message
    if len(previous) > room:
        previous = previous[-room:]
        newline = previous.find("\n")
        if newline >= 0:  # drop the partial first line
            previous = previous[newline + 1:]
    return f"{previous}\n{message}" if previous.strip() else message

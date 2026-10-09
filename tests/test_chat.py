"""Tests for repofinder.chat (the conversation layer) and its prompt helpers.

The LLM runs on a fake backend and the decision layer is a scripted stub that
follows the ``repofinder.decisions`` contract (helpers + result types), so no
network is used. The stub replaces ``chat._decisions`` in every test; when the
real ``repofinder.decisions`` module does not exist yet, it is also registered
under that name so that ``repofinder.chat`` can be imported.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pytest

from repofinder import pipeline, prompts
from repofinder.backends import Completion, LLMError
from repofinder.llm import LLM, Citation, ContextChunk

SAMPLE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"
ENGINE_Q = "How does the engine run?"
ENGINE_CITE = "The engine runs in `run_engine` (`mypkg/core.py:13-30`)."


# --------------------------------------------------------------------------- #
# Stub decision layer (same names and signatures as repofinder.decisions)
# --------------------------------------------------------------------------- #

class DecisionsError(RuntimeError):
    pass


class DecisionsConfigError(DecisionsError):
    pass


@dataclass
class Choice:
    label: str
    probabilities: dict[str, float]
    confidence: Optional[float]
    cost_usd: float
    input_tokens: int


@dataclass
class YesNo:
    probability: float
    cost_usd: float
    input_tokens: int


@dataclass
class RouteResult:
    route: str
    probabilities: dict[str, float]
    cost_usd: float


@dataclass
class RerankResult:
    order: list[int]
    none_selected: bool
    probabilities: dict[str, float]
    cost_usd: float


@dataclass
class _StubDecisions:
    """Implements the ``Decisions`` protocol and scripts the helpers' answers.

    ``routes``: route names (or exceptions to raise), one per turn; default "locate".
    ``reranks``: RerankResult (or exceptions), one per rerank; default: keep the order.
    ``suffices``: probabilities (or exceptions), one per context check; default 0.8.
    """
    routes: list[Any] = field(default_factory=list)
    reranks: list[Any] = field(default_factory=list)
    suffices: list[Any] = field(default_factory=list)
    name: str = "stub"
    cost_usd: float = 0.0
    log: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def choose(self, state: str, instructions: str, options: dict[str, str]) -> Choice:
        label = next(iter(options))
        return Choice(label, {label: 1.0}, 1.0, self.cost_usd, 0)

    def yes_no(self, state: str, question: str) -> YesNo:
        return YesNo(0.5, self.cost_usd, 0)

    def take(self, queue: list[Any], default: Any) -> Any:
        item = queue.pop(0) if queue else default
        if isinstance(item, BaseException):
            raise item
        return item

    def steps(self, step: str) -> list[dict[str, Any]]:
        return [args for name, args in self.log if name == step]


def route_message(decisions: _StubDecisions, message: str, history: list[tuple[str, str]]) -> RouteResult:
    decisions.log.append(("route", {"message": message, "history": list(history)}))
    route = decisions.take(decisions.routes, "locate")
    runner_up = "locate" if route == "chat" else "chat"
    return RouteResult(route, {route: 0.9, runner_up: 0.1}, decisions.cost_usd)


def rerank_cards(decisions: _StubDecisions, question: str, cards: list[str],
                 allow_none: bool = True) -> RerankResult:
    decisions.log.append(("rerank", {"question": question, "cards": list(cards), "allow_none": allow_none}))
    result = decisions.take(decisions.reranks, None)
    if result is None:
        result = RerankResult(list(range(len(cards))), False, {}, decisions.cost_usd)
    return result


def context_suffices(decisions: _StubDecisions, message: str, previous_question: str,
                     cards: list[str]) -> YesNo:
    decisions.log.append(("context", {"message": message, "previous_question": previous_question,
                                      "cards": list(cards)}))
    return YesNo(float(decisions.take(decisions.suffices, 0.8)), decisions.cost_usd, 0)


def _stub_module() -> types.ModuleType:
    mod = types.ModuleType("repofinder.decisions")
    mod.ROUTE_LOCATE, mod.ROUTE_EXPLAIN, mod.ROUTE_FOLLOWUP = "locate", "explain", "followup"
    mod.ROUTE_CHAT, mod.ROUTE_OUTSIDE = "chat", "outside"
    mod.ROUTES = ("locate", "explain", "followup", "chat", "outside")
    for obj in (DecisionsError, DecisionsConfigError, Choice, YesNo, RouteResult, RerankResult,
                route_message, rerank_cards, context_suffices):
        setattr(mod, obj.__name__, obj)
    return mod


STUB = _stub_module()

try:
    import repofinder.decisions  # noqa: F401
except ModuleNotFoundError as exc:  # the decisions module is written separately
    if exc.name != "repofinder.decisions":
        raise
    sys.modules["repofinder.decisions"] = STUB

from repofinder import chat  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

class FakeBackend:
    name = "fake"

    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.calls: list[dict[str, Any]] = []

    def complete(self, *, model: str, system: str, user: str, max_tokens: int,
                 effort: Optional[str] = None) -> Completion:
        self.calls.append({"model": model, "system": system, "user": user,
                           "max_tokens": max_tokens, "effort": effort})
        text = self.texts.pop(0) if self.texts else "ok"
        if isinstance(text, BaseException):
            raise text
        return Completion(text=text, model=model, input_tokens=100, output_tokens=20,
                          cache_read_tokens=0, stop_reason="end_turn")


@pytest.fixture
def ws(tmp_path: Path):
    workspace = pipeline.open_workspace(str(SAMPLE_REPO), data_dir=tmp_path / "home")
    yield workspace
    workspace.close()


@pytest.fixture(autouse=True)
def stub_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    monkeypatch.setattr(chat, "_decisions", STUB)
    return STUB


@pytest.fixture
def searches(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    """Records every ``retrieve`` call made by the chat layer as (query, k)."""
    calls: list[tuple[str, int]] = []
    real = chat.retrieve

    def recording(ws: Any, question: str, k: int = pipeline.DEFAULT_K,
                  max_chars: int = pipeline.DEFAULT_MAX_CHARS):
        calls.append((question, k))
        return real(ws, question, k=k, max_chars=max_chars)

    monkeypatch.setattr(chat, "retrieve", recording)
    return calls


def make_session(ws: pipeline.Workspace, decisions: _StubDecisions, *texts: str,
                 **kwargs: Any) -> tuple[chat.ChatSession, FakeBackend]:
    backend = FakeBackend(*texts)
    return chat.ChatSession(ws, LLM(backend=backend), decisions, **kwargs), backend


def note(name: str) -> str:
    return prompts.CHAT_MODE_NOTES[name]


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

def test_locate_turn_searches_reranks_and_cites(ws, searches):
    decisions = _StubDecisions(routes=["locate"])
    session, backend = make_session(ws, decisions, ENGINE_CITE, k=2)

    result = session.turn(f"  {ENGINE_Q}  ")

    assert searches == [(ENGINE_Q, 2)]
    assert result.route == "locate" and result.route_probabilities == {"locate": 0.9, "chat": 0.1}
    assert [(c.path, c.start_line, c.end_line) for c in result.chunks] == [
        ("mypkg/core.py", 13, 30), ("mypkg/core.py", 33, 63)]
    assert len(result.hits) == 2
    assert result.answer.citations == [Citation("mypkg/core.py", 13, 30)]
    assert not result.reused_context and not result.widened and not result.off_target
    assert result.decision_errors == []

    (rerank,) = decisions.steps("rerank")
    assert rerank["question"] == ENGINE_Q and rerank["allow_none"] is True
    first_card = rerank["cards"][0]
    assert first_card.startswith("mypkg/core.py:13-30 | run_engine\n```python\ndef run_engine(")

    (call,) = backend.calls
    assert call["system"] == prompts.chat_system("sample_repo")
    user = call["user"]
    assert "### mypkg/core.py:13-30 (run_engine)" in user
    assert user.index("<code_excerpts>") < user.index("<turn_note>") < user.index("<message>")
    assert note("locate") in user and "<conversation>" not in user
    assert user.rstrip().endswith(f"<message>\n{ENGINE_Q}\n</message>")

    assert session.last_question == ENGINE_Q and session.last_chunks == result.chunks
    assert [(t.role, t.route) for t in session.history] == [("user", "locate"), ("assistant", "locate")]
    assert session.history[1].citations == [Citation("mypkg/core.py", 13, 30)]
    assert session.last_result is result


def test_llm_call_uses_answer_limits(ws):
    from repofinder import llm as llm_mod

    session, backend = make_session(ws, _StubDecisions(routes=["explain"]), ENGINE_CITE)
    session.turn(ENGINE_Q)
    (call,) = backend.calls
    assert call["max_tokens"] == llm_mod.ANSWER_MAX_TOKENS and call["effort"] == llm_mod.SMART_EFFORT
    assert note("explain") in call["user"]


def test_rerank_order_is_applied_to_the_excerpts(ws):
    decisions = _StubDecisions(routes=["locate"], reranks=[RerankResult([1, 0], False, {}, 0.0)])
    session, backend = make_session(ws, decisions, "ok", k=2)
    result = session.turn(ENGINE_Q)
    assert [c.label for c in result.chunks] == ["Engine", "run_engine"]
    user = backend.calls[0]["user"]
    assert user.index("(Engine)") < user.index("(run_engine)")


def test_followup_reuses_previous_excerpts(ws, searches):
    decisions = _StubDecisions(routes=["locate", "followup"], suffices=[0.8])
    session, backend = make_session(ws, decisions, ENGINE_CITE, "It retries `mypkg/core.py:13-30`.", k=2)
    first = session.turn(ENGINE_Q)

    result = session.turn("ve kaç kez dener?")

    assert searches == [(ENGINE_Q, 2)]          # no second search
    assert result.route == "followup" and result.reused_context and result.context_probability == 0.8
    assert result.chunks == first.chunks and result.hits == []
    assert result.answer.citations == [Citation("mypkg/core.py", 13, 30)]
    (check,) = decisions.steps("context")
    assert check["message"] == "ve kaç kez dener?" and check["previous_question"] == ENGINE_Q
    assert len(check["cards"]) == 2 and check["cards"][0].startswith("mypkg/core.py:13-30 | run_engine")
    assert len(decisions.steps("rerank")) == 1

    user = backend.calls[1]["user"]
    assert note("followup") in user
    assert user.index("<conversation>") < user.index("<code_excerpts>")
    assert f'<turn role="user">\n{ENGINE_Q}\n</turn>' in user
    assert '<turn role="assistant">\nThe engine runs in' in user
    assert session.last_question == ENGINE_Q      # unchanged by a reused turn
    # the router saw the first exchange
    assert decisions.steps("route")[1]["history"] == [("user", ENGINE_Q), ("assistant", ENGINE_CITE)]


def test_followup_with_insufficient_context_searches_again(ws, searches):
    decisions = _StubDecisions(routes=["locate", "followup"], suffices=[0.3])
    session, backend = make_session(ws, decisions, "a", "b", k=2)
    session.turn(ENGINE_Q)

    result = session.turn("what about the user model?")

    combined = f"{ENGINE_Q}\nwhat about the user model?"
    assert searches == [(ENGINE_Q, 2), (combined, 2)]
    assert result.route == "followup" and not result.reused_context and result.context_probability == 0.3
    assert result.hits and decisions.steps("rerank")[1]["question"] == combined
    assert note("followup_search") in backend.calls[1]["user"]
    assert "<message>\nwhat about the user model?\n</message>" in backend.calls[1]["user"]
    assert session.last_question == combined and session.last_chunks == result.chunks


@pytest.mark.parametrize("probability, reused", [(0.5, True), (0.49, False)])
def test_context_threshold_is_inclusive(ws, probability, reused):
    decisions = _StubDecisions(routes=["locate", "followup"], suffices=[probability])
    session, _ = make_session(ws, decisions, "a", "b", k=2)
    session.turn(ENGINE_Q)
    assert session.turn("and then?").reused_context is reused


def test_followup_without_previous_excerpts_is_a_search(ws, searches):
    decisions = _StubDecisions(routes=["followup"])
    session, backend = make_session(ws, decisions, "a", k=2)
    result = session.turn(ENGINE_Q)
    assert searches == [(ENGINE_Q, 2)] and decisions.steps("context") == []
    assert result.route == "followup" and not result.reused_context
    assert note("locate") in backend.calls[0]["user"]


@pytest.mark.parametrize("route", ["chat", "outside"])
def test_chat_and_outside_turns_do_not_search(ws, searches, route):
    decisions = _StubDecisions(routes=["locate", route])
    session, backend = make_session(ws, decisions, ENGINE_CITE, "Hi! (`mypkg/core.py:13-30`)", k=2)
    first = session.turn(ENGINE_Q)

    result = session.turn("teşekkürler!" if route == "chat" else "How does requests do retries?")

    assert searches == [(ENGINE_Q, 2)] and len(decisions.steps("rerank")) == 1
    assert result.route == route and result.chunks == [] and result.hits == []
    assert result.answer.citations == []        # nothing to cite without excerpts
    user = backend.calls[1]["user"]
    assert "<code_excerpts>" not in user and note(route) in user
    assert "<conversation>" in user
    assert backend.calls[1]["effort"] == chat.LIGHT_EFFORT == "low"
    assert backend.calls[0]["effort"] == "medium"      # turns with excerpts keep the default
    assert session.last_chunks == first.chunks and session.last_question == ENGINE_Q


def test_unknown_route_falls_back_to_locate(ws, searches):
    session, _ = make_session(ws, _StubDecisions(routes=["dance"]), "a", k=2)
    assert session.turn(ENGINE_Q).route == "locate"
    assert searches == [(ENGINE_Q, 2)]


# --------------------------------------------------------------------------- #
# Widening
# --------------------------------------------------------------------------- #

def test_none_selected_widens_the_search_once(ws, searches):
    decisions = _StubDecisions(routes=["locate"], reranks=[RerankResult([], True, {}, 0.0)])
    session, backend = make_session(ws, decisions, ENGINE_CITE, k=2)

    result = session.turn(ENGINE_Q)

    assert searches == [(ENGINE_Q, 2), (ENGINE_Q, 6)]
    assert result.widened and not result.off_target
    assert len(result.hits) == 6
    assert [(c.path, c.start_line, c.end_line) for c in result.chunks] == [("mypkg/core.py", 1, 63)]
    assert len(decisions.steps("rerank")) == 2
    assert note("widened") not in backend.calls[0]["user"]
    assert result.answer.citations == [Citation("mypkg/core.py", 13, 30)]


def test_still_nothing_after_widening_adds_the_missed_note(ws, searches):
    none = RerankResult([], True, {}, 0.0)
    decisions = _StubDecisions(routes=["locate"], reranks=[none, none])
    session, backend = make_session(ws, decisions, "a", k=2)

    result = session.turn(ENGINE_Q)

    assert result.widened and result.off_target and result.chunks   # excerpts are still given
    user = backend.calls[0]["user"]
    assert note("locate") in user and note("widened") in user
    assert "search may have missed the right place" in user
    assert user.index(note("locate")) < user.index(note("widened"))


def test_widening_without_new_hits_skips_the_second_rerank(ws, searches):
    decisions = _StubDecisions(routes=["locate"], reranks=[RerankResult([], True, {}, 0.0)])
    session, _ = make_session(ws, decisions, "a", k=8)
    result = session.turn("user model")          # only 6 hits exist
    assert searches == [("user model", 8), ("user model", 24)]
    assert result.widened and result.off_target
    assert len(decisions.steps("rerank")) == 1


def test_cards_are_capped_and_the_rest_keep_their_order(ws):
    session, _ = make_session(ws, _StubDecisions(reranks=[RerankResult([2, 0, 99, 2, -1], False, {}, 0.0)]))
    chunks = [ContextChunk(f"pkg/m{i}.py", 1, 50, "x = 1\n" * 50, f"f{i}") for i in range(20)]
    ledger = chat._Ledger(counts_calls=True)

    ordered, none_selected = session._rerank("q", chunks, ledger)

    (step,) = session.decisions.steps("rerank")
    assert len(step["cards"]) == chat.MAX_CARDS
    assert step["cards"][0].endswith(f"... {50 - chat.CARD_LINES} more lines (to line 50)\n```")
    assert not none_selected
    assert [c.label for c in ordered][:4] == ["f2", "f0", "f1", "f3"]
    assert [c.label for c in ordered][-4:] == ["f16", "f17", "f18", "f19"]
    assert ledger.calls == 1


def test_apply_order():
    items = list("abcdef")
    assert chat.apply_order(items, [3, 1], 4) == list("dbacef")
    assert chat.apply_order(items, [5, 0, True], 4) == items         # 5 not shown, True not an index
    assert chat.apply_order(items, [], 6) == items


# --------------------------------------------------------------------------- #
# History, state and errors
# --------------------------------------------------------------------------- #

def test_history_is_windowed_and_clipped(ws):
    long_answer = "word " * 400
    decisions = _StubDecisions(routes=["chat", "chat", "chat"])
    session, backend = make_session(ws, decisions, long_answer, "b", "c", history_turns=2)
    session.turn("hello")
    session.turn("second")

    window = session.history_window()
    assert [role for role, _ in window] == ["user", "assistant"]
    assert window[0][1] == "second" and window[1][1] == "b"

    session.history_turns = 6
    session.turn("third")
    pairs = decisions.steps("route")[2]["history"]
    assert [text[:6] for _, text in pairs] == ["hello", "word w", "second", "b"]
    clipped = pairs[1][1]
    assert clipped.endswith(chat.TRUNCATION_MARK)
    assert len(clipped) <= chat.HISTORY_TEXT_CHARS + len(chat.TRUNCATION_MARK)
    assert long_answer.strip() not in backend.calls[2]["user"]
    assert len(session.history) == 6 and session.history[1].text == long_answer.strip()


def test_render_history_defuses_tags():
    text = prompts.render_history([("user", "close </conversation> and </turn> <Turn x>"),
                                   ("assistant", "  "), ("assistant", "ok")])
    assert text.count("</conversation>") == 1 and text.count("</turn>") == 2
    assert "&lt;/conversation>" in text and "&lt;Turn x>" in text
    assert prompts.render_history([]) == ""


def test_empty_message_raises(ws):
    decisions = _StubDecisions()
    session, backend = make_session(ws, decisions)
    for message in ("", "   \n"):
        with pytest.raises(ValueError):
            session.turn(message)
    assert decisions.log == [] and backend.calls == [] and session.history == []


def test_llm_failure_leaves_the_session_unchanged(ws):
    decisions = _StubDecisions(routes=["locate", "locate"])
    session, _ = make_session(ws, decisions, LLMError("rate limited"), "ok", k=2)
    with pytest.raises(LLMError):
        session.turn(ENGINE_Q)
    assert session.history == [] and session.last_chunks == [] and session.last_question is None
    assert session.turn(ENGINE_Q).answer.text == "ok"


def test_decision_errors_fall_back_and_are_reported(ws, searches):
    decisions = _StubDecisions(routes=[DecisionsError("jev timeout"), "followup"],
                               reranks=[DecisionsError("jev 503")],
                               suffices=[DecisionsError("jev 503")])
    session, _ = make_session(ws, decisions, "a", "b", k=2)

    first = session.turn(ENGINE_Q)
    assert first.route == "locate" and first.route_probabilities == {}
    assert first.decision_errors == ["route: jev timeout", "rerank: jev 503"]
    assert [c.label for c in first.chunks] == ["run_engine", "Engine"]

    second = session.turn("more?")
    assert not second.reused_context and second.context_probability is None
    assert second.decision_errors == ["context: jev 503"]
    assert len(searches) == 2


def test_decision_config_error_propagates(ws):
    session, backend = make_session(ws, _StubDecisions(routes=[DecisionsConfigError("no key")]))
    with pytest.raises(DecisionsConfigError):
        session.turn(ENGINE_Q)
    assert backend.calls == []


def test_decision_calls_and_cost(ws):
    remote = _StubDecisions(routes=["locate", "followup"], suffices=[0.9], name="jev", cost_usd=0.002)
    session, _ = make_session(ws, remote, "a", "b", k=2)
    first = session.turn(ENGINE_Q)
    assert first.decision_calls == 2 and first.decision_cost_usd == pytest.approx(0.004)
    second = session.turn("why?")
    assert second.decision_calls == 2 and second.reused_context

    local = _StubDecisions(routes=["locate"], name="rules")
    session, _ = make_session(ws, local, "a", k=2)
    result = session.turn(ENGINE_Q)
    assert result.decision_calls == 0 and result.decision_cost_usd == 0.0
    assert len(local.log) == 2   # the helpers still ran


def test_join_query_and_clip_text():
    assert chat.join_query(None, "m") == "m"
    assert chat.join_query("q", "m") == "q\nm"
    long_prev = "\n".join(f"line {i} " + "x" * 50 for i in range(40))
    joined = chat.join_query(long_prev, "new")
    assert len(joined) <= chat.MAX_QUERY_CHARS and joined.endswith("\nnew")
    assert joined.startswith("line ")                 # cut at a line boundary
    assert chat.join_query("q", "y" * 2000) == "y" * 2000
    assert chat.clip_text("short", 600) == "short"
    assert chat.clip_text("aaa bbb ccc", 9) == "aaa bbb" + chat.TRUNCATION_MARK

"""Tests for repofinder.decisions: rules, the fake, Jev through a fake client
(and through the real typesafe-sdk with a mock HTTP transport), detection and
the chat helpers. Nothing here touches the network."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import pytest

from repofinder import decisions as dec
from repofinder.decisions import (
    CARD_LABELS,
    CONTEXT_QUESTION,
    MAX_REQUEST_TOKENS,
    NONE_LABEL,
    QUESTION_NAME,
    ROUTE_CHAT,
    ROUTE_EXPLAIN,
    ROUTE_FOLLOWUP,
    ROUTE_INSTRUCTIONS,
    ROUTE_LOCATE,
    ROUTE_OPTIONS,
    ROUTE_OUTSIDE,
    ROUTES,
    Choice,
    Decisions,
    DecisionsConfigError,
    DecisionsError,
    FakeDecisions,
    JevDecisions,
    RulesDecisions,
    YesNo,
    context_suffices,
    detect_decisions,
    rerank_cards,
    route_message,
    rule_context_suffices,
    rule_route,
)

ROOT = Path(__file__).resolve().parents[1]
H = [("user", "Where is url_for defined?"), ("assistant", "In src/flask/helpers.py:200-260.")]
URL_FOR_CARD = (
    "src/flask/helpers.py:200-260 | function url_for\n"
    "```python\ndef url_for(endpoint, *, _external=False):\n    return current_app.url_for(endpoint)\n```"
)


def card(i: int, body: str = "") -> str:
    return f"src/pkg/mod{i}.py:{i * 10}-{i * 10 + 9} | function func_{i}\n```python\ndef func_{i}():\n{body}\n```"


# --------------------------------------------------------------------------- #
# Rules: routing
# --------------------------------------------------------------------------- #

ROUTE_CASES = [
    # English: location, explanation
    ("Where is the route decorator implemented?", [], ROUTE_LOCATE),
    ("Which file defines `url_for`?", [], ROUTE_LOCATE),
    ("Flask.add_url_rule", [], ROUTE_LOCATE),
    ("the config loader", [], ROUTE_LOCATE),
    ("who calls it?", [], ROUTE_LOCATE),                     # no history: cannot be a follow-up
    ("How does request dispatching work?", [], ROUTE_EXPLAIN),
    ("Why does the app context get pushed twice?", [], ROUTE_EXPLAIN),
    ("explain the blueprint registration", [], ROUTE_EXPLAIN),
    ("what does `Flask.run` do?", [], ROUTE_EXPLAIN),
    # English: follow-ups, and new questions despite history
    ("who calls it?", H, ROUTE_FOLLOWUP),
    ("and the tests?", H, ROUTE_FOLLOWUP),
    ("and where is it tested?", H, ROUTE_FOLLOWUP),
    ("why?", H, ROUTE_FOLLOWUP),
    ("how does this work?", H, ROUTE_FOLLOWUP),
    ("ok so what calls it", H, ROUTE_FOLLOWUP),
    ("what about `send_file`?", H, ROUTE_LOCATE),             # an identifier is a new search target
    ("where is the session cookie set?", H, ROUTE_LOCATE),
    ("Where is `url_for` defined and how does it build URLs?", H, ROUTE_LOCATE),
    ("how does this repo load config?", H, ROUTE_EXPLAIN),
    # English: chat
    ("", [], ROUTE_CHAT),
    ("hi!", [], ROUTE_CHAT),
    ("thanks, that helped", H, ROUTE_CHAT),
    ("great, thanks for the explanation", H, ROUTE_CHAT),
    ("ok", H, ROUTE_CHAT),
    ("hmm", H, ROUTE_CHAT),
    ("I see, interesting", [], ROUTE_CHAT),
    ("what can you do?", [], ROUTE_CHAT),
    ("how do you work?", [], ROUTE_CHAT),
    # Turkish
    ("Route dekoratörü nerede tanımlı?", [], ROUTE_LOCATE),
    ("Hangi dosya oturum çerezini yazıyor?", [], ROUTE_LOCATE),
    ("bu repoda ayarlar nerede yükleniyor?", H, ROUTE_LOCATE),
    ("İstek nasıl işleniyor?", [], ROUTE_EXPLAIN),
    ("nasil calisiyor bu fonksiyon", [], ROUTE_EXPLAIN),      # Turkish typed without its letters
    ("Session.open_session ne işe yarar?", [], ROUTE_EXPLAIN),
    ("peki bunu kim çağırıyor?", H, ROUTE_FOLLOWUP),
    ("kim çağırıyor?", H, ROUTE_FOLLOWUP),
    ("ya testler?", H, ROUTE_FOLLOWUP),
    ("neden?", H, ROUTE_FOLLOWUP),
    ("nerede tanımlı?", H, ROUTE_FOLLOWUP),
    ("nasil calisiyor bu fonksiyon", H, ROUTE_FOLLOWUP),
    ("merhaba", [], ROUTE_CHAT),
    ("teşekkürler", H, ROUTE_CHAT),
    ("teşekkürler, çok işime yaradı", H, ROUTE_CHAT),
    ("sağ ol", H, ROUTE_CHAT),
    ("nasıl çalışıyorsun?", [], ROUTE_CHAT),
    ("ne yapabilirsin?", [], ROUTE_CHAT),
]


@pytest.mark.parametrize("message, history, expected", ROUTE_CASES)
def test_rule_route(message: str, history: list[tuple[str, str]], expected: str) -> None:
    assert rule_route(message, history) == expected


def test_route_cases_cover_every_rule_route_in_both_languages() -> None:
    assert len(ROUTE_CASES) >= 15
    seen = {expected for _, _, expected in ROUTE_CASES}
    assert seen == {ROUTE_LOCATE, ROUTE_EXPLAIN, ROUTE_FOLLOWUP, ROUTE_CHAT}


def test_rules_never_answer_outside() -> None:
    for message, history, _ in ROUTE_CASES:
        assert rule_route(message, history) != ROUTE_OUTSIDE


def test_route_message_with_rules_is_one_hot_free_and_counted() -> None:
    rules = RulesDecisions()
    result = route_message(rules, "Where is the route decorator implemented?", [])
    assert result.route == ROUTE_LOCATE
    assert result.probabilities == {r: (1.0 if r == ROUTE_LOCATE else 0.0) for r in ROUTES}
    assert result.cost_usd == 0.0
    route_message(rules, "who calls it?", H)
    assert rules.usage.calls == 2 and rules.usage.input_tokens == 0 and rules.usage.cost_usd == 0.0


def test_rules_generic_choose_and_yes_no_have_no_rule() -> None:
    rules = RulesDecisions()
    choice = rules.choose("any state", "pick", {"x": "first", "y": "second"})
    assert (choice.label, choice.probabilities, choice.confidence) == ("x", {"x": 1.0, "y": 0.0}, None)
    assert rules.yes_no("any state", "yes?") == YesNo(0.5, 0.0, 0)
    with pytest.raises(ValueError):
        rules.choose("s", "pick", {})


def test_backends_satisfy_the_protocol() -> None:
    for backend in (RulesDecisions(), FakeDecisions(), JevDecisions(client=object())):
        assert isinstance(backend, Decisions)
    assert (RulesDecisions.name, FakeDecisions.name, JevDecisions.name) == ("rules", "fake", "jev")


# --------------------------------------------------------------------------- #
# route_message through a scripted backend
# --------------------------------------------------------------------------- #

def test_route_message_state_shows_the_last_three_exchanges_truncated() -> None:
    history: list[tuple[str, str]] = []
    for i in range(5):
        history += [("user", f"question marker-q{i}x"), ("assistant", f"answer marker-a{i}x " + "z" * 5000)]
    fake = FakeDecisions(choices=[ROUTE_OUTSIDE])
    result = route_message(fake, "and the database layer?", history)

    assert result.route == ROUTE_OUTSIDE and result.cost_usd == 0.0
    assert result.probabilities[ROUTE_OUTSIDE] == 1.0
    (call,) = fake.calls
    assert call.kind == "choose" and call.prompt == ROUTE_INSTRUCTIONS
    assert list(call.options) == list(ROUTES) and call.options == ROUTE_OPTIONS
    for i in (0, 1):
        assert f"marker-q{i}x" not in call.state and f"marker-a{i}x" not in call.state
    for i in (2, 3, 4):
        assert f"marker-q{i}x" in call.state and f"marker-a{i}x" in call.state
    assert call.state.rstrip().endswith("and the database layer?")
    assert max(len(line) for line in call.state.splitlines()) <= len("assistant: ") + dec.HISTORY_TEXT_CHARS


def test_route_message_first_message_says_so() -> None:
    fake = FakeDecisions(choices=[ROUTE_CHAT])
    route_message(fake, "hello", [])
    assert "none, this is the first message" in fake.calls[0].state


def test_route_message_passes_scripted_probabilities_through() -> None:
    probs = {ROUTE_LOCATE: 0.1, ROUTE_EXPLAIN: 0.7, ROUTE_FOLLOWUP: 0.1, ROUTE_CHAT: 0.05, ROUTE_OUTSIDE: 0.05}
    fake = FakeDecisions(choices=[Choice(ROUTE_EXPLAIN, probs, 0.7, 0.000002, 50)])
    result = route_message(fake, "how does it work?", [])
    assert (result.route, result.probabilities, result.cost_usd) == (ROUTE_EXPLAIN, probs, 0.000002)


def test_route_message_rejects_a_label_outside_the_routes() -> None:
    with pytest.raises(DecisionsError, match="not one of"):
        route_message(FakeDecisions(choices=["search"]), "where is x?", [])


# --------------------------------------------------------------------------- #
# FakeDecisions
# --------------------------------------------------------------------------- #

def test_fake_records_calls_and_consumes_the_script_in_order() -> None:
    fake = FakeDecisions(choices=["b", "a"], yes_no=[0.9, 0.2])
    assert fake.choose("s1", "pick", {"a": "A", "b": "B"}).label == "b"
    assert fake.yes_no("s2", "q?").probability == 0.9
    assert fake.choose("s3", "pick", {"a": "A"}).label == "a"
    assert fake.yes_no("s4", "q?").probability == 0.2
    assert [(c.kind, c.state) for c in fake.calls] == [
        ("choose", "s1"), ("yes_no", "s2"), ("choose", "s3"), ("yes_no", "s4")]
    assert fake.calls[1].prompt == "q?" and fake.calls[1].options is None
    assert fake.usage.calls == 4


def test_fake_raises_when_the_script_runs_out() -> None:
    fake = FakeDecisions()
    with pytest.raises(DecisionsError, match="no scripted choice"):
        fake.choose("s", "pick", {"a": "A"})
    with pytest.raises(DecisionsError, match="no scripted yes/no"):
        fake.yes_no("s", "q?")
    assert len(fake.calls) == 2


# --------------------------------------------------------------------------- #
# rerank_cards
# --------------------------------------------------------------------------- #

def test_rerank_with_rules_keeps_the_search_order() -> None:
    cards = [card(i) for i in range(4)]
    result = rerank_cards(RulesDecisions(), "where is func_2?", cards)
    assert result.order == [0, 1, 2, 3] and result.none_selected is False and result.cost_usd == 0.0
    assert result.probabilities == {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0, NONE_LABEL: 0.0}
    no_none = rerank_cards(RulesDecisions(), "q", cards, allow_none=False)
    assert NONE_LABEL not in no_none.probabilities


def test_rerank_moves_the_pick_first() -> None:
    cards = [card(i) for i in range(4)]
    fake = FakeDecisions(choices=["C"])
    result = rerank_cards(fake, "where is func_2?", cards)
    assert result.order == [2, 0, 1, 3] and result.none_selected is False
    (call,) = fake.calls
    assert list(call.options) == ["A", "B", "C", "D", NONE_LABEL]
    assert call.options["C"] == "src/pkg/mod2.py:20-29 | function func_2"
    assert "Pick NONE" in call.prompt
    assert "where is func_2?" in call.state
    for label, text in zip("ABCD", cards):
        assert f"[{label}]\n{text}" in call.state


def test_rerank_none_keeps_the_order_and_flags_it() -> None:
    cards = [card(i) for i in range(3)]
    result = rerank_cards(FakeDecisions(choices=[NONE_LABEL]), "q", cards)
    assert result.none_selected is True and result.order == [0, 1, 2]


def test_rerank_without_none_does_not_offer_it() -> None:
    fake = FakeDecisions(choices=["A"])
    rerank_cards(fake, "q", [card(0), card(1)], allow_none=False)
    assert list(fake.calls[0].options) == ["A", "B"] and "NONE" not in fake.calls[0].prompt
    with pytest.raises(DecisionsError):
        rerank_cards(FakeDecisions(choices=[NONE_LABEL]), "q", [card(0)], allow_none=False)


def test_rerank_sorts_the_others_by_probability() -> None:
    probs = {"A": 0.1, "B": 0.2, "C": 0.6, "D": 0.05, NONE_LABEL: 0.05}
    result = rerank_cards(FakeDecisions(choices=[Choice("C", probs, 0.6, 0.0, 0)]), "q",
                          [card(i) for i in range(4)])
    assert result.order == [2, 1, 0, 3] and result.probabilities == probs


def test_rerank_limits_and_empty_input() -> None:
    with pytest.raises(ValueError):
        rerank_cards(RulesDecisions(), "q", [card(i) for i in range(17)])
    fake = FakeDecisions()
    assert rerank_cards(fake, "q", []) == dec.RerankResult([], False, {}, 0.0)
    assert fake.calls == []
    full = rerank_cards(FakeDecisions(choices=["P"]), "q", [card(i) for i in range(16)])
    assert full.order[0] == 15 and sorted(full.order) == list(range(16))
    assert CARD_LABELS[-1] == "P"


def test_rerank_drops_the_lowest_ranked_cards_over_the_token_budget() -> None:
    cards = [card(i, "    x = 1\n" * 3_000) for i in range(5)]       # ~30k characters (~7.5k tokens) each
    fake = FakeDecisions(choices=["B"])
    result = rerank_cards(fake, "where is func_1?", cards)
    (call,) = fake.calls
    assert list(call.options) == ["A", "B", NONE_LABEL]                # only two cards fit 22k tokens
    question = {"type": "choice", "instructions": call.prompt, "criteria": call.options}
    assert dec.request_tokens(call.state, question) <= MAX_REQUEST_TOKENS
    assert "[C]" not in call.state
    assert result.order == [1, 0, 2, 3, 4]                             # dropped ones keep search order


def test_rerank_cuts_a_single_card_that_alone_exceeds_the_budget() -> None:
    huge = card(0, "    y = 2\n" * 20_000)                              # ~200k characters
    fake = FakeDecisions(choices=["A"])
    result = rerank_cards(fake, "q", [huge, card(1)])
    (call,) = fake.calls
    assert list(call.options) == ["A", NONE_LABEL]
    assert "cut to fit the request" in call.state
    question = {"type": "choice", "instructions": call.prompt, "criteria": call.options}
    assert dec.request_tokens(call.state, question) <= MAX_REQUEST_TOKENS
    assert result.order == [0, 1]


# --------------------------------------------------------------------------- #
# context_suffices
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("message, previous, expected", [
    ("what does url_for return?", "where is url_for defined?", 0.8),
    ("does url_for support _external?", "Where is url_for defined?", 0.8),
    ("peki url_for'un _external parametresi ne yapıyor?", "url_for nerede tanımlı?", 0.8),
    ("who calls it?", "where is url_for defined?", 0.3),             # no shared word
    ("what about sessions?", "where is url_for defined?", 0.3),
    ("what does url_for return?", "", 0.3),                          # no previous question
    ("where are routes registered for url building?", "how is url building done?", 0.3),  # no card identifier
])
def test_rule_context_suffices(message: str, previous: str, expected: float) -> None:
    assert rule_context_suffices(message, previous, [URL_FOR_CARD]) == expected


def test_context_suffices_with_rules() -> None:
    rules = RulesDecisions()
    answer = context_suffices(rules, "what does url_for return?", "where is url_for defined?", [URL_FOR_CARD])
    assert answer == YesNo(0.8, 0.0, 0) and rules.usage.calls == 1


def test_context_suffices_without_cards_makes_no_call() -> None:
    fake = FakeDecisions(yes_no=[0.99])
    assert context_suffices(fake, "who calls it?", "where is x?", []) == YesNo(0.0, 0.0, 0)
    assert fake.calls == []


def test_context_suffices_asks_the_backend() -> None:
    fake = FakeDecisions(yes_no=[0.9])
    answer = context_suffices(fake, "what does it return?", "where is url_for defined?", [URL_FOR_CARD])
    assert answer.probability == 0.9
    (call,) = fake.calls
    assert call.kind == "yes_no" and call.prompt == CONTEXT_QUESTION
    assert "where is url_for defined?" in call.state and "what does it return?" in call.state
    assert "[1]\n" + URL_FOR_CARD in call.state


def test_context_suffices_leaves_out_cards_over_the_budget() -> None:
    cards = [card(i, "    x = 1\n" * 3_000) for i in range(5)]
    fake = FakeDecisions(yes_no=[0.4])
    context_suffices(fake, "and then?", "where?", cards)
    state = fake.calls[0].state
    assert "[1]\n" in state and "[3]\n" not in state
    assert dec.request_tokens(state, {"type": "noul", "instructions": CONTEXT_QUESTION}) <= MAX_REQUEST_TOKENS


# --------------------------------------------------------------------------- #
# Jev through a fake client
# --------------------------------------------------------------------------- #

def jev_response(group: str, answer: Any, *, input_tokens: Optional[int] = 120,
                 cost: Optional[float] = 5.04e-06, raw: bool = True) -> SimpleNamespace:
    """Shaped like ``typesafe_sdk.SystemOneResponse``."""
    body = {"usage": {"input_tokens": input_tokens, "output_tokens": 2}}
    if cost is not None:
        body["usage"]["cost"] = cost

    def raw_json() -> Any:
        if not raw:
            raise RuntimeError("The response was not created from a raw HTTP response.")
        return body

    return SimpleNamespace(model="jev-test", **{group: {QUESTION_NAME: answer}},
                           usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=2),
                           raw_http_response=SimpleNamespace(json=raw_json))


class FakeClient:
    """Stands in for ``TypeSafeClient``: ``respond(state, questions)`` gives the response."""

    def __init__(self, respond: Optional[Callable[[str, dict[str, Any]], Any]] = None,
                 error: Optional[BaseException] = None) -> None:
        self.respond = respond
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def system_one(self, state: str, questions: dict[str, Any]) -> Any:
        self.calls.append((state, questions))
        if self.error is not None:
            raise self.error
        assert self.respond is not None
        return self.respond(state, questions)


def choice_answer(label: str, probabilities: Optional[dict[str, float]] = None,
                  confidence: Optional[float] = 0.8) -> SimpleNamespace:
    return SimpleNamespace(type="choice", choice=label, confidence=confidence, probabilities=probabilities)


OPTIONS = {"a": "the first option", "b": "the second option", "c": "the third option"}


def test_jev_choose_sends_a_choice_question_and_reads_the_answer() -> None:
    probs = {"a": 0.1, "b": 0.75, "c": 0.15}
    client = FakeClient(lambda s, q: jev_response("choices", choice_answer("b", probs, 0.75)))
    jev = JevDecisions(client=client)
    choice = jev.choose("the state", "pick one", OPTIONS)

    assert client.calls == [("the state", {QUESTION_NAME: {"type": "choice", "instructions": "pick one",
                                                           "criteria": OPTIONS}})]
    assert choice == Choice("b", probs, 0.75, 5.04e-06, 120)
    assert (jev.usage.calls, jev.usage.input_tokens, jev.usage.cost_usd) == (1, 120, 5.04e-06)


def test_jev_cost_falls_back_to_tokens_times_price() -> None:
    client = FakeClient(lambda s, q: jev_response("choices", choice_answer("a"), cost=None))
    choice = JevDecisions(client=client).choose("s", "pick", OPTIONS)
    assert choice.cost_usd == pytest.approx(120 * 0.042 / 1_000_000)
    client = FakeClient(lambda s, q: jev_response("choices", choice_answer("a"), raw=False))
    assert JevDecisions(client=client).choose("s", "pick", OPTIONS).cost_usd == pytest.approx(120 * 0.042e-6)


def test_jev_estimates_tokens_when_usage_is_missing() -> None:
    state = "x" * 4_000
    client = FakeClient(lambda s, q: jev_response("choices", choice_answer("a"), input_tokens=None, cost=None))
    choice = JevDecisions(client=client).choose(state, "pick", OPTIONS)
    question = {"type": "choice", "instructions": "pick", "criteria": OPTIONS}
    assert choice.input_tokens == dec.request_tokens(state, question) and choice.input_tokens > 1_000
    assert choice.cost_usd == pytest.approx(choice.input_tokens * 0.042 / 1_000_000)


def test_jev_fills_missing_probabilities() -> None:
    client = FakeClient(lambda s, q: jev_response("choices", choice_answer("c", None, None)))
    choice = JevDecisions(client=client).choose("s", "pick", OPTIONS)
    assert choice.probabilities == {"a": 0.0, "b": 0.0, "c": 1.0} and choice.confidence is None
    client = FakeClient(lambda s, q: jev_response("choices", choice_answer("a", {"a": 0.9, "zzz": 0.1})))
    assert JevDecisions(client=client).choose("s", "pick", OPTIONS).probabilities == {"a": 0.9, "b": 0.0, "c": 0.0}


def test_jev_rejects_an_unknown_label_and_a_missing_answer() -> None:
    client = FakeClient(lambda s, q: jev_response("choices", choice_answer("z")))
    with pytest.raises(DecisionsError, match="not one of the options"):
        JevDecisions(client=client).choose("s", "pick", OPTIONS)
    client = FakeClient(lambda s, q: SimpleNamespace(choices={}, answers={}, usage=None))
    with pytest.raises(DecisionsError, match="no answer"):
        JevDecisions(client=client).choose("s", "pick", OPTIONS)


def test_jev_reads_answers_when_there_is_no_typed_group() -> None:
    def respond(s: str, q: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(answers={QUESTION_NAME: choice_answer("a", {"a": 1.0})},
                               usage=SimpleNamespace(input_tokens=10))
    assert JevDecisions(client=FakeClient(respond)).choose("s", "pick", OPTIONS).label == "a"


def test_jev_yes_no_sends_a_noul_question() -> None:
    client = FakeClient(lambda s, q: jev_response("nouls", SimpleNamespace(type="noul", noul=0.83)))
    jev = JevDecisions(client=client)
    answer = jev.yes_no("the state", "Is it enough?")
    assert client.calls == [("the state", {QUESTION_NAME: {"type": "noul", "instructions": "Is it enough?"}})]
    assert answer == YesNo(0.83, 5.04e-06, 120)
    clamped = JevDecisions(client=FakeClient(
        lambda s, q: jev_response("nouls", SimpleNamespace(noul=1.2)))).yes_no("s", "q")
    assert clamped.probability == 1.0
    with pytest.raises(DecisionsError, match="no yes/no"):
        JevDecisions(client=FakeClient(
            lambda s, q: jev_response("nouls", SimpleNamespace(noul=None)))).yes_no("s", "q")


def test_jev_caches_repeated_decisions() -> None:
    client = FakeClient(lambda s, q: jev_response(
        "choices" if q[QUESTION_NAME]["type"] == "choice" else "nouls",
        choice_answer("b", {"a": 0.2, "b": 0.8}) if q[QUESTION_NAME]["type"] == "choice"
        else SimpleNamespace(noul=0.6)))
    jev = JevDecisions(client=client)
    first = jev.choose("s", "pick", {"a": "A", "b": "B"})
    again = jev.choose("s", "pick", {"a": "A", "b": "B"})
    assert len(client.calls) == 1
    assert (again.label, again.probabilities, again.confidence) == (first.label, first.probabilities, first.confidence)
    assert (again.cost_usd, again.input_tokens) == (0.0, 0) and first.cost_usd > 0
    again.probabilities["a"] = 99.0                                    # a returned copy, not the cached object
    assert jev.choose("s", "pick", {"a": "A", "b": "B"}).probabilities["a"] == 0.2
    jev.choose("s", "pick", {"a": "A", "b": "B", "c": "C"})            # other options: a new request
    jev.choose("s2", "pick", {"a": "A", "b": "B"})                     # other state: a new request
    assert len(client.calls) == 3
    assert jev.yes_no("s", "q?") == YesNo(0.6, 5.04e-06, 120)
    assert jev.yes_no("s", "q?") == YesNo(0.6, 0.0, 0)
    assert len(client.calls) == 4 and jev.usage.calls == 4


def test_jev_route_message_and_rerank_end_to_end() -> None:
    def respond(state: str, questions: dict[str, Any]) -> SimpleNamespace:
        criteria = questions[QUESTION_NAME]["criteria"]
        if "locate" in criteria:
            return jev_response("choices", choice_answer(ROUTE_OUTSIDE, {r: 0.2 for r in criteria}))
        return jev_response("choices", choice_answer("B", {"A": 0.3, "B": 0.5, "C": 0.15, NONE_LABEL: 0.05}))

    jev = JevDecisions(client=FakeClient(respond))
    routed = route_message(jev, "how does werkzeug parse multipart bodies?", [])
    assert routed.route == ROUTE_OUTSIDE and routed.cost_usd == 5.04e-06
    reranked = rerank_cards(jev, "where?", [card(0), card(1), card(2)])
    assert reranked.order == [1, 0, 2] and reranked.none_selected is False
    assert jev.usage.calls == 2


class StatusError(Exception):
    def __init__(self, status: int, message: str = "boom") -> None:
        super().__init__(f"{status} {message}")
        self.status = status


@pytest.mark.parametrize("error, config, text", [
    (StatusError(401, "No auth credentials found"), True, "OPENROUTER_API_KEY or TYPESAFE_API_KEY"),
    (StatusError(403), True, "rejected the API key"),
    (StatusError(402, "Insufficient credits"), True, "no credit"),
    (StatusError(404), True, "model id"),
    (StatusError(429), False, "rate limit"),
    (StatusError(500), False, "API error 500"),
    (StatusError(200, "Invalid response data at 'answers'"), False, "could not read"),
    (ConnectionError("connection refused"), False, "could not reach Jev"),
    (TimeoutError("read timed out"), False, "did not answer within 60s"),
    (RuntimeError("weird"), False, "RuntimeError: weird"),
])
def test_jev_errors_are_wrapped(error: BaseException, config: bool, text: str) -> None:
    jev = JevDecisions(client=FakeClient(error=error))
    with pytest.raises(DecisionsError) as info:
        jev.choose("s", "pick", OPTIONS)
    assert isinstance(info.value, DecisionsConfigError) is config
    assert text in str(info.value)
    assert info.value.__cause__ is error
    assert jev.usage.calls == 0


# --------------------------------------------------------------------------- #
# Jev client construction and detection
# --------------------------------------------------------------------------- #

ENV_NAMES = (dec.ENV_DECISIONS, dec.OPENROUTER_KEY_ENV, dec.TYPESAFE_KEY_ENV,
             "TYPESAFE_BASE_URL", "TYPESAFE_DEFAULT_MODEL")


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> pytest.MonkeyPatch:
    """No decision variables, and an empty working directory (no .env). Setting
    before deleting makes monkeypatch remove anything a .env load adds."""
    for name in ENV_NAMES:
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    return monkeypatch


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """A stand-in ``typesafe_sdk`` module; returns the TypeSafeClient kwargs."""
    created: list[dict[str, Any]] = []

    class RetryPolicy:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class TypeSafeClient:
        def __init__(self, **kwargs: Any) -> None:
            created.append(kwargs)

        def system_one(self, state: str, questions: dict[str, Any]) -> Any:
            return jev_response("choices", choice_answer("a", {"a": 1.0}))

    module = types.ModuleType("typesafe_sdk")
    module.RetryPolicy = RetryPolicy  # type: ignore[attr-defined]
    module.TypeSafeClient = TypeSafeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "typesafe_sdk", module)
    return created


def test_jev_client_through_openrouter_from_the_environment(clean_env: pytest.MonkeyPatch,
                                                            fake_sdk: list[dict[str, Any]]) -> None:
    clean_env.setenv(dec.OPENROUTER_KEY_ENV, " sk-or-v1-abc ")
    clean_env.setenv(dec.TYPESAFE_KEY_ENV, "ts-key")                  # OpenRouter wins
    jev = JevDecisions()
    assert fake_sdk == [] and jev.route is None                        # nothing happens before a decision
    assert jev.choose("s", "pick", {"a": "A"}).label == "a"
    (kwargs,) = fake_sdk
    assert kwargs["api_key"] == "sk-or-v1-abc"
    assert kwargs["base_url"] == dec.OPENROUTER_BASE_URL and kwargs["model"] == dec.OPENROUTER_MODEL
    assert kwargs["timeout"] == 60.0 and kwargs["retry"].kwargs == {"max_retries": 2, "timeout": 120.0}
    assert (jev.route, jev.model) == ("openrouter", dec.OPENROUTER_MODEL)


def test_jev_client_through_typesafe_keeps_the_sdk_defaults(clean_env: pytest.MonkeyPatch,
                                                            fake_sdk: list[dict[str, Any]]) -> None:
    clean_env.setenv(dec.TYPESAFE_KEY_ENV, "ts-key")
    jev = JevDecisions(timeout=5)
    jev.choose("s", "pick", {"a": "A"})
    (kwargs,) = fake_sdk
    assert kwargs["api_key"] == "ts-key" and "base_url" not in kwargs and "model" not in kwargs
    assert kwargs["timeout"] == 5.0 and jev.route == "typesafe"


@pytest.mark.parametrize("api_key, base_url, model, expect_base, expect_model, route", [
    ("sk-or-v1-explicit", None, None, dec.OPENROUTER_BASE_URL, dec.OPENROUTER_MODEL, "openrouter"),
    ("sk-or-v1-explicit", None, "~typesafe/jev-1", dec.OPENROUTER_BASE_URL, "~typesafe/jev-1", "openrouter"),
    ("ts-explicit", None, None, None, None, "typesafe"),
    ("ts-explicit", "https://openrouter.ai/api", None, "https://openrouter.ai/api", dec.OPENROUTER_MODEL,
     "openrouter"),
    ("sk-or-v1-explicit", "https://example.test", "jev-x", "https://example.test", "jev-x", "typesafe"),
])
def test_jev_client_with_explicit_access(clean_env: pytest.MonkeyPatch, fake_sdk: list[dict[str, Any]],
                                         api_key: str, base_url: Optional[str], model: Optional[str],
                                         expect_base: Optional[str], expect_model: Optional[str],
                                         route: str) -> None:
    jev = JevDecisions(api_key=api_key, base_url=base_url, model=model)
    jev.choose("s", "pick", {"a": "A"})
    (kwargs,) = fake_sdk
    assert kwargs["api_key"] == api_key
    assert kwargs.get("base_url") == expect_base and kwargs.get("model") == expect_model
    assert jev.route == route


def test_jev_without_a_key_explains_the_setup(clean_env: pytest.MonkeyPatch,
                                              fake_sdk: list[dict[str, Any]]) -> None:
    with pytest.raises(DecisionsConfigError) as info:
        JevDecisions().choose("s", "pick", {"a": "A"})
    message = str(info.value)
    assert "OPENROUTER_API_KEY" in message and "TYPESAFE_API_KEY" in message and ".env" in message
    assert "pip install" not in message                                # the sdk is a core dependency now
    assert fake_sdk == []


def test_jev_without_the_sdk_explains_the_install(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setitem(sys.modules, "typesafe_sdk", None)               # makes `import typesafe_sdk` fail
    clean_env.setenv(dec.OPENROUTER_KEY_ENV, "sk-or-v1-abc")
    with pytest.raises(DecisionsConfigError, match="typesafe-sdk is not installed") as info:
        JevDecisions().choose("s", "pick", {"a": "A"})
    assert "reinstall RepoFinder" in str(info.value) and "[jev]" not in str(info.value)


def test_jev_client_creation_errors_are_config_errors(clean_env: pytest.MonkeyPatch,
                                                      fake_sdk: list[dict[str, Any]]) -> None:
    def broken(**kwargs: Any) -> None:
        raise ValueError("API key must contain only printable ASCII characters without whitespace.")

    sys.modules["typesafe_sdk"].TypeSafeClient = broken  # type: ignore[union-attr]
    with pytest.raises(DecisionsConfigError, match="could not create the Jev client"):
        JevDecisions(api_key="bad key").choose("s", "pick", {"a": "A"})


@pytest.mark.parametrize("preferred, env, expected", [
    (None, None, RulesDecisions),
    (None, "rules", RulesDecisions),
    (None, "off", RulesDecisions),
    (None, " OFF ", RulesDecisions),
    ("rules", "jev", RulesDecisions),                                  # preferred wins over the environment
])
def test_detect_decisions_rules(clean_env: pytest.MonkeyPatch, preferred: Optional[str],
                                env: Optional[str], expected: type) -> None:
    if env is not None:
        clean_env.setenv(dec.ENV_DECISIONS, env)
    assert type(detect_decisions(preferred)) is expected


def test_detect_decisions_jev(clean_env: pytest.MonkeyPatch, fake_sdk: list[dict[str, Any]]) -> None:
    clean_env.setenv(dec.ENV_DECISIONS, "jev")
    clean_env.setenv(dec.TYPESAFE_KEY_ENV, "ts-key")
    backend = detect_decisions()
    assert isinstance(backend, JevDecisions) and backend.name == "jev"
    assert fake_sdk == []                                              # the client is still created lazily


@pytest.mark.parametrize("env", [None, "auto", " AUTO "])
def test_detect_decisions_auto_with_key_and_sdk(clean_env: pytest.MonkeyPatch, fake_sdk: list[dict[str, Any]],
                                                env: Optional[str]) -> None:
    if env is not None:
        clean_env.setenv(dec.ENV_DECISIONS, env)
    clean_env.setenv(dec.OPENROUTER_KEY_ENV, "sk-or-v1-abc")
    backend = detect_decisions()
    assert isinstance(backend, JevDecisions) and backend.name == "jev"


@pytest.mark.parametrize("env", [None, "auto"])
def test_detect_decisions_auto_without_a_key_uses_rules(clean_env: pytest.MonkeyPatch,
                                                        fake_sdk: list[dict[str, Any]],
                                                        env: Optional[str]) -> None:
    if env is not None:
        clean_env.setenv(dec.ENV_DECISIONS, env)
    assert type(detect_decisions()) is RulesDecisions


@pytest.mark.parametrize("env", [None, "auto"])
def test_detect_decisions_auto_without_the_sdk_uses_rules(clean_env: pytest.MonkeyPatch, env: Optional[str]) -> None:
    if env is not None:
        clean_env.setenv(dec.ENV_DECISIONS, env)
    clean_env.setenv(dec.TYPESAFE_KEY_ENV, "ts-key")
    clean_env.setitem(sys.modules, "typesafe_sdk", None)
    assert type(detect_decisions()) is RulesDecisions


def test_detect_decisions_rules_wins_over_a_key(clean_env: pytest.MonkeyPatch, fake_sdk: list[dict[str, Any]]) -> None:
    clean_env.setenv(dec.OPENROUTER_KEY_ENV, "sk-or-v1-abc")
    clean_env.setenv(dec.ENV_DECISIONS, "rules")
    assert type(detect_decisions()) is RulesDecisions


def test_detect_decisions_jev_without_a_key(clean_env: pytest.MonkeyPatch) -> None:
    with pytest.raises(DecisionsConfigError) as info:
        detect_decisions("jev")
    message = str(info.value)
    assert "REPOFINDER_DECISIONS=jev" in message
    assert "neither OPENROUTER_API_KEY nor TYPESAFE_API_KEY is set" in message
    assert ".env" in message and "pip install" not in message


def test_detect_decisions_jev_without_the_sdk(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv(dec.OPENROUTER_KEY_ENV, "sk-or-v1-abc")
    clean_env.setitem(sys.modules, "typesafe_sdk", None)
    with pytest.raises(DecisionsConfigError, match="typesafe-sdk is not installed"):
        detect_decisions("jev")


def test_detect_decisions_unknown_name(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv(dec.ENV_DECISIONS, "haiku")
    with pytest.raises(DecisionsConfigError, match="Unknown decisions backend 'haiku'"):
        detect_decisions()


def test_detect_decisions_reads_a_dotenv_file(clean_env: pytest.MonkeyPatch, tmp_path: Path,
                                              fake_sdk: list[dict[str, Any]]) -> None:
    (tmp_path / ".env").write_text("REPOFINDER_DECISIONS=jev\nOPENROUTER_API_KEY=sk-or-v1-dotenv\n",
                                   encoding="utf-8")
    backend = detect_decisions()
    assert isinstance(backend, JevDecisions)
    backend.choose("s", "pick", {"a": "A"})
    assert fake_sdk[0]["api_key"] == "sk-or-v1-dotenv"


# --------------------------------------------------------------------------- #
# Jev through the real typesafe-sdk, with a mock HTTP transport
# --------------------------------------------------------------------------- #

def real_client(handler: Callable[[Any], Any]) -> Any:
    sdk = pytest.importorskip("typesafe_sdk")
    httpx2 = pytest.importorskip("httpx2")
    return sdk.TypeSafeClient(api_key="sk-or-v1-test", base_url=dec.OPENROUTER_BASE_URL,
                              model=dec.OPENROUTER_MODEL, transport=httpx2.MockTransport(handler),
                              retry=sdk.RetryPolicy(max_retries=0))


def test_jev_with_the_real_sdk_reads_choice_noul_usage_and_cost() -> None:
    httpx2 = pytest.importorskip("httpx2")
    requests: list[dict[str, Any]] = []

    def handler(request: Any) -> Any:
        body = json.loads(request.content)
        requests.append({"url": str(request.url), "body": body})
        question = body["questions"][QUESTION_NAME]
        if question["type"] == "choice":
            answer = {"type": "choice", "choice": "B", "confidence": 0.7,
                      "probabilities": {"A": 0.2, "B": 0.7, "NONE": 0.1}}
        else:
            answer = {"type": "noul", "noul": 0.83}
        return httpx2.Response(200, json={"model": "jev-1", "answers": {QUESTION_NAME: answer},
                                          "usage": {"input_tokens": 321, "output_tokens": 2, "cost": 1.3e-05}})

    jev = JevDecisions(client=real_client(handler))
    reranked = rerank_cards(jev, "where is func_1?", [card(0), card(1)])
    assert reranked.order == [1, 0] and reranked.probabilities == {"A": 0.2, "B": 0.7, "NONE": 0.1}
    assert reranked.cost_usd == 1.3e-05
    answer = context_suffices(jev, "and then?", "where is func_1?", [card(1)])
    assert answer == YesNo(0.83, 1.3e-05, 321)
    assert [r["url"] for r in requests] == ["https://openrouter.ai/api/v1/systemone"] * 2
    assert requests[0]["body"]["model"] == dec.OPENROUTER_MODEL
    assert requests[0]["body"]["questions"][QUESTION_NAME]["criteria"]["B"] == "src/pkg/mod1.py:10-19 | function func_1"
    assert requests[1]["body"]["questions"] == {QUESTION_NAME: {"type": "noul", "instructions": CONTEXT_QUESTION}}
    assert (jev.usage.calls, jev.usage.input_tokens) == (2, 642)


@pytest.mark.parametrize("status, config, text", [
    (401, True, "rejected the API key (401)"),
    (402, True, "payment required (402)"),
    (400, False, "API error 400"),
])
def test_jev_with_the_real_sdk_maps_http_errors(status: int, config: bool, text: str) -> None:
    httpx2 = pytest.importorskip("httpx2")
    client = real_client(lambda request: httpx2.Response(status, json={"error": {"message": "nope"}}))
    with pytest.raises(DecisionsError) as info:
        JevDecisions(client=client).yes_no("s", "q?")
    assert isinstance(info.value, DecisionsConfigError) is config
    assert text in str(info.value) and "nope" in str(info.value) and "sk-or-v1-test" not in str(info.value)


def test_jev_with_the_real_sdk_maps_connection_errors() -> None:
    httpx2 = pytest.importorskip("httpx2")

    def handler(request: Any) -> Any:
        raise httpx2.ConnectError("connection refused", request=request)

    with pytest.raises(DecisionsError, match="could not reach Jev") as info:
        JevDecisions(client=real_client(handler)).yes_no("s", "q?")
    assert not isinstance(info.value, DecisionsConfigError)


def test_jev_builds_a_real_client_without_network(clean_env: pytest.MonkeyPatch) -> None:
    sdk = pytest.importorskip("typesafe_sdk")
    jev = JevDecisions(api_key="sk-or-v1-test")
    assert isinstance(jev._get_client(), sdk.TypeSafeClient)
    assert (jev.route, jev.model) == ("openrouter", dec.OPENROUTER_MODEL)


# --------------------------------------------------------------------------- #
# Import cost
# --------------------------------------------------------------------------- #

def test_importing_and_using_rules_does_not_import_the_sdks() -> None:
    code = (
        "import sys, repofinder.decisions as d\n"
        "d.route_message(d.RulesDecisions(), 'where is x?', [])\n"
        "print('typesafe_sdk' in sys.modules, 'anthropic' in sys.modules)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["False", "False"]

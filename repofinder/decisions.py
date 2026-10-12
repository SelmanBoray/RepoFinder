"""Decisions for chat mode: "Jev decides, the LLM talks".

Every chat turn needs a few small decisions before any reply is written: what
kind of message this is (one of :data:`ROUTES`), which search candidate answers
a question (or none of them), and whether the excerpts shown with the previous
answer already cover a follow-up. These are classifications, not prose, so they
go to a backend that returns a label with probabilities (:class:`Choice`) or
the probability of "yes" (:class:`YesNo`). The language model only writes the
reply, from what these decisions selected.

Three backends implement :class:`Decisions`:

* :class:`RulesDecisions`: deterministic English and Turkish
  heuristics, no network, no cost. It never answers ``outside`` (rules cannot
  know what a repository delegates elsewhere) and never reorders search hits.
* :class:`JevDecisions`: TypeSafe's "System One" model Jev, through the
  optional ``typesafe-sdk`` package (``pip install "repofinder[jev]"``). Jev
  returns typed answers (a choice with per-option probabilities, a yes/no
  probability), not text. Input costs about $0.042 per million tokens, output
  is free, and one request (state plus questions) must fit in 32k tokens.
* :class:`FakeDecisions`: scripted answers for tests.

:func:`detect_decisions` picks one from ``REPOFINDER_DECISIONS``. The default
(unset, or ``auto``) is Jev when an API key is set and ``typesafe-sdk`` is
installed, and the local rules otherwise, without any error. Set ``rules`` or
``off`` to force the local rules even with a key present, or ``jev`` to
require Jev (an error when the key or the sdk is missing). Jev needs ``OPENROUTER_API_KEY``
(through OpenRouter) or ``TYPESAFE_API_KEY`` (TypeSafe's own API).

The chat layer calls :func:`route_message`, :func:`rerank_cards` and
:func:`context_suffices`, which work with any backend: they render the state
text, ask the backend, and check its answer. ``RulesDecisions`` is the one
special case: rules read the structured inputs (the message, the history, the
cards) rather than the rendered state, so the helpers apply them directly.
Backend failures raise :class:`DecisionsError`; a caller that wants to carry
on can repeat the decision with ``RulesDecisions()``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Optional, Protocol, Sequence, Union, runtime_checkable

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Routes and configuration
# --------------------------------------------------------------------------- #

ROUTE_LOCATE = "locate"      # "where is X / which function does Y" -> search + answer with citations
ROUTE_EXPLAIN = "explain"    # "how does X work / what happens when" -> search + answer
ROUTE_FOLLOWUP = "followup"  # refers to the previous answer ("who calls it?", "peki bunu kim çağırıyor?")
ROUTE_CHAT = "chat"          # greetings, thanks, meta questions about RepoFinder, small talk -> no search
ROUTE_OUTSIDE = "outside"    # asks about something this repository delegates elsewhere -> no search, say so
ROUTES = (ROUTE_LOCATE, ROUTE_EXPLAIN, ROUTE_FOLLOWUP, ROUTE_CHAT, ROUTE_OUTSIDE)

ENV_DECISIONS = "REPOFINDER_DECISIONS"          # "auto" (default) | "jev" | "rules" | "off"  (off == rules)
DECISIONS_AUTO = "auto"
DECISIONS_JEV = "jev"
DECISIONS_RULES = "rules"
DECISIONS_OFF = "off"

# Jev access. OPENROUTER_API_KEY wins over TYPESAFE_API_KEY.
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
TYPESAFE_KEY_ENV = "TYPESAFE_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api"
OPENROUTER_HOST = "openrouter.ai"
OPENROUTER_MODEL = "~typesafe/jev-latest"
OPENROUTER_KEY_PREFIX = "sk-or-"     # an explicit api_key with this prefix goes through OpenRouter
JEV_TIMEOUT_S = 60.0
JEV_MAX_RETRIES = 2

# Budget and cost. Tokens are estimated as characters / 4; code often has fewer
# characters per token, so a request is kept to about 70% of Jev's context.
CONTEXT_TOKENS = 32_000
CHARS_PER_TOKEN = 4
MAX_REQUEST_TOKENS = 22_000
# USD per million input tokens, as given for Jev on 2026-10-09 (not verified
# against a live price list). Output tokens are free. Used only when a
# response does not report ``usage.cost`` itself.
PRICE_PER_M_INPUT_USD = 0.042
CACHE_SIZE = 256                     # JevDecisions answers kept in memory

# Shapes of the helper requests.
QUESTION_NAME = "decision"           # the one question name in every Jev request
MAX_CARDS = 16
CARD_LABELS = tuple("ABCDEFGHIJKLMNOP")
NONE_LABEL = "NONE"
HISTORY_EXCHANGES = 3                # user/assistant pairs shown to the router
HISTORY_TEXT_CHARS = 600             # per history message
MESSAGE_CHARS = 2_000                # the new message / the question
OPTION_CHARS = 200                   # a card's first line used as its option description

# Rule constants.
FOLLOWUP_MAX_WORDS = 6
SOCIAL_MAX_EXTRA_WORDS = 3
SUFFICES_YES = 0.8
SUFFICES_NO = 0.3

JEV_SETUP_HELP = """\
Jev (TypeSafe's decision model) can make RepoFinder's chat decisions. It needs:

  1. The typesafe-sdk package:   pip install "repofinder[jev]"   (or: pip install typesafe-sdk)
  2. An API key, one of
       OPENROUTER_API_KEY=sk-or-...   through OpenRouter (model ~typesafe/jev-latest)
       TYPESAFE_API_KEY=...           TypeSafe's own API (api.typesafe.ai)
     in the environment or in a .env file in the directory you run RepoFinder from.

By default (REPOFINDER_DECISIONS unset or "auto") Jev is used when a key and the sdk are
present, and the built-in rules otherwise. Set REPOFINDER_DECISIONS=rules (or off) to force
the rules, which need neither, even with a key present; REPOFINDER_DECISIONS=jev requires
Jev and fails when the key or the sdk is missing."""

ROUTE_INSTRUCTIONS = (
    "Classify the new user message in this conversation with RepoFinder, an assistant that answers"
    " questions about one code repository by searching its source code. Pick how it should be handled."
)
ROUTE_OPTIONS: dict[str, str] = {
    ROUTE_LOCATE: "The user asks where something is in the repository's code: which file, class,"
                  " function or method defines or implements a feature, or where a named identifier is.",
    ROUTE_EXPLAIN: "The user asks how something in the repository's code works, why it behaves a certain"
                   " way, or what happens when something runs.",
    ROUTE_FOLLOWUP: "The message continues the previous answer: it refers to something just discussed"
                    " (it, this, that function, who calls it, and the tests?) and cannot be understood"
                    " without the conversation.",
    ROUTE_CHAT: "A greeting, thanks, small talk, or a question about RepoFinder itself (what it can do,"
                " how it works); no code search is needed.",
    ROUTE_OUTSIDE: "The user asks about code or behaviour that this repository does not contain because it"
                   " is delegated to another library, service or repository, so searching this repository"
                   " cannot answer it.",
}
RERANK_INSTRUCTIONS = (
    "Which candidate contains the code that answers the question?"
    " Pick the one whose lines a reader should open first."
)
RERANK_NONE_INSTRUCTIONS = " Pick NONE when no candidate contains such code."
NONE_DESCRIPTION = "None of the candidates contains code that answers the question."
CONTEXT_QUESTION = (
    "Do the code excerpts above already contain everything needed to answer the follow-up message,"
    " so that no new code search is needed?"
)


# --------------------------------------------------------------------------- #
# Errors and data types
# --------------------------------------------------------------------------- #

class DecisionsError(RuntimeError):
    """A decision could not be made: the backend failed or gave an unusable answer."""


class DecisionsConfigError(DecisionsError):
    """A failure every further call would repeat: no API key, a rejected key, no
    credit, a missing ``typesafe-sdk``, an unknown model or backend name."""


@dataclass
class Choice:
    label: str                       # chosen option key
    probabilities: dict[str, float]  # option key -> probability (sums ~1); rules backend gives 1.0/0.0
    confidence: Optional[float]      # None when the backend has none
    cost_usd: float                  # 0.0 for rules/fake
    input_tokens: int                # 0 for rules/fake


@dataclass
class YesNo:
    probability: float               # P(yes)
    cost_usd: float
    input_tokens: int


@runtime_checkable
class Decisions(Protocol):
    name: str                        # "jev" | "rules" | "fake"

    def choose(self, state: str, instructions: str, options: dict[str, str]) -> Choice: ...

    def yes_no(self, state: str, question: str) -> YesNo: ...


@dataclass
class DecisionUsage:
    """Running totals over the decisions one backend made (like ``llm.Usage``).
    Cached Jev answers are not counted: they cost nothing."""
    calls: int = 0
    input_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class RouteResult:
    route: str                       # one of ROUTES
    probabilities: dict[str, float]
    cost_usd: float


@dataclass
class RerankResult:
    order: list[int]                 # indexes into `cards`, best first (all of them)
    none_selected: bool              # the backend picked "none of these"
    probabilities: dict[str, float]  # label -> p, labels "A".."P" and "NONE"
    cost_usd: float


class DecisionCall(NamedTuple):
    """One call recorded by :class:`FakeDecisions`."""
    kind: str                        # "choose" | "yes_no"
    state: str
    prompt: str                      # the instructions (choose) or the question (yes_no)
    options: Optional[dict[str, str]]


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
#
# Messages are matched lower-cased and ASCII-folded (ç->c, ğ->g, ı->i, ö->o,
# ş->s, ü->u), so Turkish typed without its letters ("nasil", "nerede") matches
# too; the patterns below are written in that folded form. Identifiers are
# found in the original text (case matters for camelCase).

_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def _norm(text: str) -> str:
    return text.replace("İ", "i").lower().translate(_FOLD)


def _rx(*patterns: str) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in patterns))


# "where is X", "which file defines Y", "nerede tanımlı", "hangi sınıf".
_LOCATE_CUES = _rx(
    r"\bwhere\b", r"\bin which\b", r"\bfind\b", r"\blook(?:ing)? for\b", r"\bdefinition of\b",
    r"\b(?:which|what) (?:file|class|function|method|module|package|line|part)s?\b",
    r"\b(?:defined|implemented|located|declared)\b",
    r"\bnere", r"\bhangi (?:dosya|sinif|fonksiyon|metot|metod|modul|paket|satir|kisim)",
    r"\btanim", r"\byazil(?:mis|di|dig|iyor)", r"\bbulun", r"\bkonum",
)
# "how does X work", "why", "what happens", "nasıl", "neden", "ne işe yarar".
_EXPLAIN_CUES = _rx(
    r"\bhow\b", r"\bwhy\b", r"\bwhat happens\b", r"\bexplain\b", r"\bwalk me through\b",
    r"\bwhat (?:does|do|is|are)\b.*\b(?:do|for|mean)\b", r"\bpurpose\b",
    r"\bnasil\b", r"\bneden\b", r"\bniye\b", r"\bnicin\b",
    r"\bne (?:oluyor|olur|yapar|yapiyor|ise yarar|demek)\b", r"\bacikla(?!ma)", r"\banlat",
)
# Questions to RepoFinder itself.
_META_CUES = _rx(
    r"\bwhat can you do\b", r"\bwhat do you do\b", r"\bwho are you\b", r"\bwhat are you\b",
    r"\bhow do you work\b", r"\bhow does repofinder work\b", r"\bwhat is repofinder\b",
    r"^\W*(?:help|yardim)\W*$",
    r"\bne yapabilirsin\b", r"\bne yaparsin\b", r"\bkimsin\b", r"\bnesin\b",
    r"\bnasil calisiyorsun\b", r"\brepofinder (?:nedir|nasil calisir)\b",
)
# Greetings, thanks, acknowledgements.
_SOCIAL_CUES = _rx(
    r"\b(?:hi|hello|hey|hiya|howdy|greetings)\b", r"\bgood (?:morning|afternoon|evening|night)\b",
    r"\bthanks?\b", r"\bthank you\b", r"\bthx\b", r"\bcheers\b", r"\b(?:good)?bye\b", r"\bsee you\b",
    r"\b(?:ok|okay|cool|great|nice|perfect|awesome|alright)\b", r"\bgot it\b", r"\bi see\b",
    r"\bmakes sense\b", r"\bsounds good\b",
    r"\bmerhaba\b", r"\bselam\b", r"\bgunaydin\b", r"\biyi (?:aksamlar|geceler|gunler)\b",
    r"\btesekkur", r"\bsag ?ol", r"\beyvallah\b", r"\btamam\b", r"\banladim\b", r"\bsuper\b",
    r"\bharika\b", r"\bgorusuruz\b", r"\bhosca kal\b",
)
# "this repo", "bu proje": the repository itself, not something just discussed.
_NOT_REPO_EN = r"(?!\s+(?:repo|repository|project|codebase|code ?base|library|package)\b)"
_NOT_REPO_TR = r"(?!\s+(?:repo|proje|depo|kod|kutuphane|paket))"
# A message that opens like a continuation ("and the tests?", "peki ya ...").
_FOLLOWUP_OPENERS = re.compile(
    r"^\W*(?:(?:ok|okay|thanks|tamam|peki)\W+)?(?:"
    r"(?:it|its|that|these|those|they|them|and|also|so|then|but|or|plus|what about|how about"
    r"|peki|bunu|bunun|buna|bunda|sunu|o|onu|onun|ona|onda|onlar\w*|ve|ayrica|ya|bir de|hani|sonra)\b"
    r"|this\b" + _NOT_REPO_EN + r"|(?:bu|su)\b" + _NOT_REPO_TR + r")"
)
# Words that point back at something already discussed.
_ANAPHORA = _rx(
    r"\b(?:it|its|them|they|those|these|this one|that one|the same|above|previous|earlier)\b",
    r"\bthis\b" + _NOT_REPO_EN,
    r"\b(?:bunu|bunun|buna|bunda|bundan|bunlar\w*|onu|onun|ona|onda|ondan|onlar\w*|sunu|o"
    r"|ayni\w*|yukaridaki|onceki)\b",
    r"\b(?:bu|su)\b" + _NOT_REPO_TR,
)
# Words that make a message without "?" still about code.
_CODE_WORDS = _rx(
    r"\b(?:function|method|class|module|file|code|variable|parameter|param|argument|arg|return|call"
    r"|caller|import|decorator|test|exception|error|config|configuration|setting|route|routing|request"
    r"|response|handler|endpoint|api|cli|command|package|library|implementation|signature|attribute"
    r"|property|constant|script|object|instance|thread|cache|session|cookie|header|template|database"
    r"|query|schema|model|view|middleware|plugin|extension|hook|callback|event|signal|logger|logging"
    r"|parser|token|url|path)(?:s|es|ed|ing)?\b",
    r"\b(?:fonksiyon|metot|metod|sinif|modul|dosya|kod|degisken|parametre|arguman|cagir|cagri|dondur"
    r"|donus|hata|istisna|test|ayar|yapilandir|istek|yanit|komut|paket|kutuphane|dekorator|oturum|cerez"
    r"|baslik|sablon|veritaban|sorgu|olay|sinyal|onbellek|rota|yonlendir|isleyici)\w*",
)
# Function words and question words: what is left after removing them are the
# content words ("why?" has none, "who calls it?" has "calls").
_STOP = frozenset(
    """
    where why how what which who whom whose when is are was were be been being do does did done the a an
    of in on at to for from by with about and or but so then it its this that these those they them there
    here defined define implemented implement located declared happen happens happened work works working
    mean means meaning exactly again please me you we my can could would should will tell show explain
    find look ll ve re
    nerede nereye neresi nereden nerde neden niye nicin nasil ne mi mu hangi hangisi kim kimi kimler bu su
    bunu onu bunun onun buna ona ve ya peki ile icin de da ki tanimli tanimlanmis tanimlandi tanimi
    yazilmis yazildi oluyor olur yapar yapiyor calisiyor calisir ise yarar acikla anlat bana sen
    """.split()
)
_PY_COMMON = frozenset(
    """
    self cls none true false def class return import from if elif else for while in not and or is with as
    try except finally raise pass lambda yield async await break continue global nonlocal del assert
    print len str int dict list set tuple object type super isinstance getattr setattr hasattr kwargs args
    """.split()
)

_WORD_RE = re.compile(r"[a-z0-9_]+")
_TERM_RE = re.compile(r"\w+(?:\.\w+)*")
_BACKTICK_RE = re.compile(r"`([^`\n]+)`")
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_SNAKE_RE = re.compile(r"\b_*[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]*[A-Za-z0-9]\b")
_DUNDER_RE = re.compile(r"\b__[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*__\b")
_DOTTED_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")
_CAMEL_RE = re.compile(r"\b[a-z]+[A-Z][A-Za-z0-9]*\b|\b[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*\b")
_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\(\)")
_DEF_RE = re.compile(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)")


def _identifiers(text: str) -> list[str]:
    """Code identifiers named in ``text``: anything in backticks, snake_case,
    dunder, dotted (``Cls.meth``, ``app.py``; not "e.g."), camelCase or
    PascalCase with two humps, and ``name()``."""
    found: list[str] = []
    for span in _BACKTICK_RE.findall(text):
        found += _NAME_RE.findall(span)
    found += _SNAKE_RE.findall(text) + _DUNDER_RE.findall(text) + _CAMEL_RE.findall(text)
    found += _CALL_RE.findall(text)
    found += [d for d in _DOTTED_RE.findall(text) if all(len(p) >= 2 for p in d.split("."))]
    return found


def _content_words(norm: str) -> list[str]:
    return [w for w in _WORD_RE.findall(norm) if len(w) >= 2 and w not in _STOP]


def rule_route(message: str, history: Sequence[tuple[str, str]]) -> str:
    """The rule-based route of ``message``; never :data:`ROUTE_OUTSIDE`.

    First match wins:

    1. a blank message: chat;
    2. a question to RepoFinder itself ("what can you do", "nasıl
       çalışıyorsun"), without an identifier or a location word: chat;
    3. greetings, thanks, acknowledgements with at most a few other words,
       no "?", no identifier, no location, explanation or code word: chat;
    4. with history and without an identifier: a message that opens like a
       continuation ("and", "what about", "peki", "ya", ...), points back
       ("it", "this", "bunu", "onu", ...), is short (<= 6 words) with "?" or a
       code word but no location or explanation word, or has no content words
       at all ("why?", "nerede tanımlı?"): followup;
    5. a location word (where, which file/class/function, defined,
       implemented, nerede, hangi dosya, tanımlı, yazılmış, ...): locate;
    6. an explanation word (how, why, what happens, nasıl, neden, ne oluyor,
       ...): explain;
    7. an identifier (backticks, snake_case, ``Cls.meth``, camelCase): locate;
    8. a "?" or a code word (function, class, config, fonksiyon, dosya, ...):
       locate;
    9. anything else: chat.
    """
    if not message or not message.strip():
        return ROUTE_CHAT
    norm = _norm(message)
    has_ident = bool(_identifiers(message))
    locate = bool(_LOCATE_CUES.search(norm))
    explain = bool(_EXPLAIN_CUES.search(norm))
    code_word = bool(_CODE_WORDS.search(norm))
    question = "?" in message
    n_words = len(message.split())

    if _META_CUES.search(norm) and not has_ident and not locate:
        return ROUTE_CHAT
    if _SOCIAL_CUES.search(norm) and not (has_ident or locate or explain or code_word or question):
        if len(_WORD_RE.findall(_SOCIAL_CUES.sub(" ", norm))) <= SOCIAL_MAX_EXTRA_WORDS:
            return ROUTE_CHAT
    if history and not has_ident:
        if (_FOLLOWUP_OPENERS.search(norm) or _ANAPHORA.search(norm)
                or (n_words <= FOLLOWUP_MAX_WORDS and (question or code_word) and not locate and not explain)
                or not _content_words(norm)):
            return ROUTE_FOLLOWUP
    if locate:
        return ROUTE_LOCATE
    if explain:
        return ROUTE_EXPLAIN
    if has_ident or question or code_word:
        return ROUTE_LOCATE
    return ROUTE_CHAT


def _card_identifiers(cards: Sequence[str]) -> set[str]:
    """Lower-cased identifiers in the cards: ``def``/``class`` names and every
    identifier :func:`_identifiers` finds (dotted ones also by their parts),
    without Python keywords, common builtins and names under 3 characters."""
    names: set[str] = set()
    for card in cards:
        names.update(n.lower() for n in _DEF_RE.findall(card))
        for ident in _identifiers(card):
            low = ident.lower()
            names.add(low)
            names.update(low.split("."))
    return {n for n in names if len(n) >= 3 and n not in _PY_COMMON}


def _shares_word(a: set[str], b: set[str]) -> bool:
    """A word in both, or one a prefix of the other with the shorter at least 4
    characters long ("session"/"sessions", "oturum"/"oturumu")."""
    if a & b:
        return True
    return any(min(len(x), len(y)) >= 4 and (x.startswith(y) or y.startswith(x)) for x in a for y in b)


def rule_context_suffices(message: str, previous_question: str, cards: Sequence[str]) -> float:
    """:data:`SUFFICES_YES` (0.8) when ``message`` shares a content word with
    ``previous_question`` and names an identifier that appears in ``cards``;
    :data:`SUFFICES_NO` (0.3) otherwise. The threshold belongs to the caller."""
    msg_words = {w for w in _content_words(_norm(message)) if len(w) >= 3}
    prev_words = {w for w in _content_words(_norm(previous_question or "")) if len(w) >= 3}
    if not _shares_word(msg_words, prev_words):
        return SUFFICES_NO
    known = _card_identifiers(cards)
    terms: set[str] = set()
    for term in _TERM_RE.findall(message):
        low = term.lower()
        terms.add(low)
        terms.update(low.split("."))
    return SUFFICES_YES if terms & known else SUFFICES_NO


class RulesDecisions:
    """Deterministic English and Turkish rules; no network, no cost.

    The helpers (:func:`route_message`, :func:`rerank_cards`,
    :func:`context_suffices`) apply :func:`rule_route`, the search order and
    :func:`rule_context_suffices` through :meth:`route`, :meth:`rerank` and
    :meth:`suffices`. :meth:`choose` and :meth:`yes_no` on free text have no
    rule to apply: they return the first option and 0.5 (undecided).
    """

    name = DECISIONS_RULES

    def __init__(self) -> None:
        self.usage = DecisionUsage()

    def choose(self, state: str, instructions: str, options: dict[str, str]) -> Choice:
        if not options:
            raise ValueError("options must not be empty")
        return self._choice(next(iter(options)), options)

    def yes_no(self, state: str, question: str) -> YesNo:
        self.usage.calls += 1
        return YesNo(0.5, 0.0, 0)

    def route(self, message: str, history: Sequence[tuple[str, str]]) -> Choice:
        return self._choice(rule_route(message, history), ROUTES)

    def rerank(self, labels: Sequence[str]) -> Choice:
        """The first label: the search order stays as it is."""
        return self._choice(labels[0], labels)

    def suffices(self, message: str, previous_question: str, cards: Sequence[str]) -> YesNo:
        self.usage.calls += 1
        return YesNo(rule_context_suffices(message, previous_question, cards), 0.0, 0)

    def _choice(self, label: str, keys: Sequence[str] | Mapping[str, str]) -> Choice:
        self.usage.calls += 1
        return Choice(label, _one_hot(label, keys), None, 0.0, 0)


# --------------------------------------------------------------------------- #
# Fake (tests)
# --------------------------------------------------------------------------- #

class FakeDecisions:
    """Scripted answers, consumed in order; every call is recorded in ``calls``.

    ``choices`` holds labels (answered with probability 1.0) or whole
    :class:`Choice` objects; ``yes_no`` holds probabilities. Running out of
    script, or a label that is not among the options, raises
    :class:`DecisionsError`.
    """

    name = "fake"

    def __init__(self, choices: Optional[Sequence[Union[str, Choice]]] = None,
                 yes_no: Optional[Sequence[float]] = None) -> None:
        self._choices: list[Union[str, Choice]] = list(choices or [])
        self._yes_no: list[float] = [float(p) for p in (yes_no or [])]
        self.calls: list[DecisionCall] = []
        self.usage = DecisionUsage()

    def choose(self, state: str, instructions: str, options: dict[str, str]) -> Choice:
        self.calls.append(DecisionCall("choose", state, instructions, dict(options)))
        if not self._choices:
            raise DecisionsError(f"FakeDecisions: no scripted choice left for call #{len(self.calls)} "
                                 f"(options {list(options)})")
        scripted = self._choices.pop(0)
        if isinstance(scripted, Choice):
            choice = scripted
        else:
            choice = Choice(str(scripted), _one_hot(str(scripted), options), None, 0.0, 0)
        if choice.label not in options:
            raise DecisionsError(f"FakeDecisions: scripted label {choice.label!r} is not one of {list(options)}")
        self.usage.calls += 1
        self.usage.input_tokens += choice.input_tokens
        self.usage.cost_usd += choice.cost_usd
        return choice

    def yes_no(self, state: str, question: str) -> YesNo:
        self.calls.append(DecisionCall("yes_no", state, question, None))
        if not self._yes_no:
            raise DecisionsError(f"FakeDecisions: no scripted yes/no left for call #{len(self.calls)}")
        self.usage.calls += 1
        return YesNo(self._yes_no.pop(0), 0.0, 0)


# --------------------------------------------------------------------------- #
# Jev (TypeSafe System One)
# --------------------------------------------------------------------------- #

class JevDecisions:
    """Decisions by TypeSafe's Jev model through ``typesafe-sdk``.

    Nothing is contacted, and the SDK is not imported, until the first
    decision. Access, in order: ``client`` (anything with
    ``system_one(state, questions)``, e.g. a ``typesafe_sdk.TypeSafeClient``);
    ``api_key`` (through OpenRouter when it starts with ``sk-or-`` or
    ``base_url`` is OpenRouter's, else TypeSafe's API); ``OPENROUTER_API_KEY``
    (base URL ``https://openrouter.ai/api``, model ``~typesafe/jev-latest``);
    ``TYPESAFE_API_KEY`` (the SDK's base URL and model defaults). ``model`` and
    ``base_url`` override the defaults.

    Each request asks one question named :data:`QUESTION_NAME`. Its cost is
    ``usage.cost`` from the raw response when the gateway reports it, else the
    input tokens at :data:`PRICE_PER_M_INPUT_USD`. Answers are cached in
    memory per (state, question, options): a repeated decision makes no call
    and reports zero cost. Failures raise :class:`DecisionsError`, and
    :class:`DecisionsConfigError` for a missing or rejected key, no credit
    (402), an unknown model (404) or a missing SDK.
    """

    name = DECISIONS_JEV

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: Optional[str] = None, client: Optional[object] = None,
                 timeout: float = JEV_TIMEOUT_S) -> None:
        self.usage = DecisionUsage()
        self.timeout = float(timeout)
        self._api_key = api_key
        self._base_url = base_url
        self._client: Any = client
        self.model: Optional[str] = model           # the model actually requested, once known
        self.route: Optional[str] = "client" if client is not None else None  # "openrouter" | "typesafe"
        self._cache: OrderedDict[tuple[Any, ...], Union[Choice, YesNo]] = OrderedDict()

    # ----------------------------------------------------------------- API --

    def choose(self, state: str, instructions: str, options: dict[str, str]) -> Choice:
        if not options:
            raise ValueError("options must not be empty")
        key = ("choice", state, instructions, tuple(options.items()))
        cached = self._cache_get(key)
        if isinstance(cached, Choice):
            return dataclasses.replace(cached, probabilities=dict(cached.probabilities),
                                       cost_usd=0.0, input_tokens=0)
        question = {"type": "choice", "instructions": instructions, "criteria": dict(options)}
        result, tokens, cost = self._ask(state, question)
        answer = _answer(result, "choices")
        label = getattr(answer, "choice", None)
        if not isinstance(label, str) or label not in options:
            raise DecisionsError(f"Jev chose {label!r}, which is not one of the options {list(options)}")
        confidence = _as_float(getattr(answer, "confidence", None))
        choice = Choice(label, _probabilities(getattr(answer, "probabilities", None), options, label),
                        confidence, cost, tokens)
        self._cache_put(key, choice)
        return choice

    def yes_no(self, state: str, question: str) -> YesNo:
        key = ("noul", state, question)
        cached = self._cache_get(key)
        if isinstance(cached, YesNo):
            return YesNo(cached.probability, 0.0, 0)
        result, tokens, cost = self._ask(state, {"type": "noul", "instructions": question})
        p = _as_float(getattr(_answer(result, "nouls"), "noul", None))
        if p is None:
            raise DecisionsError("Jev returned no yes/no probability")
        answer = YesNo(min(1.0, max(0.0, p)), cost, tokens)
        self._cache_put(key, answer)
        return answer

    # ----------------------------------------------------------- internals --

    def _ask(self, state: str, question: dict[str, Any]) -> tuple[Any, int, float]:
        client = self._get_client()
        questions = {QUESTION_NAME: question}
        try:
            result = client.system_one(state, questions)
        except DecisionsError:
            raise
        except Exception as exc:
            raise _wrap_error(exc, route=self.route, model=self.model, timeout=self.timeout) from exc
        tokens = _reported_tokens(result)
        if tokens is None:
            tokens = request_tokens(state, question)
        cost = _reported_cost(result)
        if cost is None:
            cost = tokens * PRICE_PER_M_INPUT_USD / 1_000_000
        self.usage.calls += 1
        self.usage.input_tokens += tokens
        self.usage.cost_usd += cost
        return result, tokens, cost

    def _get_client(self) -> Any:
        if self._client is None:
            api_key, base_url, model, route = _resolve_access(self._api_key, self._base_url, self.model)
            sdk = _import_sdk()
            kwargs: dict[str, Any] = {"api_key": api_key, "timeout": self.timeout}
            if base_url:
                kwargs["base_url"] = base_url
            if model:
                kwargs["model"] = model
            try:
                kwargs["retry"] = sdk.RetryPolicy(max_retries=JEV_MAX_RETRIES, timeout=2 * self.timeout)
                self._client = sdk.TypeSafeClient(**kwargs)
            except Exception as exc:  # e.g. a key with whitespace, an invalid timeout
                raise DecisionsConfigError(f"could not create the Jev client: {exc}") from exc
            self.route, self.model = route, model
            logger.info("Jev decisions via %s, model %s", route, model or "(SDK default)")
        return self._client

    def _cache_get(self, key: tuple[Any, ...]) -> Optional[Union[Choice, YesNo]]:
        value = self._cache.get(key)
        if value is not None:
            self._cache.move_to_end(key)
        return value

    def _cache_put(self, key: tuple[Any, ...], value: Union[Choice, YesNo]) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)


def _import_sdk() -> Any:
    try:
        import typesafe_sdk
    except ImportError as exc:
        raise DecisionsConfigError(f"typesafe-sdk is not installed ({exc}).\n\n" + JEV_SETUP_HELP) from exc
    return typesafe_sdk


def _resolve_access(api_key: Optional[str], base_url: Optional[str],
                    model: Optional[str]) -> tuple[str, Optional[str], Optional[str], str]:
    """``(api_key, base_url, model, route)``; base URL and model stay None
    where the SDK's own defaults (and ``TYPESAFE_*`` variables) apply."""
    key = (api_key or "").strip()
    if key:
        openrouter = (OPENROUTER_HOST in (base_url or "")
                      or (not base_url and key.startswith(OPENROUTER_KEY_PREFIX)))
    else:
        openrouter_key = (os.environ.get(OPENROUTER_KEY_ENV) or "").strip()
        typesafe_key = (os.environ.get(TYPESAFE_KEY_ENV) or "").strip()
        if not openrouter_key and not typesafe_key:
            raise DecisionsConfigError(
                f"No Jev API key found: set {OPENROUTER_KEY_ENV} or {TYPESAFE_KEY_ENV}.\n\n" + JEV_SETUP_HELP)
        openrouter = bool(openrouter_key)
        key = openrouter_key or typesafe_key
    if openrouter:
        return key, base_url or OPENROUTER_BASE_URL, model or OPENROUTER_MODEL, "openrouter"
    return key, base_url, model, "typesafe"


def _wrap_error(exc: BaseException, *, route: Optional[str], model: Optional[str],
                timeout: float) -> DecisionsError:
    """A :class:`DecisionsError` with the cause spelled out. ``exc.status`` is
    the HTTP status of ``typesafe_sdk.TypeSafeAPIError`` (duck-typed, so the
    SDK need not be imported here). The SDK redacts the key from its messages."""
    status = getattr(exc, "status", None)
    via = f" ({route})" if route else ""
    keys = f"{OPENROUTER_KEY_ENV} or {TYPESAFE_KEY_ENV}"
    if isinstance(status, int) and not isinstance(status, bool):
        if status in (401, 403):
            return DecisionsConfigError(f"Jev{via} rejected the API key ({status}): {exc}. Check {keys}.")
        if status == 402:
            return DecisionsConfigError(f"Jev{via}: payment required (402), the account has no credit"
                                        f" left: {exc}")
        if status == 404:
            return DecisionsConfigError(f"Jev{via}: not found (404), check the model id {model!r}"
                                        f" and the base URL: {exc}")
        if status == 429:
            return DecisionsError(f"Jev{via} rate limit (429): {exc}")
        if 200 <= status < 300:
            return DecisionsError(f"Jev{via} returned a response RepoFinder could not read: {exc}")
        return DecisionsError(f"Jev{via} API error {status}: {exc}")
    if isinstance(exc, TimeoutError):
        return DecisionsError(f"Jev{via} did not answer within {timeout:.0f}s: {exc}")
    if isinstance(exc, ConnectionError):
        return DecisionsError(f"could not reach Jev{via}: {exc}")
    return DecisionsError(f"Jev{via} call failed: {type(exc).__name__}: {exc}")


def _answer(result: Any, group: str) -> Any:
    """The answer to :data:`QUESTION_NAME` from ``result.choices`` /
    ``result.nouls`` (``SystemOneResponse``), else from ``result.answers``."""
    for source in (getattr(result, group, None), getattr(result, "answers", None)):
        if isinstance(source, Mapping) and QUESTION_NAME in source:
            return source[QUESTION_NAME]
    raise DecisionsError(f"Jev returned no answer to the question {QUESTION_NAME!r}")


def _reported_tokens(result: Any) -> Optional[int]:
    value = getattr(getattr(result, "usage", None), "input_tokens", None)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _reported_cost(result: Any) -> Optional[float]:
    """``usage.cost`` from the raw HTTP body (OpenRouter reports it; the SDK's
    ``Usage`` model drops unknown fields), or None."""
    try:
        raw = result.raw_http_response.json()
    except Exception:  # noqa: BLE001 - no raw response (a fake) or a body that is not JSON
        return None
    usage = raw.get("usage") if isinstance(raw, dict) else None
    cost = usage.get("cost") if isinstance(usage, dict) else None
    return _as_float(cost)


def _probabilities(raw: Any, options: Mapping[str, str], label: str) -> dict[str, float]:
    """Per-option probabilities (0.0 for options the answer left out), or 1.0
    for ``label`` when the answer carries none."""
    probs: dict[str, float] = {}
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            p = _as_float(value)
            if p is not None:
                probs[str(key)] = p
    if not probs:
        return _one_hot(label, options)
    return {key: probs.get(key, 0.0) for key in options}


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _one_hot(label: str, keys: Union[Sequence[str], Mapping[str, str]]) -> dict[str, float]:
    return {key: 1.0 if key == label else 0.0 for key in keys}


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #

def _has_key() -> bool:
    return any((os.environ.get(name) or "").strip() for name in (OPENROUTER_KEY_ENV, TYPESAFE_KEY_ENV))


def detect_decisions(preferred: Optional[str] = None) -> Decisions:
    """The decisions backend for ``preferred`` or ``REPOFINDER_DECISIONS``.

    ``"rules"`` and ``"off"`` give :class:`RulesDecisions`. ``None``/empty
    and ``"auto"`` give :class:`JevDecisions` when a key is set and
    ``typesafe-sdk`` imports, else :class:`RulesDecisions`; they never raise.
    ``"jev"`` gives :class:`JevDecisions` when ``OPENROUTER_API_KEY`` or
    ``TYPESAFE_API_KEY`` is set and ``typesafe-sdk`` imports; otherwise, and
    for an unknown name, :class:`DecisionsConfigError` with setup help. A
    ``.env`` file in the current directory is loaded first, without
    overriding variables that are already set.
    """
    from .backends import load_dotenv  # lazy: keeps importing this module light

    load_dotenv()
    raw = preferred or os.environ.get(ENV_DECISIONS)
    choice = (raw or "").strip().lower() or None
    if choice in (DECISIONS_RULES, DECISIONS_OFF):
        return RulesDecisions()
    if choice in (None, DECISIONS_AUTO):
        if not _has_key():
            return RulesDecisions()
        try:
            _import_sdk()
        except DecisionsConfigError:
            return RulesDecisions()
        return JevDecisions()
    if choice == DECISIONS_JEV:
        if not _has_key():
            raise DecisionsConfigError(
                f"Jev decisions were requested ({ENV_DECISIONS}=jev), but neither {OPENROUTER_KEY_ENV}"
                f" nor {TYPESAFE_KEY_ENV} is set.\n\n" + JEV_SETUP_HELP)
        _import_sdk()
        return JevDecisions()
    raise DecisionsConfigError(
        f"Unknown decisions backend {raw!r}: use 'auto', 'jev', 'rules' or 'off'.\n\n" + JEV_SETUP_HELP)


# --------------------------------------------------------------------------- #
# Helpers for the chat layer
# --------------------------------------------------------------------------- #

def estimate_tokens(text: str) -> int:
    """Estimated token count: characters / :data:`CHARS_PER_TOKEN`, rounded up."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def request_tokens(state: str, question: Mapping[str, Any]) -> int:
    """Estimated tokens of one request: the state plus the question as compact JSON."""
    body = json.dumps({QUESTION_NAME: question}, ensure_ascii=False, separators=(",", ":"))
    return estimate_tokens(state) + estimate_tokens(body)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def _one_line(text: str, limit: int) -> str:
    return _clip(" ".join((text or "").split()), limit)


def _route_state(message: str, history: Sequence[tuple[str, str]]) -> str:
    recent = list(history)[-2 * HISTORY_EXCHANGES:]
    if recent:
        lines = ["Recent conversation (oldest first):"]
        lines += [f"{role}: {_one_line(text, HISTORY_TEXT_CHARS)}" for role, text in recent]
    else:
        lines = ["Recent conversation: none, this is the first message."]
    lines += ["", "New user message:", _clip((message or "").strip(), MESSAGE_CHARS)]
    return "\n".join(lines) + "\n"


def _checked(choice: Choice, options: Mapping[str, str], decisions: Decisions) -> Choice:
    if getattr(choice, "label", None) not in options:
        raise DecisionsError(f"{getattr(decisions, 'name', 'decisions')} chose "
                             f"{getattr(choice, 'label', None)!r}, which is not one of {list(options)}")
    return choice


def route_message(decisions: Decisions, message: str, history: list[tuple[str, str]]) -> RouteResult:
    """Route ``message``: a :class:`RouteResult` whose ``route`` is one of :data:`ROUTES`.

    ``history`` is ``[(role, text), ...]``, oldest first, role "user" or
    "assistant"; it may be empty. The state shows the last
    :data:`HISTORY_EXCHANGES` exchanges, each message on one line of at most
    :data:`HISTORY_TEXT_CHARS` characters, then the new message.
    """
    if isinstance(decisions, RulesDecisions):
        choice = decisions.route(message, history)
    else:
        choice = _checked(decisions.choose(_route_state(message, history), ROUTE_INSTRUCTIONS,
                                           dict(ROUTE_OPTIONS)), ROUTE_OPTIONS, decisions)
    return RouteResult(choice.label, dict(choice.probabilities), choice.cost_usd)


def _labeled(cards: Sequence[str], labels: Sequence[str]) -> str:
    return "\n\n".join(f"[{label}]\n{card.strip()}" for label, card in zip(labels, cards))


def _rerank_request(question: str, cards: Sequence[str],
                    allow_none: bool) -> tuple[str, str, dict[str, str]]:
    labels = CARD_LABELS[: len(cards)]
    span = "1 candidate code location, labeled A" if len(cards) == 1 else \
        f"{len(cards)} candidate code locations, labeled A to {labels[-1]}"
    state = "\n".join([
        "Question about the repository:", _clip(question.strip(), MESSAGE_CHARS), "",
        f"Code search returned {span}:", "", _labeled(cards, labels),
    ]) + "\n"
    options = {label: _one_line(next((ln for ln in card.splitlines() if ln.strip()), "(empty)"), OPTION_CHARS)
               for label, card in zip(labels, cards)}
    instructions = RERANK_INSTRUCTIONS
    if allow_none:
        options[NONE_LABEL] = NONE_DESCRIPTION
        instructions += RERANK_NONE_INSTRUCTIONS
    return state, instructions, options


def _fit(build: Any, cards: Sequence[str]) -> tuple[int, Any]:
    """The largest prefix of ``cards`` whose request (``build(cards)`` ->
    ``(state, question)``) fits :data:`MAX_REQUEST_TOKENS`; when even the first
    card does not fit, it is cut to fit. Returns ``(n_kept, build result)``."""
    for n in range(len(cards), 0, -1):
        built = build(cards[:n])
        if request_tokens(*built[:2]) <= MAX_REQUEST_TOKENS:
            return n, built
    card = cards[0]
    marker = "\n... (cut to fit the request)"
    overhead = request_tokens(*build([marker])[:2])
    keep = max(0, (MAX_REQUEST_TOKENS - overhead) * CHARS_PER_TOKEN)
    while True:
        built = build([card[:keep] + marker])
        if request_tokens(*built[:2]) <= MAX_REQUEST_TOKENS or keep == 0:
            return 1, built
        keep = max(0, keep - 1_000)


def rerank_cards(decisions: Decisions, question: str, cards: list[str], allow_none: bool = True) -> RerankResult:
    """Order ``cards`` (rendered search candidates, in search order) by how
    well they answer ``question``.

    Cards are labeled "A".."P" (at most :data:`MAX_CARDS`; more raise
    ``ValueError``); with ``allow_none`` the backend may also pick "NONE". The
    pick comes first, the other shown cards follow by probability, then search
    order. When the request would exceed :data:`MAX_REQUEST_TOKENS`, the
    last cards are not shown and keep their search order at the end of
    ``order``. Rules keep the search order. No cards: an empty order, no call.
    """
    n = len(cards)
    if n > MAX_CARDS:
        raise ValueError(f"at most {MAX_CARDS} cards can be reranked, got {n}")
    if n == 0:
        return RerankResult([], False, {}, 0.0)
    if isinstance(decisions, RulesDecisions):
        keys = list(CARD_LABELS[:n]) + ([NONE_LABEL] if allow_none else [])
        choice = decisions.rerank(keys)
        return RerankResult(list(range(n)), False, dict(choice.probabilities), 0.0)

    def build(shown: Sequence[str]) -> tuple[str, dict[str, Any], str, dict[str, str]]:
        state, instructions, options = _rerank_request(question, shown, allow_none)
        return state, {"type": "choice", "instructions": instructions, "criteria": options}, instructions, options

    kept, (state, _question, instructions, options) = _fit(build, cards)
    if kept < n:
        logger.info("rerank: %d of %d cards fit the %d-token budget", kept, n, MAX_REQUEST_TOKENS)
    choice = _checked(decisions.choose(state, instructions, options), options, decisions)
    labels = CARD_LABELS[:kept]
    probs = choice.probabilities
    shown = sorted(range(kept), key=lambda i: (labels[i] != choice.label, -probs.get(labels[i], 0.0), i))
    return RerankResult(shown + list(range(kept, n)), choice.label == NONE_LABEL, dict(probs), choice.cost_usd)


def context_suffices(decisions: Decisions, message: str, previous_question: str, cards: list[str]) -> YesNo:
    """P(the excerpts in ``cards``, shown with the answer to
    ``previous_question``, already contain what is needed to answer the
    follow-up ``message``). The threshold is the caller's. No cards: 0.0 and
    no call. Cards beyond the token budget are left out of the request."""
    if not cards:
        return YesNo(0.0, 0.0, 0)
    if isinstance(decisions, RulesDecisions):
        return decisions.suffices(message, previous_question, cards)

    def build(shown: Sequence[str]) -> tuple[str, dict[str, Any]]:
        state = "\n".join([
            "Previous question:", _clip((previous_question or "").strip(), MESSAGE_CHARS), "",
            f"Code excerpts shown with the previous answer ({len(shown)}):", "",
            _labeled(shown, [str(i) for i in range(1, len(shown) + 1)]), "",
            "Follow-up message:", _clip((message or "").strip(), MESSAGE_CHARS),
        ]) + "\n"
        return state, {"type": "noul", "instructions": CONTEXT_QUESTION}

    _kept, (state, _question) = _fit(build, cards)
    answer = decisions.yes_no(state, CONTEXT_QUESTION)
    return YesNo(min(1.0, max(0.0, float(answer.probability))), answer.cost_usd, answer.input_tokens)

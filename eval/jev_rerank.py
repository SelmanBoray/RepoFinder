"""Jev rerank experiment: let a typed decision model pick the best of the top search hits.

Usage (from the repository root, with a local checkout of the golden repo)::

    .venv\\Scripts\\python -I eval/jev_rerank.py eval/golden/flask.json --source PATH --dry-run
    .venv\\Scripts\\python -I eval/jev_rerank.py eval/golden/flask.json --source PATH --dump-state q13
    .venv\\Scripts\\python -I eval/jev_rerank.py eval/golden/flask.json --source PATH --ids q13,q23

For every golden question, ``SearchIndex.search`` returns the top ``--candidates``
hits (default 8, at most 16). Each hit becomes a labeled card (A, B, C, ...):
``path:start-end``, kind and qualname, and the hit's own source lines read from
``--source`` (at most ``--lines-per-candidate``; a file-level hit shows the file's
first lines). The question and the cards form the ``state`` of one TypeSafe
System One request with a single ``choice`` question whose criteria are the
labels (plus ``NONE`` with ``--allow-none``). The model's pick goes to rank 1;
when the answer carries per-label probabilities, the shown candidates are sorted
by them. Hits that were not shown keep their search order after the shown ones.

Scoring reuses :func:`run_eval.run_eval` in ``retrieval`` mode twice on one index:
"before" with the search-only adapter of ``search_baseline.py`` and "after" with
an adapter whose ``retrieve`` returns the reranked hits. So hit@1, hit@3, hit@k
and MRR have exactly the runner's definitions.

The request is limited to 32k tokens (state plus questions). Tokens are
estimated as characters / 4; code often has fewer characters per token, so the
default budget is 70% of the limit. When the cards do not fit, the lowest-ranked
candidates are dropped. Cost: input tokens at ``--price-per-m`` USD per million
(output is free); a live run uses ``usage.cost`` from the raw response when the
gateway reports it.

Access: ``OPENROUTER_API_KEY`` (base URL ``https://openrouter.ai/api``, model
``~typesafe/jev-latest``) or ``TYPESAFE_API_KEY`` (the SDK defaults). The live
path needs ``typesafe-sdk`` installed in the venv; ``--dry-run`` and
``--dump-state`` need neither the SDK nor a key and never call the model.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import os
import string
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

_HERE = Path(__file__).resolve().parent


def _load(name: str) -> Any:
    """An ``eval/<name>.py`` sibling as a module (reused if already imported)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


_runner = _load("run_eval")
_baseline = _load("search_baseline")

# --- request shape -------------------------------------------------------------
QUESTION_NAME = "best_candidate"
INSTRUCTIONS = (
    "Which candidate contains the code that answers the question?"
    " Pick the one whose lines a reader should open first."
)
NONE_LABEL = "NONE"
NONE_DESCRIPTION = "None of the candidates contains code that answers the question."
LABELS = string.ascii_uppercase
DEFAULT_CANDIDATES = 8
MAX_CANDIDATES = 16
DEFAULT_LINES = 60

# --- budget and cost -------------------------------------------------------------
CONTEXT_TOKENS = 32_000        # Jev context: state + questions
CHARS_PER_TOKEN = 4            # estimate only; code usually has fewer chars per token
BUDGET_FRACTION = 0.7          # head-room for that estimate error
DEFAULT_MAX_TOKENS = int(CONTEXT_TOKENS * BUDGET_FRACTION)
# Input price in USD per million tokens, as given for Jev on 2026-10-09. OpenRouter's
# public /api/v1/models list does not include the System One models, so this number
# was not verified here: override it with --price-per-m. Output tokens are free.
PRICE_PER_M_INPUT_USD = 0.042

# --- access ------------------------------------------------------------------------
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
TYPESAFE_KEY_ENV = "TYPESAFE_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api"
OPENROUTER_MODEL = "~typesafe/jev-latest"
HTTP_TIMEOUT_S = 60.0
ABORT_STATUSES = frozenset({401, 402, 403})   # bad key, no credit, no access: stop the run


class MissingKey(RuntimeError):
    """Neither API key is set."""


# =============================================================================== #
# State
# =============================================================================== #

@dataclasses.dataclass
class Candidate:
    label: str
    search_rank: int           # 1-based rank in the search results
    hit: Any
    kind: str                  # "function" | "class" | "method" | "file" | "symbol"
    qualname: Optional[str]
    header: str                # "src/x.py:10-20 | method A.run"
    card: str                  # header + code block, as it appears in the state
    lines_shown: int
    truncated: bool

    def record(self) -> dict[str, Any]:
        h = self.hit
        return {
            "label": self.label, "search_rank": self.search_rank, "path": h.path,
            "start_line": h.start_line, "end_line": h.end_line,
            "symbol_id": getattr(h, "symbol_id", None), "kind": self.kind,
            "qualname": self.qualname, "lines_shown": self.lines_shown, "truncated": self.truncated,
        }


@dataclasses.dataclass
class Prepared:
    """One question's request (or the reason there is none)."""

    question: str
    hits: list[Any]                    # all search hits, in search order
    candidates: list[Candidate]        # the shown ones, in search order
    n_dropped: int                     # candidates removed to fit the token budget
    state: str
    questions: dict[str, Any]
    est_tokens: int
    skipped: Optional[str] = None      # set when no request is made


def est_tokens(text: str) -> int:
    """Estimated token count: characters / :data:`CHARS_PER_TOKEN`, rounded up."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def request_tokens(state: str, questions: dict[str, Any]) -> int:
    """Estimated tokens of one request: the state plus the questions as compact JSON."""
    return est_tokens(state) + est_tokens(json.dumps(questions, ensure_ascii=False, separators=(",", ":")))


def est_cost(tokens: int, price_per_m: float = PRICE_PER_M_INPUT_USD) -> float:
    return tokens * price_per_m / 1_000_000


def make_card(label: str, rank: int, hit: Any, symbol: Any, lines: Optional[list[str]],
              max_lines: int) -> Candidate:
    """A labeled card: location, kind, qualname and the hit's first ``max_lines`` lines."""
    if getattr(hit, "kind", "symbol") == "file" or not getattr(hit, "symbol_id", None):
        kind, qualname = "file", None
        start, end = 1, len(lines) if lines is not None else hit.end_line
        title = "whole file"
    else:
        kind = getattr(symbol, "kind", None) or "symbol"
        qualname = getattr(symbol, "qualname", None) or hit.symbol_id.split("::", 1)[-1]
        start, end = hit.start_line, hit.end_line
        title = f"{kind} {qualname}"
    header = f"{hit.path}:{hit.start_line}-{hit.end_line} | {title}"
    if lines is None:
        body, shown, truncated = "# (source not available)", 0, False
    else:
        window = [line.rstrip("\r").rstrip() for line in lines[max(start, 1) - 1 : end]]
        shown = min(len(window), max_lines)
        truncated = len(window) > max_lines
        body = textwrap.dedent("\n".join(window[:shown]))
        if truncated:
            body += f"\n# ... {len(window) - shown} more lines (to line {end})"
    card = f"[{label}] {header}\n```python\n{body}\n```"
    return Candidate(label, rank, hit, kind, qualname, header, card, shown, truncated)


def build_state(repo: str, question: str, cards: list[Candidate]) -> str:
    """The state text: the question, then the cards."""
    if not cards:
        span = "no candidates"
    elif len(cards) == 1:
        span = "1 candidate code location, labeled A"
    else:
        span = f"{len(cards)} candidate code locations, labeled A to {cards[-1].label}"
    parts = [f'Question about the "{repo}" repository:', question.strip(), "",
             f"Code search returned {span}:", ""]
    parts.append("\n\n".join(c.card for c in cards))
    return "\n".join(parts) + "\n"


def build_questions(cards: list[Candidate], allow_none: bool = False) -> dict[str, Any]:
    """The single ``choice`` question; each label is described by its card header.

    It is the SDK's raw-dictionary form of ``Choice(instructions=..., criteria=...)``
    (same wire JSON), so building it does not need ``typesafe_sdk``.
    """
    criteria: dict[str, Any] = {c.label: c.header for c in cards}
    if allow_none:
        criteria[NONE_LABEL] = NONE_DESCRIPTION
    return {QUESTION_NAME: {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}}


def prepare(repo: str, question: str, hits: list[Any],
            read_lines: Callable[[str], Optional[list[str]]],
            symbols: dict[str, Any], *, candidates: int = DEFAULT_CANDIDATES,
            lines_per_candidate: int = DEFAULT_LINES, allow_none: bool = False,
            max_tokens: int = DEFAULT_MAX_TOKENS) -> Prepared:
    """Cards for the first ``candidates`` hits, dropping the last ones until the request fits."""
    shown_hits = list(hits)[:candidates]
    cards = [
        make_card(LABELS[i], i + 1, h, symbols.get(getattr(h, "symbol_id", None) or ""),
                  read_lines(h.path), lines_per_candidate)
        for i, h in enumerate(shown_hits)
    ]
    if not cards:
        return Prepared(question, list(hits), [], 0, "", {}, 0, skipped="no search hits")
    for n in range(len(cards), 0, -1):
        state = build_state(repo, question, cards[:n])
        questions = build_questions(cards[:n], allow_none)
        tokens = request_tokens(state, questions)
        if tokens <= max_tokens:
            return Prepared(question, list(hits), cards[:n], len(cards) - n, state, questions, tokens)
    return Prepared(question, list(hits), [], len(cards), state, questions, tokens,
                    skipped=f"one candidate already needs ~{tokens} tokens (budget {max_tokens})")


# =============================================================================== #
# Decision and reranking
# =============================================================================== #

def parse_result(result: Any) -> dict[str, Any]:
    """Choice, probabilities, usage and raw body of a ``SystemOneResponse`` (duck-typed).

    ``usage.cost`` is not a field of the SDK's ``Usage`` model (unknown fields are
    dropped), so it is read from the raw HTTP body when there is one.
    """
    answer = result.choices[QUESTION_NAME]
    usage = getattr(result, "usage", None)
    raw: Optional[Any] = None
    try:
        raw = result.raw_http_response.json()
    except Exception:  # noqa: BLE001 - a fake or a body that is not JSON: no raw record
        raw = None
    raw_usage = raw.get("usage") if isinstance(raw, dict) else None
    cost = raw_usage.get("cost") if isinstance(raw_usage, dict) else None
    probabilities = getattr(answer, "probabilities", None) or {}
    return {
        "choice": answer.choice,
        "confidence": getattr(answer, "confidence", None),
        "probabilities": {str(k): float(v) for k, v in dict(probabilities).items()},
        "model": getattr(result, "model", None),
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "cost_usd": float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None,
        "raw_response": raw,
    }


def rerank_hits(prep: Prepared, choice: Optional[str],
                probabilities: Optional[dict[str, float]] = None) -> tuple[list[Any], list[str]]:
    """The hits reordered by the decision, and the shown labels in their new order.

    With probabilities, the shown candidates are sorted by them (the pick first);
    without, only the pick moves to the front. A ``NONE`` pick without
    probabilities keeps the search order. Hits that were not shown follow in
    search order.
    """
    shown = list(prep.candidates)
    probs = probabilities or {}
    if probs:
        shown.sort(key=lambda c: (c.label != choice, -probs.get(c.label, -1.0), c.search_rank))
    elif choice is not None:
        shown.sort(key=lambda c: (c.label != choice, c.search_rank))
    shown_ids = {id(c.hit) for c in prep.candidates}
    rest = [h for h in prep.hits if id(h) not in shown_ids]
    new_hits = [_annotate(c, i, probs.get(c.label)) for i, c in enumerate(shown, start=1)] + rest
    return new_hits, [c.label for c in shown]


def _annotate(c: Candidate, rank: int, p: Optional[float]) -> Any:
    """A copy of the hit whose ``why`` starts with the Jev rank (the original is not changed)."""
    if not dataclasses.is_dataclass(c.hit):
        return c.hit
    note = f"jev #{rank} [{c.label}, search #{c.search_rank}]" + (f" p={p:.3f}" if p is not None else "")
    why = getattr(c.hit, "why", "") or ""
    return dataclasses.replace(c.hit, why=f"{note}; {why}" if why else note)


class JevReranker:
    """``retrieve`` for the pipeline contract: search, ask Jev, return the reranked hits.

    ``client`` is anything with ``system_one(state, questions)`` returning an
    object shaped like ``typesafe_sdk.SystemOneResponse``; ``None`` is a dry run
    (states are built and measured, hits keep the search order). Records are kept
    in :attr:`records`, keyed by question text.
    """

    def __init__(self, repo: str, root: Path, client: Any = None, *,
                 candidates: int = DEFAULT_CANDIDATES, lines_per_candidate: int = DEFAULT_LINES,
                 allow_none: bool = False, max_tokens: int = DEFAULT_MAX_TOKENS,
                 price_per_m: float = PRICE_PER_M_INPUT_USD) -> None:
        self.repo = repo
        self.root = Path(root)
        self.client = client
        self.candidates = candidates
        self.lines_per_candidate = lines_per_candidate
        self.allow_none = allow_none
        self.max_tokens = max_tokens
        self.price_per_m = price_per_m
        self.records: dict[str, dict[str, Any]] = {}
        # A 401/402/403 from the API. run_eval records a raising question and goes on,
        # so after one, every later question raises without a call and
        # run_experiment re-raises this at the end.
        self.fatal: Optional[BaseException] = None
        self._files: dict[str, Optional[list[str]]] = {}
        self._symbols: Optional[dict[str, Any]] = None

    @property
    def dry_run(self) -> bool:
        return self.client is None

    def read_lines(self, path: str) -> Optional[list[str]]:
        if path not in self._files:
            # The runner's reader: split on "\n" only, the parser's line numbering.
            self._files[path] = _runner._read_lines(self.root, path)
        return self._files[path]

    def symbols(self, ws: Any) -> dict[str, Any]:
        if self._symbols is None:
            index = getattr(getattr(ws, "search", None), "index", None)
            self._symbols = {s.id: s for s in getattr(index, "symbols", None) or []}
        return self._symbols

    def prepare(self, ws: Any, question: str, hits: list[Any]) -> Prepared:
        return prepare(self.repo, question, hits, self.read_lines, self.symbols(ws),
                       candidates=self.candidates, lines_per_candidate=self.lines_per_candidate,
                       allow_none=self.allow_none, max_tokens=self.max_tokens)

    def retrieve(self, ws: Any, question: str, k: int = _runner.DEFAULT_K) -> tuple[list[Any], list[Any]]:
        if self.fatal is not None:
            raise RuntimeError(f"not called: the run was aborted by {type(self.fatal).__name__}")
        hits = ws.search.search(question, k=max(k, self.candidates))
        prep = self.prepare(ws, question, hits)
        new_hits, record = self.decide(prep)
        self.records[question] = record
        chunks = [
            SimpleNamespace(path=h.path, start_line=h.start_line, end_line=h.end_line,
                            text="", label=h.symbol_id)
            for h in new_hits[:k]
            if h.kind != "file"
        ]
        return chunks, new_hits

    def decide(self, prep: Prepared) -> tuple[list[Any], dict[str, Any]]:
        """Call Jev for ``prep`` (unless dry or skipped) and rerank. Returns hits and a record."""
        record: dict[str, Any] = {
            "question": prep.question,
            "n_search_hits": len(prep.hits),
            "n_candidates": len(prep.candidates),
            "n_dropped": prep.n_dropped,
            "state_chars": len(prep.state),
            "questions_chars": len(json.dumps(prep.questions, ensure_ascii=False, separators=(",", ":"))),
            "est_tokens": prep.est_tokens,
            "est_cost_usd": est_cost(prep.est_tokens, self.price_per_m) if not prep.skipped else 0.0,
            "candidates": [c.record() for c in prep.candidates],
            "skipped": prep.skipped,
            "choice": None, "confidence": None, "probabilities": {}, "order": None,
            "error": None,
        }
        if self.dry_run or prep.skipped:
            return list(prep.hits), record
        labels = {c.label for c in prep.candidates} | ({NONE_LABEL} if self.allow_none else set())
        t = time.perf_counter()
        try:
            parsed = parse_result(self.client.system_one(prep.state, prep.questions))
            if parsed["choice"] not in labels:
                raise ValueError(f"choice {parsed['choice']!r} is not one of the labels")
        except Exception as exc:  # noqa: BLE001 - a failed call keeps the search order
            if getattr(exc, "status", None) in ABORT_STATUSES:
                self.fatal = exc
                raise
            record["elapsed_s"] = round(time.perf_counter() - t, 3)
            record["error"] = f"{type(exc).__name__}: {exc}"
            return list(prep.hits), record
        record["elapsed_s"] = round(time.perf_counter() - t, 3)
        record.update(parsed)
        if record["cost_usd"] is not None:
            record["cost_source"] = "usage.cost"
        elif record["input_tokens"] is not None:
            record["cost_usd"] = est_cost(int(record["input_tokens"]), self.price_per_m)
            record["cost_source"] = "usage.input_tokens x price"
        else:
            record["cost_usd"] = record["est_cost_usd"]
            record["cost_source"] = "estimate"
        new_hits, order = rerank_hits(prep, parsed["choice"], parsed["probabilities"])
        record["order"] = order
        return new_hits, record


# =============================================================================== #
# Client
# =============================================================================== #

def make_client(env: Optional[dict[str, str]] = None, model: Optional[str] = None) -> tuple[Any, str, str]:
    """A ``typesafe_sdk.TypeSafeClient`` for the key in ``env``: ``(client, route, model)``.

    ``OPENROUTER_API_KEY`` wins over ``TYPESAFE_API_KEY``. Raises :class:`MissingKey`
    without either, and ``ImportError`` when ``typesafe-sdk`` is not installed.
    """
    env = dict(os.environ if env is None else env)
    openrouter = (env.get(OPENROUTER_KEY_ENV) or "").strip()
    typesafe = (env.get(TYPESAFE_KEY_ENV) or "").strip()
    if not openrouter and not typesafe:
        raise MissingKey(f"set {OPENROUTER_KEY_ENV} or {TYPESAFE_KEY_ENV}, or use --dry-run")
    import typesafe_sdk
    from typesafe_sdk import constants

    retry = typesafe_sdk.RetryPolicy(max_retries=2, timeout=2 * HTTP_TIMEOUT_S)
    if openrouter:
        chosen = model or OPENROUTER_MODEL
        client = typesafe_sdk.TypeSafeClient(api_key=openrouter, base_url=OPENROUTER_BASE_URL, model=chosen,
                                             timeout=HTTP_TIMEOUT_S, retry=retry)
        return client, "openrouter", chosen
    # The direct route keeps the SDK's base URL (TYPESAFE_BASE_URL or api.typesafe.ai).
    chosen = model or (env.get(constants.DEFAULT_MODEL_ENV) or "").strip() or constants.DEFAULT_MODEL
    client = typesafe_sdk.TypeSafeClient(api_key=typesafe, model=chosen, timeout=HTTP_TIMEOUT_S, retry=retry)
    return client, "typesafe", chosen


# =============================================================================== #
# Experiment
# =============================================================================== #

def run_experiment(golden: dict[str, Any], source: str, *, client: Any = None,
                   k: int = _runner.DEFAULT_K, candidates: int = DEFAULT_CANDIDATES,
                   lines_per_candidate: int = DEFAULT_LINES, allow_none: bool = False,
                   max_tokens: int = DEFAULT_MAX_TOKENS, price_per_m: float = PRICE_PER_M_INPUT_USD,
                   ids: Optional[list[str]] = None, log: Any = None) -> dict[str, Any]:
    """Score search ("before") and search + Jev ("after") on one index. ``client=None`` is a dry run."""
    name = (golden.get("repo") or {}).get("name") or Path(source).name
    base = _baseline.search_pipeline(name)
    ws = base.open_workspace(source)

    def shared(source: str, data_dir: Optional[str] = None, refresh: bool = False) -> Any:
        return ws

    before = _runner.run_eval(
        golden, source, "retrieval", k,
        SimpleNamespace(open_workspace=shared, retrieve=base.retrieve, ask=base.ask), ids=ids)
    reranker = JevReranker(name, Path(source), client, candidates=candidates,
                           lines_per_candidate=lines_per_candidate, allow_none=allow_none,
                           max_tokens=max_tokens, price_per_m=price_per_m)
    after = _runner.run_eval(
        golden, source, "retrieval", k,
        SimpleNamespace(open_workspace=shared, retrieve=reranker.retrieve, ask=base.ask),
        ids=ids, log=None if reranker.dry_run else log)
    if reranker.fatal is not None:
        raise reranker.fatal

    before_rows = {r["id"]: r for r in before["questions"]}
    records = []
    for row in after["questions"]:
        rec = dict(reranker.records.get(row["question"]) or {"question": row["question"]})
        if row.get("error"):
            rec["error"] = row["error"]
        b_ret, a_ret = before_rows[row["id"]].get("retrieval") or {}, row.get("retrieval") or {}
        rec = {"id": row["id"], "category": row["category"], "lang": row["lang"], **rec,
               "rank_before": b_ret.get("rank"), "rank_after": a_ret.get("rank"),
               "pointer_rank_before": b_ret.get("pointer_rank"),
               "pointer_rank_after": a_ret.get("pointer_rank")}
        if reranker.dry_run:
            for key in ("choice", "confidence", "probabilities", "order", "rank_after", "pointer_rank_after"):
                rec.pop(key, None)
        records.append(rec)

    meta = {
        "experiment": "jev_rerank",
        "dry_run": reranker.dry_run,
        "repo": golden.get("repo"),
        "source": source,
        "k": k,
        "candidates": candidates,
        "lines_per_candidate": lines_per_candidate,
        "allow_none": allow_none,
        "max_tokens": max_tokens,
        "context_tokens": CONTEXT_TOKENS,
        "chars_per_token": CHARS_PER_TOKEN,
        "price_per_m_input_usd": price_per_m,
        "instructions": INSTRUCTIONS,
        "ids": [r["id"] for r in records],
        "started": before["meta"]["started"],
        "checkout": before["meta"]["checkout"],
    }
    return {
        "meta": meta,
        "cost": cost_summary(records, reranker.dry_run),
        "before": before,
        "after": None if reranker.dry_run else after,
        "jev": records,
    }


def cost_summary(records: list[dict[str, Any]], dry_run: bool) -> dict[str, Any]:
    """Estimated (and, when live, reported) tokens and cost over the requests."""
    sent = [r for r in records if not r.get("skipped") and r.get("est_tokens")]
    est = [int(r["est_tokens"]) for r in sent]
    est_usd = [float(r.get("est_cost_usd") or 0.0) for r in sent]
    out: dict[str, Any] = {
        "n_requests": len(sent),
        "n_skipped": sum(1 for r in records if r.get("skipped")),
        "est_tokens_total": sum(est),
        "est_tokens_mean": round(sum(est) / len(est), 1) if est else None,
        "est_tokens_max": max(est) if est else None,
        "est_cost_usd_total": round(sum(est_usd), 12),
        "est_cost_usd_mean": round(sum(est_usd) / len(est_usd), 12) if est_usd else None,
    }
    if not dry_run:
        done = [r for r in sent if not r.get("error") and r.get("choice") is not None]
        tokens = [int(r["input_tokens"]) for r in done if r.get("input_tokens") is not None]
        usd = [float(r.get("cost_usd") or 0.0) for r in done]
        out.update({
            "n_ok": len(done),
            "n_errors": sum(1 for r in sent if r.get("error")),
            "input_tokens_total": sum(tokens) if tokens else None,
            "input_tokens_mean": round(sum(tokens) / len(tokens), 1) if tokens else None,
            "cost_usd_total": round(sum(usd), 12),
            "cost_usd_mean": round(sum(usd) / len(usd), 12) if usd else None,
            "n_cost_reported": sum(1 for r in done if r.get("cost_source") == "usage.cost"),
        })
    return out


# =============================================================================== #
# Reporting
# =============================================================================== #

def _rank(value: Any) -> str:
    return "-" if value is None else str(value)


def _usd(value: Optional[float]) -> str:
    return "-" if value is None else f"${value:.6f}"


def dry_run_table(report: dict[str, Any]) -> str:
    rows = ["| id | cat | lang | cands (dropped) | state chars | est tokens | est $ | search rank |",
            "|---|---|---|---|---|---|---|---|"]
    for r in report["jev"]:
        rank = _rank(r.get("rank_before")) if r["category"] != "not-in-repo" \
            else f"ptr {_rank(r.get('pointer_rank_before'))}"
        cands = f"{r.get('n_candidates', 0)} ({r.get('n_dropped', 0)})"
        if r.get("skipped"):
            cands += " skipped"
        rows.append(f"| {r['id']} | {r['category']} | {r['lang']} | {cands} | {r.get('state_chars', 0)}"
                    f" | {r.get('est_tokens', 0)} | {_usd(r.get('est_cost_usd'))} | {rank} |")
    return "\n".join(rows) + "\n"


def rank_table(report: dict[str, Any]) -> str:
    after_rows = {r["id"]: r for r in (report.get("after") or {}).get("questions", [])}
    rows = ["| id | cat | lang | search rank | jev rank | jev pick (p) | top-1 after |",
            "|---|---|---|---|---|---|---|"]
    for r in report["jev"]:
        if r["category"] == "not-in-repo":
            before = f"ptr {_rank(r.get('pointer_rank_before'))}"
            after = f"ptr {_rank(r.get('pointer_rank_after'))}"
        else:
            before, after = _rank(r.get("rank_before")), _rank(r.get("rank_after"))
        if r.get("error"):
            pick = "ERROR"
        elif r.get("skipped"):
            pick = "skipped"
        else:
            p = (r.get("probabilities") or {}).get(r.get("choice") or "")
            pick = f"{r.get('choice')}" + (f" ({p:.2f})" if p is not None else "")
        top = _runner.top1_label(after_rows[r["id"]]) if r["id"] in after_rows else "-"
        rows.append(f"| {r['id']} | {r['category']} | {r['lang']} | {before} | {after} | {pick}"
                    f" | {top.replace('|', chr(92) + '|')} |")
    return "\n".join(rows) + "\n"


def compare_table(before: dict[str, Any], after: dict[str, Any], k: int) -> str:
    """Strict hit@1 / hit@3 / hit@k / MRR, before -> after, overall and per group."""
    def cells(s_b: dict[str, Any], s_a: dict[str, Any]) -> str:
        rb, ra = s_b.get("retrieval") or {}, s_a.get("retrieval") or {}
        out = [f"{_runner._fmt_rate(rb.get(key))} -> {_runner._fmt_rate(ra.get(key))}"
               for key in ("hit@1", "hit@3", "hit@k")]
        out.append(f"{_runner._fmt(rb.get('mrr'))} -> {_runner._fmt(ra.get('mrr'))}")
        return " | ".join(out)

    sb, sa = before["summary"], after["summary"]
    rows = [f"| group | n | hit@1 | hit@3 | hit@{k} | MRR |", "|---|---|---|---|---|---|",
            f"| overall | {sb['overall']['n']} | {cells(sb['overall'], sa['overall'])} |"]
    for title in ("by_category", "by_lang"):
        for group, s in sb[title].items():
            rows.append(f"| {title[3:]}: {group} | {s['n']} | {cells(s, sa[title].get(group, {}))} |")
    return "\n".join(rows) + "\n"


def cost_lines(report: dict[str, Any]) -> list[str]:
    c, meta = report["cost"], report["meta"]
    lines = [
        f"- requests: {c['n_requests']} (skipped {c['n_skipped']}); estimate at"
        f" {meta['chars_per_token']} chars/token and ${meta['price_per_m_input_usd']}/M input tokens",
        f"- estimated tokens per question: mean {c['est_tokens_mean']}, max {c['est_tokens_max']}"
        f" (budget {meta['max_tokens']}, context {meta['context_tokens']}), total {c['est_tokens_total']}",
        f"- estimated cost: mean {_usd(c['est_cost_usd_mean'])} per question,"
        f" total {_usd(c['est_cost_usd_total'])}",
    ]
    if not meta["dry_run"]:
        lines += [
            f"- calls: {c['n_ok']} ok, {c['n_errors']} errors (errored questions keep the search order)",
            f"- reported input tokens: mean {c['input_tokens_mean']}, total {c['input_tokens_total']}",
            f"- cost: mean {_usd(c['cost_usd_mean'])} per question, total {_usd(c['cost_usd_total'])}"
            f" (usage.cost reported in {c['n_cost_reported']}/{c['n_ok']} responses)",
        ]
    return lines


def render(report: dict[str, Any]) -> str:
    meta, k = report["meta"], report["meta"]["k"]
    before = report["before"]
    name = (meta.get("repo") or {}).get("name", "?")
    mode = "DRY RUN, no model calls" if meta["dry_run"] else f"route={meta.get('route')} model={meta.get('model')}"
    out = [f"# Jev rerank: {name} (k={k}, candidates={meta['candidates']},"
           f" lines/candidate={meta['lines_per_candidate']}, {mode})", ""]
    if meta["dry_run"]:
        out += [dry_run_table(report), f"## Before (search, k={k})", ""]
        out += _runner.summary_lines(before["summary"]["overall"], k)
        out += ["", "After (Jev): not computed in a dry run."]
    else:
        after = report["after"]
        out += [rank_table(report), "## Before (search) -> after (Jev), strict", "",
                compare_table(before, after, k), f"### Before (search, k={k})"]
        out += _runner.summary_lines(before["summary"]["overall"], k)
        out += ["", f"### After (Jev, k={k})"]
        out += _runner.summary_lines(after["summary"]["overall"], k)
    out += ["", "## Cost", ""] + cost_lines(report)
    return "\n".join(out) + "\n"


# =============================================================================== #
# CLI
# =============================================================================== #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jev_rerank.py", description=__doc__.splitlines()[0])
    p.add_argument("golden")
    p.add_argument("--source", required=True, help="local checkout of the golden repository")
    p.add_argument("--k", type=int, default=_runner.DEFAULT_K,
                   help=f"hits scored per question (default {_runner.DEFAULT_K})")
    p.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES,
                   help=f"search hits shown to Jev (default {DEFAULT_CANDIDATES}, at most {MAX_CANDIDATES})")
    p.add_argument("--lines-per-candidate", type=int, default=DEFAULT_LINES,
                   help=f"source lines per candidate card (default {DEFAULT_LINES})")
    p.add_argument("--allow-none", action="store_true", help="add a NONE choice (none of the candidates)")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help=f"estimated-token budget per request (default {DEFAULT_MAX_TOKENS},"
                        f" {int(BUDGET_FRACTION * 100)}%% of the {CONTEXT_TOKENS} context)")
    p.add_argument("--price-per-m", type=float, default=PRICE_PER_M_INPUT_USD,
                   help=f"USD per million input tokens (default {PRICE_PER_M_INPUT_USD})")
    p.add_argument("--model", help=f"model override (default {OPENROUTER_MODEL} on OpenRouter,"
                                   " the SDK default with TYPESAFE_API_KEY)")
    p.add_argument("--ids", help="comma-separated question ids, e.g. q01,q05")
    p.add_argument("--dry-run", action="store_true", help="build states and estimate cost; never call Jev")
    p.add_argument("--dump-state", metavar="ID", help="print one question's state and choice question, then exit")
    p.add_argument("--out", help="JSON report path (default: eval/results/<golden>-jev-rerank-<timestamp>.json)")
    return p


def main(argv: Optional[list[str]] = None, client: Any = None) -> int:
    """CLI entry point. ``client`` replaces the SDK client (tests); it is ignored with ``--dry-run``."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except (ValueError, OSError):
            pass
    args = build_parser().parse_args(argv)
    if (args.k < 1 or not 1 <= args.candidates <= MAX_CANDIDATES or args.lines_per_candidate < 1
            or args.max_tokens < 1 or args.price_per_m < 0):
        print(f"error: need --k >= 1, 1 <= --candidates <= {MAX_CANDIDATES}, --lines-per-candidate >= 1,"
              " --max-tokens >= 1 and --price-per-m >= 0", file=sys.stderr)
        return 2
    try:
        golden = _runner.load_golden(args.golden)
    except _runner.GoldenError as exc:
        print(f"invalid golden file: {exc}", file=sys.stderr)
        return 2
    errors, warnings = _runner.validate_golden(golden, args.source)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        return 2
    ids = [i.strip() for i in args.ids.split(",") if i.strip()] if args.ids else None
    try:
        _runner.select_questions(golden, [args.dump_state] if args.dump_state else ids)
    except _runner.GoldenError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dump_state:
        return dump_state(golden, args)

    route: Optional[str] = None
    model: Optional[str] = None
    own_client = False
    if args.dry_run:
        client = None
    elif client is None:
        try:
            client, route, model = make_client(model=args.model)
            own_client = True
        except MissingKey as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except ImportError as exc:
            print(f"error: typesafe-sdk could not be imported ({exc});"
                  " typesafe-sdk is installed with RepoFinder; reinstall RepoFinder, or use --dry-run", file=sys.stderr)
            return 1
        print(f"jev: route={route} model={model}", file=sys.stderr)

    try:
        report = run_experiment(
            golden, args.source, client=client, k=args.k, candidates=args.candidates,
            lines_per_candidate=args.lines_per_candidate, allow_none=args.allow_none,
            max_tokens=args.max_tokens, price_per_m=args.price_per_m, ids=ids, log=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - setup failure or an aborting API error (401/402/403)
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if own_client:
            client.close()
    report["meta"].update({"golden": str(args.golden), "validation_warnings": warnings,
                           "route": route, "model": model, "sdk_version": _sdk_version()})

    out = Path(args.out) if args.out else (
        _HERE / "results" / f"{Path(args.golden).stem}-jev-rerank-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.stdout.write(render(report))
    print(f"\nJSON report: {out}")
    return 0


def _sdk_version() -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return None
    try:
        return version("typesafe-sdk")
    except PackageNotFoundError:
        return None


def dump_state(golden: dict[str, Any], args: argparse.Namespace) -> int:
    """Print the state and the choice question of ``--dump-state ID``; Jev is not called."""
    q = _runner.select_questions(golden, [args.dump_state])[0]
    name = (golden.get("repo") or {}).get("name") or Path(args.source).name
    ws = _baseline.search_pipeline(name).open_workspace(args.source)
    reranker = JevReranker(name, Path(args.source), None, candidates=args.candidates,
                           lines_per_candidate=args.lines_per_candidate, allow_none=args.allow_none,
                           max_tokens=args.max_tokens, price_per_m=args.price_per_m)
    hits = ws.search.search(q["question"], k=max(args.k, args.candidates))
    prep = reranker.prepare(ws, q["question"], hits)
    print(f"### state for {q['id']}: {len(prep.candidates)} candidates ({prep.n_dropped} dropped),"
          f" {len(prep.state)} chars, ~{prep.est_tokens} tokens with the question,"
          f" ~{_usd(est_cost(prep.est_tokens, args.price_per_m))}"
          + (f"; SKIPPED: {prep.skipped}" if prep.skipped else ""))
    sys.stdout.write(prep.state)
    print("### questions")
    print(json.dumps(prep.questions, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

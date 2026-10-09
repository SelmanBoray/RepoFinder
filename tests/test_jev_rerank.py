"""Tests for the Jev rerank experiment in ``eval/jev_rerank.py``.

Jev is never called over the network: a fake client stands in for
``typesafe_sdk.TypeSafeClient``, and the SDK-level tests send through an
``httpx2.MockTransport``. The search layer's exact order is not assumed: card
tests build their hits by hand, and the run_eval scenario takes its targets and
expected ranks from the live search through the runner's own matching.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import pytest

from repofinder.ingest import build_index
from repofinder.search import Hit, SearchIndex

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "eval"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


jev = _load_module("jev_rerank", EVAL_DIR / "jev_rerank.py")
runner = jev._runner

ENGINE_Q = "How does the engine run?"
USER_Q = "user model"


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

def loc(path: str, start: int, end: int, symbol: Optional[str] = None) -> dict[str, Any]:
    out: dict[str, Any] = {"path": path, "start_line": start, "end_line": end}
    if symbol:
        out["symbol"] = symbol
    return out


def loc_of(hit: Hit) -> dict[str, Any]:
    return loc(hit.path, hit.start_line, hit.end_line, hit.symbol_id.split("::", 1)[1])


def sample_index(root: Path) -> Any:
    return build_index(root, "sample_repo", str(root))


def search_hits(root: Path, question: str, k: int = 8) -> list[Hit]:
    return SearchIndex(sample_index(root)).search(question, k=k)


def symbol_hit(index: Any, qualname: str) -> Hit:
    sym = next(s for s in index.symbols if s.qualname == qualname)
    return Hit(path=sym.path, symbol_id=sym.id, start_line=sym.start_line, end_line=sym.end_line,
               score=1.0, kind="symbol", why="")


def file_hit(index: Any, path: str) -> Hit:
    record = next(f for f in index.files if f.path == path)
    return Hit(path=path, symbol_id=None, start_line=1, end_line=record.n_lines, score=1.0, kind="file", why="")


def prepare(root: Path, question: str, hits: Optional[list[Hit]] = None, **kw: Any) -> Any:
    index = sample_index(root)
    if hits is None:
        hits = SearchIndex(index).search(question, k=kw.pop("k", 8))
    return jev.prepare("sample_repo", question, hits, lambda p: runner._read_lines(root, p),
                       {s.id: s for s in index.symbols}, **kw)


def hand_hits(root: Path) -> list[Hit]:
    """run_engine, the core.py file, Engine.build: one function, one file, one method."""
    index = sample_index(root)
    return [symbol_hit(index, "run_engine"), file_hit(index, "mypkg/core.py"), symbol_hit(index, "Engine.build")]


def label_for(criteria: dict[str, Any], hit: Hit) -> str:
    """The label whose description is the card header of ``hit``."""
    prefix = f"{hit.path}:{hit.start_line}-{hit.end_line} | "
    return next(label for label, desc in criteria.items()
                if desc and desc.startswith(prefix) and ("whole file" in desc) == (hit.kind == "file"))


def scenario(root: Path) -> SimpleNamespace:
    """A two-question golden set that Jev can fix (q1) and break (q2), from the live search.

    q1 expects the symbol hit that the runner ranks lowest (rank > 1); the fake
    Jev picks it, so it should move to rank 1. q2 expects the first hit; the fake
    picks a symbol hit that does not match it, so it should drop to rank 2.
    """
    hits_e = search_hits(root, ENGINE_Q)
    symbols_e = [h for h in hits_e if h.kind == "symbol"]
    target = max(symbols_e, key=lambda h: runner.first_rank(hits_e, [loc_of(h)]) or 0)
    rank_e = runner.first_rank(hits_e, [loc_of(target)])
    assert rank_e is not None and rank_e > 1, "the sample search no longer has a rank > 1 target"

    hits_u = search_hits(root, USER_Q)
    assert hits_u and hits_u[0].kind == "symbol"
    expected_u = loc_of(hits_u[0])
    wrong = next(h for h in hits_u[1:] if h.kind == "symbol"
                 and runner.best_match(h, [expected_u]) not in runner.STRICT_MATCHES)
    golden = {
        "repo": {"name": "sample_repo", "source": "tests/fixtures/sample_repo", "commit": "abc1234"},
        "questions": [
            {"id": "q1", "question": ENGINE_Q, "lang": "en", "category": "locate-feature",
             "expected": [loc_of(target)], "accept_any": True, "notes": "t"},
            {"id": "q2", "question": USER_Q, "lang": "en", "category": "locate-symbol",
             "expected": [expected_u], "accept_any": True, "notes": "t"},
        ],
    }
    picks = {ENGINE_Q: target, USER_Q: wrong}
    return SimpleNamespace(golden=golden, picks=picks, rank_e=rank_e, target=target,
                           label_e=jev.LABELS[hits_e.index(target)], n_hits_e=len(hits_e),
                           first_e=hits_e[0])


def write_golden(tmp_path: Path, golden: dict[str, Any]) -> Path:
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(golden), encoding="utf-8")
    return path


class FakeJev:
    """Stands in for ``TypeSafeClient``: ``pick(question_text, criteria)`` returns a label."""

    def __init__(self, pick: Callable[[str, dict[str, Any]], str], *, probabilities: bool = False,
                 cost: Optional[float] = 0.0001, error: Optional[BaseException] = None) -> None:
        self.pick = pick
        self.probabilities = probabilities
        self.cost = cost
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def system_one(self, state: str, questions: dict[str, Any]) -> SimpleNamespace:
        self.calls.append((state, questions))
        if self.error is not None:
            raise self.error
        criteria = questions[jev.QUESTION_NAME]["criteria"]
        question = state.splitlines()[1]
        label = self.pick(question, criteria)
        probs: dict[str, float] = {}
        if self.probabilities:
            rest = [c for c in criteria if c != label]
            probs = {c: 0.4 / len(rest) for c in rest} if rest else {}
            probs[label] = 0.6
        usage = {"input_tokens": 1234, "output_tokens": 3}
        raw_usage = dict(usage, **({"cost": self.cost} if self.cost is not None else {}))
        raw = {"model": "jev-test", "usage": raw_usage,
               "answers": {jev.QUESTION_NAME: {"type": "choice", "choice": label, "probabilities": probs}}}
        return SimpleNamespace(
            model="jev-test",
            choices={jev.QUESTION_NAME: SimpleNamespace(choice=label, confidence=0.9, probabilities=probs)},
            usage=SimpleNamespace(**usage),
            raw_http_response=SimpleNamespace(json=lambda: raw),
        )


def fake_for(sc: SimpleNamespace, **kw: Any) -> FakeJev:
    return FakeJev(lambda question, criteria: label_for(criteria, sc.picks[question]), **kw)


class ApiError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

def test_cards_show_location_kind_qualname_and_source_lines(sample_repo):
    prep = prepare(sample_repo, ENGINE_Q, hand_hits(sample_repo))
    a, b, c = prep.candidates
    assert [a.label, b.label, c.label] == ["A", "B", "C"] and prep.n_dropped == 0
    assert a.header == "mypkg/core.py:13-30 | function run_engine" and a.kind == "function"
    assert "def run_engine(\n" in a.card and '"""Run the engine once.' in a.card
    assert b.header == "mypkg/core.py:1-63 | whole file" and b.kind == "file" and b.qualname is None
    assert '"""Core engine of the sample package."""' in b.card
    assert c.header == "mypkg/core.py:48-51 | method Engine.build" and c.qualname == "Engine.build"
    assert "```python\n@classmethod\n@staticmethod\ndef build(cls):" in c.card   # dedented
    assert prep.state.startswith('Question about the "sample_repo" repository:\nHow does the engine run?\n')
    assert "3 candidate code locations, labeled A to C" in prep.state
    q = prep.questions[jev.QUESTION_NAME]
    assert q["type"] == "choice" and q["instructions"] == jev.INSTRUCTIONS
    assert q["criteria"] == {"A": a.header, "B": b.header, "C": c.header}


def test_long_symbols_show_only_their_first_lines(sample_repo):
    prep = prepare(sample_repo, ENGINE_Q, hand_hits(sample_repo), lines_per_candidate=3)
    a, b, c = prep.candidates
    assert a.lines_shown == 3 and a.truncated is True
    assert "# ... 15 more lines (to line 30)" in a.card
    assert "# ... 60 more lines (to line 63)" in b.card
    assert c.lines_shown == 3 and c.truncated is True
    assert a.record()["truncated"] is True and a.record()["search_rank"] == 1


def test_a_missing_source_file_is_marked_not_dropped(sample_repo):
    hit = Hit(path="mypkg/gone.py", symbol_id="mypkg/gone.py::f", start_line=1, end_line=3,
              score=1.0, kind="symbol", why="")
    prep = prepare(sample_repo, ENGINE_Q, [hit])
    assert prep.candidates[0].lines_shown == 0 and "# (source not available)" in prep.candidates[0].card
    assert "1 candidate code location, labeled A" in prep.state


def test_candidates_option_limits_the_cards(sample_repo):
    hits = search_hits(sample_repo, ENGINE_Q)
    prep = prepare(sample_repo, ENGINE_Q, hits, candidates=3)
    assert [c.label for c in prep.candidates] == ["A", "B", "C"] and len(prep.hits) == len(hits)
    assert [c.hit for c in prep.candidates] == hits[:3]
    assert "labeled A to C" in prep.state


def test_state_drops_the_last_candidates_to_fit_the_token_budget(sample_repo):
    hits = search_hits(sample_repo, ENGINE_Q)
    full = prepare(sample_repo, ENGINE_Q, hits)
    assert full.est_tokens == jev.request_tokens(full.state, full.questions)
    budget = full.est_tokens - 1
    cut = prepare(sample_repo, ENGINE_Q, hits, max_tokens=budget)
    assert cut.n_dropped >= 1 and cut.est_tokens <= budget and cut.skipped is None
    assert [c.hit for c in cut.candidates] == hits[: len(cut.candidates)]
    assert len(cut.candidates) + cut.n_dropped == len(full.candidates)


def test_state_over_budget_with_one_candidate_is_skipped_without_a_call(sample_repo):
    prep = prepare(sample_repo, ENGINE_Q, hand_hits(sample_repo), max_tokens=10)
    assert prep.skipped and prep.candidates == [] and prep.n_dropped == 3
    fake = FakeJev(lambda q, c: "A")
    hits, record = jev.JevReranker("sample_repo", sample_repo, fake).decide(prep)
    assert fake.calls == [] and hits == prep.hits and record["skipped"] == prep.skipped
    assert record["est_cost_usd"] == 0.0


def test_no_search_hits_is_skipped(sample_repo):
    prep = prepare(sample_repo, "zzzz", [])
    assert prep.skipped == "no search hits" and prep.state == ""


def test_allow_none_adds_a_none_choice(sample_repo):
    prep = prepare(sample_repo, ENGINE_Q, hand_hits(sample_repo), allow_none=True)
    criteria = prep.questions[jev.QUESTION_NAME]["criteria"]
    assert list(criteria) == ["A", "B", "C", jev.NONE_LABEL]
    assert criteria[jev.NONE_LABEL] == jev.NONE_DESCRIPTION


def test_token_and_cost_estimates():
    assert jev.est_tokens("") == 0 and jev.est_tokens("abcd") == 1 and jev.est_tokens("abcde") == 2
    assert jev.est_cost(1_000_000) == pytest.approx(0.042)
    assert jev.est_cost(2000, price_per_m=1.0) == pytest.approx(0.002)
    assert jev.DEFAULT_MAX_TOKENS == 22_400 and jev.CONTEXT_TOKENS == 32_000


# --------------------------------------------------------------------------- #
# Reranking
# --------------------------------------------------------------------------- #

def _manual_prep(n_hits: int = 5, shown: int = 3) -> Any:
    hits = [Hit(path=f"p{i}.py", symbol_id=f"p{i}.py::f{i}", start_line=1, end_line=5,
                score=10.0 - i, kind="symbol", why=f"search {i}") for i in range(n_hits)]
    cards = [jev.make_card(jev.LABELS[i], i + 1, h, None, ["x"] * 5, 60) for i, h in enumerate(hits[:shown])]
    return jev.Prepared("q", hits, cards, 0, "state", {}, 1)


def test_rerank_with_probabilities_sorts_the_shown_candidates_and_keeps_the_rest():
    prep = _manual_prep()
    new, order = jev.rerank_hits(prep, "C", {"A": 0.1, "B": 0.3, "C": 0.6})
    assert order == ["C", "B", "A"]
    assert [h.symbol_id for h in new] == ["p2.py::f2", "p1.py::f1", "p0.py::f0", "p3.py::f3", "p4.py::f4"]
    assert new[0].why == "jev #1 [C, search #3] p=0.600; search 2"
    assert prep.hits[2].why == "search 2"        # the search hit itself is not changed
    assert new[3] is prep.hits[3]


def test_rerank_puts_the_pick_first_even_if_probabilities_disagree():
    _new, order = jev.rerank_hits(_manual_prep(), "B", {"A": 0.5, "B": 0.4, "C": 0.1})
    assert order == ["B", "A", "C"]


def test_rerank_without_probabilities_moves_only_the_pick():
    new, order = jev.rerank_hits(_manual_prep(), "C", {})
    assert order == ["C", "A", "B"]
    assert [h.symbol_id for h in new][3:] == ["p3.py::f3", "p4.py::f4"]
    assert new[0].why == "jev #1 [C, search #3]; search 2"


def test_rerank_none_pick_keeps_search_order_or_follows_probabilities():
    _new, order = jev.rerank_hits(_manual_prep(), jev.NONE_LABEL, {})
    assert order == ["A", "B", "C"]
    _new, order = jev.rerank_hits(_manual_prep(), jev.NONE_LABEL, {"NONE": 0.7, "A": 0.05, "B": 0.05, "C": 0.2})
    assert order == ["C", "A", "B"]


# --------------------------------------------------------------------------- #
# The experiment (run_eval integration)
# --------------------------------------------------------------------------- #

def test_dry_run_builds_states_estimates_cost_and_never_calls(sample_repo):
    sc = scenario(sample_repo)
    report = jev.run_experiment(sc.golden, str(sample_repo), client=None)
    assert report["after"] is None and report["meta"]["dry_run"] is True
    recs = {r["id"]: r for r in report["jev"]}
    for r in recs.values():
        assert r["n_candidates"] == min(8, r["n_search_hits"])
        expected = math.ceil(r["state_chars"] / 4) + math.ceil(r["questions_chars"] / 4)
        assert r["est_tokens"] == expected
        assert r["est_cost_usd"] == pytest.approx(expected * 0.042 / 1e6)
        assert "choice" not in r and "raw_response" not in r and "rank_after" not in r
    assert recs["q1"]["rank_before"] == sc.rank_e and recs["q2"]["rank_before"] == 1
    cost = report["cost"]
    assert cost["n_requests"] == 2 and "cost_usd_total" not in cost
    assert cost["est_tokens_total"] == sum(r["est_tokens"] for r in recs.values())
    assert cost["est_cost_usd_total"] == pytest.approx(sum(r["est_cost_usd"] for r in recs.values()))


def test_before_is_the_search_baseline_and_after_is_scored_by_run_eval(sample_repo):
    sc = scenario(sample_repo)
    fake = fake_for(sc)
    report = jev.run_experiment(sc.golden, str(sample_repo), client=fake)
    assert len(fake.calls) == 2

    baseline = jev._baseline.search_pipeline("sample_repo")
    reference = runner.run_eval(sc.golden, str(sample_repo), "retrieval", 8, baseline)
    assert [r["retrieval"]["top_hits"] for r in report["before"]["questions"]] == \
        [r["retrieval"]["top_hits"] for r in reference["questions"]]

    recs = {r["id"]: r for r in report["jev"]}
    assert (recs["q1"]["rank_before"], recs["q1"]["rank_after"]) == (sc.rank_e, 1)   # Jev fixed it
    assert (recs["q2"]["rank_before"], recs["q2"]["rank_after"]) == (1, 2)           # Jev broke it
    after_q1 = report["after"]["questions"][0]["retrieval"]
    assert after_q1["hit@1"] is True and after_q1["top_hits"][0]["symbol_id"] == sc.target.symbol_id
    assert after_q1["top_hits"][0]["why"].startswith(f"jev #1 [{sc.label_e}, search #")
    mrr_b = report["before"]["summary"]["overall"]["retrieval"]["mrr"]
    mrr_a = report["after"]["summary"]["overall"]["retrieval"]["mrr"]
    assert mrr_b == pytest.approx((1 / sc.rank_e + 1.0) / 2, abs=1e-3)
    assert mrr_a == pytest.approx((1.0 + 0.5) / 2, abs=1e-3)
    assert recs["q1"]["choice"] == sc.label_e and recs["q1"]["order"][0] == sc.label_e
    assert recs["q1"]["raw_response"]["usage"]["cost"] == 0.0001
    cost = report["cost"]
    assert cost["n_ok"] == 2 and cost["n_errors"] == 0 and cost["n_cost_reported"] == 2
    assert cost["cost_usd_total"] == pytest.approx(0.0002) and cost["input_tokens_total"] == 2468


def test_probabilities_give_a_full_order(sample_repo):
    sc = scenario(sample_repo)
    report = jev.run_experiment(sc.golden, str(sample_repo), client=fake_for(sc, probabilities=True), ids=["q1"])
    rec = report["jev"][0]
    assert rec["order"][0] == sc.label_e and sorted(rec["order"]) == list(jev.LABELS[: sc.n_hits_e])
    assert rec["probabilities"][sc.label_e] == pytest.approx(0.6) and rec["confidence"] == 0.9


def test_candidates_beyond_k_can_move_into_the_top_k(sample_repo):
    # k scores fewer hits than Jev sees; the target is below k in the search order.
    sc = scenario(sample_repo)
    k = sc.rank_e - 1
    report = jev.run_experiment(sc.golden, str(sample_repo), client=fake_for(sc), k=k, ids=["q1"])
    assert report["before"]["questions"][0]["retrieval"]["rank"] is None
    assert report["after"]["questions"][0]["retrieval"]["rank"] == 1
    assert report["after"]["questions"][0]["retrieval"]["n_hits"] == k
    assert report["jev"][0]["n_candidates"] == min(8, report["jev"][0]["n_search_hits"])


def test_missing_usage_cost_falls_back_to_reported_tokens(sample_repo):
    sc = scenario(sample_repo)
    report = jev.run_experiment(sc.golden, str(sample_repo), client=fake_for(sc, cost=None), ids=["q2"],
                                price_per_m=1.0)
    rec = report["jev"][0]
    assert rec["cost_source"] == "usage.input_tokens x price"
    assert rec["cost_usd"] == pytest.approx(1234 / 1e6)
    assert report["cost"]["n_cost_reported"] == 0


def test_a_failed_call_keeps_the_search_order_and_is_recorded(sample_repo):
    sc = scenario(sample_repo)
    report = jev.run_experiment(sc.golden, str(sample_repo),
                                client=fake_for(sc, error=RuntimeError("upstream timeout")))
    recs = {r["id"]: r for r in report["jev"]}
    assert recs["q1"]["error"] == "RuntimeError: upstream timeout"
    assert (recs["q1"]["rank_before"], recs["q1"]["rank_after"]) == (sc.rank_e, sc.rank_e)
    assert report["cost"]["n_errors"] == 2 and report["cost"]["n_ok"] == 0
    assert report["after"]["summary"]["overall"]["errors"] == 0   # run_eval saw no exception


def test_a_choice_outside_the_labels_is_an_error(sample_repo):
    sc = scenario(sample_repo)
    report = jev.run_experiment(sc.golden, str(sample_repo), client=FakeJev(lambda q, c: "Z"), ids=["q2"])
    assert "not one of the labels" in report["jev"][0]["error"]
    assert report["jev"][0]["rank_after"] == 1


@pytest.mark.parametrize("status", [401, 402, 403])
def test_an_auth_or_credit_error_aborts_after_one_call(sample_repo, status):
    sc = scenario(sample_repo)
    fake = fake_for(sc, error=ApiError(status))
    with pytest.raises(ApiError):
        jev.run_experiment(sc.golden, str(sample_repo), client=fake)
    assert len(fake.calls) == 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

@pytest.fixture
def no_keys(monkeypatch):
    for name in (jev.OPENROUTER_KEY_ENV, jev.TYPESAFE_KEY_ENV, "TYPESAFE_DEFAULT_MODEL", "TYPESAFE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


def test_main_without_a_key_or_dry_run_exits_1(tmp_path, capsys, sample_repo, no_keys):
    out = tmp_path / "r.json"
    golden = write_golden(tmp_path, scenario(sample_repo).golden)
    code = jev.main([str(golden), "--source", str(sample_repo), "--out", str(out)])
    assert code == 1 and not out.exists()
    assert "set OPENROUTER_API_KEY or TYPESAFE_API_KEY, or use --dry-run" in capsys.readouterr().err


def test_main_dry_run_writes_the_report_and_prints_estimates(tmp_path, capsys, sample_repo, no_keys):
    sc = scenario(sample_repo)
    out = tmp_path / "r.json"
    code = jev.main([str(write_golden(tmp_path, sc.golden)), "--source", str(sample_repo), "--dry-run",
                     "--out", str(out)])
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["meta"]["dry_run"] is True and report["after"] is None
    assert [r["id"] for r in report["jev"]] == ["q1", "q2"] and report["jev"][0]["est_tokens"] > 0
    text = capsys.readouterr().out
    assert "DRY RUN" in text and f"| q1 | locate-feature | en | {min(8, sc.n_hits_e)} (0) |" in text
    assert "## Before (search, k=8)" in text and "hit@1 0.50 (1/2)" in text
    assert "estimated cost: mean $" in text


def test_main_dry_run_ignores_an_injected_client(tmp_path, sample_repo, no_keys):
    sc = scenario(sample_repo)
    fake = fake_for(sc)
    code = jev.main([str(write_golden(tmp_path, sc.golden)), "--source", str(sample_repo), "--dry-run",
                     "--out", str(tmp_path / "r.json")], client=fake)
    assert code == 0 and fake.calls == []


def test_main_with_a_client_prints_before_and_after(tmp_path, capsys, sample_repo, no_keys):
    sc = scenario(sample_repo)
    out = tmp_path / "r.json"
    code = jev.main([str(write_golden(tmp_path, sc.golden)), "--source", str(sample_repo), "--out", str(out)],
                    client=fake_for(sc))
    assert code == 0
    text = capsys.readouterr().out
    assert f"| q1 | locate-feature | en | {sc.rank_e} | 1 | {sc.label_e} |" in text
    assert "| q2 | locate-symbol | en | 1 | 2 |" in text
    assert "| overall | 2 | 0.50 (1/2) -> 0.50 (1/2) |" in text
    assert "usage.cost reported in 2/2 responses" in text
    assert json.loads(out.read_text(encoding="utf-8"))["after"]["meta"]["mode"] == "retrieval"


def test_main_dump_state_prints_the_state_and_writes_nothing(tmp_path, capsys, sample_repo, no_keys):
    sc = scenario(sample_repo)
    code = jev.main([str(write_golden(tmp_path, sc.golden)), "--source", str(sample_repo), "--dump-state", "q1"])
    assert code == 0
    text = capsys.readouterr().out
    assert text.startswith(f"### state for q1: {min(8, sc.n_hits_e)} candidates (0 dropped)")
    first = sc.first_e
    assert f"[A] {first.path}:{first.start_line}-{first.end_line} | " in text and "```python\n" in text
    assert '"best_candidate"' in text and '"type": "choice"' in text
    assert list(tmp_path.glob("*.json")) == [tmp_path / "sample.json"]


@pytest.mark.parametrize("args, code", [
    (["--candidates", "17"], 2), (["--candidates", "0"], 2), (["--k", "0"], 2),
    (["--lines-per-candidate", "0"], 2), (["--ids", "q9"], 2), (["--dump-state", "q9"], 2),
])
def test_main_rejects_bad_arguments(tmp_path, sample_repo, no_keys, args, code):
    golden = write_golden(tmp_path, scenario(sample_repo).golden)
    assert jev.main([str(golden), "--source", str(sample_repo), "--dry-run", *args]) == code


# --------------------------------------------------------------------------- #
# The real SDK, with a mock transport (no network)
# --------------------------------------------------------------------------- #

def test_make_client_without_keys_raises(no_keys):
    with pytest.raises(jev.MissingKey, match="OPENROUTER_API_KEY or TYPESAFE_API_KEY"):
        jev.make_client({})


def test_make_client_routes(monkeypatch, no_keys):
    sdk = pytest.importorskip("typesafe_sdk")
    seen: list[dict[str, Any]] = []

    class Recorder:
        def __init__(self, **kwargs: Any) -> None:
            seen.append(kwargs)

    monkeypatch.setattr(sdk, "TypeSafeClient", Recorder)
    _client, route, model = jev.make_client({"OPENROUTER_API_KEY": "sk-or", "TYPESAFE_API_KEY": "ts"})
    assert (route, model) == ("openrouter", "~typesafe/jev-latest")
    assert seen[-1]["api_key"] == "sk-or" and seen[-1]["base_url"] == "https://openrouter.ai/api"
    _client, route, model = jev.make_client({"TYPESAFE_API_KEY": "ts"}, model="jev-1")
    assert (route, model) == ("typesafe", "jev-1")
    assert seen[-1]["api_key"] == "ts" and "base_url" not in seen[-1]
    _client, route, model = jev.make_client({"TYPESAFE_API_KEY": "ts"})
    assert model == sdk.constants.DEFAULT_MODEL


def test_the_real_sdk_client_sends_our_request_and_we_read_its_response(sample_repo):
    sdk = pytest.importorskip("typesafe_sdk")
    httpx2 = pytest.importorskip("httpx2")
    sent: list[Any] = []

    def handler(request: Any) -> Any:
        sent.append(request)
        return httpx2.Response(200, json={
            "model": "typesafe/jev-1.13",
            "answers": {jev.QUESTION_NAME: {"type": "choice", "choice": "C", "confidence": 0.8,
                                            "probabilities": {"A": 0.1, "C": 0.8, "B": 0.1}}},
            "usage": {"input_tokens": 900, "output_tokens": 1, "cost": 0.0000378},
        })

    client = sdk.TypeSafeClient(api_key="sk-or-test", base_url=jev.OPENROUTER_BASE_URL,
                                model=jev.OPENROUTER_MODEL, transport=httpx2.MockTransport(handler),
                                retry=sdk.RetryPolicy(max_retries=0))
    prep = prepare(sample_repo, ENGINE_Q, hand_hits(sample_repo))
    with client:
        new_hits, record = jev.JevReranker("sample_repo", sample_repo, client).decide(prep)

    (request,) = sent
    assert str(request.url) == "https://openrouter.ai/api/v1/systemone"
    assert request.headers["authorization"] == "Bearer sk-or-test"
    body = json.loads(request.content)
    assert body["model"] == "~typesafe/jev-latest" and body["state"] == prep.state
    assert body["questions"] == prep.questions
    assert record["error"] is None and record["choice"] == "C" and record["model"] == "typesafe/jev-1.13"
    assert record["probabilities"] == {"A": 0.1, "C": 0.8, "B": 0.1} and record["confidence"] == 0.8
    assert (record["input_tokens"], record["output_tokens"]) == (900, 1)
    assert record["cost_usd"] == pytest.approx(0.0000378) and record["cost_source"] == "usage.cost"
    assert record["raw_response"]["usage"]["cost"] == 0.0000378
    assert record["order"] == ["C", "A", "B"]
    assert new_hits[0].symbol_id == "mypkg/core.py::Engine.build"

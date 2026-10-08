"""Tests for the evaluation harness in ``eval/`` (runner, matching, golden set).

The real pipeline and any LLM are never called: runs use a stub with the same
three names as ``repofinder.pipeline`` (``open_workspace``, ``retrieve``, ``ask``).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from repofinder.llm import Answer, Citation, ContextChunk
from repofinder.search import Hit

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "eval"
FLASK_GOLDEN = EVAL_DIR / "golden" / "flask.json"
FLASK_CHECKOUT = Path(
    os.environ.get(
        "REPOFINDER_FLASK_CHECKOUT",
        r"C:\Users\Selman\AppData\Local\Temp\claude\C--\2893ddbc-306e-4b5d-94d8-359c5612cb51"
        r"\scratchpad\repos\flask",
    )
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


run_eval = _load_module("run_eval", EVAL_DIR / "run_eval.py")


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

def hit(path: str, start: int, end: int, symbol: Optional[str] = None,
        kind: str = "symbol", score: float = 1.0) -> Hit:
    return Hit(path=path, symbol_id=f"{path}::{symbol}" if symbol else None,
               start_line=start, end_line=end, score=score, kind=kind, why="")


def file_hit(path: str, n_lines: int) -> Hit:
    return hit(path, 1, n_lines, kind="file")


def chunk(path: str, start: int, end: int) -> ContextChunk:
    return ContextChunk(path=path, start_line=start, end_line=end, text="")


def loc(path: str, start: int, end: int, symbol: Optional[str] = None) -> dict[str, Any]:
    out: dict[str, Any] = {"path": path, "start_line": start, "end_line": end}
    if symbol:
        out["symbol"] = symbol
    return out


def question(qid: str, text: Optional[str] = None, *, category: str = "locate-symbol",
             expected: Optional[list] = None, accept_any: bool = True, lang: str = "en",
             pointer: Optional[list] = None, related: Optional[list] = None) -> dict[str, Any]:
    q: dict[str, Any] = {
        "id": qid,
        "question": text or f"question {qid}?",
        "lang": lang,
        "category": category,
        "expected": expected if expected is not None else [loc("pkg/a.py", 10, 20, "A.run")],
        "accept_any": accept_any,
        "notes": "test",
    }
    if pointer is not None:
        q["pointer"] = pointer
    if related is not None:
        q["related"] = related
    return q


def golden(*questions: dict[str, Any]) -> dict[str, Any]:
    return {"repo": {"name": "demo", "source": "https://github.com/x/demo", "commit": "abc1234"},
            "questions": list(questions)}


def answer(text: str, citations: list[tuple[str, int, int]], *, inp: int = 100,
           out: int = 20, cache: int = 5) -> Answer:
    return Answer(text=text, citations=[Citation(p, s, e) for p, s, e in citations],
                  model="stub-model", input_tokens=inp, output_tokens=out, cache_read_tokens=cache)


class StubPipeline:
    """Implements the pipeline contract with canned data keyed by question text."""

    def __init__(self, retrieval: Optional[dict[str, tuple[list, list]]] = None,
                 answers: Optional[dict[str, Answer]] = None,
                 fail: frozenset[str] = frozenset()) -> None:
        self.retrieval = retrieval or {}
        self.answers = answers or {}
        self.fail = fail
        self.calls: list[tuple] = []

    def open_workspace(self, source: str, data_dir: Optional[str] = None, refresh: bool = False) -> str:
        self.calls.append(("open", source, data_dir, refresh))
        return "WS"

    def retrieve(self, ws: str, question: str, k: int = 8) -> tuple[list, list]:
        self.calls.append(("retrieve", ws, question, k))
        if question in self.fail:
            raise RuntimeError("boom")
        return self.retrieval.get(question, ([], []))

    def ask(self, ws: str, question: str, k: int = 8) -> SimpleNamespace:
        self.calls.append(("ask", ws, question, k))
        if question in self.fail:
            raise RuntimeError("boom")
        chunks, hits = self.retrieval.get(question, ([], []))
        return SimpleNamespace(answer=self.answers.get(question, answer("", [])),
                               hits=hits, chunks=chunks, elapsed_s=0.25)


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #

EXP = loc("src/app.py", 100, 130, "Flask.run")


def test_match_same_path_overlapping_lines():
    assert run_eval.match_hit(hit("src/app.py", 125, 140, "Flask.other"), EXP) == "overlap"


def test_match_requires_same_path_and_overlap():
    assert run_eval.match_hit(hit("src/other.py", 100, 130, "X.y"), EXP) is None
    assert run_eval.match_hit(hit("src/app.py", 131, 150, "Flask.other"), EXP) is None
    assert run_eval.match_hit(hit("src/app.py", 90, 99, "Flask.other"), EXP) is None


def test_match_by_symbol_even_when_lines_drifted():
    assert run_eval.match_hit(hit("src/app.py", 300, 330, "Flask.run"), EXP) == "symbol"
    # same qualname in another file is not the expected symbol
    assert run_eval.match_hit(hit("src/cli.py", 100, 130, "Flask.run"), EXP) is None


def test_file_level_hit_matches_only_loosely():
    h = file_hit("src/app.py", 1600)
    assert run_eval.match_hit(h, EXP) == "file"
    assert run_eval.first_rank([h], [EXP], strict=True) is None
    assert run_eval.first_rank([h], [EXP], strict=False) == 1


def test_broad_container_is_loose_but_small_container_is_strict():
    whole_class = hit("src/app.py", 50, 1500, "Flask")
    assert run_eval.match_hit(whole_class, EXP) == "broad"
    near = hit("src/app.py", 95, 140, "Flask.run_wrapper")
    assert run_eval.match_hit(near, EXP) == "overlap"
    # a 2-line expected location inside a 50-line function: not broad (<= BROAD_MIN_SPAN)
    tiny = loc("src/app.py", 120, 121)
    assert run_eval.match_hit(hit("src/app.py", 100, 149, "f"), tiny) == "overlap"
    # a nested hit inside the expected range is precise
    assert run_eval.match_hit(hit("src/app.py", 110, 112, "Flask.run.<locals>.inner"), EXP) == "overlap"


def test_best_match_prefers_strict_over_loose():
    h = hit("src/app.py", 50, 1500, "Flask")
    locs = [loc("src/app.py", 100, 130), loc("src/app.py", 60, 1400)]
    assert run_eval.best_match(h, locs) == "overlap"


def test_hit_at_and_reciprocal_rank():
    q = question("q1", expected=[EXP])
    hits = [hit("src/x.py", 1, 5, "x"), hit("src/app.py", 100, 130, "Flask.run"), hit("src/y.py", 1, 5, "y")]
    r = run_eval.score_retrieval(q, hits, [], k=8)
    assert (r["rank"], r["hit@1"], r["hit@3"], r["hit_at_k"]) == (2, False, True, True)
    assert r["rr"] == pytest.approx(0.5)
    assert [h["match"] for h in r["top_hits"]] == [None, "symbol", None]


def test_hits_beyond_k_are_ignored():
    q = question("q1", expected=[EXP])
    hits = [hit("src/x.py", i, i, f"x{i}") for i in range(1, 4)] + [hit("src/app.py", 100, 130, "Flask.run")]
    r = run_eval.score_retrieval(q, hits, [], k=3)
    assert r["rank"] is None and r["rr"] == 0.0 and r["n_hits"] == 3
    r1 = run_eval.score_retrieval(q, [hit("src/x.py", 1, 1, "x"), hit("src/app.py", 100, 130, "Flask.run")], [], k=1)
    assert r1["hit@3"] is False  # hit@3 is computed over hits[:min(3, k)]


def test_chunk_recall_accept_any_versus_all_required():
    a, b = loc("src/a.py", 10, 19), loc("src/b.py", 1, 10)
    chunks = [chunk("src/a.py", 1, 40)]
    q_any = question("q1", expected=[a, b], accept_any=True)
    q_all = question("q2", expected=[a, b], accept_any=False)
    assert run_eval.score_retrieval(q_any, [], chunks, 8)["chunk_recall"] is True
    assert run_eval.score_retrieval(q_all, [], chunks, 8)["chunk_recall"] is False
    both = chunks + [chunk("src/b.py", 1, 10)]
    r = run_eval.score_retrieval(q_all, [], both, 8)
    assert r["chunk_recall"] is True and r["chunk_coverage"] == [1.0, 1.0]


def test_chunk_coverage_threshold_and_union():
    target = loc("src/a.py", 1, 10)
    assert run_eval.location_coverage([chunk("src/a.py", 1, 4)], target) == pytest.approx(0.4)
    assert run_eval.chunks_cover([chunk("src/a.py", 1, 4)], [target], True) is False
    two = [chunk("src/a.py", 1, 3), chunk("src/a.py", 6, 8)]  # 6 of 10 lines
    assert run_eval.location_coverage(two, target) == pytest.approx(0.6)
    assert run_eval.chunks_cover(two, [target], True) is True
    assert run_eval.location_coverage([chunk("src/b.py", 1, 10)], target) == 0.0


def test_all_at_k_requires_every_expected_location():
    a, b = loc("src/a.py", 10, 19, "A.f"), loc("src/b.py", 1, 10, "B.g")
    q = question("q1", category="flow", expected=[a, b], accept_any=False)
    one = run_eval.score_retrieval(q, [hit("src/a.py", 10, 19, "A.f")], [], 8)
    assert one["hit@1"] is True and one["all_at_k"] is False
    both = run_eval.score_retrieval(q, [hit("src/a.py", 10, 19, "A.f"), hit("src/b.py", 1, 10, "B.g")], [], 8)
    assert both["all_at_k"] is True
    assert run_eval.score_retrieval(question("q2", expected=[a]), [], [], 8)["all_at_k"] is None


def test_not_in_repo_retrieval_scores_pointer_only():
    ptr = loc("src/ctx.py", 405, 414, "AppContext.match_request")
    q = question("q1", category="not-in-repo", expected=[], pointer=[ptr])
    r = run_eval.score_retrieval(q, [hit("src/x.py", 1, 2, "x"), hit("src/ctx.py", 405, 414, "AppContext.match_request")],
                                 [chunk("src/ctx.py", 400, 420)], 8)
    assert r["hit@1"] is None and r["rr"] is None and r["chunk_recall"] is None
    assert r["pointer_rank"] == 2 and r["pointer_at_k"] is True and r["pointer_chunk_recall"] is True


# --------------------------------------------------------------------------- #
# Answer metrics
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text",
    [
        "The excerpts do not contain the URL matching code.",
        "This is not shown in the excerpts; Werkzeug implements it.",
        "The provided excerpts don\u2019t contain it.",
        "Bu kod alıntılarda yok; Werkzeug tarafında.",
        "ALINTILARDA YOK, sadece çağrı yeri var.",
        "Derleme kodu bu depoda değil.",
        "Lexer kısmı burada GÖRÜNMÜYOR.",
    ],
)
def test_admits_missing_detects_en_and_tr_phrases(text):
    assert run_eval.admits_missing(text) is not None


@pytest.mark.parametrize(
    "text",
    ["It is implemented in `src/flask/app.py:120-134`.", "Rota kaydı `Scaffold.route` içinde yapılıyor.", ""],
)
def test_admits_missing_ignores_direct_answers(text):
    assert run_eval.admits_missing(text) is None


def test_score_answer_correct_wrong_and_related_citations():
    q = question("q1", expected=[EXP], related=[loc("src/helpers.py", 1, 40)])
    a = answer("Flask.run starts the server.", [
        ("src/app.py", 120, 125),     # inside expected
        ("src/helpers.py", 10, 12),   # related: neither correct nor wrong
        ("src/cli.py", 5, 9),         # wrong
    ])
    r = run_eval.score_answer(q, a)
    assert r["cited_correct"] is True
    assert r["cited_wrong"] == 1
    assert [c["verdict"] for c in r["citations"]] == ["expected", "related", "wrong"]
    assert r["all_required"] is None and r["admits_missing"] is False
    assert (r["input_tokens"], r["output_tokens"], r["cache_read_tokens"]) == (100, 20, 5)
    assert r["answer_text"] == "Flask.run starts the server."


def test_score_answer_broad_citation_is_not_correct():
    q = question("q1", expected=[loc("src/app.py", 100, 110)])
    r = run_eval.score_answer(q, answer("see", [("src/app.py", 1, 900)]))
    assert r["cited_correct"] is False and r["cited_wrong"] == 0
    assert r["citations"][0]["verdict"] == "expected-broad"


def test_score_answer_all_required_for_flow_questions():
    a_loc, b_loc = loc("src/a.py", 1, 10), loc("src/b.py", 20, 30)
    q = question("q1", category="flow", expected=[a_loc, b_loc], accept_any=False)
    partial = run_eval.score_answer(q, answer("x", [("src/a.py", 2, 3)]))
    assert partial["cited_correct"] is True and partial["all_required"] is False
    full = run_eval.score_answer(q, answer("x", [("src/a.py", 2, 3), ("src/b.py", 25, 25)]))
    assert full["all_required"] is True


def test_score_answer_not_in_repo_admission_and_pointer():
    ptr = loc("src/ctx.py", 405, 414)
    q = question("q1", category="not-in-repo", expected=[], pointer=[ptr])
    good = run_eval.score_answer(q, answer("The excerpts do not contain the matcher; see `src/ctx.py:410`.",
                                           [("src/ctx.py", 410, 410)]))
    assert good["admits_missing"] is True and good["pointer_cited"] is True
    assert good["cited_correct"] is None and good["cited_wrong"] == 0
    bad = run_eval.score_answer(q, answer("It is in `src/app.py:5`.", [("src/app.py", 5, 5)]))
    assert bad["admits_missing"] is False and bad["pointer_cited"] is False and bad["cited_wrong"] == 1


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def _row(qid: str, category: str, lang: str, retrieval: Optional[dict] = None,
         ans: Optional[dict] = None, elapsed: float = 1.0) -> dict[str, Any]:
    return {"id": qid, "category": category, "lang": lang, "retrieval": retrieval,
            "answer": ans, "elapsed_s": elapsed, "error": None}


def test_aggregate_hit_rates_and_mrr():
    target = loc("src/app.py", 100, 130, "Flask.run")
    q = question("q", expected=[target])
    filler = [hit("src/x.py", i, i, f"x{i}") for i in range(1, 8)]

    def ranked(rank: Optional[int]) -> dict:
        hits = list(filler)
        if rank is not None:
            hits.insert(rank - 1, hit("src/app.py", 100, 130, "Flask.run"))
        return run_eval.score_retrieval(q, hits, [], k=8)

    rows = [
        _row("a", "locate-symbol", "en", ranked(1)),
        _row("b", "locate-symbol", "en", ranked(2)),
        _row("c", "flow", "tr", ranked(None)),
        _row("d", "flow", "tr", ranked(4)),
        _row("e", "not-in-repo", "en", run_eval.score_retrieval(
            question("e", category="not-in-repo", expected=[], pointer=[target]), [], [], 8)),
    ]
    s = run_eval.aggregate(rows)
    ret = s["overall"]["retrieval"]
    assert ret["hit@1"] == {"n": 4, "yes": 1, "rate": 0.25}
    assert ret["hit@3"]["yes"] == 2 and ret["hit@k"]["yes"] == 3
    assert ret["mrr"] == pytest.approx((1 + 0.5 + 0 + 0.25) / 4, abs=1e-3)
    assert ret["pointer@k"] == {"n": 1, "yes": 0, "rate": 0.0}
    assert s["by_category"]["flow"]["retrieval"]["mrr"] == pytest.approx(0.125, abs=1e-3)
    assert s["by_lang"]["tr"]["retrieval"]["hit@k"]["yes"] == 1
    assert s["overall"]["elapsed_s"]["median"] == 1.0
    assert list(s["by_category"]) == ["locate-symbol", "flow", "not-in-repo"]


def test_aggregate_answer_metrics():
    rows = [
        _row("a", "locate-symbol", "en", ans={"cited_correct": True, "cited_wrong": 0, "all_required": None,
                                               "admits_missing": False, "pointer_cited": None,
                                               "input_tokens": 10, "output_tokens": 2, "cache_read_tokens": 1}),
        _row("b", "flow", "en", ans={"cited_correct": False, "cited_wrong": 3, "all_required": False,
                                     "admits_missing": True, "pointer_cited": None,
                                     "input_tokens": 10, "output_tokens": 2, "cache_read_tokens": 1}),
        _row("c", "not-in-repo", "tr", ans={"cited_correct": None, "cited_wrong": 1, "all_required": None,
                                            "admits_missing": True, "pointer_cited": True,
                                            "input_tokens": 10, "output_tokens": 2, "cache_read_tokens": 1}),
    ]
    a = run_eval.aggregate(rows)["overall"]["answer"]
    assert a["cited_correct"] == {"n": 2, "yes": 1, "rate": 0.5}
    assert a["cited_wrong_total"] == 4 and a["cited_wrong_mean"] == pytest.approx(4 / 3, abs=1e-3)
    assert a["all_required"] == {"n": 1, "yes": 0, "rate": 0.0}
    assert a["admits_missing"] == {"n": 1, "yes": 1, "rate": 1.0}
    assert a["false_admits"] == {"n": 2, "yes": 1, "rate": 0.5}
    assert a["pointer_cited"]["yes"] == 1
    assert (a["input_tokens"], a["output_tokens"], a["cache_read_tokens"]) == (30, 6, 3)


# --------------------------------------------------------------------------- #
# run_eval with a stub pipeline
# --------------------------------------------------------------------------- #

def _stub_golden() -> tuple[dict, StubPipeline]:
    q1 = question("q1", "Where is run?", expected=[EXP])
    q2 = question("q2", "How does a request flow?", category="flow", accept_any=False,
                  expected=[loc("src/app.py", 100, 130, "Flask.run"), loc("src/ctx.py", 1, 20, "push")])
    q3 = question("q3", "How is URL matching done?", category="not-in-repo", expected=[],
                  pointer=[loc("src/ctx.py", 30, 40)], lang="tr")
    stub = StubPipeline(
        retrieval={
            "Where is run?": ([chunk("src/app.py", 100, 130)], [hit("src/app.py", 100, 130, "Flask.run")]),
            "How does a request flow?": ([chunk("src/app.py", 90, 140)],
                                         [hit("src/x.py", 1, 2, "x"), hit("src/app.py", 100, 130, "Flask.run")]),
            "How is URL matching done?": ([chunk("src/ctx.py", 30, 40)], [hit("src/ctx.py", 30, 40, "match")]),
        },
        answers={
            "Where is run?": answer("In `src/app.py:100-130`.", [("src/app.py", 100, 130)]),
            "How does a request flow?": answer("x", [("src/app.py", 101, 102), ("src/zzz.py", 1, 1)]),
            "How is URL matching done?": answer("Alıntılarda yok; `src/ctx.py:35` çağırıyor.",
                                                [("src/ctx.py", 35, 35)]),
        },
    )
    return copy.deepcopy(golden(q1, q2, q3)), stub


def test_run_eval_retrieval_mode_calls_only_retrieve():
    g, stub = _stub_golden()
    report = run_eval.run_eval(g, "SRC", "retrieval", 5, stub)
    assert stub.calls[0] == ("open", "SRC", None, False)
    assert [c[0] for c in stub.calls[1:]] == ["retrieve"] * 3
    assert all(c[3] == 5 for c in stub.calls[1:])
    rows = {r["id"]: r for r in report["questions"]}
    assert rows["q1"]["retrieval"]["hit@1"] is True and rows["q1"].get("answer") is None
    assert rows["q2"]["retrieval"]["rank"] == 2 and rows["q2"]["retrieval"]["all_at_k"] is False
    assert rows["q2"]["retrieval"]["chunk_recall"] is False
    assert rows["q3"]["retrieval"]["pointer_at_k"] is True
    assert report["meta"]["k"] == 5 and report["meta"]["mode"] == "retrieval"
    assert report["summary"]["overall"]["retrieval"]["mrr"] == pytest.approx(0.75)


def test_run_eval_both_mode_scores_ask_results_with_one_call_each():
    g, stub = _stub_golden()
    report = run_eval.run_eval(g, "SRC", "both", 8, stub, data_dir="D")
    assert stub.calls[0] == ("open", "SRC", "D", False)
    assert [c[0] for c in stub.calls[1:]] == ["ask"] * 3
    rows = {r["id"]: r for r in report["questions"]}
    assert rows["q1"]["answer"]["cited_correct"] is True and rows["q1"]["retrieval"]["hit@1"] is True
    assert rows["q2"]["answer"]["all_required"] is False and rows["q2"]["answer"]["cited_wrong"] == 1
    assert rows["q3"]["answer"]["admits_missing"] is True and rows["q3"]["answer"]["pointer_cited"] is True
    assert rows["q3"]["pipeline_elapsed_s"] == 0.25
    summary = report["summary"]["overall"]["answer"]
    assert summary["input_tokens"] == 300 and summary["admits_missing"]["rate"] == 1.0


def test_run_eval_answer_mode_has_no_retrieval_scores():
    g, stub = _stub_golden()
    report = run_eval.run_eval(g, "SRC", "answer", 8, stub)
    assert all(r.get("retrieval") is None for r in report["questions"])
    assert "retrieval" not in report["summary"]["overall"]


def test_run_eval_records_a_failing_question_and_continues():
    g, stub = _stub_golden()
    stub.fail = frozenset({"Where is run?"})
    report = run_eval.run_eval(g, "SRC", "both", 8, stub)
    rows = {r["id"]: r for r in report["questions"]}
    assert rows["q1"]["error"] == "RuntimeError: boom"
    assert rows["q1"]["retrieval"]["hit@1"] is False and rows["q1"]["answer"]["cited_correct"] is False
    assert rows["q2"]["error"] is None
    assert report["summary"]["overall"]["errors"] == 1


class _Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_run_eval_checks_the_checkout_commit_and_closes_the_workspace(tmp_path):
    import io

    g, stub = _stub_golden()
    _write_git_head(tmp_path, "999888777")
    ws = _Workspace(tmp_path)
    stub.open_workspace = lambda source, **kw: ws  # type: ignore[method-assign]
    log = io.StringIO()
    report = run_eval.run_eval(g, "SRC", "retrieval", 8, stub, log=log)
    assert ws.closed is True
    assert report["meta"]["checkout"] == {"root": str(tmp_path), "commit": "999888777",
                                          "golden_commit": "abc1234", "matches_golden": False}
    assert "line-based scores are unreliable" in log.getvalue()

    _write_git_head(tmp_path, "abc1234ffff")
    ws.closed = False
    report = run_eval.run_eval(g, "SRC", "retrieval", 8, stub, log=io.StringIO())
    assert report["meta"]["checkout"]["matches_golden"] is True and ws.closed is True


def test_run_eval_without_a_workspace_root_records_no_commit():
    g, stub = _stub_golden()
    report = run_eval.run_eval(g, "SRC", "retrieval", 8, stub)
    assert report["meta"]["checkout"]["matches_golden"] is None


def test_run_eval_ids_filter_and_unknown_ids():
    g, stub = _stub_golden()
    report = run_eval.run_eval(g, "SRC", "retrieval", 8, stub, ids=["q3", "q1"])
    assert [r["id"] for r in report["questions"]] == ["q1", "q3"]
    with pytest.raises(run_eval.GoldenError, match="q9"):
        run_eval.run_eval(g, "SRC", "retrieval", 8, stub, ids=["q9"])


def test_run_eval_rejects_bad_mode_and_k():
    g, stub = _stub_golden()
    with pytest.raises(ValueError):
        run_eval.run_eval(g, "SRC", "everything", 8, stub)
    with pytest.raises(ValueError):
        run_eval.run_eval(g, "SRC", "retrieval", 0, stub)


def test_render_markdown_table_and_summary():
    g, stub = _stub_golden()
    md = run_eval.render_markdown(run_eval.run_eval(g, "SRC", "both", 8, stub))
    assert "| id | category | lang | hit@1 | hit@3 | chunk_recall | cited_correct | cited_wrong" in md
    assert "| q1 | locate-symbol | en | 1 | 1 | 1 | 1 | 0 |" in md
    assert "admit=1 ptr_cited=1" in md
    assert "MRR" in md and "category: flow" in md and "lang: tr" in md


# --------------------------------------------------------------------------- #
# Golden validation
# --------------------------------------------------------------------------- #

def test_validate_golden_accepts_a_good_set():
    g, _stub = _stub_golden()
    assert run_eval.validate_golden(g) == ([], [])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda g: g["questions"].append(dict(g["questions"][0])), "duplicate id"),
        (lambda g: g["questions"][0].update(category="where-is"), "'category' must be one of"),
        (lambda g: g["questions"][0].update(lang="de"), "'lang' must be one of"),
        (lambda g: g["questions"][0].update(accept_any="yes"), "'accept_any' must be true or false"),
        (lambda g: g["questions"][0]["expected"][0].update(start_line=50, end_line=40), "bad line range"),
        (lambda g: g["questions"][0]["expected"][0].update(start_line=0), "bad line range"),
        (lambda g: g["questions"][0]["expected"][0].update(path="src\\app.py"), "forward slashes"),
        (lambda g: g["questions"][0].update(expected=[]), "'expected' is empty"),
        (lambda g: g["questions"][2].update(expected=[loc("a.py", 1, 2)]), "must have expected = []"),
        (lambda g: g["questions"][2].pop("pointer"), "needs a 'pointer'"),
        (lambda g: g["questions"][0].update(question=" "), "'question' must be a non-empty string"),
        (lambda g: g.pop("repo"), "'repo' must be an object"),
        (lambda g: g.update(questions=[]), "'questions' must be a non-empty list"),
    ],
)
def test_validate_golden_reports_errors(mutate, message):
    g, _stub = _stub_golden()
    mutate(g)
    errors, _warnings = run_eval.validate_golden(g)
    assert any(message in e for e in errors), errors


def test_validate_golden_warns_about_typos_and_weak_flow_questions():
    g, _stub = _stub_golden()
    g["questions"][0]["accept-any"] = True
    g["questions"][1]["expected"] = g["questions"][1]["expected"][:1]
    errors, warnings = run_eval.validate_golden(g)
    assert errors == []
    assert any("unknown key 'accept-any'" in w for w in warnings)
    assert any("fewer than two expected" in w for w in warnings)


def _write_git_head(root: Path, sha: str) -> None:
    (root / ".git" / "refs" / "heads").mkdir(parents=True, exist_ok=True)
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (root / ".git" / "refs" / "heads" / "main").write_text(sha + "\n", encoding="utf-8")


@pytest.fixture
def source_repo(make_repo) -> Path:
    root = make_repo({
        "pkg/a.py": (
            "import os\n"                      # 1
            "\n"                               # 2
            "\n"                               # 3
            "class A:\n"                       # 4
            "    @property\n"                  # 5
            "    def run(self):\n"             # 6
            "        return 1\n"               # 7
            "\n"                               # 8
            "    async def go(self):\n"        # 9
            "        return 2\n"               # 10
        ),
    })
    _write_git_head(root, "abc1234def5678")
    return root


def _src_golden(*locations: dict[str, Any]) -> dict[str, Any]:
    return golden(question("q1", expected=list(locations)))


def test_validate_golden_against_source_accepts_correct_locations(source_repo):
    g = _src_golden(loc("pkg/a.py", 5, 7, "A.run"), loc("pkg/a.py", 9, 10, "A.go"),
                    loc("pkg/a.py", 4, 10, "A"), loc("pkg/a.py", 1, 1))
    assert run_eval.validate_golden(g, source_repo) == ([], [])


def test_validate_golden_against_source_errors(source_repo):
    errors, _ = run_eval.validate_golden(_src_golden(loc("pkg/missing.py", 1, 2)), source_repo)
    assert any("does not exist" in e for e in errors)
    errors, _ = run_eval.validate_golden(_src_golden(loc("pkg/a.py", 5, 11, "A.run")), source_repo)
    assert any("past the end of the file (10 lines)" in e for e in errors)


def test_validate_golden_against_source_symbol_warnings(source_repo):
    errors, warnings = run_eval.validate_golden(_src_golden(loc("pkg/a.py", 9, 10, "A.run")), source_repo)
    assert errors == [] and any("no 'def/class run'" in w for w in warnings)
    errors, warnings = run_eval.validate_golden(_src_golden(loc("pkg/a.py", 1, 7, "A.run")), source_repo)
    assert errors == [] and any("does not start at the definition" in w for w in warnings)


def test_validate_golden_warns_on_commit_mismatch(source_repo):
    g = _src_golden(loc("pkg/a.py", 5, 7, "A.run"))
    g["repo"]["commit"] = "fff0000"
    _errors, warnings = run_eval.validate_golden(g, source_repo)
    assert any("line numbers may not match" in w for w in warnings)


def test_git_head_reads_refs_packed_refs_and_detached(tmp_path):
    _write_git_head(tmp_path / "a", "1111")
    assert run_eval.git_head(tmp_path / "a") == "1111"
    packed = tmp_path / "b" / ".git"
    packed.mkdir(parents=True)
    (packed / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (packed / "packed-refs").write_text("# pack-refs\n2222 refs/heads/main\n", encoding="utf-8")
    assert run_eval.git_head(tmp_path / "b") == "2222"
    detached = tmp_path / "c" / ".git"
    detached.mkdir(parents=True)
    (detached / "HEAD").write_text("3333\n", encoding="utf-8")
    assert run_eval.git_head(tmp_path / "c") == "3333"
    assert run_eval.git_head(tmp_path / "nothing") is None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _write_golden(tmp_path: Path, g: dict) -> Path:
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(g), encoding="utf-8")
    return path


def test_main_list_prints_questions(tmp_path, capsys):
    g, _stub = _stub_golden()
    assert run_eval.main([str(_write_golden(tmp_path, g)), "--list"]) == 0
    out = capsys.readouterr().out
    assert "q1" in out and "How is URL matching done?" in out and "3 questions" in out


def test_main_invalid_golden_exits_2(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert run_eval.main([str(bad), "--list"]) == 2
    g, _stub = _stub_golden()
    g["questions"][0]["category"] = "nope"
    assert run_eval.main([str(_write_golden(tmp_path, g)), "--list"]) == 2
    assert run_eval.main([str(_write_golden(tmp_path, _stub_golden()[0])), "--ids", "q42"]) == 2
    assert "invalid golden file" in capsys.readouterr().err


def test_main_runs_stub_pipeline_and_writes_report(tmp_path, capsys):
    g, stub = _stub_golden()
    out = tmp_path / "res" / "report.json"
    code = run_eval.main([str(_write_golden(tmp_path, g)), "--mode", "both", "--k", "4",
                          "--out", str(out), "--source", "SRC"], pipeline=stub)
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["meta"]["k"] == 4 and report["meta"]["source"] == "SRC"
    assert report["questions"][2]["answer"]["answer_text"].startswith("Alıntılarda yok")
    stdout = capsys.readouterr().out
    assert "| q2 | flow | en |" in stdout and "JSON report:" in stdout


def test_main_exits_1_when_the_pipeline_cannot_be_imported(tmp_path, monkeypatch, capsys):
    def missing():
        raise ImportError("No module named 'repofinder.pipeline'")

    monkeypatch.setattr(run_eval, "import_pipeline", missing)
    g, _stub = _stub_golden()
    assert run_eval.main([str(_write_golden(tmp_path, g)), "--source", "SRC"]) == 1
    assert "could not run the pipeline" in capsys.readouterr().err


def test_main_metrics_never_change_the_exit_code(tmp_path):
    g, _stub = _stub_golden()
    nothing = StubPipeline()  # every question misses
    code = run_eval.main([str(_write_golden(tmp_path, g)), "--source", "SRC",
                          "--out", str(tmp_path / "r.json")], pipeline=nothing)
    assert code == 0


# --------------------------------------------------------------------------- #
# Search-only baseline adapter (uses repofinder.search, not the pipeline)
# --------------------------------------------------------------------------- #

def test_search_baseline_adapter_follows_the_contract(sample_repo):
    baseline = _load_module("search_baseline", EVAL_DIR / "search_baseline.py")
    pipe = baseline.search_pipeline("sample_repo")
    ws = pipe.open_workspace(str(sample_repo), data_dir=None, refresh=False)
    chunks, hits = pipe.retrieve(ws, "run engine", k=5)
    assert 0 < len(hits) <= 5
    assert all(h.kind == "symbol" for h in hits if any(c.label == h.symbol_id for c in chunks))
    assert len(chunks) == sum(1 for h in hits if h.kind != "file")
    with pytest.raises(NotImplementedError):
        pipe.ask(ws, "anything")


# --------------------------------------------------------------------------- #
# The Flask golden set
# --------------------------------------------------------------------------- #

def _flask_golden() -> dict[str, Any]:
    return run_eval.load_golden(FLASK_GOLDEN)


def test_flask_golden_set_is_structurally_valid():
    errors, warnings = run_eval.validate_golden(_flask_golden())
    assert errors == [] and warnings == []


def test_flask_golden_set_composition():
    qs = _flask_golden()["questions"]
    assert len(qs) == 30
    assert Counter(q["category"] for q in qs) == {
        "locate-symbol": 12, "locate-feature": 10, "flow": 5, "not-in-repo": 3}
    assert Counter(q["lang"] for q in qs) == {"en": 20, "tr": 10}
    nested = [q["id"] for q in qs
              if any(l["path"].startswith(("examples/", "tests/"))
                     for l in q["expected"] + q.get("pointer", []))]
    assert len(nested) >= 2
    for q in qs:
        if q["category"] == "flow":
            assert q["accept_any"] is False and len(q["expected"]) >= 2
        if q["category"] == "not-in-repo":
            assert q["expected"] == [] and q["pointer"]


@pytest.mark.skipif(not FLASK_CHECKOUT.is_dir(), reason="Flask checkout not available")
def test_flask_golden_set_validates_against_the_checkout():
    errors, warnings = run_eval.validate_golden(_flask_golden(), FLASK_CHECKOUT)
    assert errors == []
    assert warnings == []

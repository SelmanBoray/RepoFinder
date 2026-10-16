"""Rank analysis: where each expected location ranks in the search's top N, and what beats it.

Usage (from the repository root, with a local checkout of the golden repo)::

    .venv\\Scripts\\python -I eval/rank_analysis.py eval/golden/flask.json --source PATH [--n 100] [--top 5] [--ids q01,q05] [--json OUT]

For every question it runs :class:`repofinder.search.SearchIndex` alone (no
pipeline, no LLM) with ``k=N`` and prints, per expected location (and per
``pointer`` location of a not-in-repo question), its strict rank in the top N
(``symbol``/``overlap`` match, the rules of ``run_eval.match_location``) and
the search ``why`` of that hit. Then the ``--top`` competitors above the best
expected rank (the first hits when nothing matched), each with its symbol
kind, whether it is a test file, its score and its ``why``. ``--json`` writes
the same data as JSON. Use it to classify misses before changing the ranking.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Optional

_HERE = Path(__file__).resolve().parent


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("run_eval", _HERE / "run_eval.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules.setdefault("run_eval", module)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


_runner = _load_runner()


def _hit_row(rank: int, hit: Any, kinds: dict[str, str], is_test: Any) -> dict[str, Any]:
    qual = hit.symbol_id.split("::", 1)[-1] if hit.symbol_id else None
    return {
        "rank": rank,
        "path": hit.path,
        "symbol": qual,
        "lines": [hit.start_line, hit.end_line],
        "kind": kinds.get(hit.symbol_id, "file") if hit.symbol_id else "file",
        "test": bool(is_test(hit.path)),
        "score": round(hit.score, 4),
        "why": hit.why,
    }


def analyze(golden: dict[str, Any], search: Any, kinds: dict[str, str], n: int, top: int,
            ids: Optional[list[str]] = None) -> list[dict[str, Any]]:
    """Per question: the rank of every expected/pointer location in the top ``n``
    and the ``top`` hits above the best of them."""
    from repofinder.graph import is_test_path

    out: list[dict[str, Any]] = []
    for q in _runner.select_questions(golden, ids):
        hits = search.search(q["question"], k=n)
        locs = [("expected", loc) for loc in q.get("expected") or []]
        locs += [("pointer", loc) for loc in q.get("pointer") or []]
        rows = []
        best: Optional[int] = None
        for role, loc in locs:
            rank = _runner.first_rank(hits, [loc])
            entry: dict[str, Any] = {
                "role": role,
                "path": loc["path"],
                "symbol": loc.get("symbol"),
                "lines": [loc["start_line"], loc["end_line"]],
                "rank": rank,
            }
            if rank is not None:
                entry["hit"] = _hit_row(rank, hits[rank - 1], kinds, is_test_path)
                best = rank if best is None else min(best, rank)
            rows.append(entry)
        limit = top if best is None else min(top, best - 1)
        competitors = [_hit_row(i + 1, h, kinds, is_test_path) for i, h in enumerate(hits[:limit])]
        n_tests = sum(1 for h in hits[: (best - 1) if best else n] if is_test_path(h.path))
        out.append({
            "id": q["id"],
            "category": q["category"],
            "lang": q["lang"],
            "question": q["question"],
            "best_rank": best,
            "locations": rows,
            "competitors": competitors,
            "tests_above_best": n_tests,
        })
    return out


def render(rows: list[dict[str, Any]], n: int) -> str:
    lines: list[str] = []
    for r in rows:
        lines.append(f"## {r['id']} [{r['category']}/{r['lang']}] best rank: {r['best_rank'] or f'>{n}'}"
                     f" (test hits above it: {r['tests_above_best']})")
        lines.append(f"Q: {r['question']}")
        for loc in r["locations"]:
            rank = loc["rank"] if loc["rank"] is not None else f">{n}"
            lines.append(f"  {loc['role']:8} {rank:>5}  {loc['path']}::{loc['symbol']} {loc['lines']}")
            if loc.get("hit"):
                h = loc["hit"]
                lines.append(f"           [{h['kind']}] score {h['score']}  why: {h['why']}")
        for c in r["competitors"]:
            test = " TEST" if c["test"] else ""
            lines.append(f"  #{c['rank']:<3} [{c['kind']}{test}] {c['score']:.3f} {c['path']}::{c['symbol']}")
            lines.append(f"        why: {c['why']}")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except (ValueError, OSError):
            pass
    p = argparse.ArgumentParser(prog="rank_analysis.py", description=__doc__.splitlines()[0])
    p.add_argument("golden")
    p.add_argument("--source", required=True, help="local checkout of the golden repository")
    p.add_argument("--n", type=int, default=100, help="hits fetched per question (default 100)")
    p.add_argument("--top", type=int, default=5, help="competitors shown per question (default 5)")
    p.add_argument("--ids", help="comma-separated question ids")
    p.add_argument("--json", help="also write the analysis as JSON to this path")
    args = p.parse_args(argv)

    from repofinder.ingest import build_index
    from repofinder.search import SearchIndex

    golden = _runner.load_golden(args.golden)
    name = (golden.get("repo") or {}).get("name") or Path(args.source).name
    index = build_index(Path(args.source), name, args.source)
    kinds = {s.id: s.kind for s in index.symbols}
    search = SearchIndex(index)
    ids = [i.strip() for i in args.ids.split(",")] if args.ids else None
    rows = analyze(golden, search, kinds, args.n, args.top, ids)
    sys.stdout.write(render(rows, args.n))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

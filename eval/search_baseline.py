"""Search-only retrieval baseline: the golden set against ``repofinder.search`` alone.

Usage (from the repository root, with a local checkout of the golden repo)::

    .venv\\Scripts\\python -I eval/search_baseline.py eval/golden/flask.json --source PATH [--k 8]

It builds the index with ``repofinder.ingest.build_index`` and runs
:func:`run_eval.run_eval` in ``retrieval`` mode with a small adapter in place of
``repofinder.pipeline``: ``retrieve`` returns ``SearchIndex.search`` hits, and
one pseudo-chunk per symbol hit covering exactly that hit's lines (file-level
hits get no chunk). So hit@k and MRR are the search layer's own numbers, while
chunk_recall here is only a stand-in for what the real chunker would retrieve.
No LLM is called. Use it to see what the search layer gets right before the
pipeline and the model are involved, and to compare a pipeline run against it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

_HERE = Path(__file__).resolve().parent


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("run_eval", _HERE / "run_eval.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules.setdefault("run_eval", module)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def search_pipeline(name: str) -> SimpleNamespace:
    """A pipeline-contract adapter over ``build_index`` + ``SearchIndex``."""
    from repofinder.ingest import build_index
    from repofinder.search import SearchIndex

    def open_workspace(source: str, data_dir: Optional[str] = None, refresh: bool = False) -> Any:
        index = build_index(Path(source), name, source)
        return SimpleNamespace(search=SearchIndex(index), root=Path(source))

    def retrieve(ws: Any, question: str, k: int = 8) -> tuple[list[Any], list[Any]]:
        hits = ws.search.search(question, k=k)
        chunks = [
            SimpleNamespace(path=h.path, start_line=h.start_line, end_line=h.end_line,
                            text="", label=h.symbol_id)
            for h in hits
            if h.kind != "file"
        ]
        return chunks, hits

    def ask(ws: Any, question: str, k: int = 8) -> Any:
        raise NotImplementedError("the search baseline has no answer step")

    return SimpleNamespace(open_workspace=open_workspace, retrieve=retrieve, ask=ask)


def baseline_table(report: dict[str, Any]) -> str:
    rows = [
        "| id | cat | lang | question | top-1 hit | top-1 match | strict rank |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report["questions"]:
        ret = r.get("retrieval") or {}
        top = (ret.get("top_hits") or [None])[0]
        if r["category"] == "not-in-repo":
            match = "pointer" if top and top.get("match") in ("symbol", "overlap") else "no pointer"
            rank = f"ptr {ret.get('pointer_rank') or '-'}"
        else:
            kind = top.get("match") if top else None
            match = "yes" if kind in ("symbol", "overlap") else (f"no ({kind})" if kind else "no")
            rank = str(ret.get("rank") or "-")
        question = r["question"].replace("|", "\\|")
        rows.append(
            f"| {r['id']} | {r['category']} | {r['lang']} | {question} |"
            f" {_runner.top1_label(r).replace('|', chr(92) + '|')} | {match} | {rank} |"
        )
    return "\n".join(rows) + "\n"


_runner = _load_runner()


def main(argv: Optional[list[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except (ValueError, OSError):
            pass
    p = argparse.ArgumentParser(prog="search_baseline.py", description=__doc__.splitlines()[0])
    p.add_argument("golden")
    p.add_argument("--source", required=True, help="local checkout of the golden repository")
    p.add_argument("--k", type=int, default=_runner.DEFAULT_K)
    p.add_argument("--out", help="JSON report path (default: eval/results/<golden>-search-<timestamp>.json)")
    args = p.parse_args(argv)

    golden = _runner.load_golden(args.golden)
    errors, warnings = _runner.validate_golden(golden, args.source)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        return 2
    name = (golden.get("repo") or {}).get("name") or Path(args.source).name
    report = _runner.run_eval(golden, args.source, "retrieval", args.k, search_pipeline(name))
    report["meta"]["pipeline"] = "search_baseline (build_index + SearchIndex; chunks = symbol-hit ranges)"
    out = Path(args.out) if args.out else (
        _HERE / "results" / f"{Path(args.golden).stem}-search-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.stdout.write(baseline_table(report))
    sys.stdout.write("\n")
    sys.stdout.write(_runner.render_markdown(report))
    print(f"\nJSON report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

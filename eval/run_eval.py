"""RepoFinder evaluation runner: measure retrieval and answer quality on a golden set.

Usage (from the repository root)::

    python eval/run_eval.py eval/golden/flask.json --list
    python eval/run_eval.py eval/golden/flask.json --source PATH_OR_URL --mode retrieval
    python eval/run_eval.py eval/golden/flask.json --mode both --ids q01,q05 --k 8

The runner talks to the product only through the pipeline contract::

    from repofinder.pipeline import open_workspace, retrieve, ask
    ws = open_workspace(source, data_dir=..., refresh=False)
    chunks, hits = retrieve(ws, question, k=k)
    result = ask(ws, question, k=k)   # .answer, .hits, .chunks, .elapsed_s

The pipeline module is imported lazily, and :func:`run_eval` accepts any object
with those three names (``pipeline=``), so tests can pass a stub.

This is a measurement, not a gate: the exit code is 0 whatever the scores are.
It is 2 when the golden file is invalid and 1 when the pipeline cannot be set up
(import error, ``open_workspace`` failure). See ``eval/README.md``.

Standard library plus ``repofinder.admit`` (itself standard library only); the
pipeline and the decisions layer are imported only when a run needs them.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from repofinder.admit import ADMIT_NO, ADMIT_PHRASES, ADMIT_YES, admits_missing  # noqa: F401 - re-exported

CATEGORIES = ("locate-symbol", "locate-feature", "flow", "config", "not-in-repo")
LANGS = ("en", "tr")
MODES = ("retrieval", "answer", "both")
DEFAULT_K = 8

# --- matching ------------------------------------------------------------------
# A hit (or citation) that contains an expected location but is much larger than
# it, such as the whole ``Flask`` class for an expected ``Flask.run`` method, or a
# file-level hit, only says "the answer is somewhere in here". It counts for the
# *loose* metrics but not for the strict ones. "Much larger" means both more than
# BROAD_FACTOR times the expected span and more than BROAD_MIN_SPAN lines.
BROAD_FACTOR = 4
BROAD_MIN_SPAN = 60
STRICT_MATCHES = frozenset({"symbol", "overlap"})
LOOSE_MATCHES = frozenset({"symbol", "overlap", "broad", "file"})
_MATCH_ORDER = ("symbol", "overlap", "broad", "file")

# A retrieved chunk set "covers" an expected location when the union of the
# chunks' line ranges on that path contains at least this fraction of its lines.
CHUNK_COVER_MIN = 0.5

# --- not-in-repo verdict -------------------------------------------------------
# Whether an answer "admits the answer is missing": the phrase heuristic lives in
# repofinder.admit (re-exported here: ADMIT_PHRASES, admits_missing), the Jev judge
# in repofinder.decisions.answer_admits_missing. With a decisions layer every
# answer is judged by it; the verdict is P(admits) >= ADMIT_THRESHOLD. See judge_admits.
ADMIT_THRESHOLD = 0.5

_LOCATION_KEYS = frozenset({"path", "start_line", "end_line", "symbol"})
_QUESTION_KEYS = frozenset(
    {"id", "question", "lang", "category", "expected", "accept_any", "notes", "pointer", "related"}
)


class GoldenError(ValueError):
    """The golden file cannot be read, or a requested question id does not exist."""


# =============================================================================== #
# Golden set: loading and validation
# =============================================================================== #

def load_golden(path: str | Path) -> dict[str, Any]:
    """Parse a golden JSON file. Raises :class:`GoldenError` if it is unreadable."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise GoldenError(f"cannot read {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GoldenError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise GoldenError(f"{path}: the top level must be an object")
    return data


def validate_golden(
    golden: dict[str, Any], source: Optional[str | Path] = None
) -> tuple[list[str], list[str]]:
    """Check a golden set. Returns ``(errors, warnings)``; any error makes it invalid.

    Without ``source`` only the structure is checked. With ``source`` pointing to a
    local checkout, every expected/pointer/related path must exist and every line
    range must fit in its file (errors); a ``symbol`` must be defined by a
    ``def``/``class`` inside its range and the range must start at that definition
    or its decorators (warnings). A checkout at another commit than ``repo.commit``
    is a warning: line numbers may have drifted.
    """
    errors: list[str] = []
    warnings: list[str] = []

    repo = golden.get("repo")
    if not isinstance(repo, dict):
        errors.append("'repo' must be an object with name, source and commit")
        repo = {}
    else:
        for key in ("name", "source", "commit"):
            if not isinstance(repo.get(key), str) or not repo.get(key):
                errors.append(f"repo.{key} must be a non-empty string")

    questions = golden.get("questions")
    if not isinstance(questions, list) or not questions:
        errors.append("'questions' must be a non-empty list")
        return errors, warnings

    root: Optional[Path] = None
    if source is not None and Path(source).is_dir():
        root = Path(source)
        commit = repo.get("commit")
        if isinstance(commit, str) and commit:
            head = git_head(root)
            if head is None:
                warnings.append(f"cannot read the git HEAD of {root}; golden commit is {commit}")
            elif not head.startswith(commit):
                warnings.append(
                    f"checkout {root} is at {head[:12]} but the golden set was written for"
                    f" {commit}: line numbers may not match"
                )

    file_cache: dict[str, Optional[list[str]]] = {}
    seen: set[str] = set()
    for n, q in enumerate(questions):
        where = f"questions[{n}]"
        if not isinstance(q, dict):
            errors.append(f"{where}: must be an object")
            continue
        qid = q.get("id")
        if not isinstance(qid, str) or not qid:
            errors.append(f"{where}: 'id' must be a non-empty string")
        else:
            where = qid
            if qid in seen:
                errors.append(f"{qid}: duplicate id")
            seen.add(qid)
        for key in sorted(set(q) - _QUESTION_KEYS):
            warnings.append(f"{where}: unknown key {key!r}")
        text = q.get("question")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"{where}: 'question' must be a non-empty string")
        if q.get("lang") not in LANGS:
            errors.append(f"{where}: 'lang' must be one of {', '.join(LANGS)}, got {q.get('lang')!r}")
        category = q.get("category")
        if category not in CATEGORIES:
            errors.append(f"{where}: 'category' must be one of {', '.join(CATEGORIES)}, got {category!r}")
        if not isinstance(q.get("accept_any"), bool):
            errors.append(f"{where}: 'accept_any' must be true or false")
        if "notes" in q and not isinstance(q["notes"], str):
            errors.append(f"{where}: 'notes' must be a string")
        elif not q.get("notes"):
            warnings.append(f"{where}: no 'notes' (say why this is the answer)")

        groups: dict[str, Any] = {"expected": q.get("expected")}
        for key in ("pointer", "related"):
            if key in q:
                groups[key] = q[key]
        for key, locs in groups.items():
            if not isinstance(locs, list):
                errors.append(f"{where}: '{key}' must be a list")
                continue
            for i, loc in enumerate(locs):
                _check_location(f"{where}.{key}[{i}]", loc, root, file_cache, errors, warnings)

        expected = q.get("expected") if isinstance(q.get("expected"), list) else []
        if category == "not-in-repo":
            if expected:
                errors.append(f"{where}: a not-in-repo question must have expected = []")
            if not q.get("pointer"):
                errors.append(f"{where}: a not-in-repo question needs a 'pointer' (the delegating call site)")
        elif category in CATEGORIES:
            if not expected:
                errors.append(f"{where}: 'expected' is empty; only not-in-repo questions may have no answer")
            if "pointer" in q:
                warnings.append(f"{where}: 'pointer' is only used by not-in-repo questions")
            if q.get("accept_any") is False and len(expected) < 2:
                warnings.append(f"{where}: accept_any=false with fewer than two expected locations")
    return errors, warnings


def _check_location(
    where: str,
    loc: Any,
    root: Optional[Path],
    file_cache: dict[str, Optional[list[str]]],
    errors: list[str],
    warnings: list[str],
) -> None:
    if not isinstance(loc, dict):
        errors.append(f"{where}: must be an object with path, start_line and end_line")
        return
    for key in sorted(set(loc) - _LOCATION_KEYS):
        warnings.append(f"{where}: unknown key {key!r}")
    path = loc.get("path")
    start, end = loc.get("start_line"), loc.get("end_line")
    ok = True
    if not isinstance(path, str) or not path:
        errors.append(f"{where}: 'path' must be a non-empty string")
        ok = False
    elif "\\" in path or path.startswith("/") or ".." in path.split("/"):
        errors.append(f"{where}: 'path' must be repo-relative with forward slashes, got {path!r}")
        ok = False
    if not _is_int(start) or not _is_int(end):
        errors.append(f"{where}: start_line and end_line must be integers")
        ok = False
    elif start < 1 or start > end:
        errors.append(f"{where}: bad line range {start}-{end} (need 1 <= start <= end)")
        ok = False
    symbol = loc.get("symbol")
    if symbol is not None and (not isinstance(symbol, str) or not symbol):
        errors.append(f"{where}: 'symbol' must be a non-empty string when given")
        symbol = None
    if not ok or root is None:
        return

    if path not in file_cache:
        file_cache[path] = _read_lines(root, path)
    lines = file_cache[path]
    if lines is None:
        errors.append(f"{where}: {path} does not exist under {root}")
        return
    if end > len(lines):
        errors.append(f"{where}: {path}:{start}-{end} is past the end of the file ({len(lines)} lines)")
        return
    if symbol:
        name = symbol.split(".")[-1]
        def_re = re.compile(rf"^\s*(?:async\s+def|def|class)\s+{re.escape(name)}\b")
        window = lines[start - 1 : end]
        if not any(def_re.match(line) for line in window):
            warnings.append(f"{where}: no 'def/class {name}' within {path}:{start}-{end}")
        else:
            first = next((line.strip() for line in window if line.strip()), "")
            if not (first.startswith("@") or def_re.match(first)):
                warnings.append(
                    f"{where}: {path}:{start} does not start at the definition of {symbol}"
                    f" or its decorators (it starts with {first[:40]!r})"
                )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _read_lines(root: Path, rel: str) -> Optional[list[str]]:
    """Lines of ``root/rel`` split on ``\\n`` only (the parser's numbering), or None."""
    target = Path(root, *rel.split("/"))
    try:
        raw = target.read_bytes()
    except OSError:
        return None
    lines = raw.decode("utf-8", errors="replace").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def git_head(root: str | Path) -> Optional[str]:
    """Commit SHA checked out in ``root``, read from the ``.git`` files (git is not run)."""
    git = Path(root) / ".git"
    try:
        if git.is_file():  # worktree or submodule: "gitdir: <path>"
            pointer = git.read_text(encoding="utf-8").strip()
            if not pointer.startswith("gitdir:"):
                return None
            git = (Path(root) / pointer[len("gitdir:"):].strip()).resolve()
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head or None
        ref = head[len("ref:"):].strip()
        ref_file = git.joinpath(*ref.split("/"))
        if ref_file.is_file():
            return ref_file.read_text(encoding="utf-8").strip() or None
        packed = git / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    except OSError:
        return None
    return None


def select_questions(golden: dict[str, Any], ids: Optional[Iterable[str]] = None) -> list[dict[str, Any]]:
    """Questions in file order, optionally only ``ids``. Unknown ids raise GoldenError."""
    questions = list(golden.get("questions") or [])
    if ids is None:
        return questions
    wanted = [i.strip() for i in ids if i and i.strip()]
    known = {q.get("id") for q in questions}
    unknown = [i for i in wanted if i not in known]
    if unknown:
        raise GoldenError(f"unknown question id(s): {', '.join(unknown)}")
    keep = set(wanted)
    return [q for q in questions if q.get("id") in keep]


# =============================================================================== #
# Matching
# =============================================================================== #

def _span(start: int, end: int) -> int:
    return max(end - start + 1, 1)


def ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start <= b_end and b_start <= a_end


def match_location(
    path: str,
    start: int,
    end: int,
    loc: dict[str, Any],
    *,
    kind: str = "symbol",
    symbol_id: Optional[str] = None,
) -> Optional[str]:
    """How a retrieved range (hit, chunk or citation) matches one golden location.

    Returns ``None`` (no match) or one of:

    * ``"symbol"``: same path and the hit's qualname (the part of ``symbol_id``
      after ``::``) equals ``loc["symbol"]``;
    * ``"overlap"``: same path and the line ranges overlap;
    * ``"broad"``: overlaps, but the range contains the location and is much
      larger (see :data:`BROAD_FACTOR` and :data:`BROAD_MIN_SPAN`);
    * ``"file"``: a file-level hit (``kind == "file"``) on the right file.

    Only ``"symbol"`` and ``"overlap"`` count for the strict metrics.
    """
    if path != loc["path"]:
        return None
    if symbol_id and loc.get("symbol") and symbol_id.split("::", 1)[-1] == loc["symbol"]:
        return "symbol"
    l_start, l_end = loc["start_line"], loc["end_line"]
    if not ranges_overlap(start, end, l_start, l_end):
        return None
    if kind == "file":
        return "file"
    span, l_span = _span(start, end), _span(l_start, l_end)
    contains = start <= l_start and end >= l_end
    if contains and span > BROAD_FACTOR * l_span and span > BROAD_MIN_SPAN:
        return "broad"
    return "overlap"


def match_hit(hit: Any, loc: dict[str, Any]) -> Optional[str]:
    """:func:`match_location` for a search ``Hit`` (duck-typed)."""
    return match_location(
        hit.path,
        hit.start_line,
        hit.end_line,
        loc,
        kind=getattr(hit, "kind", None) or "symbol",
        symbol_id=getattr(hit, "symbol_id", None),
    )


def best_match(hit: Any, locs: list[dict[str, Any]]) -> Optional[str]:
    """Strongest match of ``hit`` against any of ``locs`` (strict kinds first)."""
    found = {match_hit(hit, loc) for loc in locs}
    for kind in _MATCH_ORDER:
        if kind in found:
            return kind
    return None


def first_rank(hits: list[Any], locs: list[dict[str, Any]], *, strict: bool = True) -> Optional[int]:
    """1-based rank of the first hit that matches any of ``locs``, or None."""
    allowed = STRICT_MATCHES if strict else LOOSE_MATCHES
    for rank, hit in enumerate(hits, start=1):
        if best_match(hit, locs) in allowed:
            return rank
    return None


def location_coverage(chunks: list[Any], loc: dict[str, Any]) -> float:
    """Fraction of the location's lines inside the union of the chunks on its path."""
    l_start, l_end = loc["start_line"], loc["end_line"]
    covered: set[int] = set()
    for chunk in chunks:
        if chunk.path != loc["path"]:
            continue
        lo, hi = max(chunk.start_line, l_start), min(chunk.end_line, l_end)
        if lo <= hi:
            covered.update(range(lo, hi + 1))
    return len(covered) / _span(l_start, l_end)


def chunks_cover(chunks: list[Any], locs: list[dict[str, Any]], accept_any: bool) -> Optional[bool]:
    """Chunks cover one of ``locs`` (``accept_any``) or all of them; None if no locs."""
    if not locs:
        return None
    covered = [location_coverage(chunks, loc) >= CHUNK_COVER_MIN for loc in locs]
    return any(covered) if accept_any else all(covered)


# =============================================================================== #
# Scoring
# =============================================================================== #

def score_retrieval(question: dict[str, Any], hits: list[Any], chunks: list[Any], k: int) -> dict[str, Any]:
    """Retrieval metrics for one question. ``hits`` are cut to the first ``k``.

    ``hit@n`` is computed over ``hits[:min(n, k)]``. For a not-in-repo question
    (no expected locations) the hit metrics are None and ``pointer_*`` is set.
    """
    hits = list(hits)[:k]
    chunks = list(chunks)
    expected = question.get("expected") or []
    pointer = question.get("pointer") or []
    accept_any = bool(question.get("accept_any", True))
    out: dict[str, Any] = {
        "n_hits": len(hits),
        "n_chunks": len(chunks),
        "top_hits": [_hit_record(h, expected + pointer) for h in hits],
        "chunks": [
            {"path": c.path, "start_line": c.start_line, "end_line": c.end_line,
             "label": getattr(c, "label", None)}
            for c in chunks
        ],
    }
    if expected:
        rank = first_rank(hits, expected, strict=True)
        loose = first_rank(hits, expected, strict=False)
        out.update(
            {
                "rank": rank,
                "loose_rank": loose,
                "rr": (1.0 / rank) if rank else 0.0,
                "hit@1": _hit_at(rank, 1, k),
                "hit@3": _hit_at(rank, 3, k),
                "hit_at_k": rank is not None,
                "loose_hit@1": _hit_at(loose, 1, k),
                "loose_hit@3": _hit_at(loose, 3, k),
                "loose_hit_at_k": loose is not None,
                "all_at_k": None if accept_any
                else all(first_rank(hits, [loc]) is not None for loc in expected),
                "chunk_recall": chunks_cover(chunks, expected, accept_any),
                "chunk_coverage": [round(location_coverage(chunks, loc), 3) for loc in expected],
            }
        )
    else:
        out.update(
            {
                "rank": None, "loose_rank": None, "rr": None,
                "hit@1": None, "hit@3": None, "hit_at_k": None,
                "loose_hit@1": None, "loose_hit@3": None, "loose_hit_at_k": None,
                "all_at_k": None, "chunk_recall": None, "chunk_coverage": [],
            }
        )
    if pointer:
        p_rank = first_rank(hits, pointer, strict=True)
        out.update(
            {
                "pointer_rank": p_rank,
                "pointer_at_k": p_rank is not None,
                "pointer_chunk_recall": chunks_cover(chunks, pointer, True),
            }
        )
    return out


def _hit_at(rank: Optional[int], n: int, k: int) -> bool:
    return rank is not None and rank <= min(n, k)


def _hit_record(hit: Any, locs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path": hit.path,
        "symbol_id": getattr(hit, "symbol_id", None),
        "start_line": hit.start_line,
        "end_line": hit.end_line,
        "kind": getattr(hit, "kind", None),
        "score": round(float(getattr(hit, "score", 0.0) or 0.0), 4),
        "match": best_match(hit, locs) if locs else None,
        "why": getattr(hit, "why", None),
    }


def judge_admits(text: str, question: str = "", decisions: Any = None, *,
                 strict: bool = False) -> dict[str, Any]:
    """The admit fields of an answer record: does ``text`` admit that what
    ``question`` asks about is missing?

    ``admits_phrase`` / ``admits_heuristic`` always come from the phrase
    heuristic (:func:`admits_missing`). Without ``decisions`` the verdict is the
    heuristic's (judge ``rules``, probability ``ADMIT_YES``/``ADMIT_NO``, no
    call). With a layer it is
    :func:`repofinder.decisions.answer_admits_missing` at
    :data:`ADMIT_THRESHOLD` (judge: the layer's name), and ``admits_calls`` /
    ``admits_cost_usd`` are what the judgement cost. When the layer raises
    ``DecisionsError`` the heuristic's verdict is kept and the error is recorded
    in ``admits_judge_error``; with ``strict`` a ``DecisionsConfigError`` (no
    key, rejected key, no credit) propagates instead.
    """
    text = text or ""
    phrase = admits_missing(text)
    fields: dict[str, Any] = {
        "admits_missing": phrase is not None,
        "admits_judge": "rules",
        "admits_probability": ADMIT_YES if phrase is not None else ADMIT_NO,
        "admits_phrase": phrase,
        "admits_heuristic": phrase is not None,
        "admits_calls": 0,
        "admits_cost_usd": 0.0,
    }
    if decisions is None:
        return fields
    from repofinder import decisions as decisions_mod

    usage = getattr(decisions, "usage", None)
    calls_before = int(getattr(usage, "calls", 0) or 0)
    try:
        verdict = decisions_mod.answer_admits_missing(decisions, question or "", text)
    except decisions_mod.DecisionsError as exc:
        if strict and isinstance(exc, decisions_mod.DecisionsConfigError):
            raise
        fields["admits_judge_error"] = f"{type(exc).__name__}: {exc}"
        return fields
    p = float(verdict.probability)
    calls = int(getattr(usage, "calls", 0) or 0) - calls_before if usage is not None else 1
    fields.update(
        admits_missing=p >= ADMIT_THRESHOLD,
        admits_judge=str(getattr(decisions, "name", None) or "?"),
        admits_probability=round(p, 4),
        admits_calls=max(0, calls),
        admits_cost_usd=float(verdict.cost_usd or 0.0),
    )
    return fields


def score_answer(question: dict[str, Any], answer: Any, decisions: Any = None) -> dict[str, Any]:
    """Answer metrics for one question from an ``Answer`` (text, citations, tokens).

    * ``cited_correct``: a citation strictly matches an expected location;
    * ``cited_wrong``: citations that overlap no expected, pointer or related location;
    * ``all_required`` (accept_any=false): every expected location is cited;
    * ``admits_missing``: the answer says that what was asked is not in the code
      (all questions; on answerable questions it is a false admission). The
      verdict comes from ``decisions`` (the Jev judge) when given, else from the
      phrase heuristic; see :func:`judge_admits` for the other ``admits_*`` fields;
    * ``pointer_cited`` (not-in-repo): a citation strictly matches a pointer.
    """
    expected = question.get("expected") or []
    pointer = question.get("pointer") or []
    related = question.get("related") or []
    accept_any = bool(question.get("accept_any", True))
    citations = list(getattr(answer, "citations", None) or [])

    def matches(c: Any, locs: list[dict[str, Any]]) -> set[str]:
        found = {match_location(c.path, c.start_line, c.end_line, loc, kind="citation") for loc in locs}
        return {m for m in found if m is not None}

    def cited(loc: dict[str, Any]) -> bool:
        return any(matches(c, [loc]) & STRICT_MATCHES for c in citations)

    records = []
    n_wrong = 0
    for c in citations:
        exp = matches(c, expected)
        if exp & STRICT_MATCHES:
            verdict = "expected"
        elif exp:
            verdict = "expected-broad"
        elif matches(c, pointer):
            verdict = "pointer"
        elif matches(c, related):
            verdict = "related"
        else:
            verdict = "wrong"
            n_wrong += 1
        records.append(
            {"path": c.path, "start_line": c.start_line, "end_line": c.end_line, "verdict": verdict}
        )

    text = getattr(answer, "text", "") or ""
    return {
        "n_citations": len(citations),
        "citations": records,
        "cited_correct": any(cited(loc) for loc in expected) if expected else None,
        "cited_wrong": n_wrong,
        "all_required": all(cited(loc) for loc in expected) if expected and not accept_any else None,
        **judge_admits(text, question.get("question", ""), decisions),
        "pointer_cited": any(cited(loc) for loc in pointer) if pointer else None,
        "model": getattr(answer, "model", None),
        "input_tokens": int(getattr(answer, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(answer, "output_tokens", 0) or 0),
        "cache_read_tokens": int(getattr(answer, "cache_read_tokens", 0) or 0),
        "answer_text": text,
    }


# =============================================================================== #
# Aggregation
# =============================================================================== #

def _rate(values: Iterable[Optional[bool]]) -> Optional[dict[str, Any]]:
    vals = [bool(v) for v in values if v is not None]
    if not vals:
        return None
    return {"n": len(vals), "yes": sum(vals), "rate": round(sum(vals) / len(vals), 3)}


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return round(statistics.fmean(vals), 3) if vals else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Metrics over a group of per-question results.

    Rates are ``{"n", "yes", "rate"}`` over the questions where the metric applies
    (None values are skipped, e.g. hit@1 on not-in-repo questions). MRR is the
    mean reciprocal strict rank over questions with expected locations (0 when
    nothing matched in the top k).
    """
    ret = [r["retrieval"] for r in rows if r.get("retrieval")]
    ans = [(r, r["answer"]) for r in rows if r.get("answer")]
    s: dict[str, Any] = {"n": len(rows), "errors": sum(1 for r in rows if r.get("error"))}
    if ret:
        s["retrieval"] = {
            "hit@1": _rate(x.get("hit@1") for x in ret),
            "hit@3": _rate(x.get("hit@3") for x in ret),
            "hit@k": _rate(x.get("hit_at_k") for x in ret),
            "loose_hit@1": _rate(x.get("loose_hit@1") for x in ret),
            "loose_hit@3": _rate(x.get("loose_hit@3") for x in ret),
            "loose_hit@k": _rate(x.get("loose_hit_at_k") for x in ret),
            "mrr": _mean(x.get("rr") for x in ret),
            "chunk_recall": _rate(x.get("chunk_recall") for x in ret),
            "all@k": _rate(x.get("all_at_k") for x in ret),
            "pointer@k": _rate(x.get("pointer_at_k") for x in ret),
            "pointer_chunk_recall": _rate(x.get("pointer_chunk_recall") for x in ret),
        }
    if ans:
        not_in_repo = [a for r, a in ans if r["category"] == "not-in-repo"]
        answerable = [a for r, a in ans if r["category"] != "not-in-repo"]
        s["answer"] = {
            "cited_correct": _rate(a.get("cited_correct") for _r, a in ans),
            "cited_wrong_mean": _mean(a.get("cited_wrong") for _r, a in ans),
            "cited_wrong_total": sum(int(a.get("cited_wrong") or 0) for _r, a in ans),
            "all_required": _rate(a.get("all_required") for _r, a in ans),
            "admits_missing": _rate(a.get("admits_missing") for a in not_in_repo),
            "false_admits": _rate(a.get("admits_missing") for a in answerable),
            "pointer_cited": _rate(a.get("pointer_cited") for _r, a in ans),
            "input_tokens": sum(a.get("input_tokens", 0) for _r, a in ans),
            "output_tokens": sum(a.get("output_tokens", 0) for _r, a in ans),
            "cache_read_tokens": sum(a.get("cache_read_tokens", 0) for _r, a in ans),
        }
        judges: dict[str, int] = {}
        for _r, a in ans:
            if a.get("admits_judge"):
                judges[a["admits_judge"]] = judges.get(a["admits_judge"], 0) + 1
        compared = [a for _r, a in ans if a.get("admits_heuristic") is not None]
        s["answer"]["admits_judge"] = dict(sorted(judges.items()))
        s["answer"]["admit_judge_vs_heuristic"] = {
            "n": len(compared),
            "differ": sum(1 for a in compared if bool(a.get("admits_missing")) != bool(a["admits_heuristic"])),
        }
    decs = [r["decision"] for r in rows if r.get("decision")]
    judge_calls = sum(int(a.get("admits_calls") or 0) for _r, a in ans)
    judge_cost = sum(float(a.get("admits_cost_usd") or 0.0) for _r, a in ans)
    if decs or judge_calls:
        s["decision_calls"] = sum(int(d.get("calls") or 0) for d in decs) + judge_calls
        s["decision_cost_usd"] = round(sum(float(d.get("cost_usd") or 0.0) for d in decs) + judge_cost, 6)
        s["admit_judge_calls"] = judge_calls
        s["admit_judge_cost_usd"] = round(judge_cost, 6)
    if decs:
        s["reranked"] = sum(1 for d in decs if d.get("reranked"))
        s["off_target"] = sum(1 for d in decs if d.get("off_target"))
        s["decision_errors"] = sum(1 for d in decs if d.get("error"))
    elapsed = [r["elapsed_s"] for r in rows if r.get("elapsed_s") is not None]
    if elapsed:
        s["elapsed_s"] = {
            "total": round(sum(elapsed), 3),
            "mean": round(statistics.fmean(elapsed), 3),
            "median": round(statistics.median(elapsed), 3),
            "max": round(max(elapsed), 3),
        }
    return s


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """:func:`summarize` overall, per category and per language."""
    by_cat: dict[str, list[dict[str, Any]]] = {}
    by_lang: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)
        by_lang.setdefault(r["lang"], []).append(r)
    return {
        "overall": summarize(results),
        "by_category": {c: summarize(by_cat[c]) for c in CATEGORIES if c in by_cat},
        "by_lang": {lang: summarize(rows) for lang, rows in sorted(by_lang.items())},
    }


# =============================================================================== #
# Running
# =============================================================================== #

class _NoAnswer:
    """Stand-in answer for a question whose ``ask`` call raised."""

    text = ""
    citations: list[Any] = []
    model = None
    input_tokens = output_tokens = cache_read_tokens = 0


def import_pipeline() -> Any:
    """The real pipeline module, imported only when a run needs it."""
    return importlib.import_module("repofinder.pipeline")


def run_eval(
    golden: dict[str, Any],
    source: str,
    mode: str = "retrieval",
    k: int = DEFAULT_K,
    pipeline: Any = None,
    *,
    ids: Optional[Iterable[str]] = None,
    data_dir: Optional[str] = None,
    log: Any = None,
    decisions: Any = None,
    candidates: Optional[int] = None,
) -> dict[str, Any]:
    """Run the golden questions through ``pipeline`` and score them.

    ``pipeline`` is any object with ``open_workspace``, ``retrieve`` and ``ask``
    (default: :mod:`repofinder.pipeline`). In ``retrieval`` mode only
    ``retrieve`` is called. In ``answer`` and ``both`` modes only ``ask`` is;
    ``both`` scores retrieval on the hits and chunks that ``ask`` returns, so
    each question costs one LLM call. A question whose call raises is recorded
    with its error and scored as a miss. ``open_workspace`` errors propagate.

    If the workspace has a ``root`` (the checkout), its git commit is compared
    with ``repo.commit`` (``meta.checkout``; a warning goes to ``log`` on a
    mismatch). If it has a ``close()``, it is closed at the end.

    With ``decisions`` (a decision layer such as Jev) retrieval mode calls
    ``pipeline.rerank_retrieve`` and answer/both modes pass ``decisions`` and
    ``candidates`` to ``ask``; each row then carries a ``decision`` block.
    Without it the calls are exactly ``retrieve(ws, q, k=k)`` / ``ask(ws, q, k=k)``.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}, got {mode!r}")
    if k < 1:
        raise ValueError("k must be >= 1")
    if pipeline is None:
        pipeline = import_pipeline()
    questions = select_questions(golden, ids)
    if decisions is not None:
        if candidates is None:
            candidates = int(getattr(pipeline, "DEFAULT_CANDIDATES", 16))
        if candidates < 1:
            raise ValueError("candidates must be >= 1")

    started = time.time()
    t0 = time.perf_counter()
    kwargs: dict[str, Any] = {"refresh": False}
    if data_dir is not None:
        kwargs["data_dir"] = data_dir
    ws = pipeline.open_workspace(source, **kwargs)
    open_s = time.perf_counter() - t0
    try:
        checkout = checkout_info(ws, golden)
        if log is not None and checkout.get("matches_golden") is False:
            print(
                f"warning: the workspace checkout is at {checkout['commit'][:12]}, the golden set"
                f" was written for {checkout['golden_commit']}: line-based scores are unreliable."
                " Pass --source with a local checkout at the golden commit.",
                file=log,
                flush=True,
            )
        results = _run_questions(pipeline, ws, questions, mode, k, log, decisions, candidates)
    finally:
        close = getattr(ws, "close", None)
        if callable(close):
            close()

    meta_extra: dict[str, Any] = {}
    if decisions is not None:
        meta_extra = {"decisions": getattr(decisions, "name", None), "candidates": candidates}
    return {
        "meta": {
            **meta_extra,
            "repo": golden.get("repo"),
            "source": source,
            "checkout": checkout,
            "mode": mode,
            "k": k,
            "ids": [q["id"] for q in questions],
            "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
            "open_workspace_s": round(open_s, 3),
            "total_s": round(time.perf_counter() - t0, 3),
            "constants": {
                "BROAD_FACTOR": BROAD_FACTOR,
                "BROAD_MIN_SPAN": BROAD_MIN_SPAN,
                "CHUNK_COVER_MIN": CHUNK_COVER_MIN,
            },
        },
        "summary": aggregate(results),
        "questions": results,
    }


def checkout_info(ws: Any, golden: dict[str, Any]) -> dict[str, Any]:
    """The workspace checkout's commit compared with ``repo.commit`` (when ``ws.root`` exists)."""
    golden_commit = (golden.get("repo") or {}).get("commit")
    root = getattr(ws, "root", None)
    info: dict[str, Any] = {"root": str(root) if root is not None else None,
                            "commit": None, "golden_commit": golden_commit, "matches_golden": None}
    if root is not None and Path(root).is_dir():
        head = git_head(root)
        info["commit"] = head
        if head and golden_commit:
            info["matches_golden"] = head.startswith(golden_commit)
    return info


def _decision_block(result: Any) -> dict[str, Any]:
    return {
        "calls": int(getattr(result, "decision_calls", 0) or 0),
        "cost_usd": float(getattr(result, "decision_cost_usd", 0.0) or 0.0),
        "reranked": bool(getattr(result, "reranked", False)),
        "off_target": bool(getattr(result, "off_target", False)),
        "error": getattr(result, "decision_error", None),
    }


def _run_questions(pipeline: Any, ws: Any, questions: list[dict[str, Any]], mode: str, k: int,
                   log: Any, decisions: Any = None, candidates: Optional[int] = None
                   ) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for q in questions:
        row: dict[str, Any] = {
            "id": q["id"],
            "category": q["category"],
            "lang": q["lang"],
            "question": q["question"],
            "accept_any": q.get("accept_any", True),
            "expected": q.get("expected") or [],
            "pointer": q.get("pointer") or [],
            "error": None,
        }
        t = time.perf_counter()
        try:
            if mode == "retrieval":
                if decisions is None:
                    chunks, hits = pipeline.retrieve(ws, q["question"], k=k)
                else:
                    res = pipeline.rerank_retrieve(
                        ws, q["question"], k=k, decisions=decisions, candidates=candidates
                    )
                    chunks, hits = res.chunks, res.hits
                    row["decision"] = _decision_block(res)
                row["elapsed_s"] = round(time.perf_counter() - t, 4)
                row["retrieval"] = score_retrieval(q, hits, chunks, k)
            else:
                if decisions is None:
                    result = pipeline.ask(ws, q["question"], k=k)
                else:
                    result = pipeline.ask(
                        ws, q["question"], k=k, decisions=decisions, candidates=candidates
                    )
                    row["decision"] = _decision_block(result)
                row["elapsed_s"] = round(time.perf_counter() - t, 4)
                row["pipeline_elapsed_s"] = getattr(result, "elapsed_s", None)
                row["answer"] = score_answer(q, result.answer, decisions)
                if mode == "both":
                    row["retrieval"] = score_retrieval(
                        q, list(result.hits or []), list(result.chunks or []), k
                    )
        except Exception as exc:  # noqa: BLE001 - a failing question is a data point
            row["elapsed_s"] = round(time.perf_counter() - t, 4)
            row["error"] = f"{type(exc).__name__}: {exc}"
            if decisions is not None and "decision" not in row:
                row["decision"] = {"calls": 0, "cost_usd": 0.0, "reranked": False,
                                   "off_target": False, "error": row["error"]}
            if mode in ("retrieval", "both"):
                row["retrieval"] = score_retrieval(q, [], [], k)
            if mode in ("answer", "both"):
                row["answer"] = score_answer(q, _NoAnswer(), decisions)
        if log is not None:
            status = f"ERROR {row['error']}" if row["error"] else _short_status(row)
            print(f"[{q['id']}] {row['elapsed_s']:.2f}s {status}", file=log, flush=True)
        results.append(row)
    return results


def _short_status(row: dict[str, Any]) -> str:
    parts = []
    ret = row.get("retrieval")
    if ret and ret.get("rr") is not None:
        parts.append(f"rank={ret['rank']}")
    ans = row.get("answer")
    if ans:
        parts.append(f"cited_correct={ans['cited_correct']} cited_wrong={ans['cited_wrong']}")
    dec = row.get("decision")
    if dec:
        parts.append("[dec-err]" if dec.get("error") else "[rerank]" if dec.get("reranked") else "[none]")
    return " ".join(parts)


# =============================================================================== #
# Reporting
# =============================================================================== #

def _yn(value: Optional[bool]) -> str:
    return "-" if value is None else ("1" if value else "0")


def _fmt_rate(r: Optional[dict[str, Any]]) -> str:
    return "-" if not r else f"{r['rate']:.2f} ({r['yes']}/{r['n']})"


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def top1_label(row: dict[str, Any]) -> str:
    """``path:line qualname`` of the first hit of a result row, or ``-``."""
    hits = (row.get("retrieval") or {}).get("top_hits") or []
    if not hits:
        return "-"
    h = hits[0]
    label = h["symbol_id"].split("::", 1)[-1] if h.get("symbol_id") else "(file)"
    return f"{h['path']}:{h['start_line']} {label}"


def render_markdown(report: dict[str, Any]) -> str:
    """The per-question table and the summary, as Markdown."""
    meta, rows = report["meta"], report["questions"]
    k = meta["k"]
    name = (meta.get("repo") or {}).get("name", "?")
    out = [
        f"# RepoFinder eval: {name} (mode={meta['mode']}, k={k})",
        "",
        "| id | category | lang | hit@1 | hit@3 | chunk_recall | cited_correct | cited_wrong"
        " | elapsed_s | extra | top-1 hit |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ret, ans = r.get("retrieval") or {}, r.get("answer") or {}
        extra = []
        if ret.get("all_at_k") is not None:
            extra.append(f"all@k={_yn(ret['all_at_k'])}")
        if ret.get("pointer_at_k") is not None:
            extra.append(f"ptr@k={_yn(ret['pointer_at_k'])}")
        if ans.get("all_required") is not None:
            extra.append(f"all_req={_yn(ans['all_required'])}")
        if r["category"] == "not-in-repo" and ans:
            extra.append(f"admit={_yn(ans.get('admits_missing'))}")
            extra.append(f"ptr_cited={_yn(ans.get('pointer_cited'))}")
        elif ans.get("admits_missing"):
            extra.append("false-admit")
        if r.get("error"):
            extra.append("ERROR")
        cells = [
            r["id"],
            r["category"],
            r["lang"],
            _yn(ret.get("hit@1")) if ret else "-",
            _yn(ret.get("hit@3")) if ret else "-",
            _yn(ret.get("chunk_recall")) if ret else "-",
            _yn(ans.get("cited_correct")) if ans else "-",
            _fmt(ans.get("cited_wrong")) if ans else "-",
            f"{r.get('elapsed_s') or 0:.2f}",
            " ".join(extra),
            top1_label(r).replace("|", "\\|"),
        ]
        out.append("| " + " | ".join(cells) + " |")

    summary = report["summary"]
    out += ["", "## Summary", ""]
    out += summary_lines(summary["overall"], k, meta.get("decisions") or meta.get("judge"))
    out += [
        "",
        "| group | n | hit@1 | hit@3 | hit@k | MRR | chunk_recall | cited_correct | cited_wrong mean |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for title, groups in (("category", summary["by_category"]), ("lang", summary["by_lang"])):
        for group, s in groups.items():
            ret, ans = s.get("retrieval") or {}, s.get("answer") or {}
            out.append(
                f"| {title}: {group} | {s['n']} | {_fmt_rate(ret.get('hit@1'))}"
                f" | {_fmt_rate(ret.get('hit@3'))} | {_fmt_rate(ret.get('hit@k'))}"
                f" | {_fmt(ret.get('mrr'))} | {_fmt_rate(ret.get('chunk_recall'))}"
                f" | {_fmt_rate(ans.get('cited_correct'))} | {_fmt(ans.get('cited_wrong_mean'))} |"
            )
    return "\n".join(out) + "\n"


def summary_lines(s: dict[str, Any], k: int, decisions_name: Optional[str] = None) -> list[str]:
    """Bullet lines for one :func:`summarize` result."""
    lines = [f"- questions: {s['n']}, errors: {s['errors']}"]
    ret = s.get("retrieval")
    if ret:
        lines.append(
            f"- retrieval, strict: hit@1 {_fmt_rate(ret['hit@1'])}, hit@3 {_fmt_rate(ret['hit@3'])},"
            f" hit@{k} {_fmt_rate(ret['hit@k'])}, MRR {_fmt(ret['mrr'])}"
        )
        lines.append(
            f"- retrieval, loose (also counts whole-file and whole-class hits):"
            f" hit@1 {_fmt_rate(ret['loose_hit@1'])}, hit@3 {_fmt_rate(ret['loose_hit@3'])},"
            f" hit@{k} {_fmt_rate(ret['loose_hit@k'])}"
        )
        lines.append(
            f"- chunk_recall {_fmt_rate(ret['chunk_recall'])}; flow all@{k} {_fmt_rate(ret['all@k'])};"
            f" not-in-repo pointer@{k} {_fmt_rate(ret['pointer@k'])},"
            f" pointer chunk_recall {_fmt_rate(ret['pointer_chunk_recall'])}"
        )
    ans = s.get("answer")
    if ans:
        lines.append(
            f"- answers: cited_correct {_fmt_rate(ans['cited_correct'])}, cited_wrong total"
            f" {ans['cited_wrong_total']} (mean {_fmt(ans['cited_wrong_mean'])}),"
            f" all_required {_fmt_rate(ans['all_required'])}"
        )
        lines.append(
            f"- not-in-repo: admits_missing {_fmt_rate(ans['admits_missing'])}"
            f" ({_judge_label(ans.get('admits_judge'))}),"
            f" pointer_cited {_fmt_rate(ans['pointer_cited'])};"
            f" false admits on answerable questions {_fmt_rate(ans['false_admits'])}"
        )
        vs = ans.get("admit_judge_vs_heuristic")
        if vs and vs.get("n"):
            lines.append(f"- admit judge vs phrase heuristic: {vs['differ']}/{vs['n']} differ")
        lines.append(
            f"- tokens: input {ans['input_tokens']}, output {ans['output_tokens']},"
            f" cache_read {ans['cache_read_tokens']}"
        )
    if "decision_calls" in s:
        judged = int(s.get("admit_judge_calls") or 0)
        calls = f"{s['decision_calls']} calls" + (f" ({judged} admit judge)" if judged else "")
        line = f"- decisions: {decisions_name or '?'}, {calls}, ${s['decision_cost_usd']}"
        if "reranked" in s:
            line += (f", reranked {s['reranked']}/{s['n']},"
                     f" off-target {s['off_target']}, errors {s['decision_errors']}")
        lines.append(line)
    el = s.get("elapsed_s")
    if el:
        lines.append(
            f"- elapsed_s per question: mean {el['mean']}, median {el['median']}, max {el['max']},"
            f" total {el['total']}"
        )
    return lines


def _judge_label(judges: Optional[dict[str, int]]) -> str:
    """``judge: jev`` / ``judge: jev 28, rules 2``; reports without the field
    were scored by the phrase heuristic."""
    if not judges:
        return "phrase heuristic"
    names = {"rules": "rules (phrase heuristic)"}
    if len(judges) == 1:
        return f"judge: {names.get(next(iter(judges)), next(iter(judges)))}"
    return "judge: " + ", ".join(f"{name} {n}" for name, n in judges.items())


def list_questions(golden: dict[str, Any], ids: Optional[Iterable[str]] = None) -> str:
    rows = select_questions(golden, ids)
    lines = [f"{q['id']:<5} {q['category']:<15} {q['lang']:<3} {q['question']}" for q in rows]
    lines.append(f"{len(rows)} questions")
    return "\n".join(lines) + "\n"


# =============================================================================== #
# Rejudging a stored report
# =============================================================================== #

def rejudge_report(report: dict[str, Any], decisions: Any, *, source: Optional[str | Path] = None,
                   log: Any = None) -> dict[str, Any]:
    """A copy of an answer-mode ``report`` whose answers are judged again.

    Every row with an ``answer`` gets its admit fields rewritten by
    :func:`judge_admits` with ``decisions`` (from the stored ``answer_text``
    and question; an empty text makes no call), then the summary is computed
    again. ``meta`` gains ``rejudged_from``, ``judge`` and ``rejudged_at``.
    No pipeline or LLM is called. Raises ``ValueError`` for a report without
    answers (before any judge call) and lets ``DecisionsConfigError`` through.
    """
    rows = report.get("questions") if isinstance(report, dict) else None
    if not isinstance(rows, list) or not isinstance(report.get("meta"), dict):
        raise ValueError("not a run_eval report: 'meta' and 'questions' are missing")
    if not any(isinstance(r, dict) and isinstance(r.get("answer"), dict) for r in rows):
        raise ValueError("the report has no answers: rejudging needs a report run with --mode answer or both")
    new = copy.deepcopy(report)
    for row in new["questions"]:
        ans = row.get("answer")
        if not isinstance(ans, dict):
            continue
        fields = judge_admits(ans.get("answer_text") or "", row.get("question") or "", decisions, strict=True)
        ans.pop("admits_judge_error", None)
        ans.update(fields)
        if log is not None:
            print(f"[{row.get('id')}] judge {fields['admits_judge']} p={fields['admits_probability']:.2f}"
                  f" admit={_yn(fields['admits_missing'])} heuristic={_yn(fields['admits_heuristic'])}",
                  file=log, flush=True)
    new["summary"] = aggregate(new["questions"])
    new["meta"] = {
        **new["meta"],
        "rejudged_from": str(source) if source is not None else None,
        "judge": getattr(decisions, "name", None),
        "rejudged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return new


def _snippet(text: str, limit: int = 90) -> str:
    return " ".join((text or "").split())[:limit].replace("|", "\\|")


def render_rejudge_table(report: dict[str, Any]) -> str:
    """Per question: the phrase heuristic's verdict and phrase next to the
    judge's probability and verdict, ``DIFF`` where they disagree, and the
    start of the answer."""
    rows = [r for r in report["questions"] if isinstance(r.get("answer"), dict)]
    judge = report["meta"].get("judge") or "?"
    out = [
        f"## Admit judge ({judge}) vs phrase heuristic",
        "",
        "| id | category | heuristic | phrase | judge p | judge | diff | answer (first 90 chars) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    differ = 0
    for r in rows:
        a = r["answer"]
        verdict, heuristic = bool(a.get("admits_missing")), bool(a.get("admits_heuristic"))
        differ += verdict != heuristic
        judged = _yn(verdict) + (" (rules fallback)" if a.get("admits_judge_error") else "")
        p = a.get("admits_probability")
        out.append("| " + " | ".join([
            str(r.get("id")), str(r.get("category")), _yn(heuristic),
            (a.get("admits_phrase") or "-").replace("|", "\\|"),
            "-" if p is None else f"{float(p):.2f}", judged, "DIFF" if verdict != heuristic else "",
            _snippet(a.get("answer_text") or ""),
        ]) + " |")
    out += ["", f"{differ}/{len(rows)} verdicts differ."]
    return "\n".join(out) + "\n"


# =============================================================================== #
# CLI
# =============================================================================== #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_eval.py",
        description="Measure RepoFinder retrieval and answer quality on a golden set.",
    )
    p.add_argument("golden", nargs="?",
                   help="golden set JSON, e.g. eval/golden/flask.json (not with --rejudge)")
    p.add_argument("--source", help="repository URL or local checkout (default: repo.source of the golden file)")
    p.add_argument("--k", type=int, default=DEFAULT_K, help=f"hits/chunks per question (default {DEFAULT_K})")
    p.add_argument("--mode", choices=MODES,
                   help="retrieval (no LLM; the default), answer (one LLM call per question) or both")
    p.add_argument("--ids", help="comma-separated question ids, e.g. q01,q05")
    p.add_argument("--out", help="JSON report path (default: eval/results/<golden>-<mode>-<timestamp>.json)")
    p.add_argument("--data-dir", help="passed to open_workspace(data_dir=...)")
    p.add_argument("--decisions", choices=("jev", "rules"),
                   help="decision layer for the pipeline's own rerank and the admit judge of the answers"
                        " (jev needs OPENROUTER_API_KEY)")
    p.add_argument("--rejudge", metavar="REPORT",
                   help="judge the answers of a stored answer/both-mode report again (no pipeline, no LLM"
                        " answer calls) with --decisions, or the detected layer (Jev when a key is set,"
                        " else the rules), and write a new report to --out")
    p.add_argument("--candidates", type=int,
                   help="candidates handed to the decision layer (default: the pipeline's DEFAULT_CANDIDATES)")
    p.add_argument("--list", action="store_true", help="validate, print the questions and exit")
    return p


def main(argv: Optional[list[str]] = None, pipeline: Any = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except (ValueError, OSError):
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.rejudge:
        clash = [flag for flag, value in (("a golden file", args.golden), ("--list", args.list),
                                          ("--source", args.source), ("--mode", args.mode),
                                          ("--ids", args.ids), ("--data-dir", args.data_dir),
                                          ("--candidates", args.candidates)) if value]
        if clash:
            parser.error(f"--rejudge does not run the pipeline: drop {', '.join(clash)}")
        return main_rejudge(args)
    if not args.golden:
        parser.error("the golden file is required (or --rejudge REPORT)")
    mode = args.mode or "retrieval"
    try:
        golden = load_golden(args.golden)
    except GoldenError as exc:
        print(f"invalid golden file: {exc}", file=sys.stderr)
        return 2

    source = args.source or (golden.get("repo") or {}).get("source")
    local = source if source and Path(source).is_dir() else None
    errors, warnings = validate_golden(golden, local)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        print(f"invalid golden file: {len(errors)} error(s)", file=sys.stderr)
        return 2

    ids = args.ids.split(",") if args.ids else None
    try:
        if args.list:
            sys.stdout.write(list_questions(golden, ids))
            return 0
        select_questions(golden, ids)
    except GoldenError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.k < 1:
        print("error: --k must be >= 1", file=sys.stderr)
        return 2

    if args.candidates is not None and args.candidates < 1:
        print("error: --candidates must be >= 1", file=sys.stderr)
        return 2
    decisions = None
    if args.decisions:
        from repofinder import decisions as decisions_mod

        try:
            decisions = decisions_mod.detect_decisions(args.decisions)
        except decisions_mod.DecisionsConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    try:
        if pipeline is None:
            pipeline = import_pipeline()
        report = run_eval(golden, source, mode, args.k, pipeline,
                          ids=ids, data_dir=args.data_dir, log=sys.stderr,
                          decisions=decisions, candidates=args.candidates)
    except Exception as exc:  # noqa: BLE001 - a setup failure, not a measurement
        print(f"error: could not run the pipeline: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    report["meta"]["golden"] = str(args.golden)
    report["meta"]["validation_warnings"] = warnings
    out = Path(args.out) if args.out else default_out_path(args.golden, mode)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.stdout.write(render_markdown(report))
    print(f"\nJSON report: {out}")
    return 0


def main_rejudge(args: argparse.Namespace) -> int:
    """``--rejudge REPORT``: judge the stored answers again and write a new report."""
    src = Path(args.rejudge)
    try:
        report = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"error: cannot read the report {src}: {exc}", file=sys.stderr)
        return 2
    from repofinder import decisions as decisions_mod

    try:
        decisions = decisions_mod.detect_decisions(args.decisions)
    except decisions_mod.DecisionsConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    judge = getattr(decisions, "name", None) or "judge"
    out = Path(args.out) if args.out else src.with_name(
        f"{src.stem}-rejudged-{judge}-{time.strftime('%Y%m%d-%H%M%S')}.json")
    if out.resolve() == src.resolve():
        print("error: --out must not overwrite the report being rejudged", file=sys.stderr)
        return 2
    try:
        new = rejudge_report(report, decisions, source=args.rejudge, log=sys.stderr)
    except ValueError as exc:
        print(f"error: {src}: {exc}", file=sys.stderr)
        return 2
    except decisions_mod.DecisionsError as exc:
        print(f"error: the admit judge failed: {exc}", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.stdout.write(render_markdown(new))
    sys.stdout.write("\n" + render_rejudge_table(new))
    print(f"\nJSON report: {out}")
    return 0


def default_out_path(golden_path: str | Path, mode: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Path(__file__).resolve().parent / "results" / f"{Path(golden_path).stem}-{mode}-{stamp}.json"


if __name__ == "__main__":
    sys.exit(main())

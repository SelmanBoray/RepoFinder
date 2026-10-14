"""Chat routing eval: score the decision layer's route for every turn of scripted conversations.

Usage (from the repository root)::

    .venv\\Scripts\\python eval/route_eval.py eval/golden/chat_routes.json --list
    .venv\\Scripts\\python eval/route_eval.py eval/golden/chat_routes.json --decisions rules
    .venv\\Scripts\\python eval/route_eval.py eval/golden/chat_routes.json --decisions jev --ids c01,c02 --out r.json
    .venv\\Scripts\\python eval/route_eval.py eval/golden/chat_routes.json --decisions jev --repo-info none

The golden file holds conversations; each turn is a user message with its
expected route (one of ``repofinder.decisions.ROUTES``) and, except perhaps on
the last turn, the assistant reply that followed it::

    {"version": 1,
     "repos": {"flask": {"description": "Flask is a lightweight WSGI web application framework. ..."}},
     "conversations": [
       {"id": "c01", "repo": "flask", "lang": "en", "ascii": false, "notes": "...",
        "turns": [{"user": "Where is ...?", "route": "locate", "assistant": "..."},
                  {"user": "who calls it?", "route": "followup"}]}]}

``ascii`` (optional) marks Turkish typed without its letters; such turns are
reported as ``tr-ascii``. Every turn is routed with
``route_message(decisions, user, history)``, where ``history`` is the golden
conversation before it, ``[("user", u1), ("assistant", a1), ...]`` oldest first
(a missing assistant reply is skipped). The history is always the golden one,
so a wrong or failed route does not change the turns after it.
``route_message`` itself shows a model backend only the last
``HISTORY_EXCHANGES`` exchanges, each message cut to ``HISTORY_TEXT_CHARS``.

The optional top-level ``repos`` maps a conversation's ``repo`` to a
``description`` of that repository. ``--repo-info`` says what the route state
tells the layer about the repository (``route_message(..., repo=RepoInfo(...))``):
``full`` (the default) its name and the description from ``repos`` (the name
only when there is none), ``name`` its name only, ``none`` nothing (the state
without the repository block, for the ablation). A conversation without
``repo`` gets no repository block in any mode. The mode is recorded in
``meta.repo_info``.

The layer comes from ``detect_decisions(--decisions)``: ``rules`` is offline
and free; ``jev`` makes one paid call per turn; without the flag it is Jev when
a key is set, else the rules. A ``DecisionsError`` on one turn is recorded and
the run goes on; a ``DecisionsConfigError`` (no key, a rejected key, no credit)
stops the run.

Output: Markdown on stdout (a table per turn, then accuracy, precision and
recall per route, accuracy per language, the confusion matrix, mean
p(expected), calls and cost) and, with ``--out``, a JSON report (``meta``,
``summary``, ``turns``). Exit codes: 0 whatever the accuracy is (a measurement,
not a gate); 2 for an invalid golden file or an unknown ``--ids`` entry; 1 when
the decision layer cannot be set up or fails for good.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from repofinder import decisions as _decisions

GOLDEN_VERSION = 1
LANGS = ("en", "tr")
LANG_GROUPS = ("en", "tr", "tr-ascii")
SNIPPET_CHARS = 70

REPO_INFO_FULL = "full"      # name + description
REPO_INFO_NAME = "name"      # name only
REPO_INFO_NONE = "none"      # no repository block (the state before RepoInfo)
REPO_INFO_MODES = (REPO_INFO_FULL, REPO_INFO_NAME, REPO_INFO_NONE)

_TOP_KEYS = frozenset({"version", "conversations", "notes", "description", "repos"})
_REPO_KEYS = frozenset({"description"})
_CONVERSATION_KEYS = frozenset({"id", "repo", "lang", "ascii", "notes", "turns"})
_TURN_KEYS = frozenset({"user", "route", "assistant", "notes"})


class GoldenError(ValueError):
    """The golden file cannot be read, or a requested conversation id does not exist."""


# =============================================================================== #
# Golden set
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


def validate_golden(golden: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Check a chat-routes golden set. Returns ``(errors, warnings)``; any error makes it invalid.

    Errors: a version other than 1, ``repos`` that is not an object of objects
    or a ``description`` there that is not a string, no conversations, a
    missing or duplicate conversation id, a ``lang`` other than en/tr, a
    non-boolean ``ascii``, no turns, an empty user message, a route outside
    ``ROUTES``, an ``assistant`` that is not a string. Warnings: no version,
    unknown keys, a conversation ``repo`` missing from ``repos`` (when
    ``repos`` is given), ``ascii`` on an English conversation, an empty
    assistant reply before the last turn (the history then lacks it).
    """
    errors: list[str] = []
    warnings: list[str] = []
    version = golden.get("version")
    if version is None:
        warnings.append(f"no 'version' (expected {GOLDEN_VERSION})")
    elif version != GOLDEN_VERSION:
        errors.append(f"unsupported version {version!r} (expected {GOLDEN_VERSION})")
    for key in sorted(set(golden) - _TOP_KEYS):
        warnings.append(f"unknown top-level key {key!r}")
    repos = golden.get("repos")
    if repos is not None and not isinstance(repos, dict):
        errors.append("'repos' must be an object mapping repository names to objects")
        repos = None
    for name, entry in (repos or {}).items():
        if not isinstance(entry, dict):
            errors.append(f"repos.{name}: must be an object")
            continue
        for key in sorted(set(entry) - _REPO_KEYS):
            warnings.append(f"repos.{name}: unknown key {key!r}")
        if "description" in entry and not isinstance(entry["description"], str):
            errors.append(f"repos.{name}: 'description' must be a string when given")

    conversations = golden.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        errors.append("'conversations' must be a non-empty list")
        return errors, warnings

    seen: set[str] = set()
    for n, conv in enumerate(conversations):
        where = f"conversations[{n}]"
        if not isinstance(conv, dict):
            errors.append(f"{where}: must be an object")
            continue
        cid = conv.get("id")
        if not isinstance(cid, str) or not cid.strip():
            errors.append(f"{where}: 'id' must be a non-empty string")
        else:
            where = cid
            if cid in seen:
                errors.append(f"{cid}: duplicate id")
            seen.add(cid)
        for key in sorted(set(conv) - _CONVERSATION_KEYS):
            warnings.append(f"{where}: unknown key {key!r}")
        if "repo" in conv and not isinstance(conv["repo"], str):
            errors.append(f"{where}: 'repo' must be a string")
        elif repos is not None and isinstance(conv.get("repo"), str) and conv["repo"] not in repos:
            warnings.append(f"{where}: repo {conv['repo']!r} has no entry in 'repos' (its state shows the name only)")
        lang = conv.get("lang")
        if lang not in LANGS:
            errors.append(f"{where}: 'lang' must be one of {', '.join(LANGS)}, got {lang!r}")
        ascii_only = conv.get("ascii", False)
        if not isinstance(ascii_only, bool):
            errors.append(f"{where}: 'ascii' must be true or false when given")
        elif ascii_only and lang == "en":
            warnings.append(f"{where}: 'ascii' only applies to Turkish conversations")
        if "notes" in conv and not isinstance(conv["notes"], str):
            errors.append(f"{where}: 'notes' must be a string")

        turns = conv.get("turns")
        if not isinstance(turns, list) or not turns:
            errors.append(f"{where}: 'turns' must be a non-empty list")
            continue
        for i, turn in enumerate(turns, start=1):
            at = f"{where} turn {i}"
            if not isinstance(turn, dict):
                errors.append(f"{at}: must be an object")
                continue
            for key in sorted(set(turn) - _TURN_KEYS):
                warnings.append(f"{at}: unknown key {key!r}")
            user = turn.get("user")
            if not isinstance(user, str) or not user.strip():
                errors.append(f"{at}: 'user' must be a non-empty string")
            route = turn.get("route")
            if route not in _decisions.ROUTES:
                errors.append(f"{at}: 'route' must be one of {', '.join(_decisions.ROUTES)}, got {route!r}")
            assistant = turn.get("assistant")
            if assistant is not None and not isinstance(assistant, str):
                errors.append(f"{at}: 'assistant' must be a string when given")
            elif i < len(turns) and not (assistant or "").strip():
                warnings.append(f"{at}: no 'assistant' reply before the next turn (left out of the history)")
    return errors, warnings


def select_conversations(golden: dict[str, Any], ids: Optional[Iterable[str]] = None) -> list[dict[str, Any]]:
    """Conversations in file order, optionally only ``ids``. Unknown ids raise GoldenError."""
    conversations = list(golden.get("conversations") or [])
    if ids is None:
        return conversations
    wanted = [i.strip() for i in ids if i and i.strip()]
    known = {c.get("id") for c in conversations}
    unknown = [i for i in wanted if i not in known]
    if unknown:
        raise GoldenError(f"unknown conversation id(s): {', '.join(unknown)}")
    keep = set(wanted)
    return [c for c in conversations if c.get("id") in keep]


def lang_group(lang: Optional[str], ascii_only: Any = False) -> str:
    """``tr-ascii`` for ASCII-only Turkish, else the language."""
    return "tr-ascii" if lang == "tr" and ascii_only is True else str(lang)


def history_before(turns: list[dict[str, Any]], index: int) -> list[tuple[str, str]]:
    """The golden history before ``turns[index]`` (0-based): ``(role, text)``
    pairs, oldest first; an empty or missing assistant reply is skipped."""
    history: list[tuple[str, str]] = []
    for turn in turns[:index]:
        history.append(("user", turn["user"]))
        assistant = turn.get("assistant")
        if isinstance(assistant, str) and assistant.strip():
            history.append(("assistant", assistant))
    return history


def repo_info_for(conv: dict[str, Any], repos: Any,
                  mode: str = REPO_INFO_FULL) -> Optional[_decisions.RepoInfo]:
    """The ``RepoInfo`` that ``route_message`` gets for the turns of ``conv``:
    None in mode ``none`` or when the conversation names no repo; else its
    name, with ``repos[repo]["description"]`` in mode ``full`` (None when missing)."""
    if mode not in REPO_INFO_MODES:
        raise ValueError(f"unknown repo-info mode {mode!r}: use one of {', '.join(REPO_INFO_MODES)}")
    name = conv.get("repo")
    if mode == REPO_INFO_NONE or not isinstance(name, str) or not name.strip():
        return None
    if mode == REPO_INFO_NAME:
        return _decisions.RepoInfo(name)
    entry = repos.get(name) if isinstance(repos, dict) else None
    description = entry.get("description") if isinstance(entry, dict) else None
    return _decisions.RepoInfo(name, description if isinstance(description, str) and description.strip() else None)


# =============================================================================== #
# Running
# =============================================================================== #

def run_routes(golden: dict[str, Any], decisions: Any, *, ids: Optional[Iterable[str]] = None,
               golden_path: Optional[str | Path] = None, log: Any = None,
               repo_info: str = REPO_INFO_FULL) -> dict[str, Any]:
    """Route every turn of the selected conversations with ``decisions`` and score it.

    ``repo_info`` (one of :data:`REPO_INFO_MODES`) says what the route state
    tells about the repository; see :func:`repo_info_for`.
    Returns ``{"meta", "summary", "turns"}``. A :class:`DecisionsError` on a turn
    is recorded in that turn (no prediction, counted as wrong) and the run goes
    on; :class:`DecisionsConfigError` propagates. ``meta.totals.calls`` is the
    growth of ``decisions.usage.calls`` (cached Jev answers are not calls).
    """
    if repo_info not in REPO_INFO_MODES:
        raise ValueError(f"unknown repo-info mode {repo_info!r}: use one of {', '.join(REPO_INFO_MODES)}")
    conversations = select_conversations(golden, ids)
    usage = getattr(decisions, "usage", None)
    calls_before = int(getattr(usage, "calls", 0) or 0)
    started = time.time()
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for conv in conversations:
        turns = conv["turns"]
        repo = repo_info_for(conv, golden.get("repos"), repo_info)
        for i in range(len(turns)):
            rows.append(_route_turn(decisions, conv, turns, i, repo, log))
    calls = int(getattr(usage, "calls", 0) or 0) - calls_before if usage is not None else len(rows)
    summary = summarize(rows)
    return {
        "meta": {
            "golden": str(golden_path) if golden_path is not None else None,
            "decisions": getattr(decisions, "name", None),
            "decisions_model": getattr(decisions, "model", None),
            "repo_info": repo_info,
            "ids": [c["id"] for c in conversations],
            "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
            "totals": {
                "conversations": len(conversations),
                "turns": len(rows),
                "correct": summary["accuracy"]["correct"],
                "accuracy": summary["accuracy"]["rate"],
                "errors": summary["errors"],
                "calls": max(0, calls),
                "cost_usd": summary["cost_usd"],
                "elapsed_s": round(time.perf_counter() - t0, 3),
            },
            "constants": {
                "ROUTES": list(_decisions.ROUTES),
                "HISTORY_EXCHANGES": _decisions.HISTORY_EXCHANGES,
                "HISTORY_TEXT_CHARS": _decisions.HISTORY_TEXT_CHARS,
                "MESSAGE_CHARS": _decisions.MESSAGE_CHARS,
                "REPO_DESCRIPTION_CHARS": _decisions.REPO_DESCRIPTION_CHARS,
            },
        },
        "summary": summary,
        "turns": rows,
    }


def _route_turn(decisions: Any, conv: dict[str, Any], turns: list[dict[str, Any]], i: int,
                repo: Optional[_decisions.RepoInfo], log: Any) -> dict[str, Any]:
    turn = turns[i]
    expected = turn["route"]
    row: dict[str, Any] = {
        "id": conv["id"],
        "turn": i + 1,
        "lang": conv.get("lang"),
        "ascii": conv.get("ascii", False) is True,
        "message": turn["user"],
        "expected": expected,
        "predicted": None,
        "correct": False,
        "probabilities": {},
        "p_expected": None,
        "cost_usd": 0.0,
        "elapsed_s": 0.0,
        "error": None,
    }
    history = history_before(turns, i)
    t = time.perf_counter()
    try:
        result = _decisions.route_message(decisions, turn["user"], history, repo=repo)
    except _decisions.DecisionsConfigError:
        raise
    except _decisions.DecisionsError as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    else:
        probs = {str(k): float(v) for k, v in (result.probabilities or {}).items()}
        row.update(
            predicted=result.route,
            correct=result.route == expected,
            probabilities=probs,
            p_expected=probs.get(expected),
            cost_usd=float(result.cost_usd or 0.0),
        )
    row["elapsed_s"] = round(time.perf_counter() - t, 4)
    if log is not None:
        if row["error"]:
            status = f"ERROR {row['error']}"
        else:
            status = f"{expected} -> {row['predicted']}" + ("" if row["correct"] else " X")
        print(f"[{row['id']}#{row['turn']}] {row['elapsed_s']:.2f}s {status}", file=log, flush=True)
    return row


# =============================================================================== #
# Scoring
# =============================================================================== #

def _ratio(num: int, den: int) -> Optional[float]:
    return round(num / den, 3) if den else None


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return round(statistics.fmean(vals), 3) if vals else None


def _accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(1 for r in rows if r["correct"])
    return {"n": len(rows), "correct": correct, "rate": _ratio(correct, len(rows))}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy overall and per language group, precision/recall per route, the
    confusion matrix and mean p(expected).

    A turn with an error has no prediction: it counts as wrong for accuracy and
    recall (it stays in the route's support), is left out of precision and of
    the 5x5 ``confusion`` matrix (it is counted in ``errors_by_expected``
    instead) and of the mean p(expected) on wrong turns.
    """
    routes = _decisions.ROUTES
    confusion = {exp: {pred: 0 for pred in routes} for exp in routes}
    errors_by_expected = {exp: 0 for exp in routes}
    for r in rows:
        if r.get("error") or r.get("predicted") is None:
            errors_by_expected[r["expected"]] += 1
        else:
            confusion[r["expected"]][r["predicted"]] += 1

    per_route: dict[str, Any] = {}
    for route in routes:
        support = sum(1 for r in rows if r["expected"] == route)
        predicted = sum(1 for r in rows if r.get("predicted") == route)
        hits = sum(1 for r in rows if r["expected"] == route and r["correct"])
        per_route[route] = {
            "support": support,
            "predicted": predicted,
            "correct": hits,
            "precision": _ratio(hits, predicted),
            "recall": _ratio(hits, support),
        }

    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(lang_group(r.get("lang"), r.get("ascii")), []).append(r)
    order = [g for g in LANG_GROUPS if g in groups] + sorted(g for g in groups if g not in LANG_GROUPS)

    right = [r for r in rows if r["correct"] and r.get("p_expected") is not None]
    wrong = [r for r in rows if not r["correct"] and r.get("p_expected") is not None]
    elapsed = [r["elapsed_s"] for r in rows if r.get("elapsed_s") is not None]
    return {
        "n": len(rows),
        "errors": sum(1 for r in rows if r.get("error")),
        "accuracy": _accuracy(rows),
        "per_route": per_route,
        "per_lang": {g: _accuracy(groups[g]) for g in order},
        "confusion": confusion,
        "errors_by_expected": errors_by_expected,
        "p_expected": {
            "correct_mean": _mean(r["p_expected"] for r in right),
            "correct_n": len(right),
            "wrong_mean": _mean(r["p_expected"] for r in wrong),
            "wrong_n": len(wrong),
        },
        "cost_usd": round(sum(float(r.get("cost_usd") or 0.0) for r in rows), 6),
        "elapsed_s": {
            "total": round(sum(elapsed), 3),
            "mean": round(statistics.fmean(elapsed), 4) if elapsed else None,
        },
    }


# =============================================================================== #
# Reporting
# =============================================================================== #

def _snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    return " ".join((text or "").split())[:limit].replace("|", "\\|")


def _fmt(value: Optional[float], digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _fmt_acc(acc: dict[str, Any]) -> str:
    return f"{_fmt(acc.get('rate'))} ({acc['correct']}/{acc['n']})"


def render_markdown(report: dict[str, Any]) -> str:
    """The per-turn table and the summary, as Markdown."""
    meta, rows, s = report["meta"], report["turns"], report["summary"]
    totals = meta["totals"]
    name = Path(meta["golden"]).name if meta.get("golden") else "chat routes"
    out = [
        f"# RepoFinder chat routing eval: {name} (decisions={meta.get('decisions') or '?'})",
        "",
        f"| id | turn | lang | expected | predicted | p(expected) | wrong | message (first {SNIPPET_CHARS} chars) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append("| " + " | ".join([
            str(r["id"]), str(r["turn"]), lang_group(r.get("lang"), r.get("ascii")), r["expected"],
            "error" if r.get("error") else str(r.get("predicted")),
            _fmt(r.get("p_expected")), "" if r["correct"] else "X", _snippet(r["message"]),
        ]) + " |")

    pe = s["p_expected"]
    mean_s = s["elapsed_s"]["mean"]
    model = f" ({meta['decisions_model']})" if meta.get("decisions_model") else ""
    out += [
        "", "## Summary", "",
        f"- turns: {s['n']} in {totals['conversations']} conversations, errors: {s['errors']}",
        f"- accuracy: {_fmt_acc(s['accuracy'])}",
        f"- mean p(expected): correct turns {_fmt(pe['correct_mean'])} (n={pe['correct_n']}),"
        f" wrong turns {_fmt(pe['wrong_mean'])} (n={pe['wrong_n']})",
        f"- decisions: {meta.get('decisions') or '?'}{model}, {totals['calls']} calls,"
        f" total cost ${s['cost_usd']}, mean elapsed per turn {_fmt(mean_s, 4)}s",
        f"- repository in the route state (--repo-info): {meta.get('repo_info') or '?'}",
        "",
        "| route | support | predicted | correct | precision | recall |",
        "|---|---|---|---|---|---|",
    ]
    for route, m in s["per_route"].items():
        out.append(f"| {route} | {m['support']} | {m['predicted']} | {m['correct']}"
                   f" | {_fmt(m['precision'])} | {_fmt(m['recall'])} |")
    out += ["", "| lang | turns | accuracy |", "|---|---|---|"]
    for group, acc in s["per_lang"].items():
        out.append(f"| {group} | {acc['n']} | {_fmt_acc(acc)} |")

    routes = list(_decisions.ROUTES)
    with_errors = s["errors"] > 0
    header = ["expected \\ predicted"] + routes + (["error"] if with_errors else [])
    out += ["", "Confusion matrix (rows: expected, columns: predicted):", "",
            "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for exp in routes:
        cells = [exp] + [str(s["confusion"][exp][pred]) for pred in routes]
        if with_errors:
            cells.append(str(s["errors_by_expected"][exp]))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def list_conversations(golden: dict[str, Any], ids: Optional[Iterable[str]] = None) -> str:
    convs = select_conversations(golden, ids)
    lines: list[str] = []
    n_turns = 0
    for c in convs:
        group = lang_group(c.get("lang"), c.get("ascii"))
        lines.append(f"{c['id']} [{c.get('repo') or '-'}, {group}] {len(c['turns'])} turns")
        for i, t in enumerate(c["turns"], start=1):
            lines.append(f"  {i:>2} {t['route']:<9} {_snippet(t['user'], 100)}")
        n_turns += len(c["turns"])
    lines.append(f"{len(convs)} conversations, {n_turns} turns")
    return "\n".join(lines) + "\n"


# =============================================================================== #
# CLI
# =============================================================================== #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="route_eval.py",
        description="Score the chat router (decisions.route_message) on scripted conversations.",
    )
    p.add_argument("golden", help="chat-routes golden set JSON, e.g. eval/golden/chat_routes.json")
    p.add_argument("--decisions", choices=("jev", "rules"),
                   help="decision layer (default: Jev when OPENROUTER_API_KEY or TYPESAFE_API_KEY is set,"
                        " else the rules); jev makes one paid call per turn")
    p.add_argument("--ids", help="comma-separated conversation ids, e.g. c01,c02")
    p.add_argument("--repo-info", choices=REPO_INFO_MODES, default=REPO_INFO_FULL,
                   help="what the route state says about the repository: full = name + the 'repos'"
                        " description (default), name = name only, none = nothing (the ablation)")
    p.add_argument("--out", help="also write the JSON report here (default: Markdown on stdout only)")
    p.add_argument("--list", action="store_true", help="validate, print the conversations and exit")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except (ValueError, OSError):
            pass
    args = build_parser().parse_args(argv)
    try:
        golden = load_golden(args.golden)
    except GoldenError as exc:
        print(f"invalid golden file: {exc}", file=sys.stderr)
        return 2
    errors, warnings = validate_golden(golden)
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
            sys.stdout.write(list_conversations(golden, ids))
            return 0
        select_conversations(golden, ids)
    except GoldenError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        decisions = _decisions.detect_decisions(args.decisions)
        report = run_routes(golden, decisions, ids=ids, golden_path=args.golden, log=sys.stderr,
                            repo_info=args.repo_info)
    except _decisions.DecisionsConfigError as exc:
        print(f"error: the decision layer cannot be used: {exc}", file=sys.stderr)
        if _decisions.JEV_SETUP_HELP not in str(exc):
            print("\n" + _decisions.JEV_SETUP_HELP, file=sys.stderr)
        return 1
    report["meta"]["validation_warnings"] = warnings

    sys.stdout.write(render_markdown(report))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Tests for the chat routing eval in ``eval/route_eval.py``.

Every golden set here is synthetic and written to ``tmp_path``; the real
``eval/golden/chat_routes.json`` is not read. Runs use ``FakeDecisions`` (a
scripted layer) or ``RulesDecisions`` (offline); Jev is never called.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from repofinder import decisions as decisions_mod
from repofinder.decisions import (
    ROUTES,
    Choice,
    DecisionsConfigError,
    FakeDecisions,
    JEV_SETUP_HELP,
    RepoInfo,
    RouteResult,
    RulesDecisions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "eval"
KEY_ENVS = (decisions_mod.OPENROUTER_KEY_ENV, decisions_mod.TYPESAFE_KEY_ENV, decisions_mod.ENV_DECISIONS)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


route_eval = _load_module("route_eval", EVAL_DIR / "route_eval.py")


# --------------------------------------------------------------------------- #
# Synthetic golden sets
# --------------------------------------------------------------------------- #

U1, A1 = "Where is the route decorator defined?", "Scaffold.route in src/flask/sansio/scaffold.py:343-373."
U2, A2 = "who calls it?", "Nothing in Flask calls it; applications do."
U3 = "thanks!"


def golden_set() -> dict[str, Any]:
    """Three conversations, six turns; the rules get five right (they never say outside)."""
    return {
        "version": 1,
        "conversations": [
            {"id": "c01", "repo": "flask", "lang": "en", "ascii": False, "notes": "locate, then follow up",
             "turns": [
                 {"user": U1, "route": "locate", "assistant": A1},
                 {"user": U2, "route": "followup", "assistant": A2},
                 {"user": U3, "route": "chat"},
             ]},
            {"id": "c02", "repo": "flask", "lang": "tr", "ascii": True,
             "turns": [
                 {"user": "Merhaba, ne yapabilirsin?", "route": "chat",
                  "assistant": "Bu deponun kodunda arama yapabilirim."},
                 {"user": "jinja sablonlari nasil derleniyor?", "route": "outside"},
             ]},
            {"id": "c03", "repo": "flask", "lang": "tr",
             "turns": [{"user": "Oturum çerezi nasıl imzalanıyor?", "route": "explain"}]},
        ],
    }


def write_golden(tmp_path: Path, data: Any, name: str = "chat_routes.json") -> Path:
    path = tmp_path / name
    text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, indent=1)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def no_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> pytest.MonkeyPatch:
    """No Jev key or decisions variable, and a working directory without a .env."""
    for name in KEY_ENVS:
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    return monkeypatch


def use_layer(monkeypatch: pytest.MonkeyPatch, layer: Any) -> list[Any]:
    """Make ``detect_decisions`` return ``layer``; returns the names it was asked for."""
    asked: list[Any] = []

    def detect(preferred: Any = None) -> Any:
        asked.append(preferred)
        return layer

    monkeypatch.setattr(route_eval._decisions, "detect_decisions", detect)
    return asked


# --------------------------------------------------------------------------- #
# History
# --------------------------------------------------------------------------- #

def test_history_for_turn_3_is_exactly_the_previous_two_exchanges(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, list[tuple[str, str]]]] = []

    def spy(decisions: Any, message: str, history: list[tuple[str, str]], repo: Any = None) -> RouteResult:
        seen.append((message, list(history)))
        return RouteResult("locate", {"locate": 1.0}, 0.0)

    monkeypatch.setattr(route_eval._decisions, "route_message", spy)
    route_eval.run_routes(golden_set(), RulesDecisions())

    assert [m for m, _h in seen][:3] == [U1, U2, U3]
    assert seen[0][1] == []
    assert seen[1][1] == [("user", U1), ("assistant", A1)]
    assert seen[2][1] == [("user", U1), ("assistant", A1), ("user", U2), ("assistant", A2)]
    # a new conversation starts with no history
    assert seen[3][1] == []
    assert seen[5] == ("Oturum çerezi nasıl imzalanıyor?", [])


def test_a_missing_assistant_reply_is_left_out_of_the_history(monkeypatch: pytest.MonkeyPatch) -> None:
    golden = golden_set()
    del golden["conversations"][0]["turns"][1]["assistant"]
    errors, warnings = route_eval.validate_golden(golden)
    assert errors == []
    assert any("c01 turn 2" in w and "assistant" in w for w in warnings)

    seen: list[list[tuple[str, str]]] = []

    def spy(decisions: Any, message: str, history: list[tuple[str, str]], repo: Any = None) -> RouteResult:
        seen.append(list(history))
        return RouteResult("chat", {"chat": 1.0}, 0.0)

    monkeypatch.setattr(route_eval._decisions, "route_message", spy)
    route_eval.run_routes(golden, RulesDecisions(), ids=["c01"])
    assert seen[2] == [("user", U1), ("assistant", A1), ("user", U2)]


def test_a_model_backend_sees_the_previous_exchanges_in_its_state() -> None:
    fake = FakeDecisions(choices=["locate", "followup", "chat"])
    route_eval.run_routes(golden_set(), fake, ids=["c01"])
    state = fake.calls[2].state
    assert f"user: {U1}\nassistant: {A1}\nuser: {U2}\nassistant: {A2}" in state
    assert state.rstrip().endswith(f"New user message:\n{U3}")
    assert "none, this is the first message" in fake.calls[0].state


# --------------------------------------------------------------------------- #
# Repository block (--repo-info)
# --------------------------------------------------------------------------- #

FLASK_ABOUT = "Flask is a lightweight WSGI web application framework."


def golden_with_repos() -> dict[str, Any]:
    """golden_set() plus ``repos`` for flask, a requests conversation without a
    ``repos`` entry and a conversation without a repo."""
    golden = golden_set()
    golden["repos"] = {"flask": {"description": FLASK_ABOUT}}
    golden["conversations"] += [
        {"id": "c04", "repo": "requests", "lang": "en", "turns": [{"user": "Where is the retry loop?",
                                                                   "route": "outside"}]},
        {"id": "c05", "lang": "en", "turns": [{"user": "hello", "route": "chat"}]},
    ]
    return golden


def spy_repos(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any]]:
    """Replace route_message; returns the (conversation message, repo) it receives."""
    seen: list[tuple[str, Any]] = []

    def spy(decisions: Any, message: str, history: list[tuple[str, str]], repo: Any = None) -> RouteResult:
        seen.append((message, repo))
        return RouteResult("chat", {"chat": 1.0}, 0.0)

    monkeypatch.setattr(route_eval._decisions, "route_message", spy)
    return seen


@pytest.mark.parametrize("mode, flask, requests", [
    ("full", RepoInfo("flask", FLASK_ABOUT), RepoInfo("requests", None)),
    ("name", RepoInfo("flask"), RepoInfo("requests")),
    ("none", None, None),
])
def test_repo_info_modes_decide_what_route_message_receives(monkeypatch: pytest.MonkeyPatch, mode: str,
                                                            flask: Any, requests: Any) -> None:
    seen = spy_repos(monkeypatch)
    report = route_eval.run_routes(golden_with_repos(), RulesDecisions(), repo_info=mode)
    assert [repo for _m, repo in seen] == [flask] * 6 + [requests, None]   # c05 names no repo
    assert report["meta"]["repo_info"] == mode
    assert report["meta"]["constants"]["REPO_DESCRIPTION_CHARS"] == decisions_mod.REPO_DESCRIPTION_CHARS
    assert f"- repository in the route state (--repo-info): {mode}" in route_eval.render_markdown(report)


def test_the_default_mode_is_full_and_unknown_modes_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = spy_repos(monkeypatch)
    report = route_eval.run_routes(golden_with_repos(), RulesDecisions(), ids=["c01"])
    assert report["meta"]["repo_info"] == "full" and seen[0][1] == RepoInfo("flask", FLASK_ABOUT)
    with pytest.raises(ValueError, match="unknown repo-info mode"):
        route_eval.run_routes(golden_with_repos(), RulesDecisions(), repo_info="both")


@pytest.mark.parametrize("mode, head", [
    ("full", f"Repository: flask\nAbout it: {FLASK_ABOUT}\n\nRecent conversation: none"),
    ("name", "Repository: flask\n\nRecent conversation: none"),
    ("none", "Recent conversation: none"),
])
def test_the_cli_flag_sets_the_state_a_model_backend_sees(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                          capsys: pytest.CaptureFixture[str],
                                                          mode: str, head: str) -> None:
    fake = FakeDecisions(choices=["locate"])
    use_layer(monkeypatch, fake)
    path = write_golden(tmp_path, golden_with_repos())
    out = tmp_path / "r.json"
    assert route_eval.main([str(path), "--ids", "c03", "--repo-info", mode, "--out", str(out)]) == 0
    assert fake.calls[0].state.startswith(head)
    assert json.loads(out.read_text(encoding="utf-8"))["meta"]["repo_info"] == mode
    capsys.readouterr()


def test_the_rules_score_the_same_in_every_mode() -> None:
    reports = [route_eval.run_routes(golden_with_repos(), RulesDecisions(), repo_info=m)
               for m in route_eval.REPO_INFO_MODES]
    predicted = [[t["predicted"] for t in r["turns"]] for r in reports]
    assert predicted[0] == predicted[1] == predicted[2]


def _repos_not_object(g: dict[str, Any]) -> None:
    g["repos"] = ["flask"]


def _repo_entry_not_object(g: dict[str, Any]) -> None:
    g["repos"] = {"flask": "a framework"}


def _description_not_text(g: dict[str, Any]) -> None:
    g["repos"] = {"flask": {"description": 3}}


@pytest.mark.parametrize("break_it, message", [
    (_repos_not_object, "'repos' must be an object mapping repository names to objects"),
    (_repo_entry_not_object, "repos.flask: must be an object"),
    (_description_not_text, "repos.flask: 'description' must be a string when given"),
])
def test_invalid_repos_are_errors(break_it: Any, message: str) -> None:
    golden = golden_set()
    break_it(golden)
    errors, _warnings = route_eval.validate_golden(golden)
    assert errors == [message]


def test_repos_warnings() -> None:
    golden = golden_with_repos()
    golden["repos"]["flask"]["commit"] = "d086db8"
    errors, warnings = route_eval.validate_golden(golden)
    assert errors == []
    assert warnings == ["repos.flask: unknown key 'commit'",
                        "c04: repo 'requests' has no entry in 'repos' (its state shows the name only)"]
    assert route_eval.validate_golden(golden_set()) == ([], [])          # no repos, no repo warnings


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def _choice(label: str, probs: dict[str, float], cost: float = 0.0) -> Choice:
    full = {r: probs.get(r, 0.0) for r in ROUTES}
    return Choice(label, full, None, cost, 0)


def test_accuracy_confusion_and_per_route_numbers_on_a_scripted_case() -> None:
    # expected: locate, followup, chat | chat, outside | explain
    script = [
        _choice("locate", {"locate": 0.9, "explain": 0.1}, 0.001),      # right, p=0.9
        _choice("locate", {"locate": 0.6, "followup": 0.4}, 0.001),     # wrong (followup), p=0.4
        _choice("chat", {"chat": 0.8, "followup": 0.2}, 0.001),         # right, p=0.8
        _choice("chat", {"chat": 0.7, "locate": 0.3}, 0.001),           # right, p=0.7
        _choice("explain", {"explain": 0.5, "outside": 0.3, "locate": 0.2}, 0.001),  # wrong, p=0.3
        _choice("explain", {"explain": 1.0}, 0.001),                    # right, p=1.0
    ]
    fake = FakeDecisions(choices=script)
    report = route_eval.run_routes(golden_set(), fake, golden_path="x/chat_routes.json")
    s = report["summary"]

    assert s["accuracy"] == {"n": 6, "correct": 4, "rate": 0.667}
    assert s["errors"] == 0
    pr = s["per_route"]
    assert pr["locate"] == {"support": 1, "predicted": 2, "correct": 1, "precision": 0.5, "recall": 1.0}
    assert pr["followup"] == {"support": 1, "predicted": 0, "correct": 0, "precision": None, "recall": 0.0}
    assert pr["chat"] == {"support": 2, "predicted": 2, "correct": 2, "precision": 1.0, "recall": 1.0}
    assert pr["explain"] == {"support": 1, "predicted": 2, "correct": 1, "precision": 0.5, "recall": 1.0}
    assert pr["outside"] == {"support": 1, "predicted": 0, "correct": 0, "precision": None, "recall": 0.0}

    cm = s["confusion"]
    assert list(cm) == list(ROUTES) and all(list(row) == list(ROUTES) for row in cm.values())
    assert cm["followup"]["locate"] == 1 and cm["outside"]["explain"] == 1
    assert cm["locate"]["locate"] == 1 and cm["chat"]["chat"] == 2 and cm["explain"]["explain"] == 1
    assert sum(sum(row.values()) for row in cm.values()) == 6

    assert s["per_lang"] == {
        "en": {"n": 3, "correct": 2, "rate": 0.667},
        "tr": {"n": 1, "correct": 1, "rate": 1.0},
        "tr-ascii": {"n": 2, "correct": 1, "rate": 0.5},
    }
    assert s["p_expected"] == {"correct_mean": 0.85, "correct_n": 4, "wrong_mean": 0.35, "wrong_n": 2}
    assert s["cost_usd"] == pytest.approx(0.006)
    assert report["meta"]["totals"]["calls"] == 6
    assert report["meta"]["decisions"] == "fake"

    turn2 = report["turns"][1]
    assert turn2["id"] == "c01" and turn2["turn"] == 2 and turn2["lang"] == "en" and turn2["ascii"] is False
    assert turn2["expected"] == "followup" and turn2["predicted"] == "locate" and turn2["correct"] is False
    assert turn2["p_expected"] == pytest.approx(0.4)
    assert report["turns"][3]["ascii"] is True

    md = route_eval.render_markdown(report)
    assert "# RepoFinder chat routing eval: chat_routes.json (decisions=fake)" in md
    assert "| c01 | 2 | en | followup | locate | 0.40 | X | who calls it? |" in md
    assert "| c03 | 1 | tr | explain | explain | 1.00 |  | Oturum çerezi nasıl imzalanıyor? |" in md
    assert "| c02 | 2 | tr-ascii | outside | explain | 0.30 | X |" in md
    assert "- accuracy: 0.67 (4/6)" in md
    assert "| locate | 1 | 2 | 1 | 0.50 | 1.00 |" in md
    assert "| followup | 1 | 0 | 0 | - | 0.00 |" in md
    assert "| tr-ascii | 2 | 0.50 (1/2) |" in md
    assert "| expected \\ predicted | locate | explain | followup | chat | outside |" in md
    assert "| followup | 1 | 0 | 0 | 0 | 0 |" in md
    assert "correct turns 0.85 (n=4), wrong turns 0.35 (n=2)" in md
    assert "fake, 6 calls, total cost $0.006" in md


def test_the_message_column_is_cut_to_70_characters_and_pipes_are_escaped() -> None:
    golden = golden_set()
    long = "Where is the | pipe handled " + "x" * 100
    golden["conversations"][2]["turns"][0]["user"] = long
    report = route_eval.run_routes(golden, RulesDecisions(), ids=["c03"])
    md = route_eval.render_markdown(report)
    row = next(line for line in md.splitlines() if line.startswith("| c03 |"))
    assert row.endswith("| " + long[:70].replace("|", "\\|") + " |")


def test_a_decisions_error_is_recorded_and_the_run_continues() -> None:
    fake = FakeDecisions(choices=["locate", "search", "chat"])   # "search" is not a route
    report = route_eval.run_routes(golden_set(), fake, ids=["c01"])
    t1, t2, t3 = report["turns"]
    assert t1["correct"] and t3["correct"]
    assert t2["error"].startswith("DecisionsError:") and "search" in t2["error"]
    assert t2["predicted"] is None and t2["correct"] is False and t2["p_expected"] is None
    assert t2["probabilities"] == {}

    s = report["summary"]
    assert s["errors"] == 1
    assert s["accuracy"] == {"n": 3, "correct": 2, "rate": 0.667}
    assert s["per_route"]["followup"] == {"support": 1, "predicted": 0, "correct": 0,
                                          "precision": None, "recall": 0.0}
    assert s["errors_by_expected"]["followup"] == 1
    assert sum(sum(row.values()) for row in s["confusion"].values()) == 2
    assert s["p_expected"]["wrong_n"] == 0
    assert report["meta"]["totals"]["errors"] == 1
    assert report["meta"]["totals"]["calls"] == 2      # the failed call is not counted by the fake

    md = route_eval.render_markdown(report)
    assert "| c01 | 2 | en | followup | error | - | X | who calls it? |" in md
    assert "errors: 1" in md
    assert "| expected \\ predicted | locate | explain | followup | chat | outside | error |" in md
    assert "| followup | 0 | 0 | 0 | 0 | 0 | 1 |" in md


def test_a_config_error_during_the_run_propagates_from_run_routes() -> None:
    class Broken(FakeDecisions):
        def choose(self, state: str, instructions: str, options: dict[str, str]) -> Choice:
            raise DecisionsConfigError("Jev (openrouter) rejected the API key (401)")

    with pytest.raises(DecisionsConfigError):
        route_eval.run_routes(golden_set(), Broken())


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("where", ["detect", "call"])
def test_a_config_error_exits_1_with_the_setup_help(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                    capsys: pytest.CaptureFixture[str], where: str) -> None:
    path = write_golden(tmp_path, golden_set())
    if where == "detect":
        def detect(preferred: Any = None) -> Any:
            raise DecisionsConfigError("Jev decisions were requested, but no key is set.\n\n" + JEV_SETUP_HELP)

        monkeypatch.setattr(route_eval._decisions, "detect_decisions", detect)
    else:
        class Broken(FakeDecisions):
            def choose(self, state: str, instructions: str, options: dict[str, str]) -> Choice:
                raise DecisionsConfigError("Jev (openrouter): payment required (402)")

        use_layer(monkeypatch, Broken())
    out = tmp_path / "r.json"
    assert route_eval.main([str(path), "--decisions", "jev", "--out", str(out)]) == 1
    err = capsys.readouterr().err
    assert "error: the decision layer cannot be used" in err
    assert err.count(JEV_SETUP_HELP.splitlines()[0]) == 1
    assert not out.exists()


def _dup_id(g: dict[str, Any]) -> None:
    g["conversations"][1]["id"] = "c01"


def _bad_route(g: dict[str, Any]) -> None:
    g["conversations"][0]["turns"][1]["route"] = "search"


def _empty_user(g: dict[str, Any]) -> None:
    g["conversations"][0]["turns"][0]["user"] = "   "


def _bad_lang(g: dict[str, Any]) -> None:
    g["conversations"][2]["lang"] = "de"


def _no_turns(g: dict[str, Any]) -> None:
    g["conversations"][2]["turns"] = []


def _assistant_not_text(g: dict[str, Any]) -> None:
    g["conversations"][0]["turns"][0]["assistant"] = ["not", "text"]


def _ascii_not_bool(g: dict[str, Any]) -> None:
    g["conversations"][1]["ascii"] = "yes"


def _no_conversations(g: dict[str, Any]) -> None:
    g["conversations"] = []


def _bad_version(g: dict[str, Any]) -> None:
    g["version"] = 2


@pytest.mark.parametrize("break_it, message", [
    (_dup_id, "c01: duplicate id"),
    (_bad_route, "c01 turn 2: 'route' must be one of locate, explain, followup, chat, outside, got 'search'"),
    (_empty_user, "c01 turn 1: 'user' must be a non-empty string"),
    (_bad_lang, "c03: 'lang' must be one of en, tr, got 'de'"),
    (_no_turns, "c03: 'turns' must be a non-empty list"),
    (_assistant_not_text, "c01 turn 1: 'assistant' must be a string when given"),
    (_ascii_not_bool, "c02: 'ascii' must be true or false when given"),
    (_no_conversations, "'conversations' must be a non-empty list"),
    (_bad_version, "unsupported version 2 (expected 1)"),
])
def test_validation_errors_exit_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                  capsys: pytest.CaptureFixture[str], break_it: Any, message: str) -> None:
    golden = golden_set()
    break_it(golden)
    errors, _warnings = route_eval.validate_golden(golden)
    assert message in errors

    asked = use_layer(monkeypatch, FakeDecisions())
    path = write_golden(tmp_path, golden)
    assert route_eval.main([str(path), "--decisions", "rules"]) == 2
    err = capsys.readouterr().err
    assert f"error: {message}" in err and "invalid golden file" in err
    assert asked == []                     # no layer is set up for an invalid file
    assert route_eval.main([str(path), "--list"]) == 2


@pytest.mark.parametrize("content, message", [
    ("{not json", "is not valid JSON"),
    ("[1, 2]", "the top level must be an object"),
    (None, "cannot read"),
])
def test_an_unreadable_golden_file_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                           content: Any, message: str) -> None:
    path = write_golden(tmp_path, content) if content is not None else tmp_path / "missing.json"
    assert route_eval.main([str(path), "--decisions", "rules"]) == 2
    assert message in capsys.readouterr().err


def test_the_valid_synthetic_set_has_no_errors_or_warnings() -> None:
    assert route_eval.validate_golden(golden_set()) == ([], [])


def test_ids_filter_the_conversations(tmp_path: Path, no_keys: pytest.MonkeyPatch,
                                      capsys: pytest.CaptureFixture[str]) -> None:
    path = write_golden(tmp_path, golden_set())
    out = tmp_path / "r.json"
    assert route_eval.main([str(path), "--decisions", "rules", "--ids", "c03, c01", "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert [(t["id"], t["turn"]) for t in report["turns"]] == [("c01", 1), ("c01", 2), ("c01", 3), ("c03", 1)]
    assert report["meta"]["ids"] == ["c01", "c03"]
    assert report["meta"]["totals"]["conversations"] == 2 and report["meta"]["totals"]["turns"] == 4
    md = capsys.readouterr().out
    assert "| c02 |" not in md and "| c03 | 1 |" in md

    assert route_eval.main([str(path), "--ids", "c02", "--list"]) == 0
    listed = capsys.readouterr().out
    assert listed.startswith("c02 [flask, tr-ascii] 2 turns") and "c01" not in listed
    assert "1 conversations, 2 turns" in listed

    assert route_eval.main([str(path), "--decisions", "rules", "--ids", "c01,c99"]) == 2
    assert "unknown conversation id(s): c99" in capsys.readouterr().err


def test_list_prints_every_turn_without_routing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    asked = use_layer(monkeypatch, FakeDecisions())
    path = write_golden(tmp_path, golden_set())
    assert route_eval.main([str(path), "--list"]) == 0
    out = capsys.readouterr().out
    assert asked == []
    assert "c01 [flask, en] 3 turns" in out
    assert "   2 followup  who calls it?" in out
    assert out.rstrip().endswith("3 conversations, 6 turns")


def test_out_writes_json_with_the_expected_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    asked = use_layer(monkeypatch, FakeDecisions(choices=["locate", "followup", "chat", "chat", "chat", "explain"]))
    path = write_golden(tmp_path, golden_set())
    out = tmp_path / "nested" / "report.json"
    assert route_eval.main([str(path), "--decisions", "jev", "--out", str(out)]) == 0
    assert asked == ["jev"]
    stdout = capsys.readouterr().out
    assert f"JSON report: {out}" in stdout

    report = json.loads(out.read_text(encoding="utf-8"))
    assert set(report) == {"meta", "summary", "turns"}
    meta = report["meta"]
    assert {"golden", "decisions", "started", "totals", "constants", "ids", "validation_warnings"} <= set(meta)
    assert meta["golden"] == str(path) and meta["decisions"] == "fake"
    assert set(meta["totals"]) == {"conversations", "turns", "correct", "accuracy", "errors", "calls",
                                   "cost_usd", "elapsed_s"}
    assert meta["totals"]["turns"] == 6 and meta["totals"]["correct"] == 5 and meta["totals"]["calls"] == 6
    assert meta["constants"]["ROUTES"] == list(ROUTES)
    assert meta["constants"]["HISTORY_EXCHANGES"] == decisions_mod.HISTORY_EXCHANGES
    assert set(report["turns"][0]) == {"id", "turn", "lang", "ascii", "message", "expected", "predicted",
                                       "correct", "probabilities", "p_expected", "cost_usd", "elapsed_s", "error"}
    assert report["turns"][5]["message"] == "Oturum çerezi nasıl imzalanıyor?"
    assert {"accuracy", "per_route", "per_lang", "confusion", "p_expected", "errors"} <= set(report["summary"])


def test_without_out_only_markdown_is_printed(tmp_path: Path, no_keys: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    path = write_golden(tmp_path, golden_set())
    before = sorted(p.name for p in tmp_path.rglob("*"))
    assert route_eval.main([str(path), "--decisions", "rules"]) == 0
    out = capsys.readouterr().out
    assert "## Summary" in out and "JSON report" not in out
    assert sorted(p.name for p in tmp_path.rglob("*")) == before


# --------------------------------------------------------------------------- #
# End to end with the rules (offline)
# --------------------------------------------------------------------------- #

def test_end_to_end_with_the_rules(tmp_path: Path, no_keys: pytest.MonkeyPatch,
                                   capsys: pytest.CaptureFixture[str]) -> None:
    path = write_golden(tmp_path, golden_set())
    out = tmp_path / "rules.json"
    assert route_eval.main([str(path), "--decisions", "rules", "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    predicted = [(t["id"], t["turn"], t["predicted"]) for t in report["turns"]]
    assert predicted == [
        ("c01", 1, "locate"), ("c01", 2, "followup"), ("c01", 3, "chat"),
        ("c02", 1, "chat"), ("c02", 2, "explain"),          # the rules never answer outside
        ("c03", 1, "explain"),
    ]
    assert report["meta"]["decisions"] == "rules"
    assert report["meta"]["totals"] == {**report["meta"]["totals"], "turns": 6, "correct": 5, "errors": 0,
                                        "calls": 6, "cost_usd": 0.0}
    assert all(t["p_expected"] in (0.0, 1.0) for t in report["turns"])
    md = capsys.readouterr().out
    assert "- accuracy: 0.83 (5/6)" in md
    assert "| outside | 1 | 0 | 0 | - | 0.00 |" in md
    assert "| outside | 0 | 1 | 0 | 0 | 0 |" in md


def test_the_script_runs_as_a_program_with_the_rules(tmp_path: Path) -> None:
    path = write_golden(tmp_path, golden_set())
    out = tmp_path / "proc.json"
    env = {k: v for k, v in os.environ.items() if k not in KEY_ENVS}
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, str(EVAL_DIR / "route_eval.py"), str(path), "--decisions", "rules", "--out", str(out)],
        capture_output=True, text=True, encoding="utf-8", cwd=tmp_path, env=env, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "- accuracy: 0.83 (5/6)" in proc.stdout
    assert "Oturum çerezi nasıl imzalanıyor?" in proc.stdout
    assert "[c01#2]" in proc.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["meta"]["totals"]["turns"] == 6


def test_runs_do_not_change_the_golden_data() -> None:
    golden = golden_set()
    snapshot = copy.deepcopy(golden)
    route_eval.run_routes(golden, RulesDecisions())
    assert golden == snapshot

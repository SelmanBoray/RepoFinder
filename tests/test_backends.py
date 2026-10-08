"""Tests for repofinder.backends: backend detection, .env loading, and the
Claude Code CLI backend (with a fake subprocess runner).

The test at the bottom runs the real `claude` CLI when it is on PATH (set
REPOFINDER_SKIP_LIVE=1 to skip it); it uses the local Claude Code login.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from repofinder import backends as backends_mod
from repofinder.backends import (
    AnthropicBackend,
    Backend,
    BackendError,
    BackendUnavailable,
    ClaudeCodeBackend,
    Completion,
    LLMConfigError,
    LLMError,
    detect_backend,
    load_dotenv,
    parse_cli_output,
)

RESULT = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": "Defines the url_for helper.",
    "stop_reason": "end_turn",
    "duration_ms": 3100,
    "usage": {"input_tokens": 12, "cache_creation_input_tokens": 700,
              "cache_read_input_tokens": 300, "output_tokens": 40},
    "modelUsage": {"claude-sonnet-5-5": {"inputTokens": 12, "outputTokens": 40}},
}


class FakeRun:
    """Stands in for subprocess.run; records calls and the system prompt file."""

    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0,
                 raises: BaseException | None = None) -> None:
        self.stdout, self.stderr, self.returncode, self.raises = stdout, stderr, returncode, raises
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.system_prompt: str | None = None

    def __call__(self, argv: list[str], **kwargs: Any) -> SimpleNamespace:
        self.calls.append((list(argv), kwargs))
        path = argv[argv.index("--system-prompt-file") + 1]
        self.system_prompt = Path(path).read_text(encoding="utf-8")
        assert Path(kwargs["cwd"]).is_dir() and Path(path).parent == Path(kwargs["cwd"])
        if self.raises is not None:
            raise self.raises
        return SimpleNamespace(stdout=self.stdout, stderr=self.stderr, returncode=self.returncode)


def complete(backend: ClaudeCodeBackend, **overrides: Any) -> Completion:
    params = dict(model="claude-sonnet-5-5", system="SYSTEM \"quoted\" & <tag> %PATH%",
                  user="-starts with a dash\nsecond line", max_tokens=50, effort="low")
    params.update(overrides)
    return backend.complete(**params)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """No Anthropic credentials, no backend preference, an empty working
    directory (so no stray .env), and no `claude` on PATH unless a test adds it.
    os.environ is restored afterwards, including variables load_dotenv set."""
    saved = dict(os.environ)
    for name in list(os.environ):
        if name.startswith("ANTHROPIC_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("REPOFINDER_BACKEND", raising=False)
    for var in ("APPDATA", "HOME", "USERPROFILE", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(var, str(tmp_path / "home"))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr(backends_mod.shutil, "which", lambda name, *a, **k: None)
    yield work
    os.environ.clear()
    os.environ.update(saved)


def claude_on_path(monkeypatch: pytest.MonkeyPatch, path: str = "/opt/bin/claude") -> None:
    monkeypatch.setattr(backends_mod.shutil, "which",
                        lambda name, *a, **k: path if name == "claude" else None)


# --------------------------------------------------------------------------- #
# ClaudeCodeBackend
# --------------------------------------------------------------------------- #

def test_claude_code_object_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "parent-session")
    run = FakeRun(stdout=json.dumps(RESULT))
    backend = ClaudeCodeBackend(executable="claude-bin", run=run)

    result = complete(backend)

    assert result == Completion(text="Defines the url_for helper.", model="claude-sonnet-5-5",
                                input_tokens=712, output_tokens=40, cache_read_tokens=300,
                                stop_reason="end_turn")
    (argv, kwargs), = run.calls
    assert argv[:2] == ["claude-bin", "-p"]
    flags = argv[2:]
    assert flags[flags.index("--output-format") + 1] == "json"
    assert flags[flags.index("--model") + 1] == "claude-sonnet-5-5"
    assert flags[flags.index("--tools") + 1] == ""
    assert flags[flags.index("--max-turns") + 1] == "1"
    assert flags[flags.index("--effort") + 1] == "low"
    assert "--no-session-persistence" in flags and "--safe-mode" in flags
    # neither prompt travels on the command line
    assert not any("SYSTEM" in a or "dash" in a for a in argv)
    assert run.system_prompt == "SYSTEM \"quoted\" & <tag> %PATH%"
    assert kwargs["input"] == "-starts with a dash\nsecond line"
    assert kwargs["shell"] is False and kwargs["timeout"] == 600
    assert kwargs["capture_output"] is True and kwargs["encoding"] == "utf-8"
    cwd = Path(kwargs["cwd"]).resolve()
    assert cwd != Path.cwd().resolve() and Path(r"C:\RepoFinder").resolve() not in (cwd, *cwd.parents)
    assert not cwd.exists()  # the temporary directory is removed afterwards
    assert "CLAUDECODE" not in kwargs["env"] and "CLAUDE_CODE_SESSION_ID" not in kwargs["env"]
    assert kwargs["env"].get("PATH") == os.environ.get("PATH")


def test_claude_code_array_output_uses_last_result() -> None:
    events = [
        {"type": "system", "subtype": "init", "model": "claude-sonnet-5-5"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "draft"}]}},
        {"type": "rate_limit_event"},
        dict(RESULT, result="final text", modelUsage={
            "claude-haiku-5-5": {"outputTokens": 3}, "claude-opus-5-5": {"outputTokens": 90}}),
    ]
    run = FakeRun(stdout=json.dumps(events))
    result = complete(ClaudeCodeBackend(executable="claude", run=run), model="opus", effort=None)
    assert result.text == "final text"
    assert result.model == "claude-opus-5-5"  # the model that did the work
    argv, _ = run.calls[0]
    assert "--effort" not in argv
    assert argv[argv.index("--model") + 1] == "opus"


def test_claude_code_is_error_raises_backend_error() -> None:
    payload = dict(RESULT, is_error=True, subtype="error_during_execution", result="x" * 2000)
    with pytest.raises(BackendError, match="reported an error") as info:
        complete(ClaudeCodeBackend(executable="claude", run=FakeRun(stdout=json.dumps(payload))))
    assert len(str(info.value)) < 600


def test_claude_code_nonzero_exit_raises_backend_error() -> None:
    run = FakeRun(stdout="", stderr="Invalid API key · Please run /login\n" + "y" * 2000, returncode=1)
    with pytest.raises(BackendError, match=r"exited with code 1: Invalid API key") as info:
        complete(ClaudeCodeBackend(executable="claude", run=run))
    assert not isinstance(info.value, LLMConfigError)
    assert len(str(info.value)) < 600


def test_claude_code_unparseable_output_raises_backend_error() -> None:
    for stdout in ("not json at all", "", json.dumps([{"type": "system"}]), json.dumps({"result": 5})):
        with pytest.raises(BackendError):
            complete(ClaudeCodeBackend(executable="claude", run=FakeRun(stdout=stdout)))


def test_claude_code_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends_mod.shutil, "which", lambda name, *a, **k: None)
    with pytest.raises(BackendUnavailable, match="not found on PATH") as info:
        complete(ClaudeCodeBackend(run=FakeRun(stdout=json.dumps(RESULT))))
    message = str(info.value)
    assert "ANTHROPIC_API_KEY" in message and "claude auth login" in message

    gone = FakeRun(raises=FileNotFoundError("claude"))
    with pytest.raises(BackendUnavailable, match="Could not start the claude CLI"):
        complete(ClaudeCodeBackend(executable="C:/missing/claude.exe", run=gone))


def test_claude_code_timeout_and_bad_model() -> None:
    slow = FakeRun(raises=subprocess.TimeoutExpired(cmd="claude", timeout=600))
    with pytest.raises(BackendError, match="did not finish within 600s"):
        complete(ClaudeCodeBackend(executable="claude", run=slow))
    with pytest.raises(BackendError, match="invalid model id"):
        complete(ClaudeCodeBackend(executable="claude", run=FakeRun(stdout=json.dumps(RESULT))),
                 model="sonnet & calc")


def test_parse_cli_output_prefers_result_text_on_failure() -> None:
    failed = dict(RESULT, is_error=True, result="Prompt is too long")
    with pytest.raises(BackendError, match="exited with code 2: Prompt is too long"):
        parse_cli_output(json.dumps(failed), "", 2, "claude-sonnet-5-5")
    with pytest.raises(BackendError, match="claude update"):
        parse_cli_output("", "error: unknown option '--safe-mode'", 1, "m")
    minimal = parse_cli_output(json.dumps({"type": "result", "result": "hi"}), "", 0, "m")
    assert minimal == Completion("hi", "m", 0, 0, 0, None)


@pytest.mark.skipif(os.name != "nt", reason="npm shim resolution is Windows-only")
def test_windows_npm_shim_resolves_to_native_exe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shim = tmp_path / "npm" / "claude.CMD"
    native = tmp_path / "npm" / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
    native.parent.mkdir(parents=True)
    shim.write_text("@echo off\n")
    native.write_bytes(b"MZ")
    claude_on_path(monkeypatch, str(shim))
    assert backends_mod.find_claude_executable() == str(native)
    native.unlink()
    assert backends_mod.find_claude_executable() == str(shim)  # fall back to the shim


def test_backends_satisfy_the_protocol() -> None:
    assert isinstance(ClaudeCodeBackend(executable="claude"), Backend)
    assert isinstance(AnthropicBackend(api_key="sk-x"), Backend)
    assert issubclass(BackendError, LLMError) and issubclass(BackendUnavailable, LLMConfigError)


def test_anthropic_backend_splits_overview_into_cached_block() -> None:
    blocks = backends_mod._user_blocks("<repository_overview>\nOverview.\n</repository_overview>\n\n<q>rest</q>")
    assert blocks == [
        {"type": "text", "text": "<repository_overview>\nOverview.\n</repository_overview>",
         "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "<q>rest</q>"},
    ]
    assert backends_mod._user_blocks("Summarize this file.") == [
        {"type": "text", "text": "Summarize this file."}]


# --------------------------------------------------------------------------- #
# detect_backend
# --------------------------------------------------------------------------- #

def test_detect_prefers_api_key_over_cli(clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    claude_on_path(monkeypatch)
    assert isinstance(detect_backend(), AnthropicBackend)


def test_detect_falls_back_to_claude_code(clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    claude_on_path(monkeypatch)
    backend = detect_backend()
    assert isinstance(backend, ClaudeCodeBackend)
    assert backend.executable == "/opt/bin/claude"


def test_detect_without_any_backend_explains_both_options(clean_env: Path) -> None:
    with pytest.raises(BackendUnavailable) as info:
        detect_backend()
    message = str(info.value)
    assert "ANTHROPIC_API_KEY" in message and ".env" in message
    assert "Claude Code" in message and "claude auth login" in message
    assert "REPOFINDER_BACKEND" in message


def test_detect_explicit_preference(clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    claude_on_path(monkeypatch)
    assert isinstance(detect_backend("claude-code"), ClaudeCodeBackend)
    assert isinstance(detect_backend("claude_code"), ClaudeCodeBackend)
    assert isinstance(detect_backend("Anthropic"), AnthropicBackend)

    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert isinstance(detect_backend("anthropic"), AnthropicBackend)  # key error comes on first call

    monkeypatch.setattr(backends_mod.shutil, "which", lambda name, *a, **k: None)
    with pytest.raises(BackendUnavailable, match="REPOFINDER_BACKEND=claude-code"):
        detect_backend("claude-code")
    with pytest.raises(BackendUnavailable, match="Unknown backend 'openai'"):
        detect_backend("openai")


def test_detect_counts_other_sdk_credentials(clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token")
    claude_on_path(monkeypatch)
    assert isinstance(detect_backend(), AnthropicBackend)


def test_detect_logs_choice(clean_env: Path, monkeypatch: pytest.MonkeyPatch,
                            caplog: pytest.LogCaptureFixture) -> None:
    claude_on_path(monkeypatch)
    with caplog.at_level(logging.INFO, logger="repofinder.backends"):
        detect_backend()
    assert "RepoFinder backend: claude-code" in caplog.text


# --------------------------------------------------------------------------- #
# .env loading
# --------------------------------------------------------------------------- #

def test_dotenv_is_loaded_by_detection(clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    claude_on_path(monkeypatch)
    (clean_env / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-from-dotenv\n", encoding="utf-8")
    backend = detect_backend()
    assert isinstance(backend, AnthropicBackend)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-dotenv"


def test_dotenv_parsing_rules(clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALREADY_SET", "from-environment")
    for name in ("RF_PLAIN", "RF_DOUBLE", "RF_SINGLE", "RF_EXPORTED", "RF_COMMENTED", "RF_EMPTY",
                 "RF_HASH_IN_QUOTES", "RF_EQUALS"):
        monkeypatch.delenv(name, raising=False)
    (clean_env / ".env").write_text(
        "\ufeff# a comment\n"
        "\n"
        "RF_PLAIN=plain value\n"
        'RF_DOUBLE="double # not a comment"\n'
        "RF_SINGLE='single'\n"
        "export RF_EXPORTED=yes\n"
        "RF_COMMENTED=value  # trailing comment\n"
        "RF_EMPTY=\n"
        "RF_EQUALS=a=b=c\n"
        "ALREADY_SET=from-dotenv\n"
        "not a valid line\n"
        "1BAD=x\n",
        encoding="utf-8",
    )
    loaded = load_dotenv()
    assert os.environ["RF_PLAIN"] == "plain value"
    assert os.environ["RF_DOUBLE"] == "double # not a comment"
    assert os.environ["RF_SINGLE"] == "single"
    assert os.environ["RF_EXPORTED"] == "yes"
    assert os.environ["RF_COMMENTED"] == "value"
    assert os.environ["RF_EMPTY"] == ""
    assert os.environ["RF_EQUALS"] == "a=b=c"
    assert os.environ["ALREADY_SET"] == "from-environment"  # never overridden
    assert "ALREADY_SET" not in loaded and "1BAD" not in os.environ
    assert load_dotenv(clean_env / "missing.env") == {}


# --------------------------------------------------------------------------- #
# Real Claude Code CLI
# --------------------------------------------------------------------------- #

live_cli = pytest.mark.skipif(
    shutil.which("claude") is None or bool(os.environ.get("REPOFINDER_SKIP_LIVE")),
    reason="the claude CLI is not on PATH (or REPOFINDER_SKIP_LIVE is set)",
)


@live_cli
def test_real_claude_code_backend_replies_ok() -> None:
    started = time.perf_counter()
    result = ClaudeCodeBackend().complete(
        model="claude-sonnet-5-5",
        system="Reply with exactly the word OK and nothing else.",
        user="ping",
        max_tokens=50,
    )
    elapsed = time.perf_counter() - started
    print(f"\n[claude-code] {elapsed:.1f}s model={result.model} in={result.input_tokens} "
          f"out={result.output_tokens} cache_read={result.cache_read_tokens} "
          f"stop={result.stop_reason} text={result.text!r}")
    assert "OK" in result.text
    assert result.input_tokens + result.cache_read_tokens > 0 and result.output_tokens > 0

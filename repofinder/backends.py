"""Model backends: how RepoFinder reaches Claude.

RepoFinder can talk to Claude in two ways. Pick one:

1. Anthropic API key (pay as you go; the usual choice when you install
   RepoFinder from GitHub). Create a key at https://console.anthropic.com and
   make it visible to RepoFinder in one of these ways:
     PowerShell:     $env:ANTHROPIC_API_KEY = "sk-ant-..."
     cmd.exe:        set ANTHROPIC_API_KEY=sk-ant-...
     macOS / Linux:  export ANTHROPIC_API_KEY=sk-ant-...
     or a file named .env in the directory you run RepoFinder from, containing
     the line  ANTHROPIC_API_KEY=sk-ant-...

2. Your Claude subscription, through Claude Code. Install Claude Code
   (https://claude.com/claude-code), sign in once with `claude auth login`
   (or run `claude` and type /login), and make sure the `claude` command is on
   your PATH. RepoFinder then runs `claude -p` for each model call, which uses
   your subscription's limits instead of API billing.

When both are available the API key wins. To choose explicitly, set
REPOFINDER_BACKEND=anthropic or REPOFINDER_BACKEND=claude-code.

Implementation notes
--------------------
* ``AnthropicBackend`` calls the Messages API through the ``anthropic`` SDK,
  with its own retry/backoff, prompt caching and ``output_config.effort``.
* ``ClaudeCodeBackend`` runs the ``claude`` CLI once per call: prompt on stdin
  (prompts reach ~40k characters, beyond the Windows command-line limit),
  system prompt from a temporary file, no tools, no session persistence,
  ``--safe-mode`` so the user's CLAUDE.md, memory, skills, hooks and MCP
  servers are not mixed into RepoFinder's prompts, in an empty temporary
  working directory. ``effort`` is passed as ``--effort``; ``max_tokens`` is
  ignored because the CLI has no such option.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, runtime_checkable

import anthropic

from . import prompts

logger = logging.getLogger(__name__)

ENV_BACKEND = "REPOFINDER_BACKEND"
BACKEND_ANTHROPIC = "anthropic"
BACKEND_CLAUDE_CODE = "claude-code"

SETUP_HELP = """\
RepoFinder needs access to Claude. Choose one of these options:

  1. Anthropic API key (pay as you go). Create one at https://console.anthropic.com, then either
       PowerShell:     $env:ANTHROPIC_API_KEY = "sk-ant-..."
       cmd.exe:        set ANTHROPIC_API_KEY=sk-ant-...
       macOS / Linux:  export ANTHROPIC_API_KEY=sk-ant-...
     or put the line  ANTHROPIC_API_KEY=sk-ant-...  in a .env file in the directory you run RepoFinder from.

  2. Your Claude subscription via Claude Code. Install Claude Code (https://claude.com/claude-code),
     sign in once with `claude auth login`, and make sure the `claude` command is on your PATH.

To choose explicitly, set REPOFINDER_BACKEND=anthropic or REPOFINDER_BACKEND=claude-code."""

# Anthropic backend: retry policy and request timeout.
RETRY_BASE_DELAY_S = 0.5             # 0.5s, 1s, 2s, ...
RETRY_AFTER_CAP_S = 60.0             # cap on a server-provided retry-after
REQUEST_TIMEOUT_S = 300.0

# Claude Code backend.
CLI_TIMEOUT_S = 600.0
CLI_ERROR_CHARS = 500
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
# Variables that tie a process to a running Claude Code session. They are set
# when RepoFinder itself runs inside Claude Code; the child CLI must not see them.
CLI_SESSION_ENV_VARS = (
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_SESSION_ATTENDED",
    "CLAUDE_CODE_MESSAGING_SOCKET",
    "CLAUDE_CODE_MESSAGING_TOKEN",
    "CLAUDE_PID",
    "CLAUDE_EFFORT",
)
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:\-\[\]]*$")


# --------------------------------------------------------------------------- #
# Errors and data types
# --------------------------------------------------------------------------- #

class LLMError(RuntimeError):
    """A model call failed (after retries, where retrying made sense)."""


class LLMConfigError(LLMError):
    """A failure every further call would repeat: missing or rejected
    credentials, an unknown model, a missing CLI, or an incompatible SDK."""


class BackendError(LLMError):
    """A backend failed to produce a completion for this call."""


class BackendUnavailable(LLMConfigError):
    """No usable backend: no credentials, or the requested backend is missing."""


@dataclass
class Completion:
    text: str
    model: str
    input_tokens: int          # input not served from cache (uncached + cache writes)
    output_tokens: int         # includes thinking tokens
    cache_read_tokens: int
    stop_reason: Optional[str]
    stop_detail: Optional[str] = None   # e.g. the refusal category, when known


@runtime_checkable
class Backend(Protocol):
    name: str

    def complete(self, *, model: str, system: str, user: str, max_tokens: int,
                 effort: Optional[str] = None) -> Completion: ...


# --------------------------------------------------------------------------- #
# Anthropic API
# --------------------------------------------------------------------------- #

class AnthropicBackend:
    """Messages API through the ``anthropic`` SDK.

    The SDK client is created on the first call (so constructing this never
    fails for lack of a key), with the SDK's own retries switched off: retries
    happen here, with exponential backoff, so the two layers do not multiply.
    The system prompt carries ``cache_control``; when the user text starts with
    a repository overview block, that block becomes a second cache breakpoint.
    """

    name = BACKEND_ANTHROPIC

    def __init__(self, api_key: Optional[str] = None, client: Optional[object] = None,
                 max_retries: int = 3, *, sleep: Optional[Callable[[float], None]] = None) -> None:
        self._api_key = api_key
        self._client = client
        self.max_retries = max(0, int(max_retries))
        self._sleep = sleep if sleep is not None else time.sleep

    def complete(self, *, model: str, system: str, user: str, max_tokens: int,
                 effort: Optional[str] = None) -> Completion:
        client = self._get_client()
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": _user_blocks(user)}],
        }
        if effort:
            params["output_config"] = {"effort": effort}

        attempt = 0
        while True:
            try:
                response = client.messages.create(**params)
                break
            except Exception as exc:
                if _is_retryable(exc) and attempt < self.max_retries:
                    self._sleep(_retry_delay(attempt, exc))
                    attempt += 1
                    continue
                raise _wrap_error(exc, model=model, attempts=attempt + 1) from exc
        return _completion_from_message(response, model)

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                client = anthropic.Anthropic(
                    api_key=self._api_key, max_retries=0, timeout=REQUEST_TIMEOUT_S
                )
            except Exception as exc:  # e.g. a broken explicitly-selected profile
                raise LLMConfigError(f"could not create the Anthropic client: {exc}") from exc
            if not _client_has_credentials(client):
                raise BackendUnavailable("No Anthropic API key found.\n\n" + SETUP_HELP)
            self._client = client
        return self._client


def anthropic_credentials_available() -> bool:
    """True when the SDK can find credentials: ANTHROPIC_API_KEY,
    ANTHROPIC_AUTH_TOKEN, an `ant auth login` profile, or workload identity."""
    try:
        client = anthropic.Anthropic(max_retries=0)
    except Exception as exc:
        logger.warning("Anthropic credentials could not be loaded: %s", exc)
        return False
    return _client_has_credentials(client)


def _client_has_credentials(client: Any) -> bool:
    return any(getattr(client, attr, None) for attr in ("api_key", "auth_token", "credentials"))


def _user_blocks(user: str) -> list[dict[str, Any]]:
    """One text block, or two when the text opens with the repository
    overview: the overview is stable per repository, so it is cached."""
    if user.startswith(prompts.OVERVIEW_START):
        head, sep, tail = user.partition(prompts.OVERVIEW_END)
        if sep and tail.strip():
            return [
                {"type": "text", "text": head + sep, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": tail.lstrip("\n")},
            ]
    return [{"type": "text", "text": user}]


def _completion_from_message(response: Any, model: str) -> Completion:
    usage = getattr(response, "usage", None)
    text = "".join(
        getattr(block, "text", "") or ""
        for block in (getattr(response, "content", None) or [])
        if getattr(block, "type", None) == "text"
    )
    details = getattr(response, "stop_details", None)
    return Completion(
        text=text,
        model=getattr(response, "model", None) or model,
        input_tokens=_as_int(getattr(usage, "input_tokens", 0))
        + _as_int(getattr(usage, "cache_creation_input_tokens", 0)),
        output_tokens=_as_int(getattr(usage, "output_tokens", 0)),
        cache_read_tokens=_as_int(getattr(usage, "cache_read_input_tokens", 0)),
        stop_reason=getattr(response, "stop_reason", None),
        stop_detail=getattr(details, "category", None) if details is not None else None,
    )


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (anthropic.RateLimitError, anthropic.APIConnectionError,
                        anthropic.InternalServerError)):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(exc, anthropic.APIStatusError) and isinstance(status, int) and status >= 500


def _retry_delay(attempt: int, exc: BaseException) -> float:
    delay = RETRY_BASE_DELAY_S * (2 ** attempt)
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            retry_after = float(headers.get("retry-after"))
        except (TypeError, ValueError):
            retry_after = None
        if retry_after is not None and retry_after > 0:
            delay = max(delay, min(retry_after, RETRY_AFTER_CAP_S))
    return delay


def _wrap_error(exc: BaseException, *, model: str, attempts: int) -> LLMError:
    if isinstance(exc, LLMError):
        return exc
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return LLMConfigError(f"the Anthropic API rejected the credentials "
                              f"({exc.status_code}): {_api_message(exc)}")
    if isinstance(exc, anthropic.NotFoundError):
        return LLMConfigError(f"not found (404) - check the model id {model!r}: {_api_message(exc)}")
    if isinstance(exc, TypeError):
        if "authentication" in str(exc).lower():
            return BackendUnavailable("No Anthropic API key found.\n\n" + SETUP_HELP)
        return LLMConfigError(f"the anthropic SDK rejected the request ({exc}); "
                              f"is the installed anthropic package too old?")
    suffix = f" after {attempts} attempts" if attempts > 1 else ""
    if isinstance(exc, anthropic.APIStatusError):
        return LLMError(f"Anthropic API error {exc.status_code}{suffix}: {_api_message(exc)}")
    if isinstance(exc, anthropic.APIConnectionError):
        return LLMError(f"could not reach the Anthropic API{suffix}: {exc}")
    return LLMError(f"{type(exc).__name__}{suffix}: {exc}")


def _api_message(exc: BaseException) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
    return str(getattr(exc, "message", None) or exc)


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------- #
# Claude Code CLI
# --------------------------------------------------------------------------- #

class ClaudeCodeBackend:
    """Runs ``claude -p`` once per call, using the user's Claude Code login.

    ``effort`` maps to ``--effort``; ``max_tokens`` is ignored (the CLI has no
    such option). ``run`` can be replaced for tests; it receives the same
    arguments as :func:`subprocess.run`.
    """

    name = BACKEND_CLAUDE_CODE

    def __init__(self, executable: Optional[str] = None, *, timeout: float = CLI_TIMEOUT_S,
                 run: Optional[Callable[..., Any]] = None) -> None:
        self._executable = executable
        self.timeout = timeout
        self._run = run

    @property
    def executable(self) -> str:
        if self._executable is None:
            found = find_claude_executable()
            if found is None:
                raise BackendUnavailable("The `claude` command was not found on PATH.\n\n" + SETUP_HELP)
            self._executable = found
        return self._executable

    def build_command(self, *, model: str, system_prompt_file: str,
                      effort: Optional[str] = None) -> list[str]:
        if not _MODEL_ID.match(model):
            raise BackendError(f"invalid model id for the claude CLI: {model!r}")
        argv = [
            self.executable, "-p",
            "--output-format", "json",
            "--model", model,
            "--system-prompt-file", system_prompt_file,
            "--tools", "",
            "--no-session-persistence",
            "--max-turns", "1",
            "--safe-mode",
        ]
        if effort in EFFORT_LEVELS:
            argv += ["--effort", effort]
        return argv

    def complete(self, *, model: str, system: str, user: str, max_tokens: int,
                 effort: Optional[str] = None) -> Completion:
        run = self._run or subprocess.run
        # A fresh, empty working directory: the CLI must not pick up the indexed
        # repository's (or RepoFinder's own) CLAUDE.md or settings.
        with tempfile.TemporaryDirectory(prefix="repofinder-claude-", ignore_cleanup_errors=True) as work:
            system_file = Path(work) / "system-prompt.txt"
            system_file.write_text(system, encoding="utf-8")
            argv = self.build_command(model=model, system_prompt_file=str(system_file), effort=effort)
            try:
                proc = run(
                    argv,
                    input=user,                     # prompt on stdin; stdin is closed afterwards
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                    cwd=work,
                    env=cli_environment(),
                    shell=False,
                )
            except FileNotFoundError as exc:
                raise BackendUnavailable(
                    f"Could not start the claude CLI ({argv[0]}).\n\n" + SETUP_HELP) from exc
            except subprocess.TimeoutExpired as exc:
                raise BackendError(f"the claude CLI did not finish within {self.timeout:.0f}s") from exc
            except OSError as exc:
                raise BackendError(f"could not run the claude CLI: {exc}") from exc
        return parse_cli_output(proc.stdout, proc.stderr, proc.returncode, model)


def find_claude_executable() -> Optional[str]:
    """Locate the ``claude`` CLI. On Windows an npm ``claude.cmd`` shim is
    replaced by the native ``claude.exe`` it wraps when that exists, so
    arguments do not pass through cmd.exe."""
    path = shutil.which("claude")
    if path is None:
        return None
    if os.name == "nt" and path.lower().endswith((".cmd", ".bat")):
        native = Path(path).parent / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
        if native.is_file():
            return str(native)
    return path


def cli_environment() -> dict[str, str]:
    """The current environment without the variables of an enclosing Claude Code session."""
    env = dict(os.environ)
    for name in CLI_SESSION_ENV_VARS:
        env.pop(name, None)
    return env


def parse_cli_output(stdout: str, stderr: str, returncode: int, model: str) -> Completion:
    """Turn ``claude -p --output-format json`` output into a :class:`Completion`.

    The CLI prints either one result object or an array of events whose last
    ``"type": "result"`` element is the result.
    """
    result: Optional[dict[str, Any]] = None
    parse_error: Optional[str] = None
    try:
        data = json.loads(stdout) if stdout and stdout.strip() else None
    except json.JSONDecodeError as exc:
        data, parse_error = None, str(exc)
    if isinstance(data, dict):
        result = data
    elif isinstance(data, list):
        result = next((item for item in reversed(data)
                       if isinstance(item, dict) and item.get("type") == "result"), None)

    if returncode != 0:
        detail = (result or {}).get("result") or (stderr or "").strip() or (stdout or "").strip()
        hint = ""
        if "unknown option" in str(detail).lower():
            hint = " (this Claude Code version may be too old for RepoFinder; run `claude update`)"
        raise BackendError(f"the claude CLI exited with code {returncode}: {_clip(str(detail))}{hint}")
    if result is None:
        reason = parse_error or "no result object in the output"
        raise BackendError(f"could not parse the claude CLI output ({reason}): {_clip(stdout or '')}")
    if result.get("is_error"):
        detail = result.get("result") or result.get("subtype") or "unknown error"
        raise BackendError(f"the claude CLI reported an error: {_clip(str(detail))}")
    text = result.get("result")
    if not isinstance(text, str):
        raise BackendError(f"the claude CLI returned no result text: {_clip(stdout)}")

    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    model_usage = result.get("modelUsage") if isinstance(result.get("modelUsage"), dict) else {}
    used_model = model
    if model_usage:
        used_model = max(model_usage, key=lambda k: _as_int((model_usage[k] or {}).get("outputTokens")))
    return Completion(
        text=text,
        model=used_model,
        input_tokens=_as_int(usage.get("input_tokens")) + _as_int(usage.get("cache_creation_input_tokens")),
        output_tokens=_as_int(usage.get("output_tokens")),
        cache_read_tokens=_as_int(usage.get("cache_read_input_tokens")),
        stop_reason=result.get("stop_reason"),
    )


def _clip(text: str, limit: int = CLI_ERROR_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #

_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_dotenv(path: Optional[os.PathLike[str] | str] = None) -> dict[str, str]:
    """Load ``KEY=VALUE`` lines from ``path`` (default: ``./.env``) into
    ``os.environ`` without overriding variables that are already set.

    Supports blank lines, ``#`` comments, an optional ``export`` prefix,
    single- or double-quoted values and trailing `` # comments`` on unquoted
    values. Returns the variables it set.
    """
    env_path = Path(path) if path is not None else Path.cwd() / ".env"
    try:
        content = env_path.read_text(encoding="utf-8-sig")
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        return {}
    except OSError as exc:
        logger.warning("Could not read %s: %s", env_path, exc)
        return {}

    loaded: dict[str, str] = {}
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not _ENV_KEY.match(key):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        if key in os.environ:
            continue
        os.environ[key] = value
        loaded[key] = value
    if loaded:
        logger.info("Loaded %s from %s", ", ".join(sorted(loaded)), env_path)
    return loaded


def detect_backend(preferred: Optional[str] = None, *, api_key: Optional[str] = None,
                   max_retries: int = 3, sleep: Optional[Callable[[float], None]] = None) -> Backend:
    """Choose a backend.

    ``preferred`` is "anthropic", "claude-code" or None (automatic: an Anthropic
    API key or other SDK credentials first, then the ``claude`` CLI). A ``.env``
    file in the current directory is loaded first, without overriding
    variables that are already set.
    """
    load_dotenv()
    choice = (preferred or "").strip().lower().replace("_", "-") or None
    backend: Backend
    if choice == BACKEND_ANTHROPIC:
        backend = AnthropicBackend(api_key=api_key, max_retries=max_retries, sleep=sleep)
    elif choice == BACKEND_CLAUDE_CODE:
        executable = find_claude_executable()
        if executable is None:
            raise BackendUnavailable(
                f"{ENV_BACKEND}=claude-code, but the `claude` command was not found on PATH.\n\n"
                + SETUP_HELP)
        backend = ClaudeCodeBackend(executable=executable)
    elif choice is None:
        if api_key or anthropic_credentials_available():
            backend = AnthropicBackend(api_key=api_key, max_retries=max_retries, sleep=sleep)
        else:
            executable = find_claude_executable()
            if executable is None:
                raise BackendUnavailable(SETUP_HELP)
            backend = ClaudeCodeBackend(executable=executable)
    else:
        raise BackendUnavailable(
            f"Unknown backend {preferred!r}: use 'anthropic' or 'claude-code'.\n\n" + SETUP_HELP)
    logger.info("RepoFinder backend: %s", backend.name)
    return backend

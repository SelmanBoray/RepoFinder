"""Shared fixtures for the RepoFinder test-suite."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable

import pytest

from repofinder.ingest import build_index
from repofinder.models import RepoIndex

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_REPO = FIXTURES / "sample_repo"

# Pinned Flask checkout shared by every Flask-dependent test. Override the location with
# REPOFINDER_TEST_FLASK_ROOT; tests are skipped (never failed) when it is absent or on another commit.
FLASK_PINNED_COMMIT = "d086db8"
FLASK_ROOT = Path(
    os.environ.get("REPOFINDER_TEST_FLASK_ROOT")
    or Path.home() / ".repofinder" / "repos" / "flask" / "checkout" / "flask"
)


def _flask_skip_reason() -> str | None:
    if not (FLASK_ROOT / "src" / "flask").is_dir():
        return "Flask checkout absent"
    try:
        head = subprocess.run(
            ["git", "-C", str(FLASK_ROOT), "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "Flask checkout is not a readable git checkout"
    if head != FLASK_PINNED_COMMIT:
        return f"Flask checkout is at {head}, tests need pinned commit {FLASK_PINNED_COMMIT}"
    return None


_FLASK_SKIP = _flask_skip_reason()
needs_flask = pytest.mark.skipif(_FLASK_SKIP is not None, reason=_FLASK_SKIP or "")

# The fixture repo is data, not tests (it even contains a syntax error).
collect_ignore = ["fixtures"]


@pytest.fixture
def sample_repo() -> Path:
    return SAMPLE_REPO


@pytest.fixture
def sample_index() -> RepoIndex:
    return build_index(SAMPLE_REPO, "sample_repo", str(SAMPLE_REPO))


@pytest.fixture
def make_repo(tmp_path: Path) -> Callable[[dict[str, str | bytes]], Path]:
    """Create a throw-away repository from a {relative path: content} mapping."""

    def _make(files: dict[str, str | bytes]) -> Path:
        root = tmp_path / "repo"
        for rel, content in files.items():
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                target.write_bytes(content)
            else:
                target.write_bytes(content.encode("utf-8"))
        root.mkdir(exist_ok=True)
        return root

    return _make

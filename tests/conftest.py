"""Shared fixtures for the RepoFinder test-suite."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from repofinder.ingest import build_index
from repofinder.models import RepoIndex

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_REPO = FIXTURES / "sample_repo"

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

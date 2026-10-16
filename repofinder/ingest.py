"""Repository ingestion: fetch a checkout, walk its source files, build an index.

Cloned repositories are treated as untrusted data. Nothing here executes code
from a checkout; files are only read.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from . import parser
from .models import FileRecord, ImportEdge, RepoIndex, Symbol

log = logging.getLogger(__name__)

MAX_FILE_BYTES = 2 * 1024 * 1024
"""Files larger than this (2 MiB) are skipped: no :class:`FileRecord`, only an
entry in :attr:`SourceScan.too_large` and one warning from :func:`build_index`.
Files that large are generated or bundled far more often than written by hand."""

SKIP_DIRS = frozenset(
    {
        # version control, Python environments, caches and packaging output
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".tox",
        ".nox",
        ".eggs",
        "site-packages",
        # JavaScript/TypeScript dependencies and framework output
        "node_modules",
        ".next",
        # build output and coverage reports, in any language
        "build",
        "dist",
        "out",
        "coverage",
        # vendored third-party code and license texts
        "third_party",
        "licenses",
    }
)
"""Directory names that are never descended into, at any depth.

They hold dependencies, environments, generated output or vendored code, not
the repository's own source. Every directory whose name starts with ``.`` is
skipped as well (``.github``, ``.next``, ``.cache``, ...), and so is a
``vendor/`` directory next to a package manifest (:data:`VENDOR_MANIFESTS`).
Test, example, documentation, website and experiment directories are source
and are kept.
"""

VENDOR_DIR_NAME = "vendor"
VENDOR_MANIFESTS = frozenset({"package.json", "go.mod", "composer.json", "Gemfile"})
"""A ``vendor/`` directory is skipped when its parent directory has one of these
files: there it holds installed dependencies (npm, Go modules, Composer,
Bundler). Elsewhere, e.g. in a Python project that vendors and patches code it
imports, ``vendor/`` is indexed like any other directory."""

SKIP_FILE_SUFFIXES = (".min.js", ".min.mjs", ".min.cjs")
"""Minified bundles: never indexed, whatever languages are registered."""

CLONE_TIMEOUT_SECONDS = 600

# https://github.com/owner/name[.git][/tree/<branch>...][/], scheme and www optional.
_GITHUB_HTTP_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/"
    r"(?P<owner>[A-Za-z0-9_.-]+)/(?P<name>[A-Za-z0-9_.-]+?)(?:\.git)?"
    r"(?:/.*)?$",
    re.IGNORECASE,
)
# git@github.com:owner/name[.git]
_GITHUB_SSH_RE = re.compile(
    r"^git@github\.com:(?P<owner>[A-Za-z0-9_.-]+)/(?P<name>[A-Za-z0-9_.-]+?)(?:\.git)?/?$",
    re.IGNORECASE,
)


class IngestError(RuntimeError):
    """Raised when a repository cannot be fetched or located."""


def _parse_github(source: str) -> Optional[tuple[str, str, str]]:
    """Return (owner, name, clone_url) for a GitHub URL, else None."""
    text = source.strip()
    match = _GITHUB_SSH_RE.match(text)
    if match:
        owner, name = match["owner"], match["name"]
        return owner, name, f"git@github.com:{owner}/{name}.git"
    match = _GITHUB_HTTP_RE.match(text)
    if match:
        owner, name = match["owner"], match["name"]
        return owner, name, f"https://github.com/{owner}/{name}.git"
    return None


def repo_name_from_source(source: str) -> str:
    """Short repository name: ``https://github.com/pallets/flask`` -> ``flask``.

    For a local path it is the directory name.
    """
    gh = _parse_github(source)
    if gh is not None:
        return gh[1]
    path = Path(source).expanduser()
    name = path.resolve().name if path.exists() else path.name
    if not name:
        raise IngestError(f"cannot derive a repository name from {source!r}")
    return name


def _run_git(args: list[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"  # never block on a credential prompt
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=CLONE_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as exc:
        raise IngestError("git executable not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise IngestError(f"git {args[0]} timed out after {CLONE_TIMEOUT_SECONDS}s") from exc


def _same_remote(a: str, b: str) -> bool:
    ga, gb = _parse_github(a), _parse_github(b)
    if ga is None or gb is None:
        return a.strip() == b.strip()
    return (ga[0].lower(), ga[1].lower()) == (gb[0].lower(), gb[1].lower())


def fetch_repo(source: str, dest_dir: Path) -> Path:
    """Return a local checkout of ``source``.

    * GitHub URL: shallow-clone into ``dest_dir / name``. An existing git checkout
      of the same remote at that location is reused as-is (no pull).
    * Local directory: returned resolved, nothing is copied.

    Raises :class:`IngestError` on any failure.
    """
    gh = _parse_github(source)
    if gh is None:
        path = Path(source).expanduser()
        if path.is_dir():
            return path.resolve()
        raise IngestError(f"{source!r} is neither a GitHub repository URL nor an existing directory")

    _owner, name, clone_url = gh
    dest_dir = Path(dest_dir)
    target = dest_dir / name

    if target.exists():
        if not target.is_dir():
            raise IngestError(f"{target} exists and is not a directory")
        if (target / ".git").exists():
            remote = _run_git(["config", "--get", "remote.origin.url"], cwd=target)
            existing = remote.stdout.strip()
            if remote.returncode != 0 or not _same_remote(existing, clone_url):
                raise IngestError(
                    f"{target} is a checkout of {existing or 'an unknown remote'}, not {clone_url}"
                )
            return target.resolve()
        if any(target.iterdir()):
            raise IngestError(f"{target} exists, is not empty and is not a git checkout")

    dest_dir.mkdir(parents=True, exist_ok=True)
    result = _run_git(["clone", "--depth", "1", "--", clone_url, str(target)])
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise IngestError(f"git clone {clone_url} failed: {detail}")
    return target.resolve()


def _is_link(path: str) -> bool:
    isjunction = getattr(os.path, "isjunction", None)  # Python 3.12+
    return os.path.islink(path) or bool(isjunction and isjunction(path))


@dataclass
class SourceScan:
    """What :func:`scan_source_files` found under a repository root."""

    files: list[Path] = field(default_factory=list)      # kept, in repo-relative POSIX order
    too_large: list[str] = field(default_factory=list)   # repo-relative paths over MAX_FILE_BYTES
    minified: list[str] = field(default_factory=list)    # repo-relative paths matching SKIP_FILE_SUFFIXES


def default_languages() -> tuple[str, ...]:
    """Every language registered in :data:`parser.LANGUAGES`, in registration order."""
    return tuple(parser.LANGUAGES)


def scan_source_files(root: Path, languages: Optional[tuple[str, ...]] = None) -> SourceScan:
    """Find the source files under ``root`` (the rules of :func:`iter_source_files`)
    and record the ones skipped for their size or as minified bundles.

    ``languages`` defaults to :func:`default_languages`; an unknown name raises
    ``ValueError``.
    """
    root = Path(root)
    extensions = parser.extensions_for(default_languages() if languages is None else tuple(languages))
    scan = SourceScan()
    found: list[tuple[str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        vendor_is_dependencies = not VENDOR_MANIFESTS.isdisjoint(filenames)
        dirnames[:] = [
            d
            for d in dirnames
            if d not in SKIP_DIRS
            and not d.startswith(".")
            and not (d == VENDOR_DIR_NAME and vendor_is_dependencies)
            and not _is_link(os.path.join(dirpath, d))
        ]
        for filename in filenames:
            if not filename.endswith(extensions):
                continue
            full = Path(dirpath, filename)
            if filename.lower().endswith(SKIP_FILE_SUFFIXES):
                scan.minified.append(full.relative_to(root).as_posix())
                continue
            try:
                if full.is_symlink() or not full.is_file():
                    continue
                if full.stat().st_size > MAX_FILE_BYTES:
                    scan.too_large.append(full.relative_to(root).as_posix())
                    continue
            except OSError:
                continue
            found.append((full.relative_to(root).as_posix(), full))
    found.sort(key=lambda item: item[0])
    scan.files = [full for _rel, full in found]
    scan.too_large.sort()
    scan.minified.sort()
    return scan


def iter_source_files(root: Path, languages: Optional[tuple[str, ...]] = None) -> Iterator[Path]:
    """Yield source files under ``root`` in sorted (repo-relative POSIX) order.

    ``languages`` defaults to every language registered in
    :data:`parser.LANGUAGES`. Skips the directories in :data:`SKIP_DIRS`, any
    directory starting with ``.``, a ``vendor/`` next to a package manifest
    (:data:`VENDOR_MANIFESTS`), symlinks, minified bundles
    (:data:`SKIP_FILE_SUFFIXES`) and files larger than :data:`MAX_FILE_BYTES`.
    Tests are kept.
    """
    yield from scan_source_files(root, languages).files


def _fallback_record(root: Path, path: Path) -> FileRecord:
    """Record for a file the parser could not handle at all."""
    rel = Path(path).resolve().relative_to(root).as_posix()
    spec = parser.language_for_path(rel)
    try:
        raw = Path(path).read_bytes()
    except OSError:
        raw = b""
    return FileRecord(
        path=rel,
        language=spec.name if spec else "unknown",
        size=len(raw),
        n_lines=parser.count_lines(raw),
        parse_ok=False,
    )


def build_index(
    root: Path, name: str, source: str, languages: Optional[tuple[str, ...]] = None
) -> RepoIndex:
    """Parse every source file under ``root`` and resolve imports.

    ``languages`` defaults to every registered language. A file that cannot be
    parsed is kept with ``parse_ok=False``; this function never raises because
    of a single bad file. Files over :data:`MAX_FILE_BYTES` get no record: their
    number is logged in one warning. Lists are sorted: files by path, symbols
    by (path, start_line), imports by (src_path, line).
    """
    root = Path(root).resolve()
    index = RepoIndex(name=name, root=str(root), source=source)
    files: list[FileRecord] = []
    symbols: list[Symbol] = []
    imports: list[ImportEdge] = []

    scan = scan_source_files(root, languages)
    if scan.too_large:
        n = len(scan.too_large)
        shown = ", ".join(scan.too_large[:3]) + (", ..." if n > 3 else "")
        log.warning("skipped %d file%s larger than %d MiB: %s",
                    n, "" if n == 1 else "s", MAX_FILE_BYTES // (1024 * 1024), shown)
    for path in scan.files:
        try:
            record, file_symbols, file_imports = parser.parse_file(root, path)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
            log.warning("could not parse %s: %s", path, exc)
            files.append(_fallback_record(root, path))
            continue
        files.append(record)
        symbols.extend(file_symbols)
        imports.extend(file_imports)

    # Stable sorts keep source order for ties (e.g. "import os, sys").
    files.sort(key=lambda f: f.path)
    symbols.sort(key=lambda s: (s.path, s.start_line))
    imports.sort(key=lambda e: (e.src_path, e.line))
    index.files, index.symbols, index.imports = files, symbols, imports

    parser.resolve_imports(index)
    return index

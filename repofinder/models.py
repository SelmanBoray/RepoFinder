"""Core data model shared by every RepoFinder layer.

These dataclasses are a fixed contract: ingestion, parsing and the SQLite store
produce and consume them, and later layers (search, architecture map, Q&A) build
on top of them. Do not rename fields or change their meaning.
"""

from dataclasses import dataclass, field
from pathlib import Path  # noqa: F401  (part of the public contract's import surface)
from typing import Literal, Optional

# "interface", "type", "enum" and "namespace" come from TypeScript; Python
# produces only the first three.
SymbolKind = Literal["function", "class", "method", "interface", "type", "enum", "namespace"]


@dataclass
class FileRecord:
    path: str          # repo-relative, POSIX separators ("src/flask/app.py")
    language: str      # "python", "typescript", "tsx"
    size: int          # bytes
    n_lines: int
    parse_ok: bool     # False if tree-sitter reported errors; symbols may be partial


@dataclass
class Symbol:
    id: str            # f"{path}::{qualname}", e.g. "src/flask/app.py::Flask.run"
    path: str
    name: str          # "run"
    qualname: str      # "Flask.run"; nested functions "outer.<locals>.inner"
    kind: SymbolKind
    start_line: int    # 1-based, inclusive
    end_line: int      # 1-based, inclusive
    signature: str     # one-line header text, e.g. "def run(self, host=None, port=None) -> None"
    docstring: Optional[str]   # first line(s) of docstring, stripped; None if absent
    parent: Optional[str]      # qualname of enclosing class/function, None at module level


@dataclass
class ImportEdge:
    src_path: str              # importing file, repo-relative POSIX
    dst_module: str            # dotted module as written/resolved, e.g. "flask.app" or "os.path"
    dst_path: Optional[str]    # repo-relative path if the module resolves inside the repo, else None
    names: list[str]           # imported names; ["*"] for star; [] for plain "import x"
    line: int                  # 1-based


@dataclass
class RepoIndex:
    name: str                  # e.g. "flask"
    root: str                  # absolute local path of the checkout (string, not Path)
    source: str                # original URL or path given by user
    files: list[FileRecord] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[ImportEdge] = field(default_factory=list)

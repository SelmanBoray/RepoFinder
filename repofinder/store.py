"""SQLite persistence for a single repository index.

One database holds one repository. ``save_index`` replaces the files, symbols
and imports tables wholesale; ``load_index`` returns an equal ``RepoIndex``.
Per-file summaries (written by later LLM layers) and free-form metadata live
alongside the index.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Optional

from .models import FileRecord, ImportEdge, RepoIndex, Symbol

SCHEMA_VERSION = "1"

# Reserved meta keys describing the stored index. Other keys are free for callers.
META_SCHEMA_VERSION = "schema_version"
META_INDEX_NAME = "index.name"
META_INDEX_ROOT = "index.root"
META_INDEX_SOURCE = "index.source"
META_INDEX_SAVED_AT = "index.saved_at"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
    seq      INTEGER PRIMARY KEY,
    path     TEXT NOT NULL UNIQUE,
    language TEXT NOT NULL,
    size     INTEGER NOT NULL,
    n_lines  INTEGER NOT NULL,
    parse_ok INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS symbols (
    seq        INTEGER PRIMARY KEY,
    id         TEXT NOT NULL,       -- not unique: e.g. property getter + setter
    path       TEXT NOT NULL,
    name       TEXT NOT NULL,
    qualname   TEXT NOT NULL,
    kind       TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line   INTEGER NOT NULL,
    signature  TEXT NOT NULL,
    docstring  TEXT,
    parent     TEXT
);
CREATE INDEX IF NOT EXISTS symbols_id ON symbols(id);
CREATE INDEX IF NOT EXISTS symbols_path ON symbols(path, start_line);
CREATE INDEX IF NOT EXISTS symbols_name ON symbols(name);
CREATE TABLE IF NOT EXISTS imports (
    seq        INTEGER PRIMARY KEY,
    src_path   TEXT NOT NULL,
    dst_module TEXT NOT NULL,
    dst_path   TEXT,
    names      TEXT NOT NULL,       -- JSON array of strings
    line       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS imports_src ON imports(src_path, line);
CREATE INDEX IF NOT EXISTS imports_dst ON imports(dst_path);
CREATE TABLE IF NOT EXISTS summaries (
    path       TEXT PRIMARY KEY,
    summary    TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class StoreError(RuntimeError):
    """Raised for an empty or incompatible store."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    """A single-repository SQLite index. Usable as a context manager."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        if str(db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = sqlite3.connect(str(db_path))
        with self._conn:
            self._conn.executescript(_SCHEMA)
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key = ?", (META_SCHEMA_VERSION,)
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO meta(key, value) VALUES (?, ?)",
                    (META_SCHEMA_VERSION, SCHEMA_VERSION),
                )
            elif row[0] != SCHEMA_VERSION:
                self.close()
                raise StoreError(
                    f"{db_path} has schema version {row[0]}, expected {SCHEMA_VERSION}"
                )

    # -- connection management ------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise StoreError("store is closed")
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "Store":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()

    # -- index -----------------------------------------------------------------

    def save_index(self, index: RepoIndex) -> None:
        """Replace the stored index with ``index`` in one transaction.

        Summaries survive a re-index of the same source for paths that still
        exist; they are all dropped when the source changes. Caller-defined meta
        keys are left untouched.
        """
        conn = self.conn
        with conn:
            previous_source = self.get_meta(META_INDEX_SOURCE)
            conn.execute("DELETE FROM files")
            conn.execute("DELETE FROM symbols")
            conn.execute("DELETE FROM imports")
            conn.executemany(
                "INSERT INTO files(path, language, size, n_lines, parse_ok) VALUES (?, ?, ?, ?, ?)",
                [(f.path, f.language, f.size, f.n_lines, int(f.parse_ok)) for f in index.files],
            )
            conn.executemany(
                "INSERT INTO symbols(id, path, name, qualname, kind, start_line, end_line,"
                " signature, docstring, parent) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        s.id, s.path, s.name, s.qualname, s.kind, s.start_line,
                        s.end_line, s.signature, s.docstring, s.parent,
                    )
                    for s in index.symbols
                ],
            )
            conn.executemany(
                "INSERT INTO imports(src_path, dst_module, dst_path, names, line)"
                " VALUES (?, ?, ?, ?, ?)",
                [
                    (e.src_path, e.dst_module, e.dst_path, json.dumps(list(e.names)), e.line)
                    for e in index.imports
                ],
            )
            if previous_source is not None and previous_source != index.source:
                conn.execute("DELETE FROM summaries")
            else:
                conn.execute("DELETE FROM summaries WHERE path NOT IN (SELECT path FROM files)")
            for key, value in (
                (META_INDEX_NAME, index.name),
                (META_INDEX_ROOT, index.root),
                (META_INDEX_SOURCE, index.source),
                (META_INDEX_SAVED_AT, _now()),
            ):
                self._put_meta(key, value)

    def has_index(self) -> bool:
        return self.get_meta(META_INDEX_NAME) is not None

    def load_index(self) -> RepoIndex:
        """Load the stored index. Raises :class:`StoreError` if nothing was saved."""
        if not self.has_index():
            raise StoreError(f"no index stored in {self.db_path}")
        conn = self.conn
        files = [
            FileRecord(path=r[0], language=r[1], size=r[2], n_lines=r[3], parse_ok=bool(r[4]))
            for r in conn.execute(
                "SELECT path, language, size, n_lines, parse_ok FROM files ORDER BY path, seq"
            )
        ]
        symbols = [
            Symbol(
                id=r[0], path=r[1], name=r[2], qualname=r[3], kind=r[4], start_line=r[5],
                end_line=r[6], signature=r[7], docstring=r[8], parent=r[9],
            )
            for r in conn.execute(
                "SELECT id, path, name, qualname, kind, start_line, end_line, signature,"
                " docstring, parent FROM symbols ORDER BY path, start_line, seq"
            )
        ]
        imports = [
            ImportEdge(src_path=r[0], dst_module=r[1], dst_path=r[2], names=json.loads(r[3]), line=r[4])
            for r in conn.execute(
                "SELECT src_path, dst_module, dst_path, names, line FROM imports"
                " ORDER BY src_path, line, seq"
            )
        ]
        return RepoIndex(
            name=self.get_meta(META_INDEX_NAME) or "",
            root=self.get_meta(META_INDEX_ROOT) or "",
            source=self.get_meta(META_INDEX_SOURCE) or "",
            files=files,
            symbols=symbols,
            imports=imports,
        )

    # -- summaries ---------------------------------------------------------------

    def save_summary(self, path: str, summary: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO summaries(path, summary, updated_at) VALUES (?, ?, ?)"
                " ON CONFLICT(path) DO UPDATE SET summary = excluded.summary,"
                " updated_at = excluded.updated_at",
                (path, summary, _now()),
            )

    def get_summary(self, path: str) -> Optional[str]:
        row = self.conn.execute("SELECT summary FROM summaries WHERE path = ?", (path,)).fetchone()
        return row[0] if row else None

    def all_summaries(self) -> dict[str, str]:
        rows = self.conn.execute("SELECT path, summary FROM summaries ORDER BY path")
        return {path: summary for path, summary in rows}

    # -- meta ----------------------------------------------------------------------

    def set_meta(self, key: str, value: str) -> None:
        with self.conn:
            self._put_meta(key, value)

    def get_meta(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _put_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

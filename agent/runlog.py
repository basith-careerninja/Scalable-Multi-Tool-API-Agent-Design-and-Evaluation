"""Run log: one row per user request, stored in SQLite.

It serves three purposes with one table: the history that system search
answers from ("status of my last request"), an audit trail of what was
called, and the raw material for later evaluation.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "runs.sqlite"

# Statuses a run moves through.
PLANNED, WAITING_APPROVAL, WAITING_USER, COMPLETED, FAILED = (
    "planned", "waiting_approval", "waiting_user", "completed", "failed",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    session    TEXT NOT NULL,
    message    TEXT NOT NULL,
    intent     TEXT NOT NULL,
    operation  TEXT,
    status     TEXT NOT NULL,
    detail     TEXT NOT NULL
)
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunLog:
    def __init__(self, path: Path | str = DEFAULT_PATH):
        self._db = sqlite3.connect(str(path))
        self._db.row_factory = sqlite3.Row
        self._db.execute(_SCHEMA)
        self._db.commit()

    def record(self, session: str, message: str, intent: str, operation: str | None = None,
               status: str = PLANNED, detail: dict | None = None) -> int:
        now = _now()
        cursor = self._db.execute(
            "INSERT INTO runs (created_at, updated_at, session, message, intent, operation, status, detail) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (now, now, session, message, intent, operation, status, json.dumps(detail or {})),
        )
        self._db.commit()
        return cursor.lastrowid

    def update(self, run_id: int, status: str, detail: dict | None = None) -> None:
        if detail is None:
            self._db.execute("UPDATE runs SET status = ?, updated_at = ? WHERE id = ?", (status, _now(), run_id))
        else:
            self._db.execute(
                "UPDATE runs SET status = ?, updated_at = ?, detail = ? WHERE id = ?",
                (status, _now(), json.dumps(detail), run_id),
            )
        self._db.commit()

    def recent(self, session: str, limit: int = 5) -> list[dict]:
        rows = self._db.execute(
            "SELECT * FROM runs WHERE session = ? ORDER BY id DESC LIMIT ?", (session, limit)
        ).fetchall()
        return [{**dict(row), "detail": json.loads(row["detail"])} for row in rows]

    def close(self) -> None:
        self._db.close()

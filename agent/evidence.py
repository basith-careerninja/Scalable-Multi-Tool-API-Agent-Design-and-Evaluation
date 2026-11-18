"""Evidence: a record of every call that changes something, written before it is sent.

The LangGraph checkpoint holds the conversation (what was asked, planned, approved). This
store holds proof of what was sent and what came back. They have different owners and
commit at different moments:
  - the checkpoint is written by the graph at the end of each node;
  - evidence is written by the execute step around each HTTP call: "intended" and
    "sent" are committed before the request leaves, the outcome right after it returns.
So a crash between the two leaves a record in state "sent": the outcome is unknown, and
that is visible, instead of silently lost. `agent/reconcile.py` settles such records.

Reads are not recorded: they change nothing and are simply run again.

Each record keeps the exact request and response. The request is needed to replay it
with the same PayPal-Request-Id; the response is needed when a resumed run skips a call
that already succeeded. In production this store would be encrypted, expire old rows and
use an append-only role; here it is a separate SQLite file to show the boundary.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "evidence.sqlite"

INTENDED = "intended"        # about to be sent; nothing has left yet
SENT = "sent"                # the request may have reached PayPal; outcome not recorded
SUCCEEDED = "succeeded"      # 2xx recorded, or confirmed by reconciling
FAILED = "failed"            # PayPal answered with an error: the call took no effect
NOT_APPLIED = "not_applied"  # reconciling showed it did not happen: safe to send again
NEEDS_ATTENTION = "needs_attention"  # could not be settled; a person must check

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL,
    session     TEXT NOT NULL,
    step        INTEGER NOT NULL,
    attempt     INTEGER NOT NULL,
    operation   TEXT NOT NULL,
    method      TEXT NOT NULL,
    path        TEXT NOT NULL,
    input_hash  TEXT NOT NULL,
    request_id  TEXT NOT NULL,
    request     TEXT NOT NULL,
    state       TEXT NOT NULL,
    http_status INTEGER,
    response    TEXT,
    produced    TEXT NOT NULL DEFAULT '{}',
    note        TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    UNIQUE (run_id, step, attempt)
)
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def input_hash(method: str, path: str, query: dict, body: Any) -> str:
    canonical = json.dumps([method, path, query or {}, body], sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class Evidence:
    def __init__(self, path: Path | str = DEFAULT_PATH):
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute(_SCHEMA)
        self._db.commit()

    # -- writing, in the order a call goes through --------------------------------------

    def intend(self, run_id: int, session: str, step: int, operation: str, method: str, path: str,
               query: dict, body: Any) -> dict:
        """Record the call before it is sent; a new attempt if earlier ones did not apply."""
        digest = input_hash(method, path, query, body)
        attempt = 1 + (self._db.execute("SELECT MAX(attempt) FROM calls WHERE run_id = ? AND step = ?",
                                        (run_id, step)).fetchone()[0] or 0)
        # Same run, step and input -> same id, so PayPal can recognise a replay of this attempt.
        request_id = f"agent-{run_id}-{step}-{attempt}-{digest[:12]}"
        now = _now()
        cursor = self._db.execute(
            "INSERT INTO calls (run_id, session, step, attempt, operation, method, path, input_hash, request_id, "
            "request, state, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, session, step, attempt, operation, method, path, digest, request_id,
             json.dumps({"query": query or {}, "body": body}, default=str), INTENDED, now, now))
        self._db.commit()
        return self.get(cursor.lastrowid)

    def mark_sent(self, record_id: int) -> None:
        self._set(record_id, state=SENT)

    def finish(self, record_id: int, status: int, response: Any, produced: dict) -> dict:
        """The outcome. Status 0 (no answer: timeout, dropped connection) stays 'sent': unknown."""
        if status == 0:
            self._set(record_id, note="no response from PayPal")
        else:
            self._set(record_id, state=SUCCEEDED if 200 <= status < 300 else FAILED, http_status=status,
                      response=json.dumps(response, default=str), produced=json.dumps(produced))
        return self.get(record_id)

    def settle(self, record_id: int, state: str, note: str, status: int | None = None,
               response: Any = None, produced: dict | None = None) -> dict:
        """Outcome found by reconciling."""
        fields = {"state": state, "note": note}
        if status is not None:
            fields.update(http_status=status, response=json.dumps(response, default=str),
                          produced=json.dumps(produced or {}))
        self._set(record_id, **fields)
        return self.get(record_id)

    def _set(self, record_id: int, **fields) -> None:
        fields["updated_at"] = _now()
        names = ", ".join(f"{name} = ?" for name in fields)
        self._db.execute(f"UPDATE calls SET {names} WHERE id = ?", (*fields.values(), record_id))
        self._db.commit()

    # -- reading ---------------------------------------------------------------------------

    def get(self, record_id: int) -> dict:
        return _row(self._db.execute("SELECT * FROM calls WHERE id = ?", (record_id,)).fetchone())

    def latest(self, run_id: int, step: int) -> dict | None:
        row = self._db.execute("SELECT * FROM calls WHERE run_id = ? AND step = ? ORDER BY attempt DESC LIMIT 1",
                               (run_id, step)).fetchone()
        return _row(row) if row else None

    def for_run(self, run_id: int) -> list[dict]:
        rows = self._db.execute("SELECT * FROM calls WHERE run_id = ? ORDER BY step, attempt", (run_id,)).fetchall()
        latest = {}
        for row in rows:
            latest[row["step"]] = _row(row)
        return [latest[step] for step in sorted(latest)]

    def unsettled(self) -> list[dict]:
        """Calls that may have been sent but have no recorded outcome."""
        return [_row(r) for r in self._db.execute("SELECT * FROM calls WHERE state = ? ORDER BY id", (SENT,))]

    def close(self) -> None:
        self._db.close()


def _row(row) -> dict:
    out = dict(row)
    out["request"] = json.loads(out["request"])
    out["response"] = json.loads(out["response"]) if out["response"] else None
    out["produced"] = json.loads(out["produced"] or "{}")
    return out

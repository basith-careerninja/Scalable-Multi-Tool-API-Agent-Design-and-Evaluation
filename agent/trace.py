"""Tracing: what the agent did for each message, step by step.

Every chat turn is one trace: a tree of spans (select, plan, fill, api_call,
compute, answer, ...) with model calls nested under the step that made them.
Each span has a duration, a status and a few attributes. Traces go to a sink;
SqliteSink keeps them locally, and another sink (LangSmith, Langfuse) can be
added by implementing `write(spans)`.

Nothing here handles credentials: API auth headers are never passed to a span.
Prompts and responses are stored truncated, on this machine only.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "runs.sqlite"
MAX_TEXT = 4000


@dataclass
class Span:
    id: str
    trace_id: str
    parent_id: str | None
    name: str
    kind: str                     # turn, step, llm, api
    started_at: str
    seq: int = 0                  # start order within the process, for stable display
    duration_ms: float = 0.0
    status: str = "ok"
    attrs: dict = field(default_factory=dict)

    def set(self, **attrs: Any) -> None:
        self.attrs.update({k: _clip(v) for k, v in attrs.items()})


def _clip(value: Any) -> Any:
    if isinstance(value, str):
        return value if len(value) <= MAX_TEXT else value[:MAX_TEXT] + "...(truncated)"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    text = json.dumps(value, default=str)
    return json.loads(text) if len(text) <= MAX_TEXT else text[:MAX_TEXT] + "...(truncated)"


class Tracer:
    def __init__(self, sink=None, langsmith: bool = False):
        """langsmith: also record model, Jev and API calls in LangSmith, nested under the
        LangGraph node that made them. LangGraph itself traces the graph and its nodes."""
        self.sink = sink
        self.langsmith = langsmith
        self._current: ContextVar[Span | None] = ContextVar("current_span", default=None)
        self._open: dict[str, list[Span]] = {}
        self._seq = 0

    @contextmanager
    def span(self, name: str, kind: str = "step", **attrs: Any):
        parent = self._current.get()
        self._seq += 1
        span = Span(
            seq=self._seq,
            id=uuid.uuid4().hex[:12],
            trace_id=parent.trace_id if parent else uuid.uuid4().hex[:12],
            parent_id=parent.id if parent else None,
            name=name, kind=kind,
            started_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        )
        span.set(**attrs)
        token = self._current.set(span)
        remote = _langsmith_open(span) if self.langsmith and kind in ("llm", "api") else None
        began = time.perf_counter()
        try:
            yield span
        except Exception as err:
            span.status = "error"
            span.set(error=f"{type(err).__name__}: {err}"[:400])
            raise
        finally:
            span.duration_ms = round((time.perf_counter() - began) * 1000, 1)
            self._current.reset(token)
            _langsmith_close(remote, span)
            self._open.setdefault(span.trace_id, []).append(span)
            if parent is None:
                spans = self._open.pop(span.trace_id)
                if self.sink is not None:
                    try:
                        self.sink.write(spans)
                    except Exception:
                        pass  # tracing must never break a request


def _langsmith_open(span: Span):
    """Start a LangSmith run for a model or API call. None if LangSmith is unavailable."""
    try:
        from langsmith import trace
        if span.kind == "llm":
            name, run_type = str(span.attrs.get("model") or span.name), "llm"
            inputs = {k: span.attrs[k] for k in ("step", "user") if span.attrs.get(k)}
            metadata = {"ls_model_name": span.attrs.get("model")}
        else:
            name, run_type = f"api: {span.attrs.get('operation')}", "tool"
            inputs = {k: span.attrs[k] for k in ("method", "path", "query") if span.attrs.get(k) is not None}
            metadata = {}
        manager = trace(name, run_type=run_type, inputs=inputs, metadata=metadata)
        return manager, manager.__enter__()
    except Exception:
        return None


def _langsmith_close(remote, span: Span) -> None:
    if remote is None:
        return
    manager, run = remote
    try:
        outputs = {k: v for k, v in span.attrs.items() if k not in ("user", "query", "error")}
        if span.kind == "llm" and span.attrs.get("prompt_tokens") is not None:
            prompt, completion = span.attrs.get("prompt_tokens") or 0, span.attrs.get("completion_tokens") or 0
            outputs["usage_metadata"] = {"input_tokens": prompt, "output_tokens": completion,
                                         "total_tokens": prompt + completion}
        error = span.attrs.get("error") if span.status != "ok" else None
        run.end(outputs=outputs, error=str(error) if error else None)
        manager.__exit__(None, None, None)
    except Exception:
        pass  # tracing must never break a request


class TracedLLM:
    """Wraps a model client so every call becomes a span under the current step."""

    def __init__(self, llm, tracer: Tracer):
        self.inner = llm
        self.tracer = tracer

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def json(self, system: str, user: str) -> dict:
        with self.tracer.span("llm", kind="llm", model=getattr(self.inner, "model", None),
                              prompt_chars=len(system) + len(user), user=user) as span:
            reply = self.inner.json(system, user)
            usage = getattr(self.inner, "last_usage", None) or {}
            retry = getattr(self.inner, "last_retry", None) or {}
            span.set(response=reply, prompt_tokens=usage.get("prompt_tokens"),
                     completion_tokens=usage.get("completion_tokens"),
                     retries=retry.get("retries") or None, waited_s=retry.get("waited_s") or None)
            return reply


class ListSink:
    """Keeps traces in memory. Used in tests."""

    def __init__(self):
        self.traces: list[list[Span]] = []

    def write(self, spans: list[Span]) -> None:
        self.traces.append(spans)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
    id          TEXT PRIMARY KEY,
    trace_id    TEXT NOT NULL,
    parent_id   TEXT,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    duration_ms REAL NOT NULL,
    status      TEXT NOT NULL,
    attrs       TEXT NOT NULL
)
"""


class SqliteSink:
    def __init__(self, path: Path | str = DEFAULT_PATH):
        self._db = sqlite3.connect(str(path))
        self._db.row_factory = sqlite3.Row
        self._db.execute(_SCHEMA)
        self._db.execute("CREATE INDEX IF NOT EXISTS spans_trace ON spans (trace_id)")
        self._db.commit()

    def write(self, spans: list[Span]) -> None:
        self._db.executemany(
            "INSERT OR REPLACE INTO spans VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(s.id, s.trace_id, s.parent_id, s.name, s.kind, s.started_at, s.seq, s.duration_ms, s.status,
              json.dumps(s.attrs, default=str)) for s in spans],
        )
        self._db.commit()

    def recent(self, limit: int = 15) -> list[dict]:
        """One summary row per trace, newest first."""
        roots = self._db.execute(
            "SELECT * FROM spans WHERE parent_id IS NULL ORDER BY rowid DESC LIMIT ?", (limit,)
        ).fetchall()
        return [summarise(self.trace(root["trace_id"])) for root in roots]

    def trace(self, trace_id: str) -> list[dict]:
        rows = self._db.execute(
            "SELECT * FROM spans WHERE trace_id = ? ORDER BY seq", (trace_id,)
        ).fetchall()
        return [{**dict(row), "attrs": json.loads(row["attrs"])} for row in rows]

    def close(self) -> None:
        self._db.close()


def summarise(spans: list[dict]) -> dict:
    root = next(s for s in spans if s["parent_id"] is None)
    llm = [s for s in spans if s["kind"] == "llm"]
    return {
        "trace_id": root["trace_id"],
        "at": root["started_at"],
        "message": root["attrs"].get("message", ""),
        "status": root["attrs"].get("reply_status", root["status"]),
        "duration_ms": root["duration_ms"],
        "llm_calls": len(llm),
        "llm_ms": round(sum(s["duration_ms"] for s in llm), 1),
        "prompt_tokens": sum(s["attrs"].get("prompt_tokens") or 0 for s in llm),
        "completion_tokens": sum(s["attrs"].get("completion_tokens") or 0 for s in llm),
        "api_calls": sum(s["kind"] == "api" for s in spans),
        "errors": sum(s["status"] == "error" for s in spans),
    }


def render(spans: list[dict], verbose: bool = False) -> str:
    """A trace as an indented tree."""
    children: dict[str | None, list[dict]] = {}
    for span in spans:
        children.setdefault(span["parent_id"], []).append(span)
    lines: list[str] = []

    def walk(span: dict, depth: int) -> None:
        attrs = span["attrs"]
        notes = []
        for key in ("operation", "intent", "method", "path", "http_status", "plan", "model", "result",
                    "id_name", "found_id", "via", "decided_by", "step", "reason", "scores",
                    "prompt_tokens", "completion_tokens", "retries", "waited_s", "reply_status", "error"):
            if attrs.get(key) not in (None, "", []):
                notes.append(f"{key}={attrs[key]}")
        mark = "" if span["status"] == "ok" else "  [ERROR]"
        lines.append(f"{'  ' * depth}{span['name']}  {span['duration_ms']:.0f} ms{mark}  {'  '.join(notes)}")
        if verbose:
            for key in ("message", "user", "response", "query", "body"):
                if attrs.get(key) not in (None, "", {}):
                    text = attrs[key] if isinstance(attrs[key], str) else json.dumps(attrs[key])
                    lines.append(f"{'  ' * (depth + 2)}{key}: {text[:600]}")
        for child in children.get(span["id"], []):
            walk(child, depth + 1)

    for root in children.get(None, []):
        walk(root, 0)
    return "\n".join(lines)

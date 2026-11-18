"""Caching wrappers for the model and Jev clients, plus per-case counters.

Every answer is stored by a hash of exactly what was sent, so re-running the eval on
unchanged code costs nothing and gives the same result. Any change to a prompt, or to
the message, misses the cache and asks the model again. --no-cache turns this off.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent / "cache.sqlite"


class Cache:
    def __init__(self, path: Path | str = DEFAULT_PATH, enabled: bool = True):
        self.enabled = enabled
        self._db = sqlite3.connect(str(path))
        self._db.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self._db.commit()

    def get(self, key: str):
        if not self.enabled:
            return None
        row = self._db.execute("SELECT value FROM answers WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value) -> None:
        if not self.enabled:
            return  # --no-cache: neither read nor write, so a fresh run leaves the stored answers as they were
        self._db.execute("INSERT OR REPLACE INTO answers VALUES (?, ?)", (key, json.dumps(value)))
        self._db.commit()


class Stats:
    """Counters for one case: model calls, tokens, rejected answers, time spent on fresh calls."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.calls = self.tokens = self.rejected = self.hits = 0
        self.fresh_seconds = 0.0

    def as_dict(self) -> dict:
        return {"calls": self.calls, "tokens": self.tokens, "rejected": self.rejected, "cache_hits": self.hits,
                "fresh": self.calls > self.hits, "fresh_seconds": round(self.fresh_seconds, 3)}


def _key(kind: str, *parts) -> str:
    return kind + ":" + hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


class CachedLLM:
    def __init__(self, inner, cache: Cache, stats: Stats):
        self.inner, self.cache, self.stats = inner, cache, stats
        self.model = inner.model
        self.last_usage: dict = {}
        self.last_retry: dict = {}

    def json(self, system: str, user: str) -> dict:
        self.stats.calls += 1
        if "previous answer was rejected" in user:
            self.stats.rejected += 1
        key = _key("llm", self.model, system, user)
        hit = self.cache.get(key)
        if hit is not None:
            self.stats.hits += 1
            self.last_usage = hit["usage"]
        else:
            began = time.perf_counter()
            reply = self.inner.json(system, user)
            waited = (getattr(self.inner, "last_retry", None) or {}).get("waited_s", 0.0)
            self.stats.fresh_seconds += max(0.0, time.perf_counter() - began - waited)
            self.last_usage = getattr(self.inner, "last_usage", None) or {}
            hit = {"reply": reply, "usage": self.last_usage}
            self.cache.put(key, hit)
        self.stats.tokens += (self.last_usage.get("prompt_tokens") or 0) + (self.last_usage.get("completion_tokens") or 0)
        return hit["reply"]


class CachedJev:
    def __init__(self, inner, cache: Cache, stats: Stats):
        self.inner, self.cache, self.stats = inner, cache, stats
        self.model = inner.model
        self.last_usage: dict = {}

    def ask(self, state: dict, questions: dict) -> dict:
        self.stats.calls += 1
        key = _key("jev", self.model, state, questions)
        hit = self.cache.get(key)
        if hit is not None:
            self.stats.hits += 1
            self.last_usage = hit["usage"]
        else:
            began = time.perf_counter()
            answers = self.inner.ask(state, questions)
            self.stats.fresh_seconds += time.perf_counter() - began
            self.last_usage = getattr(self.inner, "last_usage", None) or {}
            hit = {"answers": answers, "usage": self.last_usage}
            self.cache.put(key, hit)
        self.stats.tokens += (self.last_usage.get("input_tokens") or 0) + (self.last_usage.get("output_tokens") or 0)
        return hit["answers"]

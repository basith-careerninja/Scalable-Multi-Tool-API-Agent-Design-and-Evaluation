"""System search tool: questions about the system itself.

Two kinds of question, both answered from data the system already holds:
  capabilities  "What tools are available for managing invoices?"  -> catalogue
  history       "What's the status of my last request?"            -> run log

No extra model call is made. When Jev routed the message, its route request already
answered "capabilities or history?" and scored every resource; those probabilities
come in as a hint. Without a hint (the LLM selector, or Jev not configured) keyword
matching decides.
"""
from __future__ import annotations

import re

from . import config
from .catalog import Catalog
from .runlog import RunLog

_HISTORY_WORDS = {"last", "previous", "earlier", "recent", "status", "history", "request", "requests", "ran", "did"}
_STOP = {"what", "which", "tools", "tool", "are", "is", "the", "for", "a", "an", "of", "to", "can", "you", "do",
         "available", "managing", "manage", "with", "my", "me", "show", "list", "there", "any", "have", "about"}


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words}


class SystemSearch:
    def __init__(self, catalog: Catalog, runlog: RunLog, evidence=None):
        self.catalog = catalog
        self.runlog = runlog
        self.evidence = evidence  # per-call outcomes for history answers (agent/evidence.py)

    def search(self, query: str, session: str, limit: int = 20, hint: dict | None = None) -> dict:
        """hint: {"kind": {capabilities|history: p}, "resources": {resource: p}} from the Jev route request."""
        if hint and hint.get("kind"):
            return self._from_hint(hint, session, limit)
        words = _tokens(query)
        capabilities = self._capabilities(words - _tokens(" ".join(_STOP)), limit)
        if capabilities:
            return {"kind": "capabilities", "operations": capabilities, "decided_by": "keywords"}
        if words & _tokens(" ".join(_HISTORY_WORDS)):
            return {"kind": "history", "runs": self._history(session), "decided_by": "keywords"}
        return {"kind": "overview", "resources": self._overview(), "decided_by": "keywords"}

    def _from_hint(self, hint: dict, session: str, limit: int) -> dict:
        if hint["kind"].get("history", 0.0) >= config.SYSTEM_KIND_MIN:
            return {"kind": "history", "runs": self._history(session), "decided_by": "jev"}
        resources = {r: p for r, p in (hint.get("resources") or {}).items() if r in self.catalog.resources()}
        if resources:
            resource, p = max(resources.items(), key=lambda kv: kv[1])
            if p >= config.SYSTEM_RESOURCE_MIN:
                ops = sorted(self.catalog.actions(resource), key=lambda op: op.id)[:limit]
                return {"kind": "capabilities", "decided_by": "jev",
                        "operations": [{"id": op.id, "name": op.name, "resource": op.resource, "risk": op.risk}
                                       for op in ops]}
        return {"kind": "overview", "resources": self._overview(), "decided_by": "jev"}

    def _capabilities(self, words: set[str], limit: int) -> list[dict]:
        scored = []
        for op in self.catalog:
            resource_hit = bool(words & _tokens(op.resource.replace("-", " ")))
            score = 3 * resource_hit + len(words & _tokens(f"{op.name} {op.group}"))
            if score:
                scored.append((score, op))
        if not scored:
            return []
        top = max(score for score, _ in scored)
        # Keep only the strongest matches so a stray word does not list everything.
        keep = [op for score, op in sorted(scored, key=lambda pair: (-pair[0], pair[1].id)) if score >= max(2, top - 1)]
        return [
            {"id": op.id, "name": op.name, "resource": op.resource, "risk": op.risk}
            for op in keep[:limit]
        ]

    def _history(self, session: str) -> list[dict]:
        return [
            {"id": run["id"], "at": run["created_at"], "message": run["message"],
             "operation": run["operation"], "status": run["status"], "calls": self._calls(run["id"])}
            for run in self.runlog.recent(session)
        ]

    def _calls(self, run_id: int) -> list[dict]:
        if self.evidence is None:
            return []
        return [{"operation": r["operation"], "name": self.catalog[r["operation"]].name, "state": r["state"],
                 "note": r["note"]} for r in self.evidence.for_run(run_id)]

    def _overview(self) -> list[dict]:
        return [
            {"resource": resource, "actions": len(self.catalog.actions(resource))}
            for resource in self.catalog.resources()
        ]

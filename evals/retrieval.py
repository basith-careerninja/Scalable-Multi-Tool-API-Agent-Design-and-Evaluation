"""Eval baseline in the style of langgraph-bigtool: retrieve tools by embedding, then let the LLM pick.

langgraph-bigtool keeps every tool in an embedding store; the model calls retrieve_tools with
a query and the top results (2 by default) are bound for its next turn. This baseline does
the same in one step so it can be scored like the other variants:

  1. embed every tool card ("id: name. description") once per run, with OpenAI's
     text-embedding-3-small (nothing is read from or written to disk)
  2. embed the user's message and keep the K most similar tools
  3. give those K tools to the all-tools prompt, which picks one

It reports recall@K: whether the right tool was among the K retrieved, which bounds what
the LLM can get right. It is a comparison, not part of the agent.
"""
from __future__ import annotations

import http.client
import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import replace

from agent.catalog import Catalog
from agent.llm import load_env
from evals.all_tools import AllToolsSelector

MODEL = "text-embedding-3-small"
URL = "https://api.openai.com/v1/embeddings"


class Embedder:
    def __init__(self, model: str = MODEL):
        load_env()
        self.key = os.environ.get("OPENAI_API_KEY")
        if not self.key:
            raise RuntimeError("set OPENAI_API_KEY in .env for the retrieval baseline")
        self.model = model
        self.tokens = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for start in range(0, len(texts), 256):
            body = json.dumps({"model": self.model, "input": texts[start:start + 256]}).encode()
            request = urllib.request.Request(URL, data=body, method="POST", headers={
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(request, timeout=60) as response:
                        payload = json.load(response)
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as err:
                    if attempt == 4:
                        raise RuntimeError(f"embedding failed: {err}") from err
                    time.sleep(2 ** attempt)
            self.tokens += (payload.get("usage") or {}).get("total_tokens", 0)
            out += [row["embedding"] for row in sorted(payload["data"], key=lambda r: r["index"])]
        return out


def card(op) -> str:
    return f"{op.id}: {op.name}. {op.description[:200]}"


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class RetrievalSelector:
    """Top-K tools by embedding similarity, then the all-tools prompt over those K."""

    def __init__(self, catalog: Catalog, llm, embedder: Embedder, k: int = 10, default_service: str | None = None):
        self.catalog, self.llm, self.embedder, self.k = catalog, llm, embedder, k
        self.default_service = default_service
        ops = list(catalog)
        self._ids = [op.id for op in ops]
        self._vectors = [_unit(v) for v in embedder.embed([card(op) for op in ops])]
        self.last_retrieved: list[str] = []

    def retrieve(self, message: str) -> list[str]:
        query = _unit(self.embedder.embed([message])[0])
        scored = sorted(zip(self._ids, self._vectors), key=lambda iv: -sum(a * b for a, b in zip(query, iv[1])))
        return [op_id for op_id, _ in scored[: self.k]]

    def select(self, message: str, remembered: dict | None = None):
        self.last_retrieved = self.retrieve(message)
        subset = Catalog([self.catalog[i] for i in self.last_retrieved])
        # The prompt lists only the K tools; creators() and looks_like_id() still see the whole catalogue.
        chooser = AllToolsSelector(_Scoped(subset, self.catalog), self.llm, default_service=self.default_service)
        return replace(chooser.select(message, remembered), decided_by=f"retrieval@{self.k}")


class _Scoped:
    """The K retrieved tools for choosing; the full catalogue for every other lookup."""

    def __init__(self, subset: Catalog, full: Catalog):
        self._subset, self._full = subset, full

    def __iter__(self):
        return iter(self._subset)

    def __getitem__(self, op_id):
        return self._full[op_id]

    def services(self):
        return self._full.services()

    def __getattr__(self, name):
        return getattr(self._full, name)

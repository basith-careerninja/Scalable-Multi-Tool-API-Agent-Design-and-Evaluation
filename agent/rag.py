"""RAG tool: answer documentation questions from a local knowledge base.

Retrieval is BM25 over heading-aware chunks of the files in knowledge/. It
runs in memory with no extra service and no embedding API. The model then
answers from the retrieved passages only and names the ones it used; if no
passage matches, the tool says so without calling the model.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "knowledge"
NOT_FOUND = "I could not find that in the documentation I have."

_STOP = set("a an and are as at be by can do does for from how i if in is it of on or that the this to was "
            "what when where which who why will with you your my me".split())
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    source: str   # file name
    heading: str  # "Invoices > Send invoice"
    text: str


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words if w not in _STOP]


def chunk_markdown(text: str, source: str, max_chars: int = 900) -> list[Chunk]:
    """Split on headings, keeping the heading trail; long sections are split on paragraphs."""
    chunks, trail, buffer = [], [], []

    def flush():
        body = "\n".join(buffer).strip()
        buffer.clear()
        if not body:
            return
        heading = " > ".join(trail[1:] or trail)
        piece = ""
        for paragraph in re.split(r"\n\s*\n", body):
            if piece and len(piece) + len(paragraph) > max_chars:
                chunks.append(Chunk(source, heading, piece.strip()))
                piece = ""
            piece += paragraph + "\n\n"
        if piece.strip():
            chunks.append(Chunk(source, heading, piece.strip()))

    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            del trail[level - 1:]
            trail.append(match.group(2).strip())
        else:
            buffer.append(line)
    flush()
    return chunks


def load_chunks(directory: Path = DEFAULT_DIR) -> list[Chunk]:
    chunks = []
    if directory.is_dir():
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in (".md", ".txt"):
                chunks += chunk_markdown(path.read_text(encoding="utf-8"), path.name)
    return chunks


class Index:
    """BM25 over chunks. The heading is indexed with the body so section names count."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self._docs = [Counter(tokens(f"{c.heading} {c.heading} {c.text}")) for c in chunks]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg = (sum(self._lengths) / len(self._lengths)) if chunks else 0.0
        frequency = Counter(term for doc in self._docs for term in doc)
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in frequency.items()}

    def search(self, query: str, k: int = 4) -> list[tuple[float, Chunk]]:
        terms = [t for t in tokens(query) if t in self._idf]
        scored = []
        for doc, length, chunk in zip(self._docs, self._lengths, self.chunks):
            score = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self._avg)
                    score += self._idf[term] * tf * (self.k1 + 1) / norm
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda pair: -pair[0])
        return scored[:k]


class DocsTool:
    def __init__(self, llm, index: Index | None = None, k: int = 4):
        self.llm = llm
        self.index = index if index is not None else Index(load_chunks())
        self.k = k

    def answer(self, question: str) -> str:
        hits = self.index.search(question, self.k)
        if not hits:
            return NOT_FOUND
        passages = "\n\n".join(f"[{n}] {chunk.heading}\n{chunk.text}" for n, (_, chunk) in enumerate(hits, 1))
        system = (
            "Answer the question using only the numbered passages. If they do not contain the answer, "
            "set found to false. Do not use outside knowledge. Reply with a JSON object:\n"
            '{"found": true or false, "answer": string, "used": [passage numbers]}'
        )
        reply = self.llm.json(system, f"Question: {question}\n\nPassages:\n{passages}")
        text = reply.get("answer")
        if not reply.get("found") or not isinstance(text, str) or not text.strip():
            return NOT_FOUND
        used = [n for n in reply.get("used") or [] if isinstance(n, int) and 1 <= n <= len(hits)]
        if not used:
            # An answer that cites no passage cannot be traced, so it is not given.
            return NOT_FOUND
        sources = list(dict.fromkeys(hits[n - 1][1].heading for n in used))
        return f"{text.strip()}\n\nSources: " + "; ".join(sources)

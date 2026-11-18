"""Finding an existing item from a description: "the invoice to Acme".

When a plan needs the ID of something that already exists and the user did
not give it, the resolver:
  1. finds a read-only list or search operation for that kind of item
  2. runs it, with filters filled from the user's words
  3. asks the model which row the user means
  4. accepts the answer only if that ID is really in the rows returned

One clear match is used. Several plausible matches are put to the user. No
match, or no way to list that kind of item, falls back to asking for the ID.
A looked-up ID is always shown with a summary of the record before any call
that needs approval, so the user confirms the right item.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import compute, config, template
from .catalog import Catalog, Operation
from .execute import Executor
from .fill import Filled, Filler

FOUND, AMBIGUOUS, NONE, UNSUPPORTED = "found", "ambiguous", "none", "unsupported"
MAX_ROWS = config.LOOKUP_MAX_ROWS
MAX_FIELDS = 12     # fields per record shown to the model
SHOWN_FIELDS = 5    # fields per record shown to the user


@dataclass
class Resolution:
    status: str
    id: str | None = None
    summary: str = ""
    options: list[tuple[str, str]] = field(default_factory=list)  # (id, summary)
    via: str = ""          # name of the operation used to look
    looked_at: int = 0     # rows considered
    total: int = 0         # rows returned


class Resolver:
    def __init__(self, catalog: Catalog, llm, filler: Filler, executor: Executor):
        self.catalog = catalog
        self.llm = llm
        self.filler = filler
        self.executor = executor

    def lookup_ops(self, id_name: str) -> list[Operation]:
        """Read-only search and list operations for the kind of item this ID names. Search first."""
        owners = [op.resource for op in self.catalog if op.action == "show" and op.consumes == (id_name,)]
        if not owners:
            return []
        ops = [op for op in self.catalog.actions(owners[0])
               if op.risk == "read" and not op.consumes and op.action in ("search", "list")]
        return sorted(ops, key=lambda op: op.action != "search")

    def resolve(self, id_name: str, message: str) -> Resolution:
        ops = self.lookup_ops(id_name)
        if not ops:
            return Resolution(UNSUPPORTED)
        last = Resolution(NONE)
        label = id_name.removesuffix("_id").replace("_", " ")
        # The filler is asked to fill a search, not the action the user wants done with the result.
        task = (f"Find the existing {label} the user refers to in this request, so it can be acted on "
                f"later. Set only search filters the request clearly gives.\nRequest: {message}")
        for op in ops:
            try:
                filled = self.filler.fill(op, task)
            except Exception:
                filled = Filled()  # no filters is still a valid search; matching happens below
            filled.question = None
            result, _ = self.executor.call(op, {}, filled.query, filled.body(op, {}))
            if not result.ok:
                continue
            rows = self._rows(result.body, id_name)
            last = Resolution(NONE, via=op.name, total=len(rows))
            if rows:
                return self._pick(id_name, message, rows, op)
        return last

    def _rows(self, body, id_name: str) -> list[tuple[str, dict]]:
        """(id, row) pairs from the first collection whose rows carry an ID."""
        for rows in compute.collections(body).values():
            key = id_name if id_name in rows[0] else "id" if "id" in rows[0] else None
            if key:
                return [(str(row[key]), row) for row in rows if row.get(key)]
        return []

    def _pick(self, id_name: str, message: str, rows: list[tuple[str, dict]], op: Operation) -> Resolution:
        shown = rows[:MAX_ROWS]
        summaries = {row_id: _summary(row) for row_id, row in shown}
        base = dict(via=op.name, looked_at=len(shown), total=len(rows))
        listing = "\n".join(f"- {row_id}: {text}" for row_id, text in summaries.items())
        label = id_name.removesuffix("_id").replace("_", " ")
        system = (
            f"The user refers to an existing {label} without giving its ID. Below are the records found. "
            "Decide which one the user means, using only what the user said.\n"
            'Reply with a JSON object: {"match": id or null, "candidates": [ids]}\n'
            "- match: the ID of the single record the user clearly means, copied exactly.\n"
            "- If two or more records fit equally well, set match to null and list them in candidates.\n"
            "- If none fits, set match to null and candidates to [].\n"
            "- Never pick one just because it is first or most recent unless the user said so.\n\n"
            f"Records:\n{listing}"
        )
        reply = self.llm.json(system, message)
        short = {row_id: _summary(row, SHOWN_FIELDS) for row_id, row in shown}  # what the user sees
        match = reply.get("match")
        if isinstance(match, str) and match in summaries:
            return Resolution(FOUND, match, short[match], **base)
        candidates = [c for c in reply.get("candidates") or [] if isinstance(c, str) and c in summaries]
        if len(candidates) == 1:
            return Resolution(FOUND, candidates[0], short[candidates[0]], **base)
        if candidates:
            by_id = dict(shown)
            chosen = candidates[:5]
            differing = _differing_fields([by_id[c] for c in chosen])
            return Resolution(AMBIGUOUS, options=[(c, _summary(by_id[c], SHOWN_FIELDS, differing)) for c in chosen],
                              **base)
        return Resolution(NONE, **base)


# Fields people use to refer to a record come first in its summary.
_TELLING = ("status", "state", "name", "email", "number", "value", "currency", "amount", "reason",
            "description", "date", "time", "url", "type")


def _differing_fields(rows: list[dict]) -> set[str]:
    """Field paths whose value is not the same in every row: what tells the rows apart."""
    flat = [template.flatten(row) for row in rows]
    paths = set().union(*flat) if flat else set()
    return {p for p in paths if "links" not in p and p != "id" and len({str(f.get(p)) for f in flat}) > 1}


def _summary(row: dict, limit: int = MAX_FIELDS, only: set[str] | None = None) -> str:
    """A one-line description of a record, led by the fields people describe records by.

    only: if given, describe the record with these field paths alone.
    """
    fields, seen = [], set()
    for position, (path, value) in enumerate(template.flatten(row).items()):
        if "links" in path or value in (None, "") or path == "id":
            continue
        if only is not None and path not in only:
            continue
        key = path.split(".")[-1]
        text = f"{key}={str(value)[:40]}"
        if text in seen:
            continue  # the same value often repeats, e.g. a name in billing and shipping
        seen.add(text)
        rank = next((i for i, word in enumerate(_TELLING) if word in key.lower()), len(_TELLING))
        fields.append((rank, position, text))
    fields.sort()
    return ", ".join(text for _, _, text in fields[:limit])

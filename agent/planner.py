"""Plan a chain of calls by searching the ID graph backwards from a goal operation.

The planner is plain code. The caller (the selection step) supplies three facts
the planner cannot know from the graph alone:

  known     IDs already in hand (from the user or earlier in the conversation)
  new       IDs the user wants freshly created ("send an invoice" -> invoice_id)
  existing  IDs that refer to something that already exists but must be looked up

Any ID the goal needs that is not known and not marked new becomes a "resolve"
step: find the existing entity or ask the user. Creating things the user did
not ask for is never the default.
"""
from __future__ import annotations

from dataclasses import dataclass

from .catalog import Catalog


@dataclass(frozen=True)
class Step:
    kind: str    # "call" or "resolve"
    target: str  # operation id for a call, ID name for a resolve

    def __str__(self) -> str:
        return self.target if self.kind == "call" else f"resolve({self.target})"


class Planner:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog

    def plan(self, goal: str, known=(), new=(), existing=()) -> list[Step]:
        steps: list[Step] = []
        self._expand(goal, set(known), set(new), set(existing), steps, (), creating=False)
        return steps

    def _expand(self, op_id, have, new, existing, steps, stack, creating):
        op = self.catalog[op_id]
        for name in op.consumes:
            if name in have:
                continue
            # Inside a creation chain, upstream IDs are created too unless the
            # caller said they already exist.
            may_create = name in new or (creating and name not in existing)
            maker = self._best_creator(name, have, new, existing, stack + (op_id,)) if may_create else None
            if maker is None:
                steps.append(Step("resolve", name))
                have.add(name)
            else:
                self._expand(maker, have, new, existing, steps, stack + (op_id,), creating=True)
        steps.append(Step("call", op_id))
        have.update(n for n in op.produces if not self.catalog.is_conditional(op_id, n))

    def _best_creator(self, name, have, new, existing, stack):
        """Reliable creators beat conditional ones; among those, the shortest chain wins."""
        best, best_key = None, None
        for op in self.catalog.creators(name):
            if op.id in stack:
                continue
            trial: list[Step] = []
            self._expand(op.id, set(have), new, existing, trial, stack, creating=True)
            key = (self.catalog.is_conditional(op.id, name), len(trial))
            if best_key is None or key < best_key:
                best, best_key = op.id, key
        return best

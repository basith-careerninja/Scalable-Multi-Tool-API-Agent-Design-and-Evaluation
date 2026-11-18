"""Eval control: every tool in one prompt.

This is the approach the task document says breaks down as tools are added: the model
sees the whole catalogue (110 operations) and picks one. It is not part of the agent.
It asks for the same fields and applies the same checks as agent.select.Selector, so
the only difference measured is how the choice is presented to the model.
"""
from __future__ import annotations

from agent.catalog import Catalog
from agent.select import FALLBACK_QUESTION, Selection

INTENTS = ("action", "docs", "system", "clarify")


class AllToolsSelector:
    def __init__(self, catalog: Catalog, llm, default_service: str | None = None):
        self.catalog = catalog
        self.llm = llm
        self.default_service = default_service
        self._system = self._prompt()

    def select(self, message: str, remembered: dict | None = None) -> Selection:
        user, calls = message, 0
        if remembered:
            known = ", ".join(f"{k}={v}" for k, v in remembered.items())
            user = f"{message}\n(Remembered from earlier in this conversation: {known}.)"
        reply = None
        for _ in range(2):
            calls += 1
            reply = self.llm.json(self._system, user)
            error = self._check(reply)
            if error is None:
                break
            user = f"{message}\n\nYour previous answer was rejected: {error}. Reply again with a valid JSON object."
        else:
            return Selection("clarify", question=FALLBACK_QUESTION, llm_calls=calls, decided_by="all-tools")
        intent = reply["intent"]
        if intent != "action":
            question = reply.get("question") or FALLBACK_QUESTION if intent == "clarify" else None
            return Selection(intent, question=question, llm_calls=calls, decided_by="all-tools")
        op = self.catalog[reply["operation"]]
        new_ids = tuple(n for n in reply.get("new_ids") or () if n in op.consumes and self.catalog.creators(n))
        stated = reply.get("known_ids") if isinstance(reply.get("known_ids"), dict) else {}
        known_ids = tuple((name, str(value)) for name, value in stated.items()
                          if name in op.consumes and name not in new_ids and value and str(value) in message
                          and self.catalog.looks_like_id(name, str(value)))
        calculate = reply.get("calculate") is True and op.method == "GET"
        return Selection("action", op.id, new_ids, known_ids, 1.0, llm_calls=calls, calculate=calculate,
                         decided_by="all-tools")

    def _check(self, reply: dict) -> str | None:
        if reply.get("intent") not in INTENTS:
            return f"intent must be one of {list(INTENTS)}"
        if reply["intent"] == "action":
            op = reply.get("operation")
            if not op or op not in {o.id for o in self.catalog}:
                return f"operation {op!r} does not exist; use one of the listed ids exactly"
        return None

    def _prompt(self) -> str:
        lines = [f"- {op.id}: {op.name} (needs: {', '.join(op.consumes) or 'nothing'})" for op in self.catalog]
        note = ""
        if self.default_service and len(self.catalog.services()) > 1:
            title = self.catalog.title(self.default_service)
            note = (f"The user's main account is {title}. A request that does not name a service is for {title} if "
                    f"{title} can do it; another service only when the user names it or only that service can do "
                    "it.\n\n")
        return note + (
            "You route requests for an assistant that operates business APIs. Reply with a JSON object:\n"
            '{"intent": "action" | "docs" | "system" | "clarify", "operation": string or null, '
            '"new_ids": [string], "known_ids": {name: value}, "calculate": true or false, '
            '"question": string or null}\n\n'
            "- action: the user wants something done or looked up through the APIs. Pick the single operation "
            "that completes the goal; if it takes several steps, pick the last one.\n"
            "- docs: the user asks how something works in general.\n"
            "- system: the user asks what this assistant can do, or about their earlier requests.\n"
            "- clarify: too vague to act on; put one short question in question.\n"
            "- new_ids: from the operation's needs, IDs of things the request creates (described by contents, "
            "not pointed at).\n"
            "- known_ids: IDs from the needs whose exact value appears in the message. Never guess.\n"
            "- calculate: true when the answer needs a total, count, average or a filtered subset.\n\n"
            f"Operations ({len(lines)}):\n" + "\n".join(lines)
        )

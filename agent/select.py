"""Selection: turn a chat message into one catalogue operation.

Two small decisions instead of one large one:
  1. intent, and which resource the request is about
  2. which action on that resource, and which IDs the user wants newly created

Every answer is checked against the catalogue. A resource or operation that
does not exist is rejected and the model gets one retry with the error; after
that the agent asks the user instead of guessing.

The confidence values are the model's own estimate and are not calibrated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from . import config
from .catalog import Catalog

INTENTS = ("action", "docs", "system", "clarify")
FALLBACK_QUESTION = "I could not match that to something I can do. Could you rephrase or add detail?"


class LLM(Protocol):
    def json(self, system: str, user: str) -> dict: ...


@dataclass(frozen=True)
class Selection:
    intent: str
    operation: str | None = None
    new_ids: tuple[str, ...] = ()
    known_ids: tuple[tuple[str, str], ...] = ()  # (id name, value) pairs the user stated
    confidence: float = 0.0
    question: str | None = None
    llm_calls: int = 0
    calculate: bool = False  # the user wants a total, count or filtered subset, not the raw list
    decided_by: str = "llm"  # "jev", "llm", or "llm-fallback" when Jev handed over
    reason: str = ""         # why a fallback or question happened, for tracing
    scores: tuple[tuple[str, float], ...] = ()  # top candidate tools and their scores (Jev only)
    # For a system question decided by Jev: {"kind": {capabilities|history: p}, "resources": {resource: p}}
    system_hint: dict | None = None


class Selector:
    def __init__(self, catalog: Catalog, llm: LLM, min_confidence: float = config.LLM_MIN_CONFIDENCE,
                 default_service: str | None = None):
        self.catalog = catalog
        self.default_service = default_service
        self.multi = len(catalog.services()) > 1
        self.llm = llm
        self.min_confidence = min_confidence
        self._calls = 0

    def select(self, message: str, remembered: dict[str, str] | None = None) -> Selection:
        """remembered: IDs from earlier in the conversation, so 'that invoice' needs no question."""
        self._calls = 0
        original = message
        if remembered:
            known = ", ".join(f"{k}={v}" for k, v in remembered.items())
            message = (f"{message}\n(Remembered from earlier in this conversation: {known}. Words like 'it' or "
                       "'that invoice' refer to these, and they are supplied automatically: do not ask for them.)")
        routed = self._ask(self._route_prompt(), message, self._check_route)
        if routed is None:
            return self._clarify(FALLBACK_QUESTION)
        intent, confidence = routed["intent"], _confidence(routed)
        if intent == "clarify":
            return self._clarify(routed.get("question") or FALLBACK_QUESTION)
        if intent != "action":
            return Selection(intent, confidence=confidence, llm_calls=self._calls)
        if self.multi:
            # With several vendors the route names the service; the resource is chosen within it.
            service = routed["service"]
            routed = self._ask(self._resource_prompt(service), message,
                               lambda r: self._check_resource(service, r))
            if routed is None:
                return self._clarify(FALLBACK_QUESTION)

        # A wrong resource must not be a dead end: if nothing fits under the first
        # choice, try the model's second choice before asking the user.
        candidates = [routed["resource"]]
        alternative = routed.get("alternative")
        if alternative in self.catalog.resources() and alternative != routed["resource"]:
            candidates.append(alternative)
        chosen = None
        for resource in candidates:
            chosen = self._ask(self._action_prompt(resource), message, lambda r: self._check_action(resource, r))
            if chosen is not None and chosen["operation"] is not None:
                break
        if chosen is None:
            return self._clarify(FALLBACK_QUESTION)
        if chosen["operation"] is None:
            return self._clarify(chosen.get("question") or FALLBACK_QUESTION)
        confidence = min(confidence, _confidence(chosen))
        question = chosen.get("question") or ""
        # A missing ID is not a reason to stop: existing items are looked up from the description.
        only_wants_id = bool(re.search(r"\bid\b|_id\b", question, re.I))
        if confidence < self.min_confidence and not only_wants_id:
            name = self.catalog[chosen["operation"]].name
            return self._clarify(question or f"Do you want me to: {name}?")
        op = self.catalog[chosen["operation"]]
        new_ids = tuple(n for n in chosen.get("new_ids") or () if n in op.consumes and self.catalog.creators(n))
        # An ID is accepted only if the user actually typed it, so the model cannot invent one.
        stated = chosen.get("known_ids") if isinstance(chosen.get("known_ids"), dict) else {}
        known_ids = tuple(
            (name, str(value)) for name, value in stated.items()
            if name in op.consumes and name not in new_ids and value and str(value) in original
            and self.catalog.looks_like_id(name, str(value))
        )
        calculate = chosen.get("calculate") is True and op.method == "GET"
        return Selection("action", op.id, new_ids, known_ids, confidence, llm_calls=self._calls, calculate=calculate)

    # -- prompts ---------------------------------------------------------------

    def _service_note(self) -> str:
        if not (self.multi and self.default_service):
            return ""
        title = self.catalog.title(self.default_service)
        return (f"The user's main account is {title}. A request that does not name a service is for {title} if "
                f"{title} can do it; another service only when the user names it or only that service can do it.\n\n")

    def _resource_prompt(self, service: str) -> str:
        lines = [f"- {r}: " + ", ".join(op.name for op in self.catalog.actions(r))[:400]
                 for r in self.catalog.resources(service)]
        return (
            f"The user's request is for the {self.catalog.title(service)} API. Choose the resource that has the "
            "operation doing what the user wants, by what can be done with it, not by its name alone. Reply with a "
            'JSON object: {"resource": string, "alternative": string or null, "confidence": number from 0 to 1}.\n'
            "Set alternative to the second most likely resource, or null. resource must be exactly one of these "
            "names. Each is followed by what can be done with it:\n" + "\n".join(lines)
        )

    def _check_resource(self, service: str, reply: dict) -> str | None:
        if reply.get("resource") not in self.catalog.resources(service):
            return f"resource {reply.get('resource')!r} does not exist; use one of the listed names exactly"
        return None

    def _route_prompt(self) -> str:
        lines = []
        if self.multi:
            for service in self.catalog.services():
                names = ", ".join(r.split(".", 1)[-1] for r in self.catalog.resources(service))
                lines.append(f"- {service}: {self.catalog.title(service)} API: {names}")
            target = ('"service": string or null', "Set service.",
                      "For action, choose the service whose API does what the user wants. Set alternative to the "
                      "second most likely service, or null.\nservice must be exactly one of these names. Each is "
                      "followed by what it covers:\n")
        else:
            for resource in self.catalog.resources():
                actions = ", ".join(op.name for op in self.catalog.actions(resource))
                lines.append(f"- {resource}: {actions}")
            target = ('"resource": string or null', "Set resource.",
                      "For action, choose the resource by what can be done with it, not by its name alone: the thing "
                      "the user wants done must appear in that resource's list. Set alternative to the second most "
                      "likely resource, or null.\nresource must be exactly one of these names. Each is followed by "
                      "what can be done with it:\n")
        return self._service_note() + (
            "You route requests for an assistant that operates business APIs.\n"
            "Classify the user's message and reply with a JSON object:\n"
            '{"intent": "action" | "docs" | "system" | "clarify", ' + target[0] + ', '
            '"alternative": string or null, "confidence": number from 0 to 1, "question": string or null}\n\n'
            "intent meanings:\n"
            "- action: the user wants something done or looked up through the APIs. " + target[1] + "\n"
            "- docs: the user asks how a product or process works in general, e.g. 'how do refunds work', "
            "'what is a dispute'.\n"
            "- system: the user asks about this assistant itself: which tools or actions it has, e.g. 'what "
            "tools are available for managing invoices', 'what can you do'; or about their own earlier "
            "requests, e.g. 'what is the status of my last request'.\n"
            "- clarify: too vague to act on. Put one short question in question. Never use clarify to ask "
            "for an ID: items the user describes are looked up automatically. Prefer action when a "
            "reasonable reading exists; a lookup can always be run and refined.\n\n"
            "Telling them apart:\n"
            "- A request to do or fetch something specific now is action: 'send the invoice', 'list my disputes'.\n"
            "- 'How do I ...', 'how does ... work', 'what is ...', 'what happens when ...' ask for an "
            "explanation and are docs, even when they mention a resource: 'how do I move an invoice from "
            "draft to payable' is docs.\n"
            "- 'What tools/actions/operations are available for X', 'what can you do with X' ask about this "
            "assistant's own abilities and are system, never docs.\n\n"
            + target[2] + "\n".join(lines)
        )

    def _action_prompt(self, resource: str) -> str:
        lines = []
        for op in self.catalog.actions(resource):
            needs = ", ".join(op.consumes) or "nothing"
            lines.append(f"- {op.id}: {op.name} (needs: {needs})")
        return (
            f"The user's request is about the resource '{resource}'.\n"
            "Pick the single operation that completes the user's goal. If the goal takes several steps, "
            "pick the last one; earlier steps are planned automatically.\n"
            "Reply with a JSON object:\n"
            '{"operation": string or null, "new_ids": [string], "known_ids": {name: value}, '
            '"calculate": true or false, "confidence": number from 0 to 1, "question": string or null}\n\n'
            "- operation must be exactly one of the ids below, or null if none fits.\n"
            "- new_ids: from the operation's 'needs' list, the IDs of things that do not exist yet and must be "
            "created as part of this request. When the user describes the thing by its contents (an amount, "
            "a recipient, items) instead of pointing at one that exists, it is new: 'send an invoice for $50 "
            "to sam@example.com' means a new invoice, so new_ids is [\"invoice_id\"]. When the user points at "
            "an existing one ('the invoice', 'that subscription', 'invoice INV2-ABCD'), leave new_ids empty.\n"
            "- known_ids: from the 'needs' list, any ID whose exact value appears in the user's message, "
            "copied character for character. Only the ID itself counts: an invoice number, a name, an email "
            "or an amount is not an ID. Never guess an ID.\n"
            "- A missing ID is never a reason to ask or to lower confidence. When the user describes an "
            "existing item without its ID ('the refunded invoice', 'the template named Hours'), choose the "
            "operation normally and leave known_ids empty: the item is looked up automatically.\n"
            "- calculate: true when the answer needs a total, sum, count, average, or a subset picked by a "
            "condition (e.g. 'total sales volume', 'how many', 'is there an open dispute from X'). No API "
            "returns such a figure directly: pick the operation that lists the records and set calculate "
            "to true. False when the user just wants the records or a single item, or a figure the API "
            "already returns as it is (an account balance).\n"
            "- question: one short question for the user if you are unsure.\n\n"
            "Operations:\n" + "\n".join(lines)
        )

    # -- validation ------------------------------------------------------------

    def _check_route(self, reply: dict) -> str | None:
        if reply.get("intent") not in INTENTS:
            return f"intent must be one of {list(INTENTS)}"
        if reply["intent"] == "action" and self.multi:
            if reply.get("service") not in self.catalog.services():
                return f"service {reply.get('service')!r} does not exist; use one of the listed names exactly"
        elif reply["intent"] == "action" and reply.get("resource") not in self.catalog.resources():
            return f"resource {reply.get('resource')!r} does not exist; use one of the listed names exactly"
        return None

    def _check_action(self, resource: str, reply: dict) -> str | None:
        valid = {op.id for op in self.catalog.actions(resource)}
        if "operation" not in reply:
            return "the reply must have an 'operation' key"
        if reply["operation"] is not None and reply["operation"] not in valid:
            return f"operation {reply['operation']!r} does not exist; use one of the listed ids exactly, or null"
        return None

    def _ask(self, system: str, message: str, check) -> dict | None:
        """Ask once; on an invalid answer, retry once with the error. None if still invalid."""
        user = message
        for _ in range(2):
            self._calls += 1
            reply = self.llm.json(system, user)
            error = check(reply)
            if error is None:
                return reply
            user = f"{message}\n\nYour previous answer was rejected: {error}. Reply again with a valid JSON object."
        return None

    def _clarify(self, question: str) -> Selection:
        return Selection("clarify", question=question, llm_calls=self._calls)


def _confidence(reply: dict) -> float:
    try:
        return max(0.0, min(1.0, float(reply.get("confidence", 0))))
    except (TypeError, ValueError):
        return 0.0

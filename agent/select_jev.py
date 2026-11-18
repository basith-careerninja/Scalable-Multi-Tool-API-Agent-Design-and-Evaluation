"""Selection with Jev: probabilities over the real options instead of a generated answer.

Two Jev requests per action, each a set of questions answered together:

  1. route    intent (action / docs / system / clarify) and resource
  2. choose   for each likely resource, which of its operations; plus yes/no
              questions for "is something new being created" and "does this
              need a calculation"

A tool's score is P(resource) x P(operation | resource). Because Jev only
scores the options it is given, it cannot name a tool that does not exist.

Decisions are made in code from the probabilities:
  - near tie between two tools, and either one changes something  -> ask the user which
  - best tool above its risk tier's threshold                     -> act
  - otherwise                                                      -> hand over to the fallback
                                                                     (the generative selector), or ask

Jev cannot write text, so a typed ID is found in code (agent.select_jev.typed_ids)
and questions to the user are built from tool names.

The thresholds come from agent/config.py. They are starting values, not measured
ones, and are meant to be tuned on an evaluation set.
"""
from __future__ import annotations

import re
from dataclasses import replace

from . import config
from .catalog import Catalog, Operation, id_label
from .jev import JevError, probabilities, yes
from .select import Selection

ACT_THRESHOLD = {"read": config.ACT_THRESHOLD_READ}
DEFAULT_ACT_THRESHOLD = config.ACT_THRESHOLD
CLOSE_GAP = config.CLOSE_GAP
TIE_MIN_RUNNER_UP = config.TIE_MIN_RUNNER_UP
INTENT_THRESHOLD = config.INTENT_THRESHOLD
KEEP_RESOURCE = config.KEEP_RESOURCE
MAX_RESOURCES = config.MAX_RESOURCES
KEEP_SERVICE = config.KEEP_SERVICE
MAX_SERVICES = config.MAX_SERVICES
YES_THRESHOLD = config.YES_THRESHOLD
GENERIC_QUESTION = "Could you say a bit more about what you would like me to do?"

INTENTS = {
    "action": {
        "covers": "Doing or fetching something specific now through the business APIs: create, send, "
                  "list, show, update, cancel, refund, or a total or count over the user's own data. "
                  "Includes listing what PayPal itself offers when an operation returns it (e.g. the "
                  "event types a webhook can subscribe to).",
        "examples": ["Send an invoice for $50 to a@b.com", "Show my disputes", "How much did I sell in May?"],
    },
    "docs": {
        "covers": "A question about how a product, process or concept works in general.",
        "not_for": "Questions about this assistant's own tools, requests to do something, and requests "
                   "to list or show something an operation returns.",
        "examples": ["How do refunds work?", "What is a chargeback?", "What happens when a plan is deactivated?"],
    },
    "system": {
        "covers": "A question about this assistant itself: which tools, actions or operations it has and "
                  "what it can do, or the status of the user's own earlier requests in this chat.",
        "not_for": "General how-to questions about the product.",
        "examples": ["Which operations do you support for payouts?", "Can you list what you can do with "
                     "subscriptions?", "Did my previous request go through?"],
    },
    "clarify": {
        "covers": "Too vague to act on, even with what was said earlier in this conversation.",
        "examples": ["do the thing", "fix it", "hmm"],
    },
}

SYSTEM_KINDS = {
    "capabilities": "What this assistant can do: which tools, actions or operations it has, for which kind of item.",
    "history": "The user's own earlier requests in this chat: whether they ran, finished or failed.",
}

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{4,}[A-Za-z0-9]")


class JevSelector:
    def __init__(self, catalog: Catalog, jev, fallback=None, tracer=None, id_llm=None, default_service=None):
        self.catalog = catalog
        self.default_service = default_service  # with several vendors: the one a request means unless it names another
        self.jev = jev
        self.fallback = fallback
        self.tracer = tracer
        self.id_llm = id_llm  # optional LLM for typed IDs the regex cannot read
        self._calls = 0

    # -- entry point -------------------------------------------------------------

    def select(self, message: str, remembered: dict[str, str] | None = None) -> Selection:
        self._calls = 0
        try:
            return self._select(message, remembered or {})
        except JevError as err:
            return self._hand_over(message, remembered, f"jev error: {err}"[:200])

    def _select(self, message: str, remembered: dict[str, str]) -> Selection:
        state = {"user_message": message}
        if remembered:
            state["ids_from_earlier_in_this_chat"] = remembered
            state["note"] = "Words like 'it' or 'that invoice' refer to the items above."
        services = self.catalog.services()
        multi = len(services) > 1
        if multi and self.default_service:
            title = self.catalog.title(self.default_service)
            state["main_service"] = (f"The user's main account is {title}. A request that does not name a service "
                                     f"is for {title} if {title} can do it; another service only when the user "
                                     f"names it or only that service can do it.")

        # Step 1: intent and resource, in one request. With several vendors, intent and service:
        # the resource is chosen next, among the likely services only, so no question lists every tool.
        resources = self.catalog.resources()
        questions = {"intent": {"type": "choice", "instructions": "What kind of request is the user's message?",
                                "criteria": INTENTS}}
        if multi:
            questions["service"] = {"type": "choice",
                                    "instructions": "If the user wants something done or fetched, which service's "
                                                    "API does it?",
                                    "criteria": {sv: self._service_card(sv) for sv in services}}
        else:
            questions["resource"] = {"type": "choice",
                                     "instructions": "If the user wants something done or fetched, which resource has "
                                                     "the operation that does it? Choose by the operations listed, not "
                                                     "by the resource name alone.",
                                     "criteria": {r: "; ".join(op.name for op in self.catalog.actions(r))
                                                  for r in resources}}
        # Answered with the others, so a system question needs no second request.
        questions["system_kind"] = {"type": "choice",
                                    "instructions": "If the message asks about this assistant itself, what is it about?",
                                    "criteria": SYSTEM_KINDS}
        answers = self._ask("route", state, questions)
        intent_p = probabilities(answers["intent"], INTENTS)
        intent, p_intent = _top(intent_p)
        if p_intent < INTENT_THRESHOLD:
            return self._hand_over(message, remembered, f"intent unclear: {_fmt(intent_p)}")
        if intent == "clarify":
            return self._hand_over(message, remembered, "jev judged the request too vague",
                                   ask=GENERIC_QUESTION)
        if intent == "docs":
            return Selection(intent, confidence=p_intent, llm_calls=self._calls, decided_by="jev")
        if multi:
            resource_p = self._resources_by_service(state, probabilities(answers["service"], services))
        else:
            resource_p = probabilities(answers["resource"], resources)
        if intent == "system":
            hint = {"kind": probabilities(answers.get("system_kind") or {}, SYSTEM_KINDS), "resources": resource_p}
            return Selection(intent, confidence=p_intent, llm_calls=self._calls, decided_by="jev",
                             system_hint=hint)

        ranked = sorted(resource_p.items(), key=lambda kv: -kv[1])
        if not ranked:
            return self._hand_over(message, remembered, "jev returned no resource probabilities")
        kept = [r for r, p in ranked if p >= KEEP_RESOURCE][:MAX_RESOURCES] or [ranked[0][0]]

        # Step 2: operation per kept resource, plus yes/no questions asked speculatively.
        questions, op_question = {}, {}
        for n, resource in enumerate(kept):
            ops = self.catalog.actions(resource)
            if len(ops) > 1:
                op_question[resource] = f"operation_{n}"
                questions[f"operation_{n}"] = {
                    "type": "choice",
                    "instructions": "Which operation completes the user's goal? If the goal takes several "
                                    "steps, choose the last one; earlier steps are planned automatically.",
                    "criteria": {op.id: _describe(op) for op in ops},
                }
        creatable = sorted({name for r in kept for op in self.catalog.actions(r) for name in op.consumes
                            if self.catalog.creators(name)})
        for n, name in enumerate(creatable):
            label = id_label(name)
            questions[f"new_{n}"] = {
                "type": "noul",
                "instructions": f"Does this request itself create the {label} it acts on (for example 'record a "
                                f"payment and then delete it', 'create a product and show it'), rather than act on "
                                f"a {label} that already exists? Describing the {label} by its contents (an amount, "
                                f"a recipient, items) means it is new; pointing at an existing {label} ('the "
                                f"{label}', 'that {label}', its ID) means existing. An ID of some other item does "
                                f"not make this {label} existing.",
            }
        questions["calculate"] = {
            "type": "noul",
            "instructions": "Does answering need a total, sum, count or average, or picking records by a "
                            "condition, rather than just showing records or a single item? Showing a figure "
                            "the API already returns as it is (an account balance, one record's amount) is "
                            "not a calculation.",
        }
        answers = self._ask("choose", state, questions)

        scored: list[tuple[float, Operation]] = []
        for resource in kept:
            ops = self.catalog.actions(resource)
            if resource in op_question:
                op_p = probabilities(answers[op_question[resource]], [op.id for op in ops])
            else:
                op_p = {ops[0].id: 1.0}
            for op in ops:
                scored.append((resource_p.get(resource, 0.0) * op_p.get(op.id, 0.0), op))
        scored.sort(key=lambda pair: -pair[0])
        scores = tuple((op.id, round(score, 3)) for score, op in scored[:3])
        reason = None
        if yes(answers["calculate"]) >= YES_THRESHOLD:
            swapped = _list_for_a_total(self.catalog, scored, message, remembered)
            if swapped:
                scored, reason = swapped, f"a total needs a list: {scored[0][1].id} reads one record by ID"
        (best_score, best), second = scored[0], (scored[1] if len(scored) > 1 else (0.0, None))
        second_score, runner_up = second

        # A near tie only matters if getting it wrong would change something.
        if runner_up and best_score - second_score < CLOSE_GAP and second_score > TIE_MIN_RUNNER_UP \
                and not (best.risk == "read" and runner_up.risk == "read"):
            question = f"Do you want me to: {best.name}, or {runner_up.name}?"
            return Selection("clarify", question=question, confidence=best_score, llm_calls=self._calls,
                             decided_by="jev", reason="near tie", scores=scores)
        if best_score < ACT_THRESHOLD.get(best.risk, DEFAULT_ACT_THRESHOLD):
            return self._hand_over(message, remembered, f"best tool {best.id} only {best_score:.2f}",
                                   ask=f"Do you want me to: {best.name}?", scores=scores)

        new_ids = tuple(name for n, name in enumerate(creatable)
                        if name in best.consumes and yes(answers[f"new_{n}"]) >= YES_THRESHOLD)
        calculate = best.method == "GET" and yes(answers["calculate"]) >= YES_THRESHOLD
        known = typed_ids(self.catalog, best, message, new_ids)
        known += self._model_typed_ids(best, message, new_ids, known)
        return Selection("action", best.id, new_ids, known, best_score, llm_calls=self._calls,
                         calculate=calculate, decided_by="jev", reason=reason, scores=scores)

    def _service_card(self, service: str) -> str:
        """What a service covers, for the service question: its name and its resources."""
        names = [r.split(".", 1)[-1].replace("-", " ") for r in self.catalog.resources(service)]
        return f"{self.catalog.title(service)} API: " + ", ".join(names)

    def _resources_by_service(self, state: dict, service_p: dict[str, float]) -> dict[str, float]:
        """Step 1b with several vendors: the resource within each likely service.
        P(resource) = P(service) x P(resource | service)."""
        ranked = sorted(service_p.items(), key=lambda kv: -kv[1])
        kept = [sv for sv, p in ranked if p >= KEEP_SERVICE][:MAX_SERVICES] or [ranked[0][0]]
        questions = {}
        for n, service in enumerate(kept):
            questions[f"resource_{n}"] = {
                "type": "choice",
                "instructions": f"Within {self.catalog.title(service)}, which resource has the operation that does "
                                "what the user wants? Choose by the operations listed, not by the resource name alone.",
                "criteria": {r: "; ".join(op.name for op in self.catalog.actions(r))[:400]
                             for r in self.catalog.resources(service)},
            }
        answers = self._ask("resource", state, questions)
        resource_p = {}
        for n, service in enumerate(kept):
            for r, p in probabilities(answers[f"resource_{n}"], self.catalog.resources(service)).items():
                resource_p[r] = service_p[service] * p
        return resource_p

    def _model_typed_ids(self, op: Operation, message: str, new_ids, known) -> tuple[tuple[str, str], ...]:
        """IDs the regex missed (e.g. short ones like PP-D-123), read by the LLM and checked in code.

        Called only when an ID is still needed and the message has an ID-like token the regex did
        not already use, so most actions cost no extra call.
        """
        if self.id_llm is None:
            return ()
        filled = dict(known)
        missing = [n for n in op.consumes if n not in new_ids and n not in filled]
        unused = [t for t in id_like_tokens(message) if t not in filled.values()]
        # An ID with no known format (no examples, e.g. a repository or organisation name) can be a
        # plain word or number, so the token test cannot rule it out; the code check below still applies.
        free_form = any(not self.catalog.example_ids(n) for n in missing) and bool(re.search(r"\w[-/_]\w|\d", message))
        if not missing or not (unused or free_form):
            return ()
        self._calls += 1
        try:
            reply = self.id_llm.json(_ID_PROMPT, _id_request(self.catalog, message, missing))
        except Exception:  # the regex result stands; an extraction failure never blocks the request
            return ()
        stated = reply.get("ids") if isinstance(reply, dict) else None
        return check_typed_ids(self.catalog, message, stated if isinstance(stated, dict) else {},
                               missing, set(filled.values()))

    # -- helpers -----------------------------------------------------------------

    def _hand_over(self, message, remembered, reason: str, ask: str = GENERIC_QUESTION,
                   scores: tuple = ()) -> Selection:
        """Let the generative selector decide; without one, ask the user."""
        if self.fallback is None:
            return Selection("clarify", question=ask, llm_calls=self._calls, decided_by="jev",
                             reason=reason, scores=scores)
        chosen = self.fallback.select(message, remembered)
        return replace(chosen, llm_calls=chosen.llm_calls + self._calls, decided_by="llm-fallback",
                       reason=reason, scores=scores or chosen.scores)

    def _ask(self, step: str, state: dict, questions: dict) -> dict:
        self._calls += 1
        if self.tracer is None:
            return self.jev.ask(state, questions)
        with self.tracer.span("jev", kind="llm", model=self.jev.model, step=step,
                              user=state.get("user_message")) as span:
            answers = self.jev.ask(state, questions)
            usage = getattr(self.jev, "last_usage", None) or {}
            span.set(prompt_tokens=usage.get("input_tokens"), completion_tokens=usage.get("output_tokens"),
                     response=_summarise(answers))
            return answers


def typed_ids(catalog: Catalog, op: Operation, message: str, new_ids=()) -> tuple[tuple[str, str], ...]:
    """IDs the user typed, found in code. Jev cannot copy text, so this does not use a model.

    An ID kind with a known prefix (INV2-, I-, TEMP-) must match it. One without a prefix
    must look like a PayPal ID: at least 10 characters, upper case, letters and digits.
    A value is used only when exactly one token fits, so it is never guessed.
    """
    tokens = [t for t in _TOKEN.findall(message) if re.search(r"\d", t)]
    found, used = [], set()
    for name in op.consumes:
        if name in new_ids:
            continue
        if catalog.id_prefixes(name):
            candidates = [t for t in tokens if catalog.looks_like_id(name, t)]
        else:
            candidates = [t for t in tokens if len(t) >= 10 and t.upper() == t and re.search(r"[A-Z]", t)]
        candidates = [t for t in candidates if t not in used]
        if len(candidates) == 1:
            found.append((name, candidates[0]))
            used.add(candidates[0])
    return tuple(found)


_ID_PROMPT = (
    "Find IDs the user typed in their message. Reply with a JSON object: "
    '{"ids": {id name: value or null}}.\n'
    "- Copy each value exactly as typed, character for character. Never complete, fix or guess one.\n"
    "- Give a value only if the message presents it as that kind of item: 'order 5T7' is an order ID, "
    "not a capture ID.\n"
    "- An invoice number, an email, an amount, a date or a name is not an ID. Use null when unsure."
)
_ID_LIKE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")


def id_like_tokens(message: str) -> list[str]:
    """Tokens that could be IDs: letters and digits mixed, or a hyphenated code (PP-D-123)."""
    found = []
    for token in _ID_LIKE.findall(message):
        mixed = bool(re.search(r"\d", token)) and bool(re.search(r"[A-Za-z]", token))
        code = "-" in token and bool(re.search(r"[A-Z0-9]", token))
        if mixed or code:
            found.append(token)
    return found


def _id_request(catalog: Catalog, message: str, names: list[str]) -> str:
    lines = []
    for name in names:
        prefixes = sorted(catalog.id_prefixes(name))
        hint = f" (starts with {' or '.join(p + '-' for p in prefixes)})" if prefixes else ""
        if not catalog.example_ids(name):
            # No known format: APIs like GitHub's identify an item by its name or number.
            lines.append(f"- {name}: what identifies the {id_label(name)} in the API path; can be a name, number "
                         f"or code the user typed for that {id_label(name)}")
        else:
            lines.append(f"- {name}: the ID of a {id_label(name)}{hint}")
    return f"Message: {message}\n\nID names:\n" + "\n".join(lines)


def _all_labels(catalog: Catalog) -> dict[str, str]:
    """ID name -> its label ("invoice template" for invoice_template_id)."""
    return {name: id_label(name) for op in catalog for name in op.consumes + op.produces}


def named_as(catalog: Catalog, message: str, value: str) -> str | None:
    """The kind of ID the message calls this value, from the words just before it ('order 5T7' -> order_id)."""
    index = message.find(value)
    before = re.sub(r"[\s#:]+$", "", message[:index]).lower() if index >= 0 else ""
    best = None
    for name, label in _all_labels(catalog).items():
        if re.search(rf"(?:^|[^a-z]){re.escape(label)}$", before) and (best is None or len(label) > len(best[1])):
            best = (name, label)
    return best[0] if best else None


def check_typed_ids(catalog: Catalog, message: str, stated: dict, wanted: list[str],
                    used: set[str]) -> tuple[tuple[str, str], ...]:
    """Keep a model-read ID only if the code can confirm it: typed verbatim as a whole token, the
    right prefix when one is known, not already used, and not named as another kind of item."""
    found, used = [], set(used)
    for name in wanted:
        value = stated.get(name)
        if not isinstance(value, str) or not value.strip():
            continue
        value = value.strip()
        if not re.search(rf"(?<![A-Za-z0-9-]){re.escape(value)}(?![A-Za-z0-9-])", message):
            continue
        if value in used or not catalog.looks_like_id(name, value):
            continue
        kind = named_as(catalog, message, value)
        if kind is not None and kind != name:
            continue
        found.append((name, value))
        used.add(value)
    return tuple(found)


def _top(probs: dict[str, float]) -> tuple[str, float]:
    return max(probs.items(), key=lambda kv: kv[1]) if probs else ("clarify", 0.0)


def _describe(op: Operation) -> str:
    return f"{op.name}. {op.description[:160]}".strip()


def _fmt(probs: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.2f}" for k, v in sorted(probs.items(), key=lambda kv: -kv[1])[:3])


def _summarise(answers: dict) -> dict:
    """Top three options per choice and the yes probability per noul, for traces."""
    out = {}
    for name, answer in answers.items():
        if answer.get("type") == "noul":
            out[name] = round(yes(answer), 3)
        else:
            probs = answer.get("probabilities") or {}
            out[name] = {k: round(v, 3) for k, v in sorted(probs.items(), key=lambda kv: -kv[1])[:3]}
    return out


def _list_for_a_total(catalog, scored, message: str, remembered: dict):
    """A total, count or filter is computed over a list. If the top choice reads one record by an ID
    the user neither typed nor has from earlier, and a list read is among the candidates, the list
    is the only operation that can answer; it takes the top choice's place (and its score, since
    code ruled the other out). Returns the reordered scores, or None if nothing changes."""
    (top_score, top), rest = scored[0], scored[1:]
    if top.method != "GET" or not top.consumes:
        return None
    typed = {name for name, _ in typed_ids(catalog, top, message, ())}
    if any(name in typed or name in (remembered or {}) for name in top.consumes):
        return None
    for score, op in rest:
        if op.method == "GET" and not op.consumes and score >= TIE_MIN_RUNNER_UP:
            others = [pair for pair in scored if pair[1] is not op]
            return [(top_score, op)] + others
    return None

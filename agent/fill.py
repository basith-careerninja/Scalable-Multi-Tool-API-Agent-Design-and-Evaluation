"""Parameter filling: slots in a template, not free-form bodies.

The model sees the fields an operation accepts (from the collection's example
request) and returns values only for the ones the user's request determines.
Code then:
  - rejects any field the operation does not have
  - fits each value to the example's type
  - fills ID placeholders from the plan's state, never from the model
  - drops free text the user did not write (a model-invented item name, say)
  - sets currency codes from the currency the user wrote ("$50" -> USD)
Example values are shown for format only and are never sent unless the model
sets them, so sample data from the collection cannot leak into a real call.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from . import template
from .catalog import Operation

MAX_FIELDS = 160
REQUIRED_FIELDS = Path(__file__).resolve().parent.parent / "data" / "required_fields.json"


@lru_cache(maxsize=1)
def required_defaults() -> dict:
    """operation -> {field: value sent when unset}: fields PayPal requires that the collection does
    not mark, found on the sandbox (data/required_fields.json)."""
    try:
        data = json.loads(REQUIRED_FIELDS.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return {op: {path: entry["value"] for path, entry in fields.items()}
            for op, fields in data.items() if not op.startswith("_")}


def add_required(op_id: str, body_values: dict) -> dict:
    """Fill in required fields the user left unset; returns what was added."""
    added = {}
    for path, value in required_defaults().get(op_id, {}).items():
        if not any(p == path or p.startswith(path + ".") or p.startswith(path + "[") for p in body_values):
            body_values[path] = value
            added[path] = value
    return added


@dataclass
class Filled:
    query: dict[str, Any] = field(default_factory=dict)
    body_values: dict[str, Any] = field(default_factory=dict)  # path -> value
    question: str | None = None
    llm_calls: int = 0
    dropped: dict[str, Any] = field(default_factory=dict)  # values removed because the user never said them
    needs_currency: bool = False  # an amount is set but the user named no currency
    hint: str | None = None       # a detail the model said only the user can supply

    def body(self, op: Operation, ids: dict[str, str]) -> Any:
        """Final request body: model-set values plus ID placeholders resolved from state."""
        if op.body_template is None:
            return None
        values = dict(self.body_values)
        for path, leaf in template.flatten(op.body_template).items():
            name = template.variable(leaf)
            if name and name in ids:
                values[path] = ids[name]
        built = template.build(values)
        if built is None:
            return [] if isinstance(op.body_template, list) else {}
        return built


class Filler:
    def __init__(self, llm, today: date | None = None):
        self.llm = llm
        self.today = today

    def fields(self, op: Operation) -> dict[str, Any]:
        """Body fields the model may set: path -> example value. ID placeholders are excluded."""
        if op.body_template is None:
            return {}
        out = {}
        for path, leaf in template.flatten(op.body_template).items():
            name = template.variable(leaf)
            if name and name in op.consumes:
                continue
            out[path] = "" if name else leaf
        return dict(list(out.items())[:MAX_FIELDS])

    def needs_model(self, op: Operation) -> bool:
        return bool(op.query or self.fields(op))

    def fill(self, op: Operation, message: str) -> Filled:
        if not self.needs_model(op):
            return Filled()
        fields = self.fields(op)
        query_names = {q["name"] for q in op.query}
        shapes = {template.shape(p): leaf for p, leaf in fields.items()}
        system = self._prompt(op, fields)
        user, result = message, Filled()
        for attempt in range(2):
            result.llm_calls += 1
            reply = self.llm.json(system, user)
            # On the retry, fields that do not exist are dropped rather than failing the request:
            # they are never sent, and the valid fields are still the user's.
            error = self._apply(reply, query_names, shapes, result, strict=attempt == 0)
            if error is None:
                result.dropped.update(drop_ungrounded(result.body_values, shapes, message))
                default_quantities(result.body_values, shapes)
                add_required(op.id, result.body_values)
                result.needs_currency = bool(set_currency(op, result.body_values, shapes, message))
                # The fill step never asks. What it found missing is passed on, and the readiness
                # check asks one question for the whole plan (agent/ready.py).
                hint, result.question = result.question, None
                asks_for_id = hint and any(name in hint for name in op.consumes)
                # Code sets currencies from the user's words, so the model's currency question is dropped.
                asks_currency = hint and "currency" in hint.lower()
                result.hint = hint if self._may_ask(op) and not asks_for_id and not asks_currency else None
                return result
            user = f"{message}\n\nYour previous answer was rejected: {error}. Reply again with a valid JSON object."
        result.query, result.body_values = {}, {}
        result.question = f"I could not work out the details for '{op.name}'. What values should I use?"
        return result

    def _apply(self, reply: dict, query_names: set, shapes: dict, result: Filled,
               strict: bool = True) -> str | None:
        query, body = reply.get("query") or {}, reply.get("body") or {}
        if not isinstance(query, dict) or not isinstance(body, dict):
            return "'query' and 'body' must be JSON objects"
        result.query, result.body_values = {}, {}
        for name, value in query.items():
            if name not in query_names:
                if strict:
                    return f"query parameter {name!r} does not exist"
                result.dropped[f"query.{name}"] = value
                continue
            if value is not None and value != "":
                result.query[name] = template.json_scalar(value)
        for path, value in body.items():
            key = template.shape(path)
            if key not in shapes:
                if strict:
                    return f"body field {path!r} does not exist; use only the listed paths"
                result.dropped[path] = value
                continue
            if value is None:
                continue
            try:
                result.body_values[path] = template.coerce(shapes[key], value)
            except ValueError as err:
                if strict:
                    return f"body field {path!r} {err}"
                result.dropped[path] = value
        missing = reply.get("missing")
        result.question = missing if isinstance(missing, str) and missing.strip() else None
        return None

    @staticmethod
    def _may_ask(op: Operation) -> bool:
        # A read is safe to run with whatever filters we have; asking first only slows the user down.
        return op.method != "GET"

    def _prompt(self, op: Operation, fields: dict) -> str:
        today = (self.today or date.today()).isoformat()
        lines = [
            f"You fill in the parameters for one API call: {op.name}.",
            op.description,
            f"Today's date is {today}.",
            "",
            "Reply with a JSON object:",
            '{"query": {name: value}, "body": {path: value}, "missing": string or null}',
            "",
            "Rules:",
            "- Set only values that the user's request states or clearly implies.",
            "- Text fields (names, descriptions, notes, subjects) must use the user's own words. "
            "Never write text of your own, such as a generic item name; leave the field out instead.",
            "- The examples show format only. Never copy an example value.",
            "- Leave out everything else; unset fields are not sent.",
            "- For a list, use the same path with [0], [1], ... for each entry.",
            "- Identifiers such as invoice_id or order_id are supplied automatically from earlier steps. "
            "Never ask for them and never set them.",
            "- 'missing' is a last resort. Use it only when the call cannot be attempted at all without a "
            "value that only the user can supply and that the request is about, such as who to send to "
            "or how much. Never ask for optional fields, search filters, or the account owner's own "
            "details (name, address, logo): the API applies defaults. When in doubt, leave it null.",
        ]
        # Rules about fields this call does not have only invite the model to invent them.
        if any(path.endswith("currency_code") for path in fields):
            lines.append("- Currency codes are set automatically from the user's message. "
                         "Never ask about the currency.")
        if any(".unit_amount." in path for path in fields):
            lines.append("- An amount the user gives for something sold is an item: set its unit_amount.value "
                         "and quantity even when the item has no name yet. The agent asks for the name itself.")
        if op.query:
            lines += ["", "Query parameters (name: example - meaning):"]
            lines += [f"- {q['name']}: {_short(q['example'])} - {q['description']}".rstrip(" -") for q in op.query]
        if fields:
            lines += ["", "Body fields (path: example):"]
            lines += [f"- {path}: {_short(leaf)}" for path, leaf in fields.items()]
        return "\n".join(lines)


_CODE = re.compile(r"^[A-Z0-9_]+$")          # enums and codes: NET_10, US, PERCENT
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")
_WORD = re.compile(r"[a-z0-9]+")
_INSTRUCTION = re.compile(r"^\((If what the user added|Apply this change).*\)$", re.M)

CURRENCY_CODES = {"USD", "EUR", "GBP", "INR", "JPY", "AUD", "CAD", "CHF", "CNY", "SGD", "HKD", "NZD",
                  "SEK", "NOK", "DKK", "MXN", "BRL", "PLN", "CZK", "HUF", "ILS", "PHP", "THB", "TWD",
                  "MYR"}
_CURRENCY_SIGNS = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR", "¥": "JPY"}
_CURRENCY_WORDS = {"dollar": "USD", "dollars": "USD", "euro": "EUR", "euros": "EUR", "pound": "GBP",
                   "pounds": "GBP", "rupee": "INR", "rupees": "INR", "yen": "JPY"}


def user_text(message: str) -> str:
    """What the user wrote: the merged request without the instructions the agent adds to it."""
    return _INSTRUCTION.sub("", message)


def _grounded(value: Any, text: str) -> bool:
    """Whether a text value comes from what the user wrote. Codes, dates and numbers are
    formats the model converts to, so they pass; emails and URLs must appear as written;
    free text must use only words the user wrote."""
    if not isinstance(value, str) or not value.strip():
        return True
    if _CODE.match(value) or _DATE.match(value) or _NUMBER.match(value):
        return True
    lowered = text.lower()
    if "@" in value or value.lower().startswith(("http://", "https://", "www.")):
        return value.lower() in lowered
    said = set(_WORD.findall(lowered))
    said |= {w[:-1] for w in said if w.endswith("s")} | {w + "s" for w in said}
    return all(word in said for word in _WORD.findall(value.lower()))


def drop_ungrounded(body_values: dict[str, Any], shapes: dict[str, Any], message: str) -> dict[str, Any]:
    """Remove text values the user never wrote; returns what was removed."""
    text = user_text(message)
    dropped = {}
    for path, value in list(body_values.items()):
        if path.endswith("currency_code"):
            continue  # decided by set_currency
        if isinstance(shapes.get(template.shape(path)), str) and not _grounded(value, text):
            dropped[path] = body_values.pop(path)
    return dropped


def default_quantities(body_values: dict[str, Any], shapes: dict[str, Any]) -> None:
    """A list entry with a price but no quantity is one unit of that price ("an invoice for $50").
    The value shows in the approval preview like any other field."""
    priced = {path.split(".unit_amount.")[0] for path in body_values if ".unit_amount." in path}
    for entry in priced:
        quantity = f"{entry}.quantity"
        example = shapes.get(template.shape(quantity))
        if example is not None and quantity not in body_values:
            body_values[quantity] = "1" if isinstance(example, str) else 1


def stated_currency(message: str) -> str | None:
    """The one currency the user wrote, as an ISO code; None if none or more than one."""
    text = user_text(message)
    found = {code for code in re.findall(r"\b[A-Z]{3}\b", text) if code in CURRENCY_CODES}
    found |= {code for sign, code in _CURRENCY_SIGNS.items() if sign in text}
    found |= {code for word, code in _CURRENCY_WORDS.items() if re.search(rf"\b{word}\b", text, re.I)}
    return found.pop() if len(found) == 1 else None


def set_currency(op: Operation, body_values: dict[str, Any], shapes: dict[str, Any], message: str) -> str | None:
    """Set every currency code the body needs to the currency the user wrote.

    A code is needed beside any amount that is set (items[0].unit_amount.value), and for a
    document-level code (detail.currency_code) when its object is sent at all. The model's
    own currency codes are replaced: a currency is never guessed. Returns a question when a
    code is needed and the user named no single currency."""
    if op.body_template is None:
        return None
    needed = set()
    for path in body_values:
        parent = path.rsplit(".", 1)[0] if "." in path else ""
        candidate = f"{parent}.currency_code" if parent else "currency_code"
        if template.shape(candidate) in shapes and not path.endswith("currency_code"):
            needed.add(candidate)
    present = set(body_values)
    for path, leaf in template.flatten(op.body_template).items():
        name = template.variable(leaf)
        if name and name in op.consumes:
            present.add(path)
    ancestors = {p.rsplit(".", i)[0] for p in present for i in range(1, p.count(".") + 1)}
    for path in template.flatten(op.body_template):
        if path.endswith("currency_code") and "[" not in path and "." in path                 and path.rsplit(".", 1)[0] in ancestors:
            needed.add(path)
    for path in [p for p in body_values if p.endswith("currency_code")]:
        needed.add(path)
    if not needed:
        return None
    currency = stated_currency(message)
    if currency is None:
        for path in needed:
            body_values.pop(path, None)
        return f"Which currency should I use for '{op.name}'? For example USD or EUR."
    for path in needed:
        body_values[path] = currency
    return None


def _short(value: Any, limit: int = 40) -> str:
    text = value if isinstance(value, str) else template.json_scalar(value)
    return text if len(text) <= limit else text[:limit] + "..."

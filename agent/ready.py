"""Readiness: before approval, check the planned calls have every detail only the user can supply.

Two layers decide what is missing:
  - code rules that the model cannot waive: a priced item needs a name, a request that
    will send something needs the recipient's email, an amount needs a stated currency;
  - an LLM check over the calls about to be sent, which may add details the API needs.
Both feed ONE question to the user. The answer is then read into the missing fields only
(`AnswerReader`), with code checking each value: the field must be one that was asked
for, text must be the user's own words, an email must appear as typed.

Everything here works on plain call dicts ({"op", "query", "body_values", ...}) so it can
live in the LangGraph state.
"""
from __future__ import annotations

import json
import re
from typing import Any

from . import template
from .catalog import Catalog, Operation
from .fill import CURRENCY_CODES, Filler, _grounded, stated_currency

CURRENCY = "currency"   # a missing currency, applied to every call that needs one
HINT = "hint"           # a free-form detail the fill model said it needed
MAX_LLM_MISSING = 4
EMAIL_PATH = "primary_recipients[0].billing_info.email_address"
_ENTRY = re.compile(r"^(\w+\[\d+\])\.")


def shapes_for(op: Operation) -> dict[str, Any]:
    return {template.shape(p): leaf for p, leaf in Filler(None).fields(op).items()}


def label_for(path: str) -> str:
    """A plain description of a field, for questions."""
    if path == CURRENCY:
        return "the currency (for example USD)"
    if path.endswith("email_address"):
        return "the recipient's email address"
    entry = _ENTRY.match(path)
    if entry and path.endswith(".name"):
        index = int(re.search(r"\[(\d+)\]", path).group(1))
        thing = entry.group(1).split("[")[0].removesuffix("s").replace("_", " ")
        return f"a name for the {thing}" + (f" {index + 1}" if index else "") + " (what it is for)"
    if path.endswith("amount.value") or path.endswith("price.value"):
        return "the amount"
    if path.endswith(".quantity"):
        return "the quantity"
    if ".name." in path and "billing_info" in path:
        return "the recipient's name"
    words = [w for w in re.sub(r"\[\d+\]", "", path).split(".") if w][-2:]
    return "the " + " ".join(words).replace("_", " ")


def code_rules(catalog: Catalog, calls: list[dict]) -> list[dict]:
    """Missing details by rule. Also moves an item's description into its missing name: the
    model sometimes puts the user's words there, and they are the user's own words either way."""
    sends = any(catalog[c["op"]].risk == "outbound" for c in calls)
    missing, currency_asked = [], False
    for i, call in enumerate(calls):
        op, values = catalog[call["op"]], call["body_values"]
        shapes = shapes_for(op)
        priced = sorted({m.group(1) for p in values if (m := _ENTRY.match(p)) and ".unit_amount." in p})
        for entry in priced:
            name = f"{entry}.name"
            if template.shape(name) not in shapes or name in values:
                continue
            description = f"{entry}.description"
            if values.get(description):
                values[name] = values.pop(description)
            else:
                missing.append({"call": i, "path": name, "label": label_for(name), "rule": True})
        has_email = any(p.startswith("primary_recipients[") and p.endswith("email_address") for p in values)
        if sends and template.shape(EMAIL_PATH) in shapes and not has_email:
            missing.append({"call": i, "path": EMAIL_PATH, "label": label_for(EMAIL_PATH), "rule": True})
        if call.get("needs_currency") and not currency_asked:
            currency_asked = True
            missing.append({"call": i, "path": CURRENCY, "label": label_for(CURRENCY), "rule": True})
        if call.get("hint"):
            missing.append({"call": i, "path": HINT, "label": call["hint"], "rule": False})
    return missing


def code_question(missing: list[dict], calls: list[dict], rounds: int) -> str:
    """A question built from the missing labels. After the first round it also says what is
    already known, so the user can see the answers landed."""
    labels = list(dict.fromkeys(m["label"] for m in missing if m["path"] != HINT))
    hints = list(dict.fromkeys(m["label"] for m in missing if m["path"] == HINT))
    parts = (["To finish this I still need " + _join(labels) + "."] if labels else []) + hints
    question = " ".join(parts)
    if rounds:
        have = _known_summary(calls)
        if have:
            question = f"So far I have: {have}.\n{question}"
        question += " (Or say 'cancel'.)"
    return question


def _known_summary(calls: list[dict]) -> str:
    parts = []
    for call in calls:
        for path, value in call["body_values"].items():
            if isinstance(value, str) and value and not path.endswith("currency_code") and len(parts) < 6:
                parts.append(f"{label_for(path).removeprefix('the ')} '{value}'")
    return "; ".join(parts)


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


class ReadyChecker:
    """The LLM layer: what else must the user tell us before these calls can succeed?"""

    def __init__(self, llm):
        self.llm = llm

    def check(self, catalog: Catalog, request: str, answers: list[dict], calls: list[dict],
              rule_missing: list[dict], asked_before: set | None = None) -> tuple[list[dict], str | None]:
        """Returns (extra missing details, one question covering every missing detail).

        asked_before: (call, path) pairs this check already asked for. A detail only the LLM
        thinks is needed is asked once; if the user's answer does not give it, it is dropped,
        so a wrong guess about what the API needs cannot loop."""
        writes = [(i, c) for i, c in enumerate(calls) if catalog[c["op"]].method != "GET"]
        if not writes:
            return [], None
        reply = self.llm.json(self._system(), self._user(catalog, request, answers, writes, rule_missing))
        validated = self._validate(catalog, calls, reply.get("missing"), rule_missing)
        extra = [m for m in validated if (m["call"], m["path"]) not in (asked_before or set())]
        if len(extra) < len(validated):
            return extra, None  # its question would ask again what was asked once; code words it instead
        question = reply.get("question")
        question = question.strip() if isinstance(question, str) and question.strip() else None
        return extra, question

    @staticmethod
    def _system() -> str:
        return "\n".join([
            "You check API calls an agent is about to send for a user, before the user approves them.",
            "Decide whether any detail is still missing that the API needs and that only the user can supply.",
            "",
            'Reply with a JSON object: {"missing": [{"call": number, "path": string, "why": string}], '
            '"question": string or null}',
            "",
            "Rules:",
            "- Count only what the call needs to succeed or to do what the user asked: who it is for, "
            "what it is for, how much, a date only the user can choose.",
            "- Never list optional fields, the account owner's own details (name, address, logo, phone), "
            "identifiers (supplied automatically), or currency (set from the user's words).",
            "- Anything the API defaults is optional: issue dates, due dates, payment terms, notes, "
            "references, tax, discounts. Ask for one only if the user's request mentions it.",
            "- Most requests are complete. When in doubt, nothing is missing.",
            "- A value already set counts as given, even if you would have worded it differently.",
            "- 'path' must be one of the call's listed fields.",
            "- 'question': ONE short, friendly question asking for every missing detail, the ones under "
            "'Already known to be missing' included. Plain words, no field paths. null if nothing is missing.",
        ])

    @staticmethod
    def _user(catalog, request, answers, writes, rule_missing) -> str:
        lines = [f"User request: {request}"]
        if answers:
            lines.append("Answers the user gave to earlier questions:")
            lines += [f"- Q: {a['question']}\n  A: {a['answer']}" for a in answers]
        for i, call in writes:
            op = catalog[call["op"]]
            lines += ["", f"Call {i}: {op.name} ({op.method} {op.path})",
                      f"Set so far: {json.dumps(call['body_values'])}",
                      "Fields: " + ", ".join(Filler(None).fields(op))]
        if rule_missing:
            lines += ["", "Already known to be missing: " + "; ".join(m["label"] for m in rule_missing)]
        return "\n".join(lines)

    @staticmethod
    def _validate(catalog, calls, items, rule_missing) -> list[dict]:
        if not isinstance(items, list):
            return []
        taken = {(m["call"], m["path"]) for m in rule_missing}
        out = []
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("call"), int):
                continue
            i, path = item["call"], item.get("path")
            if not (0 <= i < len(calls)) or not isinstance(path, str):
                continue
            op = catalog[calls[i]["op"]]
            if op.method == "GET" or path.startswith("invoicer.") or path.endswith("currency_code"):
                continue
            if template.shape(path) not in shapes_for(op) or path in calls[i]["body_values"]:
                continue
            if (i, path) in taken:
                continue
            taken.add((i, path))
            out.append({"call": i, "path": path, "label": label_for(path), "rule": False,
                        "why": str(item.get("why") or "")[:200]})
        return out[:MAX_LLM_MISSING]


class AnswerReader:
    """Reads the user's reply into the missing fields only."""

    def __init__(self, llm):
        self.llm = llm

    def read(self, question: str, request: str, missing: list[dict], answer: str) -> tuple[dict, bool]:
        """Returns ({(call, path): raw value}, new_request)."""
        listed = "\n".join(f"- {m['path']}: {m['label']}" for m in missing)
        system = "\n".join([
            "The user was asked for details missing from their request. Read their reply.",
            'Reply with a JSON object: {"new_request": boolean, "values": {path: value}}',
            "",
            "Rules:",
            "- new_request is true only if the reply asks for something different instead of answering.",
            "- values: for each listed path the reply gives, the value in the user's own words. Leave out "
            "paths the reply does not give. Never invent a value.",
            "- For 'currency' give the ISO code (USD, EUR, ...). For 'hint' give the part of the reply that "
            "answers it.",
        ])
        user = f"Request: {request}\nQuestion asked: {question}\nMissing:\n{listed}\nUser reply: {answer}"
        reply = self.llm.json(system, user)
        values = reply.get("values") if isinstance(reply.get("values"), dict) else {}
        by_path = {}
        for m in missing:
            if m["path"] in values and values[m["path"]] not in (None, ""):
                by_path[(m["call"], m["path"])] = values[m["path"]]
        return by_path, bool(reply.get("new_request"))


def accept(catalog: Catalog, calls: list[dict], key: tuple[int, str], value: Any, answer: str,
           request: str) -> bool:
    """Put one read value into its call if code agrees it is the user's. Returns whether it was kept."""
    i, path = key
    call = calls[i]
    if path == CURRENCY:
        code = stated_currency(answer) or (str(value).upper() if str(value).upper() in CURRENCY_CODES
                                           and str(value).upper() in answer.upper() else None)
        return bool(code) and _set_currency_everywhere(catalog, calls, code)
    if path == HINT:
        return False  # handled by re-filling the call
    shapes = shapes_for(catalog[call["op"]])
    try:
        value = template.coerce(shapes[template.shape(path)], value)
    except (KeyError, ValueError):
        return False
    if not _grounded(value, f"{answer}\n{request}"):
        return False
    call["body_values"][path] = value
    return True


def _set_currency_everywhere(catalog: Catalog, calls: list[dict], code: str) -> bool:
    from .fill import set_currency
    for call in calls:
        if call.get("needs_currency"):
            set_currency(catalog[call["op"]], call["body_values"], shapes_for(catalog[call["op"]]), code)
            call["needs_currency"] = False
    return True

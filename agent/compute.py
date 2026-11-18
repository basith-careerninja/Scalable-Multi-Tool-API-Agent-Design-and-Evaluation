"""Compute step: totals, counts and filters done in code, not by the model.

The model is shown the *shape* of an API result (which lists it holds and
which fields their rows have) and writes a small calculation spec. Code checks
the spec against that shape and runs it with exact decimal arithmetic over
every row. The model never adds numbers and never needs to see all the rows,
so a long list cannot be silently cut off before it is summed.

Spec:
  {"collection": "transaction_details",
   "filters": [{"field": "...", "op": "eq", "value": "S"}],
   "group_by": "..." or null,
   "aggregate": {"fn": "sum", "field": "..."}}
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from . import template

OPS = ("eq", "ne", "gt", "ge", "lt", "le", "contains", "in")
FUNCTIONS = ("count", "sum", "avg", "min", "max")
_NUMERIC_OPS = {"gt": lambda a, b: a > b, "ge": lambda a, b: a >= b, "lt": lambda a, b: a < b, "le": lambda a, b: a <= b}
_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")
MAX_FIELDS = 70


class SpecError(ValueError):
    pass


# -- describing a result -----------------------------------------------------------

def collections(body: Any) -> dict[str, list]:
    """Lists of objects in a response, by path. Looks at the top level and one level down."""
    found = {}
    if isinstance(body, list) and body and isinstance(body[0], dict):
        found[""] = body
    if isinstance(body, dict):
        for key, value in body.items():
            if isinstance(value, list) and value and isinstance(value[0], dict) and key != "links":
                found[key] = value
            elif isinstance(value, dict):
                for sub, inner in value.items():
                    if isinstance(inner, list) and inner and isinstance(inner[0], dict) and sub != "links":
                        found[f"{key}.{sub}"] = inner
    return found


def fields(rows: list) -> dict[str, Any]:
    """Field paths present in the rows -> an example value. Array positions are ignored."""
    seen: dict[str, Any] = {}
    for row in rows[:50]:
        for path, value in template.flatten(row).items():
            key = template.shape(path).replace("[*]", "")
            if "links" not in key:
                seen.setdefault(key, value)
    return seen


def get(row: Any, path: str) -> Any:
    node = row
    for name, index in _TOKEN.findall(path):
        if isinstance(node, list):
            node = node[0] if node else None
        if index:
            node = node[int(index)] if isinstance(node, list) and int(index) < len(node) else None
        elif isinstance(node, dict):
            node = node.get(name)
        else:
            return None
    return node[0] if isinstance(node, list) and node else node


def number(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None


# -- validating and running a spec ---------------------------------------------------

def validate(spec: dict, body: Any) -> dict:
    """Return a cleaned spec, or raise SpecError naming what is wrong."""
    available = collections(body)
    name = spec.get("collection")
    if name not in available:
        raise SpecError(f"collection {name!r} does not exist; choose one of {list(available)}")
    known = fields(available[name])

    def check_field(path, what):
        if path not in known:
            raise SpecError(f"{what} field {path!r} does not exist in '{name}'; use only the listed fields")
        return path

    aggregate = spec.get("aggregate") or {}
    fn = aggregate.get("fn")
    if fn not in FUNCTIONS:
        raise SpecError(f"aggregate.fn must be one of {list(FUNCTIONS)}")
    field = aggregate.get("field")
    if fn != "count":
        check_field(field, "aggregate")

    filters = []
    for item in spec.get("filters") or []:
        if not isinstance(item, dict) or item.get("op") not in OPS:
            raise SpecError(f"each filter needs field, op and value; op must be one of {list(OPS)}")
        check_field(item.get("field"), "filter")
        if item["op"] in _NUMERIC_OPS and number(item.get("value")) is None:
            raise SpecError(f"filter on {item['field']!r} with {item['op']} needs a numeric value")
        if item["op"] == "in" and not isinstance(item.get("value"), list):
            raise SpecError("the 'in' filter needs a list value")
        filters.append({"field": item["field"], "op": item["op"], "value": item.get("value")})

    group_by = spec.get("group_by") or None
    if group_by:
        check_field(group_by, "group_by")
    auto_grouped = False
    if fn != "count" and not group_by and field.endswith(".value"):
        # Money in different currencies must never be added together.
        currency = field[: -len("value")] + "currency_code"
        if currency in known:
            group_by, auto_grouped = currency, True

    return {"collection": name, "filters": filters, "group_by": group_by,
            "aggregate": {"fn": fn, "field": None if fn == "count" else field},
            "auto_grouped_by_currency": auto_grouped}


def _matches(row: Any, item: dict) -> bool:
    actual, op, wanted = get(row, item["field"]), item["op"], item["value"]
    if op in _NUMERIC_OPS:
        a, b = number(actual), number(wanted)
        return a is not None and b is not None and _NUMERIC_OPS[op](a, b)
    if op == "contains":
        return actual is not None and str(wanted).lower() in str(actual).lower()
    if op == "in":
        return str(actual) in {str(v) for v in wanted}
    a, b = number(actual), number(wanted)
    equal = (a == b) if a is not None and b is not None else str(actual) == str(wanted)
    return equal if op == "eq" else not equal


def run(spec: dict, body: Any) -> dict:
    """Run a validated spec. Every row is considered; nothing is sampled or truncated."""
    rows = collections(body)[spec["collection"]]
    matched = [row for row in rows if all(_matches(row, f) for f in spec["filters"])]
    fn, field = spec["aggregate"]["fn"], spec["aggregate"]["field"]

    groups: dict[str, list] = {}
    for row in matched:
        key = str(get(row, spec["group_by"])) if spec["group_by"] else "all"
        groups.setdefault(key, []).append(row)

    skipped = 0
    values: dict[str, str] = {}
    for key, members in groups.items():
        if fn == "count":
            values[key] = str(len(members))
            continue
        numbers = [number(get(row, field)) for row in members]
        skipped += sum(n is None for n in numbers)
        numbers = [n for n in numbers if n is not None]
        if not numbers:
            continue
        result = {"sum": sum(numbers), "avg": sum(numbers) / len(numbers),
                  "min": min(numbers), "max": max(numbers)}[fn]
        values[key] = format(result, "f")
    return {
        "result": values if spec["group_by"] else values.get("all", "0"),
        "rows_total": len(rows),
        "rows_matched": len(matched),
        "rows_skipped_not_numeric": skipped,
        "description": describe(spec),
    }


def describe(spec: dict) -> str:
    fn, field = spec["aggregate"]["fn"], spec["aggregate"]["field"]
    text = "count of records" if fn == "count" else f"{fn} of {field}"
    if spec["filters"]:
        text += " where " + " and ".join(f"{f['field']} {f['op']} {f['value']}" for f in spec["filters"])
    if spec["group_by"]:
        text += f", per {spec['group_by']}"
    return text


# -- asking the model for a spec -------------------------------------------------------

class ComputePlanner:
    def __init__(self, llm):
        self.llm = llm

    def plan(self, question: str, body: Any, applied_query: dict | None = None) -> dict | None:
        """A validated spec, or None when no calculation applies or the model cannot produce one.

        applied_query: filters the API call already applied, so they are not repeated here.
        """
        available = collections(body)
        if not available:
            return None
        system = self._prompt(available)
        if applied_query:
            system += ("\n\nThe API call already applied these query parameters, so the rows are already "
                       f"limited by them. Do not filter on them again: {applied_query}")
        user = question
        for _ in range(2):
            reply = self.llm.json(system, user)
            if not reply.get("collection"):
                return None
            try:
                return validate(reply, body)
            except SpecError as err:
                user = f"{question}\n\nYour previous answer was rejected: {err}. Reply again with a valid JSON object."
        return None

    def _prompt(self, available: dict[str, list]) -> str:
        lines = [
            "The user asked a question that needs a calculation over API results. You do not do the "
            "arithmetic. Write a calculation spec; code will run it exactly over every row.",
            "",
            "Reply with a JSON object:",
            '{"collection": string, "filters": [{"field": string, "op": string, "value": any}], '
            '"group_by": string or null, "aggregate": {"fn": string, "field": string or null}}',
            f"- op is one of: {', '.join(OPS)}.",
            f"- fn is one of: {', '.join(FUNCTIONS)}. For count, field is null.",
            "- Use only the collections and fields listed. Examples show format only.",
            "- When adding money, group by its currency field.",
            "- Add filters only for conditions the user stated or that the question plainly requires "
            "(for example, sales means incoming amounts, so amount greater than 0).",
            '- If no calculation fits, reply {"collection": null}.',
        ]
        for name, rows in available.items():
            lines += ["", f"Collection '{name}' ({len(rows)} rows). Fields (path: example):"]
            lines += [f"- {path}: {str(value)[:40]}" for path, value in list(fields(rows).items())[:MAX_FIELDS]]
        return "\n".join(lines)

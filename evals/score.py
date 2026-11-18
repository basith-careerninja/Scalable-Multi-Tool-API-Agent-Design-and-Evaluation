"""Eval cases and scoring. Pure functions: no model or API calls here.

A case says what the right outcome of a message is (see evals/README.md). A system's
answer is a list of "steps", one per goal it found in the message:
    {"intent", "op", "new", "ids", "calc", "decided_by", "question"}
A system that splits a message into several goals returns one step per goal.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXTERNAL_FILE = "cases_external.jsonl"  # reported apart, never tuned on
CASE_FILES = ("cases.jsonl", "cases_user.jsonl", EXTERNAL_FILE)
INTENTS = ("action", "docs", "system", "ask")


# -- cases ----------------------------------------------------------------------------

def load_cases(names=CASE_FILES, folder: Path = HERE, op_ids: set[str] | None = None) -> list[dict]:
    """All cases, validated, each with a "split": tune, test or external."""
    cases, seen = [], set()
    for name in names:
        path = Path(folder) / name
        if not path.exists():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"{name} line {number}: not valid JSON ({err})") from err
            problems = validate_case(case, op_ids)
            if problems:
                raise ValueError(f"{name} line {number} ({case.get('id')}): {'; '.join(problems)}")
            if case["id"] in seen:
                raise ValueError(f"{name} line {number}: duplicate id {case['id']}")
            seen.add(case["id"])
            case.setdefault("source", name)
            case["file"] = name
            cases.append(case)
    assign_splits(cases)
    return cases


def validate_case(case: dict, op_ids: set[str] | None = None) -> list[str]:
    problems = [f"missing {key}" for key in ("id", "category", "message", "expect") if key not in case]
    if problems:
        return problems
    expect = case["expect"]
    if expect.get("intent") not in INTENTS:
        problems.append(f"expect.intent must be one of {list(INTENTS)}")
    if expect.get("intent") == "action":
        steps = expect.get("steps") or []
        if not steps:
            problems.append("an action case needs expect.steps")
        for step in steps:
            for op in [step.get("op"), *step.get("alt", [])]:
                if not op:
                    problems.append("every step needs an op")
                elif op_ids is not None and op not in op_ids:
                    problems.append(f"unknown operation {op!r}")
        if expect.get("plan") and len(steps) != 1:
            problems.append("expect.plan is only for single-goal cases")
    if "e2e_live" in case:
        problems += validate_live(case["e2e_live"], op_ids)
    return problems


# -- live end to end (PayPal sandbox) ----------------------------------------------------

APPROVAL_LEVELS = ("none", "confirm", "explicit")


def validate_live(block: dict, op_ids: set[str] | None = None) -> list[str]:
    problems = []
    if block.get("approval") not in APPROVAL_LEVELS:
        problems.append(f"e2e_live.approval must be one of {list(APPROVAL_LEVELS)}")
    if "message" in block and not isinstance(block["message"], str):
        problems.append("e2e_live.message must be a string")
    answers = block.get("answers", [])
    if not (isinstance(answers, list) and all(isinstance(a, str) for a in answers)):
        problems.append("e2e_live.answers must be a list of strings (replies to the agent's questions, in order)")
    ops = list(block.get("calls", []))
    for flow in block.get("id_flow", []):
        if not (isinstance(flow, list) and len(flow) == 3):
            problems.append("each e2e_live.id_flow entry is [producer op, id name, consumer op]")
            continue
        ops += [flow[0], flow[2]]
    if op_ids is not None:
        problems += [f"e2e_live: unknown operation {op!r}" for op in ops if op not in op_ids]
    return problems


def live_message(case: dict, sandbox_ids: dict) -> str:
    """The message for a live run: e2e_live.message (with {placeholders} from sandbox_ids.json) or the case's own."""
    template = case["e2e_live"].get("message", case["message"])
    try:
        return template.format_map(sandbox_ids)
    except KeyError as err:
        raise KeyError(f"needs a {err.args[0]}, and the sandbox has none to read") from None


def _collapse(ops: list[str]) -> list[str]:
    """Consecutive repeats of one operation are its later pages, not separate calls."""
    return [op for i, op in enumerate(ops) if i == 0 or op != ops[i - 1]]


def live_checks(expect: dict, planned: list[str], level: str | None, waiting: bool,
                before: list[dict], after: list[dict]) -> tuple[dict[str, bool], list[str]]:
    """Behaviour checks for one live case. Pure.

    planned: operation IDs of the plan. level: the approval level the agent chose.
    waiting: the first reply asked for approval. before/after: calls recorded during the
    first turn and during the approval turn, each {"op", "risk", "status", "path", "body", "produced"}.
    """
    checks, problems = {}, []
    sent = before + after
    want = expect["approval"]
    checks["approval_asked_when_required"] = waiting == (want != "none") and (not waiting or level == want)
    if not checks["approval_asked_when_required"]:
        problems.append(f"approval: asked={waiting} level={level}, expected {want}")
    if waiting:
        # Read-only look-ups may run before approval; nothing that changes anything may.
        early = [c["op"] for c in before if c["risk"] != "read"]
        checks["nothing_sent_before_approval"] = not early
        if early:
            problems.append(f"sent {early} before approval")
    executed = _collapse([c["op"] for c in (after if waiting else before)])
    if planned:
        checks["calls_in_planned_order"] = executed[-len(planned):] == planned
        if not checks["calls_in_planned_order"]:
            problems.append(f"sent {executed}, planned {planned}")
    if "calls" in expect:
        checks["planned_calls"] = planned == expect["calls"]
        if not checks["planned_calls"]:
            problems.append(f"planned {planned}, expected {expect['calls']}")
    checks["all_2xx"] = bool(sent) and all(200 <= c["status"] < 300 for c in sent)
    if not checks["all_2xx"]:
        bad = [f"{c['op']} {c['status']}" for c in sent if not 200 <= c["status"] < 300]
        problems.append(f"HTTP errors: {bad}" if bad else "no call was sent")
    for producer, name, consumer in expect.get("id_flow", []):
        value = next((c["produced"].get(name) for c in sent if c["op"] == producer and name in c["produced"]), None)
        used = next((c for c in sent if c["op"] == consumer), None)
        ok = bool(value) and used is not None and (value in used["path"] or value in json.dumps(used["body"]))
        checks[f"id_flow {name}"] = ok
        if not ok:
            problems.append(f"{name} from {producer} ({value!r}) did not reach {consumer}")
    return checks, problems


def assign_splits(cases: list[dict]) -> None:
    """Half of each category for tuning, half held out for reporting. External cases stay apart."""
    by_category = defaultdict(list)
    for case in cases:
        if case.get("file") == EXTERNAL_FILE:
            case["split"] = "external"
        else:
            by_category[case["category"]].append(case)
    for group in by_category.values():
        for n, case in enumerate(sorted(group, key=lambda c: c["id"])):
            case["split"] = "tune" if n % 2 == 0 else "test"


# -- selection --------------------------------------------------------------------------

def to_steps(selections) -> list[dict]:
    return [{"intent": s.intent, "op": s.operation, "new": sorted(s.new_ids), "ids": dict(s.known_ids),
             "calc": bool(s.calculate), "decided_by": s.decided_by, "question": s.question}
            for s in selections]


def overall_intent(steps: list[dict]) -> str:
    if not steps or any(s["intent"] == "clarify" for s in steps):
        return "ask"
    if any(s["intent"] == "action" for s in steps):
        return "action"
    return steps[0]["intent"]


def step_checks(expected: dict, got: dict) -> dict[str, bool]:
    return {
        "op": got.get("op") == expected["op"] or got.get("op") in expected.get("alt", []),
        "new": sorted(expected.get("new", [])) == sorted(got.get("new", [])),
        "ids": expected.get("ids", {}) == got.get("ids", {}),
        "calc": expected.get("calc") is None or expected["calc"] == got.get("calc"),
    }


def match_steps(expected: list[dict], got: list[dict]) -> list[tuple[dict, dict]] | None:
    """Pair each expected step with a distinct step that has an acceptable operation. Order-free."""
    if len(expected) != len(got):
        return None
    pairs, used = [], set()
    for exp in expected:
        index = next((i for i, g in enumerate(got) if i not in used and step_checks(exp, g)["op"]), None)
        if index is None:
            return None
        used.add(index)
        pairs.append((exp, got[index]))
    return pairs


def score_split_chain(case: dict, steps: list[dict], combined_plan: list[str] | None) -> dict[str, Any] | None:
    """Score a split of a message the case treats as ONE goal with a planned chain.

    "Create a product, then a plan for it, then subscribe sam" is one goal (subscribe) with
    the chain product.create -> plan.create -> subscription.create; the agent may instead
    split it into three actions. What matters is what would be sent, so the split counts
    as right when the calls its actions plan, in order (`combined_plan`), are exactly the
    expected chain and its last action is the goal. Creation of each ID is then covered by
    that plan match, so the per-step "new" check does not apply; typed IDs and the
    calculation flag are still checked, on the last action (the goal step they belong to).

    Returns None when this rule does not apply, and the ordinary scoring is used."""
    plan, want = case["expect"].get("plan"), case["expect"].get("steps") or []
    actions = [s for s in steps if s["intent"] == "action"]
    if plan is None or len(want) != 1 or len(actions) < 2 or overall_intent(steps) != "action":
        return None
    goal, last = want[0], actions[-1]
    if combined_plan != plan or not step_checks(goal, last)["op"]:
        return None
    # The case's expected IDs belong to its goal step, so they are compared with the last
    # action (the goal). IDs typed for earlier actions have no expectation in the case.
    typed = dict(last.get("ids") or {})
    problems = []
    if goal.get("ids", {}) != typed:
        problems.append(f"{goal['op']}: ids got {typed!r}, expected {goal.get('ids', {})!r}")
    if goal.get("calc") is not None and goal["calc"] != last.get("calc"):
        problems.append(f"{goal['op']}: calc got {last.get('calc')!r}, expected {goal['calc']!r}")
    return {"intent_ok": True, "got_intent": "action", "asked": False, "should_ask": False, "dropped": False,
            "tool_ok": True, "plan_ok": True, "full_ok": not problems, "problems": problems,
            "scored_as": "split matching the planned chain"}


def score_selection(case: dict, steps: list[dict]) -> dict[str, Any]:
    expected = case["expect"]["intent"]
    got = overall_intent(steps)
    result = {"intent_ok": expected == got, "got_intent": got, "asked": got == "ask",
              "should_ask": expected == "ask", "dropped": False, "problems": []}
    if expected != "action":
        result["tool_ok"] = result["full_ok"] = result["intent_ok"]
        if not result["intent_ok"]:
            result["problems"].append(f"intent {got}, expected {expected}")
        return result
    want = case["expect"]["steps"]
    actions = [s for s in steps if s["intent"] == "action"]
    pairs = match_steps(want, actions) if got == "action" else None
    result["tool_ok"] = pairs is not None
    result["dropped"] = got == "action" and len(actions) < len(want)
    if got == "ask":
        result["problems"].append(f"asked instead of acting: {steps[0].get('question') if steps else ''}")
    elif got != "action":
        result["problems"].append(f"intent {got}, expected action")
    elif pairs is None:
        result["problems"].append(f"operations {[s['op'] for s in actions]}, expected {[w['op'] for w in want]}")
    else:
        for exp, step in pairs:
            for check, ok in step_checks(exp, step).items():
                if not ok:
                    result["problems"].append(f"{exp['op']}: {check} got {step.get(check)!r}, "
                                              f"expected {exp.get(check, [] if check == 'new' else {})!r}")
    result["full_ok"] = result["tool_ok"] and not result["problems"]
    return result


# -- end to end ---------------------------------------------------------------------------

def _number(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None


def value_matches(expected: Any, actual: Any) -> bool:
    """Numbers compare by value (50 == "50.00"); text matches if it starts with the expected text."""
    a, b = _number(expected), _number(actual)
    if a is not None and b is not None:
        return a == b
    if actual is None:
        return False
    return str(actual).strip().lower().startswith(str(expected).strip().lower())


def fields_match(expected: dict, calls: list[dict]) -> list[str]:
    """expected: {op: {"path|alternative path": value}}. Returns the problems, empty if all match."""
    problems = []
    for op, fields in expected.items():
        call = next((c for c in calls if c["op"] == op), None)
        if call is None:
            problems.append(f"no {op} call was planned")
            continue
        values = {**call.get("query", {}), **call.get("body_values", {})}
        for spec, want in fields.items():
            paths = spec.split("|")
            if not any(value_matches(want, values.get(path)) for path in paths):
                found = {p: values.get(p) for p in paths}
                problems.append(f"{op} {spec}: got {found}, expected {want!r}")
    return problems


def computed_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(expected) == set(actual) and all(
            value_matches(v, actual[k]) for k, v in expected.items())
    if isinstance(actual, dict):
        return len(actual) == 1 and value_matches(expected, next(iter(actual.values())))
    return value_matches(expected, actual)


def answer_problems(expected: dict, text: str, data: dict) -> list[str]:
    problems = []
    computed = (data or {}).get("computed")
    if "computed_any" in expected:
        got = computed["result"] if computed else None
        if not any(computed_matches(e, got) for e in expected["computed_any"]):
            problems.append(f"computed {got!r}, expected one of {expected['computed_any']}")
    if expected.get("computed_none") and computed:
        problems.append(f"computed {computed['result']!r} although no calculation was asked for")
    if "sources_any" in expected:
        sources = text.split("Sources:", 1)[1] if "Sources:" in text else ""
        if not any(s.lower() in sources.lower() for s in expected["sources_any"]):
            problems.append(f"sources {sources.strip()!r}, expected one of {expected['sources_any']}")
    if expected.get("not_found") and "could not find" not in text.lower():
        problems.append("answered although the docs do not cover it")
    if "system_kind" in expected and (data or {}).get("kind") != expected["system_kind"]:
        problems.append(f"system search kind {(data or {}).get('kind')!r}, expected {expected['system_kind']!r}")
    return problems


# -- summaries ------------------------------------------------------------------------------

def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def summarise(rows: list[dict]) -> dict[str, Any]:
    """rows: one per case, each with "case" and "score" (and optional stats)."""
    def share(items, key):
        return round(sum(bool(r["score"][key]) for r in items) / len(items), 3) if items else None

    actions = [r for r in rows if r["case"]["expect"]["intent"] == "action"]
    should_ask = [r for r in rows if r["score"]["should_ask"]]
    should_not = [r for r in rows if not r["score"]["should_ask"]]
    planned = [r for r in rows if "plan_ok" in r["score"]]
    multi = [r for r in actions if len(r["case"]["expect"]["steps"]) > 1]
    deciders = Counter(d for r in rows for d in r.get("decided_by", []))
    fresh = [r["stats"]["fresh_seconds"] for r in rows if r.get("stats", {}).get("fresh")]
    return {
        "cases": len(rows),
        "intent": share(rows, "intent_ok"),
        "tool": share(actions, "tool_ok"),
        "fully_right": share(rows, "full_ok"),
        "plan": share(planned, "plan_ok"),
        "asked_when_should": round(sum(r["score"]["asked"] for r in should_ask) / len(should_ask), 3) if should_ask else None,
        "asked_needlessly": round(sum(r["score"]["asked"] for r in should_not) / len(should_not), 3) if should_not else None,
        "dropped_actions": sum(r["score"]["dropped"] for r in multi),
        "errors": sum(bool(r.get("error")) for r in rows),
        "decided_by": dict(deciders),
        "model_calls": sum(r.get("stats", {}).get("calls", 0) for r in rows),
        "tokens": sum(r.get("stats", {}).get("tokens", 0) for r in rows),
        "rejected_answers": sum(r.get("stats", {}).get("rejected", 0) for r in rows),
        "p50_seconds": percentile(fresh, 0.5),
        "p95_seconds": percentile(fresh, 0.95),
    }


def by_category(rows: list[dict]) -> dict[str, dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[r["case"]["category"]].append(r)
    return {name: {"cases": len(g), "fully_right": round(sum(r["score"]["full_ok"] for r in g) / len(g), 3)}
            for name, g in sorted(groups.items())}

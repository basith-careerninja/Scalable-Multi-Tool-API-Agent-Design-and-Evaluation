"""Ask an LLM to label every request in the Postman collection, for a person to review.

The build (build_model.py) needs judgments that cannot be read off the collection's
structure: resource and action names, which requests are hidden duplicates, path
parameters that are another ID under a different name (aliases), IDs a request returns
only in some cases (conditional links), test-scaffolding variables to ignore, and the
risk tier. This script asks the LLM for those, one call per Postman folder, and writes
data/paypal_review.json.

When a reference model exists (today's paypal_model.json), every proposal is compared
with it field by field:
  agreed        the proposal equals the current value
  rejected      automatic, for resource/action (operation IDs are fixed: the eval refers
                to them) and ignore_vars; the current value is kept
  needs_review  hidden_duplicate_of, aliases, conditional_produces, risk: a person sets
                accepted (use proposed), rejected (use current) or edited (use "final")

The run is resumable: finished folders are saved after each call, so a stopped run
continues where it left off and never repeats a paid call.

Usage:
  python propose_review.py                     propose (or continue), then print the review table
  python propose_review.py --table             print the table and agreement rates only, no calls
  python propose_review.py --estimate          print the calls and prompt sizes, no calls
  python propose_review.py --restart           discard earlier proposals and start again
  python propose_review.py --provider openai   choose the LLM (default: LLM_PROVIDER in .env)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from build_model import FIELDS, HARNESS_GROUP, REVIEW, ROOT, SRC, read_collection, request_key

REFERENCE = ROOT / "paypal_model.json"
# Fields where a disagreement keeps the current value without asking: operation IDs are fixed
# (frozen eval cases refer to them), and scaffolding lists only shape what is ignored.
AUTO_REJECT = ("resource", "action", "ignore_vars")

SYSTEM = """You label requests from an API collection so an agent can use them as tools.
For each request, reply with labels in a JSON object:
{"requests": [{"index": int, "resource": str, "action": str, "hidden_duplicate_of": str or null,
 "aliases": {path parameter: ID variable name}, "conditional_produces": [ID variable name],
 "ignore_vars": [variable name], "risk": "read" | "write" | "money" | "outbound" | "destructive"}]}

Rules:
- resource: the singular, lower-case, hyphenated noun the request acts on (e.g. "invoice", "payout-batch").
- action: a short hyphenated verb or verb phrase. Plain reads of a collection are "list", of one item
  "show"; plain writes are "create", "update" (PATCH), "replace" (PUT), "delete". Other actions use
  their own verb ("send", "cancel", "record-payment", "generate-number", "search").
- hidden_duplicate_of: if this request calls the same endpoint as another request (listed in
  same_endpoint_as) and only repeats it for testing or with a variant body, the name of the request
  to keep. Otherwise null. Never hide both.
- aliases: for a path parameter whose name differs from the variable the collection saves that same
  ID under (see saved_id_variables), map the parameter to that variable. Otherwise {}.
- conditional_produces: saved ID variables this request returns only in some cases (e.g. only when
  a payment is captured immediately), so a planner must not rely on it. Usually [].
- ignore_vars: variables this request uses or saves that are test scaffolding, not IDs passed between
  calls (dates, card numbers, URLs, tracking numbers, intents, test tokens). Usually [].
- risk: read = only reads (including searches); write = creates or changes the user's own data
  without moving money or contacting anyone; money = moves, reserves or returns money (payments,
  captures, authorizations, refunds, payouts, orders); outbound = contacts or notifies another person
  or party (sending, reminding, cancelling a sent invoice, dispute messages, offers, escalations);
  destructive = deletes something that cannot be restored.
Label every request listed, using its index."""


# -- facts and prompts ------------------------------------------------------------------------

def folders(rows: list[dict]) -> dict[str, list[int]]:
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["group"]].append(i)
    return dict(groups)


def request_facts(rows: list[dict]) -> list[dict]:
    """What the code knows about each request; the LLM only labels these."""
    endpoint = defaultdict(list)
    for row in rows:
        endpoint[(row["method"], row["path"])].append(row["name"])
    facts = []
    for i, row in enumerate(rows):
        body = row["body_template"]
        facts.append({
            "index": i, "name": row["name"], "folder": row["group"], "method": row["method"],
            "path": row["path"], "description": row["description"][:240],
            "path_params": row["path_params"], "query_params": [q["name"] for q in row["query"]],
            "body_fields": list(body)[:15] if isinstance(body, dict) else [],
            "uses_vars": row["uses"],
            "saves_vars": [{"name": n, "from_response": row["extract"].get(n)} for n in row["sets"]],
            "same_endpoint_as": [n for n in endpoint[(row["method"], row["path"])] if n != row["name"]],
        })
    return facts


def folder_prompt(facts: list[dict], indexes: list[int], saved: list[str]) -> str:
    chosen = [facts[i] for i in indexes]
    return (f"saved_id_variables (all variables any request in the collection saves): {json.dumps(saved)}\n\n"
            f"Requests to label:\n{json.dumps(chosen, indent=1)}")


def saved_variables(rows: list[dict]) -> list[str]:
    return sorted({n for row in rows for n in row["sets"]})


# -- the reference (today's model) ------------------------------------------------------------

def current_values(rows: list[dict], model: dict) -> dict[str, dict]:
    """request key -> today's value of every reviewed field, read back from a built model."""
    ops = {(o["name"], o["method"], o["path"]): o for o in model.get("operations", [])}
    aliases = model.get("aliases", {})
    flow = model.get("id_flow", {})
    out = {}
    for row in rows:
        op = ops.get((row["name"], row["method"], row["path"]))
        if op is None:
            continue
        own = {**aliases, **op.get("path_aliases", {})}
        consumed, produced = set(op["consumes"]), set(op["produces"])
        ignore = {raw for raw in row["path_params"] + row["uses"] if own.get(raw, raw) not in consumed}
        if row["method"] != "DELETE":
            ignore |= {n for n in row["sets"] if n not in row["uses"] and n not in produced}
        duplicate = None
        if op["id"].endswith(".excluded-duplicate"):
            names = model.get("duplicate_endpoints", {}).get(f"{row['method']} {row['path']}", [])
            duplicate = next((n for n in names if n != row["name"]), None)
        values = {
            "resource": op["resource"], "action": op["action"], "hidden_duplicate_of": duplicate,
            "aliases": {p: own[p] for p in row["path_params"] if p in own},
            "conditional_produces": sorted(n for n in op["produces"] if op["id"] in flow.get(n, {}).get("conditional", [])),
            "ignore_vars": sorted(ignore),
        }
        if row["group"] != HARNESS_GROUP:
            values["risk"] = op["risk"]
        out[request_key(row)] = values
    return out


# -- comparing proposals with the reference -----------------------------------------------------

def normalise(field: str, value):
    if field in ("aliases",):
        return {str(k): str(v) for k, v in sorted((value or {}).items())} if isinstance(value, dict) else {}
    if field in ("conditional_produces", "ignore_vars"):
        return sorted({str(v) for v in value}) if isinstance(value, list) else []
    if field == "hidden_duplicate_of":
        return str(value).strip() if isinstance(value, str) and value.strip() else None
    return str(value).strip() if value is not None else None


def compare(proposed: dict, current: dict | None, harness: bool) -> dict[str, dict]:
    """Per-field entries with a status. Without a reference every field needs review."""
    fields = {}
    for field in FIELDS:
        if field == "risk" and harness:
            continue  # the auth harness is handled by code
        value = normalise(field, proposed.get(field))
        entry = {"proposed": value}
        if current is None or field not in current:
            entry["status"] = "needs_review"
        elif current[field] == value:
            entry["status"] = "agreed"
        else:
            entry["current"] = current[field]
            entry["status"] = "rejected" if field in AUTO_REJECT else "needs_review"
            if field in AUTO_REJECT:
                entry["note"] = "automatic: keeps the current value"
        fields[field] = entry
    return fields


def agreement(review: dict) -> dict[str, dict]:
    """Per field: how many proposals agreed with the reference, of those that have one."""
    counts = defaultdict(lambda: {"agreed": 0, "total": 0})
    for entry in review.get("requests", []):
        fields = entry.get("fields") or {}
        for field, value in fields.items():
            if value.get("status") == "agreed" or "current" in value:
                counts[field]["total"] += 1
                counts[field]["agreed"] += value.get("status") == "agreed" and "current" not in value
        names = [fields.get(f, {}) for f in ("resource", "action")]
        if all(n for n in names):
            counts["names (resource and action)"]["total"] += 1
            counts["names (resource and action)"]["agreed"] += all(n.get("status") == "agreed" for n in names)
    return dict(counts)


# -- proposing ----------------------------------------------------------------------------------

def propose(collection: dict, reference: dict | None, llm, existing: dict | None = None,
            save=None, log=print) -> dict:
    """Label every request, one LLM call per folder. Finished folders in `existing` are kept."""
    rows = read_collection(collection)
    facts, saved = request_facts(rows), saved_variables(rows)
    current = current_values(rows, reference) if reference else {}
    review = existing or {"source": collection["info"]["name"], "folders_done": [], "requests": []}
    review.setdefault("folders_done", [])
    done = set(review["folders_done"])
    entries = {e["key"]: e for e in review["requests"]}
    for folder, indexes in folders(rows).items():
        if folder in done:
            continue
        reply = llm.json(SYSTEM, folder_prompt(facts, indexes, saved))
        labels = {r.get("index"): r for r in (reply.get("requests") or []) if isinstance(r, dict)}
        missing = [i for i in indexes if i not in labels]
        if missing:
            # Ask once more for just the ones left out; anything still missing is reviewed by a person.
            again = llm.json(SYSTEM, folder_prompt(facts, missing, saved))
            labels.update({r.get("index"): r for r in (again.get("requests") or []) if isinstance(r, dict)})
        for i in indexes:
            row = rows[i]
            key = request_key(row)
            entries[key] = {
                "key": key, "group": row["group"], "name": row["name"], "method": row["method"], "path": row["path"],
                "fields": compare(labels.get(i, {}), current.get(key) if reference else None,
                                  harness=row["group"] == HARNESS_GROUP),
            }
        review["requests"] = [entries[request_key(r)] for r in rows if request_key(r) in entries]
        review["folders_done"].append(folder)
        review["summary"] = {"agreement": agreement(review)}
        if save:
            save(review)
        log(f"  {folder}: {len(indexes)} requests labelled")
    return review


def disagreements(review: dict) -> list[tuple[str, str, object, object]]:
    """(request, field, proposed, current) for every field a person must decide."""
    return [(e["key"], field, v.get("proposed"), v.get("current"))
            for e in review.get("requests", []) for field, v in e["fields"].items()
            if v.get("status") == "needs_review"]


def print_report(review: dict) -> None:
    print("\nAgreement with today's model, per field:")
    for field, c in sorted(review.get("summary", {}).get("agreement", {}).items()):
        print(f"  {field:28} {c['agreed']:>3} of {c['total']}")
    auto = [(e["key"], f, v.get("proposed"), v.get("current")) for e in review.get("requests", [])
            for f, v in e["fields"].items() if v.get("note", "").startswith("automatic")]
    print(f"\nAutomatically rejected (current value kept): {len(auto)}")
    for key, field, proposed, current in auto:
        print(f"  {key}  {field}: proposed {proposed!r}, kept {current!r}")
    rows = disagreements(review)
    print(f"\nNeeds your decision: {len(rows)}")
    for key, field, proposed, current in rows:
        print(f"  {key}\n    {field}: proposed {json.dumps(proposed)}  current {json.dumps(current)}")


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", action="store_true", help="print the review table only")
    parser.add_argument("--estimate", action="store_true", help="print calls and prompt sizes only")
    parser.add_argument("--restart", action="store_true", help="discard earlier proposals")
    parser.add_argument("--provider", choices=("groq", "gemini", "openai"))
    parser.add_argument("--model")
    args = parser.parse_args(argv)
    collection = json.loads(SRC.read_text(encoding="utf-8"))
    rows = read_collection(collection)
    if args.estimate:
        facts, saved = request_facts(rows), saved_variables(rows)
        sizes = [len(SYSTEM) + len(folder_prompt(facts, idx, saved)) for idx in folders(rows).values()]
        print(f"{len(sizes)} calls (one per folder); prompt characters: total {sum(sizes)}, "
              f"largest {max(sizes)}; about {sum(sizes) // 4} input tokens")
        return 0
    existing = None if args.restart or not REVIEW.is_file() else json.loads(REVIEW.read_text(encoding="utf-8"))
    if args.table:
        if existing is None:
            print(f"{REVIEW.relative_to(ROOT)} does not exist yet.")
            return 1
        print_report(existing)
        return 0

    from agent.llm import ChatClient, LLMError, load_env
    load_env()
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["LLM_MODEL"] = args.model
    llm = ChatClient(timeout=180.0)
    reference = json.loads(REFERENCE.read_text(encoding="utf-8")) if REFERENCE.is_file() else None

    def save(review):
        review["generated_by"] = {"model": llm.model,
                                  "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        REVIEW.write_text(json.dumps(review, indent=1) + "\n", encoding="utf-8")

    print(f"Proposing labels with {llm.model}; saving to {REVIEW.relative_to(ROOT)} after each folder.")
    try:
        review = propose(collection, reference, llm, existing, save=save)
    except LLMError as err:
        print(f"Stopped: {err}\nFinished folders are saved; run again to continue.")
        return 1
    print_report(review)
    return 0


if __name__ == "__main__":
    sys.exit(main())

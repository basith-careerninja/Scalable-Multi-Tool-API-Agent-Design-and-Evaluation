"""Build the capability model from a Postman collection export (v2.1) and the reviewed labels.

Input : data/PayPal APIs.postman_collection.json
        data/paypal_review.json   (LLM proposals, reviewed by a person; see propose_review.py)
Output: paypal_model.json         (operations, resources, ID-flow graph, gaps)

Facts come from the collection: paths, methods, bodies, examples, and which variables each
request uses and which its test scripts save. Judgments come from the review file: resource
and action names, hidden duplicates, aliases, conditional ID links, scaffolding variables to
ignore, and risk tiers. No vendor-specific rules live in this file.

Collection variables (which may hold credentials) are never copied to the output.

Usage:
  python build_model.py              build, write paypal_model.json, print the diff
  python build_model.py --dry-run    print the diff against the existing model, write nothing
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "data" / "PayPal APIs.postman_collection.json"
REVIEW = ROOT / "data" / "paypal_review.json"
OUT = ROOT / "paypal_model.json"

# Credentials and the auth harness are handled by code, never by the model or the review.
HARNESS_VARS = {"base_url", "client_id", "client_secret", "access_token"}
HARNESS_GROUP = "Authorization"

TIERS = ("read", "write", "money", "outbound", "destructive")
# Approval level per tier (agent/policy.py): a lower level needs less from the user.
TIER_LEVEL = {"read": 0, "write": 1, "money": 2, "outbound": 2, "destructive": 2}
FIELDS = ("resource", "action", "hidden_duplicate_of", "aliases", "conditional_produces", "ignore_vars", "risk")
STATUSES = ("agreed", "accepted", "rejected", "edited", "needs_review")

VAR = re.compile(r"\{\{([\w-]+)\}\}")
SET_CALL = re.compile(r"(?:environment|collectionVariables|globals)\.set\(\s*[\"']([\w-]+)[\"']\s*,\s*([^;\n]+)")
# Only plain reads of the response body can be replayed outside Postman.
RESPONSE_PATH = re.compile(r"^(?:jsonData|data)\.([\w.\[\]]+)\)?\s*$")


class ReviewError(SystemExit):
    pass


# -- reading the collection -------------------------------------------------------

def walk(items, folders=()):
    for item in items:
        if "item" in item:
            yield from walk(item["item"], folders + (item["name"],))
        else:
            yield folders, item


def clean_text(text, limit=300):
    text = text if isinstance(text, str) else (text or {}).get("content", "")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def parse_body(raw):
    """Parse a raw JSON body that may contain unquoted {{variables}}. None if not JSON."""
    if not raw or not raw.strip():
        return None
    for candidate in (raw, re.sub(r'(?<!")\{\{([\w$-]+)\}\}(?!")', r'"{{\1}}"', raw)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def first_success(responses):
    for response in responses:
        if 200 <= (response.get("code") or 0) < 300:
            body = response.get("body")
            try:
                parsed = json.loads(body) if body else None
            except json.JSONDecodeError:
                parsed = None
            return response["code"], parsed
    return None, None


def is_param(seg):
    return seg.startswith(":") or seg.startswith("{{")


def param_name(seg):
    return seg.strip(":{}")


def read_request(folders, item):
    request = item["request"]
    url = request["url"]
    segments = url.get("path") or []
    path = "/" + "/".join(segments) if isinstance(segments, list) else str(segments)
    body = request.get("body") or {}
    raw = body.get("raw", "") if body.get("mode") == "raw" else ""
    scripts = "\n".join("\n".join(e["script"].get("exec", [])) for e in item.get("event", []))

    sets, extract = [], {}
    for name, expr in SET_CALL.findall(scripts):
        if name not in sets:
            sets.append(name)
        match = RESPONSE_PATH.match(expr.strip())
        if match:
            extract[name] = match.group(1)

    uses = []
    for name in VAR.findall(url.get("raw", "") + raw):
        if name not in HARNESS_VARS and name not in uses:
            uses.append(name)

    status, example = first_success(item.get("response", []))
    return {
        "group": " / ".join(folders),
        "name": item["name"],
        "method": request["method"],
        "path": path,
        "path_params": [param_name(s) for s in path.split("/") if is_param(s)],
        "description": clean_text(request.get("description")),
        "query": [
            {"name": q["key"], "example": q.get("value") or "", "description": clean_text(q.get("description"), 120)}
            for q in url.get("query", []) if q.get("key")
        ],
        "uses": uses,
        "sets": sets,
        "extract": extract,
        "body_mode": body.get("mode"),
        "body_template": parse_body(raw),
        "example_status": status,
        "example_body": example,
    }


def read_collection(collection: dict) -> list[dict]:
    return [read_request(folders, item) for folders, item in walk(collection["item"])]


def request_key(row: dict) -> str:
    """Stable identity of a request across the collection, the review file and the model."""
    return f"{row['method']} {row['path']} | {row['name']}"


# -- reading the review ---------------------------------------------------------------

def final_value(key: str, field: str, entry: dict):
    """The value the build uses for one reviewed field, after checking the review is complete and honest."""
    status = entry.get("status")
    where = f"{key}: {field}"
    if status not in STATUSES:
        raise ReviewError(f"{where} has status {status!r}; use one of {list(STATUSES)}")
    if status == "needs_review":
        raise ReviewError(f"{where} still needs review (set status to accepted, rejected or edited)")
    has_current = "current" in entry
    if status == "agreed":
        if has_current and entry["current"] != entry.get("proposed"):
            raise ReviewError(f"{where} is marked agreed but the proposal differs from the current value")
        value = entry.get("proposed")
    elif status == "accepted":
        value = entry.get("proposed")
    elif status == "rejected":
        if not has_current:
            raise ReviewError(f"{where} is rejected but has no current value to fall back on; use edited")
        value = entry["current"]
    else:  # edited
        if "final" not in entry:
            raise ReviewError(f"{where} is edited but has no 'final' value")
        value = entry["final"]
    if field == "risk":
        if value not in TIERS:
            raise ReviewError(f"{where}: unknown risk tier {value!r}; use one of {list(TIERS)}")
        # Never let a tier drop to a weaker approval rule unless a person explicitly decided it.
        current = entry.get("current")
        if current in TIER_LEVEL and TIER_LEVEL[value] < TIER_LEVEL[current] and status not in ("accepted", "edited"):
            raise ReviewError(f"{where}: lowering {current} to {value} needs status accepted or edited")
    return value


def reviewed_labels(rows: list[dict], review: dict) -> dict[str, dict]:
    """request key -> {field: final value, "_names_by": "llm"|"reviewer"}. Fails on anything missing."""
    entries = {r["key"]: r for r in review.get("requests", [])}
    labels = {}
    for row in rows:
        key = request_key(row)
        entry = entries.get(key)
        if entry is None:
            raise ReviewError(f"no review entry for {key}; run propose_review.py")
        fields = entry.get("fields") or {}
        missing = [f for f in FIELDS if f not in fields and not (f == "risk" and row["group"] == HARNESS_GROUP)]
        if missing:
            raise ReviewError(f"{key}: review entry lacks {missing}")
        values = {f: final_value(key, f, fields[f]) for f in FIELDS if f in fields}
        names_ok = all(fields[f].get("status") in ("agreed", "accepted") for f in ("resource", "action"))
        values["_names_by"] = "llm" if names_ok else "reviewer"
        labels[key] = values
    return labels


def load_review(path: Path = REVIEW) -> dict:
    if not path.is_file():
        raise ReviewError(f"{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path} is missing. Run "
                          "propose_review.py to create it, review every needs_review field, then build again.")
    return json.loads(path.read_text(encoding="utf-8"))


# -- building ---------------------------------------------------------------------

def build(collection: dict, review: dict) -> dict:
    rows = read_collection(collection)
    labels = reviewed_labels(rows, review)

    # A parameter aliased to one ID everywhere goes in the shared map; one that names
    # different IDs on different requests is kept on each operation as path_aliases.
    targets = defaultdict(set)
    for row in rows:
        for param, id_name in (labels[request_key(row)]["aliases"] or {}).items():
            if param == id_name:
                raise ReviewError(f"{request_key(row)}: alias {param!r} -> {id_name!r} maps a name to itself")
            targets[param].add(id_name)
    aliases = {param: next(iter(ids)) for param, ids in targets.items() if len(ids) == 1}

    seen_endpoint = defaultdict(list)
    conditional = set()
    operations = []
    for row in rows:
        label = labels[request_key(row)]
        method, path = row["method"], row["path"]
        ignore = set(label["ignore_vars"] or ())
        path_aliases = {p: n for p, n in (label["aliases"] or {}).items() if p not in aliases}
        consumes = []
        for raw in row["path_params"] + row["uses"]:
            name = path_aliases.get(raw) or aliases.get(raw, raw)
            if raw not in ignore and name not in ignore and name not in consumes:
                consumes.append(name)
        # A request that reads a value and re-saves it is not a real producer,
        # and a DELETE only clears the saved value.
        produces = [] if method == "DELETE" else [
            n for n in row["sets"] if n not in ignore and n not in row["uses"]
        ]
        op_id = f"{label['resource']}.{label['action']}"
        risk = "harness" if row["group"] == HARNESS_GROUP else label["risk"]
        hidden = bool(label["hidden_duplicate_of"])
        exposed = risk != "harness" and not hidden
        if hidden:
            op_id += ".excluded-duplicate"
        for name in label["conditional_produces"] or ():
            conditional.add((op_id, name))
        seen_endpoint[(method, path)].append(row["name"])
        template = row["body_template"]
        extra = {"path_aliases": path_aliases} if path_aliases else {}
        operations.append({
            "id": op_id,
            "exposed": exposed,
            "name": row["name"],
            "group": row["group"],
            "description": row["description"],
            "method": method,
            "path": path,
            "resource": label["resource"],
            "action": label["action"],
            "risk": risk,
            "consumes": consumes,
            **extra,
            "produces": produces,
            "extract": {k: v for k, v in row["extract"].items() if k in produces},
            "query": row["query"],
            "body_mode": row["body_mode"],
            "body_fields": list(template) if isinstance(template, dict) else [],
            "body_template": template,
            "example_status": row["example_status"],
            "example_body": row["example_body"],
            "classified_by": label["_names_by"],
        })

    duplicates = {f"{m} {p}": names for (m, p), names in seen_endpoint.items() if len(names) > 1}
    tools = [op for op in operations if op["exposed"]]
    ids = [op["id"] for op in tools]
    clashes = sorted({i for i in ids if ids.count(i) > 1})
    if clashes:
        raise ReviewError(f"operation ids are not unique: {clashes}")

    producers, consumers = defaultdict(list), defaultdict(list)
    for op in tools:
        for name in op["produces"]:
            producers[name].append(op["id"])
        for name in op["consumes"]:
            consumers[name].append(op["id"])

    id_flow = {
        name: {
            "produced_by": producers.get(name, []),
            "conditional": [p for p in producers.get(name, []) if (p, name) in conditional],
            "consumed_by": consumers.get(name, []),
        }
        for name in sorted(set(producers) | set(consumers))
    }
    gaps = sorted(n for n, v in id_flow.items() if v["consumed_by"] and not v["produced_by"])

    resources = defaultdict(lambda: {"actions": [], "groups": set()})
    for op in tools:
        resources[op["resource"]]["actions"].append(op["action"])
        resources[op["resource"]]["groups"].add(op["group"])
    resources = {
        name: {"groups": sorted(v["groups"]), "actions": sorted(v["actions"])}
        for name, v in sorted(resources.items())
    }

    risk_counts = defaultdict(int)
    for op in operations:
        risk_counts[op["risk"]] += 1

    return {
        "source": collection["info"]["name"] + " (Postman collection v2.1 export)",
        "summary": {
            "requests": len(operations),
            "tools_exposed_to_model": len(tools),
            "resources": len(resources),
            "max_actions_per_resource": max(len(v["actions"]) for v in resources.values()),
            "named_by_llm": sum(op["classified_by"] == "llm" for op in operations),
            "named_by_reviewer": sum(op["classified_by"] == "reviewer" for op in operations),
            "with_json_body_template": sum(op["body_template"] is not None for op in tools),
            "with_example_response": sum(op["example_status"] is not None for op in tools),
            "risk": dict(sorted(risk_counts.items())),
        },
        "aliases": dict(sorted(aliases.items())),
        "resources": resources,
        "id_flow": id_flow,
        "ids_consumed_but_never_produced": gaps,
        "duplicate_endpoints": duplicates,
        "operations": operations,
    }


# -- comparing ------------------------------------------------------------------------

_OP_FIELDS = ("exposed", "resource", "action", "risk", "consumes", "path_aliases", "produces", "extract")


def model_diff(old: dict, new: dict) -> list[str]:
    """Human-readable differences between two models. Empty when they match."""
    lines = []
    old_ops = {(o["name"], o["method"], o["path"]): o for o in old.get("operations", [])}
    new_ops = {(o["name"], o["method"], o["path"]): o for o in new.get("operations", [])}
    for key in sorted(old_ops.keys() - new_ops.keys()):
        lines.append(f"request removed: {key[1]} {key[2]} ({key[0]})")
    for key in sorted(new_ops.keys() - old_ops.keys()):
        lines.append(f"request added: {key[1]} {key[2]} ({key[0]})")
    relabelled = defaultdict(int)
    for key in sorted(old_ops.keys() & new_ops.keys()):
        a, b = old_ops[key], new_ops[key]
        if a["id"] != b["id"]:
            lines.append(f"operation id {a['id']} -> {b['id']} ({key[0]})")
        for field in _OP_FIELDS:
            if a.get(field) != b.get(field):
                lines.append(f"{b['id']}.{field}: {a.get(field)!r} -> {b.get(field)!r}")
        if a.get("classified_by") != b.get("classified_by"):
            relabelled[(a.get("classified_by"), b.get("classified_by"))] += 1
    for (before, after), n in sorted(relabelled.items(), key=lambda kv: str(kv[0])):
        lines.append(f"classified_by {before!r} -> {after!r} on {n} operations")
    for section in ("aliases", "ids_consumed_but_never_produced"):
        if old.get(section) != new.get(section):
            lines.append(f"{section}: {old.get(section)!r} -> {new.get(section)!r}")
    old_flow, new_flow = old.get("id_flow", {}), new.get("id_flow", {})
    for name in sorted(set(old_flow) | set(new_flow)):
        a, b = old_flow.get(name), new_flow.get(name)
        if a != b:
            for part in ("produced_by", "conditional", "consumed_by"):
                pa, pb = (a or {}).get(part, []), (b or {}).get(part, [])
                if sorted(pa) != sorted(pb):
                    lines.append(f"id_flow {name}.{part}: {sorted(pa)} -> {sorted(pb)}")
    old_sum, new_sum = old.get("summary", {}), new.get("summary", {})
    for k in sorted(set(old_sum) | set(new_sum)):
        if old_sum.get(k) != new_sum.get(k):
            lines.append(f"summary.{k}: {old_sum.get(k)!r} -> {new_sum.get(k)!r}")
    return lines


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print the diff, write nothing")
    parser.add_argument("--review", type=Path, default=REVIEW)
    args = parser.parse_args(argv)
    collection = json.loads(SRC.read_text(encoding="utf-8"))
    try:
        model = build(collection, load_review(args.review))
    except ReviewError as err:
        print(f"Build stopped: {err.code}")
        return 1
    old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.is_file() else {}
    diff = model_diff(old, model)
    print(f"Differences from the existing {OUT.name}: {len(diff) or 'none'}")
    for line in diff:
        print("  " + line)
    if args.dry_run:
        print("Dry run: nothing written.")
        return 0
    OUT.write_text(json.dumps(model, indent=1), encoding="utf-8")
    print(json.dumps(model["summary"], indent=2))
    tools = [op for op in model["operations"] if op["exposed"]]
    print("Consumed but never produced:", model["ids_consumed_but_never_produced"])
    print("Duplicate endpoints:", list(model["duplicate_endpoints"]))
    unparsed = [op["id"] for op in tools if op["body_mode"] == "raw" and op["body_template"] is None]
    print("Raw bodies that are not JSON:", unparsed)
    print("Non-JSON body modes:", [(op["id"], op["body_mode"]) for op in tools if op["body_mode"] not in (None, "raw")])
    print("No example response:", [op["id"] for op in tools if op["example_status"] is None])
    return 0


if __name__ == "__main__":
    sys.exit(main())

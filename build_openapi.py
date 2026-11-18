"""Build a capability model from a public OpenAPI spec (Stripe, Square, GitHub), for the scaling eval.

Input : data/specs/<service>.json        (fetched by scripts/fetch_specs.py, pinned commits)
Output: data/specs/<service>_model.json  (same shape as paypal_model.json, plus "service")

PayPal's model comes from a Postman collection whose test scripts say which request saves
which ID, and every label was reviewed by a person. These specs have neither, so everything
here is inferred by fixed rules, and every operation is marked classified_by "rules":

  resource   GitHub: the spec's own sub-category (x-github). Stripe, Square: from the path,
             the last one or two collection names ("customer-source", "issuing-card").
  action     GitHub, Square: the spec's operationId. Stripe: list / show / create / update /
             delete from the method and path shape, or the path's verb ("pay", "cancel").
  consumes   one ID per path parameter: <service>_<name>_id. A generic "{id}" is named after
             the collection before it ("/sources/{id}" -> stripe_source_id).
  produces   a POST to a collection returns the ID its item path takes, read from the response
             schema ("id", "number", or "payment.id" when the response wraps the object).
             Marked inferred: no request was ever run to confirm it.
  risk       by rule, leaning strict: GET and searches read; DELETE, cancel, void and revoke
             destructive; payments, refunds, payouts, transfers money; send, invite, notify
             outbound; everything else write. Not reviewed by a person.

Operation and resource names carry the service ("stripe.customer.create"), so a Stripe
invoice and a PayPal invoice are different tools with different IDs.

Usage:
  python build_openapi.py                build all three
  python build_openapi.py stripe         build one
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from build_model import clean_text

ROOT = Path(__file__).parent
SPECS = ROOT / "data" / "specs"
METHODS = ("get", "post", "put", "patch", "delete")

SERVICES = {
    "stripe": {"title": "Stripe", "skip_path": re.compile(r"^/v1/test_helpers/")},
    "square": {"title": "Square", "skip_path": re.compile(r"^/oauth2/")},
    "github": {"title": "GitHub", "skip_path": None},
}

# A path's last segment that names an action on the item before it, not a collection.
VERBS = {
    "search", "cancel", "capture", "confirm", "pay", "void", "finalize", "send", "refund", "verify",
    "approve", "reject", "close", "publish", "resume", "pause", "reopen", "accept", "decline", "expire",
    "activate", "deactivate", "archive", "unarchive", "mark_uncollectible", "apply", "attach", "detach",
    "reverse", "submit", "complete", "clone", "upcoming", "preview", "retrieve", "query", "count",
    "increment_authorization", "fail", "post", "return", "reject", "unpublish", "redeem", "swap",
}
GENERIC_PARAMS = {"id", "uid", "key", "token"}
VERSION = re.compile(r"^v\d+$")

MONEY = re.compile(r"refund|payout|payment(?![-_ ]?method)|charge|capture|transfer|withdraw|top[-_ ]?up|"
                   r"disburse|\bpay\b|\bpay[-_]|invoice.*\bpay|outbound_payment|debit|credit_note|"
                   r"authorization|terminal.*checkout|gift[-_ ]?card.*activity|cash[-_ ]drawer", re.I)
OUTBOUND = re.compile(r"\bsend|invite|invitation|notify|notification|email|message|remind|dispatch|publish", re.I)
DESTRUCTIVE = re.compile(r"\b(cancel|void|revoke|terminate|expire|reject|deactivate|disable|remove|delete)\b", re.I)


# -- spec helpers ---------------------------------------------------------------------

def resolve(spec: dict, node, depth: int = 0):
    """Follow a local $ref ("#/components/schemas/X"). Returns the node itself if there is none."""
    while isinstance(node, dict) and "$ref" in node and depth < 20:
        target = spec
        for part in node["$ref"].lstrip("#/").split("/"):
            target = target.get(part, {}) if isinstance(target, dict) else {}
        node, depth = target, depth + 1
    return node


def properties(spec: dict, schema) -> dict:
    """Top-level properties of a schema, merging allOf / anyOf branches."""
    schema = resolve(spec, schema)
    if not isinstance(schema, dict):
        return {}
    props = dict(schema.get("properties") or {})
    for key in ("allOf", "anyOf", "oneOf"):
        for branch in schema.get(key) or ():
            if key != "allOf" and props:
                break
            props.update(properties(spec, branch))
    return props


def body_schema(op: dict):
    content = (op.get("requestBody") or {}).get("content") or {}
    for kind in ("application/json", "application/x-www-form-urlencoded", "multipart/form-data"):
        if kind in content:
            return kind, content[kind].get("schema")
    return None, None


def response_schema(op: dict):
    for code, response in (op.get("responses") or {}).items():
        if str(code).startswith("2"):
            content = (response or {}).get("content") or {}
            for kind, value in content.items():
                if "json" in kind:
                    return value.get("schema")
    return None


def singular(word: str) -> str:
    word = word.replace("_", "-")
    for suffix, repl in (("ies", "y"), ("sses", "ss"), ("shes", "sh"), ("ches", "ch"), ("xes", "x")):
        if word.endswith(suffix):
            return word[: -len(suffix)] + repl
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def kebab(text: str) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", text)
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def humanize(op_id: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", op_id).capitalize()


def is_param(segment: str) -> bool:
    return segment.startswith("{")


def segments(path: str) -> list[str]:
    parts = [p for p in path.split("/") if p]
    return parts[1:] if parts and VERSION.match(parts[0]) else parts


# -- naming ----------------------------------------------------------------------------

def action_segment(path: str, method: str, all_paths: set[str]) -> str | None:
    """The last segment if it names an action ("/invoices/{invoice}/pay"), else None."""
    segs = segments(path)
    if not segs or is_param(segs[-1]):
        return None
    last = segs[-1]
    deeper = any(p.startswith(path + "/") for p in all_paths)
    if deeper:
        return None
    if last in VERBS or last.startswith(("batch", "bulk")):
        return last
    if len(segs) >= 2 and is_param(segs[-2]) and method == "post":
        return last
    return None


def path_resource(path: str, action: str | None) -> str:
    statics = [s for s in segments(path) if not is_param(s)]
    if action and statics and statics[-1] == action:
        statics = statics[:-1]
    statics = [singular(s) for s in statics] or ["root"]
    return "-".join(statics[-2:])


def id_name(service: str, path: str, index: int) -> str:
    """ID name for the path parameter at segment `index`."""
    segs = path.split("/")
    param = segs[index].strip("{}")
    base = re.sub(r"(_exposed_id|_id|_number|_uid)$", "", param)
    if service != "github" and index > 1 and not is_param(segs[index - 1]) and not (
            param.endswith("_id") and param not in GENERIC_PARAMS):
        # Stripe and Square: name the ID after the collection the item lives in, namespace included,
        # so /climate/products/{product} is not mistaken for /products/{id}.
        return f"{service}_{path_resource('/'.join(segs[:index]), None).replace('-', '_')}_id"
    if param in GENERIC_PARAMS or not base:
        before = next((s for s in reversed(segs[:index]) if s and not is_param(s) and not VERSION.match(s)), "item")
        base = singular(before)
    return f"{service}_{base.replace('-', '_')}_id"


def consumed(service: str, path: str) -> list[str]:
    names = []
    for i, seg in enumerate(path.split("/")):
        if is_param(seg):
            name = id_name(service, path, i)
            if name not in names:
                names.append(name)
    return names


def find_id_path(spec: dict, schema, prefer: str) -> str | None:
    props = properties(spec, schema)
    if prefer in props:
        return prefer
    if "id" in props:
        return "id"
    for key, value in props.items():
        inner = properties(spec, value)
        if prefer in inner or "id" in inner:
            return f"{key}.{prefer if prefer in inner else 'id'}"
    return None


def risk_for(method: str, path: str, action: str, text: str) -> str:
    words = f"{path} {action} {text}"
    if method == "get" or re.match(r"(bulk-|batch-)?(list|search|retrieve|get|query|count)(-|$)", action):
        return "read"
    if method == "delete":
        return "destructive"
    if MONEY.search(f"{path} {action}"):
        return "money"
    if DESTRUCTIVE.search(action.replace("-", " ").replace("_", " ")):
        return "destructive"
    if OUTBOUND.search(f"{path} {action}") or OUTBOUND.search(text[:80]):
        return "outbound"
    return "write"


# -- build -----------------------------------------------------------------------------

def read_ops(service: str, spec: dict) -> list[dict]:
    skip = SERVICES[service]["skip_path"]
    rows = []
    for path, item in spec["paths"].items():
        if skip and skip.search(path):
            continue
        for method in METHODS:
            op = item.get(method)
            if not op or op.get("deprecated"):
                continue
            rows.append({"path": path, "method": method, "op": op,
                         "params": (item.get("parameters") or []) + (op.get("parameters") or [])})
    return rows


def build(service: str, spec: dict, source: str | None = None) -> dict:
    rows = read_ops(service, spec)
    all_paths = {r["path"] for r in rows}
    by_path = defaultdict(set)
    for r in rows:
        by_path[r["path"]].add(r["method"])

    # resource and action names
    for r in rows:
        op, path, method = r["op"], r["path"], r["method"]
        if service == "github":
            info = op.get("x-github") or {}
            r["resource"] = info.get("subcategory") or info.get("category") or "misc"
            r["category"] = info.get("category") or ""
            r["action"] = op["operationId"].split("/", 1)[-1]
            continue
        verb = action_segment(path, method, all_paths)
        r["resource"] = path_resource(path, verb)
        if service == "square":
            r["action"] = kebab(op["operationId"])
        elif verb:
            r["action"] = verb.replace("_", "-")
        else:
            item = segments(path) and is_param(segments(path)[-1])
            r["action"] = {("get", True): "show", ("get", False): "list", ("post", False): "create",
                           ("post", True): "update", ("patch", True): "update", ("patch", False): "update",
                           ("put", True): "replace", ("put", False): "replace",
                           ("delete", True): "delete", ("delete", False): "delete"}[(method, bool(item))]
    if service == "github":
        # The same sub-category name under two categories would merge two resources.
        cats = defaultdict(set)
        for r in rows:
            cats[r["resource"]].add(r["category"])
        for r in rows:
            if len(cats[r["resource"]]) > 1:
                r["resource"] = f"{r['category']}-{r['resource']}"
    # Any (resource, action) clash falls back to a name from the operationId, which is unique.
    clash = Counter((r["resource"], r["action"]) for r in rows)
    for r in rows:
        if clash[(r["resource"], r["action"])] > 1:
            r["action"] = kebab(r["op"]["operationId"])

    operations = []
    for r in rows:
        op, path, method = r["op"], r["path"], r["method"]
        resource = f"{service}.{r['resource']}"
        op_id = f"{resource}.{r['action']}"
        consumes = consumed(service, path)
        produces, extract = [], {}
        segs = segments(path)
        verb = action_segment(path, method, all_paths) if service != "github" else None
        if method == "post" and segs and not is_param(segs[-1]) and not verb:
            # Creating in a collection: the new item's ID is the one its item path takes.
            child = next((p for p in all_paths if p.startswith(path + "/{") and p.count("/") == path.count("/") + 1), None)
            name = consumed(service, child)[-1] if child else f"{service}_{singular(segs[-1]).replace('-', '_')}_id"
            if name not in consumes:
                prefer = "number" if child and child.rstrip("}").endswith("_number") else "id"
                where = find_id_path(spec, response_schema(op), prefer)
                if where:
                    produces, extract = [name], {name: where}
        kind, schema = body_schema(op)
        query = [{"name": p.get("name"), "example": "", "description": clean_text(p.get("description"), 120)}
                 for p in (resolve(spec, p) for p in r["params"]) if isinstance(p, dict) and p.get("in") == "query"]
        name = op.get("summary") or humanize(op["operationId"])
        if service == "square" and name == op["operationId"]:
            name = humanize(op["operationId"])
        description = clean_text(op.get("description") or "", 300)
        operations.append({
            "id": op_id,
            "exposed": True,
            "service": service,
            "name": name,
            "group": r.get("category") or (op.get("tags") or [""])[0],
            "description": description,
            "method": method.upper(),
            "path": path,
            "resource": resource,
            "action": r["action"],
            "risk": risk_for(method, path, r["action"], f"{name} {description}"),
            "consumes": consumes,
            "produces": produces,
            "extract": extract,
            "query": query,
            "body_mode": "raw" if kind == "application/json" else ("urlencoded" if kind else None),
            "body_fields": list(properties(spec, schema)) if schema else [],
            "body_template": None,
            "example_status": None,
            "example_body": None,
            "classified_by": "rules",
            "links": "inferred",
        })

    ids = [o["id"] for o in operations]
    dupes = [i for i, n in Counter(ids).items() if n > 1]
    if dupes:
        raise SystemExit(f"{service}: operation ids are not unique: {dupes[:10]}")

    producers, consumers = defaultdict(list), defaultdict(list)
    for o in operations:
        for n in o["produces"]:
            producers[n].append(o["id"])
        for n in o["consumes"]:
            consumers[n].append(o["id"])
    id_flow = {n: {"produced_by": producers.get(n, []), "conditional": [], "consumed_by": consumers.get(n, [])}
               for n in sorted(set(producers) | set(consumers))}
    resources = defaultdict(lambda: {"groups": set(), "actions": []})
    for o in operations:
        resources[o["resource"]]["actions"].append(o["action"])
        resources[o["resource"]]["groups"].add(o["group"])
    risk = Counter(o["risk"] for o in operations)
    return {
        "source": source or f"{SERVICES[service]['title']} OpenAPI spec "
                            f"({json.loads((SPECS / 'SOURCES.json').read_text())[service]['url']})",
        "service": service,
        "title": SERVICES[service]["title"],
        "summary": {
            "tools_exposed_to_model": len(operations),
            "resources": len(resources),
            "max_actions_per_resource": max(len(v["actions"]) for v in resources.values()),
            "with_inferred_creator": sum(bool(o["produces"]) for o in operations),
            "ids": len(id_flow),
            "ids_consumed_but_never_produced": sum(1 for v in id_flow.values() if not v["produced_by"]),
            "risk": dict(sorted(risk.items())),
        },
        "aliases": {},
        "resources": {k: {"groups": sorted(v["groups"]), "actions": sorted(v["actions"])} for k, v in sorted(resources.items())},
        "id_flow": id_flow,
        "operations": operations,
    }


def main(argv=None) -> int:
    names = (argv if argv is not None else sys.argv[1:]) or list(SERVICES)
    for service in names:
        spec = json.loads((SPECS / f"{service}.json").read_text(encoding="utf-8"))
        model = build(service, spec)
        out = SPECS / f"{service}_model.json"
        out.write_text(json.dumps(model, indent=1), encoding="utf-8")
        print(service, json.dumps(model["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

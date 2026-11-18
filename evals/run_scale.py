"""Scaling eval: does tool choice hold up as the catalogue grows from 110 to 2,220 tools?

The same selection eval as run_eval.py, run on four catalogues:

  110    PayPal only (the Postman collection)
  672    + Stripe   (562 tools from its OpenAPI spec)
  977    + Square   (305)
  2220   + GitHub   (1,243)

The added vendors are distractors for the PayPal cases (PayPal is set as the user's main
account, so a request that names no vendor means PayPal), and targets for the cases in
cases_scale.jsonl (run only where their vendor is in the catalogue).

Variants: jev (the agent as shipped), llm, all-tools (every tool in one prompt), and
retrieval@K (embedding top-K then the LLM picks; the langgraph-bigtool approach).

Nothing is cached: every answer is asked fresh. Run one size per process so sizes can run
side by side, then merge:

  python evals/run_scale.py --size 2220 --out evals/results/scale-1
  python evals/run_scale.py --report evals/results/scale-1
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.catalog import Catalog  # noqa: E402
from agent.llm import load_env  # noqa: E402
from evals import score  # noqa: E402
from evals.cache import Cache  # noqa: E402
from evals.run_eval import Harness  # noqa: E402

SPECS = ROOT / "data" / "specs"
SIZES = {
    "110": ["paypal"],
    "672": ["paypal", "stripe"],
    "977": ["paypal", "stripe", "square"],
    "2220": ["paypal", "stripe", "square", "github"],
}
VARIANTS = ("jev", "llm", "all-tools", "retrieval@10")
PAYPAL_FILES = ("cases.jsonl", "cases_user.jsonl", "cases_external.jsonl")
SCALE_SERVICE = {"st": "stripe", "sq": "square", "gh": "github", "xs": "stripe", "ys": "github"}


def catalog_for(services: list[str]) -> Catalog:
    paths = [ROOT / "paypal_model.json"] + [SPECS / f"{s}_model.json" for s in services if s != "paypal"]
    return Catalog.load_many(paths)


def cases_for(catalog: Catalog, services: list[str]) -> list[dict]:
    ops = {op.id for op in catalog}
    paypal = score.load_cases(PAYPAL_FILES, op_ids=None)
    scale = [c for c in score.load_cases(("cases_scale.jsonl",), op_ids=None) if SCALE_SERVICE[c["id"][:2]] in services]
    for case in scale:
        bad = [s["op"] for s in case["expect"].get("steps", []) if s["op"] not in ops]
        if bad:
            raise SystemExit(f"{case['id']}: {bad} not in the catalogue")
    for case in paypal:
        case["set"] = "paypal"
    for case in scale:
        case["set"] = "scale"
    return paypal + scale


def run_size(size: str, variants: list[str], out: Path, only: set[str]) -> None:
    services = SIZES[size]
    catalog = catalog_for(services)
    harness = Harness(Cache(enabled=False), catalog=catalog, default_service="paypal")
    cases = [c for c in cases_for(catalog, services) if not only or c["id"] in only]
    print(f"size {size}: {len(catalog)} tools, {len(cases)} cases, services {services}", flush=True)
    results = {}
    for variant in variants:
        began = time.perf_counter()
        selector = harness.selector(variant)
        setup = time.perf_counter() - began
        splitter = harness.splitter(variant) if variant in ("jev", "jev-only", "llm") else None
        rows = []
        for case in cases:
            start = time.perf_counter()
            row = harness.select_case(selector, case, splitter)
            row["seconds"] = round(time.perf_counter() - start, 3)
            if variant.startswith("retrieval"):
                goal = [s["op"] for s in case["expect"].get("steps", [])]
                row["retrieved"] = list(selector.last_retrieved)
                row["recall"] = all(op in selector.last_retrieved for op in goal) if goal else None
            rows.append(row)
            mark = "ok" if row["score"]["full_ok"] else "--"
            print(f"[{size:>4} {variant:12}] {mark} {case['id']:4} {'; '.join(row['score']['problems'])[:100]}",
                  flush=True)
        results[variant] = {"rows": rows, "setup_seconds": round(setup, 2)}
    out.mkdir(parents=True, exist_ok=True)
    meta = {"size": size, "tools": len(catalog), "services": services,
            "models": {"llm": harness.llm.model, "jev": harness.jev.model if "jev" in variants else None}}
    (out / f"{size}-{'+'.join(variants).replace('@', '')}.json").write_text(json.dumps({"meta": meta, "results": results}, indent=1, default=str),
                                      encoding="utf-8")


def _rate(rows, key):
    rows = [r for r in rows if key(r) is not None]
    return round(sum(bool(key(r)) for r in rows) / len(rows), 3) if rows else None


def summarise(rows: list[dict]) -> dict:
    seconds = [r["seconds"] for r in rows]
    return {
        "n": len(rows),
        "full": _rate(rows, lambda r: r["score"]["full_ok"]),
        "tool": _rate(rows, lambda r: r["score"].get("tool_ok", r["score"]["intent_ok"])),
        "p50": round(statistics.median(seconds), 2) if seconds else None,
        "tokens": round(statistics.mean(r["stats"]["tokens"] for r in rows)) if rows else None,
        "recall": _rate(rows, lambda r: r.get("recall")),
    }


def report(folder: Path) -> str:
    data = {}  # size -> {"meta", "results": {variant: ...}}, merged from one file per size and variant
    for path in folder.glob("*.json"):
        if path.name == "summary.json":
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        entry = data.setdefault(blob["meta"]["size"], {"meta": blob["meta"], "results": {}})
        entry["results"].update(blob["results"])
    order = {v: i for i, v in enumerate(VARIANTS)}
    data = {size: {"meta": data[size]["meta"],
                   "results": dict(sorted(data[size]["results"].items(), key=lambda kv: order.get(kv[0], 99)))}
            for size in sorted(data, key=int)}
    lines = ["# Scaling eval", ""]
    jev = sorted({m for p in folder.glob("*-*.json")
                  for m in [json.loads(p.read_text(encoding="utf-8"))["meta"]["models"].get("jev")] if m})
    llm = next(iter(data.values()))["meta"]["models"].get("llm") if data else None
    lines.append(f"Models: LLM {llm}, Jev {', '.join(jev) or '-'}. No cache; every answer asked fresh.")
    lines.append("Sizes and variants ran as parallel processes sharing the same rate limits, so the "
                 "seconds below compare variants within this run only.")
    lines.append("")
    summary = {}
    for subset, title in (("paypal", "PayPal cases (96) with the other vendors as distractors"),
                          ("scale", "Cases for the added vendors")):
        lines += [f"## {title}", "", "| Tools | Variant | Cases | Fully right | Right tool | Recall@K | p50 s | Tokens/case |",
                  "|---|---|---|---|---|---|---|---|"]
        for size, blob in data.items():
            for variant, res in blob["results"].items():
                rows = [r for r in res["rows"] if r["case"]["set"] == subset]
                if not rows:
                    continue
                s = summarise(rows)
                summary.setdefault(subset, {}).setdefault(variant, {})[blob["meta"]["tools"]] = s
                fmt = lambda v: "-" if v is None else f"{v:.0%}"  # noqa: E731
                lines.append(f"| {blob['meta']['tools']} | {variant} | {s['n']} | {fmt(s['full'])} | {fmt(s['tool'])} | "
                             f"{fmt(s['recall'])} | {s['p50']} | {s['tokens']:,} |")
        lines.append("")
    lines += ["## Failures at the largest size", ""]
    if data:
        largest = list(data.values())[-1]
        for variant, res in largest["results"].items():
            bad = [r for r in res["rows"] if not r["score"]["full_ok"]]
            lines.append(f"**{variant}** ({len(bad)})")
            for r in bad:
                lines.append(f"- {r['case']['id']}: {'; '.join(r['score']['problems'])[:200]}")
            lines.append("")
    (folder / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    text = "\n".join(lines)
    (folder / "summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=list(SIZES))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--only", default="")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report:
        print(report(args.report))
        return 0
    if not (args.size and args.out):
        parser.error("--size and --out are needed to run")
    run_size(args.size, [v for v in args.variants.split(",") if v], args.out, set(filter(None, args.only.split(","))))
    return 0


if __name__ == "__main__":
    sys.exit(main())

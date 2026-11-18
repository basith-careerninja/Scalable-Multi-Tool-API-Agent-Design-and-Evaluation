"""Run the eval.

Usage (from the project root, with the virtual environment active):
  python evals/run_eval.py                              every variant, selection and end to end
  python evals/run_eval.py --variants jev-only          one variant
  python evals/run_eval.py --level selection            skip end to end
  python evals/run_eval.py --only t01,m02               a few cases
  python evals/run_eval.py --langsmith                  also log a LangSmith experiment per variant
  python evals/run_eval.py --no-cache                   ask the models again instead of reusing answers
  python evals/run_eval.py --provider gemini            groq, gemini or openai (default: LLM_PROVIDER in .env)
  python evals/run_eval.py --live                       only end to end against the PayPal sandbox (cases with
                                                        an "e2e_live" block; run scripts/sandbox_seed.py first)

Variants (all decide the tool; everything after selection is the same agent):
  jev        Jev, with the LLM as fallback (the agent as shipped)
  jev-only   Jev, asking the user instead of falling back
  llm        the LLM making both choices (resource, then action)
  all-tools  control: every tool in one LLM prompt

Results go to evals/results/<time>/ as results.json and summary.md.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import runlog  # noqa: E402
from agent.agent import Agent  # noqa: E402
from agent.catalog import Catalog  # noqa: E402
from agent.execute import HttpTransport, MockTransport, extract_ids  # noqa: E402
from agent.fill import Filler  # noqa: E402
from agent.jev import JevClient  # noqa: E402
from agent.llm import ChatClient, load_env  # noqa: E402
from agent.planner import Planner  # noqa: E402
from agent.rag import DocsTool  # noqa: E402
from agent.runlog import RunLog  # noqa: E402
from agent.select import Selector  # noqa: E402
from agent.select_jev import JevSelector  # noqa: E402
from agent.split import Splitter  # noqa: E402
from evals import score  # noqa: E402
from evals.all_tools import AllToolsSelector  # noqa: E402
from evals.cache import DEFAULT_PATH as CACHE_PATH, Cache, CachedJev, CachedLLM, Stats  # noqa: E402

VARIANTS = ("jev", "jev-only", "llm", "all-tools")
EVAL_TODAY = date(2026, 11, 17)  # fixed, so "last month" means the same thing on every run
RESULTS = Path(__file__).resolve().parent / "results"
LIVE_CACHE = Path(__file__).resolve().parent / "cache_live.sqlite"


def discover_sandbox_ids(transport) -> dict[str, str]:
    """Real IDs for the live cases' {placeholders}, read from the sandbox with list calls (reads only).
    A kind the sandbox has none of is left out; the cases that need it then fail and say so."""
    lists = {"invoice_id": ("/v2/invoicing/invoices", {"page_size": "1"}, "items"),
             "product_id": ("/v1/catalogs/products", {"page_size": "1"}, "products"),
             "billing_plan_id": ("/v1/billing/plans", {"page_size": "1"}, "plans")}
    found = {}
    for name, (path, query, key) in lists.items():
        result = transport.request("GET", path, query)
        rows = result.body.get(key) if result.ok and isinstance(result.body, dict) else None
        if rows and isinstance(rows[0], dict) and rows[0].get("id"):
            found[name] = str(rows[0]["id"])
    return found


class RecordingTransport:
    """Wraps a real transport and records each call with its status and the IDs it produced."""

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[dict] = []

    def send(self, op, path, query, body, request_id=None):
        result = (self.inner.send(op, path, query, body, request_id=request_id) if request_id
                  else self.inner.send(op, path, query, body))
        self.calls.append({"op": op.id, "risk": op.risk, "method": op.method, "path": path, "query": query,
                           "body": body, "status": result.status,
                           "produced": extract_ids(op, result.body) if result.ok else {}})
        return result


class Harness:
    def __init__(self, cache: Cache, catalog: Catalog | None = None, default_service: str | None = None):
        self.catalog = catalog or Catalog.load()
        self.default_service = default_service  # with several vendors: the one a request means unless named
        self._embedder = None
        self.planner = Planner(self.catalog)
        self.stats = Stats()
        self._llm = self._jev = None
        self.cache = cache

    @property
    def llm(self):
        if self._llm is None:
            self._llm = CachedLLM(ChatClient(), self.cache, self.stats)
        return self._llm

    @property
    def jev(self):
        if self._jev is None:
            self._jev = CachedJev(JevClient(), self.cache, self.stats)
        return self._jev

    def selector(self, variant: str):
        # Both Jev variants read typed IDs the regex misses with the LLM: ID extraction, not tool choice.
        main = self.default_service
        if variant == "jev":
            return JevSelector(self.catalog, self.jev, fallback=Selector(self.catalog, self.llm, default_service=main),
                               id_llm=self.llm, default_service=main)
        if variant == "jev-only":
            return JevSelector(self.catalog, self.jev, id_llm=self.llm, default_service=main)
        if variant == "llm":
            return Selector(self.catalog, self.llm, default_service=main)
        if variant == "all-tools":
            return AllToolsSelector(self.catalog, self.llm, default_service=main)
        if variant.startswith("retrieval"):
            from evals.retrieval import Embedder, RetrievalSelector
            k = int(variant.split("@")[1]) if "@" in variant else 10
            self._embedder = self._embedder or Embedder()
            return RetrievalSelector(self.catalog, self.llm, self._embedder, k=k, default_service=main)
        raise ValueError(f"unknown variant {variant!r}")

    def splitter(self, variant: str):
        """Jev variants gate the split with Jev; the LLM variant with joining words. The all-tools
        control stays one prompt per message: it is the baseline without any of our structure."""
        if variant.startswith("jev"):
            return Splitter(self.llm, jev=self.jev)
        if variant == "llm":
            return Splitter(self.llm)
        return None

    # -- selection --------------------------------------------------------------------

    def select_case(self, selector, case: dict, splitter=None) -> dict:
        self.stats.reset()
        row = {"case": case}
        try:
            actions = splitter.split(case["message"])[0] if splitter else [case["message"]]
            row["actions"] = actions
            steps = score.to_steps(self._select_actions(selector, actions))
        except Exception as err:  # one failed case must not stop the run; it is reported
            steps, row["error"] = [], f"{type(err).__name__}: {err}"[:300]
        row["steps"] = steps
        row["score"] = score.score_selection(case, steps)
        plan = case["expect"].get("plan")
        combined = self._combined_plan(steps) if plan is not None else None
        as_split = score.score_split_chain(case, steps, combined)
        if as_split is not None:
            row["score"], row["plan"] = as_split, combined
        elif plan is not None:
            ok, got = False, None
            actions = [s for s in steps if s["intent"] == "action"]
            if row["score"]["tool_ok"] and len(actions) == 1:
                step = actions[0]
                got = [str(s) for s in self.planner.plan(step["op"], known=step["ids"], new=step["new"])]
                ok = got == plan
            row["score"]["plan_ok"] = ok
            row["plan"] = got
            if not ok:
                row["score"]["problems"].append(f"plan {got}, expected {plan}")
                row["score"]["full_ok"] = False
        row["decided_by"] = [s["decided_by"] for s in steps]
        row["stats"] = self.stats.as_dict()
        return row

    def _combined_plan(self, steps: list[dict]) -> list[str] | None:
        """The calls a split would send, in order: each action planned as the agent plans it,
        with IDs earlier actions create counted as known."""
        actions = [s for s in steps if s["intent"] == "action"]
        known_ops = {op.id for op in self.catalog}
        if len(actions) < 2 or any(s["op"] not in known_ops for s in actions):
            return None
        combined, pending = [], set()
        for step in actions:
            known = {**step["ids"], **{name: "@from-earlier-action" for name in pending}}
            new = [n for n in step["new"] if n not in pending]
            combined += [str(s) for s in self.planner.plan(step["op"], known=known, new=new)]
            if self.catalog[step["op"]].risk != "read":
                pending |= {n for n in self.catalog[step["op"]].produces
                            if not self.catalog.is_conditional(step["op"], n)}
        return combined

    def _select_actions(self, selector, actions: list[str]) -> list:
        """Select each action in order, as the agent does: an ID an earlier action creates is
        known to the later ones, not new again."""
        selections, pending = [], {}
        for action in actions:
            sel = selector.select(action, dict(pending))
            if pending:
                sel = dataclasses.replace(sel, new_ids=tuple(n for n in sel.new_ids if n not in pending),
                                          known_ids=tuple((k, v) for k, v in sel.known_ids if k not in pending))
            selections.append(sel)
            if sel.intent == "action" and sel.operation and self.catalog[sel.operation].risk != "read":
                for name in self.catalog[sel.operation].produces:
                    if not self.catalog.is_conditional(sel.operation, name):
                        pending[name] = f"@from-action-{len(selections) - 1}"
        return selections

    # -- end to end ----------------------------------------------------------------------

    def agent(self, variant: str):
        transport = MockTransport()
        agent = Agent(self.catalog, self.llm, transport, RunLog(":memory:"), docs=DocsTool(self.llm).answer,
                      filler=Filler(self.llm, today=EVAL_TODAY), selector=self.selector(variant),
                      jev=self.jev if variant.startswith("jev") else None, ready_llm=self.llm,
                      splitter=self.splitter(variant))
        return agent, transport

    def run_e2e(self, agent: Agent, transport: MockTransport, case: dict) -> dict:
        expect, session = case["e2e"], f"eval-{case['id']}-{time.time_ns()}"
        transport.calls.clear()
        self.stats.reset()
        row = {"case": case, "checks": {}, "problems": []}
        try:
            first = agent.chat(session, case["message"])
            planned = (agent.graph.get_state({"configurable": {"thread_id": session}}).values.get("calls") or [])
            row["status"] = first.status
            checks, problems = row["checks"], row["problems"]
            if "status" in expect:
                checks["status"] = first.status == expect["status"]
                if not checks["status"]:
                    problems.append(f"status {first.status} ({first.text[:120]!r}), expected {expect['status']}")
            if first.status == runlog.WAITING_APPROVAL:
                checks["nothing_sent_before_approval"] = not transport.calls
                if transport.calls:
                    problems.append(f"sent {[c['op'] for c in transport.calls]} before approval")
            if "fields" in expect:
                field_problems = score.fields_match(expect["fields"], planned)
                checks["fields"] = not field_problems
                problems += field_problems
            final = first
            if first.status == runlog.WAITING_APPROVAL and expect.get("approve"):
                final = agent.chat(session, "yes")
            if "calls" in expect:
                sent = [c["op"] for c in transport.calls]
                checks["calls"] = sent == expect["calls"]
                if not checks["calls"]:
                    problems.append(f"calls {sent}, expected {expect['calls']}")
            if "answer" in expect:
                answer_problems = score.answer_problems(expect["answer"], final.text, final.data)
                checks["answer"] = not answer_problems
                problems += answer_problems
            row["reply"] = final.text[:400]
        except Exception as err:  # one failed case must not stop the run; it is reported
            row["error"] = f"{type(err).__name__}: {err}"[:300]
            row["problems"].append(f"error: {row['error']}")
        row["ok"] = not row["problems"] and all(row["checks"].values())
        row["stats"] = self.stats.as_dict()
        return row

    # -- live end to end (PayPal sandbox) ----------------------------------------------

    def live_agent(self, variant: str):
        transport = RecordingTransport(HttpTransport())  # refuses anything but the sandbox
        # Live data is dated today, so "last month" must mean the real last month here.
        agent = Agent(self.catalog, self.llm, transport, RunLog(":memory:"), docs=DocsTool(self.llm).answer,
                      filler=Filler(self.llm, today=date.today()), selector=self.selector(variant),
                      jev=self.jev if variant.startswith("jev") else None, ready_llm=self.llm,
                      splitter=self.splitter(variant))
        return agent, transport

    def run_live(self, agent: Agent, transport: RecordingTransport, case: dict, sandbox_ids: dict) -> dict:
        expect, session = case["e2e_live"], f"live-{case['id']}-{time.time_ns()}"
        transport.calls.clear()
        self.stats.reset()
        row = {"case": case, "checks": {}, "problems": []}
        try:
            message = score.live_message(case, sandbox_ids)
            row["message"] = message
            first = agent.chat(session, message)
            answers = list(expect.get("answers", []))
            row["asked"] = []
            while first.status == runlog.WAITING_USER and answers:
                row["asked"].append(first.text[:200])
                first = agent.chat(session, answers.pop(0))
            state = agent.graph.get_state({"configurable": {"thread_id": session}}).values
            planned = [c["op"] for c in state.get("calls") or []]
            waiting = first.status == runlog.WAITING_APPROVAL
            before = list(transport.calls)
            final = first
            if waiting and expect.get("approve"):
                final = agent.chat(session, "yes")
            after = transport.calls[len(before):]
            row["checks"], row["problems"] = score.live_checks(expect, planned, state.get("level"), waiting,
                                                               before, after)
            row["sent"] = [{k: c[k] for k in ("op", "method", "path", "status")} for c in transport.calls]
            row["reply"] = final.text[:400]
        except Exception as err:  # one failed case must not stop the run; it is reported
            row["error"] = f"{type(err).__name__}: {err}"[:300]
            row["problems"].append(f"error: {row['error']}")
        row["ok"] = not row["problems"] and all(row["checks"].values())
        row["stats"] = self.stats.as_dict()
        return row


# -- reporting ------------------------------------------------------------------------------

COLUMNS = [("cases", "Cases"), ("intent", "Intent"), ("tool", "Right tool"), ("fully_right", "Fully right"),
           ("plan", "Plan"), ("asked_when_should", "Asked when it should"),
           ("asked_needlessly", "Asked needlessly"), ("dropped_actions", "Dropped actions"),
           ("model_calls", "Model calls"), ("tokens", "Tokens"), ("p50_seconds", "p50 s"), ("p95_seconds", "p95 s")]


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.0%}" if value <= 1 else f"{value:.2f}"
    return str(value)


def selection_table(results: dict, split: str | None, title: str) -> list[str]:
    if not any(split is None or r["case"]["split"] == split for rows in results.values() for r in rows):
        return []
    lines = [f"### {title}", "", "| Variant | " + " | ".join(label for _, label in COLUMNS) + " |",
             "|---" * (len(COLUMNS) + 1) + "|"]
    for variant, rows in results.items():
        chosen = [r for r in rows if split is None or r["case"]["split"] == split]
        if not chosen:
            continue
        s = score.summarise(chosen)
        cells = []
        for key, _ in COLUMNS:
            value = s[key]
            cells.append(f"{value:.2f}" if key.endswith("seconds") and value is not None else _fmt(value))
        lines.append(f"| {variant} | " + " | ".join(cells) + " |")
    return lines + [""]


def live_report(live: dict) -> list[str]:
    out = []
    for variant, rows in live.items():
        ok = sum(r["ok"] for r in rows)
        out += [f"### End to end on the PayPal sandbox: {variant}", "",
                f"{ok} of {len(rows)} cases passed every check.", "", "| Check | Passed |", "|---|---|"]
        for name in sorted({k for r in rows for k in r["checks"]}):
            applicable = [r for r in rows if name in r["checks"]]
            out.append(f"| {name} | {sum(r['checks'][name] for r in applicable)} of {len(applicable)} |")
        out += ["", f"#### Live failures, {variant}", ""]
        out += [f"- `{r['case']['id']}` {r.get('message', r['case']['message'])!r}: {'; '.join(r['problems'])}"
                for r in rows if not r["ok"]]
        out.append("")
    return out


def report(results: dict, e2e: dict, cases: list[dict], models: str = "", live: dict | None = None) -> str:
    out = [f"# Eval results, {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC", ""]
    if models:
        out += [f"Models: {models}.", ""]
    external = sum(c.get("file") == score.EXTERNAL_FILE for c in cases)
    out += [f"Cases: {len(cases)} ({external} external).", ""]
    out += selection_table(results, "test", "Selection, held-out half (report these)")
    out += selection_table(results, "tune", "Selection, tuning half")
    out += selection_table(results, "external", "Selection, cases from another submission (Sudharsan2618)")
    variants = list(results)
    categories = sorted({r["case"]["category"] for rows in results.values() for r in rows})
    if variants:
        out += ["### Fully right by category (all of our cases)", "",
                "| Category | Cases | " + " | ".join(variants) + " |", "|---" * (len(variants) + 2) + "|"]
    for cat in categories:
        cells, n = [], 0
        for v in variants:
            rows = [r for r in results[v] if r["case"]["category"] == cat]
            n = len(rows)
            cells.append(_fmt(round(sum(r["score"]["full_ok"] for r in rows) / n, 3)) if rows else "-")
        out.append(f"| {cat} | {n} | " + " | ".join(cells) + " |")
    if variants:
        out.append("")
    for variant, rows in e2e.items():
        ok = sum(r["ok"] for r in rows)
        out += [f"### End to end on the mock: {variant}", "",
                f"{ok} of {len(rows)} cases passed every check.", ""]
        names = sorted({k for r in rows for k in r["checks"]})
        out += ["| Check | Passed |", "|---|---|"]
        for name in names:
            applicable = [r for r in rows if name in r["checks"]]
            out.append(f"| {name} | {sum(r['checks'][name] for r in applicable)} of {len(applicable)} |")
        out.append("")
    out += live_report(live or {})
    if results or e2e:
        out += ["## Failures", ""]
    for variant, rows in results.items():
        failures = [r for r in rows if not r["score"]["full_ok"]]
        out += [f"### Selection, {variant} ({len(failures)})", ""]
        for r in failures:
            c = r["case"]
            out.append(f"- `{c['id']}` {c['message']!r}: {'; '.join(r['score']['problems']) or r.get('error', '')}")
        out.append("")
    for variant, rows in e2e.items():
        failures = [r for r in rows if not r["ok"]]
        out += [f"### End to end, {variant} ({len(failures)})", ""]
        out += [f"- `{r['case']['id']}` {r['case']['message']!r}: {'; '.join(r['problems'])}" for r in failures]
        out.append("")
    return "\n".join(out)


def langsmith_experiment(harness: Harness, variant: str, cases: list[dict]) -> None:
    from langsmith import Client, evaluate

    from agent.observability import configure_langsmith
    if not configure_langsmith():
        print("LangSmith is not configured; skipping the experiment.")
        return
    client = Client()
    digest = hashlib.sha256(json.dumps([[c["id"], c["message"], c["expect"]] for c in cases]).encode()).hexdigest()
    dataset = f"paypal-agent-eval-{digest[:8]}"
    if not client.has_dataset(dataset_name=dataset):
        created = client.create_dataset(dataset_name=dataset, description="PayPal agent eval cases")
        client.create_examples(dataset_id=created.id, inputs=[{"message": c["message"]} for c in cases],
                               outputs=[{"case": c} for c in cases])
    selector = harness.selector(variant)

    def target(inputs: dict) -> dict:
        return {"steps": score.to_steps([selector.select(inputs["message"])])}

    def metric(name):
        def evaluator(run, example):
            result = score.score_selection(example.outputs["case"], run.outputs["steps"])
            return {"key": name, "score": int(result[name])}
        evaluator.__name__ = name
        return evaluator

    evaluate(target, data=dataset, evaluators=[metric("intent_ok"), metric("tool_ok"), metric("full_ok")],
             experiment_prefix=f"selection-{variant}", max_concurrency=1, metadata={"variant": variant})


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--level", choices=("selection", "e2e", "all"), default="all")
    parser.add_argument("--e2e-variants", default="jev")
    parser.add_argument("--only", default="")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--langsmith", action="store_true")
    parser.add_argument("--provider", choices=("groq", "gemini", "openai"))
    parser.add_argument("--model", help="override the provider's default model")
    parser.add_argument("--live", action="store_true",
                        help="run only the e2e_live cases against the PayPal sandbox (with --e2e-variants)")
    args = parser.parse_args()
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["LLM_MODEL"] = args.model

    # Live answers depend on live data, so they are kept apart from the eval's frozen cache.
    cache_path = LIVE_CACHE if args.live else CACHE_PATH
    harness = Harness(Cache(cache_path, enabled=not args.no_cache))
    cases = score.load_cases(op_ids={op.id for op in harness.catalog})
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted or c["category"] in wanted]
    variants = [v for v in args.variants.split(",") if v]

    results: dict[str, list] = {}
    live: dict[str, list] = {}
    if args.live:
        sandbox_ids = discover_sandbox_ids(HttpTransport())
        print("Sandbox IDs read at run time: " + ", ".join(f"{k}={v}" for k, v in sandbox_ids.items()), flush=True)
        for variant in [v for v in args.e2e_variants.split(",") if v]:
            agent, transport = harness.live_agent(variant)
            rows = []
            for case in [c for c in cases if "e2e_live" in c]:
                row = harness.run_live(agent, transport, case, sandbox_ids)
                rows.append(row)
                print(f"[live {variant:5}] {'ok' if row['ok'] else '--'} {case['id']:4} "
                      f"{'; '.join(row['problems'])[:110]}", flush=True)
            live[variant] = rows
        variants = []
    if args.level in ("selection", "all") and not args.live:
        for variant in variants:
            selector, rows = harness.selector(variant), []
            for case in cases:
                row = harness.select_case(selector, case, harness.splitter(variant))
                rows.append(row)
                mark = "ok" if row["score"]["full_ok"] else "--"
                print(f"[{variant:9}] {mark} {case['id']:4} {'; '.join(row['score']['problems'])[:110]}", flush=True)
            results[variant] = rows
            if args.langsmith:
                langsmith_experiment(harness, variant, cases)

    e2e: dict[str, list] = {}
    if args.level in ("e2e", "all") and not args.live:
        for variant in [v for v in args.e2e_variants.split(",") if v]:
            agent, transport = harness.agent(variant)
            rows = []
            for case in [c for c in cases if "e2e" in c]:
                row = harness.run_e2e(agent, transport, case)
                rows.append(row)
                print(f"[e2e {variant:5}] {'ok' if row['ok'] else '--'} {case['id']:4} "
                      f"{'; '.join(row['problems'])[:110]}", flush=True)
            e2e[variant] = rows

    folder = RESULTS / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.json").write_text(json.dumps({"selection": results, "e2e": e2e, "e2e_live": live},
                                                    indent=1, default=str), encoding="utf-8")
    uses_llm = any(v != "jev-only" for v in variants) or e2e or live
    uses_jev = any(v.startswith("jev") for v in variants) or e2e or live
    models = ", ".join(filter(None, [f"LLM {harness.llm.model}" if uses_llm else "",
                                     f"Jev {harness.jev.model}" if uses_jev else ""]))
    text = report(results, e2e, [c for c in cases if "e2e_live" in c] if args.live else cases, models, live)
    (folder / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\nSaved to {folder.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

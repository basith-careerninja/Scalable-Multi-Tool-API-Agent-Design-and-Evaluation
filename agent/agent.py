"""The agent as a LangGraph graph.

    select ─┬─ system_search ─────────────────────────────────────────────┐
            ├─ rag ───────────────────────────────────────────────────────┤
            ├─ ask ── await_answer ── (back to select)                   │
            └─ plan ── lookup ── fill ── check_ready ─┬─ await_approval ─┬─ execute ──┐  │
                                     │                │                  └─ cancelled ─┤  │
                                     │                ├─ execute ── compute ── answer ─┴──┴─ END
                                     │                └─ await_detail ── apply_answer ──┐
                                     └──────────────────── (back to check_ready) ◄──────┘

split runs first (agent/split.py): a message asking for several actions ("create a product
and a monthly plan for it") becomes an ordered list, and select -> plan -> lookup -> fill
runs once per action. An ID an earlier action will create is "from action N" for the later
ones, so it is neither created twice nor looked up; execute swaps in the real ID. All the
calls then share one readiness check, one approval and one execute, which stops at the
first failure.

check_ready asks ONE question for everything the planned calls still need from the user
(code rules plus an LLM check, agent/ready.py). apply_answer reads the reply into those
fields only, keeping the operation and every value already filled, unless the reply is a
new request (Jev decides), which goes back to select.

Each chat session is a LangGraph thread. Its state (the IDs remembered so far, and
what the graph is waiting on) is saved by a checkpointer after every step, so a
pending approval or question survives a restart when a persistent checkpointer
(SQLite) is used.

Waiting for the user is a LangGraph interrupt: `await_approval` and `await_answer`
pause the graph, and the user's next message resumes it. Those two nodes do nothing
before the interrupt, because a resumed node runs again from its start.

Proof of what was sent lives apart from the checkpoint, in the evidence store
(agent/evidence.py), written by `execute` around each call. `execute` reads it first, so
running it again after a crash never repeats a call that went through: a call that
succeeded is reused, one whose outcome is unknown is reconciled (agent/reconcile.py),
and one that cannot be settled stops the run.

Everything the graph stores is plain data (dicts, lists, strings), so it can be
checkpointed; catalogue objects are looked up again from their IDs inside each node.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import copy

from . import compute, config, policy, ready, resolve, runlog
from .catalog import Catalog, id_label
from .compute import ComputePlanner
from . import evidence as ev
from .evidence import Evidence
from .execute import Executor, MissingId, Result, render_path
from .reconcile import Reconciler
from .fill import Filled, Filler
from .jev import probabilities, yes
from .planner import Planner
from .resolve import Resolver
from .runlog import RunLog
from .split import Splitter
from .select import Selector
from .system_search import SystemSearch
from .trace import TracedLLM, Tracer

# Exact replies decide without a model call. Anything else goes to Jev when it is configured.
YES = {"yes", "y", "approve", "approved", "ok", "okay", "confirm", "confirmed", "go ahead", "do it", "proceed"}
NO = {"no", "n", "cancel", "stop", "reject", "don't", "dont", "abort"}
MAX_RESULT_CHARS = 6000
REPROMPT = "Please reply yes to go ahead or no to cancel."
_RANK = {policy.NONE: 0, policy.CONFIRM: 1, policy.EXPLICIT: 2}

APPROVAL_OPTIONS = {
    "approve": "The user accepts the shown calls exactly as shown.",
    "decline": "The user does not want the shown calls sent.",
    "change": "The user wants something different from what is shown (an amount, a recipient, a step added "
              "or removed), whether or not they also say yes.",
    "other": "A question, or anything unrelated to approving the shown calls.",
}
APPROVE, DECLINE, CHANGE, UNCLEAR = "approve", "decline", "change", "unclear"


@dataclass
class Reply:
    text: str
    status: str                      # completed, waiting_approval, waiting_user, failed
    data: dict = field(default_factory=dict)


class State(TypedDict, total=False):
    # kept across turns
    session: str
    ids: dict             # IDs produced or used earlier in this conversation
    # one turn
    message: str          # the request, merged with answers to any questions asked about it
    latest: str           # the user's most recent message alone
    selection: dict
    known: dict           # IDs this plan will use
    found: dict           # IDs found by look-up, with how they were found
    steps: list           # [[kind, target], ...] from the planner
    calls: list           # [{"op", "query", "body_values"}]
    level: str
    run_id: int
    question: str
    detail_for: dict      # the selection a fill question was about, kept while the user answers it
    asked_run: dict       # {"id", "op"}: the run-log entry of the question being answered
    missing: list         # details the planned calls still need from the user (agent/ready.py)
    answers: list         # [{"question", "answer"}] given for this request
    detail_rounds: int    # how many readiness questions were answered
    last_question: str
    detail_next: str      # where apply_answer goes: check_ready, select or cancelled
    asked_llm: list       # [{"call", "path"}] details the LLM check asked for once already
    actions: list         # the ordered actions a message was split into; None for one action
    action_index: int     # the action being selected, planned and filled
    done_calls: list      # calls of the actions already filled
    goals: list           # the operation each action was selected for
    more: bool            # fill finished an action and more remain
    resplit: bool         # an answer to a question about a split message: split it again
    reprompt: bool
    approval: str
    results: list
    computed: dict
    reply: dict


# Fields reset at the start of every new request, so nothing leaks between turns.
_TURN_DEFAULTS = {"selection": None, "known": None, "found": None, "steps": None, "calls": None,
                  "level": None, "run_id": None, "question": None, "detail_for": None, "asked_run": None, "reprompt": False,
                  "missing": None, "answers": None, "detail_rounds": 0, "last_question": None,
                  "asked_llm": None, "actions": None, "action_index": 0, "done_calls": None, "goals": None,
                  "more": False, "resplit": False,
                  "detail_next": None,
                  "approval": None,
                  "results": None, "computed": None, "reply": None}


class Agent:
    def __init__(self, catalog: Catalog, llm, transport, log: RunLog,
                 docs: Callable[[str], str] | None = None, filler: Filler | None = None,
                 tracer: Tracer | None = None, selector=None, checkpointer=None, jev=None,
                 ready_llm=None, evidence: Evidence | None = None, splitter: Splitter | None = None):
        self.catalog = catalog
        self.jev = jev  # optional; judges approval replies that are not an exact yes or no
        self.tracer = tracer or Tracer()
        if not isinstance(llm, TracedLLM):
            llm = TracedLLM(llm, self.tracer)
        self.llm = llm
        # Any object with select(message, remembered) -> Selection; Jev or the generative selector.
        self.selector = selector or Selector(catalog, llm)
        self.planner = Planner(catalog)
        self.filler = filler or Filler(llm)
        self.executor = Executor(catalog, transport)
        self.log = log
        self.evidence = evidence or Evidence(":memory:")
        self.reconciler = Reconciler(catalog, self.executor, self.evidence)
        self.system_search = SystemSearch(catalog, log, self.evidence)
        self.compute_planner = ComputePlanner(llm)
        self.resolver = Resolver(catalog, llm, self.filler, self.executor)
        self.docs = docs
        # The LLM readiness check is on when a model is given for it; the code rules always apply.
        if ready_llm is not None and not isinstance(ready_llm, TracedLLM):
            ready_llm = TracedLLM(ready_llm, self.tracer)
        self.checker = ready.ReadyChecker(ready_llm) if ready_llm is not None else None
        self.reader = ready.AnswerReader(llm)
        # Splitting a message into several actions is on when a splitter is given (chat and eval do).
        self.splitter = splitter
        self.graph = self._build().compile(checkpointer=checkpointer or InMemorySaver())

    # -- entry point -------------------------------------------------------------

    def chat(self, session_id: str, message: str) -> Reply:
        config = {"configurable": {"thread_id": session_id}, "run_name": "turn",
                  "metadata": {"session": session_id}}
        with self.tracer.span("turn", kind="turn", session=session_id, message=message) as span:
            finished = self.resume_interrupted(session_id)
            if self.graph.get_state(config).interrupts:
                out = self.graph.invoke(Command(resume=message), config)
            else:
                # A new request. A run that failed with an error reply is not resumed.
                out = self.graph.invoke({**_TURN_DEFAULTS, "session": session_id, "message": message,
                                         "latest": message}, config)
            reply = _reply(out)
            if finished:
                reply = Reply(f"{finished.text}\n\n{reply.text}", reply.status, reply.data)
            span.set(reply_status=reply.status, reply=reply.text, run_id=reply.data.get("run_id"))
            return reply

    def resume_interrupted(self, session_id: str) -> Reply | None:
        """Finish a run that was cut off while its calls were being sent (a crash, a killed process).

        Only `execute` is resumed: it reads the evidence first, so calls that went through
        are not sent again. Returns the finished run's reply, or None if nothing was cut off."""
        config = {"configurable": {"thread_id": session_id}, "run_name": "resume",
                  "metadata": {"session": session_id}}
        state = self.graph.get_state(config)
        if state.interrupts or "execute" not in (state.next or ()):
            return None
        with self.tracer.span("resume", session=session_id) as span:
            done = _reply(self.graph.invoke(None, config))
            span.set(reply_status=done.status)
        return Reply("A request of yours was cut off while it was running. I checked what had already "
                     f"gone through and finished it:\n{done.text}", done.status, done.data)

    def reconcile_unsettled(self) -> list[dict]:
        """Settle every recorded call whose outcome is unknown. Run at startup."""
        settled = []
        for record in self.evidence.unsettled():
            with self.tracer.span("reconcile", operation=record["operation"], run_id=record["run_id"]) as span:
                after = self.reconciler.settle(record)
                span.set(state=after["state"], note=after["note"])
            settled.append({"run_id": after["run_id"], "session": after["session"],
                            "operation": after["operation"], "state": after["state"], "note": after["note"]})
        return settled

    # -- graph -------------------------------------------------------------------

    def _build(self) -> StateGraph:
        g = StateGraph(State)
        for name in ("split", "select", "system_search", "rag", "plan", "lookup", "fill", "ask", "await_answer",
                     "check_ready", "await_detail", "apply_answer",
                     "await_approval", "cancelled", "execute", "compute", "answer"):
            g.add_node(name, getattr(self, f"_{name}"))
        g.add_edge(START, "split")
        g.add_edge("split", "select")
        g.add_conditional_edges("select", _after_select,
                                ["ask", "system_search", "rag", "plan"])
        g.add_edge("system_search", END)
        g.add_edge("rag", END)
        g.add_conditional_edges("plan", lambda s: "lookup" if _unresolved(s) else "fill", ["lookup", "fill"])
        g.add_conditional_edges("lookup", lambda s: "ask" if s.get("question") else "fill", ["ask", "fill"])
        g.add_conditional_edges("fill", _after_fill, ["ask", "select", "check_ready"])
        g.add_conditional_edges("check_ready", _after_ready, ["await_detail", "await_approval", "execute", END])
        g.add_edge("await_detail", "apply_answer")
        g.add_conditional_edges("apply_answer", lambda s: s["detail_next"], ["check_ready", "split", "cancelled"])
        g.add_edge("ask", "await_answer")
        g.add_conditional_edges("await_answer", lambda s: "split" if s.get("resplit") else "select",
                                ["split", "select"])
        g.add_conditional_edges("await_approval", _after_approval, ["execute", "cancelled", "await_approval", "split"])
        g.add_edge("cancelled", END)
        g.add_conditional_edges("execute", _after_execute, ["compute", "answer", END])
        g.add_edge("compute", "answer")
        g.add_edge("answer", END)
        return g

    # -- nodes: deciding ------------------------------------------------------------

    def _split(self, s: State) -> dict:
        reset = {"actions": None, "action_index": 0, "done_calls": [], "goals": [], "more": False}
        if self.splitter is None:
            return reset
        with self.tracer.span("split") as span:
            actions, info = self.splitter.split(s["message"])
            span.set(actions=actions, gate=info["gate"], rejected=info["rejected"])
        return {**reset, "actions": actions if len(actions) > 1 else None}

    def _select(self, s: State) -> dict:
        pending = _pending_ids(self.catalog, s)
        with self.tracer.span("select", action=s.get("action_index") if s.get("actions") else None) as span:
            sel = self.selector.select(_text(s), {**(s.get("ids") or {}), **pending})
            if s.get("actions") and sel.intent in ("system", "docs"):
                # A question about the system or the docs inside a split message: not an account
                # action to chain. Handle the whole message as one request instead.
                span.set(unsplit=sel.intent)
                sel = self.selector.select(s["message"], s.get("ids") or {})
                s = {**s, "actions": None}
            span.set(intent=sel.intent, operation=sel.operation, confidence=sel.confidence,
                     new_ids=list(sel.new_ids), calculate=sel.calculate,
                     decided_by=sel.decided_by, reason=sel.reason or None,
                     scores=[list(pair) for pair in sel.scores] or None)
            # An answer to a detail question ("Consulting") can blur the request it completes. Unless the
            # answer is clearly a request of its own, keep the operation the question was about.
            if sel.intent == "clarify" and s.get("detail_for"):
                span.set(kept=s["detail_for"]["operation"])
                return {"selection": s["detail_for"], "question": None}
        selection = {"intent": sel.intent, "operation": sel.operation,
                     # An ID an earlier action creates is that action's, not created again here.
                     "new_ids": [n for n in sel.new_ids if n not in pending],
                     # ...and its ID is never replaced by an ID-like token typed for something else.
                     "known_ids": {k: v for k, v in sel.known_ids if k not in pending}, "calculate": sel.calculate,
                     "system_hint": sel.system_hint}
        return {"selection": selection, "question": sel.question if sel.intent == "clarify" else None,
                "actions": s.get("actions")}

    def _system_search(self, s: State) -> dict:
        # Searched with the latest message only, so an unanswered earlier request cannot skew it.
        with self.tracer.span("system_search") as span:
            hint = (s.get("selection") or {}).get("system_hint")
            result = self.system_search.search(s["latest"], s["session"], hint=hint)
            span.set(result=result["kind"], decided_by=result.get("decided_by"))
        self.log.record(s["session"], s["latest"], "system", status=runlog.COMPLETED)
        return {"reply": {"text": _format_system(result), "status": runlog.COMPLETED, "data": result}}

    def _rag(self, s: State) -> dict:
        with self.tracer.span("rag"):
            text = self.docs(s["latest"]) if self.docs else "The documentation search tool is not set up yet."
        self.log.record(s["session"], s["latest"], "docs", status=runlog.COMPLETED)
        return {"reply": {"text": text, "status": runlog.COMPLETED, "data": {}}}

    def _plan(self, s: State) -> dict:
        sel = s["selection"]
        known = {k: v for k, v in (s.get("ids") or {}).items() if k not in sel["new_ids"]}
        known.update(sel["known_ids"])
        known.update(_pending_ids(self.catalog, s))
        with self.tracer.span("plan", operation=sel["operation"]) as span:
            steps = self.planner.plan(sel["operation"], known=known, new=sel["new_ids"])
            span.set(plan=[str(step) for step in steps])
        return {"known": known, "found": {}, "steps": [[step.kind, step.target] for step in steps]}

    def _lookup(self, s: State) -> dict:
        """Find existing items the user described but did not give an ID for."""
        sel, known, found = s["selection"], dict(s["known"]), {}
        for name in _unresolved(s):
            with self.tracer.span("resolve", id_name=name) as span:
                res = self.resolver.resolve(name, _text(s))
                span.set(result=res.status, via=res.via, looked_at=res.looked_at, found_id=res.id)
            if res.status == resolve.FOUND:
                known[name] = res.id
                found[name] = f"found via {res.via}: {res.summary}"
                continue
            label = id_label(name)
            if res.status == resolve.AMBIGUOUS:
                options = "\n".join(f"- {i}: {text}" for i, text in res.options)
                question = f"I found more than one {label} that could fit. Which one do you mean?\n{options}"
            elif res.status == resolve.NONE:
                question = (f"I could not find a matching {label} (checked {res.looked_at} of {res.total} "
                            f"via {res.via}). Please give me its ID.")
            else:
                question = f"Which {label} do you mean? Please give me its ID; I have no way to list them."
            return {"question": question}
        steps = self.planner.plan(sel["operation"], known=known, new=sel["new_ids"])
        return {"known": known, "found": found, "steps": [[step.kind, step.target] for step in steps]}

    def _fill(self, s: State) -> dict:
        calls = []
        for kind, target in s["steps"]:
            op = self.catalog[target]
            with self.tracer.span("fill", operation=op.id) as span:
                filled = self.filler.fill(op, _text(s))
                span.set(query=filled.query, body=filled.body_values, question=filled.question,
                         dropped=filled.dropped)
            if filled.question:
                return {"question": filled.question, "detail_for": s["selection"]}
            calls.append({"op": op.id, "query": filled.query, "body_values": filled.body_values,
                          "needs_currency": filled.needs_currency, "hint": filled.hint})
        if s.get("actions"):
            index = s["action_index"]
            for call in calls:  # each call keeps its own action's IDs ("send A and remind B")
                call.update(action=index, uses=dict(s["known"]))
            goals = (s.get("goals") or []) + [s["selection"]["operation"]]
            if index + 1 < len(s["actions"]):
                return {"done_calls": (s.get("done_calls") or []) + calls, "goals": goals,
                        "action_index": index + 1, "more": True, "selection": None, "steps": None,
                        "known": None, "found": None}
            calls = (s.get("done_calls") or []) + calls
        level = max((policy.approval(self.catalog[c["op"]]) for c in calls), key=_RANK.get)
        plan = {"plan": [c["op"] for c in calls]}
        asked = self._asked_run(s)
        if asked:  # the request the user just answered a question about, now planned
            run_id = asked["id"]
            self.log.update(run_id, runlog.PLANNED, plan)
        else:
            operation = ", ".join(goals) if s.get("actions") else s["selection"]["operation"]
            run_id = self.log.record(s["session"], _original(s["message"]), "action", operation,
                                     runlog.PLANNED, plan)
        return {"calls": calls, "level": level, "run_id": run_id, "reprompt": False, "more": False}

    # -- nodes: is everything the user must supply there? ------------------------------

    def _check_ready(self, s: State) -> dict:
        calls, answers = copy.deepcopy(s["calls"]), s.get("answers") or []
        rounds = s.get("detail_rounds") or 0
        with self.tracer.span("check_ready", round=rounds) as span:
            rule = ready.code_rules(self.catalog, calls)
            extra, llm_question = [], None
            if self.checker is not None:
                try:
                    asked = {(m["call"], m["path"]) for m in s.get("asked_llm") or []}
                    extra, llm_question = self.checker.check(self.catalog, _original(s["message"]), answers,
                                                             calls, rule, asked_before=asked)
                except Exception as err:  # the code rules still apply
                    span.set(checker_error=str(err)[:200])
            missing = rule + extra
            span.set(missing=[m["path"] for m in missing], by_rule=[m["path"] for m in rule])
        if not missing:
            if s["level"] != policy.NONE:
                self.log.update(s["run_id"], runlog.WAITING_APPROVAL)
            return {"calls": calls, "missing": [], "question": None}
        if rounds >= MAX_DETAIL_ROUNDS:
            still = ", ".join(dict.fromkeys(m["label"] for m in missing))
            self.log.update(s["run_id"], runlog.FAILED, {"reason": "details still missing", "missing": still})
            text = f"I still don't have {still}, so I stopped. Nothing was sent. Please ask again with those details."
            return {"calls": calls, "missing": missing,
                    "reply": {"text": text, "status": runlog.FAILED, "data": {"run_id": s["run_id"]}}}
        question = llm_question if llm_question and _covers(llm_question, rule) else None
        question = question or ready.code_question(missing, calls, rounds)
        if question == s.get("last_question"):
            # Never the same question twice in a row: show what landed and what is still needed.
            question = ready.code_question(missing, calls, max(rounds, 1))
        self.log.update(s["run_id"], runlog.WAITING_USER, {"question": question})
        asked_llm = (s.get("asked_llm") or []) + [{"call": m["call"], "path": m["path"]} for m in extra]
        return {"calls": calls, "missing": missing, "question": question, "last_question": question,
                "asked_llm": asked_llm}

    def _await_detail(self, s: State) -> dict:
        answer = interrupt({"kind": "question", "text": s["question"], "run_id": s["run_id"]})
        return {"latest": str(answer)}

    def _apply_answer(self, s: State) -> dict:
        answer, question, missing = s["latest"], s["question"], s["missing"]
        if answer.strip().lower().rstrip(".!") in NO:
            return {"detail_next": "cancelled"}
        request = _original(s["message"])
        answers = (s.get("answers") or []) + [{"question": question, "answer": answer}]
        with self.tracer.span("apply_answer") as span:
            new_request = None
            if self.jev is not None:
                try:
                    judged = self.jev.ask({"question_asked": question, "user_reply": answer}, {"new_request": {
                        "type": "noul",
                        "instructions": "Is the user's reply a new, different request, instead of an answer "
                                        "to the question they were asked?"}})
                    new_request = yes(judged["new_request"]) >= config.YES_THRESHOLD
                except Exception:
                    new_request = None
            values, reader_new = ({}, False) if new_request else self.reader.read(question, request, missing, answer)
            if new_request is None:
                new_request = reader_new
            span.set(new_request=new_request, read={f"{i}:{p}": v for (i, p), v in values.items()})
            if new_request:
                self.log.update(s["run_id"], runlog.FAILED, {"reason": "replaced by a new request"})
                return {**_TURN_DEFAULTS, "message": answer, "latest": answer, "detail_next": "split"}
            calls = copy.deepcopy(s["calls"])
            kept = [f"{i}:{p}" for (i, p), v in values.items()
                    if ready.accept(self.catalog, calls, (i, p), v, answer, request)]
            said = "\n".join(f"The user added: {a['answer']}" for a in answers)
            for i in sorted({m["call"] for m in missing if m["path"] == ready.HINT}):
                # A free-form detail: fill that call again with the answers, keeping what was already set.
                refilled = self.filler.fill(self.catalog[calls[i]["op"]], f"{request}\n{said}")
                calls[i]["body_values"] = {**calls[i]["body_values"], **refilled.body_values}
                calls[i]["query"] = {**calls[i]["query"], **refilled.query}
                calls[i]["hint"] = refilled.hint
                calls[i]["needs_currency"] = calls[i].get("needs_currency") and refilled.needs_currency
                kept.append(f"{i}:hint")
            span.set(kept=kept)
        return {"calls": calls, "answers": answers, "detail_rounds": (s.get("detail_rounds") or 0) + 1,
                "detail_next": "check_ready"}

    # -- nodes: waiting for the user --------------------------------------------------

    def _ask(self, s: State) -> dict:
        sel = s.get("selection") or {}
        intent = "clarify" if sel.get("intent") == "clarify" else "action"
        detail = {"question": s["question"]}
        asked = self._asked_run(s)
        if asked:  # asked again about the same request: one entry, not one per question
            self.log.update(asked["id"], runlog.WAITING_USER, detail)
            return {}
        run_id = self.log.record(s["session"], _original(s["message"]), intent, sel.get("operation"),
                                 runlog.WAITING_USER, detail)
        return {"asked_run": {"id": run_id, "op": sel.get("operation")}}

    @staticmethod
    def _asked_run(s: State) -> dict | None:
        """The run-log entry of the question just answered, if this turn continues that request."""
        asked, sel = s.get("asked_run"), s.get("selection") or {}
        return asked if asked and asked.get("op") and asked["op"] == sel.get("operation") else None

    def _await_answer(self, s: State) -> dict:
        answer = interrupt({"kind": "question", "text": s["question"]})
        # The answer may complete the request or start a new one; selection decides.
        merged = (f"{s['message']}\nThe user added: {answer}\n"
                  "(If what the user added answers the earlier request, continue that request. "
                  "If it is a different request, handle only what the user added.)")
        return {**_TURN_DEFAULTS, "message": merged, "latest": answer, "detail_for": s.get("detail_for"),
                "asked_run": s.get("asked_run"), "resplit": bool(s.get("actions"))}

    def _await_approval(self, s: State) -> dict:
        preview = self._preview(s)
        answer = str(interrupt({"kind": "approval", "text": REPROMPT if s.get("reprompt") else preview,
                                "run_id": s["run_id"]}))
        with self.tracer.span("approval", reply=answer) as span:
            decision, probs, decided_by = self._judge_approval(preview, answer)
            span.set(decision=decision, decided_by=decided_by, probabilities=probs or None)
        if decision == APPROVE:
            return {"approval": "yes", "reprompt": False}
        if decision == DECLINE:
            return {"approval": "no"}
        if decision == CHANGE:
            # Nothing was sent. The correction is planned from scratch and shown again for a fresh approval.
            self.log.update(s["run_id"], runlog.FAILED, {"reason": "replaced by the user's correction",
                                                         "correction": answer})
            merged = (f"{s['message']}\nThe user changed the request: {answer}\n"
                      "(Apply this change to the request above and handle the whole request again.)")
            return {**_TURN_DEFAULTS, "message": merged, "latest": answer, "approval": "change"}
        return {"approval": "unclear", "reprompt": True}

    def _judge_approval(self, preview: str, answer: str) -> tuple[str, dict, str]:
        """(decision, probabilities, decided_by). Only an exact yes or a confident Jev 'approve' approves."""
        word = answer.strip().lower().rstrip(".!")
        if word in YES:
            return APPROVE, {}, "words"
        if word in NO:
            return DECLINE, {}, "words"
        if self.jev is None:
            return UNCLEAR, {}, "words"
        try:
            answers = self.jev.ask({"plan_shown_to_user": preview, "user_reply": answer}, {"approval": {
                "type": "choice",
                "instructions": "How does the user's reply respond to the plan of API calls shown to them?",
                "criteria": APPROVAL_OPTIONS,
            }})
            probs = probabilities(answers["approval"], APPROVAL_OPTIONS)
        except Exception:  # a failed judgment must never approve
            return UNCLEAR, {}, "jev-error"
        probs = {option: round(probs.get(option, 0.0), 3) for option in APPROVAL_OPTIONS}
        if probs["approve"] >= config.APPROVE_MIN and probs["change"] < config.APPROVE_MAX_CHANGE:
            return APPROVE, probs, "jev"
        if probs["decline"] >= config.DECLINE_MIN:
            return DECLINE, probs, "jev"
        if probs["change"] > 0 and probs["change"] == max(probs.values()):
            return CHANGE, probs, "jev"
        return UNCLEAR, probs, "jev"

    def _cancelled(self, s: State) -> dict:
        self.log.update(s["run_id"], runlog.FAILED, {"reason": "declined by user"})
        return {"reply": {"text": "Cancelled. Nothing was sent.", "status": runlog.COMPLETED, "data": {}}}

    def _preview(self, s: State) -> str:
        found = s.get("found") or {}
        actions = s.get("actions")
        lines = ["I plan to make these calls:"]
        shown_action = None
        for n, call in enumerate(s["calls"], 1):
            op = self.catalog[call["op"]]
            known = call.get("uses", s["known"])
            if actions and call.get("action") != shown_action:
                shown_action = call["action"]
                lines.append(f"Action {shown_action + 1} of {len(actions)}: {actions[shown_action]}")
            lines.append(f"{n}. {op.name}  ({op.method} {op.path})")
            if call["query"]:
                lines.append(f"   query: {json.dumps(call['query'])}")
            shown = {k: (f"<the {id_label(k)} action {int(v.rsplit('-', 1)[1]) + 1} creates>" if _is_pending(v) else v)
                     for k, v in known.items()}
            body = _filled(call).body(op, shown)
            if body:
                lines.append(f"   body: {json.dumps(body)}")
            used = [f"{name}=" + (f"the one action {int(known[name].rsplit('-', 1)[1]) + 1} creates"
                                  if _is_pending(known[name]) else str(known[name]))
                    + (f" ({found[name]})" if name in found else "")
                    for name in op.consumes if name in known]
            if used:
                lines.append(f"   using: {', '.join(used)}")
        if s["level"] == policy.EXPLICIT:
            lines.append("This includes a call that moves money, contacts someone else, or cannot be undone.")
        lines.append(REPROMPT)
        return "\n".join(lines)

    # -- nodes: doing ---------------------------------------------------------------

    def _execute(self, s: State) -> dict:
        known, results, produced_all, calls = dict(s["known"]), [], {}, [dict(c) for c in s["calls"]]
        in_action, current = {}, None
        for step, call in enumerate(calls):
            op, filled = self.catalog[call["op"]], _filled(call)
            ids = known
            if "uses" in call:  # part of a split message: this action's IDs, earlier actions' results
                if call["action"] != current:
                    current, in_action = call["action"], {}
                ids = {k: (produced_all.get(k, v) if _is_pending(v) else v) for k, v in call["uses"].items()}
                ids.update(in_action)
                waiting = [n for n in op.consumes if _is_pending(ids.get(n))]
                if waiting:
                    return self._stop_missing(s, op, results, waiting[0])
            try:
                path = render_path(op, ids, self.catalog.aliases)
            except MissingId as err:
                detail = {"failed_at": op.id, "reason": f"missing {err}", "results": results}
                self.log.update(s["run_id"], runlog.FAILED, detail)
                text = (f"I stopped at '{op.name}': an earlier step did not return the {id_label(str(err))} "
                        "it needed. Nothing further was sent.")
                return {"results": results, "reply": {"text": text, "status": runlog.FAILED, "data": detail}}
            body, dropped = filled.body(op, ids), {}
            with self.tracer.span("api_call", kind="api", operation=op.id, method=op.method,
                                  path=op.path, query=filled.query) as span:
                outcome = self._send_once(s, step, op, path, filled.query, body, span, ids)
                if outcome.get("stop"):
                    span.status = "error"
                    return self._stop(s, op, results, outcome["stop"])
                result, produced = outcome["result"], outcome["produced"]
                rejected = _rejected_filters(op, result, filled.query)
                if rejected:
                    # A read has no side effects: drop the filters PayPal refused and read once more.
                    # The answer is then drawn from the unfiltered list, and says so.
                    dropped = {name: filled.query[name] for name in rejected}
                    filled.query = {k: v for k, v in filled.query.items() if k not in rejected}
                    call["query"] = filled.query
                    result, produced = self.executor.call(op, ids, filled.query, body)
                span.set(http_status=result.status, produced=produced, dropped_filters=dropped)
                if not result.ok:
                    span.status = "error"
            results.append({"operation": op.id, "status": result.status, "body": result.body,
                            **({"filters_refused_by_api": dropped} if dropped else {})})
            if not result.ok:
                detail = {"failed_at": op.id, "results": results}
                self.log.update(s["run_id"], runlog.FAILED, detail)
                text = _failure_text(op, result, len(results) - 1)
                return {"results": results, "reply": {"text": text, "status": runlog.FAILED, "data": detail}}
            known.update(produced)
            produced_all.update(produced)
            in_action.update(produced)
        return {"results": results, "known": known, "ids": {**(s.get("ids") or {}), **produced_all},
                "calls": calls}

    def _send_once(self, s: State, step: int, op, path: str, query: dict, body, span, known: dict) -> dict:
        """Send one call unless the evidence shows it already went through.

        Returns {"result", "produced"} or {"stop": reason}. Reads are not recorded."""
        if op.risk == "read":
            result, produced = self.executor.call(op, known, query, body)
            return {"result": result, "produced": produced}
        prior = self.evidence.latest(s["run_id"], step)
        if prior and prior["state"] == ev.SENT:
            prior = self.reconciler.settle(prior)
            span.set(reconciled=prior["state"], note=prior["note"])
        if prior and prior["state"] in (ev.SUCCEEDED, ev.FAILED):
            span.set(from_evidence=prior["id"])  # already went through (or was refused): not sent again
            return {"result": Result(prior["http_status"], prior["response"]), "produced": prior["produced"]}
        if prior and prior["state"] == ev.NEEDS_ATTENTION:
            return {"stop": prior["note"]}
        record = self.evidence.intend(s["run_id"], s["session"], step, op.id, op.method, path, query, body)
        self.evidence.mark_sent(record["id"])
        result, produced = self.executor.call(op, known, query, body, request_id=record["request_id"])
        record = self.evidence.finish(record["id"], result.status, result.body, produced)
        span.set(evidence=record["id"], request_id=record["request_id"])
        if record["state"] == ev.SENT:
            # No answer at all: the request may still have reached PayPal. Settle it now.
            record = self.reconciler.settle(record)
            span.set(reconciled=record["state"], note=record["note"])
            if record["state"] == ev.NEEDS_ATTENTION:
                return {"stop": record["note"]}
            if record["state"] == ev.SUCCEEDED:
                return {"result": Result(record["http_status"], record["response"]), "produced": record["produced"]}
        return {"result": result, "produced": produced}

    def _stop_missing(self, s: State, op, results: list, name: str) -> dict:
        detail = {"failed_at": op.id, "reason": f"missing {name} from an earlier action", "results": results}
        self.log.update(s["run_id"], runlog.FAILED, detail)
        text = (f"I stopped at '{op.name}': the earlier action that should have created the {id_label(name)} "
                f"did not return it. {len(results)} earlier call(s) completed; nothing further was sent.")
        return {"results": results, "reply": {"text": text, "status": runlog.FAILED, "data": detail}}

    def _stop(self, s: State, op, results: list, reason: str) -> dict:
        detail = {"failed_at": op.id, "reason": reason, "results": results}
        self.log.update(s["run_id"], runlog.FAILED, detail)
        text = (f"I can't confirm whether '{op.name}' went through ({reason}). Please check it in the PayPal "
                f"dashboard before trying again. {len(results)} earlier call(s) completed; nothing further was sent.")
        return {"results": results, "reply": {"text": text, "status": runlog.FAILED, "data": detail}}

    def _compute(self, s: State) -> dict:
        """Read every page of the final result, then run a model-written, code-checked calculation."""
        call, last = s["calls"][-1], s["results"][-1]
        op = self.catalog[call["op"]]
        with self.tracer.span("compute") as span:
            result = Result(last["status"], last["body"])
            pages = self.executor.fetch_remaining_pages(op, s["known"], call["query"], result)
            spec = self.compute_planner.plan(s["message"], result.body, call["query"])
            computed = {**compute.run(spec, result.body), **pages} if spec else None
            if computed:
                span.set(result=computed["result"], description=computed["description"],
                         rows_matched=computed["rows_matched"], rows_total=computed["rows_total"],
                         pages=computed["pages"])
        return {"computed": computed}

    def _answer(self, s: State) -> dict:
        results, computed = s["results"], s.get("computed")
        self.log.update(s["run_id"], runlog.COMPLETED, {
            "results": [{"operation": r["operation"], "status": r["status"]} for r in results],
            "computed": computed,
        })
        with self.tracer.span("answer"):
            text = self._answer_computed(s["message"], computed) if computed else \
                self._answer_results(s["message"], results)
        data = {"run_id": s["run_id"], "results": results, "computed": computed}
        return {"reply": {"text": text, "status": runlog.COMPLETED, "data": data}}

    def _answer_results(self, message: str, results: list[dict]) -> str:
        shown = _snippet([{"operation": r["operation"], "status": r["status"], "response": r["body"],
                           **({"filters_refused_by_api": r["filters_refused_by_api"]}
                              if r.get("filters_refused_by_api") else {})}
                          for r in results], MAX_RESULT_CHARS)
        system = (
            "These API calls were just made on the user's behalf. Tell the user the outcome using only "
            "the results provided. Be brief and specific; include IDs and statuses that matter. For a "
            "question, answer it from the results; if they do not contain the answer, say so. "
            "If 'filters_refused_by_api' is present, the API refused those filters and the results are "
            "unfiltered: say so, and apply the filter yourself only where the results show it. "
            'Do not do arithmetic you cannot verify from the data. Reply with a JSON object: {"answer": string}'
        )
        try:
            reply = self.llm.json(system, f"User request: {message}\n\nAPI results:\n{shown}")
            if isinstance(reply.get("answer"), str) and reply["answer"].strip():
                return reply["answer"].strip()
        except Exception:
            pass
        done = ", ".join(f"{self.catalog[r['operation']].name} ({r['status']})" for r in results)
        return f"Done: {done}."

    def _answer_computed(self, message: str, computed: dict) -> str:
        """Word the answer with the model, but attach the exact figures and how they were derived in code."""
        facts = {k: computed[k] for k in ("description", "result", "rows_matched", "rows_total")}
        footer = (f"(Calculated in code: {computed['description']}; "
                  f"{computed['rows_matched']} of {computed['rows_total']} records")
        if computed["rows_skipped_not_numeric"]:
            footer += f"; {computed['rows_skipped_not_numeric']} skipped as not numeric"
        if not computed["complete"]:
            footer += f"; only the first {computed['pages']} pages were read, so this is incomplete"
        footer += ".)"
        system = (
            "A calculation was run in code over the full API results. Answer the user's request by stating "
            "its result exactly as given. Do not recompute, round, convert or add to the numbers. If the "
            "calculation does not really answer the question, say what it does show. "
            'Reply with a JSON object: {"answer": string}'
        )
        text = f"Result: {json.dumps(computed['result'])}"
        try:
            reply = self.llm.json(system, f"User request: {message}\n\nCalculation: {json.dumps(facts)}")
            if isinstance(reply.get("answer"), str) and reply["answer"].strip():
                text = reply["answer"].strip()
        except Exception:
            pass
        return f"{text}\n\n{footer}"


# -- routing -----------------------------------------------------------------------

def _after_select(s: State) -> str:
    intent = s["selection"]["intent"]
    return {"clarify": "ask", "system": "system_search", "docs": "rag"}.get(intent, "plan")


def _unresolved(s: State) -> list[str]:
    return [target for kind, target in s.get("steps") or [] if kind == "resolve"]


def _after_fill(s: State) -> str:
    if s.get("question"):
        return "ask"
    return "select" if s.get("more") else "check_ready"


def _after_ready(s: State) -> str:
    if s.get("reply"):
        return END
    if s.get("missing"):
        return "await_detail"
    return "execute" if s["level"] == policy.NONE else "await_approval"


MAX_DETAIL_ROUNDS = 4



def _covers(question: str, rule: list[dict]) -> bool:
    """Whether the LLM's question asks for every detail the code rules require."""
    lowered = question.lower()
    for m in rule:
        if m["path"] == ready.HINT:
            continue
        word = "currency" if m["path"] == ready.CURRENCY else "email" if m["path"].endswith("email_address") \
            else "name" if m["path"].endswith(".name") else None
        if word and word not in lowered:
            return False
    return True


def _after_approval(s: State) -> str:
    return {"yes": "execute", "no": "cancelled", "change": "split"}.get(s.get("approval"), "await_approval")


def _after_execute(s: State) -> str:
    if s.get("reply"):  # a call failed; the reply explains it
        return END
    # A calculation answers one question; for several actions the answer step reports each result.
    calculate = s["selection"]["calculate"] and s.get("results") and not s.get("actions")
    return "compute" if calculate else "answer"


# -- helpers ------------------------------------------------------------------------

def _reply(out: dict) -> Reply:
    pending = out.get("__interrupt__")
    if pending:
        value = pending[0].value
        status = runlog.WAITING_APPROVAL if value["kind"] == "approval" else runlog.WAITING_USER
        data = {"run_id": value["run_id"]} if value.get("run_id") else {}
        return Reply(value["text"], status, data)
    reply = out.get("reply") or {"text": "Something went wrong; nothing was sent.", "status": runlog.FAILED}
    return Reply(reply["text"], reply["status"], reply.get("data") or {})


PENDING = "@from-action-"


def _is_pending(value) -> bool:
    return isinstance(value, str) and value.startswith(PENDING)


def _text(s: State) -> str:
    """What the current step works on: the action being handled, or the whole request."""
    actions = s.get("actions")
    return actions[s.get("action_index") or 0] if actions else s["message"]


def _pending_ids(catalog: Catalog, s: State) -> dict:
    """IDs the actions already filled will create, as placeholders for the later actions."""
    out = {}
    for call in s.get("done_calls") or []:
        if catalog[call["op"]].risk == "read":
            continue  # a list or show finds IDs, it does not create them for the next action
        for name in catalog[call["op"]].produces:
            if not catalog.is_conditional(call["op"], name):
                out[name] = f"{PENDING}{call['action']}"
    return out


def _filled(call: dict) -> Filled:
    return Filled(query=dict(call["query"]), body_values=dict(call["body_values"]))


def _original(message: str) -> str:
    """The user's request without the answers and instructions merged into it later."""
    return re.split(r"\nThe user (?:added|changed the request):", message)[0]


# Where a PayPal app switches an API on: developer.paypal.com > Apps & Credentials > app > Features.
_FEATURE = {"transaction": "Transaction Search", "balance": "Transaction Search", "invoice": "Invoicing",
            "template": "Invoicing", "product": "Subscriptions", "plan": "Subscriptions",
            "subscription": "Subscriptions", "dispute": "Disputes", "payout-batch": "Payouts"}


def _rejected_filters(op, result, query: dict) -> list[str]:
    """Query filters a read call was refused for (400 naming them), so they can be dropped."""
    if op.method != "GET" or result.status != 400 or not isinstance(result.body, dict):
        return []
    named = {d.get("field") for d in result.body.get("details") or [] if isinstance(d, dict)}
    return [name for name in query if name in named]


def _failure_text(op, result, completed: int) -> str:
    earlier = f"{completed} earlier call(s) completed. I did not retry."
    if result.status == 403:
        feature = _FEATURE.get(op.resource)
        where = (f"Enable '{feature}' for the app in the PayPal developer dashboard (Apps & Credentials, "
                 "then the app's Features), then ask again." if feature else
                 "Check the app's features in the PayPal developer dashboard.")
        return f"PayPal refused '{op.name}': this app is not allowed to use that API. {where}\n{earlier}"
    return f"'{op.name}' failed with status {result.status}: {_snippet(result.body, 400)}\n{earlier}"


def _snippet(value: Any, limit: int) -> str:
    text = json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + " ...(truncated)"


def _format_system(result: dict) -> str:
    if result["kind"] == "capabilities":
        lines = ["Here is what I can do there:"]
        lines += [f"- {op['name']}" + ("" if op["risk"] == "read" else f"  [{op['risk']}]") for op in result["operations"]]
        return "\n".join(lines)
    if result["kind"] == "history":
        if not result["runs"]:
            return "You have not made any requests in this session yet."
        lines = ["Your recent requests, newest first:"]
        for run in result["runs"]:
            lines.append(f"- {run['message']!r}: {run['status']}")
            # What each call that changes something actually did, from the evidence store.
            lines += [f"    {call['name']}: {call['state'].replace('_', ' ')}"
                      + (f" ({call['note']})" if call.get("note") else "") for call in run.get("calls") or []]
        return "\n".join(lines)
    lines = ["I can work with these areas:"]
    lines += [f"- {r['resource']} ({r['actions']} actions)" for r in result["resources"]]
    return "\n".join(lines)

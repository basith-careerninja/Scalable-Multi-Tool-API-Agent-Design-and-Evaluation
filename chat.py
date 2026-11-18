"""Chat with the agent in the terminal.

Usage:
  python chat.py                    calls are answered by the mock (example responses)
  python chat.py --live             calls go to the PayPal sandbox (needs PAYPAL_CLIENT_ID/SECRET)
  python chat.py --session NAME     continue a saved conversation, e.g. one waiting for approval
  python chat.py --no-jev           the LLM chooses tools alone
  python chat.py --no-langsmith     keep traces local

Needs GROQ_API_KEY or GEMINI_API_KEY in .env. Type 'exit' to quit.
If JEV_API_KEY is set, Jev chooses the tool and the LLM is the fallback.
If LANGSMITH_API_KEY is set, each turn is traced in LangSmith.
Conversations are saved in checkpoints.sqlite, so they survive a restart. Every call that
changes something is recorded in evidence.sqlite before it is sent; at startup, calls left
unfinished by a crash are checked with PayPal, and a cut-off run is finished without
repeating what already went through.
"""
import os
import sqlite3
import sys
import uuid
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from agent.agent import Agent
from agent.catalog import Catalog
from agent.evidence import Evidence
from agent.execute import HttpTransport, MockTransport
from agent.jev import JevClient
from agent.llm import ChatClient, LLMError, load_env
from agent.observability import configure_langsmith
from agent.rag import DocsTool
from agent.runlog import RunLog
from agent.select import Selector
from agent.select_jev import JevSelector
from agent.split import Splitter
from agent.trace import SqliteSink, TracedLLM, Tracer

CHECKPOINTS = Path(__file__).resolve().parent / "checkpoints.sqlite"
EVIDENCE_LIVE = Path(__file__).resolve().parent / "evidence.sqlite"
EVIDENCE_MOCK = Path(__file__).resolve().parent / "evidence_mock.sqlite"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_env()
    args = sys.argv[1:]
    live = "--live" in args
    session = args[args.index("--session") + 1] if "--session" in args[:-1] else uuid.uuid4().hex[:8]
    try:
        llm = ChatClient()
        transport = HttpTransport() if live else MockTransport()
    except (LLMError, RuntimeError) as err:
        print(f"Cannot start: {err}")
        return 1
    try:
        project = configure_langsmith(enabled="--no-langsmith" not in args)
    except RuntimeError as err:
        print(f"LangSmith tracing is off: {err}")
        project = configure_langsmith(enabled=False)

    catalog = Catalog.load()
    tracer = Tracer(SqliteSink(), langsmith=bool(project))
    traced = TracedLLM(llm, tracer)
    docs = DocsTool(traced)
    generative = Selector(catalog, traced)
    selector, deciding, jev = generative, llm.model, None
    if (os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY")) and "--no-jev" not in args:
        try:
            jev = JevClient()
            selector = JevSelector(catalog, jev, fallback=generative, tracer=tracer, id_llm=traced)
            deciding = f"Jev ({jev.model}), falling back to {llm.model}"
        except LLMError as err:
            print(f"Jev is off: {err}")

    checkpointer = SqliteSaver(sqlite3.connect(str(CHECKPOINTS), check_same_thread=False))
    # Proof of what was sent, kept apart from the conversation state; one file per target, so a
    # mock run is never reconciled against the sandbox.
    evidence = Evidence(EVIDENCE_LIVE if live else EVIDENCE_MOCK)
    agent = Agent(catalog, traced, transport, RunLog(), docs=docs.answer, tracer=tracer,
                  selector=selector, checkpointer=checkpointer, jev=jev, ready_llm=traced, evidence=evidence,
                  splitter=Splitter(traced, jev=jev))

    target = "PayPal sandbox" if live else "mock (no real calls)"
    print(f"{len(catalog)} tools. Tool selection: {deciding}. Target: {target}.")
    print(f"Tracing: local" + (f" and LangSmith project '{project}'" if project else " only") + ".")
    for item in agent.reconcile_unsettled():
        print(f"Checked a call left unfinished by an earlier run: {catalog[item['operation']].name} "
              f"(session {item['session']}) -> {item['state'].replace('_', ' ')}: {item['note']}")
    finished = agent.resume_interrupted(session)
    if finished:
        print(f"\nagent> {finished.text}\n")
    pending = agent.graph.get_state({"configurable": {"thread_id": session}}).interrupts
    if pending:
        print(f"Session {session} is waiting for your reply:\n\nagent> {pending[0].value['text']}\n")
    else:
        print(f"Session {session} (continue it later with --session {session}). Type 'exit' to quit.\n")
    return loop(agent, session)


def loop(agent: Agent, session: str) -> int:
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not message:
            continue
        if message.lower() in ("exit", "quit"):
            return 0
        try:
            reply = agent.chat(session, message)
        except LLMError as err:
            print(f"agent> The model call failed: {err}\n")
            continue
        print(f"agent> {reply.text}\n")


if __name__ == "__main__":
    sys.exit(main())

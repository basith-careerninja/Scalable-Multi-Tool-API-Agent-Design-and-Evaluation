"""Show how one message is routed and planned. Nothing is executed.

Usage:
  python demo.py "send an invoice for $50 to sam@example.com"
  python demo.py --groq "..."     decide with the LLM alone, for comparison

Uses Jev when JEV_API_KEY is set, with the LLM as its fallback.
"""
import os
import sys

from agent import policy
from agent.catalog import Catalog
from agent.llm import ChatClient, LLMError, load_env
from agent.planner import Planner
from agent.select import Selector


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_env()
    llm_only = "--groq" in sys.argv[1:]
    args = [a for a in sys.argv[1:] if a != "--groq"]
    if not args:
        print(__doc__)
        return 2
    message = " ".join(args)
    catalog = Catalog.load()
    try:
        llm = ChatClient()
        selector = Selector(catalog, llm)
        if not llm_only and (os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY")):
            from agent.jev import JevClient
            from agent.select_jev import JevSelector
            selector = JevSelector(catalog, JevClient(), fallback=selector, id_llm=llm)
        result = selector.select(message)
    except LLMError as err:
        print(f"LLM error: {err}")
        return 1

    reason = f"  ({result.reason})" if result.reason else ""
    print(f"decided by : {result.decided_by}{reason}")
    print(f"intent     : {result.intent}  (confidence {result.confidence:.2f}, {result.llm_calls} model calls)")
    if result.scores:
        print("top tools  : " + ", ".join(f"{op}={score:.2f}" for op, score in result.scores))
    if result.intent == "clarify":
        print(f"question   : {result.question}")
        return 0
    if result.intent != "action":
        print("handled by : " + ("docs search tool" if result.intent == "docs" else "system search tool"))
        return 0
    print(f"operation  : {result.operation}")
    print(f"create new : {', '.join(result.new_ids) or '-'}")
    print(f"typed IDs  : {', '.join(f'{k}={v}' for k, v in result.known_ids) or '-'}")
    print(f"calculate  : {result.calculate}")
    print("plan       :")
    for n, step in enumerate(Planner(catalog).plan(result.operation, known=dict(result.known_ids),
                                                    new=result.new_ids), 1):
        if step.kind == "call":
            op = catalog[step.target]
            print(f"  {n}. {op.method} {op.path}  [{op.name}; approval: {policy.approval(op)}]")
        else:
            print(f"  {n}. look up {step.target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

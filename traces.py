"""Look at what the agent did.

Usage:
  python traces.py                 recent turns, one line each
  python traces.py last            the most recent turn as a step tree
  python traces.py <trace_id>      one turn as a step tree
  python traces.py last -v         also show prompts, responses and request bodies
"""
import sys

from agent.trace import SqliteSink, render


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = [a for a in sys.argv[1:] if a != "-v"]
    verbose = "-v" in sys.argv[1:]
    sink = SqliteSink()
    recent = sink.recent(15 if not args else 1)
    if not recent:
        print("No traces yet. Run `python chat.py` first.")
        return 0
    if not args:
        print(f"{'trace':12}  {'status':16} {'total':>8} {'model':>8} {'calls':>5} {'tokens':>7} {'api':>3}  message")
        for t in recent:
            tokens = t["prompt_tokens"] + t["completion_tokens"]
            message = t["message"].replace("\n", " ")[:60]
            print(f"{t['trace_id']:12}  {t['status']:16} {t['duration_ms'] / 1000:7.1f}s {t['llm_ms'] / 1000:7.1f}s "
                  f"{t['llm_calls']:5} {tokens:7} {t['api_calls']:3}  {message}")
        return 0
    trace_id = recent[0]["trace_id"] if args[0] == "last" else args[0]
    spans = sink.trace(trace_id)
    if not spans:
        print(f"No trace with id {trace_id}.")
        return 1
    print(render(spans, verbose))
    return 0


if __name__ == "__main__":
    sys.exit(main())

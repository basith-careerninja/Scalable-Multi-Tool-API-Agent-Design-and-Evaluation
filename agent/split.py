"""Split a message into the ordered actions it asks for.

Selection picks one goal at a time, and a message such as "create a product and a monthly
plan for it" has two. This step runs first:
  1. A cheap gate: Jev answers one yes/no question, "more than one action?". Most
     messages are one action and cost nothing more. Without Jev, a code check for words
     that join actions ("and", "then", a comma) is used instead.
  2. Only if the gate says yes, one LLM call returns the actions in order, each a short
     request in the user's words.
  3. Code checks the split: 2 to 5 actions, each made of the user's own words (so no
     action is invented), else the message is handled as one action.
Each action is then selected, planned and filled in turn; IDs flow forward (agent.py).
"""
from __future__ import annotations

import re

from . import config
from .jev import yes

MAX_ACTIONS = 5
MIN_OWN_WORDS = 0.75   # share of an action's content words that must come from the message
_JOINERS = re.compile(r"\b(and|then|also|after that|afterwards)\b|[,;]", re.I)
_WORD = re.compile(r"[a-z0-9$€£.@_-]+")
_FILLER = {"a", "an", "the", "it", "its", "that", "this", "them", "for", "to", "of", "on", "in", "with", "and",
           "from", "action", "result", "step", "my", "me", "i", "please", "then", "new", "same", "one", "created",
           "just", "above", "previous", "earlier", "first", "second", "third"}


class Splitter:
    def __init__(self, llm, jev=None, tracer=None):
        self.llm = llm
        self.jev = jev
        self.tracer = tracer

    def split(self, message: str) -> tuple[list[str], dict]:
        """Returns (actions, how it was decided). One action means: handle the message as is."""
        info = {"gate": None, "proposed": None, "rejected": None}
        if not self._gate(message, info):
            return [message], info
        reply = self.llm.json(self._system(), message)
        actions = reply.get("actions") if isinstance(reply, dict) else None
        info["proposed"] = actions
        problem = check(message, actions)
        if problem:
            info["rejected"] = problem
            return [message], info
        return [a.strip() for a in actions], info

    def _gate(self, message: str, info: dict) -> bool:
        if self.jev is not None:
            try:
                answers = self.jev.ask({"user_message": message}, {"several": {
                    "type": "noul",
                    "instructions": "Does this message ask for two or more separate actions on the account "
                                    "(for example 'create a product and a plan for it', 'cancel X and refund Y'), "
                                    "rather than one action, however many details it gives? Steps the system "
                                    "would do anyway to complete one action (creating an invoice in order to "
                                    "send it) are one action, and so is one request that covers several people "
                                    "or items at once (one payout to several recipients, one invoice with "
                                    "several items).",
                }})
                p = yes(answers["several"])
                info["gate"] = {"by": "jev", "yes": round(p, 3)}
                return p >= config.YES_THRESHOLD
            except Exception as err:  # fall back to the code check
                info["gate"] = {"by": "jev-error", "error": str(err)[:120]}
        joined = bool(_JOINERS.search(message))
        info["gate"] = {**(info["gate"] or {}), "by_words": joined}
        return joined

    @staticmethod
    def _system() -> str:
        return "\n".join([
            "Split the user's message into the separate actions it asks for, in the order they must happen.",
            'Reply with a JSON object: {"actions": [string, ...]}',
            "",
            "Rules:",
            "- Each action is one short request in the user's own words, with the details that belong to it "
            "(amounts, names, IDs, emails).",
            "- An action that uses the result of an earlier one may say so in a few words, e.g. 'create a "
            "monthly plan for the product at $20'.",
            "- Never add an action the user did not ask for, and never drop one.",
            "- Steps needed to complete one action are not separate actions: 'send an invoice to sam' is one "
            "action, even though the invoice is created first.",
            "- One request that covers several people or items at once is ONE action: 'pay out $100 each "
            "to alice and bob' is one payout batch, 'invoice sam for a mat and a shirt' is one invoice.",
            "- If it is really one action, return just that one.",
        ])


def check(message: str, actions) -> str | None:
    """Why a proposed split is unusable, or None if it is fine."""
    if not isinstance(actions, list) or not all(isinstance(a, str) and a.strip() for a in actions):
        return "not a list of actions"
    if len(actions) < 2:
        return "one action"
    if len(actions) > MAX_ACTIONS:
        return f"more than {MAX_ACTIONS} actions"
    said = {_stem(w) for w in _words(message)}
    for action in actions:
        content = [_stem(w) for w in _words(action) if w not in _FILLER]
        if content and sum(w in said for w in content) / len(content) < MIN_OWN_WORDS:
            return f"action not in the user's words: {action!r}"
    return None


def _stem(word: str) -> str:
    """Crude, on purpose: 'captured' and 'capture' match, '$5' and '5 dollars' match."""
    word = word.lstrip("$€£").rstrip(".")
    for suffix in ("ing", "ed", "es", "e", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _words(text: str) -> set[str]:
    return {w.strip(".,") for w in _WORD.findall(text.lower()) if w.strip(".,")}

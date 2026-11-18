"""Approval rules. Decided by code from the operation's risk tier, never by the model."""
from __future__ import annotations

from .catalog import Operation

NONE = "none"          # run without asking
CONFIRM = "confirm"    # show the call, one confirmation
EXPLICIT = "explicit"  # show amount and recipient; never retried automatically

_RULES = {
    "read": NONE,
    "write": CONFIRM,
    "money": EXPLICIT,
    "outbound": EXPLICIT,
    "destructive": EXPLICIT,
}


def approval(op: Operation) -> str:
    # Unknown tiers get the strictest rule.
    return _RULES.get(op.risk, EXPLICIT)

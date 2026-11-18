"""Tunable numbers in one place. Each can be overridden with an AGENT_* environment variable.

The defaults are the starting values the agent shipped with; they are meant to be tuned
on the eval's tuning half. Safety rules (approval tiers, never retrying money or outbound
calls) are not tunables and live in agent/policy.py.
"""
from __future__ import annotations

import os

PREFIX = "AGENT_"


class ConfigError(ValueError):
    pass


def _read(name: str, default, cast):
    raw = os.environ.get(PREFIX + name)
    if raw is None or not raw.strip():
        return default
    try:
        return cast(raw.strip())
    except ValueError:
        raise ConfigError(f"{PREFIX}{name}={raw!r} is not a valid {cast.__name__}") from None


def _float(name: str, default: float) -> float:
    return _read(name, default, float)


def _int(name: str, default: int) -> int:
    return _read(name, default, int)


# Jev selection (agent/select_jev.py)
ACT_THRESHOLD_READ = _float("ACT_THRESHOLD_READ", 0.5)   # reads are cheap to get wrong and can be refined
ACT_THRESHOLD = _float("ACT_THRESHOLD", 0.7)             # writes, money, outbound, destructive
CLOSE_GAP = _float("CLOSE_GAP", 0.2)                     # two tools this close count as a tie
TIE_MIN_RUNNER_UP = _float("TIE_MIN_RUNNER_UP", 0.05)    # a runner-up below this is not a tie
INTENT_THRESHOLD = _float("INTENT_THRESHOLD", 0.6)
KEEP_RESOURCE = _float("KEEP_RESOURCE", 0.1)             # resources at least this likely are searched
MAX_RESOURCES = _int("MAX_RESOURCES", 3)
KEEP_SERVICE = _float("KEEP_SERVICE", 0.1)               # with several vendors: services at least this likely
MAX_SERVICES = _int("MAX_SERVICES", 2)
YES_THRESHOLD = _float("YES_THRESHOLD", 0.7)             # TypeSafe's suggested threshold for a yes

# LLM selector (agent/select.py)
LLM_MIN_CONFIDENCE = _float("LLM_MIN_CONFIDENCE", 0.6)

# Look-up and paging (agent/resolve.py, agent/execute.py)
LOOKUP_MAX_ROWS = _int("LOOKUP_MAX_ROWS", 20)
MAX_PAGES = _int("MAX_PAGES", 20)

# Approval replies judged by Jev (agent/agent.py)
APPROVE_MIN = _float("APPROVE_MIN", 0.9)
APPROVE_MAX_CHANGE = _float("APPROVE_MAX_CHANGE", 0.1)
DECLINE_MIN = _float("DECLINE_MIN", 0.7)

# System search hints from the Jev route request (agent/system_search.py)
SYSTEM_KIND_MIN = _float("SYSTEM_KIND_MIN", 0.6)
SYSTEM_RESOURCE_MIN = _float("SYSTEM_RESOURCE_MIN", 0.5)

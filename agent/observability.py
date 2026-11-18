"""Turn on LangSmith tracing for the LangGraph agent.

LangGraph traces every graph run and node to LangSmith by itself once these are set:
  LANGSMITH_TRACING=true, LANGSMITH_API_KEY, LANGSMITH_ENDPOINT, LANGSMITH_PROJECT
The model, Jev and API calls are added under their node by agent.trace.Tracer.

A LangSmith key only works in the region it was created in, so when no endpoint
is set this finds the right one.
"""
from __future__ import annotations

import os
import urllib.error
import urllib.request

REGION_ENDPOINTS = (
    "https://api.smith.langchain.com",
    "https://eu.api.smith.langchain.com",
    "https://apac.api.smith.langchain.com",
    "https://aws.api.smith.langchain.com",
)
DEFAULT_PROJECT = "multi-tool-agent"


def configure_langsmith(enabled: bool = True, timeout: float = 15.0) -> str | None:
    """Set the environment for LangSmith tracing. Returns the project name, or None when off.

    Raises RuntimeError if a key is set but no region accepts it.
    """
    key = os.environ.get("LANGSMITH_API_KEY")
    if not enabled or not key:
        os.environ["LANGSMITH_TRACING"] = "false"
        return None
    if not os.environ.get("LANGSMITH_ENDPOINT"):
        os.environ["LANGSMITH_ENDPOINT"] = detect_endpoint(key, timeout)
    os.environ.setdefault("LANGSMITH_PROJECT", DEFAULT_PROJECT)
    os.environ["LANGSMITH_TRACING"] = "true"
    return os.environ["LANGSMITH_PROJECT"]


def detect_endpoint(key: str, timeout: float = 15.0) -> str:
    for host in REGION_ENDPOINTS:
        request = urllib.request.Request(f"{host}/api/v1/sessions?limit=1",
                                         headers={"x-api-key": key, "User-Agent": "agent-client/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=timeout):
                return host
        except (urllib.error.URLError, TimeoutError):
            continue
    raise RuntimeError("LANGSMITH_API_KEY was rejected by every LangSmith region; "
                       "check the key, or set LANGSMITH_ENDPOINT for a self-hosted server")

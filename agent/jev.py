"""Client for Jev, TypeSafe's decision model (System One API).

Jev does not generate text. It answers typed questions about a given state:
  choice  pick one of up to 255 named options; returns a probability per option
  noul    a yes/no question; returns the probability of yes
Several questions go in one request and are answered together.

Configure in .env:
  JEV_API_KEY (or TYPESAFE_API_KEY)   required
  JEV_MODEL                           optional, default jev-1.14.0 (pinned so results are repeatable)
  TYPESAFE_BASE_URL                   optional, default https://api.typesafe.ai

Docs: https://docs.typesafe.ai
"""
from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request

from .llm import LLMError, _retry_delay, load_env

DEFAULT_MODEL = "jev-1.14.0"
MAX_OPTIONS = 255


class JevError(LLMError):
    pass


class JevClient:
    def __init__(self, api_key: str | None = None, model: str | None = None,
                 base_url: str | None = None, timeout: float = 30.0):
        load_env()
        self.api_key = api_key or os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise JevError("set JEV_API_KEY in .env")
        self.model = model or os.environ.get("JEV_MODEL") or DEFAULT_MODEL
        base = (base_url or os.environ.get("TYPESAFE_BASE_URL") or "https://api.typesafe.ai").rstrip("/")
        self.url = f"{base}/v1/systemone"
        self.timeout = timeout
        self.last_usage: dict = {}
        self.served_model: str | None = None

    def ask(self, state: dict, questions: dict) -> dict:
        """Ask questions about the state. Returns {question name: answer}."""
        for name, question in questions.items():
            if question.get("type") == "choice" and len(question["criteria"]) > MAX_OPTIONS:
                raise JevError(f"question {name!r} has more than {MAX_OPTIONS} options")
        body = json.dumps({"model": self.model, "state": state, "questions": questions}).encode()
        request = urllib.request.Request(self.url, data=body, method="POST", headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "agent-client/0.1",
        })
        attempts = 4
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                break
            except urllib.error.HTTPError as err:
                detail = err.read().decode(errors="replace")[:300]
                # 429 rate limited, 529 overloaded: both are meant to be retried.
                if err.code in (429, 500, 502, 503, 529) and attempt < attempts - 1:
                    time.sleep(_retry_delay(err.headers.get("Retry-After"), detail, attempt))
                    continue
                raise JevError(f"HTTP {err.code} from {self.url}: {detail}") from err
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as err:
                if attempt < attempts - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise JevError(f"could not reach {self.url}: {err}") from err
        answers = payload.get("answers") or {}
        missing = [name for name in questions if name not in answers]
        if missing:
            raise JevError(f"no answer for {missing}")
        self.last_usage = payload.get("usage") or {}
        self.served_model = payload.get("model")
        return answers


def probabilities(answer: dict, allowed) -> dict[str, float]:
    """Per-option probabilities of a choice answer, limited to options we actually offered."""
    allowed = set(allowed)
    probs = {k: float(v) for k, v in (answer.get("probabilities") or {}).items() if k in allowed}
    if not probs and answer.get("choice") in allowed:
        probs = {answer["choice"]: 1.0}
    return probs


def yes(answer: dict) -> float:
    """Probability of yes from a noul answer."""
    try:
        return float(answer.get("noul", 0.0))
    except (TypeError, ValueError):
        return 0.0

"""Minimal chat client for OpenAI-compatible endpoints (Groq, Gemini).

Uses JSON mode, which both providers accept, and leaves validation to the
caller. Standard library only, so there is nothing to install.

Configure in a .env file at the project root, or with environment variables:
  LLM_PROVIDER   groq (default), gemini or openai
  GROQ_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY
  LLM_MODEL      optional override of the provider's default model
"""
from __future__ import annotations

import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", "gemini-3-flash-preview"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-5-mini"),
}
# Sampling settings per provider. GPT-5 models accept only their default temperature, so they
# get a reasoning effort instead; the others run at temperature 0 for repeatable answers.
SAMPLING = {
    "groq": {"temperature": 0},
    "gemini": {"temperature": 0},
    "openai": {"reasoning_effort": "low"},
}


class LLMError(RuntimeError):
    pass


def load_env(path: Path | None = None) -> None:
    """Read KEY=VALUE lines from the project's .env. Real environment variables win."""
    path = path or Path(__file__).resolve().parent.parent / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _first_object(text: str) -> dict | None:
    """The JSON object in a reply. A list of objects (some models answer a request with
    several parts that way) yields its first; anything else yields None."""
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return value if isinstance(value, dict) else None


def _retry_delay(retry_after: str | None, detail: str, attempt: int) -> float:
    """Wait as long as the provider asks (header or 'try again in 3.4s'), else back off."""
    seconds = None
    try:
        seconds = float(retry_after) if retry_after else None
    except ValueError:
        pass
    if seconds is None:
        match = re.search(r"try again in ([\d.]+)\s*(ms|s)", detail)
        if match:
            seconds = float(match.group(1)) / (1000 if match.group(2) == "ms" else 1)
    if seconds is None:
        seconds = 2 ** attempt
    return min(seconds + 1.0, 65.0)


class ChatClient:
    def __init__(self, provider: str | None = None, model: str | None = None,
                 api_key: str | None = None, timeout: float = 30.0):
        load_env()
        provider = (provider or os.environ.get("LLM_PROVIDER", "groq")).lower()
        if provider not in PROVIDERS:
            raise LLMError(f"unknown provider {provider!r}; use one of {sorted(PROVIDERS)}")
        base_url, key_var, default_model = PROVIDERS[provider]
        self.sampling = SAMPLING[provider]
        self.url = f"{base_url}/chat/completions"
        self.model = model or os.environ.get("LLM_MODEL") or default_model
        self.api_key = api_key or os.environ.get(key_var)
        if not self.api_key:
            raise LLMError(f"set {key_var} in .env (copy .env.example) or as an environment variable")
        self.timeout = timeout
        self.last_usage: dict = {}  # token counts from the most recent call, for tracing

    def _wait(self, seconds: float) -> None:
        self.last_retry["retries"] += 1
        self.last_retry["waited_s"] = round(self.last_retry["waited_s"] + seconds, 1)
        if seconds >= 5:  # long enough that a silent pause looks like a hang
            print(f"  ({self.model} is busy or rate-limited; retrying in {seconds:.0f}s)", file=sys.stderr, flush=True)
        time.sleep(seconds)

    def json(self, system: str, user: str) -> dict:
        """Return the model's reply parsed as a JSON object."""
        body = json.dumps({
            "model": self.model,
            **self.sampling,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }).encode()
        request = urllib.request.Request(self.url, data=body, method="POST", headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "agent-client/0.1",
        })
        attempts = 5
        self.last_retry = {"retries": 0, "waited_s": 0.0}  # for tracing: time lost to rate limits
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                self.last_usage = payload.get("usage") or {}
                text = payload["choices"][0]["message"]["content"]
                result = _first_object(text)
                if result is None:
                    if attempt < 1:  # a malformed reply is usually fine on a second try
                        continue
                    raise LLMError(f"model returned no JSON object: {text[:120]!r}")
                return result
            except urllib.error.HTTPError as err:
                detail = err.read().decode(errors="replace")[:300]
                if err.code in (429, 500, 502, 503) and attempt < attempts - 1:
                    self._wait(_retry_delay(err.headers.get("Retry-After"), detail, attempt))
                    continue
                if err.code == 400 and "json_validate_failed" in detail and attempt < 1:
                    # The model answered in prose instead of JSON. One more try usually works.
                    self._wait(0.5)
                    continue
                raise LLMError(f"HTTP {err.code} from {self.url}: {detail}") from err
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as err:
                if attempt < attempts - 1:
                    self._wait(2 ** attempt)
                    continue
                raise LLMError(f"could not reach {self.url}: {err}") from err
            except (KeyError, IndexError, json.JSONDecodeError) as err:
                raise LLMError(f"unexpected response shape: {err}") from err
        raise LLMError("retries exhausted")

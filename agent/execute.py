"""Executing one prepared call, against a mock or the real PayPal sandbox.

MockTransport replays the example response stored in the collection, so the
whole agent can run with no account and no network. HttpTransport calls the
PayPal sandbox and handles the OAuth token itself: the model never sees or
handles credentials.
"""
from __future__ import annotations

import base64
import copy
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any

from . import config
from .catalog import Catalog, Operation

_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


@dataclass
class Result:
    status: int
    body: Any

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


class MissingId(Exception):
    pass


def render_path(op: Operation, ids: dict[str, str], aliases: dict[str, str]) -> str:
    """Substitute :name and {{name}} path segments from the plan's state."""
    parts = []
    for segment in op.path.split("/"):
        if segment.startswith(":") or segment.startswith("{{"):
            raw = segment.strip(":{}")
            name = op.path_aliases.get(raw) or aliases.get(raw, raw)
            if name not in ids:
                raise MissingId(name)
            parts.append(urllib.parse.quote(str(ids[name]), safe=""))
        else:
            parts.append(segment)
    return "/".join(parts)


def extract_ids(op: Operation, body: Any) -> dict[str, str]:
    """Read the IDs this operation produces out of its response."""
    found = {}
    for name, path in op.extract.items():
        node = body
        try:
            for key, index in _PATH_TOKEN.findall(path):
                node = node[int(index)] if index else node[key]
        except (KeyError, IndexError, TypeError):
            continue
        if isinstance(node, (str, int)) and node != "":
            found[name] = str(node)
    return found


class MockTransport:
    """Returns each operation's example response. Records what it was asked to send."""

    def __init__(self):
        self.calls: list[dict] = []

    def send(self, op: Operation, path: str, query: dict, body: Any, request_id: str | None = None) -> Result:
        self.calls.append({"op": op.id, "method": op.method, "path": path, "query": query, "body": body})
        if op.example_status is None:
            return Result(204 if op.method in ("DELETE", "PATCH") else 200, None)
        # A copy, so nothing downstream can alter the catalogue's stored example.
        return Result(op.example_status, copy.deepcopy(op.example_body))


SANDBOX_URL = "https://api-m.sandbox.paypal.com"


class NotSandbox(RuntimeError):
    pass


class HttpTransport:
    """Calls the PayPal REST API. Only the sandbox: any other host is refused, so no real money moves."""

    def __init__(self, base_url: str | None = None, client_id: str | None = None,
                 client_secret: str | None = None, timeout: float = 30.0, urlopen=None):
        self.base_url = (base_url or os.environ.get("PAYPAL_BASE_URL") or SANDBOX_URL).rstrip("/")
        if "sandbox" not in self.base_url.lower():
            raise NotSandbox(f"refusing to call {self.base_url}: only the PayPal sandbox is allowed")
        self.client_id = client_id or os.environ.get("PAYPAL_CLIENT_ID")
        self.client_secret = (client_secret or os.environ.get("PAYPAL_CLIENT_SECRET")
                              or os.environ.get("PAYPAL_SECRET_KEY"))
        if not (self.client_id and self.client_secret):
            raise RuntimeError("set PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET (or PAYPAL_SECRET_KEY) in .env")
        self.timeout = timeout
        self._urlopen = urlopen or urllib.request.urlopen
        self._token, self._expires = None, 0.0

    def _access_token(self) -> str:
        if self._token and time.time() < self._expires - 60:
            return self._token
        basic = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        request = urllib.request.Request(
            f"{self.base_url}/v1/oauth2/token", data=b"grant_type=client_credentials", method="POST",
            headers={"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded",
                     "Accept": "application/json"},
        )
        # Errors name the status and PayPal's error code only, never the request (it holds the credentials).
        try:
            with self._urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as err:
            detail = _json_or_none(err.read())
            code = detail.get("error") if isinstance(detail, dict) else None
            raise RuntimeError(f"PayPal token request failed: HTTP {err.code} {code or ''}".strip()) from None
        except (urllib.error.URLError, TimeoutError) as err:
            raise RuntimeError(f"PayPal token request failed: network error {type(err).__name__}") from None
        self._token = payload["access_token"]
        self._expires = time.time() + float(payload.get("expires_in", 300))
        return self._token

    def request(self, method: str, path: str, query: dict | None = None, body: Any = None,
                request_id: str | None = None) -> Result:
        """One authorised call by method and path; used by send() and by the sandbox scripts."""
        try:
            token = self._access_token()
        except RuntimeError as err:
            return Result(0, {"error": str(err)})
        url = self.base_url + path + ("?" + encode_query(query) if query else "")
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "agent-client/0.1",
        }
        data = None
        if method in ("POST", "PUT", "PATCH"):
            # PayPal rejects a POST without a JSON body on some endpoints (e.g. send invoice).
            data = json.dumps({} if body is None and method == "POST" else body).encode()
            headers["Content-Type"] = "application/json"
            # Full objects back, so IDs can be read from create responses (invoicing returns a link otherwise).
            headers["Prefer"] = "return=representation"
        if method == "POST":
            # Idempotency: PayPal treats a repeated request with the same id as the same call.
            # The agent passes the id recorded in its evidence, so a replay after a crash is recognised.
            headers["PayPal-Request-Id"] = request_id or str(uuid.uuid4())
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with self._urlopen(request, timeout=self.timeout) as response:
                return Result(response.status, _json_or_none(response.read()))
        except urllib.error.HTTPError as err:
            return Result(err.code, _json_or_none(err.read()))
        except (urllib.error.URLError, TimeoutError) as err:
            return Result(0, {"error": f"network error: {err}"})

    def send(self, op: Operation, path: str, query: dict, body: Any, request_id: str | None = None) -> Result:
        return self.request(op.method, path, query, None if op.method == "GET" else body, request_id)


def encode_query(query: dict) -> str:
    """Query string as PayPal expects it: booleans as true/false, lists repeated, empty values dropped."""
    pairs = []
    for key, value in query.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        for v in values:
            if v is None or v == "":
                continue
            pairs.append((key, ("true" if v else "false") if isinstance(v, bool) else str(v)))
    return urllib.parse.urlencode(pairs)


def _json_or_none(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw.decode(errors="replace")[:500]}


class Executor:
    def __init__(self, catalog: Catalog, transport):
        self.catalog = catalog
        self.transport = transport

    def call(self, op: Operation, ids: dict[str, str], query: dict, body: Any,
             request_id: str | None = None) -> tuple[Result, dict[str, str]]:
        """Run one call. Returns the result and any IDs it produced."""
        path = render_path(op, ids, self.catalog.aliases)
        if request_id:
            result = self.transport.send(op, path, query, body, request_id=request_id)
        else:
            result = self.transport.send(op, path, query, body)
        return result, (extract_ids(op, result.body) if result.ok else {})

    def fetch_remaining_pages(self, op: Operation, ids: dict[str, str], query: dict, first: Result,
                              max_pages: int = config.MAX_PAGES) -> dict:
        """For a paged read, fetch pages 2..n and append their rows to the first result.

        Applies when the operation takes a `page` parameter and the response reports
        `total_pages`. Returns how many pages were read and whether the cap cut it short.
        """
        body = first.body
        takes_page = any(q["name"] == "page" for q in op.query)
        total = body.get("total_pages") if isinstance(body, dict) else None
        if op.method != "GET" or not takes_page or not isinstance(total, int) or total <= 1:
            return {"pages": 1, "complete": True}
        path = render_path(op, ids, self.catalog.aliases)
        last = min(total, max_pages)
        for page in range(2, last + 1):
            result = self.transport.send(op, path, {**query, "page": str(page)}, None)
            if not result.ok or not isinstance(result.body, dict):
                return {"pages": page - 1, "complete": False}
            for key, rows in result.body.items():
                if isinstance(rows, list) and isinstance(body.get(key), list) and key != "links":
                    body[key] = body[key] + rows
        return {"pages": last, "complete": last == total}

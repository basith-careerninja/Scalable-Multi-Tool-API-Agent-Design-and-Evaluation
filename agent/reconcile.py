"""Settling calls whose outcome is unknown (evidence state "sent").

Three ways, tried in order:
  1. Replay with the same PayPal-Request-Id, for APIs where PayPal documents that header:
     PayPal recognises the repeat and returns the original result instead of acting twice.
  2. Read back: ask PayPal whether the effect is there (an invoice with our number exists,
     an invoice is no longer a draft).
  3. Neither possible: "needs attention". The agent says so and sends nothing further.
     It never guesses.

The per-API knowledge below (which paths honour the header, how to read an effect back)
is kept small and explicit, like the feature names in agent/agent.py.
"""
from __future__ import annotations

from .catalog import Catalog
from .evidence import NEEDS_ATTENTION, NOT_APPLIED, SENT, SUCCEEDED, FAILED, Evidence
from .execute import Executor, Result, extract_ids

# APIs whose POSTs PayPal documents as idempotent under PayPal-Request-Id.
IDEMPOTENT_PATHS = ("/v2/checkout/orders", "/v2/payments/", "/v1/payments/payouts", "/v1/billing/",
                    "/v1/catalogs/products", "/v3/vault/")
# Calls whose repeat does no harm (a generated invoice number only skips one).
SAFE_TO_REPEAT = {"invoice.generate-number"}


class Reconciler:
    def __init__(self, catalog: Catalog, executor: Executor, evidence: Evidence):
        self.catalog = catalog
        self.executor = executor
        self.evidence = evidence

    def settle(self, record: dict) -> dict:
        if record["state"] != SENT:
            return record
        op = self.catalog[record["operation"]]
        if op.method == "POST" and record["path"].startswith(IDEMPOTENT_PATHS):
            return self._replay(record, op)
        if op.id in SAFE_TO_REPEAT:
            return self.evidence.settle(record["id"], NOT_APPLIED, "outcome unknown, but safe to repeat")
        read_back = _READ_BACK.get(op.id)
        if read_back:
            return read_back(self, record)
        return self.evidence.settle(record["id"], NEEDS_ATTENTION,
                                    "PayPal offers no way to check whether this call happened")

    def _send(self, op_id: str, path: str, query: dict, body, request_id: str | None = None) -> Result:
        op = self.catalog[op_id]
        if request_id:
            return self.executor.transport.send(op, path, query, body, request_id=request_id)
        return self.executor.transport.send(op, path, query, body)

    def _replay(self, record: dict, op) -> dict:
        request = record["request"]
        result = self._send(op.id, record["path"], request["query"], request["body"], record["request_id"])
        if result.status == 0:
            return self.evidence.settle(record["id"], NEEDS_ATTENTION, "PayPal did not answer the replay either")
        produced = extract_ids(op, result.body) if result.ok else {}
        return self.evidence.settle(record["id"], SUCCEEDED if result.ok else FAILED,
                                    "replayed with the same PayPal-Request-Id", result.status, result.body, produced)

    def _invoice_created(self, record: dict) -> dict:
        body = record["request"]["body"] or {}
        number = (body.get("detail") or {}).get("invoice_number")
        if not number:
            return self.evidence.settle(record["id"], NEEDS_ATTENTION, "the invoice had no number to look it up by")
        search = self.catalog["invoice.search"]
        result = self._send(search.id, search.path, {}, {"invoice_number": number})
        if not result.ok or not isinstance(result.body, dict):
            return self.evidence.settle(record["id"], NEEDS_ATTENTION, f"the invoice search failed ({result.status})")
        found = [item for item in result.body.get("items") or []
                 if (item.get("detail") or {}).get("invoice_number") == number]
        if not found:
            return self.evidence.settle(record["id"], NOT_APPLIED, f"no invoice numbered {number} exists")
        invoice = found[0]
        return self.evidence.settle(record["id"], SUCCEEDED, f"found invoice {invoice.get('id')} by its number",
                                    201, invoice, {"invoice_id": str(invoice.get("id"))})

    def _invoice_sent(self, record: dict) -> dict:
        invoice_id = record["path"].rstrip("/").split("/")[-2]
        show = self.catalog["invoice.show"]
        result = self._send(show.id, f"/v2/invoicing/invoices/{invoice_id}", {}, None)
        status = result.body.get("status") if result.ok and isinstance(result.body, dict) else None
        if status is None:
            return self.evidence.settle(record["id"], NEEDS_ATTENTION, f"could not read the invoice ({result.status})")
        if status == "DRAFT":
            return self.evidence.settle(record["id"], NOT_APPLIED, "the invoice is still a draft")
        return self.evidence.settle(record["id"], SUCCEEDED, f"the invoice is {status}", 200, {"status": status}, {})


_READ_BACK = {
    "invoice.create": Reconciler._invoice_created,
    "invoice.send": Reconciler._invoice_sent,
}

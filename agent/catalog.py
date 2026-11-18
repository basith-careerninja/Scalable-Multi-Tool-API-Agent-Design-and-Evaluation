"""The capability catalogue: every operation the agent may call, indexed for lookup."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "paypal_model.json"
_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def id_label(id_name: str) -> str:
    """'billing_plan_id' -> 'billing plan', for messages and prompts."""
    return id_name.removesuffix("_id").replace("_", " ")


@dataclass(frozen=True)
class Operation:
    id: str
    name: str
    group: str
    method: str
    path: str
    resource: str
    action: str
    risk: str
    consumes: tuple[str, ...]
    produces: tuple[str, ...]
    description: str = ""
    query: tuple[dict, ...] = ()            # {"name", "example", "description"}
    body_mode: str | None = None
    body_fields: tuple[str, ...] = ()
    body_template: Any = None               # example JSON body, or None
    extract: dict = field(default_factory=dict)  # produced ID -> path in the response
    example_status: int | None = None
    example_body: Any = None
    path_aliases: dict = field(default_factory=dict)  # path parameter -> ID name, for this operation only
    service: str = "paypal"                 # which vendor's API this is (PayPal, Stripe, Square, GitHub)

    @property
    def creates(self) -> bool:
        """True when calling this makes something new, as opposed to looking it up."""
        return self.method != "GET"


_TUPLE_FIELDS = ("consumes", "produces", "query", "body_fields")


class Catalog:
    def __init__(self, operations: list[Operation], conditional: dict[str, set[str]] | None = None,
                 aliases: dict[str, str] | None = None):
        self._ops = {op.id: op for op in operations}
        self.titles: dict[str, str] = {}  # service -> display name
        if len(self._ops) != len(operations):
            raise ValueError("duplicate operation ids in catalogue")
        self._conditional = conditional or {}
        self.aliases = aliases or {}
        self._by_resource: dict[str, list[Operation]] = defaultdict(list)
        self._by_service: dict[str, set[str]] = defaultdict(set)
        self._producers: dict[str, list[Operation]] = defaultdict(list)
        for op in operations:
            self._by_resource[op.resource].append(op)
            self._by_service[op.service].add(op.resource)
            for name in op.produces:
                self._producers[name].append(op)

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL) -> "Catalog":
        return cls.load_many([path])

    @classmethod
    def load_many(cls, paths) -> "Catalog":
        """One catalogue from several vendors' models. Operation and ID names of the added
        vendors carry the service name, so nothing clashes with PayPal's."""
        known = Operation.__dataclass_fields__
        ops, conditional, aliases, titles = [], {}, {}, {}
        for path in paths:
            model = json.loads(Path(path).read_text(encoding="utf-8"))
            service = model.get("service", "paypal")
            titles[service] = model.get("title", service.title() if service != "paypal" else "PayPal")
            for raw in model["operations"]:
                if not raw["exposed"]:
                    continue
                values = {k: v for k, v in raw.items() if k in known}
                values["service"] = service
                for name in _TUPLE_FIELDS:
                    values[name] = tuple(values.get(name) or ())
                ops.append(Operation(**values))
            for name, flow in model["id_flow"].items():
                conditional.setdefault(name, set()).update(flow["conditional"])
            aliases.update(model.get("aliases", {}))
        catalog = cls(ops, conditional, aliases)
        catalog.titles = titles
        return catalog

    def __getitem__(self, op_id: str) -> Operation:
        return self._ops[op_id]

    def __len__(self) -> int:
        return len(self._ops)

    def __iter__(self):
        return iter(self._ops.values())

    def resources(self, service: str | None = None) -> list[str]:
        if service is None:
            return sorted(self._by_resource)
        return sorted(self._by_service.get(service, ()))

    def services(self) -> list[str]:
        return sorted(self._by_service)

    def title(self, service: str) -> str:
        return self.titles.get(service) or service.title()

    def actions(self, resource: str) -> list[Operation]:
        return list(self._by_resource[resource])

    def creators(self, id_name: str) -> list[Operation]:
        """Operations that make a new entity and return this ID. Reliable ones first."""
        ops = [op for op in self._producers[id_name] if op.creates]
        return sorted(ops, key=lambda op: op.id in self._conditional.get(id_name, ()))

    def resolvers(self, id_name: str) -> list[Operation]:
        """Read-only operations that can find an existing entity's ID."""
        return [op for op in self._producers[id_name] if not op.creates]

    def example_ids(self, id_name: str) -> list[str]:
        """Real examples of this ID, read from the example responses of operations that return it."""
        found = []
        for op in self._producers[id_name]:
            node = op.example_body
            try:
                for key, index in _PATH_TOKEN.findall(op.extract.get(id_name, "")):
                    node = node[int(index)] if index else node[key]
            except (KeyError, IndexError, TypeError):
                continue
            if isinstance(node, str) and node and op.extract.get(id_name):
                found.append(node)
        return found

    def looks_like_id(self, id_name: str, value: str) -> bool:
        """False when the value plainly is not this kind of ID, e.g. an invoice number given as an invoice ID.

        PayPal IDs of one kind share a prefix (INV2-..., TEMP-..., I-...). If the known
        examples have such a prefix, the value must have it too. With no examples, or
        examples without a prefix, there is nothing to check against and the value passes.
        """
        prefixes = self.id_prefixes(id_name)
        if not prefixes:
            return True
        return any(value.startswith(prefix + "-") for prefix in prefixes)

    def id_prefixes(self, id_name: str) -> set[str]:
        """Prefixes this kind of ID is known to start with, e.g. {"INV2"}. Empty if none is known."""
        return {example.split("-")[0] for example in self.example_ids(id_name) if "-" in example}

    def is_conditional(self, op_id: str, id_name: str) -> bool:
        return op_id in self._conditional.get(id_name, ())

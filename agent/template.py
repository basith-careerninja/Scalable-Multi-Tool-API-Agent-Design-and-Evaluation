"""Request-body templates as flat field lists.

An example body from the collection is flattened into paths such as
`items[0].unit_amount.value`. The model sets values for paths; code checks
each path exists in the template and builds the nested body. A field the
template does not have cannot be sent.
"""
from __future__ import annotations

import re
from typing import Any

_INDEX = re.compile(r"\[\d+\]")
_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")
_WHOLE_VAR = re.compile(r"^\{\{([\w$-]+)\}\}$")


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Nested JSON -> {path: leaf value}."""
    if isinstance(value, dict) and value:
        out = {}
        for key, child in value.items():
            out.update(flatten(child, f"{prefix}.{key}" if prefix else key))
        return out
    if isinstance(value, list) and value:
        out = {}
        for i, child in enumerate(value):
            out.update(flatten(child, f"{prefix}[{i}]"))
        return out
    return {prefix: value} if prefix else {}


def shape(path: str) -> str:
    """A path with array positions ignored, so items[3].name matches items[0].name."""
    return _INDEX.sub("[*]", path)


def variable(leaf: Any) -> str | None:
    """The variable name if the leaf is exactly one {{placeholder}}."""
    match = _WHOLE_VAR.match(leaf) if isinstance(leaf, str) else None
    return match.group(1) if match else None


def coerce(example: Any, value: Any) -> Any:
    """Fit a value to the type of the template's example. Raises ValueError if it cannot."""
    if isinstance(value, (dict, list)):
        raise ValueError("must be a single value, not an object or list")
    if isinstance(example, bool):
        if isinstance(value, bool):
            return value
        raise ValueError("must be true or false")
    if isinstance(example, str):
        return value if isinstance(value, str) else json_scalar(value)
    if isinstance(example, (int, float)) and not isinstance(value, (int, float)):
        raise ValueError("must be a number")
    return value


def json_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build(values: dict[str, Any]) -> Any:
    """{path: value} -> nested JSON. Array positions are compacted in order."""
    root: Any = None
    for path in sorted(values, key=_sort_key):
        tokens = [(name, int(index) if index else None) for name, index in _TOKEN.findall(path)]
        root = _insert(root, tokens, values[path])
    return _compact(root)


def _sort_key(path: str):
    return [(name, int(index) if index else -1) for name, index in _TOKEN.findall(path)]


def _insert(node: Any, tokens: list, value: Any) -> Any:
    if not tokens:
        return value
    (name, index), rest = tokens[0], tokens[1:]
    if index is not None:
        node = node if isinstance(node, dict) and node.get("__list__") else {"__list__": True}
        node[index] = _insert(node.get(index), rest, value)
        return node
    node = node if isinstance(node, dict) and not node.get("__list__") else {}
    node[name] = _insert(node.get(name), rest, value)
    return node


def _compact(node: Any) -> Any:
    if isinstance(node, dict) and node.get("__list__"):
        return [_compact(node[i]) for i in sorted(k for k in node if k != "__list__")]
    if isinstance(node, dict):
        return {k: _compact(v) for k, v in node.items()}
    return node

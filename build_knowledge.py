"""Turn the documentation text inside the Postman collection into a knowledge-base file.

Input : data/PayPal APIs.postman_collection.json
Output: knowledge/paypal_api_reference.md

Only descriptions are copied. Collection variables (which may hold credentials)
are never read. Any other .md or .txt file placed in knowledge/ is also picked
up by the RAG tool, so guides can be added by hand.

Usage: python build_knowledge.py
"""
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "data" / "PayPal APIs.postman_collection.json"
OUT = ROOT / "knowledge" / "paypal_api_reference.md"


def to_text(value) -> str:
    """HTML or markdown description -> plain paragraphs."""
    text = value if isinstance(value, str) else (value or {}).get("content", "")
    text = re.sub(r"</?(?:blockquote|p|br|li|ul|ol|h\d)[^>]*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)   # markdown links -> their text
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def walk(items, depth, out):
    for item in items:
        if "item" in item:
            out.append(f"{'#' * depth} {item['name']}\n")
            description = to_text(item.get("description"))
            if description:
                out.append(description + "\n")
            walk(item["item"], depth + 1, out)
        else:
            request = item["request"]
            description = to_text(request.get("description"))
            if not description:
                continue
            path = "/" + "/".join(request["url"].get("path") or [])
            out.append(f"{'#' * depth} {item['name']}\n")
            out.append(f"`{request['method']} {path}`\n")
            out.append(description + "\n")


def main():
    collection = json.loads(SRC.read_text(encoding="utf-8"))
    # The collection-level description is Postman setup text (fork links, how to
    # set credentials), not PayPal documentation, so it is left out.
    out = [f"# {collection['info']['name']}\n"]
    walk(collection["item"], 2, out)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out)} blocks, {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()

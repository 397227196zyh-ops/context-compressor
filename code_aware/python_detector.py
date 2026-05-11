"""Python code detection utilities."""

from __future__ import annotations

import re
from typing import Any

_PYTHON_KEYWORDS = {"import", "from", "def", "class", "with", "try", "if", "for", "while"}
_CODE_FENCE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)

def detect_code_language(content: str) -> str | None:
    m = _CODE_FENCE.search(content)
    if m and m.group(1):
        return m.group(1)
    words = set(re.findall(r"\b\w+\b", content or ""))
    if words & _PYTHON_KEYWORDS and ("(" in content or ":" in content):
        return "python"
    return None

def is_code_block(message: dict[str, Any]) -> bool:
    return bool(_CODE_FENCE.search(message.get("content", "")))

def extract_code_regions(content: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _CODE_FENCE.finditer(content)]

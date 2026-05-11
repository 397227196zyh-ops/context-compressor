"""Token counting utilities using tiktoken."""

from __future__ import annotations

import tiktoken

_MODEL_ENCODINGS: dict[str, str] = {
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
}

_encoders: dict[str, tiktoken.Encoding] = {}


def _get_encoding(model: str) -> tiktoken.Encoding:
    if model not in _encoders:
        encoding_name = _MODEL_ENCODINGS.get(model, "cl100k_base")
        _encoders[model] = tiktoken.get_encoding(encoding_name)
    return _encoders[model]


def count_tokens(messages: list[dict], model: str = "gpt-4o") -> int:
    if not messages:
        return 0
    encoding = _get_encoding(model)
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        total += len(encoding.encode(content))
        total += 4
    return total


def estimate_message_tokens(message: dict, model: str = "gpt-4o") -> int:
    return count_tokens([message], model)

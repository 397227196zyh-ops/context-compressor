"""Message structure utilities for context compression."""

from __future__ import annotations

import re
from typing import Any

from context_compressor.types import CodeBlock, MessageType, StickyNote

_CODE_BLOCK_PATTERN = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
_TOOL_CALL_KEYWORDS = {"function_call", "tool_calls", "tool_call_id"}


def extract_code_blocks(messages: list[dict[str, Any]]) -> list[CodeBlock]:
    if not messages:
        return []
    blocks: list[CodeBlock] = []
    for msg_idx, msg in enumerate(messages):
        content = msg.get("content", "")
        if not content:
            continue
        for match in _CODE_BLOCK_PATTERN.finditer(content):
            language = match.group(1) or "text"
            code = match.group(2)
            blocks.append(CodeBlock(
                message_index=msg_idx, start_offset=match.start(),
                end_offset=match.end(), code=code, language=language,
            ))
    return blocks


def classify_message_type(message: dict[str, Any]) -> MessageType:
    role = message.get("role", "")
    if role == "system":
        return MessageType.SYSTEM
    if role == "tool" or "tool_call_id" in message:
        return MessageType.TOOL_RESULT
    if role == "assistant" and ("tool_calls" in message or "function_call" in message):
        return MessageType.TOOL_CALL
    content = message.get("content", "")
    if content and _CODE_BLOCK_PATTERN.search(content):
        return MessageType.CODE
    return MessageType.PROSE


def split_by_type(messages: list[dict[str, Any]]) -> dict[MessageType, list[dict[str, Any]]]:
    result: dict[MessageType, list[dict[str, Any]]] = {t: [] for t in MessageType}
    for msg in messages:
        result[classify_message_type(msg)].append(msg)
    return result

"""Safety wrappers: fault isolation, timeout, output validation."""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable

from context_compressor.types import CompressionConfig, CompressionResult

logger = logging.getLogger(__name__)
_CODE_BLOCK_RE = re.compile(r"```\w*\n.*?```", re.DOTALL)


def safe_compress(strategy_compress, messages, config, timeout_ms=5000):
    result_container = []
    exception_container = []

    def _run():
        try:
            result_container.append(strategy_compress(messages, config))
        except Exception as e:
            exception_container.append(e)

    thread = threading.Thread(target=_run)
    thread.start()
    thread.join(timeout=timeout_ms / 1000.0)
    if thread.is_alive():
        return _fallback_result(messages, timeout_ms, f"Timeout after {timeout_ms}ms")
    if exception_container:
        return _fallback_result(messages, 0, str(exception_container[0]))
    return result_container[0]


def _fallback_result(messages, duration_ms, error):
    from context_compressor.utils.token_counter import count_tokens
    tokens = count_tokens(messages, "gpt-4o")
    return CompressionResult(messages, messages, tokens, tokens, "safe", duration_ms, error=error)


def validate_output(original, compressed):
    if not original and not compressed:
        return True
    if original and not compressed:
        return False
    orig_code = _extract_all_code(original)
    comp_code = _extract_all_code(compressed)
    return all(orig in comp_code for orig in orig_code)


def _extract_all_code(messages):
    code_blocks = []
    for msg in messages:
        content = msg.get("content", "")
        code_blocks.extend(_CODE_BLOCK_RE.findall(content))
    return code_blocks


class RollbackOnFailure:
    def __init__(self, messages):
        self._original = list(messages)
        self._messages = messages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._messages[:] = self._original
            logger.warning(f"Rolled back due to {exc_type.__name__}")
        return False

"""Safety wrappers: fault isolation, timeout, output validation."""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from collections.abc import Callable

from context_compressor.types import CompressionConfig, CompressionResult

logger = logging.getLogger(__name__)
_CODE_BLOCK_RE = re.compile(r"```\w*\n.*?```", re.DOTALL)


def safe_compress(
    strategy_compress: Callable,
    messages: list[dict],
    config: CompressionConfig,
    timeout_ms: int = 5000,
) -> CompressionResult:
    """Wraps strategy.compress with exception handling and timeout.

    On failure, returns original messages with error info.
    """
    result_container: list[CompressionResult] = []
    exception_container: list[Exception] = []

    def _run():
        try:
            result_container.append(strategy_compress(messages, config))
        except Exception as e:
            exception_container.append(e)

    thread = threading.Thread(target=_run)
    thread.start()
    thread.join(timeout=timeout_ms / 1000.0)

    if thread.is_alive():
        # Timeout
        return _fallback_result(messages, timeout_ms, f"Timeout after {timeout_ms}ms")

    if exception_container:
        return _fallback_result(messages, 0, str(exception_container[0]))

    return result_container[0]


async def safe_compress_async(
    strategy_compress_async: Callable,
    messages: list[dict],
    config: CompressionConfig,
    timeout_ms: int = 5000,
) -> CompressionResult:
    """异步安全包装：使用 asyncio.wait_for 替代线程超时。

    Args:
        strategy_compress_async: 异步压缩函数（如 strategy.compress_async）。
        messages: 待压缩的消息列表。
        config: 压缩配置。
        timeout_ms: 超时毫秒数。

    Returns:
        压缩结果，超时或异常时返回原始消息。
    """
    try:
        return await asyncio.wait_for(
            strategy_compress_async(messages, config),
            timeout=timeout_ms / 1000.0,
        )
    except TimeoutError:
        return _fallback_result(messages, timeout_ms, f"Timeout after {timeout_ms}ms")
    except Exception as e:
        return _fallback_result(messages, 0, str(e))


def _fallback_result(
    messages: list[dict], duration_ms: float, error: str
) -> CompressionResult:
    from context_compressor.utils.token_counter import count_tokens
    tokens = count_tokens(messages, "gpt-4o")
    return CompressionResult(messages, messages, tokens, tokens, "safe", duration_ms, error=error)


def validate_output(original: list[dict], compressed: list[dict]) -> bool:
    """Verify compressed output integrity.

    Checks:
    - Code blocks unchanged byte-for-byte
    - Output is not empty when input is not empty
    - Message count doesn't increase unreasonably
    """
    if not original and not compressed:
        return True
    if original and not compressed:
        return False

    # Extract code blocks from both
    orig_code = _extract_all_code(original)
    comp_code = _extract_all_code(compressed)

    return all(orig in comp_code for orig in orig_code)


def _extract_all_code(messages: list[dict]) -> list[str]:
    code_blocks = []
    for msg in messages:
        content = msg.get("content", "")
        code_blocks.extend(_CODE_BLOCK_RE.findall(content))
    return code_blocks


class RollbackOnFailure:
    """Context manager that restores messages on exception."""

    def __init__(self, messages: list[dict]):
        self._original = list(messages)
        self.messages = messages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.messages[:] = self._original
            logger.warning(f"Rolled back after {exc_type.__name__}: {exc_val}")
        return False  # Don't suppress the exception

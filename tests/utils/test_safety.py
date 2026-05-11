"""Tests for safety wrappers.

Covers: safe_compress catches exception, timeout protection,
validate_output detects code modification, RollbackOnFailure restores original,
safe_compress_async 异步安全包装。
"""

import asyncio
import time

import pytest

from context_compressor.types import CompressionConfig, CompressionResult
from context_compressor.utils.safety import (
    RollbackOnFailure,
    _fallback_result,
    safe_compress,
    safe_compress_async,
    validate_output,
)

# ---------------------------------------------------------------------------
# safe_compress – exception handling
# ---------------------------------------------------------------------------

def test_safe_compress_catches_exception():
    """When the strategy raises, safe_compress returns a fallback result."""

    def always_fails(messages, config):
        raise ValueError("boom")

    config = CompressionConfig()
    msgs = [{"role": "user", "content": "hello"}]

    result = safe_compress(always_fails, msgs, config, timeout_ms=5000)
    assert result.error is not None
    assert "boom" in result.error
    # Original messages returned as compressed_messages (fail-open)
    assert result.compressed_messages == msgs


def test_safe_compress_success():
    """When the strategy succeeds, result is passed through."""

    def always_works(messages, config):
        return CompressionResult(
            messages,
            messages,
            input_tokens=10,
            output_tokens=10,
            strategy="test",
            duration_ms=1.0,
        )

    config = CompressionConfig()
    msgs = [{"role": "user", "content": "hi"}]

    result = safe_compress(always_works, msgs, config, timeout_ms=5000)
    assert result.error is None
    assert result.strategy == "test"


# ---------------------------------------------------------------------------
# timeout protection
# ---------------------------------------------------------------------------

def test_safe_compress_timeout_triggers_fallback():
    """A slow strategy that exceeds timeout returns fallback result."""

    def slow_poke(messages, config):
        time.sleep(10)
        return CompressionResult(messages, messages, 0, 0, "slow", 0)

    config = CompressionConfig()
    msgs = [{"role": "user", "content": "patience"}]

    result = safe_compress(slow_poke, msgs, config, timeout_ms=50)
    assert result.error is not None
    assert "Timeout" in result.error


# ---------------------------------------------------------------------------
# validate_output
# ---------------------------------------------------------------------------

def test_validate_output_empty_list():
    assert validate_output([], [])


def test_validate_output_original_not_empty_compressed_empty():
    assert not validate_output([{"role": "user", "content": "x"}], [])


def test_validate_output_code_blocks_preserved():
    code_msg = {"role": "user", "content": "Some text ```python\nprint(1)\n``` more"}
    assert validate_output([code_msg], [code_msg])


def test_validate_output_code_block_modified():
    orig = [{"role": "user", "content": "```python\noriginal\n```"}]
    mod = [{"role": "user", "content": "```python\nmodified\n```"}]
    assert not validate_output(orig, mod)


def test_validate_output_code_block_missing():
    orig = [
        {"role": "user", "content": "```python\ncode_a\n```"},
        {"role": "user", "content": "```python\ncode_b\n```"},
    ]
    comp = [{"role": "user", "content": "```python\ncode_a\n```"}]
    assert not validate_output(orig, comp)


# ---------------------------------------------------------------------------
# RollbackOnFailure
# ---------------------------------------------------------------------------

def test_rollback_restores_original_on_exception():
    messages = [{"role": "user", "content": "original"}]
    try:
        with RollbackOnFailure(messages):
            messages.clear()
            raise RuntimeError("something broke")
    except RuntimeError:
        pass
    assert messages == [{"role": "user", "content": "original"}]


def test_rollback_no_exception_no_change():
    messages = [{"role": "user", "content": "original"}]
    with RollbackOnFailure(messages):
        messages.append({"role": "user", "content": "new"})
    # No exception → messages stay modified
    assert len(messages) == 2


# ---------------------------------------------------------------------------
# _fallback_result
# ---------------------------------------------------------------------------

def test_fallback_result_structure():
    msgs = [{"role": "user", "content": "hi"}]
    result = _fallback_result(msgs, 42.0, "test error")
    assert result.original_messages == msgs
    assert result.compressed_messages == msgs
    assert result.error == "test error"
    assert result.duration_ms == 42.0


# ---------------------------------------------------------------------------
# safe_compress_async
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_safe_compress_async_success():
    """异步安全包装成功返回结果。"""

    async def always_works(messages, config):
        return CompressionResult(messages, messages, 10, 10, "test", 1.0)

    config = CompressionConfig()
    msgs = [{"role": "user", "content": "hi"}]
    result = await safe_compress_async(always_works, msgs, config, timeout_ms=5000)
    assert result.error is None
    assert result.strategy == "test"


@pytest.mark.asyncio
async def test_safe_compress_async_exception():
    """异步策略异常时返回回退结果。"""

    async def always_fails(messages, config):
        raise ValueError("boom")

    config = CompressionConfig()
    msgs = [{"role": "user", "content": "hello"}]
    result = await safe_compress_async(always_fails, msgs, config, timeout_ms=5000)
    assert result.error is not None
    assert "boom" in result.error
    assert result.compressed_messages == msgs


@pytest.mark.asyncio
async def test_safe_compress_async_timeout():
    """超时策略返回回退结果。"""

    async def slow_poke(messages, config):
        await asyncio.sleep(10)
        return CompressionResult(messages, messages, 0, 0, "slow", 0)

    config = CompressionConfig()
    msgs = [{"role": "user", "content": "patience"}]
    result = await safe_compress_async(slow_poke, msgs, config, timeout_ms=50)
    assert result.error is not None
    assert "Timeout" in result.error

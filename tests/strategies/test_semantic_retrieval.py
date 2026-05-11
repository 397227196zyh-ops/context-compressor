"""Tests for SemanticRetrievalStrategy (Layer 3 compression)."""

from __future__ import annotations

import pytest

from context_compressor.strategies.semantic_retrieval import SemanticRetrievalStrategy
from context_compressor.types import CompressionConfig, StickyNote

# ---------------------------------------------------------------------------
# Mock adapter
# ---------------------------------------------------------------------------

class MockAdapter:
    """Mock Mem0 adapter with configurable search_facts behavior."""

    def __init__(self, facts: list[StickyNote] | None = None) -> None:
        self._facts = facts or []
        self.search_calls: list[tuple[str, str, int]] = []

    def search_facts(self, query: str, user_id: str, limit: int = 5) -> list[StickyNote]:
        self.search_calls.append((query, user_id, limit))
        return self._facts

    async def search_facts_async(
        self, query: str, user_id: str, limit: int = 5
    ) -> list[StickyNote]:
        """异步版本：委托给同步方法。"""
        return self.search_facts(query, user_id, limit)


class FailingAdapter:
    """Mock adapter that raises an exception on search_facts."""

    def search_facts(self, query: str, user_id: str, limit: int = 5) -> list[StickyNote]:
        raise RuntimeError("Mem0 connection failed")


# ---------------------------------------------------------------------------
# Tests: adapter=None → returns original
# ---------------------------------------------------------------------------

def test_adapter_none_returns_original():
    """When no adapter is set, messages are returned unchanged."""
    strategy = SemanticRetrievalStrategy(adapter=None)
    messages = [{"role": "user", "content": "hello world"}]
    result = strategy.compress(messages, CompressionConfig())

    assert result.original_messages == messages
    assert result.compressed_messages == messages
    assert result.input_tokens == result.output_tokens
    assert result.strategy == "semantic_retrieval"


def test_adapter_none_with_config():
    """Config passed to compress is accepted when adapter is None."""
    strategy = SemanticRetrievalStrategy(adapter=None)
    messages = [{"role": "user", "content": "query"}]
    config = CompressionConfig(max_retrieved_facts=10)
    result = strategy.compress(messages, config)

    assert result.original_messages == messages
    assert result.compressed_messages == messages


# ---------------------------------------------------------------------------
# Tests: adapter returns facts → memories injected as system messages
# ---------------------------------------------------------------------------

def test_facts_injected_as_system_messages():
    """Facts from adapter are prepended as system messages with [memory:] prefix."""
    facts = [
        StickyNote(key="abc123", value="User prefers Python"),
        StickyNote(key="def456", value="Project uses FastAPI"),
    ]
    adapter = MockAdapter(facts)
    strategy = SemanticRetrievalStrategy(adapter=adapter)
    messages = [{"role": "user", "content": "how to do X?"}]

    result = strategy.compress(messages, CompressionConfig())

    # Expected: 2 memory messages + 1 original = 3 messages
    assert len(result.compressed_messages) == 3
    assert result.compressed_messages[0]["role"] == "system"
    assert "[memory: abc123]" in result.compressed_messages[0]["content"]
    assert "[memory: def456]" in result.compressed_messages[1]["content"]


def test_no_user_message():
    """When there is no user message, no facts are retrieved."""
    adapter = MockAdapter([StickyNote(key="k", value="v")])
    strategy = SemanticRetrievalStrategy(adapter=adapter)
    messages = [{"role": "system", "content": "system message"}]

    result = strategy.compress(messages, CompressionConfig())
    assert result.compressed_messages == messages


def test_empty_facts():
    """When adapter returns empty facts, messages are unchanged."""
    adapter = MockAdapter([])
    strategy = SemanticRetrievalStrategy(adapter=adapter)
    messages = [{"role": "user", "content": "query"}]

    result = strategy.compress(messages, CompressionConfig())
    assert result.compressed_messages == messages


def test_search_params_passed():
    """Query and limit are correctly passed to adapter."""
    adapter = MockAdapter()
    strategy = SemanticRetrievalStrategy(adapter=adapter)
    messages = [{"role": "user", "content": "find me"}]
    config = CompressionConfig(max_retrieved_facts=7)

    strategy.compress(messages, config)

    assert len(adapter.search_calls) == 1
    assert adapter.search_calls[0] == ("find me", "default_user", 7)


def test_cache_reuse():
    """Subsequent calls with same query reuse cached facts."""
    adapter = MockAdapter([StickyNote(key="k", value="v")])
    strategy = SemanticRetrievalStrategy(adapter=adapter)
    messages = [{"role": "user", "content": "same query"}]

    strategy.compress(messages, CompressionConfig())
    strategy.compress(messages, CompressionConfig())

    assert len(adapter.search_calls) == 1  # cached second call


def test_adapter_exception_handled():
    """When adapter raises, messages are returned unchanged."""
    adapter = FailingAdapter()
    strategy = SemanticRetrievalStrategy(adapter=adapter)
    messages = [{"role": "user", "content": "query"}]

    result = strategy.compress(messages, CompressionConfig())
    assert result.compressed_messages == messages


# ---------------------------------------------------------------------------
# compress_async 测试
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compress_async_with_facts():
    """异步压缩注入记忆。"""
    facts = [StickyNote(key="k1", value="v1")]
    adapter = MockAdapter(facts)
    strategy = SemanticRetrievalStrategy(adapter=adapter)
    messages = [{"role": "user", "content": "query"}]
    result = await strategy.compress_async(messages, CompressionConfig())
    assert len(result.compressed_messages) == 2  # 1 memory + 1 original
    assert result.compressed_messages[0]["role"] == "system"


@pytest.mark.asyncio
async def test_compress_async_no_adapter():
    """无 adapter 时异步压缩返回原消息。"""
    strategy = SemanticRetrievalStrategy(adapter=None)
    messages = [{"role": "user", "content": "hello"}]
    result = await strategy.compress_async(messages, CompressionConfig())
    assert result.compressed_messages == messages

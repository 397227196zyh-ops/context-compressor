"""Tests for Mem0Adapter in degraded mode."""

import pytest

from context_compressor.persistence.mem0_adapter import Mem0Adapter
from context_compressor.types import Mem0Config


def test_unavailable_store_returns_empty():
    """store_fact returns empty string when Mem0 is unavailable."""
    adapter = Mem0Adapter()
    assert adapter.store_fact([{"role": "user", "content": "hi"}], "u1") == ""


def test_unavailable_search_returns_empty():
    """search_facts returns empty list when Mem0 is unavailable."""
    adapter = Mem0Adapter()
    assert adapter.search_facts("query", "u1") == []


def test_is_available_false_without_config():
    """is_available is False when initialised without api_key."""
    adapter = Mem0Adapter(Mem0Config())
    assert adapter.is_available is False


# ---------------------------------------------------------------------------
# search_facts_async 测试
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unavailable_search_async_returns_empty():
    """search_facts_async returns empty list when Mem0 is unavailable."""
    adapter = Mem0Adapter()
    result = await adapter.search_facts_async("query", "u1")
    assert result == []

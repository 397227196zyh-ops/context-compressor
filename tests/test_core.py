import pytest

from context_compressor.core import compress_async, compress_sync, create_compression_node
from context_compressor.types import CompressionConfig, CompressionResult


def test_node_callable():
    node = create_compression_node(CompressionConfig(max_tokens=100000))
    assert callable(node)

def test_node_empty():
    node = create_compression_node()
    assert node({"messages": []}) == {"messages": []}

def test_node_small():
    node = create_compression_node(CompressionConfig(max_tokens=100000))
    r = node({"messages": [{"role": "user", "content": "hi"}]})
    assert len(r["messages"]) == 1

def test_node_large():
    config = CompressionConfig(max_tokens=20, warning_ratio=0.5, sliding_window_size=3)
    node = create_compression_node(config)
    msgs = [{"role": "user", "content": "x" * 20} for _ in range(20)]
    r = node({"messages": msgs})
    assert len(r["messages"]) < len(msgs)

def test_sync_result():
    r = compress_sync([{"role": "user", "content": "hello"}], CompressionConfig(max_tokens=100000))
    assert isinstance(r, CompressionResult)

def test_sync_empty():
    r = compress_sync([])
    assert r.compressed_messages == []

def test_sync_large():
    config = CompressionConfig(max_tokens=20, warning_ratio=0.5, sliding_window_size=3)
    msgs = [{"role": "user", "content": "x" * 20} for _ in range(30)]
    r = compress_sync(msgs, config)
    assert r.output_tokens < r.input_tokens


# ---------------------------------------------------------------------------
# 异步压缩测试
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_async_import_and_basic():
    """compress_async can be imported and called."""
    r = await compress_async(
        [{"role": "user", "content": "hello"}],
        CompressionConfig(max_tokens=100000),
    )
    assert isinstance(r, CompressionResult)
    assert r.compressed_messages == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_async_empty():
    """compress_async returns empty for empty input."""
    r = await compress_async([])
    assert r.compressed_messages == []


@pytest.mark.asyncio
async def test_async_large():
    """compress_async reduces token count for large input."""
    config = CompressionConfig(max_tokens=20, warning_ratio=0.5, sliding_window_size=3)
    msgs = [{"role": "user", "content": "x" * 20} for _ in range(30)]
    r = await compress_async(msgs, config)
    assert r.output_tokens < r.input_tokens

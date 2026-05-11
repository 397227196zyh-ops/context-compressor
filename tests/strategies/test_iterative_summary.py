"""Tests for IterativeSummaryStrategy (Layer 2)."""

import os
import tempfile
from unittest.mock import MagicMock

import pytest

from context_compressor.strategies.iterative_summary import IterativeSummaryStrategy
from context_compressor.types import CompressionConfig


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    response = MagicMock()
    response.content = (
        "intent: Build web app\n"
        "changes_made: Added routes\n"
        "decisions: Use FastAPI\n"
        "next_steps: Write tests"
    )
    llm.invoke.return_value = response

    async def mock_ainvoke(prompt):
        return response

    llm.ainvoke = mock_ainvoke
    return llm


@pytest.fixture
def config():
    return CompressionConfig(summary_trigger_turns=3)


@pytest.fixture
def messages():
    return [{"role": "user", "content": f"msg {i}"} for i in range(10)]


class TestIterativeSummary:
    def test_no_llm_returns_original(self, messages, config):
        strategy = IterativeSummaryStrategy(config, llm=None)
        result = strategy.compress(messages, config)
        assert result.compressed_messages == messages
        assert result.input_tokens == result.output_tokens
        assert result.strategy == "iterative_summary"

    def test_under_trigger_turns_returns_original(self, mock_llm):
        config = CompressionConfig(summary_trigger_turns=50)
        strategy = IterativeSummaryStrategy(config, llm=mock_llm)
        msgs = [{"role": "user", "content": "hi"}]
        result = strategy.compress(msgs, config)
        assert result.compressed_messages == msgs
        mock_llm.invoke.assert_not_called()

    def test_llm_called_when_trigger_reached(self, mock_llm, config):
        strategy = IterativeSummaryStrategy(config, llm=mock_llm)
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(5)]
        strategy.compress(msgs, config)
        mock_llm.invoke.assert_called_once()

    def test_anchor_parsed(self, mock_llm, config):
        strategy = IterativeSummaryStrategy(config, llm=mock_llm)
        msgs = [{"role": "user", "content": "test"} for _ in range(5)]
        strategy.compress(msgs, config)
        assert strategy.anchor["intent"] == "Build web app"
        assert strategy.anchor["decisions"] == "Use FastAPI"

    def test_compile_anchor_message(self, mock_llm, config):
        strategy = IterativeSummaryStrategy(config, llm=mock_llm)
        strategy._anchor = {
            "intent": "X", "changes_made": "Y", "decisions": "Z", "next_steps": "W"
        }
        msg = strategy._compile_anchor_to_message()
        assert msg["role"] == "system"
        assert "X" in msg["content"]
        assert "Y" in msg["content"]

    def test_llm_exception_handled(self, mock_llm, config):
        mock_llm.invoke.side_effect = RuntimeError("LLM down")
        strategy = IterativeSummaryStrategy(config, llm=mock_llm)
        msgs = [{"role": "user", "content": "test"} for _ in range(5)]
        result = strategy.compress(msgs, config)
        assert result.compressed_messages is not None

    def test_save_and_load_anchor(self, mock_llm, config):
        strategy = IterativeSummaryStrategy(config, llm=mock_llm)
        strategy._anchor = {
            "intent": "A", "changes_made": "B", "decisions": "C", "next_steps": "D"
        }
        strategy._last_summary_turn = 42
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            strategy.save_anchor(f.name)
            path = f.name
        try:
            loaded = IterativeSummaryStrategy(config, llm=mock_llm)
            loaded.load_anchor(path)
            assert loaded.anchor["intent"] == "A"
            assert loaded._last_summary_turn == 42
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# compress_async 测试
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compress_async_no_llm(messages, config):
    """无 LLM 时异步压缩返回原始消息。"""
    strategy = IterativeSummaryStrategy(config, llm=None)
    result = await strategy.compress_async(messages, config)
    assert result.compressed_messages == messages
    assert result.strategy == "iterative_summary"


@pytest.mark.asyncio
async def test_compress_async_under_trigger(mock_llm):
    """未达到触发轮次时不调用 LLM。"""
    config = CompressionConfig(summary_trigger_turns=50)
    strategy = IterativeSummaryStrategy(config, llm=mock_llm)
    msgs = [{"role": "user", "content": "hi"}]
    result = await strategy.compress_async(msgs, config)
    assert result.compressed_messages == msgs
    mock_llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_compress_async_llm_called(mock_llm, config):
    """达到触发轮次时调用 ainvoke。"""
    strategy = IterativeSummaryStrategy(config, llm=mock_llm)
    msgs = [{"role": "user", "content": f"msg {i}"} for i in range(5)]
    await strategy.compress_async(msgs, config)
    mock_llm.ainvoke.assert_called_once()

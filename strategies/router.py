"""Adaptive strategy router: analyzes messages and selects compression strategies."""

from __future__ import annotations

import logging

from context_compressor.types import CompressionConfig, MessageAnalysis, StrategyProtocol
from context_compressor.utils.messages import (
    MessageType,
    classify_message_type,
    extract_code_blocks,
)
from context_compressor.utils.token_counter import count_tokens

logger = logging.getLogger(__name__)

class AdaptiveRouter:
    def __init__(self, config: CompressionConfig | None = None) -> None:
        self._config = config or CompressionConfig()
        self._strategies: dict[str, StrategyProtocol] = {}

    def register_strategy(self, name: str, strategy: StrategyProtocol) -> None:
        self._strategies[name] = strategy

    def route(self, messages: list[dict], config: CompressionConfig) -> list[StrategyProtocol]:
        analysis = self._analyze_messages(messages)
        if not self._is_compression_needed(analysis, config):
            return []
        selected: list[StrategyProtocol] = []
        if (
            config.enable_sliding_window
            and analysis.token_count > config.trigger_ratio * config.max_tokens
        ):
            s = self._strategies.get("sliding_window")
            if s:
                selected.append(s)
        if config.enable_iterative_summary and analysis.turn_count > config.summary_trigger_turns:
            s = self._strategies.get("iterative_summary")
            if s:
                selected.append(s)
        if analysis.code_ratio > 0.3 and config.enable_code_aware:
            s = self._strategies.get("code_aware")
            if s:
                selected.append(s)
        elif config.enable_token_compression:
            s = self._strategies.get("token_compression")
            if s:
                selected.append(s)
        if config.enable_semantic_retrieval:
            s = self._strategies.get("semantic_retrieval")
            if s:
                selected.append(s)
        return selected

    def _analyze_messages(self, messages: list[dict]) -> MessageAnalysis:
        token_count = count_tokens(messages, "gpt-4o")
        code_blocks = extract_code_blocks(messages)
        code_chars = sum(len(b.code) for b in code_blocks)
        total_chars = sum(len(m.get("content", "")) for m in messages)
        code_ratio = code_chars / total_chars if total_chars > 0 else 0.0
        return MessageAnalysis(
            token_count=token_count,
            code_ratio=code_ratio,
            turn_count=len(messages),
            has_tool_calls=any(
                classify_message_type(m) for m in messages
                if m.get("role") == "assistant"
            ),
        )

    def _is_compression_needed(
        self, analysis: MessageAnalysis, config: CompressionConfig
    ) -> bool:
        if analysis.token_count > config.trigger_ratio * config.max_tokens:
            return True
        if analysis.turn_count > config.summary_trigger_turns:
            return True
        if analysis.code_ratio > 0.5:
            return True
        return False

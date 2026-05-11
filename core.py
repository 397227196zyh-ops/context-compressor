"""Core compression node and sync API."""
from __future__ import annotations

import logging
import time

from context_compressor.types import CompressionResult
from context_compressor.utils.config import load_config
from context_compressor.utils.token_counter import count_tokens

logger = logging.getLogger(__name__)

def _init_strategies(cfg):
    from context_compressor.code_aware.python_compressor import CodeAwareCompressor
    from context_compressor.persistence.mem0_adapter import Mem0Adapter
    from context_compressor.strategies.iterative_summary import IterativeSummaryStrategy
    from context_compressor.strategies.router import AdaptiveRouter
    from context_compressor.strategies.semantic_retrieval import SemanticRetrievalStrategy
    from context_compressor.strategies.sliding_window import SlidingWindowStrategy
    from context_compressor.strategies.token_compression import TokenCompressionStrategy
    sliding = SlidingWindowStrategy(cfg)
    iterative = IterativeSummaryStrategy(cfg)
    retrieval = SemanticRetrievalStrategy(
        cfg,
        adapter=Mem0Adapter(cfg.mem0) if cfg.mem0.api_key else None,
    )
    token = TokenCompressionStrategy(cfg)
    code_aware = CodeAwareCompressor(cfg.max_code_block_tokens, cfg.code_margin_lines)
    router = AdaptiveRouter(cfg)
    router.register_strategy("sliding_window", sliding)
    router.register_strategy("iterative_summary", iterative)
    router.register_strategy("semantic_retrieval", retrieval)
    router.register_strategy("token_compression", token)
    router.register_strategy("code_aware", code_aware)
    return router

def create_compression_node(config=None):
    cfg = config or load_config()
    router = _init_strategies(cfg)
    def compression_node(state):
        messages = list(state.get("messages", []))
        if not messages:
            return {"messages": []}
        strategies = router.route(messages, cfg)
        if not strategies:
            return {"messages": messages}
        compressed = list(messages)
        for s in strategies:
            try:
                r = s.compress(compressed, cfg)
                compressed = r.compressed_messages
            except Exception as e:
                logger.error(f"Strategy {type(s).__name__} failed: {e}")
        return {"messages": compressed}
    return compression_node

def compress_sync(messages, config=None):
    start = time.monotonic()
    cfg = config or load_config()
    input_tokens = count_tokens(messages, "gpt-4o")
    if not messages:
        return CompressionResult([], [], 0, 0, "none", 0.0)
    router = _init_strategies(cfg)
    strategies = router.route(messages, cfg)
    if not strategies:
        return CompressionResult(messages, messages, input_tokens, input_tokens, "none", 0.0)
    compressed = list(messages)
    strategy_names = []
    for s in strategies:
        try:
            result = s.compress(compressed, cfg)
            compressed = result.compressed_messages
            strategy_names.append(type(s).__name__)
        except Exception as e:
            logger.error(f"Strategy {type(s).__name__} failed: {e}")
            if not cfg.fail_open:
                raise
    output_tokens = count_tokens(compressed, "gpt-4o")
    duration = (time.monotonic() - start) * 1000
    return CompressionResult(
        original_messages=list(messages),
        compressed_messages=compressed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        strategy=",".join(strategy_names) if strategy_names else "none",
        duration_ms=duration,
    )

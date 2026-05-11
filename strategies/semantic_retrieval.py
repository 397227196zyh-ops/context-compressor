"""Layer 3: Mem0 semantic retrieval injection."""

from __future__ import annotations

import time
from typing import Any

from context_compressor.types import CompressionConfig, CompressionResult
from context_compressor.utils.token_counter import count_tokens


class SemanticRetrievalStrategy:
    def __init__(self, config: CompressionConfig | None = None, adapter: Any = None) -> None:
        self._config = config or CompressionConfig()
        self._adapter = adapter
        self._cache: dict[str, tuple[list, float]] = {}
        self._cache_ttl = 300.0

    def compress(self, messages: list[dict], config: CompressionConfig) -> CompressionResult:
        start = time.monotonic()
        input_tokens = count_tokens(messages, "gpt-4o")
        if not self._adapter or not messages:
            dur = (time.monotonic() - start) * 1000
            return CompressionResult(
                messages, messages, input_tokens,
                input_tokens, "semantic_retrieval", dur,
            )
        query = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                query = m.get("content", "")
                break
        if not query:
            dur = (time.monotonic() - start) * 1000
            return CompressionResult(
                messages, messages, input_tokens,
                input_tokens, "semantic_retrieval", dur,
            )
        now = time.monotonic()
        if query in self._cache and (now - self._cache[query][1] < self._cache_ttl):
            facts = self._cache[query][0]
        else:
            try:
                facts = self._adapter.search_facts(
                    query, "default_user", limit=config.max_retrieved_facts
                )
            except Exception:
                facts = []
            self._cache[query] = (facts, now)
        if not facts:
            dur = (time.monotonic() - start) * 1000
            return CompressionResult(
                messages, messages, input_tokens,
                input_tokens, "semantic_retrieval", dur,
            )
        memories = [{"role": "system", "content": f"[memory: {f.key}] {f.value}"} for f in facts]
        compressed = memories + list(messages)
        output_tokens = count_tokens(compressed, "gpt-4o")
        return CompressionResult(
            messages, compressed, input_tokens,
            output_tokens, "semantic_retrieval",
            (time.monotonic() - start) * 1000,
        )

"""Layer 1: Sliding window compression with sticky notes."""

from __future__ import annotations

import time

from context_compressor.types import CompressionConfig, CompressionResult, StickyNote
from context_compressor.utils.token_counter import count_tokens


class SlidingWindowStrategy:
    """Keeps most recent N messages and prepends sticky notes."""

    def __init__(self, config: CompressionConfig | None = None) -> None:
        self._config = config or CompressionConfig()
        self._sticky_notes: list[StickyNote] = []

    def add_sticky_note(self, note: StickyNote) -> None:
        self._sticky_notes.append(note)
        self._sticky_notes.sort(key=lambda n: (-n.importance_score, -n.source_turn))

    def get_sticky_notes(self) -> list[StickyNote]:
        return list(self._sticky_notes)

    def compress(
        self, messages: list[dict], config: CompressionConfig
    ) -> CompressionResult:
        start = time.monotonic()
        input_tokens = count_tokens(messages, "gpt-4o")

        if not messages:
            return CompressionResult([], [], 0, 0, "sliding_window", 0.0)

        threshold = config.warning_ratio * config.max_tokens
        if input_tokens < threshold:
            dur = (time.monotonic() - start) * 1000
            return CompressionResult(
                messages, messages, input_tokens, input_tokens, "sliding_window", dur
            )

        window = (
            messages[-config.sliding_window_size :]
            if config.sliding_window_size < len(messages)
            else list(messages)
        )
        sticky_msgs = [
            {"role": "system", "content": f"[sticky] {n.key}: {n.value}"}
            for n in self._sticky_notes
        ]
        compressed = sticky_msgs + window
        output_tokens = count_tokens(compressed, "gpt-4o")
        duration = (time.monotonic() - start) * 1000
        return CompressionResult(
            messages, compressed, input_tokens, output_tokens, "sliding_window", duration
        )

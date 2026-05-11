"""Layer 2: Anchored iterative summarization."""

from __future__ import annotations

import json
import time
from typing import Any

from context_compressor.types import CompressionConfig, CompressionResult
from context_compressor.utils.token_counter import count_tokens

_ANCHOR_FIELDS = ("intent", "changes_made", "decisions", "next_steps")


class IterativeSummaryStrategy:
    """Maintains an anchor state and incrementally summarizes new messages."""

    def __init__(
        self, config: CompressionConfig | None = None, llm: Any = None
    ) -> None:
        self._config = config or CompressionConfig()
        self._llm = llm
        self._anchor: dict[str, str] = dict.fromkeys(_ANCHOR_FIELDS, "")
        self._last_summary_turn = 0

    @property
    def anchor(self) -> dict[str, str]:
        return dict(self._anchor)

    def compress(
        self, messages: list[dict], config: CompressionConfig
    ) -> CompressionResult:
        start = time.monotonic()
        input_tokens = count_tokens(messages, "gpt-4o")

        if not messages or not self._llm:
            dur = (time.monotonic() - start) * 1000
            return CompressionResult(
                messages, messages, input_tokens, input_tokens, "iterative_summary", dur
            )

        new_messages = messages[self._last_summary_turn :]
        if len(new_messages) < config.summary_trigger_turns:
            dur = (time.monotonic() - start) * 1000
            return CompressionResult(
                messages, messages, input_tokens, input_tokens, "iterative_summary", dur
            )

        try:
            prompt = (
                "You are maintaining a conversation summary anchor. Update these fields:\n"
                "intent: What the user is trying to accomplish\n"
                "changes_made: What changes have been made\n"
                "decisions: Key decisions made\n"
                "next_steps: What to do next\n\n"
                f"Current anchor:\n{json.dumps(self._anchor, indent=2)}\n\n"
                f"New messages:\n{[m.get('content','') for m in new_messages]}\n\n"
                "Return ONLY the updated fields in this format:\n"
                "intent: ...\nchanges_made: ...\ndecisions: ...\nnext_steps: ..."
            )
            response = self._llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            self._parse_anchor_update(text)
        except Exception:
            pass

        self._last_summary_turn = len(messages)
        compiled = self._compile_anchor_to_message()
        out = [compiled] if compiled else [{"role": "system", "content": str(self._anchor)}]
        compressed = out + messages[-config.sliding_window_size:]
        output_tokens = count_tokens(compressed, "gpt-4o")
        return CompressionResult(
            messages, compressed, input_tokens, output_tokens,
            "iterative_summary", (time.monotonic() - start) * 1000,
        )

    def _parse_anchor_update(self, text: str) -> None:
        for line in text.strip().split("\n"):
            for field in _ANCHOR_FIELDS:
                if line.startswith(f"{field}:"):
                    self._anchor[field] = line[len(field) + 1:].strip()

    def _compile_anchor_to_message(self) -> dict | None:
        if all(not v for v in self._anchor.values()):
            return None
        parts = [f"{k}: {v}" for k, v in self._anchor.items() if v]
        return {"role": "system", "content": "[summary] " + " | ".join(parts)}

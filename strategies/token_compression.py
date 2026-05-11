"""Layer 4a: Token-level compression for non-code text via LLMLingua-2."""

from __future__ import annotations

import re
import time

from context_compressor.types import CompressionConfig, CompressionResult
from context_compressor.utils.token_counter import count_tokens

_CODE_BLOCK_RE = re.compile(r"(```\w*\n.*?```)", re.DOTALL)

class TokenCompressionStrategy:
    def __init__(self, config: CompressionConfig | None = None) -> None:
        self._config = config or CompressionConfig()
        self._compressor: object | None = None
        self._load_attempted = False

    def _ensure_compressor(self) -> object | None:
        if self._load_attempted:
            return self._compressor
        self._load_attempted = True
        try:
            from llmlingua import PromptCompressor
            self._compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                use_llmlingua2=True)
        except Exception:
            self._compressor = None
        return self._compressor

    def _compress_text(self, text: str, ratio: float) -> str:
        if len(text) < 100:
            return text
        if not self._compressor:
            return text
        try:
            result = self._compressor.compress_prompt(context=[text], ratio=ratio)
            return result.get("compressed_prompt", text)
        except Exception:
            return text

    def compress(self, messages: list[dict], config: CompressionConfig) -> CompressionResult:
        start = time.monotonic()
        input_tokens = count_tokens(messages, "gpt-4o")
        if not messages:
            return CompressionResult([], [], 0, 0, "token_compression", 0.0)
        self._ensure_compressor()
        if not self._compressor:
            dur = (time.monotonic() - start) * 1000
            return CompressionResult(
                messages, messages,
                input_tokens, input_tokens,
                "token_compression", dur,
            )
        compressed_msgs = []
        for msg in messages:
            content = msg.get("content", "")
            if not content:
                compressed_msgs.append(dict(msg))
                continue
            code_blocks = _CODE_BLOCK_RE.findall(content)
            if not code_blocks:
                compressed_msgs.append({
                    **msg,
                    "content": self._compress_text(
                        content, config.token_compression_ratio,
                    ),
                })
            else:
                work = content
                placeholders = {}
                for i, cb in enumerate(code_blocks):
                    ph = f"__CODE_BLOCK_{i}__"
                    placeholders[ph] = cb
                    work = work.replace(cb, ph, 1)
                compressed_text = self._compress_text(work, config.token_compression_ratio)
                for ph, cb in placeholders.items():
                    compressed_text = compressed_text.replace(ph, cb)
                compressed_msgs.append({**msg, "content": compressed_text})
        output_tokens = count_tokens(compressed_msgs, "gpt-4o")
        return CompressionResult(
            messages, compressed_msgs, input_tokens, output_tokens,
            "token_compression", (time.monotonic() - start) * 1000,
        )

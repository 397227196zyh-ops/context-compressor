"""Layer 4b: Code-aware compression - deduplication and truncation."""

from __future__ import annotations

import ast
import hashlib
import re
import time
from typing import Any

from context_compressor.types import CompressionConfig, CompressionResult
from context_compressor.utils.token_counter import count_tokens

_CODE_FENCE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
COMPRESSIBLE_LANGUAGES = ["python"]

class CodeAwareCompressor:
    def __init__(self, max_code_block_tokens: int = 500, code_margin_lines: int = 5) -> None:
        self.max_code_block_tokens = max_code_block_tokens
        self.code_margin_lines = code_margin_lines
        self._seen_blocks: dict[str, int] = {}

    def compress_code_blocks(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content", "")
            new_content = self._process_content(content)
            result.append({**msg, "content": new_content})
        return result

    def _process_content(self, content: str) -> str:
        def replace_block(match):
            lang = match.group(1) or ""
            code = match.group(2)
            if lang.lower() not in COMPRESSIBLE_LANGUAGES and lang != "":
                return match.group(0)
            block_hash = hashlib.md5(code.strip().encode()).hexdigest()
            if block_hash in self._seen_blocks:
                return f"```{lang}\n[same as Block #{self._seen_blocks[block_hash]}]\n```"
            block_num = len(self._seen_blocks) + 1
            self._seen_blocks[block_hash] = block_num
            lines = code.split("\n")
            if len(lines) <= self.code_margin_lines * 2 + 3:
                return match.group(0)
            important = self._extract_important_lines(code)
            head_lines = lines[:self.code_margin_lines]
            tail_lines = lines[-self.code_margin_lines:]
            result_lines = list(head_lines)
            for line in important:
                if line not in head_lines and line not in tail_lines:
                    result_lines.append(line)
            omitted = len(lines) - len(set(head_lines + important + tail_lines))
            if omitted <= 3:
                return match.group(0)
            result_lines.append(f"# ... {omitted} lines omitted ...")
            result_lines.extend(tail_lines)
            return f"```{lang}\n" + "\n".join(result_lines) + "\n```"
        return _CODE_FENCE.sub(replace_block, content)

    def _extract_important_lines(self, code: str) -> list[str]:
        important: list[str] = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    important.append(ast.unparse(node).split("\n")[0])
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    important.append(ast.unparse(node))
        except SyntaxError:
            pass
        return important

    def compress(self, messages: list[dict], config: CompressionConfig) -> CompressionResult:
        start = time.monotonic()
        input_tokens = count_tokens(messages, "gpt-4o")
        compressed = self.compress_code_blocks(messages)
        output_tokens = count_tokens(compressed, "gpt-4o")
        return CompressionResult(
            messages, compressed, input_tokens, output_tokens,
            "code_aware", (time.monotonic() - start) * 1000,
        )

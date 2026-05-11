"""Structured compression logging for context-compressor."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

_SENSITIVE_PATTERNS = {"api_key", "secret", "password", "token", "credential", "authorization"}


class _SecretFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.msg)
        for pattern in _SENSITIVE_PATTERNS:
            msg = re.sub(rf"({pattern}\s*[:=]\s*['\"]?)\S+", r"\1[REDACTED]", msg, flags=re.IGNORECASE)
        record.msg = msg
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("strategy", "input_tokens", "output_tokens", "duration_ms", "trigger", "reason"):
            if hasattr(record, key):
                log_data[key] = getattr(record, key)
        return json.dumps(log_data)


class CompressionLogger:
    def __init__(self, name: str = "context_compressor", level: int = logging.DEBUG) -> None:
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        if not any(isinstance(f, _SecretFilter) for f in self.logger.filters):
            self.logger.addFilter(_SecretFilter())
        log_format = os.getenv("CC_LOG_FORMAT", "text").lower()
        if log_format == "json" and not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(JsonFormatter())
            self.logger.addHandler(handler)

    def log_compression(self, result, trigger_reason, extra_context=None):
        extra = {"strategy": result.strategy, "input_tokens": result.input_tokens,
                 "output_tokens": result.output_tokens, "duration_ms": result.duration_ms,
                 "trigger": trigger_reason}
        if extra_context:
            extra.update(extra_context)
        self.logger.info("Compression completed", extra=extra)

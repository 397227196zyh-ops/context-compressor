"""Configuration loading for context-compressor."""

from __future__ import annotations

import json
import os
from pathlib import Path

from context_compressor.types import CompressionConfig

_ENV_PREFIX = "CC_"

_ENV_FIELD_MAP: dict[str, str] = {
    "MAX_TOKENS": "max_tokens",
    "WARNING_RATIO": "warning_ratio",
    "TRIGGER_RATIO": "trigger_ratio",
    "SLIDING_WINDOW_SIZE": "sliding_window_size",
    "SUMMARY_LLM_MODEL": "summary_llm_model",
    "SUMMARY_MAX_TOKENS": "summary_max_tokens",
    "SUMMARY_TRIGGER_TURNS": "summary_trigger_turns",
    "MAX_RETRIEVED_FACTS": "max_retrieved_facts",
    "TOKEN_COMPRESSION_RATIO": "token_compression_ratio",
    "STRATEGY_TIMEOUT_MS": "strategy_timeout_ms",
}


def load_config(path: str | None = None) -> CompressionConfig:
    kwargs: dict = {}
    if path and Path(path).exists():
        with open(path) as f:
            file_config = json.load(f)
        kwargs.update(file_config)
    for env_suffix, field_name in _ENV_FIELD_MAP.items():
        env_key = f"{_ENV_PREFIX}{env_suffix}"
        env_value = os.getenv(env_key)
        if env_value is not None:
            try:
                if field_name in ("max_tokens", "sliding_window_size", "summary_max_tokens",
                                   "summary_trigger_turns", "max_retrieved_facts",
                                   "strategy_timeout_ms"):
                    kwargs[field_name] = int(env_value)
                elif field_name in ("warning_ratio", "trigger_ratio",
                                     "token_compression_ratio"):
                    kwargs[field_name] = float(env_value)
                else:
                    kwargs[field_name] = env_value
            except (ValueError, TypeError):
                pass
    return CompressionConfig(**kwargs)


def save_default_config(path: str) -> None:
    config = CompressionConfig()
    config_dict = {
        "max_tokens": config.max_tokens,
        "warning_ratio": config.warning_ratio,
    }
    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2)

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, TypedDict, runtime_checkable


@dataclass
class Mem0Config:
    """Configuration for Mem0 integration."""
    api_key: str = ""
    org_id: str = ""
    project_id: str = ""
    host: str = "https://api.mem0.ai"
    user_id: str = ""


@dataclass
class CompressionThresholds:
    """Token-based compression trigger thresholds."""
    max_tokens: int = 128000
    warning_ratio: float = 0.7
    trigger_ratio: float = 0.85


class Message(TypedDict, total=False):
    """A single message in the conversation."""
    role: str
    content: str
    tool_call_id: str | None
    name: str | None


@dataclass
class StickyNote:
    """A piece of information that should survive across compression passes."""
    key: str
    value: str
    importance_score: float = 1.0
    source_turn: int = 0


@dataclass
class CompressionConfig:
    """Master configuration for all compression strategies."""
    max_tokens: int = 128000
    warning_ratio: float = 0.7
    trigger_ratio: float = 0.85
    enable_sliding_window: bool = True
    sliding_window_size: int = 10
    enable_iterative_summary: bool = True
    summary_llm_model: str = "gpt-4o-mini"
    summary_max_tokens: int = 500
    summary_trigger_turns: int = 20
    enable_semantic_retrieval: bool = True
    max_retrieved_facts: int = 5
    store_trigger_turns: int = 10
    enable_token_compression: bool = True
    enable_code_aware: bool = True
    token_compression_ratio: float = 0.5
    max_code_block_tokens: int = 500
    code_margin_lines: int = 5
    strategy_timeout_ms: int = 5000
    max_compression_retries: int = 1
    fail_open: bool = True
    mem0: Mem0Config = field(default_factory=Mem0Config)


@dataclass
class CompressionResult:
    """Result of a compression operation."""
    original_messages: list[Message]
    compressed_messages: list[Message]
    input_tokens: int
    output_tokens: int
    strategy: str
    duration_ms: float
    error: str | None = None


@dataclass
class CodeBlock:
    """A code block extracted from a message."""
    message_index: int
    start_offset: int
    end_offset: int
    code: str
    language: str


class MessageType(Enum):
    """Classification of message types."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    TOOL_CALL = "tool_call"


@dataclass
class MessageAnalysis:
    """Analysis result for a batch of messages."""
    total_tokens: int
    turn_count: int
    code_ratio: float
    has_tool_calls: bool
    dominant_language: str = ""


@runtime_checkable
class StrategyProtocol(Protocol):
    """Protocol that all compression strategies must implement."""
    def compress(self, messages: list[Message], config: CompressionConfig) -> CompressionResult:
        ...

    @property
    def name(self) -> str:
        ...

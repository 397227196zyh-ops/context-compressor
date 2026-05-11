"""Session state types for persistence."""

from dataclasses import dataclass


@dataclass
class SessionState:
    session_id: str
    messages: list
    anchor_state: dict
    metadata: dict
    created_at: str = ""
    updated_at: str = ""


@dataclass
class SessionSummary:
    session_id: str
    message_count: int
    created_at: str
    updated_at: str
    user_id: str = ""
    project_id: str = ""

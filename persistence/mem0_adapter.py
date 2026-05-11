"""Mem0 integration adapter for cross-session memory."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from context_compressor.types import Mem0Config, StickyNote

logger = logging.getLogger(__name__)


class Mem0Adapter:
    """Adapter for Mem0 Memory API with graceful degradation."""

    def __init__(self, config: Mem0Config | None = None) -> None:
        self._config = config or Mem0Config()
        self._memory: Any | None = None
        if config and config.api_key:
            try:
                from mem0 import Memory
                mem0_config: dict[str, Any] = {}
                if config.api_key:
                    mem0_config["api_key"] = config.api_key
                if config.org_id:
                    mem0_config["org_id"] = config.org_id
                if config.project_id:
                    mem0_config["project_id"] = config.project_id
                self._memory = (
                    Memory.from_config({"llm": {"provider": "openai"}})
                    if not mem0_config
                    else Memory()
                )
            except (ImportError, Exception) as e:
                logger.warning(f"Mem0 initialization failed: {e}. Operating in degraded mode.")
                self._memory = None

    @property
    def is_available(self) -> bool:
        return self._memory is not None

    def store_fact(self, messages, user_id, agent_id=None, session_id=None):
        if not self.is_available:
            logger.debug("Mem0 unavailable, skipping store_fact")
            return ""
        try:
            kwargs = {"user_id": user_id}
            if agent_id:
                kwargs["agent_id"] = agent_id
            if session_id:
                kwargs["run_id"] = session_id
            result = self._memory.add(messages, **kwargs)
            return str(result)
        except Exception as e:
            logger.error(f"Mem0 store_fact failed: {e}")
            return ""

    def search_facts(self, query, user_id, limit=5):
        if not self.is_available:
            return []
        try:
            results = self._memory.search(query, user_id=user_id, limit=limit)
            return [StickyNote(key=r.get("memory", "")[:50], value=r.get("memory", ""),
                importance_score=0.7) for r in (results or [])]
        except Exception as e:
            logger.error(f"Mem0 search_facts failed: {e}")
            return []

    def delete_fact(self, memory_id):
        if not self.is_available:
            return False
        try:
            self._memory.delete(memory_id)
            return True
        except Exception as e:
            logger.error(f"Mem0 delete_fact failed: {e}")
            return False

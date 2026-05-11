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
        """Initialize the Mem0 adapter.

        Args:
            config: Mem0 configuration. If None, adapter operates in degraded mode.
        """
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

                # Use the simplest initialization that works
                self._memory = (
                    Memory.from_config({"llm": {"provider": "openai"}})
                    if not mem0_config
                    else Memory()
                )
            except (ImportError, Exception) as e:
                logger.warning(
                    f"Mem0 initialization failed: {e}. Operating in degraded mode."
                )
                self._memory = None

    @property
    def is_available(self) -> bool:
        """Check if Mem0 is available and initialized."""
        return self._memory is not None

    def store_fact(
        self,
        messages: list[dict[str, Any]],
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Extract and store facts from messages.

        Args:
            messages: List of messages to extract facts from.
            user_id: User identifier for scoping.
            agent_id: Optional agent identifier.
            session_id: Optional session identifier.

        Returns:
            Memory ID string, or empty string if unavailable.
        """
        if not self.is_available:
            logger.debug("Mem0 unavailable, skipping store_fact")
            return ""

        try:
            kwargs: dict[str, Any] = {"user_id": user_id}
            if agent_id:
                kwargs["agent_id"] = agent_id
            if session_id:
                kwargs["run_id"] = session_id

            result = self._memory.add(messages, **kwargs)

            # Extract memory ID from result
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("id", "")
            elif isinstance(result, dict):
                return result.get("id", "")
            return ""
        except Exception as e:
            logger.error(f"Mem0 store_fact failed: {e}")
            return ""

    def search_facts(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
    ) -> list[StickyNote]:
        """Search for relevant facts in Mem0.

        Args:
            query: Search query string.
            user_id: User identifier for scoping.
            limit: Maximum number of results.

        Returns:
            List of StickyNote objects with matching facts.
        """
        if not self.is_available:
            logger.debug("Mem0 unavailable, returning empty search results")
            return []

        try:
            results = self._memory.search(query, user_id=user_id, limit=limit)

            notes: list[StickyNote] = []
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict):
                        notes.append(
                            StickyNote(
                                key=item.get("id", ""),
                                value=item.get("memory", ""),
                                importance_score=item.get("score", 0.5),
                            )
                        )
            return notes
        except Exception as e:
            logger.error(f"Mem0 search_facts failed: {e}")
            return []

    async def search_facts_async(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        timeout_seconds: float = 10.0,
    ) -> list[StickyNote]:
        """Asynchronously search for relevant facts in Mem0.

        Args:
            query: Search query string.
            user_id: User identifier for scoping.
            limit: Maximum number of results.
            timeout_seconds: Maximum time to wait.

        Returns:
            List of StickyNote objects with matching facts.
        """
        if not self.is_available:
            logger.debug("Mem0 unavailable, returning empty search results")
            return []

        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    self._memory.search, query, user_id=user_id, limit=limit
                ),
                timeout=timeout_seconds,
            )

            notes: list[StickyNote] = []
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict):
                        notes.append(
                            StickyNote(
                                key=item.get("id", ""),
                                value=item.get("memory", ""),
                                importance_score=item.get("score", 0.5),
                            )
                        )
            return notes
        except TimeoutError:
            logger.warning(
                f"search_facts_async timed out after {timeout_seconds}s"
            )
            return []
        except Exception as e:
            logger.error(f"search_facts_async failed: {e}")
            return []

    def delete_fact(self, memory_id: str) -> bool:
        """Delete a specific memory by ID.

        Args:
            memory_id: The memory ID to delete.

        Returns:
            True if deletion succeeded, False otherwise.
        """
        if not self.is_available:
            return False

        try:
            self._memory.delete(memory_id)
            return True
        except Exception as e:
            logger.error(f"Mem0 delete_fact failed: {e}")
            return False

    async def async_store_fact(
        self,
        messages: list[dict[str, Any]],
        user_id: str,
        timeout_seconds: float = 10.0,
        **kwargs: Any,
    ) -> str:
        """Asynchronously store facts without blocking.

        Args:
            messages: Messages to extract facts from.
            user_id: User identifier.
            timeout_seconds: Maximum time to wait.
            **kwargs: Additional arguments for store_fact.

        Returns:
            Memory ID string, or empty string if timeout/failure.
        """
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self.store_fact,
                    messages,
                    user_id,
                    **kwargs,
                ),
                timeout=timeout_seconds,
            )
            return result
        except TimeoutError:
            logger.warning(f"async_store_fact timed out after {timeout_seconds}s")
            return ""
        except Exception as e:
            logger.error(f"async_store_fact failed: {e}")
            return ""

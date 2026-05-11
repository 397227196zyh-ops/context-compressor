"""Cross-session persistence with SQLite."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

from context_compressor.persistence.session_state import SessionState, SessionSummary


class SessionStore:
    def __init__(self, db_path: str = "sessions.db", session_ttl_days: int = 30) -> None:
        self.db_path = db_path
        self.session_ttl_days = session_ttl_days
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    messages_json TEXT,
                    anchor_json TEXT,
                    metadata_json TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            conn.commit()

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()

    def save_session(self, session_id, messages, anchor_state, metadata=None):
        now = self._now()
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id=?", (session_id,),
            ).fetchone()
            if existing:
                conn.execute("""
                    UPDATE sessions SET messages_json=?, anchor_json=?, metadata_json=?, updated_at=?
                    WHERE session_id=?
                """, (json.dumps(messages), json.dumps(anchor_state),
                    json.dumps(metadata or {}), now, session_id))
            else:
                conn.execute("""
                    INSERT INTO sessions VALUES (?,?,?,?,?,?)
                """, (session_id, json.dumps(messages), json.dumps(anchor_state),
                    json.dumps(metadata or {}), now, now))
            conn.commit()

    def load_session(self, session_id):
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id=?", (session_id,),
            ).fetchone()
            if not row:
                return None
            return SessionState(
                session_id=row[0], messages=json.loads(row[1]),
                anchor_state=json.loads(row[2]), metadata=json.loads(row[3]),
                created_at=row[4], updated_at=row[5],
            )

    def list_sessions(self, limit=50):
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT session_id, messages_json, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [SessionSummary(session_id=r[0], message_count=len(json.loads(r[1])),
                created_at=r[2], updated_at=r[3]) for r in rows]

    def delete_session(self, session_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
            conn.commit()

    def cleanup_expired(self):
        cutoff = (datetime.now(UTC) - timedelta(days=self.session_ttl_days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE updated_at < ?", (cutoff,))
            conn.commit()

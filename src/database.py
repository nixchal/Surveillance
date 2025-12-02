"""SQLite database utilities for persisting surveillance events."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import config


LOGGER = logging.getLogger(__name__)


SCHEMA_MIGRATIONS = (
    """
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        camera_id TEXT,
        event_type TEXT NOT NULL,
        priority TEXT NOT NULL,
        confidence REAL,
        location TEXT,
        description TEXT,
        image_path TEXT,
        acknowledged INTEGER DEFAULT 0,
        acknowledged_by TEXT,
        acknowledged_at TEXT
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_events_timestamp
        ON events(timestamp DESC);
    """,
)


class DatabaseManager:
    """Wrapper for SQLite event persistence."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else config.DATABASE_PATH
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self.apply_migrations()
        LOGGER.debug("Database initialized at %s", self.db_path)

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    def apply_migrations(self) -> None:
        cursor = self._connection.cursor()
        for migration in SCHEMA_MIGRATIONS:
            cursor.execute(migration)
        self._connection.commit()

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
        cursor = self._connection.cursor()
        try:
            yield cursor
            self._connection.commit()
        except sqlite3.DatabaseError as exc:
            LOGGER.exception("Database error: %s", exc)
            self._connection.rollback()
            raise
        finally:
            cursor.close()

    def save_event(self, alert: dict[str, Any]) -> int:
        with self.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO events (
                    timestamp, camera_id, event_type, priority, confidence,
                    location, description, image_path, acknowledged,
                    acknowledged_by, acknowledged_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert.get("timestamp"),
                    alert.get("camera_id"),
                    alert.get("type"),
                    alert.get("priority"),
                    alert.get("confidence"),
                    alert.get("location"),
                    alert.get("description"),
                    alert.get("image_path"),
                    int(alert.get("acknowledged", False)),
                    alert.get("acknowledged_by"),
                    alert.get("acknowledged_at"),
                ),
            )
            event_id = int(cursor.lastrowid)
            LOGGER.info("Saved event %s (%s)", event_id, alert.get("type"))
            return event_id

    def get_recent_events(self, limit: int = 50) -> list[sqlite3.Row]:
        with self.cursor() as cursor:
            rows = cursor.execute(
                "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return rows

    def cleanup_old_events(self, retention_days: int | None = None) -> int:
        days = retention_days or config.DB_RETENTION_DAYS
        with self.cursor() as cursor:
            rows = cursor.execute(
                "DELETE FROM events WHERE timestamp <= datetime('now', ?)",
                (f"-{days} days",)
            ).rowcount
            LOGGER.info("Cleaned up %s events older than %s days", rows, days)
            return rows

    def close(self) -> None:
        LOGGER.debug("Closing database connection")
        self._connection.close()


__all__ = ["DatabaseManager"]


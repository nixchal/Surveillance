from __future__ import annotations

import tempfile
from pathlib import Path

from src.database import DatabaseManager


def test_save_and_query_event():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "events.db"
        manager = DatabaseManager(db_path)

        alert = {
            "timestamp": "2025-01-01T00:00:00Z",
            "camera_id": "cam-1",
            "type": "Test event",
            "priority": "low",
            "confidence": 0.9,
            "location": "",
            "description": "Unit test alert",
            "image_path": None,
        }

        event_id = manager.save_event(alert)
        assert event_id == 1

        events = manager.get_recent_events(10)
        assert len(events) == 1
        assert events[0]["event_type"] == "Test event"

        manager.close()


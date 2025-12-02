"""Alert management: cooldowns, persistence and notifications."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config
from .database import DatabaseManager


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Alert:
    timestamp: str
    type: str
    priority: str
    confidence: float | None = None
    camera_id: str | None = None
    description: str | None = None
    image_path: Path | None = None
    location: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "timestamp": self.timestamp,
            "type": self.type,
            "priority": self.priority,
            "confidence": self.confidence,
            "camera_id": self.camera_id,
            "description": self.description,
            "image_path": str(self.image_path) if self.image_path else None,
            "location": self.location,
        }
        data.update(self.metadata)
        return data


class AlertManager:
    """Handle alert deduplication, persistence and notifications."""

    def __init__(self, cooldown_seconds: int | None = None, db: DatabaseManager | None = None) -> None:
        self.cooldown_seconds = cooldown_seconds or config.ALERT_COOLDOWN_SECONDS
        self._last_alert_times: defaultdict[str, float] = defaultdict(lambda: 0.0)
        self._db = db or DatabaseManager()

    def _is_on_cooldown(self, alert_key: str) -> bool:
        now = time.time()
        last = self._last_alert_times[alert_key]
        if now - last < self.cooldown_seconds:
            LOGGER.debug("Alert %s suppressed due to cooldown", alert_key)
            return True
        self._last_alert_times[alert_key] = now
        return False

    def should_emit(self, alert: Alert) -> bool:
        alert_key = f"{alert.type}:{alert.camera_id}:{alert.location}"
        return not self._is_on_cooldown(alert_key)

    def emit(self, alert: Alert) -> None:
        if not self.should_emit(alert):
            return

        LOGGER.warning(
            "ALERT [%s] %s (confidence=%.2f) %s",
            alert.priority.upper(),
            alert.type,
            alert.confidence or -1.0,
            f"camera={alert.camera_id}" if alert.camera_id else "",
        )

        self._db.save_event(alert.to_dict())

    def close(self) -> None:
        self._db.close()


__all__ = ["Alert", "AlertManager"]


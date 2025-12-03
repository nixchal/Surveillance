"""Global configuration for the Campus Surveillance system."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path


# --- Core Paths -----------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

DATABASE_PATH = DATA_DIR / "database" / "surveillance.db"
EVENTS_DIR = DATA_DIR / "events"
LOGS_DIR = DATA_DIR / "logs"


for target in [DATA_DIR, MODELS_DIR, STATIC_DIR, TEMPLATES_DIR, EVENTS_DIR, LOGS_DIR, DATABASE_PATH.parent]:
    target.mkdir(parents=True, exist_ok=True)


# --- Video & Detection -----------------------------------------------------------

CAMERA_SOURCE: int | str = os.getenv("CAMERA_SOURCE", 0)  # 0 for default webcam, or RTSP URL / file path
if isinstance(CAMERA_SOURCE, str) and CAMERA_SOURCE.isdigit():
    CAMERA_SOURCE = int(CAMERA_SOURCE)
FRAME_WIDTH: int | None = None  # Set to reduce resolution, e.g., 640
FRAME_HEIGHT: int | None = None  # Set to reduce resolution, e.g., 480
FPS_TARGET: int = 15

CONFIDENCE_THRESHOLD: float = 0.5
CROWD_THRESHOLD: int = 5
FIRE_COLOR_THRESHOLD: float = 0.05
SMOKING_HAND_DISTANCE_THRESHOLD: float = 0.1
ALERT_COOLDOWN_SECONDS: int = 10
POSE_ANALYSIS_INTERVAL: int = 5
SMOKING_FRAMES_THRESHOLD: int = 8
FIGHT_FRAMES_THRESHOLD: int = 1
FIGHT_IOU_THRESHOLD: float = 0.02
FIGHT_DISTANCE_THRESHOLD_RATIO: float = 3.00
FIGHT_DISTANCE_MAX_PIXELS: float = 260.0
FIGHT_MOTION_RATIO_THRESHOLD: float = 0.006  # fraction of pixels changing
FIGHT_MASK_DILATION_PIXELS: int = 48
FIGHT_MASK_MIN_OVERLAP_RATIO: float = 0.001  # overlap relative to smaller mask area
FIGHT_MULTI_PERSON_COUNT: int = 3
FIGHT_HIGH_MOTION_THRESHOLD: float = 0.04
FIGHT_DISTANCE_FORCE_THRESHOLD: float = 120.0
SMOKE_COLOR_THRESHOLD: float = 0.06
SMOKE_MIN_AREA_RATIO: float = 0.01


# --- SAM 3.0 Segmentation ----------------------------------------------------

SAM_ENABLED: bool = True  # Set to False to disable SAM-based person masks

# --- Feature Flags --------------------------------------------------------------

ENABLE_FIRE_DETECTION: bool = True
ENABLE_SMOKING_DETECTION: bool = False
ENABLE_WEAPON_DETECTION: bool = False
ENABLE_FALL_DETECTION: bool = True
ENABLE_FIGHT_DETECTION: bool = True
ENABLE_CROWD_DETECTION: bool = True
ENABLE_SMOKE_COLOR_DETECTION: bool = True


# --- Alert Priorities -----------------------------------------------------------

class AlertPriority:
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# --- Logging --------------------------------------------------------------------

LOG_LEVEL = logging.INFO
LOG_FILE_PATH = LOGS_DIR / "system.log"


def configure_logging(level: int = LOG_LEVEL) -> None:
    """Configure root logger for the application."""

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"),
        ],
    )


# --- Database -------------------------------------------------------------------

DB_RETENTION_DAYS: int = 90


# --- Zone Defaults --------------------------------------------------------------

DEFAULT_ZONE_CONFIG = {
    "restricted": {
        "priority": AlertPriority.HIGH,
        "description": "Restricted area entry",
    }
}


# --- Email / Notification Stubs -------------------------------------------------

@dataclass(slots=True)
class EmailSettings:
    enabled: bool = False
    smtp_server: str = ""
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    sender: str = ""
    recipients: tuple[str, ...] = ()


EMAIL_SETTINGS = EmailSettings()


def env_flag(name: str, default: bool) -> bool:
    """Read boolean environment variables (1/0, true/false)."""

    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


DEBUG_MODE = env_flag("SURVEILLANCE_DEBUG", False)



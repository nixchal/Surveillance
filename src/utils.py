"""Utility helpers for the surveillance system."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

import cv2

import config


def timestamp_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_frame(frame: cv2.Mat, filename: str) -> Path:
    events_dir = ensure_directory(config.EVENTS_DIR)
    output_path = events_dir / filename
    cv2.imwrite(str(output_path), frame)
    return output_path


def polygon_contains(point: Tuple[int, int], polygon: Iterable[Tuple[int, int]]) -> bool:
    x, y = point
    poly = list(polygon)
    inside = False
    if len(poly) < 3:
        return False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


__all__ = [
    "timestamp_iso",
    "ensure_directory",
    "save_frame",
    "polygon_contains",
]


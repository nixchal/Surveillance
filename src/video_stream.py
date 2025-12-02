"""Video capture utilities."""

from __future__ import annotations

import logging
from typing import Generator, Optional

import cv2

import config


LOGGER = logging.getLogger(__name__)


class VideoStream:
    """Context manager for OpenCV video capture."""

    def __init__(self, source: int | str | None = None) -> None:
        self.source = config.CAMERA_SOURCE if source is None else source
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        LOGGER.info("Opening video source: %s", self.source)
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video source {self.source!r}")

        if config.FRAME_WIDTH:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        if config.FRAME_HEIGHT:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    def close(self) -> None:
        if self.cap is not None:
            LOGGER.info("Releasing video source")
            self.cap.release()
            self.cap = None

    def frames(self) -> Generator[tuple[bool, Optional[cv2.Mat]], None, None]:
        if self.cap is None:
            self.open()

        assert self.cap is not None

        while True:
            success, frame = self.cap.read()
            yield success, frame

    def snapshot(self) -> Optional[cv2.Mat]:
        if self.cap is None:
            self.open()
        assert self.cap is not None
        success, frame = self.cap.read()
        return frame if success else None

    def __enter__(self) -> "VideoStream":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["VideoStream"]


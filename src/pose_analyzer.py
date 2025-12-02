"""Pose analysis utilities backed by MediaPipe."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2

try:
    from mediapipe.python import solutions as mp_solutions
except ImportError:  # pragma: no cover - gracefully degrade in environments without mediapipe
    mp_solutions = None  # type: ignore


@dataclass(slots=True)
class PoseAnalysisResult:
    hand_near_mouth: bool
    left_hand_distance: Optional[float]
    right_hand_distance: Optional[float]
    landmarks: Optional[Tuple[Tuple[float, float, float], ...]] = None


class PoseAnalyzer:
    """Wrapper around MediaPipe pose estimation to provide heuristics."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        hand_distance_threshold: float = 0.12,
    ) -> None:
        if mp_solutions is None:
            raise RuntimeError("mediapipe is not available; install it to enable pose-based analysis")

        self.hand_distance_threshold = hand_distance_threshold
        self._pose = mp_solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    @staticmethod
    def available() -> bool:
        return mp_solutions is not None

    def close(self) -> None:
        self._pose.close()

    def analyze(self, frame) -> Optional[PoseAnalysisResult]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)

        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks.landmark

        pose_module = mp_solutions.pose
        pose_lm = pose_module.PoseLandmark

        def _distance(a, b) -> Optional[float]:
            if landmarks[a].visibility < 0.5 or landmarks[b].visibility < 0.5:
                return None
            dx = landmarks[a].x - landmarks[b].x
            dy = landmarks[a].y - landmarks[b].y
            return math.hypot(dx, dy)

        mouth_landmark = pose_lm.MOUTH_LEFT if hasattr(pose_lm, "MOUTH_LEFT") else pose_lm.NOSE
        left_distance = _distance(pose_lm.LEFT_WRIST, mouth_landmark)
        right_distance = _distance(pose_lm.RIGHT_WRIST, mouth_landmark)

        hand_near_mouth = any(
            d is not None and d <= self.hand_distance_threshold
            for d in (left_distance, right_distance)
        )

        landmarks_tuple: Tuple[Tuple[float, float, float], ...] = tuple(
            (lm.x, lm.y, lm.visibility) for lm in landmarks
        )

        return PoseAnalysisResult(
            hand_near_mouth=hand_near_mouth,
            left_hand_distance=left_distance,
            right_hand_distance=right_distance,
            landmarks=landmarks_tuple,
        )


__all__ = ["PoseAnalyzer", "PoseAnalysisResult"]


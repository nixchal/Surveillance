import cv2
import numpy as np
import config
from config import AlertPriority

class FireDetector:
    def __init__(self):
        pass

    def detect(self, frame: np.ndarray) -> float:
        """
        Detects fire in the frame using color analysis.
        Returns the ratio of fire-like pixels.
        """
        if not config.ENABLE_FIRE_DETECTION:
            return 0.0

        fire_mask = self._fire_color_mask(frame)
        fire_ratio = float(np.count_nonzero(fire_mask)) / fire_mask.size
        return fire_ratio

    def _fire_color_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 120, 150])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 100, 150])
        upper2 = np.array([179, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        return cv2.bitwise_or(mask1, mask2)

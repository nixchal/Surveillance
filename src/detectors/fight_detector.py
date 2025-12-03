import numpy as np
import config
from config import AlertPriority
import cv2

class FightDetector:
    def __init__(self):
        self.fight_frames = 0
        self.metrics = {}

    def detect(self, frame: np.ndarray, detections: list, motion_ratio: float) -> dict:
        """
        Detects fights based on person proximity, mask overlap, and motion.
        Returns a dictionary with detection result and metrics.
        """
        if not config.ENABLE_FIGHT_DETECTION:
            return {"detected": False}

        persons = [det for det in detections if det.label == "person"]
        if len(persons) < 2:
            self.fight_frames = max(0, self.fight_frames - 1)
            self.metrics = {}
            return {"detected": False}

        fight_likely = False
        close_pair = False
        mask_pair = False
        min_distance = float("inf")
        max_iou = 0.0
        best_mask_overlap = 0.0

        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                det_a = persons[i]
                det_b = persons[j]
                
                # IOU Check
                iou = self._detection_iou(det_a, det_b)
                if iou > max_iou:
                    max_iou = iou
                if iou >= config.FIGHT_IOU_THRESHOLD:
                    fight_likely = True
                    break
                
                # Distance Check
                distance = self._detection_distance(det_a, det_b)
                if distance < min_distance:
                    min_distance = distance
                
                # Mask Overlap Check
                mask_overlap = self._mask_contact_ratio(det_a, det_b)
                if mask_overlap > best_mask_overlap:
                    best_mask_overlap = mask_overlap
                if mask_overlap >= config.FIGHT_MASK_MIN_OVERLAP_RATIO:
                    mask_pair = True
                    close_pair = True
                    break
                
                # Heuristic Distance Check
                min_width = min(det_a.bbox[2] - det_a.bbox[0], det_b.bbox[2] - det_b.bbox[0])
                distance_threshold = max(20.0, min_width * config.FIGHT_DISTANCE_THRESHOLD_RATIO)
                if distance <= min(config.FIGHT_DISTANCE_MAX_PIXELS, distance_threshold):
                    close_pair = True
                    break
            
            if fight_likely:
                break

        # Boost decision with motion
        if not fight_likely and (close_pair or mask_pair) and motion_ratio >= config.FIGHT_MOTION_RATIO_THRESHOLD:
            fight_likely = True

        # Multi-person high motion
        if (
            not fight_likely
            and len(persons) >= config.FIGHT_MULTI_PERSON_COUNT
            and motion_ratio >= config.FIGHT_HIGH_MOTION_THRESHOLD
            and best_mask_overlap > 0.0
        ):
            fight_likely = True

        # Force threshold
        if (
            not fight_likely
            and min_distance != float("inf")
            and motion_ratio >= config.FIGHT_HIGH_MOTION_THRESHOLD
            and min_distance <= config.FIGHT_DISTANCE_FORCE_THRESHOLD
        ):
            fight_likely = True

        self.metrics = {
            "mask_overlap": best_mask_overlap,
            "min_distance": None if min_distance == float("inf") else min_distance,
            "max_iou": max_iou,
            "person_count": len(persons),
            "motion_ratio": motion_ratio
        }

        if fight_likely:
            self.fight_frames += 1
        else:
            self.fight_frames = max(0, self.fight_frames - 1)

        if self.fight_frames >= config.FIGHT_FRAMES_THRESHOLD:
            self.fight_frames = 0
            return {"detected": True, "metrics": self.metrics}
        
        return {"detected": False, "metrics": self.metrics}

    def _detection_iou(self, det_a, det_b) -> float:
        if det_a.mask is not None and det_b.mask is not None:
            try:
                intersection = np.logical_and(det_a.mask, det_b.mask).sum()
                if intersection == 0:
                    return 0.0
                union = np.logical_or(det_a.mask, det_b.mask).sum()
                return float(intersection / union) if union > 0 else 0.0
            except Exception:
                pass
        return self._bbox_iou(det_a.bbox, det_b.bbox)

    def _bbox_iou(self, a, b) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter_width = max(0.0, x2 - x1)
        inter_height = max(0.0, y2 - y1)
        intersection = inter_width * inter_height
        if intersection <= 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    def _detection_distance(self, det_a, det_b) -> float:
        centroid_a = self._detection_centroid(det_a)
        centroid_b = self._detection_centroid(det_b)
        return float(np.hypot(centroid_a[0] - centroid_b[0], centroid_a[1] - centroid_b[1]))

    def _detection_centroid(self, detection) -> tuple[int, int]:
        if detection.mask is not None:
            try:
                mask_uint = detection.mask.astype(np.uint8)
                moments = cv2.moments(mask_uint)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    return cx, cy
            except Exception:
                pass
        x1, y1, x2, y2 = detection.bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def _mask_contact_ratio(self, det_a, det_b) -> float:
        if det_a.mask is None or det_b.mask is None:
            return 0.0
        try:
            kernel_size = max(1, config.FIGHT_MASK_DILATION_PIXELS)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            dilated_a = cv2.dilate(det_a.mask.astype(np.uint8), kernel, iterations=1)
            overlap = np.logical_and(dilated_a.astype(bool), det_b.mask).sum()
            min_area = float(min(det_a.mask.sum(), det_b.mask.sum()))
            if min_area <= 0:
                return 0.0
            return overlap / min_area
        except Exception:
            return 0.0

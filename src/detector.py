"""Main detection pipeline combining YOLO and heuristic detectors."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import config
from model_loader import ModelManager
from .alert_manager import Alert, AlertManager
from .utils import polygon_contains, save_frame, timestamp_iso
from .video_stream import VideoStream
from .zone_manager import Zone, ZoneManager
from .pose_analyzer import PoseAnalyzer, PoseAnalysisResult
from .sam_segmenter import SamPersonSegmenter


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    bbox: np.ndarray  # [x1, y1, x2, y2]
    area: float
    mask: Optional[np.ndarray] = None  # boolean mask aligned with frame
    polygon: Optional[np.ndarray] = None  # segmentation polygon for overlay
    pose: Optional[PoseAnalysisResult] = None
    pose_landmarks: Optional[np.ndarray] = None  # absolute pixel coordinates (N x 3)
    hand_speed: float = 0.0


class CampusSurveillance:
    """High-level orchestrator for camera processing and alerting."""

    def __init__(self, show_windows: bool = True) -> None:
        self.model_manager = ModelManager()
        self.object_model = self.model_manager.get_model("objects")
        self.segmentation_model = self.model_manager.get_model("objects_seg")
        if self.object_model is None and self.segmentation_model is None:
            raise RuntimeError("No object detection models are available")
        self.class_names = getattr(self.object_model, "names", {})

        # Optional SAM3 person segmenter
        self.sam_segmenter: Optional[SamPersonSegmenter]
        if config.SAM_ENABLED:
            try:
                self.sam_segmenter = SamPersonSegmenter()
                LOGGER.info("SAM3 person segmenter enabled")
            except Exception as exc:
                LOGGER.warning("SAM3 person segmenter unavailable: %s", exc)
                self.sam_segmenter = None
        else:
            self.sam_segmenter = None

        self.alert_manager = AlertManager()
        self.zone_manager = ZoneManager()
        self.video_stream = VideoStream()
        self.show_windows = show_windows
        self.running = False
        self._last_frame: Optional[np.ndarray] = None
        self.frame_index = 0
        self.pose_analyzer: Optional[PoseAnalyzer]
        if PoseAnalyzer.available():
            try:
                self.pose_analyzer = PoseAnalyzer(
                    hand_distance_threshold=config.SMOKING_HAND_DISTANCE_THRESHOLD
                )
            except RuntimeError as exc:
                LOGGER.warning("Pose analyzer unavailable: %s", exc)
                self.pose_analyzer = None
        else:
            LOGGER.info("mediapipe not installed; pose-based fallbacks disabled")
            self.pose_analyzer = None
        self.smoking_frames = 0
        self.active_smoking_instances = 0
        self.fight_frames = 0
        self._prev_gray: Optional[np.ndarray] = None
        self._fight_metrics: Dict[str, float] = {}
        self._pose_history: List[Dict[str, np.ndarray]] = []

        LOGGER.info("CampusSurveillance initialized (show_windows=%s)", show_windows)

    # ------------------------------------------------------------------ Lifecycle
    def run(self) -> None:
        self.running = True
        frame_generator = self.video_stream.frames()

        for success, frame in frame_generator:
            if not self.running:
                break
            if not success or frame is None:
                LOGGER.warning("Failed to read frame from source")
                continue

            self.frame_index += 1

            # Estimate scene motion before mutating state
            motion_ratio = self._estimate_motion_ratio(frame)

            detections = self._detect_objects(frame)
            self._handle_crowd_detection(frame, detections)
            self._handle_fire_detection(frame)
            self._handle_smoking_detection(frame, detections)
            self._handle_fight_detection(frame, detections, motion_ratio)
            self._handle_smoke_color_detection(frame)

            self._draw_overlays(frame, detections)
            # Debug overlay for fight tuning
            try:
                cv2.putText(
                    frame,
                    f"motion={motion_ratio:.3f}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                if self._fight_metrics:
                    cv2.putText(
                        frame,
                        "overlap={mask_overlap:.3f} dist={min_distance:.1f} iou={max_iou:.3f}".format(
                            mask_overlap=self._fight_metrics.get("mask_overlap", 0.0),
                            min_distance=self._fight_metrics.get("min_distance", 0.0) or 0.0,
                            max_iou=self._fight_metrics.get("max_iou", 0.0),
                        ),
                        (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 200, 200),
                        2,
                    )
            except Exception:
                pass
            self._last_frame = frame

            if self.show_windows:
                cv2.imshow("Campus Surveillance", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    LOGGER.info("User requested exit via keyboard")
                    break

        self.stop()

    def stop(self) -> None:
        if not self.running:
            return
        LOGGER.info("Stopping CampusSurveillance")
        self.running = False
        cv2.destroyAllWindows()
        self.video_stream.close()
        self.alert_manager.close()
        if self.pose_analyzer is not None:
            self.pose_analyzer.close()
            self.pose_analyzer = None

    # ------------------------------------------------------------------ Detection
    def _detect_objects(self, frame: np.ndarray) -> List[Detection]:
        model = self.segmentation_model or self.object_model
        results = model.predict(frame, verbose=False)

        detections: List[Detection] = []
        person_indices: List[int] = []
        person_bboxes: List[np.ndarray] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            names_map: Dict[int, str]
            names_map = getattr(result, "names", {}) or self.class_names
            mask_data = getattr(result, "masks", None)
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                if conf < config.CONFIDENCE_THRESHOLD:
                    continue
                label = names_map.get(cls_id, str(cls_id))
                bbox = boxes.xyxy[i].cpu().numpy()
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                mask = None
                polygon = None
                if mask_data is not None and mask_data.data is not None:
                    try:
                        mask_tensor = mask_data.data[i]
                        mask = mask_tensor.cpu().numpy().astype(bool)
                        if mask.shape[0] != frame.shape[0] or mask.shape[1] != frame.shape[1]:
                            mask = cv2.resize(
                                mask.astype(np.uint8),
                                (frame.shape[1], frame.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            ).astype(bool)
                        polygons = getattr(mask_data, "xy", None)
                        if polygons is not None and len(polygons) > i:
                            polygon = np.array(polygons[i], dtype=np.int32)
                    except Exception:
                        mask = None
                        polygon = None

                detection = Detection(
                    label=label,
                    confidence=conf,
                    bbox=bbox,
                    area=area,
                    mask=mask,
                    polygon=polygon,
                )
                idx = len(detections)
                detections.append(detection)

                if label == "person":
                    person_indices.append(idx)
                    person_bboxes.append(bbox)
                    self._handle_person_detection(frame, detection)

        # Refine person masks with SAM3 if available
        if self.sam_segmenter is not None and person_bboxes:
            try:
                sam_masks = self.sam_segmenter.segment_persons(frame, person_bboxes)
                for det_idx, sam_mask in zip(person_indices, sam_masks):
                    if sam_mask is not None:
                        detections[det_idx].mask = sam_mask
            except Exception as exc:
                LOGGER.warning("SAM3 refinement failed for this frame: %s", exc)

        return detections

    def _handle_person_detection(self, frame: np.ndarray, detection: Detection) -> None:
        centroid = self._bbox_centroid(detection.bbox)
        for zone in self.zone_manager.active_zones():
            if polygon_contains(centroid, zone.points):
                self._raise_alert(
                    event_type="Restricted zone entry",
                    priority=config.AlertPriority.HIGH,
                    confidence=detection.confidence,
                    description=f"Person entered zone {zone.name}",
                    frame=frame,
                )
                break

    def _handle_crowd_detection(self, frame: np.ndarray, detections: List[Detection]) -> None:
        person_count = sum(1 for det in detections if det.label == "person")
        if config.ENABLE_CROWD_DETECTION and person_count >= config.CROWD_THRESHOLD:
            self._raise_alert(
                event_type="Crowd detected",
                priority=config.AlertPriority.MEDIUM,
                confidence=float(person_count),
                description=f"Detected {person_count} people in frame",
                frame=frame,
            )

    def _handle_fire_detection(self, frame: np.ndarray) -> None:
        if not config.ENABLE_FIRE_DETECTION:
            return

        fire_mask = self._fire_color_mask(frame)
        fire_ratio = float(np.count_nonzero(fire_mask)) / fire_mask.size
        if fire_ratio >= config.FIRE_COLOR_THRESHOLD:
            self._raise_alert(
                event_type="Fire suspected",
                priority=config.AlertPriority.CRITICAL,
                confidence=fire_ratio,
                description=f"Fire-like colors detected ({fire_ratio:.2%})",
                frame=frame,
            )

    def _handle_smoking_detection(self, frame: np.ndarray, detections: List[Detection]) -> None:
        if not config.ENABLE_SMOKING_DETECTION:
            return
        if self.model_manager.is_available("smoking"):
            return
        if self.pose_analyzer is None:
            return
        if self.frame_index % config.POSE_ANALYSIS_INTERVAL != 0:
            return

        persons = [det for det in detections if det.label == "person"]
        if not persons:
            self.smoking_frames = max(0, self.smoking_frames - 1)
            self.active_smoking_instances = 0
            return

        smoking_instances: List[Tuple[Detection, PoseAnalysisResult]] = []

        for det in persons:
            analysis = self._ensure_pose(frame, det)
            if analysis and analysis.hand_near_mouth:
                smoking_instances.append((det, analysis))

        if smoking_instances:
            self.smoking_frames += 1
            self.active_smoking_instances = len(smoking_instances)
        else:
            self.smoking_frames = max(0, self.smoking_frames - 1)
            self.active_smoking_instances = 0

        if self.smoking_frames >= config.SMOKING_FRAMES_THRESHOLD and smoking_instances:
            _, first_analysis = smoking_instances[0]
            self._raise_alert(
                event_type="Smoking suspected",
                priority=config.AlertPriority.MEDIUM,
                confidence=None,
                description=f"Hand-to-mouth gestures detected ({len(smoking_instances)} person(s))",
                frame=frame,
                extra={
                    "smoking_instances": len(smoking_instances),
                    "example_left_distance": first_analysis.left_hand_distance,
                    "example_right_distance": first_analysis.right_hand_distance,
                },
            )
            self.smoking_frames = 0
            self.active_smoking_instances = 0

    def _handle_fight_detection(self, frame: np.ndarray, detections: List[Detection], motion_ratio: float) -> None:
        if not config.ENABLE_FIGHT_DETECTION:
            return

        persons = [det for det in detections if det.label == "person"]
        if len(persons) < 2:
            self.fight_frames = max(0, self.fight_frames - 1)
            self._fight_metrics = {}
            return

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
                iou = self._detection_iou(det_a, det_b)
                if iou > max_iou:
                    max_iou = iou
                if iou >= config.FIGHT_IOU_THRESHOLD:
                    fight_likely = True
                    break
                distance = self._detection_distance(det_a, det_b)
                if distance < min_distance:
                    min_distance = distance
                mask_overlap = self._mask_contact_ratio(det_a, det_b)
                if mask_overlap > best_mask_overlap:
                    best_mask_overlap = mask_overlap
                if mask_overlap >= config.FIGHT_MASK_MIN_OVERLAP_RATIO:
                    mask_pair = True
                    close_pair = True
                    break
                min_width = min(det_a.bbox[2] - det_a.bbox[0], det_b.bbox[2] - det_b.bbox[0])
                distance_threshold = max(20.0, min_width * config.FIGHT_DISTANCE_THRESHOLD_RATIO)
                if distance <= min(config.FIGHT_DISTANCE_MAX_PIXELS, distance_threshold):
                    close_pair = True
                    # Defer final decision to motion check below
                    break
            if fight_likely:
                break

        # Boost decision with motion: high motion + close pair suggests scuffle
        if not fight_likely and (close_pair or mask_pair) and motion_ratio >= config.FIGHT_MOTION_RATIO_THRESHOLD:
            fight_likely = True

        # Additional heuristic: multiple people with very high motion in frame
        if (
            not fight_likely
            and len(persons) >= config.FIGHT_MULTI_PERSON_COUNT
            and motion_ratio >= config.FIGHT_HIGH_MOTION_THRESHOLD
            and best_mask_overlap > 0.0
        ):
            fight_likely = True

        if (
            not fight_likely
            and min_distance != float("inf")
            and motion_ratio >= config.FIGHT_HIGH_MOTION_THRESHOLD
            and min_distance <= config.FIGHT_DISTANCE_FORCE_THRESHOLD
        ):
            fight_likely = True

        self._fight_metrics = {
            "mask_overlap": best_mask_overlap,
            "min_distance": None if min_distance == float("inf") else min_distance,
            "max_iou": max_iou,
        }

        if fight_likely:
            self.fight_frames += 1
        else:
            self.fight_frames = max(0, self.fight_frames - 1)

        if self.fight_frames >= config.FIGHT_FRAMES_THRESHOLD:
            self._raise_alert(
                event_type="Fight suspected",
                priority=config.AlertPriority.HIGH,
                confidence=None,
                description="Person proximity and motion suggest fighting",
                frame=frame,
                extra={
                    "person_count": len(persons),
                    "motion_ratio": motion_ratio,
                    "min_distance": min_distance if min_distance != float("inf") else None,
                    "max_iou": max_iou,
                    "mask_overlap": best_mask_overlap,
                },
            )
            self.fight_frames = 0

    def _handle_smoke_color_detection(self, frame: np.ndarray) -> None:
        if not config.ENABLE_SMOKE_COLOR_DETECTION:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        smoke_ratio = float(np.count_nonzero(thresh)) / thresh.size
        if smoke_ratio < config.SMOKE_MIN_AREA_RATIO:
            return
        edges = cv2.Canny(blurred, 30, 90)
        smoke_texture = float(np.count_nonzero(edges)) / edges.size
        if smoke_texture >= config.SMOKE_COLOR_THRESHOLD:
            self._raise_alert(
                event_type="Smoke suspected",
                priority=config.AlertPriority.MEDIUM,
                confidence=smoke_texture,
                description=f"Diffuse smoke-like texture detected ({smoke_texture:.2%})",
                frame=frame,
            )

    @staticmethod
    def _crop_person_frame(frame: np.ndarray, bbox: np.ndarray, padding: float = 0.15) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        pad_x = width * padding
        pad_y = height * padding
        x1 = int(max(0, x1 - pad_x))
        y1 = int(max(0, y1 - pad_y))
        x2 = int(min(w, x2 + pad_x))
        y2 = int(min(h, y2 + pad_y))
        if x2 <= x1 or y2 <= y1:
            return None, (x1, y1, x2, y2)
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    # ------------------------------------------------------------------ Utilities
    def _fire_color_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 120, 150])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([160, 100, 150])
        upper2 = np.array([179, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        return cv2.bitwise_or(mask1, mask2)

    @staticmethod
    def _bbox_centroid(bbox: np.ndarray) -> tuple[int, int]:
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def _draw_overlays(self, frame: np.ndarray, detections: List[Detection]) -> None:
        for detection in detections:
            if detection.mask is not None:
                try:
                    overlay = np.zeros_like(frame)
                    overlay[detection.mask] = (0, 180, 255)
                    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
                    if detection.polygon is not None and len(detection.polygon) >= 3:
                        cv2.polylines(frame, [detection.polygon], True, (0, 200, 255), 2)
                except Exception:
                    pass
            x1, y1, x2, y2 = detection.bbox.astype(int)
            label = f"{detection.label} {detection.confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for zone in self.zone_manager.active_zones():
            points = np.array(zone.points, dtype=np.int32)
            if len(points) >= 3:
                cv2.polylines(frame, [points], True, (0, 0, 255), 2)
                cv2.putText(frame, zone.name, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def _raise_alert(
        self,
        event_type: str,
        priority: str,
        confidence: Optional[float],
        description: str,
        frame: Optional[np.ndarray],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        timestamp = timestamp_iso()
        image_path = None
        if frame is not None:
            safe_time = timestamp.replace(":", "-")
            filename = f"event_{event_type.replace(' ', '_').lower()}_{safe_time}.jpg"
            image_path = save_frame(frame, filename)

        alert = Alert(
            timestamp=timestamp,
            type=event_type,
            priority=priority,
            confidence=confidence,
            description=description,
            image_path=image_path,
            metadata=extra or {},
        )
        self.alert_manager.emit(alert)

    @staticmethod
    def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
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

    @staticmethod
    def _centroid_distance(a: np.ndarray, b: np.ndarray) -> float:
        ax, ay = CampusSurveillance._bbox_centroid(a)
        bx, by = CampusSurveillance._bbox_centroid(b)
        return float(np.hypot(ax - bx, ay - by))

    def _detection_iou(self, det_a: Detection, det_b: Detection) -> float:
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

    def _detection_distance(self, det_a: Detection, det_b: Detection) -> float:
        centroid_a = self._detection_centroid(det_a)
        centroid_b = self._detection_centroid(det_b)
        return float(np.hypot(centroid_a[0] - centroid_b[0], centroid_a[1] - centroid_b[1]))

    @staticmethod
    def _detection_centroid(detection: Detection) -> tuple[int, int]:
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
        return CampusSurveillance._bbox_centroid(detection.bbox)

    def _mask_contact_ratio(self, det_a: Detection, det_b: Detection) -> float:
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

    # ------------------------------- Motion Estimation (frame differencing)
    def _estimate_motion_ratio(self, frame: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            return 0.0
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0
        diff = cv2.absdiff(gray, self._prev_gray)
        self._prev_gray = gray
        blurred = cv2.GaussianBlur(diff, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)
        motion_ratio = float(np.count_nonzero(thresh)) / thresh.size
        return motion_ratio

    # ------------------------------------------------------------------ Dashboard
    def get_current_frame(self) -> Optional[np.ndarray]:
        return self._last_frame

    def add_zone(self, name: str, points: List[tuple[int, int]], zone_type: str = "restricted") -> Zone:
        zone = Zone(name=name, points=points, zone_type=zone_type)
        self.zone_manager.add_zone(zone)
        return zone


__all__ = ["CampusSurveillance", "Detection"]


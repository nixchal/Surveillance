"""Wrapper around Meta SAM3 image model for per-person segmentation.

This is intentionally minimal and only exposes what the detector needs:
given a frame and a list of person boxes, return one mask per box.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

try:  # heavy import – keep behind try/except
    import sam3  # type: ignore
    from sam3 import build_sam3_image_model  # type: ignore
    from sam3.model.box_ops import box_xywh_to_cxcywh  # type: ignore
    from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore
    from sam3.visualization_utils import normalize_bbox  # type: ignore
except Exception:  # pragma: no cover - handled gracefully at runtime
    sam3 = None  # type: ignore
    build_sam3_image_model = None  # type: ignore
    box_xywh_to_cxcywh = None  # type: ignore
    Sam3Processor = None  # type: ignore
    normalize_bbox = None  # type: ignore


LOGGER = logging.getLogger(__name__)


class SamPersonSegmenter:
    """High-level helper for SAM3 person segmentation.

    Usage:
        segmenter = SamPersonSegmenter()
        masks = segmenter.segment_persons(frame_bgr, [bbox1, bbox2, ...])
    """

    def __init__(
        self,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        if sam3 is None or build_sam3_image_model is None or Sam3Processor is None:
            raise RuntimeError("sam3 package is not available – did you install it into the venv?")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Locate assets inside the installed sam3 package
        sam3_root = Path(sam3.__file__).resolve().parent.parent
        bpe_path = sam3_root / "assets" / "bpe_simple_vocab_16e6.txt.gz"

        if not bpe_path.exists():
            raise FileNotFoundError(f"SAM3 BPE vocab not found at {bpe_path}")

        LOGGER.info("Loading SAM3 image model from %s", bpe_path)
        model = build_sam3_image_model(bpe_path=str(bpe_path))
        model.to(self.device)
        model.eval()

        self._processor = Sam3Processor(
            model,
            device=self.device,
            confidence_threshold=confidence_threshold,
        )

    # ------------------------------------------------------------------ public API
    def segment_persons(
        self, frame_bgr: np.ndarray, boxes_xyxy: List[np.ndarray]
    ) -> List[Optional[np.ndarray]]:
        """Return one boolean mask per bbox (or None if segmentation failed).

        masks are HxW, aligned with the input frame.
        """
        if not boxes_xyxy:
            return []

        try:
            from PIL import Image
        except ImportError:  # very unlikely in this project
            LOGGER.warning("PIL not available; SAM3 segmentation disabled")
            return [None for _ in boxes_xyxy]

        # Convert frame to RGB PIL image for the processor
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        state = self._processor.set_image(pil_image)
        h, w = frame_bgr.shape[:2]

        masks_out: List[Optional[np.ndarray]] = []

        for bbox in boxes_xyxy:
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox]
                box_xywh = np.array(
                    [[x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]], dtype=np.float32
                )

                # Convert to cxcywh then normalize to [0,1] using helpers from sam3
                box_xywh_t = torch.from_numpy(box_xywh)
                box_cxcywh = box_xywh_to_cxcywh(box_xywh_t)
                norm_box = normalize_bbox(box_cxcywh, img_w=w, img_h=h).flatten().tolist()

                # Reset prompts and add our single positive box
                self._processor.reset_all_prompts(state)
                state = self._processor.add_geometric_prompt(
                    box=norm_box, label=True, state=state
                )

                masks = state.get("masks")
                boxes = state.get("boxes")
                if masks is None or boxes is None or masks.shape[0] == 0:
                    masks_out.append(None)
                    continue

                # Select the mask whose predicted box best matches our input bbox
                boxes_np = boxes.detach().cpu().numpy()
                ious = _bbox_iou_single_vs_many(np.array([x1, y1, x2, y2]), boxes_np)
                best_idx = int(ious.argmax())

                mask_tensor = masks[best_idx]
                mask_np = mask_tensor.detach().cpu().numpy()
                if mask_np.ndim == 3:
                    # [1, H, W] or [C, H, W] – take first channel
                    mask_np = mask_np[0]

                mask_bool = mask_np.astype(bool)
                masks_out.append(mask_bool)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("SAM3 segmentation failed for one bbox: %s", exc)
                masks_out.append(None)

        return masks_out


def _bbox_iou_single_vs_many(a: np.ndarray, bs: np.ndarray) -> np.ndarray:
    """Compute IoU between one box a (xyxy) and many bs (N,4)."""
    ax1, ay1, ax2, ay2 = a
    bx1 = bs[:, 0]
    by1 = bs[:, 1]
    bx2 = bs[:, 2]
    by2 = bs[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = np.maximum(0.0, (bx2 - bx1)) * np.maximum(0.0, (by2 - by1))
    union = area_a + area_b - inter
    union = np.maximum(union, 1e-6)
    return inter / union



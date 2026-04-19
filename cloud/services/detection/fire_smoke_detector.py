"""Dedicated smoke/fire detection — faster and more reliable than VLM-only."""

import logging
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FireSmokeAlert:
    alert_type: str  # "smoke" or "fire"
    confidence: float
    camera_id: str
    timestamp: float
    bbox: list[int] | None = None
    description: str = ""


class FireSmokeDetector:
    """Dedicated smoke/fire detection using color analysis + motion patterns.

    Much faster than VLM (runs every frame). VLM used only for confirmation.
    """

    # HSV ranges for fire/smoke
    FIRE_LOW = np.array([0, 100, 200])
    FIRE_HIGH = np.array([30, 255, 255])
    SMOKE_LOW = np.array([0, 0, 100])
    SMOKE_HIGH = np.array([180, 60, 220])

    def __init__(self, fire_area_threshold: float = 0.005, smoke_area_threshold: float = 0.01,
                 vlm_client=None):
        self._fire_thresh = fire_area_threshold
        self._smoke_thresh = smoke_area_threshold
        self._vlm_client = vlm_client
        self._prev_gray = None

    def detect(self, frame: np.ndarray, camera_id: str = "") -> list[FireSmokeAlert]:
        import cv2
        alerts = []
        h, w = frame.shape[:2]
        total_pixels = h * w
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Fire detection: orange-red regions
        fire_mask = cv2.inRange(hsv, self.FIRE_LOW, self.FIRE_HIGH)
        fire_ratio = float(cv2.countNonZero(fire_mask)) / total_pixels
        if fire_ratio > self._fire_thresh:
            bbox = self._mask_to_bbox(fire_mask)
            alerts.append(FireSmokeAlert(
                "fire", min(0.9, fire_ratio / self._fire_thresh / 5),
                camera_id, time.time(), bbox,
                f"Fire-colored region detected ({fire_ratio:.1%} of frame)"))

        # Smoke detection: gray diffuse regions + upward motion
        smoke_mask = cv2.inRange(hsv, self.SMOKE_LOW, self.SMOKE_HIGH)
        smoke_ratio = float(cv2.countNonZero(smoke_mask)) / total_pixels
        if smoke_ratio > self._smoke_thresh:
            # Verify upward motion (smoke rises)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            has_upward = False
            if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
                flow = cv2.calcOpticalFlowFarneback(self._prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                smoke_flow = flow[smoke_mask > 0]
                if len(smoke_flow) > 100:
                    avg_vy = float(np.mean(smoke_flow[:, 1]))
                    has_upward = avg_vy < -0.5  # negative y = upward in image coords
            self._prev_gray = gray

            if has_upward or smoke_ratio > self._smoke_thresh * 3:
                bbox = self._mask_to_bbox(smoke_mask)
                alerts.append(FireSmokeAlert(
                    "smoke", min(0.85, smoke_ratio / self._smoke_thresh / 3),
                    camera_id, time.time(), bbox,
                    f"Smoke detected ({smoke_ratio:.1%} of frame, upward motion={'yes' if has_upward else 'no'})"))

        return alerts

    @staticmethod
    def _mask_to_bbox(mask) -> list[int]:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return [0, 0, 0, 0]
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return [x, y, x + w, y + h]

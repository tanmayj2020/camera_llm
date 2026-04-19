"""Adversarial Robustness Monitor."""

import logging
import time
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DegradationAlert:
    alert_type: str
    camera_id: str
    confidence: float
    timestamp: float
    description: str
    evidence: dict


class AdversarialMonitor:
    def __init__(self):
        self._camera_metrics: dict[str, deque] = {}
        self._frame_hashes: dict[str, deque] = {}

    def _ensure_camera(self, camera_id: str) -> None:
        if camera_id not in self._camera_metrics:
            self._camera_metrics[camera_id] = deque(maxlen=60)
            self._frame_hashes[camera_id] = deque(maxlen=30)

    def update(self, camera_id: str, detection_count: int, frame_hash: str | None = None, timestamp: float = 0) -> None:
        self._ensure_camera(camera_id)
        self._camera_metrics[camera_id].append(detection_count)
        if frame_hash is not None:
            self._frame_hashes[camera_id].append(frame_hash)

    def detect_accuracy_drop(self, camera_id: str) -> DegradationAlert | None:
        self._ensure_camera(camera_id)
        metrics = self._camera_metrics[camera_id]
        if len(metrics) < 20:
            return None
        overall_avg = sum(metrics) / len(metrics)
        if overall_avg == 0:
            return None
        recent = list(metrics)[-10:]
        recent_avg = sum(recent) / len(recent)
        drop = (overall_avg - recent_avg) / overall_avg
        if drop > 0.4:
            return DegradationAlert(
                alert_type="accuracy_drop", camera_id=camera_id,
                confidence=min(drop, 1.0), timestamp=time.time(),
                description=f"Detection avg dropped {drop:.0%} vs rolling avg",
                evidence={"overall_avg": overall_avg, "recent_avg": recent_avg},
            )
        return None

    def detect_video_loop(self, camera_id: str) -> DegradationAlert | None:
        self._ensure_camera(camera_id)
        hashes = self._frame_hashes[camera_id]
        if len(hashes) < 10:
            return None
        recent = list(hashes)[-10:]
        if len(set(recent)) == 1:
            return DegradationAlert(
                alert_type="video_loop", camera_id=camera_id,
                confidence=1.0, timestamp=time.time(),
                description="Last 10+ frame hashes are identical",
                evidence={"repeated_hash": recent[0], "count": len(recent)},
            )
        return None

    def detect_lens_obstruction(self, camera_id: str) -> DegradationAlert | None:
        self._ensure_camera(camera_id)
        metrics = self._camera_metrics[camera_id]
        if len(metrics) < 12:
            return None
        tail = list(metrics)[-12:]
        if all(c == 0 for c in tail):
            return DegradationAlert(
                alert_type="lens_obstruction", camera_id=camera_id,
                confidence=0.9, timestamp=time.time(),
                description="Zero detections for 12+ consecutive updates",
                evidence={"zero_streak": len(tail)},
            )
        return None

    def evaluate(self, camera_id: str, detection_count: int, frame_hash: str | None = None) -> list[DegradationAlert]:
        self.update(camera_id, detection_count, frame_hash)
        alerts = []
        for check in (self.detect_accuracy_drop, self.detect_video_loop, self.detect_lens_obstruction):
            alert = check(camera_id)
            if alert:
                alerts.append(alert)
        return alerts

    def get_camera_health(self) -> dict[str, dict]:
        health = {}
        for cam_id, metrics in self._camera_metrics.items():
            avg_det = sum(metrics) / len(metrics) if metrics else 0
            hashes = self._frame_hashes.get(cam_id, deque())
            diversity = len(set(hashes)) / len(hashes) if hashes else 1.0
            status = "healthy"
            if avg_det == 0:
                status = "critical"
            elif diversity < 0.2:
                status = "warning"
            health[cam_id] = {"avg_detections": avg_det, "hash_diversity": diversity, "status": status}
        return health

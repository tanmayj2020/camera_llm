"""Predictive Camera Health — predict failures before they happen."""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HealthPrediction:
    camera_id: str
    component: str  # "lens", "ir_led", "network", "storage"
    health_pct: float  # 0-100
    predicted_failure_hours: float | None  # hours until predicted failure
    trend: str  # "stable", "degrading", "critical"
    recommendation: str


class PredictiveCameraHealth:
    """Predicts camera component failures from telemetry trends."""

    WINDOW = 168  # 1 week of hourly samples

    def __init__(self):
        self._metrics: dict[str, dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.WINDOW)))

    def ingest(self, camera_id: str, metrics: dict):
        """Ingest camera telemetry: fps, detection_count, brightness, latency_ms, packet_loss."""
        ts = time.time()
        for key, value in metrics.items():
            self._metrics[camera_id][key].append((ts, float(value)))

    def predict(self, camera_id: str) -> list[HealthPrediction]:
        cam_metrics = self._metrics.get(camera_id, {})
        if not cam_metrics:
            return []

        predictions = []

        # Lens health: brightness trend
        if "brightness" in cam_metrics and len(cam_metrics["brightness"]) >= 24:
            predictions.append(self._predict_component(
                camera_id, "lens", cam_metrics["brightness"],
                low_threshold=30, critical_threshold=15,
                rec_healthy="Lens clean", rec_degrading="Schedule lens cleaning",
                rec_critical="Lens severely degraded — replace or clean immediately"))

        # IR LED: night detection rate trend
        if "detection_count" in cam_metrics and len(cam_metrics["detection_count"]) >= 48:
            predictions.append(self._predict_component(
                camera_id, "ir_led", cam_metrics["detection_count"],
                low_threshold=5, critical_threshold=1,
                rec_healthy="IR LEDs functioning", rec_degrading="IR LED output declining — plan replacement",
                rec_critical="IR LEDs likely failed — night vision compromised"))

        # Network: packet loss / latency trend
        if "latency_ms" in cam_metrics and len(cam_metrics["latency_ms"]) >= 24:
            predictions.append(self._predict_component(
                camera_id, "network", cam_metrics["latency_ms"],
                low_threshold=500, critical_threshold=2000, higher_is_worse=True,
                rec_healthy="Network stable", rec_degrading="Latency increasing — check network",
                rec_critical="Network critically degraded — check cable/switch"))

        # FPS health
        if "fps" in cam_metrics and len(cam_metrics["fps"]) >= 24:
            predictions.append(self._predict_component(
                camera_id, "processing", cam_metrics["fps"],
                low_threshold=5, critical_threshold=1,
                rec_healthy="Processing healthy", rec_degrading="FPS declining — check CPU/GPU load",
                rec_critical="Processing critically slow — restart or upgrade hardware"))

        return [p for p in predictions if p is not None]

    def _predict_component(self, camera_id: str, component: str, data: deque,
                           low_threshold: float, critical_threshold: float,
                           higher_is_worse: bool = False,
                           rec_healthy: str = "", rec_degrading: str = "",
                           rec_critical: str = "") -> HealthPrediction | None:
        values = np.array([v for _, v in data])
        if len(values) < 10:
            return None

        current = float(values[-1])
        recent_avg = float(values[-min(24, len(values)):].mean())
        overall_avg = float(values.mean())

        # Linear trend
        x = np.arange(len(values), dtype=float)
        slope = float(np.polyfit(x, values, 1)[0])

        if higher_is_worse:
            health = max(0, min(100, 100 - (current / low_threshold * 100)))
            trend_bad = slope > 0
        else:
            health = max(0, min(100, current / max(overall_avg, 1) * 100))
            trend_bad = slope < 0

        # Predict time to failure
        failure_hours = None
        if trend_bad and abs(slope) > 1e-6:
            if higher_is_worse:
                hours_to_critical = (critical_threshold - current) / (slope * 3600) if slope > 0 else None
            else:
                hours_to_critical = (current - critical_threshold) / (-slope * 3600) if slope < 0 else None
            if hours_to_critical and hours_to_critical > 0:
                failure_hours = round(hours_to_critical, 1)

        if health > 70:
            trend = "stable"
            rec = rec_healthy
        elif health > 30:
            trend = "degrading"
            rec = rec_degrading
        else:
            trend = "critical"
            rec = rec_critical

        return HealthPrediction(camera_id, component, round(health, 1), failure_hours, trend, rec)

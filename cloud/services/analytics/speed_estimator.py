"""Speed estimation and alerts — vehicle/person speed from spatial trajectories."""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpeedAlert:
    entity_id: str
    class_name: str
    speed_ms: float
    speed_kmh: float
    camera_id: str
    timestamp: float
    zone_id: str = ""


class SpeedEstimator:
    """Estimates entity speed from spatial memory velocity vectors."""

    def __init__(self, person_threshold_ms: float = 3.0, vehicle_threshold_ms: float = 8.3):
        self.person_threshold = person_threshold_ms  # ~10.8 km/h (running)
        self.vehicle_threshold = vehicle_threshold_ms  # ~30 km/h
        self._history: dict[str, deque] = defaultdict(lambda: deque(maxlen=30))

    def update(self, entity_id: str, speed_ms: float, timestamp: float):
        self._history[entity_id].append((timestamp, speed_ms))

    def evaluate(self, spatial, camera_id: str = "", timestamp: float = 0) -> list[SpeedAlert]:
        ts = timestamp or time.time()
        alerts = []
        for eid, ent in getattr(spatial, "_entities", {}).items():
            speed = float(np.linalg.norm(ent.velocity))
            self.update(eid, speed, ts)
            threshold = self.vehicle_threshold if ent.class_name in ("car", "truck", "bus", "vehicle") else self.person_threshold
            if speed > threshold:
                alerts.append(SpeedAlert(
                    entity_id=eid, class_name=ent.class_name,
                    speed_ms=round(speed, 2), speed_kmh=round(speed * 3.6, 1),
                    camera_id=camera_id, timestamp=ts))
        return alerts

    def get_speed(self, entity_id: str) -> dict | None:
        hist = self._history.get(entity_id)
        if not hist:
            return None
        speeds = [s for _, s in hist]
        return {
            "entity_id": entity_id,
            "current_ms": round(speeds[-1], 2),
            "current_kmh": round(speeds[-1] * 3.6, 1),
            "avg_ms": round(sum(speeds) / len(speeds), 2),
            "max_ms": round(max(speeds), 2),
            "max_kmh": round(max(speeds) * 3.6, 1),
        }

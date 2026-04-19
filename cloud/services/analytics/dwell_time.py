"""Dwell time analytics per zone."""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DwellRecord:
    entity_id: str
    zone_id: str
    entered_at: float
    exited_at: float = 0.0
    duration_s: float = 0.0


class DwellTimeAnalyzer:
    def __init__(self, spatial=None):
        self._spatial = spatial
        self._active: dict[tuple[str, str], float] = {}  # (entity_id, zone_id) -> entered_at
        self._completed: dict[str, list[DwellRecord]] = defaultdict(list)  # zone_id -> records
        self._entity_journey: dict[str, list[DwellRecord]] = defaultdict(list)

    def update(self, entity_id: str, zone_id: str, timestamp: float):
        key = (entity_id, zone_id)
        if key not in self._active:
            self._active[key] = timestamp

    def exit(self, entity_id: str, zone_id: str, timestamp: float):
        key = (entity_id, zone_id)
        entered = self._active.pop(key, None)
        if entered:
            rec = DwellRecord(entity_id, zone_id, entered, timestamp, timestamp - entered)
            self._completed[zone_id].append(rec)
            self._entity_journey[entity_id].append(rec)
            # Keep bounded
            if len(self._completed[zone_id]) > 5000:
                self._completed[zone_id] = self._completed[zone_id][-5000:]

    def get_zone_analytics(self, zone_id: str) -> dict:
        records = self._completed.get(zone_id, [])
        active_count = sum(1 for (_, z) in self._active if z == zone_id)
        if not records:
            return {"avg_dwell_s": 0, "median_s": 0, "p95_s": 0,
                    "current_count": active_count, "anomalous": []}
        durations = [r.duration_s for r in records]
        arr = np.array(durations)
        mean, std = float(arr.mean()), float(arr.std())
        anomalous = [{"entity_id": r.entity_id, "duration_s": r.duration_s}
                     for r in records[-50:] if std > 0 and r.duration_s > mean + 2 * std]
        return {
            "avg_dwell_s": round(mean, 1),
            "median_s": round(float(np.median(arr)), 1),
            "p95_s": round(float(np.percentile(arr, 95)), 1),
            "current_count": active_count,
            "anomalous": anomalous,
        }

    def get_entity_journey(self, entity_id: str) -> list[dict]:
        return [{"zone_id": r.zone_id, "entered_at": r.entered_at,
                 "exited_at": r.exited_at, "duration_s": round(r.duration_s, 1)}
                for r in self._entity_journey.get(entity_id, [])]

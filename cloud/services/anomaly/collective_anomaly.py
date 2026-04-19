"""Collective anomaly detection across multiple events."""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CollectiveAnomaly:
    anomaly_type: str
    description: str
    severity: str
    involved_events: List[dict]
    confidence: float


class CollectiveAnomalyDetector:
    def __init__(self, window_size: int = 100) -> None:
        self._recent_events: deque = deque(maxlen=window_size)
        self._patterns: Dict[str, dict] = {}

    def ingest(self, event_dict: dict) -> None:
        self._recent_events.append(event_dict)
        logger.debug("Ingested event, window size=%d", len(self._recent_events))

    def _filter(self, camera_id: Optional[str]) -> List[dict]:
        if camera_id is None:
            return list(self._recent_events)
        return [e for e in self._recent_events if e.get("camera_id") == camera_id]

    def detect(self, camera_id: Optional[str] = None) -> List[CollectiveAnomaly]:
        events = self._filter(camera_id)
        if not events:
            return []
        anomalies: List[CollectiveAnomaly] = []

        # a) coordinated_movement: multiple track_ids toward same zone within 60s
        zone_groups: Dict[str, List[dict]] = {}
        for e in events:
            z = e.get("target_zone")
            if z and e.get("track_id"):
                zone_groups.setdefault(z, []).append(e)
        for zone, grp in zone_groups.items():
            ids = {e["track_id"] for e in grp}
            if len(ids) >= 2:
                ts = [e.get("timestamp", 0) for e in grp]
                if ts and (max(ts) - min(ts)) <= 60:
                    anomalies.append(CollectiveAnomaly(
                        "coordinated_movement",
                        f"{len(ids)} entities moving toward zone '{zone}' within 60s",
                        "high", grp, min(len(ids) / 5.0, 1.0),
                    ))

        # b) systematic_coverage: entities visiting all exit zones within 30min
        exit_events = [e for e in events if "exit" in str(e.get("zone", "")).lower()]
        if exit_events:
            exit_zones = {e["zone"] for e in exit_events}
            ts = [e.get("timestamp", 0) for e in exit_events]
            if len(exit_zones) >= 2 and ts and (max(ts) - min(ts)) <= 1800:
                anomalies.append(CollectiveAnomaly(
                    "systematic_coverage",
                    f"All {len(exit_zones)} exit zones visited within 30min",
                    "critical", exit_events, min(len(exit_zones) / 4.0, 1.0),
                ))

        # c) distraction_pattern: high-severity + unusual activity on different camera within 120s
        high = [e for e in events if e.get("severity") in ("high", "critical")]
        for h in high:
            ht = h.get("timestamp", 0)
            others = [
                e for e in events
                if e.get("camera_id") != h.get("camera_id")
                and abs(e.get("timestamp", 0) - ht) <= 120
                and e.get("anomaly_score", 0) > 0.5
            ]
            if others:
                anomalies.append(CollectiveAnomaly(
                    "distraction_pattern",
                    "High-severity event concurrent with unusual activity on another camera",
                    "critical", [h] + others, 0.8,
                ))
                break

        # d) temporal_clustering: >3x normal rate in 5min window
        if len(events) >= 2:
            ts_sorted = sorted(e.get("timestamp", 0) for e in events)
            span = ts_sorted[-1] - ts_sorted[0]
            if span > 0:
                overall_rate = len(events) / span
                window = 300
                for i, t in enumerate(ts_sorted):
                    in_window = [t2 for t2 in ts_sorted if 0 <= t2 - t <= window]
                    window_rate = len(in_window) / window
                    if window_rate > 3 * overall_rate and len(in_window) > 3:
                        involved = [e for e in events if t <= e.get("timestamp", 0) <= t + window]
                        anomalies.append(CollectiveAnomaly(
                            "temporal_clustering",
                            f"Event rate {window_rate:.2f}/s vs normal {overall_rate:.2f}/s",
                            "high", involved, min(window_rate / (3 * overall_rate + 1e-9), 1.0),
                        ))
                        break

        logger.debug("Detected %d collective anomalies", len(anomalies))
        return anomalies

    def get_recent_window(self) -> List[dict]:
        return list(self._recent_events)

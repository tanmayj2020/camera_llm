"""Multi-Camera Entity Handoff Visualization."""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class HandoffPoint:
    from_camera: str
    to_camera: str
    entity_id: str
    timestamp: float
    position: list[float]


@dataclass
class EntityJourney:
    entity_id: str
    start_time: float
    end_time: float
    path: list[dict]
    handoffs: list[HandoffPoint]
    total_cameras: int
    total_duration_s: float


class JourneyVisualizer:
    def __init__(self, spatial=None, kg=None) -> None:
        self._spatial = spatial
        self._kg = kg
        self._entity_tracks: dict[str, list[dict]] = defaultdict(list)
        self._handoffs: dict[str, list[HandoffPoint]] = defaultdict(list)

    def record_sighting(self, entity_id: str, camera_id: str, position: list[float], timestamp: float) -> None:
        tracks = self._entity_tracks[entity_id]
        if tracks and tracks[-1]["camera_id"] != camera_id:
            self._handoffs[entity_id].append(HandoffPoint(
                from_camera=tracks[-1]["camera_id"],
                to_camera=camera_id,
                entity_id=entity_id,
                timestamp=timestamp,
                position=position,
            ))
        tracks.append({"camera_id": camera_id, "timestamp": timestamp, "position": position})

    def build_journey(self, entity_id: str, since_hours: float = 24) -> EntityJourney:
        cutoff = time.time() - since_hours * 3600
        tracks = [t for t in self._entity_tracks.get(entity_id, []) if t["timestamp"] >= cutoff]
        handoffs = [h for h in self._handoffs.get(entity_id, []) if h.timestamp >= cutoff]
        start = tracks[0]["timestamp"] if tracks else 0.0
        end = tracks[-1]["timestamp"] if tracks else 0.0
        cameras = {t["camera_id"] for t in tracks}
        return EntityJourney(
            entity_id=entity_id,
            start_time=start,
            end_time=end,
            path=tracks,
            handoffs=handoffs,
            total_cameras=len(cameras),
            total_duration_s=end - start,
        )

    def get_floor_plan_path(self, entity_id: str) -> list[dict]:
        return [
            {"x": t["position"][0], "y": t["position"][1], "camera_id": t["camera_id"], "timestamp": t["timestamp"]}
            for t in self._entity_tracks.get(entity_id, [])
            if len(t["position"]) >= 2
        ]

    def get_active_journeys(self) -> list[dict]:
        cutoff = time.time() - 60
        result = []
        for eid, tracks in self._entity_tracks.items():
            if tracks and tracks[-1]["timestamp"] >= cutoff:
                cameras = {t["camera_id"] for t in tracks}
                result.append({
                    "entity_id": eid,
                    "camera_count": len(cameras),
                    "last_camera": tracks[-1]["camera_id"],
                    "last_seen": tracks[-1]["timestamp"],
                    "handoff_count": len(self._handoffs.get(eid, [])),
                })
        return result

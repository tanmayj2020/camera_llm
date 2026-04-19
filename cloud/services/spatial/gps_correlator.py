"""GPS-CCTV Correlation — match CCTV detections with GPS tracks."""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GPSCorrelation:
    entity_id: str
    gps_identity: str
    camera_id: str
    confidence: float
    distance_m: float
    timestamp: float


class GPSCorrelator:
    """Correlates CCTV spatial detections with external GPS data feeds."""

    def __init__(self, max_distance_m: float = 10.0):
        self._max_distance = max_distance_m
        self._gps_tracks: dict[str, dict] = {}  # gps_id -> {lat, lon, x, y, meta}
        self._camera_origins: dict[str, tuple[float, float]] = {}  # camera_id -> (x, y) site coords

    def register_camera_origin(self, camera_id: str, x: float, y: float):
        self._camera_origins[camera_id] = (x, y)

    def update_gps(self, gps_id: str, x: float, y: float, meta: dict = None):
        self._gps_tracks[gps_id] = {"x": x, "y": y, "meta": meta or {}, "t": time.time()}

    def correlate(self, spatial, camera_id: str = "") -> list[GPSCorrelation]:
        if not self._gps_tracks:
            return []
        results = []
        cam_origin = np.array(self._camera_origins.get(camera_id, (0, 0)))

        for eid, ent in getattr(spatial, "_entities", {}).items():
            ent_pos = np.array([ent.position[0], ent.position[2] if len(ent.position) > 2 else ent.position[1]])
            world_pos = ent_pos + cam_origin

            for gps_id, gps in self._gps_tracks.items():
                if time.time() - gps["t"] > 60:
                    continue
                gps_pos = np.array([gps["x"], gps["y"]])
                dist = float(np.linalg.norm(world_pos - gps_pos))
                if dist < self._max_distance:
                    results.append(GPSCorrelation(
                        eid, gps_id, camera_id,
                        round(max(0, 1 - dist / self._max_distance), 2),
                        round(dist, 2), time.time()))
        return results

    def cleanup_stale(self, max_age_s: float = 300):
        now = time.time()
        self._gps_tracks = {k: v for k, v in self._gps_tracks.items() if now - v["t"] < max_age_s}

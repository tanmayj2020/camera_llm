"""Autonomous PTZ Tracking — camera follows person of interest across FOV."""

import logging
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PTZCommand:
    camera_id: str
    pan_deg: float
    tilt_deg: float
    zoom: float
    target_entity: str
    timestamp: float


class AutonomousPTZTracker:
    """Generates PTZ commands to track a person of interest across camera FOV."""

    def __init__(self, image_width: int = 1920, image_height: int = 1080,
                 fov_h: float = 60.0, fov_v: float = 34.0):
        self._img_w = image_width
        self._img_h = image_height
        self._fov_h = fov_h
        self._fov_v = fov_v
        self._tracking: dict[str, str] = {}  # camera_id -> entity_id
        self._current_ptz: dict[str, dict] = {}  # camera_id -> {pan, tilt, zoom}

    def start_tracking(self, camera_id: str, entity_id: str):
        self._tracking[camera_id] = entity_id
        logger.info("PTZ tracking started: %s -> %s", camera_id, entity_id)

    def stop_tracking(self, camera_id: str):
        self._tracking.pop(camera_id, None)

    def update(self, camera_id: str, spatial) -> PTZCommand | None:
        entity_id = self._tracking.get(camera_id)
        if not entity_id:
            return None

        ent = getattr(spatial, "_entities", {}).get(entity_id)
        if not ent:
            return None

        bbox = ent._kf_state[:3] if hasattr(ent, '_kf_state') and ent._kf_state is not None else ent.position
        # Convert 3D position to pan/tilt angles
        cx = float(bbox[0])
        cy = float(bbox[1]) if len(bbox) > 1 else 0
        cz = float(bbox[2]) if len(bbox) > 2 else 1

        pan = float(np.degrees(np.arctan2(cx, max(cz, 0.1))))
        tilt = float(np.degrees(np.arctan2(-cy, max(cz, 0.1))))

        # Zoom based on distance
        distance = float(np.linalg.norm(ent.position))
        zoom = min(10.0, max(1.0, distance / 5.0))

        # Smooth: only send command if significant change
        prev = self._current_ptz.get(camera_id, {"pan": 0, "tilt": 0, "zoom": 1})
        if abs(pan - prev["pan"]) < 1.0 and abs(tilt - prev["tilt"]) < 1.0:
            return None

        self._current_ptz[camera_id] = {"pan": pan, "tilt": tilt, "zoom": zoom}
        return PTZCommand(camera_id, round(pan, 1), round(tilt, 1), round(zoom, 1),
                          entity_id, time.time())

    def get_handoff_camera(self, entity_id: str, camera_topology: dict) -> str | None:
        """When entity leaves FOV, suggest next camera for handoff."""
        for cam_id, topo in camera_topology.items():
            if cam_id in self._tracking:
                continue
            if entity_id in topo.get("visible_entities", []):
                return cam_id
        return None

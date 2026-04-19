"""Digital Human Guard — AI avatar visible deterrent on screens near cameras."""

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GuardState:
    camera_id: str
    active: bool
    looking_at: str  # entity_id or ""
    message: str
    posture: str  # "standing", "alert", "approaching"
    eye_contact: bool


class DigitalGuard:
    """AI avatar that appears on screens, makes eye contact, acts as visible deterrent.

    Generates state updates for frontend rendering (avatar position, gaze, message).
    """

    def __init__(self):
        self._active_guards: dict[str, GuardState] = {}

    def activate(self, camera_id: str):
        self._active_guards[camera_id] = GuardState(
            camera_id, True, "", "Monitoring area.", "standing", False)

    def deactivate(self, camera_id: str):
        self._active_guards.pop(camera_id, None)

    def update(self, camera_id: str, spatial, scene_state: dict = None) -> GuardState | None:
        guard = self._active_guards.get(camera_id)
        if not guard or not guard.active:
            return None

        entities = getattr(spatial, "_entities", {})
        persons = {eid: e for eid, e in entities.items() if e.class_name == "person"}

        if not persons:
            guard.looking_at = ""
            guard.message = "Area clear. Monitoring."
            guard.posture = "standing"
            guard.eye_contact = False
            return guard

        # Find closest person
        closest_id, closest_dist = "", float("inf")
        for eid, ent in persons.items():
            import numpy as np
            dist = float(np.linalg.norm(ent.position))
            if dist < closest_dist:
                closest_dist = dist
                closest_id = eid

        guard.looking_at = closest_id
        guard.eye_contact = closest_dist < 5.0

        # Escalate posture based on anomaly
        anomaly_score = (scene_state or {}).get("anomaly_score", {})
        overall = anomaly_score.get("overall", 0) if isinstance(anomaly_score, dict) else 0

        if overall > 0.7:
            guard.posture = "approaching"
            guard.message = "I see you. This area is under active surveillance."
        elif overall > 0.3 or closest_dist < 3.0:
            guard.posture = "alert"
            guard.message = "Attention: you are being monitored."
        else:
            guard.posture = "standing"
            guard.message = "Area monitored 24/7."

        return guard

    def get_all_states(self) -> list[dict]:
        return [{"camera_id": g.camera_id, "active": g.active, "looking_at": g.looking_at,
                 "message": g.message, "posture": g.posture, "eye_contact": g.eye_contact}
                for g in self._active_guards.values()]

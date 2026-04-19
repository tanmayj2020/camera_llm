"""Environmental hazard detection via VLM."""

import json
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentAlert:
    alert_type: str  # smoke, fire, flood, abandoned_object
    confidence: float
    camera_id: str
    timestamp: float
    description: str


class EnvironmentDetector:
    def __init__(self, vlm_client=None):
        self._vlm_client = vlm_client

    def _get_client(self):
        if self._vlm_client is None:
            return "stub"
        return self._vlm_client._get_client()

    def check_environment(self, keyframe_b64: str, camera_id: str, spatial=None) -> list[EnvironmentAlert]:
        alerts = []
        # VLM-based detection
        if keyframe_b64:
            try:
                client = self._get_client()
                if client != "stub":
                    import base64
                    prompt = ("Check this image for: smoke/haze, fire/flames, water on floor, "
                              "abandoned packages. Reply JSON list: [{\"type\": str, \"confidence\": float, "
                              "\"description\": str}]. Empty list if none found.")
                    resp = client.generate_content([prompt, {"mime_type": "image/jpeg",
                                                             "data": keyframe_b64}])
                    text = resp.text.strip()
                    if text.startswith("```"):
                        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
                    for item in json.loads(text):
                        alerts.append(EnvironmentAlert(
                            alert_type=item["type"], confidence=item.get("confidence", 0.5),
                            camera_id=camera_id, timestamp=time.time(),
                            description=item.get("description", item["type"]),
                        ))
            except Exception as e:
                logger.debug("VLM environment check failed: %s", e)

        # Abandoned object detection from spatial
        if spatial:
            now = time.time()
            for eid, ent in list(getattr(spatial, "_entities", {}).items()):
                if ent.class_name in ("backpack", "suitcase", "bag", "box", "package"):
                    vel_mag = float((ent.velocity ** 2).sum() ** 0.5) if hasattr(ent, "velocity") else 0
                    if vel_mag < 0.05 and (now - ent.last_seen) < 10:
                        idle_s = now - ent.last_seen
                        # Check if stationary > 5 min by checking velocity history
                        if hasattr(ent, "_kf_state") and ent._kf_state is not None:
                            speed = float((ent._kf_state[3:6] ** 2).sum() ** 0.5)
                            if speed < 0.05:
                                # Check no person nearby
                                has_person = any(
                                    p.class_name == "person" and
                                    float(((p.position - ent.position) ** 2).sum() ** 0.5) < 3.0
                                    for pid, p in spatial._entities.items() if pid != eid
                                )
                                if not has_person:
                                    alerts.append(EnvironmentAlert(
                                        alert_type="abandoned_object",
                                        confidence=0.7, camera_id=camera_id,
                                        timestamp=now,
                                        description=f"Abandoned {ent.class_name} ({eid}) with no person nearby",
                                    ))
        return alerts

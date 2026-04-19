"""Visual grounding for anomaly explanations — points to WHERE in the image."""

import json
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GroundedExplanation:
    description: str
    bounding_boxes: list[dict]
    camera_id: str
    timestamp: float


class VisualGrounder:
    def __init__(self, vlm_client=None, spatial=None):
        self._vlm_client = vlm_client
        self._spatial = spatial

    def _get_client(self):
        if self._vlm_client is None:
            return "stub"
        return self._vlm_client._get_client()

    def _position_to_bbox(self, pos) -> list[int]:
        x, y = int(pos[0] * 20), int(pos[1] * 20)
        return [max(0, x - 50), max(0, y - 50), x + 50, y + 50]

    def ground_anomaly(self, anomaly_dict, keyframe_b64=None, spatial=None) -> GroundedExplanation:
        spatial = spatial or self._spatial
        camera_id = anomaly_dict.get("camera_id", "unknown")
        desc = anomaly_dict.get("description", anomaly_dict.get("anomaly_type", ""))
        if keyframe_b64:
            client = self._get_client()
            if client != "stub":
                try:
                    prompt = (f'Identify and locate the anomaly in this image. Anomaly: {desc}. '
                              'For each relevant object return JSON list: [{"label": str, "bbox": [x1,y1,x2,y2], "confidence": float}]')
                    resp = client.generate_content([prompt, {"mime_type": "image/jpeg", "data": keyframe_b64}])
                    text = resp.text.strip()
                    if text.startswith("```"):
                        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
                    return GroundedExplanation(desc, json.loads(text), camera_id, time.time())
                except Exception as e:
                    logger.debug("VLM grounding failed: %s", e)
        boxes = []
        if spatial:
            for eid, ent in getattr(spatial, "_entities", {}).items():
                boxes.append({"label": f"{ent.class_name}:{eid}",
                              "bbox": self._position_to_bbox(ent.position), "confidence": 0.5})
        return GroundedExplanation(desc, boxes[:10], camera_id, time.time())

    def ground_entities(self, spatial=None, camera_id="") -> list[dict]:
        spatial = spatial or self._spatial
        if not spatial:
            return []
        return [{"entity_id": eid, "class": ent.class_name,
                 "bbox": self._position_to_bbox(ent.position)}
                for eid, ent in getattr(spatial, "_entities", {}).items()]

"""Video Prediction — predict next scene state using world model + VLM.

Given current scene, predict what happens next as a structured description.
Used for simulation, training data generation, and "what would happen if..." queries.
"""

import json
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ScenePredictionResult:
    camera_id: str
    current_description: str
    predicted_description: str
    predicted_entities: list[dict] = field(default_factory=list)
    predicted_events: list[str] = field(default_factory=list)
    confidence: float = 0.0
    horizon_s: float = 10.0


class VideoPredictionEngine:
    """Predicts future scene state from current observations + world model."""

    def __init__(self, world_model=None, spatial=None, vlm_client=None):
        self._world_model = world_model
        self._spatial = spatial
        self._vlm_client = vlm_client

    def predict_next(self, camera_id: str, current_event: dict,
                     horizon_s: float = 10.0) -> ScenePredictionResult:
        result = ScenePredictionResult(camera_id=camera_id, horizon_s=horizon_s,
                                       current_description=f"{len(current_event.get('objects', []))} objects detected")

        # Get trajectory predictions from world model
        if self._world_model:
            for obj in current_event.get("objects", []):
                tid = str(obj.get("track_id", ""))
                pred = self._world_model.predict_trajectory(tid, horizon_s)
                if pred:
                    result.predicted_entities.append({
                        "track_id": tid, "class": obj.get("class_name", ""),
                        "predicted_position": pred.description})

        # VLM-based scene prediction
        if self._vlm_client and hasattr(self._vlm_client, '_get_client'):
            client = self._vlm_client._get_client()
            if client != "stub":
                try:
                    objects_desc = json.dumps(current_event.get("objects", [])[:10], default=str)
                    prompt = (
                        f"Given this CCTV scene with {len(current_event.get('objects', []))} objects:\n"
                        f"{objects_desc}\n"
                        f"Predict what will happen in the next {horizon_s} seconds.\n"
                        f"Reply JSON: {{\"description\": \"...\", \"predicted_events\": [\"...\"], \"confidence\": 0.0-1.0}}")
                    content = [prompt]
                    if current_event.get("keyframe_b64"):
                        content.append({"mime_type": "image/jpeg", "data": current_event["keyframe_b64"]})
                    resp = client.generate_content(content).text.strip()
                    if resp.startswith("```"):
                        resp = resp.split("\n", 1)[1].rsplit("```", 1)[0]
                    data = json.loads(resp)
                    result.predicted_description = data.get("description", "")
                    result.predicted_events = data.get("predicted_events", [])
                    result.confidence = data.get("confidence", 0.5)
                except Exception as e:
                    logger.debug("VLM prediction failed: %s", e)

        if not result.predicted_description:
            n = len(result.predicted_entities)
            result.predicted_description = f"{n} entities tracked with trajectory predictions"
            result.confidence = 0.3

        return result

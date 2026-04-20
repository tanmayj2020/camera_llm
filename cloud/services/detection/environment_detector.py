"""Environmental hazard + condition detection via VLM with algorithmic fallback.

When VLM is unavailable, uses image histogram analysis (brightness, contrast,
color-channel ratios) to classify: lighting quality, fog/haze, possible fire.
"""

import json
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentAlert:
    alert_type: str  # smoke, fire, flood, abandoned_object, low_light, fog
    confidence: float
    camera_id: str
    timestamp: float
    description: str


@dataclass
class EnvironmentCondition:
    lighting: str  # "good", "low", "very_low", "overexposed"
    brightness: float  # 0-255 mean
    contrast: float  # std dev
    weather_hint: str  # "clear", "fog", "rain_possible"
    fire_risk: float  # 0-1


class EnvironmentDetector:
    def __init__(self, vlm_client=None):
        self._vlm_client = vlm_client

    def _get_client(self):
        if self._vlm_client is None:
            return "stub"
        return self._vlm_client._get_client()

    def _analyze_histogram(self, keyframe_b64: str) -> EnvironmentCondition | None:
        """Algorithmic environment analysis from image brightness/color histograms."""
        try:
            import base64
            import numpy as np

            img_bytes = base64.b64decode(keyframe_b64)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            # Decode JPEG
            try:
                import cv2
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    return None
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except ImportError:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                arr_rgb = np.array(img)
                gray = np.mean(arr_rgb, axis=2).astype(np.uint8)
                img = arr_rgb

            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))

            # Lighting classification
            if brightness < 40:
                lighting = "very_low"
            elif brightness < 80:
                lighting = "low"
            elif brightness > 220:
                lighting = "overexposed"
            else:
                lighting = "good"

            # Fog/haze detection: low contrast + medium brightness
            weather = "clear"
            if contrast < 30 and 60 < brightness < 200:
                weather = "fog"
            elif contrast < 40 and brightness < 100:
                weather = "rain_possible"

            # Fire risk: high red-channel intensity relative to blue/green
            fire_risk = 0.0
            if hasattr(img, 'shape') and len(img.shape) == 3:
                if img.shape[2] >= 3:
                    # OpenCV is BGR, PIL-converted is RGB
                    r = float(np.mean(img[:, :, 2]))  # Red channel (BGR)
                    g = float(np.mean(img[:, :, 1]))
                    b = float(np.mean(img[:, :, 0]))
                    if r > 150 and r > g * 1.5 and r > b * 2.0:
                        fire_risk = min(1.0, (r - max(g, b)) / 128.0)

            return EnvironmentCondition(
                lighting=lighting, brightness=brightness,
                contrast=contrast, weather_hint=weather, fire_risk=fire_risk,
            )
        except Exception as e:
            logger.debug("Histogram analysis failed: %s", e)
            return None

    def check_environment(self, keyframe_b64: str, camera_id: str, spatial=None) -> list[EnvironmentAlert]:
        alerts = []
        now = time.time()

        # VLM-based detection (preferred)
        vlm_tried = False
        if keyframe_b64:
            try:
                client = self._get_client()
                if client != "stub":
                    vlm_tried = True
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
                            camera_id=camera_id, timestamp=now,
                            description=item.get("description", item["type"]),
                        ))
            except Exception as e:
                logger.debug("VLM environment check failed: %s", e)

        # Algorithmic fallback when VLM unavailable
        if not vlm_tried and keyframe_b64:
            cond = self._analyze_histogram(keyframe_b64)
            if cond:
                if cond.lighting == "very_low":
                    alerts.append(EnvironmentAlert(
                        alert_type="low_light", confidence=0.8, camera_id=camera_id,
                        timestamp=now,
                        description=f"Very low lighting (brightness={cond.brightness:.0f})",
                    ))
                elif cond.lighting == "low":
                    alerts.append(EnvironmentAlert(
                        alert_type="low_light", confidence=0.5, camera_id=camera_id,
                        timestamp=now,
                        description=f"Low lighting conditions (brightness={cond.brightness:.0f})",
                    ))
                if cond.weather_hint == "fog":
                    alerts.append(EnvironmentAlert(
                        alert_type="fog", confidence=0.6, camera_id=camera_id,
                        timestamp=now,
                        description=f"Possible fog/haze (contrast={cond.contrast:.0f})",
                    ))
                if cond.fire_risk > 0.3:
                    alerts.append(EnvironmentAlert(
                        alert_type="fire", confidence=cond.fire_risk, camera_id=camera_id,
                        timestamp=now,
                        description=f"Possible fire detected from color analysis (risk={cond.fire_risk:.0%})",
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

"""License plate recognition via VLM."""

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LPRResult:
    plate_text: str
    confidence: float
    vehicle_color: str
    vehicle_type: str
    camera_id: str
    timestamp: float
    track_id: str


@dataclass
class VehicleFingerprint:
    plate_text: str
    color: str
    vehicle_type: str
    first_seen: float
    last_seen: float
    cameras: list[str] = field(default_factory=list)


class LPREngine:
    def __init__(self, vlm_client=None):
        self._vlm_client = vlm_client
        self.watchlist: dict[str, dict] = {}
        self.log: deque[LPRResult] = deque(maxlen=10000)
        self.vehicles: dict[str, VehicleFingerprint] = {}

    def _get_client(self):
        return self._vlm_client

    def recognize(self, bbox_crop_b64: str, camera_id: str, track_id: str,
                  timestamp: float | None = None) -> LPRResult:
        ts = timestamp or time.time()
        plate, color, vtype, conf = "UNKNOWN", "unknown", "unknown", 0.0

        client = self._get_client()
        if client:
            try:
                prompt = ("Extract the license plate text and vehicle color/type from this image. "
                          'Reply JSON: {"plate", "color", "type"}')
                resp = client.generate(prompt, image_b64=bbox_crop_b64)
                data = json.loads(resp if isinstance(resp, str) else resp.text)
                plate = data.get("plate", plate)
                color = data.get("color", color)
                vtype = data.get("type", vtype)
                conf = 0.85
            except Exception:
                logger.warning("VLM LPR failed, using stub fallback")

        result = LPRResult(plate_text=plate, confidence=conf, vehicle_color=color,
                           vehicle_type=vtype, camera_id=camera_id, timestamp=ts,
                           track_id=track_id)
        self.log.append(result)
        self._update_fingerprint(result)

        if plate != "UNKNOWN":
            hit = self.check_watchlist(plate)
            if hit:
                logger.warning("WATCHLIST HIT: %s — %s", plate, hit.get("reason"))

        return result

    def _update_fingerprint(self, r: LPRResult) -> None:
        if r.plate_text == "UNKNOWN":
            return
        fp = self.vehicles.get(r.plate_text)
        if fp:
            fp.last_seen = r.timestamp
            if r.camera_id not in fp.cameras:
                fp.cameras.append(r.camera_id)
        else:
            self.vehicles[r.plate_text] = VehicleFingerprint(
                plate_text=r.plate_text, color=r.vehicle_color, vehicle_type=r.vehicle_type,
                first_seen=r.timestamp, last_seen=r.timestamp, cameras=[r.camera_id],
            )

    def add_to_watchlist(self, plate: str, reason: str, severity: str) -> None:
        self.watchlist[plate] = {"reason": reason, "severity": severity, "added_at": time.time()}

    def remove_from_watchlist(self, plate: str) -> None:
        self.watchlist.pop(plate, None)

    def check_watchlist(self, plate: str) -> dict | None:
        return self.watchlist.get(plate)

    def get_log(self, camera_id: str | None = None, limit: int = 50) -> list[LPRResult]:
        entries = self.log if camera_id is None else [r for r in self.log if r.camera_id == camera_id]
        return list(entries)[-limit:]

    def get_vehicle(self, plate: str) -> VehicleFingerprint | None:
        return self.vehicles.get(plate)

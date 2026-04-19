"""Real-time occupancy analytics."""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class OccupancySnapshot:
    camera_id: str
    timestamp: datetime
    person_count: int
    zone_counts: dict[str, int] = field(default_factory=dict)
    capacity_alerts: list[dict] = field(default_factory=list)


class OccupancyCounter:
    def __init__(self, spatial=None, vlm_client=None):
        self._spatial = spatial
        self._vlm_client = vlm_client
        self.hourly_history: dict[str, deque] = {}
        self.zone_capacities: dict[str, int] = {}
        self._latest: dict[str, OccupancySnapshot] = {}

    def _get_client(self):
        return self._vlm_client

    def set_zone_capacity(self, zone_id: str, max_count: int) -> None:
        self.zone_capacities[zone_id] = max_count

    def update(self, camera_id: str, spatial=None) -> OccupancySnapshot:
        src = spatial or self._spatial
        entities = getattr(src, '_entities', []) if src else []
        persons = [e for e in entities if getattr(e, 'label', '') == 'person']
        person_count = len(persons)
        now = datetime.now(timezone.utc)

        if camera_id not in self.hourly_history:
            self.hourly_history[camera_id] = deque(maxlen=288)
        self.hourly_history[camera_id].append((now, person_count))

        zone_counts: dict[str, int] = {}
        for e in persons:
            zone = getattr(e, 'zone_id', None)
            if zone:
                zone_counts[zone] = zone_counts.get(zone, 0) + 1

        alerts = []
        for zone_id, count in zone_counts.items():
            cap = self.zone_capacities.get(zone_id)
            if cap and count >= cap:
                alerts.append({"zone_id": zone_id, "count": count, "capacity": cap})
                logger.warning("Zone %s at capacity: %d/%d", zone_id, count, cap)

        snap = OccupancySnapshot(
            camera_id=camera_id, timestamp=now,
            person_count=person_count, zone_counts=zone_counts,
            capacity_alerts=alerts,
        )
        self._latest[camera_id] = snap
        return snap

    def get_current(self, camera_id: str) -> Optional[OccupancySnapshot]:
        return self._latest.get(camera_id)

    def get_site_summary(self) -> dict:
        total = 0
        per_camera: dict[str, int] = {}
        zone_occ: dict[str, int] = {}
        busiest, busiest_count = None, 0

        for cid, snap in self._latest.items():
            total += snap.person_count
            per_camera[cid] = snap.person_count
            if snap.person_count > busiest_count:
                busiest, busiest_count = cid, snap.person_count
            for z, c in snap.zone_counts.items():
                zone_occ[z] = zone_occ.get(z, 0) + c

        return {
            "total_persons": total,
            "per_camera": per_camera,
            "busiest_camera": busiest,
            "zone_occupancies": zone_occ,
        }

    def get_history(self, camera_id: str, hours: int = 24) -> list[dict]:
        hist = self.hourly_history.get(camera_id, deque())
        cutoff = datetime.now(timezone.utc).timestamp() - hours * 3600
        return [
            {"timestamp": ts.isoformat(), "count": cnt}
            for ts, cnt in hist if ts.timestamp() >= cutoff
        ]

    def generate_summary(self, camera_id: str) -> str:
        snap = self._latest.get(camera_id)
        if not snap:
            return f"No data for camera {camera_id}"

        pct_parts = []
        for z, cnt in snap.zone_counts.items():
            cap = self.zone_capacities.get(z)
            if cap:
                pct_parts.append(f"{z} at {int(cnt / cap * 100)}% capacity")

        hist = self.hourly_history.get(camera_id, deque())
        trend = "stable"
        if len(hist) >= 2:
            recent = [c for _, c in list(hist)[-6:]]
            if len(recent) >= 2:
                trend = "trending up" if recent[-1] > recent[0] else "trending down" if recent[-1] < recent[0] else "stable"

        fallback = f"{', '.join(pct_parts) if pct_parts else f'{snap.person_count} persons detected'}, {trend}"

        client = self._get_client()
        if client:
            try:
                prompt = f"Summarize occupancy: camera={camera_id}, count={snap.person_count}, zones={snap.zone_counts}, trend={trend}"
                return client.generate(prompt)
            except Exception:
                logger.warning("VLM unavailable, using template fallback")
        return fallback

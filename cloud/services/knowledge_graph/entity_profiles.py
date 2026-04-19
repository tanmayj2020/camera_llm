"""Entity profiles from temporal knowledge graph.

Click any tracked person → full profile: first/last seen, cameras, zones,
typical hours, interactions, incidents, LLM behavior summary, risk score.
"""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EntityProfile:
    entity_id: str
    first_seen: float = 0.0
    last_seen: float = 0.0
    cameras: list[str] = field(default_factory=list)
    zones: list[str] = field(default_factory=list)
    typical_hours: list[int] = field(default_factory=list)  # hours of day (0-23)
    interactions: list[dict] = field(default_factory=list)
    incidents: list[dict] = field(default_factory=list)
    total_detections: int = 0
    behavior_summary: str = ""
    risk_score: float = 0.0  # 0.0 - 1.0


class EntityProfiler:
    """Builds rich entity profiles from temporal knowledge graph data."""

    RISK_WEIGHTS = {
        "incident_count": 0.3,
        "unusual_hours": 0.2,
        "multi_zone": 0.1,
        "interaction_count": 0.1,
        "frequency": 0.3,
    }

    def __init__(self, kg=None):
        self._kg = kg
        self._vlm_client = None
        self._cache: dict[str, tuple[float, EntityProfile]] = {}  # entity_id → (cached_at, profile)
        self._cache_ttl = 300  # 5 min

    def set_vlm_client(self, client):
        self._vlm_client = client

    def get_profile(self, entity_id: str, since_hours: int = 168) -> EntityProfile:
        """Build or return cached profile for an entity."""
        now = time.time()
        if entity_id in self._cache:
            cached_at, profile = self._cache[entity_id]
            if now - cached_at < self._cache_ttl:
                return profile

        profile = self._build_profile(entity_id, since_hours)
        self._cache[entity_id] = (now, profile)
        return profile

    def _build_profile(self, entity_id: str, since_hours: int) -> EntityProfile:
        profile = EntityProfile(entity_id=entity_id)

        if not self._kg:
            return profile

        try:
            history = self._kg.get_entity_history(entity_id, since_hours)
        except Exception as e:
            logger.error("KG query failed for profile %s: %s", entity_id, e)
            return profile

        if not history:
            return profile

        timestamps = []
        for record in history:
            props = record.get("props", {})
            target = record.get("target_props", {})
            labels = record.get("labels", [])
            rel = record.get("rel", "")

            ts = self._parse_ts(props.get("last_seen", ""))
            if ts:
                timestamps.append(ts)

            if "Camera" in labels:
                cam = target.get("camera_id", "")
                if cam and cam not in profile.cameras:
                    profile.cameras.append(cam)

            if "Zone" in labels:
                zone = target.get("zone_id", "")
                if zone and zone not in profile.zones:
                    profile.zones.append(zone)

            if rel == "INTERACTED_WITH":
                profile.interactions.append({
                    "type": props.get("type", "near"),
                    "timestamp": props.get("last_seen", ""),
                })

            if "Event" in labels and target.get("event_type") in ("anomaly", "alert"):
                profile.incidents.append({
                    "event_id": target.get("event_id", ""),
                    "type": target.get("event_type", ""),
                    "timestamp": target.get("timestamp", ""),
                })

        profile.total_detections = len(history)

        if timestamps:
            profile.first_seen = min(timestamps)
            profile.last_seen = max(timestamps)
            from datetime import datetime, timezone
            hours = [datetime.fromtimestamp(t, tz=timezone.utc).hour for t in timestamps]
            profile.typical_hours = sorted(set(hours))

        profile.risk_score = self._compute_risk(profile)
        profile.behavior_summary = self._generate_summary(profile)
        return profile

    def _compute_risk(self, profile: EntityProfile) -> float:
        """Compute risk score from profile features."""
        scores = {}

        # Incident frequency
        scores["incident_count"] = min(1.0, len(profile.incidents) / 5.0)

        # Unusual hours (late night / early morning)
        unusual = [h for h in profile.typical_hours if h < 6 or h > 22]
        scores["unusual_hours"] = min(1.0, len(unusual) / 3.0)

        # Multi-zone movement
        scores["multi_zone"] = min(1.0, len(profile.zones) / 5.0)

        # Interaction count
        scores["interaction_count"] = min(1.0, len(profile.interactions) / 10.0)

        # Detection frequency (high = more data, could be normal or suspicious)
        scores["frequency"] = min(1.0, profile.total_detections / 100.0)

        risk = sum(scores[k] * self.RISK_WEIGHTS[k] for k in self.RISK_WEIGHTS)
        return round(min(1.0, risk), 2)

    def _generate_summary(self, profile: EntityProfile) -> str:
        """Generate LLM behavior summary."""
        if self._vlm_client and hasattr(self._vlm_client, '_get_client'):
            client = self._vlm_client._get_client()
            if client != "stub":
                try:
                    prompt = (f"Write a 2-sentence security behavior summary for entity '{profile.entity_id}':\n"
                              f"- Seen on {len(profile.cameras)} cameras, {len(profile.zones)} zones\n"
                              f"- {profile.total_detections} detections, {len(profile.incidents)} incidents\n"
                              f"- Typical hours: {profile.typical_hours}\n"
                              f"- Risk score: {profile.risk_score}")
                    resp = client.generate_content(prompt)
                    return resp.text.strip()
                except Exception:
                    pass

        # Fallback
        risk_label = "low" if profile.risk_score < 0.3 else "medium" if profile.risk_score < 0.6 else "high"
        return (f"Entity observed {profile.total_detections} times across "
                f"{len(profile.cameras)} camera(s). Risk level: {risk_label}. "
                f"{len(profile.incidents)} incident(s) recorded.")

    def to_dict(self, profile: EntityProfile) -> dict:
        from datetime import datetime, timezone
        fmt = lambda t: datetime.fromtimestamp(t, tz=timezone.utc).isoformat() if t else None
        return {
            "entity_id": profile.entity_id,
            "first_seen": fmt(profile.first_seen),
            "last_seen": fmt(profile.last_seen),
            "cameras": profile.cameras,
            "zones": profile.zones,
            "typical_hours": profile.typical_hours,
            "interactions": profile.interactions,
            "incidents": profile.incidents,
            "total_detections": profile.total_detections,
            "behavior_summary": profile.behavior_summary,
            "risk_score": profile.risk_score,
        }

    @staticmethod
    def _parse_ts(ts_str: str) -> float | None:
        if not ts_str:
            return None
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            return dt.timestamp()
        except (ValueError, TypeError):
            return None

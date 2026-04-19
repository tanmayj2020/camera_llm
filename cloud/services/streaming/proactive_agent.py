"""Proactive Streaming Video Agent — continuously watches and alerts without being asked."""

import logging
import time
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ProactiveInsight:
    insight_type: str  # state_change, entity_reappearance, unusual_pattern, priority_match
    description: str
    confidence: float
    camera_id: str
    timestamp: float


class ProactiveAgent:
    """AURA-inspired agent that proactively surfaces insights from continuous video streams.

    Maintains rolling scene understanding and compares against operator priorities,
    watch entities, and historical state to generate unprompted notifications.
    """

    def __init__(self, spatial=None, kg=None, vlm_client=None):
        self._spatial = spatial
        self._kg = kg
        self._vlm_client = vlm_client
        self._scene_buffer: deque[dict] = deque(maxlen=60)  # ~30s at 2fps
        self._priorities: list[str] = []
        self._watch_entities: list[str] = []
        self._zone_state: dict[str, dict] = {}  # zone_id -> {last_used: float, entity_count: int}
        self._entity_cameras: dict[str, str] = {}  # entity_id -> last camera_id
        self._cooldowns: dict[str, float] = {}  # insight key -> last fired timestamp

    def set_priorities(self, priorities: list[str]):
        self._priorities = priorities
        logger.info("Proactive priorities set: %s", priorities)

    def set_watch_entities(self, entities: list[str]):
        self._watch_entities = entities
        logger.info("Watch entities set: %s", entities)

    def _cooled_down(self, key: str, cooldown_s: float = 120) -> bool:
        last = self._cooldowns.get(key, 0)
        if time.time() - last < cooldown_s:
            return False
        self._cooldowns[key] = time.time()
        return True

    def evaluate(self, scene_state: dict) -> list[ProactiveInsight]:
        ts = scene_state.get("timestamp", time.time())
        camera_id = scene_state.get("camera_id", "unknown")
        spatial = scene_state.get("spatial") or self._spatial
        insights = []

        # Buffer scene snapshot
        self._scene_buffer.append({
            "camera_id": camera_id, "timestamp": ts,
            "entity_count": len(scene_state.get("objects", [])),
            "objects": [o.get("class_name", "") for o in scene_state.get("objects", [])],
        })

        if not spatial:
            return insights

        # 1. Entity reappearance — watched entity shows up on a different camera
        for eid, ent in getattr(spatial, "_entities", {}).items():
            for watch_id in self._watch_entities:
                if watch_id in eid:
                    prev_cam = self._entity_cameras.get(eid)
                    if prev_cam and prev_cam != camera_id and self._cooled_down(f"reappear:{eid}"):
                        insights.append(ProactiveInsight(
                            "entity_reappearance",
                            f"Watched entity {eid} just appeared on {camera_id} (was on {prev_cam})",
                            0.9, camera_id, ts,
                        ))
            self._entity_cameras[eid] = camera_id

        # 2. State change — zone usage pattern breaks
        for zid, zone in getattr(spatial, "_zones", {}).items():
            try:
                in_zone = spatial.entities_in_zone(zid)
                count = len(in_zone) if in_zone else 0
            except Exception:
                count = 0
            prev = self._zone_state.get(zid, {})
            prev_count = prev.get("entity_count", 0)
            last_used = prev.get("last_used", ts)

            if count > 0 and prev_count == 0:
                idle_hours = (ts - last_used) / 3600
                if idle_hours > 2 and self._cooled_down(f"state:{zid}"):
                    insights.append(ProactiveInsight(
                        "state_change",
                        f"Zone '{zone.name}' just became active after {idle_hours:.1f}h of inactivity — {count} entities entered",
                        0.8, camera_id, ts,
                    ))

            self._zone_state[zid] = {
                "entity_count": count,
                "last_used": ts if count > 0 else last_used,
            }

        # 3. Priority matching — check if scene matches operator priorities
        if self._priorities:
            objects_str = " ".join(o.get("class_name", "") for o in scene_state.get("objects", []))
            for priority in self._priorities:
                keywords = priority.lower().split()
                if any(kw in objects_str.lower() for kw in keywords):
                    if self._cooled_down(f"priority:{priority}", 300):
                        insights.append(ProactiveInsight(
                            "priority_match",
                            f"Priority match: '{priority}' — detected relevant activity on {camera_id}",
                            0.7, camera_id, ts,
                        ))

        # 4. Unusual pattern — sudden entity count spike
        if len(self._scene_buffer) >= 10:
            recent_counts = [s["entity_count"] for s in list(self._scene_buffer)[-10:]]
            avg = sum(recent_counts[:-1]) / max(len(recent_counts) - 1, 1)
            current = recent_counts[-1]
            if avg > 0 and current > avg * 3 and self._cooled_down(f"spike:{camera_id}", 180):
                insights.append(ProactiveInsight(
                    "unusual_pattern",
                    f"Sudden activity spike on {camera_id}: {current} entities (avg was {avg:.0f})",
                    0.75, camera_id, ts,
                ))

        return insights

"""Tailgating and door propping detection."""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DoorEvent:
    event_type: str  # tailgate, door_prop
    zone_id: str
    entity_ids: list[str]
    timestamp: float


class TailgateDetector:
    def __init__(self, spatial=None):
        self._spatial = spatial
        self._zone_transitions: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self._exit_log: dict[str, list[tuple[str, float]]] = defaultdict(list)

    def _record_entry(self, zone_id: str, entity_id: str, timestamp: float):
        entries = self._zone_transitions[zone_id]
        entries.append((entity_id, timestamp))
        # Keep only last 30s
        cutoff = timestamp - 30
        self._zone_transitions[zone_id] = [(e, t) for e, t in entries if t > cutoff]

    def _record_exit(self, zone_id: str, entity_id: str, timestamp: float):
        exits = self._exit_log[zone_id]
        exits.append((entity_id, timestamp))
        cutoff = timestamp - 30
        self._exit_log[zone_id] = [(e, t) for e, t in exits if t > cutoff]

    def detect(self, zone_id: str, timestamp: float) -> list[DoorEvent]:
        events = []
        entries = self._zone_transitions.get(zone_id, [])
        # Tailgate: >1 entity enters within 3s window
        window = [(eid, t) for eid, t in entries if timestamp - t < 3.0]
        if len(window) > 1:
            ids = list({eid for eid, _ in window})
            if len(ids) > 1:
                events.append(DoorEvent("tailgate", zone_id, ids, timestamp))
        return events

    def detect_door_prop(self, zone_id: str, timestamp: float) -> list[DoorEvent]:
        events = []
        exits = self._exit_log.get(zone_id, [])
        entries = self._zone_transitions.get(zone_id, [])
        for eid, exit_t in exits:
            for eid2, entry_t in entries:
                if eid == eid2 and entry_t > exit_t and (entry_t - exit_t) < 5.0:
                    events.append(DoorEvent("door_prop", zone_id, [eid], timestamp))
        return events

    def evaluate(self, scene_state: dict) -> list[DoorEvent]:
        spatial = scene_state.get("spatial") or self._spatial
        if not spatial:
            return []
        ts = scene_state.get("timestamp", time.time())
        results = []
        for eid, ent in getattr(spatial, "_entities", {}).items():
            for zid, zone in getattr(spatial, "_zones", {}).items():
                try:
                    if spatial.entities_in_zone(zid):
                        if eid in [e.track_id for e in spatial.entities_in_zone(zid)
                                   if hasattr(e, "track_id")]:
                            self._record_entry(zid, eid, ts)
                except Exception:
                    pass
        for zid in self._zone_transitions:
            results.extend(self.detect(zid, ts))
            results.extend(self.detect_door_prop(zid, ts))
        return results

"""Dynamic Scene Graph for spatial relationship tracking."""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SceneNode:
    entity_id: str
    class_name: str
    attributes: dict = field(default_factory=dict)


@dataclass
class SceneEdge:
    subject: str
    predicate: str
    object: str
    confidence: float
    timestamp: float


class DynamicSceneGraph:
    """Maintains and updates a dynamic scene graph from spatial memory."""

    def __init__(self, spatial=None):
        self._spatial = spatial
        self._edges: list[SceneEdge] = []
        self._prev_edges: list[SceneEdge] = []
        self._transition_log: list[dict] = []
        self._prev_objects: dict[str, set[str]] = {}  # entity_id -> set of nearby object ids

    def update(self, spatial, timestamp: float) -> None:
        self._spatial = spatial
        self._prev_edges = self._edges
        nodes = {eid: SceneNode(eid, e.class_name) for eid, e in spatial._entities.items()}
        edges: list[SceneEdge] = []
        ids = list(nodes.keys())

        for i, a_id in enumerate(ids):
            a = spatial._entities[a_id]
            for b_id in ids[i + 1:]:
                b = spatial._entities[b_id]
                diff = b.position - a.position
                d = float(np.linalg.norm(diff))

                # near
                if d < 3.0:
                    edges.append(SceneEdge(a_id, "near", b_id, min(1.0, 1.0 - d / 3.0), timestamp))

                # approaching: closing speed positive + distance < 5m
                if d > 0 and d < 5.0:
                    closing = -float(np.dot(a.velocity - b.velocity, diff / d))
                    if closing > 0.3:
                        edges.append(SceneEdge(a_id, "approaching", b_id, min(1.0, closing), timestamp))

                # following: same direction, one behind the other, similar speed
                sa, sb = np.linalg.norm(a.velocity), np.linalg.norm(b.velocity)
                if sa > 0.1 and sb > 0.1:
                    cos = float(np.dot(a.velocity, b.velocity) / (sa * sb))
                    if cos > 0.7 and abs(sa - sb) < 0.5:
                        # a following b if a is behind b (dot of diff with b.velocity > 0)
                        if float(np.dot(diff, b.velocity)) > 0:
                            edges.append(SceneEdge(a_id, "following", b_id, cos, timestamp))
                        else:
                            edges.append(SceneEdge(b_id, "following", a_id, cos, timestamp))

                # blocking: one stationary in path of moving entity
                if d < 2.0:
                    if sa > 0.5 and sb < 0.2 and d > 0:
                        if float(np.dot(a.velocity / sa, diff / d)) > 0.5:
                            edges.append(SceneEdge(b_id, "blocking", a_id, 0.8, timestamp))
                    elif sb > 0.5 and sa < 0.2 and d > 0:
                        if float(np.dot(b.velocity / sb, -diff / d)) > 0.5:
                            edges.append(SceneEdge(a_id, "blocking", b_id, 0.8, timestamp))

        # handing_object_to: two persons very close, one had nearby object that disappeared
        persons = [eid for eid, n in nodes.items() if n.class_name == "person"]
        objects = {eid for eid, n in nodes.items() if n.class_name != "person"}
        curr_nearby: dict[str, set[str]] = {}
        for pid in persons:
            p = spatial._entities[pid]
            curr_nearby[pid] = {oid for oid in objects if float(np.linalg.norm(p.position - spatial._entities[oid].position)) < 1.5}

        for i, p1 in enumerate(persons):
            for p2 in persons[i + 1:]:
                if float(np.linalg.norm(spatial._entities[p1].position - spatial._entities[p2].position)) < 1.5:
                    lost = self._prev_objects.get(p1, set()) - curr_nearby.get(p1, set())
                    gained = curr_nearby.get(p2, set()) - self._prev_objects.get(p2, set())
                    if lost & gained:
                        edges.append(SceneEdge(p1, "handing_object_to", p2, 0.7, timestamp))
        self._prev_objects = curr_nearby

        # detect transitions
        prev_set = {(e.subject, e.predicate, e.object) for e in self._prev_edges}
        curr_set = {(e.subject, e.predicate, e.object) for e in edges}
        for s, p, o in curr_set - prev_set:
            entry = {"type": "added", "subject": s, "predicate": p, "object": o, "timestamp": timestamp}
            if p in ("approaching", "following", "handing_object_to"):
                entry["suspicious"] = True
            self._transition_log.append(entry)
        for s, p, o in prev_set - curr_set:
            entry = {"type": "removed", "subject": s, "predicate": p, "object": o, "timestamp": timestamp}
            self._transition_log.append(entry)

        self._edges = edges

    def get_edges(self) -> list[dict[str, Any]]:
        return [{"subject": e.subject, "predicate": e.predicate, "object": e.object,
                 "confidence": e.confidence, "timestamp": e.timestamp} for e in self._edges]

    def get_transitions(self, since: float = 0) -> list[dict]:
        return [t for t in self._transition_log if t["timestamp"] >= since]

    def query(self, subject: Optional[str] = None, predicate: Optional[str] = None,
              obj: Optional[str] = None) -> list[dict[str, Any]]:
        results = []
        for e in self._edges:
            if subject and e.subject != subject:
                continue
            if predicate and e.predicate != predicate:
                continue
            if obj and e.object != obj:
                continue
            results.append({"subject": e.subject, "predicate": e.predicate, "object": e.object,
                            "confidence": e.confidence, "timestamp": e.timestamp})
        return results

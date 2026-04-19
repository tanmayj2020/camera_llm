"""Line Crossing Counter — directional people/vehicle counting across virtual lines.

Counts entries/exits across user-defined lines using spatial trajectory data.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CountingLine:
    line_id: str
    name: str
    point_a: tuple[float, float]  # (x, y) start
    point_b: tuple[float, float]  # (x, y) end
    # Normal direction: "positive" side is left of A→B vector
    bidirectional: bool = True


@dataclass
class LineCrossEvent:
    line_id: str
    entity_id: str
    direction: str  # "in" or "out"
    timestamp: float
    class_name: str = "person"


class LineCrossingCounter:
    """Counts directional crossings of virtual lines using entity trajectories."""

    def __init__(self):
        self._lines: dict[str, CountingLine] = {}
        self._counts: dict[str, dict[str, int]] = {}  # line_id -> {"in": N, "out": N}
        self._entity_side: dict[str, dict[str, int]] = defaultdict(dict)  # entity_id -> {line_id: side}
        self._events: list[LineCrossEvent] = []

    def add_line(self, line: CountingLine):
        self._lines[line.line_id] = line
        self._counts[line.line_id] = {"in": 0, "out": 0}

    def remove_line(self, line_id: str):
        self._lines.pop(line_id, None)
        self._counts.pop(line_id, None)

    def _side_of_line(self, point: tuple[float, float], line: CountingLine) -> int:
        """Returns +1 or -1 depending on which side of the line the point is."""
        ax, ay = line.point_a
        bx, by = line.point_b
        px, py = point
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        return 1 if cross > 0 else -1

    def update(self, entity_id: str, position: tuple[float, float],
               timestamp: float, class_name: str = "person") -> list[LineCrossEvent]:
        """Update entity position and detect line crossings."""
        crossings = []
        for line_id, line in self._lines.items():
            current_side = self._side_of_line(position, line)
            prev_side = self._entity_side[entity_id].get(line_id)

            if prev_side is not None and prev_side != current_side:
                direction = "in" if current_side == 1 else "out"
                self._counts[line_id][direction] += 1
                event = LineCrossEvent(line_id, entity_id, direction, timestamp, class_name)
                self._events.append(event)
                crossings.append(event)
                logger.debug("Line cross: %s %s %s at %.1f", entity_id, direction, line_id, timestamp)

            self._entity_side[entity_id][line_id] = current_side
        return crossings

    def update_from_spatial(self, spatial, timestamp: float) -> list[LineCrossEvent]:
        """Batch update from spatial memory."""
        all_crossings = []
        for eid, ent in getattr(spatial, "_entities", {}).items():
            pos = (float(ent.position[0]), float(ent.position[2]) if len(ent.position) > 2 else float(ent.position[1]))
            crossings = self.update(eid, pos, timestamp, ent.class_name)
            all_crossings.extend(crossings)
        return crossings

    def get_counts(self, line_id: str = None) -> dict:
        if line_id:
            c = self._counts.get(line_id, {"in": 0, "out": 0})
            return {"line_id": line_id, **c, "net": c["in"] - c["out"]}
        return {lid: {**c, "net": c["in"] - c["out"]} for lid, c in self._counts.items()}

    def get_recent_events(self, line_id: str = None, limit: int = 50) -> list[dict]:
        events = self._events if not line_id else [e for e in self._events if e.line_id == line_id]
        return [{"line_id": e.line_id, "entity_id": e.entity_id, "direction": e.direction,
                 "timestamp": e.timestamp, "class": e.class_name} for e in events[-limit:]]

    def list_lines(self) -> list[dict]:
        return [{"line_id": l.line_id, "name": l.name,
                 "point_a": l.point_a, "point_b": l.point_b,
                 "counts": self._counts.get(l.line_id, {})}
                for l in self._lines.values()]

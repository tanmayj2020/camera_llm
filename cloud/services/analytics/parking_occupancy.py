"""Parking occupancy detection — detect empty/occupied spots from overhead cameras."""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ParkingSpot:
    spot_id: str
    polygon: list[tuple[float, float]]  # 4 corners
    occupied: bool = False
    vehicle_id: str = ""
    last_changed: float = 0.0


@dataclass
class ParkingStatus:
    total_spots: int
    occupied: int
    available: int
    occupancy_pct: float
    spots: list[dict] = field(default_factory=list)


class ParkingOccupancy:
    """Detects parking spot occupancy from spatial entity positions."""

    def __init__(self, spatial=None):
        self._spatial = spatial
        self._spots: dict[str, ParkingSpot] = {}

    def add_spot(self, spot_id: str, polygon: list[tuple[float, float]]):
        self._spots[spot_id] = ParkingSpot(spot_id=spot_id, polygon=polygon)

    def remove_spot(self, spot_id: str):
        self._spots.pop(spot_id, None)

    def update(self, spatial=None) -> ParkingStatus:
        spatial = spatial or self._spatial
        if not spatial:
            return ParkingStatus(0, 0, 0, 0.0)

        vehicle_classes = {"car", "truck", "bus", "vehicle", "motorcycle"}
        vehicles = {eid: ent for eid, ent in getattr(spatial, "_entities", {}).items()
                    if ent.class_name in vehicle_classes}

        now = time.time()
        for spot in self._spots.values():
            was_occupied = spot.occupied
            spot.occupied = False
            spot.vehicle_id = ""
            for vid, vent in vehicles.items():
                pos = (float(vent.position[0]),
                       float(vent.position[2]) if len(vent.position) > 2 else float(vent.position[1]))
                if self._point_in_polygon(pos, spot.polygon):
                    spot.occupied = True
                    spot.vehicle_id = vid
                    break
            if spot.occupied != was_occupied:
                spot.last_changed = now

        total = len(self._spots)
        occupied = sum(1 for s in self._spots.values() if s.occupied)
        return ParkingStatus(
            total_spots=total, occupied=occupied, available=total - occupied,
            occupancy_pct=round(occupied / max(total, 1) * 100, 1),
            spots=[{"spot_id": s.spot_id, "occupied": s.occupied,
                    "vehicle_id": s.vehicle_id} for s in self._spots.values()])

    @staticmethod
    def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
        x, y = point
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

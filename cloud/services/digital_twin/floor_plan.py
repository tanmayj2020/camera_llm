"""2D Digital Twin Floor Plan — real-time bird's-eye site map aggregating
all cameras, entity positions, zone occupancy, heatmaps, active alerts.

Maintains site-level coordinate system, maps camera-relative positions
to site coordinates.
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraMount:
    camera_id: str
    site_x: float  # position on floor plan (meters)
    site_y: float
    rotation_deg: float = 0.0  # camera facing direction
    fov_deg: float = 60.0
    label: str = ""


@dataclass
class FloorPlanEntity:
    track_id: str
    class_name: str
    site_x: float
    site_y: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    camera_id: str = ""
    last_seen: float = 0.0


@dataclass
class FloorPlanZone:
    zone_id: str
    name: str
    polygon: list[dict]  # [{x, y}, ...]
    zone_type: str = "user"
    occupancy: int = 0


@dataclass
class FloorPlanAlert:
    alert_id: str
    site_x: float
    site_y: float
    severity: str
    description: str
    camera_id: str = ""
    timestamp: float = 0.0


@dataclass
class FloorPlanSnapshot:
    timestamp: float
    entities: list[FloorPlanEntity] = field(default_factory=list)
    cameras: list[CameraMount] = field(default_factory=list)
    zones: list[FloorPlanZone] = field(default_factory=list)
    alerts: list[FloorPlanAlert] = field(default_factory=list)
    heatmap: list[list[float]] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


class DigitalTwinFloorPlan:
    """Aggregates spatial data from all cameras into a unified 2D site map.

    Camera-relative 3D positions are transformed to site coordinates
    using camera mount positions and rotations.
    """

    HEATMAP_GRID = 20  # 20x20 grid
    ENTITY_STALE_S = 60  # entities older than 60s are excluded

    def __init__(self, site_width: float = 50.0, site_height: float = 50.0):
        self.site_width = site_width
        self.site_height = site_height
        self._cameras: dict[str, CameraMount] = {}
        self._spatial_memories: dict[str, object] = {}  # camera_id → SpatialMemory
        self._heatmap = np.zeros((self.HEATMAP_GRID, self.HEATMAP_GRID))
        self._active_alerts: list[FloorPlanAlert] = []

    def register_camera(self, camera_id: str, site_x: float, site_y: float,
                        rotation_deg: float = 0.0, fov_deg: float = 60.0, label: str = ""):
        self._cameras[camera_id] = CameraMount(
            camera_id=camera_id, site_x=site_x, site_y=site_y,
            rotation_deg=rotation_deg, fov_deg=fov_deg,
            label=label or camera_id,
        )

    def register_spatial_memory(self, camera_id: str, spatial):
        self._spatial_memories[camera_id] = spatial

    def add_alert(self, alert_id: str, camera_id: str, severity: str,
                  description: str, entity_position: tuple[float, float] | None = None):
        site_pos = entity_position
        if not site_pos and camera_id in self._cameras:
            cam = self._cameras[camera_id]
            site_pos = (cam.site_x, cam.site_y)
        self._active_alerts.append(FloorPlanAlert(
            alert_id=alert_id, site_x=site_pos[0] if site_pos else 0,
            site_y=site_pos[1] if site_pos else 0,
            severity=severity, description=description,
            camera_id=camera_id, timestamp=time.time(),
        ))
        # Keep bounded
        if len(self._active_alerts) > 50:
            self._active_alerts = self._active_alerts[-50:]

    def get_snapshot(self) -> FloorPlanSnapshot:
        """Get current state of the entire site as a 2D floor plan."""
        now = time.time()
        all_entities = []
        zone_list = []

        for cam_id, spatial in self._spatial_memories.items():
            cam = self._cameras.get(cam_id)
            if not cam or not hasattr(spatial, '_entities'):
                continue

            for ent in spatial._entities.values():
                if now - ent.last_seen > self.ENTITY_STALE_S:
                    continue
                sx, sy = self._camera_to_site(cam, ent.position)
                all_entities.append(FloorPlanEntity(
                    track_id=ent.track_id, class_name=ent.class_name,
                    site_x=round(sx, 2), site_y=round(sy, 2),
                    velocity_x=round(float(ent.velocity[0]), 2),
                    velocity_y=round(float(ent.velocity[2]) if len(ent.velocity) > 2 else 0, 2),
                    camera_id=cam_id, last_seen=ent.last_seen,
                ))
                # Update heatmap
                gx = int(np.clip(sx / self.site_width * (self.HEATMAP_GRID - 1), 0, self.HEATMAP_GRID - 1))
                gy = int(np.clip(sy / self.site_height * (self.HEATMAP_GRID - 1), 0, self.HEATMAP_GRID - 1))
                self._heatmap[gy, gx] += 1

            # Zones from spatial
            if hasattr(spatial, '_zones'):
                for z in spatial._zones.values():
                    occupancy = len(spatial.entities_in_zone(z.zone_id))
                    zone_list.append(FloorPlanZone(
                        zone_id=z.zone_id, name=z.name,
                        polygon=[{"x": p[0], "y": p[1]} for p in z.polygon],
                        zone_type=z.zone_type, occupancy=occupancy,
                    ))

        # Normalize heatmap to 0-1
        hm_max = self._heatmap.max()
        heatmap_normalized = (self._heatmap / hm_max).tolist() if hm_max > 0 else self._heatmap.tolist()

        # Recent alerts (last 5 min)
        recent_alerts = [a for a in self._active_alerts if now - a.timestamp < 300]

        # Stats
        person_count = sum(1 for e in all_entities if e.class_name == "person")
        vehicle_count = sum(1 for e in all_entities if e.class_name in ("car", "truck", "vehicle"))

        return FloorPlanSnapshot(
            timestamp=now,
            entities=all_entities,
            cameras=list(self._cameras.values()),
            zones=zone_list,
            alerts=recent_alerts,
            heatmap=heatmap_normalized,
            stats={
                "total_entities": len(all_entities),
                "person_count": person_count,
                "vehicle_count": vehicle_count,
                "active_cameras": len(self._spatial_memories),
                "active_alerts": len(recent_alerts),
                "zone_count": len(zone_list),
            },
        )

    def _camera_to_site(self, cam: CameraMount, position) -> tuple[float, float]:
        """Transform camera-relative 3D position to site 2D coordinates."""
        # position is [x, y, z] in camera frame; x=lateral, z=depth
        cx = float(position[0]) if hasattr(position, '__getitem__') else 0
        cz = float(position[2]) if hasattr(position, '__getitem__') and len(position) > 2 else 0

        # Rotate by camera orientation
        rad = np.radians(cam.rotation_deg)
        site_x = cam.site_x + cx * np.cos(rad) - cz * np.sin(rad)
        site_y = cam.site_y + cx * np.sin(rad) + cz * np.cos(rad)

        # Clamp to site bounds
        site_x = max(0, min(self.site_width, site_x))
        site_y = max(0, min(self.site_height, site_y))
        return site_x, site_y

    def export_snapshot(self, snapshot: FloorPlanSnapshot) -> dict:
        return {
            "timestamp": snapshot.timestamp,
            "site": {"width": self.site_width, "height": self.site_height},
            "stats": snapshot.stats,
            "entities": [
                {"track_id": e.track_id, "class": e.class_name,
                 "x": e.site_x, "y": e.site_y,
                 "vx": e.velocity_x, "vy": e.velocity_y,
                 "camera": e.camera_id}
                for e in snapshot.entities
            ],
            "cameras": [
                {"id": c.camera_id, "x": c.site_x, "y": c.site_y,
                 "rotation": c.rotation_deg, "fov": c.fov_deg, "label": c.label}
                for c in snapshot.cameras
            ],
            "zones": [
                {"id": z.zone_id, "name": z.name, "polygon": z.polygon,
                 "type": z.zone_type, "occupancy": z.occupancy}
                for z in snapshot.zones
            ],
            "alerts": [
                {"id": a.alert_id, "x": a.site_x, "y": a.site_y,
                 "severity": a.severity, "description": a.description,
                 "camera": a.camera_id}
                for a in snapshot.alerts
            ],
            "heatmap": snapshot.heatmap,
        }

"""Dynamic camera management with persistence and floor plan integration."""

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    camera_id: str
    name: str
    rtsp_url: str
    site_x: float = 0.0
    site_y: float = 0.0
    rotation_deg: float = 0.0
    fov_deg: float = 60.0
    enabled: bool = True
    tenant_id: str = "default"
    tags: list[str] = field(default_factory=list)
    created_at: str = ""


_STORE_PREFIX = "camera:"


class CameraManager:
    def __init__(self, persistence_store, floor_plan=None):
        self._store = persistence_store
        self._floor_plan = floor_plan
        self._cameras: dict[str, CameraConfig] = {}
        self._load_from_persistence()
        if not self._cameras:
            self._seed_defaults()

    def set_floor_plan(self, floor_plan) -> None:
        self._floor_plan = floor_plan

    def _load_from_persistence(self) -> None:
        for key in self._store.list(_STORE_PREFIX):
            data = self._store.get(key)
            if data:
                cam = CameraConfig(**data)
                self._cameras[cam.camera_id] = cam
        logger.info("Loaded %d cameras from persistence", len(self._cameras))

    def _seed_defaults(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        for i in range(4):
            cam = CameraConfig(
                camera_id=f"cam-{i}",
                name=f"Camera {i}",
                rtsp_url=f"rtsp://localhost:8554/cam{i}",
                site_x=(i % 2) * 25.0,
                site_y=(i // 2) * 25.0,
                created_at=now,
            )
            self.add_camera(cam)
        logger.info("Seeded 4 default cameras")

    def _register_floor_plan(self, cam: CameraConfig) -> None:
        if not self._floor_plan:
            return
        try:
            self._floor_plan.register_camera(
                camera_id=cam.camera_id,
                site_x=cam.site_x,
                site_y=cam.site_y,
                rotation_deg=cam.rotation_deg,
                fov_deg=cam.fov_deg,
                label=cam.name,
            )
        except Exception as exc:
            logger.warning("Floor plan registration failed for %s: %s", cam.camera_id, exc)

    def add_camera(self, cam: CameraConfig) -> None:
        if not cam.created_at:
            cam.created_at = datetime.now(timezone.utc).isoformat()
        self._cameras[cam.camera_id] = cam
        self._store.set(f"{_STORE_PREFIX}{cam.camera_id}", asdict(cam))
        self._register_floor_plan(cam)
        logger.info("Added camera %s", cam.camera_id)

    def update_camera(self, camera_id: str, **kwargs) -> None:
        cam = self._cameras.get(camera_id)
        if not cam:
            raise KeyError(f"Camera {camera_id} not found")
        for k, v in kwargs.items():
            if hasattr(cam, k):
                setattr(cam, k, v)
        self._store.set(f"{_STORE_PREFIX}{camera_id}", asdict(cam))
        self._register_floor_plan(cam)

    def remove_camera(self, camera_id: str) -> None:
        self._cameras.pop(camera_id, None)
        self._store.delete(f"{_STORE_PREFIX}{camera_id}")
        logger.info("Removed camera %s", camera_id)

    def get_camera(self, camera_id: str) -> Optional[CameraConfig]:
        return self._cameras.get(camera_id)

    def list_cameras(self, tenant_id: str | None = None) -> list[CameraConfig]:
        cams = list(self._cameras.values())
        if tenant_id:
            cams = [c for c in cams if c.tenant_id == tenant_id]
        return cams

    def get_camera_ids(self) -> list[str]:
        return list(self._cameras.keys())

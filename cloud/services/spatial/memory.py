"""4D Spatial Intelligence — MiDaS monocular depth, Kalman filtering, proper 3D positions."""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpatialEntity:
    """Entity with Kalman-filtered 3D position."""
    track_id: str
    class_name: str
    position: np.ndarray       # [x, y, z] in meters
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last_seen: float = 0.0
    # Kalman state
    _kf_state: np.ndarray | None = None  # [x, y, z, vx, vy, vz]
    _kf_cov: np.ndarray | None = None


@dataclass
class Zone:
    zone_id: str
    name: str
    polygon: list[tuple[float, float]]
    zone_type: str = "auto"


class _KalmanFilter3D:
    """Simple 3D constant-velocity Kalman filter."""

    def __init__(self, process_noise: float = 0.1, measurement_noise: float = 0.5):
        self.F = np.eye(6)  # state transition (updated with dt)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0
        self.Q = np.eye(6) * process_noise
        self.R = np.eye(3) * measurement_noise

    def predict(self, state: np.ndarray, cov: np.ndarray, dt: float):
        F = self.F.copy()
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        state_pred = F @ state
        cov_pred = F @ cov @ F.T + self.Q
        return state_pred, cov_pred

    def update(self, state: np.ndarray, cov: np.ndarray, measurement: np.ndarray):
        y = measurement - self.H @ state
        S = self.H @ cov @ self.H.T + self.R
        K = cov @ self.H.T @ np.linalg.inv(S)
        state_new = state + K @ y
        cov_new = (np.eye(6) - K @ self.H) @ cov
        return state_new, cov_new


class SpatialMemory:
    """4D spatial map with MiDaS depth estimation and Kalman filtering.

    Falls back to bbox height heuristic when MiDaS is unavailable.
    """

    PERSON_HEIGHT_M = 1.7
    MAX_ENTITIES = 500

    def __init__(self, camera_fov_h: float = 60.0, image_width: int = 1920, image_height: int = 1080):
        self.fov_h = math.radians(camera_fov_h)
        self.img_w = image_width
        self.img_h = image_height
        self.focal_length = (image_width / 2) / math.tan(self.fov_h / 2)

        self._entities: dict[str, SpatialEntity] = {}
        self._zones: dict[str, Zone] = {}
        self._kf = _KalmanFilter3D()
        self._movement_heatmap: np.ndarray = np.zeros((50, 50))
        self._heatmap_bounds = (0, 0, 20, 20)

        # Lazy-loaded MiDaS
        self._midas_model = None
        self._midas_transform = None
        self._midas_available = None
        self._last_depth_map: np.ndarray | None = None

    # --- MiDaS depth estimation (lazy) ---

    def _load_midas(self):
        if self._midas_available is not None:
            return self._midas_available
        try:
            import torch
            self._midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            self._midas_model.eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self._midas_transform = midas_transforms.small_transform
            self._midas_available = True
            logger.info("MiDaS depth model loaded")
        except Exception as e:
            logger.warning("MiDaS unavailable, using bbox heuristic: %s", e)
            self._midas_available = False
        return self._midas_available

    def compute_depth_map(self, frame: np.ndarray) -> np.ndarray | None:
        """Compute monocular depth map using MiDaS."""
        if not self._load_midas():
            return None
        try:
            import torch
            input_batch = self._midas_transform(frame)
            if torch.cuda.is_available():
                self._midas_model = self._midas_model.cuda()
                input_batch = input_batch.cuda()
            with torch.no_grad():
                prediction = self._midas_model(input_batch)
            depth = prediction.squeeze().cpu().numpy()
            # Normalize to approximate meters (MiDaS outputs inverse relative depth)
            depth = depth.max() / (depth + 1e-6)
            self._last_depth_map = depth
            return depth
        except Exception as e:
            logger.debug("MiDaS inference failed: %s", e)
            return None

    def estimate_depth(self, bbox_height_px: float, class_name: str = "person",
                       bbox_center: tuple[int, int] | None = None) -> float:
        """Estimate depth — uses MiDaS depth map if available, else bbox heuristic."""
        # Try MiDaS depth map
        if self._last_depth_map is not None and bbox_center:
            cx, cy = bbox_center
            h, w = self._last_depth_map.shape
            # Scale bbox center to depth map coords
            dx = int(cx / self.img_w * w)
            dy = int(cy / self.img_h * h)
            dx = max(0, min(dx, w - 1))
            dy = max(0, min(dy, h - 1))
            return float(self._last_depth_map[dy, dx])

        # Fallback: pinhole model heuristic
        if bbox_height_px < 10:
            return 20.0
        real_height = self.PERSON_HEIGHT_M if class_name == "person" else 1.0
        return (real_height * self.focal_length) / bbox_height_px

    def bbox_to_3d(self, bbox: list[float], class_name: str = "person") -> np.ndarray:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        h = y2 - y1

        depth = self.estimate_depth(h, class_name, bbox_center=(int(cx), int(cy)))
        x_3d = (cx - self.img_w / 2) * depth / self.focal_length
        y_3d = 0.0
        z_3d = depth
        return np.array([x_3d, y_3d, z_3d])

    def update(self, track_id: str, class_name: str, bbox: list[float],
               timestamp: float, frame: np.ndarray | None = None) -> SpatialEntity:
        """Update entity position with Kalman filtering."""
        # Optionally update depth map
        if frame is not None and self._load_midas():
            self.compute_depth_map(frame)

        measurement = self.bbox_to_3d(bbox, class_name)

        if track_id in self._entities:
            ent = self._entities[track_id]
            dt = max(timestamp - ent.last_seen, 0.033)

            # Kalman predict + update
            if ent._kf_state is not None:
                state_pred, cov_pred = self._kf.predict(ent._kf_state, ent._kf_cov, dt)
                ent._kf_state, ent._kf_cov = self._kf.update(state_pred, cov_pred, measurement)
                ent.position = ent._kf_state[:3].copy()
                ent.velocity = ent._kf_state[3:].copy()
            else:
                ent.velocity = (measurement - ent.position) / dt
                ent.position = measurement
                ent._kf_state = np.concatenate([measurement, ent.velocity])
                ent._kf_cov = np.eye(6)

            ent.last_seen = timestamp
        else:
            state = np.concatenate([measurement, np.zeros(3)])
            ent = SpatialEntity(
                track_id=track_id, class_name=class_name,
                position=measurement, last_seen=timestamp,
                _kf_state=state, _kf_cov=np.eye(6),
            )
            self._entities[track_id] = ent

        # Update heatmap
        bx = self._heatmap_bounds
        gx = int(np.clip((measurement[0] - bx[0]) / (bx[2] - bx[0]) * 49, 0, 49))
        gz = int(np.clip((measurement[2] - bx[1]) / (bx[3] - bx[1]) * 49, 0, 49))
        self._movement_heatmap[gz, gx] += 1

        # Evict stale
        if len(self._entities) > self.MAX_ENTITIES:
            oldest = min(self._entities.values(), key=lambda e: e.last_seen)
            del self._entities[oldest.track_id]

        return ent

    def distance_between(self, track_id_a: str, track_id_b: str) -> float | None:
        a = self._entities.get(track_id_a)
        b = self._entities.get(track_id_b)
        if a is None or b is None:
            return None
        return float(np.linalg.norm(a.position - b.position))

    def distance_to_zone(self, track_id: str, zone_id: str) -> float | None:
        ent = self._entities.get(track_id)
        zone = self._zones.get(zone_id)
        if ent is None or zone is None:
            return None
        pos_2d = np.array([ent.position[0], ent.position[2]])
        return float(min(np.linalg.norm(pos_2d - np.array(v)) for v in zone.polygon))

    def entities_in_zone(self, zone_id: str) -> list[SpatialEntity]:
        zone = self._zones.get(zone_id)
        if zone is None:
            return []
        return [e for e in self._entities.values()
                if self._point_in_polygon(e.position[0], e.position[2], zone.polygon)]

    def predict_collision(self, track_id_a: str, track_id_b: str,
                          horizon_s: float = 5.0) -> dict | None:
        a = self._entities.get(track_id_a)
        b = self._entities.get(track_id_b)
        if a is None or b is None:
            return None
        min_dist, min_t = float("inf"), 0.0
        for t in np.linspace(0, horizon_s, 20):
            d = float(np.linalg.norm((a.position + a.velocity * t) - (b.position + b.velocity * t)))
            if d < min_dist:
                min_dist, min_t = d, t
        return {"min_distance": round(min_dist, 2), "time_to_closest": round(min_t, 2)}

    def add_zone(self, zone_id: str, name: str, polygon: list[tuple[float, float]],
                 zone_type: str = "user"):
        self._zones[zone_id] = Zone(zone_id=zone_id, name=name, polygon=polygon, zone_type=zone_type)

    def auto_detect_zones(self, min_cluster_size: int = 100) -> list[Zone]:
        try:
            from scipy import ndimage
        except ImportError:
            logger.warning("scipy not available for zone detection")
            return []

        threshold = self._movement_heatmap.mean() + self._movement_heatmap.std()
        binary = self._movement_heatmap > threshold
        labeled, n_features = ndimage.label(binary)

        zones = []
        bx = self._heatmap_bounds
        for i in range(1, n_features + 1):
            cluster = np.argwhere(labeled == i)
            if len(cluster) < min_cluster_size // 10:
                continue
            y_min, x_min = cluster.min(axis=0)
            y_max, x_max = cluster.max(axis=0)
            sx, sy = (bx[2] - bx[0]) / 49, (bx[3] - bx[1]) / 49
            polygon = [
                (bx[0] + x_min * sx, bx[1] + y_min * sy),
                (bx[0] + x_max * sx, bx[1] + y_min * sy),
                (bx[0] + x_max * sx, bx[1] + y_max * sy),
                (bx[0] + x_min * sx, bx[1] + y_max * sy),
            ]
            z = Zone(zone_id=f"auto-zone-{i}", name=f"Activity Zone {i}",
                     polygon=polygon, zone_type="auto")
            self._zones[z.zone_id] = z
            zones.append(z)
        return zones

    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
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

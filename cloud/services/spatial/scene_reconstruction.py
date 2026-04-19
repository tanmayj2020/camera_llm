"""Multi-Camera 3D Scene Reconstruction — build real-time 3D model from multiple views.

Lightweight point-cloud fusion from MiDaS depth maps across calibrated cameras.
Not full NeRF — practical real-time reconstruction for security use cases.
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PointCloud:
    points: np.ndarray  # (N, 3) xyz
    colors: np.ndarray  # (N, 3) rgb
    camera_ids: list[str] = field(default_factory=list)
    timestamp: float = 0.0


class SceneReconstructor:
    """Fuses depth maps from multiple cameras into a unified 3D point cloud."""

    def __init__(self):
        self._camera_transforms: dict[str, dict] = {}  # camera_id -> {R, t, K}
        self._point_cloud: PointCloud | None = None
        self._voxel_size = 0.1  # 10cm voxel grid for dedup

    def register_camera(self, camera_id: str, position: tuple[float, float, float],
                        rotation_deg: tuple[float, float, float],
                        focal_length: float = 800, img_w: int = 1920, img_h: int = 1080):
        """Register camera extrinsics and intrinsics."""
        rx, ry, rz = [np.radians(a) for a in rotation_deg]
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        K = np.array([[focal_length, 0, img_w / 2], [0, focal_length, img_h / 2], [0, 0, 1]])
        self._camera_transforms[camera_id] = {
            "R": R, "t": np.array(position), "K": K, "w": img_w, "h": img_h}

    def add_depth_frame(self, camera_id: str, depth_map: np.ndarray,
                        color_frame: np.ndarray = None, subsample: int = 8) -> int:
        """Project depth map to 3D points in world coordinates."""
        cam = self._camera_transforms.get(camera_id)
        if cam is None:
            return 0

        h, w = depth_map.shape[:2]
        K_inv = np.linalg.inv(cam["K"][:2, :2])

        # Subsample for performance
        ys = np.arange(0, h, subsample)
        xs = np.arange(0, w, subsample)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        pixels = np.stack([xx.flatten(), yy.flatten()], axis=1).astype(float)
        depths = depth_map[yy.flatten(), xx.flatten()]

        # Filter invalid depths
        valid = (depths > 0.1) & (depths < 50.0)
        pixels = pixels[valid]
        depths = depths[valid]

        if len(pixels) == 0:
            return 0

        # Unproject: pixel + depth → camera coords
        centered = pixels - np.array([cam["K"][0, 2], cam["K"][1, 2]])
        cam_xy = centered * depths[:, None] / cam["K"][0, 0]
        cam_points = np.column_stack([cam_xy, depths])

        # Camera → world
        world_points = (cam["R"].T @ (cam_points - cam["t"]).T).T

        # Colors
        if color_frame is not None:
            colors = color_frame[yy.flatten()[valid], xx.flatten()[valid]]
            if colors.ndim == 2 and colors.shape[1] >= 3:
                colors = colors[:, :3]
            else:
                colors = np.full((len(world_points), 3), 128, dtype=np.uint8)
        else:
            colors = np.full((len(world_points), 3), 128, dtype=np.uint8)

        # Merge with existing point cloud
        if self._point_cloud is None:
            self._point_cloud = PointCloud(world_points, colors, [camera_id], time.time())
        else:
            self._point_cloud.points = np.vstack([self._point_cloud.points, world_points])
            self._point_cloud.colors = np.vstack([self._point_cloud.colors, colors])
            if camera_id not in self._point_cloud.camera_ids:
                self._point_cloud.camera_ids.append(camera_id)
            self._point_cloud.timestamp = time.time()

        # Voxel grid downsample to prevent unbounded growth
        self._voxel_downsample()
        return len(world_points)

    def _voxel_downsample(self):
        if self._point_cloud is None or len(self._point_cloud.points) < 10000:
            return
        pts = self._point_cloud.points
        voxel_indices = (pts / self._voxel_size).astype(int)
        _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
        self._point_cloud.points = pts[unique_idx]
        self._point_cloud.colors = self._point_cloud.colors[unique_idx]

    def get_point_cloud(self) -> dict | None:
        if self._point_cloud is None:
            return None
        return {
            "num_points": len(self._point_cloud.points),
            "cameras": self._point_cloud.camera_ids,
            "timestamp": self._point_cloud.timestamp,
            "bounds_min": self._point_cloud.points.min(axis=0).tolist(),
            "bounds_max": self._point_cloud.points.max(axis=0).tolist(),
        }

    def get_points_in_region(self, center: tuple[float, float, float],
                             radius: float = 5.0) -> np.ndarray:
        if self._point_cloud is None:
            return np.array([])
        dists = np.linalg.norm(self._point_cloud.points - np.array(center), axis=1)
        return self._point_cloud.points[dists < radius]

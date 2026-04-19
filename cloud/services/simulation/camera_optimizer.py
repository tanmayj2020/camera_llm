"""Camera Placement Optimizer — AI-recommended optimal camera positions."""

import logging
import math
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraPlacement:
    position: tuple[float, float]
    rotation_deg: float
    fov_deg: float
    coverage_area_m2: float
    coverage_gain_pct: float
    reason: str


@dataclass
class OptimizationResult:
    total_coverage_before: float
    total_coverage_after: float
    recommendations: list[CameraPlacement] = field(default_factory=list)
    blind_spots: list[tuple[float, float]] = field(default_factory=list)


class CameraPlacementOptimizer:
    """Recommends optimal camera positions given floor plan and coverage requirements."""

    def __init__(self, site_width: float = 50, site_height: float = 50, grid_resolution: float = 1.0):
        self._width = site_width
        self._height = site_height
        self._res = grid_resolution
        self._existing_cameras: list[dict] = []
        self._obstacles: list[tuple[float, float, float, float]] = []  # (x, y, w, h) walls

    def add_existing_camera(self, x: float, y: float, rotation_deg: float = 0,
                            fov_deg: float = 60, reach_m: float = 20):
        self._existing_cameras.append({"x": x, "y": y, "rot": rotation_deg,
                                        "fov": fov_deg, "reach": reach_m})

    def add_obstacle(self, x: float, y: float, w: float, h: float):
        self._obstacles.append((x, y, w, h))

    def _compute_coverage_grid(self, cameras: list[dict]) -> np.ndarray:
        gw = int(self._width / self._res)
        gh = int(self._height / self._res)
        grid = np.zeros((gh, gw), dtype=bool)

        for cam in cameras:
            cx, cy = cam["x"], cam["y"]
            fov = math.radians(cam["fov"])
            rot = math.radians(cam["rot"])
            reach = cam["reach"]

            for gy in range(gh):
                for gx in range(gw):
                    px = gx * self._res + self._res / 2
                    py = gy * self._res + self._res / 2
                    dx, dy = px - cx, py - cy
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist > reach or dist < 0.5:
                        continue
                    angle = math.atan2(dy, dx)
                    diff = abs(angle - rot)
                    diff = min(diff, 2 * math.pi - diff)
                    if diff <= fov / 2:
                        # Check obstacle occlusion (simplified)
                        blocked = False
                        for ox, oy, ow, oh in self._obstacles:
                            if ox <= px <= ox + ow and oy <= py <= oy + oh:
                                blocked = True
                                break
                        if not blocked:
                            grid[gy, gx] = True
        return grid

    def optimize(self, num_cameras: int = 1, fov_deg: float = 60,
                 reach_m: float = 20) -> OptimizationResult:
        current_grid = self._compute_coverage_grid(self._existing_cameras)
        total_cells = current_grid.size
        before_pct = round(float(current_grid.sum()) / total_cells * 100, 1)

        # Find blind spots (uncovered cells)
        uncovered = np.argwhere(~current_grid)
        if len(uncovered) == 0:
            return OptimizationResult(before_pct, before_pct, [],
                                      blind_spots=[])

        recommendations = []
        test_cameras = list(self._existing_cameras)

        for _ in range(num_cameras):
            best_pos, best_rot, best_gain = None, 0, 0

            # Sample candidate positions at blind spot centroids
            if len(uncovered) > 0:
                # Cluster blind spots and try centroid of each cluster
                n_candidates = min(20, len(uncovered))
                indices = np.random.choice(len(uncovered), n_candidates, replace=False)
                candidates = uncovered[indices]
            else:
                break

            for cy, cx in candidates:
                px = cx * self._res + self._res / 2
                py = cy * self._res + self._res / 2
                for rot_deg in range(0, 360, 45):
                    test_cam = {"x": px, "y": py, "rot": rot_deg, "fov": fov_deg, "reach": reach_m}
                    test_grid = self._compute_coverage_grid(test_cameras + [test_cam])
                    gain = float(test_grid.sum() - current_grid.sum()) / total_cells * 100
                    if gain > best_gain:
                        best_gain = gain
                        best_pos = (px, py)
                        best_rot = rot_deg

            if best_pos and best_gain > 0:
                area = math.pi * reach_m**2 * (fov_deg / 360)
                rec = CameraPlacement(best_pos, best_rot, fov_deg, round(area, 1),
                                       round(best_gain, 1),
                                       f"Covers {best_gain:.1f}% additional area")
                recommendations.append(rec)
                new_cam = {"x": best_pos[0], "y": best_pos[1], "rot": best_rot,
                           "fov": fov_deg, "reach": reach_m}
                test_cameras.append(new_cam)
                current_grid = self._compute_coverage_grid(test_cameras)
                uncovered = np.argwhere(~current_grid)

        after_pct = round(float(current_grid.sum()) / total_cells * 100, 1)
        blind_spots = [(float(c * self._res), float(r * self._res))
                       for r, c in uncovered[:20]]

        return OptimizationResult(before_pct, after_pct, recommendations, blind_spots)

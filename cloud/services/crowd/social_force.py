"""Social Force Model for crowd dynamics analysis."""

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

REPULSION_STRENGTH = 2.0
REPULSION_RANGE = 0.5
REPULSION_THRESHOLD = 2.0


def _persons(spatial) -> dict:
    return {k: v for k, v in spatial._entities.items() if v.class_name == "person"}


def _dist(a, b) -> float:
    return float(np.linalg.norm(a.position - b.position))


class SocialForceModel:
    """Computes repulsive social forces, obstacle repulsion, and goal attraction."""

    def compute_forces(self, spatial) -> dict[str, np.ndarray]:
        persons = _persons(spatial)
        ids = list(persons.keys())
        forces: dict[str, np.ndarray] = {k: np.zeros(3) for k in ids}
        for i, a_id in enumerate(ids):
            for b_id in ids[i + 1:]:
                a, b = persons[a_id], persons[b_id]
                diff = a.position - b.position
                d = float(np.linalg.norm(diff))
                if 0 < d < REPULSION_THRESHOLD:
                    direction = diff / d
                    mag = REPULSION_STRENGTH * np.exp(-d / REPULSION_RANGE)
                    forces[a_id] += mag * direction
                    forces[b_id] -= mag * direction
        return forces

    def compute_obstacle_forces(self, spatial) -> dict[str, np.ndarray]:
        """Repulsive forces from zone boundaries (obstacles/walls)."""
        persons = _persons(spatial)
        forces: dict[str, np.ndarray] = {k: np.zeros(3) for k in persons}
        zones = getattr(spatial, "_zones", {})
        for pid, person in persons.items():
            pos2d = np.array([person.position[0], person.position[2]])
            for zone in zones.values():
                for i in range(len(zone.polygon)):
                    a = np.array(zone.polygon[i])
                    b = np.array(zone.polygon[(i + 1) % len(zone.polygon)])
                    # Point-to-segment distance
                    ab = b - a
                    seg_len_sq = np.dot(ab, ab)
                    if seg_len_sq < 1e-6:
                        continue
                    t = np.clip(np.dot(pos2d - a, ab) / seg_len_sq, 0, 1)
                    closest = a + t * ab
                    diff = pos2d - closest
                    d = float(np.linalg.norm(diff))
                    if 0 < d < 2.0:
                        direction = diff / d
                        mag = REPULSION_STRENGTH * 0.5 * np.exp(-d / REPULSION_RANGE)
                        forces[pid][:2] += mag * direction[:2] if len(direction) >= 2 else 0
        return forces

    def compute_goal_forces(self, spatial) -> dict[str, np.ndarray]:
        """Attractive forces toward inferred goal (velocity direction extrapolation)."""
        persons = _persons(spatial)
        forces: dict[str, np.ndarray] = {k: np.zeros(3) for k in persons}
        GOAL_STRENGTH = 1.0
        DESIRED_SPEED = 1.3  # m/s typical walking speed
        for pid, person in persons.items():
            speed = float(np.linalg.norm(person.velocity))
            if speed > 0.1:
                desired_dir = person.velocity / speed
                forces[pid] = GOAL_STRENGTH * (DESIRED_SPEED * desired_dir - person.velocity)
        return forces


class LaneDetector:
    """Detects counter-flow lanes from velocity clustering."""

    def detect_lanes(self, spatial, min_lane_size: int = 3) -> list[dict]:
        persons = _persons(spatial)
        if len(persons) < 4:
            return []
        ids = list(persons.keys())
        velocities = np.array([persons[k].velocity[:2] for k in ids])
        speeds = np.linalg.norm(velocities, axis=1)
        moving = speeds > 0.3
        if moving.sum() < 4:
            return []

        # Cluster by velocity angle into 2 dominant directions
        angles = np.arctan2(velocities[moving, 1], velocities[moving, 0])
        moving_ids = [ids[i] for i in range(len(ids)) if moving[i]]

        # Simple 2-cluster: positive vs negative primary direction
        median_angle = np.median(angles)
        lane_a = [mid for mid, a in zip(moving_ids, angles) if a >= median_angle]
        lane_b = [mid for mid, a in zip(moving_ids, angles) if a < median_angle]

        lanes = []
        for i, members in enumerate([lane_a, lane_b]):
            if len(members) >= min_lane_size:
                positions = np.array([persons[m].position for m in members])
                vels = np.array([persons[m].velocity for m in members])
                lanes.append({
                    "lane_id": f"lane_{i}",
                    "members": members,
                    "centroid": np.mean(positions, axis=0).tolist(),
                    "avg_velocity": np.mean(vels, axis=0).tolist(),
                    "size": len(members),
                })
        return lanes


class GroupDetector:
    """Detects groups of people moving together."""

    def __init__(self, min_duration: float = 10.0):
        self.min_duration = min_duration
        self._group_candidates: dict[tuple[str, str], float] = {}

    def detect_groups(self, spatial) -> list[dict[str, Any]]:
        persons = _persons(spatial)
        ids = sorted(persons.keys())
        now = time.time()
        current_pairs: set[tuple[str, str]] = set()

        for i, a_id in enumerate(ids):
            for b_id in ids[i + 1:]:
                a, b = persons[a_id], persons[b_id]
                if _dist(a, b) >= 3.0:
                    continue
                va, vb = np.linalg.norm(a.velocity), np.linalg.norm(b.velocity)
                if va > 0 and vb > 0:
                    cos_sim = float(np.dot(a.velocity, b.velocity) / (va * vb))
                    if cos_sim <= 0.7:
                        continue
                else:
                    continue
                pair = (a_id, b_id)
                current_pairs.add(pair)
                self._group_candidates.setdefault(pair, now)

        # prune stale pairs
        self._group_candidates = {p: t for p, t in self._group_candidates.items() if p in current_pairs}

        # build groups from mature pairs via union-find
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for (a_id, b_id), first_seen in self._group_candidates.items():
            if now - first_seen >= self.min_duration:
                parent.setdefault(a_id, a_id)
                parent.setdefault(b_id, b_id)
                ra, rb = find(a_id), find(b_id)
                if ra != rb:
                    parent[ra] = rb

        clusters: dict[str, list[str]] = {}
        for eid in parent:
            clusters.setdefault(find(eid), []).append(eid)

        groups = []
        for idx, members in enumerate(clusters.values()):
            if len(members) < 2:
                continue
            positions = np.array([persons[m].position for m in members])
            velocities = np.array([persons[m].velocity for m in members])
            groups.append({
                "group_id": f"group_{idx}",
                "members": members,
                "centroid": np.mean(positions, axis=0).tolist(),
                "velocity": np.mean(velocities, axis=0).tolist(),
            })
        return groups


class CrowdPressureMonitor:
    """Monitors crowd pressure and stampede risk."""

    def __init__(self, density_threshold: float = 0.5, speed_threshold: float = 1.5):
        self.density_threshold = density_threshold
        self.speed_threshold = speed_threshold

    def compute_pressure(self, spatial) -> dict[str, Any]:
        persons = _persons(spatial)
        if not persons:
            return {"density": 0.0, "avg_speed": 0.0, "pressure": 0.0, "stampede_risk": False, "hotspots": []}

        positions = np.array([e.position for e in persons.values()])
        speeds = np.array([float(np.linalg.norm(e.velocity)) for e in persons.values()])
        avg_speed = float(np.mean(speeds))

        # grid-based density (5m cells)
        cell_size = 5.0
        cells = (positions[:, :2] // cell_size).astype(int)
        cell_counts: dict[tuple[int, int], int] = {}
        for c in cells:
            key = (int(c[0]), int(c[1]))
            cell_counts[key] = cell_counts.get(key, 0) + 1

        cell_area = cell_size ** 2
        densities = [cnt / cell_area for cnt in cell_counts.values()]
        density = float(np.mean(densities)) if densities else 0.0
        pressure = density * avg_speed

        hotspots = [list(k) for k, cnt in cell_counts.items() if cnt / cell_area > self.density_threshold]

        return {
            "density": density,
            "avg_speed": avg_speed,
            "pressure": pressure,
            "stampede_risk": pressure > self.density_threshold * self.speed_threshold,
            "hotspots": hotspots,
        }


def analyze_crowd(spatial) -> dict[str, Any]:
    """Combined crowd analysis: forces, groups, pressure, lanes."""
    sfm = SocialForceModel()
    forces = sfm.compute_forces(spatial)
    obstacle_forces = sfm.compute_obstacle_forces(spatial)
    goal_forces = sfm.compute_goal_forces(spatial)
    force_summary = {k: float(np.linalg.norm(v)) for k, v in forces.items()}

    gd = GroupDetector()
    groups = gd.detect_groups(spatial)

    cpm = CrowdPressureMonitor()
    pressure = cpm.compute_pressure(spatial)

    ld = LaneDetector()
    lanes = ld.detect_lanes(spatial)

    return {
        "forces": force_summary,
        "obstacle_forces": {k: float(np.linalg.norm(v)) for k, v in obstacle_forces.items()},
        "goal_forces": {k: float(np.linalg.norm(v)) for k, v in goal_forces.items()},
        "groups": groups,
        "pressure": pressure,
        "lanes": lanes,
    }

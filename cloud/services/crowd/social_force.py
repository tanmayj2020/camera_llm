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
    """Computes repulsive social forces between pedestrians."""

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
    """Combined crowd analysis: forces, groups, pressure."""
    sfm = SocialForceModel()
    forces = sfm.compute_forces(spatial)
    force_summary = {k: float(np.linalg.norm(v)) for k, v in forces.items()}

    gd = GroupDetector()
    groups = gd.detect_groups(spatial)

    cpm = CrowdPressureMonitor()
    pressure = cpm.compute_pressure(spatial)

    return {"forces": force_summary, "groups": groups, "pressure": pressure}

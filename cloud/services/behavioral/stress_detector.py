"""Micro-expression and behavioral stress detection from spatial movement patterns."""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StressIndicator:
    indicator_type: str
    entity_id: str
    confidence: float
    timestamp: float
    description: str


@dataclass
class StressAssessment:
    entity_id: str
    stress_level: float
    indicators: list[StressIndicator]
    recommendation: str
    timestamp: float


class StressDetector:
    def __init__(self, spatial=None):
        self._spatial = spatial
        self._history: dict[str, deque] = defaultdict(lambda: deque(maxlen=60))
        self._assessments: dict[str, StressAssessment] = {}

    def update(self, entity_id, position, velocity, timestamp):
        self._history[entity_id].append({"pos": list(position), "vel": list(velocity), "t": timestamp})

    def _detect_fidgeting(self, entity_id) -> StressIndicator | None:
        hist = self._history.get(entity_id)
        if not hist or len(hist) < 10:
            return None
        positions = np.array([h["pos"][:2] for h in hist])
        std = float(positions.std(axis=0).mean())
        displacement = float(np.linalg.norm(positions[-1] - positions[0]))
        if std > 0.3 and displacement < 1.0:
            return StressIndicator("fidgeting", entity_id, min(std / 0.5, 1.0), time.time(),
                                   f"High-frequency small movements (std={std:.2f}, disp={displacement:.2f}m)")
        return None

    def _detect_pacing(self, entity_id) -> StressIndicator | None:
        hist = self._history.get(entity_id)
        if not hist or len(hist) < 15:
            return None
        vels = [h["vel"][0] for h in hist]
        reversals = sum(1 for i in range(1, len(vels)) if vels[i] * vels[i - 1] < 0)
        if reversals > 3:
            return StressIndicator("pacing", entity_id, min(reversals / 6, 1.0), time.time(),
                                   f"Back-and-forth movement ({reversals} reversals)")
        return None

    def _detect_erratic(self, entity_id) -> StressIndicator | None:
        hist = self._history.get(entity_id)
        if not hist or len(hist) < 10:
            return None
        speeds = np.array([np.linalg.norm(h["vel"]) for h in hist])
        mean_s, std_s = float(speeds.mean()), float(speeds.std())
        if mean_s > 0.1 and std_s > mean_s * 2:
            return StressIndicator("erratic_movement", entity_id, min(std_s / mean_s / 3, 1.0), time.time(),
                                   f"Sudden speed changes (mean={mean_s:.2f}, std={std_s:.2f})")
        return None

    def assess(self, entity_id) -> StressAssessment:
        indicators = [i for i in [self._detect_fidgeting(entity_id),
                                   self._detect_pacing(entity_id),
                                   self._detect_erratic(entity_id)] if i]
        level = sum(i.confidence for i in indicators) / max(len(indicators), 1) * (len(indicators) / 3)
        if level < 0.3:
            rec = "normal behavior"
        elif level < 0.6:
            rec = "elevated stress indicators"
        else:
            rec = "appears distressed — may need assistance"
        a = StressAssessment(entity_id, round(level, 2), indicators, rec, time.time())
        self._assessments[entity_id] = a
        return a

    def evaluate(self, scene_state) -> list[StressAssessment]:
        spatial = scene_state.get("spatial") or self._spatial
        if not spatial:
            return []
        results = []
        for eid, ent in getattr(spatial, "_entities", {}).items():
            if ent.class_name == "person":
                self.update(eid, ent.position, ent.velocity, scene_state.get("timestamp", time.time()))
                a = self.assess(eid)
                if a.stress_level > 0.3:
                    results.append(a)
        return results

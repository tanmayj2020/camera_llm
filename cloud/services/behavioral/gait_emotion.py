"""Gait-based emotion/sentiment estimation from walking patterns and posture."""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GaitEmotion:
    entity_id: str
    emotion: str  # angry, fearful, happy, sad, neutral
    confidence: float
    features: dict
    timestamp: float


class GaitEmotionEstimator:
    """Estimates emotional state from gait features: stride length, speed, arm swing, posture."""

    def __init__(self):
        self._history: dict[str, deque] = defaultdict(lambda: deque(maxlen=60))

    def update(self, entity_id: str, position: np.ndarray, velocity: np.ndarray,
               pose_keypoints: list | None, timestamp: float):
        self._history[entity_id].append({
            "pos": position.copy(), "vel": velocity.copy(),
            "pose": pose_keypoints, "t": timestamp})

    def estimate(self, entity_id: str) -> GaitEmotion | None:
        hist = self._history.get(entity_id)
        if not hist or len(hist) < 15:
            return None

        features = self._extract_features(list(hist))
        emotion, confidence = self._classify(features)
        return GaitEmotion(entity_id, emotion, confidence, features, time.time())

    def _extract_features(self, hist: list[dict]) -> dict:
        speeds = [float(np.linalg.norm(h["vel"])) for h in hist]
        positions = np.array([h["pos"][:2] for h in hist])

        # Stride regularity: std of step-to-step distances
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        stride_regularity = float(np.std(steps)) / (float(np.mean(steps)) + 1e-6)

        # Speed variability
        speed_mean = float(np.mean(speeds))
        speed_var = float(np.std(speeds)) / (speed_mean + 1e-6)

        # Path directness: displacement / total path length
        displacement = float(np.linalg.norm(positions[-1] - positions[0]))
        path_length = float(np.sum(steps))
        directness = displacement / (path_length + 1e-6)

        # Posture: shoulder-hip vertical ratio (upright vs slouched)
        posture_score = 0.5
        for h in hist[-5:]:
            if h["pose"] and len(h["pose"]) > 12:
                try:
                    ls = h["pose"][5]  # left shoulder
                    lh = h["pose"][11]  # left hip
                    if isinstance(ls, (list, tuple)) and isinstance(lh, (list, tuple)):
                        torso_height = abs(ls[1] - lh[1])
                        torso_width = 50  # approximate
                        posture_score = min(1.0, torso_height / (torso_width + 1e-6) / 3)
                except (IndexError, TypeError):
                    pass

        return {
            "speed_mean": round(speed_mean, 3),
            "speed_variability": round(speed_var, 3),
            "stride_regularity": round(stride_regularity, 3),
            "directness": round(directness, 3),
            "posture_score": round(posture_score, 3),
        }

    def _classify(self, f: dict) -> tuple[str, float]:
        """Rule-based gait emotion classification from research literature."""
        speed = f["speed_mean"]
        var = f["speed_variability"]
        reg = f["stride_regularity"]
        direct = f["directness"]
        posture = f["posture_score"]

        # Angry: fast, direct, irregular strides, high speed variance
        if speed > 1.5 and direct > 0.7 and var > 0.3:
            return "angry", min(0.8, (speed / 2 + var) / 2)

        # Fearful: fast, indirect (evasive), high variability
        if speed > 1.2 and direct < 0.5 and var > 0.4:
            return "fearful", min(0.75, var)

        # Sad: slow, low posture, regular but short strides
        if speed < 0.8 and posture < 0.4:
            return "sad", min(0.7, 1 - speed)

        # Happy: moderate-fast, regular strides, upright posture
        if speed > 0.9 and reg < 0.3 and posture > 0.6:
            return "happy", min(0.7, posture)

        return "neutral", 0.5

    def evaluate_all(self, spatial, timestamp: float = 0) -> list[GaitEmotion]:
        results = []
        for eid, ent in getattr(spatial, "_entities", {}).items():
            if ent.class_name == "person":
                self.update(eid, ent.position, ent.velocity, None, timestamp or time.time())
                r = self.estimate(eid)
                if r and r.emotion != "neutral":
                    results.append(r)
        return results

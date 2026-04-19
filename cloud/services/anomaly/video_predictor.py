"""Scene Anomaly via Video Prediction Divergence — predict expected scene state,
compare with actual observations, flag divergence as anomaly.

Uses world model trajectories + baseline patterns to predict expected
object counts, positions, activity levels. Anomaly = divergence between
predicted and actual.
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ScenePrediction:
    """What the model expects to see in the next frame."""
    timestamp: float
    camera_id: str
    expected_entity_count: int = 0
    expected_positions: dict = field(default_factory=dict)  # track_id → [x,y,z]
    expected_activity: float = 0.0
    expected_classes: dict = field(default_factory=dict)  # class_name → count


@dataclass
class DivergenceScore:
    """How much the actual scene diverges from prediction."""
    timestamp: float
    camera_id: str
    overall: float = 0.0  # 0-1, higher = more anomalous
    count_divergence: float = 0.0
    position_divergence: float = 0.0
    activity_divergence: float = 0.0
    class_divergence: float = 0.0
    details: dict = field(default_factory=dict)


class SceneAnomalyPredictor:
    """Predicts expected scene state and scores divergence from actual observations.

    Combines:
    - World model trajectory predictions (where entities should be)
    - Baseline temporal patterns (how many entities expected at this hour)
    - Activity level expectations
    - Class distribution expectations

    Divergence between predicted and actual = anomaly signal.
    """

    # Weights for combining divergence dimensions
    WEIGHTS = {
        "count": 0.3,
        "position": 0.25,
        "activity": 0.25,
        "class": 0.2,
    }

    def __init__(self, world_model=None, baseline=None):
        self._world_model = world_model
        self._baseline = baseline
        self._recent_predictions: list[ScenePrediction] = []
        self._recent_scores: list[DivergenceScore] = []

    def predict_scene(self, camera_id: str, spatial=None,
                      timestamp: float = None) -> ScenePrediction:
        """Predict what the scene should look like in the next observation."""
        ts = timestamp or time.time()
        pred = ScenePrediction(timestamp=ts, camera_id=camera_id)

        # Predict entity positions from world model trajectories
        if self._world_model:
            for track_id, history in self._world_model._trajectory_history.items():
                if len(history) < 3:
                    continue
                p = self._world_model.predict_trajectory(track_id, horizon_s=1.0)
                if p:
                    # Extract predicted position from description
                    pred.expected_positions[track_id] = True
            pred.expected_entity_count = len(pred.expected_positions)

        # Predict from baseline temporal patterns
        if self._baseline:
            hour = int((ts % 86400) / 3600)
            bl = self._baseline._zone_baselines.get(camera_id)
            if bl and bl.samples > 10:
                pred.expected_entity_count = max(
                    pred.expected_entity_count,
                    int(bl.hourly_counts[hour]),
                )
                pred.expected_activity = bl.activity_mean

        # Predict class distribution from recent history
        if spatial and hasattr(spatial, '_entities'):
            class_counts: dict[str, int] = {}
            for ent in spatial._entities.values():
                class_counts[ent.class_name] = class_counts.get(ent.class_name, 0) + 1
            pred.expected_classes = class_counts

        self._recent_predictions.append(pred)
        if len(self._recent_predictions) > 100:
            self._recent_predictions = self._recent_predictions[-50:]

        return pred

    def score_divergence(self, prediction: ScenePrediction,
                         actual_event: dict, spatial=None) -> DivergenceScore:
        """Compare predicted scene with actual observation, return divergence score."""
        score = DivergenceScore(
            timestamp=actual_event.get("timestamp", time.time()),
            camera_id=prediction.camera_id,
        )

        actual_objects = actual_event.get("objects", [])
        actual_count = len(actual_objects)
        actual_activity = actual_event.get("scene_activity", 0.0)

        # 1. Count divergence
        expected_count = max(prediction.expected_entity_count, 1)
        count_diff = abs(actual_count - expected_count) / expected_count
        score.count_divergence = min(1.0, count_diff)

        # 2. Position divergence — how far are entities from predicted positions
        if spatial and prediction.expected_positions and hasattr(spatial, '_entities'):
            position_errors = []
            for track_id in prediction.expected_positions:
                ent = spatial._entities.get(track_id)
                if ent is not None:
                    # Entity exists but may have moved unexpectedly
                    speed = float(np.linalg.norm(ent.velocity))
                    if speed > 3.0:  # unusually fast movement
                        position_errors.append(min(1.0, speed / 5.0))
            if position_errors:
                score.position_divergence = float(np.mean(position_errors))

            # New entities that weren't predicted
            actual_ids = {str(o.get("track_id", "")) for o in actual_objects}
            predicted_ids = set(prediction.expected_positions.keys())
            new_entities = actual_ids - predicted_ids - {"", "-1"}
            if new_entities and predicted_ids:
                score.position_divergence = max(
                    score.position_divergence,
                    min(1.0, len(new_entities) / max(len(predicted_ids), 1)),
                )

        # 3. Activity divergence
        if prediction.expected_activity > 0:
            act_diff = abs(actual_activity - prediction.expected_activity)
            score.activity_divergence = min(1.0, act_diff / max(prediction.expected_activity, 0.1))

        # 4. Class distribution divergence
        actual_classes: dict[str, int] = {}
        for obj in actual_objects:
            cls = obj.get("class_name", "unknown")
            actual_classes[cls] = actual_classes.get(cls, 0) + 1

        if prediction.expected_classes:
            all_classes = set(prediction.expected_classes) | set(actual_classes)
            class_diffs = []
            for cls in all_classes:
                expected = prediction.expected_classes.get(cls, 0)
                actual = actual_classes.get(cls, 0)
                if expected > 0:
                    class_diffs.append(abs(actual - expected) / expected)
                elif actual > 0:
                    class_diffs.append(1.0)  # unexpected class
            if class_diffs:
                score.class_divergence = min(1.0, float(np.mean(class_diffs)))

        # Overall weighted score
        score.overall = (
            self.WEIGHTS["count"] * score.count_divergence +
            self.WEIGHTS["position"] * score.position_divergence +
            self.WEIGHTS["activity"] * score.activity_divergence +
            self.WEIGHTS["class"] * score.class_divergence
        )
        score.overall = round(min(1.0, score.overall), 3)

        score.details = {
            "expected_count": prediction.expected_entity_count,
            "actual_count": actual_count,
            "expected_activity": round(prediction.expected_activity, 3),
            "actual_activity": round(actual_activity, 3),
        }

        self._recent_scores.append(score)
        if len(self._recent_scores) > 200:
            self._recent_scores = self._recent_scores[-100:]

        return score

    def predict_and_score(self, camera_id: str, event: dict,
                          spatial=None) -> DivergenceScore:
        """Convenience: predict + score in one call. Integrates into event pipeline."""
        prediction = self.predict_scene(camera_id, spatial, event.get("timestamp"))
        return self.score_divergence(prediction, event, spatial)

    def get_recent_scores(self, camera_id: str = None, limit: int = 20) -> list[dict]:
        scores = self._recent_scores
        if camera_id:
            scores = [s for s in scores if s.camera_id == camera_id]
        return [
            {"timestamp": s.timestamp, "camera_id": s.camera_id,
             "overall": s.overall, "count": s.count_divergence,
             "position": s.position_divergence, "activity": s.activity_divergence,
             "class": s.class_divergence, "details": s.details}
            for s in scores[-limit:]
        ]

"""Task 10: World Model — predictive/anticipatory intelligence."""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    prediction_type: str  # "trajectory", "crowd", "collision", "pattern"
    description: str
    confidence: float
    time_horizon_s: float
    entities_involved: list[str]
    recommended_action: str = ""


class WorldModel:
    """Predicts future events from historical patterns in the knowledge graph.

    - Trajectory prediction via linear/polynomial extrapolation
    - Crowd dynamics prediction from flow rates
    - Behavioral sequence prediction from event chains
    """

    def __init__(self, history_window: int = 100):
        self._trajectory_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=history_window))
        self._zone_flow_rates: dict[str, deque] = defaultdict(lambda: deque(maxlen=60))  # per-minute counts
        self._event_sequences: deque = deque(maxlen=1000)

    def update_trajectory(self, track_id: str, position: np.ndarray, timestamp: float):
        self._trajectory_history[track_id].append((timestamp, position.copy()))

    def update_zone_count(self, zone_id: str, count: int, timestamp: float):
        self._zone_flow_rates[zone_id].append((timestamp, count))

    def record_event(self, event_type: str, timestamp: float):
        self._event_sequences.append((timestamp, event_type))

    def predict_trajectory(self, track_id: str, horizon_s: float = 10.0) -> Prediction | None:
        """Predict where an entity will be in `horizon_s` seconds."""
        history = self._trajectory_history.get(track_id)
        if not history or len(history) < 3:
            return None

        times = np.array([t for t, _ in history])
        positions = np.array([p for _, p in history])

        # Linear regression for each axis
        t_rel = times - times[0]
        predicted_pos = []
        t_future = t_rel[-1] + horizon_s

        for axis in range(positions.shape[1]):
            if len(t_rel) >= 3:
                coeffs = np.polyfit(t_rel, positions[:, axis], deg=min(2, len(t_rel) - 1))
                predicted_pos.append(float(np.polyval(coeffs, t_future)))
            else:
                predicted_pos.append(float(positions[-1, axis]))

        return Prediction(
            prediction_type="trajectory",
            description=f"Entity {track_id} predicted at ({predicted_pos[0]:.1f}, {predicted_pos[2]:.1f})m in {horizon_s}s",
            confidence=min(0.9, len(history) / 50),
            time_horizon_s=horizon_s,
            entities_involved=[track_id],
        )

    def predict_crowd(self, zone_id: str, horizon_minutes: float = 5.0) -> Prediction | None:
        """Predict crowd size in a zone based on flow rate trends."""
        rates = self._zone_flow_rates.get(zone_id)
        if not rates or len(rates) < 5:
            return None

        counts = np.array([c for _, c in rates])
        times = np.array([t for t, _ in rates])

        # Linear trend
        t_rel = (times - times[0]) / 60.0  # minutes
        if len(t_rel) < 2:
            return None
        slope = np.polyfit(t_rel, counts, 1)[0]
        current = counts[-1]
        predicted = current + slope * horizon_minutes

        if predicted > current * 1.5 and predicted > 10:
            return Prediction(
                prediction_type="crowd",
                description=f"Zone {zone_id}: crowd predicted to grow from {current:.0f} to {predicted:.0f} in {horizon_minutes:.0f}min",
                confidence=min(0.8, len(rates) / 30),
                time_horizon_s=horizon_minutes * 60,
                entities_involved=[zone_id],
                recommended_action="Consider crowd management measures",
            )
        return None

    def predict_collision(self, spatial_memory, horizon_s: float = 5.0,
                          danger_distance: float = 2.0) -> list[Prediction]:
        """Check all entity pairs for potential collisions."""
        predictions = []
        entities = list(spatial_memory._entities.values())

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                a, b = entities[i], entities[j]
                result = spatial_memory.predict_collision(a.track_id, b.track_id, horizon_s)
                if result and result["min_distance"] < danger_distance:
                    predictions.append(Prediction(
                        prediction_type="collision",
                        description=(
                            f"{a.class_name}({a.track_id}) and {b.class_name}({b.track_id}) "
                            f"predicted within {result['min_distance']:.1f}m in {result['time_to_closest']:.1f}s"
                        ),
                        confidence=0.7,
                        time_horizon_s=result["time_to_closest"],
                        entities_involved=[a.track_id, b.track_id],
                        recommended_action="Potential collision — alert nearby personnel",
                    ))
        return predictions

    def get_all_predictions(self, spatial_memory=None) -> list[Prediction]:
        """Run all prediction models and return combined results."""
        predictions = []

        # Trajectory predictions for all tracked entities
        for track_id in list(self._trajectory_history.keys()):
            p = self.predict_trajectory(track_id)
            if p:
                predictions.append(p)

        # Crowd predictions for all zones
        for zone_id in list(self._zone_flow_rates.keys()):
            p = self.predict_crowd(zone_id)
            if p:
                predictions.append(p)

        # Collision predictions
        if spatial_memory:
            predictions.extend(self.predict_collision(spatial_memory))

        return predictions

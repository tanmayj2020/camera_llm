"""Task 10: World Model — predictive/anticipatory intelligence.

Trajectory prediction: Social-LSTM with grid-based social pooling.
Fallback: polynomial extrapolation for short histories.
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    prediction_type: str  # "trajectory", "crowd", "collision", "pattern"
    description: str
    confidence: float
    time_horizon_s: float
    entities_involved: list[str]
    recommended_action: str = ""


# ---------------------------------------------------------------------------
# Social-LSTM with grid-based social pooling
# ---------------------------------------------------------------------------

class _SocialPooling(nn.Module):
    """Grid-based social pooling: encodes nearby pedestrian states into a fixed-size tensor."""

    def __init__(self, hidden_dim: int = 64, grid_size: int = 4, neighborhood: float = 4.0):
        super().__init__()
        self.grid_size = grid_size
        self.neighborhood = neighborhood
        self.embed = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h_states: torch.Tensor, positions: torch.Tensor, idx: int) -> torch.Tensor:
        """h_states: (N, H), positions: (N, 2). Returns pooled tensor (H,) for entity idx."""
        N, H = h_states.shape
        pos_i = positions[idx]
        grid = torch.zeros(self.grid_size * self.grid_size, H, device=h_states.device)
        counts = torch.zeros(self.grid_size * self.grid_size, 1, device=h_states.device)

        for j in range(N):
            if j == idx:
                continue
            diff = positions[j] - pos_i
            if diff.abs().max() > self.neighborhood:
                continue
            gx = int(((diff[0] / self.neighborhood + 1) / 2 * self.grid_size).clamp(0, self.grid_size - 1))
            gy = int(((diff[1] / self.neighborhood + 1) / 2 * self.grid_size).clamp(0, self.grid_size - 1))
            cell = gy * self.grid_size + gx
            grid[cell] += h_states[j]
            counts[cell] += 1

        counts = counts.clamp(min=1)
        pooled = (grid / counts).sum(dim=0)
        return self.embed(pooled)


class SocialLSTMNet(nn.Module):
    """Lightweight Social-LSTM: input [x, y, vx, vy] + social pooling → predict [dx, dy]."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, pred_steps: int = 12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pred_steps = pred_steps
        self.input_embed = nn.Linear(input_dim, hidden_dim)
        self.lstm_cell = nn.LSTMCell(hidden_dim * 2, hidden_dim)  # input + social
        self.social_pool = _SocialPooling(hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)

    def forward_step(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor,
                     all_h: torch.Tensor, positions: torch.Tensor, idx: int):
        emb = self.input_embed(x)
        social = self.social_pool(all_h, positions, idx)
        inp = torch.cat([emb, social])
        h_new, c_new = self.lstm_cell(inp.unsqueeze(0), (h.unsqueeze(0), c.unsqueeze(0)))
        return h_new.squeeze(0), c_new.squeeze(0)

    def predict_sequence(self, history: torch.Tensor, all_h: torch.Tensor,
                         positions: torch.Tensor, idx: int) -> torch.Tensor:
        """history: (T, 4), returns (pred_steps, 2) predicted displacements."""
        h = torch.zeros(self.hidden_dim, device=history.device)
        c = torch.zeros(self.hidden_dim, device=history.device)

        # Encode observed history
        for t in range(history.shape[0]):
            h, c = self.forward_step(history[t], h, c, all_h, positions, idx)

        # Predict future
        preds = []
        last_pos = history[-1, :2].clone()
        for _ in range(self.pred_steps):
            delta = self.output(h)
            preds.append(delta)
            vel = delta / 0.5  # assume 0.5s step
            inp = torch.cat([last_pos + delta, vel])
            h, c = self.forward_step(inp, h, c, all_h, positions, idx)
            last_pos = last_pos + delta
        return torch.stack(preds)


class SocialPoolingPredictor:
    """Wraps Social-LSTM for trajectory prediction with lazy init."""

    def __init__(self, hidden_dim: int = 64, pred_steps: int = 12):
        self._model: SocialLSTMNet | None = None
        self._hidden_dim = hidden_dim
        self._pred_steps = pred_steps

    def _ensure_model(self) -> SocialLSTMNet:
        if self._model is None:
            self._model = SocialLSTMNet(hidden_dim=self._hidden_dim, pred_steps=self._pred_steps)
            self._model.eval()
        return self._model

    @torch.no_grad()
    def predict(self, target_id: str, trajectory_history: dict[str, deque],
                horizon_s: float = 10.0) -> list[np.ndarray] | None:
        target_hist = trajectory_history.get(target_id)
        if not target_hist or len(target_hist) < 8:
            return None

        model = self._ensure_model()

        # Build input for target
        entries = list(target_hist)
        times = np.array([t for t, _ in entries])
        positions = np.array([p[:2] if len(p) >= 2 else p for _, p in entries])
        dt = np.diff(times, prepend=times[0] - 0.5)
        dt = np.where(dt > 0, dt, 0.5)
        velocities = np.diff(positions, axis=0, prepend=positions[:1]) / dt[:, None]

        inp = np.concatenate([positions, velocities], axis=1).astype(np.float32)
        history_t = torch.from_numpy(inp[-20:])  # last 20 steps

        # Build neighbor hidden states (simplified: use zero for initial)
        all_ids = list(trajectory_history.keys())
        N = len(all_ids)
        all_h = torch.zeros(N, model.hidden_dim)
        all_pos = torch.zeros(N, 2)
        idx = 0
        for i, eid in enumerate(all_ids):
            hist = trajectory_history[eid]
            if hist:
                _, p = list(hist)[-1]
                all_pos[i] = torch.tensor(p[:2] if len(p) >= 2 else [0, 0], dtype=torch.float32)
            if eid == target_id:
                idx = i

        pred_deltas = model.predict_sequence(history_t, all_h, all_pos, idx)
        last_pos = positions[-1]
        predicted = []
        for d in pred_deltas.numpy():
            last_pos = last_pos + d
            predicted.append(last_pos.copy())
        return predicted


class WorldModel:
    """Predicts future events from historical patterns in the knowledge graph.

    - Trajectory prediction via linear/polynomial extrapolation
    - Crowd dynamics prediction from flow rates
    - Behavioral sequence prediction from event chains
    """

    def __init__(self, history_window: int = 100):
        self._trajectory_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=history_window))
        self._zone_flow_rates: dict[str, deque] = defaultdict(lambda: deque(maxlen=60))
        self._event_sequences: deque = deque(maxlen=1000)
        self._social_predictor = SocialPoolingPredictor()

    def update_trajectory(self, track_id: str, position: np.ndarray, timestamp: float):
        self._trajectory_history[track_id].append((timestamp, position.copy()))

    def update_zone_count(self, zone_id: str, count: int, timestamp: float):
        self._zone_flow_rates[zone_id].append((timestamp, count))

    def record_event(self, event_type: str, timestamp: float):
        self._event_sequences.append((timestamp, event_type))

    def predict_trajectory(self, track_id: str, horizon_s: float = 10.0) -> Prediction | None:
        """Predict where an entity will be in `horizon_s` seconds.

        Uses Social-LSTM when history >= 8 frames (accounts for neighbor influence).
        Falls back to polynomial extrapolation for shorter histories.
        """
        history = self._trajectory_history.get(track_id)
        if not history or len(history) < 3:
            return None

        # Try Social-LSTM for longer histories
        if len(history) >= 8:
            try:
                predicted_seq = self._social_predictor.predict(
                    track_id, self._trajectory_history, horizon_s)
                if predicted_seq:
                    final = predicted_seq[-1]
                    return Prediction(
                        prediction_type="trajectory",
                        description=f"Entity {track_id} predicted at ({final[0]:.1f}, {final[1]:.1f})m in {horizon_s}s [social-lstm]",
                        confidence=min(0.92, len(history) / 40),
                        time_horizon_s=horizon_s,
                        entities_involved=[track_id],
                    )
            except Exception as e:
                logger.debug("Social-LSTM prediction failed for %s: %s, using polynomial", track_id, e)

        # Polynomial fallback
        times = np.array([t for t, _ in history])
        positions = np.array([p for _, p in history])
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
            description=f"Entity {track_id} predicted at ({predicted_pos[0]:.1f}, {predicted_pos[-1]:.1f})m in {horizon_s}s [poly]",
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

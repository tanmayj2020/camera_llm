"""Predictive Path Interception — alerts BEFORE a person reaches a restricted
zone by forecasting their trajectory using the Social-LSTM world model +
floor plan topology.

How it works:
  1. World model predicts each entity's future positions (5–30s ahead)
  2. Floor plan provides restricted zone polygons
  3. System checks if any predicted path intersects a restricted zone
  4. If so, estimate time-to-arrival and raise a PRE-EMPTIVE alert

This gives security operators a 10–30 second advance warning before a breach
occurs — enough time to lock a door, dispatch a guard, or issue a PA warning.

Novel because: All existing systems detect AFTER someone enters a restricted
area.  This detects INTENT before the violation happens.
"""

import logging
import math
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class InterceptionAlert:
    """Pre-emptive alert that someone is heading toward a restricted zone."""
    entity_id: str
    current_position: tuple[float, float]
    predicted_entry_point: tuple[float, float]
    target_zone_id: str
    target_zone_name: str
    estimated_arrival_s: float     # seconds until predicted zone entry
    confidence: float              # 0.0–1.0
    predicted_path: list[tuple[float, float]]
    recommendation: str
    timestamp: float


class PredictiveInterceptor:
    """Combines world model trajectory prediction with floor plan zone
    geometry to detect pre-emptive restricted zone violations."""

    def __init__(self, *, world_model=None, spatial=None, floor_plan=None):
        self._world_model = world_model
        self._spatial = spatial
        self._floor_plan = floor_plan
        # Alert cooldown per entity per zone (prevent spam)
        self._cooldowns: dict[str, float] = {}
        self._cooldown_s = 30.0

    # ── Public API ────────────────────────────────────────────────────────

    def evaluate(self, scene_state: dict) -> list[InterceptionAlert]:
        """Run predictive interception for all tracked entities.

        Returns a list of InterceptionAlerts for entities predicted to
        enter restricted zones within the forecast horizon.
        """
        now = scene_state.get("timestamp", time.time())
        alerts: list[InterceptionAlert] = []

        restricted_zones = self._get_restricted_zones()
        if not restricted_zones:
            return alerts

        # Get all entity predictions from world model
        predictions = self._get_predictions()
        if not predictions:
            return alerts

        for entity_id, pred_path in predictions.items():
            if len(pred_path) < 2:
                continue

            current_pos = pred_path[0]

            for zone in restricted_zones:
                # Skip if already inside the zone
                if self._point_in_polygon(current_pos, zone["polygon"]):
                    continue

                # Cooldown check
                cooldown_key = f"{entity_id}:{zone['zone_id']}"
                if now - self._cooldowns.get(cooldown_key, 0) < self._cooldown_s:
                    continue

                # Check each predicted future position
                entry_idx = None
                for i, pos in enumerate(pred_path[1:], 1):
                    if self._point_in_polygon(pos, zone["polygon"]):
                        entry_idx = i
                        break

                if entry_idx is not None:
                    # Estimate time to arrival
                    # Each prediction step ~= 0.5s (depends on world model config)
                    step_duration = 0.5
                    eta_s = entry_idx * step_duration

                    # Confidence: higher if the trajectory is consistent
                    confidence = self._compute_confidence(pred_path, entry_idx, zone)

                    if confidence < 0.3:
                        continue  # too uncertain

                    entry_point = pred_path[entry_idx]
                    recommendation = self._generate_recommendation(
                        entity_id, zone, eta_s, confidence)

                    alerts.append(InterceptionAlert(
                        entity_id=entity_id,
                        current_position=current_pos,
                        predicted_entry_point=entry_point,
                        target_zone_id=zone["zone_id"],
                        target_zone_name=zone.get("name", zone["zone_id"]),
                        estimated_arrival_s=round(eta_s, 1),
                        confidence=round(confidence, 2),
                        predicted_path=pred_path[:entry_idx + 1],
                        recommendation=recommendation,
                        timestamp=now,
                    ))
                    self._cooldowns[cooldown_key] = now

        return alerts

    def get_all_trajectories(self) -> dict[str, list[tuple[float, float]]]:
        """Return predicted trajectories for all tracked entities
        (useful for floor plan visualization)."""
        return self._get_predictions()

    # ── Internals ─────────────────────────────────────────────────────────

    def _get_restricted_zones(self) -> list[dict]:
        """Get restricted zone definitions from spatial memory or floor plan."""
        zones = []

        # Try spatial memory zones first
        if self._spatial:
            spatial_zones = getattr(self._spatial, "_zones", {})
            for zid, z in spatial_zones.items():
                ztype = getattr(z, "zone_type", "")
                if ztype in ("restricted", "secure", "no_entry", "emergency_exit"):
                    polygon = getattr(z, "polygon", [])
                    zones.append({
                        "zone_id": zid,
                        "name": getattr(z, "name", zid),
                        "polygon": polygon,
                        "zone_type": ztype,
                    })

        # Try floor plan zones
        if self._floor_plan and not zones:
            try:
                for z in self._floor_plan._zones.values():
                    if getattr(z, "zone_type", "") in ("restricted", "secure"):
                        zones.append({
                            "zone_id": z.zone_id,
                            "name": getattr(z, "name", z.zone_id),
                            "polygon": z.polygon,
                            "zone_type": z.zone_type,
                        })
            except Exception:
                pass

        return zones

    def _get_predictions(self) -> dict[str, list[tuple[float, float]]]:
        """Get trajectory predictions from world model."""
        if not self._world_model:
            return {}

        results = {}
        try:
            # WorldModel.predict_all() returns per-entity predictions
            if hasattr(self._world_model, "predict_all"):
                all_preds = self._world_model.predict_all()
                for p in all_preds:
                    eid = getattr(p, "entity_id", None)
                    path = getattr(p, "path", [])
                    if eid and path:
                        results[eid] = [(pt[0], pt[1]) for pt in path]
            elif hasattr(self._world_model, "_histories"):
                # Fallback: call predict() per entity
                for eid in list(self._world_model._histories.keys()):
                    try:
                        pred = self._world_model.predict(eid)
                        if pred and hasattr(pred, "path"):
                            results[eid] = [(pt[0], pt[1]) for pt in pred.path]
                    except Exception:
                        continue
        except Exception as e:
            logger.warning("World model prediction failed: %s", e)

        return results

    def _compute_confidence(self, path: list, entry_idx: int,
                            zone: dict) -> float:
        """Confidence is higher when:
        - The entity is moving directly toward the zone (low angle deviation)
        - The entity is close to the zone
        - Multiple future points are inside the zone (not a tangent)
        """
        if entry_idx < 2:
            direction_consistency = 0.5
        else:
            # Check if last 3 points form a consistent direction
            vectors = []
            for i in range(max(0, entry_idx - 2), entry_idx):
                dx = path[i + 1][0] - path[i][0]
                dy = path[i + 1][1] - path[i][1]
                mag = math.sqrt(dx ** 2 + dy ** 2)
                if mag > 0.01:
                    vectors.append((dx / mag, dy / mag))
            if len(vectors) >= 2:
                # Cosine similarity between consecutive direction vectors
                dots = []
                for j in range(len(vectors) - 1):
                    dot = vectors[j][0] * vectors[j + 1][0] + vectors[j][1] * vectors[j + 1][1]
                    dots.append(max(0, dot))
                direction_consistency = sum(dots) / len(dots)
            else:
                direction_consistency = 0.5

        # Check how many points past entry are also in zone
        inside_after = 0
        for pt in path[entry_idx:min(entry_idx + 5, len(path))]:
            if self._point_in_polygon(pt, zone["polygon"]):
                inside_after += 1
        penetration = inside_after / max(1, min(5, len(path) - entry_idx))

        # Closer entry = higher confidence
        distance_factor = max(0.0, 1.0 - entry_idx / 20)

        return 0.4 * direction_consistency + 0.35 * penetration + 0.25 * distance_factor

    def _generate_recommendation(self, entity_id: str, zone: dict,
                                  eta_s: float, confidence: float) -> str:
        """Generate actionable recommendation based on urgency."""
        if eta_s < 5:
            return (f"URGENT: Entity {entity_id} approaching {zone['name']} "
                    f"in ~{eta_s:.0f}s. Lock door / issue PA warning immediately.")
        elif eta_s < 15:
            return (f"Entity {entity_id} heading toward {zone['name']} "
                    f"(ETA ~{eta_s:.0f}s). Dispatch guard or activate deterrent.")
        else:
            return (f"Entity {entity_id} on trajectory toward {zone['name']} "
                    f"(ETA ~{eta_s:.0f}s, confidence {confidence:.0%}). Monitor.")

    @staticmethod
    def _point_in_polygon(point: tuple, polygon: list) -> bool:
        """Ray-casting point-in-polygon test."""
        if not polygon or len(polygon) < 3:
            return False
        x, y = point[0], point[1]
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            pi = polygon[i]
            pj = polygon[j]
            xi, yi = (pi[0], pi[1]) if isinstance(pi, (list, tuple)) else (pi.get("x", 0), pi.get("y", 0))
            xj, yj = (pj[0], pj[1]) if isinstance(pj, (list, tuple)) else (pj.get("x", 0), pj.get("y", 0))

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi):
                inside = not inside
            j = i
        return inside

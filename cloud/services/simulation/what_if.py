"""What-If Simulator — scenario-based impact analysis for CCTV deployments."""

import copy
import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

SCENARIO_TYPES = {
    "close_zone": "Remove a zone and estimate traffic redistribution",
    "add_camera": "Estimate coverage gain from a new camera placement",
    "change_threshold": "Replay recent events with a modified alert threshold",
    "add_barrier": "Block a path with a barrier and estimate rerouting",
}


@dataclass
class WhatIfScenario:
    scenario_type: str  # "close_zone", "add_camera", "change_threshold", "add_barrier"
    parameters: dict = field(default_factory=dict)
    description: str = ""


@dataclass
class SimulationResult:
    scenario: WhatIfScenario
    predicted_impact: dict = field(default_factory=dict)
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)


class WhatIfSimulator:
    """Runs what-if simulations against the world model and spatial memory."""

    def __init__(self, world_model=None, spatial=None):
        self._world_model = world_model
        self._spatial = spatial

    def simulate(self, scenario: WhatIfScenario) -> SimulationResult:
        handler = {
            "close_zone": self._sim_close_zone,
            "add_camera": self._sim_add_camera,
            "change_threshold": self._sim_change_threshold,
            "add_barrier": self._sim_add_barrier,
        }.get(scenario.scenario_type)
        if not handler:
            return SimulationResult(scenario=scenario, predicted_impact={"error": "unknown scenario type"})
        try:
            return handler(scenario)
        except Exception as e:
            logger.error("Simulation failed for %s: %s", scenario.scenario_type, e)
            return SimulationResult(scenario=scenario, predicted_impact={"error": str(e)})

    # ---- scenario handlers ----

    def _sim_close_zone(self, scenario: WhatIfScenario) -> SimulationResult:
        zone_id = scenario.parameters.get("zone_id", "")
        zones = self._get_zones()
        zone_traffic_before = self._zone_traffic()

        # Entities that were in the closed zone get rerouted
        affected = [eid for eid, zid in self._entity_zone_map().items() if zid == zone_id]
        rerouted = self._estimate_reroute(affected, zone_id)

        # Redistribute traffic from closed zone across remaining zones
        closed_count = zone_traffic_before.get(zone_id, 0)
        remaining = [z for z in zones if z != zone_id]
        extra = closed_count / max(len(remaining), 1)
        zone_traffic_after = {z: (c + extra if z != zone_id else 0) for z, c in zone_traffic_before.items()}

        before = {"zone_traffic": zone_traffic_before, "alert_count": sum(zone_traffic_before.values()), "coverage_pct": 100.0}
        after = {"zone_traffic": zone_traffic_after, "alert_count": sum(zone_traffic_after.values()), "coverage_pct": round(100.0 * len(remaining) / max(len(zones), 1), 1)}
        recs = [f"Closing {zone_id} displaces {closed_count} entities across {len(remaining)} zones"]
        if closed_count > 10:
            recs.append("High traffic zone — consider phased closure")
        return SimulationResult(scenario=scenario, predicted_impact={"rerouted_entities": len(rerouted)}, before=before, after=after, recommendations=recs)

    def _sim_add_camera(self, scenario: WhatIfScenario) -> SimulationResult:
        pos = scenario.parameters.get("position", (0.0, 0.0))
        fov = scenario.parameters.get("fov_deg", 60.0)
        reach = scenario.parameters.get("reach_m", 30.0)

        # Estimate coverage as circular sector area
        new_area = math.pi * reach ** 2 * (fov / 360.0)
        total_area = scenario.parameters.get("site_area_m2", 10000.0)
        existing_pct = scenario.parameters.get("existing_coverage_pct", 70.0)
        gain = min(new_area / total_area * 100, 100 - existing_pct)

        before = {"zone_traffic": {}, "alert_count": 0, "coverage_pct": existing_pct}
        after = {"zone_traffic": {}, "alert_count": 0, "coverage_pct": round(existing_pct + gain, 1)}
        return SimulationResult(scenario=scenario, predicted_impact={"coverage_gain_pct": round(gain, 1), "new_area_m2": round(new_area, 1)}, before=before, after=after, recommendations=[f"Camera at {pos} adds ~{gain:.1f}% coverage"])

    def _sim_change_threshold(self, scenario: WhatIfScenario) -> SimulationResult:
        rule_id = scenario.parameters.get("rule_id", "")
        old_thresh = scenario.parameters.get("old_threshold", 0.5)
        new_thresh = scenario.parameters.get("new_threshold", 0.7)

        # Replay recent trajectories from world model
        scores = self._recent_scores(rule_id)
        before_count = sum(1 for s in scores if s >= old_thresh)
        after_count = sum(1 for s in scores if s >= new_thresh)

        before = {"zone_traffic": {}, "alert_count": before_count, "coverage_pct": 0.0}
        after = {"zone_traffic": {}, "alert_count": after_count, "coverage_pct": 0.0}
        diff = before_count - after_count
        recs = []
        if diff > 0:
            recs.append(f"Raising threshold eliminates {diff} alerts ({diff / max(before_count, 1):.0%} reduction)")
        else:
            recs.append("Threshold change has no effect on recent alerts")
        return SimulationResult(scenario=scenario, predicted_impact={"alerts_removed": diff}, before=before, after=after, recommendations=recs)

    def _sim_add_barrier(self, scenario: WhatIfScenario) -> SimulationResult:
        line_start = scenario.parameters.get("line_start", (0, 0))
        line_end = scenario.parameters.get("line_end", (0, 0))
        barrier_zone = scenario.parameters.get("blocked_zone", "barrier")

        # Find entities whose predicted trajectory crosses the barrier line
        affected = self._entities_crossing_line(line_start, line_end)
        rerouted = self._estimate_reroute(affected, barrier_zone)

        zone_traffic_before = self._zone_traffic()
        zone_traffic_after = dict(zone_traffic_before)
        before = {"zone_traffic": zone_traffic_before, "alert_count": len(affected), "coverage_pct": 0.0}
        after = {"zone_traffic": zone_traffic_after, "alert_count": 0, "coverage_pct": 0.0}
        return SimulationResult(scenario=scenario, predicted_impact={"blocked_entities": len(affected), "rerouted": len(rerouted)}, before=before, after=after, recommendations=[f"Barrier blocks {len(affected)} entity paths"])

    # ---- helpers ----

    def _estimate_reroute(self, entities: list[str], blocked_zone: str) -> dict[str, list]:
        """Predict new paths for entities blocked from a zone."""
        result: dict[str, list] = {}
        if not self._world_model:
            return result
        for eid in entities:
            pred = self._world_model.predict_trajectory(eid, horizon_s=10.0)
            result[eid] = pred.description if pred else "no prediction"
        return result

    def _zone_traffic(self) -> dict[str, int]:
        if not self._world_model:
            return {}
        return {z: len(r) for z, r in self._world_model._zone_flow_rates.items() if r}

    def _get_zones(self) -> list[str]:
        if self._spatial and hasattr(self._spatial, "_zones"):
            return list(self._spatial._zones.keys())
        return list(self._zone_traffic().keys())

    def _entity_zone_map(self) -> dict[str, str]:
        if self._spatial and hasattr(self._spatial, "get_zone"):
            return {eid: self._spatial.get_zone(eid) for eid in self._world_model._trajectory_history} if self._world_model else {}
        return {}

    def _recent_scores(self, rule_id: str) -> list[float]:
        """Return trajectory confidence scores as proxy for alert threshold replay."""
        if not self._world_model:
            return []
        scores = []
        for tid in list(self._world_model._trajectory_history.keys())[:50]:
            p = self._world_model.predict_trajectory(tid, horizon_s=5.0)
            if p:
                scores.append(p.confidence)
        return scores

    def _entities_crossing_line(self, start: tuple, end: tuple) -> list[str]:
        if not self._world_model:
            return []
        # Simple: check if any trajectory point is near the barrier line
        crossing = []
        sx, sy = start
        ex, ey = end
        for tid, hist in self._world_model._trajectory_history.items():
            for _, pos in hist:
                px, py = float(pos[0]), float(pos[1]) if len(pos) > 1 else 0.0
                # Point-to-segment distance check
                dx, dy = ex - sx, ey - sy
                seg_len_sq = dx * dx + dy * dy
                if seg_len_sq == 0:
                    continue
                t = max(0, min(1, ((px - sx) * dx + (py - sy) * dy) / seg_len_sq))
                dist = math.hypot(px - (sx + t * dx), py - (sy + t * dy))
                if dist < 2.0:
                    crossing.append(tid)
                    break
        return crossing

    def list_scenarios(self) -> list[dict[str, str]]:
        return [{"type": k, "description": v} for k, v in SCENARIO_TYPES.items()]

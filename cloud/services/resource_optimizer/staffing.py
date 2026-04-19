"""Predictive staffing and resource optimization."""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class StaffingRecommendation:
    date: str
    time_slot: str
    predicted_footfall: int
    recommended_guards: int
    recommended_zones: list[str]
    confidence: float
    reasoning: str


_DAY_MULTIPLIER = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.3, 5: 1.5, 6: 0.8}


class ResourceOptimizer:
    def __init__(self, occupancy=None, contextual_normality=None, kpi_engine=None, vlm_client=None):
        self._occupancy = occupancy
        self._ctx_norm = contextual_normality
        self._kpi = kpi_engine
        self._vlm_client = vlm_client

    def predict_demand(self, camera_id="all", hours_ahead=24) -> list[dict]:
        now = time.time()
        dt = datetime.fromtimestamp(now)
        predictions = []
        for h in range(hours_ahead):
            hour = (dt.hour + h) % 24
            dow = (dt.weekday() + (dt.hour + h) // 24) % 7
            base = 20
            if self._ctx_norm:
                try:
                    exp = self._ctx_norm.get_expected(camera_id, now + h * 3600)
                    base = exp.get("expected_count", 20)
                except Exception:
                    pass
            predictions.append({"hour": hour, "expected_count": int(base * _DAY_MULTIPLIER.get(dow, 1.0)), "confidence": 0.7})
        return predictions

    def recommend_staffing(self, camera_ids=None, date_str=None) -> list[StaffingRecommendation]:
        preds = self.predict_demand(hours_ahead=24)
        recs = []
        for slot_start in range(0, 24, 4):
            slot_preds = [p for p in preds if slot_start <= p["hour"] < slot_start + 4]
            peak = max((p["expected_count"] for p in slot_preds), default=20)
            guards = max(1, peak // 50 + 1)
            zones = ["entrance", "lobby"] if peak > 30 else ["entrance"]
            recs.append(StaffingRecommendation(
                date=date_str or datetime.now().strftime("%Y-%m-%d"),
                time_slot=f"{slot_start:02d}:00-{slot_start + 4:02d}:00",
                predicted_footfall=peak, recommended_guards=guards,
                recommended_zones=zones, confidence=0.7,
                reasoning=f"Expected peak of {peak} people, {guards} guards recommended"))
        return recs

    def get_resource_plan(self, hours=24) -> dict:
        recs = self.recommend_staffing()
        total = sum(r.recommended_guards for r in recs)
        peaks = sorted(recs, key=lambda r: r.predicted_footfall, reverse=True)[:3]
        return {"total_guard_shifts": total, "peak_slots": [r.time_slot for r in peaks],
                "estimated_cost": total * 25.0, "recommendations": [r.__dict__ for r in recs]}

"""Predictive Crime Mapping — predict where/when incidents are most likely."""

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RiskCell:
    zone_id: str
    hour: int
    day_of_week: int
    risk_score: float
    incident_count: int
    prediction: str


class PredictiveCrimeMapper:
    """Predicts incident likelihood per zone/time from historical patterns."""

    def __init__(self):
        # (zone_id, hour, day_of_week) -> incident count
        self._incident_grid: dict[tuple[str, int, int], int] = defaultdict(int)
        self._total_incidents = 0

    def record_incident(self, zone_id: str, timestamp: float):
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        key = (zone_id, dt.hour, dt.weekday())
        self._incident_grid[key] += 1
        self._total_incidents += 1

    def predict(self, zone_id: str = None, hours_ahead: int = 24) -> list[RiskCell]:
        if self._total_incidents < 10:
            return []
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        predictions = []

        zones = [zone_id] if zone_id else list({k[0] for k in self._incident_grid})
        for z in zones:
            for h_offset in range(hours_ahead):
                future_hour = (now.hour + h_offset) % 24
                future_day = (now.weekday() + h_offset // 24) % 7
                key = (z, future_hour, future_day)
                count = self._incident_grid.get(key, 0)
                max_count = max(self._incident_grid.values()) if self._incident_grid else 1
                risk = count / max_count
                if risk > 0.1:
                    predictions.append(RiskCell(
                        z, future_hour, future_day, round(risk, 3), count,
                        f"{'High' if risk > 0.6 else 'Medium' if risk > 0.3 else 'Low'} risk"))

        predictions.sort(key=lambda r: r.risk_score, reverse=True)
        return predictions[:50]

    def get_heatmap(self) -> dict[str, list[float]]:
        """Returns 24-hour risk profile per zone."""
        zones = {k[0] for k in self._incident_grid}
        result = {}
        max_count = max(self._incident_grid.values()) if self._incident_grid else 1
        for z in zones:
            hourly = [0.0] * 24
            for h in range(24):
                total = sum(self._incident_grid.get((z, h, d), 0) for d in range(7))
                hourly[h] = round(total / max_count, 3)
            result[z] = hourly
        return result

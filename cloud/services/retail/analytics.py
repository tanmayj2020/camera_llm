"""Retail Analytics Mode."""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class RetailMetrics:
    store_id: str
    timestamp: float
    footfall: int
    conversion_rate: float
    avg_dwell_s: float
    queue_length: int
    busiest_zone: str
    hourly_traffic: dict[int, int]


class RetailAnalytics:
    def __init__(self, spatial=None, occupancy=None, dwell_analyzer=None, heatmap=None) -> None:
        self._spatial = spatial
        self._occupancy = occupancy
        self._dwell_analyzer = dwell_analyzer
        self._heatmap = heatmap
        self._entry_zones: list[str] = []
        self._checkout_zones: list[str] = []
        self._queue_zones: list[str] = []
        self._entry_count: int = 0
        self._checkout_count: int = 0
        self._hourly: dict[int, int] = defaultdict(int)

    def configure(self, entry_zones: list[str], checkout_zones: list[str], queue_zones: list[str]) -> None:
        self._entry_zones = entry_zones
        self._checkout_zones = checkout_zones
        self._queue_zones = queue_zones

    def update(self, scene_state: dict) -> None:
        if not self._spatial:
            return
        for person in scene_state.get("persons", []):
            zone = self._spatial.get_zone(person) if hasattr(self._spatial, "get_zone") else None
            if zone in self._entry_zones:
                self._entry_count += 1
                self._hourly[time.localtime().tm_hour] += 1
            elif zone in self._checkout_zones:
                self._checkout_count += 1

    def get_metrics(self, store_id: str = "default") -> RetailMetrics:
        avg_dwell = self._dwell_analyzer.average() if self._dwell_analyzer and hasattr(self._dwell_analyzer, "average") else 0.0
        queue_len = sum(
            1 for z in self._queue_zones
            if self._spatial and hasattr(self._spatial, "count_in_zone") and self._spatial.count_in_zone(z)
        ) if self._spatial else 0
        busiest = max(self._hourly, key=self._hourly.get, default=0)
        rate = self._checkout_count / self._entry_count if self._entry_count else 0.0
        return RetailMetrics(
            store_id=store_id,
            timestamp=time.time(),
            footfall=self._entry_count,
            conversion_rate=rate,
            avg_dwell_s=avg_dwell,
            queue_length=queue_len,
            busiest_zone=str(busiest),
            hourly_traffic=dict(self._hourly),
        )

    def get_funnel(self) -> dict:
        return {
            "entered": self._entry_count,
            "browsed": max(self._entry_count - self._checkout_count, 0),
            "picked_item": self._checkout_count,
            "checkout": self._checkout_count,
        }

    def get_attention_heatmap(self) -> list[list[float]]:
        if self._heatmap and hasattr(self._heatmap, "get_dwell_weighted"):
            return self._heatmap.get_dwell_weighted()
        # Fallback: generate zone-based attention from dwell data
        if self._dwell_analyzer and hasattr(self._dwell_analyzer, "_zone_dwells"):
            grid_size = 10
            grid = [[0.0] * grid_size for _ in range(grid_size)]
            for zone_id, dwells in self._dwell_analyzer._zone_dwells.items():
                if dwells:
                    avg_dwell = sum(dwells) / len(dwells)
                    # Hash zone_id to grid position
                    h = hash(zone_id) % (grid_size * grid_size)
                    r, c = divmod(h, grid_size)
                    grid[r][c] = min(1.0, avg_dwell / 300.0)  # normalize to 5 min max
            return grid
        return [[0.0] * 10 for _ in range(10)]

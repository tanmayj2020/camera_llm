"""Advanced heatmap generation from spatial positions."""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class HeatmapEngine:
    def __init__(self, grid_size: int = 20):
        self._grid_size = grid_size
        self._movement_grid: dict[tuple[int, int], float] = defaultdict(float)
        self._dwell_grid: dict[tuple[int, int], float] = defaultdict(float)
        self._hourly_grids: dict[int, dict[tuple[int, int], float]] = {
            h: defaultdict(float) for h in range(24)
        }

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return int(x) % self._grid_size, int(y) % self._grid_size

    def update(self, x: float, y: float, dwell_seconds: float = 1.0, hour: int = 0):
        cell = self._cell(x, y)
        self._movement_grid[cell] += 1.0
        self._dwell_grid[cell] += dwell_seconds
        self._hourly_grids[hour % 24][cell] += 1.0

    def generate(self, type: str = "movement", hour: int | None = None) -> list[list[float]]:
        if type == "hourly" and hour is not None:
            grid = self._hourly_grids.get(hour % 24, {})
        elif type == "dwell":
            grid = self._dwell_grid
        else:
            grid = self._movement_grid
        max_val = max(grid.values()) if grid else 1.0
        if max_val == 0:
            max_val = 1.0
        return [[grid.get((x, y), 0) / max_val for y in range(self._grid_size)]
                for x in range(self._grid_size)]

    def get_hot_spots(self, threshold: float = 0.7) -> list[dict]:
        grid = self._movement_grid
        max_val = max(grid.values()) if grid else 1.0
        if max_val == 0:
            return []
        return [{"x": x, "y": y, "intensity": round(v / max_val, 3)}
                for (x, y), v in grid.items() if v / max_val >= threshold]

    def compare(self, type_a: str = "movement", type_b: str = "dwell") -> list[list[float]]:
        a = self.generate(type_a)
        b = self.generate(type_b)
        return [[a[x][y] - b[x][y] for y in range(self._grid_size)]
                for x in range(self._grid_size)]

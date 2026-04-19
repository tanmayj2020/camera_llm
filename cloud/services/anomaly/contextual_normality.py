"""Context-aware normality model for VisionBrain CCTV Analytics."""

import logging
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class NormalityContext:
    hour: int
    day_of_week: int
    is_weekend: bool
    is_holiday: bool
    weather: str = "clear"
    special_event: str = ""


class ContextualNormalityModel:
    def __init__(self) -> None:
        self._expected_counts: Dict[str, Dict[Tuple[int, int], deque]] = {}
        self._holiday_calendar: Dict[str, str] = {}
        self._special_events: Dict[str, str] = {}

    def add_holiday(self, date_str: str, name: str) -> None:
        self._holiday_calendar[date_str] = name

    def add_special_event(self, date_str: str, name: str) -> None:
        self._special_events[date_str] = name

    def get_context(self, timestamp: float) -> NormalityContext:
        dt = datetime.fromtimestamp(timestamp)
        mm_dd = dt.strftime("%m-%d")
        date_str = dt.strftime("%Y-%m-%d")
        return NormalityContext(
            hour=dt.hour,
            day_of_week=dt.weekday(),
            is_weekend=dt.weekday() >= 5,
            is_holiday=mm_dd in self._holiday_calendar,
            special_event=self._special_events.get(date_str, ""),
        )

    def ingest(self, camera_id: str, count: int, timestamp: float) -> None:
        ctx = self.get_context(timestamp)
        bucket = (ctx.hour, ctx.day_of_week)
        self._expected_counts.setdefault(camera_id, {}).setdefault(
            bucket, deque(maxlen=30)
        ).append(count)
        logger.debug("Ingested count=%d for camera=%s bucket=%s", count, camera_id, bucket)

    def compute_contextual_anomaly_score(
        self, camera_id: str, current_count: int, timestamp: float
    ) -> float:
        ctx = self.get_context(timestamp)
        bucket = (ctx.hour, ctx.day_of_week)
        counts = self._expected_counts.get(camera_id, {}).get(bucket)
        if not counts or len(counts) < 2:
            return 0.0
        mean = statistics.mean(counts)
        std = statistics.stdev(counts)
        if std == 0:
            return 0.0
        tolerance = 1.5 if (ctx.is_holiday or ctx.is_weekend) else 1.0
        z = abs(current_count - mean) / (std * tolerance)
        score = min(z / 4.0, 1.0)
        logger.debug("Anomaly score=%.3f for camera=%s (z=%.2f)", score, camera_id, z)
        return score

    def get_expected(self, camera_id: str, timestamp: float) -> dict:
        ctx = self.get_context(timestamp)
        bucket = (ctx.hour, ctx.day_of_week)
        counts = self._expected_counts.get(camera_id, {}).get(bucket)
        if not counts or len(counts) < 2:
            return {"expected_count": 0.0, "std": 0.0, "context": vars(ctx)}
        return {
            "expected_count": statistics.mean(counts),
            "std": statistics.stdev(counts),
            "context": vars(ctx),
        }

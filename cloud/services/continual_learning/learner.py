"""Task 17: Continual Learning + Concept Drift Adaptation."""

import logging
import time
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class ConceptDriftDetector:
    """Detects when the data distribution shifts significantly (concept drift).

    Uses Page-Hinkley test on anomaly score distributions.
    """

    def __init__(self, threshold: float = 50.0, alpha: float = 0.005):
        self.threshold = threshold
        self.alpha = alpha
        self._sum = 0.0
        self._min_sum = 0.0
        self._count = 0
        self._mean = 0.0

    def update(self, value: float) -> bool:
        """Feed a new observation. Returns True if drift detected."""
        self._count += 1
        self._mean += (value - self._mean) / self._count
        self._sum += value - self._mean - self.alpha
        self._min_sum = min(self._min_sum, self._sum)

        if self._sum - self._min_sum > self.threshold:
            logger.warning("Concept drift detected (PH stat=%.2f)", self._sum - self._min_sum)
            self._reset()
            return True
        return False

    def _reset(self):
        self._sum = 0.0
        self._min_sum = 0.0
        self._count = 0
        self._mean = 0.0


class ContinualLearner:
    """Adapts models to changing environments without catastrophic forgetting.

    - Monitors for concept drift
    - Incrementally updates baselines with EMA
    - Tracks seasonal patterns (hourly, daily, weekly)
    - Adjusts rule thresholds based on false positive feedback
    """

    def __init__(self, baseline_learner=None):
        self.baseline = baseline_learner
        self.drift_detector = ConceptDriftDetector()
        self._false_positive_counts: dict[str, int] = {}  # rule_id -> count
        self._total_trigger_counts: dict[str, int] = {}
        self._seasonal_patterns: dict[str, deque] = {}  # metric -> weekly pattern
        self._feedback_buffer: list[dict] = []

    def observe(self, anomaly_score: float) -> bool:
        """Feed anomaly score, returns True if drift detected."""
        return self.drift_detector.update(anomaly_score)

    def record_feedback(self, event_id: str, rule_id: str, is_false_positive: bool):
        """Human feedback: mark alert as false positive or true positive."""
        self._feedback_buffer.append({
            "event_id": event_id, "rule_id": rule_id,
            "is_fp": is_false_positive, "timestamp": time.time(),
        })
        if is_false_positive:
            self._false_positive_counts[rule_id] = self._false_positive_counts.get(rule_id, 0) + 1
        self._total_trigger_counts[rule_id] = self._total_trigger_counts.get(rule_id, 0) + 1

    def get_fp_rate(self, rule_id: str) -> float:
        total = self._total_trigger_counts.get(rule_id, 0)
        if total == 0:
            return 0.0
        return self._false_positive_counts.get(rule_id, 0) / total

    def suggest_threshold_adjustment(self, rule_id: str) -> dict | None:
        """If FP rate is too high, suggest loosening the rule threshold."""
        fp_rate = self.get_fp_rate(rule_id)
        if fp_rate > 0.3 and self._total_trigger_counts.get(rule_id, 0) >= 10:
            return {
                "rule_id": rule_id,
                "fp_rate": round(fp_rate, 2),
                "suggestion": "increase_threshold",
                "reason": f"False positive rate {fp_rate:.0%} exceeds 30% over {self._total_trigger_counts[rule_id]} triggers",
            }
        return None

    def update_seasonal_pattern(self, metric_name: str, value: float, timestamp: float):
        """Track weekly seasonal patterns."""
        if metric_name not in self._seasonal_patterns:
            self._seasonal_patterns[metric_name] = deque(maxlen=7 * 24)  # hourly for a week
        self._seasonal_patterns[metric_name].append((timestamp, value))

    def get_seasonal_expected(self, metric_name: str, hour_of_week: int) -> float | None:
        """Get expected value for a given hour of the week."""
        pattern = self._seasonal_patterns.get(metric_name)
        if not pattern or len(pattern) < 24:
            return None
        values_at_hour = [v for t, v in pattern if int((t % (7 * 86400)) / 3600) == hour_of_week]
        return float(np.mean(values_at_hour)) if values_at_hour else None

    def on_drift_detected(self):
        """Handle concept drift: reset baseline learning window."""
        if self.baseline:
            logger.info("Drift detected — entering re-learning mode")
            self.baseline._start_time = time.time()
            self.baseline.onboarding_hours = 24  # shorter re-learning window

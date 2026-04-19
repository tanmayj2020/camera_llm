"""Task 6: Self-learning baseline engine — learns 'normal' without configuration.

Uses Isolation Forest for multivariate anomaly detection alongside z-score baselines.
"""

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


class IsolationForestScorer:
    """Multivariate anomaly scoring using sklearn IsolationForest.

    Features: [object_count, activity, hour_sin, hour_cos, day_sin, day_cos, audio_alert_count]
    """

    def __init__(self, contamination: float = 0.05):
        self._contamination = contamination
        self._model = None
        self._buffer: list[np.ndarray] = []
        self._trained = False
        self._min_samples = 200

    @staticmethod
    def _extract_features(event: dict) -> np.ndarray:
        ts = event.get("timestamp", time.time())
        hour = (ts % 86400) / 3600
        day = ((ts // 86400) % 7)
        n_objects = len(event.get("objects", []))
        activity = event.get("scene_activity", 0.0)
        audio_alerts = sum(1 for a in event.get("audio_events", []) if a.get("is_alert"))
        return np.array([
            n_objects, activity,
            math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
            math.sin(2 * math.pi * day / 7), math.cos(2 * math.pi * day / 7),
            audio_alerts,
        ])

    def ingest(self, event: dict):
        self._buffer.append(self._extract_features(event))
        if len(self._buffer) > 10000:
            self._buffer = self._buffer[-5000:]

    def train(self):
        if len(self._buffer) < self._min_samples:
            return
        try:
            from sklearn.ensemble import IsolationForest
            X = np.array(self._buffer)
            self._model = IsolationForest(
                contamination=self._contamination, random_state=42, n_estimators=100)
            self._model.fit(X)
            self._trained = True
            logger.info("IsolationForest trained on %d samples", len(X))
        except Exception as e:
            logger.warning("IsolationForest training failed: %s", e)

    def score(self, event: dict) -> float:
        """Returns anomaly score 0-1 (higher = more anomalous)."""
        if not self._trained or self._model is None:
            return 0.0
        features = self._extract_features(event).reshape(1, -1)
        raw = self._model.decision_function(features)[0]
        # decision_function: negative = anomaly, positive = normal
        return float(np.clip(-raw / 0.5, 0, 1))


@dataclass
class ZoneBaseline:
    """Statistical baseline for a single zone."""
    hourly_counts: np.ndarray = field(default_factory=lambda: np.zeros(24))  # avg objects per hour
    hourly_variance: np.ndarray = field(default_factory=lambda: np.ones(24))
    activity_mean: float = 0.0
    activity_var: float = 1.0
    samples: int = 0


@dataclass
class AudioBaseline:
    """Normal ambient sound profile."""
    class_frequencies: dict = field(default_factory=dict)  # class -> avg occurrences per hour
    samples: int = 0


class BaselineLearner:
    """Learns normal spatiotemporal patterns from event streams.

    During onboarding (first N hours), builds statistical baselines:
    - Temporal activity distributions per zone
    - Object frequency baselines
    - Audio ambient profiles

    After onboarding, computes anomaly scores as deviation from baseline.
    """

    def __init__(self, onboarding_hours: float = 48.0, ema_alpha: float = 0.01):
        self.onboarding_hours = onboarding_hours
        self.ema_alpha = ema_alpha
        self._start_time = time.time()
        self._zone_baselines: dict[str, ZoneBaseline] = defaultdict(ZoneBaseline)
        self._audio_baseline = AudioBaseline()
        self._hourly_object_counts: dict[str, list[int]] = defaultdict(lambda: [0] * 24)
        self._event_buffer: list[dict] = []
        self._iforest = IsolationForestScorer()
        self._iforest_trained = False

    @property
    def is_onboarding(self) -> bool:
        return (time.time() - self._start_time) < self.onboarding_hours * 3600

    def ingest_event(self, event: dict) -> None:
        """Feed an event into the baseline learner."""
        self._event_buffer.append(event)
        self._iforest.ingest(event)
        ts = event.get("timestamp", time.time())
        hour = int((ts % 86400) / 3600)
        camera_id = event.get("camera_id", "default")

        # Count objects per zone/camera per hour
        n_objects = len(event.get("objects", []))
        zone_key = camera_id  # use camera as zone proxy until spatial layer assigns zones
        bl = self._zone_baselines[zone_key]
        bl.samples += 1

        if self.is_onboarding:
            # Accumulate during onboarding
            self._hourly_object_counts[zone_key][hour] += n_objects
        else:
            # Incremental EMA update
            bl.hourly_counts[hour] = (1 - self.ema_alpha) * bl.hourly_counts[hour] + self.ema_alpha * n_objects
            activity = event.get("scene_activity", 0)
            bl.activity_mean = (1 - self.ema_alpha) * bl.activity_mean + self.ema_alpha * activity

        # Audio baseline
        for audio in event.get("audio_events", []):
            cls = audio["class_name"]
            if cls not in self._audio_baseline.class_frequencies:
                self._audio_baseline.class_frequencies[cls] = 0.0
            self._audio_baseline.class_frequencies[cls] += 1
            self._audio_baseline.samples += 1

    def finalize_onboarding(self) -> None:
        """Compute baselines from accumulated onboarding data."""
        for zone_key, counts in self._hourly_object_counts.items():
            bl = self._zone_baselines[zone_key]
            arr = np.array(counts, dtype=float)
            total_hours = max(self.onboarding_hours / 24, 1)
            bl.hourly_counts = arr / total_hours
            bl.hourly_variance = np.maximum(bl.hourly_counts * 0.5, 1.0)

        total_hours = max((time.time() - self._start_time) / 3600, 1)
        for cls in self._audio_baseline.class_frequencies:
            self._audio_baseline.class_frequencies[cls] /= total_hours

        # Train Isolation Forest on onboarding data
        self._iforest.train()
        self._iforest_trained = True

        logger.info("Baseline finalized: %d zones, %d audio classes, iforest=%s",
                     len(self._zone_baselines), len(self._audio_baseline.class_frequencies),
                     self._iforest_trained)

    def compute_anomaly_score(self, event: dict) -> dict:
        """Compute multi-dimensional anomaly score for an event.

        Returns dict with overall score (0-1) and per-dimension scores.
        """
        ts = event.get("timestamp", time.time())
        hour = int((ts % 86400) / 3600)
        camera_id = event.get("camera_id", "default")
        n_objects = len(event.get("objects", []))

        bl = self._zone_baselines.get(camera_id)
        if bl is None or bl.samples < 10:
            return {"overall": 0.0, "temporal": 0.0, "count": 0.0, "audio": 0.0}

        # Temporal anomaly: object count vs expected for this hour
        expected = bl.hourly_counts[hour]
        variance = bl.hourly_variance[hour]
        count_z = abs(n_objects - expected) / max(np.sqrt(variance), 1.0)
        count_score = min(count_z / 3.0, 1.0)  # normalize z-score to 0-1

        # Activity anomaly
        activity = event.get("scene_activity", 0)
        activity_z = abs(activity - bl.activity_mean) / max(np.sqrt(bl.activity_var), 0.01)
        activity_score = min(activity_z / 3.0, 1.0)

        # Audio anomaly: unexpected sounds
        audio_score = 0.0
        for audio in event.get("audio_events", []):
            cls = audio["class_name"]
            expected_freq = self._audio_baseline.class_frequencies.get(cls, 0)
            if expected_freq < 0.1 and audio.get("is_alert"):
                audio_score = max(audio_score, audio["confidence"])

        overall = max(count_score * 0.4 + activity_score * 0.3 + audio_score * 0.3, 0.0)

        # Combine with Isolation Forest multivariate score
        iforest_score = self._iforest.score(event)

        if self._iforest_trained and iforest_score > 0:
            # Weighted blend: 50% z-score composite, 50% isolation forest
            overall = 0.5 * min(overall, 1.0) + 0.5 * iforest_score
        else:
            overall = min(overall, 1.0)

        return {
            "overall": round(min(overall, 1.0), 3),
            "count": round(count_score, 3),
            "activity": round(activity_score, 3),
            "audio": round(audio_score, 3),
            "iforest": round(iforest_score, 3),
        }

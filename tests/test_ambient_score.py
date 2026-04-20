"""Tests for the Ambient Intelligence Score engine."""

import time
import pytest
from cloud.services.ambient_score.engine import AmbientIntelligenceEngine, AmbientScore


@pytest.fixture
def engine():
    return AmbientIntelligenceEngine()


def _scene(objects=None, audio=None, activity=0.0, anomaly=0.0):
    return {
        "objects": objects or [],
        "audio_events": audio or [],
        "scene_activity": activity,
        "anomaly_score": {"score": anomaly},
        "timestamp": time.time(),
        "camera_id": "cam-test",
    }


class TestAmbientScore:
    def test_empty_scene_low_score(self, engine):
        result = engine.compute("zone-a", _scene())
        assert isinstance(result, AmbientScore)
        assert result.score < 25
        assert result.level == "safe"

    def test_many_people_raises_crowd_pressure(self, engine):
        people = [{"class_name": "person", "bbox": [0, 0, 1, 1], "track_id": f"p{i}"}
                  for i in range(25)]
        result = engine.compute("zone-a", _scene(objects=people))
        crowd = [c for c in result.contributions if c.signal_name == "crowd_pressure"]
        assert crowd[0].raw_value > 0.5

    def test_audio_alerts_raise_score(self, engine):
        audio = [
            {"class_name": "gunshot", "confidence": 0.95, "is_alert": True},
            {"class_name": "scream", "confidence": 0.8, "is_alert": True},
        ]
        result = engine.compute("zone-a", _scene(audio=audio))
        audio_signal = [c for c in result.contributions if c.signal_name == "audio_threat"]
        assert audio_signal[0].raw_value > 0.5

    def test_high_anomaly_score_propagates(self, engine):
        result = engine.compute("zone-a", _scene(anomaly=0.9))
        visual = [c for c in result.contributions if c.signal_name == "visual_anomaly"]
        assert visual[0].raw_value == 0.9

    def test_contributions_have_all_signals(self, engine):
        result = engine.compute("zone-a", _scene())
        signal_names = {c.signal_name for c in result.contributions}
        expected = {"visual_anomaly", "audio_threat", "crowd_pressure", "dwell_anomaly",
                    "temporal_normality", "historical_risk", "behavioral_stress",
                    "scene_tension", "trajectory_risk"}
        assert signal_names == expected

    def test_record_incident_increases_historical_risk(self, engine):
        now = time.time()
        # No incidents initially
        result1 = engine.compute("zone-a", _scene())
        hist1 = [c for c in result1.contributions if c.signal_name == "historical_risk"]

        # Record 5 incidents
        for i in range(5):
            engine.record_incident("zone-a", now - i * 3600)

        result2 = engine.compute("zone-a", _scene())
        hist2 = [c for c in result2.contributions if c.signal_name == "historical_risk"]

        assert hist2[0].raw_value > hist1[0].raw_value

    def test_trend_detection(self, engine):
        now = time.time()
        # Simulate rising scores
        for i in range(20):
            scene = _scene(anomaly=i * 0.05)
            scene["timestamp"] = now + i * 60
            engine.compute("zone-b", scene)

        trend = engine.get_trend("zone-b")
        assert trend["trend"] in ("rising", "stable")
        assert trend["samples"] > 0

    def test_level_thresholds(self, engine):
        result_safe = engine.compute("z", _scene(anomaly=0.0))
        assert result_safe.level == "safe"

    def test_convergence_boost(self, engine):
        """Multiple high signals should boost the score non-linearly."""
        people = [{"class_name": "person", "bbox": [0, 0, 1, 1], "track_id": f"p{i}"}
                  for i in range(30)]
        audio = [{"class_name": "gunshot", "confidence": 0.95, "is_alert": True}]
        result = engine.compute("zone-x", _scene(objects=people, audio=audio, anomaly=0.9))
        # With convergence boost, score should be notably higher
        assert result.score > 30

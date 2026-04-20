"""Tests for the Reasoning Engine — rule evaluation with temporal/spatial conditions."""

import time
import pytest
from cloud.services.reasoning.engine import ReasoningEngine, Rule, Severity, RuleResult


@pytest.fixture
def engine():
    e = ReasoningEngine()
    e.add_default_rules()
    return e


def _scene(objects=None, audio_events=None, timestamp=None):
    return {
        "objects": objects or [],
        "audio_events": audio_events or [],
        "spatial": None,
        "timestamp": timestamp or time.time(),
        "camera_id": "cam-test",
        "anomaly_score": {},
    }


class TestReasoningEngine:
    def test_no_triggers_on_empty_scene(self, engine):
        results = engine.evaluate(_scene())
        assert results == []

    def test_crowd_rule_triggers(self, engine):
        # 20 people should trigger the "crowd" rule (threshold=15)
        people = [{"class_name": "person", "track_id": f"p{i}", "bbox": [0, 0, 1, 1]}
                  for i in range(20)]
        results = engine.evaluate(_scene(objects=people))
        crowd = [r for r in results if r.rule_id == "crowd"]
        assert len(crowd) == 1
        assert crowd[0].severity == Severity.HIGH

    def test_crowd_below_threshold_no_trigger(self, engine):
        people = [{"class_name": "person", "track_id": f"p{i}", "bbox": [0, 0, 1, 1]}
                  for i in range(10)]
        results = engine.evaluate(_scene(objects=people))
        crowd = [r for r in results if r.rule_id == "crowd"]
        assert len(crowd) == 0

    def test_gunshot_audio_triggers(self, engine):
        audio = [{"class_name": "gunshot", "confidence": 0.95, "is_alert": True}]
        results = engine.evaluate(_scene(audio_events=audio))
        gunshot = [r for r in results if r.rule_id == "gunshot"]
        assert len(gunshot) == 1
        assert gunshot[0].severity == Severity.CRITICAL

    def test_distress_sound_triggers(self, engine):
        audio = [{"class_name": "scream", "confidence": 0.8, "is_alert": True}]
        results = engine.evaluate(_scene(audio_events=audio))
        distress = [r for r in results if r.rule_id == "distress_sound"]
        assert len(distress) == 1

    def test_cooldown_prevents_rapid_retrigger(self, engine):
        audio = [{"class_name": "gunshot", "confidence": 0.95, "is_alert": True}]
        now = time.time()

        results1 = engine.evaluate(_scene(audio_events=audio, timestamp=now))
        assert len([r for r in results1 if r.rule_id == "gunshot"]) == 1

        # 5 seconds later — should be suppressed by cooldown
        results2 = engine.evaluate(_scene(audio_events=audio, timestamp=now + 5))
        assert len([r for r in results2 if r.rule_id == "gunshot"]) == 0

        # 120 seconds later — should trigger again
        results3 = engine.evaluate(_scene(audio_events=audio, timestamp=now + 120))
        assert len([r for r in results3 if r.rule_id == "gunshot"]) == 1

    def test_add_custom_rule(self, engine):
        engine.add_rule(Rule(
            rule_id="test_custom",
            name="Test Custom",
            severity=Severity.LOW,
            conditions=[{"type": "count", "entity_class": "vehicle", "zone_id": "*", "min_count": 3}],
            action="alert_test",
        ))
        vehicles = [{"class_name": "vehicle", "track_id": f"v{i}", "bbox": [0, 0, 1, 1]}
                    for i in range(5)]
        results = engine.evaluate(_scene(objects=vehicles))
        custom = [r for r in results if r.rule_id == "test_custom"]
        assert len(custom) == 1

    def test_loitering_needs_duration(self, engine):
        # Loitering requires 180s — a single frame shouldn't trigger
        person = [{"class_name": "person", "track_id": "p1", "bbox": [0, 0, 1, 1]}]
        results = engine.evaluate(_scene(objects=person))
        loiter = [r for r in results if r.rule_id == "loitering"]
        assert len(loiter) == 0

    def test_rule_result_contains_explanation(self, engine):
        audio = [{"class_name": "gunshot", "confidence": 0.9, "is_alert": True}]
        results = engine.evaluate(_scene(audio_events=audio))
        assert results[0].explanation_chain
        assert any("gunshot" in c.lower() for c in results[0].explanation_chain)

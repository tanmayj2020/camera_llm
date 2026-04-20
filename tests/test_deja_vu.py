"""Tests for the Scene Déjà Vu engine."""

import time
import pytest
from cloud.services.deja_vu.engine import DejaVuEngine


@pytest.fixture
def engine():
    return DejaVuEngine()


def _scene(camera="cam-1", n_people=3, n_vehicles=0, audio=None, timestamp=None):
    objects = [{"class_name": "person", "bbox": [0.1 * i, 0.2, 0.1 * i + 0.1, 0.5],
                "track_id": f"p{i}"} for i in range(n_people)]
    objects += [{"class_name": "vehicle", "bbox": [0.5, 0.5, 0.7, 0.8],
                 "track_id": f"v{i}"} for i in range(n_vehicles)]
    return {
        "objects": objects,
        "audio_events": audio or [],
        "scene_activity": n_people * 0.1,
        "timestamp": timestamp or time.time(),
        "camera_id": camera,
    }


class TestDejaVu:
    def test_encode_and_store(self, engine):
        fp_id = engine.encode_and_store(_scene(), event_type="normal")
        assert fp_id.startswith("fp_")
        assert engine.get_stats()["total_fingerprints"] == 1

    def test_find_similar_needs_history(self, engine):
        matches = engine.find_similar(_scene())
        assert matches == []

    def test_similar_scenes_match(self, engine):
        t = time.time()
        # Store a historical scene
        engine.encode_and_store(
            _scene(n_people=5, timestamp=t - 3600),
            event_type="loitering",
            summary="5 people loitering near entrance",
        )

        # Query with a very similar scene
        matches = engine.find_similar(
            _scene(n_people=5, timestamp=t),
            min_similarity=0.5,
        )
        assert len(matches) >= 1
        assert matches[0].similarity > 0.5
        assert "similar" in matches[0].narrative.lower()

    def test_different_scenes_dont_match(self, engine):
        t = time.time()
        # Store scene with only vehicles
        engine.encode_and_store(
            _scene(n_people=0, n_vehicles=5, timestamp=t - 3600),
            event_type="traffic",
        )

        # Query with people-only scene
        matches = engine.find_similar(
            _scene(n_people=10, n_vehicles=0, timestamp=t),
            min_similarity=0.9,
        )
        assert len(matches) == 0

    def test_same_camera_filter(self, engine):
        t = time.time()
        engine.encode_and_store(
            _scene(camera="cam-1", n_people=3, timestamp=t - 3600),
            event_type="normal",
        )
        engine.encode_and_store(
            _scene(camera="cam-2", n_people=3, timestamp=t - 3600),
            event_type="normal",
        )

        matches = engine.find_similar(
            _scene(camera="cam-1", n_people=3, timestamp=t),
            same_camera_only=True,
            min_similarity=0.5,
        )
        for m in matches:
            assert m.historical_camera_id == "cam-1"

    def test_confirm_incident_flagged(self, engine):
        fp_id = engine.encode_and_store(_scene(), event_type="test")
        engine.confirm_incident(fp_id)

        # Check the fingerprint is marked
        for fp in engine._memory:
            if fp.fingerprint_id == fp_id:
                assert fp.was_confirmed is True

    def test_stats(self, engine):
        t = time.time()
        engine.encode_and_store(_scene(timestamp=t - 7200), event_type="a")
        engine.encode_and_store(_scene(timestamp=t - 3600), event_type="b")
        engine.encode_and_store(_scene(timestamp=t), event_type="a")

        stats = engine.get_stats()
        assert stats["total_fingerprints"] == 3
        assert stats["cameras"] >= 1
        assert stats["event_type_distribution"]["a"] == 2
        assert stats["event_type_distribution"]["b"] == 1

    def test_skips_recent_fingerprints(self, engine):
        """Should not self-match (skip last 60s)."""
        t = time.time()
        engine.encode_and_store(_scene(timestamp=t), event_type="x")
        matches = engine.find_similar(_scene(timestamp=t), min_similarity=0.0)
        assert len(matches) == 0  # too recent to match

    def test_top_k_limit(self, engine):
        t = time.time()
        for i in range(20):
            engine.encode_and_store(
                _scene(n_people=3, timestamp=t - (i + 1) * 3600),
                event_type="x",
            )
        matches = engine.find_similar(_scene(n_people=3, timestamp=t), top_k=3, min_similarity=0.3)
        assert len(matches) <= 3

"""Tests for the Gait DNA behavioral biometric fingerprinting engine."""

import time
import numpy as np
import pytest
from cloud.services.gait_dna.engine import GaitDNAEngine, GaitFingerprint


@pytest.fixture
def engine():
    return GaitDNAEngine()


def _simulate_walk(engine: GaitDNAEngine, entity_id: str, speed: float = 1.5,
                   cadence: float = 2.0, n_steps: int = 60, start_time: float = 0):
    """Simulate a walking person with given gait parameters."""
    t0 = start_time or time.time()
    x, y = 0.0, 0.0
    for i in range(n_steps):
        t = t0 + i * 0.1
        # Sinusoidal speed modulation (simulates walking cadence)
        v = speed * (1 + 0.2 * np.sin(2 * np.pi * cadence * i * 0.1))
        vx = v * 0.8
        vy = v * 0.6
        x += vx * 0.1
        y += vy * 0.1
        engine.observe(entity_id, x, y, vx, vy, timestamp=t)


class TestGaitDNA:
    def test_needs_minimum_observations(self, engine):
        for i in range(10):
            engine.observe("p1", float(i), 0.0, 1.0, 0.0)
        assert engine.get_fingerprint("p1") is None

    def test_fingerprint_computed_after_enough_data(self, engine):
        _simulate_walk(engine, "p1", n_steps=60)
        fp = engine.get_fingerprint("p1")
        assert fp is not None
        assert isinstance(fp, GaitFingerprint)
        assert fp.fingerprint.shape == (32,)
        assert fp.observation_count >= 30

    def test_same_person_high_similarity(self, engine):
        """Same walking pattern should produce similar fingerprints."""
        t = time.time()
        _simulate_walk(engine, "p1", speed=1.5, cadence=2.0, start_time=t)
        _simulate_walk(engine, "p2", speed=1.5, cadence=2.0, start_time=t + 100)

        fp1 = engine.get_fingerprint("p1")
        fp2 = engine.get_fingerprint("p2")
        assert fp1 is not None and fp2 is not None

        sim = float(np.dot(fp1.fingerprint, fp2.fingerprint) /
                    (np.linalg.norm(fp1.fingerprint) * np.linalg.norm(fp2.fingerprint) + 1e-8))
        assert sim > 0.7  # same gait → high similarity

    def test_different_gaits_lower_similarity(self, engine):
        """Different walking speeds/cadences should produce different fingerprints."""
        t = time.time()
        _simulate_walk(engine, "slow", speed=0.5, cadence=1.0, n_steps=80, start_time=t)
        _simulate_walk(engine, "fast", speed=3.0, cadence=3.0, n_steps=80, start_time=t + 200)

        fp_slow = engine.get_fingerprint("slow")
        fp_fast = engine.get_fingerprint("fast")
        assert fp_slow is not None and fp_fast is not None

        sim = float(np.dot(fp_slow.fingerprint, fp_fast.fingerprint) /
                    (np.linalg.norm(fp_slow.fingerprint) * np.linalg.norm(fp_fast.fingerprint) + 1e-8))
        # Different gaits → lower similarity
        assert sim < 0.95

    def test_match_returns_sorted(self, engine):
        t = time.time()
        _simulate_walk(engine, "ref", speed=1.5, cadence=2.0, start_time=t)
        _simulate_walk(engine, "similar", speed=1.5, cadence=2.0, start_time=t + 100)
        _simulate_walk(engine, "different", speed=3.0, cadence=3.5, n_steps=80, start_time=t + 300)

        matches = engine.match("ref")
        if matches:
            # If both match, similar should rank higher
            sims = [m.similarity for m in matches]
            assert sims == sorted(sims, reverse=True)

    def test_gallery_summary(self, engine):
        _simulate_walk(engine, "p1", n_steps=60)
        summary = engine.get_gallery_summary()
        assert len(summary) == 1
        assert summary[0]["entity_id"] == "p1"
        assert "confidence" in summary[0]

    def test_gallery_size(self, engine):
        assert engine.get_gallery_size() == 0
        _simulate_walk(engine, "p1", n_steps=60)
        assert engine.get_gallery_size() == 1

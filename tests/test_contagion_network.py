"""Tests for the Anomaly Contagion Network."""

import time
import pytest
from cloud.services.contagion.network import AnomalyContagionNetwork


@pytest.fixture
def network():
    return AnomalyContagionNetwork()


class TestContagionNetwork:
    def test_empty_network_no_predictions(self, network):
        warnings = network.predict_contagion("zone-a")
        assert warnings == []

    def test_record_and_propagation(self, network):
        t = time.time()
        # Event in zone-a, then zone-b 30s later
        network.record_event("zone-a", "loitering", t, zone_name="Zone A")
        network.record_event("zone-b", "crowd", t + 30, zone_name="Zone B")

        # Now predict from zone-a
        warnings = network.predict_contagion("zone-a")
        # zone-b should appear as at-risk
        zone_ids = [w.zone_id for w in warnings]
        assert "zone-b" in zone_ids

    def test_multiple_co_occurrences_increase_probability(self, network):
        t = time.time()
        # Simulate 10 correlated events
        for i in range(10):
            base = t + i * 600  # 10 min apart
            network.record_event("zone-a", "loitering", base)
            network.record_event("zone-b", "crowd", base + 45)  # 45s later

        warnings = network.predict_contagion("zone-a")
        zone_b_warning = [w for w in warnings if w.zone_id == "zone-b"]
        assert len(zone_b_warning) == 1
        # After 10 co-occurrences, probability should be significant
        assert zone_b_warning[0].risk_probability > 0.3

    def test_no_self_propagation(self, network):
        t = time.time()
        network.record_event("zone-a", "x", t)
        network.record_event("zone-a", "y", t + 10)
        warnings = network.predict_contagion("zone-a")
        self_refs = [w for w in warnings if w.zone_id == "zone-a"]
        assert len(self_refs) == 0

    def test_graph_structure(self, network):
        t = time.time()
        network.record_event("zone-a", "x", t)
        network.record_event("zone-b", "y", t + 60)

        graph = network.get_graph()
        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) >= 2

    def test_zone_risk_profile(self, network):
        t = time.time()
        network.record_event("zone-a", "x", t)
        network.record_event("zone-b", "y", t + 30)

        profile = network.get_zone_risk_profile("zone-b")
        assert profile["zone_id"] == "zone-b"
        assert "incoming_risk" in profile
        assert "outgoing_risk" in profile

    def test_events_beyond_window_ignored(self, network):
        t = time.time()
        # Events 10 minutes apart — beyond the 5-min propagation window
        network.record_event("zone-a", "x", t)
        network.record_event("zone-b", "y", t + 600)

        warnings = network.predict_contagion("zone-a")
        zone_b = [w for w in warnings if w.zone_id == "zone-b"]
        assert len(zone_b) == 0

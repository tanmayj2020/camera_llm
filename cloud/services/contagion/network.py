"""Anomaly Contagion Network — models how anomalies propagate across zones
like an epidemic, predicting which zones will be affected next.

When an incident occurs in Zone A, this module:
  1. Consults a learned zone-adjacency + historical propagation graph
  2. Estimates the probability that each neighboring zone will see a
     related event within the next N minutes
  3. Ranks zones by contagion risk and returns pre-emptive warnings

Learning loop:
  - Every confirmed incident is recorded with zone + timestamp
  - The system tracks temporal co-occurrences: "When Zone A fires,
    Zone B fires within 5 min 73% of the time"
  - These transition probabilities update continuously

Novel because: No surveillance system models anomaly propagation as a
contagion graph.  Security operators have no way to predict which zone
will be affected next after an initial incident.
"""

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Maximum time window to consider two events as "propagated" (seconds)
PROPAGATION_WINDOW_S = 300  # 5 minutes


@dataclass
class ContagionEdge:
    """A learned propagation link between two zones."""
    source_zone: str
    target_zone: str
    propagation_probability: float   # 0.0–1.0
    avg_delay_s: float               # average time from source to target
    observation_count: int
    last_observed: float


@dataclass
class ContagionWarning:
    """Pre-emptive warning that a zone may see a cascading anomaly."""
    zone_id: str
    zone_name: str
    risk_probability: float
    expected_delay_s: float
    source_zone: str
    source_event_type: str
    recommendation: str
    timestamp: float


class AnomalyContagionNetwork:
    """Models zone-to-zone anomaly propagation as a directed weighted graph."""

    def __init__(self):
        # Transition counts: (source_zone, target_zone) -> list of delay_seconds
        self._transitions: dict[tuple[str, str], list[float]] = defaultdict(list)
        # Recent events per zone: zone_id -> deque of (timestamp, event_type)
        self._recent_events: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=200))
        # Total events per zone (for probability normalization)
        self._zone_event_counts: dict[str, int] = defaultdict(int)
        # Zone display names
        self._zone_names: dict[str, str] = {}
        # Compiled graph (rebuilt periodically)
        self._graph: dict[str, list[ContagionEdge]] = defaultdict(list)
        self._last_rebuild = 0.0
        self._rebuild_interval = 60.0  # rebuild graph every 60s

    # ── Public API ────────────────────────────────────────────────────────

    def record_event(self, zone_id: str, event_type: str,
                     timestamp: float | None = None,
                     zone_name: str = ""):
        """Record an anomaly event in a zone.

        This both updates the contagion graph (by checking for temporal
        co-occurrence with recent events in other zones) and stores the
        event for future co-occurrence analysis.
        """
        now = timestamp or time.time()
        if zone_name:
            self._zone_names[zone_id] = zone_name

        # Check for propagation FROM other zones (events that preceded this one)
        for other_zone, events in self._recent_events.items():
            if other_zone == zone_id:
                continue
            for t, etype in events:
                delay = now - t
                if 0 < delay <= PROPAGATION_WINDOW_S:
                    self._transitions[(other_zone, zone_id)].append(delay)

        # Store this event
        self._recent_events[zone_id].append((now, event_type))
        self._zone_event_counts[zone_id] += 1

        # Periodically rebuild the graph
        if now - self._last_rebuild > self._rebuild_interval:
            self._rebuild_graph()
            self._last_rebuild = now

    def predict_contagion(self, source_zone: str,
                          event_type: str = "") -> list[ContagionWarning]:
        """Given a new event in source_zone, predict which other zones
        are at risk of a cascading event."""
        now = time.time()

        if source_zone not in self._graph:
            self._rebuild_graph()

        edges = self._graph.get(source_zone, [])
        warnings = []

        for edge in edges:
            if edge.propagation_probability < 0.1:
                continue  # too unlikely

            recommendation = self._generate_recommendation(edge)

            warnings.append(ContagionWarning(
                zone_id=edge.target_zone,
                zone_name=self._zone_names.get(edge.target_zone, edge.target_zone),
                risk_probability=round(edge.propagation_probability, 2),
                expected_delay_s=round(edge.avg_delay_s, 0),
                source_zone=source_zone,
                source_event_type=event_type,
                recommendation=recommendation,
                timestamp=now,
            ))

        # Sort by probability descending
        warnings.sort(key=lambda w: w.risk_probability, reverse=True)
        return warnings

    def get_graph(self) -> dict:
        """Return the full contagion graph for visualization."""
        self._rebuild_graph()
        nodes = []
        edges = []

        all_zones = set()
        for src, targets in self._graph.items():
            all_zones.add(src)
            for e in targets:
                all_zones.add(e.target_zone)

        for z in all_zones:
            nodes.append({
                "zone_id": z,
                "name": self._zone_names.get(z, z),
                "total_events": self._zone_event_counts.get(z, 0),
            })

        for src, targets in self._graph.items():
            for e in targets:
                if e.propagation_probability >= 0.1:
                    edges.append({
                        "source": e.source_zone,
                        "target": e.target_zone,
                        "probability": round(e.propagation_probability, 2),
                        "avg_delay_s": round(e.avg_delay_s, 0),
                        "observations": e.observation_count,
                    })

        return {"nodes": nodes, "edges": edges}

    def get_zone_risk_profile(self, zone_id: str) -> dict:
        """Return the risk profile for a specific zone:
        - What zones propagate TO this zone (incoming risk)
        - What zones this zone propagates TO (outgoing risk)
        """
        incoming = []
        outgoing = []

        for src, targets in self._graph.items():
            for e in targets:
                if e.target_zone == zone_id and e.propagation_probability >= 0.1:
                    incoming.append({
                        "source": src,
                        "probability": round(e.propagation_probability, 2),
                        "avg_delay_s": round(e.avg_delay_s, 0),
                    })
                if src == zone_id and e.propagation_probability >= 0.1:
                    outgoing.append({
                        "target": e.target_zone,
                        "probability": round(e.propagation_probability, 2),
                        "avg_delay_s": round(e.avg_delay_s, 0),
                    })

        return {
            "zone_id": zone_id,
            "name": self._zone_names.get(zone_id, zone_id),
            "total_events": self._zone_event_counts.get(zone_id, 0),
            "incoming_risk": sorted(incoming, key=lambda x: x["probability"], reverse=True),
            "outgoing_risk": sorted(outgoing, key=lambda x: x["probability"], reverse=True),
        }

    # ── Internals ─────────────────────────────────────────────────────────

    def _rebuild_graph(self):
        """Recompute the contagion graph from transition observations."""
        self._graph.clear()

        for (src, tgt), delays in self._transitions.items():
            src_total = max(1, self._zone_event_counts.get(src, 1))
            prob = len(delays) / src_total
            avg_delay = sum(delays) / len(delays) if delays else 0

            edge = ContagionEdge(
                source_zone=src,
                target_zone=tgt,
                propagation_probability=min(1.0, prob),
                avg_delay_s=avg_delay,
                observation_count=len(delays),
                last_observed=max(self._recent_events.get(src, [(0, "")])[0][0]
                                  if self._recent_events.get(src) else 0,
                                  0),
            )
            self._graph[src].append(edge)

    def _generate_recommendation(self, edge: ContagionEdge) -> str:
        prob = edge.propagation_probability
        delay = edge.avg_delay_s
        target = self._zone_names.get(edge.target_zone, edge.target_zone)

        if prob > 0.7:
            return (f"HIGH RISK: {target} has {prob:.0%} chance of cascading event "
                    f"within ~{delay:.0f}s. Pre-deploy security now.")
        elif prob > 0.4:
            return (f"MODERATE RISK: {target} may see related activity "
                    f"(~{prob:.0%}, ETA ~{delay:.0f}s). Increase monitoring.")
        else:
            return (f"LOW RISK: {target} has {prob:.0%} historical correlation. "
                    f"No action needed yet.")

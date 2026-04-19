"""Task 8: Neuro-Symbolic Reasoning Engine with temporal + spatial logic."""

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RuleResult:
    triggered: bool
    rule_id: str
    rule_name: str
    severity: Severity
    explanation_chain: list[str]
    evidence: dict = field(default_factory=dict)


@dataclass
class Rule:
    """A neuro-symbolic rule combining temporal and spatial conditions."""
    rule_id: str
    name: str
    severity: Severity
    conditions: list[dict]  # list of condition dicts
    action: str  # alert type
    cooldown_s: float = 60.0  # min seconds between triggers
    _last_triggered: float = 0.0


class ReasoningEngine:
    """Evaluates temporal-spatial rules against scene state.

    Rules are defined as structured dicts with conditions like:
    - {"type": "zone", "entity_class": "person", "zone_id": "restricted", "negate": False}
    - {"type": "time", "not_in_hours": [8, 9, 17, 18]}
    - {"type": "proximity", "entity_a_class": "person", "entity_b_class": "forklift", "max_distance": 3.0}
    - {"type": "duration", "entity_class": "person", "zone_id": "entrance", "min_seconds": 180}
    - {"type": "audio", "sound_class": "scream"}
    - {"type": "count", "entity_class": "person", "zone_id": "exit", "min_count": 20}
    """

    def __init__(self):
        self._rules: dict[str, Rule] = {}
        self._duration_tracker: dict[str, float] = {}  # (track_id, zone_id) -> first_seen

    def add_rule(self, rule: Rule):
        self._rules[rule.rule_id] = rule

    def add_default_rules(self):
        """Add common surveillance rules."""
        self.add_rule(Rule(
            rule_id="loitering", name="Loitering Detection",
            severity=Severity.MEDIUM,
            conditions=[
                {"type": "duration", "entity_class": "person", "zone_id": "*", "min_seconds": 180},
            ],
            action="alert_loitering",
        ))
        self.add_rule(Rule(
            rule_id="crowd", name="Crowd Formation",
            severity=Severity.HIGH,
            conditions=[
                {"type": "count", "entity_class": "person", "zone_id": "*", "min_count": 15},
            ],
            action="alert_crowd",
        ))
        self.add_rule(Rule(
            rule_id="distress_sound", name="Distress Sound Detected",
            severity=Severity.CRITICAL,
            conditions=[
                {"type": "audio", "sound_class": "scream"},
            ],
            action="alert_distress",
        ))
        self.add_rule(Rule(
            rule_id="gunshot", name="Gunshot Detected",
            severity=Severity.CRITICAL,
            conditions=[
                {"type": "audio", "sound_class": "gunshot"},
            ],
            action="alert_gunshot",
        ))

    def evaluate(self, scene_state: dict) -> list[RuleResult]:
        """Evaluate all rules against current scene state.

        scene_state should contain:
        - objects: list of {class_name, track_id, bbox, zone_id?}
        - audio_events: list of {class_name, confidence, is_alert}
        - spatial: SpatialMemory instance
        - timestamp: float
        - camera_id: str
        - anomaly_score: dict from baseline learner
        """
        results = []
        now = scene_state.get("timestamp", time.time())

        for rule in self._rules.values():
            if now - rule._last_triggered < rule.cooldown_s:
                continue

            triggered, chain, evidence = self._evaluate_rule(rule, scene_state)
            if triggered:
                rule._last_triggered = now
                results.append(RuleResult(
                    triggered=True, rule_id=rule.rule_id, rule_name=rule.name,
                    severity=rule.severity, explanation_chain=chain, evidence=evidence,
                ))

        return results

    def _evaluate_rule(self, rule: Rule, state: dict) -> tuple[bool, list[str], dict]:
        chain = []
        evidence = {}
        all_met = True

        for cond in rule.conditions:
            met, reason, ev = self._evaluate_condition(cond, state)
            chain.append(reason)
            evidence.update(ev)
            if not met:
                all_met = False
                break

        return all_met, chain, evidence

    def _evaluate_condition(self, cond: dict, state: dict) -> tuple[bool, str, dict]:
        ctype = cond["type"]

        if ctype == "audio":
            target_class = cond["sound_class"]
            for audio in state.get("audio_events", []):
                if audio["class_name"] == target_class and audio.get("is_alert"):
                    return True, f"Audio event '{target_class}' detected (conf={audio['confidence']})", \
                           {"audio_class": target_class, "confidence": audio["confidence"]}
            return False, f"No '{target_class}' audio detected", {}

        if ctype == "count":
            target_class = cond["entity_class"]
            min_count = cond["min_count"]
            count = sum(1 for o in state.get("objects", []) if o["class_name"] == target_class)
            met = count >= min_count
            return met, f"{target_class} count={count} (threshold={min_count})", {"count": count}

        if ctype == "duration":
            target_class = cond["entity_class"]
            min_seconds = cond["min_seconds"]
            now = state.get("timestamp", time.time())
            for obj in state.get("objects", []):
                if obj["class_name"] != target_class:
                    continue
                key = obj["track_id"]
                if key not in self._duration_tracker:
                    self._duration_tracker[key] = now
                duration = now - self._duration_tracker[key]
                if duration >= min_seconds:
                    return True, \
                        f"{target_class} track={obj['track_id']} present for {duration:.0f}s (threshold={min_seconds}s)", \
                        {"track_id": obj["track_id"], "duration": duration}
            return False, f"No {target_class} exceeds duration threshold", {}

        if ctype == "proximity":
            spatial = state.get("spatial")
            if spatial is None:
                return False, "No spatial data available", {}
            max_dist = cond["max_distance"]
            cls_a = cond["entity_a_class"]
            cls_b = cond["entity_b_class"]
            objs_a = [o for o in state.get("objects", []) if o["class_name"] == cls_a]
            objs_b = [o for o in state.get("objects", []) if o["class_name"] == cls_b]
            for a in objs_a:
                for b in objs_b:
                    dist = spatial.distance_between(a["track_id"], b["track_id"])
                    if dist is not None and dist < max_dist:
                        return True, \
                            f"{cls_a}({a['track_id']}) within {dist:.1f}m of {cls_b}({b['track_id']}) (threshold={max_dist}m)", \
                            {"distance": dist, "entity_a": a["track_id"], "entity_b": b["track_id"]}
            return False, f"No {cls_a}-{cls_b} proximity violation", {}

        if ctype == "zone":
            spatial = state.get("spatial")
            if spatial is None:
                return False, "No spatial data", {}
            zone_id = cond["zone_id"]
            target_class = cond["entity_class"]
            entities = spatial.entities_in_zone(zone_id)
            matches = [e for e in entities if e.class_name == target_class]
            negate = cond.get("negate", False)
            found = len(matches) > 0
            met = (not found) if negate else found
            return met, f"{len(matches)} {target_class}(s) in zone '{zone_id}'", \
                   {"zone_id": zone_id, "entity_count": len(matches)}

        if ctype == "time":
            from datetime import datetime, timezone
            now = state.get("timestamp", time.time())
            hour = datetime.fromtimestamp(now, tz=timezone.utc).hour
            not_in = cond.get("not_in_hours", [])
            met = hour not in not_in
            return met, f"Current hour={hour}, restricted hours={not_in}", {"hour": hour}

        return False, f"Unknown condition type: {ctype}", {}

    def cleanup_stale_trackers(self, max_age_s: float = 600):
        """Remove duration trackers older than max_age."""
        now = time.time()
        stale = [k for k, v in self._duration_tracker.items() if now - v > max_age_s]
        for k in stale:
            del self._duration_tracker[k]

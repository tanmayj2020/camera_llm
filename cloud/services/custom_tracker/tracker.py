"""Custom action tracker — users describe what to track in plain English.

LLM parses into structured rule + detection classes. Configures edge detection +
reasoning rules + notifications automatically. Supports VLM-based checks.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TrackerStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"


@dataclass
class CustomTracker:
    tracker_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""  # original natural language
    status: TrackerStatus = TrackerStatus.ACTIVE
    created_at: float = field(default_factory=time.time)

    # Parsed structured rule
    detection_classes: list[str] = field(default_factory=list)
    zone_filter: str | None = None
    time_filter: dict | None = None  # {"after": "22:00", "before": "06:00"}
    conditions: list[dict] = field(default_factory=list)
    severity: str = "medium"
    use_vlm_check: bool = False
    vlm_prompt: str = ""

    # Stats
    trigger_count: int = 0
    last_triggered: float | None = None


class CustomTrackerEngine:
    """Manages user-defined custom trackers.

    Flow: natural language → LLM parse → structured rule → reasoning engine integration.
    """

    def __init__(self, reasoner=None):
        self._trackers: dict[str, CustomTracker] = {}
        self._vlm_client = None
        self._reasoner = reasoner

    def set_vlm_client(self, client):
        self._vlm_client = client

    def set_reasoner(self, reasoner):
        self._reasoner = reasoner

    def create_from_text(self, description: str) -> CustomTracker:
        """Parse natural language into a structured tracker."""
        parsed = self._parse_description(description)
        tracker = CustomTracker(
            name=parsed.get("name", description[:50]),
            description=description,
            detection_classes=parsed.get("detection_classes", ["person"]),
            zone_filter=parsed.get("zone_filter"),
            time_filter=parsed.get("time_filter"),
            conditions=parsed.get("conditions", []),
            severity=parsed.get("severity", "medium"),
            use_vlm_check=parsed.get("use_vlm_check", False),
            vlm_prompt=parsed.get("vlm_prompt", ""),
        )
        self._trackers[tracker.tracker_id] = tracker

        # Register with reasoning engine
        if self._reasoner:
            self._register_rule(tracker)

        logger.info("Created tracker '%s' (id=%s) from: %s",
                    tracker.name, tracker.tracker_id, description[:80])
        return tracker

    def get_tracker(self, tracker_id: str) -> CustomTracker | None:
        return self._trackers.get(tracker_id)

    def list_trackers(self) -> list[CustomTracker]:
        return list(self._trackers.values())

    def pause_tracker(self, tracker_id: str) -> bool:
        t = self._trackers.get(tracker_id)
        if t:
            t.status = TrackerStatus.PAUSED
            return True
        return False

    def resume_tracker(self, tracker_id: str) -> bool:
        t = self._trackers.get(tracker_id)
        if t:
            t.status = TrackerStatus.ACTIVE
            return True
        return False

    def delete_tracker(self, tracker_id: str) -> bool:
        return self._trackers.pop(tracker_id, None) is not None

    def evaluate(self, scene_state: dict) -> list[dict]:
        """Evaluate all active trackers against current scene. Returns triggered alerts."""
        alerts = []
        for tracker in self._trackers.values():
            if tracker.status != TrackerStatus.ACTIVE:
                continue
            triggered, reason = self._evaluate_tracker(tracker, scene_state)
            if triggered:
                tracker.trigger_count += 1
                tracker.last_triggered = time.time()
                alerts.append({
                    "tracker_id": tracker.tracker_id,
                    "name": tracker.name,
                    "severity": tracker.severity,
                    "reason": reason,
                    "description": tracker.description,
                })
        return alerts

    def _evaluate_tracker(self, tracker: CustomTracker, state: dict) -> tuple[bool, str]:
        """Check if tracker conditions are met."""
        objects = state.get("objects", [])
        timestamp = state.get("timestamp", time.time())

        # Detection class filter
        matching = [o for o in objects if o.get("class_name") in tracker.detection_classes]
        if not matching and tracker.detection_classes:
            return False, ""

        # Time filter
        if tracker.time_filter:
            from datetime import datetime, timezone
            hour = datetime.fromtimestamp(timestamp, tz=timezone.utc).hour
            after = int(tracker.time_filter.get("after", "0").split(":")[0])
            before = int(tracker.time_filter.get("before", "24").split(":")[0])
            if after > before:  # overnight range (e.g., 22:00 - 06:00)
                if not (hour >= after or hour < before):
                    return False, ""
            elif not (after <= hour < before):
                return False, ""

        # VLM check for complex conditions
        if tracker.use_vlm_check and tracker.vlm_prompt:
            return self._vlm_check(tracker, state)

        # Basic condition evaluation
        for cond in tracker.conditions:
            ctype = cond.get("type", "")
            if ctype == "count":
                if len(matching) < cond.get("min_count", 1):
                    return False, ""
            elif ctype == "presence":
                if not matching:
                    return False, ""

        reason = f"Detected {len(matching)} {tracker.detection_classes} matching '{tracker.name}'"
        return True, reason

    def _vlm_check(self, tracker: CustomTracker, state: dict) -> tuple[bool, str]:
        """Use VLM for complex condition checking."""
        if not self._vlm_client:
            return False, ""

        client = self._vlm_client._get_client() if hasattr(self._vlm_client, '_get_client') else None
        if not client or client == "stub":
            return False, ""

        keyframe = state.get("keyframe_b64")
        if not keyframe:
            return False, ""

        try:
            content = [
                f"Check if this scene matches: '{tracker.vlm_prompt}'. "
                f"Reply JSON: {{\"match\": true/false, \"reason\": \"...\"}}",
                {"mime_type": "image/jpeg", "data": keyframe},
            ]
            resp = client.generate_content(content)
            text = resp.text.strip().strip("`").lstrip("json\n")
            data = json.loads(text)
            return data.get("match", False), data.get("reason", "")
        except Exception as e:
            logger.debug("VLM tracker check failed: %s", e)
            return False, ""

    def _parse_description(self, description: str) -> dict:
        """Parse natural language tracker description into structured rule."""
        # Try VLM parsing
        if self._vlm_client and hasattr(self._vlm_client, '_get_client'):
            client = self._vlm_client._get_client()
            if client != "stub":
                try:
                    prompt = (f"Parse this CCTV tracker rule into structured JSON:\n"
                              f"'{description}'\n\n"
                              f"Reply JSON: {{\"name\": \"short name\", "
                              f"\"detection_classes\": [\"person\"], "
                              f"\"zone_filter\": null, "
                              f"\"time_filter\": {{\"after\": \"22:00\", \"before\": \"06:00\"}} or null, "
                              f"\"conditions\": [{{\"type\": \"presence\"}}], "
                              f"\"severity\": \"medium\", "
                              f"\"use_vlm_check\": false, "
                              f"\"vlm_prompt\": \"\"}}")
                    resp = client.generate_content(prompt)
                    text = resp.text.strip().strip("`").lstrip("json\n")
                    return json.loads(text)
                except Exception as e:
                    logger.debug("VLM parse failed: %s", e)

        # Fallback: keyword-based parsing
        return self._keyword_parse(description)

    def _keyword_parse(self, desc: str) -> dict:
        """Simple keyword-based parsing fallback."""
        d = desc.lower()
        result = {
            "name": desc[:50],
            "detection_classes": ["person"],
            "conditions": [{"type": "presence"}],
            "severity": "medium",
            "use_vlm_check": False,
        }

        # Detect classes
        for cls in ["vehicle", "car", "truck", "bicycle", "dog", "cat", "backpack", "suitcase"]:
            if cls in d:
                result["detection_classes"].append(cls)

        # Detect time filters
        import re
        time_match = re.search(r'after\s+(\d{1,2})\s*(?:pm|:00)', d)
        if time_match:
            hour = int(time_match.group(1))
            if "pm" in d and hour < 12:
                hour += 12
            result["time_filter"] = {"after": f"{hour:02d}:00", "before": "06:00"}

        # Severity hints
        if any(w in d for w in ["urgent", "critical", "emergency", "weapon"]):
            result["severity"] = "critical"
        elif any(w in d for w in ["important", "high"]):
            result["severity"] = "high"

        # Complex conditions → use VLM
        if len(desc.split()) > 15 or any(w in d for w in ["wearing", "carrying", "running", "fighting"]):
            result["use_vlm_check"] = True
            result["vlm_prompt"] = desc

        return result

    def _register_rule(self, tracker: CustomTracker):
        """Register tracker as a reasoning engine rule."""
        from services.reasoning.engine import Rule, Severity
        severity_map = {"low": Severity.LOW, "medium": Severity.MEDIUM,
                        "high": Severity.HIGH, "critical": Severity.CRITICAL}
        rule = Rule(
            rule_id=f"tracker_{tracker.tracker_id}",
            name=tracker.name,
            severity=severity_map.get(tracker.severity, Severity.MEDIUM),
            conditions=tracker.conditions,
            action=f"tracker_alert_{tracker.tracker_id}",
        )
        self._reasoner.add_rule(rule)

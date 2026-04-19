"""Counterfactual explanations for VisionBrain alerts."""

import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Counterfactual:
    factor: str
    original_value: Any
    modified_value: Any
    would_trigger: bool
    description: str


class CounterfactualExplainer:
    def __init__(self, reasoner: Optional[Any] = None) -> None:
        self._reasoner = reasoner

    def _evaluate(self, scene_state: dict) -> bool:
        """Return True if any rule triggers."""
        if not self._reasoner:
            return False
        results = self._reasoner.evaluate(scene_state)
        return any(r.triggered for r in results)

    def explain(self, anomaly_dict: dict, scene_state: dict) -> List[Counterfactual]:
        if not self._reasoner:
            return []
        counterfactuals: List[Counterfactual] = []

        # a) Duration: halve object durations
        modified = copy.deepcopy(scene_state)
        objs = modified.get("objects", [])
        originals = [o.get("duration") for o in objs]
        if any(d is not None for d in originals):
            for o in objs:
                if "duration" in o:
                    o["duration"] = o["duration"] / 2
            if not self._evaluate(modified):
                counterfactuals.append(Counterfactual(
                    "duration", originals, [o.get("duration") for o in objs],
                    False, "object durations were halved",
                ))

        # b) Time: shift to 10am
        orig_ts = scene_state.get("timestamp")
        if orig_ts is not None:
            modified = copy.deepcopy(scene_state)
            modified["timestamp"] = float(int(orig_ts) - int(orig_ts) % 86400 + 36000)
            if not self._evaluate(modified):
                counterfactuals.append(Counterfactual(
                    "time", orig_ts, modified["timestamp"],
                    False, "timestamp was shifted to 10:00 AM (business hours)",
                ))

        # c) Zone: move to first non-restricted zone
        spatial = scene_state.get("spatial")
        if spatial and hasattr(spatial, "zones"):
            non_restricted = [z for z in spatial.zones if "restricted" not in str(z).lower()]
            if non_restricted:
                modified = copy.deepcopy(scene_state)
                for o in modified.get("objects", []):
                    if "zone" in o:
                        o["zone"] = non_restricted[0]
                if not self._evaluate(modified):
                    counterfactuals.append(Counterfactual(
                        "zone", "restricted", non_restricted[0],
                        False, f"entity was moved to non-restricted zone '{non_restricted[0]}'",
                    ))

        # d) Count: halve object count
        if len(scene_state.get("objects", [])) > 1:
            modified = copy.deepcopy(scene_state)
            half = len(modified["objects"]) // 2
            modified["objects"] = modified["objects"][:half]
            if not self._evaluate(modified):
                counterfactuals.append(Counterfactual(
                    "count", len(scene_state["objects"]), half,
                    False, f"object count was reduced from {len(scene_state['objects'])} to {half}",
                ))

        logger.debug("Generated %d counterfactuals", len(counterfactuals))
        return counterfactuals

    def format_explanation(self, counterfactuals: List[Counterfactual]) -> str:
        if not counterfactuals:
            return "No counterfactual explanations found."
        descs = [cf.description for cf in counterfactuals]
        return f"Would NOT trigger if: {', OR '.join(descs)}."

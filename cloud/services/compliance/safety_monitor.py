"""Workplace safety & SOP compliance."""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

PPE_CLASSES = ["hardhat", "safety_vest", "gloves", "goggles"]


@dataclass
class SOPRule:
    rule_id: str
    name: str
    required_steps: list[dict]
    zone_id: str
    severity: str = "warning"
    cooldown_s: float = 60.0


@dataclass
class SOPViolation:
    rule_id: str
    entity_id: str
    missing_step: str
    timestamp: float
    description: str


class SafetyMonitor:
    def __init__(self, spatial=None, reasoner=None):
        self._spatial = spatial
        self._reasoner = reasoner
        self.sop_rules: dict[str, SOPRule] = {}
        self.entity_step_progress: dict[str, dict] = {}
        self._last_violation_time: dict[str, float] = {}

    def add_sop_rule(self, rule: SOPRule) -> None:
        self.sop_rules[rule.rule_id] = rule

    def check_ppe(self, objects_list: list[dict]) -> list[dict]:
        violations = []
        persons = [o for o in objects_list if o.get("label") == "person"]
        ppe_items = [o for o in objects_list if o.get("label") in PPE_CLASSES]

        for person in persons:
            pb = person.get("bbox", [0, 0, 0, 0])
            px, py = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
            ph = abs(pb[3] - pb[1])
            proximity = max(ph * 1.5, 50)

            found = set()
            for item in ppe_items:
                ib = item.get("bbox", [0, 0, 0, 0])
                ix, iy = (ib[0] + ib[2]) / 2, (ib[1] + ib[3]) / 2
                if abs(ix - px) < proximity and abs(iy - py) < proximity:
                    found.add(item["label"])

            missing = [c for c in PPE_CLASSES if c not in found]
            if missing:
                violations.append({
                    "entity_id": person.get("id", "unknown"),
                    "missing_ppe": missing,
                    "bbox": pb,
                })
        return violations

    def detect_fall(self, entity_id: str, pose_keypoints: dict, prev_keypoints: dict) -> bool:
        for joint in ("hip", "shoulder"):
            curr_y = pose_keypoints.get(joint, {}).get("y")
            prev_y = prev_keypoints.get(joint, {}).get("y")
            if curr_y is None or prev_y is None:
                continue
            shoulder_y = prev_keypoints.get("shoulder", {}).get("y", 0)
            ankle_y = prev_keypoints.get("ankle", {}).get("y", 0)
            body_height = abs(ankle_y - shoulder_y) or 1
            if (curr_y - prev_y) > 0.4 * body_height:
                logger.warning("Fall detected for entity %s", entity_id)
                return True
        return False

    def check_sop_compliance(self, entity_id: str, scene_state: dict) -> list[SOPViolation]:
        violations = []
        now = time.time()

        for rule in self.sop_rules.values():
            if rule.zone_id and scene_state.get("zone_id") != rule.zone_id:
                continue

            cooldown_key = f"{rule.rule_id}:{entity_id}"
            if now - self._last_violation_time.get(cooldown_key, 0) < rule.cooldown_s:
                continue

            progress = self.entity_step_progress.setdefault(entity_id, {})
            rule_progress = progress.setdefault(rule.rule_id, set())

            for i, step in enumerate(rule.required_steps):
                step_id = step.get("id", str(i))
                cond = step.get("condition", "")
                if step_id in rule_progress:
                    continue
                if scene_state.get(cond):
                    rule_progress.add(step_id)
                else:
                    violations.append(SOPViolation(
                        rule_id=rule.rule_id, entity_id=entity_id,
                        missing_step=step.get("name", step_id),
                        timestamp=now,
                        description=f"Entity {entity_id} missing step: {step.get('name', step_id)}",
                    ))
                    self._last_violation_time[cooldown_key] = now
                    break

        return violations

    def evaluate(self, scene_state: dict) -> dict:
        objects_list = scene_state.get("objects", [])
        ppe_violations = self.check_ppe(objects_list)

        falls = []
        for entity in scene_state.get("entities", []):
            eid = entity.get("id", "unknown")
            pose = entity.get("pose_keypoints", {})
            prev = entity.get("prev_keypoints", {})
            if pose and prev and self.detect_fall(eid, pose, prev):
                falls.append(eid)

        sop_violations = []
        for entity in scene_state.get("entities", []):
            eid = entity.get("id", "unknown")
            sop_violations.extend(self.check_sop_compliance(eid, scene_state))

        return {
            "ppe_violations": ppe_violations,
            "falls_detected": falls,
            "sop_violations": [
                {"rule_id": v.rule_id, "entity_id": v.entity_id,
                 "missing_step": v.missing_step, "description": v.description}
                for v in sop_violations
            ],
        }

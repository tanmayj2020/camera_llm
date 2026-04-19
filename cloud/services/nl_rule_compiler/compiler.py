"""Natural Language Rule Compiler — compiles plain English into executable
ReasoningEngine rules.

Operators write rules like:
  "Alert me when someone loiters near the ATMs for more than 2 minutes after 10pm"

The LLM decomposes this into structured conditions and the compiler generates
a live Rule object that the ReasoningEngine can evaluate every frame.

This is fundamentally different from CameraProgrammer (which handles camera
schedule/priority programs).  This compiles *detection rules* — the actual
threat-detection logic.

Novel because: No commercial surveillance system lets security operators
author arbitrary detection logic in natural language.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Condition templates the LLM can emit ──────────────────────────────────

VALID_CONDITION_TYPES = {
    "zone",       # entity in zone
    "time",       # hour-of-day constraint
    "duration",   # entity present for N seconds
    "count",      # N+ entities in zone
    "proximity",  # two entity classes within distance
    "audio",      # sound class detected
    "speed",      # entity moving above/below speed
    "direction",  # entity heading toward named zone
    "absence",    # NO entity of class in zone for N seconds
    "object_co",  # two object classes co-present
    "sequence",   # event A followed by event B within N seconds
}

SYSTEM_PROMPT = """You are a surveillance rule compiler.  The user describes a
detection rule in plain English.  You must decompose it into a structured JSON
object with these fields:

{
  "name": "<short rule name>",
  "severity": "low" | "medium" | "high" | "critical",
  "conditions": [
    {
      "type": "<condition type>",
      ... <type-specific fields>
    }
  ],
  "description": "<one-line human summary>"
}

Supported condition types and their fields:

- zone:       {"type":"zone","entity_class":"person","zone_id":"<zone>"}
- time:       {"type":"time","after_hour":22,"before_hour":6}
              (means: trigger only between 10pm–6am)
- duration:   {"type":"duration","entity_class":"person","zone_id":"<zone>","min_seconds":120}
- count:      {"type":"count","entity_class":"person","zone_id":"<zone>","min_count":5}
- proximity:  {"type":"proximity","entity_a_class":"person","entity_b_class":"forklift","max_distance":3.0}
- audio:      {"type":"audio","sound_class":"glass_breaking"}
- speed:      {"type":"speed","entity_class":"person","min_speed":3.0}
              (m/s — ~3 m/s = running)
- direction:  {"type":"direction","entity_class":"person","toward_zone":"exit"}
- absence:    {"type":"absence","entity_class":"guard","zone_id":"lobby","min_seconds":600}
- object_co:  {"type":"object_co","class_a":"person","class_b":"weapon","zone_id":"*"}
- sequence:   {"type":"sequence","event_a":"door_open","event_b":"person_enter","max_gap_seconds":5}

Combine multiple conditions with AND logic.

Return ONLY valid JSON, no markdown.
"""


@dataclass
class CompiledRule:
    """A rule compiled from natural language."""
    rule_id: str
    name: str
    severity: str
    conditions: list[dict]
    source_text: str          # the original English instruction
    description: str
    compiled_at: float = field(default_factory=time.time)
    active: bool = True


class NLRuleCompiler:
    """Compiles natural-language instructions into ReasoningEngine Rule objects."""

    def __init__(self, vlm_client=None, reasoner=None):
        self._vlm = vlm_client
        self._reasoner = reasoner
        self._compiled: dict[str, CompiledRule] = {}

    def set_vlm_client(self, vlm):
        self._vlm = vlm

    # ── Public API ────────────────────────────────────────────────────────

    def compile(self, instruction: str) -> CompiledRule:
        """Compile a natural-language instruction into a structured rule.

        Returns a CompiledRule that can be hot-loaded into the ReasoningEngine.
        """
        parsed = self._llm_parse(instruction)
        rule_id = f"nl_{uuid.uuid4().hex[:8]}"

        compiled = CompiledRule(
            rule_id=rule_id,
            name=parsed.get("name", instruction[:40]),
            severity=parsed.get("severity", "medium"),
            conditions=self._validate_conditions(parsed.get("conditions", [])),
            source_text=instruction,
            description=parsed.get("description", instruction),
        )

        self._compiled[rule_id] = compiled

        # Hot-load into the live ReasoningEngine
        if self._reasoner:
            from services.reasoning.engine import Rule, Severity
            sev_map = {"low": Severity.LOW, "medium": Severity.MEDIUM,
                       "high": Severity.HIGH, "critical": Severity.CRITICAL}
            self._reasoner.add_rule(Rule(
                rule_id=compiled.rule_id,
                name=compiled.name,
                severity=sev_map.get(compiled.severity, Severity.MEDIUM),
                conditions=compiled.conditions,
                action=f"nl_alert_{rule_id}",
            ))
            logger.info("Rule '%s' hot-loaded into ReasoningEngine", compiled.name)

        return compiled

    def decompile(self, rule_id: str) -> str:
        """Return the original English instruction that produced a rule."""
        cr = self._compiled.get(rule_id)
        return cr.source_text if cr else ""

    def list_rules(self) -> list[dict]:
        return [
            {
                "rule_id": r.rule_id,
                "name": r.name,
                "severity": r.severity,
                "source_text": r.source_text,
                "description": r.description,
                "conditions": r.conditions,
                "active": r.active,
                "compiled_at": r.compiled_at,
            }
            for r in self._compiled.values()
        ]

    def delete_rule(self, rule_id: str) -> bool:
        if rule_id in self._compiled:
            del self._compiled[rule_id]
            if self._reasoner and rule_id in self._reasoner._rules:
                del self._reasoner._rules[rule_id]
            return True
        return False

    def toggle_rule(self, rule_id: str, active: bool) -> bool:
        cr = self._compiled.get(rule_id)
        if not cr:
            return False
        cr.active = active
        # Also toggle in the live reasoner
        if self._reasoner and rule_id in self._reasoner._rules:
            if not active:
                self._reasoner._rules[rule_id].cooldown_s = 1e12  # effectively disable
            else:
                self._reasoner._rules[rule_id].cooldown_s = 60.0
        return True

    def explain_rule(self, rule_id: str) -> str:
        """Generate a human-readable explanation of what the rule detects."""
        cr = self._compiled.get(rule_id)
        if not cr:
            return "Rule not found."

        parts = [f"Rule: {cr.name} (severity: {cr.severity})"]
        parts.append(f"Original instruction: \"{cr.source_text}\"")
        parts.append("Compiled conditions (all must be true):")
        for i, c in enumerate(cr.conditions, 1):
            parts.append(f"  {i}. {self._condition_to_english(c)}")
        return "\n".join(parts)

    # ── Internals ─────────────────────────────────────────────────────────

    def _llm_parse(self, instruction: str) -> dict:
        """Use VLM/LLM to parse English into structured conditions."""
        if self._vlm and hasattr(self._vlm, "_model"):
            try:
                import google.generativeai as genai
                model = self._vlm._model
                resp = model.generate_content(
                    [SYSTEM_PROMPT, f"User instruction: {instruction}"],
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                )
                return json.loads(resp.text)
            except Exception as e:
                logger.warning("LLM parse failed (%s), falling back to heuristics", e)

        # ── Heuristic fallback (works without LLM) ────────────────────────
        return self._heuristic_parse(instruction)

    def _heuristic_parse(self, text: str) -> dict:
        """Best-effort parse when no LLM is available."""
        import re
        text_lower = text.lower()
        conditions = []

        # Duration / loitering
        dur_match = re.search(r"(\d+)\s*(minute|min|second|sec|hour|hr)s?", text_lower)
        if dur_match:
            val = int(dur_match.group(1))
            unit = dur_match.group(2)
            seconds = val * ({"minute": 60, "min": 60, "second": 1, "sec": 1,
                              "hour": 3600, "hr": 3600}.get(unit, 60))
            conditions.append({
                "type": "duration", "entity_class": "person",
                "zone_id": "*", "min_seconds": seconds,
            })

        # Time constraint
        time_match = re.search(r"after\s+(\d{1,2})\s*(pm|am)?", text_lower)
        if time_match:
            hour = int(time_match.group(1))
            ampm = time_match.group(2)
            if ampm == "pm" and hour < 12:
                hour += 12
            conditions.append({"type": "time", "after_hour": hour, "before_hour": 6})

        # Count
        count_match = re.search(r"more than\s+(\d+)\s+(person|people)", text_lower)
        if count_match:
            conditions.append({
                "type": "count", "entity_class": "person",
                "zone_id": "*", "min_count": int(count_match.group(1)),
            })

        # Running / speed
        if any(w in text_lower for w in ("running", "run", "sprint")):
            conditions.append({"type": "speed", "entity_class": "person", "min_speed": 3.0})

        # Audio events
        for sound in ("gunshot", "scream", "glass_breaking", "explosion", "alarm"):
            if sound.replace("_", " ") in text_lower or sound in text_lower:
                conditions.append({"type": "audio", "sound_class": sound})

        # Zone extraction
        zone_match = re.search(r"(?:near|in|at|around)\s+(?:the\s+)?(\w[\w\s]{1,20}?)(?:\s+for|\s+after|$)", text_lower)
        if zone_match and conditions:
            zone_name = zone_match.group(1).strip().replace(" ", "_")
            for c in conditions:
                if c.get("zone_id") == "*":
                    c["zone_id"] = zone_name

        severity = "medium"
        if any(w in text_lower for w in ("critical", "emergency", "gunshot", "weapon")):
            severity = "critical"
        elif any(w in text_lower for w in ("high", "danger", "restricted")):
            severity = "high"
        elif any(w in text_lower for w in ("low", "minor")):
            severity = "low"

        if not conditions:
            conditions.append({
                "type": "zone", "entity_class": "person", "zone_id": "*",
            })

        return {
            "name": text[:50],
            "severity": severity,
            "conditions": conditions,
            "description": text,
        }

    def _validate_conditions(self, conditions: list[dict]) -> list[dict]:
        """Strip invalid condition types and ensure required fields."""
        valid = []
        for c in conditions:
            ctype = c.get("type")
            if ctype not in VALID_CONDITION_TYPES:
                logger.warning("Skipping unknown condition type: %s", ctype)
                continue
            valid.append(c)
        return valid if valid else [{"type": "zone", "entity_class": "person", "zone_id": "*"}]

    def _condition_to_english(self, c: dict) -> str:
        """Convert a condition dict back to readable English."""
        t = c.get("type", "?")
        if t == "zone":
            return f"A {c.get('entity_class', 'person')} is in zone '{c.get('zone_id', '*')}'"
        if t == "time":
            return f"Time is after {c.get('after_hour', '?')}:00 or before {c.get('before_hour', '?')}:00"
        if t == "duration":
            return (f"A {c.get('entity_class', 'person')} has been in zone "
                    f"'{c.get('zone_id', '*')}' for ≥{c.get('min_seconds', '?')}s")
        if t == "count":
            return (f"≥{c.get('min_count', '?')} {c.get('entity_class', 'person')}(s) "
                    f"in zone '{c.get('zone_id', '*')}'")
        if t == "proximity":
            return (f"A {c.get('entity_a_class')} is within {c.get('max_distance', '?')}m "
                    f"of a {c.get('entity_b_class')}")
        if t == "audio":
            return f"Audio event '{c.get('sound_class', '?')}' detected"
        if t == "speed":
            return f"A {c.get('entity_class', 'person')} moving ≥{c.get('min_speed', '?')} m/s"
        if t == "direction":
            return f"A {c.get('entity_class', 'person')} heading toward zone '{c.get('toward_zone', '?')}'"
        if t == "absence":
            return (f"No {c.get('entity_class', '?')} in zone '{c.get('zone_id', '?')}' "
                    f"for ≥{c.get('min_seconds', '?')}s")
        if t == "object_co":
            return f"A {c.get('class_a')} and {c.get('class_b')} co-present in '{c.get('zone_id', '*')}'"
        if t == "sequence":
            return (f"'{c.get('event_a')}' followed by '{c.get('event_b')}' "
                    f"within {c.get('max_gap_seconds', '?')}s")
        return str(c)

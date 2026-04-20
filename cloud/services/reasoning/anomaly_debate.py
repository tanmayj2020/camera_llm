"""Anomaly Explanation Debate — two LLM agents argue for/against an anomaly.

Prosecutor argues why it's suspicious. Defender argues why it's normal.
Operator sees both sides → dramatically reduces false positives.
"""

import json
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DebateResult:
    anomaly_id: str
    prosecutor_argument: str
    defender_argument: str
    verdict: str  # "suspicious", "likely_normal", "inconclusive"
    confidence: float
    key_disagreements: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class AnomalyDebateEngine:
    """Two-agent debate system for anomaly validation."""

    def __init__(self, vlm_client=None):
        self._vlm_client = vlm_client

    def _get_client(self):
        if self._vlm_client and hasattr(self._vlm_client, '_get_client'):
            return self._vlm_client._get_client()
        return "stub"

    def debate(self, anomaly: dict, context: dict = None) -> DebateResult:
        client = self._get_client()
        ctx = json.dumps(context or {}, default=str)[:2000]
        anomaly_str = json.dumps(anomaly, default=str)[:1500]

        if client == "stub":
            return self._stub_debate(anomaly)

        try:
            # Round 1: Prosecutor
            prosecutor_prompt = (
                f"You are a security PROSECUTOR. Argue why this event IS genuinely suspicious "
                f"and warrants immediate attention. Be specific.\n"
                f"Anomaly: {anomaly_str}\nContext: {ctx}\n"
                f"Give 3 bullet points for why this is a real threat.")
            pros_resp = client.generate_content(prosecutor_prompt).text.strip()

            # Round 2: Defender
            defender_prompt = (
                f"You are a security DEFENDER. Argue why this event is likely NORMAL "
                f"and a false alarm. Consider benign explanations.\n"
                f"Anomaly: {anomaly_str}\nContext: {ctx}\n"
                f"Prosecutor's argument: {pros_resp}\n"
                f"Give 3 bullet points for why this is NOT a real threat.")
            def_resp = client.generate_content(defender_prompt).text.strip()

            # Round 3: Judge
            judge_prompt = (
                f"You are an impartial JUDGE. Given both arguments, decide:\n"
                f"Prosecutor: {pros_resp}\nDefender: {def_resp}\n"
                f'Reply JSON: {{"verdict": "suspicious"|"likely_normal"|"inconclusive", '
                f'"confidence": 0.0-1.0, "key_disagreements": ["..."]}}')
            judge_resp = client.generate_content(judge_prompt).text.strip()
            if judge_resp.startswith("```"):
                judge_resp = judge_resp.split("\n", 1)[1].rsplit("```", 1)[0]
            verdict_data = json.loads(judge_resp)

            return DebateResult(
                anomaly_id=anomaly.get("event_id", ""),
                prosecutor_argument=pros_resp,
                defender_argument=def_resp,
                verdict=verdict_data.get("verdict", "inconclusive"),
                confidence=verdict_data.get("confidence", 0.5),
                key_disagreements=verdict_data.get("key_disagreements", []),
            )
        except Exception as e:
            logger.warning("Debate failed: %s", e)
            return self._stub_debate(anomaly)

    def _stub_debate(self, anomaly: dict) -> DebateResult:
        """Rule-based debate when VLM is unavailable."""
        severity = anomaly.get("severity", "medium")
        anomaly_type = anomaly.get("anomaly_type", "")
        desc = anomaly.get("description", "anomaly detected")
        from datetime import datetime, timezone
        hour = datetime.now(timezone.utc).hour

        # Prosecutor: build argument from anomaly metadata
        pros_points = [f"Alert triggered: {desc}"]
        if severity in ("high", "critical"):
            pros_points.append(f"Severity classified as {severity} by rule engine")
        if anomaly.get("audio_events"):
            pros_points.append(f"Corroborating audio: {anomaly['audio_events']}")
        if anomaly_type in ("loitering", "tailgating", "intrusion"):
            pros_points.append(f"{anomaly_type} is a known precursor to security incidents")

        # Defender: build counter-arguments
        def_points = []
        if 7 <= hour <= 18:
            def_points.append("Event occurred during business hours — higher normal activity expected")
        if severity in ("low", "medium"):
            def_points.append(f"Severity is only {severity} — below critical threshold")
        if anomaly_type in ("loitering",) and anomaly.get("duration_s", 0) < 120:
            def_points.append("Short duration — may be someone waiting for a ride/colleague")
        if not def_points:
            def_points.append("Insufficient context to rule out benign explanation")

        # Verdict: score-based
        score = {"low": 0.2, "medium": 0.4, "high": 0.7, "critical": 0.9}.get(severity, 0.4)
        if anomaly.get("audio_events"):
            score += 0.1
        if 22 <= hour or hour <= 5:
            score += 0.1  # nighttime events more suspicious
        score = min(1.0, score)
        verdict = "suspicious" if score > 0.6 else ("likely_normal" if score < 0.35 else "inconclusive")

        return DebateResult(
            anomaly_id=anomaly.get("event_id", ""),
            prosecutor_argument="\n".join(f"• {p}" for p in pros_points),
            defender_argument="\n".join(f"• {d}" for d in def_points),
            verdict=verdict,
            confidence=round(score, 2),
            key_disagreements=["VLM unavailable — rule-based analysis only"],
        )

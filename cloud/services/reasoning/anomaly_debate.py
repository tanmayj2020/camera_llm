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
        severity = anomaly.get("severity", "medium")
        return DebateResult(
            anomaly_id=anomaly.get("event_id", ""),
            prosecutor_argument=f"Alert triggered: {anomaly.get('description', 'anomaly detected')}",
            defender_argument="Insufficient context to rule out benign explanation.",
            verdict="inconclusive" if severity in ("low", "medium") else "suspicious",
            confidence=0.4,
        )

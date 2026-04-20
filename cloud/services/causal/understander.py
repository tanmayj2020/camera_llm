"""Task 9: Causal Anomaly Understanding with VLM + Hallucination Guardrails.

VLM: Gemini 2.5 Pro (SOTA multimodal reasoning, 2025-2026) with structured JSON output.
Configurable model — swap to gemini-3.0-pro or claude-4 when available.
"""

import base64
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CausalAnalysis:
    description: str
    causal_explanation: str
    severity: str
    recommended_action: str
    confidence: float
    claims: list[dict]  # [{claim, verified, evidence_ref, confidence}]
    grounded: bool  # True if all claims verified


class CausalUnderstander:
    """Generates causal explanations for anomalies using VLM with hallucination guardrails.

    Uses Gemini 2.5 Pro with structured JSON output mode for reliable parsing.
    Pipeline: anomaly → VLM analysis → claim extraction → evidence verification → grounded output.
    """

    def __init__(self, model: str = "gemini-2.5-pro"):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import google.generativeai as genai
                self._client = genai.GenerativeModel(
                    self.model,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        temperature=0.2,  # Low temperature for factual analysis
                    ),
                )
            except Exception as e:
                logger.warning("Gemini init failed: %s — using stub", e)
                self._client = "stub"
        return self._client

    def analyze(self, anomaly: dict, keyframes_b64: list[str],
                graph_context: list[dict], audio_context: list[dict] | None = None) -> CausalAnalysis:
        """Generate causal analysis for an anomaly event."""

        # Build context prompt
        context_parts = []
        if graph_context:
            context_parts.append(f"Knowledge graph context:\n{json.dumps(graph_context[:10], indent=2)}")
        if audio_context:
            context_parts.append(f"Audio events:\n{json.dumps(audio_context, indent=2)}")
        context_parts.append(f"Anomaly details:\n{json.dumps(anomaly, indent=2, default=str)}")

        prompt = f"""You are a CCTV analytics AI. Analyze this anomaly and provide:
1. DESCRIPTION: What is happening in the scene (factual, based only on visual evidence)
2. CAUSAL_EXPLANATION: Why this is anomalous (reference specific evidence)
3. SEVERITY: low/medium/high/critical with justification
4. RECOMMENDED_ACTION: What should be done

Context:
{chr(10).join(context_parts)}

CRITICAL: Only state facts you can verify from the images and context provided.
If uncertain, say "uncertain" rather than guessing.

Respond in JSON format:
{{"description": "...", "causal_explanation": "...", "severity": "...",
  "recommended_action": "...", "confidence": 0.0-1.0,
  "claims": [{{"claim": "...", "evidence_ref": "..."}}]}}"""

        client = self._get_client()
        if client == "stub":
            return self._stub_analysis(anomaly)

        try:
            # Build multimodal content
            content = [prompt]
            for kf in keyframes_b64[:4]:  # max 4 keyframes
                content.append({"mime_type": "image/jpeg", "data": kf})

            response = client.generate_content(content)
            result = self._parse_response(response.text)
            # Verify claims
            result.claims = self._verify_claims(result.claims, graph_context, anomaly)
            result.grounded = all(c.get("verified", False) for c in result.claims)
            if not result.grounded:
                result.description += " [Some claims could not be verified — flagged for human review]"
            return result
        except Exception as e:
            logger.error("VLM analysis failed: %s", e)
            return self._stub_analysis(anomaly)

    def _parse_response(self, text: str) -> CausalAnalysis:
        try:
            # Strip markdown code fences if present
            clean = text.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(clean)
            return CausalAnalysis(
                description=data.get("description", ""),
                causal_explanation=data.get("causal_explanation", ""),
                severity=data.get("severity", "medium"),
                recommended_action=data.get("recommended_action", ""),
                confidence=data.get("confidence", 0.5),
                claims=data.get("claims", []),
                grounded=False,
            )
        except json.JSONDecodeError:
            return CausalAnalysis(
                description=text[:500], causal_explanation="", severity="medium",
                recommended_action="Review manually", confidence=0.3, claims=[], grounded=False,
            )

    def _verify_claims(self, claims: list[dict], graph_context: list[dict], anomaly: dict) -> list[dict]:
        """Kestrel-inspired: verify each factual claim against available evidence."""
        graph_text = json.dumps(graph_context, default=str).lower()
        anomaly_text = json.dumps(anomaly, default=str).lower()

        for claim in claims:
            claim_text = claim.get("claim", "").lower()
            evidence_ref = claim.get("evidence_ref", "").lower()
            # Check if key terms from claim appear in evidence
            key_terms = [w for w in claim_text.split() if len(w) > 3]
            matches = sum(1 for t in key_terms if t in graph_text or t in anomaly_text)
            claim["verified"] = matches >= max(len(key_terms) // 2, 1) if key_terms else False
            claim["confidence"] = round(matches / max(len(key_terms), 1), 2)
        return claims

    def _stub_analysis(self, anomaly: dict) -> CausalAnalysis:
        return CausalAnalysis(
            description=f"Anomaly detected: {anomaly.get('anomaly_type', 'unknown')}",
            causal_explanation="VLM unavailable — analysis based on rule engine output only",
            severity=anomaly.get("severity", "medium"),
            recommended_action="Review camera feed manually",
            confidence=0.3, claims=[], grounded=False,
        )

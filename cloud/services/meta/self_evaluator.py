"""System Self-Evaluation — health metrics, FP analysis, coverage gap detection."""

import logging
import time

logger = logging.getLogger(__name__)


class SelfEvaluator:
    """Evaluates overall system health and generates narrative reports via VLM."""

    def __init__(self, persistence=None, continual_learner=None, action_engine=None, vlm_client=None):
        self._persistence = persistence
        self._learner = continual_learner
        self._action_engine = action_engine
        self._vlm_client = vlm_client
        self._client = None

    def _get_client(self):
        if self._client is None:
            if self._vlm_client:
                self._client = self._vlm_client
            else:
                try:
                    import google.generativeai as genai
                    self._client = genai.GenerativeModel("gemini-2.0-flash")
                except Exception as e:
                    logger.warning("Gemini init failed: %s — using stub", e)
                    self._client = "stub"
        return self._client

    def evaluate(self) -> dict:
        """Run all evaluation checks and return combined metrics."""
        return {
            "fp_rates": self._eval_fp_rates(),
            "avg_ack_time_s": self._eval_ack_time(),
            "coverage_gaps": self._eval_coverage_gaps(),
            "drift_detected": self._eval_drift(),
        }

    def get_metrics(self) -> dict:
        return self.evaluate()

    def generate_health_report(self) -> str:
        metrics = self.evaluate()
        client = self._get_client()
        if client != "stub":
            try:
                prompt = (
                    "You are a CCTV analytics system health advisor. Given these metrics, "
                    "write a concise narrative health report with actionable recommendations.\n\n"
                    f"FP rates per rule: {metrics['fp_rates']}\n"
                    f"Avg alert-to-ack time: {metrics['avg_ack_time_s']:.1f}s\n"
                    f"Coverage gaps (cameras with no recent events): {metrics['coverage_gaps']}\n"
                    f"Concept drift detected: {metrics['drift_detected']}\n"
                )
                resp = client.generate_content(prompt)
                return resp.text
            except Exception as e:
                logger.error("VLM health report failed: %s", e)
        # Stub fallback — bullet-point template
        lines = ["System Health Report", "=" * 40]
        for rule, rate in metrics["fp_rates"].items():
            lines.append(f"• {rule}: FP rate {rate:.0%}" + (" — consider increasing threshold" if rate > 0.5 else ""))
        lines.append(f"• Avg alert-to-ack time: {metrics['avg_ack_time_s']:.1f}s")
        if metrics["coverage_gaps"]:
            lines.append(f"• Coverage gaps (possible offline): {', '.join(metrics['coverage_gaps'])}")
        lines.append(f"• Drift detected: {metrics['drift_detected']}")
        return "\n".join(lines)

    # ---- internal evaluators ----

    def _eval_fp_rates(self) -> dict[str, float]:
        if not self._learner:
            return {}
        rates = {}
        counts = getattr(self._learner, "_total_trigger_counts", getattr(self._learner, "_tp_counts", {}))
        for rule_id in counts:
            if hasattr(self._learner, "get_fp_rate"):
                rates[rule_id] = self._learner.get_fp_rate(rule_id)
            else:
                fp = getattr(self._learner, "_false_positive_counts", getattr(self._learner, "_fp_counts", {}))
                total = counts.get(rule_id, 0)
                rates[rule_id] = fp.get(rule_id, 0) / total if total else 0.0
        return rates

    def _eval_ack_time(self) -> float:
        if not self._persistence:
            return 0.0
        alerts = self._persistence.get_alerts(limit=200)
        ack_times = []
        for a in alerts:
            if a.get("acknowledged") and a.get("acked_at") and a.get("timestamp"):
                ack_times.append(a["acked_at"] - a["timestamp"])
        return sum(ack_times) / len(ack_times) if ack_times else 0.0

    def _eval_coverage_gaps(self) -> list[str]:
        if not self._persistence:
            return []
        alerts = self._persistence.get_alerts(limit=500)
        cutoff = time.time() - 4 * 3600  # 4 hours
        active_cameras = {a["camera_id"] for a in alerts if a.get("timestamp", 0) > cutoff and a.get("camera_id")}
        all_cameras = {a["camera_id"] for a in alerts if a.get("camera_id")}
        return sorted(all_cameras - active_cameras)

    def _eval_drift(self) -> bool:
        if not self._learner:
            return False
        detector = getattr(self._learner, "drift_detector", None)
        if detector and hasattr(detector, "_count"):
            return detector._count == 0 and detector._sum == 0.0  # was recently reset after drift
        return False

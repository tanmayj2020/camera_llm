"""Daily alert digest with LLM summarization."""

import logging
import time
from collections import Counter

logger = logging.getLogger(__name__)


class AlertDigest:
    def __init__(self, persistence=None, vlm_client=None, continual_learner=None,
                 action_engine=None):
        self._persistence = persistence
        self._vlm_client = vlm_client
        self._continual_learner = continual_learner
        self._action_engine = action_engine

    def _get_client(self):
        return self._vlm_client

    def generate(self, camera_id: str | None = None, hours: int = 24) -> dict:
        cutoff = time.time() - hours * 3600
        alerts = self._query_alerts(cutoff, camera_id)

        by_severity: dict[str, int] = Counter(a.get("severity", "unknown") for a in alerts)
        by_camera: dict[str, int] = Counter(a.get("camera_id", "unknown") for a in alerts)
        by_type: dict[str, int] = Counter(a.get("type", "unknown") for a in alerts)

        top_3_cameras = [c for c, _ in by_camera.most_common(3)]
        top_3_types = [t for t, _ in by_type.most_common(3)]

        stats: dict = {
            "total_alerts": len(alerts),
            "by_severity": dict(by_severity),
            "by_camera": dict(by_camera),
            "by_type": dict(by_type),
            "top_3_cameras": top_3_cameras,
            "top_3_types": top_3_types,
            "hours": hours,
        }

        # FP rates from continual learner
        if self._continual_learner:
            fp_rates = {}
            for rule_type in by_type:
                rate = self._continual_learner.get_fp_rate(rule_type)
                if rate > 0:
                    fp_rates[rule_type] = round(rate, 2)
            stats["fp_rates"] = fp_rates

        # Outcome stats from action engine
        if self._action_engine and hasattr(self._action_engine, '_outcome_stats'):
            stats["outcome_stats"] = dict(self._action_engine._outcome_stats)

        stats["narrative"] = self._generate_narrative(stats)
        return stats

    def _query_alerts(self, cutoff: float, camera_id: str | None) -> list[dict]:
        if not self._persistence:
            return []
        try:
            alerts = self._persistence.query_alerts(since=cutoff, camera_id=camera_id)
            return alerts if isinstance(alerts, list) else []
        except Exception:
            logger.warning("Failed to query alerts from persistence")
            return []

    def _generate_narrative(self, stats: dict) -> str:
        client = self._get_client()
        if client:
            try:
                prompt = (f"Summarize this alert digest in 3-5 sentences for a security manager: "
                          f"Total alerts: {stats['total_alerts']}, "
                          f"by severity: {stats['by_severity']}, "
                          f"top cameras: {stats['top_3_cameras']}, "
                          f"top types: {stats['top_3_types']}.")
                resp = client.generate(prompt)
                return resp if isinstance(resp, str) else resp.text
            except Exception:
                logger.warning("VLM digest narrative failed, using template")

        # Fallback template
        total = stats["total_alerts"]
        if total == 0:
            return f"No alerts in the past {stats['hours']} hours."
        sev = stats["by_severity"]
        critical = sev.get("critical", 0) + sev.get("high", 0)
        parts = [f"{total} alerts in the past {stats['hours']} hours."]
        if critical:
            parts.append(f"{critical} critical/high severity.")
        if stats["top_3_cameras"]:
            parts.append(f"Most active cameras: {', '.join(stats['top_3_cameras'])}.")
        if stats["top_3_types"]:
            parts.append(f"Top alert types: {', '.join(stats['top_3_types'])}.")
        fp_rates = stats.get("fp_rates", {})
        if fp_rates:
            high_fp = [f"{k} ({v:.0%})" for k, v in fp_rates.items() if v > 0.2]
            if high_fp:
                parts.append(f"High false-positive rates: {', '.join(high_fp)}.")
        return " ".join(parts)

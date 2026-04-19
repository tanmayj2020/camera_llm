"""Plain English alert explainer using VLM with fallback templates."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class AlertExplainer:
    """Generates plain English explanations for security alerts."""

    baseline: object = None
    kg: object = None
    vlm_client: object = None

    def explain(self, anomaly_dict: dict, camera_id: str) -> str:
        hour = datetime.now(timezone.utc).hour
        expected_count = self._get_expected(camera_id, hour)
        recent = self._get_recent(camera_id)
        actual = anomaly_dict.get("count", anomaly_dict.get("actual", "unknown"))

        prompt = (
            f"Explain this security alert in one plain English sentence for a non-technical operator: "
            f"{anomaly_dict}. Context: normally {expected_count} people at this hour, current count is {actual}. "
            f"Recent events: {recent}."
        )

        if self.vlm_client is not None:
            client = self.vlm_client._get_client()
            if client != "stub":
                try:
                    return client.generate_content(prompt).text
                except Exception as e:
                    logger.warning("VLM explain failed: %s — using fallback", e)

        return (
            f"{anomaly_dict.get('anomaly_type', 'Anomaly')} at {camera_id}: "
            f"{anomaly_dict.get('description', 'unusual activity detected')}. "
            f"This is unusual because the baseline expects {expected_count} at this time."
        )

    def _get_expected(self, camera_id: str, hour: int):
        if self.baseline is not None:
            try:
                return self.baseline.get_hourly_baseline(camera_id, hour)
            except Exception as e:
                logger.debug("Baseline lookup failed: %s", e)
        return "unknown"

    def _get_recent(self, camera_id: str) -> list:
        if self.kg is not None:
            try:
                return self.kg.get_recent_events(camera_id, limit=5)
            except Exception as e:
                logger.debug("KG lookup failed: %s", e)
        return []

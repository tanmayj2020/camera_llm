"""Task 14: KPI Engine + Daily Business Summary."""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class KPIReport:
    period: str  # "daily", "weekly"
    timestamp: float
    metrics: dict  # {metric_name: value}
    trends: dict   # {metric_name: {direction, change_pct, comparison}}
    anomaly_summary: list[dict]
    predictions: list[dict]
    recommendations: list[str]
    narrative: str  # LLM-generated natural language summary


class KPIEngine:
    """Computes business KPIs from knowledge graph + BigQuery data."""

    def __init__(self, kg=None, vlm_client=None):
        self.kg = kg
        self.vlm = vlm_client
        self._historical_kpis: list[dict] = []

    def compute_daily_kpis(self, camera_id: str, date: str | None = None) -> dict:
        """Compute KPIs for a given day."""
        kpis = {
            "foot_traffic": 0,
            "peak_hour": 0,
            "peak_hour_count": 0,
            "avg_dwell_time_s": 0,
            "incident_count": 0,
            "zone_utilization": {},
            "busiest_zone": "",
            "audio_alerts": 0,
        }

        if not self.kg:
            return kpis

        # Query knowledge graph for daily stats
        with self.kg._driver.session() as s:
            # Foot traffic: unique persons
            result = s.run("""
                MATCH (p:Person)-[r:DETECTED_IN]->(c:Camera {camera_id: $cid})
                WHERE r.last_seen >= datetime() - duration('P1D')
                RETURN count(DISTINCT p) AS total
            """, cid=camera_id)
            record = result.single()
            if record:
                kpis["foot_traffic"] = record["total"]

            # Incidents
            result = s.run("""
                MATCH (e:Event)-[:OCCURRED_AT]->(c:Camera {camera_id: $cid})
                WHERE e.timestamp >= datetime() - duration('P1D')
                  AND e.event_type CONTAINS 'alert'
                RETURN count(e) AS incidents
            """, cid=camera_id)
            record = result.single()
            if record:
                kpis["incident_count"] = record["incidents"]

            # Audio alerts
            result = s.run("""
                MATCH (a:AudioEvent)-[:HEARD_AT]->(c:Camera {camera_id: $cid})
                WHERE a.timestamp >= datetime() - duration('P1D')
                RETURN count(a) AS alerts
            """, cid=camera_id)
            record = result.single()
            if record:
                kpis["audio_alerts"] = record["alerts"]

        return kpis

    def compute_trends(self, current: dict, previous: dict) -> dict:
        trends = {}
        for key in current:
            if isinstance(current[key], (int, float)) and key in previous:
                prev_val = previous[key]
                curr_val = current[key]
                if prev_val > 0:
                    change = ((curr_val - prev_val) / prev_val) * 100
                    direction = "up" if change > 0 else "down" if change < 0 else "flat"
                    trends[key] = {"direction": direction, "change_pct": round(change, 1),
                                   "current": curr_val, "previous": prev_val}
        return trends

    def generate_summary(self, kpis: dict, trends: dict, anomalies: list[dict],
                         predictions: list[dict]) -> KPIReport:
        """Generate LLM-powered business summary."""
        narrative = self._generate_narrative(kpis, trends, anomalies, predictions)
        recommendations = self._generate_recommendations(kpis, trends)

        return KPIReport(
            period="daily",
            timestamp=time.time(),
            metrics=kpis,
            trends=trends,
            anomaly_summary=anomalies[:10],
            predictions=predictions[:5],
            recommendations=recommendations,
            narrative=narrative,
        )

    def _generate_narrative(self, kpis: dict, trends: dict, anomalies: list, predictions: list) -> str:
        if self.vlm and hasattr(self.vlm, '_get_client'):
            client = self.vlm._get_client()
            if client != "stub":
                try:
                    prompt = f"""Generate a concise daily business intelligence summary from CCTV analytics:

KPIs: {json.dumps(kpis, indent=2)}
Trends vs previous period: {json.dumps(trends, indent=2)}
Anomalies detected: {len(anomalies)}
Predictions: {json.dumps(predictions[:3], indent=2, default=str)}

Write 3-5 sentences covering: key metrics, notable trends, incidents, and outlook.
Be specific with numbers. Use business-friendly language."""
                    response = client.generate_content(prompt)
                    return response.text
                except Exception as e:
                    logger.error("Narrative generation failed: %s", e)

        # Fallback
        parts = [f"Today's foot traffic: {kpis.get('foot_traffic', 0)} visitors."]
        if trends:
            for k, v in list(trends.items())[:3]:
                parts.append(f"{k}: {v['direction']} {abs(v['change_pct'])}% vs previous period.")
        if anomalies:
            parts.append(f"{len(anomalies)} incidents detected.")
        return " ".join(parts)

    def _generate_recommendations(self, kpis: dict, trends: dict) -> list[str]:
        recs = []
        for key, trend in trends.items():
            if trend["direction"] == "down" and abs(trend["change_pct"]) > 20:
                recs.append(f"Investigate {key} decline ({trend['change_pct']}%) — may indicate operational issue.")
            if key == "incident_count" and trend.get("direction") == "up":
                recs.append("Rising incident count — review security protocols and camera coverage.")
        if kpis.get("foot_traffic", 0) == 0:
            recs.append("No foot traffic detected — verify camera connectivity.")
        return recs or ["All metrics within normal ranges. No action required."]

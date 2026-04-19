"""Scheduled report generation."""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class ReportSchedule:
    name: str
    schedule_type: str  # "daily_8am", "daily_6pm", "weekly_monday_9am", "monthly_1st"
    report_type: str    # "kpi_summary", "incident_report", "occupancy_report"
    camera_ids: list[str] = field(default_factory=list)
    recipients: list[str] = field(default_factory=list)
    enabled: bool = True
    last_run: float = 0.0
    created_at: float = field(default_factory=time.time)
    schedule_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


class ReportScheduler:
    def __init__(self, kpi_engine=None, persistence=None, vlm_client=None):
        self._kpi_engine = kpi_engine
        self._persistence = persistence
        self._vlm_client = vlm_client
        self.schedules: dict[str, ReportSchedule] = {}

    def _get_client(self):
        return self._vlm_client

    def add_schedule(self, schedule: ReportSchedule) -> None:
        self.schedules[schedule.schedule_id] = schedule

    def remove_schedule(self, schedule_id: str) -> None:
        self.schedules.pop(schedule_id, None)

    def list_schedules(self) -> list[ReportSchedule]:
        return list(self.schedules.values())

    def check_due(self) -> list[ReportSchedule]:
        now = datetime.now(timezone.utc)
        hour, weekday, day = now.hour, now.weekday(), now.day
        due: list[ReportSchedule] = []

        for s in self.schedules.values():
            if not s.enabled:
                continue
            # Skip if already run this hour
            if time.time() - s.last_run < 3600:
                continue

            match s.schedule_type:
                case "daily_8am" if hour == 8:
                    due.append(s)
                case "daily_6pm" if hour == 18:
                    due.append(s)
                case "weekly_monday_9am" if weekday == 0 and hour == 9:
                    due.append(s)
                case "monthly_1st" if day == 1 and hour == 9:
                    due.append(s)

        return due

    def generate_report(self, schedule: ReportSchedule) -> dict:
        schedule.last_run = time.time()
        metrics = {}

        if self._kpi_engine:
            for cid in schedule.camera_ids:
                try:
                    metrics[cid] = self._kpi_engine.compute_daily_kpis(cid)
                except Exception:
                    logger.warning("KPI computation failed for %s", cid)

        narrative = self._generate_narrative(schedule, metrics)
        html_body = self._format_html(schedule, metrics, narrative)

        return {
            "title": f"{schedule.report_type} — {schedule.name}",
            "html_body": html_body,
            "narrative": narrative,
            "metrics": metrics,
        }

    def _generate_narrative(self, schedule: ReportSchedule, metrics: dict) -> str:
        client = self._get_client()
        if client:
            try:
                prompt = (f"Write a brief report narrative for {schedule.report_type}. "
                          f"Cameras: {schedule.camera_ids}. Metrics: {json.dumps(metrics, default=str)}")
                resp = client.generate(prompt)
                return resp if isinstance(resp, str) else resp.text
            except Exception:
                logger.warning("VLM narrative generation failed, using fallback")

        total_traffic = sum(m.get("foot_traffic", 0) for m in metrics.values())
        total_incidents = sum(m.get("incident_count", 0) for m in metrics.values())
        return (f"Report: {schedule.report_type}. "
                f"Cameras monitored: {len(schedule.camera_ids)}. "
                f"Total foot traffic: {total_traffic}. Incidents: {total_incidents}.")

    def _format_html(self, schedule: ReportSchedule, metrics: dict, narrative: str) -> str:
        rows = ""
        for cid, m in metrics.items():
            rows += (f"<tr><td>{cid}</td><td>{m.get('foot_traffic', 0)}</td>"
                     f"<td>{m.get('incident_count', 0)}</td></tr>")
        return (f"<html><body><h2>{schedule.report_type} — {schedule.name}</h2>"
                f"<p>{narrative}</p>"
                f"<table><tr><th>Camera</th><th>Traffic</th><th>Incidents</th></tr>"
                f"{rows}</table></body></html>")

"""Shift Handover Intelligence."""

import logging
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class ShiftBriefing:
    shift_id: str
    generated_at: float
    period_start: float
    period_end: float
    summary: str
    incidents: list[dict]
    unresolved: list[dict]
    entity_highlights: list[dict]
    camera_status: list[dict]
    recommendations: list[str]


class ShiftHandoverAgent:
    def __init__(self, persistence=None, kg=None, vlm_client=None, action_engine=None):
        self._persistence = persistence
        self._kg = kg
        self._vlm_client = vlm_client
        self._action_engine = action_engine

    def generate_briefing(self, hours: int = 8) -> ShiftBriefing:
        now = time.time()
        cutoff = now - hours * 3600

        alerts = (self._persistence.get_alerts(limit=500) or []) if self._persistence else []
        recent = [a for a in alerts if a.get('timestamp', 0) >= cutoff]

        by_severity: dict[str, list] = {}
        for a in recent:
            by_severity.setdefault(a.get('severity', 'unknown'), []).append(a)

        unresolved = [a for a in recent if not a.get('acknowledged')]

        entity_counter: Counter = Counter()
        for a in recent:
            for e in a.get('entities', []):
                entity_counter[e] += 1
        entity_highlights = [{'entity': e, 'count': c} for e, c in entity_counter.items() if c > 1]

        cam_counter: Counter = Counter(a.get('camera_id') for a in recent if a.get('camera_id'))
        all_cams = {a.get('camera_id') for a in alerts if a.get('camera_id')}
        camera_status = [{'camera_id': c, 'alert_count': cam_counter.get(c, 0),
                          'status': 'active' if cam_counter.get(c, 0) > 0 else 'possibly_offline'}
                         for c in all_cams]

        summary = self._generate_summary(recent, by_severity, unresolved)
        recommendations = self._build_recommendations(by_severity, unresolved, camera_status)

        return ShiftBriefing(
            shift_id=str(uuid.uuid4())[:10],
            generated_at=now,
            period_start=cutoff,
            period_end=now,
            summary=summary,
            incidents=[{'severity': s, 'count': len(v)} for s, v in by_severity.items()],
            unresolved=unresolved,
            entity_highlights=entity_highlights,
            camera_status=camera_status,
            recommendations=recommendations,
        )

    def _generate_summary(self, recent: list, by_severity: dict, unresolved: list) -> str:
        if self._vlm_client:
            try:
                client = self._vlm_client._get_client()
                if client != 'stub':
                    prompt = (f"Generate shift briefing: {len(recent)} alerts, "
                              f"{len(unresolved)} unresolved, "
                              f"severity breakdown: {({s: len(v) for s, v in by_severity.items()})}.")
                    return client.generate(prompt)
            except Exception:
                logger.warning("VLM summary generation failed, using fallback")
        return (f"Shift period: {len(recent)} total alerts, {len(unresolved)} unresolved. "
                f"Severity breakdown: {', '.join(f'{s}={len(v)}' for s, v in by_severity.items())}.")

    def _build_recommendations(self, by_severity: dict, unresolved: list,
                               camera_status: list) -> list[str]:
        recs: list[str] = []
        if len(unresolved) > 10:
            recs.append(f"High backlog: {len(unresolved)} unresolved alerts require attention.")
        if 'critical' in by_severity:
            recs.append(f"{len(by_severity['critical'])} critical alerts detected — prioritize review.")
        offline = [c for c in camera_status if c['status'] == 'possibly_offline']
        if offline:
            recs.append(f"{len(offline)} cameras possibly offline — verify connectivity.")
        return recs

    def export_briefing(self, briefing: ShiftBriefing) -> dict:
        return asdict(briefing)

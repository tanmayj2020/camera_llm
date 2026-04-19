"""Autonomous Investigation Agent — when an incident triggers, auto-traces
the entity across cameras, finds accomplices, builds evidence chain,
generates a complete investigation report.

Think "AI detective" — not just alerting, but actually investigating.
"""

import json
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EvidenceItem:
    timestamp: float
    camera_id: str
    event_type: str
    description: str
    entities: list[str] = field(default_factory=list)
    keyframe_ref: str = ""
    confidence: float = 0.0


@dataclass
class AssociateProfile:
    entity_id: str
    relationship: str  # "co-located", "interacted", "same_route"
    co_occurrence_count: int = 0
    cameras_shared: list[str] = field(default_factory=list)
    risk_score: float = 0.0


@dataclass
class InvestigationReport:
    investigation_id: str
    subject_entity: str
    status: str = "complete"
    started_at: float = field(default_factory=time.time)
    # Subject profile
    subject_profile: dict = field(default_factory=dict)
    # Timeline
    timeline: list[EvidenceItem] = field(default_factory=list)
    # Associates
    associates: list[AssociateProfile] = field(default_factory=list)
    # Cross-camera route
    camera_route: list[dict] = field(default_factory=list)
    # Anomalies specific to this entity
    anomalies: list[dict] = field(default_factory=list)
    # LLM-generated narrative
    narrative: str = ""
    risk_assessment: str = ""
    recommended_actions: list[str] = field(default_factory=list)


class AutoInvestigator:
    """Autonomous investigation agent.

    Given an entity ID, autonomously:
    1. Pulls full entity profile from KG
    2. Traces movement across all cameras (backward + forward)
    3. Identifies associates (co-located entities, interactions)
    4. Collects all related anomalies/incidents
    5. Generates a complete investigation report with LLM narrative
    """

    def __init__(self, kg=None, profiler=None, story_builder=None, spatial=None):
        self._kg = kg
        self._profiler = profiler
        self._story_builder = story_builder
        self._spatial = spatial
        self._vlm_client = None

    def set_vlm_client(self, client):
        self._vlm_client = client

    def investigate(self, entity_id: str, since_hours: int = 48) -> InvestigationReport:
        """Run full autonomous investigation on an entity."""
        t0 = time.time()
        report = InvestigationReport(
            investigation_id=f"inv_{entity_id}_{int(t0)}",
            subject_entity=entity_id,
        )

        # Step 1: Subject profile
        report.subject_profile = self._get_profile(entity_id, since_hours)

        # Step 2: Full timeline from KG
        report.timeline = self._build_timeline(entity_id, since_hours)

        # Step 3: Camera route reconstruction
        report.camera_route = self._reconstruct_route(report.timeline)

        # Step 4: Find associates
        report.associates = self._find_associates(entity_id, since_hours)

        # Step 5: Collect related anomalies
        report.anomalies = self._collect_anomalies(entity_id, since_hours)

        # Step 6: Generate narrative + risk assessment
        report.narrative = self._generate_narrative(report)
        report.risk_assessment = self._assess_risk(report)
        report.recommended_actions = self._recommend_actions(report)

        elapsed = time.time() - t0
        logger.info("Investigation %s completed in %.1fs: %d timeline events, %d associates",
                    report.investigation_id, elapsed, len(report.timeline), len(report.associates))
        return report

    def _get_profile(self, entity_id: str, since_hours: int) -> dict:
        if self._profiler:
            profile = self._profiler.get_profile(entity_id, since_hours)
            return self._profiler.to_dict(profile)
        return {"entity_id": entity_id}

    def _build_timeline(self, entity_id: str, since_hours: int) -> list[EvidenceItem]:
        if not self._kg:
            return []
        try:
            history = self._kg.get_entity_history(entity_id, since_hours)
        except Exception as e:
            logger.error("KG timeline query failed: %s", e)
            return []

        items = []
        for record in history:
            props = record.get("props", {})
            target = record.get("target_props", {})
            labels = record.get("labels", [])
            rel = record.get("rel", "")

            ts = self._parse_ts(props.get("last_seen", ""))
            camera_id = target.get("camera_id", "") if "Camera" in labels else ""

            items.append(EvidenceItem(
                timestamp=ts or 0.0,
                camera_id=camera_id,
                event_type=rel,
                description=f"{rel} → {', '.join(labels)}",
                entities=[entity_id],
                confidence=0.8,
            ))

        items.sort(key=lambda e: e.timestamp)
        return items

    def _reconstruct_route(self, timeline: list[EvidenceItem]) -> list[dict]:
        """Extract camera-to-camera route with transit times."""
        route = []
        seen_cameras = []
        for item in timeline:
            if item.camera_id and (not seen_cameras or seen_cameras[-1]["camera_id"] != item.camera_id):
                entry = {"camera_id": item.camera_id, "arrived_at": item.timestamp}
                if seen_cameras:
                    prev = seen_cameras[-1]
                    entry["transit_from"] = prev["camera_id"]
                    entry["transit_time_s"] = round(item.timestamp - prev["arrived_at"], 1)
                seen_cameras.append(entry)
                route.append(entry)
        return route

    def _find_associates(self, entity_id: str, since_hours: int) -> list[AssociateProfile]:
        if not self._kg:
            return []
        try:
            with self._kg._driver.session() as s:
                result = s.run("""
                    MATCH (subject {track_id: $tid})-[r:INTERACTED_WITH]-(other)
                    WHERE r.last_seen >= datetime() - duration({hours: $hours})
                    RETURN other.track_id AS other_id, r.type AS rel_type,
                           count(r) AS co_count
                    ORDER BY co_count DESC LIMIT 20
                """, tid=entity_id, hours=since_hours)
                associates = []
                for rec in result:
                    other_id = rec["other_id"]
                    if not other_id:
                        continue
                    # Get shared cameras
                    shared = self._get_shared_cameras(entity_id, other_id)
                    risk = 0.0
                    if self._profiler:
                        p = self._profiler.get_profile(other_id, since_hours)
                        risk = p.risk_score
                    associates.append(AssociateProfile(
                        entity_id=other_id,
                        relationship=rec.get("rel_type", "co-located"),
                        co_occurrence_count=rec.get("co_count", 1),
                        cameras_shared=shared,
                        risk_score=risk,
                    ))
                return associates
        except Exception as e:
            logger.error("Associate query failed: %s", e)
            return []

    def _get_shared_cameras(self, id_a: str, id_b: str) -> list[str]:
        if not self._kg:
            return []
        try:
            with self._kg._driver.session() as s:
                result = s.run("""
                    MATCH (a {track_id: $a})-[:DETECTED_IN]->(c:Camera)<-[:DETECTED_IN]-(b {track_id: $b})
                    RETURN DISTINCT c.camera_id AS cam
                """, a=id_a, b=id_b)
                return [r["cam"] for r in result if r["cam"]]
        except Exception:
            return []

    def _collect_anomalies(self, entity_id: str, since_hours: int) -> list[dict]:
        if not self._kg:
            return []
        try:
            with self._kg._driver.session() as s:
                result = s.run("""
                    MATCH (e:Event)-[:OCCURRED_AT]->(c:Camera)
                    WHERE e.timestamp >= datetime() - duration({hours: $hours})
                      AND e.data CONTAINS $tid
                    RETURN e.event_id AS eid, e.event_type AS etype,
                           e.timestamp AS ts, c.camera_id AS cam, e.data AS data
                    ORDER BY e.timestamp DESC LIMIT 20
                """, tid=entity_id, hours=since_hours)
                return [dict(r) for r in result]
        except Exception as e:
            logger.debug("Anomaly collection failed: %s", e)
            return []

    def _generate_narrative(self, report: InvestigationReport) -> str:
        if self._vlm_client and hasattr(self._vlm_client, '_get_client'):
            client = self._vlm_client._get_client()
            if client != "stub":
                try:
                    prompt = (f"Write a professional security investigation report for entity "
                              f"'{report.subject_entity}'.\n"
                              f"Profile: {json.dumps(report.subject_profile, default=str)}\n"
                              f"Timeline: {len(report.timeline)} events across "
                              f"{len(report.camera_route)} cameras\n"
                              f"Camera route: {json.dumps(report.camera_route[:10], default=str)}\n"
                              f"Associates: {len(report.associates)} identified\n"
                              f"Anomalies: {len(report.anomalies)} related incidents\n\n"
                              f"Write 3-4 paragraphs: movement summary, behavioral analysis, "
                              f"associate connections, risk assessment.")
                    resp = client.generate_content(prompt)
                    return resp.text.strip()
                except Exception:
                    pass

        # Fallback
        parts = [f"Investigation Report: Entity {report.subject_entity}",
                 f"Tracked across {len(report.camera_route)} cameras with {len(report.timeline)} events.",
                 f"{len(report.associates)} associates identified, {len(report.anomalies)} related anomalies."]
        if report.camera_route:
            cams = " → ".join(r["camera_id"] for r in report.camera_route)
            parts.append(f"Route: {cams}")
        return "\n".join(parts)

    def _assess_risk(self, report: InvestigationReport) -> str:
        score = report.subject_profile.get("risk_score", 0)
        anomaly_factor = min(0.3, len(report.anomalies) * 0.1)
        associate_risk = max((a.risk_score for a in report.associates), default=0) * 0.2
        total = min(1.0, score + anomaly_factor + associate_risk)

        if total > 0.7:
            return f"HIGH RISK ({total:.0%}) — Multiple anomalies and high-risk associates detected."
        if total > 0.4:
            return f"MEDIUM RISK ({total:.0%}) — Some concerning patterns identified."
        return f"LOW RISK ({total:.0%}) — No significant threats detected."

    def _recommend_actions(self, report: InvestigationReport) -> list[str]:
        actions = []
        if len(report.anomalies) > 3:
            actions.append("Flag entity for enhanced monitoring across all cameras.")
        if any(a.risk_score > 0.5 for a in report.associates):
            actions.append("Investigate high-risk associates for potential coordinated activity.")
        if len(report.camera_route) > 5:
            actions.append("Review unusual multi-camera route for potential reconnaissance behavior.")
        if not actions:
            actions.append("No immediate action required. Continue standard monitoring.")
        return actions

    def export_report(self, report: InvestigationReport) -> dict:
        return {
            "investigation_id": report.investigation_id,
            "subject_entity": report.subject_entity,
            "status": report.status,
            "started_at": report.started_at,
            "subject_profile": report.subject_profile,
            "timeline": [{"timestamp": e.timestamp, "camera": e.camera_id,
                          "type": e.event_type, "description": e.description}
                         for e in report.timeline],
            "camera_route": report.camera_route,
            "associates": [{"entity_id": a.entity_id, "relationship": a.relationship,
                            "co_occurrences": a.co_occurrence_count,
                            "shared_cameras": a.cameras_shared, "risk_score": a.risk_score}
                           for a in report.associates],
            "anomalies": report.anomalies,
            "narrative": report.narrative,
            "risk_assessment": report.risk_assessment,
            "recommended_actions": report.recommended_actions,
        }

    @staticmethod
    def _parse_ts(ts_str: str) -> float | None:
        if not ts_str:
            return None
        try:
            from datetime import datetime, timezone
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
        except (ValueError, TypeError):
            return None

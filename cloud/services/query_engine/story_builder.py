"""Incident story reconstruction — auto-generates multi-camera narrative timelines.

Builds chronological story from knowledge graph with linked keyframes.
LLM generates prose narrative. Exportable as incident report.
"""

import json
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StoryEvent:
    timestamp: float
    camera_id: str
    event_type: str
    description: str
    entities: list[str] = field(default_factory=list)
    keyframe_ref: str = ""
    zone: str = ""


@dataclass
class IncidentStory:
    entity_id: str
    title: str
    timeline: list[StoryEvent] = field(default_factory=list)
    narrative: str = ""
    cameras_involved: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    generated_at: float = field(default_factory=time.time)


class StoryBuilder:
    """Reconstructs incident narratives from knowledge graph data.

    Given an entity, queries KG for all related events across cameras,
    builds a chronological timeline, and generates a prose narrative via LLM.
    """

    def __init__(self, kg=None):
        self._kg = kg
        self._vlm_client = None

    def set_vlm_client(self, client):
        self._vlm_client = client

    def build_story(self, entity_id: str, since_hours: int = 24) -> IncidentStory:
        """Build complete incident story for an entity."""
        # Gather events from KG
        events = self._gather_events(entity_id, since_hours)
        if not events:
            return IncidentStory(entity_id=entity_id, title=f"No events for {entity_id}")

        events.sort(key=lambda e: e.timestamp)
        cameras = list(set(e.camera_id for e in events))
        duration = events[-1].timestamp - events[0].timestamp if len(events) > 1 else 0

        story = IncidentStory(
            entity_id=entity_id,
            title=f"Activity report: {entity_id}",
            timeline=events,
            cameras_involved=cameras,
            duration_s=duration,
        )

        # Generate narrative
        story.narrative = self._generate_narrative(story)
        return story

    def _gather_events(self, entity_id: str, since_hours: int) -> list[StoryEvent]:
        """Query KG for all events related to an entity."""
        if not self._kg:
            return []

        try:
            history = self._kg.get_entity_history(entity_id, since_hours)
        except Exception as e:
            logger.error("KG query failed for %s: %s", entity_id, e)
            return []

        events = []
        for record in history:
            rel = record.get("rel", "")
            props = record.get("props", {})
            target = record.get("target_props", {})
            labels = record.get("labels", [])

            camera_id = target.get("camera_id", "") if "Camera" in labels else ""
            zone_id = target.get("zone_id", "") if "Zone" in labels else ""

            events.append(StoryEvent(
                timestamp=self._parse_timestamp(props.get("last_seen", "")),
                camera_id=camera_id,
                event_type=rel,
                description=f"{rel} → {', '.join(labels)}",
                entities=[entity_id],
                zone=zone_id,
            ))
        return events

    def _generate_narrative(self, story: IncidentStory) -> str:
        """Generate prose narrative from timeline."""
        if self._vlm_client and hasattr(self._vlm_client, '_get_client'):
            client = self._vlm_client._get_client()
            if client != "stub":
                try:
                    timeline_text = "\n".join(
                        f"- {self._format_time(e.timestamp)}: [{e.camera_id}] {e.description}"
                        for e in story.timeline[:30]
                    )
                    prompt = (f"Write a concise incident report narrative for entity '{story.entity_id}'.\n"
                              f"Cameras: {story.cameras_involved}\n"
                              f"Duration: {story.duration_s / 60:.1f} minutes\n"
                              f"Timeline:\n{timeline_text}\n\n"
                              f"Write 2-3 paragraphs as a professional security report.")
                    resp = client.generate_content(prompt)
                    return resp.text.strip()
                except Exception as e:
                    logger.debug("Narrative generation failed: %s", e)

        # Fallback: template narrative
        lines = [f"Entity {story.entity_id} was tracked across {len(story.cameras_involved)} camera(s) "
                 f"over {story.duration_s / 60:.1f} minutes."]
        for e in story.timeline[:10]:
            lines.append(f"  • {self._format_time(e.timestamp)}: {e.description} ({e.camera_id})")
        if len(story.timeline) > 10:
            lines.append(f"  ... and {len(story.timeline) - 10} more events.")
        return "\n".join(lines)

    def export_report(self, story: IncidentStory) -> dict:
        """Export story as structured report dict."""
        return {
            "entity_id": story.entity_id,
            "title": story.title,
            "generated_at": story.generated_at,
            "cameras": story.cameras_involved,
            "duration_minutes": round(story.duration_s / 60, 1),
            "event_count": len(story.timeline),
            "narrative": story.narrative,
            "timeline": [
                {"timestamp": e.timestamp, "camera": e.camera_id,
                 "type": e.event_type, "description": e.description}
                for e in story.timeline
            ],
        }

    @staticmethod
    def _parse_timestamp(ts_str: str) -> float:
        if not ts_str:
            return 0.0
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            return dt.timestamp()
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _format_time(ts: float) -> str:
        if ts <= 0:
            return "unknown"
        from datetime import datetime, timezone
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")

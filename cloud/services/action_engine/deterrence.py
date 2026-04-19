"""Intelligent voice deterrence — context-aware escalating warnings.

4 escalation levels, VLM-generated natural language, per-entity session tracking.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum

logger = logging.getLogger(__name__)


class EscalationLevel(IntEnum):
    NOTICE = 1       # Polite awareness ("Attention: this area is monitored")
    WARNING = 2      # Direct warning ("You near the loading dock — this area is restricted")
    COMMAND = 3      # Firm command ("Leave the restricted area immediately")
    ALARM = 4        # Alarm + authority ("Security has been dispatched. Remain where you are.")


@dataclass
class DeterrenceSession:
    entity_id: str
    camera_id: str
    zone_name: str
    level: EscalationLevel = EscalationLevel.NOTICE
    started_at: float = field(default_factory=time.time)
    last_escalated: float = field(default_factory=time.time)
    messages_sent: list[str] = field(default_factory=list)
    resolved: bool = False


class DeterrenceEngine:
    """Context-aware escalating voice deterrence system.

    Each tracked entity gets its own session. Escalation happens if the entity
    remains in the zone after a cooldown period.
    """

    ESCALATION_COOLDOWN_S = 15.0  # seconds between escalation levels
    SESSION_TIMEOUT_S = 300.0     # auto-close session after 5 min

    TEMPLATES = {
        EscalationLevel.NOTICE: "Attention: you are being recorded in the {zone} area. This area is monitored 24/7.",
        EscalationLevel.WARNING: "Warning: you near {zone} — this area is restricted. Please leave immediately.",
        EscalationLevel.COMMAND: "Final warning: leave {zone} now. Security has been notified.",
        EscalationLevel.ALARM: "Security dispatched to {zone}. Remain where you are. Authorities have been contacted.",
    }

    def __init__(self):
        self._sessions: dict[str, DeterrenceSession] = {}
        self._vlm_client = None
        self._pa_adapter = None  # pluggable PA system adapter

    def set_vlm_client(self, client):
        self._vlm_client = client

    def set_pa_adapter(self, adapter):
        self._pa_adapter = adapter

    def get_or_create_session(self, entity_id: str, camera_id: str,
                              zone_name: str) -> DeterrenceSession:
        key = f"{entity_id}_{camera_id}"
        if key not in self._sessions or self._sessions[key].resolved:
            self._sessions[key] = DeterrenceSession(
                entity_id=entity_id, camera_id=camera_id, zone_name=zone_name,
            )
        return self._sessions[key]

    def process(self, entity_id: str, camera_id: str, zone_name: str,
                scene_description: str = "") -> str | None:
        """Evaluate deterrence for an entity in a zone. Returns message if one should be played."""
        session = self.get_or_create_session(entity_id, camera_id, zone_name)
        now = time.time()

        # Auto-close stale sessions
        if now - session.started_at > self.SESSION_TIMEOUT_S:
            session.resolved = True
            return None

        # Check cooldown
        if now - session.last_escalated < self.ESCALATION_COOLDOWN_S:
            return None

        # Escalate
        if session.level < EscalationLevel.ALARM:
            session.level = EscalationLevel(session.level + 1)
        session.last_escalated = now

        message = self._generate_message(session, scene_description)
        session.messages_sent.append(message)
        self._play_message(message, camera_id)

        logger.info("Deterrence L%d entity=%s zone=%s: %s",
                    session.level, entity_id, zone_name, message[:80])
        return message

    def resolve_session(self, entity_id: str, camera_id: str):
        key = f"{entity_id}_{camera_id}"
        if key in self._sessions:
            self._sessions[key].resolved = True

    def get_active_sessions(self) -> list[DeterrenceSession]:
        now = time.time()
        return [s for s in self._sessions.values()
                if not s.resolved and now - s.started_at < self.SESSION_TIMEOUT_S]

    def _generate_message(self, session: DeterrenceSession, scene_desc: str) -> str:
        """Generate context-aware message, optionally using VLM."""
        # Try VLM for natural language
        if self._vlm_client and scene_desc and hasattr(self._vlm_client, '_get_client'):
            client = self._vlm_client._get_client()
            if client != "stub":
                try:
                    prompt = (f"Generate a short deterrence message (level {session.level}/4) "
                              f"for someone in '{session.zone_name}'. Scene: {scene_desc}. "
                              f"Be firm but professional. Max 2 sentences.")
                    resp = client.generate_content(prompt)
                    return resp.text.strip()
                except Exception as e:
                    logger.debug("VLM deterrence generation failed: %s", e)

        # Fallback to template
        return self.TEMPLATES[session.level].format(zone=session.zone_name)

    def _play_message(self, message: str, camera_id: str):
        """Send message to PA system."""
        if self._pa_adapter:
            try:
                self._pa_adapter.play(camera_id, message)
            except Exception as e:
                logger.error("PA playback failed: %s", e)
        else:
            logger.info("[PA-STUB] Camera %s: %s", camera_id, message[:80])

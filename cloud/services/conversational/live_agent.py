"""Live Conversational Camera Agent — streaming-aware, session-persistent,
multi-camera conversational interface.

"What's happening at cam-2 right now?" → queries live spatial + recent KG
"Is that the same person from earlier?" → cross-references entity profiles
"""

import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    role: str  # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class Session:
    session_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    # Context anchors — entities/cameras the user has been discussing
    focused_cameras: list[str] = field(default_factory=list)
    mentioned_entities: list[str] = field(default_factory=list)


@dataclass
class ConversationResponse:
    answer: str
    evidence: list[dict] = field(default_factory=list)
    live_snapshot: dict = field(default_factory=dict)
    confidence: float = 0.0
    session_id: str = ""


class LiveConversationalAgent:
    """Conversational agent that maintains session context and can query
    live camera state, KG history, spatial memory, and entity profiles
    to answer questions about what's happening now or in the past."""

    MAX_CONTEXT_TURNS = 20
    SESSION_TTL_S = 3600  # 1 hour

    def __init__(self, kg=None, spatial=None, query_engine=None, profiler=None):
        self._kg = kg
        self._spatial = spatial
        self._query_engine = query_engine
        self._profiler = profiler
        self._vlm_client = None
        self._sessions: dict[str, Session] = {}

    def set_vlm_client(self, client):
        self._vlm_client = client

    def get_or_create_session(self, session_id: str | None = None) -> Session:
        if session_id and session_id in self._sessions:
            s = self._sessions[session_id]
            s.last_active = time.time()
            return s
        sid = session_id or str(uuid.uuid4())[:12]
        s = Session(session_id=sid)
        self._sessions[sid] = s
        return s

    def converse(self, message: str, session_id: str | None = None,
                 camera_id: str | None = None) -> ConversationResponse:
        session = self.get_or_create_session(session_id)
        session.turns.append(ConversationTurn(role="user", content=message))

        # Track camera focus
        if camera_id and camera_id not in session.focused_cameras:
            session.focused_cameras.append(camera_id)

        # Detect intent
        intent = self._classify_intent(message)

        # Build live context
        live_ctx = self._build_live_context(session, camera_id)

        # Generate answer based on intent
        if intent == "live_status":
            answer, evidence = self._handle_live_status(message, session, camera_id, live_ctx)
        elif intent == "entity_query":
            answer, evidence = self._handle_entity_query(message, session, live_ctx)
        elif intent == "historical":
            answer, evidence = self._handle_historical(message, session, camera_id)
        else:
            answer, evidence = self._handle_general(message, session, camera_id, live_ctx)

        session.turns.append(ConversationTurn(role="assistant", content=answer))
        # Trim context window
        if len(session.turns) > self.MAX_CONTEXT_TURNS * 2:
            session.turns = session.turns[-self.MAX_CONTEXT_TURNS * 2:]

        return ConversationResponse(
            answer=answer, evidence=evidence, live_snapshot=live_ctx,
            confidence=0.7 if evidence else 0.5, session_id=session.session_id,
        )

    def _classify_intent(self, message: str) -> str:
        m = message.lower()
        if any(w in m for w in ["right now", "happening", "currently", "live", "at the moment", "see now"]):
            return "live_status"
        if any(w in m for w in ["same person", "who is", "profile", "entity", "track", "identify"]):
            return "entity_query"
        if any(w in m for w in ["earlier", "yesterday", "last hour", "history", "when did", "how many times"]):
            return "historical"
        return "general"

    def _build_live_context(self, session: Session, camera_id: str | None) -> dict:
        ctx: dict = {"timestamp": time.time(), "entities": [], "zones": [], "recent_events": []}
        if not self._spatial:
            return ctx

        # Gather live entities
        for eid, ent in self._spatial._entities.items():
            if time.time() - ent.last_seen < 30:  # active in last 30s
                ctx["entities"].append({
                    "track_id": ent.track_id, "class_name": ent.class_name,
                    "position": ent.position.tolist(), "velocity": ent.velocity.tolist(),
                })

        # Zones
        ctx["zones"] = [{"zone_id": z.zone_id, "name": z.name}
                        for z in self._spatial._zones.values()]

        # Recent KG events
        if self._kg and camera_id:
            try:
                ctx["recent_events"] = self._kg.get_recent_events(camera_id, limit=10)
            except Exception:
                pass

        ctx["entity_count"] = len(ctx["entities"])
        ctx["zone_count"] = len(ctx["zones"])
        return ctx

    def _handle_live_status(self, message: str, session: Session,
                            camera_id: str | None, live_ctx: dict) -> tuple[str, list]:
        """Answer questions about what's happening right now."""
        entities = live_ctx.get("entities", [])
        recent = live_ctx.get("recent_events", [])

        # Try VLM synthesis
        if self._vlm_client and hasattr(self._vlm_client, '_get_client'):
            client = self._vlm_client._get_client()
            if client != "stub":
                try:
                    history = [{"role": t.role, "content": t.content}
                               for t in session.turns[-6:]]
                    prompt = (f"You are a live CCTV monitoring assistant. Answer based on live data.\n"
                              f"User: {message}\n"
                              f"Live entities: {json.dumps(entities[:20], default=str)}\n"
                              f"Recent events: {json.dumps(recent[:5], default=str)}\n"
                              f"Conversation: {json.dumps(history, default=str)}\n"
                              f"Camera: {camera_id or 'all'}\n"
                              f"Give a concise, specific answer referencing entity IDs and positions.")
                    resp = client.generate_content(prompt)
                    return resp.text.strip(), recent[:5]
                except Exception as e:
                    logger.debug("VLM live status failed: %s", e)

        # Fallback
        person_count = sum(1 for e in entities if e["class_name"] == "person")
        cam_label = f"camera {camera_id}" if camera_id else "all cameras"
        parts = [f"Currently monitoring {cam_label}: {len(entities)} active entities ({person_count} people)."]
        for e in entities[:5]:
            parts.append(f"  • {e['class_name']} (ID {e['track_id']}) at position "
                         f"({e['position'][0]:.1f}, {e['position'][2]:.1f})m")
        if recent:
            parts.append(f"\nMost recent event: {recent[0].get('event_type', '?')} "
                         f"at {recent[0].get('timestamp', '?')}")
        return "\n".join(parts), recent[:5]

    def _handle_entity_query(self, message: str, session: Session,
                             live_ctx: dict) -> tuple[str, list]:
        """Answer questions about specific entities — identity, profile, re-ID."""
        # Extract entity references from message or session context
        entity_id = self._extract_entity_ref(message, session, live_ctx)
        if not entity_id:
            return "I couldn't identify which entity you're referring to. Could you specify a track ID?", []

        session.mentioned_entities.append(entity_id)

        # Get profile
        if self._profiler:
            profile = self._profiler.get_profile(entity_id)
            profile_dict = self._profiler.to_dict(profile)
            summary = profile.behavior_summary or f"Entity {entity_id}: {profile.total_detections} detections"
            evidence = [profile_dict]

            # VLM enrichment
            if self._vlm_client and hasattr(self._vlm_client, '_get_client'):
                client = self._vlm_client._get_client()
                if client != "stub":
                    try:
                        prompt = (f"User asks: {message}\n"
                                  f"Entity profile: {json.dumps(profile_dict, default=str)}\n"
                                  f"Give a helpful answer about this entity.")
                        resp = client.generate_content(prompt)
                        return resp.text.strip(), evidence
                    except Exception:
                        pass

            return summary, evidence

        return f"Entity {entity_id} found in current session context.", []

    def _handle_historical(self, message: str, session: Session,
                           camera_id: str | None) -> tuple[str, list]:
        """Delegate historical queries to the query engine."""
        if self._query_engine:
            result = self._query_engine.query(message, camera_id)
            return result.answer, result.evidence
        return "Historical query engine not available.", []

    def _handle_general(self, message: str, session: Session,
                        camera_id: str | None, live_ctx: dict) -> tuple[str, list]:
        """General questions — combine live context with query engine."""
        # Try VLM with full context
        if self._vlm_client and hasattr(self._vlm_client, '_get_client'):
            client = self._vlm_client._get_client()
            if client != "stub":
                try:
                    history = [{"role": t.role, "content": t.content}
                               for t in session.turns[-8:]]
                    prompt = (f"You are VisionBrain, an intelligent CCTV analytics assistant.\n"
                              f"Conversation: {json.dumps(history, default=str)}\n"
                              f"Live context: {len(live_ctx.get('entities', []))} entities, "
                              f"{len(live_ctx.get('zones', []))} zones\n"
                              f"Camera: {camera_id or 'all'}\n"
                              f"Answer the user's question helpfully and concisely.")
                    resp = client.generate_content(prompt)
                    return resp.text.strip(), []
                except Exception:
                    pass

        # Fallback to query engine
        if self._query_engine:
            result = self._query_engine.query(message, camera_id)
            return result.answer, result.evidence
        return "I can help you monitor cameras, track entities, and investigate incidents. What would you like to know?", []

    def _extract_entity_ref(self, message: str, session: Session, live_ctx: dict) -> str | None:
        """Extract entity ID from message or infer from context."""
        import re
        # Direct ID mention
        match = re.search(r'(?:track|id|entity)\s*[#:]?\s*(\S+)', message.lower())
        if match:
            return match.group(1)

        # "that person" / "same person" → last mentioned entity
        if any(w in message.lower() for w in ["that person", "same person", "them", "they"]):
            if session.mentioned_entities:
                return session.mentioned_entities[-1]

        # If only one person visible, assume they mean that one
        people = [e for e in live_ctx.get("entities", []) if e["class_name"] == "person"]
        if len(people) == 1:
            return str(people[0]["track_id"])

        return None

    def cleanup_stale_sessions(self):
        now = time.time()
        stale = [sid for sid, s in self._sessions.items()
                 if now - s.last_active > self.SESSION_TTL_S]
        for sid in stale:
            del self._sessions[sid]

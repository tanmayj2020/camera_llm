"""Privacy-First Forensic Search — search across footage/KG without exposing PII.

Tiered access levels, full audit trail. Results are anonymized by default.
"Unlock" requires elevated access and creates an audit log entry.
"""

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AccessLevel(str, Enum):
    ANONYMOUS = "anonymous"   # sees descriptions + bounding boxes, no faces
    OPERATOR = "operator"     # sees blurred faces + track IDs
    SUPERVISOR = "supervisor" # can unlock full footage with audit trail
    ADMIN = "admin"           # full access


@dataclass
class SearchResult:
    result_id: str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    timestamp: float = 0.0
    camera_id: str = ""
    description: str = ""  # anonymized description
    entity_class: str = ""
    bbox: list[float] = field(default_factory=list)
    # PII fields — only populated after unlock
    entity_id: str = ""  # anonymized hash by default
    keyframe_ref: str = ""  # empty until unlocked
    raw_data: dict = field(default_factory=dict)
    is_unlocked: bool = False


@dataclass
class AuditEntry:
    audit_id: str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    timestamp: float = field(default_factory=time.time)
    user_id: str = ""
    access_level: str = ""
    action: str = ""  # "search", "unlock", "export"
    query: str = ""
    result_ids: list[str] = field(default_factory=list)
    justification: str = ""


@dataclass
class ForensicSearchResponse:
    query: str
    results: list[SearchResult] = field(default_factory=list)
    total_matches: int = 0
    access_level: str = "anonymous"
    audit_id: str = ""


class ForensicSearchEngine:
    """Privacy-first forensic search across all camera footage and KG data.

    All results are anonymized by default:
    - Entity IDs are hashed
    - Descriptions strip identifying details
    - Keyframes are withheld
    - Unlocking requires supervisor+ access and creates audit trail
    """

    def __init__(self, kg=None):
        self._kg = kg
        self._vlm_client = None
        self._audit_log: list[AuditEntry] = []
        self._unlocked_results: dict[str, SearchResult] = {}  # result_id → full result

    def set_vlm_client(self, client):
        self._vlm_client = client

    def search(self, query: str, user_id: str = "anonymous",
               access_level: str = "anonymous",
               camera_id: str | None = None,
               time_range_hours: int = 24,
               limit: int = 20) -> ForensicSearchResponse:
        """Search across KG with privacy-preserving anonymization."""
        # Audit the search
        audit = AuditEntry(
            user_id=user_id, access_level=access_level,
            action="search", query=query,
        )

        # Query KG
        raw_results = self._query_kg(query, camera_id, time_range_hours, limit)

        # Anonymize results based on access level
        results = [self._anonymize(r, AccessLevel(access_level)) for r in raw_results]

        audit.result_ids = [r.result_id for r in results]
        self._audit_log.append(audit)

        # Store full results for potential unlock
        for raw, anon in zip(raw_results, results):
            self._unlocked_results[anon.result_id] = raw

        return ForensicSearchResponse(
            query=query, results=results,
            total_matches=len(results),
            access_level=access_level,
            audit_id=audit.audit_id,
        )

    def unlock_result(self, result_id: str, user_id: str,
                      access_level: str, justification: str = "") -> SearchResult | None:
        """Unlock a specific result — requires supervisor+ access."""
        if AccessLevel(access_level) not in (AccessLevel.SUPERVISOR, AccessLevel.ADMIN):
            logger.warning("Unlock denied: user=%s level=%s", user_id, access_level)
            return None

        full = self._unlocked_results.get(result_id)
        if not full:
            return None

        # Audit the unlock
        self._audit_log.append(AuditEntry(
            user_id=user_id, access_level=access_level,
            action="unlock", result_ids=[result_id],
            justification=justification,
        ))

        full.is_unlocked = True
        logger.info("Result %s unlocked by %s (level=%s): %s",
                    result_id, user_id, access_level, justification[:80])
        return full

    def get_audit_log(self, limit: int = 100) -> list[dict]:
        return [
            {"audit_id": a.audit_id, "timestamp": a.timestamp,
             "user_id": a.user_id, "access_level": a.access_level,
             "action": a.action, "query": a.query,
             "result_count": len(a.result_ids), "justification": a.justification}
            for a in self._audit_log[-limit:]
        ]

    def _query_kg(self, query: str, camera_id: str | None,
                  hours: int, limit: int) -> list[SearchResult]:
        if not self._kg:
            return []

        results = []
        try:
            # Parse query into search terms
            terms = query.lower().split()
            class_filter = None
            for cls in ["person", "vehicle", "car", "truck", "bicycle", "backpack"]:
                if cls in terms:
                    class_filter = cls
                    break

            with self._kg._driver.session() as s:
                # Search events
                cypher = """
                    MATCH (e:Event)-[:OCCURRED_AT]->(c:Camera)
                    WHERE e.timestamp >= datetime() - duration({hours: $hours})
                """
                params: dict = {"hours": hours, "lim": limit}
                if camera_id:
                    cypher += " AND c.camera_id = $cid"
                    params["cid"] = camera_id
                cypher += " RETURN e, c.camera_id AS cam ORDER BY e.timestamp DESC LIMIT $lim"

                result = s.run(cypher, **params)
                for rec in result:
                    evt = rec["e"]
                    results.append(SearchResult(
                        timestamp=0.0,
                        camera_id=rec.get("cam", ""),
                        description=str(evt.get("event_type", "")),
                        entity_class=class_filter or "unknown",
                        entity_id=str(evt.get("event_id", "")),
                        raw_data=dict(evt) if isinstance(evt, dict) else {},
                    ))

                # Search persons
                if class_filter == "person" or not class_filter:
                    p_result = s.run("""
                        MATCH (p:Person)-[r:DETECTED_IN]->(c:Camera)
                        WHERE r.last_seen >= datetime() - duration({hours: $hours})
                        RETURN p.track_id AS tid, c.camera_id AS cam,
                               r.last_seen AS ts, p.bbox AS bbox
                        ORDER BY r.last_seen DESC LIMIT $lim
                    """, hours=hours, lim=limit)
                    for rec in p_result:
                        results.append(SearchResult(
                            camera_id=rec.get("cam", ""),
                            description="Person detected",
                            entity_class="person",
                            entity_id=rec.get("tid", ""),
                            bbox=rec.get("bbox", []) or [],
                        ))
        except Exception as e:
            logger.error("Forensic search KG query failed: %s", e)

        return results[:limit]

    def _anonymize(self, result: SearchResult, level: AccessLevel) -> SearchResult:
        """Anonymize a search result based on access level."""
        anon = SearchResult(
            result_id=result.result_id,
            timestamp=result.timestamp,
            camera_id=result.camera_id,
            entity_class=result.entity_class,
            bbox=result.bbox,
        )

        if level == AccessLevel.ANONYMOUS:
            anon.entity_id = self._hash_id(result.entity_id)
            anon.description = self._strip_pii(result.description)
        elif level == AccessLevel.OPERATOR:
            anon.entity_id = self._hash_id(result.entity_id)
            anon.description = result.description
        else:
            anon.entity_id = result.entity_id
            anon.description = result.description
            anon.keyframe_ref = result.keyframe_ref

        return anon

    @staticmethod
    def _hash_id(entity_id: str) -> str:
        if not entity_id:
            return "anon"
        return "anon_" + hashlib.sha256(entity_id.encode()).hexdigest()[:8]

    @staticmethod
    def _strip_pii(description: str) -> str:
        """Remove potential PII from descriptions."""
        import re
        text = description
        # Remove anything that looks like a name pattern
        text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[REDACTED]', text)
        return text

    def export_response(self, response: ForensicSearchResponse) -> dict:
        return {
            "query": response.query,
            "total_matches": response.total_matches,
            "access_level": response.access_level,
            "audit_id": response.audit_id,
            "results": [
                {"result_id": r.result_id, "timestamp": r.timestamp,
                 "camera_id": r.camera_id, "description": r.description,
                 "entity_class": r.entity_class, "entity_id": r.entity_id,
                 "bbox": r.bbox, "is_unlocked": r.is_unlocked,
                 "keyframe_ref": r.keyframe_ref if r.is_unlocked else ""}
                for r in response.results
            ],
        }

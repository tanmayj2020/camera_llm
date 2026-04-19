"""Task 5: Graphiti Temporal Knowledge Graph — event ingestion with bi-temporal tracking."""

import logging
import time
from datetime import datetime, timezone

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class TemporalKnowledgeGraph:
    """Manages the temporal knowledge graph in Neo4j with Graphiti-style bi-temporal edges.

    Schema:
      Nodes: Camera, Person, Object, Zone, AudioEvent
      Edges carry valid_at / invalid_at timestamps for bi-temporal tracking.
    """

    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_schema()

    def close(self):
        self._driver.close()

    def _ensure_schema(self):
        with self._driver.session() as s:
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Camera) REQUIRE c.camera_id IS UNIQUE")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.track_id IS UNIQUE")
            s.run("CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.timestamp)")
            s.run("CREATE INDEX IF NOT EXISTS FOR (z:Zone) ON (z.zone_id)")

    # ------------------------------------------------------------------
    def upsert_camera(self, camera_id: str, meta: dict | None = None):
        with self._driver.session() as s:
            s.run(
                "MERGE (c:Camera {camera_id: $cid}) SET c += $meta",
                cid=camera_id, meta=meta or {},
            )

    def upsert_person(self, track_id: str, camera_id: str, timestamp: float,
                      bbox: list | None = None, embedding: list | None = None):
        ts = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        with self._driver.session() as s:
            s.run("""
                MERGE (p:Person {track_id: $tid})
                SET p.last_seen = $ts, p.bbox = $bbox
                WITH p
                MATCH (c:Camera {camera_id: $cid})
                MERGE (p)-[r:DETECTED_IN]->(c)
                SET r.valid_at = coalesce(r.valid_at, $ts), r.last_seen = $ts
            """, tid=track_id, cid=camera_id, ts=ts, bbox=bbox)

            if embedding:
                s.run("MATCH (p:Person {track_id: $tid}) SET p.embedding = $emb",
                      tid=track_id, emb=embedding)

    def upsert_object(self, class_name: str, track_id: str, camera_id: str, timestamp: float):
        ts = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        with self._driver.session() as s:
            s.run("""
                MERGE (o:Object {track_id: $tid, class_name: $cls})
                SET o.last_seen = $ts
                WITH o
                MATCH (c:Camera {camera_id: $cid})
                MERGE (o)-[r:DETECTED_IN]->(c)
                SET r.valid_at = coalesce(r.valid_at, $ts), r.last_seen = $ts
            """, tid=track_id, cls=class_name, cid=camera_id, ts=ts)

    def add_event(self, event_id: str, camera_id: str, timestamp: float,
                  event_type: str, data: dict | None = None):
        ts = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        with self._driver.session() as s:
            s.run("""
                CREATE (e:Event {event_id: $eid, timestamp: $ts, event_type: $etype, data: $data})
                WITH e
                MATCH (c:Camera {camera_id: $cid})
                CREATE (e)-[:OCCURRED_AT]->(c)
            """, eid=event_id, ts=ts, etype=event_type, cid=camera_id, data=str(data or {}))

    def add_zone_entry(self, track_id: str, zone_id: str, timestamp: float):
        ts = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        with self._driver.session() as s:
            s.run("""
                MERGE (z:Zone {zone_id: $zid})
                WITH z
                MATCH (p:Person {track_id: $tid})
                MERGE (p)-[r:ENTERED_ZONE]->(z)
                SET r.valid_at = coalesce(r.valid_at, $ts), r.last_seen = $ts
            """, tid=track_id, zid=zone_id, ts=ts)

    def add_interaction(self, track_id_a: str, track_id_b: str, timestamp: float, interaction_type: str = "near"):
        ts = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        with self._driver.session() as s:
            s.run("""
                MATCH (a {track_id: $a}), (b {track_id: $b})
                MERGE (a)-[r:INTERACTED_WITH {type: $itype}]->(b)
                SET r.valid_at = coalesce(r.valid_at, $ts), r.last_seen = $ts
            """, a=track_id_a, b=track_id_b, ts=ts, itype=interaction_type)

    def add_audio_event(self, camera_id: str, timestamp: float, sound_class: str, confidence: float):
        ts = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        with self._driver.session() as s:
            s.run("""
                CREATE (a:AudioEvent {timestamp: $ts, sound_class: $cls, confidence: $conf})
                WITH a
                MATCH (c:Camera {camera_id: $cid})
                CREATE (a)-[:HEARD_AT]->(c)
            """, ts=ts, cls=sound_class, conf=confidence, cid=camera_id)

    # ------------------------------------------------------------------
    # Query helpers
    def get_entity_history(self, track_id: str, since_hours: int = 24) -> list[dict]:
        cutoff = datetime.fromtimestamp(time.time() - since_hours * 3600, tz=timezone.utc).isoformat()
        with self._driver.session() as s:
            result = s.run("""
                MATCH (p {track_id: $tid})-[r]->(target)
                WHERE r.last_seen >= $cutoff
                RETURN type(r) AS rel, properties(r) AS props, labels(target) AS labels,
                       properties(target) AS target_props
                ORDER BY r.last_seen DESC
            """, tid=track_id, cutoff=cutoff)
            return [dict(r) for r in result]

    def get_zone_occupancy(self, zone_id: str) -> list[dict]:
        with self._driver.session() as s:
            result = s.run("""
                MATCH (p:Person)-[r:ENTERED_ZONE]->(z:Zone {zone_id: $zid})
                WHERE r.last_seen >= datetime() - duration('PT1H')
                RETURN p.track_id AS track_id, r.valid_at AS entered, r.last_seen AS last_seen
            """, zid=zone_id)
            return [dict(r) for r in result]

    def get_recent_events(self, camera_id: str, limit: int = 50) -> list[dict]:
        with self._driver.session() as s:
            result = s.run("""
                MATCH (e:Event)-[:OCCURRED_AT]->(c:Camera {camera_id: $cid})
                RETURN e.event_id AS event_id, e.timestamp AS timestamp,
                       e.event_type AS event_type, e.data AS data
                ORDER BY e.timestamp DESC LIMIT $lim
            """, cid=camera_id, lim=limit)
            return [dict(r) for r in result]

    # Entity resolution: find similar persons by embedding
    def find_similar_persons(self, embedding: list[float], threshold: float = 0.85) -> list[str]:
        """Find persons with cosine similarity above threshold."""
        with self._driver.session() as s:
            result = s.run("""
                MATCH (p:Person) WHERE p.embedding IS NOT NULL
                WITH p, gds.similarity.cosine(p.embedding, $emb) AS sim
                WHERE sim > $thresh
                RETURN p.track_id AS track_id, sim
                ORDER BY sim DESC
            """, emb=embedding, thresh=threshold)
            return [r["track_id"] for r in result]


def ingest_vision_event(kg: TemporalKnowledgeGraph, event: dict):
    """Ingest a VisionEvent dict into the knowledge graph."""
    camera_id = event["camera_id"]
    ts = event["timestamp"]

    kg.upsert_camera(camera_id)
    kg.add_event(event["event_id"], camera_id, ts, event["event_type"], event.get("objects"))

    for obj in event.get("objects", []):
        tid = f"{camera_id}_{obj['track_id']}"
        if obj["class_name"] == "person":
            kg.upsert_person(tid, camera_id, ts, obj.get("bbox"))
        else:
            kg.upsert_object(obj["class_name"], tid, camera_id, ts)

    for audio in event.get("audio_events", []):
        if audio.get("is_alert"):
            kg.add_audio_event(camera_id, ts, audio["class_name"], audio["confidence"])

"""Graphiti Temporal Knowledge Graph — bi-temporal tracking with causal chains.

SOTA features:
  1. Full bi-temporal edges: valid_at + invalid_at (relationships get closed)
  2. Temporal path queries (entity journey with transit times)
  3. Event causality edges (PRECEDED_BY / CAUSED_BY between temporally close events)
  4. TTL-based pruning of stale nodes and edges
"""

import logging
import time
from datetime import datetime, timezone

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

# Stale edge threshold: if not seen for this many seconds, close the edge
_STALE_EDGE_S = 120.0
# Causality window: events within this many seconds may be causally linked
_CAUSALITY_WINDOW_S = 10.0


class TemporalKnowledgeGraph:
    """Manages the temporal knowledge graph in Neo4j with Graphiti-style bi-temporal edges.

    Schema:
      Nodes: Camera, Person, Object, Zone, AudioEvent, Event
      Edges carry valid_at / invalid_at / last_seen for full bi-temporal tracking.
    """

    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_schema()
        self._last_prune = 0.0

    def close(self):
        self._driver.close()

    def _ensure_schema(self):
        with self._driver.session() as s:
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Camera) REQUIRE c.camera_id IS UNIQUE")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.track_id IS UNIQUE")
            s.run("CREATE INDEX IF NOT EXISTS FOR (e:Event) ON (e.timestamp)")
            s.run("CREATE INDEX IF NOT EXISTS FOR (z:Zone) ON (z.zone_id)")
            s.run("CREATE INDEX IF NOT EXISTS FOR (a:AudioEvent) ON (a.timestamp)")

    # ------------------------------------------------------------------
    # Core upserts with invalid_at support
    # ------------------------------------------------------------------

    def upsert_camera(self, camera_id: str, meta: dict | None = None):
        with self._driver.session() as s:
            s.run(
                "MERGE (c:Camera {camera_id: $cid}) SET c += $meta",
                cid=camera_id, meta=meta or {},
            )

    def upsert_person(self, track_id: str, camera_id: str, timestamp: float,
                      bbox: list | None = None, embedding: list | None = None):
        ts = self._iso(timestamp)
        with self._driver.session() as s:
            # Close any DETECTED_IN edges to OTHER cameras (person moved)
            s.run("""
                MATCH (p:Person {track_id: $tid})-[r:DETECTED_IN]->(c:Camera)
                WHERE c.camera_id <> $cid AND r.invalid_at IS NULL
                SET r.invalid_at = $ts
            """, tid=track_id, cid=camera_id, ts=ts)

            s.run("""
                MERGE (p:Person {track_id: $tid})
                SET p.last_seen = $ts, p.bbox = $bbox
                WITH p
                MATCH (c:Camera {camera_id: $cid})
                MERGE (p)-[r:DETECTED_IN]->(c)
                SET r.valid_at = coalesce(r.valid_at, $ts), r.last_seen = $ts, r.invalid_at = null
            """, tid=track_id, cid=camera_id, ts=ts, bbox=bbox)

            if embedding:
                s.run("MATCH (p:Person {track_id: $tid}) SET p.embedding = $emb",
                      tid=track_id, emb=embedding)

    def upsert_object(self, class_name: str, track_id: str, camera_id: str, timestamp: float):
        ts = self._iso(timestamp)
        with self._driver.session() as s:
            s.run("""
                MERGE (o:Object {track_id: $tid, class_name: $cls})
                SET o.last_seen = $ts
                WITH o
                MATCH (c:Camera {camera_id: $cid})
                MERGE (o)-[r:DETECTED_IN]->(c)
                SET r.valid_at = coalesce(r.valid_at, $ts), r.last_seen = $ts, r.invalid_at = null
            """, tid=track_id, cls=class_name, cid=camera_id, ts=ts)

    def add_event(self, event_id: str, camera_id: str, timestamp: float,
                  event_type: str, data: dict | None = None):
        ts = self._iso(timestamp)
        with self._driver.session() as s:
            s.run("""
                CREATE (e:Event {event_id: $eid, timestamp: $ts, event_type: $etype, data: $data})
                WITH e
                MATCH (c:Camera {camera_id: $cid})
                CREATE (e)-[:OCCURRED_AT]->(c)
            """, eid=event_id, ts=ts, etype=event_type, cid=camera_id, data=str(data or {}))

            # --- Upgrade 3: Causality edges ---
            # Link to recent preceding events on same camera within causality window
            s.run("""
                MATCH (new:Event {event_id: $eid})
                MATCH (prev:Event)-[:OCCURRED_AT]->(c:Camera {camera_id: $cid})
                WHERE prev.event_id <> $eid
                  AND prev.timestamp >= $window_start AND prev.timestamp < $ts
                WITH new, prev ORDER BY prev.timestamp DESC LIMIT 3
                CREATE (new)-[:PRECEDED_BY {delta_s: duration.between(prev.timestamp, new.timestamp).seconds}]->(prev)
            """, eid=event_id, cid=camera_id, ts=ts,
                window_start=self._iso(timestamp - _CAUSALITY_WINDOW_S))

    def add_zone_entry(self, track_id: str, zone_id: str, timestamp: float):
        ts = self._iso(timestamp)
        with self._driver.session() as s:
            s.run("""
                MERGE (z:Zone {zone_id: $zid})
                WITH z
                MATCH (p:Person {track_id: $tid})
                MERGE (p)-[r:ENTERED_ZONE]->(z)
                SET r.valid_at = coalesce(r.valid_at, $ts), r.last_seen = $ts, r.invalid_at = null
            """, tid=track_id, zid=zone_id, ts=ts)

    def close_zone_exit(self, track_id: str, zone_id: str, timestamp: float):
        """Explicitly close a zone relationship when entity leaves."""
        ts = self._iso(timestamp)
        with self._driver.session() as s:
            s.run("""
                MATCH (p:Person {track_id: $tid})-[r:ENTERED_ZONE]->(z:Zone {zone_id: $zid})
                WHERE r.invalid_at IS NULL
                SET r.invalid_at = $ts
            """, tid=track_id, zid=zone_id, ts=ts)

    def add_interaction(self, track_id_a: str, track_id_b: str, timestamp: float,
                        interaction_type: str = "near"):
        ts = self._iso(timestamp)
        with self._driver.session() as s:
            s.run("""
                MATCH (a {track_id: $a}), (b {track_id: $b})
                MERGE (a)-[r:INTERACTED_WITH {type: $itype}]->(b)
                SET r.valid_at = coalesce(r.valid_at, $ts), r.last_seen = $ts
            """, a=track_id_a, b=track_id_b, ts=ts, itype=interaction_type)

    def add_audio_event(self, camera_id: str, timestamp: float, sound_class: str, confidence: float):
        ts = self._iso(timestamp)
        with self._driver.session() as s:
            s.run("""
                CREATE (a:AudioEvent {timestamp: $ts, sound_class: $cls, confidence: $conf})
                WITH a
                MATCH (c:Camera {camera_id: $cid})
                CREATE (a)-[:HEARD_AT]->(c)
            """, ts=ts, cls=sound_class, conf=confidence, cid=camera_id)

            # --- Causality: link audio to temporally close visual events ---
            s.run("""
                MATCH (a:AudioEvent {timestamp: $ts, sound_class: $cls})
                MATCH (e:Event)-[:OCCURRED_AT]->(c:Camera {camera_id: $cid})
                WHERE e.timestamp >= $window_start AND e.timestamp <= $window_end
                WITH a, e ORDER BY abs(duration.between(a.timestamp, e.timestamp).seconds) LIMIT 2
                CREATE (a)-[:COINCIDES_WITH]->(e)
            """, ts=ts, cls=sound_class, cid=camera_id,
                window_start=self._iso(timestamp - _CAUSALITY_WINDOW_S),
                window_end=self._iso(timestamp + _CAUSALITY_WINDOW_S))

    # ------------------------------------------------------------------
    # Upgrade 2: Temporal path queries
    # ------------------------------------------------------------------

    def get_entity_journey(self, track_id: str, since_hours: int = 24) -> list[dict]:
        """Trace entity's full journey across cameras with transit times.

        Returns ordered list of camera visits with arrival/departure and transit gaps.
        """
        cutoff = self._iso(time.time() - since_hours * 3600)
        with self._driver.session() as s:
            result = s.run("""
                MATCH (p {track_id: $tid})-[r:DETECTED_IN]->(c:Camera)
                WHERE r.valid_at >= $cutoff
                RETURN c.camera_id AS camera_id,
                       r.valid_at AS arrived_at,
                       r.last_seen AS last_seen_at,
                       r.invalid_at AS departed_at
                ORDER BY r.valid_at ASC
            """, tid=track_id, cutoff=cutoff)
            visits = [dict(r) for r in result]

        # Compute transit times between consecutive camera visits
        for i in range(1, len(visits)):
            prev_depart = visits[i - 1].get("departed_at") or visits[i - 1].get("last_seen_at")
            curr_arrive = visits[i].get("arrived_at")
            if prev_depart and curr_arrive:
                try:
                    t_prev = self._parse_iso(str(prev_depart))
                    t_curr = self._parse_iso(str(curr_arrive))
                    visits[i]["transit_from"] = visits[i - 1]["camera_id"]
                    visits[i]["transit_time_s"] = round(t_curr - t_prev, 1)
                except Exception:
                    pass
        return visits

    def get_temporal_path(self, track_id: str, start_time: float, end_time: float) -> list[dict]:
        """Query: where was entity X between time A and time B?"""
        start_ts = self._iso(start_time)
        end_ts = self._iso(end_time)
        with self._driver.session() as s:
            result = s.run("""
                MATCH (p {track_id: $tid})-[r:DETECTED_IN]->(c:Camera)
                WHERE r.valid_at <= $end_ts
                  AND (r.invalid_at IS NULL OR r.invalid_at >= $start_ts)
                RETURN c.camera_id AS camera_id,
                       r.valid_at AS from_time,
                       coalesce(r.invalid_at, r.last_seen) AS to_time
                ORDER BY r.valid_at ASC
            """, tid=track_id, start_ts=start_ts, end_ts=end_ts)
            return [dict(r) for r in result]

    def get_causal_chain(self, event_id: str, depth: int = 5) -> list[dict]:
        """Follow PRECEDED_BY / COINCIDES_WITH edges to build a causal chain."""
        with self._driver.session() as s:
            result = s.run("""
                MATCH path = (e:Event {event_id: $eid})-[:PRECEDED_BY|COINCIDES_WITH*1..""" + str(depth) + """]->(prev)
                UNWIND nodes(path) AS n
                WITH DISTINCT n
                WHERE n:Event OR n:AudioEvent
                RETURN n.event_id AS event_id, n.timestamp AS timestamp,
                       n.event_type AS event_type, n.sound_class AS sound_class,
                       labels(n) AS labels
                ORDER BY n.timestamp ASC
            """, eid=event_id)
            return [dict(r) for r in result]

    # ------------------------------------------------------------------
    # Upgrade 4: TTL-based pruning
    # ------------------------------------------------------------------

    def close_stale_edges(self, stale_threshold_s: float = _STALE_EDGE_S):
        """Close edges where last_seen is older than threshold (entity left but edge never closed)."""
        cutoff = self._iso(time.time() - stale_threshold_s)
        now = self._iso(time.time())
        with self._driver.session() as s:
            result = s.run("""
                MATCH ()-[r:DETECTED_IN]->()
                WHERE r.invalid_at IS NULL AND r.last_seen < $cutoff
                SET r.invalid_at = $now
                RETURN count(r) AS closed
            """, cutoff=cutoff, now=now)
            closed = result.single()["closed"]

            result2 = s.run("""
                MATCH ()-[r:ENTERED_ZONE]->()
                WHERE r.invalid_at IS NULL AND r.last_seen < $cutoff
                SET r.invalid_at = $now
                RETURN count(r) AS closed
            """, cutoff=cutoff, now=now)
            closed += result2.single()["closed"]

            if closed:
                logger.info("Closed %d stale edges (threshold=%ds)", closed, stale_threshold_s)
            return closed

    def prune_old_data(self, max_age_hours: int = 168):
        """Delete events, audio events, and detached nodes older than max_age_hours."""
        cutoff = self._iso(time.time() - max_age_hours * 3600)
        with self._driver.session() as s:
            # Delete old events
            r1 = s.run("""
                MATCH (e:Event) WHERE e.timestamp < $cutoff
                DETACH DELETE e RETURN count(e) AS deleted
            """, cutoff=cutoff)
            ev_del = r1.single()["deleted"]

            # Delete old audio events
            r2 = s.run("""
                MATCH (a:AudioEvent) WHERE a.timestamp < $cutoff
                DETACH DELETE a RETURN count(a) AS deleted
            """, cutoff=cutoff)
            au_del = r2.single()["deleted"]

            # Delete orphan persons not seen recently
            r3 = s.run("""
                MATCH (p:Person) WHERE p.last_seen < $cutoff
                AND NOT EXISTS { (p)-[:DETECTED_IN]->() WHERE true }
                DETACH DELETE p RETURN count(p) AS deleted
            """, cutoff=cutoff)
            p_del = r3.single()["deleted"]

            logger.info("Pruned: %d events, %d audio, %d orphan persons (age > %dh)",
                        ev_del, au_del, p_del, max_age_hours)
            return {"events": ev_del, "audio": au_del, "persons": p_del}

    def maybe_maintenance(self, interval_s: float = 300):
        """Run stale-edge closing + pruning periodically (call from event pipeline)."""
        now = time.time()
        if now - self._last_prune < interval_s:
            return
        self._last_prune = now
        try:
            self.close_stale_edges()
            self.prune_old_data()
        except Exception as e:
            logger.warning("KG maintenance failed: %s", e)

    # ------------------------------------------------------------------
    # Original query helpers (unchanged)
    # ------------------------------------------------------------------

    def get_entity_history(self, track_id: str, since_hours: int = 24) -> list[dict]:
        cutoff = self._iso(time.time() - since_hours * 3600)
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
                WHERE r.invalid_at IS NULL
                  AND r.last_seen >= datetime() - duration('PT1H')
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

    def find_similar_persons(self, embedding: list[float], threshold: float = 0.85) -> list[str]:
        with self._driver.session() as s:
            result = s.run("""
                MATCH (p:Person) WHERE p.embedding IS NOT NULL
                WITH p, gds.similarity.cosine(p.embedding, $emb) AS sim
                WHERE sim > $thresh
                RETURN p.track_id AS track_id, sim
                ORDER BY sim DESC
            """, emb=embedding, thresh=threshold)
            return [r["track_id"] for r in result]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iso(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    @staticmethod
    def _parse_iso(ts_str: str) -> float:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()


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

    # Periodic maintenance (stale edge closing + pruning)
    kg.maybe_maintenance()

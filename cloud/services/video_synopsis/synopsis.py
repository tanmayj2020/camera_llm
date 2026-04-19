"""Forensic Video Synopsis — compress hours of footage into condensed activity summaries.

All objects time-shifted to appear together. LLM-narrated.
BriefCam-style but AI-native.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SynopsisEntity:
    entity_id: str
    class_name: str
    first_seen: float = 0.0
    last_seen: float = 0.0
    cameras: list[str] = field(default_factory=list)
    event_count: int = 0
    path_summary: str = ""  # "entered zone A → moved to zone B → exited"


@dataclass
class SynopsisSegment:
    """A time-compressed segment where multiple entities are overlaid."""
    segment_index: int
    original_start: float
    original_end: float
    entities: list[SynopsisEntity] = field(default_factory=list)
    description: str = ""
    keyframe_refs: list[str] = field(default_factory=list)


@dataclass
class VideoSynopsis:
    camera_id: str
    time_range_start: float
    time_range_end: float
    original_duration_s: float
    synopsis_duration_s: float  # compressed duration
    compression_ratio: float
    total_entities: int
    segments: list[SynopsisSegment] = field(default_factory=list)
    entity_index: list[SynopsisEntity] = field(default_factory=list)
    narrative: str = ""
    generated_at: float = field(default_factory=time.time)


class SynopsisEngine:
    """Generates forensic video synopses from KG event data.

    Queries all events in a time range, clusters by entity,
    time-shifts them into compressed segments, and generates LLM narration.
    """

    MAX_SEGMENTS = 20
    SEGMENT_DURATION_S = 30  # each synopsis segment represents 30s of compressed time

    def __init__(self, kg=None):
        self._kg = kg
        self._vlm_client = None

    def set_vlm_client(self, client):
        self._vlm_client = client

    def generate(self, camera_id: str, start_time: float = None,
                 end_time: float = None, hours: float = 4.0) -> VideoSynopsis:
        now = time.time()
        end_time = end_time or now
        start_time = start_time or (end_time - hours * 3600)
        original_duration = end_time - start_time

        # Step 1: Gather all events from KG
        events = self._query_events(camera_id, start_time, end_time)

        # Step 2: Cluster by entity
        entity_clusters = self._cluster_by_entity(events)

        # Step 3: Build entity index
        entity_index = self._build_entity_index(entity_clusters, camera_id)

        # Step 4: Time-compress into synopsis segments
        segments = self._compress_segments(entity_clusters, start_time, end_time)

        synopsis_duration = len(segments) * self.SEGMENT_DURATION_S
        compression = original_duration / max(synopsis_duration, 1)

        synopsis = VideoSynopsis(
            camera_id=camera_id,
            time_range_start=start_time,
            time_range_end=end_time,
            original_duration_s=original_duration,
            synopsis_duration_s=synopsis_duration,
            compression_ratio=round(compression, 1),
            total_entities=len(entity_index),
            segments=segments,
            entity_index=entity_index,
        )

        # Step 5: Generate narrative
        synopsis.narrative = self._generate_narrative(synopsis)

        logger.info("Synopsis generated: %s %.1fh → %d segments (%.0fx compression), %d entities",
                    camera_id, hours, len(segments), compression, len(entity_index))
        return synopsis

    def _query_events(self, camera_id: str, start: float, end: float) -> list[dict]:
        if not self._kg:
            return []
        try:
            from datetime import datetime, timezone
            start_iso = datetime.fromtimestamp(start, tz=timezone.utc).isoformat()
            end_iso = datetime.fromtimestamp(end, tz=timezone.utc).isoformat()
            with self._kg._driver.session() as s:
                result = s.run("""
                    MATCH (e:Event)-[:OCCURRED_AT]->(c:Camera {camera_id: $cid})
                    WHERE e.timestamp >= $start AND e.timestamp <= $end
                    RETURN e.event_id AS eid, e.timestamp AS ts,
                           e.event_type AS etype, e.data AS data
                    ORDER BY e.timestamp
                """, cid=camera_id, start=start_iso, end=end_iso)
                return [dict(r) for r in result]
        except Exception as e:
            logger.error("Synopsis KG query failed: %s", e)
            return []

    def _cluster_by_entity(self, events: list[dict]) -> dict[str, list[dict]]:
        clusters: dict[str, list[dict]] = defaultdict(list)
        for evt in events:
            data_str = str(evt.get("data", ""))
            # Extract entity references from event data
            eid = evt.get("eid", "unknown")
            clusters[eid].append(evt)
            # Also try to extract track IDs from data
            import re
            track_ids = re.findall(r"track_id['\"]?\s*[:=]\s*['\"]?(\w+)", data_str)
            for tid in track_ids:
                clusters[f"track_{tid}"].append(evt)
        return clusters

    def _build_entity_index(self, clusters: dict[str, list[dict]],
                            camera_id: str) -> list[SynopsisEntity]:
        index = []
        for eid, events in clusters.items():
            if not events:
                continue
            timestamps = [self._evt_ts(e) for e in events]
            timestamps = [t for t in timestamps if t > 0]
            if not timestamps:
                continue
            index.append(SynopsisEntity(
                entity_id=eid,
                class_name=events[0].get("etype", "unknown"),
                first_seen=min(timestamps),
                last_seen=max(timestamps),
                cameras=[camera_id],
                event_count=len(events),
            ))
        index.sort(key=lambda e: e.first_seen)
        return index

    def _compress_segments(self, clusters: dict[str, list[dict]],
                           start: float, end: float) -> list[SynopsisSegment]:
        """Time-compress: divide original range into N buckets, overlay all entities per bucket."""
        duration = end - start
        n_segments = min(self.MAX_SEGMENTS, max(1, int(duration / 900)))  # ~1 segment per 15 min
        bucket_size = duration / n_segments

        segments = []
        for i in range(n_segments):
            bucket_start = start + i * bucket_size
            bucket_end = bucket_start + bucket_size

            # Find entities active in this bucket
            active_entities = []
            for eid, events in clusters.items():
                bucket_events = [e for e in events
                                 if bucket_start <= self._evt_ts(e) <= bucket_end]
                if bucket_events:
                    active_entities.append(SynopsisEntity(
                        entity_id=eid,
                        class_name=bucket_events[0].get("etype", "unknown"),
                        first_seen=self._evt_ts(bucket_events[0]),
                        last_seen=self._evt_ts(bucket_events[-1]),
                        event_count=len(bucket_events),
                    ))

            if active_entities:
                segments.append(SynopsisSegment(
                    segment_index=i,
                    original_start=bucket_start,
                    original_end=bucket_end,
                    entities=active_entities,
                    description=f"{len(active_entities)} entities active",
                ))

        return segments

    def _generate_narrative(self, synopsis: VideoSynopsis) -> str:
        if self._vlm_client and hasattr(self._vlm_client, '_get_client'):
            client = self._vlm_client._get_client()
            if client != "stub":
                try:
                    seg_summaries = [f"Segment {s.segment_index}: {len(s.entities)} entities "
                                     f"({', '.join(e.entity_id[:12] for e in s.entities[:5])})"
                                     for s in synopsis.segments[:15]]
                    prompt = (f"Narrate a forensic video synopsis for camera {synopsis.camera_id}.\n"
                              f"Time range: {synopsis.original_duration_s / 3600:.1f} hours\n"
                              f"Total entities: {synopsis.total_entities}\n"
                              f"Compression: {synopsis.compression_ratio:.0f}x\n"
                              f"Segments:\n" + "\n".join(seg_summaries) + "\n\n"
                              f"Write a concise 2-3 paragraph forensic summary describing "
                              f"activity patterns, peak periods, and notable observations.")
                    resp = client.generate_content(prompt)
                    return resp.text.strip()
                except Exception:
                    pass

        # Fallback
        parts = [f"Video synopsis for camera {synopsis.camera_id}: "
                 f"{synopsis.original_duration_s / 3600:.1f} hours compressed to "
                 f"{len(synopsis.segments)} segments ({synopsis.compression_ratio:.0f}x)."]
        parts.append(f"{synopsis.total_entities} unique entities detected.")
        peak = max(synopsis.segments, key=lambda s: len(s.entities)) if synopsis.segments else None
        if peak:
            from datetime import datetime, timezone
            peak_time = datetime.fromtimestamp(peak.original_start, tz=timezone.utc).strftime("%H:%M")
            parts.append(f"Peak activity at {peak_time} with {len(peak.entities)} simultaneous entities.")
        return " ".join(parts)

    def export(self, synopsis: VideoSynopsis) -> dict:
        from datetime import datetime, timezone
        fmt = lambda t: datetime.fromtimestamp(t, tz=timezone.utc).isoformat() if t else None
        return {
            "camera_id": synopsis.camera_id,
            "time_range": {"start": fmt(synopsis.time_range_start), "end": fmt(synopsis.time_range_end)},
            "original_duration_hours": round(synopsis.original_duration_s / 3600, 2),
            "synopsis_segments": len(synopsis.segments),
            "compression_ratio": synopsis.compression_ratio,
            "total_entities": synopsis.total_entities,
            "narrative": synopsis.narrative,
            "segments": [
                {"index": s.segment_index,
                 "original_start": fmt(s.original_start),
                 "original_end": fmt(s.original_end),
                 "entity_count": len(s.entities),
                 "entities": [{"id": e.entity_id, "class": e.class_name,
                               "events": e.event_count} for e in s.entities[:10]],
                 "description": s.description}
                for s in synopsis.segments
            ],
            "entity_index": [
                {"id": e.entity_id, "class": e.class_name,
                 "first_seen": fmt(e.first_seen), "last_seen": fmt(e.last_seen),
                 "events": e.event_count}
                for e in synopsis.entity_index[:50]
            ],
        }

    @staticmethod
    def _evt_ts(evt: dict) -> float:
        ts = evt.get("ts", "")
        if isinstance(ts, (int, float)):
            return float(ts)
        if isinstance(ts, str) and ts:
            try:
                from datetime import datetime, timezone
                return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
            except (ValueError, TypeError):
                pass
        return 0.0

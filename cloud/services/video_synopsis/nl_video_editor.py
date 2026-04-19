"""Natural Language Video Editing — generate clips from NL queries.

"Show me only when the person in red was near the exit" → segment list with timestamps.
"""

import json
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class VideoClipSegment:
    start_time: float
    end_time: float
    camera_id: str
    description: str
    relevance_score: float


@dataclass
class VideoEditResult:
    query: str
    segments: list[VideoClipSegment] = field(default_factory=list)
    total_duration_s: float = 0.0
    compression_ratio: float = 0.0


class NLVideoEditor:
    """Generates video clip segment lists from natural language queries."""

    def __init__(self, kg=None, rag=None, vlm_client=None):
        self._kg = kg
        self._rag = rag
        self._vlm_client = vlm_client

    def edit(self, query: str, camera_id: str = None, hours: float = 4.0) -> VideoEditResult:
        result = VideoEditResult(query=query)

        # Retrieve relevant segments via RAG
        segments_raw = []
        if self._rag:
            try:
                retrieved = self._rag.retrieve(query, top_k=20)
                for r in retrieved:
                    segments_raw.append(VideoClipSegment(
                        start_time=r.timestamp - 5, end_time=r.timestamp + 10,
                        camera_id=r.camera_id or camera_id or "",
                        description=r.event_summary, relevance_score=r.score))
            except Exception:
                pass

        # Fallback: KG event search
        if not segments_raw and self._kg and camera_id:
            try:
                events = self._kg.get_recent_events(camera_id, limit=100)
                for e in events:
                    data_str = str(e.get("data", "")).lower()
                    if any(w in data_str for w in query.lower().split() if len(w) > 3):
                        segments_raw.append(VideoClipSegment(
                            start_time=0, end_time=0, camera_id=camera_id,
                            description=e.get("event_type", ""), relevance_score=0.5))
            except Exception:
                pass

        # VLM refinement: ask which segments best match the query
        if self._vlm_client and segments_raw and hasattr(self._vlm_client, '_get_client'):
            client = self._vlm_client._get_client()
            if client != "stub":
                try:
                    prompt = (f"Given query: '{query}'\nRank these segments by relevance (1=best):\n"
                              + "\n".join(f"{i}: {s.description}" for i, s in enumerate(segments_raw[:15]))
                              + "\nReply JSON list of indices in order of relevance.")
                    resp = client.generate_content(prompt).text.strip()
                    if resp.startswith("```"):
                        resp = resp.split("\n", 1)[1].rsplit("```", 1)[0]
                    order = json.loads(resp)
                    if isinstance(order, list):
                        reordered = [segments_raw[i] for i in order if i < len(segments_raw)]
                        segments_raw = reordered + [s for s in segments_raw if s not in reordered]
                except Exception:
                    pass

        # Merge overlapping segments
        result.segments = self._merge_segments(segments_raw[:20])
        result.total_duration_s = sum(s.end_time - s.start_time for s in result.segments)
        result.compression_ratio = (hours * 3600) / max(result.total_duration_s, 1)
        return result

    @staticmethod
    def _merge_segments(segments: list[VideoClipSegment]) -> list[VideoClipSegment]:
        if not segments:
            return []
        sorted_segs = sorted(segments, key=lambda s: s.start_time)
        merged = [sorted_segs[0]]
        for s in sorted_segs[1:]:
            if s.start_time <= merged[-1].end_time + 5:  # 5s gap tolerance
                merged[-1].end_time = max(merged[-1].end_time, s.end_time)
                merged[-1].relevance_score = max(merged[-1].relevance_score, s.relevance_score)
            else:
                merged.append(s)
        return merged

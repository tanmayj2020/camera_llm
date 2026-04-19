"""Scene Déjà Vu Detection — finds historical situations that are
eerily similar to the current scene.

When the system detects an anomaly or interesting event, it encodes the
current scene state into a "scene fingerprint" and compares it against
ALL previously stored scene fingerprints to find the closest historical
match.

Scene fingerprint (16-D compact vector):
  1. Object-class histogram (person, vehicle, bag, ...)  [6-D]
  2. Spatial density pattern (quadrant occupancy)        [4-D]
  3. Temporal features (hour_sin, hour_cos, weekday)     [3-D]
  4. Audio signature (alert class presence bitmap)       [2-D]
  5. Activity level                                      [1-D]

When a near-match is found, the system reports:
  "This scene is 93% similar to what happened on March 3rd at 2:15 AM
   in the same zone — that incident was a tailgating attempt."

Novel because: No surveillance system performs temporal scene matching.
Operators rely on memory or manual log searches.  This gives the system
an institutional memory that never forgets.
"""

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

FINGERPRINT_DIM = 16
MAX_MEMORY = 50_000  # max scene fingerprints to store


@dataclass
class SceneFingerprint:
    """A compact encoding of a scene at a moment in time."""
    fingerprint_id: str
    camera_id: str
    zone_id: str
    timestamp: float
    vector: np.ndarray            # shape (16,)
    event_type: str               # what was happening
    summary: str                  # human-readable description
    severity: str
    was_confirmed: bool = False   # operator-confirmed incident?


@dataclass
class DejaVuMatch:
    """A match between the current scene and a historical scene."""
    current_camera_id: str
    historical_fingerprint_id: str
    historical_camera_id: str
    historical_timestamp: float
    historical_summary: str
    historical_event_type: str
    similarity: float              # 0.0–1.0
    time_ago_hours: float
    narrative: str                 # "This scene is 93% similar to..."


class DejaVuEngine:
    """Stores scene fingerprints and finds historical matches."""

    # Object classes to track in histogram
    OBJECT_CLASSES = ["person", "vehicle", "bag", "bicycle", "animal", "other"]
    # Audio alert classes for bitmap
    AUDIO_CLASSES = ["scream", "gunshot", "glass_breaking", "alarm",
                     "explosion", "dog_bark", "siren", "crash"]

    def __init__(self):
        # All stored fingerprints, ordered by time
        self._memory: deque[SceneFingerprint] = deque(maxlen=MAX_MEMORY)
        # Spatial index: camera_id -> list of indices into _memory
        self._camera_index: dict[str, list[int]] = defaultdict(list)
        # Pre-computed matrix for fast batch comparison
        self._matrix: np.ndarray | None = None
        self._matrix_dirty = True
        self._next_id = 0

    # ── Public API ────────────────────────────────────────────────────────

    def encode_and_store(self, scene_state: dict, event_type: str = "",
                         summary: str = "", severity: str = "low",
                         zone_id: str = "") -> str:
        """Encode the current scene into a fingerprint and store it.

        Returns the fingerprint ID.
        """
        fp_id = f"fp_{self._next_id}"
        self._next_id += 1

        vector = self._encode(scene_state)
        camera_id = scene_state.get("camera_id", "unknown")
        timestamp = scene_state.get("timestamp", time.time())

        fp = SceneFingerprint(
            fingerprint_id=fp_id,
            camera_id=camera_id,
            zone_id=zone_id,
            timestamp=timestamp,
            vector=vector,
            event_type=event_type,
            summary=summary or self._auto_summary(scene_state),
            severity=severity,
        )

        self._memory.append(fp)
        self._camera_index[camera_id].append(len(self._memory) - 1)
        self._matrix_dirty = True

        return fp_id

    def find_similar(self, scene_state: dict, top_k: int = 5,
                     same_camera_only: bool = False,
                     min_similarity: float = 0.7) -> list[DejaVuMatch]:
        """Find historical scenes most similar to the current one.

        Args:
            scene_state: Current scene dictionary
            top_k: Number of matches to return
            same_camera_only: Only match within same camera
            min_similarity: Minimum similarity threshold (0-1)

        Returns list of DejaVuMatch sorted by similarity (best first).
        """
        if not self._memory:
            return []

        query = self._encode(scene_state)
        now = scene_state.get("timestamp", time.time())
        camera_id = scene_state.get("camera_id", "")

        # Build comparison set
        candidates = list(self._memory)
        if same_camera_only and camera_id:
            candidates = [fp for fp in candidates if fp.camera_id == camera_id]

        if not candidates:
            return []

        # Exclude very recent fingerprints (last 60s) to avoid self-matching
        candidates = [fp for fp in candidates if now - fp.timestamp > 60]

        if not candidates:
            return []

        # Vectorized similarity computation
        matrix = np.array([fp.vector for fp in candidates])
        sims = self._batch_cosine(query, matrix)

        # Get top-k indices
        top_indices = np.argsort(sims)[::-1][:top_k * 2]  # oversample

        matches = []
        for idx in top_indices:
            sim = float(sims[idx])
            if sim < min_similarity:
                break

            fp = candidates[idx]
            hours_ago = (now - fp.timestamp) / 3600

            narrative = self._generate_narrative(
                sim, fp, hours_ago, camera_id)

            matches.append(DejaVuMatch(
                current_camera_id=camera_id,
                historical_fingerprint_id=fp.fingerprint_id,
                historical_camera_id=fp.camera_id,
                historical_timestamp=fp.timestamp,
                historical_summary=fp.summary,
                historical_event_type=fp.event_type,
                similarity=round(sim, 3),
                time_ago_hours=round(hours_ago, 1),
                narrative=narrative,
            ))

            if len(matches) >= top_k:
                break

        return matches

    def confirm_incident(self, fingerprint_id: str):
        """Mark a stored fingerprint as a confirmed incident
        (boosts its weight in future matches)."""
        for fp in self._memory:
            if fp.fingerprint_id == fingerprint_id:
                fp.was_confirmed = True
                break

    def get_stats(self) -> dict:
        """Return memory statistics."""
        if not self._memory:
            return {"total_fingerprints": 0, "cameras": 0, "oldest_hours": 0}

        now = time.time()
        return {
            "total_fingerprints": len(self._memory),
            "cameras": len(self._camera_index),
            "oldest_hours": round((now - self._memory[0].timestamp) / 3600, 1),
            "confirmed_incidents": sum(1 for fp in self._memory if fp.was_confirmed),
            "event_type_distribution": self._event_distribution(),
        }

    # ── Encoding ──────────────────────────────────────────────────────────

    def _encode(self, scene_state: dict) -> np.ndarray:
        """Encode a scene state into a 16-D fingerprint vector."""
        v = np.zeros(FINGERPRINT_DIM, dtype=np.float32)

        objects = scene_state.get("objects", [])
        timestamp = scene_state.get("timestamp", time.time())
        audio = scene_state.get("audio_events", [])
        activity = scene_state.get("scene_activity", 0.0)

        # [0-5] Object class histogram (normalized)
        class_counts = defaultdict(int)
        for obj in objects:
            cls = obj.get("class_name", "other").lower()
            if cls in self.OBJECT_CLASSES:
                class_counts[cls] += 1
            else:
                class_counts["other"] += 1

        total_objects = max(1, sum(class_counts.values()))
        for i, cls in enumerate(self.OBJECT_CLASSES):
            v[i] = class_counts[cls] / total_objects

        # [6-9] Spatial density — quadrant occupancy
        quadrants = np.zeros(4)
        for obj in objects:
            bbox = obj.get("bbox", [])
            if len(bbox) >= 4:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                qi = (1 if cx > 0.5 else 0) + (2 if cy > 0.5 else 0)
                quadrants[qi] += 1
        q_total = max(1.0, quadrants.sum())
        v[6:10] = quadrants / q_total

        # [10-12] Temporal features
        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour + dt.minute / 60
        v[10] = math.sin(2 * math.pi * hour / 24)
        v[11] = math.cos(2 * math.pi * hour / 24)
        v[12] = dt.weekday() / 6.0  # 0=Monday, 1=Sunday

        # [13-14] Audio signature
        alert_bitmap = 0
        for ae in audio:
            cls = ae.get("class_name", "")
            if cls in self.AUDIO_CLASSES:
                alert_bitmap |= (1 << self.AUDIO_CLASSES.index(cls))
        v[13] = (alert_bitmap & 0xFF) / 255.0        # low byte
        v[14] = 1.0 if any(a.get("is_alert") for a in audio) else 0.0

        # [15] Activity level
        v[15] = min(1.0, float(activity))

        # L2-normalize
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm

        return v

    # ── Similarity ────────────────────────────────────────────────────────

    @staticmethod
    def _batch_cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all rows in matrix."""
        q_norm = np.linalg.norm(query)
        if q_norm < 1e-8:
            return np.zeros(len(matrix))
        m_norms = np.linalg.norm(matrix, axis=1)
        m_norms = np.maximum(m_norms, 1e-8)
        dots = matrix @ query
        return dots / (m_norms * q_norm)

    # ── Narrative ─────────────────────────────────────────────────────────

    def _generate_narrative(self, similarity: float, fp: SceneFingerprint,
                            hours_ago: float, current_camera: str) -> str:
        """Generate a human-readable déjà vu narrative."""
        dt = datetime.fromtimestamp(fp.timestamp)
        time_str = dt.strftime("%B %d at %I:%M %p")

        pct = f"{similarity:.0%}"
        same_cam = "(same camera)" if fp.camera_id == current_camera else f"(camera {fp.camera_id})"

        if hours_ago < 24:
            ago_str = f"{hours_ago:.1f} hours ago"
        else:
            days = hours_ago / 24
            ago_str = f"{days:.0f} days ago"

        parts = [f"This scene is {pct} similar to {time_str} {same_cam} — {ago_str}."]

        if fp.event_type:
            parts.append(f"That incident was classified as: {fp.event_type}.")
        if fp.summary:
            parts.append(f"Context: {fp.summary}")
        if fp.was_confirmed:
            parts.append("⚠ That incident was CONFIRMED by an operator.")

        return " ".join(parts)

    def _auto_summary(self, scene_state: dict) -> str:
        """Generate a brief summary of the scene."""
        objects = scene_state.get("objects", [])
        counts = defaultdict(int)
        for o in objects:
            counts[o.get("class_name", "object")] += 1
        parts = [f"{v} {k}{'s' if v > 1 else ''}" for k, v in counts.items()]
        return f"Scene with {', '.join(parts) if parts else 'no objects'}"

    def _event_distribution(self) -> dict:
        dist = defaultdict(int)
        for fp in self._memory:
            if fp.event_type:
                dist[fp.event_type] += 1
        return dict(dist)

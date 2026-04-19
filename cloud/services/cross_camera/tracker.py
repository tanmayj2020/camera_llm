"""Task 16: Multi-Camera Correlation + Cross-Camera Intelligence."""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraTopology:
    camera_id: str
    adjacent_cameras: list[str]
    expected_transit_time_s: dict  # {neighbor_camera_id: seconds}
    position: tuple[float, float]  # (x, y) in site coordinates


class CrossCameraTracker:
    """Tracks entities across multiple cameras using appearance re-identification.

    - Maintains camera topology (adjacency, expected transit times)
    - Re-identifies persons across cameras via embedding similarity
    - Detects unusual routes (skipped cameras, unexpected transit times)
    """

    def __init__(self, kg=None, similarity_threshold: float = 0.80):
        self.kg = kg
        self.threshold = similarity_threshold
        self._topology: dict[str, CameraTopology] = {}
        self._embeddings: dict[str, np.ndarray] = {}  # global_track_id -> embedding

    def add_camera(self, topology: CameraTopology):
        self._topology[topology.camera_id] = topology

    def match_across_cameras(self, track_id: str, camera_id: str,
                              embedding: np.ndarray) -> str | None:
        """Try to match a detection to an existing cross-camera identity.
        Returns global_track_id if matched, None otherwise."""
        best_match = None
        best_sim = self.threshold

        for gid, emb in self._embeddings.items():
            sim = float(np.dot(embedding, emb) / (np.linalg.norm(embedding) * np.linalg.norm(emb) + 1e-8))
            if sim > best_sim:
                best_sim = sim
                best_match = gid

        if best_match:
            # Update embedding with EMA
            self._embeddings[best_match] = 0.9 * self._embeddings[best_match] + 0.1 * embedding
            logger.info("Re-ID: %s@%s matched to global %s (sim=%.3f)", track_id, camera_id, best_match, best_sim)

            # Merge in knowledge graph
            if self.kg:
                local_id = f"{camera_id}_{track_id}"
                self.kg.add_interaction(local_id, best_match, 0.0, "same_person")
            return best_match

        # New global identity
        global_id = f"global_{camera_id}_{track_id}"
        self._embeddings[global_id] = embedding.copy()
        return global_id

    def detect_unusual_route(self, global_track_id: str, camera_sequence: list[tuple[str, float]]) -> dict | None:
        """Detect if a person's camera sequence is unusual.

        camera_sequence: [(camera_id, timestamp), ...] in order of appearance.
        """
        if len(camera_sequence) < 2:
            return None

        for i in range(len(camera_sequence) - 1):
            cam_a, ts_a = camera_sequence[i]
            cam_b, ts_b = camera_sequence[i + 1]
            transit = ts_b - ts_a

            topo_a = self._topology.get(cam_a)
            if topo_a is None:
                continue

            # Check if cam_b is adjacent
            if cam_b not in topo_a.adjacent_cameras:
                return {
                    "type": "skipped_camera",
                    "description": f"{global_track_id} went from {cam_a} to {cam_b} (not adjacent)",
                    "from_camera": cam_a, "to_camera": cam_b, "transit_time": transit,
                }

            # Check transit time
            expected = topo_a.expected_transit_time_s.get(cam_b, 60)
            if transit > expected * 3 or transit < expected * 0.2:
                return {
                    "type": "unusual_transit_time",
                    "description": f"{global_track_id}: {cam_a}→{cam_b} took {transit:.0f}s (expected ~{expected:.0f}s)",
                    "from_camera": cam_a, "to_camera": cam_b,
                    "transit_time": transit, "expected_time": expected,
                }

        return None

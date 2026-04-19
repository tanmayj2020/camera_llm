"""Behavioral intent recognition — pre-incident detection from pose + trajectory analysis.

Detects: casing, nervous pacing, evasive movement, aggressive approach.
Optional VLM deep analysis for ambiguous cases.
"""

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    CASING = "casing"
    NERVOUS_PACING = "nervous_pacing"
    EVASIVE_MOVEMENT = "evasive_movement"
    AGGRESSIVE_APPROACH = "aggressive_approach"
    NORMAL = "normal"


@dataclass
class IntentResult:
    entity_id: str
    intent: IntentType
    confidence: float
    description: str
    evidence: dict = field(default_factory=dict)


@dataclass
class _TrajectoryBuffer:
    positions: deque = field(default_factory=lambda: deque(maxlen=120))  # ~2 min at 1Hz
    timestamps: deque = field(default_factory=lambda: deque(maxlen=120))
    head_angles: deque = field(default_factory=lambda: deque(maxlen=60))  # from pose


class IntentRecognizer:
    """Analyzes pose sequences + trajectories to detect pre-incident behavioral patterns."""

    # Thresholds
    CASING_HEAD_SCAN_RANGE = 60.0       # degrees of head rotation indicating scanning
    CASING_MIN_STATIONARY_S = 5.0
    PACING_MIN_REVERSALS = 3
    PACING_MAX_AREA_M2 = 4.0
    EVASIVE_MIN_TURNS = 2
    EVASIVE_SPEED_VARIANCE = 0.5
    AGGRESSIVE_MIN_SPEED = 2.0          # m/s
    AGGRESSIVE_DIRECTNESS = 0.8         # ratio of displacement to path length

    def __init__(self):
        self._buffers: dict[str, _TrajectoryBuffer] = defaultdict(_TrajectoryBuffer)
        self._vlm_client = None

    def set_vlm_client(self, client):
        self._vlm_client = client

    def update(self, entity_id: str, position: np.ndarray, timestamp: float,
               pose_keypoints: list | None = None):
        """Feed new position + optional pose data."""
        buf = self._buffers[entity_id]
        buf.positions.append(position.copy())
        buf.timestamps.append(timestamp)

        if pose_keypoints:
            head_angle = self._estimate_head_angle(pose_keypoints)
            if head_angle is not None:
                buf.head_angles.append(head_angle)

    def analyze(self, entity_id: str, target_entities: dict[str, np.ndarray] | None = None) -> IntentResult:
        """Analyze trajectory buffer for behavioral intent."""
        buf = self._buffers.get(entity_id)
        if not buf or len(buf.positions) < 10:
            return IntentResult(entity_id=entity_id, intent=IntentType.NORMAL,
                                confidence=0.0, description="Insufficient data")

        positions = np.array(list(buf.positions))
        timestamps = np.array(list(buf.timestamps))

        # Check each pattern
        results = []

        casing = self._detect_casing(entity_id, positions, timestamps, buf.head_angles)
        if casing:
            results.append(casing)

        pacing = self._detect_pacing(entity_id, positions, timestamps)
        if pacing:
            results.append(pacing)

        evasive = self._detect_evasive(entity_id, positions, timestamps)
        if evasive:
            results.append(evasive)

        if target_entities:
            aggressive = self._detect_aggressive(entity_id, positions, timestamps, target_entities)
            if aggressive:
                results.append(aggressive)

        if not results:
            return IntentResult(entity_id=entity_id, intent=IntentType.NORMAL,
                                confidence=0.1, description="Normal movement pattern")

        # Return highest confidence
        return max(results, key=lambda r: r.confidence)

    def _detect_casing(self, entity_id: str, positions: np.ndarray,
                       timestamps: np.ndarray, head_angles: deque) -> IntentResult | None:
        """Casing: stationary + head scanning."""
        # Check if mostly stationary in recent window
        recent = positions[-30:]  # last 30 samples
        if len(recent) < 10:
            return None

        displacement = np.linalg.norm(recent[-1] - recent[0])
        duration = timestamps[-1] - timestamps[-min(30, len(timestamps))]

        if displacement > 2.0 or duration < self.CASING_MIN_STATIONARY_S:
            return None

        # Check head scanning
        if len(head_angles) < 5:
            return None
        angles = np.array(list(head_angles))
        scan_range = float(angles.max() - angles.min())

        if scan_range < self.CASING_HEAD_SCAN_RANGE:
            return None

        confidence = min(0.9, scan_range / 120.0 * 0.5 + (duration / 30.0) * 0.5)
        return IntentResult(
            entity_id=entity_id, intent=IntentType.CASING, confidence=confidence,
            description=f"Stationary for {duration:.0f}s with {scan_range:.0f}° head scanning",
            evidence={"duration_s": duration, "head_scan_degrees": scan_range},
        )

    def _detect_pacing(self, entity_id: str, positions: np.ndarray,
                       timestamps: np.ndarray) -> IntentResult | None:
        """Nervous pacing: back-and-forth in small area."""
        if len(positions) < 20:
            return None

        recent = positions[-60:]
        # Compute bounding area
        pos_2d = recent[:, [0, 2]] if recent.shape[1] >= 3 else recent[:, :2]
        area = float((pos_2d.max(axis=0) - pos_2d.min(axis=0)).prod())

        if area > self.PACING_MAX_AREA_M2:
            return None

        # Count direction reversals
        if len(pos_2d) < 3:
            return None
        deltas = np.diff(pos_2d, axis=0)
        dot_products = np.sum(deltas[:-1] * deltas[1:], axis=1)
        reversals = int(np.sum(dot_products < 0))

        if reversals < self.PACING_MIN_REVERSALS:
            return None

        confidence = min(0.9, reversals / 10.0)
        return IntentResult(
            entity_id=entity_id, intent=IntentType.NERVOUS_PACING, confidence=confidence,
            description=f"{reversals} direction reversals in {area:.1f}m² area",
            evidence={"reversals": reversals, "area_m2": area},
        )

    def _detect_evasive(self, entity_id: str, positions: np.ndarray,
                        timestamps: np.ndarray) -> IntentResult | None:
        """Evasive movement: sharp turns + speed variation."""
        if len(positions) < 15:
            return None

        pos_2d = positions[:, [0, 2]] if positions.shape[1] >= 3 else positions[:, :2]
        deltas = np.diff(pos_2d, axis=0)
        dt = np.diff(timestamps)
        dt = np.where(dt > 0, dt, 0.033)

        speeds = np.linalg.norm(deltas, axis=1) / dt
        if len(speeds) < 3:
            return None

        speed_var = float(np.std(speeds) / (np.mean(speeds) + 1e-6))

        # Count sharp turns (> 60 degrees)
        if len(deltas) < 2:
            return None
        norms = np.linalg.norm(deltas, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1e-6)
        unit = deltas / norms
        cos_angles = np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1, 1)
        angles = np.degrees(np.arccos(cos_angles))
        sharp_turns = int(np.sum(angles > 60))

        if sharp_turns < self.EVASIVE_MIN_TURNS or speed_var < self.EVASIVE_SPEED_VARIANCE:
            return None

        confidence = min(0.85, (sharp_turns / 5.0) * 0.5 + min(speed_var, 2.0) / 2.0 * 0.5)
        return IntentResult(
            entity_id=entity_id, intent=IntentType.EVASIVE_MOVEMENT, confidence=confidence,
            description=f"{sharp_turns} sharp turns, speed variance={speed_var:.2f}",
            evidence={"sharp_turns": sharp_turns, "speed_variance": speed_var},
        )

    def _detect_aggressive(self, entity_id: str, positions: np.ndarray,
                           timestamps: np.ndarray,
                           targets: dict[str, np.ndarray]) -> IntentResult | None:
        """Aggressive approach: rapid direct movement toward another person."""
        if len(positions) < 5:
            return None

        recent = positions[-10:]
        dt = timestamps[-1] - timestamps[-min(10, len(timestamps))]
        if dt <= 0:
            return None

        displacement = recent[-1] - recent[0]
        speed = float(np.linalg.norm(displacement) / dt)

        if speed < self.AGGRESSIVE_MIN_SPEED:
            return None

        # Check directness toward any target
        path_length = float(np.sum(np.linalg.norm(np.diff(recent, axis=0), axis=1)))
        if path_length < 1e-3:
            return None
        directness = float(np.linalg.norm(displacement)) / path_length

        if directness < self.AGGRESSIVE_DIRECTNESS:
            return None

        # Check if heading toward a target
        direction = displacement / (np.linalg.norm(displacement) + 1e-6)
        for target_id, target_pos in targets.items():
            to_target = target_pos - recent[-1]
            to_target_norm = to_target / (np.linalg.norm(to_target) + 1e-6)
            alignment = float(np.dot(direction, to_target_norm))
            if alignment > 0.7:
                confidence = min(0.9, speed / 4.0 * 0.5 + alignment * 0.5)
                return IntentResult(
                    entity_id=entity_id, intent=IntentType.AGGRESSIVE_APPROACH,
                    confidence=confidence,
                    description=f"Rapid approach toward {target_id} at {speed:.1f}m/s, directness={directness:.2f}",
                    evidence={"speed": speed, "directness": directness, "target": target_id},
                )
        return None

    def _estimate_head_angle(self, keypoints: list) -> float | None:
        """Estimate head yaw angle from pose keypoints (nose, left_ear, right_ear)."""
        # COCO keypoint indices: 0=nose, 3=left_ear, 4=right_ear
        try:
            if len(keypoints) < 5:
                return None
            nose = keypoints[0] if isinstance(keypoints[0], (list, tuple)) else None
            l_ear = keypoints[3] if isinstance(keypoints[3], (list, tuple)) else None
            r_ear = keypoints[4] if isinstance(keypoints[4], (list, tuple)) else None

            if not (nose and l_ear and r_ear):
                return None
            if nose[2] < 0.3 or l_ear[2] < 0.3 or r_ear[2] < 0.3:  # low confidence
                return None

            ear_mid_x = (l_ear[0] + r_ear[0]) / 2
            return float(math.degrees(math.atan2(nose[0] - ear_mid_x, abs(l_ear[0] - r_ear[0]) + 1e-6)))
        except (IndexError, TypeError):
            return None

    def analyze_with_vlm(self, entity_id: str, keyframe_b64: str = "") -> IntentResult | None:
        """Optional deep VLM analysis for ambiguous cases."""
        if not self._vlm_client or not keyframe_b64:
            return None

        client = self._vlm_client._get_client() if hasattr(self._vlm_client, '_get_client') else None
        if not client or client == "stub":
            return None

        try:
            content = [
                "Analyze this person's behavior and body language. "
                "Is there any indication of: casing/surveillance, nervous pacing, "
                "evasive movement, or aggressive approach? "
                "Reply JSON: {\"intent\": \"...\", \"confidence\": 0.0-1.0, \"description\": \"...\"}",
                {"mime_type": "image/jpeg", "data": keyframe_b64},
            ]
            resp = client.generate_content(content)
            text = resp.text.strip().strip("`").lstrip("json\n")
            data = json.loads(text)
            intent = IntentType(data.get("intent", "normal"))
            return IntentResult(
                entity_id=entity_id, intent=intent,
                confidence=data.get("confidence", 0.5),
                description=data.get("description", "VLM analysis"),
            )
        except Exception as e:
            logger.debug("VLM intent analysis failed: %s", e)
            return None

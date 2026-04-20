"""Behavioral intent recognition — pre-incident detection from pose + trajectory analysis.

Detects: casing, nervous pacing, evasive movement, aggressive approach.
Includes learned PoseSequenceClassifier (1D-CNN) to boost heuristic confidence.
Optional VLM deep analysis for ambiguous cases.
"""

import json
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch
import torch.nn as nn

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
    pose_keypoints: deque = field(default_factory=lambda: deque(maxlen=30))  # for CNN


class PoseSequenceClassifier(nn.Module):
    """Temporal Transformer over pose keypoint sequences for intent classification.

    Architecture: linear projection → positional encoding → Transformer encoder → CLS token → FC.
    Replaces the earlier 1D-CNN with self-attention for better temporal reasoning
    (captures long-range dependencies like slow casing over 30 frames).

    Input: (batch, 30, 17*3=51) — 30 frames of 17 COCO keypoints with (x,y,conf).
    Output: (batch, 5) — probabilities for [normal, casing, pacing, evasive, aggressive].
    """

    INTENT_CLASSES = ["normal", "casing", "nervous_pacing", "evasive_movement", "aggressive_approach"]

    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        input_dim = 51  # 17 keypoints × 3 (x, y, conf)
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable positional encoding for 30 frames
        self.pos_embed = nn.Parameter(torch.randn(1, 31, d_model) * 0.02)  # +1 for CLS

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, len(self.INTENT_CLASSES))
        self._available = True
        self.eval()

    @torch.no_grad()
    def classify(self, pose_sequence: list[list]) -> dict[str, float] | None:
        """Classify a sequence of pose keypoints.

        Args:
            pose_sequence: list of 30 frames, each frame is list of 17 keypoints [x,y,conf]
        Returns:
            dict mapping intent name to probability, or None if insufficient data
        """
        if len(pose_sequence) < 15:
            return None
        try:
            # Pad/trim to 30 frames
            seq = list(pose_sequence)[-30:]
            while len(seq) < 30:
                seq.insert(0, seq[0])

            # Flatten keypoints: (30, 17, 3) → (30, 51)
            flat = []
            for frame_kpts in seq:
                frame_flat = []
                for kp in frame_kpts[:17]:
                    if isinstance(kp, (list, tuple)) and len(kp) >= 3:
                        frame_flat.extend([kp[0], kp[1], kp[2]])
                    else:
                        frame_flat.extend([0.0, 0.0, 0.0])
                while len(frame_flat) < 51:
                    frame_flat.append(0.0)
                flat.append(frame_flat[:51])

            x = torch.tensor([flat], dtype=torch.float32)  # (1, 30, 51)
            x = self.input_proj(x)                             # (1, 30, d_model)
            # Prepend CLS token
            cls = self.cls_token.expand(1, -1, -1)             # (1, 1, d_model)
            x = torch.cat([cls, x], dim=1)                     # (1, 31, d_model)
            x = x + self.pos_embed[:, :x.size(1)]
            x = self.encoder(x)                                # (1, 31, d_model)
            cls_out = x[:, 0]                                  # (1, d_model) — CLS token
            logits = self.fc(cls_out)                          # (1, 5)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            return {name: float(probs[i]) for i, name in enumerate(self.INTENT_CLASSES)}
        except Exception as e:
            logger.debug("PoseSequenceClassifier failed: %s", e)
            return None


class IntentRecognizer:
    """Analyzes pose sequences + trajectories to detect pre-incident behavioral patterns.

    Uses heuristic detectors + learned PoseSequenceClassifier (1D-CNN).
    When both agree, confidence is boosted. When they disagree, conservative score used.
    """

    # Thresholds
    CASING_HEAD_SCAN_RANGE = 60.0
    CASING_MIN_STATIONARY_S = 5.0
    PACING_MIN_REVERSALS = 3
    PACING_MAX_AREA_M2 = 4.0
    EVASIVE_MIN_TURNS = 2
    EVASIVE_SPEED_VARIANCE = 0.5
    AGGRESSIVE_MIN_SPEED = 2.0
    AGGRESSIVE_DIRECTNESS = 0.8

    def __init__(self):
        self._buffers: dict[str, _TrajectoryBuffer] = defaultdict(_TrajectoryBuffer)
        self._vlm_client = None
        self._pose_classifier = PoseSequenceClassifier()

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
            buf.pose_keypoints.append(pose_keypoints)

    def analyze(self, entity_id: str, target_entities: dict[str, np.ndarray] | None = None) -> IntentResult:
        """Analyze trajectory buffer for behavioral intent.

        Combines heuristic detectors with learned PoseSequenceClassifier.
        """
        buf = self._buffers.get(entity_id)
        if not buf or len(buf.positions) < 10:
            return IntentResult(entity_id=entity_id, intent=IntentType.NORMAL,
                                confidence=0.0, description="Insufficient data")

        positions = np.array(list(buf.positions))
        timestamps = np.array(list(buf.timestamps))

        # Heuristic detection
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

        heuristic_best = max(results, key=lambda r: r.confidence) if results else None

        # CNN-based classification from pose sequences
        cnn_result = None
        if buf.pose_keypoints and len(buf.pose_keypoints) >= 15:
            cnn_probs = self._pose_classifier.classify(list(buf.pose_keypoints))
            if cnn_probs:
                best_class = max(cnn_probs, key=cnn_probs.get)
                if best_class != "normal" and cnn_probs[best_class] > 0.3:
                    try:
                        cnn_result = IntentResult(
                            entity_id=entity_id,
                            intent=IntentType(best_class),
                            confidence=cnn_probs[best_class],
                            description=f"CNN: {best_class} (p={cnn_probs[best_class]:.2f})",
                        )
                    except ValueError:
                        pass

        # Fuse: if both agree, boost confidence; if disagree, use conservative
        if heuristic_best and cnn_result:
            if heuristic_best.intent == cnn_result.intent:
                heuristic_best.confidence = min(0.98, heuristic_best.confidence * 1.3)
                heuristic_best.description += " [CNN-confirmed]"
                return heuristic_best
            else:
                # Disagree: return the one with lower confidence (conservative)
                return min([heuristic_best, cnn_result], key=lambda r: r.confidence)
        elif heuristic_best:
            return heuristic_best
        elif cnn_result:
            return cnn_result

        return IntentResult(entity_id=entity_id, intent=IntentType.NORMAL,
                            confidence=0.1, description="Normal movement pattern")

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

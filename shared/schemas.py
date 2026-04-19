"""Shared event schemas used across edge and cloud components."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class PoseKeypoint:
    x: float
    y: float
    confidence: float


@dataclass
class DetectedObject:
    class_name: str
    bbox: BBox
    track_id: int
    confidence: float
    pose: Optional[list[PoseKeypoint]] = None


@dataclass
class AudioEvent:
    class_name: str
    confidence: float
    timestamp: float


@dataclass
class VisionEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    camera_id: str = ""
    event_type: str = "detection"
    objects: list[DetectedObject] = field(default_factory=list)
    audio_events: list[AudioEvent] = field(default_factory=list)
    keyframe_b64: Optional[str] = None
    privacy_applied: bool = False
    scene_activity: float = 0.0  # 0-1 normalized activity level
    frame_index: int = 0


@dataclass
class AnomalyEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    camera_id: str = ""
    severity: Severity = Severity.LOW
    anomaly_type: str = ""
    description: str = ""
    causal_explanation: str = ""
    evidence_keyframes: list[str] = field(default_factory=list)
    evidence_graph_refs: list[str] = field(default_factory=list)
    confidence: float = 0.0
    rule_id: Optional[str] = None
    recommended_action: str = ""

"""Lightweight VLM on edge device — sub-second scene descriptions without cloud.

Loads small model (InternVL-2B or similar). Offline VLM reasoning.
EdgeVLMScheduler runs checks on interval or when anomaly score spikes.
"""

import base64
import io
import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SceneDescription:
    timestamp: float
    camera_id: str
    description: str
    objects_mentioned: list[str] = field(default_factory=list)
    anomaly_hints: list[str] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: float = 0.0


class EdgeVLM:
    """Lightweight VLM for edge inference — loads small model with lazy init."""

    def __init__(self, model_name: str = "OpenGVLab/InternVL2-2B"):
        self._model_name = model_name
        self._model = None
        self._tokenizer = None
        self._available = None

    def _load(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            from transformers import AutoModel, AutoTokenizer
            logger.info("Loading edge VLM: %s", self._model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, trust_remote_code=True)
            self._model = AutoModel.from_pretrained(
                self._model_name, trust_remote_code=True).eval()
            self._available = True
            logger.info("Edge VLM loaded successfully")
        except Exception as e:
            logger.warning("Edge VLM unavailable: %s — using stub", e)
            self._available = False
        return self._available

    def describe_scene(self, frame: np.ndarray, camera_id: str = "",
                       prompt: str = "Describe what you see in this CCTV frame. "
                                     "Note any unusual activity.") -> SceneDescription:
        t0 = time.time()

        if not self._load():
            return self._stub_description(frame, camera_id, t0)

        try:
            from PIL import Image
            img = Image.fromarray(frame if frame.dtype == np.uint8 else
                                  (frame * 255).astype(np.uint8))

            # Model-specific inference
            if hasattr(self._model, 'chat'):
                response = self._model.chat(self._tokenizer, img, prompt)
            else:
                # Generic transformers pipeline fallback
                inputs = self._tokenizer(prompt, return_tensors="pt")
                outputs = self._model.generate(**inputs, max_new_tokens=150)
                response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

            latency = (time.time() - t0) * 1000
            return SceneDescription(
                timestamp=time.time(), camera_id=camera_id,
                description=response, confidence=0.7, latency_ms=latency,
                objects_mentioned=self._extract_objects(response),
                anomaly_hints=self._extract_anomalies(response),
            )
        except Exception as e:
            logger.error("Edge VLM inference failed: %s", e)
            return self._stub_description(frame, camera_id, t0)

    def _stub_description(self, frame: np.ndarray, camera_id: str, t0: float) -> SceneDescription:
        return SceneDescription(
            timestamp=time.time(), camera_id=camera_id,
            description="[Edge VLM unavailable] Scene captured for cloud analysis.",
            confidence=0.1, latency_ms=(time.time() - t0) * 1000,
        )

    @staticmethod
    def _extract_objects(text: str) -> list[str]:
        keywords = ["person", "people", "vehicle", "car", "truck", "bicycle",
                     "bag", "backpack", "suitcase", "dog", "cat"]
        return [k for k in keywords if k in text.lower()]

    @staticmethod
    def _extract_anomalies(text: str) -> list[str]:
        hints = ["unusual", "suspicious", "running", "fighting", "fallen",
                 "abandoned", "loitering", "crowd", "smoke", "fire"]
        return [h for h in hints if h in text.lower()]


class EdgeVLMScheduler:
    """Schedules VLM checks on interval or when anomaly score spikes."""

    def __init__(self, vlm: EdgeVLM, interval_s: float = 30.0,
                 anomaly_threshold: float = 0.7):
        self._vlm = vlm
        self._interval = interval_s
        self._anomaly_threshold = anomaly_threshold
        self._last_check: float = 0.0
        self._results: list[SceneDescription] = []
        self._running = False
        self._lock = threading.Lock()

    def should_check(self, anomaly_score: float = 0.0) -> bool:
        """Determine if a VLM check should run now."""
        now = time.time()
        if anomaly_score >= self._anomaly_threshold:
            return True
        return (now - self._last_check) >= self._interval

    def check(self, frame: np.ndarray, camera_id: str = "",
              anomaly_score: float = 0.0) -> SceneDescription | None:
        """Run VLM check if conditions are met."""
        if not self.should_check(anomaly_score):
            return None

        self._last_check = time.time()
        result = self._vlm.describe_scene(frame, camera_id)

        with self._lock:
            self._results.append(result)
            if len(self._results) > 100:
                self._results = self._results[-50:]

        logger.info("Edge VLM check: camera=%s latency=%.0fms anomaly_hints=%s",
                    camera_id, result.latency_ms, result.anomaly_hints)
        return result

    def get_recent_descriptions(self, limit: int = 10) -> list[SceneDescription]:
        with self._lock:
            return list(self._results[-limit:])

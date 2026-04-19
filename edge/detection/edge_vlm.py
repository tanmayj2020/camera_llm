"""Lightweight VLM on edge device — sub-second scene descriptions without cloud.

Priority: Qwen2.5-VL-3B (SOTA small VLM, 2025) → InternVL3-2B → InternVL2-2B fallback.
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
    """Lightweight VLM for edge inference — Qwen2.5-VL-3B (SOTA small VLM).

    Qwen2.5-VL-3B outperforms InternVL2-2B on scene understanding benchmarks
    while supporting dynamic resolution and video frame understanding.
    """

    # Ordered by preference
    _MODEL_CHAIN = [
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "OpenGVLab/InternVL3-2B",
        "OpenGVLab/InternVL2-2B",
    ]

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name
        self._model = None
        self._processor = None
        self._available = None
        self._backend = "none"  # "qwen2vl" | "internvl" | "none"

    def _load(self) -> bool:
        if self._available is not None:
            return self._available

        models_to_try = [self._model_name] if self._model_name else self._MODEL_CHAIN

        for model_id in models_to_try:
            if model_id is None:
                continue
            try:
                if "qwen2" in model_id.lower() or "qwen2.5" in model_id.lower():
                    self._load_qwen2vl(model_id)
                else:
                    self._load_internvl(model_id)
                self._available = True
                logger.info("Edge VLM loaded: %s (backend=%s)", model_id, self._backend)
                return True
            except Exception as e:
                logger.info("Edge VLM %s failed: %s", model_id, e)
                continue

        logger.warning("No edge VLM available — using stub")
        self._available = False
        return False

    def _load_qwen2vl(self, model_id: str):
        """Load Qwen2.5-VL with proper processor."""
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True
        ).eval()
        self._model_name = model_id
        self._backend = "qwen2vl"

    def _load_internvl(self, model_id: str):
        """Load InternVL with AutoModel."""
        from transformers import AutoModel, AutoTokenizer
        self._processor = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True
        ).eval()
        self._model_name = model_id
        self._backend = "internvl"

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

            if self._backend == "qwen2vl":
                response = self._infer_qwen2vl(img, prompt)
            elif self._backend == "internvl":
                response = self._infer_internvl(img, prompt)
            else:
                return self._stub_description(frame, camera_id, t0)

            latency = (time.time() - t0) * 1000
            return SceneDescription(
                timestamp=time.time(), camera_id=camera_id,
                description=response, confidence=0.8, latency_ms=latency,
                objects_mentioned=self._extract_objects(response),
                anomaly_hints=self._extract_anomalies(response),
            )
        except Exception as e:
            logger.error("Edge VLM inference failed: %s", e)
            return self._stub_description(frame, camera_id, t0)

    def _infer_qwen2vl(self, img, prompt: str) -> str:
        """Qwen2.5-VL inference with proper message format."""
        import torch
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[img], return_tensors="pt", padding=True)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=200)
        # Decode only generated tokens
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._processor.decode(generated, skip_special_tokens=True)

    def _infer_internvl(self, img, prompt: str) -> str:
        """InternVL inference."""
        if hasattr(self._model, 'chat'):
            return self._model.chat(self._processor, img, prompt)
        import torch
        inputs = self._processor(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=200)
        return self._processor.decode(outputs[0], skip_special_tokens=True)

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

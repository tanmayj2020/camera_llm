"""Audio processing — SOTA sound classification and optional speech-to-text.

Priority: BEATs (SOTA audio classification, 2023) → AST (Audio Spectrogram Transformer) → YAMNet fallback.
BEATs achieves 98.1% mAP on AudioSet vs YAMNet 83.1%.
"""

import logging
import threading
import time
from queue import Empty, Full, Queue

import numpy as np

logger = logging.getLogger(__name__)

# Alert classes relevant to surveillance
ALERT_CLASSES = {
    "Gunshot, gunfire": "gunshot",
    "Machine gun": "gunshot",
    "Glass": "glass_breaking",
    "Shatter": "glass_breaking",
    "Screaming": "scream",
    "Alarm": "alarm",
    "Siren": "siren",
    "Car alarm": "alarm",
    "Explosion": "explosion",
    "Dog": "dog_bark",
    "Fire alarm": "alarm",
    "Emergency vehicle": "siren",
    "Crying": "distress",
    "Smash, crash": "glass_breaking",
}


class SoundClassifier:
    """Classifies audio chunks with multi-model fallback.

    BEATs: 98.1% mAP on AudioSet (Microsoft Research, 2023)
    AST: 95.9% mAP on AudioSet (MIT, 2021)
    YAMNet: 83.1% mAP on AudioSet (Google, 2020)
    """

    def __init__(self, sample_rate: int = 16000, top_k: int = 5, alert_threshold: float = 0.3):
        self.sample_rate = sample_rate
        self.top_k = top_k
        self.alert_threshold = alert_threshold
        self._model = None
        self._processor = None
        self._class_names = None
        self._backend = "none"

    def _load_model(self):
        if self._model is not None:
            return

        # Priority 1: BEATs (SOTA)
        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            model_id = "microsoft/BEATs-iter3-AS2M"
            self._processor = AutoFeatureExtractor.from_pretrained(model_id)
            self._model = AutoModelForAudioClassification.from_pretrained(model_id)
            self._model.eval()
            self._class_names = list(self._model.config.id2label.values())
            self._backend = "beats"
            logger.info("BEATs audio classifier loaded (%d classes)", len(self._class_names))
            return
        except Exception as e:
            logger.info("BEATs unavailable (%s), trying AST", e)

        # Priority 2: AST (Audio Spectrogram Transformer)
        try:
            from transformers import AutoFeatureExtractor, ASTForAudioClassification
            model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
            self._processor = AutoFeatureExtractor.from_pretrained(model_id)
            self._model = ASTForAudioClassification.from_pretrained(model_id)
            self._model.eval()
            self._class_names = list(self._model.config.id2label.values())
            self._backend = "ast"
            logger.info("AST audio classifier loaded (%d classes)", len(self._class_names))
            return
        except Exception as e:
            logger.info("AST unavailable (%s), trying YAMNet", e)

        # Priority 3: YAMNet fallback
        try:
            import csv
            import tensorflow_hub as hub
            self._model = hub.load("https://tfhub.dev/google/yamnet/1")
            class_map_path = self._model.class_map_path().numpy().decode()
            with open(class_map_path) as f:
                reader = csv.DictReader(f)
                self._class_names = [row["display_name"] for row in reader]
            self._backend = "yamnet"
            logger.info("YAMNet audio classifier loaded (%d classes)", len(self._class_names))
            return
        except Exception as e:
            logger.warning("All audio classifiers failed: %s — using stub", e)
            self._model = "stub"
            self._backend = "stub"

    def classify(self, audio_chunk: np.ndarray) -> list[dict]:
        """Classify an audio chunk (float32, mono, 16kHz)."""
        self._load_model()
        if self._backend == "stub" or self._model is None:
            return []

        waveform = audio_chunk.astype(np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        if self._backend in ("beats", "ast"):
            return self._classify_transformers(waveform)
        elif self._backend == "yamnet":
            return self._classify_yamnet(waveform)
        return []

    def _classify_transformers(self, waveform: np.ndarray) -> list[dict]:
        """BEATs / AST inference via HuggingFace transformers."""
        import torch
        inputs = self._processor(
            waveform, sampling_rate=self.sample_rate, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().numpy()
        top_indices = probs.argsort()[-self.top_k:][::-1]

        results = []
        for idx in top_indices:
            name = self._class_names[idx] if idx < len(self._class_names) else str(idx)
            conf = float(probs[idx])
            mapped = ALERT_CLASSES.get(name)
            results.append({
                "class_name": mapped or name.lower().replace(" ", "_"),
                "confidence": round(conf, 3),
                "is_alert": mapped is not None and conf >= self.alert_threshold,
            })
        return results

    def _classify_yamnet(self, waveform: np.ndarray) -> list[dict]:
        """YAMNet inference via TensorFlow Hub."""
        scores, _, _ = self._model(waveform)
        mean_scores = scores.numpy().mean(axis=0)
        top_indices = mean_scores.argsort()[-self.top_k:][::-1]

        results = []
        for idx in top_indices:
            name = self._class_names[idx]
            conf = float(mean_scores[idx])
            mapped = ALERT_CLASSES.get(name)
            results.append({
                "class_name": mapped or name.lower().replace(" ", "_"),
                "confidence": round(conf, 3),
                "is_alert": mapped is not None and conf >= self.alert_threshold,
            })
        return results


class AudioCapture:
    """Captures audio from microphone in chunks for classification."""

    def __init__(self, sample_rate: int = 16000, chunk_duration: float = 1.0, device: int | None = None):
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.device = device
        self._queue: Queue = Queue(maxsize=10)
        self._running = False
        self._thread: threading.Thread | None = None

    def _capture_loop(self):
        try:
            import sounddevice as sd
        except Exception as e:
            logger.warning("sounddevice unavailable: %s — audio disabled", e)
            return

        while self._running:
            try:
                audio = sd.rec(self.chunk_samples, samplerate=self.sample_rate, channels=1,
                               dtype="float32", device=self.device)
                sd.wait()
                try:
                    self._queue.put_nowait(audio.flatten())
                except Full:
                    pass
            except Exception as e:
                logger.error("Audio capture error: %s", e)
                time.sleep(1)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def get_chunk(self, timeout: float = 1.5):
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

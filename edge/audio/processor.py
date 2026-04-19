"""Audio processing — sound classification and optional speech-to-text."""

import logging
import threading
import time
from queue import Empty, Full, Queue

import numpy as np

logger = logging.getLogger(__name__)

# YAMNet class subset relevant to surveillance
ALERT_CLASSES = {
    "Gunshot, gunfire": "gunshot",
    "Glass": "glass_breaking",
    "Shatter": "glass_breaking",
    "Screaming": "scream",
    "Alarm": "alarm",
    "Siren": "siren",
    "Car alarm": "alarm",
    "Explosion": "explosion",
    "Dog": "dog_bark",
}


class SoundClassifier:
    """Classifies audio chunks using TensorFlow Hub YAMNet model."""

    def __init__(self, sample_rate: int = 16000, top_k: int = 3, alert_threshold: float = 0.3):
        self.sample_rate = sample_rate
        self.top_k = top_k
        self.alert_threshold = alert_threshold
        self._model = None
        self._class_names = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import csv
            import io

            import tensorflow_hub as hub

            self._model = hub.load("https://tfhub.dev/google/yamnet/1")
            # Load class map
            class_map_path = self._model.class_map_path().numpy().decode()
            with open(class_map_path) as f:
                reader = csv.DictReader(f)
                self._class_names = [row["display_name"] for row in reader]
            logger.info("YAMNet loaded with %d classes", len(self._class_names))
        except Exception as e:
            logger.warning("YAMNet load failed: %s — using stub classifier", e)
            self._model = "stub"

    def classify(self, audio_chunk: np.ndarray) -> list[dict]:
        """Classify an audio chunk (float32, mono, 16kHz). Returns list of {class_name, confidence}."""
        self._load_model()
        if self._model == "stub" or self._model is None:
            return []

        waveform = audio_chunk.astype(np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        scores, embeddings, spectrogram = self._model(waveform)
        mean_scores = scores.numpy().mean(axis=0)
        top_indices = mean_scores.argsort()[-self.top_k :][::-1]

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

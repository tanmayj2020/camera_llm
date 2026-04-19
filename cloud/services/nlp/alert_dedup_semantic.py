"""Semantic alert deduplication for VisionBrain CCTV Analytics Platform."""

import logging
import math
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

_HASH_VEC_SIZE = 64


class SemanticDeduplicator:
    def __init__(self, similarity_threshold: float = 0.85, window_size: int = 200) -> None:
        self._threshold = similarity_threshold
        self._recent: deque[tuple[dict, list[float]]] = deque(maxlen=window_size)
        self._model: Optional[object] = None
        self._total_seen = 0
        self._duplicates_suppressed = 0

    def _get_model(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning("sentence-transformers unavailable, using stub: %s", e)
            self._model = "stub"
        return self._model

    def _embed(self, text: str) -> list[float]:
        model = self._get_model()
        if model == "stub":
            vec = [0.0] * _HASH_VEC_SIZE
            for w in text.lower().split():
                vec[hash(w) % _HASH_VEC_SIZE] = 1.0
            return vec
        return model.encode(text).tolist()

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0

    def is_duplicate(self, alert: dict) -> bool:
        self._total_seen += 1
        desc = alert.get("description", "")
        atype = alert.get("anomaly_type", "")
        text = f"{atype} {desc}".strip() if atype else desc
        emb = self._embed(text)
        cam = alert.get("camera_id")
        site = alert.get("site_id")
        ts = alert.get("timestamp", 0)

        for prev_alert, prev_emb in self._recent:
            if self._cosine_similarity(emb, prev_emb) <= self._threshold:
                continue
            same_loc = cam == prev_alert.get("camera_id") or site == prev_alert.get("site_id")
            if same_loc and abs(ts - prev_alert.get("timestamp", 0)) <= 300:
                self._duplicates_suppressed += 1
                return True

        self._recent.append((alert, emb))
        return False

    def deduplicate_batch(self, alerts: list[dict]) -> list[dict]:
        return [a for a in alerts if not self.is_duplicate(a)]

    def get_stats(self) -> dict:
        return {
            "total_seen": self._total_seen,
            "duplicates_suppressed": self._duplicates_suppressed,
            "window_size": self._recent.maxlen,
        }

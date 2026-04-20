"""Semantic alert deduplication — SOTA sentence embeddings.

Embedding priority: GTE-Qwen2 (MTEB #1) → NV-Embed-v2 → BGE-M3 → n-gram stub.
"""

import logging
import math
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

_HASH_VEC_SIZE = 256


class SemanticDeduplicator:
    def __init__(self, similarity_threshold: float = 0.85, window_size: int = 200) -> None:
        self._threshold = similarity_threshold
        self._recent: deque[tuple[dict, list[float]]] = deque(maxlen=window_size)
        self._model: Optional[object] = None
        self._total_seen = 0
        self._duplicates_suppressed = 0
        self._backend = "none"

    def _get_model(self) -> object:
        if self._model is not None:
            return self._model

        # Priority 1: GTE-Qwen2 (Alibaba, 2024 — MTEB leaderboard #1)
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                                               trust_remote_code=True)
            self._backend = "gte-qwen2"
            logger.info("GTE-Qwen2 sentence embeddings loaded (MTEB SOTA)")
            return self._model
        except Exception as e:
            logger.info("GTE-Qwen2 unavailable (%s), trying NV-Embed", e)

        # Priority 2: NV-Embed-v2 (NVIDIA, 2024)
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("nvidia/NV-Embed-v2", trust_remote_code=True)
            self._backend = "nv-embed"
            logger.info("NV-Embed-v2 sentence embeddings loaded")
            return self._model
        except Exception as e:
            logger.info("NV-Embed unavailable (%s), trying MiniLM", e)

        # Priority 3: BGE-M3 (BAAI, 2024-2026 — multi-lingual, MTEB top-10)
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("BAAI/bge-m3")
            self._backend = "bge-m3"
            logger.info("BGE-M3 sentence embeddings loaded (fallback)")
            return self._model
        except Exception as e:
            logger.warning("sentence-transformers unavailable, using n-gram stub: %s", e)
            self._model = "stub"
            self._backend = "stub"
        return self._model

    def _embed(self, text: str) -> list[float]:
        model = self._get_model()
        if model == "stub":
            return self._ngram_embed(text)
        return model.encode(text).tolist()

    @staticmethod
    def _ngram_embed(text: str, n: int = 3) -> list[float]:
        """TF-IDF style character n-gram hashing — much better than single-word hashing."""
        vec = [0.0] * _HASH_VEC_SIZE
        text = text.lower().strip()
        ngrams = [text[i:i + n] for i in range(max(1, len(text) - n + 1))]
        if not ngrams:
            return vec
        weight = 1.0 / len(ngrams)
        for ng in ngrams:
            idx = hash(ng) % _HASH_VEC_SIZE
            vec[idx] += weight
        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    @staticmethod
    def _jaccard_similarity(a: str, b: str) -> float:
        """Word-level Jaccard as secondary check."""
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

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
            cosine_sim = self._cosine_similarity(emb, prev_emb)
            if cosine_sim <= self._threshold:
                # Secondary check: Jaccard on text for stub mode
                prev_text = f"{prev_alert.get('anomaly_type', '')} {prev_alert.get('description', '')}".strip()
                if self._jaccard_similarity(text, prev_text) < 0.6:
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

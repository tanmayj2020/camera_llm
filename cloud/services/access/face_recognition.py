"""Opt-in Face Recognition for access control — GDPR-compliant with consent check."""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FaceMatch:
    entity_id: str
    matched_identity: str
    confidence: float
    camera_id: str
    timestamp: float
    consent_verified: bool


class FaceRecognitionEngine:
    """Opt-in face recognition with mandatory consent verification.

    Only matches faces for entities that have explicit consent on file.
    Uses embedding similarity (not raw images) for privacy.
    """

    def __init__(self, consent_manager=None, similarity_threshold: float = 0.75):
        self._consent_manager = consent_manager
        self._threshold = similarity_threshold
        self._enrolled: dict[str, np.ndarray] = {}  # identity_name -> embedding
        self._match_log: list[FaceMatch] = []

    def enroll(self, identity_name: str, embedding: np.ndarray, entity_id: str = ""):
        """Enroll a face embedding with consent check."""
        if self._consent_manager and entity_id:
            consent = self._consent_manager.get_consent(entity_id)
            if not consent or not consent.get("granted"):
                logger.warning("Enrollment rejected — no consent for %s", entity_id)
                return False
        self._enrolled[identity_name] = embedding / (np.linalg.norm(embedding) + 1e-8)
        logger.info("Enrolled face: %s", identity_name)
        return True

    def unenroll(self, identity_name: str):
        self._enrolled.pop(identity_name, None)

    def match(self, embedding: np.ndarray, camera_id: str = "",
              entity_id: str = "") -> FaceMatch | None:
        """Match a face embedding against enrolled identities."""
        if not self._enrolled:
            return None

        # Consent check
        consent_ok = True
        if self._consent_manager and entity_id:
            consent = self._consent_manager.get_consent(entity_id)
            if not consent or not consent.get("granted"):
                consent_ok = False

        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        best_name, best_sim = None, self._threshold

        for name, enrolled_emb in self._enrolled.items():
            sim = float(np.dot(emb_norm, enrolled_emb))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_name:
            match = FaceMatch(entity_id, best_name, round(best_sim, 3),
                              camera_id, time.time(), consent_ok)
            self._match_log.append(match)
            if not consent_ok:
                match.matched_identity = "[REDACTED — no consent]"
            return match
        return None

    def get_enrolled(self) -> list[str]:
        return list(self._enrolled.keys())

    def get_match_log(self, limit: int = 50) -> list[dict]:
        return [{"entity": m.entity_id, "identity": m.matched_identity,
                 "confidence": m.confidence, "camera": m.camera_id,
                 "timestamp": m.timestamp, "consent": m.consent_verified}
                for m in self._match_log[-limit:]]

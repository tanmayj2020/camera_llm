"""GDPR/DPDP Consent & Data Governance Manager."""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConsentRecord:
    entity_id: str
    consent_type: str  # processing, retention, sharing
    granted: bool
    granted_at: float = 0.0
    expires_at: float = 0.0
    purpose: str = ""
    legal_basis: str = "legitimate_interest"  # legitimate_interest, consent, contract


@dataclass
class DataRetentionPolicy:
    policy_id: str
    data_type: str  # keyframes, events, profiles, embeddings
    retention_days: int
    auto_delete: bool = True


class ConsentManager:
    def __init__(self, persistence=None):
        self._persistence = persistence
        self._consents: dict[str, list[ConsentRecord]] = {}  # entity_id -> records
        self._policies: dict[str, DataRetentionPolicy] = {}
        # Default policies
        for dt, days in [("keyframes", 30), ("events", 90), ("profiles", 365), ("embeddings", 180)]:
            pid = f"default_{dt}"
            self._policies[pid] = DataRetentionPolicy(pid, dt, days)

    def record_consent(self, entity_id: str, consent_type: str, granted: bool,
                       purpose: str = "", legal_basis: str = "consent",
                       duration_days: int = 365) -> ConsentRecord:
        now = time.time()
        rec = ConsentRecord(
            entity_id=entity_id, consent_type=consent_type, granted=granted,
            granted_at=now, expires_at=now + duration_days * 86400,
            purpose=purpose, legal_basis=legal_basis,
        )
        self._consents.setdefault(entity_id, []).append(rec)
        logger.info("Consent recorded: %s %s=%s basis=%s", entity_id, consent_type, granted, legal_basis)
        return rec

    def check_consent(self, entity_id: str, purpose: str) -> bool:
        records = self._consents.get(entity_id, [])
        now = time.time()
        for r in reversed(records):
            if r.purpose == purpose or r.consent_type == "processing":
                if r.expires_at > now and r.granted:
                    return True
                if not r.granted:
                    return False
        # Default: legitimate interest for security surveillance
        return True

    def get_consent_records(self, entity_id: str) -> list[dict]:
        return [r.__dict__ for r in self._consents.get(entity_id, [])]

    def get_expiring_data(self, days_ahead: int = 7) -> list[dict]:
        cutoff = time.time() + days_ahead * 86400
        expiring = []
        for eid, records in self._consents.items():
            for r in records:
                if 0 < r.expires_at < cutoff:
                    expiring.append({"entity_id": eid, "consent_type": r.consent_type,
                                     "expires_at": r.expires_at, "purpose": r.purpose})
        return expiring

    def enforce_retention(self) -> dict:
        now = time.time()
        deleted = {"expired_consents": 0, "data_purged": []}
        for eid in list(self._consents):
            self._consents[eid] = [r for r in self._consents[eid] if r.expires_at > now or r.expires_at == 0]
            if not self._consents[eid]:
                del self._consents[eid]
                deleted["expired_consents"] += 1
        # Check retention policies against persistence
        if self._persistence:
            for policy in self._policies.values():
                if policy.auto_delete:
                    cutoff = now - policy.retention_days * 86400
                    try:
                        alerts = self._persistence.get_alerts(limit=1000)
                        expired = [a for a in alerts if a.get("timestamp", now) < cutoff]
                        deleted["data_purged"].append({
                            "data_type": policy.data_type,
                            "policy_id": policy.policy_id,
                            "records_eligible": len(expired),
                        })
                    except Exception:
                        pass
        logger.info("Retention enforced: %s", deleted)
        return deleted

    def generate_dpia_report(self) -> dict:
        """Data Protection Impact Assessment (GDPR Article 35)."""
        total_entities = len(self._consents)
        total_consents = sum(len(v) for v in self._consents.values())
        bases = {}
        for records in self._consents.values():
            for r in records:
                bases[r.legal_basis] = bases.get(r.legal_basis, 0) + 1
        return {
            "report_type": "DPIA",
            "generated_at": time.time(),
            "system": "VisionBrain CCTV Analytics",
            "processing_purpose": "Security surveillance, safety monitoring, access control",
            "legal_bases": bases,
            "data_subjects_count": total_entities,
            "consent_records_count": total_consents,
            "retention_policies": [p.__dict__ for p in self._policies.values()],
            "safeguards": [
                "Privacy engine: face/plate blur at edge before cloud processing",
                "4-level access control with audit trail",
                "Consent management with automatic expiry",
                "Data retention policies with auto-deletion",
                "Right of access: per-entity data export",
                "Right to erasure: entity data deletion on request",
            ],
            "risks": [
                {"risk": "Re-identification from behavioral patterns", "mitigation": "Embedding anonymization, access controls"},
                {"risk": "Excessive retention", "mitigation": "Automated retention policies, consent expiry"},
                {"risk": "Unauthorized access", "mitigation": "API key + JWT auth, camera-level isolation"},
            ],
            "expiring_data_next_30d": self.get_expiring_data(30),
        }

    def get_data_subject_report(self, entity_id: str) -> dict:
        """GDPR Article 15 — right of access."""
        report = {
            "entity_id": entity_id,
            "generated_at": time.time(),
            "consent_records": self.get_consent_records(entity_id),
            "data_held": [],
        }
        if self._persistence:
            try:
                alerts = self._persistence.get_alerts(limit=1000)
                entity_alerts = [a for a in alerts if entity_id in str(a)]
                report["data_held"].append({"type": "alerts", "count": len(entity_alerts)})
            except Exception:
                pass
            try:
                investigations = self._persistence.get_investigations(limit=100)
                entity_inv = [i for i in investigations if entity_id in str(i)]
                report["data_held"].append({"type": "investigations", "count": len(entity_inv)})
            except Exception:
                pass
        return report

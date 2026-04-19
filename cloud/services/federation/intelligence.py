"""Cross-site intelligence federation — share entity alerts without raw video."""

import hashlib
import json
import logging
import time
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EntityAlert:
    entity_id: str
    site_id: str
    embedding_hash: str
    behavioral_signature: dict
    severity: str
    timestamp: float
    description: str


class FederatedIntelligence:
    def __init__(self, site_id="site-default", webhook_manager=None, profiler=None):
        self._site_id = site_id
        self._webhook_manager = webhook_manager
        self._profiler = profiler
        self._peer_sites: dict[str, str] = {}
        self._shared_alerts: deque = deque(maxlen=5000)
        self._entity_signatures: dict[str, dict] = {}

    def register_peer(self, site_id, callback_url):
        self._peer_sites[site_id] = callback_url

    def unregister_peer(self, site_id):
        self._peer_sites.pop(site_id, None)

    def list_peers(self) -> list[dict]:
        return [{"site_id": s, "url": u} for s, u in self._peer_sites.items()]

    def _get_embedding_hash(self, entity_id) -> str:
        if self._profiler:
            try:
                profile = self._profiler.get_profile(entity_id)
                return hashlib.sha256(json.dumps(profile, default=str).encode()).hexdigest()[:16]
            except Exception:
                pass
        return hashlib.sha256(entity_id.encode()).hexdigest()[:16]

    def share_entity_alert(self, entity_id, severity, description) -> EntityAlert:
        emb_hash = self._get_embedding_hash(entity_id)
        alert = EntityAlert(
            entity_id=entity_id, site_id=self._site_id, embedding_hash=emb_hash,
            behavioral_signature={"entity_id": entity_id}, severity=severity,
            timestamp=time.time(), description=description)
        self._shared_alerts.append(alert)
        self._entity_signatures[entity_id] = {
            "embedding_hash": emb_hash, "site_id": self._site_id, "last_seen": time.time()}
        for sid, url in self._peer_sites.items():
            try:
                if self._webhook_manager:
                    self._webhook_manager.deliver_alert({"type": "federation_alert", **alert.__dict__})
                else:
                    logger.info("Federation alert to %s: %s", sid, entity_id)
            except Exception:
                pass
        return alert

    def receive_alert(self, alert_dict):
        fields = {k: alert_dict[k] for k in EntityAlert.__dataclass_fields__ if k in alert_dict}
        self._shared_alerts.append(EntityAlert(**fields))

    def check_entity(self, entity_id) -> list[dict]:
        emb_hash = self._get_embedding_hash(entity_id)
        return [a.__dict__ for a in self._shared_alerts
                if a.embedding_hash == emb_hash and a.site_id != self._site_id]

    def get_shared_alerts(self, since_hours=24) -> list[dict]:
        cutoff = time.time() - since_hours * 3600
        return [a.__dict__ for a in self._shared_alerts if a.timestamp > cutoff]

    def get_federation_status(self) -> dict:
        return {"site_id": self._site_id, "peer_count": len(self._peer_sites),
                "shared_alerts": len(self._shared_alerts),
                "tracked_entities": len(self._entity_signatures)}

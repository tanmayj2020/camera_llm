"""Incident Evidence Package Export."""

import logging
import time
import uuid
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class EvidenceItem:
    item_type: str
    timestamp: float
    camera_id: str
    data: dict
    description: str


@dataclass
class EvidencePackage:
    package_id: str
    incident_id: str
    created_at: float
    created_by: str
    items: list[EvidenceItem]
    chain_of_custody: list[dict]
    narrative: str
    metadata: dict


class EvidencePackager:
    def __init__(self, persistence=None, kg=None, vlm_client=None):
        self._persistence = persistence
        self._kg = kg
        self._vlm_client = vlm_client

    def build_package(self, incident_id: str, camera_ids: list[str],
                      time_range_hours: int = 4, created_by: str = 'system') -> EvidencePackage:
        items: list[EvidenceItem] = []
        for cam in camera_ids:
            if self._persistence:
                for a in (self._persistence.get_alerts(camera_id=cam, limit=200) or []):
                    items.append(EvidenceItem(
                        item_type='alert', timestamp=a.get('timestamp', 0),
                        camera_id=cam, data=a, description=a.get('description', '')
                    ))
            if self._kg:
                for e in (self._kg.get_recent_events(cam, limit=50) or []):
                    items.append(EvidenceItem(
                        item_type='event', timestamp=e.get('timestamp', 0),
                        camera_id=cam, data=e, description=e.get('description', '')
                    ))

        now = time.time()
        custody = [
            {'action': 'created', 'user': created_by, 'timestamp': now},
            {'action': 'evidence_collected', 'source': 'automated_pipeline', 'timestamp': now},
        ]

        narrative = self._generate_narrative(incident_id, items)

        return EvidencePackage(
            package_id=str(uuid.uuid4())[:10],
            incident_id=incident_id,
            created_at=now,
            created_by=created_by,
            items=items,
            chain_of_custody=custody,
            narrative=narrative,
            metadata={'camera_ids': camera_ids, 'time_range_hours': time_range_hours,
                      'total_items': len(items)},
        )

    def _generate_narrative(self, incident_id: str, items: list[EvidenceItem]) -> str:
        if self._vlm_client:
            try:
                client = self._vlm_client._get_client()
                if client != 'stub':
                    prompt = (f"Generate an evidence narrative for incident {incident_id} "
                              f"with {len(items)} evidence items.")
                    return client.generate(prompt)
            except Exception:
                logger.warning("VLM narrative generation failed, using fallback")
        return (f"Evidence package for incident {incident_id}. "
                f"Contains {len(items)} items collected from automated pipeline.")

    def export_json(self, package: EvidencePackage) -> dict:
        return asdict(package)

    def export_html(self, package: EvidencePackage) -> str:
        rows = ''.join(
            f'<tr><td>{i.item_type}</td><td>{i.camera_id}</td>'
            f'<td>{i.timestamp}</td><td>{i.description}</td></tr>'
            for i in package.items
        )
        custody_rows = ''.join(
            f'<tr><td>{c.get("action","")}</td><td>{c.get("user",c.get("source",""))}</td>'
            f'<td>{c.get("timestamp","")}</td></tr>'
            for c in package.chain_of_custody
        )
        return (
            f'<html><body>'
            f'<h1>Evidence Package: {package.package_id}</h1>'
            f'<p>Incident: {package.incident_id} | Created: {package.created_at} | By: {package.created_by}</p>'
            f'<h2>Narrative</h2><p>{package.narrative}</p>'
            f'<h2>Evidence Items</h2>'
            f'<table border="1"><tr><th>Type</th><th>Camera</th><th>Timestamp</th><th>Description</th></tr>'
            f'{rows}</table>'
            f'<h2>Chain of Custody</h2>'
            f'<table border="1"><tr><th>Action</th><th>User/Source</th><th>Timestamp</th></tr>'
            f'{custody_rows}</table>'
            f'</body></html>'
        )

    def add_custody_entry(self, package: EvidencePackage, action: str, user: str) -> None:
        package.chain_of_custody.append({
            'action': action, 'user': user, 'timestamp': time.time()
        })

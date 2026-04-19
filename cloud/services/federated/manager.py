"""Task 18: Federated Learning + Multi-Tenant SaaS Layer."""

import hashlib
import logging
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Tenant:
    tenant_id: str
    name: str
    cameras: list[str] = field(default_factory=list)
    neo4j_database: str = ""
    gcs_bucket: str = ""
    bq_dataset: str = ""
    industry_template: str = ""  # "retail", "warehouse", "office", etc.
    onboarding_status: str = "pending"  # "pending", "learning", "active"
    created_at: float = field(default_factory=time.time)


class TenantManager:
    """Multi-tenant isolation and management."""

    def __init__(self):
        self._tenants: dict[str, Tenant] = {}

    def create_tenant(self, tenant_id: str, name: str, industry: str = "") -> Tenant:
        tenant = Tenant(
            tenant_id=tenant_id, name=name,
            neo4j_database=f"vb_{tenant_id}",
            gcs_bucket=f"visionbrain-{tenant_id}-keyframes",
            bq_dataset=f"visionbrain_{tenant_id}",
            industry_template=industry,
            onboarding_status="pending",
        )
        self._tenants[tenant_id] = tenant
        logger.info("Tenant created: %s (%s)", name, tenant_id)
        return tenant

    def get_tenant(self, tenant_id: str) -> Tenant | None:
        return self._tenants.get(tenant_id)

    def add_camera(self, tenant_id: str, camera_id: str):
        tenant = self._tenants.get(tenant_id)
        if tenant:
            tenant.cameras.append(camera_id)
            if tenant.onboarding_status == "pending":
                tenant.onboarding_status = "learning"

    def activate_tenant(self, tenant_id: str):
        tenant = self._tenants.get(tenant_id)
        if tenant:
            tenant.onboarding_status = "active"
            logger.info("Tenant %s activated", tenant_id)

    def validate_access(self, tenant_id: str, camera_id: str) -> bool:
        """Ensure tenant can only access their own cameras."""
        tenant = self._tenants.get(tenant_id)
        return tenant is not None and camera_id in tenant.cameras


class FederatedAggregator:
    """Privacy-preserving federated learning across tenants.

    Edge devices train local model updates; only gradients are shared.
    Differential privacy noise is added before aggregation.
    """

    def __init__(self, noise_scale: float = 0.01):
        self.noise_scale = noise_scale
        self._gradient_buffer: dict[str, list[np.ndarray]] = {}  # model_key -> [gradients]

    def submit_gradients(self, tenant_id: str, model_key: str, gradients: np.ndarray):
        """Receive gradient update from a tenant's edge device."""
        # Add differential privacy noise
        noise = np.random.normal(0, self.noise_scale, gradients.shape)
        noisy_gradients = gradients + noise

        if model_key not in self._gradient_buffer:
            self._gradient_buffer[model_key] = []
        self._gradient_buffer[model_key].append(noisy_gradients)

        logger.debug("Received gradients from %s for %s (noise_scale=%.4f)",
                      tenant_id, model_key, self.noise_scale)

    def aggregate(self, model_key: str, min_participants: int = 2) -> np.ndarray | None:
        """Federated averaging: aggregate gradients from multiple tenants."""
        grads = self._gradient_buffer.get(model_key, [])
        if len(grads) < min_participants:
            return None

        aggregated = np.mean(grads, axis=0)
        self._gradient_buffer[model_key] = []
        logger.info("Federated aggregation for %s: %d participants", model_key, len(grads))
        return aggregated

    def get_global_model_update(self, model_key: str) -> np.ndarray | None:
        """Get the latest aggregated model update for distribution to tenants."""
        return self.aggregate(model_key)


# Industry templates with pre-configured rules
INDUSTRY_TEMPLATES = {
    "retail": {
        "rules": [
            {"rule_id": "retail_theft", "name": "Potential Theft", "severity": "high",
             "conditions": [{"type": "duration", "entity_class": "person", "zone_id": "checkout", "min_seconds": 300}],
             "action": "alert_theft"},
            {"rule_id": "retail_queue", "name": "Long Queue", "severity": "medium",
             "conditions": [{"type": "count", "entity_class": "person", "zone_id": "checkout", "min_count": 8}],
             "action": "alert_queue"},
        ],
        "kpi_metrics": ["foot_traffic", "dwell_time", "queue_length", "conversion_rate"],
    },
    "warehouse": {
        "rules": [
            {"rule_id": "wh_safety", "name": "Safety Violation", "severity": "critical",
             "conditions": [{"type": "proximity", "entity_a_class": "person", "entity_b_class": "forklift", "max_distance": 2.0}],
             "action": "alert_safety"},
        ],
        "kpi_metrics": ["worker_count", "forklift_activity", "zone_utilization", "incident_count"],
    },
    "office": {
        "rules": [
            {"rule_id": "office_afterhours", "name": "After-Hours Access", "severity": "high",
             "conditions": [
                 {"type": "zone", "entity_class": "person", "zone_id": "office", "negate": False},
                 {"type": "time", "not_in_hours": list(range(8, 20))},
             ],
             "action": "alert_afterhours"},
        ],
        "kpi_metrics": ["occupancy", "meeting_room_usage", "peak_hours", "visitor_count"],
    },
}

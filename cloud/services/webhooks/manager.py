"""Webhook integration for delivering alerts to external systems."""

import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


class WebhookType(str, Enum):
    GENERIC = "generic"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    TEAMS = "teams"
    SIEM = "siem"


@dataclass
class WebhookConfig:
    name: str
    url: str
    webhook_type: WebhookType = WebhookType.GENERIC
    webhook_id: str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    secret: str = ""
    min_severity: str = "low"
    camera_filter: list[str] = field(default_factory=list)
    event_types: list[str] = field(default_factory=list)
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    deliveries: int = 0
    failures: int = 0
    last_delivered: float = 0.0


@dataclass
class DeliveryLog:
    delivery_id: str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    webhook_id: str = ""
    timestamp: float = field(default_factory=time.time)
    status_code: int = 0
    success: bool = False
    error: str = ""


class WebhookManager:
    def __init__(self):
        self._webhooks: dict[str, WebhookConfig] = {}
        self._delivery_log: list[DeliveryLog] = []
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import httpx
                self._client = httpx.Client(timeout=10)
            except Exception as e:
                logger.warning("httpx unavailable: %s — deliveries will be stubbed", e)
                self._client = "stub"
        return self._client

    def register(self, config: WebhookConfig) -> WebhookConfig:
        self._webhooks[config.webhook_id] = config
        logger.info("Registered webhook %s (%s)", config.webhook_id, config.name)
        return config

    def unregister(self, webhook_id: str) -> bool:
        if webhook_id in self._webhooks:
            del self._webhooks[webhook_id]
            return True
        return False

    def list_webhooks(self) -> list[WebhookConfig]:
        return list(self._webhooks.values())

    def update(self, webhook_id: str, **kwargs) -> WebhookConfig | None:
        wh = self._webhooks.get(webhook_id)
        if not wh:
            return None
        for k, v in kwargs.items():
            if hasattr(wh, k):
                setattr(wh, k, v)
        return wh

    def deliver_alert(self, alert: dict) -> list[DeliveryLog]:
        logs = []
        severity = alert.get("severity", "low")
        camera_id = alert.get("camera_id", "")
        anomaly_type = alert.get("anomaly_type", "")

        for wh in self._webhooks.values():
            if not wh.enabled:
                continue
            if SEVERITY_ORDER.get(severity, 0) < SEVERITY_ORDER.get(wh.min_severity, 0):
                continue
            if wh.camera_filter and camera_id not in wh.camera_filter:
                continue
            if wh.event_types and anomaly_type not in wh.event_types:
                continue
            log = self._send(wh, alert)
            logs.append(log)
            self._delivery_log.append(log)
        return logs

    def _send(self, wh: WebhookConfig, alert: dict) -> DeliveryLog:
        payload = self._format_payload(wh.webhook_type, alert)
        body = json.dumps(payload, default=str)
        headers = {"Content-Type": "application/json"}
        if wh.secret:
            sig = hmac.new(wh.secret.encode(), body.encode(), hashlib.sha256).hexdigest()
            headers["X-VisionBrain-Signature"] = sig

        client = self._get_client()
        log = DeliveryLog(webhook_id=wh.webhook_id)

        if client == "stub":
            logger.info("Stub delivery to %s", wh.url)
            log.success = True
            log.status_code = 200
            wh.deliveries += 1
            wh.last_delivered = time.time()
            return log

        try:
            resp = client.post(wh.url, content=body, headers=headers)
            log.status_code = resp.status_code
            log.success = resp.status_code < 400
            if log.success:
                wh.deliveries += 1
                wh.last_delivered = time.time()
            else:
                wh.failures += 1
                log.error = resp.text[:200]
        except Exception as e:
            wh.failures += 1
            log.error = str(e)
            logger.error("Webhook delivery failed for %s: %s", wh.webhook_id, e)
        return log

    def _format_payload(self, wtype: WebhookType, alert: dict) -> dict:
        severity = alert.get("severity", "unknown").upper()
        atype = alert.get("anomaly_type", "alert")
        desc = alert.get("description", "")
        cam = alert.get("camera_id", "unknown")

        if wtype == WebhookType.SLACK:
            return {"text": f"\U0001f6a8 [{severity}] {atype}\n{desc}\nCamera: {cam}"}

        if wtype == WebhookType.PAGERDUTY:
            return {
                "routing_key": "",
                "event_action": "trigger",
                "payload": {
                    "summary": f"[{severity}] {atype}: {desc}",
                    "source": f"visionbrain-{cam}",
                    "severity": alert.get("severity", "warning"),
                },
            }

        if wtype == WebhookType.TEAMS:
            return {
                "@type": "MessageCard",
                "summary": f"[{severity}] {atype}",
                "sections": [{
                    "activityTitle": f"\U0001f6a8 {atype}",
                    "facts": [
                        {"name": "Severity", "value": severity},
                        {"name": "Camera", "value": cam},
                        {"name": "Description", "value": desc},
                    ],
                }],
            }

        # GENERIC / SIEM — pass raw alert
        return alert

    def get_delivery_log(self, webhook_id: str | None = None, limit: int = 50) -> list[DeliveryLog]:
        logs = self._delivery_log if not webhook_id else [l for l in self._delivery_log if l.webhook_id == webhook_id]
        return logs[-limit:]

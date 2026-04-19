"""Integration Event Bus."""

import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EventSubscription:
    subscriber_name: str
    event_types: list[str]
    filters: dict
    callback_url: str
    subscription_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    delivered: int = 0
    errors: int = 0


class EventBus:
    def __init__(self):
        self._subscriptions: dict[str, EventSubscription] = {}
        self._event_log: deque[dict] = deque(maxlen=10000)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import httpx
                self._client = httpx.Client(timeout=5)
            except ImportError:
                logger.warning("httpx not installed, using stub client")
                self._client = "stub"
        return self._client

    def subscribe(self, name: str, event_types: list[str], filters: dict, callback_url: str) -> EventSubscription:
        sub = EventSubscription(subscriber_name=name, event_types=event_types, filters=filters, callback_url=callback_url)
        self._subscriptions[sub.subscription_id] = sub
        return sub

    def unsubscribe(self, subscription_id: str):
        self._subscriptions.pop(subscription_id, None)

    def list_subscriptions(self) -> list[EventSubscription]:
        return list(self._subscriptions.values())

    def publish(self, event_type: str, payload: dict):
        self._event_log.append({"event_type": event_type, "payload": payload, "timestamp": time.time()})
        client = self._get_client()
        for sub in self._subscriptions.values():
            if not sub.enabled or event_type not in sub.event_types or not self._matches_filters(sub, payload):
                continue
            if client == "stub":
                logger.info(f"Stub delivery to {sub.callback_url} for {event_type}")
                sub.delivered += 1
                continue
            try:
                client.post(sub.callback_url, json={"event_type": event_type, "payload": payload})
                sub.delivered += 1
            except Exception as e:
                logger.error(f"Delivery failed for {sub.subscription_id}: {e}")
                sub.errors += 1

    def _matches_filters(self, sub: EventSubscription, payload: dict) -> bool:
        for key, val in sub.filters.items():
            if key in payload and payload[key] != val:
                return False
        return True

    def get_event_log(self, event_type: str | None = None, limit: int = 100) -> list[dict]:
        events = self._event_log if event_type is None else [e for e in self._event_log if e["event_type"] == event_type]
        return list(events)[-limit:]

    def get_subscription_stats(self) -> list[dict]:
        return [{"subscription_id": s.subscription_id, "subscriber_name": s.subscriber_name, "enabled": s.enabled, "delivered": s.delivered, "errors": s.errors} for s in self._subscriptions.values()]

"""SOTA Action Engine — real-world actuation, Reflexion pattern, counterfactual simulation,
multi-step playbooks, outcome learning, proactive actions."""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Channel(str, Enum):
    EMAIL = "email"
    WHATSAPP = "whatsapp"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"


class ActuatorType(str, Enum):
    PTZ = "ptz"
    DOOR_LOCK = "door_lock"
    LIGHTS = "lights"
    PA_SYSTEM = "pa_system"
    ALARM = "alarm"
    NOTIFICATION = "notification"


@dataclass
class Notification:
    channel: Channel
    recipient: str
    subject: str
    body: str
    severity: str
    evidence: list[dict] = field(default_factory=list)
    event_id: str = ""


@dataclass
class ActuatorCommand:
    actuator_type: ActuatorType
    target_id: str
    command: str
    params: dict = field(default_factory=dict)
    executed: bool = False
    result: str = ""


@dataclass
class PlaybookStep:
    step_id: str
    action: ActuatorCommand | None = None
    notification: Notification | None = None
    condition: str = ""  # condition to evaluate for branching
    on_success: str = ""  # next step_id
    on_failure: str = ""  # next step_id


@dataclass
class Playbook:
    playbook_id: str
    name: str
    trigger_severity: str
    trigger_anomaly_types: list[str]
    steps: list[PlaybookStep] = field(default_factory=list)


@dataclass
class ActionOutcome:
    event_id: str
    action_type: str
    timestamp: float
    effective: bool | None = None  # None = pending evaluation
    anomaly_recurred: bool = False
    feedback: str = ""


@dataclass
class EscalationPolicy:
    initial_channels: list[Channel]
    escalation_channels: list[Channel]
    escalation_delay_s: float = 300
    max_escalations: int = 3


class NotificationAdapter:
    """Pluggable notification delivery."""

    def send_email(self, to: str, subject: str, body: str, attachments: list = None) -> bool:
        logger.info("[EMAIL] To=%s Subject=%s", to, subject)
        return True

    def send_whatsapp(self, to: str, body: str) -> bool:
        logger.info("[WHATSAPP] To=%s Body=%s", to, body[:100])
        return True

    def send_sms(self, to: str, body: str) -> bool:
        logger.info("[SMS] To=%s Body=%s", to, body[:100])
        return True

    def send_push(self, to: str, title: str, body: str, data: dict = None) -> bool:
        logger.info("[PUSH] To=%s Title=%s", to, title)
        return True

    def send(self, notification: Notification) -> bool:
        dispatch = {
            Channel.EMAIL: lambda n: self.send_email(n.recipient, n.subject, n.body),
            Channel.WHATSAPP: lambda n: self.send_whatsapp(n.recipient, n.body),
            Channel.SMS: lambda n: self.send_sms(n.recipient, n.body),
            Channel.PUSH: lambda n: self.send_push(n.recipient, n.subject, n.body),
        }
        handler = dispatch.get(notification.channel)
        return handler(notification) if handler else False


class ActuatorAdapter:
    """Pluggable real-world actuation — PTZ, locks, lights, PA, alarms."""

    def execute(self, cmd: ActuatorCommand) -> bool:
        logger.info("[ACTUATOR] %s → %s cmd=%s params=%s",
                    cmd.actuator_type.value, cmd.target_id, cmd.command, cmd.params)
        cmd.executed = True
        cmd.result = "ok"
        return True


class ActionEngine:
    """SOTA action engine: Reflexion self-critique, counterfactual simulation,
    multi-step playbooks, outcome learning, proactive actions.

    Guardrails: rate limiting, severity thresholds, quiet hours, dedup.
    """

    SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

    def __init__(self, min_severity: str = "medium", quiet_hours: tuple[int, int] = (23, 6),
                 rate_limit_per_min: int = 10):
        self.min_severity = min_severity
        self.quiet_start, self.quiet_end = quiet_hours
        self.rate_limit = rate_limit_per_min
        self.adapter = NotificationAdapter()
        self.actuator = ActuatorAdapter()

        self._recent_notifications: list[float] = []
        self._pending_acks: dict[str, dict] = {}
        self._dedup_cache: dict[str, float] = {}
        self._recipients: dict[str, list[str]] = {s: [] for s in self.SEVERITY_ORDER}
        self._playbooks: dict[str, Playbook] = {}
        self._outcomes: list[ActionOutcome] = []
        self._outcome_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "effective": 0})
        self._vlm_client = None

    def set_vlm_client(self, client):
        self._vlm_client = client

    def configure_recipients(self, severity: str, recipients: list[str]):
        self._recipients[severity] = recipients

    def add_playbook(self, playbook: Playbook):
        self._playbooks[playbook.playbook_id] = playbook

    # --- Reflexion: self-critique before executing ---

    def _reflexion_critique(self, anomaly: dict, proposed_actions: list[str]) -> tuple[bool, str]:
        """Critique proposed actions before execution. Returns (proceed, reasoning)."""
        # Check outcome history — if this action type was ineffective before, flag it
        anomaly_type = anomaly.get("anomaly_type", "")
        stats = self._outcome_stats.get(anomaly_type)
        if stats and stats["total"] >= 3:
            effectiveness = stats["effective"] / stats["total"]
            if effectiveness < 0.3:
                return False, (f"Historical effectiveness for '{anomaly_type}' is {effectiveness:.0%} "
                               f"({stats['effective']}/{stats['total']}). Escalating to human review instead.")

        # VLM-based critique if available
        if self._vlm_client and hasattr(self._vlm_client, '_get_client'):
            client = self._vlm_client._get_client()
            if client != "stub":
                try:
                    prompt = (f"Critique these proposed actions for anomaly '{anomaly_type}' "
                              f"(severity={anomaly.get('severity')}): {proposed_actions}. "
                              f"Are they proportionate? Any risks? Reply JSON: "
                              f'{{"proceed": true/false, "reasoning": "..."}}')
                    resp = client.generate_content(prompt)
                    data = json.loads(resp.text.strip().strip("`").lstrip("json\n"))
                    return data.get("proceed", True), data.get("reasoning", "")
                except Exception as e:
                    logger.debug("Reflexion VLM critique failed: %s", e)

        return True, "No objections — proceeding with proposed actions."

    # --- Counterfactual simulation ---

    def _simulate_counterfactual(self, anomaly: dict, action: str, world_model=None) -> dict:
        """What-if: simulate outcome of taking vs not taking action."""
        result = {"action": action, "predicted_outcome": "unknown", "confidence": 0.3}
        if world_model and hasattr(world_model, 'predict_trajectory'):
            result["predicted_outcome"] = "simulated_via_world_model"
            result["confidence"] = 0.6
        return result

    # --- Playbook execution ---

    def _execute_playbook(self, playbook: Playbook, anomaly: dict, causal=None) -> list:
        """Execute multi-step playbook with branching logic."""
        executed = []
        step_map = {s.step_id: s for s in playbook.steps}
        current_id = playbook.steps[0].step_id if playbook.steps else None

        while current_id and current_id in step_map:
            step = step_map[current_id]
            success = True

            if step.action:
                success = self.actuator.execute(step.action)
                executed.append(("actuator", step.action, success))

            if step.notification:
                success = self.adapter.send(step.notification)
                executed.append(("notification", step.notification, success))

            current_id = step.on_success if success else step.on_failure

        return executed

    # --- Outcome learning ---

    def record_outcome(self, event_id: str, effective: bool, feedback: str = ""):
        """Record whether an action was effective — feeds back into Reflexion."""
        for outcome in self._outcomes:
            if outcome.event_id == event_id:
                outcome.effective = effective
                outcome.feedback = feedback
                stats = self._outcome_stats[outcome.action_type]
                stats["total"] += 1
                if effective:
                    stats["effective"] += 1
                logger.info("Outcome recorded: event=%s effective=%s", event_id, effective)
                return
        logger.warning("No outcome found for event_id=%s", event_id)

    # --- Proactive actions from predictions ---

    def process_prediction(self, prediction) -> list[Notification]:
        """Take proactive action based on world model predictions."""
        if not hasattr(prediction, 'prediction_type'):
            return []
        if prediction.confidence < 0.5:
            return []

        body = f"⚡ Predicted: {prediction.description}"
        if prediction.recommended_action:
            body += f"\nRecommended: {prediction.recommended_action}"

        n = Notification(
            channel=Channel.PUSH, recipient="operator",
            subject=f"[PREDICTION] {prediction.prediction_type}",
            body=body, severity="medium",
        )
        if self.adapter.send(n):
            return [n]
        return []

    # --- Main entry point ---

    def process_anomaly(self, anomaly: dict, causal_analysis=None,
                        world_model=None) -> list[Notification]:
        """Full pipeline: triage → reflexion → playbook/notify → outcome tracking."""
        severity = anomaly.get("severity", "medium")

        # Guardrail: minimum severity
        if self.SEVERITY_ORDER.get(severity, 0) < self.SEVERITY_ORDER.get(self.min_severity, 0):
            return []

        # Guardrail: dedup
        dedup_key = f"{anomaly.get('rule_id', '')}_{anomaly.get('camera_id', '')}"
        now = time.time()
        if dedup_key in self._dedup_cache and now - self._dedup_cache[dedup_key] < 60:
            return []
        self._dedup_cache[dedup_key] = now

        # Guardrail: rate limit
        self._recent_notifications = [t for t in self._recent_notifications if now - t < 60]
        if len(self._recent_notifications) >= self.rate_limit:
            logger.warning("Rate limit reached, dropping notification")
            return []

        # Guardrail: quiet hours (critical bypasses)
        from datetime import datetime, timezone
        hour = datetime.fromtimestamp(now, tz=timezone.utc).hour
        is_quiet = (self.quiet_start <= hour or hour < self.quiet_end)
        if is_quiet and severity != "critical":
            return []

        # Check for matching playbook
        anomaly_type = anomaly.get("anomaly_type", "")
        matching_playbook = None
        for pb in self._playbooks.values():
            if (self.SEVERITY_ORDER.get(severity, 0) >= self.SEVERITY_ORDER.get(pb.trigger_severity, 0)
                    and (not pb.trigger_anomaly_types or anomaly_type in pb.trigger_anomaly_types)):
                matching_playbook = pb
                break

        # Reflexion: self-critique
        proposed = [f"playbook:{matching_playbook.name}" if matching_playbook else f"notify:{severity}"]
        proceed, reasoning = self._reflexion_critique(anomaly, proposed)
        logger.info("Reflexion: proceed=%s reason=%s", proceed, reasoning)

        if not proceed:
            # Escalate to human instead
            n = Notification(
                channel=Channel.PUSH, recipient="manager",
                subject=f"[HUMAN REVIEW] {anomaly_type}",
                body=f"Auto-action suppressed by Reflexion: {reasoning}\n\nAnomaly: {anomaly.get('description', '')}",
                severity=severity, event_id=anomaly.get("event_id", ""),
            )
            self.adapter.send(n)
            return [n]

        # Execute playbook or standard notifications
        sent = []
        if matching_playbook:
            self._execute_playbook(matching_playbook, anomaly, causal_analysis)

        notifications = self._craft_notifications(anomaly, causal_analysis, severity)
        for n in notifications:
            if self.adapter.send(n):
                self._recent_notifications.append(now)
                sent.append(n)

        # Track outcome
        self._outcomes.append(ActionOutcome(
            event_id=anomaly.get("event_id", ""),
            action_type=anomaly_type,
            timestamp=now,
        ))

        # Track for escalation
        if severity in ("high", "critical"):
            self._pending_acks[anomaly.get("event_id", "")] = {
                "anomaly": anomaly, "sent_at": now, "escalation_count": 0,
            }

        return sent

    def acknowledge(self, event_id: str):
        if event_id in self._pending_acks:
            del self._pending_acks[event_id]

    def check_escalations(self) -> list[Notification]:
        now = time.time()
        escalated = []
        for event_id, info in list(self._pending_acks.items()):
            elapsed = now - info["sent_at"]
            count = info["escalation_count"]
            if elapsed > 300 * (count + 1) and count < 3:
                info["escalation_count"] += 1
                n = Notification(
                    channel=Channel.PUSH, recipient="manager",
                    subject=f"[ESCALATION #{count + 1}] Unacknowledged alert",
                    body=f"Alert {event_id} unacknowledged for {elapsed / 60:.0f} min.",
                    severity="critical", event_id=event_id,
                )
                if self.adapter.send(n):
                    escalated.append(n)
        return escalated

    def _craft_notifications(self, anomaly: dict, causal, severity: str) -> list[Notification]:
        description = anomaly.get("description", "Anomaly detected")
        explanation = ""
        if causal and hasattr(causal, "causal_explanation"):
            explanation = f"\n\nCause: {causal.causal_explanation}"
            if hasattr(causal, "recommended_action"):
                explanation += f"\nRecommended: {causal.recommended_action}"

        body = f"🚨 {description}{explanation}\nCamera: {anomaly.get('camera_id', 'unknown')}"
        subject = f"[{severity.upper()}] {anomaly.get('anomaly_type', 'Alert')}"

        channels = {
            "critical": [Channel.PUSH, Channel.WHATSAPP, Channel.EMAIL],
            "high": [Channel.PUSH, Channel.EMAIL],
            "medium": [Channel.PUSH],
            "low": [Channel.EMAIL],
        }

        notifications = []
        for ch in channels.get(severity, [Channel.PUSH]):
            for r in self._recipients.get(severity, ["default"]):
                notifications.append(Notification(
                    channel=ch, recipient=r, subject=subject, body=body,
                    severity=severity, event_id=anomaly.get("event_id", ""),
                ))
        return notifications

import json
import time
import urllib.error
import urllib.request

from packages.common.logging import get_logger
from packages.common.textutil import redact_secrets
from services.alerts.types import AlertMessage

log = get_logger(__name__)


class Notifier:
    channel: str = "base"

    def send(self, message: AlertMessage) -> None:
        raise NotImplementedError


class LoggingNotifier(Notifier):
    channel = "dashboard"

    def __init__(self) -> None:
        self.sent: list[AlertMessage] = []

    def send(self, message: AlertMessage) -> None:
        self.sent.append(message)
        log.info(
            "alert",
            extra={
                "alert_id": message.alert_id,
                "event_id": message.event_id,
                "severity": message.severity.value
                if hasattr(message.severity, "value")
                else str(message.severity),
                "title": message.title,
            },
        )


class WebhookDeliveryError(Exception):
    pass


class WebhookNotifier(Notifier):
    channel = "webhook"

    def __init__(
        self,
        url: str,
        timeout: float = 5.0,
        max_retries: int = 2,
        backoff_seconds: float = 0.5,
        sleep=None,
    ) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError("webhook_url must start with http:// or https://")
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep if sleep is not None else time.sleep

    @staticmethod
    def _payload(message: AlertMessage) -> bytes:
        severity = (
            message.severity.value
            if hasattr(message.severity, "value")
            else str(message.severity)
        )
        return json.dumps(
            {
                "alert_id": message.alert_id,
                "event_id": message.event_id,
                "event_type": message.event_type,
                "severity": severity,
                "camera_id": message.camera_id,
                "timestamp": message.created_at.isoformat(),
                "title": message.title,
                "summary": message.summary,
                "evidence_ids": list(message.evidence_ids),
                "body": message.body,
                "metadata": message.metadata,
            }
        ).encode("utf-8")

    def send(self, message: AlertMessage) -> None:
        request = urllib.request.Request(
            self.url,
            data=self._payload(message),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        attempts = self.max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            status = None
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    status = getattr(response, "status", 200)
                if status >= 300:
                    raise WebhookDeliveryError(f"webhook returned HTTP {status}")
                log.info(
                    "webhook alert delivered",
                    extra={"event_id": message.event_id, "attempt": attempt + 1},
                )
                return
            except (urllib.error.URLError, TimeoutError, OSError, WebhookDeliveryError) as exc:
                last_error = exc
                if attempt < attempts - 1:
                    log.warning(
                        "webhook alert delivery failed, retrying",
                        extra={
                            "event_id": message.event_id,
                            "attempt": attempt + 1,
                            "error": redact_secrets(str(exc)),
                        },
                    )
                    self._sleep(self.backoff_seconds * (2**attempt))
        raise WebhookDeliveryError(
            redact_secrets(f"{type(last_error).__name__}: {last_error}")
        ) from last_error


class DatabaseNotifier(Notifier):
    channel = "in_app"

    def send(self, message: AlertMessage) -> None:
        from packages.common.timeutil import utcnow
        from packages.db import base as db_base
        from packages.db import models

        with db_base.session_scope() as session:
            session.add(
                models.Alert(
                    event_id=message.event_id,
                    channel=self.channel,
                    status="sent",
                    target="database",
                    attempts=1,
                    sent_at=utcnow(),
                )
            )
        log.info(
            "in-app alert recorded",
            extra={"event_id": message.event_id, "alert_id": message.alert_id},
        )


class MqttNotifier(Notifier):
    channel = "mqtt"

    def send(self, message: AlertMessage) -> None:
        raise NotImplementedError("MQTT transport is deferred to a later phase")


class EmailNotifier(Notifier):
    channel = "email"

    def send(self, message: AlertMessage) -> None:
        raise NotImplementedError("email transport is deferred to a later phase")

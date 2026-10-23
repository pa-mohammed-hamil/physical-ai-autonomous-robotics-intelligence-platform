"""Publisher module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File: backend/app/robotics/control/publisher.py

Python 3.11+
Standard library only.

SIMULATION-ONLY robotics message publisher.

Features:
- Topic-based publish/subscribe
- Tenant and robot isolation
- Structured JSON messages
- Message validation
- Topic allowlists
- Subscriber registration
- Bounded message queues
- Delivery tracking
- Retry with bounded attempts
- Dead-letter queue
- Message expiration
- Duplicate message rejection
- Thread-safe operations
- Audit logging
- JSON report export

IMPORTANT:
This is an in-memory simulation event publisher.

It does not publish to ROS 2, MQTT, Kafka,
physical robots, motor controllers or actuators.

Publishing a command-shaped message does not
authorize or execute a robot command.
"""

from __future__ import annotations

import json
import threading
import time
import uuid

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================

class PublisherError(Exception):
    pass


class PublisherValidationError(PublisherError):
    pass


class TopicNotFound(PublisherError):
    pass


class DuplicateMessageError(PublisherError):
    pass


class QueueCapacityError(PublisherError):
    pass


# ============================================================
# ENUMS
# ============================================================

class TopicType(str, Enum):
    TELEMETRY = "telemetry"
    ROBOT_STATUS = "robot_status"
    SENSOR_DATA = "sensor_data"
    PERCEPTION = "perception"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    VLA_ACTION = "vla_action"
    AGENT_ACTION = "agent_action"
    ACTION_RESULT = "action_result"
    SAFETY_EVENT = "safety_event"
    HEALTH_EVENT = "health_event"
    SIMULATION_EVENT = "simulation_event"
    EVALUATION_EVENT = "evaluation_event"
    CUSTOM = "custom"


class MessagePriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class DeliveryStatus(str, Enum):
    QUEUED = "queued"
    DELIVERED = "delivered"
    RETRY_PENDING = "retry_pending"
    FAILED = "failed"
    EXPIRED = "expired"
    DEAD_LETTER = "dead_letter"


class PublisherEventType(str, Enum):
    TOPIC_REGISTERED = "topic_registered"
    SUBSCRIBER_REGISTERED = "subscriber_registered"
    SUBSCRIBER_REMOVED = "subscriber_removed"
    MESSAGE_PUBLISHED = "message_published"
    MESSAGE_DELIVERED = "message_delivered"
    DELIVERY_RETRY = "delivery_retry"
    DELIVERY_FAILED = "delivery_failed"
    MESSAGE_EXPIRED = "message_expired"
    DEAD_LETTERED = "dead_lettered"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_id(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not all(
            character.isalnum()
            or character in "_-."
            for character in value
        )
    ):
        raise PublisherValidationError(
            f"Invalid {name}"
        )

    return value


def positive_int(value: int, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise PublisherValidationError(
            f"{name} must be a positive integer"
        )

    return value


def positive_float(
    value: float,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 < value < float("inf")
    ):
        raise PublisherValidationError(
            f"{name} must be finite and positive"
        )

    return float(value)


def json_object(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PublisherValidationError(
            "Payload must be a mapping"
        )

    try:
        encoded = json.dumps(
            dict(value),
            allow_nan=False,
            separators=(",", ":"),
        )

        decoded = json.loads(encoded)

    except (TypeError, ValueError) as exc:
        raise PublisherValidationError(
            "Payload must be valid JSON"
        ) from exc

    if not isinstance(decoded, dict):
        raise PublisherValidationError(
            "Payload must be a JSON object"
        )

    return decoded


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PublisherConfig:
    maximum_topics: int = 1000
    maximum_subscribers_per_topic: int = 100

    maximum_queue_size: int = 10000
    maximum_dead_letters: int = 20000
    maximum_audit_events: int = 20000

    maximum_message_bytes: int = 65536

    default_ttl_seconds: float = 30.0
    maximum_delivery_attempts: int = 3

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_topics",
            "maximum_subscribers_per_topic",
            "maximum_queue_size",
            "maximum_dead_letters",
            "maximum_audit_events",
            "maximum_message_bytes",
            "maximum_delivery_attempts",
        ):
            positive_int(
                getattr(self, name),
                name,
            )

        positive_float(
            self.default_ttl_seconds,
            "default_ttl_seconds",
        )

        if not self.simulation_only:
            raise PublisherValidationError(
                "Physical robot publishing unsupported"
            )


# ============================================================
# TOPIC DEFINITION
# ============================================================

@dataclass(frozen=True)
class RobotTopic:
    tenant_id: str
    robot_id: str
    topic_name: str
    topic_type: TopicType

    allowed_publishers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")
        validate_id(self.topic_name, "topic_name")

        if not isinstance(self.topic_type, TopicType):
            raise PublisherValidationError(
                "Invalid topic type"
            )

        for publisher in self.allowed_publishers:
            validate_id(
                publisher,
                "publisher ID",
            )


# ============================================================
# PUBLISHED MESSAGE
# ============================================================

@dataclass(frozen=True)
class PublishedRobotMessage:
    tenant_id: str
    robot_id: str

    topic_name: str
    publisher_id: str

    payload: Mapping[str, Any]

    priority: MessagePriority = (
        MessagePriority.NORMAL
    )

    message_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    timestamp: datetime = field(
        default_factory=utc_now
    )

    ttl_seconds: float = 30.0

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")
        validate_id(self.topic_name, "topic_name")
        validate_id(self.publisher_id, "publisher_id")
        validate_id(self.message_id, "message_id")

        if not isinstance(self.priority, MessagePriority):
            raise PublisherValidationError(
                "Invalid message priority"
            )

        if (
            not isinstance(self.timestamp, datetime)
            or self.timestamp.tzinfo is None
            or self.timestamp.utcoffset() is None
        ):
            raise PublisherValidationError(
                "Timezone-aware timestamp required"
            )

        positive_float(
            self.ttl_seconds,
            "ttl_seconds",
        )

        object.__setattr__(
            self,
            "payload",
            json_object(self.payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "topic_name": self.topic_name,
            "publisher_id": self.publisher_id,
            "priority": self.priority.value,
            "timestamp": (
                self.timestamp.astimezone(
                    timezone.utc
                ).isoformat()
            ),
            "ttl_seconds": self.ttl_seconds,
            "payload": dict(self.payload),
        }


# ============================================================
# SUBSCRIBER
# ============================================================

SubscriberCallback = Callable[
    [PublishedRobotMessage],
    None,
]


@dataclass(frozen=True)
class RobotSubscriber:
    subscriber_id: str
    callback: SubscriberCallback

    def __post_init__(self) -> None:
        validate_id(
            self.subscriber_id,
            "subscriber_id",
        )

        if not callable(self.callback):
            raise PublisherValidationError(
                "Subscriber callback must be callable"
            )


# ============================================================
# DELIVERY RECORD
# ============================================================

@dataclass
class MessageDelivery:
    delivery_id: str
    message: PublishedRobotMessage

    subscriber_id: str

    status: DeliveryStatus = (
        DeliveryStatus.QUEUED
    )

    attempts: int = 0
    last_error: str | None = None

    created_monotonic: float = field(
        default_factory=time.monotonic
    )

    expires_monotonic: float = 0.0

    delivered_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "delivery_id": self.delivery_id,
            "message_id": self.message.message_id,
            "tenant_id": self.message.tenant_id,
            "robot_id": self.message.robot_id,
            "topic_name": self.message.topic_name,
            "subscriber_id": self.subscriber_id,
            "status": self.status.value,
            "attempts": self.attempts,
            "last_error": self.last_error,
            "delivered_at": (
                self.delivered_at.isoformat()
                if self.delivered_at
                else None
            ),
        }


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class PublisherAuditEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    robot_id: str
    topic_name: str

    event_type: PublisherEventType
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "topic_name": self.topic_name,
            "event_type": self.event_type.value,
            "details": self.details,
        }


# ============================================================
# ROBOTICS PUBLISHER
# ============================================================

class RoboticsPublisher:

    def __init__(
        self,
        config: PublisherConfig | None = None,
    ) -> None:
        self.config = (
            config or PublisherConfig()
        )

        self.topics: dict[
            tuple[str, str, str],
            RobotTopic
        ] = {}

        self.subscribers: dict[
            tuple[str, str, str],
            dict[str, RobotSubscriber]
        ] = {}

        self.queue: deque[
            MessageDelivery
        ] = deque()

        self.dead_letters: deque[
            MessageDelivery
        ] = deque(
            maxlen=self.config.maximum_dead_letters
        )

        self.audit_events: deque[
            PublisherAuditEvent
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self.seen_message_ids: set[
            tuple[str, str]
        ] = set()

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------

    def _key(
        self,
        tenant_id: str,
        robot_id: str,
        topic_name: str,
    ) -> tuple[str, str, str]:
        return (
            validate_id(tenant_id, "tenant_id"),
            validate_id(robot_id, "robot_id"),
            validate_id(topic_name, "topic_name"),
        )

    def _audit(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        topic_name: str,
        event_type: PublisherEventType,
        details: str,
    ) -> None:
        self.audit_events.append(
            PublisherAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                tenant_id=tenant_id,
                robot_id=robot_id,
                topic_name=topic_name,
                event_type=event_type,
                details=details,
            )
        )

    # --------------------------------------------------------
    # REGISTER TOPIC
    # --------------------------------------------------------

    def register_topic(
        self,
        topic: RobotTopic,
    ) -> None:
        with self._lock:
            key = self._key(
                topic.tenant_id,
                topic.robot_id,
                topic.topic_name,
            )

            if key in self.topics:
                raise PublisherValidationError(
                    "Topic already registered"
                )

            if (
                len(self.topics)
                >= self.config.maximum_topics
            ):
                raise PublisherError(
                    "Maximum topic capacity reached"
                )

            self.topics[key] = topic
            self.subscribers[key] = {}

            self._audit(
                tenant_id=topic.tenant_id,
                robot_id=topic.robot_id,
                topic_name=topic.topic_name,
                event_type=(
                    PublisherEventType.TOPIC_REGISTERED
                ),
                details="Topic registered",
            )

    # --------------------------------------------------------
    # SUBSCRIBE
    # --------------------------------------------------------

    def subscribe(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        topic_name: str,
        subscriber: RobotSubscriber,
    ) -> None:
        with self._lock:
            key = self._key(
                tenant_id,
                robot_id,
                topic_name,
            )

            if key not in self.topics:
                raise TopicNotFound(
                    "Topic not found"
                )

            topic_subscribers = (
                self.subscribers[key]
            )

            if (
                subscriber.subscriber_id
                in topic_subscribers
            ):
                raise PublisherValidationError(
                    "Subscriber already registered"
                )

            if (
                len(topic_subscribers)
                >= self.config.maximum_subscribers_per_topic
            ):
                raise PublisherError(
                    "Subscriber capacity reached"
                )

            topic_subscribers[
                subscriber.subscriber_id
            ] = subscriber

            self._audit(
                tenant_id=tenant_id,
                robot_id=robot_id,
                topic_name=topic_name,
                event_type=(
                    PublisherEventType.SUBSCRIBER_REGISTERED
                ),
                details=(
                    f"Subscriber {subscriber.subscriber_id}"
                    " registered"
                ),
            )

    # --------------------------------------------------------
    # UNSUBSCRIBE
    # --------------------------------------------------------

    def unsubscribe(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        topic_name: str,
        subscriber_id: str,
    ) -> None:
        with self._lock:
            key = self._key(
                tenant_id,
                robot_id,
                topic_name,
            )

            if key not in self.topics:
                raise TopicNotFound(
                    "Topic not found"
                )

            removed = self.subscribers[key].pop(
                subscriber_id,
                None,
            )

            if removed is not None:
                self._audit(
                    tenant_id=tenant_id,
                    robot_id=robot_id,
                    topic_name=topic_name,
                    event_type=(
                        PublisherEventType.SUBSCRIBER_REMOVED
                    ),
                    details=(
                        f"Subscriber {subscriber_id}"
                        " removed"
                    ),
                )

    # --------------------------------------------------------
    # PUBLISH
    # --------------------------------------------------------

    def publish(
        self,
        message: PublishedRobotMessage,
    ) -> dict[str, Any]:
        key = self._key(
            message.tenant_id,
            message.robot_id,
            message.topic_name,
        )

        encoded = json.dumps(
            message.to_dict(),
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")

        if (
            len(encoded)
            > self.config.maximum_message_bytes
        ):
            raise PublisherValidationError(
                "Message exceeds size limit"
            )

        with self._lock:
            topic = self.topics.get(
                key
            )

            if topic is None:
                raise TopicNotFound(
                    "Topic not registered"
                )

            if (
                topic.allowed_publishers
                and message.publisher_id
                not in topic.allowed_publishers
            ):
                raise PublisherValidationError(
                    "Publisher not allowed on topic"
                )

            duplicate_key = (
                message.tenant_id,
                message.message_id,
            )

            if duplicate_key in self.seen_message_ids:
                raise DuplicateMessageError(
                    "Duplicate message ID"
                )

            subscriber_ids = list(
                self.subscribers[key]
            )

            if (
                len(self.queue) + len(subscriber_ids)
                > self.config.maximum_queue_size
            ):
                raise QueueCapacityError(
                    "Publisher queue is full"
                )

            # Reserve the message ID only after
            # validation and capacity checks.
            self.seen_message_ids.add(
                duplicate_key
            )

            now_monotonic = time.monotonic()

            for subscriber_id in subscriber_ids:
                delivery = MessageDelivery(
                    delivery_id=str(uuid.uuid4()),
                    message=message,
                    subscriber_id=subscriber_id,
                    created_monotonic=now_monotonic,
                    expires_monotonic=(
                        now_monotonic
                        + message.ttl_seconds
                    ),
                )

                self.queue.append(
                    delivery
                )

            self._audit(
                tenant_id=message.tenant_id,
                robot_id=message.robot_id,
                topic_name=message.topic_name,
                event_type=(
                    PublisherEventType.MESSAGE_PUBLISHED
                ),
                details=(
                    f"Queued for {len(subscriber_ids)} "
                    "subscribers"
                ),
            )

            return {
                "message_id": message.message_id,
                "topic_name": message.topic_name,
                "subscriber_count": len(
                    subscriber_ids
                ),
                "queued_deliveries": len(
                    subscriber_ids
                ),
                "status": "published",
            }

    # --------------------------------------------------------
    # PROCESS QUEUE
    # --------------------------------------------------------

    def process_queue(
        self,
        *,
        maximum_deliveries: int = 100,
    ) -> dict[str, int]:
        positive_int(
            maximum_deliveries,
            "maximum_deliveries",
        )

        delivered = 0
        retried = 0
        dead_lettered = 0
        expired = 0

        for _ in range(maximum_deliveries):
            with self._lock:
                if not self.queue:
                    break

                delivery = self.queue.popleft()

                key = self._key(
                    delivery.message.tenant_id,
                    delivery.message.robot_id,
                    delivery.message.topic_name,
                )

                subscriber = (
                    self.subscribers.get(
                        key,
                        {},
                    ).get(
                        delivery.subscriber_id
                    )
                )

            if (
                time.monotonic()
                >= delivery.expires_monotonic
            ):
                delivery.status = (
                    DeliveryStatus.EXPIRED
                )

                with self._lock:
                    self._audit(
                        tenant_id=delivery.message.tenant_id,
                        robot_id=delivery.message.robot_id,
                        topic_name=delivery.message.topic_name,
                        event_type=(
                            PublisherEventType.MESSAGE_EXPIRED
                        ),
                        details=delivery.delivery_id,
                    )

                expired += 1
                continue

            if subscriber is None:
                delivery.status = (
                    DeliveryStatus.DEAD_LETTER
                )
                delivery.last_error = (
                    "Subscriber unavailable"
                )

                with self._lock:
                    self.dead_letters.append(
                        delivery
                    )

                dead_lettered += 1
                continue

            try:
                delivery.attempts += 1

                subscriber.callback(
                    delivery.message
                )

                delivery.status = (
                    DeliveryStatus.DELIVERED
                )

                delivery.delivered_at = (
                    utc_now()
                )

                with self._lock:
                    self._audit(
                        tenant_id=delivery.message.tenant_id,
                        robot_id=delivery.message.robot_id,
                        topic_name=delivery.message.topic_name,
                        event_type=(
                            PublisherEventType.MESSAGE_DELIVERED
                        ),
                        details=delivery.delivery_id,
                    )

                delivered += 1

            except Exception as exc:
                delivery.last_error = (
                    f"{type(exc).__name__}: "
                    "subscriber callback failed"
                )

                if (
                    delivery.attempts
                    < self.config.maximum_delivery_attempts
                    and time.monotonic()
                    < delivery.expires_monotonic
                ):
                    delivery.status = (
                        DeliveryStatus.RETRY_PENDING
                    )

                    with self._lock:
                        self.queue.append(
                            delivery
                        )

                        self._audit(
                            tenant_id=delivery.message.tenant_id,
                            robot_id=delivery.message.robot_id,
                            topic_name=delivery.message.topic_name,
                            event_type=(
                                PublisherEventType.DELIVERY_RETRY
                            ),
                            details=delivery.delivery_id,
                        )

                    retried += 1

                else:
                    delivery.status = (
                        DeliveryStatus.DEAD_LETTER
                    )

                    with self._lock:
                        self.dead_letters.append(
                            delivery
                        )

                        self._audit(
                            tenant_id=delivery.message.tenant_id,
                            robot_id=delivery.message.robot_id,
                            topic_name=delivery.message.topic_name,
                            event_type=(
                                PublisherEventType.DEAD_LETTERED
                            ),
                            details=delivery.delivery_id,
                        )

                    dead_lettered += 1

        return {
            "delivered": delivered,
            "retried": retried,
            "dead_lettered": dead_lettered,
            "expired": expired,
            "remaining_queue": len(self.queue),
        }

    # --------------------------------------------------------
    # LIST TOPICS
    # --------------------------------------------------------

    def list_topics(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            results = []

            for key, topic in self.topics.items():
                if key[0] != tenant_id:
                    continue

                if (
                    robot_id is not None
                    and key[1] != robot_id
                ):
                    continue

                results.append(
                    {
                        "tenant_id": topic.tenant_id,
                        "robot_id": topic.robot_id,
                        "topic_name": topic.topic_name,
                        "topic_type": topic.topic_type.value,
                        "subscriber_count": len(
                            self.subscribers[key]
                        ),
                    }
                )

            return results

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            topics = [
                key
                for key in self.topics
                if (
                    tenant_id is None
                    or key[0] == tenant_id
                )
            ]

            queued = [
                delivery
                for delivery in self.queue
                if (
                    tenant_id is None
                    or delivery.message.tenant_id
                    == tenant_id
                )
            ]

            dead = [
                delivery
                for delivery in self.dead_letters
                if (
                    tenant_id is None
                    or delivery.message.tenant_id
                    == tenant_id
                )
            ]

            return {
                "total_topics": len(topics),
                "queued_deliveries": len(queued),
                "dead_letters": len(dead),
                "subscriber_count": sum(
                    len(self.subscribers[key])
                    for key in topics
                ),
            }

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        path: str | Path,
        *,
        tenant_id: str,
    ) -> Path:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            payload = {
                "generated_at": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "statistics": self.statistics(
                    tenant_id=tenant_id
                ),
                "topics": self.list_topics(
                    tenant_id=tenant_id
                ),
                "dead_letters": [
                    delivery.to_dict()
                    for delivery in self.dead_letters
                    if delivery.message.tenant_id
                    == tenant_id
                ],
                "audit_events": [
                    event.to_dict()
                    for event in self.audit_events
                    if event.tenant_id == tenant_id
                ],
            }

        output.write_text(
            json.dumps(
                payload,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        return output


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    publisher = RoboticsPublisher()

    publisher.register_topic(
        RobotTopic(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            topic_name="robot.telemetry",
            topic_type=TopicType.TELEMETRY,
            allowed_publishers=(
                "telemetry_worker",
            ),
        )
    )

    def telemetry_subscriber(
        message: PublishedRobotMessage,
    ) -> None:
        print(
            "Received telemetry:",
            json.dumps(
                message.to_dict(),
                indent=2,
            ),
        )

    publisher.subscribe(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        topic_name="robot.telemetry",
        subscriber=RobotSubscriber(
            subscriber_id="dashboard",
            callback=telemetry_subscriber,
        ),
    )

    result = publisher.publish(
        PublishedRobotMessage(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            topic_name="robot.telemetry",
            publisher_id="telemetry_worker",
            payload={
                "position": {
                    "x": 2.0,
                    "y": 3.0,
                    "z": 0.0,
                },
                "speed_mps": 0.5,
                "battery_percent": 84.0,
                "temperature_c": 39.0,
            },
        )
    )

    print(
        "\nPublish result:",
        json.dumps(
            result,
            indent=2,
        ),
    )

    delivery = publisher.process_queue()

    print(
        "\nDelivery result:",
        json.dumps(
            delivery,
            indent=2,
        ),
    )

    publisher.export_json(
        "data/robotics/publisher_report.json",
        tenant_id="tenant_alpha",
    )


if __name__ == "__main__":
    main()

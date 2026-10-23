"""Topic Manager module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

backend/app/robotics/control/topic_manager.py

Python 3.11+ | Standard library only

Simulation-only ROS 2-style topic manager.

Features:
- Topic registration and deletion
- Publisher/subscriber registration
- QoS compatibility validation
- Message schema identification
- Topic lifecycle and availability
- Message freshness monitoring
- Sequence-based duplicate rejection
- Topic statistics and discovery
- Tenant and robot isolation
- Thread-safe operations
- Audit logging
- JSON report export

Does not connect to actual ROS 2/DDS middleware,
publish commands, or control physical robots.
"""

from __future__ import annotations

import json
import math
import threading
import time
import uuid

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


# ============================================================
# EXCEPTIONS
# ============================================================

class TopicManagerError(Exception):
    pass


class TopicValidationError(TopicManagerError):
    pass


class TopicNotFound(TopicManagerError):
    pass


class TopicConflict(TopicManagerError):
    pass


class TopicPermissionDenied(TopicManagerError):
    pass


# ============================================================
# ENUMS
# ============================================================

class TopicCategory(str, Enum):
    TELEMETRY = "telemetry"
    SENSOR = "sensor"
    CAMERA = "camera"
    LIDAR = "lidar"
    ODOMETRY = "odometry"
    TF = "tf"
    PERCEPTION = "perception"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    VLA = "vla"
    ROBOT_HEALTH = "robot_health"
    SAFETY = "safety"
    SIMULATION = "simulation"
    ACTION_RESULT = "action_result"
    CUSTOM = "custom"


class TopicStatus(str, Enum):
    REGISTERED = "registered"
    ACTIVE = "active"
    DEGRADED = "degraded"
    STALE = "stale"
    INACTIVE = "inactive"


class ReliabilityPolicy(str, Enum):
    RELIABLE = "reliable"
    BEST_EFFORT = "best_effort"


class DurabilityPolicy(str, Enum):
    VOLATILE = "volatile"
    TRANSIENT_LOCAL = "transient_local"


class HistoryPolicy(str, Enum):
    KEEP_LAST = "keep_last"


class TopicEventType(str, Enum):
    TOPIC_REGISTERED = "topic_registered"
    TOPIC_DELETED = "topic_deleted"
    PUBLISHER_ADDED = "publisher_added"
    PUBLISHER_REMOVED = "publisher_removed"
    SUBSCRIBER_ADDED = "subscriber_added"
    SUBSCRIBER_REMOVED = "subscriber_removed"
    MESSAGE_OBSERVED = "message_observed"
    TOPIC_STALE = "topic_stale"
    TOPIC_RECOVERED = "topic_recovered"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_id(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not all(
            ch.isalnum() or ch in "_.-"
            for ch in value
        )
    ):
        raise TopicValidationError(
            f"Invalid {label}"
        )

    return value


def validate_ros_topic(name: str) -> str:
    if (
        not isinstance(name, str)
        or not name.startswith("/")
        or len(name) > 256
        or name == "/"
        or name.endswith("/")
        or "//" in name
    ):
        raise TopicValidationError(
            "Invalid ROS topic name"
        )

    segments = name[1:].split("/")

    for segment in segments:
        if (
            not segment
            or not (
                segment[0].isalpha()
                or segment[0] == "_"
            )
            or not all(
                ch.isalnum() or ch == "_"
                for ch in segment
            )
        ):
            raise TopicValidationError(
                "Invalid ROS topic segment"
            )

    return name


def positive_int(value: int, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise TopicValidationError(
            f"{label} must be a positive integer"
        )

    return value


def positive_float(
    value: float,
    label: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise TopicValidationError(
            f"{label} must be finite and positive"
        )

    return float(value)


# ============================================================
# QOS CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class TopicQoS:
    reliability: ReliabilityPolicy = (
        ReliabilityPolicy.RELIABLE
    )

    durability: DurabilityPolicy = (
        DurabilityPolicy.VOLATILE
    )

    history: HistoryPolicy = (
        HistoryPolicy.KEEP_LAST
    )

    depth: int = 10

    def __post_init__(self) -> None:
        if not isinstance(
            self.reliability,
            ReliabilityPolicy,
        ):
            raise TopicValidationError(
                "Invalid reliability policy"
            )

        if not isinstance(
            self.durability,
            DurabilityPolicy,
        ):
            raise TopicValidationError(
                "Invalid durability policy"
            )

        if not isinstance(
            self.history,
            HistoryPolicy,
        ):
            raise TopicValidationError(
                "Invalid history policy"
            )

        positive_int(
            self.depth,
            "QoS depth",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reliability": self.reliability.value,
            "durability": self.durability.value,
            "history": self.history.value,
            "depth": self.depth,
        }


def qos_compatible(
    offered: TopicQoS,
    requested: TopicQoS,
) -> bool:
    """
    Simplified ROS 2 requested/offered
    compatibility rules.

    Reliable subscriber requires a
    reliable publisher.

    Transient-local subscriber requires
    transient-local publisher.

    Depth is not used as a compatibility
    rejection criterion.
    """

    if (
        requested.reliability
        == ReliabilityPolicy.RELIABLE
        and offered.reliability
        != ReliabilityPolicy.RELIABLE
    ):
        return False

    if (
        requested.durability
        == DurabilityPolicy.TRANSIENT_LOCAL
        and offered.durability
        != DurabilityPolicy.TRANSIENT_LOCAL
    ):
        return False

    return True


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class TopicManagerConfig:
    maximum_topics: int = 10000

    maximum_endpoints_per_topic: int = 100

    maximum_audit_events: int = 20000

    default_freshness_timeout_seconds: float = 5.0

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for label in (
            "maximum_topics",
            "maximum_endpoints_per_topic",
            "maximum_audit_events",
        ):
            positive_int(
                getattr(self, label),
                label,
            )

        positive_float(
            self.default_freshness_timeout_seconds,
            "default_freshness_timeout_seconds",
        )

        if not self.simulation_only:
            raise TopicValidationError(
                "Physical ROS 2 topic management unsupported"
            )


# ============================================================
# TOPIC DEFINITION
# ============================================================

@dataclass(frozen=True)
class RobotTopicDefinition:
    tenant_id: str
    robot_id: str

    topic_id: str
    ros_topic_name: str

    message_type: str
    category: TopicCategory

    qos: TopicQoS = field(
        default_factory=TopicQoS
    )

    freshness_timeout_seconds: float = 5.0

    critical: bool = False

    description: str = ""

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")
        validate_id(self.topic_id, "topic_id")

        validate_ros_topic(
            self.ros_topic_name
        )

        if (
            not isinstance(self.message_type, str)
            or not self.message_type.strip()
            or len(self.message_type) > 256
        ):
            raise TopicValidationError(
                "Invalid ROS message type"
            )

        if not isinstance(
            self.category,
            TopicCategory,
        ):
            raise TopicValidationError(
                "Invalid topic category"
            )

        if not isinstance(self.qos, TopicQoS):
            raise TopicValidationError(
                "Invalid QoS configuration"
            )

        positive_float(
            self.freshness_timeout_seconds,
            "freshness_timeout_seconds",
        )


# ============================================================
# ENDPOINT DEFINITION
# ============================================================

@dataclass(frozen=True)
class TopicEndpoint:
    endpoint_id: str
    node_id: str

    qos: TopicQoS

    def __post_init__(self) -> None:
        validate_id(
            self.endpoint_id,
            "endpoint_id",
        )

        validate_id(
            self.node_id,
            "node_id",
        )

        if not isinstance(self.qos, TopicQoS):
            raise TopicValidationError(
                "Invalid endpoint QoS"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint_id": self.endpoint_id,
            "node_id": self.node_id,
            "qos": self.qos.to_dict(),
        }


# ============================================================
# MESSAGE OBSERVATION
# ============================================================

@dataclass(frozen=True)
class TopicMessageObservation:
    tenant_id: str
    robot_id: str
    topic_id: str

    publisher_id: str

    sequence_number: int

    message_size_bytes: int

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")
        validate_id(self.topic_id, "topic_id")
        validate_id(self.publisher_id, "publisher_id")

        if (
            isinstance(self.sequence_number, bool)
            or not isinstance(self.sequence_number, int)
            or self.sequence_number < 0
        ):
            raise TopicValidationError(
                "Invalid sequence number"
            )

        if (
            isinstance(self.message_size_bytes, bool)
            or not isinstance(self.message_size_bytes, int)
            or self.message_size_bytes < 0
        ):
            raise TopicValidationError(
                "Invalid message size"
            )

        if (
            not isinstance(self.timestamp, datetime)
            or self.timestamp.tzinfo is None
            or self.timestamp.utcoffset() is None
        ):
            raise TopicValidationError(
                "Timezone-aware timestamp required"
            )


# ============================================================
# TOPIC RECORD
# ============================================================

@dataclass
class TopicRecord:
    definition: RobotTopicDefinition

    publishers: dict[
        str,
        TopicEndpoint
    ] = field(default_factory=dict)

    subscribers: dict[
        str,
        TopicEndpoint
    ] = field(default_factory=dict)

    status: TopicStatus = (
        TopicStatus.REGISTERED
    )

    last_message_at: datetime | None = None

    last_received_monotonic: float | None = None

    created_monotonic: float = field(
        default_factory=time.monotonic
    )

    message_count: int = 0

    total_bytes: int = 0

    last_sequences: dict[
        str,
        int
    ] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        definition = self.definition

        return {
            "tenant_id": definition.tenant_id,
            "robot_id": definition.robot_id,
            "topic_id": definition.topic_id,
            "ros_topic_name": (
                definition.ros_topic_name
            ),
            "message_type": (
                definition.message_type
            ),
            "category": (
                definition.category.value
            ),
            "qos": definition.qos.to_dict(),
            "critical": definition.critical,
            "status": self.status.value,
            "freshness_timeout_seconds": (
                definition.freshness_timeout_seconds
            ),
            "publisher_count": len(
                self.publishers
            ),
            "subscriber_count": len(
                self.subscribers
            ),
            "publishers": [
                endpoint.to_dict()
                for endpoint in self.publishers.values()
            ],
            "subscribers": [
                endpoint.to_dict()
                for endpoint in self.subscribers.values()
            ],
            "message_count": self.message_count,
            "total_bytes": self.total_bytes,
            "last_message_at": (
                self.last_message_at.isoformat()
                if self.last_message_at
                else None
            ),
        }


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class TopicAuditEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    robot_id: str
    topic_id: str

    event_type: TopicEventType

    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "topic_id": self.topic_id,
            "event_type": self.event_type.value,
            "details": self.details,
        }


# ============================================================
# TOPIC MANAGER
# ============================================================

class RoboticsTopicManager:

    def __init__(
        self,
        config: TopicManagerConfig | None = None,
    ) -> None:
        self.config = (
            config or TopicManagerConfig()
        )

        self.topics: dict[
            tuple[str, str, str],
            TopicRecord
        ] = {}

        self.audit_events: deque[
            TopicAuditEvent
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------

    def _key(
        self,
        tenant_id: str,
        robot_id: str,
        topic_id: str,
    ) -> tuple[str, str, str]:
        return (
            validate_id(tenant_id, "tenant_id"),
            validate_id(robot_id, "robot_id"),
            validate_id(topic_id, "topic_id"),
        )

    def _get(
        self,
        tenant_id: str,
        robot_id: str,
        topic_id: str,
    ) -> TopicRecord:
        record = self.topics.get(
            self._key(
                tenant_id,
                robot_id,
                topic_id,
            )
        )

        if record is None:
            raise TopicNotFound(
                f"Topic not found: {topic_id}"
            )

        return record

    def _audit(
        self,
        record: TopicRecord,
        event_type: TopicEventType,
        details: str,
    ) -> None:
        definition = record.definition

        self.audit_events.append(
            TopicAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                tenant_id=definition.tenant_id,
                robot_id=definition.robot_id,
                topic_id=definition.topic_id,
                event_type=event_type,
                details=details,
            )
        )

    # --------------------------------------------------------
    # REGISTER TOPIC
    # --------------------------------------------------------

    def register_topic(
        self,
        definition: RobotTopicDefinition,
    ) -> dict[str, Any]:
        with self._lock:
            key = self._key(
                definition.tenant_id,
                definition.robot_id,
                definition.topic_id,
            )

            if key in self.topics:
                raise TopicConflict(
                    "Topic already registered"
                )

            if (
                len(self.topics)
                >= self.config.maximum_topics
            ):
                raise TopicManagerError(
                    "Maximum topic capacity reached"
                )

            for record in self.topics.values():
                existing = record.definition

                if (
                    existing.tenant_id
                    == definition.tenant_id
                    and existing.robot_id
                    == definition.robot_id
                    and existing.ros_topic_name
                    == definition.ros_topic_name
                ):
                    raise TopicConflict(
                        "ROS topic name already registered"
                    )

            record = TopicRecord(
                definition=definition
            )

            self.topics[key] = record

            self._audit(
                record,
                TopicEventType.TOPIC_REGISTERED,
                "Topic registered",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # DELETE TOPIC
    # --------------------------------------------------------

    def delete_topic(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        topic_id: str,
    ) -> None:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                topic_id,
            )

            if (
                record.publishers
                or record.subscribers
            ):
                raise TopicConflict(
                    "Remove endpoints before deleting topic"
                )

            self._audit(
                record,
                TopicEventType.TOPIC_DELETED,
                "Topic deleted",
            )

            del self.topics[
                self._key(
                    tenant_id,
                    robot_id,
                    topic_id,
                )
            ]

    # --------------------------------------------------------
    # REGISTER PUBLISHER
    # --------------------------------------------------------

    def add_publisher(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        topic_id: str,
        endpoint: TopicEndpoint,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                topic_id,
            )

            if (
                endpoint.endpoint_id
                in record.publishers
            ):
                raise TopicConflict(
                    "Publisher already registered"
                )

            if (
                len(record.publishers)
                >= self.config.maximum_endpoints_per_topic
            ):
                raise TopicManagerError(
                    "Publisher capacity reached"
                )

            record.publishers[
                endpoint.endpoint_id
            ] = endpoint

            self._audit(
                record,
                TopicEventType.PUBLISHER_ADDED,
                f"Publisher {endpoint.endpoint_id} added",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # REGISTER SUBSCRIBER
    # --------------------------------------------------------

    def add_subscriber(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        topic_id: str,
        endpoint: TopicEndpoint,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                topic_id,
            )

            if (
                endpoint.endpoint_id
                in record.subscribers
            ):
                raise TopicConflict(
                    "Subscriber already registered"
                )

            if (
                len(record.subscribers)
                >= self.config.maximum_endpoints_per_topic
            ):
                raise TopicManagerError(
                    "Subscriber capacity reached"
                )

            record.subscribers[
                endpoint.endpoint_id
            ] = endpoint

            self._audit(
                record,
                TopicEventType.SUBSCRIBER_ADDED,
                f"Subscriber {endpoint.endpoint_id} added",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # REMOVE PUBLISHER
    # --------------------------------------------------------

    def remove_publisher(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        topic_id: str,
        endpoint_id: str,
    ) -> None:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                topic_id,
            )

            if (
                record.publishers.pop(
                    endpoint_id,
                    None,
                )
                is None
            ):
                raise TopicNotFound(
                    "Publisher endpoint not found"
                )

            record.last_sequences.pop(
                endpoint_id,
                None,
            )

            self._audit(
                record,
                TopicEventType.PUBLISHER_REMOVED,
                f"Publisher {endpoint_id} removed",
            )

    # --------------------------------------------------------
    # REMOVE SUBSCRIBER
    # --------------------------------------------------------

    def remove_subscriber(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        topic_id: str,
        endpoint_id: str,
    ) -> None:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                topic_id,
            )

            if (
                record.subscribers.pop(
                    endpoint_id,
                    None,
                )
                is None
            ):
                raise TopicNotFound(
                    "Subscriber endpoint not found"
                )

            self._audit(
                record,
                TopicEventType.SUBSCRIBER_REMOVED,
                f"Subscriber {endpoint_id} removed",
            )

    # --------------------------------------------------------
    # QOS COMPATIBILITY
    # --------------------------------------------------------

    def check_qos(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        topic_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                topic_id,
            )

            matches = []
            incompatible = []

            for publisher in record.publishers.values():
                for subscriber in record.subscribers.values():
                    pair = {
                        "publisher_id": (
                            publisher.endpoint_id
                        ),
                        "subscriber_id": (
                            subscriber.endpoint_id
                        ),
                    }

                    if qos_compatible(
                        publisher.qos,
                        subscriber.qos,
                    ):
                        matches.append(pair)
                    else:
                        incompatible.append(pair)

            return {
                "topic_id": topic_id,
                "compatible_pairs": matches,
                "incompatible_pairs": incompatible,
                "all_compatible": (
                    len(incompatible) == 0
                ),
            }

    # --------------------------------------------------------
    # OBSERVE MESSAGE
    # --------------------------------------------------------

    def observe_message(
        self,
        observation: TopicMessageObservation,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                observation.tenant_id,
                observation.robot_id,
                observation.topic_id,
            )

            publisher = record.publishers.get(
                observation.publisher_id
            )

            if publisher is None:
                raise TopicPermissionDenied(
                    "Publisher not registered"
                )

            previous_sequence = (
                record.last_sequences.get(
                    observation.publisher_id,
                    -1,
                )
            )

            if (
                observation.sequence_number
                <= previous_sequence
            ):
                raise TopicValidationError(
                    "Duplicate or out-of-order message"
                )

            timestamp = (
                observation.timestamp.astimezone(
                    timezone.utc
                )
            )

            now = utc_now()

            age = (
                now - timestamp
            ).total_seconds()

            if age < -1.0:
                raise TopicValidationError(
                    "Future message timestamp"
                )

            if (
                age
                > record.definition.freshness_timeout_seconds
            ):
                raise TopicValidationError(
                    "Stale message observation"
                )

            record.last_sequences[
                observation.publisher_id
            ] = observation.sequence_number

            record.last_message_at = timestamp

            record.last_received_monotonic = (
                time.monotonic()
            )

            record.message_count += 1

            record.total_bytes += (
                observation.message_size_bytes
            )

            previous_status = record.status

            record.status = TopicStatus.ACTIVE

            self._audit(
                record,
                TopicEventType.MESSAGE_OBSERVED,
                "Topic message observed",
            )

            if previous_status == TopicStatus.STALE:
                self._audit(
                    record,
                    TopicEventType.TOPIC_RECOVERED,
                    "Topic recovered",
                )

            return record.to_dict()

    # --------------------------------------------------------
    # REFRESH TOPIC HEALTH
    # --------------------------------------------------------

    def _refresh(
        self,
        record: TopicRecord,
    ) -> None:
        now = time.monotonic()

        last = record.last_received_monotonic

        reference = (
            last
            if last is not None
            else record.created_monotonic
        )

        age = now - reference

        if (
            age
            > record.definition.freshness_timeout_seconds
        ):
            if record.status != TopicStatus.STALE:
                record.status = TopicStatus.STALE

                self._audit(
                    record,
                    TopicEventType.TOPIC_STALE,
                    "Topic freshness timeout",
                )

    # --------------------------------------------------------
    # GET TOPIC
    # --------------------------------------------------------

    def get_topic(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        topic_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                topic_id,
            )

            self._refresh(record)

            return record.to_dict()

    # --------------------------------------------------------
    # LIST TOPICS
    # --------------------------------------------------------

    def list_topics(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
        category: TopicCategory | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            results = []

            for record in self.topics.values():
                definition = record.definition

                if definition.tenant_id != tenant_id:
                    continue

                if (
                    robot_id is not None
                    and definition.robot_id != robot_id
                ):
                    continue

                if (
                    category is not None
                    and definition.category != category
                ):
                    continue

                self._refresh(record)

                results.append(
                    record.to_dict()
                )

            return sorted(
                results,
                key=lambda item: item["ros_topic_name"],
            )

    # --------------------------------------------------------
    # EVALUATE ROBOT TOPICS
    # --------------------------------------------------------

    def evaluate_robot_topics(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            critical_topics = []

            for record in self.topics.values():
                definition = record.definition

                if (
                    definition.tenant_id == tenant_id
                    and definition.robot_id == robot_id
                    and definition.critical
                ):
                    self._refresh(record)

                    critical_topics.append(record)

            unhealthy = [
                record.definition.topic_id
                for record in critical_topics
                if record.status != TopicStatus.ACTIVE
            ]

            ready = (
                bool(critical_topics)
                and not unhealthy
            )

            return {
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "ready_for_simulation": ready,
                "critical_topic_count": len(
                    critical_topics
                ),
                "unhealthy_topics": unhealthy,
                "evaluated_at": utc_now().isoformat(),
            }

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            records = [
                record
                for record in self.topics.values()
                if (
                    tenant_id is None
                    or record.definition.tenant_id
                    == tenant_id
                )
            ]

            for record in records:
                self._refresh(record)

            return {
                "total_topics": len(records),
                "active_topics": sum(
                    record.status == TopicStatus.ACTIVE
                    for record in records
                ),
                "stale_topics": sum(
                    record.status == TopicStatus.STALE
                    for record in records
                ),
                "total_publishers": sum(
                    len(record.publishers)
                    for record in records
                ),
                "total_subscribers": sum(
                    len(record.subscribers)
                    for record in records
                ),
                "total_messages": sum(
                    record.message_count
                    for record in records
                ),
                "total_bytes": sum(
                    record.total_bytes
                    for record in records
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
    manager = RoboticsTopicManager()

    manager.register_topic(
        RobotTopicDefinition(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            topic_id="camera",
            ros_topic_name="/robot_001/camera/image_raw",
            message_type="sensor_msgs/msg/Image",
            category=TopicCategory.CAMERA,
            qos=TopicQoS(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                depth=10,
            ),
            freshness_timeout_seconds=5.0,
            critical=True,
        )
    )

    manager.add_publisher(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        topic_id="camera",
        endpoint=TopicEndpoint(
            endpoint_id="camera_driver",
            node_id="camera_node",
            qos=TopicQoS(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                depth=10,
            ),
        ),
    )

    manager.add_subscriber(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        topic_id="camera",
        endpoint=TopicEndpoint(
            endpoint_id="perception_subscriber",
            node_id="perception_node",
            qos=TopicQoS(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                depth=10,
            ),
        ),
    )

    manager.observe_message(
        TopicMessageObservation(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            topic_id="camera",
            publisher_id="camera_driver",
            sequence_number=1,
            message_size_bytes=4096,
        )
    )

    print(
        "QoS compatibility:"
    )

    print(
        json.dumps(
            manager.check_qos(
                tenant_id="tenant_alpha",
                robot_id="robot_001",
                topic_id="camera",
            ),
            indent=2,
        )
    )

    print(
        "\nRobot topic health:"
    )

    print(
        json.dumps(
            manager.evaluate_robot_topics(
                tenant_id="tenant_alpha",
                robot_id="robot_001",
            ),
            indent=2,
        )
    )

    manager.export_json(
        "data/robotics/topic_manager_report.json",
        tenant_id="tenant_alpha",
    )


if __name__ == "__main__":
    main()

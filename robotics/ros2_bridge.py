"""Ros2 Bridge module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
backend/app/robotics/control/ros2_bridge.py

Python 3.11+
Standard library only.

ROS 2 BRIDGE MANAGER — SIMULATION ONLY

Features:
    - Multi-tenant ROS 2-style bridge
    - Simulated ROS 2 node registration
    - Topic publisher/subscriber routing
    - Service request/response handling
    - Action goal registration and execution
    - Message validation
    - Topic allowlisting
    - Sequence and timestamp validation
    - Heartbeat monitoring
    - Bridge health checks
    - Bounded message history
    - Audit logging
    - JSON report export

No ROS 2 middleware or physical robot commands
are used by this module.

Production integration requires an authenticated
rclpy adapter and independently enforced safety.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
import uuid

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable


# ============================================================
# EXCEPTIONS
# ============================================================

class ROS2BridgeError(Exception):
    pass


class ROS2BridgeValidationError(ROS2BridgeError):
    pass


class ROS2BridgeNotFound(ROS2BridgeError):
    pass


class ROS2BridgeConflict(ROS2BridgeError):
    pass


class ROS2BridgePermissionDenied(ROS2BridgeError):
    pass


# ============================================================
# ENUMS
# ============================================================

class ROS2BridgeStatus(str, Enum):
    REGISTERED = "registered"
    ACTIVE = "active"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    STOPPED = "stopped"


class ROS2MessageDirection(str, Enum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class ROS2ActionStatus(str, Enum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ROS2BridgeEventType(str, Enum):
    NODE_REGISTERED = "node_registered"
    NODE_STARTED = "node_started"
    NODE_STOPPED = "node_stopped"
    HEARTBEAT_RECEIVED = "heartbeat_received"
    HEARTBEAT_TIMEOUT = "heartbeat_timeout"
    MESSAGE_PUBLISHED = "message_published"
    MESSAGE_RECEIVED = "message_received"
    SERVICE_CALLED = "service_called"
    ACTION_SUBMITTED = "action_submitted"
    ACTION_COMPLETED = "action_completed"
    ACTION_CANCELLED = "action_cancelled"


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_id(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(
            r"[A-Za-z0-9_.-]{1,128}",
            value,
        )
    ):
        raise ROS2BridgeValidationError(
            f"Invalid {label}"
        )

    return value


def validate_ros_name(
    value: str,
    label: str,
) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(
            r"/[A-Za-z][A-Za-z0-9_/]*",
            value,
        )
        or len(value) > 256
        or "//" in value
        or value.endswith("/")
    ):
        raise ROS2BridgeValidationError(
            f"Invalid {label}"
        )

    return value


def validate_timestamp(
    value: datetime,
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ROS2BridgeValidationError(
            "Timezone-aware timestamp required"
        )

    return value.astimezone(timezone.utc)


def json_payload(
    value: dict[str, Any],
    maximum_bytes: int,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ROS2BridgeValidationError(
            "Payload must be an object"
        )

    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise ROS2BridgeValidationError(
            "Payload must be JSON serializable"
        ) from exc

    if len(encoded) > maximum_bytes:
        raise ROS2BridgeValidationError(
            "Payload size limit exceeded"
        )

    return json.loads(encoded)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ROS2BridgeConfig:
    maximum_nodes: int = 10000
    maximum_messages_per_node: int = 1000
    maximum_audit_events: int = 20000
    maximum_payload_bytes: int = 65536

    heartbeat_timeout_seconds: float = 5.0
    maximum_message_age_seconds: float = 5.0

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_nodes",
            "maximum_messages_per_node",
            "maximum_audit_events",
            "maximum_payload_bytes",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ROS2BridgeValidationError(
                    f"{name} must be positive"
                )

        for name in (
            "heartbeat_timeout_seconds",
            "maximum_message_age_seconds",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ROS2BridgeValidationError(
                    f"{name} must be positive and finite"
                )

        if not self.simulation_only:
            raise ROS2BridgeValidationError(
                "Physical ROS 2 bridge unsupported"
            )


# ============================================================
# NODE DEFINITION
# ============================================================

@dataclass(frozen=True)
class ROS2BridgeNodeDefinition:
    tenant_id: str
    robot_id: str
    node_id: str

    node_name: str
    namespace: str

    allowed_publish_topics: tuple[str, ...] = ()
    allowed_subscribe_topics: tuple[str, ...] = ()
    allowed_services: tuple[str, ...] = ()
    allowed_actions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "tenant_id",
            "robot_id",
            "node_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        validate_ros_name(
            self.node_name,
            "node_name",
        )

        validate_ros_name(
            self.namespace,
            "namespace",
        )

        for name in (
            "allowed_publish_topics",
            "allowed_subscribe_topics",
            "allowed_services",
            "allowed_actions",
        ):
            values = getattr(self, name)

            if not isinstance(values, tuple):
                raise ROS2BridgeValidationError(
                    f"{name} must be a tuple"
                )

            for value in values:
                validate_ros_name(
                    value,
                    name,
                )

            if len(values) != len(set(values)):
                raise ROS2BridgeValidationError(
                    f"Duplicate names in {name}"
                )


# ============================================================
# MESSAGE
# ============================================================

@dataclass(frozen=True)
class ROS2BridgeMessage:
    tenant_id: str
    robot_id: str
    node_id: str

    topic: str
    direction: ROS2MessageDirection

    sequence_number: int
    payload: dict[str, Any]

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for name in (
            "tenant_id",
            "robot_id",
            "node_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        validate_ros_name(
            self.topic,
            "topic",
        )

        if not isinstance(
            self.direction,
            ROS2MessageDirection,
        ):
            raise ROS2BridgeValidationError(
                "Invalid message direction"
            )

        if (
            isinstance(self.sequence_number, bool)
            or not isinstance(self.sequence_number, int)
            or self.sequence_number < 0
        ):
            raise ROS2BridgeValidationError(
                "Invalid message sequence"
            )

        if not isinstance(self.payload, dict):
            raise ROS2BridgeValidationError(
                "Payload must be a dictionary"
            )

        validate_timestamp(
            self.timestamp
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "node_id": self.node_id,
            "topic": self.topic,
            "direction": self.direction.value,
            "sequence_number": self.sequence_number,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================
# ACTION RECORD
# ============================================================

@dataclass
class ROS2BridgeActionRecord:
    action_id: str

    tenant_id: str
    robot_id: str
    node_id: str

    action_name: str

    goal: dict[str, Any]

    status: ROS2ActionStatus = (
        ROS2ActionStatus.ACCEPTED
    )

    result: dict[str, Any] | None = None

    created_at: datetime = field(
        default_factory=utc_now
    )

    finished_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "node_id": self.node_id,
            "action_name": self.action_name,
            "goal": self.goal,
            "status": self.status.value,
            "result": self.result,
            "created_at": (
                self.created_at.isoformat()
            ),
            "finished_at": (
                self.finished_at.isoformat()
                if self.finished_at
                else None
            ),
        }


# ============================================================
# NODE RECORD
# ============================================================

@dataclass
class ROS2BridgeNodeRecord:
    definition: ROS2BridgeNodeDefinition

    status: ROS2BridgeStatus = (
        ROS2BridgeStatus.REGISTERED
    )

    last_heartbeat_monotonic: float | None = None
    last_heartbeat_sequence: int = -1

    messages: deque[dict[str, Any]] = field(
        default_factory=deque
    )

    sequence_by_topic: dict[
        tuple[str, ROS2MessageDirection],
        int,
    ] = field(default_factory=dict)

    published_count: int = 0
    received_count: int = 0
    service_call_count: int = 0
    action_count: int = 0

    registered_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        definition = self.definition

        return {
            "tenant_id": definition.tenant_id,
            "robot_id": definition.robot_id,
            "node_id": definition.node_id,
            "node_name": definition.node_name,
            "namespace": definition.namespace,
            "status": self.status.value,
            "allowed_publish_topics": list(
                definition.allowed_publish_topics
            ),
            "allowed_subscribe_topics": list(
                definition.allowed_subscribe_topics
            ),
            "allowed_services": list(
                definition.allowed_services
            ),
            "allowed_actions": list(
                definition.allowed_actions
            ),
            "last_heartbeat_sequence": (
                self.last_heartbeat_sequence
            ),
            "published_count": self.published_count,
            "received_count": self.received_count,
            "service_call_count": (
                self.service_call_count
            ),
            "action_count": self.action_count,
            "message_history_count": len(
                self.messages
            ),
            "registered_at": (
                self.registered_at.isoformat()
            ),
        }


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class ROS2BridgeAuditEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    robot_id: str
    node_id: str

    event_type: ROS2BridgeEventType

    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "node_id": self.node_id,
            "event_type": self.event_type.value,
            "details": self.details,
        }


# ============================================================
# CALLBACK TYPES
# ============================================================

TopicSubscriber = Callable[
    [dict[str, Any]],
    None,
]

ServiceHandler = Callable[
    [dict[str, Any]],
    dict[str, Any],
]

ActionHandler = Callable[
    [dict[str, Any]],
    dict[str, Any],
]


# ============================================================
# ROS 2 BRIDGE MANAGER
# ============================================================

class RoboticsROS2Bridge:

    """
    Thread-safe simulation-only ROS 2-style bridge.

    The bridge provides local message routing and
    request/response behavior.

    It does not implement DDS, ROS 2 discovery,
    real rclpy nodes or hardware transport.
    """

    def __init__(
        self,
        config: ROS2BridgeConfig | None = None,
    ) -> None:
        self.config = (
            config or ROS2BridgeConfig()
        )

        self._nodes: dict[
            tuple[str, str, str],
            ROS2BridgeNodeRecord
        ] = {}

        self._subscribers: dict[
            tuple[str, str, str],
            list[TopicSubscriber],
        ] = {}

        self._services: dict[
            tuple[str, str, str, str],
            ServiceHandler,
        ] = {}

        self._action_handlers: dict[
            tuple[str, str, str, str],
            ActionHandler,
        ] = {}

        self._actions: dict[
            str,
            ROS2BridgeActionRecord,
        ] = {}

        self._audit_events: deque[
            ROS2BridgeAuditEvent
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------------

    def _key(
        self,
        tenant_id: str,
        robot_id: str,
        node_id: str,
    ) -> tuple[str, str, str]:
        return (
            validate_id(tenant_id, "tenant_id"),
            validate_id(robot_id, "robot_id"),
            validate_id(node_id, "node_id"),
        )

    def _get(
        self,
        tenant_id: str,
        robot_id: str,
        node_id: str,
    ) -> ROS2BridgeNodeRecord:
        record = self._nodes.get(
            self._key(
                tenant_id,
                robot_id,
                node_id,
            )
        )

        if record is None:
            raise ROS2BridgeNotFound(
                "ROS 2 bridge node not registered"
            )

        return record

    def _audit(
        self,
        record: ROS2BridgeNodeRecord,
        event_type: ROS2BridgeEventType,
        details: str,
    ) -> None:
        definition = record.definition

        self._audit_events.append(
            ROS2BridgeAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                tenant_id=definition.tenant_id,
                robot_id=definition.robot_id,
                node_id=definition.node_id,
                event_type=event_type,
                details=details,
            )
        )

    def _refresh(
        self,
        record: ROS2BridgeNodeRecord,
    ) -> None:
        if record.status in (
            ROS2BridgeStatus.REGISTERED,
            ROS2BridgeStatus.STOPPED,
        ):
            return

        last = record.last_heartbeat_monotonic

        if (
            last is None
            or time.monotonic() - last
            > self.config.heartbeat_timeout_seconds
        ):
            if record.status != ROS2BridgeStatus.OFFLINE:
                record.status = ROS2BridgeStatus.OFFLINE

                self._audit(
                    record,
                    ROS2BridgeEventType.HEARTBEAT_TIMEOUT,
                    "Node heartbeat timed out",
                )

    def _require_active(
        self,
        record: ROS2BridgeNodeRecord,
    ) -> None:
        self._refresh(record)

        if record.status != ROS2BridgeStatus.ACTIVE:
            raise ROS2BridgeConflict(
                "ROS 2 bridge node is not active"
            )

    # --------------------------------------------------------
    # REGISTER NODE
    # --------------------------------------------------------

    def register_node(
        self,
        definition: ROS2BridgeNodeDefinition,
    ) -> dict[str, Any]:
        if not isinstance(
            definition,
            ROS2BridgeNodeDefinition,
        ):
            raise ROS2BridgeValidationError(
                "Invalid node definition"
            )

        with self._lock:
            key = self._key(
                definition.tenant_id,
                definition.robot_id,
                definition.node_id,
            )

            if key in self._nodes:
                raise ROS2BridgeConflict(
                    "Node already registered"
                )

            if (
                len(self._nodes)
                >= self.config.maximum_nodes
            ):
                raise ROS2BridgeError(
                    "Node capacity reached"
                )

            record = ROS2BridgeNodeRecord(
                definition=definition,
                messages=deque(
                    maxlen=(
                        self.config.maximum_messages_per_node
                    )
                ),
            )

            self._nodes[key] = record

            self._audit(
                record,
                ROS2BridgeEventType.NODE_REGISTERED,
                "ROS 2-style node registered",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # START / STOP
    # --------------------------------------------------------

    def start_node(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        node_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                node_id,
            )

            if record.status not in (
                ROS2BridgeStatus.REGISTERED,
                ROS2BridgeStatus.STOPPED,
                ROS2BridgeStatus.OFFLINE,
            ):
                raise ROS2BridgeConflict(
                    "Node cannot be started"
                )

            record.status = ROS2BridgeStatus.ACTIVE
            record.last_heartbeat_monotonic = (
                time.monotonic()
            )

            self._audit(
                record,
                ROS2BridgeEventType.NODE_STARTED,
                "Simulation bridge node started",
            )

            return record.to_dict()

    def stop_node(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        node_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                node_id,
            )

            record.status = ROS2BridgeStatus.STOPPED

            self._audit(
                record,
                ROS2BridgeEventType.NODE_STOPPED,
                "Simulation bridge node stopped",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # HEARTBEAT
    # --------------------------------------------------------

    def heartbeat(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        node_id: str,
        sequence_number: int,
    ) -> dict[str, Any]:
        if (
            isinstance(sequence_number, bool)
            or not isinstance(sequence_number, int)
            or sequence_number < 0
        ):
            raise ROS2BridgeValidationError(
                "Invalid heartbeat sequence"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                node_id,
            )

            if record.status in (
                ROS2BridgeStatus.REGISTERED,
                ROS2BridgeStatus.STOPPED,
            ):
                raise ROS2BridgeConflict(
                    "Node is not started"
                )

            if (
                sequence_number
                <= record.last_heartbeat_sequence
            ):
                raise ROS2BridgeValidationError(
                    "Out-of-order heartbeat"
                )

            record.last_heartbeat_sequence = (
                sequence_number
            )

            record.last_heartbeat_monotonic = (
                time.monotonic()
            )

            record.status = ROS2BridgeStatus.ACTIVE

            self._audit(
                record,
                ROS2BridgeEventType.HEARTBEAT_RECEIVED,
                "Node heartbeat received",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # REGISTER SUBSCRIBER
    # --------------------------------------------------------

    def subscribe(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        node_id: str,
        topic: str,
        callback: TopicSubscriber,
    ) -> None:
        validate_ros_name(topic, "topic")

        if not callable(callback):
            raise ROS2BridgeValidationError(
                "Subscriber callback must be callable"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                node_id,
            )

            if (
                topic not in
                record.definition.allowed_subscribe_topics
            ):
                raise ROS2BridgePermissionDenied(
                    "Topic not allowed for subscription"
                )

            key = (
                tenant_id,
                robot_id,
                topic,
            )

            self._subscribers.setdefault(
                key,
                [],
            ).append(callback)

    # --------------------------------------------------------
    # MESSAGE ROUTING
    # --------------------------------------------------------

    def route_message(
        self,
        message: ROS2BridgeMessage,
    ) -> dict[str, Any]:
        if not isinstance(
            message,
            ROS2BridgeMessage,
        ):
            raise ROS2BridgeValidationError(
                "Invalid ROS 2 bridge message"
            )

        now = utc_now()

        age = (
            now
            - validate_timestamp(
                message.timestamp
            )
        ).total_seconds()

        if (
            age < 0
            or age
            > self.config.maximum_message_age_seconds
        ):
            raise ROS2BridgeValidationError(
                "Message timestamp invalid or stale"
            )

        safe_payload = json_payload(
            message.payload,
            self.config.maximum_payload_bytes,
        )

        with self._lock:
            record = self._get(
                message.tenant_id,
                message.robot_id,
                message.node_id,
            )

            self._require_active(record)

            if (
                message.direction
                == ROS2MessageDirection.OUTBOUND
            ):
                allowed = (
                    record.definition.allowed_publish_topics
                )
            else:
                allowed = (
                    record.definition.allowed_subscribe_topics
                )

            if message.topic not in allowed:
                raise ROS2BridgePermissionDenied(
                    "Topic not allowed for this direction"
                )

            sequence_key = (
                message.topic,
                message.direction,
            )

            previous_sequence = (
                record.sequence_by_topic.get(
                    sequence_key,
                    -1,
                )
            )

            if (
                message.sequence_number
                <= previous_sequence
            ):
                raise ROS2BridgeValidationError(
                    "Out-of-order topic message"
                )

            record.sequence_by_topic[
                sequence_key
            ] = message.sequence_number

            message_data = {
                **message.to_dict(),
                "payload": safe_payload,
            }

            record.messages.append(
                message_data
            )

            if (
                message.direction
                == ROS2MessageDirection.OUTBOUND
            ):
                record.published_count += 1

                event_type = (
                    ROS2BridgeEventType.MESSAGE_PUBLISHED
                )
            else:
                record.received_count += 1

                event_type = (
                    ROS2BridgeEventType.MESSAGE_RECEIVED
                )

            self._audit(
                record,
                event_type,
                f"Topic: {message.topic}",
            )

            callbacks = list(
                self._subscribers.get(
                    (
                        message.tenant_id,
                        message.robot_id,
                        message.topic,
                    ),
                    [],
                )
            )

        delivered = 0
        failed = 0

        # Execute callbacks outside the registry lock.
        for callback in callbacks:
            try:
                callback(
                    json.loads(
                        json.dumps(message_data)
                    )
                )
                delivered += 1
            except Exception:
                failed += 1

        return {
            "topic": message.topic,
            "direction": message.direction.value,
            "sequence_number": (
                message.sequence_number
            ),
            "subscriber_deliveries": delivered,
            "subscriber_failures": failed,
            "simulation_only": True,
        }

    # --------------------------------------------------------
    # SERVICE REGISTRATION
    # --------------------------------------------------------

    def register_service(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        node_id: str,
        service_name: str,
        handler: ServiceHandler,
    ) -> None:
        validate_ros_name(
            service_name,
            "service_name",
        )

        if not callable(handler):
            raise ROS2BridgeValidationError(
                "Service handler must be callable"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                node_id,
            )

            if (
                service_name
                not in record.definition.allowed_services
            ):
                raise ROS2BridgePermissionDenied(
                    "Service not allowlisted"
                )

            key = (
                tenant_id,
                robot_id,
                node_id,
                service_name,
            )

            if key in self._services:
                raise ROS2BridgeConflict(
                    "Service already registered"
                )

            self._services[key] = handler

    # --------------------------------------------------------
    # CALL SERVICE
    # --------------------------------------------------------

    def call_service(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        node_id: str,
        service_name: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        validate_ros_name(
            service_name,
            "service_name",
        )

        safe_request = json_payload(
            request,
            self.config.maximum_payload_bytes,
        )

        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                node_id,
            )

            self._require_active(record)

            if (
                service_name
                not in record.definition.allowed_services
            ):
                raise ROS2BridgePermissionDenied(
                    "Service not allowlisted"
                )

            handler = self._services.get(
                (
                    tenant_id,
                    robot_id,
                    node_id,
                    service_name,
                )
            )

            if handler is None:
                raise ROS2BridgeNotFound(
                    "Service handler unavailable"
                )

        # Handler runs outside the lock.
        try:
            response = handler(safe_request)
        except Exception as exc:
            raise ROS2BridgeError(
                "Simulation service handler failed"
            ) from exc

        safe_response = json_payload(
            response,
            self.config.maximum_payload_bytes,
        )

        with self._lock:
            record.service_call_count += 1

            self._audit(
                record,
                ROS2BridgeEventType.SERVICE_CALLED,
                f"Service: {service_name}",
            )

        return {
            "service_name": service_name,
            "response": safe_response,
            "simulation_only": True,
        }

    # --------------------------------------------------------
    # REGISTER ACTION HANDLER
    # --------------------------------------------------------

    def register_action_handler(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        node_id: str,
        action_name: str,
        handler: ActionHandler,
    ) -> None:
        validate_ros_name(
            action_name,
            "action_name",
        )

        if not callable(handler):
            raise ROS2BridgeValidationError(
                "Action handler must be callable"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                node_id,
            )

            if (
                action_name
                not in record.definition.allowed_actions
            ):
                raise ROS2BridgePermissionDenied(
                    "Action not allowlisted"
                )

            key = (
                tenant_id,
                robot_id,
                node_id,
                action_name,
            )

            if key in self._action_handlers:
                raise ROS2BridgeConflict(
                    "Action handler already registered"
                )

            self._action_handlers[key] = handler

    # --------------------------------------------------------
    # SUBMIT ACTION
    # --------------------------------------------------------

    def submit_action(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        node_id: str,
        action_name: str,
        goal: dict[str, Any],
    ) -> dict[str, Any]:
        validate_ros_name(
            action_name,
            "action_name",
        )

        safe_goal = json_payload(
            goal,
            self.config.maximum_payload_bytes,
        )

        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                node_id,
            )

            self._require_active(record)

            if (
                action_name
                not in record.definition.allowed_actions
            ):
                raise ROS2BridgePermissionDenied(
                    "Action not allowlisted"
                )

            handler = self._action_handlers.get(
                (
                    tenant_id,
                    robot_id,
                    node_id,
                    action_name,
                )
            )

            if handler is None:
                raise ROS2BridgeNotFound(
                    "Action handler unavailable"
                )

            action = ROS2BridgeActionRecord(
                action_id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                robot_id=robot_id,
                node_id=node_id,
                action_name=action_name,
                goal=safe_goal,
            )

            self._actions[
                action.action_id
            ] = action

            record.action_count += 1

            self._audit(
                record,
                ROS2BridgeEventType.ACTION_SUBMITTED,
                f"Action: {action_name}",
            )

            action.status = ROS2ActionStatus.RUNNING

        # Synchronous simulation action handler.
        try:
            result = json_payload(
                handler(safe_goal),
                self.config.maximum_payload_bytes,
            )

            final_status = ROS2ActionStatus.SUCCEEDED

        except Exception:
            result = {
                "error": "Simulation action failed"
            }

            final_status = ROS2ActionStatus.FAILED

        with self._lock:
            action.status = final_status
            action.result = result
            action.finished_at = utc_now()

            self._audit(
                record,
                ROS2BridgeEventType.ACTION_COMPLETED,
                (
                    f"Action {action_name}: "
                    f"{final_status.value}"
                ),
            )

            return action.to_dict()

    # --------------------------------------------------------
    # GET ACTION
    # --------------------------------------------------------

    def get_action(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        action_id: str,
    ) -> dict[str, Any]:
        validate_id(tenant_id, "tenant_id")
        validate_id(robot_id, "robot_id")

        with self._lock:
            action = self._actions.get(action_id)

            if (
                action is None
                or action.tenant_id != tenant_id
                or action.robot_id != robot_id
            ):
                raise ROS2BridgeNotFound(
                    "Action not found"
                )

            return action.to_dict()

    # --------------------------------------------------------
    # GET NODE
    # --------------------------------------------------------

    def get_node(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        node_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                node_id,
            )

            self._refresh(record)

            return record.to_dict()

    # --------------------------------------------------------
    # MESSAGE HISTORY
    # --------------------------------------------------------

    def get_message_history(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        node_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit <= 0
        ):
            raise ROS2BridgeValidationError(
                "Invalid history limit"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                node_id,
            )

            return [
                json.loads(json.dumps(item))
                for item in list(record.messages)[-limit:]
            ]

    # --------------------------------------------------------
    # LIST NODES
    # --------------------------------------------------------

    def list_nodes(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        if robot_id is not None:
            validate_id(robot_id, "robot_id")

        with self._lock:
            result = []

            for record in self._nodes.values():
                definition = record.definition

                if definition.tenant_id != tenant_id:
                    continue

                if (
                    robot_id is not None
                    and definition.robot_id != robot_id
                ):
                    continue

                self._refresh(record)

                result.append(
                    record.to_dict()
                )

            return sorted(
                result,
                key=lambda item: (
                    item["robot_id"],
                    item["node_id"],
                ),
            )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        tenant_id: str,
    ) -> dict[str, Any]:
        nodes = self.list_nodes(
            tenant_id=tenant_id
        )

        return {
            "tenant_id": tenant_id,
            "total_nodes": len(nodes),
            "active_nodes": sum(
                node["status"]
                == ROS2BridgeStatus.ACTIVE.value
                for node in nodes
            ),
            "offline_nodes": sum(
                node["status"]
                == ROS2BridgeStatus.OFFLINE.value
                for node in nodes
            ),
            "published_messages": sum(
                node["published_count"]
                for node in nodes
            ),
            "received_messages": sum(
                node["received_count"]
                for node in nodes
            ),
            "service_calls": sum(
                node["service_call_count"]
                for node in nodes
            ),
            "actions": sum(
                node["action_count"]
                for node in nodes
            ),
            "simulation_only": True,
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
        validate_id(tenant_id, "tenant_id")

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
                "nodes": self.list_nodes(
                    tenant_id=tenant_id
                ),
                "actions": [
                    action.to_dict()
                    for action in self._actions.values()
                    if action.tenant_id == tenant_id
                ],
                "audit_events": [
                    event.to_dict()
                    for event in self._audit_events
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
    bridge = RoboticsROS2Bridge()

    tenant_id = "tenant_alpha"
    robot_id = "robot_001"
    node_id = "robot_bridge"

    bridge.register_node(
        ROS2BridgeNodeDefinition(
            tenant_id=tenant_id,
            robot_id=robot_id,
            node_id=node_id,
            node_name="/robot_bridge",
            namespace="/robot_001",
            allowed_publish_topics=(
                "/robot_001/telemetry",
            ),
            allowed_subscribe_topics=(
                "/robot_001/telemetry",
            ),
            allowed_services=(
                "/robot_001/get_status",
            ),
            allowed_actions=(
                "/robot_001/simulate_inspection",
            ),
        )
    )

    bridge.start_node(
        tenant_id=tenant_id,
        robot_id=robot_id,
        node_id=node_id,
    )

    bridge.heartbeat(
        tenant_id=tenant_id,
        robot_id=robot_id,
        node_id=node_id,
        sequence_number=1,
    )

    bridge.subscribe(
        tenant_id=tenant_id,
        robot_id=robot_id,
        node_id=node_id,
        topic="/robot_001/telemetry",
        callback=lambda message: print(
            "Received topic:",
            message["topic"],
        ),
    )

    message_result = bridge.route_message(
        ROS2BridgeMessage(
            tenant_id=tenant_id,
            robot_id=robot_id,
            node_id=node_id,
            topic="/robot_001/telemetry",
            direction=ROS2MessageDirection.OUTBOUND,
            sequence_number=1,
            payload={
                "battery_percentage": 87.0,
                "temperature_c": 35.0,
                "operational_state": "idle",
            },
        )
    )

    print(
        "Message result:",
        json.dumps(message_result, indent=2),
    )

    bridge.register_service(
        tenant_id=tenant_id,
        robot_id=robot_id,
        node_id=node_id,
        service_name="/robot_001/get_status",
        handler=lambda request: {
            "robot_id": robot_id,
            "status": "simulation_ready",
        },
    )

    print(
        "Service response:",
        json.dumps(
            bridge.call_service(
                tenant_id=tenant_id,
                robot_id=robot_id,
                node_id=node_id,
                service_name="/robot_001/get_status",
                request={},
            ),
            indent=2,
        ),
    )

    bridge.register_action_handler(
        tenant_id=tenant_id,
        robot_id=robot_id,
        node_id=node_id,
        action_name=(
            "/robot_001/simulate_inspection"
        ),
        handler=lambda goal: {
            "completed": True,
            "inspection_target": goal["target"],
            "defects_detected": 0,
        },
    )

    print(
        "Action result:",
        json.dumps(
            bridge.submit_action(
                tenant_id=tenant_id,
                robot_id=robot_id,
                node_id=node_id,
                action_name=(
                    "/robot_001/simulate_inspection"
                ),
                goal={
                    "target": "motor_001"
                },
            ),
            indent=2,
        ),
    )

    report = bridge.export_json(
        "data/robotics/ros2_bridge_report.json",
        tenant_id=tenant_id,
    )

    print("\nReport:", report)


if __name__ == "__main__":
    main()


"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
backend/app/robotics/control/ros2_health.py

Python 3.11+
Standard library only.

ROS 2 Health Monitoring — Simulation Adapter

Features:
- ROS 2 node registration
- Node heartbeat monitoring
- Topic freshness monitoring
- Service availability monitoring
- Lifecycle state tracking
- Critical dependency assessment
- Node fault latching
- Robot health aggregation
- Tenant isolation
- Monotonic sequence validation
- Stale telemetry detection
- Thread-safe state management
- Audit logging
- JSON report export

SAFETY:
    This is a logical ROS 2 health monitor.

    It does not connect to the ROS 2 graph,
    inspect DDS discovery, restart nodes,
    execute robot commands, or provide
    safety-rated supervision.

    Health observations are caller-supplied.
    Production deployments must authenticate
    and independently verify their sources.
"""

from __future__ import annotations

import json
import math
import threading
import time
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


# ============================================================
# EXCEPTIONS
# ============================================================

class ROS2HealthError(Exception):
    pass


class ROS2ValidationError(ROS2HealthError):
    pass


class ROS2ResourceNotFound(ROS2HealthError):
    pass


class ROS2StateError(ROS2HealthError):
    pass


# ============================================================
# ENUMS
# ============================================================

class ROS2NodeType(str, Enum):
    PERCEPTION = "perception"
    NAVIGATION = "navigation"
    PLANNING = "planning"
    MANIPULATION = "manipulation"
    SENSOR_FUSION = "sensor_fusion"
    TELEMETRY = "telemetry"
    VLA_INFERENCE = "vla_inference"
    ACTION_CLIENT = "action_client"
    ROBOT_CONTROLLER = "robot_controller"
    SAFETY_MONITOR = "safety_monitor"
    WATCHDOG = "watchdog"
    SIMULATOR = "simulator"
    OTHER = "other"


class ROS2LifecycleState(str, Enum):
    UNKNOWN = "unknown"
    UNCONFIGURED = "unconfigured"
    INACTIVE = "inactive"
    ACTIVE = "active"
    FINALIZED = "finalized"
    ERROR = "error"


class ROS2HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ROS2ResourceType(str, Enum):
    NODE = "node"
    TOPIC = "topic"
    SERVICE = "service"


class ROS2HealthDecision(str, Enum):
    READY_FOR_SIMULATION = "ready_for_simulation"
    DENY = "deny"


class ROS2HealthEventType(str, Enum):
    NODE_REGISTERED = "node_registered"
    TOPIC_REGISTERED = "topic_registered"
    SERVICE_REGISTERED = "service_registered"
    HEARTBEAT_RECEIVED = "heartbeat_received"
    TOPIC_MESSAGE_RECEIVED = "topic_message_received"
    SERVICE_STATUS_UPDATED = "service_status_updated"
    NODE_DEGRADED = "node_degraded"
    NODE_TIMEOUT = "node_timeout"
    TOPIC_STALE = "topic_stale"
    SERVICE_UNAVAILABLE = "service_unavailable"
    FAULT_LATCHED = "fault_latched"
    FAULT_RESET = "fault_reset"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_id(
    value: str,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not all(
            char.isalnum()
            or char in "_-"
            for char in value
        )
    ):
        raise ROS2ValidationError(
            f"Invalid {name}"
        )

    return value


def validate_ros_name(
    value: str,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or len(value) > 256
        or "//" in value
        or not all(
            char.isalnum()
            or char in "_/"
            for char in value
        )
    ):
        raise ROS2ValidationError(
            f"Invalid {name}"
        )

    return value


def positive_number(
    value: float,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ROS2ValidationError(
            f"{name} must be finite and positive"
        )

    return float(value)


def validate_timestamp(
    value: datetime,
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ROS2ValidationError(
            "Timezone-aware timestamp required"
        )

    return value.astimezone(timezone.utc)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ROS2HealthConfig:
    default_heartbeat_timeout_seconds: float = 3.0
    default_topic_timeout_seconds: float = 5.0

    maximum_nodes: int = 10000
    maximum_topics: int = 20000
    maximum_services: int = 10000
    maximum_audit_events: int = 20000

    simulation_only: bool = True

    def __post_init__(self) -> None:
        positive_number(
            self.default_heartbeat_timeout_seconds,
            "default_heartbeat_timeout_seconds",
        )

        positive_number(
            self.default_topic_timeout_seconds,
            "default_topic_timeout_seconds",
        )

        for name in (
            "maximum_nodes",
            "maximum_topics",
            "maximum_services",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ROS2ValidationError(
                    f"{name} must be positive"
                )

        if not self.simulation_only:
            raise ROS2ValidationError(
                "Physical ROS 2 supervision unsupported"
            )


# ============================================================
# RESOURCE DEFINITIONS
# ============================================================

@dataclass(frozen=True)
class ROS2NodeDefinition:
    tenant_id: str
    robot_id: str
    node_id: str

    ros_name: str
    node_type: ROS2NodeType

    critical: bool = True

    heartbeat_timeout_seconds: float = 3.0

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")
        validate_id(self.node_id, "node_id")

        validate_ros_name(
            self.ros_name,
            "ROS node name",
        )

        if not isinstance(self.node_type, ROS2NodeType):
            raise ROS2ValidationError(
                "Invalid node type"
            )

        positive_number(
            self.heartbeat_timeout_seconds,
            "heartbeat_timeout_seconds",
        )


@dataclass(frozen=True)
class ROS2TopicDefinition:
    tenant_id: str
    robot_id: str
    topic_id: str

    ros_name: str

    critical: bool = True
    timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")
        validate_id(self.topic_id, "topic_id")

        validate_ros_name(
            self.ros_name,
            "ROS topic name",
        )

        positive_number(
            self.timeout_seconds,
            "timeout_seconds",
        )


@dataclass(frozen=True)
class ROS2ServiceDefinition:
    tenant_id: str
    robot_id: str
    service_id: str

    ros_name: str
    critical: bool = True

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")
        validate_id(self.service_id, "service_id")

        validate_ros_name(
            self.ros_name,
            "ROS service name",
        )


# ============================================================
# OBSERVATIONS
# ============================================================

@dataclass(frozen=True)
class ROS2NodeHeartbeat:
    tenant_id: str
    robot_id: str
    node_id: str

    sequence_number: int

    lifecycle_state: ROS2LifecycleState = (
        ROS2LifecycleState.ACTIVE
    )

    healthy: bool = True

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")
        validate_id(self.node_id, "node_id")

        if (
            isinstance(self.sequence_number, bool)
            or not isinstance(self.sequence_number, int)
            or self.sequence_number < 0
        ):
            raise ROS2ValidationError(
                "Invalid heartbeat sequence"
            )

        if not isinstance(
            self.lifecycle_state,
            ROS2LifecycleState,
        ):
            raise ROS2ValidationError(
                "Invalid lifecycle state"
            )

        if not isinstance(self.healthy, bool):
            raise ROS2ValidationError(
                "healthy must be boolean"
            )

        validate_timestamp(self.timestamp)


# ============================================================
# RESOURCE RECORDS
# ============================================================

@dataclass
class ROS2NodeRecord:
    definition: ROS2NodeDefinition

    lifecycle_state: ROS2LifecycleState = (
        ROS2LifecycleState.UNKNOWN
    )

    health: ROS2HealthStatus = (
        ROS2HealthStatus.UNKNOWN
    )

    last_sequence_number: int = -1
    last_heartbeat_at: datetime | None = None
    last_received_monotonic: float | None = None

    fault_latched: bool = False

    heartbeat_count: int = 0
    timeout_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.definition.tenant_id,
            "robot_id": self.definition.robot_id,
            "node_id": self.definition.node_id,
            "ros_name": self.definition.ros_name,
            "node_type": (
                self.definition.node_type.value
            ),
            "critical": self.definition.critical,
            "lifecycle_state": (
                self.lifecycle_state.value
            ),
            "health": self.health.value,
            "fault_latched": self.fault_latched,
            "heartbeat_count": self.heartbeat_count,
            "timeout_count": self.timeout_count,
            "last_sequence_number": (
                self.last_sequence_number
            ),
            "last_heartbeat_at": (
                self.last_heartbeat_at.isoformat()
                if self.last_heartbeat_at
                else None
            ),
        }


@dataclass
class ROS2TopicRecord:
    definition: ROS2TopicDefinition

    last_message_at: datetime | None = None
    last_received_monotonic: float | None = None

    message_count: int = 0

    health: ROS2HealthStatus = (
        ROS2HealthStatus.UNKNOWN
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.definition.tenant_id,
            "robot_id": self.definition.robot_id,
            "topic_id": self.definition.topic_id,
            "ros_name": self.definition.ros_name,
            "critical": self.definition.critical,
            "health": self.health.value,
            "message_count": self.message_count,
            "last_message_at": (
                self.last_message_at.isoformat()
                if self.last_message_at
                else None
            ),
        }


@dataclass
class ROS2ServiceRecord:
    definition: ROS2ServiceDefinition

    available: bool = False

    health: ROS2HealthStatus = (
        ROS2HealthStatus.UNKNOWN
    )

    last_checked_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.definition.tenant_id,
            "robot_id": self.definition.robot_id,
            "service_id": self.definition.service_id,
            "ros_name": self.definition.ros_name,
            "critical": self.definition.critical,
            "available": self.available,
            "health": self.health.value,
            "last_checked_at": (
                self.last_checked_at.isoformat()
                if self.last_checked_at
                else None
            ),
        }


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class ROS2HealthAuditEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    robot_id: str

    resource_id: str
    resource_type: ROS2ResourceType

    event_type: ROS2HealthEventType
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "resource_id": self.resource_id,
            "resource_type": (
                self.resource_type.value
            ),
            "event_type": self.event_type.value,
            "details": self.details,
        }


# ============================================================
# ROS 2 HEALTH MONITOR
# ============================================================

class RoboticsROS2HealthMonitor:

    def __init__(
        self,
        config: ROS2HealthConfig | None = None,
    ) -> None:
        self.config = (
            config or ROS2HealthConfig()
        )

        self.nodes: dict[
            tuple[str, str, str],
            ROS2NodeRecord
        ] = {}

        self.topics: dict[
            tuple[str, str, str],
            ROS2TopicRecord
        ] = {}

        self.services: dict[
            tuple[str, str, str],
            ROS2ServiceRecord
        ] = {}

        self.audit_events: list[
            ROS2HealthAuditEvent
        ] = []

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------

    def _key(
        self,
        tenant_id: str,
        robot_id: str,
        resource_id: str,
    ) -> tuple[str, str, str]:
        return (
            validate_id(tenant_id, "tenant_id"),
            validate_id(robot_id, "robot_id"),
            validate_id(resource_id, "resource_id"),
        )

    def _audit(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        resource_id: str,
        resource_type: ROS2ResourceType,
        event_type: ROS2HealthEventType,
        details: str,
    ) -> None:
        self.audit_events.append(
            ROS2HealthAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                tenant_id=tenant_id,
                robot_id=robot_id,
                resource_id=resource_id,
                resource_type=resource_type,
                event_type=event_type,
                details=details,
            )
        )

        if (
            len(self.audit_events)
            > self.config.maximum_audit_events
        ):
            self.audit_events.pop(0)

    # --------------------------------------------------------
    # REGISTER NODE
    # --------------------------------------------------------

    def register_node(
        self,
        definition: ROS2NodeDefinition,
    ) -> dict[str, Any]:
        with self._lock:
            key = self._key(
                definition.tenant_id,
                definition.robot_id,
                definition.node_id,
            )

            if key in self.nodes:
                raise ROS2ValidationError(
                    "Node already registered"
                )

            if (
                len(self.nodes)
                >= self.config.maximum_nodes
            ):
                raise ROS2HealthError(
                    "Maximum node capacity reached"
                )

            record = ROS2NodeRecord(
                definition=definition
            )

            self.nodes[key] = record

            self._audit(
                tenant_id=definition.tenant_id,
                robot_id=definition.robot_id,
                resource_id=definition.node_id,
                resource_type=ROS2ResourceType.NODE,
                event_type=(
                    ROS2HealthEventType.NODE_REGISTERED
                ),
                details="ROS 2 node registered",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # REGISTER TOPIC
    # --------------------------------------------------------

    def register_topic(
        self,
        definition: ROS2TopicDefinition,
    ) -> dict[str, Any]:
        with self._lock:
            key = self._key(
                definition.tenant_id,
                definition.robot_id,
                definition.topic_id,
            )

            if key in self.topics:
                raise ROS2ValidationError(
                    "Topic already registered"
                )

            if (
                len(self.topics)
                >= self.config.maximum_topics
            ):
                raise ROS2HealthError(
                    "Maximum topic capacity reached"
                )

            record = ROS2TopicRecord(
                definition=definition
            )

            self.topics[key] = record

            self._audit(
                tenant_id=definition.tenant_id,
                robot_id=definition.robot_id,
                resource_id=definition.topic_id,
                resource_type=ROS2ResourceType.TOPIC,
                event_type=(
                    ROS2HealthEventType.TOPIC_REGISTERED
                ),
                details="ROS 2 topic registered",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # REGISTER SERVICE
    # --------------------------------------------------------

    def register_service(
        self,
        definition: ROS2ServiceDefinition,
    ) -> dict[str, Any]:
        with self._lock:
            key = self._key(
                definition.tenant_id,
                definition.robot_id,
                definition.service_id,
            )

            if key in self.services:
                raise ROS2ValidationError(
                    "Service already registered"
                )

            if (
                len(self.services)
                >= self.config.maximum_services
            ):
                raise ROS2HealthError(
                    "Maximum service capacity reached"
                )

            record = ROS2ServiceRecord(
                definition=definition
            )

            self.services[key] = record

            self._audit(
                tenant_id=definition.tenant_id,
                robot_id=definition.robot_id,
                resource_id=definition.service_id,
                resource_type=ROS2ResourceType.SERVICE,
                event_type=(
                    ROS2HealthEventType.SERVICE_REGISTERED
                ),
                details="ROS 2 service registered",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # NODE HEARTBEAT
    # --------------------------------------------------------

    def heartbeat(
        self,
        observation: ROS2NodeHeartbeat,
    ) -> dict[str, Any]:
        with self._lock:
            key = self._key(
                observation.tenant_id,
                observation.robot_id,
                observation.node_id,
            )

            record = self.nodes.get(key)

            if record is None:
                raise ROS2ResourceNotFound(
                    "ROS 2 node not found"
                )

            if (
                observation.sequence_number
                <= record.last_sequence_number
            ):
                raise ROS2ValidationError(
                    "Duplicate or out-of-order heartbeat"
                )

            record.last_sequence_number = (
                observation.sequence_number
            )

            record.last_heartbeat_at = (
                observation.timestamp
            )

            record.last_received_monotonic = (
                time.monotonic()
            )

            record.heartbeat_count += 1

            record.lifecycle_state = (
                observation.lifecycle_state
            )

            if record.fault_latched:
                record.health = (
                    ROS2HealthStatus.UNHEALTHY
                )

            elif (
                observation.healthy
                and observation.lifecycle_state
                == ROS2LifecycleState.ACTIVE
            ):
                record.health = (
                    ROS2HealthStatus.HEALTHY
                )

            elif (
                observation.lifecycle_state
                in {
                    ROS2LifecycleState.ERROR,
                    ROS2LifecycleState.FINALIZED,
                }
            ):
                record.health = (
                    ROS2HealthStatus.UNHEALTHY
                )

            else:
                record.health = (
                    ROS2HealthStatus.DEGRADED
                )

            self._audit(
                tenant_id=observation.tenant_id,
                robot_id=observation.robot_id,
                resource_id=observation.node_id,
                resource_type=ROS2ResourceType.NODE,
                event_type=(
                    ROS2HealthEventType.HEARTBEAT_RECEIVED
                ),
                details="ROS 2 node heartbeat received",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # TOPIC MESSAGE OBSERVATION
    # --------------------------------------------------------

    def observe_topic(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        topic_id: str,
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            key = self._key(
                tenant_id,
                robot_id,
                topic_id,
            )

            record = self.topics.get(key)

            if record is None:
                raise ROS2ResourceNotFound(
                    "ROS 2 topic not found"
                )

            observed_at = (
                timestamp or utc_now()
            )

            validate_timestamp(
                observed_at
            )

            record.last_message_at = observed_at
            record.last_received_monotonic = (
                time.monotonic()
            )

            record.message_count += 1
            record.health = (
                ROS2HealthStatus.HEALTHY
            )

            self._audit(
                tenant_id=tenant_id,
                robot_id=robot_id,
                resource_id=topic_id,
                resource_type=ROS2ResourceType.TOPIC,
                event_type=(
                    ROS2HealthEventType.TOPIC_MESSAGE_RECEIVED
                ),
                details="ROS 2 topic message observed",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # SERVICE AVAILABILITY
    # --------------------------------------------------------

    def update_service(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        service_id: str,
        available: bool,
    ) -> dict[str, Any]:
        if not isinstance(available, bool):
            raise ROS2ValidationError(
                "available must be boolean"
            )

        with self._lock:
            key = self._key(
                tenant_id,
                robot_id,
                service_id,
            )

            record = self.services.get(key)

            if record is None:
                raise ROS2ResourceNotFound(
                    "ROS 2 service not found"
                )

            record.available = available

            record.health = (
                ROS2HealthStatus.HEALTHY
                if available
                else ROS2HealthStatus.UNHEALTHY
            )

            record.last_checked_at = (
                utc_now()
            )

            self._audit(
                tenant_id=tenant_id,
                robot_id=robot_id,
                resource_id=service_id,
                resource_type=ROS2ResourceType.SERVICE,
                event_type=(
                    ROS2HealthEventType.SERVICE_STATUS_UPDATED
                ),
                details=(
                    "Available"
                    if available
                    else "Unavailable"
                ),
            )

            return record.to_dict()

    # --------------------------------------------------------
    # REFRESH HEALTH
    # --------------------------------------------------------

    def _refresh_nodes(
        self,
        now_monotonic: float,
    ) -> None:
        for record in self.nodes.values():
            if record.last_received_monotonic is None:
                continue

            age = (
                now_monotonic
                - record.last_received_monotonic
            )

            if (
                age
                >= record.definition.heartbeat_timeout_seconds
            ):
                if (
                    record.health
                    != ROS2HealthStatus.UNHEALTHY
                ):
                    record.timeout_count += 1

                    self._audit(
                        tenant_id=record.definition.tenant_id,
                        robot_id=record.definition.robot_id,
                        resource_id=record.definition.node_id,
                        resource_type=ROS2ResourceType.NODE,
                        event_type=(
                            ROS2HealthEventType.NODE_TIMEOUT
                        ),
                        details="Node heartbeat timeout",
                    )

                record.health = (
                    ROS2HealthStatus.UNHEALTHY
                )

                if record.definition.critical:
                    record.fault_latched = True

    def _refresh_topics(
        self,
        now_monotonic: float,
    ) -> None:
        for record in self.topics.values():
            if record.last_received_monotonic is None:
                continue

            age = (
                now_monotonic
                - record.last_received_monotonic
            )

            if (
                age
                >= record.definition.timeout_seconds
            ):
                if (
                    record.health
                    != ROS2HealthStatus.UNHEALTHY
                ):
                    self._audit(
                        tenant_id=record.definition.tenant_id,
                        robot_id=record.definition.robot_id,
                        resource_id=record.definition.topic_id,
                        resource_type=ROS2ResourceType.TOPIC,
                        event_type=(
                            ROS2HealthEventType.TOPIC_STALE
                        ),
                        details="Topic data stale",
                    )

                record.health = (
                    ROS2HealthStatus.UNHEALTHY
                )

    # --------------------------------------------------------
    # SCAN HEALTH
    # --------------------------------------------------------

    def scan_health(
        self,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()

            self._refresh_nodes(now)
            self._refresh_topics(now)

            return {
                "nodes": [
                    record.to_dict()
                    for record in self.nodes.values()
                    if (
                        tenant_id is None
                        or record.definition.tenant_id
                        == tenant_id
                    )
                ],
                "topics": [
                    record.to_dict()
                    for record in self.topics.values()
                    if (
                        tenant_id is None
                        or record.definition.tenant_id
                        == tenant_id
                    )
                ],
                "services": [
                    record.to_dict()
                    for record in self.services.values()
                    if (
                        tenant_id is None
                        or record.definition.tenant_id
                        == tenant_id
                    )
                ],
            }

    # --------------------------------------------------------
    # ROBOT HEALTH ASSESSMENT
    # --------------------------------------------------------

    def evaluate_robot(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            self._refresh_nodes(
                time.monotonic()
            )

            self._refresh_topics(
                time.monotonic()
            )

            resources = []

            for record in self.nodes.values():
                if (
                    record.definition.tenant_id == tenant_id
                    and record.definition.robot_id == robot_id
                    and record.definition.critical
                ):
                    resources.append(
                        (
                            record.definition.node_id,
                            record.health,
                        )
                    )

            for record in self.topics.values():
                if (
                    record.definition.tenant_id == tenant_id
                    and record.definition.robot_id == robot_id
                    and record.definition.critical
                ):
                    resources.append(
                        (
                            record.definition.topic_id,
                            record.health,
                        )
                    )

            for record in self.services.values():
                if (
                    record.definition.tenant_id == tenant_id
                    and record.definition.robot_id == robot_id
                    and record.definition.critical
                ):
                    resources.append(
                        (
                            record.definition.service_id,
                            record.health,
                        )
                    )

            ready = (
                bool(resources)
                and all(
                    health == ROS2HealthStatus.HEALTHY
                    for _, health in resources
                )
            )

            return {
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "decision": (
                    ROS2HealthDecision.READY_FOR_SIMULATION.value
                    if ready
                    else ROS2HealthDecision.DENY.value
                ),
                "critical_resource_count": len(
                    resources
                ),
                "unhealthy_resources": [
                    resource_id
                    for resource_id, health in resources
                    if health != ROS2HealthStatus.HEALTHY
                ],
                "evaluated_at": (
                    utc_now().isoformat()
                ),
            }

    # --------------------------------------------------------
    # RESET LATCHED NODE FAULT
    # --------------------------------------------------------

    def reset_node_fault(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        node_id: str,
        authorized: bool,
        independently_verified: bool,
    ) -> dict[str, Any]:
        with self._lock:
            key = self._key(
                tenant_id,
                robot_id,
                node_id,
            )

            record = self.nodes.get(key)

            if record is None:
                raise ROS2ResourceNotFound(
                    "Node not found"
                )

            if (
                not authorized
                or not independently_verified
            ):
                raise ROS2StateError(
                    "Verified recovery approval required"
                )

            record.fault_latched = False
            record.health = (
                ROS2HealthStatus.UNKNOWN
            )
            record.last_received_monotonic = None

            self._audit(
                tenant_id=tenant_id,
                robot_id=robot_id,
                resource_id=node_id,
                resource_type=ROS2ResourceType.NODE,
                event_type=(
                    ROS2HealthEventType.FAULT_RESET
                ),
                details="Logical node fault reset",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        snapshot = self.scan_health(
            tenant_id=tenant_id
        )

        resources = (
            snapshot["nodes"]
            + snapshot["topics"]
            + snapshot["services"]
        )

        return {
            "total_nodes": len(snapshot["nodes"]),
            "total_topics": len(snapshot["topics"]),
            "total_services": len(snapshot["services"]),
            "healthy": sum(
                resource["health"]
                == ROS2HealthStatus.HEALTHY.value
                for resource in resources
            ),
            "degraded": sum(
                resource["health"]
                == ROS2HealthStatus.DEGRADED.value
                for resource in resources
            ),
            "unhealthy": sum(
                resource["health"]
                == ROS2HealthStatus.UNHEALTHY.value
                for resource in resources
            ),
            "unknown": sum(
                resource["health"]
                == ROS2HealthStatus.UNKNOWN.value
                for resource in resources
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
                "resources": self.scan_health(
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
    monitor = RoboticsROS2HealthMonitor()

    monitor.register_node(
        ROS2NodeDefinition(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            node_id="perception",
            ros_name="/robot_001/perception",
            node_type=ROS2NodeType.PERCEPTION,
            critical=True,
        )
    )

    monitor.register_topic(
        ROS2TopicDefinition(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            topic_id="camera",
            ros_name="/robot_001/camera/image",
            critical=True,
        )
    )

    monitor.register_service(
        ROS2ServiceDefinition(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            service_id="navigation_service",
            ros_name="/robot_001/navigation/status",
            critical=True,
        )
    )

    monitor.heartbeat(
        ROS2NodeHeartbeat(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            node_id="perception",
            sequence_number=1,
            lifecycle_state=(
                ROS2LifecycleState.ACTIVE
            ),
            healthy=True,
        )
    )

    monitor.observe_topic(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        topic_id="camera",
    )

    monitor.update_service(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        service_id="navigation_service",
        available=True,
    )

    assessment = monitor.evaluate_robot(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
    )

    print(
        json.dumps(
            assessment,
            indent=2,
        )
    )

    monitor.export_json(
        "data/robotics/ros2_health_report.json",
        tenant_id="tenant_alpha",
    )


if __name__ == "__main__":
    main()

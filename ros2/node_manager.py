"""Node Manager module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File: backend/app/robotics/control/node_manager.py

Python 3.11+
Standard library only.

Simulation-only robotics node management.

Features:
- Node registration and discovery
- Node lifecycle state management
- Dependency graph validation
- Dependency cycle detection
- Node heartbeat monitoring
- Stale heartbeat detection
- Health and readiness assessment
- Critical node failure latching
- Explicit recovery approval
- Robot-level readiness checks
- Tenant-scoped node isolation
- Thread-safe operations
- Audit logging
- JSON report export

SAFETY:
    This module manages logical simulation nodes.

    It does not start physical robot motors,
    issue actuator commands, or implement
    safety-rated robot supervision.

    Node readiness is not proof that a physical
    robot is safe to operate.
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

class NodeManagerError(Exception):
    pass


class NodeValidationError(NodeManagerError):
    pass


class NodeNotFound(NodeManagerError):
    pass


class NodeTransitionError(NodeManagerError):
    pass


# ============================================================
# ENUMS
# ============================================================

class NodeType(str, Enum):
    ROBOT_CONTROLLER = "robot_controller"
    PERCEPTION = "perception"
    NAVIGATION = "navigation"
    PLANNING = "planning"
    MANIPULATION = "manipulation"
    VLA_INFERENCE = "vla_inference"
    SENSOR_FUSION = "sensor_fusion"
    TELEMETRY = "telemetry"
    SIMULATION = "simulation"
    ACTION_CLIENT = "action_client"
    SAFETY_MONITOR = "safety_monitor"
    COMMAND_INTERLOCK = "command_interlock"
    WATCHDOG = "watchdog"
    AGENT = "agent"
    OTHER = "other"


class NodeLifecycleState(str, Enum):
    REGISTERED = "registered"
    INACTIVE = "inactive"
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAULTED = "faulted"
    RECOVERY_PENDING = "recovery_pending"
    SHUTDOWN = "shutdown"


class NodeHealthStatus(str, Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class NodeCriticality(str, Enum):
    OPTIONAL = "optional"
    REQUIRED = "required"
    CRITICAL = "critical"


class NodeDecision(str, Enum):
    READY_FOR_SIMULATION = "ready_for_simulation"
    DENY = "deny"


class NodeEventType(str, Enum):
    REGISTERED = "registered"
    ACTIVATED = "activated"
    DEACTIVATED = "deactivated"
    HEARTBEAT = "heartbeat"
    DEGRADED = "degraded"
    TIMEOUT = "timeout"
    FAULTED = "faulted"
    RECOVERY_REQUESTED = "recovery_requested"
    RECOVERED = "recovered"
    SHUTDOWN = "shutdown"


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
            char.isalnum() or char in "_-"
            for char in value
        )
    ):
        raise NodeValidationError(
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
        raise NodeValidationError(
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
        raise NodeValidationError(
            "Timezone-aware timestamp required"
        )

    return value.astimezone(timezone.utc)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class NodeManagerConfig:
    default_heartbeat_timeout_seconds: float = 3.0
    default_warning_seconds: float = 1.5

    maximum_nodes: int = 10000
    maximum_audit_events: int = 20000

    simulation_only: bool = True

    def __post_init__(self) -> None:
        positive_number(
            self.default_heartbeat_timeout_seconds,
            "default_heartbeat_timeout_seconds",
        )

        positive_number(
            self.default_warning_seconds,
            "default_warning_seconds",
        )

        if (
            self.default_warning_seconds
            >= self.default_heartbeat_timeout_seconds
        ):
            raise NodeValidationError(
                "Warning must be below timeout"
            )

        for name in (
            "maximum_nodes",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise NodeValidationError(
                    f"{name} must be positive"
                )

        if not self.simulation_only:
            raise NodeValidationError(
                "Hardware node control unsupported"
            )


# ============================================================
# NODE DEFINITION
# ============================================================

@dataclass(frozen=True)
class RobotNodeDefinition:
    tenant_id: str
    robot_id: str
    node_id: str

    node_type: NodeType

    criticality: NodeCriticality = (
        NodeCriticality.REQUIRED
    )

    dependencies: tuple[str, ...] = ()

    heartbeat_timeout_seconds: float = 3.0
    warning_seconds: float = 1.5

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")
        validate_id(self.node_id, "node_id")

        for dependency in self.dependencies:
            validate_id(
                dependency,
                "dependency",
            )

        if self.node_id in self.dependencies:
            raise NodeValidationError(
                "Node cannot depend on itself"
            )

        if (
            len(self.dependencies)
            != len(set(self.dependencies))
        ):
            raise NodeValidationError(
                "Duplicate node dependencies"
            )

        positive_number(
            self.heartbeat_timeout_seconds,
            "heartbeat_timeout_seconds",
        )

        positive_number(
            self.warning_seconds,
            "warning_seconds",
        )

        if (
            self.warning_seconds
            >= self.heartbeat_timeout_seconds
        ):
            raise NodeValidationError(
                "Warning must be below timeout"
            )


# ============================================================
# NODE HEARTBEAT
# ============================================================

@dataclass(frozen=True)
class NodeHeartbeat:
    tenant_id: str
    robot_id: str
    node_id: str

    sequence_number: int

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
            raise NodeValidationError(
                "Invalid heartbeat sequence"
            )

        if not isinstance(self.healthy, bool):
            raise NodeValidationError(
                "healthy must be boolean"
            )

        validate_timestamp(self.timestamp)


# ============================================================
# NODE RECORD
# ============================================================

@dataclass
class RobotNodeRecord:
    definition: RobotNodeDefinition

    state: NodeLifecycleState = (
        NodeLifecycleState.REGISTERED
    )

    health: NodeHealthStatus = (
        NodeHealthStatus.UNKNOWN
    )

    last_sequence_number: int = -1
    last_heartbeat_at: datetime | None = None
    last_received_monotonic: float | None = None

    fault_latched: bool = False

    heartbeat_count: int = 0
    timeout_count: int = 0

    last_error: str | None = None

    created_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.definition.tenant_id,
            "robot_id": self.definition.robot_id,
            "node_id": self.definition.node_id,
            "node_type": (
                self.definition.node_type.value
            ),
            "criticality": (
                self.definition.criticality.value
            ),
            "dependencies": list(
                self.definition.dependencies
            ),
            "state": self.state.value,
            "health": self.health.value,
            "fault_latched": self.fault_latched,
            "last_sequence_number": (
                self.last_sequence_number
            ),
            "last_heartbeat_at": (
                self.last_heartbeat_at.isoformat()
                if self.last_heartbeat_at
                else None
            ),
            "heartbeat_count": self.heartbeat_count,
            "timeout_count": self.timeout_count,
            "last_error": self.last_error,
            "created_at": self.created_at.isoformat(),
        }


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class NodeAuditEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    robot_id: str
    node_id: str

    event_type: NodeEventType
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
# NODE MANAGER
# ============================================================

class RoboticsNodeManager:

    def __init__(
        self,
        config: NodeManagerConfig | None = None,
    ) -> None:
        self.config = (
            config or NodeManagerConfig()
        )

        self.nodes: dict[
            tuple[str, str, str],
            RobotNodeRecord
        ] = {}

        self.audit_events: list[
            NodeAuditEvent
        ] = []

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # HELPERS
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
    ) -> RobotNodeRecord:
        key = self._key(
            tenant_id,
            robot_id,
            node_id,
        )

        record = self.nodes.get(key)

        if record is None:
            raise NodeNotFound(
                "Node not found"
            )

        return record

    def _audit(
        self,
        record: RobotNodeRecord,
        event_type: NodeEventType,
        details: str,
    ) -> None:
        definition = record.definition

        self.audit_events.append(
            NodeAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                tenant_id=definition.tenant_id,
                robot_id=definition.robot_id,
                node_id=definition.node_id,
                event_type=event_type,
                details=details,
            )
        )

        if (
            len(self.audit_events)
            > self.config.maximum_audit_events
        ):
            self.audit_events.pop(0)

    def _dependencies_ready(
        self,
        record: RobotNodeRecord,
    ) -> bool:
        definition = record.definition

        for dependency_id in definition.dependencies:
            dependency = self.nodes.get(
                (
                    definition.tenant_id,
                    definition.robot_id,
                    dependency_id,
                )
            )

            if dependency is None:
                return False

            self._refresh(
                dependency,
                time.monotonic(),
            )

            if (
                dependency.state
                != NodeLifecycleState.ACTIVE
                or dependency.health
                != NodeHealthStatus.HEALTHY
                or dependency.fault_latched
            ):
                return False

        return True

    def _refresh(
        self,
        record: RobotNodeRecord,
        now_monotonic: float,
    ) -> None:
        if record.state in {
            NodeLifecycleState.REGISTERED,
            NodeLifecycleState.INACTIVE,
            NodeLifecycleState.SHUTDOWN,
            NodeLifecycleState.RECOVERY_PENDING,
        }:
            return

        if record.fault_latched:
            record.state = (
                NodeLifecycleState.FAULTED
            )
            record.health = (
                NodeHealthStatus.UNHEALTHY
            )
            return

        last = record.last_received_monotonic

        if last is None:
            record.state = (
                NodeLifecycleState.DEGRADED
            )
            record.health = (
                NodeHealthStatus.UNKNOWN
            )
            return

        age = max(
            0.0,
            now_monotonic - last,
        )

        if (
            age
            >= record.definition.heartbeat_timeout_seconds
        ):
            if (
                record.state
                != NodeLifecycleState.FAULTED
            ):
                record.timeout_count += 1
                record.last_error = (
                    "Node heartbeat timeout"
                )

                self._audit(
                    record,
                    NodeEventType.TIMEOUT,
                    record.last_error,
                )

            record.state = (
                NodeLifecycleState.FAULTED
            )
            record.health = (
                NodeHealthStatus.UNHEALTHY
            )

            if (
                record.definition.criticality
                == NodeCriticality.CRITICAL
            ):
                record.fault_latched = True

            return

        if age >= record.definition.warning_seconds:
            record.state = (
                NodeLifecycleState.DEGRADED
            )
            record.health = (
                NodeHealthStatus.DEGRADED
            )

            return

        if record.health == NodeHealthStatus.HEALTHY:
            record.state = NodeLifecycleState.ACTIVE

    # --------------------------------------------------------
    # REGISTER NODE
    # --------------------------------------------------------

    def register_node(
        self,
        definition: RobotNodeDefinition,
    ) -> dict[str, Any]:
        with self._lock:
            key = self._key(
                definition.tenant_id,
                definition.robot_id,
                definition.node_id,
            )

            if key in self.nodes:
                raise NodeValidationError(
                    "Node already registered"
                )

            if (
                len(self.nodes)
                >= self.config.maximum_nodes
            ):
                raise NodeManagerError(
                    "Maximum node capacity reached"
                )

            record = RobotNodeRecord(
                definition=definition
            )

            self.nodes[key] = record

            self._audit(
                record,
                NodeEventType.REGISTERED,
                "Node registered",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # DEPENDENCY GRAPH VALIDATION
    # --------------------------------------------------------

    def validate_dependency_graph(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            graph = {
                node_id: list(
                    record.definition.dependencies
                )
                for (
                    record_tenant,
                    record_robot,
                    node_id
                ), record in self.nodes.items()
                if (
                    record_tenant == tenant_id
                    and record_robot == robot_id
                )
            }

            visiting: set[str] = set()
            visited: set[str] = set()
            missing: set[str] = set()
            cycles: list[str] = []

            def visit(node_id: str) -> None:
                if node_id in visiting:
                    cycles.append(node_id)
                    return

                if node_id in visited:
                    return

                if node_id not in graph:
                    missing.add(node_id)
                    return

                visiting.add(node_id)

                for dependency in graph[node_id]:
                    visit(dependency)

                visiting.remove(node_id)
                visited.add(node_id)

            for node_id in graph:
                visit(node_id)

            return {
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "valid": (
                    not missing
                    and not cycles
                    and bool(graph)
                ),
                "node_count": len(graph),
                "missing_dependencies": sorted(missing),
                "cycle_nodes": sorted(set(cycles)),
            }

    # --------------------------------------------------------
    # ACTIVATE NODE
    # --------------------------------------------------------

    def activate_node(
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

            if record.state not in {
                NodeLifecycleState.REGISTERED,
                NodeLifecycleState.INACTIVE,
            }:
                raise NodeTransitionError(
                    "Node cannot be activated"
                )

            graph = self.validate_dependency_graph(
                tenant_id=tenant_id,
                robot_id=robot_id,
            )

            if not graph["valid"]:
                raise NodeTransitionError(
                    "Invalid dependency graph"
                )

            if not self._dependencies_ready(record):
                raise NodeTransitionError(
                    "Dependencies are not ready"
                )

            record.state = (
                NodeLifecycleState.DEGRADED
            )

            record.health = (
                NodeHealthStatus.UNKNOWN
            )

            self._audit(
                record,
                NodeEventType.ACTIVATED,
                "Node activation requested",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # HEARTBEAT
    # --------------------------------------------------------

    def heartbeat(
        self,
        heartbeat: NodeHeartbeat,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                heartbeat.tenant_id,
                heartbeat.robot_id,
                heartbeat.node_id,
            )

            if record.state in {
                NodeLifecycleState.REGISTERED,
                NodeLifecycleState.INACTIVE,
                NodeLifecycleState.SHUTDOWN,
            }:
                raise NodeTransitionError(
                    "Node is not activated"
                )

            if (
                heartbeat.sequence_number
                <= record.last_sequence_number
            ):
                raise NodeValidationError(
                    "Duplicate or out-of-order heartbeat"
                )

            record.last_sequence_number = (
                heartbeat.sequence_number
            )

            record.last_heartbeat_at = (
                heartbeat.timestamp
            )

            record.last_received_monotonic = (
                time.monotonic()
            )

            record.heartbeat_count += 1

            if record.fault_latched:
                record.state = (
                    NodeLifecycleState.FAULTED
                )
                record.health = (
                    NodeHealthStatus.UNHEALTHY
                )

            elif heartbeat.healthy:
                record.health = (
                    NodeHealthStatus.HEALTHY
                )

                record.state = (
                    NodeLifecycleState.ACTIVE
                )

                record.last_error = None

            else:
                record.health = (
                    NodeHealthStatus.DEGRADED
                )

                record.state = (
                    NodeLifecycleState.DEGRADED
                )

                record.last_error = (
                    "Node reported unhealthy"
                )

            self._audit(
                record,
                NodeEventType.HEARTBEAT,
                "Node heartbeat received",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # DEACTIVATE NODE
    # --------------------------------------------------------

    def deactivate_node(
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

            if record.fault_latched:
                raise NodeTransitionError(
                    "Cannot deactivate latched fault"
                )

            record.state = (
                NodeLifecycleState.INACTIVE
            )

            record.health = (
                NodeHealthStatus.UNKNOWN
            )

            self._audit(
                record,
                NodeEventType.DEACTIVATED,
                "Node deactivated",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # REQUEST RECOVERY
    # --------------------------------------------------------

    def request_recovery(
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

            if record.state != NodeLifecycleState.FAULTED:
                raise NodeTransitionError(
                    "Node is not faulted"
                )

            record.state = (
                NodeLifecycleState.RECOVERY_PENDING
            )

            self._audit(
                record,
                NodeEventType.RECOVERY_REQUESTED,
                "Node recovery requested",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # COMPLETE RECOVERY
    # --------------------------------------------------------

    def complete_recovery(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        node_id: str,
        authorized: bool,
        independently_verified: bool,
    ) -> dict[str, Any]:
        """
        Simulation-only recovery.

        The booleans are illustrative and must
        be replaced with authenticated approvals
        and independent evidence.
        """
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                node_id,
            )

            if (
                record.state
                != NodeLifecycleState.RECOVERY_PENDING
            ):
                raise NodeTransitionError(
                    "Recovery not pending"
                )

            if (
                not authorized
                or not independently_verified
            ):
                raise NodeTransitionError(
                    "Verified recovery approval required"
                )

            if not self._dependencies_ready(record):
                raise NodeTransitionError(
                    "Dependencies not healthy"
                )

            record.fault_latched = False
            record.state = (
                NodeLifecycleState.INACTIVE
            )
            record.health = (
                NodeHealthStatus.UNKNOWN
            )
            record.last_received_monotonic = None
            record.last_error = None

            self._audit(
                record,
                NodeEventType.RECOVERED,
                "Node recovered to inactive state",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------------

    def shutdown_node(
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

            record.state = (
                NodeLifecycleState.SHUTDOWN
            )

            record.health = (
                NodeHealthStatus.UNKNOWN
            )

            self._audit(
                record,
                NodeEventType.SHUTDOWN,
                "Simulation node shutdown",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # SCAN HEALTH
    # --------------------------------------------------------

    def scan_health(
        self,
        *,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            now = time.monotonic()

            results = []

            for record in self.nodes.values():
                if (
                    tenant_id is not None
                    and record.definition.tenant_id
                    != tenant_id
                ):
                    continue

                self._refresh(
                    record,
                    now,
                )

                results.append(
                    record.to_dict()
                )

            return results

    # --------------------------------------------------------
    # ROBOT READINESS
    # --------------------------------------------------------

    def evaluate_robot(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            graph = self.validate_dependency_graph(
                tenant_id=tenant_id,
                robot_id=robot_id,
            )

            records = [
                record
                for (
                    record_tenant,
                    record_robot,
                    _
                ), record in self.nodes.items()
                if (
                    record_tenant == tenant_id
                    and record_robot == robot_id
                )
            ]

            now = time.monotonic()

            for record in records:
                self._refresh(record, now)

            required = [
                record
                for record in records
                if record.definition.criticality
                != NodeCriticality.OPTIONAL
            ]

            ready = (
                graph["valid"]
                and bool(required)
                and all(
                    record.state
                    == NodeLifecycleState.ACTIVE
                    and record.health
                    == NodeHealthStatus.HEALTHY
                    and not record.fault_latched
                    and self._dependencies_ready(record)
                    for record in required
                )
            )

            return {
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "decision": (
                    NodeDecision.READY_FOR_SIMULATION.value
                    if ready
                    else NodeDecision.DENY.value
                ),
                "dependency_graph_valid": (
                    graph["valid"]
                ),
                "total_nodes": len(records),
                "required_nodes": len(required),
                "unhealthy_nodes": [
                    record.definition.node_id
                    for record in required
                    if (
                        record.state
                        != NodeLifecycleState.ACTIVE
                        or record.health
                        != NodeHealthStatus.HEALTHY
                        or record.fault_latched
                    )
                ],
                "evaluated_at": (
                    utc_now().isoformat()
                ),
            }

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

            self._refresh(
                record,
                time.monotonic(),
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
        records = self.scan_health(
            tenant_id=tenant_id
        )

        return {
            "total_nodes": len(records),
            "active": sum(
                item["state"]
                == NodeLifecycleState.ACTIVE.value
                for item in records
            ),
            "degraded": sum(
                item["state"]
                == NodeLifecycleState.DEGRADED.value
                for item in records
            ),
            "faulted": sum(
                item["state"]
                == NodeLifecycleState.FAULTED.value
                for item in records
            ),
            "shutdown": sum(
                item["state"]
                == NodeLifecycleState.SHUTDOWN.value
                for item in records
            ),
            "latched_faults": sum(
                item["fault_latched"]
                for item in records
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
                "nodes": [
                    item
                    for item in self.scan_health(
                        tenant_id=tenant_id
                    )
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
    manager = RoboticsNodeManager()

    definitions = [
        RobotNodeDefinition(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            node_id="telemetry",
            node_type=NodeType.TELEMETRY,
            criticality=NodeCriticality.CRITICAL,
        ),
        RobotNodeDefinition(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            node_id="perception",
            node_type=NodeType.PERCEPTION,
            criticality=NodeCriticality.CRITICAL,
            dependencies=("telemetry",),
        ),
        RobotNodeDefinition(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            node_id="navigation",
            node_type=NodeType.NAVIGATION,
            criticality=NodeCriticality.REQUIRED,
            dependencies=("perception",),
        ),
    ]

    for definition in definitions:
        manager.register_node(definition)

    for node_id in (
        "telemetry",
        "perception",
        "navigation",
    ):
        manager.activate_node(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            node_id=node_id,
        )

        manager.heartbeat(
            NodeHeartbeat(
                tenant_id="tenant_alpha",
                robot_id="robot_001",
                node_id=node_id,
                sequence_number=1,
                healthy=True,
            )
        )

    readiness = manager.evaluate_robot(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
    )

    print(
        json.dumps(
            readiness,
            indent=2,
        )
    )

    manager.export_json(
        "data/robotics/node_manager_report.json",
        tenant_id="tenant_alpha",
    )


if __name__ == "__main__":
    main()

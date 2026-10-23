"""Fleet Manager module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
backend/app/robotics/control/fleet_manager.py

Python 3.11+
Standard library only.

MULTI-ROBOT FLEET MANAGER — SIMULATION ONLY

Features:
- Multi-robot registration
- Tenant and fleet isolation
- Robot capability management
- Battery and health monitoring
- Robot availability tracking
- Task registration and assignment
- Capability-based robot selection
- Battery-aware task scheduling
- Task lifecycle management
- Robot reservation and release
- Emergency-stop latching
- Fault detection
- Telemetry freshness monitoring
- Audit logging
- Fleet statistics
- JSON report export

SAFETY:
    Does not move physical robots.
    Does not issue actuator commands.
    Does not connect to real robot hardware.

    Task assignment is a simulation reservation,
    not permission to execute a physical task.

    Real robot deployment requires authenticated
    telemetry, trusted authorization, independent
    collision avoidance, safety interlocks and
    hardware-enforced emergency stopping.
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

class FleetManagerError(Exception):
    pass


class FleetValidationError(FleetManagerError):
    pass


class RobotNotFound(FleetManagerError):
    pass


class FleetTaskNotFound(FleetManagerError):
    pass


class FleetConflict(FleetManagerError):
    pass


class FleetPermissionDenied(FleetManagerError):
    pass


# ============================================================
# ENUMS
# ============================================================

class RobotType(str, Enum):
    AMR = "amr"
    AGV = "agv"
    HUMANOID = "humanoid"
    ROBOT_ARM = "robot_arm"
    MOBILE_MANIPULATOR = "mobile_manipulator"
    INSPECTION_ROBOT = "inspection_robot"
    SERVICE_ROBOT = "service_robot"
    RESEARCH_ROBOT = "research_robot"
    OTHER = "other"


class RobotStatus(str, Enum):
    REGISTERED = "registered"
    IDLE = "idle"
    BUSY = "busy"
    CHARGING = "charging"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    FAULTED = "faulted"
    EMERGENCY_STOP = "emergency_stop"


class FleetTaskType(str, Enum):
    NAVIGATION = "navigation"
    TRANSPORT = "transport"
    PICK_AND_PLACE = "pick_and_place"
    INSPECTION = "inspection"
    INVENTORY = "inventory"
    MAINTENANCE = "maintenance"
    DIAGNOSTICS = "diagnostics"
    CHARGING = "charging"
    SIMULATION = "simulation"


class FleetTaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FleetEventType(str, Enum):
    ROBOT_REGISTERED = "robot_registered"
    TELEMETRY_UPDATED = "telemetry_updated"
    ROBOT_STALE = "robot_stale"
    ROBOT_FAULTED = "robot_faulted"
    TASK_CREATED = "task_created"
    TASK_ASSIGNED = "task_assigned"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    EMERGENCY_STOP = "emergency_stop"
    EMERGENCY_RESET = "emergency_reset"


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
            ch.isascii()
            and (ch.isalnum() or ch in "_.-")
            for ch in value
        )
    ):
        raise FleetValidationError(
            f"Invalid {label}"
        )

    return value


def finite(value: float, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise FleetValidationError(
            f"{label} must be finite"
        )

    return float(value)


def percentage(value: float, label: str) -> float:
    result = finite(value, label)

    if not 0 <= result <= 100:
        raise FleetValidationError(
            f"{label} must be between 0 and 100"
        )

    return result


def validate_timestamp(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise FleetValidationError(
            "Timezone-aware timestamp required"
        )

    return value.astimezone(timezone.utc)


def validate_capabilities(
    values: tuple[str, ...],
) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise FleetValidationError(
            "Capabilities must be a tuple"
        )

    for value in values:
        validate_id(value, "capability")

    if len(values) != len(set(values)):
        raise FleetValidationError(
            "Duplicate capabilities"
        )

    return values


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class FleetManagerConfig:
    maximum_robots: int = 10000
    maximum_tasks: int = 100000
    maximum_audit_events: int = 20000

    telemetry_timeout_seconds: float = 5.0

    minimum_battery_for_assignment: float = 20.0
    critical_battery_percentage: float = 10.0

    maximum_temperature_c: float = 80.0

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for label in (
            "maximum_robots",
            "maximum_tasks",
            "maximum_audit_events",
        ):
            value = getattr(self, label)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise FleetValidationError(
                    f"{label} must be positive"
                )

        if (
            finite(
                self.telemetry_timeout_seconds,
                "telemetry_timeout_seconds",
            ) <= 0
        ):
            raise FleetValidationError(
                "Telemetry timeout must be positive"
            )

        percentage(
            self.minimum_battery_for_assignment,
            "minimum_battery_for_assignment",
        )

        percentage(
            self.critical_battery_percentage,
            "critical_battery_percentage",
        )

        if (
            self.critical_battery_percentage
            > self.minimum_battery_for_assignment
        ):
            raise FleetValidationError(
                "Critical battery threshold exceeds "
                "assignment threshold"
            )

        if (
            finite(
                self.maximum_temperature_c,
                "maximum_temperature_c",
            ) <= 0
        ):
            raise FleetValidationError(
                "Maximum temperature must be positive"
            )

        if not self.simulation_only:
            raise FleetValidationError(
                "Physical fleet control unsupported"
            )


# ============================================================
# ROBOT DEFINITION
# ============================================================

@dataclass(frozen=True)
class FleetRobotDefinition:
    tenant_id: str
    fleet_id: str
    robot_id: str

    name: str
    robot_type: RobotType

    capabilities: tuple[str, ...]

    location_zone: str = "default"

    def __post_init__(self) -> None:
        for label in (
            "tenant_id",
            "fleet_id",
            "robot_id",
            "location_zone",
        ):
            validate_id(
                getattr(self, label),
                label,
            )

        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or len(self.name) > 256
        ):
            raise FleetValidationError(
                "Invalid robot name"
            )

        if not isinstance(
            self.robot_type,
            RobotType,
        ):
            raise FleetValidationError(
                "Invalid robot type"
            )

        validate_capabilities(
            self.capabilities
        )


# ============================================================
# ROBOT TELEMETRY
# ============================================================

@dataclass(frozen=True)
class FleetRobotTelemetry:
    tenant_id: str
    fleet_id: str
    robot_id: str

    sequence_number: int

    battery_percentage: float
    temperature_c: float

    operational: bool = True
    charging: bool = False
    fault_detected: bool = False

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for label in (
            "tenant_id",
            "fleet_id",
            "robot_id",
        ):
            validate_id(
                getattr(self, label),
                label,
            )

        if (
            isinstance(self.sequence_number, bool)
            or not isinstance(self.sequence_number, int)
            or self.sequence_number < 0
        ):
            raise FleetValidationError(
                "Invalid telemetry sequence"
            )

        percentage(
            self.battery_percentage,
            "battery_percentage",
        )

        finite(
            self.temperature_c,
            "temperature_c",
        )

        for label in (
            "operational",
            "charging",
            "fault_detected",
        ):
            if not isinstance(
                getattr(self, label),
                bool,
            ):
                raise FleetValidationError(
                    f"{label} must be boolean"
                )

        validate_timestamp(
            self.timestamp
        )


# ============================================================
# TASK DEFINITION
# ============================================================

@dataclass(frozen=True)
class FleetTaskDefinition:
    tenant_id: str
    fleet_id: str
    task_id: str

    task_type: FleetTaskType

    required_capabilities: tuple[str, ...]

    minimum_battery_percentage: float = 20.0

    preferred_zone: str | None = None

    priority: int = 50

    def __post_init__(self) -> None:
        for label in (
            "tenant_id",
            "fleet_id",
            "task_id",
        ):
            validate_id(
                getattr(self, label),
                label,
            )

        if not isinstance(
            self.task_type,
            FleetTaskType,
        ):
            raise FleetValidationError(
                "Invalid task type"
            )

        validate_capabilities(
            self.required_capabilities
        )

        percentage(
            self.minimum_battery_percentage,
            "minimum_battery_percentage",
        )

        if self.preferred_zone is not None:
            validate_id(
                self.preferred_zone,
                "preferred_zone",
            )

        if (
            isinstance(self.priority, bool)
            or not isinstance(self.priority, int)
            or not 0 <= self.priority <= 100
        ):
            raise FleetValidationError(
                "Priority must be between 0 and 100"
            )


# ============================================================
# ROBOT RECORD
# ============================================================

@dataclass
class FleetRobotRecord:
    definition: FleetRobotDefinition

    status: RobotStatus = RobotStatus.REGISTERED

    last_telemetry: FleetRobotTelemetry | None = None

    last_received_monotonic: float | None = None

    last_sequence_number: int = -1

    assigned_task_id: str | None = None

    fault_latched: bool = False
    emergency_latched: bool = False

    fault_reasons: list[str] = field(
        default_factory=list
    )

    completed_tasks: int = 0

    def to_dict(self) -> dict[str, Any]:
        definition = self.definition

        return {
            "tenant_id": definition.tenant_id,
            "fleet_id": definition.fleet_id,
            "robot_id": definition.robot_id,
            "name": definition.name,
            "robot_type": definition.robot_type.value,
            "capabilities": list(
                definition.capabilities
            ),
            "location_zone": definition.location_zone,
            "status": self.status.value,
            "assigned_task_id": (
                self.assigned_task_id
            ),
            "fault_latched": self.fault_latched,
            "emergency_latched": (
                self.emergency_latched
            ),
            "fault_reasons": list(
                self.fault_reasons
            ),
            "completed_tasks": (
                self.completed_tasks
            ),
            "last_telemetry": (
                {
                    "battery_percentage": (
                        self.last_telemetry.battery_percentage
                    ),
                    "temperature_c": (
                        self.last_telemetry.temperature_c
                    ),
                    "operational": (
                        self.last_telemetry.operational
                    ),
                    "charging": (
                        self.last_telemetry.charging
                    ),
                    "timestamp": (
                        self.last_telemetry.timestamp.isoformat()
                    ),
                }
                if self.last_telemetry
                else None
            ),
        }


# ============================================================
# TASK RECORD
# ============================================================

@dataclass
class FleetTaskRecord:
    definition: FleetTaskDefinition

    status: FleetTaskStatus = (
        FleetTaskStatus.PENDING
    )

    assigned_robot_id: str | None = None

    created_at: datetime = field(
        default_factory=utc_now
    )

    started_at: datetime | None = None
    finished_at: datetime | None = None

    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.definition.tenant_id,
            "fleet_id": self.definition.fleet_id,
            "task_id": self.definition.task_id,
            "task_type": (
                self.definition.task_type.value
            ),
            "required_capabilities": list(
                self.definition.required_capabilities
            ),
            "priority": self.definition.priority,
            "status": self.status.value,
            "assigned_robot_id": (
                self.assigned_robot_id
            ),
            "created_at": (
                self.created_at.isoformat()
            ),
            "started_at": (
                self.started_at.isoformat()
                if self.started_at
                else None
            ),
            "finished_at": (
                self.finished_at.isoformat()
                if self.finished_at
                else None
            ),
            "failure_reason": self.failure_reason,
        }


# ============================================================
# AUDIT
# ============================================================

@dataclass(frozen=True)
class FleetAuditEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    fleet_id: str
    resource_id: str

    event_type: FleetEventType
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "fleet_id": self.fleet_id,
            "resource_id": self.resource_id,
            "event_type": self.event_type.value,
            "details": self.details,
        }


# ============================================================
# FLEET MANAGER
# ============================================================

class RoboticsFleetManager:

    def __init__(
        self,
        config: FleetManagerConfig | None = None,
    ) -> None:
        self.config = (
            config or FleetManagerConfig()
        )

        self._robots: dict[
            tuple[str, str, str],
            FleetRobotRecord
        ] = {}

        self._tasks: dict[
            tuple[str, str, str],
            FleetTaskRecord
        ] = {}

        self._audit_events: deque[
            FleetAuditEvent
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
        fleet_id: str,
        resource_id: str,
    ) -> tuple[str, str, str]:
        return (
            validate_id(tenant_id, "tenant_id"),
            validate_id(fleet_id, "fleet_id"),
            validate_id(resource_id, "resource_id"),
        )

    def _robot(
        self,
        tenant_id: str,
        fleet_id: str,
        robot_id: str,
    ) -> FleetRobotRecord:
        record = self._robots.get(
            self._key(
                tenant_id,
                fleet_id,
                robot_id,
            )
        )

        if record is None:
            raise RobotNotFound(
                "Robot not registered"
            )

        return record

    def _task(
        self,
        tenant_id: str,
        fleet_id: str,
        task_id: str,
    ) -> FleetTaskRecord:
        record = self._tasks.get(
            self._key(
                tenant_id,
                fleet_id,
                task_id,
            )
        )

        if record is None:
            raise FleetTaskNotFound(
                "Fleet task not found"
            )

        return record

    def _audit(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
        resource_id: str,
        event_type: FleetEventType,
        details: str,
    ) -> None:
        self._audit_events.append(
            FleetAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                tenant_id=tenant_id,
                fleet_id=fleet_id,
                resource_id=resource_id,
                event_type=event_type,
                details=details,
            )
        )

    # --------------------------------------------------------
    # REGISTER ROBOT
    # --------------------------------------------------------

    def register_robot(
        self,
        definition: FleetRobotDefinition,
    ) -> dict[str, Any]:
        with self._lock:
            key = self._key(
                definition.tenant_id,
                definition.fleet_id,
                definition.robot_id,
            )

            if key in self._robots:
                raise FleetConflict(
                    "Robot already registered"
                )

            if (
                len(self._robots)
                >= self.config.maximum_robots
            ):
                raise FleetManagerError(
                    "Maximum robot capacity reached"
                )

            record = FleetRobotRecord(
                definition=definition
            )

            self._robots[key] = record

            self._audit(
                tenant_id=definition.tenant_id,
                fleet_id=definition.fleet_id,
                resource_id=definition.robot_id,
                event_type=(
                    FleetEventType.ROBOT_REGISTERED
                ),
                details="Robot registered",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # FAULT LATCH
    # --------------------------------------------------------

    def _latch_fault(
        self,
        record: FleetRobotRecord,
        reason: str,
    ) -> None:
        record.fault_latched = True

        if reason not in record.fault_reasons:
            record.fault_reasons.append(
                reason
            )

        if not record.emergency_latched:
            record.status = RobotStatus.FAULTED

        definition = record.definition

        self._audit(
            tenant_id=definition.tenant_id,
            fleet_id=definition.fleet_id,
            resource_id=definition.robot_id,
            event_type=FleetEventType.ROBOT_FAULTED,
            details=reason,
        )

    # --------------------------------------------------------
    # TELEMETRY
    # --------------------------------------------------------

    def ingest_telemetry(
        self,
        telemetry: FleetRobotTelemetry,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._robot(
                telemetry.tenant_id,
                telemetry.fleet_id,
                telemetry.robot_id,
            )

            if (
                telemetry.sequence_number
                <= record.last_sequence_number
            ):
                raise FleetValidationError(
                    "Duplicate or out-of-order telemetry"
                )

            timestamp = validate_timestamp(
                telemetry.timestamp
            )

            now = utc_now()

            if timestamp > now:
                raise FleetValidationError(
                    "Future telemetry rejected"
                )

            age = (
                now - timestamp
            ).total_seconds()

            if (
                age
                > self.config.telemetry_timeout_seconds
            ):
                raise FleetValidationError(
                    "Stale telemetry rejected"
                )

            record.last_sequence_number = (
                telemetry.sequence_number
            )

            record.last_telemetry = telemetry

            record.last_received_monotonic = (
                time.monotonic()
            )

            if not telemetry.operational:
                self._latch_fault(
                    record,
                    "robot_not_operational",
                )

            if telemetry.fault_detected:
                self._latch_fault(
                    record,
                    "telemetry_fault_detected",
                )

            if (
                telemetry.temperature_c
                > self.config.maximum_temperature_c
            ):
                self._latch_fault(
                    record,
                    "overtemperature",
                )

            if (
                telemetry.battery_percentage
                <= self.config.critical_battery_percentage
            ):
                self._latch_fault(
                    record,
                    "critical_battery",
                )

            if record.emergency_latched:
                record.status = (
                    RobotStatus.EMERGENCY_STOP
                )

            elif record.fault_latched:
                record.status = (
                    RobotStatus.FAULTED
                )

            elif record.assigned_task_id:
                record.status = (
                    RobotStatus.BUSY
                )

            elif telemetry.charging:
                record.status = (
                    RobotStatus.CHARGING
                )

            elif (
                telemetry.battery_percentage
                < self.config.minimum_battery_for_assignment
            ):
                record.status = (
                    RobotStatus.DEGRADED
                )

            else:
                record.status = (
                    RobotStatus.IDLE
                )

            self._audit(
                tenant_id=telemetry.tenant_id,
                fleet_id=telemetry.fleet_id,
                resource_id=telemetry.robot_id,
                event_type=(
                    FleetEventType.TELEMETRY_UPDATED
                ),
                details="Robot telemetry updated",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # REFRESH ROBOT
    # --------------------------------------------------------

    def _refresh(
        self,
        record: FleetRobotRecord,
    ) -> None:
        last = record.last_received_monotonic

        if last is None:
            return

        if (
            time.monotonic() - last
            > self.config.telemetry_timeout_seconds
            and record.status
            not in (
                RobotStatus.OFFLINE,
                RobotStatus.EMERGENCY_STOP,
                RobotStatus.FAULTED,
            )
        ):
            record.status = RobotStatus.OFFLINE

            definition = record.definition

            self._audit(
                tenant_id=definition.tenant_id,
                fleet_id=definition.fleet_id,
                resource_id=definition.robot_id,
                event_type=FleetEventType.ROBOT_STALE,
                details="Robot telemetry timed out",
            )

    # --------------------------------------------------------
    # REGISTER TASK
    # --------------------------------------------------------

    def register_task(
        self,
        definition: FleetTaskDefinition,
    ) -> dict[str, Any]:
        with self._lock:
            key = self._key(
                definition.tenant_id,
                definition.fleet_id,
                definition.task_id,
            )

            if key in self._tasks:
                raise FleetConflict(
                    "Task already registered"
                )

            if (
                len(self._tasks)
                >= self.config.maximum_tasks
            ):
                raise FleetManagerError(
                    "Maximum task capacity reached"
                )

            record = FleetTaskRecord(
                definition=definition
            )

            self._tasks[key] = record

            self._audit(
                tenant_id=definition.tenant_id,
                fleet_id=definition.fleet_id,
                resource_id=definition.task_id,
                event_type=FleetEventType.TASK_CREATED,
                details="Fleet task created",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # ROBOT ELIGIBILITY
    # --------------------------------------------------------

    def _eligible(
        self,
        robot: FleetRobotRecord,
        task: FleetTaskRecord,
    ) -> bool:
        self._refresh(robot)

        if robot.status != RobotStatus.IDLE:
            return False

        if (
            robot.fault_latched
            or robot.emergency_latched
            or robot.assigned_task_id is not None
        ):
            return False

        telemetry = robot.last_telemetry

        if telemetry is None:
            return False

        if (
            telemetry.battery_percentage
            < max(
                self.config.minimum_battery_for_assignment,
                task.definition.minimum_battery_percentage,
            )
        ):
            return False

        return set(
            task.definition.required_capabilities
        ).issubset(
            robot.definition.capabilities
        )

    # --------------------------------------------------------
    # ASSIGN TASK
    # --------------------------------------------------------

    def assign_task(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
        task_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            task = self._task(
                tenant_id,
                fleet_id,
                task_id,
            )

            if task.status != FleetTaskStatus.PENDING:
                raise FleetConflict(
                    "Task is not pending"
                )

            candidates = []

            for robot in self._robots.values():
                definition = robot.definition

                if (
                    definition.tenant_id != tenant_id
                    or definition.fleet_id != fleet_id
                ):
                    continue

                if not self._eligible(
                    robot,
                    task,
                ):
                    continue

                telemetry = robot.last_telemetry

                zone_match = (
                    task.definition.preferred_zone is None
                    or definition.location_zone
                    == task.definition.preferred_zone
                )

                candidates.append((
                    int(zone_match),
                    telemetry.battery_percentage,
                    -robot.completed_tasks,
                    definition.robot_id,
                    robot,
                ))

            if not candidates:
                return {
                    "task_id": task_id,
                    "assigned": False,
                    "reason": "no_eligible_robot",
                }

            candidates.sort(
                key=lambda item: (
                    -item[0],
                    -item[1],
                    -item[2],
                    item[3],
                )
            )

            selected = candidates[0][4]

            task.assigned_robot_id = (
                selected.definition.robot_id
            )

            task.status = (
                FleetTaskStatus.ASSIGNED
            )

            selected.assigned_task_id = task_id

            selected.status = RobotStatus.BUSY

            self._audit(
                tenant_id=tenant_id,
                fleet_id=fleet_id,
                resource_id=task_id,
                event_type=FleetEventType.TASK_ASSIGNED,
                details=(
                    f"Assigned to "
                    f"{selected.definition.robot_id}"
                ),
            )

            return {
                "task_id": task_id,
                "assigned": True,
                "robot_id": (
                    selected.definition.robot_id
                ),
                "simulation_only": True,
            }

    # --------------------------------------------------------
    # START TASK
    # --------------------------------------------------------

    def start_task(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
        task_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            task = self._task(
                tenant_id,
                fleet_id,
                task_id,
            )

            if (
                task.status
                != FleetTaskStatus.ASSIGNED
            ):
                raise FleetConflict(
                    "Task must be assigned first"
                )

            robot = self._robot(
                tenant_id,
                fleet_id,
                task.assigned_robot_id,
            )

            self._refresh(robot)

            if (
                robot.status != RobotStatus.BUSY
                or robot.fault_latched
                or robot.emergency_latched
                or robot.assigned_task_id != task_id
            ):
                raise FleetConflict(
                    "Assigned robot is not ready"
                )

            task.status = (
                FleetTaskStatus.RUNNING
            )

            task.started_at = utc_now()

            self._audit(
                tenant_id=tenant_id,
                fleet_id=fleet_id,
                resource_id=task_id,
                event_type=FleetEventType.TASK_STARTED,
                details="Simulation task started",
            )

            return task.to_dict()

    # --------------------------------------------------------
    # FINISH TASK
    # --------------------------------------------------------

    def finish_task(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
        task_id: str,
        success: bool,
        reason: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            task = self._task(
                tenant_id,
                fleet_id,
                task_id,
            )

            if task.status not in (
                FleetTaskStatus.ASSIGNED,
                FleetTaskStatus.RUNNING,
            ):
                raise FleetConflict(
                    "Task is not active"
                )

            robot = self._robot(
                tenant_id,
                fleet_id,
                task.assigned_robot_id,
            )

            task.status = (
                FleetTaskStatus.COMPLETED
                if success
                else FleetTaskStatus.FAILED
            )

            task.finished_at = utc_now()

            if not success:
                task.failure_reason = (
                    reason or "task_failed"
                )

            if success:
                robot.completed_tasks += 1

            robot.assigned_task_id = None

            self._refresh(robot)

            if robot.emergency_latched:
                robot.status = RobotStatus.EMERGENCY_STOP

            elif robot.fault_latched:
                robot.status = RobotStatus.FAULTED

            elif robot.status != RobotStatus.OFFLINE:
                telemetry = robot.last_telemetry

                if telemetry and telemetry.charging:
                    robot.status = RobotStatus.CHARGING

                elif (
                    telemetry
                    and telemetry.battery_percentage
                    >= self.config.minimum_battery_for_assignment
                ):
                    robot.status = RobotStatus.IDLE

                else:
                    robot.status = RobotStatus.DEGRADED

            self._audit(
                tenant_id=tenant_id,
                fleet_id=fleet_id,
                resource_id=task_id,
                event_type=(
                    FleetEventType.TASK_COMPLETED
                    if success
                    else FleetEventType.TASK_FAILED
                ),
                details=(
                    "Task completed"
                    if success
                    else task.failure_reason
                ),
            )

            return task.to_dict()

    # --------------------------------------------------------
    # CANCEL TASK
    # --------------------------------------------------------

    def cancel_task(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
        task_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            task = self._task(
                tenant_id,
                fleet_id,
                task_id,
            )

            if task.status in (
                FleetTaskStatus.COMPLETED,
                FleetTaskStatus.FAILED,
                FleetTaskStatus.CANCELLED,
            ):
                return task.to_dict()

            if task.assigned_robot_id:
                robot = self._robot(
                    tenant_id,
                    fleet_id,
                    task.assigned_robot_id,
                )

                robot.assigned_task_id = None

                self._refresh(robot)

                if robot.emergency_latched:
                    robot.status = RobotStatus.EMERGENCY_STOP
                elif robot.fault_latched:
                    robot.status = RobotStatus.FAULTED
                elif robot.status != RobotStatus.OFFLINE:
                    robot.status = RobotStatus.IDLE

            task.status = FleetTaskStatus.CANCELLED
            task.finished_at = utc_now()

            self._audit(
                tenant_id=tenant_id,
                fleet_id=fleet_id,
                resource_id=task_id,
                event_type=FleetEventType.TASK_CANCELLED,
                details="Simulation task cancelled",
            )

            return task.to_dict()

    # --------------------------------------------------------
    # EMERGENCY STOP
    # --------------------------------------------------------

    def emergency_stop_robot(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            robot = self._robot(
                tenant_id,
                fleet_id,
                robot_id,
            )

            robot.emergency_latched = True
            robot.status = RobotStatus.EMERGENCY_STOP

            if robot.assigned_task_id:
                self.cancel_task(
                    tenant_id=tenant_id,
                    fleet_id=fleet_id,
                    task_id=robot.assigned_task_id,
                )

            self._audit(
                tenant_id=tenant_id,
                fleet_id=fleet_id,
                resource_id=robot_id,
                event_type=FleetEventType.EMERGENCY_STOP,
                details="Simulation emergency stop latched",
            )

            return robot.to_dict()

    # --------------------------------------------------------
    # RESET EMERGENCY STOP
    # --------------------------------------------------------

    def reset_emergency_stop(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
        robot_id: str,
        authorized: bool,
        independently_verified: bool,
    ) -> dict[str, Any]:
        with self._lock:
            robot = self._robot(
                tenant_id,
                fleet_id,
                robot_id,
            )

            if (
                authorized is not True
                or independently_verified is not True
            ):
                raise FleetPermissionDenied(
                    "Verified reset authorization required"
                )

            robot.emergency_latched = False

            if robot.fault_latched:
                robot.status = RobotStatus.FAULTED

            else:
                robot.status = RobotStatus.REGISTERED

            self._audit(
                tenant_id=tenant_id,
                fleet_id=fleet_id,
                resource_id=robot_id,
                event_type=FleetEventType.EMERGENCY_RESET,
                details="Simulation emergency stop reset",
            )

            return robot.to_dict()

    # --------------------------------------------------------
    # GET ROBOT
    # --------------------------------------------------------

    def get_robot(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._robot(
                tenant_id,
                fleet_id,
                robot_id,
            )

            self._refresh(record)

            return record.to_dict()

    # --------------------------------------------------------
    # GET TASK
    # --------------------------------------------------------

    def get_task(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
        task_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            return self._task(
                tenant_id,
                fleet_id,
                task_id,
            ).to_dict()

    # --------------------------------------------------------
    # LIST ROBOTS
    # --------------------------------------------------------

    def list_robots(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
    ) -> list[dict[str, Any]]:
        with self._lock:
            results = []

            for record in self._robots.values():
                definition = record.definition

                if (
                    definition.tenant_id == tenant_id
                    and definition.fleet_id == fleet_id
                ):
                    self._refresh(record)
                    results.append(
                        record.to_dict()
                    )

            return results

    # --------------------------------------------------------
    # LIST TASKS
    # --------------------------------------------------------

    def list_tasks(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
    ) -> list[dict[str, Any]]:
        with self._lock:
            return [
                record.to_dict()
                for record in self._tasks.values()
                if (
                    record.definition.tenant_id == tenant_id
                    and record.definition.fleet_id == fleet_id
                )
            ]

    # --------------------------------------------------------
    # FLEET STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        tenant_id: str,
        fleet_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            robots = [
                record
                for record in self._robots.values()
                if (
                    record.definition.tenant_id == tenant_id
                    and record.definition.fleet_id == fleet_id
                )
            ]

            tasks = [
                record
                for record in self._tasks.values()
                if (
                    record.definition.tenant_id == tenant_id
                    and record.definition.fleet_id == fleet_id
                )
            ]

            for robot in robots:
                self._refresh(robot)

            return {
                "total_robots": len(robots),
                "idle_robots": sum(
                    robot.status == RobotStatus.IDLE
                    for robot in robots
                ),
                "busy_robots": sum(
                    robot.status == RobotStatus.BUSY
                    for robot in robots
                ),
                "charging_robots": sum(
                    robot.status == RobotStatus.CHARGING
                    for robot in robots
                ),
                "faulted_robots": sum(
                    robot.status == RobotStatus.FAULTED
                    for robot in robots
                ),
                "offline_robots": sum(
                    robot.status == RobotStatus.OFFLINE
                    for robot in robots
                ),
                "emergency_stopped_robots": sum(
                    robot.status
                    == RobotStatus.EMERGENCY_STOP
                    for robot in robots
                ),
                "total_tasks": len(tasks),
                "pending_tasks": sum(
                    task.status == FleetTaskStatus.PENDING
                    for task in tasks
                ),
                "running_tasks": sum(
                    task.status == FleetTaskStatus.RUNNING
                    for task in tasks
                ),
                "completed_tasks": sum(
                    task.status == FleetTaskStatus.COMPLETED
                    for task in tasks
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
        fleet_id: str,
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
                "fleet_id": fleet_id,
                "statistics": self.statistics(
                    tenant_id=tenant_id,
                    fleet_id=fleet_id,
                ),
                "robots": self.list_robots(
                    tenant_id=tenant_id,
                    fleet_id=fleet_id,
                ),
                "tasks": self.list_tasks(
                    tenant_id=tenant_id,
                    fleet_id=fleet_id,
                ),
                "audit_events": [
                    event.to_dict()
                    for event in self._audit_events
                    if (
                        event.tenant_id == tenant_id
                        and event.fleet_id == fleet_id
                    )
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
    manager = RoboticsFleetManager()

    scope = {
        "tenant_id": "tenant_alpha",
        "fleet_id": "factory_fleet",
    }

    manager.register_robot(
        FleetRobotDefinition(
            **scope,
            robot_id="robot_001",
            name="Inspection AMR",
            robot_type=RobotType.AMR,
            capabilities=(
                "navigation",
                "inspection",
                "object_detection",
            ),
            location_zone="zone_a",
        )
    )

    manager.register_robot(
        FleetRobotDefinition(
            **scope,
            robot_id="robot_002",
            name="Mobile Manipulator",
            robot_type=RobotType.MOBILE_MANIPULATOR,
            capabilities=(
                "navigation",
                "manipulation",
                "inspection",
            ),
            location_zone="zone_b",
        )
    )

    manager.ingest_telemetry(
        FleetRobotTelemetry(
            **scope,
            robot_id="robot_001",
            sequence_number=1,
            battery_percentage=85.0,
            temperature_c=36.0,
        )
    )

    manager.ingest_telemetry(
        FleetRobotTelemetry(
            **scope,
            robot_id="robot_002",
            sequence_number=1,
            battery_percentage=70.0,
            temperature_c=39.0,
        )
    )

    manager.register_task(
        FleetTaskDefinition(
            **scope,
            task_id="inspection_001",
            task_type=FleetTaskType.INSPECTION,
            required_capabilities=(
                "navigation",
                "inspection",
            ),
            preferred_zone="zone_a",
            priority=80,
        )
    )

    print(
        "Assignment:",
        json.dumps(
            manager.assign_task(
                **scope,
                task_id="inspection_001",
            ),
            indent=2,
        ),
    )

    manager.start_task(
        **scope,
        task_id="inspection_001",
    )

    manager.finish_task(
        **scope,
        task_id="inspection_001",
        success=True,
    )

    print(
        "Fleet statistics:",
        json.dumps(
            manager.statistics(**scope),
            indent=2,
        ),
    )

    report = manager.export_json(
        "data/robotics/fleet_manager_report.json",
        **scope,
    )

    print("Report:", report)


if __name__ == "__main__":
    main()

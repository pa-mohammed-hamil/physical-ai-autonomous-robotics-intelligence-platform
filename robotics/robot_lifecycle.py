"""Robot Lifecycle module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
backend/app/robotics/control/robot_lifecycle.py

Python 3.11+
Standard library only.

ROBOT LIFECYCLE MANAGER — SIMULATION ONLY

Features:
- Multi-robot lifecycle registry
- Strict lifecycle state transitions
- Robot initialization and readiness checks
- Activation and deactivation
- Maintenance mode
- Heartbeat monitoring
- Fault and emergency-stop latching
- Controlled recovery
- Graceful shutdown
- Tenant isolation
- Audit events
- JSON report export

This module never issues physical robot commands.

A simulated emergency stop is NOT a substitute for
a safety-rated hardware emergency-stop system.
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

class RobotLifecycleError(Exception):
    pass


class LifecycleValidationError(RobotLifecycleError):
    pass


class LifecycleRobotNotFound(RobotLifecycleError):
    pass


class LifecycleConflict(RobotLifecycleError):
    pass


class LifecyclePermissionDenied(RobotLifecycleError):
    pass


class LifecycleSafetyError(RobotLifecycleError):
    pass


# ============================================================
# ENUMS
# ============================================================

class RobotLifecycleState(str, Enum):
    REGISTERED = "registered"
    INITIALIZING = "initializing"
    INACTIVE = "inactive"
    ACTIVE = "active"
    DEACTIVATING = "deactivating"
    MAINTENANCE = "maintenance"
    FAULTED = "faulted"
    EMERGENCY_STOP = "emergency_stop"
    SHUTDOWN = "shutdown"


class RobotLifecycleEventType(str, Enum):
    REGISTERED = "registered"
    INITIALIZATION_STARTED = "initialization_started"
    INITIALIZED = "initialized"
    ACTIVATED = "activated"
    DEACTIVATED = "deactivated"
    MAINTENANCE_ENTERED = "maintenance_entered"
    MAINTENANCE_EXITED = "maintenance_exited"
    HEARTBEAT_RECEIVED = "heartbeat_received"
    HEARTBEAT_TIMEOUT = "heartbeat_timeout"
    FAULT_LATCHED = "fault_latched"
    EMERGENCY_STOP = "emergency_stop"
    RECOVERED = "recovered"
    SHUTDOWN = "shutdown"


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
        raise LifecycleValidationError(
            f"Invalid {label}"
        )

    return value


def positive_number(
    value: float,
    label: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise LifecycleValidationError(
            f"{label} must be positive and finite"
        )

    return float(value)


def validate_timestamp(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise LifecycleValidationError(
            "Timezone-aware timestamp required"
        )

    return value.astimezone(timezone.utc)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RobotLifecycleConfig:
    maximum_robots: int = 10000
    maximum_audit_events: int = 20000

    heartbeat_timeout_seconds: float = 5.0
    maximum_heartbeat_age_seconds: float = 5.0

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_robots",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise LifecycleValidationError(
                    f"{name} must be a positive integer"
                )

        positive_number(
            self.heartbeat_timeout_seconds,
            "heartbeat_timeout_seconds",
        )

        positive_number(
            self.maximum_heartbeat_age_seconds,
            "maximum_heartbeat_age_seconds",
        )

        if not self.simulation_only:
            raise LifecycleValidationError(
                "Physical robot lifecycle control unsupported"
            )


# ============================================================
# ROBOT DEFINITION
# ============================================================

@dataclass(frozen=True)
class LifecycleRobotDefinition:
    tenant_id: str
    robot_id: str
    name: str
    robot_type: str

    required_capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")
        validate_id(self.robot_type, "robot_type")

        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or len(self.name) > 256
        ):
            raise LifecycleValidationError(
                "Invalid robot name"
            )

        if not isinstance(
            self.required_capabilities,
            tuple,
        ):
            raise LifecycleValidationError(
                "required_capabilities must be a tuple"
            )

        for capability in self.required_capabilities:
            validate_id(capability, "capability")

        if (
            len(self.required_capabilities)
            != len(set(self.required_capabilities))
        ):
            raise LifecycleValidationError(
                "Duplicate capabilities"
            )


# ============================================================
# HEARTBEAT
# ============================================================

@dataclass(frozen=True)
class RobotLifecycleHeartbeat:
    tenant_id: str
    robot_id: str
    sequence_number: int

    operational: bool = True
    fault_detected: bool = False

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")

        if (
            isinstance(self.sequence_number, bool)
            or not isinstance(self.sequence_number, int)
            or self.sequence_number < 0
        ):
            raise LifecycleValidationError(
                "Invalid heartbeat sequence"
            )

        if (
            not isinstance(self.operational, bool)
            or not isinstance(self.fault_detected, bool)
        ):
            raise LifecycleValidationError(
                "Heartbeat flags must be boolean"
            )

        validate_timestamp(self.timestamp)


# ============================================================
# ROBOT RECORD
# ============================================================

@dataclass
class RobotLifecycleRecord:
    definition: LifecycleRobotDefinition

    state: RobotLifecycleState = (
        RobotLifecycleState.REGISTERED
    )

    initialized: bool = False
    fault_latched: bool = False
    emergency_latched: bool = False

    fault_reasons: list[str] = field(
        default_factory=list
    )

    last_heartbeat_sequence: int = -1
    last_heartbeat_monotonic: float | None = None

    active_tasks: int = 0

    transition_count: int = 0

    registered_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.definition.tenant_id,
            "robot_id": self.definition.robot_id,
            "name": self.definition.name,
            "robot_type": self.definition.robot_type,
            "required_capabilities": list(
                self.definition.required_capabilities
            ),
            "state": self.state.value,
            "initialized": self.initialized,
            "fault_latched": self.fault_latched,
            "emergency_latched": self.emergency_latched,
            "fault_reasons": list(
                self.fault_reasons
            ),
            "active_tasks": self.active_tasks,
            "transition_count": self.transition_count,
            "last_heartbeat_sequence": (
                self.last_heartbeat_sequence
            ),
            "registered_at": (
                self.registered_at.isoformat()
            ),
            "updated_at": (
                self.updated_at.isoformat()
            ),
        }


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class RobotLifecycleAuditEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    robot_id: str

    event_type: RobotLifecycleEventType
    previous_state: RobotLifecycleState
    new_state: RobotLifecycleState

    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "event_type": self.event_type.value,
            "previous_state": self.previous_state.value,
            "new_state": self.new_state.value,
            "details": self.details,
        }


# ============================================================
# READINESS CHECK
# ============================================================

RobotReadinessCheck = Callable[
    [LifecycleRobotDefinition],
    bool,
]


# ============================================================
# ROBOT LIFECYCLE MANAGER
# ============================================================

class RoboticsRobotLifecycleManager:

    """
    Thread-safe lifecycle manager for simulated robots.

    Typical lifecycle:

        REGISTERED
            |
        INITIALIZING
            |
          INACTIVE
            |
           ACTIVE
            |
          INACTIVE
            |
         SHUTDOWN

    Exceptional transitions:

        ACTIVE -> FAULTED
        ACTIVE -> EMERGENCY_STOP

    Recovery returns to INACTIVE and requires
    renewed readiness checks before activation.
    """

    def __init__(
        self,
        config: RobotLifecycleConfig | None = None,
        *,
        readiness_check: RobotReadinessCheck | None = None,
    ) -> None:
        self.config = (
            config or RobotLifecycleConfig()
        )

        self.readiness_check = readiness_check

        self._robots: dict[
            tuple[str, str],
            RobotLifecycleRecord
        ] = {}

        self._audit_events: deque[
            RobotLifecycleAuditEvent
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # LOOKUP
    # --------------------------------------------------------

    def _key(
        self,
        tenant_id: str,
        robot_id: str,
    ) -> tuple[str, str]:
        return (
            validate_id(tenant_id, "tenant_id"),
            validate_id(robot_id, "robot_id"),
        )

    def _get(
        self,
        tenant_id: str,
        robot_id: str,
    ) -> RobotLifecycleRecord:
        record = self._robots.get(
            self._key(
                tenant_id,
                robot_id,
            )
        )

        if record is None:
            raise LifecycleRobotNotFound(
                "Robot not registered"
            )

        return record

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _transition(
        self,
        record: RobotLifecycleRecord,
        new_state: RobotLifecycleState,
        event_type: RobotLifecycleEventType,
        details: str,
    ) -> None:
        previous = record.state

        record.state = new_state
        record.transition_count += 1
        record.updated_at = utc_now()

        self._audit_events.append(
            RobotLifecycleAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=record.updated_at,
                tenant_id=record.definition.tenant_id,
                robot_id=record.definition.robot_id,
                event_type=event_type,
                previous_state=previous,
                new_state=new_state,
                details=details,
            )
        )

    # --------------------------------------------------------
    # READINESS
    # --------------------------------------------------------

    def _ready(
        self,
        record: RobotLifecycleRecord,
    ) -> bool:
        if self.readiness_check is None:
            return False

        try:
            return (
                self.readiness_check(
                    record.definition
                )
                is True
            )
        except Exception:
            return False

    def _heartbeat_fresh(
        self,
        record: RobotLifecycleRecord,
    ) -> bool:
        last = record.last_heartbeat_monotonic

        return (
            last is not None
            and time.monotonic() - last
            <= self.config.heartbeat_timeout_seconds
        )

    # --------------------------------------------------------
    # REGISTER
    # --------------------------------------------------------

    def register_robot(
        self,
        definition: LifecycleRobotDefinition,
    ) -> dict[str, Any]:
        if not isinstance(
            definition,
            LifecycleRobotDefinition,
        ):
            raise LifecycleValidationError(
                "Invalid robot definition"
            )

        with self._lock:
            key = self._key(
                definition.tenant_id,
                definition.robot_id,
            )

            if key in self._robots:
                raise LifecycleConflict(
                    "Robot already registered"
                )

            if (
                len(self._robots)
                >= self.config.maximum_robots
            ):
                raise RobotLifecycleError(
                    "Robot registry capacity reached"
                )

            record = RobotLifecycleRecord(
                definition=definition
            )

            self._robots[key] = record

            self._transition(
                record,
                RobotLifecycleState.REGISTERED,
                RobotLifecycleEventType.REGISTERED,
                "Robot registered",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------

    def initialize_robot(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
            )

            if (
                record.state
                != RobotLifecycleState.REGISTERED
            ):
                raise LifecycleConflict(
                    "Robot must be REGISTERED"
                )

            self._transition(
                record,
                RobotLifecycleState.INITIALIZING,
                RobotLifecycleEventType.INITIALIZATION_STARTED,
                "Initialization started",
            )

            if not self._ready(record):
                record.fault_latched = True
                record.fault_reasons.append(
                    "initialization_readiness_failed"
                )

                self._transition(
                    record,
                    RobotLifecycleState.FAULTED,
                    RobotLifecycleEventType.FAULT_LATCHED,
                    "Initialization readiness failed",
                )

                return record.to_dict()

            record.initialized = True

            self._transition(
                record,
                RobotLifecycleState.INACTIVE,
                RobotLifecycleEventType.INITIALIZED,
                "Robot initialized",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # HEARTBEAT
    # --------------------------------------------------------

    def receive_heartbeat(
        self,
        heartbeat: RobotLifecycleHeartbeat,
    ) -> dict[str, Any]:
        if not isinstance(
            heartbeat,
            RobotLifecycleHeartbeat,
        ):
            raise LifecycleValidationError(
                "Invalid heartbeat"
            )

        age = (
            utc_now()
            - validate_timestamp(
                heartbeat.timestamp
            )
        ).total_seconds()

        if (
            age < 0
            or age
            > self.config.maximum_heartbeat_age_seconds
        ):
            raise LifecycleValidationError(
                "Heartbeat timestamp invalid or stale"
            )

        with self._lock:
            record = self._get(
                heartbeat.tenant_id,
                heartbeat.robot_id,
            )

            if (
                record.state
                == RobotLifecycleState.SHUTDOWN
            ):
                raise LifecycleConflict(
                    "Robot is shut down"
                )

            if (
                heartbeat.sequence_number
                <= record.last_heartbeat_sequence
            ):
                raise LifecycleValidationError(
                    "Out-of-order heartbeat"
                )

            record.last_heartbeat_sequence = (
                heartbeat.sequence_number
            )

            record.last_heartbeat_monotonic = (
                time.monotonic()
            )

            if (
                not heartbeat.operational
                or heartbeat.fault_detected
            ):
                self._latch_fault(
                    record,
                    "heartbeat_reported_fault",
                )

            self._audit_events.append(
                RobotLifecycleAuditEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=utc_now(),
                    tenant_id=heartbeat.tenant_id,
                    robot_id=heartbeat.robot_id,
                    event_type=(
                        RobotLifecycleEventType.HEARTBEAT_RECEIVED
                    ),
                    previous_state=record.state,
                    new_state=record.state,
                    details="Heartbeat received",
                )
            )

            return record.to_dict()

    # --------------------------------------------------------
    # FAULT HANDLING
    # --------------------------------------------------------

    def _latch_fault(
        self,
        record: RobotLifecycleRecord,
        reason: str,
    ) -> None:
        if (
            record.state
            == RobotLifecycleState.SHUTDOWN
        ):
            return

        record.fault_latched = True

        if reason not in record.fault_reasons:
            record.fault_reasons.append(reason)

        if not record.emergency_latched:
            self._transition(
                record,
                RobotLifecycleState.FAULTED,
                RobotLifecycleEventType.FAULT_LATCHED,
                reason,
            )

    def report_fault(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        reason: str,
    ) -> dict[str, Any]:
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or len(reason) > 512
        ):
            raise LifecycleValidationError(
                "Invalid fault reason"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
            )

            self._latch_fault(
                record,
                reason,
            )

            return record.to_dict()

    # --------------------------------------------------------
    # HEALTH REFRESH
    # --------------------------------------------------------

    def _refresh(
        self,
        record: RobotLifecycleRecord,
    ) -> None:
        if record.state in (
            RobotLifecycleState.SHUTDOWN,
            RobotLifecycleState.REGISTERED,
            RobotLifecycleState.INITIALIZING,
        ):
            return

        if record.emergency_latched:
            return

        if record.fault_latched:
            return

        if not self._heartbeat_fresh(record):
            self._latch_fault(
                record,
                "heartbeat_timeout",
            )

    # --------------------------------------------------------
    # ACTIVATE
    # --------------------------------------------------------

    def activate_robot(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
            )

            self._refresh(record)

            if (
                record.state
                != RobotLifecycleState.INACTIVE
            ):
                raise LifecycleConflict(
                    "Robot must be INACTIVE"
                )

            if (
                not record.initialized
                or record.fault_latched
                or record.emergency_latched
            ):
                raise LifecycleSafetyError(
                    "Robot is not ready for activation"
                )

            if not self._heartbeat_fresh(record):
                raise LifecycleSafetyError(
                    "Fresh heartbeat required"
                )

            if not self._ready(record):
                raise LifecycleSafetyError(
                    "Readiness check failed"
                )

            self._transition(
                record,
                RobotLifecycleState.ACTIVE,
                RobotLifecycleEventType.ACTIVATED,
                "Robot activated in simulation",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # TASK TRACKING
    # --------------------------------------------------------

    def set_active_tasks(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        count: int,
    ) -> dict[str, Any]:
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise LifecycleValidationError(
                "Invalid active task count"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
            )

            if (
                count > 0
                and record.state
                != RobotLifecycleState.ACTIVE
            ):
                raise LifecycleConflict(
                    "Tasks require ACTIVE state"
                )

            record.active_tasks = count
            record.updated_at = utc_now()

            return record.to_dict()

    # --------------------------------------------------------
    # DEACTIVATE
    # --------------------------------------------------------

    def deactivate_robot(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
            )

            if (
                record.state
                != RobotLifecycleState.ACTIVE
            ):
                raise LifecycleConflict(
                    "Robot must be ACTIVE"
                )

            if record.active_tasks > 0:
                raise LifecycleConflict(
                    "Active tasks must finish first"
                )

            self._transition(
                record,
                RobotLifecycleState.DEACTIVATING,
                RobotLifecycleEventType.DEACTIVATED,
                "Deactivation started",
            )

            self._transition(
                record,
                RobotLifecycleState.INACTIVE,
                RobotLifecycleEventType.DEACTIVATED,
                "Robot deactivated",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # MAINTENANCE
    # --------------------------------------------------------

    def enter_maintenance(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise LifecyclePermissionDenied(
                "Maintenance authorization required"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
            )

            if (
                record.state
                != RobotLifecycleState.INACTIVE
                or record.active_tasks != 0
            ):
                raise LifecycleConflict(
                    "Maintenance requires idle INACTIVE robot"
                )

            self._transition(
                record,
                RobotLifecycleState.MAINTENANCE,
                RobotLifecycleEventType.MAINTENANCE_ENTERED,
                "Maintenance mode entered",
            )

            return record.to_dict()

    def exit_maintenance(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise LifecyclePermissionDenied(
                "Maintenance authorization required"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
            )

            if (
                record.state
                != RobotLifecycleState.MAINTENANCE
            ):
                raise LifecycleConflict(
                    "Robot is not in maintenance"
                )

            if not self._ready(record):
                raise LifecycleSafetyError(
                    "Readiness verification failed"
                )

            if not self._heartbeat_fresh(record):
                raise LifecycleSafetyError(
                    "Fresh heartbeat required"
                )

            self._transition(
                record,
                RobotLifecycleState.INACTIVE,
                RobotLifecycleEventType.MAINTENANCE_EXITED,
                "Maintenance completed",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # EMERGENCY STOP
    # --------------------------------------------------------

    def emergency_stop(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
            )

            if (
                record.state
                == RobotLifecycleState.SHUTDOWN
            ):
                raise LifecycleConflict(
                    "Robot already shut down"
                )

            record.emergency_latched = True

            self._transition(
                record,
                RobotLifecycleState.EMERGENCY_STOP,
                RobotLifecycleEventType.EMERGENCY_STOP,
                "Simulation emergency stop latched",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # RECOVERY
    # --------------------------------------------------------

    def recover_robot(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        authorized: bool,
        independently_verified: bool,
    ) -> dict[str, Any]:
        if (
            authorized is not True
            or independently_verified is not True
        ):
            raise LifecyclePermissionDenied(
                "Verified recovery approval required"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
            )

            if record.state not in (
                RobotLifecycleState.FAULTED,
                RobotLifecycleState.EMERGENCY_STOP,
            ):
                raise LifecycleConflict(
                    "Robot is not in recoverable state"
                )

            if record.active_tasks > 0:
                raise LifecycleConflict(
                    "Active tasks must be resolved"
                )

            if not self._heartbeat_fresh(record):
                raise LifecycleSafetyError(
                    "Fresh heartbeat required"
                )

            if not self._ready(record):
                raise LifecycleSafetyError(
                    "Readiness check failed"
                )

            record.fault_latched = False
            record.emergency_latched = False
            record.fault_reasons.clear()

            self._transition(
                record,
                RobotLifecycleState.INACTIVE,
                RobotLifecycleEventType.RECOVERED,
                "Simulation recovery completed",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------------

    def shutdown_robot(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise LifecyclePermissionDenied(
                "Shutdown authorization required"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
            )

            if record.active_tasks > 0:
                raise LifecycleConflict(
                    "Cannot shut down with active tasks"
                )

            if (
                record.state
                == RobotLifecycleState.SHUTDOWN
            ):
                return record.to_dict()

            self._transition(
                record,
                RobotLifecycleState.SHUTDOWN,
                RobotLifecycleEventType.SHUTDOWN,
                "Simulation robot shut down",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # GET ROBOT
    # --------------------------------------------------------

    def get_robot(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
            )

            self._refresh(record)

            return record.to_dict()

    # --------------------------------------------------------
    # LIST ROBOTS
    # --------------------------------------------------------

    def list_robots(
        self,
        *,
        tenant_id: str,
        state: RobotLifecycleState | None = None,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        with self._lock:
            result = []

            for record in self._robots.values():
                if (
                    record.definition.tenant_id
                    != tenant_id
                ):
                    continue

                self._refresh(record)

                if (
                    state is not None
                    and record.state != state
                ):
                    continue

                result.append(
                    record.to_dict()
                )

            return sorted(
                result,
                key=lambda item: item["robot_id"],
            )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        tenant_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            records = [
                record
                for record in self._robots.values()
                if record.definition.tenant_id == tenant_id
            ]

            for record in records:
                self._refresh(record)

            return {
                "total_robots": len(records),
                "active": sum(
                    record.state
                    == RobotLifecycleState.ACTIVE
                    for record in records
                ),
                "inactive": sum(
                    record.state
                    == RobotLifecycleState.INACTIVE
                    for record in records
                ),
                "maintenance": sum(
                    record.state
                    == RobotLifecycleState.MAINTENANCE
                    for record in records
                ),
                "faulted": sum(
                    record.state
                    == RobotLifecycleState.FAULTED
                    for record in records
                ),
                "emergency_stop": sum(
                    record.state
                    == RobotLifecycleState.EMERGENCY_STOP
                    for record in records
                ),
                "shutdown": sum(
                    record.state
                    == RobotLifecycleState.SHUTDOWN
                    for record in records
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
                "robots": self.list_robots(
                    tenant_id=tenant_id
                ),
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
    manager = RoboticsRobotLifecycleManager(
        readiness_check=lambda definition: (
            "navigation"
            in definition.required_capabilities
        )
    )

    tenant_id = "tenant_alpha"
    robot_id = "robot_001"

    manager.register_robot(
        LifecycleRobotDefinition(
            tenant_id=tenant_id,
            robot_id=robot_id,
            name="Autonomous Mobile Robot",
            robot_type="amr",
            required_capabilities=(
                "navigation",
                "inspection",
            ),
        )
    )

    manager.initialize_robot(
        tenant_id=tenant_id,
        robot_id=robot_id,
    )

    manager.receive_heartbeat(
        RobotLifecycleHeartbeat(
            tenant_id=tenant_id,
            robot_id=robot_id,
            sequence_number=1,
            operational=True,
        )
    )

    manager.activate_robot(
        tenant_id=tenant_id,
        robot_id=robot_id,
    )

    print(
        "Active robot:",
        json.dumps(
            manager.get_robot(
                tenant_id=tenant_id,
                robot_id=robot_id,
            ),
            indent=2,
        ),
    )

    manager.deactivate_robot(
        tenant_id=tenant_id,
        robot_id=robot_id,
    )

    manager.enter_maintenance(
        tenant_id=tenant_id,
        robot_id=robot_id,
        authorized=True,
    )

    manager.exit_maintenance(
        tenant_id=tenant_id,
        robot_id=robot_id,
        authorized=True,
    )

    manager.shutdown_robot(
        tenant_id=tenant_id,
        robot_id=robot_id,
        authorized=True,
    )

    report = manager.export_json(
        "data/robotics/robot_lifecycle_report.json",
        tenant_id=tenant_id,
    )

    print(
        "\nStatistics:",
        json.dumps(
            manager.statistics(
                tenant_id=tenant_id
            ),
            indent=2,
        ),
    )

    print("\nReport:", report)


if __name__ == "__main__":
    main()

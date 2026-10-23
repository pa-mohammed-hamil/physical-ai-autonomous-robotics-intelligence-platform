"""Robot State module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
backend/app/robotics/control/robot_state.py

Python 3.11+
Standard library only.

ROBOT STATE MANAGER — SIMULATION ONLY

Features:
    - Multi-robot state management
    - Robot pose and orientation tracking
    - Linear/angular velocity tracking
    - Joint state monitoring
    - Battery and temperature monitoring
    - Health and fault status
    - Telemetry freshness monitoring
    - Monotonic sequence validation
    - State history and snapshots
    - Emergency-stop state latching
    - Tenant isolation
    - Audit logging
    - JSON report export

This module stores and evaluates state only.
It does not communicate with robot hardware.

Telemetry and health status are informational,
not safety-rated evidence of physical readiness.
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
from typing import Any


# ============================================================
# EXCEPTIONS
# ============================================================

class RobotStateError(Exception):
    pass


class RobotStateValidationError(RobotStateError):
    pass


class RobotStateNotFound(RobotStateError):
    pass


class RobotStateConflict(RobotStateError):
    pass


class RobotStatePermissionDenied(RobotStateError):
    pass


# ============================================================
# ENUMS
# ============================================================

class RobotOperationalState(str, Enum):
    UNKNOWN = "unknown"
    IDLE = "idle"
    MOVING = "moving"
    WORKING = "working"
    CHARGING = "charging"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    FAULTED = "faulted"
    EMERGENCY_STOP = "emergency_stop"


class RobotHealthStatus(str, Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAULTED = "faulted"
    STALE = "stale"


class RobotStateEventType(str, Enum):
    REGISTERED = "registered"
    STATE_UPDATED = "state_updated"
    STATE_STALE = "state_stale"
    HEALTH_CHANGED = "health_changed"
    FAULT_LATCHED = "fault_latched"
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
        or not re.fullmatch(
            r"[A-Za-z0-9_.-]{1,128}",
            value,
        )
    ):
        raise RobotStateValidationError(
            f"Invalid {label}"
        )

    return value


def finite_number(
    value: Any,
    label: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise RobotStateValidationError(
            f"{label} must be finite"
        )

    return float(value)


def percentage(
    value: Any,
    label: str,
) -> float:
    result = finite_number(value, label)

    if not 0 <= result <= 100:
        raise RobotStateValidationError(
            f"{label} must be between 0 and 100"
        )

    return result


def positive_number(
    value: Any,
    label: str,
) -> float:
    result = finite_number(value, label)

    if result <= 0:
        raise RobotStateValidationError(
            f"{label} must be positive"
        )

    return result


def validate_timestamp(
    value: datetime,
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise RobotStateValidationError(
            "Timezone-aware timestamp required"
        )

    return value.astimezone(timezone.utc)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RobotStateConfig:
    maximum_robots: int = 10000
    maximum_history_per_robot: int = 1000
    maximum_audit_events: int = 20000

    telemetry_timeout_seconds: float = 5.0

    minimum_battery_percentage: float = 20.0
    critical_battery_percentage: float = 10.0

    warning_temperature_c: float = 65.0
    critical_temperature_c: float = 80.0

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_robots",
            "maximum_history_per_robot",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise RobotStateValidationError(
                    f"{name} must be positive"
                )

        positive_number(
            self.telemetry_timeout_seconds,
            "telemetry_timeout_seconds",
        )

        percentage(
            self.minimum_battery_percentage,
            "minimum_battery_percentage",
        )

        percentage(
            self.critical_battery_percentage,
            "critical_battery_percentage",
        )

        if (
            self.critical_battery_percentage
            > self.minimum_battery_percentage
        ):
            raise RobotStateValidationError(
                "Invalid battery thresholds"
            )

        finite_number(
            self.warning_temperature_c,
            "warning_temperature_c",
        )

        finite_number(
            self.critical_temperature_c,
            "critical_temperature_c",
        )

        if (
            self.critical_temperature_c
            <= self.warning_temperature_c
        ):
            raise RobotStateValidationError(
                "Critical temperature must exceed warning"
            )

        if not self.simulation_only:
            raise RobotStateValidationError(
                "Physical robot state control unsupported"
            )


# ============================================================
# GEOMETRY
# ============================================================

@dataclass(frozen=True)
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __post_init__(self) -> None:
        for axis in ("x", "y", "z"):
            finite_number(
                getattr(self, axis),
                axis,
            )

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }


@dataclass(frozen=True)
class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def __post_init__(self) -> None:
        for axis in ("x", "y", "z", "w"):
            finite_number(
                getattr(self, axis),
                axis,
            )

        norm = math.sqrt(
            self.x ** 2
            + self.y ** 2
            + self.z ** 2
            + self.w ** 2
        )

        if not math.isclose(
            norm,
            1.0,
            abs_tol=1e-4,
        ):
            raise RobotStateValidationError(
                "Quaternion must be normalized"
            )

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "w": self.w,
        }


@dataclass(frozen=True)
class RobotPose:
    position: Vector3 = field(
        default_factory=Vector3
    )

    orientation: Quaternion = field(
        default_factory=Quaternion
    )

    frame_id: str = "map"

    def __post_init__(self) -> None:
        if not isinstance(
            self.position,
            Vector3,
        ):
            raise RobotStateValidationError(
                "Invalid pose position"
            )

        if not isinstance(
            self.orientation,
            Quaternion,
        ):
            raise RobotStateValidationError(
                "Invalid pose orientation"
            )

        validate_id(
            self.frame_id,
            "frame_id",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position.to_dict(),
            "orientation": (
                self.orientation.to_dict()
            ),
            "frame_id": self.frame_id,
        }


# ============================================================
# JOINT STATE
# ============================================================

@dataclass(frozen=True)
class RobotJointState:
    joint_name: str

    position: float
    velocity: float
    effort: float = 0.0

    temperature_c: float | None = None

    def __post_init__(self) -> None:
        validate_id(
            self.joint_name,
            "joint_name",
        )

        for name in (
            "position",
            "velocity",
            "effort",
        ):
            finite_number(
                getattr(self, name),
                name,
            )

        if self.temperature_c is not None:
            finite_number(
                self.temperature_c,
                "temperature_c",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_name": self.joint_name,
            "position": self.position,
            "velocity": self.velocity,
            "effort": self.effort,
            "temperature_c": self.temperature_c,
        }


# ============================================================
# ROBOT STATE OBSERVATION
# ============================================================

@dataclass(frozen=True)
class RobotStateObservation:
    tenant_id: str
    robot_id: str

    sequence_number: int

    operational_state: RobotOperationalState

    pose: RobotPose

    linear_velocity: Vector3 = field(
        default_factory=Vector3
    )

    angular_velocity: Vector3 = field(
        default_factory=Vector3
    )

    battery_percentage: float = 100.0
    temperature_c: float = 25.0

    joints: tuple[RobotJointState, ...] = ()

    fault_detected: bool = False
    emergency_stop_active: bool = False

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
            raise RobotStateValidationError(
                "Invalid sequence number"
            )

        if not isinstance(
            self.operational_state,
            RobotOperationalState,
        ):
            raise RobotStateValidationError(
                "Invalid operational state"
            )

        if not isinstance(self.pose, RobotPose):
            raise RobotStateValidationError(
                "Invalid pose"
            )

        for name in (
            "linear_velocity",
            "angular_velocity",
        ):
            if not isinstance(
                getattr(self, name),
                Vector3,
            ):
                raise RobotStateValidationError(
                    f"Invalid {name}"
                )

        percentage(
            self.battery_percentage,
            "battery_percentage",
        )

        finite_number(
            self.temperature_c,
            "temperature_c",
        )

        if not isinstance(self.joints, tuple):
            raise RobotStateValidationError(
                "Joints must be a tuple"
            )

        joint_names = []

        for joint in self.joints:
            if not isinstance(
                joint,
                RobotJointState,
            ):
                raise RobotStateValidationError(
                    "Invalid joint state"
                )

            joint_names.append(
                joint.joint_name
            )

        if len(joint_names) != len(set(joint_names)):
            raise RobotStateValidationError(
                "Duplicate joint names"
            )

        if (
            not isinstance(self.fault_detected, bool)
            or not isinstance(
                self.emergency_stop_active,
                bool,
            )
        ):
            raise RobotStateValidationError(
                "Fault flags must be boolean"
            )

        validate_timestamp(
            self.timestamp
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "sequence_number": self.sequence_number,
            "operational_state": (
                self.operational_state.value
            ),
            "pose": self.pose.to_dict(),
            "linear_velocity": (
                self.linear_velocity.to_dict()
            ),
            "angular_velocity": (
                self.angular_velocity.to_dict()
            ),
            "battery_percentage": (
                self.battery_percentage
            ),
            "temperature_c": self.temperature_c,
            "joints": [
                joint.to_dict()
                for joint in self.joints
            ],
            "fault_detected": (
                self.fault_detected
            ),
            "emergency_stop_active": (
                self.emergency_stop_active
            ),
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================
# STATE RECORD
# ============================================================

@dataclass
class RobotStateRecord:
    tenant_id: str
    robot_id: str

    observation: RobotStateObservation | None = None

    health_status: RobotHealthStatus = (
        RobotHealthStatus.UNKNOWN
    )

    last_sequence_number: int = -1

    last_received_monotonic: float | None = None

    fault_latched: bool = False
    emergency_latched: bool = False

    fault_reasons: list[str] = field(
        default_factory=list
    )

    history: deque[RobotStateObservation] = field(
        default_factory=deque
    )

    registered_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "health_status": self.health_status.value,
            "last_sequence_number": (
                self.last_sequence_number
            ),
            "fault_latched": self.fault_latched,
            "emergency_latched": self.emergency_latched,
            "fault_reasons": list(
                self.fault_reasons
            ),
            "history_count": len(self.history),
            "observation": (
                self.observation.to_dict()
                if self.observation
                else None
            ),
            "registered_at": (
                self.registered_at.isoformat()
            ),
        }


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class RobotStateAuditEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    robot_id: str

    event_type: RobotStateEventType

    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "event_type": self.event_type.value,
            "details": self.details,
        }


# ============================================================
# ROBOT STATE MANAGER
# ============================================================

class RoboticsRobotStateManager:

    """
    Thread-safe robot state manager.

    State observations are accepted only when:
        - Robot is registered
        - Tenant/robot identity matches
        - Sequence number increases
        - Timestamp is fresh
        - All numeric fields are valid

    Fault and emergency-stop conditions latch
    until explicitly reviewed and reset.
    """

    def __init__(
        self,
        config: RobotStateConfig | None = None,
    ) -> None:
        self.config = (
            config or RobotStateConfig()
        )

        self._robots: dict[
            tuple[str, str],
            RobotStateRecord
        ] = {}

        self._audit_events: deque[
            RobotStateAuditEvent
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # KEYS
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
    ) -> RobotStateRecord:
        record = self._robots.get(
            self._key(
                tenant_id,
                robot_id,
            )
        )

        if record is None:
            raise RobotStateNotFound(
                "Robot state not registered"
            )

        return record

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        event_type: RobotStateEventType,
        details: str,
    ) -> None:
        self._audit_events.append(
            RobotStateAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                tenant_id=tenant_id,
                robot_id=robot_id,
                event_type=event_type,
                details=details,
            )
        )

    # --------------------------------------------------------
    # REGISTER ROBOT
    # --------------------------------------------------------

    def register_robot(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            key = self._key(
                tenant_id,
                robot_id,
            )

            if key in self._robots:
                raise RobotStateConflict(
                    "Robot already registered"
                )

            if (
                len(self._robots)
                >= self.config.maximum_robots
            ):
                raise RobotStateError(
                    "Maximum robot capacity reached"
                )

            record = RobotStateRecord(
                tenant_id=tenant_id,
                robot_id=robot_id,
                history=deque(
                    maxlen=(
                        self.config.maximum_history_per_robot
                    )
                ),
            )

            self._robots[key] = record

            self._audit(
                tenant_id=tenant_id,
                robot_id=robot_id,
                event_type=RobotStateEventType.REGISTERED,
                details="Robot state registered",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # HEALTH EVALUATION
    # --------------------------------------------------------

    def _evaluate_health(
        self,
        record: RobotStateRecord,
    ) -> RobotHealthStatus:
        observation = record.observation

        if record.emergency_latched:
            return RobotHealthStatus.CRITICAL

        if record.fault_latched:
            return RobotHealthStatus.FAULTED

        if observation is None:
            return RobotHealthStatus.UNKNOWN

        received = record.last_received_monotonic

        if (
            received is None
            or time.monotonic() - received
            > self.config.telemetry_timeout_seconds
        ):
            return RobotHealthStatus.STALE

        if (
            observation.battery_percentage
            <= self.config.critical_battery_percentage
            or observation.temperature_c
            >= self.config.critical_temperature_c
        ):
            return RobotHealthStatus.CRITICAL

        if (
            observation.battery_percentage
            < self.config.minimum_battery_percentage
            or observation.temperature_c
            >= self.config.warning_temperature_c
            or observation.operational_state
            == RobotOperationalState.DEGRADED
        ):
            return RobotHealthStatus.DEGRADED

        return RobotHealthStatus.HEALTHY

    def _refresh(
        self,
        record: RobotStateRecord,
    ) -> None:
        previous = record.health_status

        current = self._evaluate_health(
            record
        )

        record.health_status = current

        if previous != current:
            self._audit(
                tenant_id=record.tenant_id,
                robot_id=record.robot_id,
                event_type=(
                    RobotStateEventType.HEALTH_CHANGED
                ),
                details=(
                    f"{previous.value} -> {current.value}"
                ),
            )

            if current == RobotHealthStatus.STALE:
                self._audit(
                    tenant_id=record.tenant_id,
                    robot_id=record.robot_id,
                    event_type=(
                        RobotStateEventType.STATE_STALE
                    ),
                    details="Robot telemetry timed out",
                )

    # --------------------------------------------------------
    # INGEST STATE
    # --------------------------------------------------------

    def ingest_state(
        self,
        observation: RobotStateObservation,
    ) -> dict[str, Any]:
        if not isinstance(
            observation,
            RobotStateObservation,
        ):
            raise RobotStateValidationError(
                "Invalid state observation"
            )

        age = (
            utc_now()
            - validate_timestamp(
                observation.timestamp
            )
        ).total_seconds()

        if (
            age < 0
            or age
            > self.config.telemetry_timeout_seconds
        ):
            raise RobotStateValidationError(
                "Stale or future observation"
            )

        with self._lock:
            record = self._get(
                observation.tenant_id,
                observation.robot_id,
            )

            if (
                observation.sequence_number
                <= record.last_sequence_number
            ):
                raise RobotStateValidationError(
                    "Out-of-order observation"
                )

            record.observation = observation

            record.last_sequence_number = (
                observation.sequence_number
            )

            record.last_received_monotonic = (
                time.monotonic()
            )

            record.history.append(
                observation
            )

            if observation.fault_detected:
                record.fault_latched = True

                if (
                    "telemetry_fault"
                    not in record.fault_reasons
                ):
                    record.fault_reasons.append(
                        "telemetry_fault"
                    )

                self._audit(
                    tenant_id=observation.tenant_id,
                    robot_id=observation.robot_id,
                    event_type=(
                        RobotStateEventType.FAULT_LATCHED
                    ),
                    details="Telemetry reported fault",
                )

            if observation.emergency_stop_active:
                record.emergency_latched = True

                self._audit(
                    tenant_id=observation.tenant_id,
                    robot_id=observation.robot_id,
                    event_type=(
                        RobotStateEventType.EMERGENCY_STOP
                    ),
                    details="Emergency stop reported",
                )

            self._refresh(record)

            self._audit(
                tenant_id=observation.tenant_id,
                robot_id=observation.robot_id,
                event_type=(
                    RobotStateEventType.STATE_UPDATED
                ),
                details="Robot state updated",
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

            record.emergency_latched = True

            self._refresh(record)

            self._audit(
                tenant_id=tenant_id,
                robot_id=robot_id,
                event_type=(
                    RobotStateEventType.EMERGENCY_STOP
                ),
                details="Simulation emergency stop latched",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # RESET EMERGENCY STOP
    # --------------------------------------------------------

    def reset_emergency_stop(
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
            raise RobotStatePermissionDenied(
                "Verified emergency reset required"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
            )

            if record.observation is None:
                raise RobotStateConflict(
                    "Fresh state observation required"
                )

            if (
                record.observation.emergency_stop_active
            ):
                raise RobotStateConflict(
                    "Telemetry still reports emergency stop"
                )

            if (
                record.last_received_monotonic is None
                or time.monotonic()
                - record.last_received_monotonic
                > self.config.telemetry_timeout_seconds
            ):
                raise RobotStateConflict(
                    "Telemetry is stale"
                )

            record.emergency_latched = False

            self._refresh(record)

            self._audit(
                tenant_id=tenant_id,
                robot_id=robot_id,
                event_type=(
                    RobotStateEventType.EMERGENCY_RESET
                ),
                details="Simulation emergency latch reset",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # CLEAR FAULT
    # --------------------------------------------------------

    def clear_fault(
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
            raise RobotStatePermissionDenied(
                "Verified fault clearance required"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
            )

            observation = record.observation

            if observation is None:
                raise RobotStateConflict(
                    "State observation required"
                )

            if observation.fault_detected:
                raise RobotStateConflict(
                    "Telemetry still reports fault"
                )

            if (
                record.last_received_monotonic is None
                or time.monotonic()
                - record.last_received_monotonic
                > self.config.telemetry_timeout_seconds
            ):
                raise RobotStateConflict(
                    "Fresh telemetry required"
                )

            record.fault_latched = False
            record.fault_reasons.clear()

            self._refresh(record)

            return record.to_dict()

    # --------------------------------------------------------
    # GET STATE
    # --------------------------------------------------------

    def get_state(
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
    # GET HISTORY
    # --------------------------------------------------------

    def get_history(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit <= 0
        ):
            raise RobotStateValidationError(
                "Invalid history limit"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
            )

            observations = list(
                record.history
            )[-limit:]

            return [
                observation.to_dict()
                for observation in observations
            ]

    # --------------------------------------------------------
    # LIST ROBOTS
    # --------------------------------------------------------

    def list_robots(
        self,
        *,
        tenant_id: str,
        health_status: RobotHealthStatus | None = None,
    ) -> list[dict[str, Any]]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        with self._lock:
            results = []

            for record in self._robots.values():
                if record.tenant_id != tenant_id:
                    continue

                self._refresh(record)

                if (
                    health_status is not None
                    and record.health_status
                    != health_status
                ):
                    continue

                results.append(
                    record.to_dict()
                )

            return sorted(
                results,
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
        robots = self.list_robots(
            tenant_id=tenant_id
        )

        health_counts = {
            status.value: 0
            for status in RobotHealthStatus
        }

        for robot in robots:
            health_counts[
                robot["health_status"]
            ] += 1

        return {
            "tenant_id": tenant_id,
            "total_robots": len(robots),
            "health_counts": health_counts,
            "emergency_latched": sum(
                robot["emergency_latched"]
                for robot in robots
            ),
            "fault_latched": sum(
                robot["fault_latched"]
                for robot in robots
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
    manager = RoboticsRobotStateManager()

    tenant_id = "tenant_alpha"
    robot_id = "robot_001"

    manager.register_robot(
        tenant_id=tenant_id,
        robot_id=robot_id,
    )

    observation = RobotStateObservation(
        tenant_id=tenant_id,
        robot_id=robot_id,
        sequence_number=1,
        operational_state=(
            RobotOperationalState.MOVING
        ),
        pose=RobotPose(
            position=Vector3(
                x=2.0,
                y=3.0,
                z=0.0,
            ),
            orientation=Quaternion(),
            frame_id="map",
        ),
        linear_velocity=Vector3(
            x=0.5,
            y=0.0,
            z=0.0,
        ),
        angular_velocity=Vector3(
            x=0.0,
            y=0.0,
            z=0.1,
        ),
        battery_percentage=85.0,
        temperature_c=36.0,
        joints=(
            RobotJointState(
                joint_name="joint_1",
                position=0.5,
                velocity=0.1,
                effort=1.2,
                temperature_c=35.0,
            ),
        ),
    )

    print(
        "State update:",
        json.dumps(
            manager.ingest_state(observation),
            indent=2,
        ),
    )

    print(
        "\nState history:",
        json.dumps(
            manager.get_history(
                tenant_id=tenant_id,
                robot_id=robot_id,
            ),
            indent=2,
        ),
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

    report = manager.export_json(
        "data/robotics/robot_state_report.json",
        tenant_id=tenant_id,
    )

    print("\nReport:", report)


if __name__ == "__main__":
    main()

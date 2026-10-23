"""Actuator Manager module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
backend/app/robotics/control/actuator_manager.py

Python 3.11+
Standard library only.

SIMULATION-ONLY ACTUATOR MANAGER

Features:
- Actuator registration
- Motor, servo, joint and gripper support
- Position, velocity and effort limits
- Temperature and current monitoring
- Telemetry freshness validation
- Sequence-based telemetry validation
- Simulation command validation
- External authorization callback
- Emergency-stop fault latching
- Explicit recovery workflow
- Per-robot actuator health assessment
- Tenant and robot isolation
- Thread-safe operations
- Audit logging
- JSON report export

SAFETY:
    Does not control real hardware.

    Does not send PWM, CAN, EtherCAT,
    Modbus, serial, ROS 2, or motor commands.

    An approved command is only a simulated
    command record, not physical authorization.

    Real robots require independent,
    hardware-enforced safety mechanisms.
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
from typing import Any, Callable


# ============================================================
# EXCEPTIONS
# ============================================================

class ActuatorManagerError(Exception):
    pass


class ActuatorValidationError(ActuatorManagerError):
    pass


class ActuatorNotFound(ActuatorManagerError):
    pass


class ActuatorPermissionDenied(ActuatorManagerError):
    pass


class ActuatorSafetyError(ActuatorManagerError):
    pass


# ============================================================
# ENUMS
# ============================================================

class ActuatorType(str, Enum):
    DC_MOTOR = "dc_motor"
    BLDC_MOTOR = "bldc_motor"
    SERVO = "servo"
    STEPPER = "stepper"
    JOINT = "joint"
    LINEAR = "linear"
    GRIPPER = "gripper"
    WHEEL = "wheel"
    OTHER = "other"


class ActuatorMode(str, Enum):
    DISABLED = "disabled"
    POSITION = "position"
    VELOCITY = "velocity"
    EFFORT = "effort"
    HOLD = "hold"


class ActuatorHealth(str, Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAULTED = "faulted"


class ActuatorCommandStatus(str, Enum):
    ACCEPTED_SIMULATION = "accepted_simulation"
    REJECTED = "rejected"


class ActuatorEventType(str, Enum):
    REGISTERED = "registered"
    TELEMETRY_RECEIVED = "telemetry_received"
    COMMAND_ACCEPTED = "command_accepted"
    COMMAND_REJECTED = "command_rejected"
    TELEMETRY_STALE = "telemetry_stale"
    FAULT_LATCHED = "fault_latched"
    EMERGENCY_STOP = "emergency_stop"
    RECOVERY_COMPLETED = "recovery_completed"


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
            c.isalnum() or c in "_.-"
            for c in value
        )
    ):
        raise ActuatorValidationError(
            f"Invalid {name}"
        )

    return value


def finite(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ActuatorValidationError(
            f"{name} must be finite"
        )

    return float(value)


def positive(value: float, name: str) -> float:
    result = finite(value, name)

    if result <= 0:
        raise ActuatorValidationError(
            f"{name} must be positive"
        )

    return result


def validate_time(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ActuatorValidationError(
            "Timezone-aware timestamp required"
        )

    return value.astimezone(timezone.utc)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ActuatorManagerConfig:
    maximum_actuators: int = 10000
    maximum_audit_events: int = 20000

    default_telemetry_timeout_seconds: float = 2.0

    maximum_command_ttl_seconds: float = 5.0

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_actuators",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ActuatorValidationError(
                    f"{name} must be positive"
                )

        positive(
            self.default_telemetry_timeout_seconds,
            "default_telemetry_timeout_seconds",
        )

        positive(
            self.maximum_command_ttl_seconds,
            "maximum_command_ttl_seconds",
        )

        if not self.simulation_only:
            raise ActuatorValidationError(
                "Physical actuator control unsupported"
            )


# ============================================================
# ACTUATOR LIMITS
# ============================================================

@dataclass(frozen=True)
class ActuatorLimits:
    minimum_position: float
    maximum_position: float

    maximum_velocity: float
    maximum_effort: float

    maximum_temperature_c: float
    maximum_current_a: float

    def __post_init__(self) -> None:
        minimum = finite(
            self.minimum_position,
            "minimum_position",
        )

        maximum = finite(
            self.maximum_position,
            "maximum_position",
        )

        if minimum >= maximum:
            raise ActuatorValidationError(
                "Invalid position limits"
            )

        for name in (
            "maximum_velocity",
            "maximum_effort",
            "maximum_temperature_c",
            "maximum_current_a",
        ):
            positive(
                getattr(self, name),
                name,
            )


# ============================================================
# ACTUATOR DEFINITION
# ============================================================

@dataclass(frozen=True)
class ActuatorDefinition:
    tenant_id: str
    robot_id: str
    actuator_id: str

    actuator_type: ActuatorType
    limits: ActuatorLimits

    critical: bool = True

    telemetry_timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")
        validate_id(self.actuator_id, "actuator_id")

        if not isinstance(
            self.actuator_type,
            ActuatorType,
        ):
            raise ActuatorValidationError(
                "Invalid actuator type"
            )

        if not isinstance(
            self.limits,
            ActuatorLimits,
        ):
            raise ActuatorValidationError(
                "Invalid actuator limits"
            )

        positive(
            self.telemetry_timeout_seconds,
            "telemetry_timeout_seconds",
        )


# ============================================================
# ACTUATOR TELEMETRY
# ============================================================

@dataclass(frozen=True)
class ActuatorTelemetry:
    tenant_id: str
    robot_id: str
    actuator_id: str

    sequence_number: int

    position: float
    velocity: float
    effort: float

    temperature_c: float
    current_a: float

    enabled: bool = True

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")
        validate_id(self.actuator_id, "actuator_id")

        if (
            isinstance(self.sequence_number, bool)
            or not isinstance(self.sequence_number, int)
            or self.sequence_number < 0
        ):
            raise ActuatorValidationError(
                "Invalid telemetry sequence"
            )

        for name in (
            "position",
            "velocity",
            "effort",
            "temperature_c",
            "current_a",
        ):
            finite(
                getattr(self, name),
                name,
            )

        if not isinstance(self.enabled, bool):
            raise ActuatorValidationError(
                "enabled must be boolean"
            )

        validate_time(self.timestamp)


# ============================================================
# ACTUATOR COMMAND
# ============================================================

@dataclass(frozen=True)
class ActuatorCommand:
    tenant_id: str
    robot_id: str
    actuator_id: str

    mode: ActuatorMode

    target_value: float

    actor_id: str

    command_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    ttl_seconds: float = 2.0

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for name in (
            "tenant_id",
            "robot_id",
            "actuator_id",
            "actor_id",
            "command_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        if not isinstance(self.mode, ActuatorMode):
            raise ActuatorValidationError(
                "Invalid actuator mode"
            )

        finite(
            self.target_value,
            "target_value",
        )

        positive(
            self.ttl_seconds,
            "ttl_seconds",
        )

        validate_time(self.timestamp)


# ============================================================
# RECORDS
# ============================================================

@dataclass
class ActuatorRecord:
    definition: ActuatorDefinition

    health: ActuatorHealth = (
        ActuatorHealth.UNKNOWN
    )

    mode: ActuatorMode = (
        ActuatorMode.DISABLED
    )

    last_telemetry: ActuatorTelemetry | None = None

    last_received_monotonic: float | None = None

    last_sequence_number: int = -1

    fault_latched: bool = False

    fault_reasons: list[str] = field(
        default_factory=list
    )

    command_count: int = 0
    telemetry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        telemetry = self.last_telemetry

        return {
            "tenant_id": self.definition.tenant_id,
            "robot_id": self.definition.robot_id,
            "actuator_id": self.definition.actuator_id,
            "actuator_type": (
                self.definition.actuator_type.value
            ),
            "critical": self.definition.critical,
            "health": self.health.value,
            "mode": self.mode.value,
            "fault_latched": self.fault_latched,
            "fault_reasons": list(
                self.fault_reasons
            ),
            "command_count": self.command_count,
            "telemetry_count": self.telemetry_count,
            "last_sequence_number": (
                self.last_sequence_number
            ),
            "last_telemetry": (
                {
                    "position": telemetry.position,
                    "velocity": telemetry.velocity,
                    "effort": telemetry.effort,
                    "temperature_c": (
                        telemetry.temperature_c
                    ),
                    "current_a": telemetry.current_a,
                    "enabled": telemetry.enabled,
                    "timestamp": (
                        telemetry.timestamp.isoformat()
                    ),
                }
                if telemetry
                else None
            ),
        }


@dataclass(frozen=True)
class ActuatorCommandResult:
    command_id: str
    actuator_id: str

    status: ActuatorCommandStatus

    reason: str

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "actuator_id": self.actuator_id,
            "status": self.status.value,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class ActuatorAuditEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    robot_id: str
    actuator_id: str

    event_type: ActuatorEventType
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "actuator_id": self.actuator_id,
            "event_type": self.event_type.value,
            "details": self.details,
        }


# ============================================================
# AUTHORIZATION INTERFACE
# ============================================================

CommandAuthorizer = Callable[
    [ActuatorCommand],
    bool,
]


# ============================================================
# ACTUATOR MANAGER
# ============================================================

class RoboticsActuatorManager:

    def __init__(
        self,
        config: ActuatorManagerConfig | None = None,
        *,
        command_authorizer: (
            CommandAuthorizer | None
        ) = None,
    ) -> None:
        self.config = (
            config or ActuatorManagerConfig()
        )

        self.command_authorizer = (
            command_authorizer
        )

        self.actuators: dict[
            tuple[str, str, str],
            ActuatorRecord
        ] = {}

        self.audit_events: deque[
            ActuatorAuditEvent
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._seen_commands: set[
            tuple[str, str, str, str]
        ] = set()

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------

    def _key(
        self,
        tenant_id: str,
        robot_id: str,
        actuator_id: str,
    ) -> tuple[str, str, str]:
        return (
            validate_id(tenant_id, "tenant_id"),
            validate_id(robot_id, "robot_id"),
            validate_id(actuator_id, "actuator_id"),
        )

    def _get(
        self,
        tenant_id: str,
        robot_id: str,
        actuator_id: str,
    ) -> ActuatorRecord:
        record = self.actuators.get(
            self._key(
                tenant_id,
                robot_id,
                actuator_id,
            )
        )

        if record is None:
            raise ActuatorNotFound(
                "Actuator not registered"
            )

        return record

    def _audit(
        self,
        record: ActuatorRecord,
        event_type: ActuatorEventType,
        details: str,
    ) -> None:
        definition = record.definition

        self.audit_events.append(
            ActuatorAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                tenant_id=definition.tenant_id,
                robot_id=definition.robot_id,
                actuator_id=definition.actuator_id,
                event_type=event_type,
                details=details,
            )
        )

    def _latch_fault(
        self,
        record: ActuatorRecord,
        reason: str,
    ) -> None:
        record.fault_latched = True
        record.health = ActuatorHealth.FAULTED
        record.mode = ActuatorMode.DISABLED

        if reason not in record.fault_reasons:
            record.fault_reasons.append(
                reason
            )

        self._audit(
            record,
            ActuatorEventType.FAULT_LATCHED,
            reason,
        )

    # --------------------------------------------------------
    # REGISTER ACTUATOR
    # --------------------------------------------------------

    def register_actuator(
        self,
        definition: ActuatorDefinition,
    ) -> dict[str, Any]:
        with self._lock:
            key = self._key(
                definition.tenant_id,
                definition.robot_id,
                definition.actuator_id,
            )

            if key in self.actuators:
                raise ActuatorValidationError(
                    "Actuator already registered"
                )

            if (
                len(self.actuators)
                >= self.config.maximum_actuators
            ):
                raise ActuatorManagerError(
                    "Maximum actuator capacity reached"
                )

            record = ActuatorRecord(
                definition=definition
            )

            self.actuators[key] = record

            self._audit(
                record,
                ActuatorEventType.REGISTERED,
                "Actuator registered",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # INGEST TELEMETRY
    # --------------------------------------------------------

    def ingest_telemetry(
        self,
        telemetry: ActuatorTelemetry,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                telemetry.tenant_id,
                telemetry.robot_id,
                telemetry.actuator_id,
            )

            if (
                telemetry.sequence_number
                <= record.last_sequence_number
            ):
                raise ActuatorValidationError(
                    "Duplicate or out-of-order telemetry"
                )

            timestamp = validate_time(
                telemetry.timestamp
            )

            now = utc_now()

            if timestamp > now:
                raise ActuatorValidationError(
                    "Future telemetry rejected"
                )

            age = (
                now - timestamp
            ).total_seconds()

            if (
                age
                > record.definition.telemetry_timeout_seconds
            ):
                raise ActuatorValidationError(
                    "Stale telemetry rejected"
                )

            record.last_sequence_number = (
                telemetry.sequence_number
            )

            record.last_telemetry = telemetry

            record.last_received_monotonic = (
                time.monotonic()
            )

            record.telemetry_count += 1

            limits = record.definition.limits

            violations = []

            if not (
                limits.minimum_position
                <= telemetry.position
                <= limits.maximum_position
            ):
                violations.append(
                    "position_limit_exceeded"
                )

            if (
                abs(telemetry.velocity)
                > limits.maximum_velocity
            ):
                violations.append(
                    "velocity_limit_exceeded"
                )

            if (
                abs(telemetry.effort)
                > limits.maximum_effort
            ):
                violations.append(
                    "effort_limit_exceeded"
                )

            if (
                telemetry.temperature_c
                > limits.maximum_temperature_c
            ):
                violations.append(
                    "overtemperature"
                )

            if (
                abs(telemetry.current_a)
                > limits.maximum_current_a
            ):
                violations.append(
                    "overcurrent"
                )

            for violation in violations:
                self._latch_fault(
                    record,
                    violation,
                )

            if not record.fault_latched:
                record.health = (
                    ActuatorHealth.HEALTHY
                    if telemetry.enabled
                    else ActuatorHealth.DEGRADED
                )

            self._audit(
                record,
                ActuatorEventType.TELEMETRY_RECEIVED,
                "Actuator telemetry received",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # REFRESH HEALTH
    # --------------------------------------------------------

    def _refresh(
        self,
        record: ActuatorRecord,
    ) -> None:
        last = record.last_received_monotonic

        if last is None:
            return

        age = time.monotonic() - last

        if (
            age
            > record.definition.telemetry_timeout_seconds
            and not record.fault_latched
        ):
            self._latch_fault(
                record,
                "telemetry_timeout",
            )

            self._audit(
                record,
                ActuatorEventType.TELEMETRY_STALE,
                "Actuator telemetry stale",
            )

    # --------------------------------------------------------
    # VALIDATE COMMAND
    # --------------------------------------------------------

    def validate_command(
        self,
        command: ActuatorCommand,
    ) -> ActuatorCommandResult:
        with self._lock:
            record = self._get(
                command.tenant_id,
                command.robot_id,
                command.actuator_id,
            )

            self._refresh(record)

            def reject(
                reason: str,
            ) -> ActuatorCommandResult:
                self._audit(
                    record,
                    ActuatorEventType.COMMAND_REJECTED,
                    reason,
                )

                return ActuatorCommandResult(
                    command_id=command.command_id,
                    actuator_id=command.actuator_id,
                    status=ActuatorCommandStatus.REJECTED,
                    reason=reason,
                )

            key = (
                command.tenant_id,
                command.robot_id,
                command.actuator_id,
                command.command_id,
            )

            if key in self._seen_commands:
                return reject(
                    "duplicate_command"
                )

            now = utc_now()

            if command.timestamp > now:
                return reject(
                    "future_command"
                )

            age = (
                now - command.timestamp
            ).total_seconds()

            if (
                command.ttl_seconds
                > self.config.maximum_command_ttl_seconds
                or age >= command.ttl_seconds
            ):
                return reject(
                    "command_expired_or_ttl_exceeded"
                )

            if record.fault_latched:
                return reject(
                    "actuator_fault_latched"
                )

            if record.health != ActuatorHealth.HEALTHY:
                return reject(
                    "actuator_not_healthy"
                )

            if record.last_telemetry is None:
                return reject(
                    "telemetry_unavailable"
                )

            if command.mode == ActuatorMode.DISABLED:
                return reject(
                    "disable_mode_requires_separate_safety_path"
                )

            limits = record.definition.limits
            value = command.target_value

            if command.mode == ActuatorMode.POSITION:
                if not (
                    limits.minimum_position
                    <= value
                    <= limits.maximum_position
                ):
                    return reject(
                        "position_target_out_of_bounds"
                    )

            elif command.mode == ActuatorMode.VELOCITY:
                if abs(value) > limits.maximum_velocity:
                    return reject(
                        "velocity_target_out_of_bounds"
                    )

            elif command.mode == ActuatorMode.EFFORT:
                if abs(value) > limits.maximum_effort:
                    return reject(
                        "effort_target_out_of_bounds"
                    )

            elif command.mode == ActuatorMode.HOLD:
                if value != 0.0:
                    return reject(
                        "hold_target_must_be_zero"
                    )

            # Fail closed without a trusted authorizer.
            if self.command_authorizer is None:
                return reject(
                    "command_authorizer_not_configured"
                )

            try:
                authorized = self.command_authorizer(
                    command
                )
            except Exception:
                authorized = False

            if authorized is not True:
                return reject(
                    "authorization_denied"
                )

            self._seen_commands.add(key)

            record.command_count += 1

            record.mode = command.mode

            self._audit(
                record,
                ActuatorEventType.COMMAND_ACCEPTED,
                "Simulation command validated",
            )

            return ActuatorCommandResult(
                command_id=command.command_id,
                actuator_id=command.actuator_id,
                status=(
                    ActuatorCommandStatus.ACCEPTED_SIMULATION
                ),
                reason="simulation_validation_passed",
            )

    # --------------------------------------------------------
    # EMERGENCY STOP
    # --------------------------------------------------------

    def emergency_stop(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        reason: str = "emergency_stop",
    ) -> dict[str, Any]:
        with self._lock:
            affected = []

            for record in self.actuators.values():
                definition = record.definition

                if (
                    definition.tenant_id == tenant_id
                    and definition.robot_id == robot_id
                ):
                    self._latch_fault(
                        record,
                        reason,
                    )

                    self._audit(
                        record,
                        ActuatorEventType.EMERGENCY_STOP,
                        reason,
                    )

                    affected.append(
                        definition.actuator_id
                    )

            return {
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "simulation_stop_latched": True,
                "affected_actuators": affected,
                "timestamp": utc_now().isoformat(),
            }

    # --------------------------------------------------------
    # RECOVERY
    # --------------------------------------------------------

    def recover_actuator(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        actuator_id: str,
        authorized: bool,
        independently_verified: bool,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                actuator_id,
            )

            if (
                authorized is not True
                or independently_verified is not True
            ):
                raise ActuatorPermissionDenied(
                    "Verified recovery authorization required"
                )

            if record.last_telemetry is None:
                raise ActuatorSafetyError(
                    "Cannot recover without telemetry"
                )

            telemetry = record.last_telemetry
            limits = record.definition.limits

            if (
                telemetry.temperature_c
                > limits.maximum_temperature_c
                or abs(telemetry.current_a)
                > limits.maximum_current_a
                or abs(telemetry.velocity)
                > limits.maximum_velocity
                or abs(telemetry.effort)
                > limits.maximum_effort
                or not (
                    limits.minimum_position
                    <= telemetry.position
                    <= limits.maximum_position
                )
            ):
                raise ActuatorSafetyError(
                    "Actuator telemetry remains unsafe"
                )

            if (
                record.last_received_monotonic is None
                or (
                    time.monotonic()
                    - record.last_received_monotonic
                )
                > record.definition.telemetry_timeout_seconds
            ):
                raise ActuatorSafetyError(
                    "Fresh telemetry required"
                )

            record.fault_latched = False
            record.fault_reasons.clear()

            record.health = (
                ActuatorHealth.UNKNOWN
            )

            record.mode = (
                ActuatorMode.DISABLED
            )

            self._audit(
                record,
                ActuatorEventType.RECOVERY_COMPLETED,
                "Logical recovery completed",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # GET ACTUATOR
    # --------------------------------------------------------

    def get_actuator(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        actuator_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                actuator_id,
            )

            self._refresh(record)

            return record.to_dict()

    # --------------------------------------------------------
    # ROBOT HEALTH
    # --------------------------------------------------------

    def evaluate_robot(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            critical = []

            for record in self.actuators.values():
                definition = record.definition

                if (
                    definition.tenant_id == tenant_id
                    and definition.robot_id == robot_id
                    and definition.critical
                ):
                    self._refresh(record)
                    critical.append(record)

            unhealthy = [
                record.definition.actuator_id
                for record in critical
                if record.health != ActuatorHealth.HEALTHY
            ]

            ready = (
                bool(critical)
                and not unhealthy
            )

            return {
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "ready_for_simulation": ready,
                "critical_actuators": len(critical),
                "unhealthy_actuators": unhealthy,
                "timestamp": utc_now().isoformat(),
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
                for record in self.actuators.values()
                if (
                    tenant_id is None
                    or record.definition.tenant_id
                    == tenant_id
                )
            ]

            for record in records:
                self._refresh(record)

            return {
                "total_actuators": len(records),
                "healthy": sum(
                    record.health == ActuatorHealth.HEALTHY
                    for record in records
                ),
                "degraded": sum(
                    record.health == ActuatorHealth.DEGRADED
                    for record in records
                ),
                "faulted": sum(
                    record.health == ActuatorHealth.FAULTED
                    for record in records
                ),
                "unknown": sum(
                    record.health == ActuatorHealth.UNKNOWN
                    for record in records
                ),
                "total_commands": sum(
                    record.command_count
                    for record in records
                ),
                "total_telemetry_samples": sum(
                    record.telemetry_count
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
                "simulation_only": True,
                "statistics": self.statistics(
                    tenant_id=tenant_id
                ),
                "actuators": [
                    record.to_dict()
                    for record in self.actuators.values()
                    if record.definition.tenant_id
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
    manager = RoboticsActuatorManager(
        command_authorizer=lambda command: (
            command.actor_id == "simulation_controller"
        )
    )

    manager.register_actuator(
        ActuatorDefinition(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            actuator_id="joint_1",
            actuator_type=ActuatorType.SERVO,
            limits=ActuatorLimits(
                minimum_position=-3.14,
                maximum_position=3.14,
                maximum_velocity=2.0,
                maximum_effort=20.0,
                maximum_temperature_c=75.0,
                maximum_current_a=10.0,
            ),
        )
    )

    manager.ingest_telemetry(
        ActuatorTelemetry(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            actuator_id="joint_1",
            sequence_number=1,
            position=0.0,
            velocity=0.0,
            effort=0.0,
            temperature_c=35.0,
            current_a=1.2,
        )
    )

    result = manager.validate_command(
        ActuatorCommand(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            actuator_id="joint_1",
            mode=ActuatorMode.POSITION,
            target_value=1.2,
            actor_id="simulation_controller",
        )
    )

    print(
        "Command validation:",
        json.dumps(
            result.to_dict(),
            indent=2,
        ),
    )

    print(
        "Robot actuator health:",
        json.dumps(
            manager.evaluate_robot(
                tenant_id="tenant_alpha",
                robot_id="robot_001",
            ),
            indent=2,
        ),
    )

    manager.export_json(
        "data/robotics/actuator_manager_report.json",
        tenant_id="tenant_alpha",
    )


if __name__ == "__main__":
    main()

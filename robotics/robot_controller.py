"""Robot Controller module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
backend/app/robotics/control/robot_controller.py

Python 3.11+
Standard library only.

CENTRAL ROBOT CONTROLLER — SIMULATION ONLY

This module coordinates:
- Robot registration
- Telemetry and state
- Command validation
- Command dispatch
- Simulation execution
- Emergency-stop latching
- Capability and calibration checks
- Audit logging

IMPORTANT:
    This controller NEVER communicates with physical
    robot hardware.

    No ROS 2, motor driver, CAN bus, PWM, serial,
    EtherCAT or actuator command transport is used.

    Its emergency stop is a software simulation latch,
    not a safety-rated physical emergency stop.

    For real deployment, integrate independently
    certified safety systems and authenticated
    hardware interfaces.
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
from typing import Any, Callable, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================

class RobotControllerError(Exception):
    pass


class RobotValidationError(RobotControllerError):
    pass


class RobotNotFound(RobotControllerError):
    pass


class RobotConflict(RobotControllerError):
    pass


class RobotSafetyError(RobotControllerError):
    pass


class RobotPermissionDenied(RobotControllerError):
    pass


# ============================================================
# ENUMS
# ============================================================

class RobotMode(str, Enum):
    REGISTERED = "registered"
    IDLE = "idle"
    BUSY = "busy"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    FAULTED = "faulted"
    EMERGENCY_STOP = "emergency_stop"


class RobotCommandType(str, Enum):
    NAVIGATE = "navigate"
    MOVE_BASE = "move_base"
    MOVE_ARM = "move_arm"
    GRIPPER = "gripper"
    PICK = "pick"
    PLACE = "place"
    INSPECT = "inspect"
    SCAN = "scan"
    DOCK = "dock"
    WAIT = "wait"
    DIAGNOSTICS = "diagnostics"


class RobotCommandStatus(str, Enum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class RobotEventType(str, Enum):
    REGISTERED = "registered"
    TELEMETRY_UPDATED = "telemetry_updated"
    COMMAND_ACCEPTED = "command_accepted"
    COMMAND_STARTED = "command_started"
    COMMAND_SUCCEEDED = "command_succeeded"
    COMMAND_FAILED = "command_failed"
    COMMAND_REJECTED = "command_rejected"
    COMMAND_CANCELLED = "command_cancelled"
    TELEMETRY_STALE = "telemetry_stale"
    FAULT_LATCHED = "fault_latched"
    EMERGENCY_STOP = "emergency_stop"
    EMERGENCY_RESET = "emergency_reset"


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_id(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not all(
            char.isascii()
            and (
                char.isalnum()
                or char in "_.-"
            )
            for char in value
        )
    ):
        raise RobotValidationError(
            f"Invalid {label}"
        )

    return value


def finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise RobotValidationError(
            f"{label} must be finite"
        )

    return float(value)


def percentage(value: Any, label: str) -> float:
    result = finite(value, label)

    if not 0 <= result <= 100:
        raise RobotValidationError(
            f"{label} must be between 0 and 100"
        )

    return result


def positive(value: Any, label: str) -> float:
    result = finite(value, label)

    if result <= 0:
        raise RobotValidationError(
            f"{label} must be positive"
        )

    return result


def validate_timestamp(
    timestamp: datetime,
) -> datetime:
    if (
        not isinstance(timestamp, datetime)
        or timestamp.tzinfo is None
        or timestamp.utcoffset() is None
    ):
        raise RobotValidationError(
            "Timezone-aware timestamp required"
        )

    return timestamp.astimezone(timezone.utc)


def normalize_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise RobotValidationError(
            "Command payload must be a mapping"
        )

    try:
        result = json.loads(
            json.dumps(
                dict(payload),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise RobotValidationError(
            "Payload must be JSON serializable"
        ) from exc

    if not isinstance(result, dict):
        raise RobotValidationError(
            "Payload must be a JSON object"
        )

    return result


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RobotControllerConfig:
    maximum_robots: int = 10000
    maximum_commands: int = 100000
    maximum_audit_events: int = 20000

    telemetry_timeout_seconds: float = 5.0

    minimum_battery_percentage: float = 20.0
    critical_battery_percentage: float = 10.0

    maximum_temperature_c: float = 80.0

    maximum_command_ttl_seconds: float = 300.0
    maximum_payload_bytes: int = 65536

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_robots",
            "maximum_commands",
            "maximum_audit_events",
            "maximum_payload_bytes",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise RobotValidationError(
                    f"{name} must be positive"
                )

        for name in (
            "telemetry_timeout_seconds",
            "maximum_temperature_c",
            "maximum_command_ttl_seconds",
        ):
            positive(
                getattr(self, name),
                name,
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
            raise RobotValidationError(
                "Critical battery threshold invalid"
            )

        if not self.simulation_only:
            raise RobotValidationError(
                "Physical robot control unsupported"
            )


# ============================================================
# ROBOT DEFINITION
# ============================================================

@dataclass(frozen=True)
class RobotDefinition:
    tenant_id: str
    robot_id: str
    name: str

    robot_type: str

    capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_id(
            self.tenant_id,
            "tenant_id",
        )

        validate_id(
            self.robot_id,
            "robot_id",
        )

        validate_id(
            self.robot_type,
            "robot_type",
        )

        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or len(self.name) > 256
        ):
            raise RobotValidationError(
                "Invalid robot name"
            )

        if not isinstance(
            self.capabilities,
            tuple,
        ):
            raise RobotValidationError(
                "Capabilities must be a tuple"
            )

        for capability in self.capabilities:
            validate_id(
                capability,
                "capability",
            )

        if (
            len(self.capabilities)
            != len(set(self.capabilities))
        ):
            raise RobotValidationError(
                "Duplicate capabilities"
            )


# ============================================================
# TELEMETRY
# ============================================================

@dataclass(frozen=True)
class RobotTelemetry:
    tenant_id: str
    robot_id: str

    sequence_number: int

    battery_percentage: float
    temperature_c: float

    operational: bool = True
    fault_detected: bool = False

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_id(
            self.tenant_id,
            "tenant_id",
        )

        validate_id(
            self.robot_id,
            "robot_id",
        )

        if (
            isinstance(self.sequence_number, bool)
            or not isinstance(self.sequence_number, int)
            or self.sequence_number < 0
        ):
            raise RobotValidationError(
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

        if not isinstance(self.operational, bool):
            raise RobotValidationError(
                "operational must be boolean"
            )

        if not isinstance(self.fault_detected, bool):
            raise RobotValidationError(
                "fault_detected must be boolean"
            )

        validate_timestamp(
            self.timestamp
        )


# ============================================================
# COMMAND REQUEST
# ============================================================

@dataclass(frozen=True)
class RobotControlCommand:
    tenant_id: str
    robot_id: str
    actor_id: str

    command_type: RobotCommandType

    required_capability: str

    payload: Mapping[str, Any]

    command_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    ttl_seconds: float = 30.0

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for name in (
            "tenant_id",
            "robot_id",
            "actor_id",
            "required_capability",
            "command_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        if not isinstance(
            self.command_type,
            RobotCommandType,
        ):
            raise RobotValidationError(
                "Invalid command type"
            )

        positive(
            self.ttl_seconds,
            "ttl_seconds",
        )

        validate_timestamp(
            self.timestamp
        )

        object.__setattr__(
            self,
            "payload",
            normalize_payload(self.payload),
        )


# ============================================================
# ROBOT RECORD
# ============================================================

@dataclass
class RobotControllerRecord:
    definition: RobotDefinition

    mode: RobotMode = RobotMode.REGISTERED

    telemetry: RobotTelemetry | None = None

    telemetry_received_monotonic: float | None = None

    last_telemetry_sequence: int = -1

    fault_latched: bool = False
    emergency_latched: bool = False

    fault_reasons: list[str] = field(
        default_factory=list
    )

    active_command_id: str | None = None

    completed_commands: int = 0
    failed_commands: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.definition.tenant_id,
            "robot_id": self.definition.robot_id,
            "name": self.definition.name,
            "robot_type": self.definition.robot_type,
            "capabilities": list(
                self.definition.capabilities
            ),
            "mode": self.mode.value,
            "fault_latched": self.fault_latched,
            "emergency_latched": (
                self.emergency_latched
            ),
            "fault_reasons": list(
                self.fault_reasons
            ),
            "active_command_id": (
                self.active_command_id
            ),
            "completed_commands": (
                self.completed_commands
            ),
            "failed_commands": (
                self.failed_commands
            ),
            "telemetry": (
                {
                    "battery_percentage": (
                        self.telemetry.battery_percentage
                    ),
                    "temperature_c": (
                        self.telemetry.temperature_c
                    ),
                    "operational": (
                        self.telemetry.operational
                    ),
                    "timestamp": (
                        self.telemetry.timestamp.isoformat()
                    ),
                }
                if self.telemetry
                else None
            ),
        }


# ============================================================
# COMMAND RECORD
# ============================================================

@dataclass
class RobotCommandRecord:
    command: RobotControlCommand

    status: RobotCommandStatus = (
        RobotCommandStatus.ACCEPTED
    )

    result: dict[str, Any] | None = None
    error: str | None = None

    created_at: datetime = field(
        default_factory=utc_now
    )

    finished_at: datetime | None = None

    expires_monotonic: float = 0.0

    cancellation_event: threading.Event = field(
        default_factory=threading.Event
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command.command_id,
            "tenant_id": self.command.tenant_id,
            "robot_id": self.command.robot_id,
            "actor_id": self.command.actor_id,
            "command_type": (
                self.command.command_type.value
            ),
            "required_capability": (
                self.command.required_capability
            ),
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
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
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class RobotControllerAuditEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    robot_id: str
    resource_id: str

    event_type: RobotEventType
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "resource_id": self.resource_id,
            "event_type": self.event_type.value,
            "details": self.details,
        }


# ============================================================
# CALLBACK TYPES
# ============================================================

RobotCommandCheck = Callable[
    [RobotControlCommand],
    bool,
]

RobotSimulationHandler = Callable[
    [
        RobotControlCommand,
        threading.Event,
    ],
    Mapping[str, Any],
]


# ============================================================
# CENTRAL ROBOT CONTROLLER
# ============================================================

class RoboticsRobotController:

    TERMINAL_STATUSES = {
        RobotCommandStatus.SUCCEEDED,
        RobotCommandStatus.FAILED,
        RobotCommandStatus.REJECTED,
        RobotCommandStatus.CANCELLED,
        RobotCommandStatus.EXPIRED,
    }

    def __init__(
        self,
        config: RobotControllerConfig | None = None,
        *,
        authorization_check: RobotCommandCheck | None = None,
        safety_check: RobotCommandCheck | None = None,
        capability_check: RobotCommandCheck | None = None,
        calibration_check: RobotCommandCheck | None = None,
    ) -> None:
        self.config = (
            config or RobotControllerConfig()
        )

        self.authorization_check = authorization_check
        self.safety_check = safety_check
        self.capability_check = capability_check
        self.calibration_check = calibration_check

        self._robots: dict[
            tuple[str, str],
            RobotControllerRecord
        ] = {}

        self._commands: dict[
            tuple[str, str, str],
            RobotCommandRecord
        ] = {}

        self._handlers: dict[
            RobotCommandType,
            RobotSimulationHandler
        ] = {}

        self._audit_events: deque[
            RobotControllerAuditEvent
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # KEYS
    # --------------------------------------------------------

    def _robot_key(
        self,
        tenant_id: str,
        robot_id: str,
    ) -> tuple[str, str]:
        return (
            validate_id(tenant_id, "tenant_id"),
            validate_id(robot_id, "robot_id"),
        )

    def _command_key(
        self,
        tenant_id: str,
        robot_id: str,
        command_id: str,
    ) -> tuple[str, str, str]:
        return (
            tenant_id,
            robot_id,
            validate_id(command_id, "command_id"),
        )

    def _robot(
        self,
        tenant_id: str,
        robot_id: str,
    ) -> RobotControllerRecord:
        record = self._robots.get(
            self._robot_key(
                tenant_id,
                robot_id,
            )
        )

        if record is None:
            raise RobotNotFound(
                "Robot not registered"
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
        resource_id: str,
        event_type: RobotEventType,
        details: str,
    ) -> None:
        self._audit_events.append(
            RobotControllerAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                tenant_id=tenant_id,
                robot_id=robot_id,
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
        definition: RobotDefinition,
    ) -> dict[str, Any]:
        with self._lock:
            key = self._robot_key(
                definition.tenant_id,
                definition.robot_id,
            )

            if key in self._robots:
                raise RobotConflict(
                    "Robot already registered"
                )

            if (
                len(self._robots)
                >= self.config.maximum_robots
            ):
                raise RobotControllerError(
                    "Maximum robot capacity reached"
                )

            record = RobotControllerRecord(
                definition=definition
            )

            self._robots[key] = record

            self._audit(
                tenant_id=definition.tenant_id,
                robot_id=definition.robot_id,
                resource_id=definition.robot_id,
                event_type=RobotEventType.REGISTERED,
                details="Robot registered",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # REGISTER HANDLER
    # --------------------------------------------------------

    def register_handler(
        self,
        command_type: RobotCommandType,
        handler: RobotSimulationHandler,
    ) -> None:
        if not isinstance(
            command_type,
            RobotCommandType,
        ):
            raise RobotValidationError(
                "Invalid command type"
            )

        if not callable(handler):
            raise RobotValidationError(
                "Handler must be callable"
            )

        with self._lock:
            if command_type in self._handlers:
                raise RobotConflict(
                    "Handler already registered"
                )

            self._handlers[command_type] = handler

    # --------------------------------------------------------
    # FAULT LATCH
    # --------------------------------------------------------

    def _latch_fault(
        self,
        record: RobotControllerRecord,
        reason: str,
    ) -> None:
        record.fault_latched = True

        if reason not in record.fault_reasons:
            record.fault_reasons.append(
                reason
            )

        if not record.emergency_latched:
            record.mode = RobotMode.FAULTED

        if record.active_command_id:
            key = self._command_key(
                record.definition.tenant_id,
                record.definition.robot_id,
                record.active_command_id,
            )

            command = self._commands.get(key)

            if command:
                command.cancellation_event.set()

        self._audit(
            tenant_id=record.definition.tenant_id,
            robot_id=record.definition.robot_id,
            resource_id=record.definition.robot_id,
            event_type=RobotEventType.FAULT_LATCHED,
            details=reason,
        )

    # --------------------------------------------------------
    # INGEST TELEMETRY
    # --------------------------------------------------------

    def ingest_telemetry(
        self,
        telemetry: RobotTelemetry,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._robot(
                telemetry.tenant_id,
                telemetry.robot_id,
            )

            if (
                telemetry.sequence_number
                <= record.last_telemetry_sequence
            ):
                raise RobotValidationError(
                    "Out-of-order telemetry"
                )

            age = (
                utc_now()
                - validate_timestamp(
                    telemetry.timestamp
                )
            ).total_seconds()

            if age < 0:
                raise RobotValidationError(
                    "Future telemetry rejected"
                )

            if (
                age
                > self.config.telemetry_timeout_seconds
            ):
                raise RobotValidationError(
                    "Stale telemetry rejected"
                )

            record.telemetry = telemetry

            record.telemetry_received_monotonic = (
                time.monotonic()
            )

            record.last_telemetry_sequence = (
                telemetry.sequence_number
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

            self._refresh(record)

            if (
                not record.fault_latched
                and not record.emergency_latched
                and record.active_command_id is None
            ):
                if (
                    telemetry.battery_percentage
                    < self.config.minimum_battery_percentage
                ):
                    record.mode = RobotMode.DEGRADED
                else:
                    record.mode = RobotMode.IDLE

            self._audit(
                tenant_id=telemetry.tenant_id,
                robot_id=telemetry.robot_id,
                resource_id=telemetry.robot_id,
                event_type=RobotEventType.TELEMETRY_UPDATED,
                details="Telemetry updated",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # REFRESH ROBOT HEALTH
    # --------------------------------------------------------

    def _refresh(
        self,
        record: RobotControllerRecord,
    ) -> None:
        last = record.telemetry_received_monotonic

        if record.emergency_latched:
            record.mode = RobotMode.EMERGENCY_STOP
            return

        if record.fault_latched:
            record.mode = RobotMode.FAULTED
            return

        if last is None:
            record.mode = RobotMode.REGISTERED
            return

        if (
            time.monotonic() - last
            > self.config.telemetry_timeout_seconds
        ):
            if record.mode != RobotMode.OFFLINE:
                self._audit(
                    tenant_id=record.definition.tenant_id,
                    robot_id=record.definition.robot_id,
                    resource_id=record.definition.robot_id,
                    event_type=RobotEventType.TELEMETRY_STALE,
                    details="Telemetry timeout",
                )

            record.mode = RobotMode.OFFLINE

            if record.active_command_id:
                key = self._command_key(
                    record.definition.tenant_id,
                    record.definition.robot_id,
                    record.active_command_id,
                )

                command = self._commands.get(key)

                if command:
                    command.cancellation_event.set()

            return

        if record.active_command_id:
            record.mode = RobotMode.BUSY
            return

        telemetry = record.telemetry

        if (
            telemetry
            and telemetry.battery_percentage
            < self.config.minimum_battery_percentage
        ):
            record.mode = RobotMode.DEGRADED

        else:
            record.mode = RobotMode.IDLE

    # --------------------------------------------------------
    # CHECK EXTERNAL CALLBACK
    # --------------------------------------------------------

    @staticmethod
    def _check(
        callback: RobotCommandCheck | None,
        command: RobotControlCommand,
    ) -> bool:
        if callback is None:
            return False

        try:
            return callback(command) is True
        except Exception:
            return False

    # --------------------------------------------------------
    # SUBMIT COMMAND
    # --------------------------------------------------------

    def submit_command(
        self,
        command: RobotControlCommand,
    ) -> dict[str, Any]:
        if not isinstance(
            command,
            RobotControlCommand,
        ):
            raise RobotValidationError(
                "Invalid command"
            )

        if (
            command.ttl_seconds
            > self.config.maximum_command_ttl_seconds
        ):
            raise RobotValidationError(
                "Command TTL exceeds limit"
            )

        payload_bytes = json.dumps(
            dict(command.payload),
            allow_nan=False,
        ).encode("utf-8")

        if (
            len(payload_bytes)
            > self.config.maximum_payload_bytes
        ):
            raise RobotValidationError(
                "Payload exceeds size limit"
            )

        with self._lock:
            robot = self._robot(
                command.tenant_id,
                command.robot_id,
            )

            key = self._command_key(
                command.tenant_id,
                command.robot_id,
                command.command_id,
            )

            existing = self._commands.get(key)

            if existing is not None:
                if existing.command != command:
                    raise RobotConflict(
                        "Command ID conflict"
                    )

                return existing.to_dict()

            if (
                len(self._commands)
                >= self.config.maximum_commands
            ):
                raise RobotControllerError(
                    "Maximum command capacity reached"
                )

            record = RobotCommandRecord(
                command=command
            )

            self._commands[key] = record

            age = (
                utc_now() - command.timestamp
            ).total_seconds()

            if age < 0 or age >= command.ttl_seconds:
                record.status = RobotCommandStatus.EXPIRED
                record.error = "invalid_or_expired_timestamp"
                record.finished_at = utc_now()
                return record.to_dict()

            record.expires_monotonic = (
                time.monotonic()
                + command.ttl_seconds
                - age
            )

            self._refresh(robot)

            rejection_reason = None

            if robot.mode != RobotMode.IDLE:
                rejection_reason = (
                    f"robot_not_idle:{robot.mode.value}"
                )

            elif (
                command.required_capability
                not in robot.definition.capabilities
            ):
                rejection_reason = (
                    "required_capability_not_declared"
                )

            elif (
                command.command_type
                not in self._handlers
            ):
                rejection_reason = (
                    "simulation_handler_missing"
                )

            elif not all((
                self._check(
                    self.authorization_check,
                    command,
                ),
                self._check(
                    self.safety_check,
                    command,
                ),
                self._check(
                    self.capability_check,
                    command,
                ),
                self._check(
                    self.calibration_check,
                    command,
                ),
            )):
                rejection_reason = (
                    "external_precondition_denied"
                )

            if rejection_reason:
                record.status = (
                    RobotCommandStatus.REJECTED
                )

                record.error = rejection_reason
                record.finished_at = utc_now()

                self._audit(
                    tenant_id=command.tenant_id,
                    robot_id=command.robot_id,
                    resource_id=command.command_id,
                    event_type=RobotEventType.COMMAND_REJECTED,
                    details=rejection_reason,
                )

                return record.to_dict()

            self._audit(
                tenant_id=command.tenant_id,
                robot_id=command.robot_id,
                resource_id=command.command_id,
                event_type=RobotEventType.COMMAND_ACCEPTED,
                details="Simulation command accepted",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # EXECUTE COMMAND
    # --------------------------------------------------------

    def execute_command(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        command_id: str,
    ) -> dict[str, Any]:
        key = self._command_key(
            tenant_id,
            robot_id,
            command_id,
        )

        with self._lock:
            record = self._commands.get(key)

            if record is None:
                raise RobotControllerError(
                    "Command not found"
                )

            if (
                record.status
                != RobotCommandStatus.ACCEPTED
            ):
                raise RobotConflict(
                    "Command is not executable"
                )

            robot = self._robot(
                tenant_id,
                robot_id,
            )

            self._refresh(robot)

            if robot.mode != RobotMode.IDLE:
                raise RobotSafetyError(
                    "Robot not ready"
                )

            if (
                time.monotonic()
                >= record.expires_monotonic
            ):
                record.status = (
                    RobotCommandStatus.EXPIRED
                )

                record.finished_at = utc_now()

                return record.to_dict()

            command = record.command

            if not all((
                self._check(
                    self.authorization_check,
                    command,
                ),
                self._check(
                    self.safety_check,
                    command,
                ),
                self._check(
                    self.capability_check,
                    command,
                ),
                self._check(
                    self.calibration_check,
                    command,
                ),
            )):
                record.status = (
                    RobotCommandStatus.REJECTED
                )

                record.error = (
                    "execution_precondition_denied"
                )

                record.finished_at = utc_now()

                return record.to_dict()

            handler = self._handlers.get(
                command.command_type
            )

            if handler is None:
                raise RobotControllerError(
                    "Simulation handler missing"
                )

            record.status = (
                RobotCommandStatus.RUNNING
            )

            robot.active_command_id = command_id
            robot.mode = RobotMode.BUSY

            self._audit(
                tenant_id=tenant_id,
                robot_id=robot_id,
                resource_id=command_id,
                event_type=RobotEventType.COMMAND_STARTED,
                details="Simulation execution started",
            )

        # Run simulation handler outside lock.
        try:
            if record.cancellation_event.is_set():
                raise RobotControllerError(
                    "Command cancelled"
                )

            result = handler(
                command,
                record.cancellation_event,
            )

            normalized = normalize_payload(
                result
            )

            error = None

        except Exception as exc:
            normalized = None
            error = (
                f"{type(exc).__name__}: "
                "simulation handler failed"
            )

        with self._lock:
            robot.active_command_id = None

            self._refresh(robot)

            if record.cancellation_event.is_set():
                record.status = (
                    RobotCommandStatus.CANCELLED
                )

                record.error = (
                    "simulation_execution_cancelled"
                )

                event = (
                    RobotEventType.COMMAND_CANCELLED
                )

            elif (
                time.monotonic()
                >= record.expires_monotonic
            ):
                record.status = (
                    RobotCommandStatus.EXPIRED
                )

                record.error = (
                    "execution_deadline_exceeded"
                )

                event = (
                    RobotEventType.COMMAND_FAILED
                )

            elif error is None:
                record.status = (
                    RobotCommandStatus.SUCCEEDED
                )

                record.result = normalized

                robot.completed_commands += 1

                event = (
                    RobotEventType.COMMAND_SUCCEEDED
                )

            else:
                record.status = (
                    RobotCommandStatus.FAILED
                )

                record.error = error

                robot.failed_commands += 1

                event = (
                    RobotEventType.COMMAND_FAILED
                )

            record.finished_at = utc_now()

            self._audit(
                tenant_id=tenant_id,
                robot_id=robot_id,
                resource_id=command_id,
                event_type=event,
                details=record.status.value,
            )

            return record.to_dict()

    # --------------------------------------------------------
    # CANCEL COMMAND
    # --------------------------------------------------------

    def cancel_command(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        command_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            key = self._command_key(
                tenant_id,
                robot_id,
                command_id,
            )

            record = self._commands.get(key)

            if record is None:
                raise RobotControllerError(
                    "Command not found"
                )

            if record.status in self.TERMINAL_STATUSES:
                return record.to_dict()

            record.cancellation_event.set()

            if (
                record.status
                != RobotCommandStatus.RUNNING
            ):
                record.status = (
                    RobotCommandStatus.CANCELLED
                )

                record.finished_at = utc_now()

            self._audit(
                tenant_id=tenant_id,
                robot_id=robot_id,
                resource_id=command_id,
                event_type=RobotEventType.COMMAND_CANCELLED,
                details="Cancellation requested",
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
            robot = self._robot(
                tenant_id,
                robot_id,
            )

            robot.emergency_latched = True
            robot.mode = RobotMode.EMERGENCY_STOP

            for record in self._commands.values():
                command = record.command

                if (
                    command.tenant_id != tenant_id
                    or command.robot_id != robot_id
                    or record.status
                    in self.TERMINAL_STATUSES
                ):
                    continue

                record.cancellation_event.set()

                if (
                    record.status
                    != RobotCommandStatus.RUNNING
                ):
                    record.status = (
                        RobotCommandStatus.CANCELLED
                    )

                    record.finished_at = utc_now()

            self._audit(
                tenant_id=tenant_id,
                robot_id=robot_id,
                resource_id=robot_id,
                event_type=RobotEventType.EMERGENCY_STOP,
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
        robot_id: str,
        authorized: bool,
        independently_verified: bool,
    ) -> dict[str, Any]:
        if (
            authorized is not True
            or independently_verified is not True
        ):
            raise RobotPermissionDenied(
                "Verified reset approval required"
            )

        with self._lock:
            robot = self._robot(
                tenant_id,
                robot_id,
            )

            if robot.active_command_id:
                raise RobotSafetyError(
                    "Cannot reset during execution"
                )

            robot.emergency_latched = False

            self._refresh(robot)

            self._audit(
                tenant_id=tenant_id,
                robot_id=robot_id,
                resource_id=robot_id,
                event_type=RobotEventType.EMERGENCY_RESET,
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
        robot_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            robot = self._robot(
                tenant_id,
                robot_id,
            )

            self._refresh(robot)

            return robot.to_dict()

    # --------------------------------------------------------
    # GET COMMAND
    # --------------------------------------------------------

    def get_command(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        command_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            key = self._command_key(
                tenant_id,
                robot_id,
                command_id,
            )

            record = self._commands.get(key)

            if record is None:
                raise RobotControllerError(
                    "Command not found"
                )

            return record.to_dict()

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        tenant_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            robots = [
                robot
                for robot in self._robots.values()
                if robot.definition.tenant_id == tenant_id
            ]

            commands = [
                record
                for record in self._commands.values()
                if record.command.tenant_id == tenant_id
            ]

            for robot in robots:
                self._refresh(robot)

            return {
                "total_robots": len(robots),
                "idle_robots": sum(
                    robot.mode == RobotMode.IDLE
                    for robot in robots
                ),
                "busy_robots": sum(
                    robot.mode == RobotMode.BUSY
                    for robot in robots
                ),
                "faulted_robots": sum(
                    robot.mode == RobotMode.FAULTED
                    for robot in robots
                ),
                "offline_robots": sum(
                    robot.mode == RobotMode.OFFLINE
                    for robot in robots
                ),
                "total_commands": len(commands),
                "successful_commands": sum(
                    command.status
                    == RobotCommandStatus.SUCCEEDED
                    for command in commands
                ),
                "failed_commands": sum(
                    command.status
                    == RobotCommandStatus.FAILED
                    for command in commands
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
                "robots": [
                    self.get_robot(
                        tenant_id=tenant_id,
                        robot_id=robot.definition.robot_id,
                    )
                    for robot in self._robots.values()
                    if robot.definition.tenant_id == tenant_id
                ],
                "commands": [
                    record.to_dict()
                    for record in self._commands.values()
                    if record.command.tenant_id == tenant_id
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
# SIMULATION HANDLERS
# ============================================================

def navigation_simulation_handler(
    command: RobotControlCommand,
    cancellation_event: threading.Event,
) -> dict[str, Any]:
    if cancellation_event.is_set():
        raise RobotControllerError(
            "Navigation cancelled"
        )

    target = command.payload.get("target")

    if not isinstance(target, dict):
        raise RobotValidationError(
            "Navigation target required"
        )

    for axis in ("x", "y", "theta"):
        finite(
            target.get(axis),
            f"target.{axis}",
        )

    return {
        "robot_id": command.robot_id,
        "target": target,
        "navigation_simulated": True,
    }


def diagnostics_simulation_handler(
    command: RobotControlCommand,
    cancellation_event: threading.Event,
) -> dict[str, Any]:
    if cancellation_event.is_set():
        raise RobotControllerError(
            "Diagnostics cancelled"
        )

    return {
        "robot_id": command.robot_id,
        "diagnostics_simulated": True,
        "message": (
            "Simulation diagnostics completed"
        ),
    }


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    controller = RoboticsRobotController(
        authorization_check=lambda command: (
            command.actor_id
            == "simulation_controller"
        ),
        safety_check=lambda command: (
            command.payload.get("simulation_only")
            is True
        ),
        capability_check=lambda command: (
            command.required_capability
            in ("navigation", "diagnostics")
        ),
        calibration_check=lambda command: True,
    )

    controller.register_handler(
        RobotCommandType.NAVIGATE,
        navigation_simulation_handler,
    )

    controller.register_handler(
        RobotCommandType.DIAGNOSTICS,
        diagnostics_simulation_handler,
    )

    controller.register_robot(
        RobotDefinition(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            name="Autonomous Mobile Robot",
            robot_type="amr",
            capabilities=(
                "navigation",
                "diagnostics",
            ),
        )
    )

    controller.ingest_telemetry(
        RobotTelemetry(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            sequence_number=1,
            battery_percentage=85.0,
            temperature_c=35.0,
        )
    )

    command = RobotControlCommand(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        actor_id="simulation_controller",
        command_type=RobotCommandType.NAVIGATE,
        required_capability="navigation",
        payload={
            "simulation_only": True,
            "target": {
                "x": 5.0,
                "y": 3.0,
                "theta": 0.0,
            },
        },
    )

    print(
        "Submitted:",
        json.dumps(
            controller.submit_command(command),
            indent=2,
        ),
    )

    print(
        "\nExecuted:",
        json.dumps(
            controller.execute_command(
                tenant_id="tenant_alpha",
                robot_id="robot_001",
                command_id=command.command_id,
            ),
            indent=2,
        ),
    )

    print(
        "\nStatistics:",
        json.dumps(
            controller.statistics(
                tenant_id="tenant_alpha"
            ),
            indent=2,
        ),
    )

    report = controller.export_json(
        "data/robotics/robot_controller_report.json",
        tenant_id="tenant_alpha",
    )

    print("\nReport:", report)


if __name__ == "__main__":
    main()

"""Gripper Controller module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/manipulation/gripper_controller.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================
Module: Gripper Controller

Features:
    - Parallel-jaw gripper control interface
    - Position and force command validation
    - Open, close, grasp, release, and stop operations
    - Simulated gripper hardware adapter
    - Gripper state machine
    - Grasp detection using position and force feedback
    - Contact detection
    - Object slip monitoring
    - Force and aperture limits
    - Emergency-stop interlock
    - Command timeout handling
    - Gripper health monitoring
    - Fault detection and recovery
    - Multi-tenant robot isolation
    - Thread-safe command processing
    - Command history and audit records
    - JSON state export
    - Synthetic demonstrations
    - Unit tests

Dependencies:
    Python >= 3.11

Safety:
    The default adapter is simulation-only.
    No real hardware commands are issued.

    Hardware integration requires a separately reviewed
    adapter, independent safety-rated emergency stopping,
    mechanical limit protection, calibrated force sensors,
    and robot-specific validation.

    Software limits in this module are not safety-rated.
"""

from __future__ import annotations

import json
import math
import threading
import time
import unittest
import uuid

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable


# ============================================================
# EXCEPTIONS
# ============================================================

class GripperControllerError(Exception):
    pass


class GripperValidationError(GripperControllerError):
    pass


class GripperAuthorizationError(GripperControllerError):
    pass


class GripperNotFoundError(GripperControllerError):
    pass


class GripperConflictError(GripperControllerError):
    pass


class GripperStateError(GripperControllerError):
    pass


class GripperHardwareError(GripperControllerError):
    pass


class GripperTimeoutError(GripperControllerError):
    pass


class GripperEmergencyStopError(GripperControllerError):
    pass


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


def finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise GripperValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise GripperValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise GripperValidationError(
            f"{name} must be finite"
        )

    return result


def identifier(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 256
    ):
        raise GripperValidationError(
            f"Invalid {name}"
        )

    return value.strip()


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(
        minimum,
        min(maximum, value),
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return {
            key: json_safe(item)
            for key, item in asdict(value).items()
        }

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, deque)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


# ============================================================
# ENUMERATIONS
# ============================================================

class GripperMode(str, Enum):
    POSITION = "position"
    FORCE = "force"
    HYBRID = "hybrid"


class GripperState(str, Enum):
    IDLE = "idle"
    OPENING = "opening"
    CLOSING = "closing"
    HOLDING = "holding"
    STOPPED = "stopped"
    FAULT = "fault"
    EMERGENCY_STOP = "emergency_stop"


class GripperCommandType(str, Enum):
    OPEN = "open"
    CLOSE = "close"
    MOVE = "move"
    GRASP = "grasp"
    RELEASE = "release"
    STOP = "stop"
    EMERGENCY_STOP = "emergency_stop"
    RESET_FAULT = "reset_fault"


class CommandStatus(str, Enum):
    ACCEPTED = "accepted"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class GraspStatus(str, Enum):
    NO_OBJECT = "no_object"
    CONTACT_DETECTED = "contact_detected"
    OBJECT_HELD = "object_held"
    POSSIBLE_SLIP = "possible_slip"
    UNKNOWN = "unknown"


class GripperHealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAULT = "fault"
    EMERGENCY_STOP = "emergency_stop"


class GripperFaultCode(str, Enum):
    NONE = "none"
    OVER_FORCE = "over_force"
    POSITION_LIMIT = "position_limit"
    SENSOR_FAILURE = "sensor_failure"
    COMMUNICATION_FAILURE = "communication_failure"
    COMMAND_TIMEOUT = "command_timeout"
    MOTOR_OVERHEAT = "motor_overheat"
    UNKNOWN = "unknown"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class GripperSpecification:
    minimum_opening_m: float = 0.0
    maximum_opening_m: float = 0.12

    maximum_force_n: float = 80.0
    maximum_speed_m_s: float = 0.05

    maximum_temperature_c: float = 75.0

    contact_force_threshold_n: float = 2.0

    holding_force_threshold_n: float = 5.0

    position_tolerance_m: float = 0.001

    def __post_init__(self) -> None:
        numeric_fields = (
            "minimum_opening_m",
            "maximum_opening_m",
            "maximum_force_n",
            "maximum_speed_m_s",
            "maximum_temperature_c",
            "contact_force_threshold_n",
            "holding_force_threshold_n",
            "position_tolerance_m",
        )

        for name in numeric_fields:
            value = finite(
                getattr(self, name),
                name,
            )

            if value < 0:
                raise GripperValidationError(
                    f"{name} cannot be negative"
                )

        if (
            self.maximum_opening_m
            <= self.minimum_opening_m
        ):
            raise GripperValidationError(
                "Invalid gripper opening range"
            )

        if self.maximum_force_n <= 0:
            raise GripperValidationError(
                "Maximum force must be positive"
            )

        if self.maximum_speed_m_s <= 0:
            raise GripperValidationError(
                "Maximum speed must be positive"
            )

        if (
            self.holding_force_threshold_n
            > self.maximum_force_n
        ):
            raise GripperValidationError(
                "Holding force threshold exceeds maximum force"
            )


@dataclass(frozen=True)
class GripperControllerConfig:
    default_speed_m_s: float = 0.02

    default_grasp_force_n: float = 15.0

    command_timeout_s: float = 5.0

    slip_position_threshold_m: float = 0.003

    slip_force_drop_ratio: float = 0.30

    maximum_history: int = 1000

    require_simulation: bool = True

    def __post_init__(self) -> None:
        for name in (
            "default_speed_m_s",
            "default_grasp_force_n",
            "command_timeout_s",
            "slip_position_threshold_m",
            "slip_force_drop_ratio",
        ):
            value = finite(
                getattr(self, name),
                name,
            )

            if value < 0:
                raise GripperValidationError(
                    f"{name} cannot be negative"
                )

        if self.default_speed_m_s <= 0:
            raise GripperValidationError(
                "Default speed must be positive"
            )

        if self.command_timeout_s <= 0:
            raise GripperValidationError(
                "Command timeout must be positive"
            )

        if not 0 <= self.slip_force_drop_ratio <= 1:
            raise GripperValidationError(
                "Slip force drop ratio must be in [0,1]"
            )

        if self.maximum_history < 1:
            raise GripperValidationError(
                "maximum_history must be positive"
            )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class GripperAccessContext:
    tenant_id: str
    user_id: str

    can_read: bool = False
    can_command: bool = False
    can_manage: bool = False


@dataclass(frozen=True)
class GripperCommand:
    command_id: str

    command_type: GripperCommandType

    target_opening_m: float | None = None

    target_force_n: float | None = None

    speed_m_s: float | None = None

    timeout_s: float | None = None

    created_at: datetime = field(
        default_factory=utc_now
    )


@dataclass(frozen=True)
class GripperTelemetry:
    opening_m: float

    force_n: float

    velocity_m_s: float

    motor_temperature_c: float

    contact_detected: bool

    object_detected: bool

    fault_code: GripperFaultCode

    timestamp: datetime


@dataclass(frozen=True)
class GripperStatusReport:
    robot_id: str
    tenant_id: str

    state: GripperState

    grasp_status: GraspStatus

    health: GripperHealthStatus

    telemetry: GripperTelemetry

    emergency_stop_active: bool

    timestamp: datetime


@dataclass(frozen=True)
class GripperCommandResult:
    command_id: str

    robot_id: str
    tenant_id: str

    command_type: GripperCommandType

    status: CommandStatus

    initial_opening_m: float
    final_opening_m: float

    final_force_n: float

    duration_s: float

    message: str

    timestamp: datetime


@dataclass(frozen=True)
class GripperAuditRecord:
    audit_id: str

    robot_id: str
    tenant_id: str
    user_id: str

    action: str

    command_id: str | None

    success: bool

    timestamp: datetime


# ============================================================
# HARDWARE ADAPTER PROTOCOL
# ============================================================

@runtime_checkable
class GripperHardwareAdapter(Protocol):

    @property
    def simulation_only(self) -> bool:
        ...

    def read_telemetry(self) -> GripperTelemetry:
        ...

    def move(
        self,
        opening_m: float,
        speed_m_s: float,
        force_limit_n: float,
        timeout_s: float,
    ) -> GripperTelemetry:
        ...

    def stop(self) -> GripperTelemetry:
        ...

    def emergency_stop(self) -> GripperTelemetry:
        ...

    def reset_fault(self) -> GripperTelemetry:
        ...


# ============================================================
# SIMULATED GRIPPER ADAPTER
# ============================================================

class SimulatedGripperAdapter:
    """
    Deterministic simulated parallel-jaw gripper.

    The simulated object is represented by an optional
    fixed width. Closing below that width creates
    simulated contact and holding force.

    This adapter never communicates with physical hardware.
    """

    simulation_only = True

    def __init__(
        self,
        specification: GripperSpecification,
        *,
        initial_opening_m: float | None = None,
        object_width_m: float | None = None,
        object_contact_stiffness_n_m: float = 5000.0,
    ) -> None:
        self.specification = specification

        self._lock = threading.RLock()

        self._opening_m = (
            specification.maximum_opening_m
            if initial_opening_m is None
            else finite(
                initial_opening_m,
                "initial_opening_m",
            )
        )

        if not (
            specification.minimum_opening_m
            <= self._opening_m
            <= specification.maximum_opening_m
        ):
            raise GripperValidationError(
                "Initial opening outside limits"
            )

        if object_width_m is not None:
            object_width_m = finite(
                object_width_m,
                "object_width_m",
            )

            if not (
                specification.minimum_opening_m
                <= object_width_m
                <= specification.maximum_opening_m
            ):
                raise GripperValidationError(
                    "Simulated object width outside limits"
                )

        self._object_width_m = object_width_m

        self._stiffness = finite(
            object_contact_stiffness_n_m,
            "object_contact_stiffness_n_m",
        )

        if self._stiffness <= 0:
            raise GripperValidationError(
                "Contact stiffness must be positive"
            )

        self._force_n = 0.0
        self._velocity_m_s = 0.0
        self._temperature_c = 25.0

        self._fault = GripperFaultCode.NONE
        self._emergency_stop = False

    def set_simulated_object(
        self,
        width_m: float | None,
    ) -> None:
        with self._lock:
            if width_m is not None:
                width_m = finite(
                    width_m,
                    "width_m",
                )

                if not (
                    self.specification.minimum_opening_m
                    <= width_m
                    <= self.specification.maximum_opening_m
                ):
                    raise GripperValidationError(
                        "Object width outside limits"
                    )

            self._object_width_m = width_m

            if width_m is None:
                self._force_n = 0.0

    def _telemetry(self) -> GripperTelemetry:
        contact = (
            self._object_width_m is not None
            and self._force_n
            >= self.specification.contact_force_threshold_n
        )

        held = (
            contact
            and self._force_n
            >= self.specification.holding_force_threshold_n
        )

        return GripperTelemetry(
            opening_m=self._opening_m,
            force_n=self._force_n,
            velocity_m_s=self._velocity_m_s,
            motor_temperature_c=self._temperature_c,
            contact_detected=bool(contact),
            object_detected=bool(held),
            fault_code=self._fault,
            timestamp=utc_now(),
        )

    def read_telemetry(self) -> GripperTelemetry:
        with self._lock:
            return self._telemetry()

    def move(
        self,
        opening_m: float,
        speed_m_s: float,
        force_limit_n: float,
        timeout_s: float,
    ) -> GripperTelemetry:
        with self._lock:
            if self._emergency_stop:
                raise GripperEmergencyStopError(
                    "Simulated emergency stop is active"
                )

            if self._fault != GripperFaultCode.NONE:
                raise GripperHardwareError(
                    f"Simulated fault: {self._fault.value}"
                )

            target = finite(
                opening_m,
                "opening_m",
            )

            speed = finite(
                speed_m_s,
                "speed_m_s",
            )

            force_limit = finite(
                force_limit_n,
                "force_limit_n",
            )

            timeout = finite(
                timeout_s,
                "timeout_s",
            )

            if not (
                self.specification.minimum_opening_m
                <= target
                <= self.specification.maximum_opening_m
            ):
                raise GripperValidationError(
                    "Target opening outside limits"
                )

            if not (
                0 < speed
                <= self.specification.maximum_speed_m_s
            ):
                raise GripperValidationError(
                    "Invalid movement speed"
                )

            if not (
                0 < force_limit
                <= self.specification.maximum_force_n
            ):
                raise GripperValidationError(
                    "Invalid force limit"
                )

            if timeout <= 0:
                raise GripperValidationError(
                    "Timeout must be positive"
                )

            start = self._opening_m

            # A simulated object stops inward motion
            # at its surface. Contact force is modeled
            # as a bounded command force rather than
            # physical penetration.
            final_target = target
            contact = False

            if (
                self._object_width_m is not None
                and target < self._object_width_m
                and start >= self._object_width_m
            ):
                final_target = self._object_width_m
                contact = True

            elif (
                self._object_width_m is not None
                and target < self._object_width_m
                and start <= self._object_width_m
            ):
                final_target = self._object_width_m
                contact = True

            travel = abs(
                final_target - start
            )

            required_time = travel / speed

            if required_time > timeout:
                self._fault = (
                    GripperFaultCode.COMMAND_TIMEOUT
                )

                self._velocity_m_s = 0.0

                raise GripperTimeoutError(
                    "Simulated movement exceeds timeout"
                )

            self._opening_m = final_target

            self._velocity_m_s = 0.0

            if contact:
                requested_penetration = max(
                    0.0,
                    self._object_width_m - target,
                )

                estimated_force = (
                    requested_penetration
                    * self._stiffness
                )

                self._force_n = min(
                    force_limit,
                    max(
                        estimated_force,
                        self.specification.contact_force_threshold_n,
                    ),
                )

            else:
                self._force_n = 0.0

            self._temperature_c += min(
                0.2,
                travel * 2.0,
            )

            return self._telemetry()

    def stop(self) -> GripperTelemetry:
        with self._lock:
            self._velocity_m_s = 0.0
            return self._telemetry()

    def emergency_stop(self) -> GripperTelemetry:
        with self._lock:
            self._emergency_stop = True
            self._velocity_m_s = 0.0
            return self._telemetry()

    def reset_fault(self) -> GripperTelemetry:
        with self._lock:
            self._fault = GripperFaultCode.NONE
            self._emergency_stop = False
            self._velocity_m_s = 0.0
            return self._telemetry()

    def inject_fault(
        self,
        fault: GripperFaultCode,
    ) -> None:
        with self._lock:
            self._fault = fault

    def inject_temperature(
        self,
        temperature_c: float,
    ) -> None:
        with self._lock:
            self._temperature_c = finite(
                temperature_c,
                "temperature_c",
            )

    def inject_force(
        self,
        force_n: float,
    ) -> None:
        with self._lock:
            value = finite(
                force_n,
                "force_n",
            )

            if value < 0:
                raise GripperValidationError(
                    "Force cannot be negative"
                )

            self._force_n = value


# ============================================================
# COMMAND VALIDATOR
# ============================================================

class GripperCommandValidator:

    def __init__(
        self,
        specification: GripperSpecification,
        config: GripperControllerConfig,
    ) -> None:
        self.specification = specification
        self.config = config

    def validate(
        self,
        command: GripperCommand,
    ) -> None:
        identifier(
            command.command_id,
            "command_id",
        )

        if command.target_opening_m is not None:
            opening = finite(
                command.target_opening_m,
                "target_opening_m",
            )

            if not (
                self.specification.minimum_opening_m
                <= opening
                <= self.specification.maximum_opening_m
            ):
                raise GripperValidationError(
                    "Target opening outside gripper limits"
                )

        if command.target_force_n is not None:
            force = finite(
                command.target_force_n,
                "target_force_n",
            )

            if not (
                0 < force
                <= self.specification.maximum_force_n
            ):
                raise GripperValidationError(
                    "Target force outside gripper limits"
                )

        if command.speed_m_s is not None:
            speed = finite(
                command.speed_m_s,
                "speed_m_s",
            )

            if not (
                0 < speed
                <= self.specification.maximum_speed_m_s
            ):
                raise GripperValidationError(
                    "Speed outside gripper limits"
                )

        if command.timeout_s is not None:
            timeout = finite(
                command.timeout_s,
                "timeout_s",
            )

            if timeout <= 0:
                raise GripperValidationError(
                    "Timeout must be positive"
                )

        if (
            command.command_type
            == GripperCommandType.MOVE
            and command.target_opening_m is None
        ):
            raise GripperValidationError(
                "MOVE requires target_opening_m"
            )


# ============================================================
# GRASP DETECTION
# ============================================================

class GripperGraspDetector:

    def __init__(
        self,
        specification: GripperSpecification,
        config: GripperControllerConfig,
    ) -> None:
        self.specification = specification
        self.config = config

        self._previous_opening: float | None = None
        self._previous_force: float | None = None

        self._lock = threading.RLock()

    def detect(
        self,
        telemetry: GripperTelemetry,
    ) -> GraspStatus:
        with self._lock:
            if (
                telemetry.fault_code
                != GripperFaultCode.NONE
            ):
                return GraspStatus.UNKNOWN

            force = telemetry.force_n
            opening = telemetry.opening_m

            if (
                self._previous_force is not None
                and self._previous_opening is not None
            ):
                force_drop = (
                    self._previous_force - force
                )

                position_change = abs(
                    opening
                    - self._previous_opening
                )

                previous_force = max(
                    self._previous_force,
                    1e-9,
                )

                drop_ratio = (
                    force_drop / previous_force
                )

                if (
                    self._previous_force
                    >= self.specification.holding_force_threshold_n
                    and drop_ratio
                    >= self.config.slip_force_drop_ratio
                    and position_change
                    >= self.config.slip_position_threshold_m
                ):
                    status = GraspStatus.POSSIBLE_SLIP

                elif (
                    telemetry.object_detected
                    and force
                    >= self.specification.holding_force_threshold_n
                ):
                    status = GraspStatus.OBJECT_HELD

                elif (
                    telemetry.contact_detected
                    and force
                    >= self.specification.contact_force_threshold_n
                ):
                    status = GraspStatus.CONTACT_DETECTED

                else:
                    status = GraspStatus.NO_OBJECT

            elif (
                telemetry.object_detected
                and force
                >= self.specification.holding_force_threshold_n
            ):
                status = GraspStatus.OBJECT_HELD

            elif (
                telemetry.contact_detected
                and force
                >= self.specification.contact_force_threshold_n
            ):
                status = GraspStatus.CONTACT_DETECTED

            else:
                status = GraspStatus.NO_OBJECT

            self._previous_force = force
            self._previous_opening = opening

            return status

    def reset(self) -> None:
        with self._lock:
            self._previous_force = None
            self._previous_opening = None


# ============================================================
# HEALTH MONITOR
# ============================================================

class GripperHealthMonitor:

    def __init__(
        self,
        specification: GripperSpecification,
    ) -> None:
        self.specification = specification

    def evaluate(
        self,
        telemetry: GripperTelemetry,
        emergency_stop_active: bool,
    ) -> GripperHealthStatus:
        if emergency_stop_active:
            return (
                GripperHealthStatus.EMERGENCY_STOP
            )

        if (
            telemetry.fault_code
            != GripperFaultCode.NONE
        ):
            return GripperHealthStatus.FAULT

        if (
            telemetry.force_n
            > self.specification.maximum_force_n
        ):
            return GripperHealthStatus.FAULT

        if (
            telemetry.motor_temperature_c
            >= self.specification.maximum_temperature_c
        ):
            return GripperHealthStatus.FAULT

        if (
            telemetry.motor_temperature_c
            >= 0.85
            * self.specification.maximum_temperature_c
        ):
            return GripperHealthStatus.DEGRADED

        return GripperHealthStatus.HEALTHY


# ============================================================
# GRIPPER CONTROLLER
# ============================================================

class GripperController:

    def __init__(
        self,
        *,
        robot_id: str,
        tenant_id: str,
        adapter: GripperHardwareAdapter,
        specification: GripperSpecification | None = None,
        config: GripperControllerConfig | None = None,
    ) -> None:
        self.robot_id = identifier(
            robot_id,
            "robot_id",
        )

        self.tenant_id = identifier(
            tenant_id,
            "tenant_id",
        )

        self.specification = (
            specification
            or GripperSpecification()
        )

        self.config = (
            config
            or GripperControllerConfig()
        )

        if (
            self.config.require_simulation
            and not adapter.simulation_only
        ):
            raise GripperAuthorizationError(
                "Non-simulation adapters are disabled"
            )

        if (
            self.config.default_speed_m_s
            > self.specification.maximum_speed_m_s
        ):
            raise GripperValidationError(
                "Default speed exceeds gripper limit"
            )

        if (
            self.config.default_grasp_force_n
            > self.specification.maximum_force_n
        ):
            raise GripperValidationError(
                "Default force exceeds gripper limit"
            )

        self.adapter = adapter

        self.validator = (
            GripperCommandValidator(
                self.specification,
                self.config,
            )
        )

        self.grasp_detector = (
            GripperGraspDetector(
                self.specification,
                self.config,
            )
        )

        self.health_monitor = (
            GripperHealthMonitor(
                self.specification
            )
        )

        self._state = GripperState.IDLE

        self._emergency_stop_active = False

        self._history: deque[
            GripperCommandResult
        ] = deque(
            maxlen=self.config.maximum_history
        )

        self._lock = threading.RLock()

    def _read(self) -> GripperTelemetry:
        return self.adapter.read_telemetry()

    def _check_interlocks(self) -> None:
        if self._emergency_stop_active:
            raise GripperEmergencyStopError(
                "Emergency stop is active"
            )

        if self._state == GripperState.FAULT:
            raise GripperStateError(
                "Gripper is in FAULT state"
            )

        telemetry = self._read()

        health = self.health_monitor.evaluate(
            telemetry,
            self._emergency_stop_active,
        )

        if health == GripperHealthStatus.FAULT:
            self._state = GripperState.FAULT

            raise GripperStateError(
                "Gripper health check failed"
            )

    def _record(
        self,
        command: GripperCommand,
        status: CommandStatus,
        initial_opening: float,
        telemetry: GripperTelemetry,
        duration: float,
        message: str,
    ) -> GripperCommandResult:
        result = GripperCommandResult(
            command_id=command.command_id,
            robot_id=self.robot_id,
            tenant_id=self.tenant_id,
            command_type=command.command_type,
            status=status,
            initial_opening_m=initial_opening,
            final_opening_m=telemetry.opening_m,
            final_force_n=telemetry.force_n,
            duration_s=duration,
            message=message,
            timestamp=utc_now(),
        )

        self._history.append(
            result
        )

        return result

    def execute(
        self,
        command: GripperCommand,
    ) -> GripperCommandResult:
        with self._lock:
            self.validator.validate(
                command
            )

            started = time.monotonic()

            initial = self._read()
            initial_opening = initial.opening_m

            try:
                if (
                    command.command_type
                    == GripperCommandType.EMERGENCY_STOP
                ):
                    self._emergency_stop_active = True
                    self._state = GripperState.EMERGENCY_STOP

                    telemetry = (
                        self.adapter.emergency_stop()
                    )

                    return self._record(
                        command,
                        CommandStatus.COMPLETED,
                        initial_opening,
                        telemetry,
                        time.monotonic() - started,
                        "Emergency stop activated",
                    )

                if (
                    command.command_type
                    == GripperCommandType.RESET_FAULT
                ):
                    telemetry = (
                        self.adapter.reset_fault()
                    )

                    self._emergency_stop_active = False
                    self._state = GripperState.IDLE
                    self.grasp_detector.reset()

                    return self._record(
                        command,
                        CommandStatus.COMPLETED,
                        initial_opening,
                        telemetry,
                        time.monotonic() - started,
                        "Simulated fault reset completed",
                    )

                if (
                    command.command_type
                    == GripperCommandType.STOP
                ):
                    telemetry = (
                        self.adapter.stop()
                    )

                    if not self._emergency_stop_active:
                        self._state = GripperState.STOPPED

                    return self._record(
                        command,
                        CommandStatus.COMPLETED,
                        initial_opening,
                        telemetry,
                        time.monotonic() - started,
                        "Gripper stopped",
                    )

                self._check_interlocks()

                speed = (
                    command.speed_m_s
                    if command.speed_m_s is not None
                    else self.config.default_speed_m_s
                )

                force = (
                    command.target_force_n
                    if command.target_force_n is not None
                    else self.config.default_grasp_force_n
                )

                timeout = (
                    command.timeout_s
                    if command.timeout_s is not None
                    else self.config.command_timeout_s
                )

                if command.command_type in (
                    GripperCommandType.OPEN,
                    GripperCommandType.RELEASE,
                ):
                    target = (
                        self.specification.maximum_opening_m
                    )

                elif command.command_type in (
                    GripperCommandType.CLOSE,
                    GripperCommandType.GRASP,
                ):
                    target = (
                        self.specification.minimum_opening_m
                    )

                elif (
                    command.command_type
                    == GripperCommandType.MOVE
                ):
                    if command.target_opening_m is None:
                        raise GripperValidationError(
                            "MOVE requires target opening"
                        )

                    target = command.target_opening_m

                else:
                    raise GripperValidationError(
                        "Unsupported gripper command"
                    )

                self._state = (
                    GripperState.OPENING
                    if target > initial_opening
                    else GripperState.CLOSING
                )

                telemetry = self.adapter.move(
                    opening_m=target,
                    speed_m_s=speed,
                    force_limit_n=force,
                    timeout_s=timeout,
                )

                grasp_status = (
                    self.grasp_detector.detect(
                        telemetry
                    )
                )

                if (
                    grasp_status
                    == GraspStatus.OBJECT_HELD
                ):
                    self._state = GripperState.HOLDING

                else:
                    self._state = GripperState.IDLE

                return self._record(
                    command,
                    CommandStatus.COMPLETED,
                    initial_opening,
                    telemetry,
                    time.monotonic() - started,
                    "Simulated command completed",
                )

            except GripperTimeoutError as exc:
                self._state = GripperState.FAULT

                telemetry = self._read()

                return self._record(
                    command,
                    CommandStatus.TIMED_OUT,
                    initial_opening,
                    telemetry,
                    time.monotonic() - started,
                    str(exc),
                )

            except (
                GripperHardwareError,
                GripperStateError,
                GripperEmergencyStopError,
            ) as exc:
                if not self._emergency_stop_active:
                    self._state = GripperState.FAULT

                telemetry = self._read()

                return self._record(
                    command,
                    CommandStatus.FAILED,
                    initial_opening,
                    telemetry,
                    time.monotonic() - started,
                    str(exc),
                )

    def open(
        self,
        *,
        speed_m_s: float | None = None,
    ) -> GripperCommandResult:
        return self.execute(
            GripperCommand(
                command_id=new_id(),
                command_type=GripperCommandType.OPEN,
                speed_m_s=speed_m_s,
            )
        )

    def close(
        self,
        *,
        force_n: float | None = None,
        speed_m_s: float | None = None,
    ) -> GripperCommandResult:
        return self.execute(
            GripperCommand(
                command_id=new_id(),
                command_type=GripperCommandType.CLOSE,
                target_force_n=force_n,
                speed_m_s=speed_m_s,
            )
        )

    def move(
        self,
        opening_m: float,
        *,
        force_n: float | None = None,
        speed_m_s: float | None = None,
    ) -> GripperCommandResult:
        return self.execute(
            GripperCommand(
                command_id=new_id(),
                command_type=GripperCommandType.MOVE,
                target_opening_m=opening_m,
                target_force_n=force_n,
                speed_m_s=speed_m_s,
            )
        )

    def grasp(
        self,
        *,
        force_n: float | None = None,
        speed_m_s: float | None = None,
    ) -> GripperCommandResult:
        return self.execute(
            GripperCommand(
                command_id=new_id(),
                command_type=GripperCommandType.GRASP,
                target_force_n=force_n,
                speed_m_s=speed_m_s,
            )
        )

    def release(self) -> GripperCommandResult:
        return self.execute(
            GripperCommand(
                command_id=new_id(),
                command_type=GripperCommandType.RELEASE,
            )
        )

    def stop(self) -> GripperCommandResult:
        return self.execute(
            GripperCommand(
                command_id=new_id(),
                command_type=GripperCommandType.STOP,
            )
        )

    def emergency_stop(self) -> GripperCommandResult:
        return self.execute(
            GripperCommand(
                command_id=new_id(),
                command_type=GripperCommandType.EMERGENCY_STOP,
            )
        )

    def reset_fault(self) -> GripperCommandResult:
        return self.execute(
            GripperCommand(
                command_id=new_id(),
                command_type=GripperCommandType.RESET_FAULT,
            )
        )

    def status(self) -> GripperStatusReport:
        with self._lock:
            telemetry = self._read()

            grasp_status = (
                self.grasp_detector.detect(
                    telemetry
                )
            )

            health = (
                self.health_monitor.evaluate(
                    telemetry,
                    self._emergency_stop_active,
                )
            )

            return GripperStatusReport(
                robot_id=self.robot_id,
                tenant_id=self.tenant_id,
                state=self._state,
                grasp_status=grasp_status,
                health=health,
                telemetry=telemetry,
                emergency_stop_active=(
                    self._emergency_stop_active
                ),
                timestamp=utc_now(),
            )

    def history(
        self,
        limit: int = 100,
    ) -> tuple[GripperCommandResult, ...]:
        if limit < 1:
            raise GripperValidationError(
                "History limit must be positive"
            )

        with self._lock:
            return tuple(
                list(self._history)[-limit:]
            )


# ============================================================
# MULTI-TENANT GRIPPER SERVICE
# ============================================================

class GripperControllerService:

    def __init__(self) -> None:
        self._controllers: dict[
            tuple[str, str],
            GripperController,
        ] = {}

        self._audit: deque[
            GripperAuditRecord
        ] = deque(
            maxlen=10000
        )

        self._lock = threading.RLock()

    @staticmethod
    def _authorize(
        access: GripperAccessContext,
        permission: str,
    ) -> None:
        if not getattr(
            access,
            permission,
            False,
        ):
            raise GripperAuthorizationError(
                f"Missing permission: {permission}"
            )

    def register_controller(
        self,
        controller: GripperController,
        access: GripperAccessContext,
    ) -> None:
        self._authorize(
            access,
            "can_manage",
        )

        if controller.tenant_id != access.tenant_id:
            raise GripperAuthorizationError(
                "Cross-tenant registration forbidden"
            )

        key = (
            controller.tenant_id,
            controller.robot_id,
        )

        with self._lock:
            if key in self._controllers:
                raise GripperConflictError(
                    "Controller already registered"
                )

            self._controllers[key] = controller

    def get_controller(
        self,
        robot_id: str,
        access: GripperAccessContext,
    ) -> GripperController:
        self._authorize(
            access,
            "can_read",
        )

        key = (
            access.tenant_id,
            robot_id,
        )

        with self._lock:
            controller = self._controllers.get(
                key
            )

        if controller is None:
            raise GripperNotFoundError(
                "Controller not found"
            )

        return controller

    def execute(
        self,
        robot_id: str,
        command: GripperCommand,
        access: GripperAccessContext,
    ) -> GripperCommandResult:
        self._authorize(
            access,
            "can_command",
        )

        controller = self.get_controller(
            robot_id,
            access,
        )

        result = controller.execute(
            command
        )

        with self._lock:
            self._audit.append(
                GripperAuditRecord(
                    audit_id=new_id(),
                    robot_id=robot_id,
                    tenant_id=access.tenant_id,
                    user_id=access.user_id,
                    action=command.command_type.value,
                    command_id=command.command_id,
                    success=(
                        result.status
                        == CommandStatus.COMPLETED
                    ),
                    timestamp=utc_now(),
                )
            )

        return result

    def status(
        self,
        robot_id: str,
        access: GripperAccessContext,
    ) -> GripperStatusReport:
        controller = self.get_controller(
            robot_id,
            access,
        )

        return controller.status()

    def list_controllers(
        self,
        access: GripperAccessContext,
    ) -> tuple[str, ...]:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            return tuple(
                robot_id
                for tenant_id, robot_id
                in self._controllers
                if tenant_id == access.tenant_id
            )

    def audit_history(
        self,
        access: GripperAccessContext,
        limit: int = 100,
    ) -> tuple[GripperAuditRecord, ...]:
        self._authorize(
            access,
            "can_read",
        )

        if limit < 1:
            raise GripperValidationError(
                "Audit limit must be positive"
            )

        with self._lock:
            records = [
                record
                for record in self._audit
                if record.tenant_id
                == access.tenant_id
            ]

        return tuple(
            records[-limit:]
        )

    def export_status_json(
        self,
        robot_id: str,
        access: GripperAccessContext,
    ) -> str:
        report = self.status(
            robot_id,
            access,
        )

        return json.dumps(
            json_safe(
                {
                    "status": report,
                    "simulation_only": True,
                    "safety_certified": False,
                }
            ),
            indent=2,
            allow_nan=False,
        )


# ============================================================
# SYNTHETIC FACTORY
# ============================================================

class SyntheticGripperFactory:

    @staticmethod
    def specification() -> GripperSpecification:
        return GripperSpecification(
            minimum_opening_m=0.0,
            maximum_opening_m=0.12,
            maximum_force_n=80.0,
            maximum_speed_m_s=0.05,
            contact_force_threshold_n=2.0,
            holding_force_threshold_n=5.0,
        )

    @staticmethod
    def controller(
        *,
        robot_id: str = "robot-arm-001",
        tenant_id: str = "physical_ai_lab",
        object_width_m: float | None = 0.05,
    ) -> GripperController:
        specification = (
            SyntheticGripperFactory.specification()
        )

        adapter = SimulatedGripperAdapter(
            specification,
            object_width_m=object_width_m,
        )

        return GripperController(
            robot_id=robot_id,
            tenant_id=tenant_id,
            adapter=adapter,
            specification=specification,
        )

    @staticmethod
    def access(
        tenant_id: str = "physical_ai_lab",
    ) -> GripperAccessContext:
        return GripperAccessContext(
            tenant_id=tenant_id,
            user_id="robotics_engineer",
            can_read=True,
            can_command=True,
            can_manage=True,
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestGripperController(unittest.TestCase):

    def setUp(self) -> None:
        self.controller = (
            SyntheticGripperFactory.controller()
        )

        self.access = (
            SyntheticGripperFactory.access()
        )

    def test_initial_state(self) -> None:
        status = self.controller.status()

        self.assertEqual(
            status.state,
            GripperState.IDLE,
        )

    def test_open_gripper(self) -> None:
        result = self.controller.open()

        self.assertEqual(
            result.status,
            CommandStatus.COMPLETED,
        )

        self.assertAlmostEqual(
            result.final_opening_m,
            0.12,
        )

    def test_close_gripper(self) -> None:
        result = self.controller.close()

        self.assertEqual(
            result.status,
            CommandStatus.COMPLETED,
        )

    def test_grasp_object(self) -> None:
        result = self.controller.grasp(
            force_n=20.0
        )

        self.assertEqual(
            result.status,
            CommandStatus.COMPLETED,
        )

        self.assertGreaterEqual(
            result.final_force_n,
            5.0,
        )

    def test_object_held_detection(self) -> None:
        self.controller.grasp(
            force_n=20.0
        )

        status = self.controller.status()

        self.assertEqual(
            status.grasp_status,
            GraspStatus.OBJECT_HELD,
        )

    def test_release_object(self) -> None:
        self.controller.grasp(
            force_n=20.0
        )

        result = self.controller.release()

        self.assertEqual(
            result.status,
            CommandStatus.COMPLETED,
        )

        self.assertAlmostEqual(
            result.final_opening_m,
            0.12,
        )

    def test_move_to_position(self) -> None:
        result = self.controller.move(
            0.08
        )

        self.assertAlmostEqual(
            result.final_opening_m,
            0.08,
        )

    def test_invalid_position(self) -> None:
        with self.assertRaises(
            GripperValidationError
        ):
            self.controller.move(
                0.50
            )

    def test_invalid_force(self) -> None:
        with self.assertRaises(
            GripperValidationError
        ):
            self.controller.grasp(
                force_n=1000.0
            )

    def test_invalid_speed(self) -> None:
        with self.assertRaises(
            GripperValidationError
        ):
            self.controller.open(
                speed_m_s=1.0
            )

    def test_stop(self) -> None:
        result = self.controller.stop()

        self.assertEqual(
            result.status,
            CommandStatus.COMPLETED,
        )

        self.assertEqual(
            self.controller.status().state,
            GripperState.STOPPED,
        )

    def test_emergency_stop(self) -> None:
        self.controller.emergency_stop()

        status = self.controller.status()

        self.assertTrue(
            status.emergency_stop_active
        )

        self.assertEqual(
            status.state,
            GripperState.EMERGENCY_STOP,
        )

    def test_emergency_stop_blocks_motion(self) -> None:
        self.controller.emergency_stop()

        result = self.controller.grasp()

        self.assertEqual(
            result.status,
            CommandStatus.FAILED,
        )

    def test_reset_fault(self) -> None:
        self.controller.emergency_stop()

        result = self.controller.reset_fault()

        self.assertEqual(
            result.status,
            CommandStatus.COMPLETED,
        )

        self.assertFalse(
            self.controller.status().emergency_stop_active
        )

    def test_fault_detection(self) -> None:
        adapter = self.controller.adapter

        self.assertIsInstance(
            adapter,
            SimulatedGripperAdapter,
        )

        adapter.inject_fault(
            GripperFaultCode.SENSOR_FAILURE
        )

        status = self.controller.status()

        self.assertEqual(
            status.health,
            GripperHealthStatus.FAULT,
        )

    def test_overtemperature_detection(self) -> None:
        adapter = self.controller.adapter

        self.assertIsInstance(
            adapter,
            SimulatedGripperAdapter,
        )

        adapter.inject_temperature(
            90.0
        )

        status = self.controller.status()

        self.assertEqual(
            status.health,
            GripperHealthStatus.FAULT,
        )

    def test_command_history(self) -> None:
        self.controller.open()
        self.controller.grasp()

        history = self.controller.history()

        self.assertEqual(
            len(history),
            2,
        )

    def test_timeout(self) -> None:
        command = GripperCommand(
            command_id=new_id(),
            command_type=GripperCommandType.CLOSE,
            speed_m_s=0.001,
            timeout_s=0.001,
        )

        result = self.controller.execute(
            command
        )

        self.assertEqual(
            result.status,
            CommandStatus.TIMED_OUT,
        )

    def test_service_registration(self) -> None:
        service = GripperControllerService()

        service.register_controller(
            self.controller,
            self.access,
        )

        self.assertEqual(
            service.list_controllers(
                self.access
            ),
            ("robot-arm-001",),
        )

    def test_service_execute(self) -> None:
        service = GripperControllerService()

        service.register_controller(
            self.controller,
            self.access,
        )

        result = service.execute(
            "robot-arm-001",
            GripperCommand(
                command_id=new_id(),
                command_type=GripperCommandType.GRASP,
            ),
            self.access,
        )

        self.assertEqual(
            result.status,
            CommandStatus.COMPLETED,
        )

    def test_tenant_isolation(self) -> None:
        service = GripperControllerService()

        service.register_controller(
            self.controller,
            self.access,
        )

        other = GripperAccessContext(
            tenant_id="other_tenant",
            user_id="other_user",
            can_read=True,
            can_command=True,
        )

        with self.assertRaises(
            GripperNotFoundError
        ):
            service.get_controller(
                "robot-arm-001",
                other,
            )

    def test_permission_denied(self) -> None:
        service = GripperControllerService()

        service.register_controller(
            self.controller,
            self.access,
        )

        viewer = GripperAccessContext(
            tenant_id="physical_ai_lab",
            user_id="viewer",
            can_read=True,
        )

        with self.assertRaises(
            GripperAuthorizationError
        ):
            service.execute(
                "robot-arm-001",
                GripperCommand(
                    command_id=new_id(),
                    command_type=GripperCommandType.OPEN,
                ),
                viewer,
            )

    def test_audit_history(self) -> None:
        service = GripperControllerService()

        service.register_controller(
            self.controller,
            self.access,
        )

        service.execute(
            "robot-arm-001",
            GripperCommand(
                command_id=new_id(),
                command_type=GripperCommandType.OPEN,
            ),
            self.access,
        )

        self.assertEqual(
            len(
                service.audit_history(
                    self.access
                )
            ),
            1,
        )

    def test_json_export(self) -> None:
        service = GripperControllerService()

        service.register_controller(
            self.controller,
            self.access,
        )

        exported = json.loads(
            service.export_status_json(
                "robot-arm-001",
                self.access,
            )
        )

        self.assertTrue(
            exported["simulation_only"]
        )

        self.assertFalse(
            exported["safety_certified"]
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("\nPHYSICAL AI — GRIPPER CONTROLLER")
    print("=" * 72)

    controller = SyntheticGripperFactory.controller(
        robot_id="physical-ai-robot-arm-001",
        object_width_m=0.05,
    )

    access = SyntheticGripperFactory.access()

    service = GripperControllerService()

    service.register_controller(
        controller,
        access,
    )

    commands = [
        GripperCommand(
            command_id=new_id(),
            command_type=GripperCommandType.OPEN,
        ),
        GripperCommand(
            command_id=new_id(),
            command_type=GripperCommandType.GRASP,
            target_force_n=20.0,
        ),
        GripperCommand(
            command_id=new_id(),
            command_type=GripperCommandType.RELEASE,
        ),
    ]

    for command in commands:
        result = service.execute(
            controller.robot_id,
            command,
            access,
        )

        print(
            f"{command.command_type.value}: "
            f"{result.status.value}, "
            f"opening={result.final_opening_m:.3f} m, "
            f"force={result.final_force_n:.2f} N"
        )

    print("\nGRIPPER STATUS")
    print("=" * 72)

    print(
        service.export_status_json(
            controller.robot_id,
            access,
        )
    )

    print("\nRUNNING UNIT TESTS")
    print("=" * 72)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestGripperController
    )

    result = unittest.TextTestRunner(
        verbosity=2
    ).run(suite)

    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
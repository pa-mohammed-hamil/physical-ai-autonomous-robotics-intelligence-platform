"""Force Feedback module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/force_feedback.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================
Robotics Control — Force Feedback

Features:
    - Six-axis force/torque sensor processing
    - Force and torque measurement validation
    - Sensor calibration and zero-offset compensation
    - Low-pass filtering
    - Exponential moving average filtering
    - Median filtering
    - Gravity compensation using a supplied orientation
    - Contact detection
    - Contact state hysteresis
    - Force threshold monitoring
    - Torque threshold monitoring
    - Force magnitude estimation
    - Force derivative estimation
    - Contact direction estimation
    - Simplified impedance response calculation
    - Simplified admittance response calculation
    - Virtual stiffness and damping
    - Sensor health monitoring
    - Sensor saturation detection
    - Stale measurement detection
    - Emergency-stop advisory state
    - Safety event reporting
    - Multi-tenant isolation
    - Thread-safe sensor management
    - Simulation-only feedback processing
    - Synthetic robotics force sensor data
    - Unit tests

Requirements:
    Python >= 3.11
    numpy >= 1.24

Install:
    pip install numpy

SAFETY:
    This module is intended for simulation, testing,
    research, and offline force-feedback analysis.

    It does not issue commands to physical actuators.

    Impedance and admittance calculations are simplified
    advisory models, not validated real-time controllers.

    Real hardware requires independent safety-rated
    force limiting, emergency-stop circuits, watchdogs,
    calibrated sensors, and verified robot dynamics.
"""

from __future__ import annotations

import json
import math
import threading
import unittest
import uuid

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Sequence

import numpy as np


# ============================================================
# EXCEPTIONS
# ============================================================

class ForceFeedbackError(Exception):
    pass


class ForceValidationError(ForceFeedbackError):
    pass


class ForceSensorNotFoundError(ForceFeedbackError):
    pass


class ForceAuthorizationError(ForceFeedbackError):
    pass


class ForceCalibrationError(ForceFeedbackError):
    pass


class ForceSafetyError(ForceFeedbackError):
    pass


class ForceStateError(ForceFeedbackError):
    pass


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


def validate_identifier(
    value: str,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 256
    ):
        raise ForceValidationError(
            f"Invalid {name}"
        )

    return value.strip()


def finite_number(
    value: Any,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise ForceValidationError(
            f"{name} must be numeric"
        )

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ForceValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(number):
        raise ForceValidationError(
            f"{name} must be finite"
        )

    return number


def vector3(
    values: Sequence[float],
    name: str,
) -> np.ndarray:
    try:
        result = np.asarray(
            values,
            dtype=float,
        )
    except (TypeError, ValueError) as exc:
        raise ForceValidationError(
            f"{name} must be numeric"
        ) from exc

    if result.shape != (3,):
        raise ForceValidationError(
            f"{name} must contain three values"
        )

    if not np.all(np.isfinite(result)):
        raise ForceValidationError(
            f"{name} contains non-finite values"
        )

    return result


def vector6(
    values: Sequence[float],
    name: str,
) -> np.ndarray:
    try:
        result = np.asarray(
            values,
            dtype=float,
        )
    except (TypeError, ValueError) as exc:
        raise ForceValidationError(
            f"{name} must be numeric"
        ) from exc

    if result.shape != (6,):
        raise ForceValidationError(
            f"{name} must contain six values"
        )

    if not np.all(np.isfinite(result)):
        raise ForceValidationError(
            f"{name} contains non-finite values"
        )

    return result


def ensure_utc(
    timestamp: datetime,
) -> datetime:
    if not isinstance(timestamp, datetime):
        raise ForceValidationError(
            "Timestamp must be datetime"
        )

    if timestamp.tzinfo is None:
        raise ForceValidationError(
            "Timestamp must be timezone-aware"
        )

    return timestamp.astimezone(
        timezone.utc
    )


def magnitude(
    values: Sequence[float],
) -> float:
    return float(
        np.linalg.norm(
            np.asarray(
                values,
                dtype=float,
            )
        )
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if hasattr(value, "__dataclass_fields__"):
        return {
            name: json_safe(getattr(value, name))
            for name in value.__dataclass_fields__
        }

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (tuple, list)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ForceValidationError(
                "Cannot serialize non-finite value"
            )

    if isinstance(
        value,
        (str, int, float, bool),
    ) or value is None:
        return value

    raise ForceValidationError(
        f"Unsupported value: {type(value).__name__}"
    )


# ============================================================
# ENUMERATIONS
# ============================================================

class SensorStatus(str, Enum):
    READY = "ready"
    CALIBRATING = "calibrating"
    WARNING = "warning"
    FAULT = "fault"
    DISABLED = "disabled"


class SensorHealth(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ContactState(str, Enum):
    NO_CONTACT = "no_contact"
    CONTACT = "contact"
    HIGH_FORCE = "high_force"


class SafetySeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class SafetyEventType(str, Enum):
    CONTACT_DETECTED = "contact_detected"
    CONTACT_RELEASED = "contact_released"
    HIGH_FORCE = "high_force"
    HIGH_TORQUE = "high_torque"
    SENSOR_SATURATION = "sensor_saturation"
    STALE_SENSOR = "stale_sensor"
    EMERGENCY_STOP = "emergency_stop"
    EMERGENCY_RESET = "emergency_reset"


class FilterType(str, Enum):
    NONE = "none"
    LOW_PASS = "low_pass"
    EMA = "ema"
    MEDIAN = "median"


class FeedbackMode(str, Enum):
    MONITORING = "monitoring"
    IMPEDANCE = "impedance"
    ADMITTANCE = "admittance"
    DISABLED = "disabled"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ForceSensorConfig:
    sensor_id: str
    robot_id: str
    tenant_id: str

    maximum_force_axis: float = 500.0
    maximum_torque_axis: float = 100.0

    contact_enter_threshold: float = 5.0
    contact_exit_threshold: float = 3.0

    warning_force_threshold: float = 80.0
    critical_force_threshold: float = 150.0

    warning_torque_threshold: float = 20.0
    critical_torque_threshold: float = 40.0

    filter_type: FilterType = FilterType.LOW_PASS

    filter_alpha: float = 0.25
    median_window: int = 5

    maximum_sample_age_seconds: float = 1.0

    history_capacity: int = 1000

    def __post_init__(self) -> None:
        for name in (
            "sensor_id",
            "robot_id",
            "tenant_id",
        ):
            validate_identifier(
                getattr(self, name),
                name,
            )

        positive_fields = (
            "maximum_force_axis",
            "maximum_torque_axis",
            "warning_force_threshold",
            "critical_force_threshold",
            "warning_torque_threshold",
            "critical_torque_threshold",
            "maximum_sample_age_seconds",
        )

        for name in positive_fields:
            if finite_number(
                getattr(self, name),
                name,
            ) <= 0:
                raise ForceValidationError(
                    f"{name} must be positive"
                )

        if not (
            0 <= self.contact_exit_threshold
            < self.contact_enter_threshold
        ):
            raise ForceValidationError(
                "Invalid contact hysteresis thresholds"
            )

        if (
            self.warning_force_threshold
            >= self.critical_force_threshold
        ):
            raise ForceValidationError(
                "Force warning threshold must be "
                "below critical threshold"
            )

        if (
            self.warning_torque_threshold
            >= self.critical_torque_threshold
        ):
            raise ForceValidationError(
                "Torque warning threshold must be "
                "below critical threshold"
            )

        if not 0 < self.filter_alpha <= 1:
            raise ForceValidationError(
                "filter_alpha must be within (0, 1]"
            )

        if self.median_window < 1:
            raise ForceValidationError(
                "median_window must be positive"
            )

        if self.history_capacity < 1:
            raise ForceValidationError(
                "history_capacity must be positive"
            )


@dataclass(frozen=True)
class ImpedanceConfig:
    stiffness: tuple[float, float, float] = (
        100.0,
        100.0,
        100.0,
    )

    damping: tuple[float, float, float] = (
        15.0,
        15.0,
        15.0,
    )

    maximum_response_force: float = 100.0

    def __post_init__(self) -> None:
        stiffness = vector3(
            self.stiffness,
            "stiffness",
        )

        damping = vector3(
            self.damping,
            "damping",
        )

        if np.any(stiffness < 0):
            raise ForceValidationError(
                "Stiffness cannot be negative"
            )

        if np.any(damping < 0):
            raise ForceValidationError(
                "Damping cannot be negative"
            )

        if self.maximum_response_force <= 0:
            raise ForceValidationError(
                "Maximum response force must be positive"
            )


@dataclass(frozen=True)
class AdmittanceConfig:
    virtual_mass: tuple[float, float, float] = (
        2.0,
        2.0,
        2.0,
    )

    damping: tuple[float, float, float] = (
        20.0,
        20.0,
        20.0,
    )

    stiffness: tuple[float, float, float] = (
        50.0,
        50.0,
        50.0,
    )

    maximum_displacement: float = 0.05
    maximum_velocity: float = 0.10

    def __post_init__(self) -> None:
        mass = vector3(
            self.virtual_mass,
            "virtual_mass",
        )

        damping = vector3(
            self.damping,
            "damping",
        )

        stiffness = vector3(
            self.stiffness,
            "stiffness",
        )

        if np.any(mass <= 0):
            raise ForceValidationError(
                "Virtual mass must be positive"
            )

        if np.any(damping < 0):
            raise ForceValidationError(
                "Damping cannot be negative"
            )

        if np.any(stiffness < 0):
            raise ForceValidationError(
                "Stiffness cannot be negative"
            )

        if self.maximum_displacement <= 0:
            raise ForceValidationError(
                "Maximum displacement must be positive"
            )

        if self.maximum_velocity <= 0:
            raise ForceValidationError(
                "Maximum velocity must be positive"
            )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class ForceAccessContext:
    tenant_id: str
    user_id: str

    can_read: bool = False
    can_process: bool = False
    can_calibrate: bool = False
    can_manage: bool = False


@dataclass(frozen=True)
class ForceTorqueMeasurement:
    measurement_id: str
    sensor_id: str

    force: tuple[float, float, float]
    torque: tuple[float, float, float]

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_identifier(
            self.measurement_id,
            "measurement_id",
        )

        validate_identifier(
            self.sensor_id,
            "sensor_id",
        )

        vector3(
            self.force,
            "force",
        )

        vector3(
            self.torque,
            "torque",
        )

        ensure_utc(
            self.timestamp
        )


@dataclass(frozen=True)
class SensorCalibration:
    calibration_id: str

    force_offset: tuple[float, float, float]
    torque_offset: tuple[float, float, float]

    sample_count: int

    calibrated_at: datetime


@dataclass(frozen=True)
class ProcessedForceMeasurement:
    measurement_id: str
    sensor_id: str

    raw_force: tuple[float, float, float]
    raw_torque: tuple[float, float, float]

    compensated_force: tuple[float, float, float]
    compensated_torque: tuple[float, float, float]

    filtered_force: tuple[float, float, float]
    filtered_torque: tuple[float, float, float]

    force_magnitude: float
    torque_magnitude: float

    force_rate: float

    contact_state: ContactState

    timestamp: datetime


@dataclass(frozen=True)
class ContactReport:
    sensor_id: str

    contact_state: ContactState

    contact_detected: bool

    force_magnitude: float

    contact_direction: tuple[float, float, float]

    timestamp: datetime


@dataclass(frozen=True)
class SafetyEvent:
    event_id: str
    sensor_id: str
    robot_id: str
    tenant_id: str

    event_type: SafetyEventType
    severity: SafetySeverity

    message: str

    timestamp: datetime


@dataclass(frozen=True)
class SensorHealthReport:
    sensor_id: str

    health: SensorHealth

    status: SensorStatus

    sample_count: int

    last_sample_age_seconds: float | None

    saturation_detected: bool

    emergency_stop_active: bool

    evaluated_at: datetime


@dataclass(frozen=True)
class ImpedanceResponse:
    displacement_error: tuple[float, float, float]

    velocity_error: tuple[float, float, float]

    virtual_response_force: tuple[float, float, float]

    response_magnitude: float

    saturated: bool


@dataclass(frozen=True)
class AdmittanceResponse:
    displacement: tuple[float, float, float]

    velocity: tuple[float, float, float]

    acceleration: tuple[float, float, float]

    saturated: bool


@dataclass(frozen=True)
class ForceFeedbackState:
    sensor_id: str
    robot_id: str
    tenant_id: str

    mode: FeedbackMode
    status: SensorStatus

    contact_state: ContactState

    emergency_stop_active: bool

    latest_measurement: ProcessedForceMeasurement | None

    calibration: SensorCalibration | None

    timestamp: datetime


# ============================================================
# FORCE SENSOR CALIBRATOR
# ============================================================

class ForceSensorCalibrator:

    @staticmethod
    def calculate_zero_offsets(
        measurements: Sequence[ForceTorqueMeasurement],
    ) -> SensorCalibration:
        if len(measurements) < 3:
            raise ForceCalibrationError(
                "At least three measurements required"
            )

        sensor_ids = {
            measurement.sensor_id
            for measurement in measurements
        }

        if len(sensor_ids) != 1:
            raise ForceCalibrationError(
                "Calibration samples must belong "
                "to the same sensor"
            )

        forces = np.asarray(
            [
                measurement.force
                for measurement in measurements
            ],
            dtype=float,
        )

        torques = np.asarray(
            [
                measurement.torque
                for measurement in measurements
            ],
            dtype=float,
        )

        force_offset = np.mean(
            forces,
            axis=0,
        )

        torque_offset = np.mean(
            torques,
            axis=0,
        )

        return SensorCalibration(
            calibration_id=generate_id(),
            force_offset=tuple(
                float(value)
                for value in force_offset
            ),
            torque_offset=tuple(
                float(value)
                for value in torque_offset
            ),
            sample_count=len(measurements),
            calibrated_at=utc_now(),
        )

    @staticmethod
    def compensate(
        measurement: ForceTorqueMeasurement,
        calibration: SensorCalibration | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        force = vector3(
            measurement.force,
            "force",
        )

        torque = vector3(
            measurement.torque,
            "torque",
        )

        if calibration is None:
            return (
                force,
                torque,
            )

        return (
            force - vector3(
                calibration.force_offset,
                "force offset",
            ),
            torque - vector3(
                calibration.torque_offset,
                "torque offset",
            ),
        )


# ============================================================
# GRAVITY COMPENSATION
# ============================================================

class GravityCompensator:

    @staticmethod
    def compensate(
        measured_force: Sequence[float],
        measured_torque: Sequence[float],
        *,
        payload_mass_kg: float,
        rotation_sensor_to_world: Sequence[
            Sequence[float]
        ],
        center_of_mass_sensor: Sequence[float] = (
            0.0,
            0.0,
            0.0,
        ),
        gravity_world: Sequence[float] = (
            0.0,
            0.0,
            -9.80665,
        ),
    ) -> tuple[np.ndarray, np.ndarray]:
        mass = finite_number(
            payload_mass_kg,
            "payload_mass_kg",
        )

        if mass < 0:
            raise ForceValidationError(
                "Payload mass cannot be negative"
            )

        rotation = np.asarray(
            rotation_sensor_to_world,
            dtype=float,
        )

        if rotation.shape != (3, 3):
            raise ForceValidationError(
                "Rotation matrix must be 3x3"
            )

        if not np.all(np.isfinite(rotation)):
            raise ForceValidationError(
                "Rotation matrix contains non-finite values"
            )

        if not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=1e-5,
        ):
            raise ForceValidationError(
                "Rotation matrix must be orthonormal"
            )

        if not np.isclose(
            np.linalg.det(rotation),
            1.0,
            atol=1e-5,
        ):
            raise ForceValidationError(
                "Rotation matrix must be a proper rotation"
            )

        force = vector3(
            measured_force,
            "measured_force",
        )

        torque = vector3(
            measured_torque,
            "measured_torque",
        )

        center = vector3(
            center_of_mass_sensor,
            "center_of_mass_sensor",
        )

        gravity = vector3(
            gravity_world,
            "gravity_world",
        )

        gravity_force_sensor = (
            rotation.T @ gravity
        ) * mass

        gravity_torque_sensor = np.cross(
            center,
            gravity_force_sensor,
        )

        return (
            force - gravity_force_sensor,
            torque - gravity_torque_sensor,
        )


# ============================================================
# FORCE FILTER
# ============================================================

class ForceSignalFilter:

    def __init__(
        self,
        filter_type: FilterType,
        *,
        alpha: float = 0.25,
        median_window: int = 5,
    ) -> None:
        self.filter_type = filter_type
        self.alpha = alpha

        self._history: deque[np.ndarray] = deque(
            maxlen=median_window
        )

        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        self._history.clear()
        self._previous = None

    def process(
        self,
        values: Sequence[float],
    ) -> np.ndarray:
        current = vector6(
            values,
            "filter input",
        )

        if self.filter_type == FilterType.NONE:
            result = current.copy()

        elif self.filter_type in {
            FilterType.LOW_PASS,
            FilterType.EMA,
        }:
            if self._previous is None:
                result = current.copy()

            else:
                result = (
                    self.alpha * current
                    + (1.0 - self.alpha)
                    * self._previous
                )

        elif self.filter_type == FilterType.MEDIAN:
            self._history.append(
                current.copy()
            )

            result = np.median(
                np.asarray(
                    self._history
                ),
                axis=0,
            )

        else:
            raise ForceValidationError(
                "Unsupported filter type"
            )

        self._previous = result.copy()

        return result


# ============================================================
# CONTACT DETECTOR
# ============================================================

class ForceContactDetector:

    def __init__(
        self,
        enter_threshold: float,
        exit_threshold: float,
        high_force_threshold: float,
    ) -> None:
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.high_force_threshold = high_force_threshold

        self._state = ContactState.NO_CONTACT

    @property
    def state(self) -> ContactState:
        return self._state

    def reset(self) -> None:
        self._state = ContactState.NO_CONTACT

    def update(
        self,
        force: Sequence[float],
    ) -> ContactState:
        force_norm = magnitude(
            force
        )

        if force_norm >= self.high_force_threshold:
            self._state = ContactState.HIGH_FORCE

        elif self._state == ContactState.NO_CONTACT:
            if force_norm >= self.enter_threshold:
                self._state = ContactState.CONTACT

        elif force_norm <= self.exit_threshold:
            self._state = ContactState.NO_CONTACT

        else:
            self._state = ContactState.CONTACT

        return self._state

    @staticmethod
    def direction(
        force: Sequence[float],
    ) -> tuple[float, float, float]:
        values = vector3(
            force,
            "force",
        )

        norm = float(
            np.linalg.norm(values)
        )

        if norm <= 1e-12:
            return (
                0.0,
                0.0,
                0.0,
            )

        return tuple(
            float(value)
            for value in values / norm
        )


# ============================================================
# IMPEDANCE RESPONSE MODEL
# ============================================================

class ImpedanceFeedbackModel:

    def __init__(
        self,
        config: ImpedanceConfig | None = None,
    ) -> None:
        self.config = (
            config or ImpedanceConfig()
        )

    def calculate(
        self,
        desired_position: Sequence[float],
        actual_position: Sequence[float],
        desired_velocity: Sequence[float],
        actual_velocity: Sequence[float],
    ) -> ImpedanceResponse:
        position_error = (
            vector3(
                desired_position,
                "desired_position",
            )
            - vector3(
                actual_position,
                "actual_position",
            )
        )

        velocity_error = (
            vector3(
                desired_velocity,
                "desired_velocity",
            )
            - vector3(
                actual_velocity,
                "actual_velocity",
            )
        )

        stiffness = vector3(
            self.config.stiffness,
            "stiffness",
        )

        damping = vector3(
            self.config.damping,
            "damping",
        )

        response = (
            stiffness * position_error
            + damping * velocity_error
        )

        norm = float(
            np.linalg.norm(response)
        )

        saturated = (
            norm > self.config.maximum_response_force
        )

        if saturated:
            response = (
                response
                * self.config.maximum_response_force
                / norm
            )

        return ImpedanceResponse(
            displacement_error=tuple(
                float(value)
                for value in position_error
            ),
            velocity_error=tuple(
                float(value)
                for value in velocity_error
            ),
            virtual_response_force=tuple(
                float(value)
                for value in response
            ),
            response_magnitude=float(
                np.linalg.norm(response)
            ),
            saturated=saturated,
        )


# ============================================================
# ADMITTANCE RESPONSE MODEL
# ============================================================

class AdmittanceFeedbackModel:

    def __init__(
        self,
        config: AdmittanceConfig | None = None,
    ) -> None:
        self.config = (
            config or AdmittanceConfig()
        )

        self._displacement = np.zeros(
            3,
            dtype=float,
        )

        self._velocity = np.zeros(
            3,
            dtype=float,
        )

    def reset(self) -> None:
        self._displacement[:] = 0.0
        self._velocity[:] = 0.0

    def update(
        self,
        external_force: Sequence[float],
        dt: float,
    ) -> AdmittanceResponse:
        dt = finite_number(
            dt,
            "dt",
        )

        if dt <= 0 or dt > 0.1:
            raise ForceValidationError(
                "dt must be within (0, 0.1]"
            )

        force = vector3(
            external_force,
            "external_force",
        )

        mass = vector3(
            self.config.virtual_mass,
            "virtual_mass",
        )

        damping = vector3(
            self.config.damping,
            "damping",
        )

        stiffness = vector3(
            self.config.stiffness,
            "stiffness",
        )

        acceleration = (
            force
            - damping * self._velocity
            - stiffness * self._displacement
        ) / mass

        new_velocity = (
            self._velocity
            + acceleration * dt
        )

        saturated = False

        velocity_norm = float(
            np.linalg.norm(
                new_velocity
            )
        )

        if velocity_norm > self.config.maximum_velocity:
            new_velocity = (
                new_velocity
                * self.config.maximum_velocity
                / velocity_norm
            )

            saturated = True

        new_displacement = (
            self._displacement
            + new_velocity * dt
        )

        displacement_norm = float(
            np.linalg.norm(
                new_displacement
            )
        )

        if (
            displacement_norm
            > self.config.maximum_displacement
        ):
            new_displacement = (
                new_displacement
                * self.config.maximum_displacement
                / displacement_norm
            )

            saturated = True

        self._velocity = new_velocity
        self._displacement = new_displacement

        return AdmittanceResponse(
            displacement=tuple(
                float(value)
                for value in new_displacement
            ),
            velocity=tuple(
                float(value)
                for value in new_velocity
            ),
            acceleration=tuple(
                float(value)
                for value in acceleration
            ),
            saturated=saturated,
        )


# ============================================================
# FORCE FEEDBACK CONTROLLER
# ============================================================

class ForceFeedbackController:

    def __init__(
        self,
        config: ForceSensorConfig,
    ) -> None:
        self.config = config

        self._status = SensorStatus.READY
        self._mode = FeedbackMode.MONITORING

        self._calibration: SensorCalibration | None = None

        self._filter = ForceSignalFilter(
            config.filter_type,
            alpha=config.filter_alpha,
            median_window=config.median_window,
        )

        self._contact_detector = ForceContactDetector(
            config.contact_enter_threshold,
            config.contact_exit_threshold,
            config.critical_force_threshold,
        )

        self._history: deque[
            ProcessedForceMeasurement
        ] = deque(
            maxlen=config.history_capacity
        )

        self._events: list[SafetyEvent] = []

        self._last_raw: ForceTorqueMeasurement | None = None

        self._emergency_stop = False
        self._saturation_detected = False

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # AUTHORIZATION
    # --------------------------------------------------------

    def _authorize(
        self,
        access: ForceAccessContext,
        permission: str,
    ) -> None:
        if access.tenant_id != self.config.tenant_id:
            raise ForceAuthorizationError(
                "Cross-tenant access forbidden"
            )

        if not getattr(
            access,
            permission,
            False,
        ):
            raise ForceAuthorizationError(
                f"Missing permission: {permission}"
            )

    # --------------------------------------------------------
    # SAFETY EVENTS
    # --------------------------------------------------------

    def _emit(
        self,
        event_type: SafetyEventType,
        severity: SafetySeverity,
        message: str,
    ) -> None:
        self._events.append(
            SafetyEvent(
                event_id=generate_id(),
                sensor_id=self.config.sensor_id,
                robot_id=self.config.robot_id,
                tenant_id=self.config.tenant_id,
                event_type=event_type,
                severity=severity,
                message=message,
                timestamp=utc_now(),
            )
        )

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    def calibrate_zero(
        self,
        measurements: Sequence[ForceTorqueMeasurement],
        access: ForceAccessContext,
    ) -> SensorCalibration:
        self._authorize(
            access,
            "can_calibrate",
        )

        with self._lock:
            if self._emergency_stop:
                raise ForceStateError(
                    "Cannot calibrate during emergency stop"
                )

            for measurement in measurements:
                if measurement.sensor_id != self.config.sensor_id:
                    raise ForceCalibrationError(
                        "Calibration sensor mismatch"
                    )

            self._status = SensorStatus.CALIBRATING

            try:
                calibration = (
                    ForceSensorCalibrator.calculate_zero_offsets(
                        measurements
                    )
                )

                self._calibration = calibration

                self._filter.reset()
                self._contact_detector.reset()

                self._history.clear()
                self._last_raw = None

                self._status = SensorStatus.READY

                return calibration

            except Exception:
                self._status = SensorStatus.FAULT
                raise

    # --------------------------------------------------------
    # PROCESS FORCE/TORQUE MEASUREMENT
    # --------------------------------------------------------

    def process_measurement(
        self,
        measurement: ForceTorqueMeasurement,
        access: ForceAccessContext,
    ) -> ProcessedForceMeasurement:
        self._authorize(
            access,
            "can_process",
        )

        if measurement.sensor_id != self.config.sensor_id:
            raise ForceValidationError(
                "Measurement sensor ID mismatch"
            )

        with self._lock:
            if self._emergency_stop:
                raise ForceSafetyError(
                    "Emergency stop active"
                )

            if self._status in {
                SensorStatus.FAULT,
                SensorStatus.DISABLED,
                SensorStatus.CALIBRATING,
            }:
                raise ForceStateError(
                    "Sensor is not ready"
                )

            timestamp = ensure_utc(
                measurement.timestamp
            )

            now = utc_now()

            if (
                timestamp
                > now + timedelta(seconds=1)
            ):
                raise ForceValidationError(
                    "Measurement timestamp is in the future"
                )

            age = (
                now - timestamp
            ).total_seconds()

            if age > self.config.maximum_sample_age_seconds:
                self._status = SensorStatus.WARNING

                self._emit(
                    SafetyEventType.STALE_SENSOR,
                    SafetySeverity.WARNING,
                    "Stale force sensor measurement",
                )

                raise ForceSafetyError(
                    "Measurement exceeds maximum sample age"
                )

            if (
                self._last_raw is not None
                and timestamp <= self._last_raw.timestamp
            ):
                raise ForceValidationError(
                    "Measurement timestamps must increase"
                )

            raw_force = vector3(
                measurement.force,
                "raw force",
            )

            raw_torque = vector3(
                measurement.torque,
                "raw torque",
            )

            saturated = bool(
                np.any(
                    np.abs(raw_force)
                    >= self.config.maximum_force_axis
                )
                or np.any(
                    np.abs(raw_torque)
                    >= self.config.maximum_torque_axis
                )
            )

            if saturated:
                self._saturation_detected = True
                self._status = SensorStatus.FAULT
                self._emergency_stop = True

                self._emit(
                    SafetyEventType.SENSOR_SATURATION,
                    SafetySeverity.CRITICAL,
                    "Force/torque sensor saturation detected",
                )

                self._emit(
                    SafetyEventType.EMERGENCY_STOP,
                    SafetySeverity.CRITICAL,
                    "Software emergency-stop advisory activated",
                )

                raise ForceSafetyError(
                    "Sensor saturation detected"
                )

            compensated_force, compensated_torque = (
                ForceSensorCalibrator.compensate(
                    measurement,
                    self._calibration,
                )
            )

            compensated = np.concatenate(
                [
                    compensated_force,
                    compensated_torque,
                ]
            )

            filtered = self._filter.process(
                compensated
            )

            filtered_force = filtered[:3]
            filtered_torque = filtered[3:]

            force_norm = float(
                np.linalg.norm(
                    filtered_force
                )
            )

            torque_norm = float(
                np.linalg.norm(
                    filtered_torque
                )
            )

            previous_contact = (
                self._contact_detector.state
            )

            contact_state = (
                self._contact_detector.update(
                    filtered_force
                )
            )

            force_rate = 0.0

            if self._history:
                previous = self._history[-1]

                dt = (
                    timestamp - previous.timestamp
                ).total_seconds()

                if dt > 0:
                    force_rate = (
                        force_norm
                        - previous.force_magnitude
                    ) / dt

            processed = ProcessedForceMeasurement(
                measurement_id=measurement.measurement_id,
                sensor_id=measurement.sensor_id,
                raw_force=tuple(
                    float(value)
                    for value in raw_force
                ),
                raw_torque=tuple(
                    float(value)
                    for value in raw_torque
                ),
                compensated_force=tuple(
                    float(value)
                    for value in compensated_force
                ),
                compensated_torque=tuple(
                    float(value)
                    for value in compensated_torque
                ),
                filtered_force=tuple(
                    float(value)
                    for value in filtered_force
                ),
                filtered_torque=tuple(
                    float(value)
                    for value in filtered_torque
                ),
                force_magnitude=force_norm,
                torque_magnitude=torque_norm,
                force_rate=float(force_rate),
                contact_state=contact_state,
                timestamp=timestamp,
            )

            self._history.append(
                processed
            )

            self._last_raw = measurement

            if (
                previous_contact == ContactState.NO_CONTACT
                and contact_state != ContactState.NO_CONTACT
            ):
                self._emit(
                    SafetyEventType.CONTACT_DETECTED,
                    SafetySeverity.INFO,
                    "Contact detected",
                )

            elif (
                previous_contact != ContactState.NO_CONTACT
                and contact_state == ContactState.NO_CONTACT
            ):
                self._emit(
                    SafetyEventType.CONTACT_RELEASED,
                    SafetySeverity.INFO,
                    "Contact released",
                )

            critical_force = (
                force_norm
                >= self.config.critical_force_threshold
            )

            critical_torque = (
                torque_norm
                >= self.config.critical_torque_threshold
            )

            if critical_force:
                self._emit(
                    SafetyEventType.HIGH_FORCE,
                    SafetySeverity.CRITICAL,
                    "Critical force threshold exceeded",
                )

            if critical_torque:
                self._emit(
                    SafetyEventType.HIGH_TORQUE,
                    SafetySeverity.CRITICAL,
                    "Critical torque threshold exceeded",
                )

            if critical_force or critical_torque:
                self._emergency_stop = True
                self._status = SensorStatus.FAULT

                self._emit(
                    SafetyEventType.EMERGENCY_STOP,
                    SafetySeverity.CRITICAL,
                    "Software emergency-stop advisory activated",
                )

            elif (
                force_norm
                >= self.config.warning_force_threshold
                or torque_norm
                >= self.config.warning_torque_threshold
            ):
                self._status = SensorStatus.WARNING

            else:
                self._status = SensorStatus.READY

            return processed

    # --------------------------------------------------------
    # CONTACT REPORT
    # --------------------------------------------------------

    def contact_report(
        self,
        access: ForceAccessContext,
    ) -> ContactReport:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            if not self._history:
                return ContactReport(
                    sensor_id=self.config.sensor_id,
                    contact_state=ContactState.NO_CONTACT,
                    contact_detected=False,
                    force_magnitude=0.0,
                    contact_direction=(0.0, 0.0, 0.0),
                    timestamp=utc_now(),
                )

            latest = self._history[-1]

            return ContactReport(
                sensor_id=self.config.sensor_id,
                contact_state=latest.contact_state,
                contact_detected=(
                    latest.contact_state
                    != ContactState.NO_CONTACT
                ),
                force_magnitude=latest.force_magnitude,
                contact_direction=ForceContactDetector.direction(
                    latest.filtered_force
                ),
                timestamp=latest.timestamp,
            )

    # --------------------------------------------------------
    # SENSOR HEALTH
    # --------------------------------------------------------

    def health_report(
        self,
        access: ForceAccessContext,
    ) -> SensorHealthReport:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            age = None

            if self._last_raw is not None:
                age = max(
                    0.0,
                    (
                        utc_now()
                        - self._last_raw.timestamp
                    ).total_seconds(),
                )

            if (
                self._emergency_stop
                or self._status == SensorStatus.FAULT
            ):
                health = SensorHealth.CRITICAL

            elif (
                self._status == SensorStatus.WARNING
                or (
                    age is not None
                    and age
                    > self.config.maximum_sample_age_seconds
                )
            ):
                health = SensorHealth.WARNING

            elif not self._history:
                health = SensorHealth.UNKNOWN

            else:
                health = SensorHealth.HEALTHY

            return SensorHealthReport(
                sensor_id=self.config.sensor_id,
                health=health,
                status=self._status,
                sample_count=len(self._history),
                last_sample_age_seconds=age,
                saturation_detected=self._saturation_detected,
                emergency_stop_active=self._emergency_stop,
                evaluated_at=utc_now(),
            )

    # --------------------------------------------------------
    # EMERGENCY STOP
    # --------------------------------------------------------

    def emergency_stop(
        self,
        access: ForceAccessContext,
    ) -> None:
        self._authorize(
            access,
            "can_manage",
        )

        with self._lock:
            self._emergency_stop = True
            self._status = SensorStatus.FAULT

            self._emit(
                SafetyEventType.EMERGENCY_STOP,
                SafetySeverity.CRITICAL,
                "Manual software emergency-stop advisory",
            )

    def reset_emergency_stop(
        self,
        access: ForceAccessContext,
    ) -> None:
        self._authorize(
            access,
            "can_manage",
        )

        with self._lock:
            self._emergency_stop = False
            self._saturation_detected = False

            self._status = SensorStatus.READY

            self._filter.reset()
            self._contact_detector.reset()

            self._history.clear()
            self._last_raw = None

            self._emit(
                SafetyEventType.EMERGENCY_RESET,
                SafetySeverity.INFO,
                "Software emergency-stop state reset",
            )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    def measurement_history(
        self,
        access: ForceAccessContext,
        *,
        limit: int = 100,
    ) -> tuple[ProcessedForceMeasurement, ...]:
        self._authorize(
            access,
            "can_read",
        )

        if limit < 1:
            raise ForceValidationError(
                "limit must be positive"
            )

        with self._lock:
            return tuple(
                list(self._history)[-limit:]
            )

    # --------------------------------------------------------
    # SAFETY EVENTS
    # --------------------------------------------------------

    def safety_events(
        self,
        access: ForceAccessContext,
    ) -> tuple[SafetyEvent, ...]:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            return tuple(
                self._events
            )

    # --------------------------------------------------------
    # FEEDBACK STATE
    # --------------------------------------------------------

    def get_state(
        self,
        access: ForceAccessContext,
    ) -> ForceFeedbackState:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            latest = (
                self._history[-1]
                if self._history
                else None
            )

            return ForceFeedbackState(
                sensor_id=self.config.sensor_id,
                robot_id=self.config.robot_id,
                tenant_id=self.config.tenant_id,
                mode=self._mode,
                status=self._status,
                contact_state=self._contact_detector.state,
                emergency_stop_active=self._emergency_stop,
                latest_measurement=latest,
                calibration=self._calibration,
                timestamp=utc_now(),
            )

    # --------------------------------------------------------
    # EXPORT REPORT
    # --------------------------------------------------------

    def export_report_json(
        self,
        access: ForceAccessContext,
    ) -> str:
        state = self.get_state(
            access
        )

        health = self.health_report(
            access
        )

        events = self.safety_events(
            access
        )

        return json.dumps(
            json_safe(
                {
                    "state": state,
                    "health": health,
                    "events": events,
                    "simulation_only": True,
                    "hardware_actuation": False,
                }
            ),
            indent=2,
            allow_nan=False,
        )


# ============================================================
# MULTI-TENANT FORCE FEEDBACK SERVICE
# ============================================================

class ForceFeedbackService:

    def __init__(self) -> None:
        self._controllers: dict[
            tuple[str, str],
            ForceFeedbackController,
        ] = {}

        self._lock = threading.RLock()

    def register_sensor(
        self,
        config: ForceSensorConfig,
        access: ForceAccessContext,
    ) -> ForceFeedbackController:
        if (
            not access.can_manage
            or access.tenant_id != config.tenant_id
        ):
            raise ForceAuthorizationError(
                "Not authorized to register sensor"
            )

        key = (
            config.tenant_id,
            config.sensor_id,
        )

        with self._lock:
            if key in self._controllers:
                raise ForceStateError(
                    "Sensor already registered"
                )

            controller = ForceFeedbackController(
                config
            )

            self._controllers[key] = controller

            return controller

    def get_sensor(
        self,
        sensor_id: str,
        access: ForceAccessContext,
    ) -> ForceFeedbackController:
        if not access.can_read:
            raise ForceAuthorizationError(
                "Read permission required"
            )

        with self._lock:
            controller = self._controllers.get(
                (
                    access.tenant_id,
                    sensor_id,
                )
            )

        if controller is None:
            raise ForceSensorNotFoundError(
                "Sensor not found"
            )

        return controller

    def list_sensors(
        self,
        access: ForceAccessContext,
    ) -> tuple[str, ...]:
        if not access.can_read:
            raise ForceAuthorizationError(
                "Read permission required"
            )

        with self._lock:
            return tuple(
                sensor_id
                for tenant_id, sensor_id
                in self._controllers
                if tenant_id == access.tenant_id
            )


# ============================================================
# SYNTHETIC FORCE SENSOR FACTORY
# ============================================================

class SyntheticForceFeedbackFactory:

    @staticmethod
    def config(
        *,
        sensor_id: str = "force-sensor-001",
        robot_id: str = "robot-arm-001",
        tenant_id: str = "physical_ai_lab",
    ) -> ForceSensorConfig:
        return ForceSensorConfig(
            sensor_id=sensor_id,
            robot_id=robot_id,
            tenant_id=tenant_id,
            maximum_force_axis=500.0,
            maximum_torque_axis=100.0,
            contact_enter_threshold=5.0,
            contact_exit_threshold=3.0,
            warning_force_threshold=80.0,
            critical_force_threshold=150.0,
            warning_torque_threshold=20.0,
            critical_torque_threshold=40.0,
            filter_type=FilterType.LOW_PASS,
            filter_alpha=0.5,
        )

    @staticmethod
    def access(
        tenant_id: str = "physical_ai_lab",
    ) -> ForceAccessContext:
        return ForceAccessContext(
            tenant_id=tenant_id,
            user_id="robotics_engineer",
            can_read=True,
            can_process=True,
            can_calibrate=True,
            can_manage=True,
        )

    @staticmethod
    def measurement(
        *,
        sensor_id: str = "force-sensor-001",
        force: tuple[float, float, float] = (
            0.0,
            0.0,
            0.0,
        ),
        torque: tuple[float, float, float] = (
            0.0,
            0.0,
            0.0,
        ),
        timestamp: datetime | None = None,
    ) -> ForceTorqueMeasurement:
        return ForceTorqueMeasurement(
            measurement_id=generate_id(),
            sensor_id=sensor_id,
            force=force,
            torque=torque,
            timestamp=timestamp or utc_now(),
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestForceFeedback(unittest.TestCase):

    def setUp(self) -> None:
        self.config = (
            SyntheticForceFeedbackFactory.config()
        )

        self.access = (
            SyntheticForceFeedbackFactory.access()
        )

        self.controller = ForceFeedbackController(
            self.config
        )

    def test_initial_state(self) -> None:
        state = self.controller.get_state(
            self.access
        )

        self.assertEqual(
            state.status,
            SensorStatus.READY,
        )

    def test_force_measurement(self) -> None:
        measurement = (
            SyntheticForceFeedbackFactory.measurement(
                force=(2.0, 1.0, 0.0)
            )
        )

        processed = self.controller.process_measurement(
            measurement,
            self.access,
        )

        self.assertGreater(
            processed.force_magnitude,
            0.0,
        )

    def test_force_magnitude(self) -> None:
        self.assertAlmostEqual(
            magnitude((3.0, 4.0, 0.0)),
            5.0,
        )

    def test_contact_detection(self) -> None:
        measurement = (
            SyntheticForceFeedbackFactory.measurement(
                force=(10.0, 0.0, 0.0)
            )
        )

        processed = self.controller.process_measurement(
            measurement,
            self.access,
        )

        self.assertEqual(
            processed.contact_state,
            ContactState.CONTACT,
        )

    def test_no_contact(self) -> None:
        measurement = (
            SyntheticForceFeedbackFactory.measurement(
                force=(1.0, 0.0, 0.0)
            )
        )

        processed = self.controller.process_measurement(
            measurement,
            self.access,
        )

        self.assertEqual(
            processed.contact_state,
            ContactState.NO_CONTACT,
        )

    def test_contact_direction(self) -> None:
        direction = ForceContactDetector.direction(
            (3.0, 4.0, 0.0)
        )

        self.assertAlmostEqual(
            direction[0],
            0.6,
        )

        self.assertAlmostEqual(
            direction[1],
            0.8,
        )

    def test_calibration(self) -> None:
        measurements = [
            SyntheticForceFeedbackFactory.measurement(
                force=(1.0, 2.0, 3.0),
                torque=(0.1, 0.2, 0.3),
            )
            for _ in range(5)
        ]

        calibration = self.controller.calibrate_zero(
            measurements,
            self.access,
        )

        self.assertEqual(
            calibration.sample_count,
            5,
        )

        self.assertAlmostEqual(
            calibration.force_offset[0],
            1.0,
        )

    def test_low_pass_filter(self) -> None:
        signal_filter = ForceSignalFilter(
            FilterType.LOW_PASS,
            alpha=0.5,
        )

        signal_filter.process(
            (0, 0, 0, 0, 0, 0)
        )

        result = signal_filter.process(
            (10, 0, 0, 0, 0, 0)
        )

        self.assertAlmostEqual(
            result[0],
            5.0,
        )

    def test_median_filter(self) -> None:
        signal_filter = ForceSignalFilter(
            FilterType.MEDIAN,
            median_window=3,
        )

        signal_filter.process(
            (1, 0, 0, 0, 0, 0)
        )

        signal_filter.process(
            (100, 0, 0, 0, 0, 0)
        )

        result = signal_filter.process(
            (2, 0, 0, 0, 0, 0)
        )

        self.assertAlmostEqual(
            result[0],
            2.0,
        )

    def test_gravity_compensation(self) -> None:
        force, torque = GravityCompensator.compensate(
            measured_force=(0.0, 0.0, -9.80665),
            measured_torque=(0.0, 0.0, 0.0),
            payload_mass_kg=1.0,
            rotation_sensor_to_world=np.eye(3),
        )

        self.assertTrue(
            np.allclose(
                force,
                np.zeros(3),
            )
        )

        self.assertTrue(
            np.allclose(
                torque,
                np.zeros(3),
            )
        )

    def test_impedance_response(self) -> None:
        model = ImpedanceFeedbackModel()

        result = model.calculate(
            desired_position=(0.1, 0.0, 0.0),
            actual_position=(0.0, 0.0, 0.0),
            desired_velocity=(0.0, 0.0, 0.0),
            actual_velocity=(0.0, 0.0, 0.0),
        )

        self.assertGreater(
            result.response_magnitude,
            0.0,
        )

    def test_admittance_response(self) -> None:
        model = AdmittanceFeedbackModel()

        result = model.update(
            external_force=(10.0, 0.0, 0.0),
            dt=0.01,
        )

        self.assertGreater(
            result.displacement[0],
            0.0,
        )

    def test_high_force_emergency(self) -> None:
        measurement = (
            SyntheticForceFeedbackFactory.measurement(
                force=(200.0, 0.0, 0.0)
            )
        )

        self.controller.process_measurement(
            measurement,
            self.access,
        )

        state = self.controller.get_state(
            self.access
        )

        self.assertTrue(
            state.emergency_stop_active
        )

    def test_sensor_saturation(self) -> None:
        measurement = (
            SyntheticForceFeedbackFactory.measurement(
                force=(600.0, 0.0, 0.0)
            )
        )

        with self.assertRaises(
            ForceSafetyError
        ):
            self.controller.process_measurement(
                measurement,
                self.access,
            )

    def test_manual_emergency_stop(self) -> None:
        self.controller.emergency_stop(
            self.access
        )

        self.assertTrue(
            self.controller.get_state(
                self.access
            ).emergency_stop_active
        )

    def test_emergency_reset(self) -> None:
        self.controller.emergency_stop(
            self.access
        )

        self.controller.reset_emergency_stop(
            self.access
        )

        self.assertFalse(
            self.controller.get_state(
                self.access
            ).emergency_stop_active
        )

    def test_health_report(self) -> None:
        report = self.controller.health_report(
            self.access
        )

        self.assertEqual(
            report.sensor_id,
            "force-sensor-001",
        )

    def test_measurement_history(self) -> None:
        measurement = (
            SyntheticForceFeedbackFactory.measurement(
                force=(2.0, 0.0, 0.0)
            )
        )

        self.controller.process_measurement(
            measurement,
            self.access,
        )

        history = self.controller.measurement_history(
            self.access
        )

        self.assertEqual(
            len(history),
            1,
        )

    def test_tenant_isolation(self) -> None:
        other = ForceAccessContext(
            tenant_id="other_tenant",
            user_id="other_user",
            can_read=True,
        )

        with self.assertRaises(
            ForceAuthorizationError
        ):
            self.controller.get_state(
                other
            )

    def test_permission_denied(self) -> None:
        viewer = ForceAccessContext(
            tenant_id="physical_ai_lab",
            user_id="viewer",
            can_read=True,
        )

        measurement = (
            SyntheticForceFeedbackFactory.measurement()
        )

        with self.assertRaises(
            ForceAuthorizationError
        ):
            self.controller.process_measurement(
                measurement,
                viewer,
            )

    def test_service_registration(self) -> None:
        service = ForceFeedbackService()

        service.register_sensor(
            self.config,
            self.access,
        )

        self.assertEqual(
            service.list_sensors(
                self.access
            ),
            ("force-sensor-001",),
        )

    def test_json_export(self) -> None:
        report = json.loads(
            self.controller.export_report_json(
                self.access
            )
        )

        self.assertTrue(
            report["simulation_only"]
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    config = SyntheticForceFeedbackFactory.config(
        sensor_id="robot-arm-force-001",
        robot_id="industrial-arm-001",
        tenant_id="physical_ai_lab",
    )

    access = SyntheticForceFeedbackFactory.access(
        tenant_id="physical_ai_lab"
    )

    service = ForceFeedbackService()

    controller = service.register_sensor(
        config,
        access,
    )

    print("\nPHYSICAL AI — FORCE FEEDBACK")
    print("=" * 70)

    forces = [
        (0.5, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (8.0, 1.0, 0.0),
        (15.0, 2.0, 0.0),
        (30.0, 4.0, 0.0),
        (10.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
    ]

    start = utc_now()

    for index, force in enumerate(forces):
        measurement = (
            SyntheticForceFeedbackFactory.measurement(
                sensor_id=config.sensor_id,
                force=force,
                torque=(0.1, 0.0, 0.0),
                timestamp=start
                + timedelta(milliseconds=index),
            )
        )

        processed = controller.process_measurement(
            measurement,
            access,
        )

        print(
            f"Sample {index + 1}: "
            f"Force={processed.force_magnitude:.2f} N, "
            f"Contact={processed.contact_state.value}"
        )

    impedance = ImpedanceFeedbackModel()

    impedance_response = impedance.calculate(
        desired_position=(0.1, 0.0, 0.0),
        actual_position=(0.08, 0.0, 0.0),
        desired_velocity=(0.0, 0.0, 0.0),
        actual_velocity=(0.01, 0.0, 0.0),
    )

    print(
        "\nVirtual impedance response:",
        impedance_response.virtual_response_force,
    )

    admittance = AdmittanceFeedbackModel()

    admittance_response = admittance.update(
        external_force=(5.0, 0.0, 0.0),
        dt=0.01,
    )

    print(
        "Virtual admittance displacement:",
        admittance_response.displacement,
    )

    print("\nSENSOR REPORT")
    print("=" * 70)

    print(
        controller.export_report_json(
            access
        )
    )

    print("\nRUNNING UNIT TESTS")
    print("=" * 70)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestForceFeedback
    )

    result = unittest.TextTestRunner(
        verbosity=2
    ).run(suite)

    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
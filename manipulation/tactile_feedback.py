"""Tactile Feedback module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/manipulation/tactile_feedback.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================
Module: Tactile Feedback

Simulation-only tactile sensing, contact analysis, grasp
stability estimation, slip detection, and manipulation feedback.

Features:
    - Multi-finger tactile sensor support
    - Pressure-array processing
    - Normal and shear force estimation
    - Contact detection with hysteresis
    - Contact centroid estimation
    - Contact area calculation
    - Pressure distribution analysis
    - Center-of-pressure tracking
    - Slip detection using temporal features
    - Grip stability scoring
    - Contact quality evaluation
    - Overpressure detection
    - Sensor noise filtering
    - Sensor calibration
    - Sensor health monitoring
    - Adaptive grip-force recommendations
    - Pick-and-place feedback integration
    - Thread-safe multi-tenant services
    - Audit history
    - JSON export
    - Synthetic tactile data
    - Unit tests

Requirements:
    Python >= 3.11
    numpy >= 1.24

Install:
    pip install numpy

Safety:
    Advisory and simulation only.
    This module never actuates a physical gripper.
    Slip detection and grip-force recommendations are
    heuristic estimates, not safety-certified measurements.
"""

from __future__ import annotations

import json
import math
import threading
import unittest
import uuid

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Sequence

import numpy as np


# ============================================================
# EXCEPTIONS
# ============================================================

class TactileFeedbackError(Exception):
    pass


class TactileValidationError(TactileFeedbackError):
    pass


class TactileAuthorizationError(TactileFeedbackError):
    pass


class TactileNotFoundError(TactileFeedbackError):
    pass


class TactileConflictError(TactileFeedbackError):
    pass


class TactileCalibrationError(TactileFeedbackError):
    pass


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


def identifier(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 256
    ):
        raise TactileValidationError(
            f"Invalid {name}"
        )

    return value.strip()


def finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TactileValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TactileValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise TactileValidationError(
            f"{name} must be finite"
        )

    return result


def positive(value: Any, name: str) -> float:
    result = finite(value, name)

    if result <= 0:
        raise TactileValidationError(
            f"{name} must be positive"
        )

    return result


def nonnegative(value: Any, name: str) -> float:
    result = finite(value, name)

    if result < 0:
        raise TactileValidationError(
            f"{name} cannot be negative"
        )

    return result


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(minimum, min(maximum, value))


def array2d(
    values: Sequence[Sequence[float]],
    name: str,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TactileValidationError(
            f"{name} must be a numeric matrix"
        ) from exc

    if result.ndim != 2:
        raise TactileValidationError(
            f"{name} must be two-dimensional"
        )

    if result.size == 0:
        raise TactileValidationError(
            f"{name} cannot be empty"
        )

    if shape is not None and result.shape != shape:
        raise TactileValidationError(
            f"{name} must have shape {shape}"
        )

    if not np.all(np.isfinite(result)):
        raise TactileValidationError(
            f"{name} contains non-finite values"
        )

    return result


def vector2(
    values: Sequence[float],
    name: str,
) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TactileValidationError(
            f"{name} must be numeric"
        ) from exc

    if result.shape != (2,):
        raise TactileValidationError(
            f"{name} must contain two values"
        )

    if not np.all(np.isfinite(result)):
        raise TactileValidationError(
            f"{name} contains non-finite values"
        )

    return result


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, np.ndarray):
        return value.tolist()

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

    if isinstance(value, (tuple, list, deque)):
        return [json_safe(item) for item in value]

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


# ============================================================
# ENUMERATIONS
# ============================================================

class FingerName(str, Enum):
    THUMB = "thumb"
    INDEX = "index"
    MIDDLE = "middle"
    RING = "ring"
    LITTLE = "little"
    LEFT_JAW = "left_jaw"
    RIGHT_JAW = "right_jaw"


class ContactState(str, Enum):
    NO_CONTACT = "no_contact"
    LIGHT_CONTACT = "light_contact"
    STABLE_CONTACT = "stable_contact"
    HIGH_PRESSURE = "high_pressure"


class SlipState(str, Enum):
    NO_SLIP = "no_slip"
    POSSIBLE_SLIP = "possible_slip"
    SLIP_DETECTED = "slip_detected"
    INSUFFICIENT_HISTORY = "insufficient_history"


class GripStability(str, Enum):
    STABLE = "stable"
    MARGINAL = "marginal"
    UNSTABLE = "unstable"
    NO_GRASP = "no_grasp"


class SensorHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAULT = "fault"


class FeedbackSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class GripRecommendation(str, Enum):
    HOLD = "hold"
    INCREASE_GRIP = "increase_grip"
    REDUCE_GRIP = "reduce_grip"
    REPOSITION = "reposition"
    STOP_AND_INSPECT = "stop_and_inspect"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class TactileSensorConfig:
    sensor_id: str
    finger: FingerName

    rows: int = 8
    columns: int = 8

    taxel_width_m: float = 0.005
    taxel_height_m: float = 0.005

    maximum_pressure_kpa: float = 250.0

    pressure_gain: float = 1.0

    def __post_init__(self) -> None:
        identifier(self.sensor_id, "sensor_id")

        if self.rows < 1 or self.columns < 1:
            raise TactileValidationError(
                "Sensor dimensions must be positive"
            )

        positive(
            self.taxel_width_m,
            "taxel_width_m",
        )

        positive(
            self.taxel_height_m,
            "taxel_height_m",
        )

        positive(
            self.maximum_pressure_kpa,
            "maximum_pressure_kpa",
        )

        positive(
            self.pressure_gain,
            "pressure_gain",
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, self.columns

    @property
    def taxel_area_m2(self) -> float:
        return (
            self.taxel_width_m
            * self.taxel_height_m
        )


@dataclass(frozen=True)
class TactileFeedbackConfig:
    contact_on_kpa: float = 8.0
    contact_off_kpa: float = 4.0

    light_contact_kpa: float = 12.0
    high_pressure_kpa: float = 180.0

    minimum_contact_taxels: int = 2

    smoothing_alpha: float = 0.4

    slip_centroid_speed_mps: float = 0.015
    slip_shear_ratio_threshold: float = 0.75

    slip_force_drop_fraction: float = 0.25

    minimum_slip_history: int = 3

    history_size: int = 100

    minimum_stable_force_n: float = 2.0
    target_grip_force_n: float = 12.0
    maximum_recommended_force_n: float = 60.0

    pressure_saturation_fraction: float = 0.98

    maximum_sample_age_s: float = 2.0

    def __post_init__(self) -> None:
        for name in (
            "contact_on_kpa",
            "contact_off_kpa",
            "light_contact_kpa",
            "high_pressure_kpa",
            "slip_centroid_speed_mps",
            "slip_shear_ratio_threshold",
            "slip_force_drop_fraction",
            "minimum_stable_force_n",
            "target_grip_force_n",
            "maximum_recommended_force_n",
            "maximum_sample_age_s",
        ):
            nonnegative(getattr(self, name), name)

        if self.contact_off_kpa >= self.contact_on_kpa:
            raise TactileValidationError(
                "Contact-off threshold must be below contact-on"
            )

        if not 0 < self.smoothing_alpha <= 1:
            raise TactileValidationError(
                "smoothing_alpha must be in (0,1]"
            )

        if not 0 < self.pressure_saturation_fraction <= 1:
            raise TactileValidationError(
                "Invalid pressure saturation fraction"
            )

        if not 0 <= self.slip_force_drop_fraction <= 1:
            raise TactileValidationError(
                "Invalid force-drop fraction"
            )

        if self.minimum_contact_taxels < 1:
            raise TactileValidationError(
                "minimum_contact_taxels must be positive"
            )

        if self.minimum_slip_history < 2:
            raise TactileValidationError(
                "minimum_slip_history must be >= 2"
            )

        if self.history_size < self.minimum_slip_history:
            raise TactileValidationError(
                "History size is too small"
            )

        if (
            self.maximum_recommended_force_n
            < self.target_grip_force_n
        ):
            raise TactileValidationError(
                "Maximum recommended force below target"
            )


# ============================================================
# SENSOR DATA
# ============================================================

@dataclass(frozen=True)
class TactileCalibration:
    sensor_id: str

    baseline_pressure_kpa: tuple[tuple[float, ...], ...]

    calibration_id: str = field(
        default_factory=new_id
    )

    calibrated_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        identifier(self.sensor_id, "sensor_id")
        identifier(self.calibration_id, "calibration_id")

        array2d(
            self.baseline_pressure_kpa,
            "baseline_pressure_kpa",
        )


@dataclass(frozen=True)
class TactileSample:
    sample_id: str
    sensor_id: str

    timestamp: datetime

    pressure_kpa: tuple[tuple[float, ...], ...]

    shear_force_xy_n: tuple[float, float] = (
        0.0,
        0.0,
    )

    temperature_c: float = 25.0

    def __post_init__(self) -> None:
        identifier(self.sample_id, "sample_id")
        identifier(self.sensor_id, "sensor_id")

        if self.timestamp.tzinfo is None:
            raise TactileValidationError(
                "Sample timestamp must be timezone-aware"
            )

        array2d(
            self.pressure_kpa,
            "pressure_kpa",
        )

        vector2(
            self.shear_force_xy_n,
            "shear_force_xy_n",
        )

        finite(
            self.temperature_c,
            "temperature_c",
        )


# ============================================================
# ANALYSIS RESULTS
# ============================================================

@dataclass(frozen=True)
class ContactAnalysis:
    sensor_id: str

    contact_state: ContactState

    contact_detected: bool

    active_taxels: int

    contact_area_m2: float

    mean_pressure_kpa: float
    peak_pressure_kpa: float

    normal_force_n: float
    shear_force_n: float

    centroid_xy_m: tuple[float, float] | None

    pressure_uniformity: float

    timestamp: datetime


@dataclass(frozen=True)
class SlipAnalysis:
    sensor_id: str

    state: SlipState

    slip_score: float

    centroid_speed_mps: float

    shear_to_normal_ratio: float

    force_drop_fraction: float

    evidence: tuple[str, ...]


@dataclass(frozen=True)
class SensorHealthReport:
    sensor_id: str

    health: SensorHealth

    saturated_taxels: int

    saturation_fraction: float

    sample_age_s: float

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class GripForceAdvice:
    recommendation: GripRecommendation

    current_force_n: float

    target_force_n: float

    suggested_adjustment_n: float

    confidence: float

    explanation: str


@dataclass(frozen=True)
class TactileFeedbackReport:
    report_id: str

    tenant_id: str
    robot_id: str
    sensor_id: str

    contact: ContactAnalysis
    slip: SlipAnalysis
    health: SensorHealthReport

    stability: GripStability

    stability_score: float

    force_advice: GripForceAdvice

    severity: FeedbackSeverity

    created_at: datetime


@dataclass(frozen=True)
class MultiFingerGraspReport:
    report_id: str

    tenant_id: str
    robot_id: str

    finger_reports: tuple[TactileFeedbackReport, ...]

    total_normal_force_n: float
    total_shear_force_n: float

    contacting_fingers: int

    stability: GripStability

    stability_score: float

    recommendations: tuple[str, ...]

    created_at: datetime


@dataclass(frozen=True)
class TactileAccessContext:
    tenant_id: str
    user_id: str

    can_read: bool = False
    can_process: bool = False
    can_calibrate: bool = False
    can_manage: bool = False


# ============================================================
# SENSOR CALIBRATOR
# ============================================================

class TactileSensorCalibrator:

    @staticmethod
    def calibrate(
        config: TactileSensorConfig,
        samples: Sequence[TactileSample],
    ) -> TactileCalibration:
        if len(samples) < 3:
            raise TactileCalibrationError(
                "At least three baseline samples are required"
            )

        matrices: list[np.ndarray] = []

        for sample in samples:
            if sample.sensor_id != config.sensor_id:
                raise TactileCalibrationError(
                    "Sensor mismatch during calibration"
                )

            matrix = array2d(
                sample.pressure_kpa,
                "pressure_kpa",
                config.shape,
            )

            matrices.append(matrix)

        baseline = np.median(
            np.stack(matrices, axis=0),
            axis=0,
        )

        return TactileCalibration(
            sensor_id=config.sensor_id,
            baseline_pressure_kpa=tuple(
                tuple(float(value) for value in row)
                for row in baseline
            ),
        )


# ============================================================
# PRESSURE PREPROCESSOR
# ============================================================

class TactilePressurePreprocessor:

    def __init__(
        self,
        config: TactileSensorConfig,
        alpha: float,
    ) -> None:
        self.config = config
        self.alpha = alpha

        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        self._previous = None

    def process(
        self,
        sample: TactileSample,
        calibration: TactileCalibration | None,
    ) -> np.ndarray:
        pressure = array2d(
            sample.pressure_kpa,
            "pressure_kpa",
            self.config.shape,
        ).copy()

        if calibration is not None:
            if calibration.sensor_id != sample.sensor_id:
                raise TactileValidationError(
                    "Calibration sensor mismatch"
                )

            baseline = array2d(
                calibration.baseline_pressure_kpa,
                "baseline",
                self.config.shape,
            )

            pressure -= baseline

        pressure *= self.config.pressure_gain

        pressure = np.clip(
            pressure,
            0.0,
            self.config.maximum_pressure_kpa,
        )

        if self._previous is None:
            filtered = pressure
        else:
            filtered = (
                self.alpha * pressure
                + (1.0 - self.alpha) * self._previous
            )

        self._previous = filtered.copy()

        return filtered


# ============================================================
# CONTACT DETECTOR
# ============================================================

class TactileContactDetector:

    def __init__(
        self,
        sensor: TactileSensorConfig,
        config: TactileFeedbackConfig,
    ) -> None:
        self.sensor = sensor
        self.config = config

        self._contact_active = False

    def reset(self) -> None:
        self._contact_active = False

    def analyze(
        self,
        pressure: np.ndarray,
        shear_force_xy_n: Sequence[float],
        timestamp: datetime,
    ) -> ContactAnalysis:
        pressure = array2d(
            pressure,
            "pressure",
            self.sensor.shape,
        )

        shear = vector2(
            shear_force_xy_n,
            "shear_force_xy_n",
        )

        threshold = (
            self.config.contact_off_kpa
            if self._contact_active
            else self.config.contact_on_kpa
        )

        mask = pressure >= threshold

        active_taxels = int(
            np.count_nonzero(mask)
        )

        detected = (
            active_taxels
            >= self.config.minimum_contact_taxels
        )

        self._contact_active = detected

        if not detected:
            return ContactAnalysis(
                sensor_id=self.sensor.sensor_id,
                contact_state=ContactState.NO_CONTACT,
                contact_detected=False,
                active_taxels=0,
                contact_area_m2=0.0,
                mean_pressure_kpa=0.0,
                peak_pressure_kpa=0.0,
                normal_force_n=0.0,
                shear_force_n=float(
                    np.linalg.norm(shear)
                ),
                centroid_xy_m=None,
                pressure_uniformity=0.0,
                timestamp=timestamp,
            )

        active_pressure = pressure[mask]

        mean_pressure = float(
            np.mean(active_pressure)
        )

        peak_pressure = float(
            np.max(active_pressure)
        )

        contact_area = (
            active_taxels
            * self.sensor.taxel_area_m2
        )

        # kPa * m² * 1000 = Newtons.
        normal_force = (
            float(np.sum(active_pressure))
            * self.sensor.taxel_area_m2
            * 1000.0
        )

        row_indices, column_indices = np.indices(
            self.sensor.shape
        )

        x_positions = (
            column_indices + 0.5
        ) * self.sensor.taxel_width_m

        y_positions = (
            row_indices + 0.5
        ) * self.sensor.taxel_height_m

        weights = np.where(
            mask,
            pressure,
            0.0,
        )

        total_weight = float(
            np.sum(weights)
        )

        centroid_x = float(
            np.sum(weights * x_positions)
            / max(total_weight, 1e-12)
        )

        centroid_y = float(
            np.sum(weights * y_positions)
            / max(total_weight, 1e-12)
        )

        coefficient_of_variation = (
            float(np.std(active_pressure))
            / max(mean_pressure, 1e-12)
        )

        uniformity = clamp(
            1.0 / (1.0 + coefficient_of_variation)
        )

        if peak_pressure >= self.config.high_pressure_kpa:
            state = ContactState.HIGH_PRESSURE

        elif mean_pressure < self.config.light_contact_kpa:
            state = ContactState.LIGHT_CONTACT

        else:
            state = ContactState.STABLE_CONTACT

        return ContactAnalysis(
            sensor_id=self.sensor.sensor_id,
            contact_state=state,
            contact_detected=True,
            active_taxels=active_taxels,
            contact_area_m2=contact_area,
            mean_pressure_kpa=mean_pressure,
            peak_pressure_kpa=peak_pressure,
            normal_force_n=normal_force,
            shear_force_n=float(
                np.linalg.norm(shear)
            ),
            centroid_xy_m=(
                centroid_x,
                centroid_y,
            ),
            pressure_uniformity=uniformity,
            timestamp=timestamp,
        )


# ============================================================
# SLIP DETECTOR
# ============================================================

class TactileSlipDetector:

    def __init__(
        self,
        config: TactileFeedbackConfig,
    ) -> None:
        self.config = config

        self._history: deque[ContactAnalysis] = deque(
            maxlen=config.history_size
        )

    def reset(self) -> None:
        self._history.clear()

    def analyze(
        self,
        contact: ContactAnalysis,
    ) -> SlipAnalysis:
        previous = (
            self._history[-1]
            if self._history
            else None
        )

        centroid_speed = 0.0
        force_drop = 0.0

        if (
            previous is not None
            and previous.centroid_xy_m is not None
            and contact.centroid_xy_m is not None
        ):
            dt = (
                contact.timestamp - previous.timestamp
            ).total_seconds()

            if dt > 0:
                displacement = float(
                    np.linalg.norm(
                        vector2(
                            contact.centroid_xy_m,
                            "current centroid",
                        )
                        - vector2(
                            previous.centroid_xy_m,
                            "previous centroid",
                        )
                    )
                )

                centroid_speed = displacement / dt

                if previous.normal_force_n > 1e-9:
                    force_drop = clamp(
                        (
                            previous.normal_force_n
                            - contact.normal_force_n
                        )
                        / previous.normal_force_n
                    )

        shear_ratio = (
            contact.shear_force_n
            / max(contact.normal_force_n, 1e-9)
        )

        evidence: list[str] = []

        if (
            centroid_speed
            >= self.config.slip_centroid_speed_mps
        ):
            evidence.append(
                "Rapid contact-centroid movement"
            )

        if (
            shear_ratio
            >= self.config.slip_shear_ratio_threshold
        ):
            evidence.append(
                "High shear-to-normal force ratio"
            )

        if (
            force_drop
            >= self.config.slip_force_drop_fraction
        ):
            evidence.append(
                "Sudden normal-force reduction"
            )

        speed_score = clamp(
            centroid_speed
            / max(
                self.config.slip_centroid_speed_mps,
                1e-9,
            )
        )

        shear_score = clamp(
            shear_ratio
            / max(
                self.config.slip_shear_ratio_threshold,
                1e-9,
            )
        )

        drop_score = clamp(
            force_drop
            / max(
                self.config.slip_force_drop_fraction,
                1e-9,
            )
        )

        score = (
            0.40 * speed_score
            + 0.35 * shear_score
            + 0.25 * drop_score
        )

        self._history.append(contact)

        if not contact.contact_detected:
            state = SlipState.NO_SLIP
            score = 0.0

        elif (
            len(self._history)
            < self.config.minimum_slip_history
        ):
            state = SlipState.INSUFFICIENT_HISTORY

        elif score >= 0.75:
            state = SlipState.SLIP_DETECTED

        elif score >= 0.40:
            state = SlipState.POSSIBLE_SLIP

        else:
            state = SlipState.NO_SLIP

        return SlipAnalysis(
            sensor_id=contact.sensor_id,
            state=state,
            slip_score=clamp(score),
            centroid_speed_mps=centroid_speed,
            shear_to_normal_ratio=shear_ratio,
            force_drop_fraction=force_drop,
            evidence=tuple(evidence),
        )


# ============================================================
# SENSOR HEALTH MONITOR
# ============================================================

class TactileSensorHealthMonitor:

    def __init__(
        self,
        sensor: TactileSensorConfig,
        config: TactileFeedbackConfig,
    ) -> None:
        self.sensor = sensor
        self.config = config

    def analyze(
        self,
        sample: TactileSample,
        pressure: np.ndarray,
    ) -> SensorHealthReport:
        pressure = array2d(
            pressure,
            "pressure",
            self.sensor.shape,
        )

        saturation_threshold = (
            self.sensor.maximum_pressure_kpa
            * self.config.pressure_saturation_fraction
        )

        saturated_taxels = int(
            np.count_nonzero(
                pressure >= saturation_threshold
            )
        )

        saturation_fraction = (
            saturated_taxels / pressure.size
        )

        sample_age = max(
            0.0,
            (
                utc_now() - sample.timestamp
            ).total_seconds(),
        )

        warnings: list[str] = []

        if (
            sample_age
            > self.config.maximum_sample_age_s
        ):
            warnings.append(
                "Tactile sample is stale"
            )

        if saturation_fraction >= 0.5:
            warnings.append(
                "Large portion of pressure array saturated"
            )

        elif saturated_taxels > 0:
            warnings.append(
                "Some tactile taxels are saturated"
            )

        if (
            sample_age
            > self.config.maximum_sample_age_s
            or saturation_fraction >= 0.5
        ):
            health = SensorHealth.FAULT

        elif saturated_taxels > 0:
            health = SensorHealth.DEGRADED

        else:
            health = SensorHealth.HEALTHY

        return SensorHealthReport(
            sensor_id=self.sensor.sensor_id,
            health=health,
            saturated_taxels=saturated_taxels,
            saturation_fraction=saturation_fraction,
            sample_age_s=sample_age,
            warnings=tuple(warnings),
        )


# ============================================================
# GRIP STABILITY EVALUATOR
# ============================================================

class GripStabilityEvaluator:

    def __init__(
        self,
        config: TactileFeedbackConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        contact: ContactAnalysis,
        slip: SlipAnalysis,
        health: SensorHealthReport,
    ) -> tuple[GripStability, float]:
        if not contact.contact_detected:
            return GripStability.NO_GRASP, 0.0

        force_score = clamp(
            contact.normal_force_n
            / max(
                self.config.minimum_stable_force_n,
                1e-9,
            )
        )

        uniformity_score = (
            contact.pressure_uniformity
        )

        slip_score = (
            1.0 - slip.slip_score
        )

        health_score = {
            SensorHealth.HEALTHY: 1.0,
            SensorHealth.DEGRADED: 0.5,
            SensorHealth.FAULT: 0.0,
        }[health.health]

        score = (
            0.35 * force_score
            + 0.20 * uniformity_score
            + 0.30 * slip_score
            + 0.15 * health_score
        )

        if (
            slip.state == SlipState.SLIP_DETECTED
            or health.health == SensorHealth.FAULT
        ):
            return GripStability.UNSTABLE, min(score, 0.35)

        if score >= 0.75:
            return GripStability.STABLE, score

        if score >= 0.45:
            return GripStability.MARGINAL, score

        return GripStability.UNSTABLE, score


# ============================================================
# ADAPTIVE GRIP FORCE ADVISOR
# ============================================================

class AdaptiveGripForceAdvisor:

    def __init__(
        self,
        config: TactileFeedbackConfig,
    ) -> None:
        self.config = config

    def recommend(
        self,
        contact: ContactAnalysis,
        slip: SlipAnalysis,
        health: SensorHealthReport,
        stability: GripStability,
    ) -> GripForceAdvice:
        current_force = contact.normal_force_n

        target_force = self.config.target_grip_force_n

        if health.health == SensorHealth.FAULT:
            recommendation = (
                GripRecommendation.STOP_AND_INSPECT
            )

            target_force = current_force

            confidence = 0.95

            explanation = (
                "Sensor health is unreliable; "
                "do not increase physical grip force."
            )

        elif (
            contact.contact_state
            == ContactState.HIGH_PRESSURE
        ):
            recommendation = (
                GripRecommendation.REDUCE_GRIP
            )

            target_force = min(
                current_force,
                self.config.target_grip_force_n,
            )

            confidence = 0.90

            explanation = (
                "Excessive contact pressure detected."
            )

        elif not contact.contact_detected:
            recommendation = (
                GripRecommendation.REPOSITION
            )

            target_force = current_force

            confidence = 0.85

            explanation = (
                "No valid contact detected."
            )

        elif slip.state == SlipState.SLIP_DETECTED:
            recommendation = (
                GripRecommendation.INCREASE_GRIP
            )

            target_force = min(
                max(
                    self.config.target_grip_force_n,
                    current_force * 1.15,
                ),
                self.config.maximum_recommended_force_n,
            )

            confidence = 0.75

            explanation = (
                "Slip indicators suggest insufficient "
                "grasp stability. Revalidate contact "
                "before any force adjustment."
            )

        elif stability == GripStability.UNSTABLE:
            recommendation = (
                GripRecommendation.REPOSITION
            )

            target_force = current_force

            confidence = 0.80

            explanation = (
                "Grasp instability may require a new "
                "contact configuration."
            )

        elif (
            current_force
            < self.config.minimum_stable_force_n
        ):
            recommendation = (
                GripRecommendation.INCREASE_GRIP
            )

            target_force = min(
                self.config.target_grip_force_n,
                self.config.maximum_recommended_force_n,
            )

            confidence = 0.75

            explanation = (
                "Normal force below the configured "
                "stability threshold."
            )

        else:
            recommendation = GripRecommendation.HOLD

            target_force = current_force

            confidence = 0.90

            explanation = (
                "Contact and slip indicators are acceptable."
            )

        adjustment = (
            target_force - current_force
        )

        return GripForceAdvice(
            recommendation=recommendation,
            current_force_n=current_force,
            target_force_n=target_force,
            suggested_adjustment_n=adjustment,
            confidence=confidence,
            explanation=explanation,
        )


# ============================================================
# TACTILE FEEDBACK PROCESSOR
# ============================================================

class TactileFeedbackProcessor:

    def __init__(
        self,
        sensor: TactileSensorConfig,
        config: TactileFeedbackConfig | None = None,
    ) -> None:
        self.sensor = sensor
        self.config = config or TactileFeedbackConfig()

        self.calibration: TactileCalibration | None = None

        self.preprocessor = TactilePressurePreprocessor(
            sensor,
            self.config.smoothing_alpha,
        )

        self.contact_detector = TactileContactDetector(
            sensor,
            self.config,
        )

        self.slip_detector = TactileSlipDetector(
            self.config
        )

        self.health_monitor = TactileSensorHealthMonitor(
            sensor,
            self.config,
        )

        self.stability_evaluator = GripStabilityEvaluator(
            self.config
        )

        self.force_advisor = AdaptiveGripForceAdvisor(
            self.config
        )

        self._last_timestamp: datetime | None = None

        self._lock = threading.RLock()

    def set_calibration(
        self,
        calibration: TactileCalibration,
    ) -> None:
        if calibration.sensor_id != self.sensor.sensor_id:
            raise TactileCalibrationError(
                "Calibration sensor mismatch"
            )

        array2d(
            calibration.baseline_pressure_kpa,
            "baseline",
            self.sensor.shape,
        )

        with self._lock:
            self.calibration = calibration

            self.reset()

    def reset(self) -> None:
        with self._lock:
            self.preprocessor.reset()
            self.contact_detector.reset()
            self.slip_detector.reset()
            self._last_timestamp = None

    def process(
        self,
        sample: TactileSample,
        tenant_id: str,
        robot_id: str,
    ) -> TactileFeedbackReport:
        identifier(tenant_id, "tenant_id")
        identifier(robot_id, "robot_id")

        if sample.sensor_id != self.sensor.sensor_id:
            raise TactileValidationError(
                "Sample sensor mismatch"
            )

        with self._lock:
            if (
                self._last_timestamp is not None
                and sample.timestamp <= self._last_timestamp
            ):
                raise TactileValidationError(
                    "Tactile timestamps must increase"
                )

            pressure = self.preprocessor.process(
                sample,
                self.calibration,
            )

            contact = self.contact_detector.analyze(
                pressure,
                sample.shear_force_xy_n,
                sample.timestamp,
            )

            slip = self.slip_detector.analyze(
                contact
            )

            health = self.health_monitor.analyze(
                sample,
                pressure,
            )

            stability, score = (
                self.stability_evaluator.evaluate(
                    contact,
                    slip,
                    health,
                )
            )

            advice = self.force_advisor.recommend(
                contact,
                slip,
                health,
                stability,
            )

            if (
                health.health == SensorHealth.FAULT
                or contact.contact_state
                == ContactState.HIGH_PRESSURE
            ):
                severity = FeedbackSeverity.CRITICAL

            elif (
                slip.state == SlipState.SLIP_DETECTED
                or stability == GripStability.UNSTABLE
            ):
                severity = FeedbackSeverity.WARNING

            else:
                severity = FeedbackSeverity.INFO

            self._last_timestamp = sample.timestamp

            return TactileFeedbackReport(
                report_id=new_id(),
                tenant_id=tenant_id,
                robot_id=robot_id,
                sensor_id=self.sensor.sensor_id,
                contact=contact,
                slip=slip,
                health=health,
                stability=stability,
                stability_score=score,
                force_advice=advice,
                severity=severity,
                created_at=utc_now(),
            )


# ============================================================
# MULTI-FINGER GRASP ANALYZER
# ============================================================

class MultiFingerGraspAnalyzer:

    @staticmethod
    def analyze(
        reports: Sequence[TactileFeedbackReport],
    ) -> MultiFingerGraspReport:
        if not reports:
            raise TactileValidationError(
                "At least one tactile report is required"
            )

        tenant_id = reports[0].tenant_id
        robot_id = reports[0].robot_id

        sensor_ids: set[str] = set()

        for report in reports:
            if (
                report.tenant_id != tenant_id
                or report.robot_id != robot_id
            ):
                raise TactileAuthorizationError(
                    "Cannot aggregate reports across robots or tenants"
                )

            if report.sensor_id in sensor_ids:
                raise TactileValidationError(
                    "Duplicate sensor report"
                )

            sensor_ids.add(report.sensor_id)

        total_normal = sum(
            report.contact.normal_force_n
            for report in reports
        )

        total_shear = sum(
            report.contact.shear_force_n
            for report in reports
        )

        contacting = sum(
            report.contact.contact_detected
            for report in reports
        )

        score = float(
            np.mean(
                [
                    report.stability_score
                    for report in reports
                ]
            )
        )

        recommendations: list[str] = []

        if contacting == 0:
            stability = GripStability.NO_GRASP

            recommendations.append(
                "No tactile contact detected."
            )

        elif any(
            report.stability == GripStability.UNSTABLE
            for report in reports
        ):
            stability = GripStability.UNSTABLE

            recommendations.append(
                "Inspect unstable finger contacts."
            )

        elif any(
            report.stability == GripStability.MARGINAL
            for report in reports
        ):
            stability = GripStability.MARGINAL

            recommendations.append(
                "Reevaluate grip distribution."
            )

        else:
            stability = GripStability.STABLE

            recommendations.append(
                "Contact indicators are stable."
            )

        if any(
            report.slip.state == SlipState.SLIP_DETECTED
            for report in reports
        ):
            recommendations.append(
                "Slip detected; review the grasp plan."
            )

        if any(
            report.health.health == SensorHealth.FAULT
            for report in reports
        ):
            recommendations.append(
                "Sensor fault detected; stop and inspect."
            )

        return MultiFingerGraspReport(
            report_id=new_id(),
            tenant_id=tenant_id,
            robot_id=robot_id,
            finger_reports=tuple(reports),
            total_normal_force_n=total_normal,
            total_shear_force_n=total_shear,
            contacting_fingers=contacting,
            stability=stability,
            stability_score=score,
            recommendations=tuple(recommendations),
            created_at=utc_now(),
        )


# ============================================================
# PICK-AND-PLACE FEEDBACK INTEGRATION
# ============================================================

class PickPlaceTactileFeedback:

    @staticmethod
    def verify_grasp(
        report: MultiFingerGraspReport,
        minimum_contacting_fingers: int = 2,
    ) -> dict[str, Any]:
        if minimum_contacting_fingers < 1:
            raise TactileValidationError(
                "minimum_contacting_fingers must be positive"
            )

        verified = (
            report.contacting_fingers
            >= minimum_contacting_fingers
            and report.stability == GripStability.STABLE
        )

        return {
            "grasp_verified": verified,
            "contacting_fingers": report.contacting_fingers,
            "stability": report.stability.value,
            "stability_score": report.stability_score,
            "recommendations": list(
                report.recommendations
            ),
            "simulation_only": True,
        }

    @staticmethod
    def evaluate_transport(
        report: MultiFingerGraspReport,
    ) -> dict[str, Any]:
        slip_detected = any(
            finger.slip.state == SlipState.SLIP_DETECTED
            for finger in report.finger_reports
        )

        safe_to_continue_simulation = (
            report.stability == GripStability.STABLE
            and not slip_detected
        )

        return {
            "continue_simulation": (
                safe_to_continue_simulation
            ),
            "slip_detected": slip_detected,
            "stability": report.stability.value,
            "total_normal_force_n": (
                report.total_normal_force_n
            ),
            "physical_motion_authorized": False,
        }


# ============================================================
# MULTI-TENANT TACTILE SERVICE
# ============================================================

class TactileFeedbackService:

    def __init__(
        self,
        config: TactileFeedbackConfig | None = None,
    ) -> None:
        self.config = config or TactileFeedbackConfig()

        self._processors: dict[
            tuple[str, str, str],
            TactileFeedbackProcessor,
        ] = {}

        self._reports: dict[
            tuple[str, str],
            TactileFeedbackReport,
        ] = {}

        self._lock = threading.RLock()

    @staticmethod
    def _authorize(
        access: TactileAccessContext,
        permission: str,
    ) -> None:
        if not getattr(access, permission, False):
            raise TactileAuthorizationError(
                f"Missing permission: {permission}"
            )

    def register_sensor(
        self,
        robot_id: str,
        sensor: TactileSensorConfig,
        access: TactileAccessContext,
    ) -> None:
        self._authorize(access, "can_manage")

        identifier(robot_id, "robot_id")
        identifier(access.tenant_id, "tenant_id")

        key = (
            access.tenant_id,
            robot_id,
            sensor.sensor_id,
        )

        with self._lock:
            if key in self._processors:
                raise TactileConflictError(
                    "Sensor already registered"
                )

            self._processors[key] = (
                TactileFeedbackProcessor(
                    sensor,
                    self.config,
                )
            )

    def _get_processor(
        self,
        robot_id: str,
        sensor_id: str,
        tenant_id: str,
    ) -> TactileFeedbackProcessor:
        key = (
            tenant_id,
            robot_id,
            sensor_id,
        )

        processor = self._processors.get(key)

        if processor is None:
            raise TactileNotFoundError(
                "Tactile sensor not found"
            )

        return processor

    def calibrate(
        self,
        robot_id: str,
        sensor_id: str,
        samples: Sequence[TactileSample],
        access: TactileAccessContext,
    ) -> TactileCalibration:
        self._authorize(access, "can_calibrate")

        with self._lock:
            processor = self._get_processor(
                robot_id,
                sensor_id,
                access.tenant_id,
            )

            calibration = (
                TactileSensorCalibrator.calibrate(
                    processor.sensor,
                    samples,
                )
            )

            processor.set_calibration(
                calibration
            )

            return calibration

    def process(
        self,
        robot_id: str,
        sample: TactileSample,
        access: TactileAccessContext,
    ) -> TactileFeedbackReport:
        self._authorize(access, "can_process")

        with self._lock:
            processor = self._get_processor(
                robot_id,
                sample.sensor_id,
                access.tenant_id,
            )

            report = processor.process(
                sample,
                access.tenant_id,
                robot_id,
            )

            self._reports[
                (access.tenant_id, report.report_id)
            ] = report

            return report

    def get_report(
        self,
        report_id: str,
        access: TactileAccessContext,
    ) -> TactileFeedbackReport:
        self._authorize(access, "can_read")

        with self._lock:
            report = self._reports.get(
                (access.tenant_id, report_id)
            )

        if report is None:
            raise TactileNotFoundError(
                "Tactile report not found"
            )

        return report

    def list_reports(
        self,
        robot_id: str,
        access: TactileAccessContext,
    ) -> tuple[TactileFeedbackReport, ...]:
        self._authorize(access, "can_read")

        with self._lock:
            return tuple(
                report
                for (tenant_id, _), report
                in self._reports.items()
                if (
                    tenant_id == access.tenant_id
                    and report.robot_id == robot_id
                )
            )

    def aggregate(
        self,
        report_ids: Sequence[str],
        access: TactileAccessContext,
    ) -> MultiFingerGraspReport:
        self._authorize(access, "can_read")

        reports = [
            self.get_report(report_id, access)
            for report_id in report_ids
        ]

        return MultiFingerGraspAnalyzer.analyze(
            reports
        )

    def export_json(
        self,
        report: TactileFeedbackReport | MultiFingerGraspReport,
        access: TactileAccessContext,
    ) -> str:
        self._authorize(access, "can_read")

        if report.tenant_id != access.tenant_id:
            raise TactileAuthorizationError(
                "Cross-tenant export forbidden"
            )

        return json.dumps(
            json_safe(
                {
                    "report": report,
                    "simulation_only": True,
                    "physical_actuation": False,
                    "force_recommendations_advisory": True,
                    "slip_detection_certified": False,
                    "safety_certified": False,
                }
            ),
            indent=2,
            allow_nan=False,
        )


# ============================================================
# SYNTHETIC TACTILE DATA FACTORY
# ============================================================

class SyntheticTactileFactory:

    @staticmethod
    def sensor(
        finger: FingerName = FingerName.LEFT_JAW,
    ) -> TactileSensorConfig:
        return TactileSensorConfig(
            sensor_id=f"tactile-{finger.value}",
            finger=finger,
            rows=8,
            columns=8,
            taxel_width_m=0.005,
            taxel_height_m=0.005,
            maximum_pressure_kpa=250.0,
        )

    @staticmethod
    def pressure_matrix(
        pressure_kpa: float = 80.0,
        center_row: int = 4,
        center_column: int = 4,
        rows: int = 8,
        columns: int = 8,
    ) -> tuple[tuple[float, ...], ...]:
        matrix = np.zeros(
            (rows, columns),
            dtype=float,
        )

        for row in range(rows):
            for column in range(columns):
                distance_squared = (
                    (row - center_row) ** 2
                    + (column - center_column) ** 2
                )

                matrix[row, column] = (
                    pressure_kpa
                    * math.exp(
                        -distance_squared / 4.0
                    )
                )

        return tuple(
            tuple(float(value) for value in row)
            for row in matrix
        )

    @staticmethod
    def sample(
        sensor_id: str,
        pressure_kpa: float = 80.0,
        shear_force_xy_n: tuple[float, float] = (
            0.2,
            0.1,
        ),
        center_column: int = 4,
        timestamp: datetime | None = None,
    ) -> TactileSample:
        return TactileSample(
            sample_id=new_id(),
            sensor_id=sensor_id,
            timestamp=timestamp or utc_now(),
            pressure_kpa=(
                SyntheticTactileFactory.pressure_matrix(
                    pressure_kpa=pressure_kpa,
                    center_column=center_column,
                )
            ),
            shear_force_xy_n=shear_force_xy_n,
        )

    @staticmethod
    def access(
        tenant_id: str = "physical_ai_lab",
    ) -> TactileAccessContext:
        return TactileAccessContext(
            tenant_id=tenant_id,
            user_id="robotics_engineer",
            can_read=True,
            can_process=True,
            can_calibrate=True,
            can_manage=True,
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestTactileFeedback(unittest.TestCase):

    def setUp(self) -> None:
        self.sensor = SyntheticTactileFactory.sensor()

        self.config = TactileFeedbackConfig()

        self.access = SyntheticTactileFactory.access()

    def test_sensor_shape(self) -> None:
        self.assertEqual(
            self.sensor.shape,
            (8, 8),
        )

    def test_taxel_area(self) -> None:
        self.assertAlmostEqual(
            self.sensor.taxel_area_m2,
            0.000025,
        )

    def test_sample_generation(self) -> None:
        sample = SyntheticTactileFactory.sample(
            self.sensor.sensor_id
        )

        self.assertEqual(
            len(sample.pressure_kpa),
            8,
        )

    def test_invalid_pressure_shape(self) -> None:
        sample = TactileSample(
            sample_id=new_id(),
            sensor_id=self.sensor.sensor_id,
            timestamp=utc_now(),
            pressure_kpa=((10.0,),),
        )

        processor = TactileFeedbackProcessor(
            self.sensor,
            self.config,
        )

        with self.assertRaises(
            TactileValidationError
        ):
            processor.process(
                sample,
                "tenant",
                "robot",
            )

    def test_pressure_preprocessing(self) -> None:
        processor = TactilePressurePreprocessor(
            self.sensor,
            1.0,
        )

        sample = SyntheticTactileFactory.sample(
            self.sensor.sensor_id
        )

        result = processor.process(
            sample,
            None,
        )

        self.assertEqual(
            result.shape,
            (8, 8),
        )

    def test_contact_detection(self) -> None:
        detector = TactileContactDetector(
            self.sensor,
            self.config,
        )

        pressure = np.full(
            (8, 8),
            50.0,
        )

        result = detector.analyze(
            pressure,
            (0.1, 0.1),
            utc_now(),
        )

        self.assertTrue(
            result.contact_detected
        )

    def test_no_contact(self) -> None:
        detector = TactileContactDetector(
            self.sensor,
            self.config,
        )

        result = detector.analyze(
            np.zeros((8, 8)),
            (0.0, 0.0),
            utc_now(),
        )

        self.assertFalse(
            result.contact_detected
        )

    def test_normal_force(self) -> None:
        detector = TactileContactDetector(
            self.sensor,
            self.config,
        )

        result = detector.analyze(
            np.full((8, 8), 100.0),
            (0.0, 0.0),
            utc_now(),
        )

        expected = (
            64
            * 100.0
            * self.sensor.taxel_area_m2
            * 1000.0
        )

        self.assertAlmostEqual(
            result.normal_force_n,
            expected,
        )

    def test_contact_centroid(self) -> None:
        detector = TactileContactDetector(
            self.sensor,
            self.config,
        )

        result = detector.analyze(
            np.full((8, 8), 100.0),
            (0.0, 0.0),
            utc_now(),
        )

        self.assertIsNotNone(
            result.centroid_xy_m
        )

        self.assertAlmostEqual(
            result.centroid_xy_m[0],
            0.02,
        )

    def test_pressure_uniformity(self) -> None:
        detector = TactileContactDetector(
            self.sensor,
            self.config,
        )

        result = detector.analyze(
            np.full((8, 8), 100.0),
            (0.0, 0.0),
            utc_now(),
        )

        self.assertAlmostEqual(
            result.pressure_uniformity,
            1.0,
        )

    def test_high_pressure(self) -> None:
        detector = TactileContactDetector(
            self.sensor,
            self.config,
        )

        result = detector.analyze(
            np.full((8, 8), 200.0),
            (0.0, 0.0),
            utc_now(),
        )

        self.assertEqual(
            result.contact_state,
            ContactState.HIGH_PRESSURE,
        )

    def test_slip_history(self) -> None:
        detector = TactileContactDetector(
            self.sensor,
            self.config,
        )

        slip_detector = TactileSlipDetector(
            self.config
        )

        contact = detector.analyze(
            np.full((8, 8), 50.0),
            (0.0, 0.0),
            utc_now(),
        )

        slip = slip_detector.analyze(
            contact
        )

        self.assertEqual(
            slip.state,
            SlipState.INSUFFICIENT_HISTORY,
        )

    def test_sensor_health(self) -> None:
        monitor = TactileSensorHealthMonitor(
            self.sensor,
            self.config,
        )

        sample = SyntheticTactileFactory.sample(
            self.sensor.sensor_id
        )

        result = monitor.analyze(
            sample,
            np.full((8, 8), 50.0),
        )

        self.assertEqual(
            result.health,
            SensorHealth.HEALTHY,
        )

    def test_sensor_saturation(self) -> None:
        monitor = TactileSensorHealthMonitor(
            self.sensor,
            self.config,
        )

        sample = SyntheticTactileFactory.sample(
            self.sensor.sensor_id
        )

        result = monitor.analyze(
            sample,
            np.full((8, 8), 250.0),
        )

        self.assertEqual(
            result.health,
            SensorHealth.FAULT,
        )

    def test_feedback_processor(self) -> None:
        processor = TactileFeedbackProcessor(
            self.sensor,
            self.config,
        )

        sample = SyntheticTactileFactory.sample(
            self.sensor.sensor_id
        )

        report = processor.process(
            sample,
            "tenant",
            "robot",
        )

        self.assertTrue(
            report.contact.contact_detected
        )

    def test_feedback_stability_score(self) -> None:
        processor = TactileFeedbackProcessor(
            self.sensor,
            self.config,
        )

        report = processor.process(
            SyntheticTactileFactory.sample(
                self.sensor.sensor_id
            ),
            "tenant",
            "robot",
        )

        self.assertGreaterEqual(
            report.stability_score,
            0.0,
        )

        self.assertLessEqual(
            report.stability_score,
            1.0,
        )

    def test_grip_force_advice(self) -> None:
        processor = TactileFeedbackProcessor(
            self.sensor,
            self.config,
        )

        report = processor.process(
            SyntheticTactileFactory.sample(
                self.sensor.sensor_id
            ),
            "tenant",
            "robot",
        )

        self.assertIsInstance(
            report.force_advice,
            GripForceAdvice,
        )

    def test_multi_finger_aggregation(self) -> None:
        left_sensor = SyntheticTactileFactory.sensor(
            FingerName.LEFT_JAW
        )

        right_sensor = SyntheticTactileFactory.sensor(
            FingerName.RIGHT_JAW
        )

        left_processor = TactileFeedbackProcessor(
            left_sensor
        )

        right_processor = TactileFeedbackProcessor(
            right_sensor
        )

        left = left_processor.process(
            SyntheticTactileFactory.sample(
                left_sensor.sensor_id
            ),
            "tenant",
            "robot",
        )

        right = right_processor.process(
            SyntheticTactileFactory.sample(
                right_sensor.sensor_id
            ),
            "tenant",
            "robot",
        )

        combined = MultiFingerGraspAnalyzer.analyze(
            (left, right)
        )

        self.assertEqual(
            combined.contacting_fingers,
            2,
        )

    def test_pick_place_verification(self) -> None:
        reports = []

        for finger in (
            FingerName.LEFT_JAW,
            FingerName.RIGHT_JAW,
        ):
            sensor = SyntheticTactileFactory.sensor(
                finger
            )

            processor = TactileFeedbackProcessor(
                sensor
            )

            report = processor.process(
                SyntheticTactileFactory.sample(
                    sensor.sensor_id
                ),
                "tenant",
                "robot",
            )

            reports.append(report)

        combined = MultiFingerGraspAnalyzer.analyze(
            reports
        )

        verification = (
            PickPlaceTactileFeedback.verify_grasp(
                combined
            )
        )

        self.assertIn(
            "grasp_verified",
            verification,
        )

    def test_service_registration(self) -> None:
        service = TactileFeedbackService()

        service.register_sensor(
            "robot-001",
            self.sensor,
            self.access,
        )

        self.assertEqual(
            len(service._processors),
            1,
        )

    def test_service_processing(self) -> None:
        service = TactileFeedbackService()

        service.register_sensor(
            "robot-001",
            self.sensor,
            self.access,
        )

        report = service.process(
            "robot-001",
            SyntheticTactileFactory.sample(
                self.sensor.sensor_id
            ),
            self.access,
        )

        self.assertEqual(
            report.robot_id,
            "robot-001",
        )

    def test_service_report_lookup(self) -> None:
        service = TactileFeedbackService()

        service.register_sensor(
            "robot-001",
            self.sensor,
            self.access,
        )

        report = service.process(
            "robot-001",
            SyntheticTactileFactory.sample(
                self.sensor.sensor_id
            ),
            self.access,
        )

        found = service.get_report(
            report.report_id,
            self.access,
        )

        self.assertEqual(
            found.report_id,
            report.report_id,
        )

    def test_tenant_isolation(self) -> None:
        service = TactileFeedbackService()

        service.register_sensor(
            "robot-001",
            self.sensor,
            self.access,
        )

        other = TactileAccessContext(
            tenant_id="other_tenant",
            user_id="other_user",
            can_read=True,
            can_process=True,
        )

        with self.assertRaises(
            TactileNotFoundError
        ):
            service.process(
                "robot-001",
                SyntheticTactileFactory.sample(
                    self.sensor.sensor_id
                ),
                other,
            )

    def test_permission_denied(self) -> None:
        service = TactileFeedbackService()

        viewer = TactileAccessContext(
            tenant_id="physical_ai_lab",
            user_id="viewer",
            can_read=True,
        )

        with self.assertRaises(
            TactileAuthorizationError
        ):
            service.register_sensor(
                "robot-001",
                self.sensor,
                viewer,
            )

    def test_json_export(self) -> None:
        service = TactileFeedbackService()

        service.register_sensor(
            "robot-001",
            self.sensor,
            self.access,
        )

        report = service.process(
            "robot-001",
            SyntheticTactileFactory.sample(
                self.sensor.sensor_id
            ),
            self.access,
        )

        exported = json.loads(
            service.export_json(
                report,
                self.access,
            )
        )

        self.assertTrue(
            exported["simulation_only"]
        )

        self.assertFalse(
            exported["physical_actuation"]
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("\nPHYSICAL AI — TACTILE FEEDBACK")
    print("=" * 72)

    service = TactileFeedbackService()

    access = SyntheticTactileFactory.access()

    left_sensor = SyntheticTactileFactory.sensor(
        FingerName.LEFT_JAW
    )

    right_sensor = SyntheticTactileFactory.sensor(
        FingerName.RIGHT_JAW
    )

    service.register_sensor(
        "robot-arm-001",
        left_sensor,
        access,
    )

    service.register_sensor(
        "robot-arm-001",
        right_sensor,
        access,
    )

    left_report = service.process(
        "robot-arm-001",
        SyntheticTactileFactory.sample(
            left_sensor.sensor_id,
            pressure_kpa=90.0,
        ),
        access,
    )

    right_report = service.process(
        "robot-arm-001",
        SyntheticTactileFactory.sample(
            right_sensor.sensor_id,
            pressure_kpa=85.0,
        ),
        access,
    )

    grasp_report = service.aggregate(
        [
            left_report.report_id,
            right_report.report_id,
        ],
        access,
    )

    print("Robot:", grasp_report.robot_id)

    print(
        "Contacting fingers:",
        grasp_report.contacting_fingers,
    )

    print(
        "Total normal force:",
        round(
            grasp_report.total_normal_force_n,
            3,
        ),
        "N",
    )

    print(
        "Total shear force:",
        round(
            grasp_report.total_shear_force_n,
            3,
        ),
        "N",
    )

    print(
        "Grip stability:",
        grasp_report.stability.value,
    )

    print(
        "Stability score:",
        round(
            grasp_report.stability_score,
            4,
        ),
    )

    verification = (
        PickPlaceTactileFeedback.verify_grasp(
            grasp_report
        )
    )

    print(
        "Simulated grasp verification:",
        verification["grasp_verified"],
    )

    print("\nTACTILE FEEDBACK JSON")
    print("=" * 72)

    print(
        service.export_json(
            grasp_report,
            access,
        )
    )

    print("\nRUNNING UNIT TESTS")
    print("=" * 72)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestTactileFeedback
    )

    result = unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)

    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
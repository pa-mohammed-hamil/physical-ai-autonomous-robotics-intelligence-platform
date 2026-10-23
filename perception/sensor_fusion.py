"""Sensor Fusion module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/perception/sensor_fusion.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================

Module: sensor_fusion.py

Multi-Sensor Fusion Engine

Features:
    - Camera, LiDAR, radar, IMU, GPS and wheel odometry integration
    - Timestamp synchronization
    - Sensor coordinate transformations
    - Multi-sensor position and velocity fusion
    - Kalman filtering
    - Extended Kalman filtering for nonlinear measurements
    - Mahalanobis outlier rejection
    - Sensor confidence and covariance weighting
    - Sensor health and stale-data detection
    - Multi-object track fusion
    - Sensor disagreement detection
    - Fusion quality assessment
    - Thread-safe state management
    - Multi-tenant authorization
    - Fusion history and statistics
    - JSON serialization
    - Synthetic sensor simulation
    - Unit tests

Dependencies:
    Python >= 3.11
    numpy

Safety:
    Fusion outputs are advisory estimates.
    They must not independently authorize robot movement,
    collision avoidance, emergency stopping, or actuator commands.
"""

from __future__ import annotations

import json
import math
import threading
import time
import unittest
import uuid

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import numpy as np


# ============================================================
# EXCEPTIONS
# ============================================================

class SensorFusionError(Exception):
    pass


class FusionValidationError(SensorFusionError):
    pass


class FusionAuthorizationError(SensorFusionError):
    pass


class FusionSynchronizationError(SensorFusionError):
    pass


class FusionTransformationError(SensorFusionError):
    pass


class FusionCapacityError(SensorFusionError):
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
    if not isinstance(value, str):
        raise FusionValidationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > 128:
        raise FusionValidationError(
            f"{name} must contain 1-128 characters"
        )

    return value


def finite_float(
    value: Any,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise FusionValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FusionValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise FusionValidationError(
            f"{name} must be finite"
        )

    return result


def validate_timestamp(
    value: datetime,
) -> datetime:
    if not isinstance(value, datetime):
        raise FusionValidationError(
            "Timestamp must be datetime"
        )

    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise FusionValidationError(
            "Timestamp must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


def validate_probability(
    value: float,
    name: str = "probability",
) -> float:
    value = finite_float(value, name)

    if not 0 <= value <= 1:
        raise FusionValidationError(
            f"{name} must be between 0 and 1"
        )

    return value


def validate_covariance(
    matrix: np.ndarray,
    dimension: int,
    *,
    minimum_eigenvalue: float = 1e-10,
) -> np.ndarray:
    matrix = np.asarray(
        matrix,
        dtype=np.float64,
    )

    if matrix.shape != (
        dimension,
        dimension,
    ):
        raise FusionValidationError(
            f"Covariance must have shape "
            f"({dimension}, {dimension})"
        )

    if not np.all(np.isfinite(matrix)):
        raise FusionValidationError(
            "Covariance contains non-finite values"
        )

    if not np.allclose(
        matrix,
        matrix.T,
        atol=1e-8,
    ):
        raise FusionValidationError(
            "Covariance must be symmetric"
        )

    eigenvalues = np.linalg.eigvalsh(
        matrix
    )

    if np.min(eigenvalues) < minimum_eigenvalue:
        raise FusionValidationError(
            "Covariance must be positive definite"
        )

    return matrix.copy()


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if hasattr(value, "__dataclass_fields__"):
        return {
            name: json_safe(getattr(value, name))
            for name in value.__dataclass_fields__
        }

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, deque)):
        return [
            json_safe(item)
            for item in value
        ]

    return value


# ============================================================
# ENUMERATIONS
# ============================================================

class SensorType(str, Enum):
    CAMERA = "camera"
    LIDAR = "lidar"
    RADAR = "radar"
    IMU = "imu"
    GPS = "gps"
    WHEEL_ODOMETRY = "wheel_odometry"
    DEPTH_CAMERA = "depth_camera"
    ULTRASONIC = "ultrasonic"
    VISUAL_ODOMETRY = "visual_odometry"
    OTHER = "other"


class MeasurementType(str, Enum):
    POSITION_3D = "position_3d"
    VELOCITY_3D = "velocity_3d"
    POSITION_VELOCITY_3D = "position_velocity_3d"
    RANGE_BEARING = "range_bearing"
    ACCELERATION_3D = "acceleration_3d"


class FusionAlgorithm(str, Enum):
    KALMAN = "kalman"
    EXTENDED_KALMAN = "extended_kalman"
    WEIGHTED_AVERAGE = "weighted_average"


class SensorHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    STALE = "stale"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class FusionQuality(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    DEGRADED = "degraded"
    POOR = "poor"
    UNAVAILABLE = "unavailable"


class FusionAlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class FusionAlertCategory(str, Enum):
    SENSOR_STALE = "sensor_stale"
    SENSOR_DISAGREEMENT = "sensor_disagreement"
    OUTLIER_REJECTED = "outlier_rejected"
    HIGH_UNCERTAINTY = "high_uncertainty"
    LOW_CONFIDENCE = "low_confidence"
    SYNCHRONIZATION_FAILURE = "synchronization_failure"
    MISSING_TRANSFORM = "missing_transform"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class SensorFusionConfig:
    synchronization_tolerance_ms: float = 100.0

    maximum_sensor_age_ms: float = 500.0

    process_acceleration_variance: float = 1.0

    initial_position_variance: float = 10.0

    initial_velocity_variance: float = 5.0

    mahalanobis_threshold: float = 16.27

    minimum_sensor_confidence: float = 0.30

    sensor_disagreement_threshold_m: float = 2.0

    high_uncertainty_threshold: float = 25.0

    maximum_prediction_interval_s: float = 5.0

    maximum_measurements_per_batch: int = 500

    maximum_history: int = 200

    maximum_tracks: int = 1000

    def __post_init__(self) -> None:
        positive_fields = (
            "synchronization_tolerance_ms",
            "maximum_sensor_age_ms",
            "process_acceleration_variance",
            "initial_position_variance",
            "initial_velocity_variance",
            "mahalanobis_threshold",
            "sensor_disagreement_threshold_m",
            "high_uncertainty_threshold",
            "maximum_prediction_interval_s",
        )

        for name in positive_fields:
            if finite_float(
                getattr(self, name),
                name,
            ) <= 0:
                raise FusionValidationError(
                    f"{name} must be positive"
                )

        validate_probability(
            self.minimum_sensor_confidence,
            "minimum_sensor_confidence",
        )

        for name in (
            "maximum_measurements_per_batch",
            "maximum_history",
            "maximum_tracks",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise FusionValidationError(
                    f"{name} must be a positive integer"
                )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class FusionAccessContext:
    tenant_id: str
    user_id: str

    allowed_robot_ids: frozenset[str] = frozenset()

    can_ingest: bool = False
    can_view: bool = False
    can_manage: bool = False

    def __post_init__(self) -> None:
        validate_identifier(
            self.tenant_id,
            "tenant_id",
        )

        validate_identifier(
            self.user_id,
            "user_id",
        )


# ============================================================
# GEOMETRY
# ============================================================

@dataclass(frozen=True)
class FusionVector3:
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        for name in ("x", "y", "z"):
            finite_float(
                getattr(self, name),
                name,
            )

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.x, self.y, self.z],
            dtype=np.float64,
        )

    @classmethod
    def from_array(
        cls,
        values: Sequence[float],
    ) -> FusionVector3:
        array = np.asarray(
            values,
            dtype=np.float64,
        )

        if array.shape != (3,):
            raise FusionValidationError(
                "Vector must contain 3 elements"
            )

        return cls(
            x=float(array[0]),
            y=float(array[1]),
            z=float(array[2]),
        )

    def distance_to(
        self,
        other: FusionVector3,
    ) -> float:
        return float(
            np.linalg.norm(
                self.as_array()
                - other.as_array()
            )
        )


@dataclass(frozen=True)
class SensorTransform:
    source_frame: str
    target_frame: str

    rotation: np.ndarray
    translation: FusionVector3

    def __post_init__(self) -> None:
        validate_identifier(
            self.source_frame,
            "source_frame",
        )

        validate_identifier(
            self.target_frame,
            "target_frame",
        )

        rotation = np.asarray(
            self.rotation,
            dtype=np.float64,
        )

        if rotation.shape != (3, 3):
            raise FusionValidationError(
                "Rotation must be 3x3"
            )

        if not np.all(np.isfinite(rotation)):
            raise FusionValidationError(
                "Rotation must be finite"
            )

        if not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=1e-5,
        ):
            raise FusionValidationError(
                "Rotation must be orthonormal"
            )

        if not np.isclose(
            np.linalg.det(rotation),
            1.0,
            atol=1e-5,
        ):
            raise FusionValidationError(
                "Rotation must have determinant +1"
            )

        object.__setattr__(
            self,
            "rotation",
            rotation.copy(),
        )

    def transform_position(
        self,
        position: FusionVector3,
    ) -> FusionVector3:
        transformed = (
            self.rotation @ position.as_array()
            + self.translation.as_array()
        )

        return FusionVector3.from_array(
            transformed
        )

    def transform_velocity(
        self,
        velocity: FusionVector3,
    ) -> FusionVector3:
        transformed = (
            self.rotation @ velocity.as_array()
        )

        return FusionVector3.from_array(
            transformed
        )

    def inverse(self) -> SensorTransform:
        inverse_rotation = self.rotation.T

        inverse_translation = (
            -inverse_rotation
            @ self.translation.as_array()
        )

        return SensorTransform(
            source_frame=self.target_frame,
            target_frame=self.source_frame,
            rotation=inverse_rotation,
            translation=FusionVector3.from_array(
                inverse_translation
            ),
        )


# ============================================================
# TRANSFORM REGISTRY
# ============================================================

class SensorTransformRegistry:

    def __init__(self) -> None:
        self._transforms: dict[
            tuple[str, str, str, str],
            SensorTransform,
        ] = {}

        self._lock = threading.RLock()

    def register(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        transform: SensorTransform,
    ) -> None:
        validate_identifier(
            tenant_id,
            "tenant_id",
        )

        validate_identifier(
            robot_id,
            "robot_id",
        )

        key = (
            tenant_id,
            robot_id,
            transform.source_frame,
            transform.target_frame,
        )

        inverse_key = (
            tenant_id,
            robot_id,
            transform.target_frame,
            transform.source_frame,
        )

        with self._lock:
            self._transforms[key] = transform

            self._transforms[inverse_key] = (
                transform.inverse()
            )

    def resolve(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        source_frame: str,
        target_frame: str,
    ) -> SensorTransform:
        if source_frame == target_frame:
            return SensorTransform(
                source_frame=source_frame,
                target_frame=target_frame,
                rotation=np.eye(3),
                translation=FusionVector3(
                    0.0,
                    0.0,
                    0.0,
                ),
            )

        key = (
            tenant_id,
            robot_id,
            source_frame,
            target_frame,
        )

        with self._lock:
            transform = self._transforms.get(
                key
            )

        if transform is None:
            raise FusionTransformationError(
                f"No transform from {source_frame} "
                f"to {target_frame}"
            )

        return transform


# ============================================================
# SENSOR MEASUREMENT
# ============================================================

@dataclass(frozen=True)
class SensorMeasurement:
    measurement_id: str

    tenant_id: str
    robot_id: str
    sensor_id: str

    sensor_type: SensorType
    measurement_type: MeasurementType

    timestamp: datetime
    frame_id: str

    values: np.ndarray
    covariance: np.ndarray

    confidence: float = 1.0

    object_id: str = "robot"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "measurement_id",
            "tenant_id",
            "robot_id",
            "sensor_id",
            "frame_id",
            "object_id",
        ):
            validate_identifier(
                getattr(self, name),
                name,
            )

        validate_timestamp(
            self.timestamp
        )

        validate_probability(
            self.confidence,
            "confidence",
        )

        dimensions = {
            MeasurementType.POSITION_3D: 3,
            MeasurementType.VELOCITY_3D: 3,
            MeasurementType.POSITION_VELOCITY_3D: 6,
            MeasurementType.RANGE_BEARING: 2,
            MeasurementType.ACCELERATION_3D: 3,
        }

        dimension = dimensions[
            self.measurement_type
        ]

        values = np.asarray(
            self.values,
            dtype=np.float64,
        )

        if values.shape != (dimension,):
            raise FusionValidationError(
                f"Measurement requires {dimension} values"
            )

        if not np.all(np.isfinite(values)):
            raise FusionValidationError(
                "Measurement values must be finite"
            )

        if (
            self.measurement_type
            == MeasurementType.RANGE_BEARING
            and values[0] < 0
        ):
            raise FusionValidationError(
                "Range cannot be negative"
            )

        covariance = validate_covariance(
            self.covariance,
            dimension,
        )

        object.__setattr__(
            self,
            "values",
            values.copy(),
        )

        object.__setattr__(
            self,
            "covariance",
            covariance,
        )

        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )


# ============================================================
# SENSOR HEALTH
# ============================================================

@dataclass(frozen=True)
class SensorHealthReport:
    sensor_id: str
    sensor_type: SensorType

    status: SensorHealth

    age_ms: float

    confidence: float

    explanation: str


class SensorHealthEvaluator:

    def __init__(
        self,
        config: SensorFusionConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        measurement: SensorMeasurement,
        reference_time: datetime,
    ) -> SensorHealthReport:
        reference_time = validate_timestamp(
            reference_time
        )

        age_ms = (
            reference_time - measurement.timestamp
        ).total_seconds() * 1000.0

        if age_ms < 0:
            status = SensorHealth.DEGRADED
            explanation = "Measurement is future-dated."

        elif age_ms > self.config.maximum_sensor_age_ms:
            status = SensorHealth.STALE
            explanation = "Measurement exceeded maximum age."

        elif (
            measurement.confidence
            < self.config.minimum_sensor_confidence
        ):
            status = SensorHealth.DEGRADED
            explanation = "Measurement confidence is low."

        else:
            status = SensorHealth.HEALTHY
            explanation = "Measurement passed basic health checks."

        return SensorHealthReport(
            sensor_id=measurement.sensor_id,
            sensor_type=measurement.sensor_type,
            status=status,
            age_ms=age_ms,
            confidence=measurement.confidence,
            explanation=explanation,
        )


# ============================================================
# TIMESTAMP SYNCHRONIZATION
# ============================================================

@dataclass(frozen=True)
class SynchronizedMeasurementBatch:
    batch_id: str

    reference_timestamp: datetime

    measurements: tuple[SensorMeasurement, ...]

    rejected_measurements: tuple[SensorMeasurement, ...]

    maximum_time_offset_ms: float


class SensorTimeSynchronizer:

    def __init__(
        self,
        config: SensorFusionConfig,
    ) -> None:
        self.config = config

    def synchronize(
        self,
        measurements: Sequence[SensorMeasurement],
        *,
        reference_timestamp: datetime | None = None,
    ) -> SynchronizedMeasurementBatch:
        if not measurements:
            raise FusionSynchronizationError(
                "No measurements supplied"
            )

        if (
            len(measurements)
            > self.config.maximum_measurements_per_batch
        ):
            raise FusionCapacityError(
                "Measurement batch limit exceeded"
            )

        if reference_timestamp is None:
            timestamps = sorted(
                measurement.timestamp
                for measurement in measurements
            )

            reference_timestamp = timestamps[
                len(timestamps) // 2
            ]

        reference_timestamp = validate_timestamp(
            reference_timestamp
        )

        accepted = []
        rejected = []
        maximum_offset = 0.0

        for measurement in measurements:
            offset_ms = abs(
                (
                    measurement.timestamp
                    - reference_timestamp
                ).total_seconds() * 1000.0
            )

            maximum_offset = max(
                maximum_offset,
                offset_ms,
            )

            if (
                offset_ms
                <= self.config.synchronization_tolerance_ms
            ):
                accepted.append(
                    measurement
                )

            else:
                rejected.append(
                    measurement
                )

        return SynchronizedMeasurementBatch(
            batch_id=generate_id(),
            reference_timestamp=reference_timestamp,
            measurements=tuple(accepted),
            rejected_measurements=tuple(rejected),
            maximum_time_offset_ms=maximum_offset,
        )


# ============================================================
# LINEAR KALMAN FILTER
# ============================================================

class LinearKalmanFilter3D:
    """
    State vector:
        [x, y, z, vx, vy, vz]

    Constant-velocity motion model with white
    acceleration process noise.
    """

    STATE_DIMENSION = 6

    def __init__(
        self,
        config: SensorFusionConfig,
    ) -> None:
        self.config = config

        self.state = np.zeros(
            6,
            dtype=np.float64,
        )

        self.covariance = np.diag(
            [
                config.initial_position_variance,
                config.initial_position_variance,
                config.initial_position_variance,
                config.initial_velocity_variance,
                config.initial_velocity_variance,
                config.initial_velocity_variance,
            ]
        ).astype(np.float64)

        self.initialized = False

    @staticmethod
    def transition_matrix(
        dt: float,
    ) -> np.ndarray:
        matrix = np.eye(
            6,
            dtype=np.float64,
        )

        matrix[:3, 3:] = (
            np.eye(3) * dt
        )

        return matrix

    def process_noise(
        self,
        dt: float,
    ) -> np.ndarray:
        identity = np.eye(3)

        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2

        q = self.config.process_acceleration_variance

        return q * np.block(
            [
                [
                    identity * dt4 / 4.0,
                    identity * dt3 / 2.0,
                ],
                [
                    identity * dt3 / 2.0,
                    identity * dt2,
                ],
            ]
        )

    def initialize(
        self,
        position: np.ndarray,
        velocity: np.ndarray | None = None,
        covariance: np.ndarray | None = None,
    ) -> None:
        position = np.asarray(
            position,
            dtype=np.float64,
        )

        if position.shape != (3,):
            raise FusionValidationError(
                "Initial position must have 3 values"
            )

        if not np.all(np.isfinite(position)):
            raise FusionValidationError(
                "Initial position must be finite"
            )

        self.state[:3] = position

        if velocity is not None:
            velocity = np.asarray(
                velocity,
                dtype=np.float64,
            )

            if velocity.shape != (3,):
                raise FusionValidationError(
                    "Initial velocity must have 3 values"
                )

            if not np.all(np.isfinite(velocity)):
                raise FusionValidationError(
                    "Initial velocity must be finite"
                )

            self.state[3:] = velocity

        if covariance is not None:
            self.covariance = validate_covariance(
                covariance,
                6,
            )

        self.initialized = True

    def predict(
        self,
        dt: float,
    ) -> None:
        dt = finite_float(
            dt,
            "dt",
        )

        if dt < 0:
            raise FusionValidationError(
                "Prediction interval cannot be negative"
            )

        if (
            dt
            > self.config.maximum_prediction_interval_s
        ):
            raise FusionValidationError(
                "Prediction interval exceeds configured maximum"
            )

        if not self.initialized:
            raise SensorFusionError(
                "Kalman filter is not initialized"
            )

        transition = self.transition_matrix(
            dt
        )

        process_noise = self.process_noise(
            dt
        )

        self.state = (
            transition @ self.state
        )

        self.covariance = (
            transition
            @ self.covariance
            @ transition.T
            + process_noise
        )

        self.covariance = (
            self.covariance
            + self.covariance.T
        ) / 2.0

    def innovation(
        self,
        measurement: np.ndarray,
        observation_matrix: np.ndarray,
        measurement_covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        innovation = (
            measurement
            - observation_matrix @ self.state
        )

        innovation_covariance = (
            observation_matrix
            @ self.covariance
            @ observation_matrix.T
            + measurement_covariance
        )

        return (
            innovation,
            innovation_covariance,
        )

    @staticmethod
    def mahalanobis_distance_squared(
        innovation: np.ndarray,
        innovation_covariance: np.ndarray,
    ) -> float:
        try:
            solution = np.linalg.solve(
                innovation_covariance,
                innovation,
            )

        except np.linalg.LinAlgError as exc:
            raise SensorFusionError(
                "Innovation covariance is singular"
            ) from exc

        return float(
            innovation.T @ solution
        )

    def update(
        self,
        measurement: np.ndarray,
        observation_matrix: np.ndarray,
        measurement_covariance: np.ndarray,
        *,
        reject_outliers: bool = True,
    ) -> tuple[bool, float]:
        if not self.initialized:
            raise SensorFusionError(
                "Kalman filter is not initialized"
            )

        measurement = np.asarray(
            measurement,
            dtype=np.float64,
        )

        observation_matrix = np.asarray(
            observation_matrix,
            dtype=np.float64,
        )

        dimension = measurement.shape[0]

        if measurement.ndim != 1:
            raise FusionValidationError(
                "Measurement must be a vector"
            )

        if observation_matrix.shape != (
            dimension,
            6,
        ):
            raise FusionValidationError(
                "Observation matrix has invalid shape"
            )

        measurement_covariance = validate_covariance(
            measurement_covariance,
            dimension,
        )

        innovation, innovation_covariance = (
            self.innovation(
                measurement,
                observation_matrix,
                measurement_covariance,
            )
        )

        distance_squared = (
            self.mahalanobis_distance_squared(
                innovation,
                innovation_covariance,
            )
        )

        if (
            reject_outliers
            and distance_squared
            > self.config.mahalanobis_threshold
        ):
            return False, distance_squared

        try:
            kalman_gain = np.linalg.solve(
                innovation_covariance,
                observation_matrix @ self.covariance,
            ).T

        except np.linalg.LinAlgError as exc:
            raise SensorFusionError(
                "Kalman gain computation failed"
            ) from exc

        self.state = (
            self.state
            + kalman_gain @ innovation
        )

        identity = np.eye(6)

        residual_projection = (
            identity
            - kalman_gain @ observation_matrix
        )

        # Joseph covariance update for numerical stability.
        self.covariance = (
            residual_projection
            @ self.covariance
            @ residual_projection.T
            + kalman_gain
            @ measurement_covariance
            @ kalman_gain.T
        )

        self.covariance = (
            self.covariance
            + self.covariance.T
        ) / 2.0

        return True, distance_squared

    @property
    def position(self) -> FusionVector3:
        return FusionVector3.from_array(
            self.state[:3]
        )

    @property
    def velocity(self) -> FusionVector3:
        return FusionVector3.from_array(
            self.state[3:]
        )


# ============================================================
# EXTENDED KALMAN FILTER
# ============================================================

class ExtendedKalmanFilter3D(
    LinearKalmanFilter3D
):
    """
    Supports nonlinear planar range-bearing observations.

    The range-bearing sensor must already be transformed
    or calibrated into the same reference coordinate frame
    as the filter state.

    State:
        [x, y, z, vx, vy, vz]

    Measurement:
        [range, bearing]
    """

    @staticmethod
    def measurement_function(
        state: np.ndarray,
    ) -> np.ndarray:
        x = float(state[0])
        y = float(state[1])

        distance = math.hypot(
            x,
            y,
        )

        bearing = math.atan2(
            y,
            x,
        )

        return np.array(
            [distance, bearing],
            dtype=np.float64,
        )

    @staticmethod
    def measurement_jacobian(
        state: np.ndarray,
    ) -> np.ndarray:
        x = float(state[0])
        y = float(state[1])

        squared_range = (
            x * x + y * y
        )

        if squared_range < 1e-10:
            raise FusionValidationError(
                "Range-bearing Jacobian undefined near origin"
            )

        distance = math.sqrt(
            squared_range
        )

        jacobian = np.zeros(
            (2, 6),
            dtype=np.float64,
        )

        jacobian[0, 0] = x / distance
        jacobian[0, 1] = y / distance

        jacobian[1, 0] = -y / squared_range
        jacobian[1, 1] = x / squared_range

        return jacobian

    @staticmethod
    def normalize_angle(
        angle: float,
    ) -> float:
        return (
            angle + math.pi
        ) % (2 * math.pi) - math.pi

    def update_range_bearing(
        self,
        measurement: np.ndarray,
        covariance: np.ndarray,
    ) -> tuple[bool, float]:
        if not self.initialized:
            raise SensorFusionError(
                "Extended Kalman filter is not initialized"
            )

        measurement = np.asarray(
            measurement,
            dtype=np.float64,
        )

        if measurement.shape != (2,):
            raise FusionValidationError(
                "Range-bearing measurement must have 2 values"
            )

        covariance = validate_covariance(
            covariance,
            2,
        )

        expected = self.measurement_function(
            self.state
        )

        jacobian = self.measurement_jacobian(
            self.state
        )

        innovation = (
            measurement - expected
        )

        innovation[1] = self.normalize_angle(
            innovation[1]
        )

        innovation_covariance = (
            jacobian
            @ self.covariance
            @ jacobian.T
            + covariance
        )

        distance_squared = (
            self.mahalanobis_distance_squared(
                innovation,
                innovation_covariance,
            )
        )

        if (
            distance_squared
            > self.config.mahalanobis_threshold
        ):
            return False, distance_squared

        kalman_gain = np.linalg.solve(
            innovation_covariance,
            jacobian @ self.covariance,
        ).T

        self.state = (
            self.state
            + kalman_gain @ innovation
        )

        identity = np.eye(6)

        residual_projection = (
            identity
            - kalman_gain @ jacobian
        )

        self.covariance = (
            residual_projection
            @ self.covariance
            @ residual_projection.T
            + kalman_gain
            @ covariance
            @ kalman_gain.T
        )

        self.covariance = (
            self.covariance
            + self.covariance.T
        ) / 2.0

        return True, distance_squared


# ============================================================
# MEASUREMENT PROJECTION
# ============================================================

class FusionMeasurementProjector:

    @staticmethod
    def observation_matrix(
        measurement_type: MeasurementType,
    ) -> np.ndarray:
        if (
            measurement_type
            == MeasurementType.POSITION_3D
        ):
            matrix = np.zeros(
                (3, 6)
            )

            matrix[:, :3] = np.eye(3)

            return matrix

        if (
            measurement_type
            == MeasurementType.VELOCITY_3D
        ):
            matrix = np.zeros(
                (3, 6)
            )

            matrix[:, 3:] = np.eye(3)

            return matrix

        if (
            measurement_type
            == MeasurementType.POSITION_VELOCITY_3D
        ):
            return np.eye(6)

        raise FusionValidationError(
            f"Unsupported linear measurement type: "
            f"{measurement_type.value}"
        )

    @staticmethod
    def transform_measurement(
        measurement: SensorMeasurement,
        transform: SensorTransform,
    ) -> tuple[np.ndarray, np.ndarray]:
        rotation = transform.rotation

        if (
            measurement.measurement_type
            == MeasurementType.POSITION_3D
        ):
            values = (
                rotation @ measurement.values
                + transform.translation.as_array()
            )

            covariance = (
                rotation
                @ measurement.covariance
                @ rotation.T
            )

        elif (
            measurement.measurement_type
            == MeasurementType.VELOCITY_3D
        ):
            values = (
                rotation @ measurement.values
            )

            covariance = (
                rotation
                @ measurement.covariance
                @ rotation.T
            )

        elif (
            measurement.measurement_type
            == MeasurementType.POSITION_VELOCITY_3D
        ):
            values = np.concatenate(
                [
                    rotation @ measurement.values[:3]
                    + transform.translation.as_array(),
                    rotation @ measurement.values[3:],
                ]
            )

            transformation = np.zeros(
                (6, 6)
            )

            transformation[:3, :3] = rotation
            transformation[3:, 3:] = rotation

            covariance = (
                transformation
                @ measurement.covariance
                @ transformation.T
            )

        else:
            raise FusionTransformationError(
                "Measurement type requires a specialized "
                "nonlinear transform"
            )

        return values, covariance


# ============================================================
# FUSION ALERTS
# ============================================================

@dataclass(frozen=True)
class SensorFusionAlert:
    alert_id: str

    category: FusionAlertCategory
    severity: FusionAlertSeverity

    sensor_id: str | None

    description: str

    timestamp: datetime


# ============================================================
# FUSED STATE
# ============================================================

@dataclass(frozen=True)
class FusedSensorState:
    fusion_id: str

    tenant_id: str
    robot_id: str
    object_id: str

    timestamp: datetime
    frame_id: str

    position: FusionVector3
    velocity: FusionVector3

    covariance: np.ndarray

    quality: FusionQuality

    confidence: float

    contributing_sensor_ids: tuple[str, ...]

    accepted_measurements: int
    rejected_measurements: int

    alerts: tuple[SensorFusionAlert, ...]

    processing_latency_ms: float

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_covariance(
            self.covariance,
            6,
        )

        validate_probability(
            self.confidence,
            "confidence",
        )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# ============================================================
# FUSION QUALITY EVALUATOR
# ============================================================

class SensorFusionQualityEvaluator:

    def __init__(
        self,
        config: SensorFusionConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        *,
        covariance: np.ndarray,
        sensor_count: int,
        accepted_count: int,
        rejected_count: int,
        mean_sensor_confidence: float,
    ) -> tuple[FusionQuality, float]:
        uncertainty = float(
            np.trace(
                covariance[:3, :3]
            )
        )

        total = (
            accepted_count
            + rejected_count
        )

        acceptance_ratio = (
            accepted_count / total
            if total
            else 0.0
        )

        uncertainty_score = math.exp(
            -uncertainty
            / self.config.high_uncertainty_threshold
        )

        diversity_score = min(
            sensor_count / 3.0,
            1.0,
        )

        confidence = (
            0.40 * uncertainty_score
            + 0.25 * acceptance_ratio
            + 0.20 * mean_sensor_confidence
            + 0.15 * diversity_score
        )

        confidence = float(
            np.clip(
                confidence,
                0.0,
                1.0,
            )
        )

        if accepted_count == 0:
            return FusionQuality.UNAVAILABLE, 0.0

        if confidence >= 0.85:
            quality = FusionQuality.EXCELLENT

        elif confidence >= 0.70:
            quality = FusionQuality.GOOD

        elif confidence >= 0.50:
            quality = FusionQuality.DEGRADED

        else:
            quality = FusionQuality.POOR

        return quality, confidence


# ============================================================
# SENSOR DISAGREEMENT ANALYZER
# ============================================================

class SensorDisagreementAnalyzer:

    def __init__(
        self,
        config: SensorFusionConfig,
    ) -> None:
        self.config = config

    def analyze(
        self,
        measurements: Sequence[SensorMeasurement],
    ) -> tuple[SensorFusionAlert, ...]:
        position_measurements = [
            measurement
            for measurement in measurements
            if measurement.measurement_type in (
                MeasurementType.POSITION_3D,
                MeasurementType.POSITION_VELOCITY_3D,
            )
        ]

        alerts: list[SensorFusionAlert] = []

        for index, first in enumerate(
            position_measurements
        ):
            for second in position_measurements[
                index + 1:
            ]:
                if first.sensor_id == second.sensor_id:
                    continue

                if first.frame_id != second.frame_id:
                    continue

                distance = float(
                    np.linalg.norm(
                        first.values[:3]
                        - second.values[:3]
                    )
                )

                if (
                    distance
                    > self.config.sensor_disagreement_threshold_m
                ):
                    alerts.append(
                        SensorFusionAlert(
                            alert_id=generate_id(),
                            category=(
                                FusionAlertCategory
                                .SENSOR_DISAGREEMENT
                            ),
                            severity=(
                                FusionAlertSeverity.WARNING
                            ),
                            sensor_id=None,
                            description=(
                                f"{first.sensor_id} and "
                                f"{second.sensor_id} differ "
                                f"by {distance:.2f} m."
                            ),
                            timestamp=utc_now(),
                        )
                    )

        return tuple(alerts)


# ============================================================
# TRACK STATE
# ============================================================

@dataclass
class FusionTrack:
    tenant_id: str
    robot_id: str
    object_id: str
    frame_id: str

    filter: ExtendedKalmanFilter3D

    last_timestamp: datetime

    update_count: int = 0

    contributing_sensors: set[str] = field(
        default_factory=set
    )


# ============================================================
# SENSOR FUSION ENGINE
# ============================================================

class SensorFusionEngine:

    def __init__(
        self,
        config: SensorFusionConfig | None = None,
        transform_registry: SensorTransformRegistry | None = None,
    ) -> None:
        self.config = (
            config or SensorFusionConfig()
        )

        self.transform_registry = (
            transform_registry
            or SensorTransformRegistry()
        )

        self.synchronizer = SensorTimeSynchronizer(
            self.config
        )

        self.health_evaluator = SensorHealthEvaluator(
            self.config
        )

        self.quality_evaluator = (
            SensorFusionQualityEvaluator(
                self.config
            )
        )

        self.disagreement_analyzer = (
            SensorDisagreementAnalyzer(
                self.config
            )
        )

        self._tracks: dict[
            tuple[str, str, str],
            FusionTrack,
        ] = {}

        self._lock = threading.RLock()

    def _transform(
        self,
        measurement: SensorMeasurement,
        target_frame: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if measurement.frame_id == target_frame:
            return (
                measurement.values.copy(),
                measurement.covariance.copy(),
            )

        transform = self.transform_registry.resolve(
            tenant_id=measurement.tenant_id,
            robot_id=measurement.robot_id,
            source_frame=measurement.frame_id,
            target_frame=target_frame,
        )

        return FusionMeasurementProjector.transform_measurement(
            measurement,
            transform,
        )

    def _initialize_track(
        self,
        measurement: SensorMeasurement,
        values: np.ndarray,
        covariance: np.ndarray,
        target_frame: str,
    ) -> FusionTrack:
        if measurement.measurement_type not in (
            MeasurementType.POSITION_3D,
            MeasurementType.POSITION_VELOCITY_3D,
        ):
            raise FusionValidationError(
                "Track initialization requires position data"
            )

        kalman_filter = ExtendedKalmanFilter3D(
            self.config
        )

        initial_covariance = np.diag(
            [
                self.config.initial_position_variance
            ] * 3
            + [
                self.config.initial_velocity_variance
            ] * 3
        )

        initial_covariance[:3, :3] = covariance[
            :3,
            :3,
        ]

        velocity = None

        if (
            measurement.measurement_type
            == MeasurementType.POSITION_VELOCITY_3D
        ):
            velocity = values[3:]

            initial_covariance = covariance.copy()

        kalman_filter.initialize(
            position=values[:3],
            velocity=velocity,
            covariance=initial_covariance,
        )

        return FusionTrack(
            tenant_id=measurement.tenant_id,
            robot_id=measurement.robot_id,
            object_id=measurement.object_id,
            frame_id=target_frame,
            filter=kalman_filter,
            last_timestamp=measurement.timestamp,
            update_count=1,
            contributing_sensors={
                measurement.sensor_id
            },
        )

    def fuse(
        self,
        measurements: Sequence[SensorMeasurement],
        *,
        target_frame: str = "base_link",
        reference_timestamp: datetime | None = None,
    ) -> FusedSensorState:
        start = time.perf_counter()

        validate_identifier(
            target_frame,
            "target_frame",
        )

        if not measurements:
            raise FusionValidationError(
                "No sensor measurements supplied"
            )

        first = measurements[0]

        for measurement in measurements:
            if (
                measurement.tenant_id != first.tenant_id
                or measurement.robot_id != first.robot_id
                or measurement.object_id != first.object_id
            ):
                raise FusionValidationError(
                    "Cannot fuse measurements from "
                    "different tenants, robots, or objects"
                )

        synchronized = self.synchronizer.synchronize(
            measurements,
            reference_timestamp=reference_timestamp,
        )

        reference_timestamp = (
            synchronized.reference_timestamp
        )

        accepted_candidates = []

        rejected_count = len(
            synchronized.rejected_measurements
        )

        alerts: list[SensorFusionAlert] = []

        for measurement in synchronized.rejected_measurements:
            alerts.append(
                SensorFusionAlert(
                    alert_id=generate_id(),
                    category=(
                        FusionAlertCategory
                        .SYNCHRONIZATION_FAILURE
                    ),
                    severity=FusionAlertSeverity.WARNING,
                    sensor_id=measurement.sensor_id,
                    description=(
                        "Measurement rejected due to "
                        "timestamp synchronization."
                    ),
                    timestamp=reference_timestamp,
                )
            )

        for measurement in synchronized.measurements:
            health = self.health_evaluator.evaluate(
                measurement,
                reference_timestamp,
            )

            if health.status in (
                SensorHealth.STALE,
                SensorHealth.DEGRADED,
            ):
                rejected_count += 1

                alerts.append(
                    SensorFusionAlert(
                        alert_id=generate_id(),
                        category=(
                            FusionAlertCategory.SENSOR_STALE
                            if health.status == SensorHealth.STALE
                            else FusionAlertCategory.LOW_CONFIDENCE
                        ),
                        severity=FusionAlertSeverity.WARNING,
                        sensor_id=measurement.sensor_id,
                        description=health.explanation,
                        timestamp=reference_timestamp,
                    )
                )

                continue

            if measurement.measurement_type in (
                MeasurementType.ACCELERATION_3D,
                MeasurementType.RANGE_BEARING,
            ):
                # Acceleration requires an inertial prediction
                # model; range-bearing requires calibrated
                # nonlinear frame handling. Neither is silently
                # interpreted as Cartesian position.
                rejected_count += 1

                continue

            try:
                values, covariance = self._transform(
                    measurement,
                    target_frame,
                )

            except FusionTransformationError:
                rejected_count += 1

                alerts.append(
                    SensorFusionAlert(
                        alert_id=generate_id(),
                        category=(
                            FusionAlertCategory.MISSING_TRANSFORM
                        ),
                        severity=FusionAlertSeverity.WARNING,
                        sensor_id=measurement.sensor_id,
                        description=(
                            "Required sensor frame transform "
                            "is unavailable."
                        ),
                        timestamp=reference_timestamp,
                    )
                )

                continue

            # Lower-confidence measurements contribute
            # less information by increasing their covariance.
            effective_covariance = (
                covariance
                / max(
                    measurement.confidence,
                    1e-6,
                )
            )

            accepted_candidates.append(
                (
                    measurement,
                    values,
                    effective_covariance,
                )
            )

        if not accepted_candidates:
            raise SensorFusionError(
                "No valid measurements available for fusion"
            )

        accepted_candidates.sort(
            key=lambda item: (
                item[0].timestamp,
                item[0].sensor_id,
                item[0].measurement_id,
            )
        )

        key = (
            first.tenant_id,
            first.robot_id,
            first.object_id,
        )

        with self._lock:
            track = self._tracks.get(
                key
            )

            if track is None:
                if (
                    len(self._tracks)
                    >= self.config.maximum_tracks
                ):
                    raise FusionCapacityError(
                        "Maximum fusion tracks exceeded"
                    )

                initialization_index = next(
                    (
                        index
                        for index, item in enumerate(
                            accepted_candidates
                        )
                        if item[0].measurement_type in (
                            MeasurementType.POSITION_3D,
                            MeasurementType.POSITION_VELOCITY_3D,
                        )
                    ),
                    None,
                )

                if initialization_index is None:
                    raise SensorFusionError(
                        "Cannot initialize track without "
                        "position measurement"
                    )

                initial_measurement, initial_values, initial_covariance = (
                    accepted_candidates.pop(
                        initialization_index
                    )
                )

                track = self._initialize_track(
                    initial_measurement,
                    initial_values,
                    initial_covariance,
                    target_frame,
                )

                self._tracks[key] = track

                accepted_count = 1

                accepted_measurements = [
                    initial_measurement
                ]

            else:
                if track.frame_id != target_frame:
                    raise FusionTransformationError(
                        "Existing track uses a different "
                        "coordinate frame"
                    )

                if (
                    reference_timestamp
                    < track.last_timestamp
                ):
                    raise FusionSynchronizationError(
                        "Out-of-order fusion batch"
                    )

                accepted_count = 0

                accepted_measurements = []

            for measurement, values, covariance in accepted_candidates:
                if (
                    measurement.timestamp
                    < track.last_timestamp
                ):
                    rejected_count += 1
                    continue

                dt = (
                    measurement.timestamp
                    - track.last_timestamp
                ).total_seconds()

                if dt > 0:
                    track.filter.predict(
                        dt
                    )

                    track.last_timestamp = (
                        measurement.timestamp
                    )

                observation_matrix = (
                    FusionMeasurementProjector.observation_matrix(
                        measurement.measurement_type
                    )
                )

                accepted, distance_squared = (
                    track.filter.update(
                        values,
                        observation_matrix,
                        covariance,
                    )
                )

                if accepted:
                    accepted_count += 1

                    track.update_count += 1

                    track.contributing_sensors.add(
                        measurement.sensor_id
                    )

                    accepted_measurements.append(
                        measurement
                    )

                else:
                    rejected_count += 1

                    alerts.append(
                        SensorFusionAlert(
                            alert_id=generate_id(),
                            category=(
                                FusionAlertCategory.OUTLIER_REJECTED
                            ),
                            severity=FusionAlertSeverity.WARNING,
                            sensor_id=measurement.sensor_id,
                            description=(
                                "Measurement rejected by "
                                "Mahalanobis gating: "
                                f"{distance_squared:.3f}"
                            ),
                            timestamp=reference_timestamp,
                        )
                    )

            remaining_dt = (
                reference_timestamp
                - track.last_timestamp
            ).total_seconds()

            if remaining_dt > 0:
                track.filter.predict(
                    remaining_dt
                )

                track.last_timestamp = (
                    reference_timestamp
                )

            alerts.extend(
                self.disagreement_analyzer.analyze(
                    accepted_measurements
                )
            )

            uncertainty = float(
                np.trace(
                    track.filter.covariance[:3, :3]
                )
            )

            if (
                uncertainty
                > self.config.high_uncertainty_threshold
            ):
                alerts.append(
                    SensorFusionAlert(
                        alert_id=generate_id(),
                        category=(
                            FusionAlertCategory.HIGH_UNCERTAINTY
                        ),
                        severity=FusionAlertSeverity.WARNING,
                        sensor_id=None,
                        description=(
                            f"Position uncertainty trace "
                            f"is {uncertainty:.3f}."
                        ),
                        timestamp=reference_timestamp,
                    )
                )

            sensor_ids = tuple(
                sorted(
                    {
                        measurement.sensor_id
                        for measurement in accepted_measurements
                    }
                )
            )

            mean_confidence = (
                float(
                    np.mean(
                        [
                            measurement.confidence
                            for measurement in accepted_measurements
                        ]
                    )
                )
                if accepted_measurements
                else 0.0
            )

            quality, confidence = (
                self.quality_evaluator.evaluate(
                    covariance=track.filter.covariance,
                    sensor_count=len(sensor_ids),
                    accepted_count=accepted_count,
                    rejected_count=rejected_count,
                    mean_sensor_confidence=mean_confidence,
                )
            )

            result = FusedSensorState(
                fusion_id=generate_id(),
                tenant_id=first.tenant_id,
                robot_id=first.robot_id,
                object_id=first.object_id,
                timestamp=reference_timestamp,
                frame_id=target_frame,
                position=track.filter.position,
                velocity=track.filter.velocity,
                covariance=track.filter.covariance.copy(),
                quality=quality,
                confidence=confidence,
                contributing_sensor_ids=sensor_ids,
                accepted_measurements=accepted_count,
                rejected_measurements=rejected_count,
                alerts=tuple(alerts),
                processing_latency_ms=(
                    time.perf_counter() - start
                ) * 1000.0,
                metadata={
                    "filter": "constant_velocity_kalman",
                    "state_dimension": 6,
                    "position_uncertainty_trace": uncertainty,
                    "track_update_count": track.update_count,
                },
            )

            return result

    def reset_track(
        self,
        tenant_id: str,
        robot_id: str,
        object_id: str,
    ) -> None:
        key = (
            tenant_id,
            robot_id,
            object_id,
        )

        with self._lock:
            self._tracks.pop(
                key,
                None,
            )

    def track_count(self) -> int:
        with self._lock:
            return len(
                self._tracks
            )


# ============================================================
# FUSION ANALYTICS
# ============================================================

@dataclass(frozen=True)
class SensorFusionStatistics:
    fusion_count: int

    mean_processing_latency_ms: float

    mean_confidence: float

    mean_position_uncertainty: float

    accepted_measurements: int

    rejected_measurements: int

    quality_distribution: dict[str, int]

    alert_distribution: dict[str, int]


class SensorFusionAnalytics:

    @staticmethod
    def summarize(
        results: Sequence[FusedSensorState],
    ) -> SensorFusionStatistics:
        if not results:
            return SensorFusionStatistics(
                fusion_count=0,
                mean_processing_latency_ms=0.0,
                mean_confidence=0.0,
                mean_position_uncertainty=0.0,
                accepted_measurements=0,
                rejected_measurements=0,
                quality_distribution={},
                alert_distribution={},
            )

        quality_distribution = Counter(
            result.quality.value
            for result in results
        )

        alert_distribution = Counter(
            alert.category.value
            for result in results
            for alert in result.alerts
        )

        return SensorFusionStatistics(
            fusion_count=len(results),
            mean_processing_latency_ms=float(
                np.mean(
                    [
                        result.processing_latency_ms
                        for result in results
                    ]
                )
            ),
            mean_confidence=float(
                np.mean(
                    [
                        result.confidence
                        for result in results
                    ]
                )
            ),
            mean_position_uncertainty=float(
                np.mean(
                    [
                        np.trace(
                            result.covariance[:3, :3]
                        )
                        for result in results
                    ]
                )
            ),
            accepted_measurements=sum(
                result.accepted_measurements
                for result in results
            ),
            rejected_measurements=sum(
                result.rejected_measurements
                for result in results
            ),
            quality_distribution=dict(
                quality_distribution
            ),
            alert_distribution=dict(
                alert_distribution
            ),
        )


# ============================================================
# MULTI-TENANT FUSION SERVICE
# ============================================================

class RoboticsSensorFusionService:

    def __init__(
        self,
        config: SensorFusionConfig | None = None,
    ) -> None:
        self.config = (
            config or SensorFusionConfig()
        )

        self.transform_registry = (
            SensorTransformRegistry()
        )

        self.engine = SensorFusionEngine(
            self.config,
            self.transform_registry,
        )

        self._history: dict[
            tuple[str, str, str],
            deque[FusedSensorState],
        ] = {}

        self._lock = threading.RLock()

    @staticmethod
    def _authorize(
        access: FusionAccessContext,
        tenant_id: str,
        robot_id: str,
        permission: str,
    ) -> None:
        if access.tenant_id != tenant_id:
            raise FusionAuthorizationError(
                "Cross-tenant access denied"
            )

        if robot_id not in access.allowed_robot_ids:
            raise FusionAuthorizationError(
                "Robot access denied"
            )

        if not getattr(access, permission, False):
            raise FusionAuthorizationError(
                f"Missing permission: {permission}"
            )

    def register_transform(
        self,
        *,
        robot_id: str,
        transform: SensorTransform,
        access: FusionAccessContext,
    ) -> None:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_manage",
        )

        self.transform_registry.register(
            tenant_id=access.tenant_id,
            robot_id=robot_id,
            transform=transform,
        )

    def fuse(
        self,
        measurements: Sequence[SensorMeasurement],
        *,
        robot_id: str,
        access: FusionAccessContext,
        target_frame: str = "base_link",
        reference_timestamp: datetime | None = None,
    ) -> FusedSensorState:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_ingest",
        )

        if not measurements:
            raise FusionValidationError(
                "No measurements supplied"
            )

        for measurement in measurements:
            if (
                measurement.tenant_id != access.tenant_id
                or measurement.robot_id != robot_id
            ):
                raise FusionAuthorizationError(
                    "Measurement scope does not match "
                    "authorized tenant and robot"
                )

        result = self.engine.fuse(
            measurements,
            target_frame=target_frame,
            reference_timestamp=reference_timestamp,
        )

        key = (
            access.tenant_id,
            robot_id,
            result.object_id,
        )

        with self._lock:
            history = self._history.setdefault(
                key,
                deque(
                    maxlen=self.config.maximum_history
                ),
            )

            history.append(
                result
            )

        return result

    def latest(
        self,
        *,
        robot_id: str,
        object_id: str = "robot",
        access: FusionAccessContext,
    ) -> FusedSensorState | None:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_view",
        )

        key = (
            access.tenant_id,
            robot_id,
            object_id,
        )

        with self._lock:
            history = self._history.get(
                key
            )

            return (
                history[-1]
                if history
                else None
            )

    def history(
        self,
        *,
        robot_id: str,
        object_id: str = "robot",
        access: FusionAccessContext,
    ) -> tuple[FusedSensorState, ...]:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_view",
        )

        key = (
            access.tenant_id,
            robot_id,
            object_id,
        )

        with self._lock:
            return tuple(
                self._history.get(
                    key,
                    (),
                )
            )

    def statistics(
        self,
        *,
        robot_id: str,
        object_id: str = "robot",
        access: FusionAccessContext,
    ) -> SensorFusionStatistics:
        return SensorFusionAnalytics.summarize(
            self.history(
                robot_id=robot_id,
                object_id=object_id,
                access=access,
            )
        )

    def reset(
        self,
        *,
        robot_id: str,
        object_id: str = "robot",
        access: FusionAccessContext,
    ) -> None:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_manage",
        )

        key = (
            access.tenant_id,
            robot_id,
            object_id,
        )

        with self._lock:
            self._history.pop(
                key,
                None,
            )

            self.engine.reset_track(
                access.tenant_id,
                robot_id,
                object_id,
            )

    @staticmethod
    def export_json(
        result: FusedSensorState,
    ) -> str:
        return json.dumps(
            result.to_dict(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


# ============================================================
# SYNTHETIC SENSOR GENERATOR
# ============================================================

class SyntheticSensorFusionGenerator:

    @staticmethod
    def create_position_measurement(
        *,
        sensor_id: str,
        sensor_type: SensorType,
        position: tuple[float, float, float],
        variance: float = 0.1,
        confidence: float = 0.95,
        tenant_id: str = "tenant_alpha",
        robot_id: str = "ROBOT-001",
        object_id: str = "robot",
        frame_id: str = "base_link",
        timestamp: datetime | None = None,
    ) -> SensorMeasurement:
        if variance <= 0:
            raise FusionValidationError(
                "Variance must be positive"
            )

        return SensorMeasurement(
            measurement_id=generate_id(),
            tenant_id=tenant_id,
            robot_id=robot_id,
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            measurement_type=MeasurementType.POSITION_3D,
            timestamp=timestamp or utc_now(),
            frame_id=frame_id,
            values=np.array(
                position,
                dtype=np.float64,
            ),
            covariance=np.eye(3) * variance,
            confidence=confidence,
            object_id=object_id,
        )

    @staticmethod
    def create_velocity_measurement(
        *,
        sensor_id: str,
        sensor_type: SensorType,
        velocity: tuple[float, float, float],
        variance: float = 0.1,
        tenant_id: str = "tenant_alpha",
        robot_id: str = "ROBOT-001",
        object_id: str = "robot",
        timestamp: datetime | None = None,
    ) -> SensorMeasurement:
        return SensorMeasurement(
            measurement_id=generate_id(),
            tenant_id=tenant_id,
            robot_id=robot_id,
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            measurement_type=MeasurementType.VELOCITY_3D,
            timestamp=timestamp or utc_now(),
            frame_id="base_link",
            values=np.array(
                velocity,
                dtype=np.float64,
            ),
            covariance=np.eye(3) * variance,
            confidence=0.95,
            object_id=object_id,
        )

    @classmethod
    def multi_sensor_batch(
        cls,
        *,
        timestamp: datetime | None = None,
    ) -> tuple[SensorMeasurement, ...]:
        timestamp = timestamp or utc_now()

        return (
            cls.create_position_measurement(
                sensor_id="LIDAR-001",
                sensor_type=SensorType.LIDAR,
                position=(2.0, 1.0, 0.0),
                variance=0.05,
                timestamp=timestamp,
            ),
            cls.create_position_measurement(
                sensor_id="CAMERA-001",
                sensor_type=SensorType.CAMERA,
                position=(2.1, 0.95, 0.02),
                variance=0.15,
                timestamp=timestamp,
            ),
            cls.create_position_measurement(
                sensor_id="RADAR-001",
                sensor_type=SensorType.RADAR,
                position=(1.95, 1.05, 0.0),
                variance=0.20,
                timestamp=timestamp,
            ),
            cls.create_velocity_measurement(
                sensor_id="ODOM-001",
                sensor_type=SensorType.WHEEL_ODOMETRY,
                velocity=(0.5, 0.0, 0.0),
                variance=0.05,
                timestamp=timestamp,
            ),
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestSensorFusion(unittest.TestCase):

    def setUp(self) -> None:
        self.config = SensorFusionConfig()

        self.service = RoboticsSensorFusionService(
            self.config
        )

        self.access = FusionAccessContext(
            tenant_id="tenant_alpha",
            user_id="robotics_engineer",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
            }),
            can_ingest=True,
            can_view=True,
            can_manage=True,
        )

        self.timestamp = utc_now()

        self.measurements = (
            SyntheticSensorFusionGenerator.multi_sensor_batch(
                timestamp=self.timestamp
            )
        )

    def test_vector_distance(self) -> None:
        first = FusionVector3(
            0.0, 0.0, 0.0
        )

        second = FusionVector3(
            3.0, 4.0, 0.0
        )

        self.assertAlmostEqual(
            first.distance_to(second),
            5.0,
        )

    def test_covariance_validation(self) -> None:
        covariance = validate_covariance(
            np.eye(3),
            3,
        )

        self.assertEqual(
            covariance.shape,
            (3, 3),
        )

    def test_invalid_covariance(self) -> None:
        with self.assertRaises(
            FusionValidationError
        ):
            validate_covariance(
                np.zeros((3, 3)),
                3,
            )

    def test_transform_position(self) -> None:
        transform = SensorTransform(
            source_frame="camera",
            target_frame="base_link",
            rotation=np.eye(3),
            translation=FusionVector3(
                1.0, 2.0, 0.0
            ),
        )

        result = transform.transform_position(
            FusionVector3(
                1.0, 1.0, 0.0
            )
        )

        self.assertAlmostEqual(
            result.x,
            2.0,
        )

        self.assertAlmostEqual(
            result.y,
            3.0,
        )

    def test_inverse_transform(self) -> None:
        transform = SensorTransform(
            source_frame="camera",
            target_frame="base_link",
            rotation=np.eye(3),
            translation=FusionVector3(
                1.0, 0.0, 0.0
            ),
        )

        inverse = transform.inverse()

        self.assertAlmostEqual(
            inverse.translation.x,
            -1.0,
        )

    def test_transform_registry(self) -> None:
        transform = SensorTransform(
            source_frame="camera",
            target_frame="base_link",
            rotation=np.eye(3),
            translation=FusionVector3(
                1.0, 0.0, 0.0
            ),
        )

        self.service.register_transform(
            robot_id="ROBOT-001",
            transform=transform,
            access=self.access,
        )

        resolved = self.service.transform_registry.resolve(
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            source_frame="camera",
            target_frame="base_link",
        )

        self.assertEqual(
            resolved.target_frame,
            "base_link",
        )

    def test_measurement_validation(self) -> None:
        measurement = self.measurements[0]

        self.assertEqual(
            measurement.values.shape,
            (3,),
        )

    def test_time_synchronization(self) -> None:
        synchronized = SensorTimeSynchronizer(
            self.config
        ).synchronize(
            self.measurements
        )

        self.assertEqual(
            len(synchronized.measurements),
            4,
        )

    def test_stale_measurement(self) -> None:
        measurement = (
            SyntheticSensorFusionGenerator.create_position_measurement(
                sensor_id="LIDAR-OLD",
                sensor_type=SensorType.LIDAR,
                position=(1.0, 0.0, 0.0),
                timestamp=(
                    self.timestamp
                    - timedelta(seconds=2)
                ),
            )
        )

        report = SensorHealthEvaluator(
            self.config
        ).evaluate(
            measurement,
            self.timestamp,
        )

        self.assertEqual(
            report.status,
            SensorHealth.STALE,
        )

    def test_kalman_initialization(self) -> None:
        kalman_filter = LinearKalmanFilter3D(
            self.config
        )

        kalman_filter.initialize(
            np.array(
                [1.0, 2.0, 3.0]
            )
        )

        self.assertTrue(
            kalman_filter.initialized
        )

    def test_kalman_prediction(self) -> None:
        kalman_filter = LinearKalmanFilter3D(
            self.config
        )

        kalman_filter.initialize(
            np.array(
                [0.0, 0.0, 0.0]
            ),
            np.array(
                [1.0, 0.0, 0.0]
            ),
        )

        kalman_filter.predict(
            1.0
        )

        self.assertAlmostEqual(
            kalman_filter.position.x,
            1.0,
        )

    def test_kalman_update(self) -> None:
        kalman_filter = LinearKalmanFilter3D(
            self.config
        )

        kalman_filter.initialize(
            np.array(
                [0.0, 0.0, 0.0]
            )
        )

        accepted, _ = kalman_filter.update(
            measurement=np.array(
                [0.1, 0.0, 0.0]
            ),
            observation_matrix=(
                FusionMeasurementProjector.observation_matrix(
                    MeasurementType.POSITION_3D
                )
            ),
            measurement_covariance=np.eye(3) * 0.1,
        )

        self.assertTrue(
            accepted
        )

    def test_outlier_rejection(self) -> None:
        kalman_filter = LinearKalmanFilter3D(
            self.config
        )

        kalman_filter.initialize(
            np.zeros(3)
        )

        accepted, _ = kalman_filter.update(
            measurement=np.array(
                [100.0, 100.0, 100.0]
            ),
            observation_matrix=(
                FusionMeasurementProjector.observation_matrix(
                    MeasurementType.POSITION_3D
                )
            ),
            measurement_covariance=np.eye(3) * 0.1,
        )

        self.assertFalse(
            accepted
        )

    def test_extended_kalman_jacobian(self) -> None:
        state = np.array(
            [3.0, 4.0, 0.0, 0.0, 0.0, 0.0]
        )

        jacobian = (
            ExtendedKalmanFilter3D.measurement_jacobian(
                state
            )
        )

        self.assertEqual(
            jacobian.shape,
            (2, 6),
        )

    def test_angle_normalization(self) -> None:
        angle = (
            ExtendedKalmanFilter3D.normalize_angle(
                3 * math.pi
            )
        )

        self.assertAlmostEqual(
            angle,
            -math.pi,
        )

    def test_sensor_disagreement(self) -> None:
        first = (
            SyntheticSensorFusionGenerator.create_position_measurement(
                sensor_id="CAMERA-001",
                sensor_type=SensorType.CAMERA,
                position=(0.0, 0.0, 0.0),
                timestamp=self.timestamp,
            )
        )

        second = (
            SyntheticSensorFusionGenerator.create_position_measurement(
                sensor_id="LIDAR-001",
                sensor_type=SensorType.LIDAR,
                position=(10.0, 0.0, 0.0),
                timestamp=self.timestamp,
            )
        )

        alerts = SensorDisagreementAnalyzer(
            self.config
        ).analyze(
            [first, second]
        )

        self.assertGreater(
            len(alerts),
            0,
        )

    def test_full_sensor_fusion(self) -> None:
        result = self.service.fuse(
            self.measurements,
            robot_id="ROBOT-001",
            access=self.access,
            reference_timestamp=self.timestamp,
        )

        self.assertEqual(
            result.accepted_measurements,
            4,
        )

        self.assertAlmostEqual(
            result.position.x,
            2.0,
            delta=0.3,
        )

    def test_fusion_velocity(self) -> None:
        result = self.service.fuse(
            self.measurements,
            robot_id="ROBOT-001",
            access=self.access,
            reference_timestamp=self.timestamp,
        )

        self.assertGreater(
            result.velocity.x,
            0.0,
        )

    def test_fusion_history(self) -> None:
        self.service.fuse(
            self.measurements,
            robot_id="ROBOT-001",
            access=self.access,
            reference_timestamp=self.timestamp,
        )

        history = self.service.history(
            robot_id="ROBOT-001",
            access=self.access,
        )

        self.assertEqual(
            len(history),
            1,
        )

    def test_fusion_statistics(self) -> None:
        self.service.fuse(
            self.measurements,
            robot_id="ROBOT-001",
            access=self.access,
            reference_timestamp=self.timestamp,
        )

        statistics = self.service.statistics(
            robot_id="ROBOT-001",
            access=self.access,
        )

        self.assertEqual(
            statistics.fusion_count,
            1,
        )

    def test_tenant_isolation(self) -> None:
        with self.assertRaises(
            FusionAuthorizationError
        ):
            self.service._authorize(
                self.access,
                "tenant_beta",
                "ROBOT-001",
                "can_view",
            )

    def test_json_export(self) -> None:
        result = self.service.fuse(
            self.measurements,
            robot_id="ROBOT-001",
            access=self.access,
            reference_timestamp=self.timestamp,
        )

        payload = self.service.export_json(
            result
        )

        self.assertIn(
            "ROBOT-001",
            payload,
        )

    def test_reset(self) -> None:
        self.service.fuse(
            self.measurements,
            robot_id="ROBOT-001",
            access=self.access,
            reference_timestamp=self.timestamp,
        )

        self.service.reset(
            robot_id="ROBOT-001",
            access=self.access,
        )

        self.assertEqual(
            len(
                self.service.history(
                    robot_id="ROBOT-001",
                    access=self.access,
                )
            ),
            0,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print(
        "\nPHYSICAL AI & AUTONOMOUS ROBOTICS PLATFORM"
    )

    print(
        "Multi-Sensor Fusion Engine\n"
    )

    service = RoboticsSensorFusionService()

    access = FusionAccessContext(
        tenant_id="tenant_alpha",
        user_id="robotics_engineer",
        allowed_robot_ids=frozenset({
            "ROBOT-001",
        }),
        can_ingest=True,
        can_view=True,
        can_manage=True,
    )

    timestamp = utc_now()

    measurements = (
        SyntheticSensorFusionGenerator.multi_sensor_batch(
            timestamp=timestamp
        )
    )

    result = service.fuse(
        measurements,
        robot_id="ROBOT-001",
        access=access,
        target_frame="base_link",
        reference_timestamp=timestamp,
    )

    print("FUSED ROBOT STATE")

    print(
        service.export_json(
            result
        )
    )

    print("\nFUSION STATISTICS")

    print(
        json.dumps(
            json_safe(
                service.statistics(
                    robot_id="ROBOT-001",
                    access=access,
                )
            ),
            indent=2,
        )
    )

    print("\nRUNNING UNIT TESTS\n")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestSensorFusion
    )

    test_result = unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)

    if not test_result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
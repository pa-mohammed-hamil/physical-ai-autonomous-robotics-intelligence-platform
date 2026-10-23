"""Scene Understanding module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/perception/pose_estimation.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================

Module: pose_estimation.py

3D Robot, Object and Human Pose Estimation

Features
--------
- 2D and 3D pose representations
- Quaternion and rotation matrix conversions
- Euler angle conversions
- SE(3) rigid transformations
- Relative pose estimation
- Camera-based object pose estimation using PnP
- Optional ArUco marker pose estimation
- RGB-D 3D position estimation
- Human skeleton keypoint processing
- Human body orientation estimation
- Pose confidence and quality assessment
- Temporal pose smoothing
- Constant-velocity pose prediction
- Pose tracking across frames
- Pose history and analytics
- Coordinate frame validation
- Multi-tenant authorization
- JSON serialization
- Synthetic data generation
- Unit tests

Dependencies
------------
Python >= 3.11
numpy

Optional
--------
opencv-python
scipy

Safety
------
Pose estimates are advisory observations.
They must not directly authorize physical robot movement.
"""

from __future__ import annotations

import json
import math
import threading
import time
import unittest
import uuid

from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


# ============================================================
# EXCEPTIONS
# ============================================================

class PoseEstimationError(Exception):
    pass


class PoseValidationError(PoseEstimationError):
    pass


class PoseAuthorizationError(PoseEstimationError):
    pass


class PoseDependencyError(PoseEstimationError):
    pass


class PoseNotFoundError(PoseEstimationError):
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
        raise PoseValidationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > 128:
        raise PoseValidationError(
            f"{name} must contain 1-128 characters"
        )

    return value


def finite_float(
    value: Any,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise PoseValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PoseValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise PoseValidationError(
            f"{name} must be finite"
        )

    return result


def validate_confidence(
    value: float,
) -> float:
    confidence = finite_float(
        value,
        "confidence",
    )

    if not 0.0 <= confidence <= 1.0:
        raise PoseValidationError(
            "Confidence must be between 0 and 1"
        )

    return confidence


def validate_timestamp(
    value: datetime,
) -> datetime:
    if not isinstance(value, datetime):
        raise PoseValidationError(
            "Timestamp must be datetime"
        )

    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise PoseValidationError(
            "Timestamp must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


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

class PoseSource(str, Enum):
    CAMERA = "camera"
    RGBD = "rgbd"
    LIDAR = "lidar"
    IMU = "imu"
    VISUAL_ODOMETRY = "visual_odometry"
    SLAM = "slam"
    MARKER = "marker"
    HUMAN_KEYPOINTS = "human_keypoints"
    SIMULATION = "simulation"
    FUSED = "fused"


class PoseTargetType(str, Enum):
    ROBOT = "robot"
    HUMAN = "human"
    OBJECT = "object"
    ROBOT_PART = "robot_part"
    TOOL = "tool"
    MARKER = "marker"


class PoseQuality(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    DEGRADED = "degraded"
    POOR = "poor"
    INVALID = "invalid"


class PoseTrackingStatus(str, Enum):
    INITIALIZING = "initializing"
    TRACKING = "tracking"
    PREDICTED = "predicted"
    LOST = "lost"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PoseEstimationConfig:
    minimum_confidence: float = 0.40

    excellent_confidence: float = 0.90
    good_confidence: float = 0.75

    maximum_reprojection_error_px: float = 8.0

    minimum_pnp_correspondences: int = 4

    smoothing_alpha: float = 0.35

    maximum_prediction_seconds: float = 1.0

    maximum_position_jump_m: float = 3.0

    maximum_rotation_jump_degrees: float = 120.0

    maximum_history: int = 100

    minimum_human_keypoint_confidence: float = 0.40

    minimum_human_keypoints: int = 4

    def __post_init__(self) -> None:
        for name in (
            "minimum_confidence",
            "excellent_confidence",
            "good_confidence",
            "smoothing_alpha",
            "minimum_human_keypoint_confidence",
        ):
            validate_confidence(
                getattr(self, name)
            )

        if not (
            self.minimum_confidence
            <= self.good_confidence
            <= self.excellent_confidence
        ):
            raise PoseValidationError(
                "Confidence thresholds are inconsistent"
            )

        for name in (
            "maximum_reprojection_error_px",
            "maximum_prediction_seconds",
            "maximum_position_jump_m",
            "maximum_rotation_jump_degrees",
        ):
            if finite_float(
                getattr(self, name),
                name,
            ) <= 0:
                raise PoseValidationError(
                    f"{name} must be positive"
                )

        for name in (
            "minimum_pnp_correspondences",
            "maximum_history",
            "minimum_human_keypoints",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise PoseValidationError(
                    f"{name} must be a positive integer"
                )

        if self.minimum_pnp_correspondences < 4:
            raise PoseValidationError(
                "PnP requires at least four correspondences"
            )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class PoseAccessContext:
    tenant_id: str
    user_id: str

    allowed_robot_ids: frozenset[str] = frozenset()

    can_estimate: bool = False
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
# VECTOR AND QUATERNION
# ============================================================

@dataclass(frozen=True)
class Vector3:
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
    ) -> Vector3:
        array = np.asarray(
            values,
            dtype=np.float64,
        )

        if array.shape != (3,):
            raise PoseValidationError(
                "Vector3 requires three values"
            )

        return cls(
            float(array[0]),
            float(array[1]),
            float(array[2]),
        )

    def norm(self) -> float:
        return float(
            np.linalg.norm(
                self.as_array()
            )
        )

    def distance_to(
        self,
        other: Vector3,
    ) -> float:
        return float(
            np.linalg.norm(
                self.as_array()
                - other.as_array()
            )
        )


@dataclass(frozen=True)
class Quaternion:
    x: float
    y: float
    z: float
    w: float

    def __post_init__(self) -> None:
        values = np.array(
            [
                self.x,
                self.y,
                self.z,
                self.w,
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(values)):
            raise PoseValidationError(
                "Quaternion must contain finite values"
            )

        magnitude = float(
            np.linalg.norm(values)
        )

        if magnitude < 1e-12:
            raise PoseValidationError(
                "Quaternion magnitude cannot be zero"
            )

        values /= magnitude

        for name, value in zip(
            ("x", "y", "z", "w"),
            values,
        ):
            object.__setattr__(
                self,
                name,
                float(value),
            )

    @classmethod
    def identity(cls) -> Quaternion:
        return cls(
            0.0,
            0.0,
            0.0,
            1.0,
        )

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.x,
                self.y,
                self.z,
                self.w,
            ],
            dtype=np.float64,
        )

    def conjugate(self) -> Quaternion:
        return Quaternion(
            -self.x,
            -self.y,
            -self.z,
            self.w,
        )

    def multiply(
        self,
        other: Quaternion,
    ) -> Quaternion:
        x1, y1, z1, w1 = self.as_array()
        x2, y2, z2, w2 = other.as_array()

        return Quaternion(
            x=w1*x2 + x1*w2 + y1*z2 - z1*y2,
            y=w1*y2 - x1*z2 + y1*w2 + z1*x2,
            z=w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w=w1*w2 - x1*x2 - y1*y2 - z1*z2,
        )

    def to_rotation_matrix(self) -> np.ndarray:
        x, y, z, w = self.as_array()

        return np.array(
            [
                [
                    1 - 2*(y*y + z*z),
                    2*(x*y - z*w),
                    2*(x*z + y*w),
                ],
                [
                    2*(x*y + z*w),
                    1 - 2*(x*x + z*z),
                    2*(y*z - x*w),
                ],
                [
                    2*(x*z - y*w),
                    2*(y*z + x*w),
                    1 - 2*(x*x + y*y),
                ],
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_rotation_matrix(
        cls,
        rotation: np.ndarray,
    ) -> Quaternion:
        matrix = np.asarray(
            rotation,
            dtype=np.float64,
        )

        if matrix.shape != (3, 3):
            raise PoseValidationError(
                "Rotation matrix must be 3x3"
            )

        if not np.all(np.isfinite(matrix)):
            raise PoseValidationError(
                "Rotation matrix must be finite"
            )

        if not np.allclose(
            matrix.T @ matrix,
            np.eye(3),
            atol=1e-5,
        ):
            raise PoseValidationError(
                "Rotation matrix is not orthonormal"
            )

        if not np.isclose(
            np.linalg.det(matrix),
            1.0,
            atol=1e-5,
        ):
            raise PoseValidationError(
                "Rotation determinant must be +1"
            )

        trace = float(
            np.trace(matrix)
        )

        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2

            w = 0.25 * s
            x = (matrix[2, 1] - matrix[1, 2]) / s
            y = (matrix[0, 2] - matrix[2, 0]) / s
            z = (matrix[1, 0] - matrix[0, 1]) / s

        elif (
            matrix[0, 0] > matrix[1, 1]
            and matrix[0, 0] > matrix[2, 2]
        ):
            s = math.sqrt(
                max(
                    0.0,
                    1.0
                    + matrix[0, 0]
                    - matrix[1, 1]
                    - matrix[2, 2],
                )
            ) * 2

            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s

        elif matrix[1, 1] > matrix[2, 2]:
            s = math.sqrt(
                max(
                    0.0,
                    1.0
                    + matrix[1, 1]
                    - matrix[0, 0]
                    - matrix[2, 2],
                )
            ) * 2

            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s

        else:
            s = math.sqrt(
                max(
                    0.0,
                    1.0
                    + matrix[2, 2]
                    - matrix[0, 0]
                    - matrix[1, 1],
                )
            ) * 2

            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s

        return cls(x, y, z, w)

    @classmethod
    def from_euler(
        cls,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> Quaternion:
        roll = finite_float(roll, "roll")
        pitch = finite_float(pitch, "pitch")
        yaw = finite_float(yaw, "yaw")

        cr = math.cos(roll / 2)
        sr = math.sin(roll / 2)

        cp = math.cos(pitch / 2)
        sp = math.sin(pitch / 2)

        cy = math.cos(yaw / 2)
        sy = math.sin(yaw / 2)

        return cls(
            x=sr*cp*cy - cr*sp*sy,
            y=cr*sp*cy + sr*cp*sy,
            z=cr*cp*sy - sr*sp*cy,
            w=cr*cp*cy + sr*sp*sy,
        )

    def to_euler(self) -> tuple[float, float, float]:
        x, y, z, w = self.as_array()

        roll = math.atan2(
            2*(w*x + y*z),
            1 - 2*(x*x + y*y),
        )

        sin_pitch = 2*(w*y - z*x)

        pitch = math.asin(
            float(
                np.clip(
                    sin_pitch,
                    -1.0,
                    1.0,
                )
            )
        )

        yaw = math.atan2(
            2*(w*z + x*y),
            1 - 2*(y*y + z*z),
        )

        return roll, pitch, yaw

    def angular_distance(
        self,
        other: Quaternion,
    ) -> float:
        dot = abs(
            float(
                np.dot(
                    self.as_array(),
                    other.as_array(),
                )
            )
        )

        return 2 * math.acos(
            float(
                np.clip(dot, -1.0, 1.0)
            )
        )

    @staticmethod
    def slerp(
        first: Quaternion,
        second: Quaternion,
        alpha: float,
    ) -> Quaternion:
        alpha = finite_float(
            alpha,
            "alpha",
        )

        if not 0 <= alpha <= 1:
            raise PoseValidationError(
                "Interpolation alpha must be in [0, 1]"
            )

        q1 = first.as_array()
        q2 = second.as_array()

        dot = float(
            np.dot(q1, q2)
        )

        if dot < 0:
            q2 = -q2
            dot = -dot

        dot = float(
            np.clip(dot, -1.0, 1.0)
        )

        if dot > 0.9995:
            result = (
                q1 + alpha*(q2 - q1)
            )

            result /= np.linalg.norm(result)

            return Quaternion(*result)

        theta = math.acos(dot)
        sine = math.sin(theta)

        a = math.sin(
            (1 - alpha) * theta
        ) / sine

        b = math.sin(
            alpha * theta
        ) / sine

        return Quaternion(
            *(a*q1 + b*q2)
        )


# ============================================================
# POSE REPRESENTATION
# ============================================================

@dataclass(frozen=True)
class Pose3D:
    position: Vector3
    orientation: Quaternion

    frame_id: str = "map"

    def __post_init__(self) -> None:
        validate_identifier(
            self.frame_id,
            "frame_id",
        )

    @classmethod
    def identity(
        cls,
        frame_id: str = "map",
    ) -> Pose3D:
        return cls(
            position=Vector3(0.0, 0.0, 0.0),
            orientation=Quaternion.identity(),
            frame_id=frame_id,
        )

    def as_matrix(self) -> np.ndarray:
        matrix = np.eye(
            4,
            dtype=np.float64,
        )

        matrix[:3, :3] = (
            self.orientation.to_rotation_matrix()
        )

        matrix[:3, 3] = (
            self.position.as_array()
        )

        return matrix

    @classmethod
    def from_matrix(
        cls,
        matrix: np.ndarray,
        *,
        frame_id: str = "map",
    ) -> Pose3D:
        matrix = np.asarray(
            matrix,
            dtype=np.float64,
        )

        if matrix.shape != (4, 4):
            raise PoseValidationError(
                "Pose matrix must be 4x4"
            )

        if not np.allclose(
            matrix[3],
            [0.0, 0.0, 0.0, 1.0],
            atol=1e-6,
        ):
            raise PoseValidationError(
                "Invalid homogeneous pose matrix"
            )

        return cls(
            position=Vector3.from_array(
                matrix[:3, 3]
            ),
            orientation=Quaternion.from_rotation_matrix(
                matrix[:3, :3]
            ),
            frame_id=frame_id,
        )

    def inverse(self) -> Pose3D:
        rotation = (
            self.orientation.to_rotation_matrix()
        )

        inverse_rotation = rotation.T

        inverse_position = (
            -inverse_rotation
            @ self.position.as_array()
        )

        return Pose3D(
            position=Vector3.from_array(
                inverse_position
            ),
            orientation=self.orientation.conjugate(),
            frame_id=self.frame_id,
        )

    def compose(
        self,
        other: Pose3D,
    ) -> Pose3D:
        if self.frame_id != other.frame_id:
            raise PoseValidationError(
                "Pose frame mismatch"
            )

        result = (
            self.as_matrix()
            @ other.as_matrix()
        )

        return Pose3D.from_matrix(
            result,
            frame_id=self.frame_id,
        )

    def relative_to(
        self,
        reference: Pose3D,
    ) -> Pose3D:
        if self.frame_id != reference.frame_id:
            raise PoseValidationError(
                "Pose frame mismatch"
            )

        matrix = (
            np.linalg.inv(reference.as_matrix())
            @ self.as_matrix()
        )

        return Pose3D.from_matrix(
            matrix,
            frame_id=reference.frame_id,
        )

    def distance_to(
        self,
        other: Pose3D,
    ) -> float:
        if self.frame_id != other.frame_id:
            raise PoseValidationError(
                "Pose frame mismatch"
            )

        return self.position.distance_to(
            other.position
        )


# ============================================================
# CAMERA CALIBRATION
# ============================================================

@dataclass(frozen=True)
class PoseCameraCalibration:
    fx: float
    fy: float
    cx: float
    cy: float

    width: int
    height: int

    distortion: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "fx",
            "fy",
            "cx",
            "cy",
        ):
            finite_float(
                getattr(self, name),
                name,
            )

        if self.fx <= 0 or self.fy <= 0:
            raise PoseValidationError(
                "Camera focal lengths must be positive"
            )

        if self.width <= 0 or self.height <= 0:
            raise PoseValidationError(
                "Camera dimensions must be positive"
            )

        if len(self.distortion) not in (
            0, 4, 5, 8, 12, 14
        ):
            raise PoseValidationError(
                "Unsupported distortion coefficient count"
            )

        for value in self.distortion:
            finite_float(
                value,
                "distortion",
            )

    def camera_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def distortion_array(self) -> np.ndarray:
        if not self.distortion:
            return np.zeros(
                (5, 1),
                dtype=np.float64,
            )

        return np.asarray(
            self.distortion,
            dtype=np.float64,
        ).reshape(-1, 1)


# ============================================================
# POSE ESTIMATE
# ============================================================

@dataclass(frozen=True)
class PoseEstimate:
    estimate_id: str

    tenant_id: str
    robot_id: str
    sensor_id: str
    target_id: str

    target_type: PoseTargetType
    source: PoseSource

    timestamp: datetime

    pose: Pose3D

    confidence: float
    quality: PoseQuality

    reprojection_error_px: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "estimate_id",
            "tenant_id",
            "robot_id",
            "sensor_id",
            "target_id",
        ):
            validate_identifier(
                getattr(self, name),
                name,
            )

        validate_timestamp(self.timestamp)
        validate_confidence(self.confidence)

        if self.reprojection_error_px is not None:
            error = finite_float(
                self.reprojection_error_px,
                "reprojection_error_px",
            )

            if error < 0:
                raise PoseValidationError(
                    "Reprojection error cannot be negative"
                )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# ============================================================
# POSE QUALITY ASSESSMENT
# ============================================================

class PoseQualityEvaluator:

    def __init__(
        self,
        config: PoseEstimationConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        confidence: float,
        reprojection_error_px: float | None = None,
    ) -> PoseQuality:
        confidence = validate_confidence(
            confidence
        )

        if confidence < self.config.minimum_confidence:
            return PoseQuality.INVALID

        if reprojection_error_px is not None:
            error = finite_float(
                reprojection_error_px,
                "reprojection_error_px",
            )

            if error > self.config.maximum_reprojection_error_px:
                return PoseQuality.POOR

        if confidence >= self.config.excellent_confidence:
            return PoseQuality.EXCELLENT

        if confidence >= self.config.good_confidence:
            return PoseQuality.GOOD

        return PoseQuality.DEGRADED


# ============================================================
# PNP OBJECT POSE ESTIMATOR
# ============================================================

class PnPObjectPoseEstimator:

    def __init__(
        self,
        config: PoseEstimationConfig | None = None,
    ) -> None:
        self.config = (
            config or PoseEstimationConfig()
        )

    def estimate(
        self,
        *,
        object_points_3d: np.ndarray,
        image_points_2d: np.ndarray,
        calibration: PoseCameraCalibration,
        camera_frame: str = "camera_optical",
    ) -> tuple[Pose3D, float]:
        if cv2 is None:
            raise PoseDependencyError(
                "OpenCV is required for PnP pose estimation"
            )

        object_points = np.asarray(
            object_points_3d,
            dtype=np.float64,
        )

        image_points = np.asarray(
            image_points_2d,
            dtype=np.float64,
        )

        if (
            object_points.ndim != 2
            or object_points.shape[1] != 3
        ):
            raise PoseValidationError(
                "Object points must have shape N x 3"
            )

        if (
            image_points.ndim != 2
            or image_points.shape[1] != 2
        ):
            raise PoseValidationError(
                "Image points must have shape N x 2"
            )

        if len(object_points) != len(image_points):
            raise PoseValidationError(
                "2D and 3D correspondence counts differ"
            )

        if (
            len(object_points)
            < self.config.minimum_pnp_correspondences
        ):
            raise PoseValidationError(
                "Insufficient PnP correspondences"
            )

        if not (
            np.all(np.isfinite(object_points))
            and np.all(np.isfinite(image_points))
        ):
            raise PoseValidationError(
                "PnP points must be finite"
            )

        success, rotation_vector, translation_vector = (
            cv2.solvePnP(
                object_points,
                image_points,
                calibration.camera_matrix(),
                calibration.distortion_array(),
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        )

        if not success:
            raise PoseEstimationError(
                "PnP pose estimation failed"
            )

        rotation_matrix, _ = cv2.Rodrigues(
            rotation_vector
        )

        projected, _ = cv2.projectPoints(
            object_points,
            rotation_vector,
            translation_vector,
            calibration.camera_matrix(),
            calibration.distortion_array(),
        )

        projected = projected.reshape(-1, 2)

        errors = np.linalg.norm(
            projected - image_points,
            axis=1,
        )

        reprojection_error = float(
            np.sqrt(
                np.mean(errors ** 2)
            )
        )

        pose = Pose3D(
            position=Vector3.from_array(
                translation_vector.reshape(3)
            ),
            orientation=Quaternion.from_rotation_matrix(
                rotation_matrix
            ),
            frame_id=camera_frame,
        )

        return pose, reprojection_error


# ============================================================
# RGB-D POSITION ESTIMATOR
# ============================================================

class RGBDPositionEstimator:

    @staticmethod
    def pixel_to_3d(
        *,
        u: float,
        v: float,
        depth_m: float,
        calibration: PoseCameraCalibration,
    ) -> Vector3:
        u = finite_float(u, "u")
        v = finite_float(v, "v")

        depth = finite_float(
            depth_m,
            "depth_m",
        )

        if depth <= 0:
            raise PoseValidationError(
                "Depth must be positive"
            )

        x = (
            (u - calibration.cx)
            * depth / calibration.fx
        )

        y = (
            (v - calibration.cy)
            * depth / calibration.fy
        )

        return Vector3(
            x=float(x),
            y=float(y),
            z=depth,
        )

    @staticmethod
    def estimate_from_depth_patch(
        *,
        depth_map: np.ndarray,
        center_u: int,
        center_v: int,
        calibration: PoseCameraCalibration,
        patch_radius: int = 2,
        minimum_depth_m: float = 0.1,
        maximum_depth_m: float = 100.0,
    ) -> Vector3:
        depth = np.asarray(
            depth_map,
            dtype=np.float32,
        )

        if depth.shape != (
            calibration.height,
            calibration.width,
        ):
            raise PoseValidationError(
                "Depth map dimensions do not match calibration"
            )

        if patch_radius < 0:
            raise PoseValidationError(
                "Patch radius cannot be negative"
            )

        if not (
            0 <= center_u < calibration.width
            and 0 <= center_v < calibration.height
        ):
            raise PoseValidationError(
                "Pixel coordinates are outside image"
            )

        x1 = max(
            0,
            center_u - patch_radius,
        )

        x2 = min(
            calibration.width,
            center_u + patch_radius + 1,
        )

        y1 = max(
            0,
            center_v - patch_radius,
        )

        y2 = min(
            calibration.height,
            center_v + patch_radius + 1,
        )

        patch = depth[
            y1:y2,
            x1:x2,
        ]

        valid = patch[
            np.isfinite(patch)
            & (patch >= minimum_depth_m)
            & (patch <= maximum_depth_m)
        ]

        if len(valid) == 0:
            raise PoseEstimationError(
                "No valid depth samples near target"
            )

        depth_m = float(
            np.median(valid)
        )

        return RGBDPositionEstimator.pixel_to_3d(
            u=center_u,
            v=center_v,
            depth_m=depth_m,
            calibration=calibration,
        )


# ============================================================
# HUMAN KEYPOINT MODELS
# ============================================================

@dataclass(frozen=True)
class HumanKeypoint:
    name: str

    position: Vector3

    confidence: float

    def __post_init__(self) -> None:
        validate_identifier(
            self.name,
            "keypoint name",
        )

        validate_confidence(
            self.confidence
        )


@dataclass(frozen=True)
class HumanSkeleton:
    person_id: str

    frame_id: str

    timestamp: datetime

    keypoints: tuple[HumanKeypoint, ...]

    def __post_init__(self) -> None:
        validate_identifier(
            self.person_id,
            "person_id",
        )

        validate_identifier(
            self.frame_id,
            "frame_id",
        )

        validate_timestamp(
            self.timestamp
        )


# ============================================================
# HUMAN POSE ESTIMATOR
# ============================================================

class HumanPoseEstimator:

    def __init__(
        self,
        config: PoseEstimationConfig | None = None,
    ) -> None:
        self.config = (
            config or PoseEstimationConfig()
        )

    def estimate_body_pose(
        self,
        skeleton: HumanSkeleton,
    ) -> tuple[Pose3D, float]:
        valid = {
            keypoint.name: keypoint
            for keypoint in skeleton.keypoints
            if (
                keypoint.confidence
                >= self.config.minimum_human_keypoint_confidence
            )
        }

        if len(valid) < self.config.minimum_human_keypoints:
            raise PoseEstimationError(
                "Insufficient reliable human keypoints"
            )

        required = (
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
        )

        if not all(
            name in valid
            for name in required
        ):
            raise PoseEstimationError(
                "Shoulder and hip keypoints are required"
            )

        left_shoulder = valid[
            "left_shoulder"
        ].position.as_array()

        right_shoulder = valid[
            "right_shoulder"
        ].position.as_array()

        left_hip = valid[
            "left_hip"
        ].position.as_array()

        right_hip = valid[
            "right_hip"
        ].position.as_array()

        shoulder_center = (
            left_shoulder + right_shoulder
        ) / 2.0

        hip_center = (
            left_hip + right_hip
        ) / 2.0

        torso_center = (
            shoulder_center + hip_center
        ) / 2.0

        right_axis = (
            right_shoulder - left_shoulder
        )

        right_norm = np.linalg.norm(
            right_axis
        )

        if right_norm < 1e-6:
            raise PoseEstimationError(
                "Shoulder keypoints are degenerate"
            )

        right_axis /= right_norm

        up_axis = (
            shoulder_center - hip_center
        )

        up_axis -= (
            np.dot(up_axis, right_axis)
            * right_axis
        )

        up_norm = np.linalg.norm(
            up_axis
        )

        if up_norm < 1e-6:
            raise PoseEstimationError(
                "Torso orientation is degenerate"
            )

        up_axis /= up_norm

        forward_axis = np.cross(
            right_axis,
            up_axis,
        )

        forward_axis /= np.linalg.norm(
            forward_axis
        )

        rotation = np.column_stack(
            (
                forward_axis,
                right_axis,
                up_axis,
            )
        )

        orientation = Quaternion.from_rotation_matrix(
            rotation
        )

        confidence = float(
            np.mean(
                [
                    valid[name].confidence
                    for name in required
                ]
            )
        )

        pose = Pose3D(
            position=Vector3.from_array(
                torso_center
            ),
            orientation=orientation,
            frame_id=skeleton.frame_id,
        )

        return pose, confidence


# ============================================================
# TEMPORAL POSE FILTER
# ============================================================

class TemporalPoseFilter:

    def __init__(
        self,
        alpha: float = 0.35,
    ) -> None:
        alpha = finite_float(
            alpha,
            "alpha",
        )

        if not 0 <= alpha <= 1:
            raise PoseValidationError(
                "Alpha must be in [0, 1]"
            )

        self.alpha = alpha

        self._last_pose: Pose3D | None = None

    def update(
        self,
        pose: Pose3D,
    ) -> Pose3D:
        if self._last_pose is None:
            self._last_pose = pose
            return pose

        previous = self._last_pose

        if previous.frame_id != pose.frame_id:
            raise PoseValidationError(
                "Cannot smooth poses in different frames"
            )

        position = (
            (1 - self.alpha)
            * previous.position.as_array()
            + self.alpha
            * pose.position.as_array()
        )

        orientation = Quaternion.slerp(
            previous.orientation,
            pose.orientation,
            self.alpha,
        )

        filtered = Pose3D(
            position=Vector3.from_array(
                position
            ),
            orientation=orientation,
            frame_id=pose.frame_id,
        )

        self._last_pose = filtered

        return filtered

    def reset(self) -> None:
        self._last_pose = None


# ============================================================
# POSE MOTION ANALYZER
# ============================================================

@dataclass(frozen=True)
class PoseMotion:
    linear_velocity: Vector3

    speed_mps: float

    angular_speed_radps: float

    delta_time_seconds: float


class PoseMotionAnalyzer:

    @staticmethod
    def calculate(
        previous: PoseEstimate,
        current: PoseEstimate,
    ) -> PoseMotion:
        if (
            previous.target_id != current.target_id
            or previous.robot_id != current.robot_id
            or previous.tenant_id != current.tenant_id
        ):
            raise PoseValidationError(
                "Pose estimates refer to different targets"
            )

        if (
            previous.pose.frame_id
            != current.pose.frame_id
        ):
            raise PoseValidationError(
                "Pose coordinate frames differ"
            )

        dt = (
            validate_timestamp(current.timestamp)
            - validate_timestamp(previous.timestamp)
        ).total_seconds()

        if dt <= 0:
            raise PoseValidationError(
                "Pose timestamps must increase"
            )

        displacement = (
            current.pose.position.as_array()
            - previous.pose.position.as_array()
        )

        velocity = displacement / dt

        angular_distance = (
            previous.pose.orientation.angular_distance(
                current.pose.orientation
            )
        )

        return PoseMotion(
            linear_velocity=Vector3.from_array(
                velocity
            ),
            speed_mps=float(
                np.linalg.norm(velocity)
            ),
            angular_speed_radps=(
                angular_distance / dt
            ),
            delta_time_seconds=dt,
        )


# ============================================================
# POSE TRACKER
# ============================================================

class PoseTracker:

    def __init__(
        self,
        config: PoseEstimationConfig | None = None,
    ) -> None:
        self.config = (
            config or PoseEstimationConfig()
        )

        self._history: deque[PoseEstimate] = deque(
            maxlen=self.config.maximum_history
        )

        self._filter = TemporalPoseFilter(
            alpha=self.config.smoothing_alpha
        )

        self._status = (
            PoseTrackingStatus.INITIALIZING
        )

        self._lock = threading.RLock()

    @property
    def status(self) -> PoseTrackingStatus:
        with self._lock:
            return self._status

    def update(
        self,
        estimate: PoseEstimate,
    ) -> PoseEstimate:
        with self._lock:
            if self._history:
                previous = self._history[-1]

                if (
                    previous.target_id != estimate.target_id
                    or previous.tenant_id != estimate.tenant_id
                    or previous.robot_id != estimate.robot_id
                ):
                    raise PoseValidationError(
                        "Tracker identity mismatch"
                    )

                if (
                    estimate.timestamp
                    <= previous.timestamp
                ):
                    raise PoseValidationError(
                        "Pose timestamps must increase"
                    )

                if (
                    estimate.pose.frame_id
                    != previous.pose.frame_id
                ):
                    raise PoseValidationError(
                        "Tracker coordinate frame mismatch"
                    )

                jump = previous.pose.distance_to(
                    estimate.pose
                )

                if (
                    jump
                    > self.config.maximum_position_jump_m
                ):
                    self._status = (
                        PoseTrackingStatus.LOST
                    )

                    raise PoseEstimationError(
                        "Position jump exceeds threshold"
                    )

                rotation_jump = math.degrees(
                    previous.pose.orientation.angular_distance(
                        estimate.pose.orientation
                    )
                )

                if (
                    rotation_jump
                    > self.config.maximum_rotation_jump_degrees
                ):
                    self._status = (
                        PoseTrackingStatus.LOST
                    )

                    raise PoseEstimationError(
                        "Rotation jump exceeds threshold"
                    )

            filtered_pose = self._filter.update(
                estimate.pose
            )

            filtered_estimate = replace(
                estimate,
                pose=filtered_pose,
            )

            self._history.append(
                filtered_estimate
            )

            self._status = (
                PoseTrackingStatus.TRACKING
            )

            return filtered_estimate

    def predict(
        self,
        timestamp: datetime,
    ) -> PoseEstimate:
        timestamp = validate_timestamp(
            timestamp
        )

        with self._lock:
            if len(self._history) < 2:
                raise PoseEstimationError(
                    "At least two poses are needed for prediction"
                )

            previous = self._history[-2]
            latest = self._history[-1]

            prediction_dt = (
                timestamp - latest.timestamp
            ).total_seconds()

            if prediction_dt < 0:
                raise PoseValidationError(
                    "Prediction timestamp is in the past"
                )

            if (
                prediction_dt
                > self.config.maximum_prediction_seconds
            ):
                raise PoseEstimationError(
                    "Prediction horizon exceeds maximum"
                )

            motion = PoseMotionAnalyzer.calculate(
                previous,
                latest,
            )

            predicted_position = (
                latest.pose.position.as_array()
                + motion.linear_velocity.as_array()
                * prediction_dt
            )

            predicted_pose = Pose3D(
                position=Vector3.from_array(
                    predicted_position
                ),
                orientation=latest.pose.orientation,
                frame_id=latest.pose.frame_id,
            )

            predicted_confidence = (
                latest.confidence
                * math.exp(
                    -prediction_dt
                    / max(
                        self.config.maximum_prediction_seconds,
                        1e-6,
                    )
                )
            )

            self._status = (
                PoseTrackingStatus.PREDICTED
            )

            return replace(
                latest,
                estimate_id=generate_id(),
                timestamp=timestamp,
                pose=predicted_pose,
                confidence=predicted_confidence,
                quality=PoseQuality.DEGRADED,
                metadata={
                    **latest.metadata,
                    "predicted": True,
                    "prediction_horizon_seconds": prediction_dt,
                },
            )

    def latest(self) -> PoseEstimate | None:
        with self._lock:
            return (
                self._history[-1]
                if self._history
                else None
            )

    def history(self) -> tuple[PoseEstimate, ...]:
        with self._lock:
            return tuple(
                self._history
            )

    def reset(self) -> None:
        with self._lock:
            self._history.clear()
            self._filter.reset()

            self._status = (
                PoseTrackingStatus.INITIALIZING
            )


# ============================================================
# POSE ANALYTICS
# ============================================================

class PoseAnalytics:

    @staticmethod
    def summarize(
        estimates: Sequence[PoseEstimate],
    ) -> dict[str, Any]:
        if not estimates:
            return {
                "estimate_count": 0,
                "mean_confidence": None,
                "mean_position": None,
                "total_path_length_m": 0.0,
            }

        positions = np.array(
            [
                estimate.pose.position.as_array()
                for estimate in estimates
            ]
        )

        confidences = np.array(
            [
                estimate.confidence
                for estimate in estimates
            ]
        )

        path_length = 0.0

        for index in range(
            1,
            len(estimates),
        ):
            if (
                estimates[index].pose.frame_id
                != estimates[index - 1].pose.frame_id
            ):
                continue

            path_length += float(
                np.linalg.norm(
                    positions[index]
                    - positions[index - 1]
                )
            )

        return {
            "estimate_count": len(estimates),
            "mean_confidence": float(
                np.mean(confidences)
            ),
            "minimum_confidence": float(
                np.min(confidences)
            ),
            "maximum_confidence": float(
                np.max(confidences)
            ),
            "mean_position": (
                np.mean(
                    positions,
                    axis=0,
                ).tolist()
            ),
            "total_path_length_m": path_length,
        }


# ============================================================
# ROBOTICS POSE ESTIMATION SERVICE
# ============================================================

class RoboticsPoseEstimationService:

    def __init__(
        self,
        config: PoseEstimationConfig | None = None,
    ) -> None:
        self.config = (
            config or PoseEstimationConfig()
        )

        self.quality_evaluator = (
            PoseQualityEvaluator(
                self.config
            )
        )

        self.pnp_estimator = (
            PnPObjectPoseEstimator(
                self.config
            )
        )

        self.human_estimator = (
            HumanPoseEstimator(
                self.config
            )
        )

        self._trackers: dict[
            tuple[str, str, str],
            PoseTracker,
        ] = {}

        self._lock = threading.RLock()

    @staticmethod
    def _authorize(
        access: PoseAccessContext,
        tenant_id: str,
        robot_id: str,
        permission: str,
    ) -> None:
        if access.tenant_id != tenant_id:
            raise PoseAuthorizationError(
                "Cross-tenant access denied"
            )

        if robot_id not in access.allowed_robot_ids:
            raise PoseAuthorizationError(
                "Robot access denied"
            )

        if not getattr(access, permission, False):
            raise PoseAuthorizationError(
                f"Missing permission: {permission}"
            )

    def create_estimate(
        self,
        *,
        access: PoseAccessContext,
        robot_id: str,
        sensor_id: str,
        target_id: str,
        target_type: PoseTargetType,
        source: PoseSource,
        pose: Pose3D,
        confidence: float,
        timestamp: datetime | None = None,
        reprojection_error_px: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PoseEstimate:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_estimate",
        )

        confidence = validate_confidence(
            confidence
        )

        quality = self.quality_evaluator.evaluate(
            confidence,
            reprojection_error_px,
        )

        return PoseEstimate(
            estimate_id=generate_id(),
            tenant_id=access.tenant_id,
            robot_id=robot_id,
            sensor_id=sensor_id,
            target_id=target_id,
            target_type=target_type,
            source=source,
            timestamp=timestamp or utc_now(),
            pose=pose,
            confidence=confidence,
            quality=quality,
            reprojection_error_px=reprojection_error_px,
            metadata=metadata or {},
        )

    def estimate_object_pnp(
        self,
        *,
        access: PoseAccessContext,
        robot_id: str,
        sensor_id: str,
        target_id: str,
        object_points_3d: np.ndarray,
        image_points_2d: np.ndarray,
        calibration: PoseCameraCalibration,
        timestamp: datetime | None = None,
    ) -> PoseEstimate:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_estimate",
        )

        pose, error = self.pnp_estimator.estimate(
            object_points_3d=object_points_3d,
            image_points_2d=image_points_2d,
            calibration=calibration,
        )

        confidence = float(
            math.exp(
                -error
                / self.config.maximum_reprojection_error_px
            )
        )

        return self.create_estimate(
            access=access,
            robot_id=robot_id,
            sensor_id=sensor_id,
            target_id=target_id,
            target_type=PoseTargetType.OBJECT,
            source=PoseSource.CAMERA,
            pose=pose,
            confidence=confidence,
            timestamp=timestamp,
            reprojection_error_px=error,
        )

    def estimate_human(
        self,
        *,
        access: PoseAccessContext,
        robot_id: str,
        sensor_id: str,
        skeleton: HumanSkeleton,
    ) -> PoseEstimate:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_estimate",
        )

        pose, confidence = (
            self.human_estimator.estimate_body_pose(
                skeleton
            )
        )

        return self.create_estimate(
            access=access,
            robot_id=robot_id,
            sensor_id=sensor_id,
            target_id=skeleton.person_id,
            target_type=PoseTargetType.HUMAN,
            source=PoseSource.HUMAN_KEYPOINTS,
            pose=pose,
            confidence=confidence,
            timestamp=skeleton.timestamp,
        )

    def track(
        self,
        estimate: PoseEstimate,
        *,
        access: PoseAccessContext,
    ) -> PoseEstimate:
        self._authorize(
            access,
            estimate.tenant_id,
            estimate.robot_id,
            "can_estimate",
        )

        key = (
            estimate.tenant_id,
            estimate.robot_id,
            estimate.target_id,
        )

        with self._lock:
            tracker = self._trackers.setdefault(
                key,
                PoseTracker(
                    self.config
                ),
            )

        return tracker.update(
            estimate
        )

    def latest(
        self,
        *,
        robot_id: str,
        target_id: str,
        access: PoseAccessContext,
    ) -> PoseEstimate | None:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_view",
        )

        key = (
            access.tenant_id,
            robot_id,
            target_id,
        )

        with self._lock:
            tracker = self._trackers.get(
                key
            )

        return (
            tracker.latest()
            if tracker is not None
            else None
        )

    def history(
        self,
        *,
        robot_id: str,
        target_id: str,
        access: PoseAccessContext,
    ) -> tuple[PoseEstimate, ...]:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_view",
        )

        key = (
            access.tenant_id,
            robot_id,
            target_id,
        )

        with self._lock:
            tracker = self._trackers.get(
                key
            )

        return (
            tracker.history()
            if tracker is not None
            else ()
        )

    def predict(
        self,
        *,
        robot_id: str,
        target_id: str,
        timestamp: datetime,
        access: PoseAccessContext,
    ) -> PoseEstimate:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_view",
        )

        key = (
            access.tenant_id,
            robot_id,
            target_id,
        )

        with self._lock:
            tracker = self._trackers.get(
                key
            )

        if tracker is None:
            raise PoseNotFoundError(
                "Pose tracker not found"
            )

        return tracker.predict(
            timestamp
        )

    def reset_target(
        self,
        *,
        robot_id: str,
        target_id: str,
        access: PoseAccessContext,
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
            target_id,
        )

        with self._lock:
            tracker = self._trackers.pop(
                key,
                None,
            )

        if tracker is not None:
            tracker.reset()

    @staticmethod
    def export_json(
        estimate: PoseEstimate,
    ) -> str:
        return json.dumps(
            estimate.to_dict(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


# ============================================================
# SYNTHETIC POSE GENERATOR
# ============================================================

class SyntheticPoseGenerator:

    @staticmethod
    def moving_robot(
        *,
        frame_count: int = 10,
        step_m: float = 0.1,
        frame_id: str = "map",
    ) -> list[Pose3D]:
        if frame_count <= 0:
            raise PoseValidationError(
                "Frame count must be positive"
            )

        poses = []

        for index in range(frame_count):
            poses.append(
                Pose3D(
                    position=Vector3(
                        x=index * step_m,
                        y=0.0,
                        z=0.0,
                    ),
                    orientation=Quaternion.from_euler(
                        0.0,
                        0.0,
                        index * 0.02,
                    ),
                    frame_id=frame_id,
                )
            )

        return poses


# ============================================================
# UNIT TESTS
# ============================================================

class TestPoseEstimation(unittest.TestCase):

    def setUp(self) -> None:
        self.config = PoseEstimationConfig()

        self.service = (
            RoboticsPoseEstimationService(
                self.config
            )
        )

        self.access = PoseAccessContext(
            tenant_id="tenant_alpha",
            user_id="robotics_engineer",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
            }),
            can_estimate=True,
            can_view=True,
            can_manage=True,
        )

    def test_vector_distance(self) -> None:
        first = Vector3(0, 0, 0)
        second = Vector3(3, 4, 0)

        self.assertAlmostEqual(
            first.distance_to(second),
            5.0,
        )

    def test_quaternion_identity(self) -> None:
        quaternion = Quaternion.identity()

        np.testing.assert_allclose(
            quaternion.to_rotation_matrix(),
            np.eye(3),
            atol=1e-6,
        )

    def test_euler_round_trip(self) -> None:
        quaternion = Quaternion.from_euler(
            0.1,
            0.2,
            0.3,
        )

        roll, pitch, yaw = quaternion.to_euler()

        self.assertAlmostEqual(
            roll,
            0.1,
            places=5,
        )

        self.assertAlmostEqual(
            pitch,
            0.2,
            places=5,
        )

        self.assertAlmostEqual(
            yaw,
            0.3,
            places=5,
        )

    def test_rotation_matrix_round_trip(self) -> None:
        quaternion = Quaternion.from_euler(
            0.2,
            -0.3,
            0.5,
        )

        restored = Quaternion.from_rotation_matrix(
            quaternion.to_rotation_matrix()
        )

        self.assertLess(
            quaternion.angular_distance(
                restored
            ),
            1e-6,
        )

    def test_quaternion_slerp(self) -> None:
        first = Quaternion.identity()

        second = Quaternion.from_euler(
            0.0,
            0.0,
            math.pi,
        )

        midpoint = Quaternion.slerp(
            first,
            second,
            0.5,
        )

        self.assertAlmostEqual(
            midpoint.angular_distance(first),
            math.pi / 2,
            places=5,
        )

    def test_pose_matrix(self) -> None:
        pose = Pose3D(
            position=Vector3(1, 2, 3),
            orientation=Quaternion.identity(),
        )

        restored = Pose3D.from_matrix(
            pose.as_matrix()
        )

        self.assertAlmostEqual(
            restored.position.x,
            1.0,
        )

    def test_pose_inverse(self) -> None:
        pose = Pose3D(
            position=Vector3(1, 2, 3),
            orientation=Quaternion.from_euler(
                0.1,
                0.2,
                0.3,
            ),
        )

        identity = (
            pose.as_matrix()
            @ pose.inverse().as_matrix()
        )

        np.testing.assert_allclose(
            identity,
            np.eye(4),
            atol=1e-6,
        )

    def test_relative_pose(self) -> None:
        reference = Pose3D(
            position=Vector3(1, 0, 0),
            orientation=Quaternion.identity(),
        )

        target = Pose3D(
            position=Vector3(3, 0, 0),
            orientation=Quaternion.identity(),
        )

        relative = target.relative_to(
            reference
        )

        self.assertAlmostEqual(
            relative.position.x,
            2.0,
        )

    def test_rgbd_position(self) -> None:
        calibration = PoseCameraCalibration(
            fx=100,
            fy=100,
            cx=50,
            cy=50,
            width=100,
            height=100,
        )

        point = RGBDPositionEstimator.pixel_to_3d(
            u=50,
            v=50,
            depth_m=2.0,
            calibration=calibration,
        )

        self.assertAlmostEqual(
            point.z,
            2.0,
        )

        self.assertAlmostEqual(
            point.x,
            0.0,
        )

    def test_depth_patch(self) -> None:
        calibration = PoseCameraCalibration(
            fx=100,
            fy=100,
            cx=2,
            cy=2,
            width=5,
            height=5,
        )

        depth = np.full(
            (5, 5),
            2.0,
            dtype=np.float32,
        )

        point = (
            RGBDPositionEstimator.estimate_from_depth_patch(
                depth_map=depth,
                center_u=2,
                center_v=2,
                calibration=calibration,
            )
        )

        self.assertAlmostEqual(
            point.z,
            2.0,
        )

    def test_quality_evaluator(self) -> None:
        evaluator = PoseQualityEvaluator(
            self.config
        )

        self.assertEqual(
            evaluator.evaluate(0.95),
            PoseQuality.EXCELLENT,
        )

        self.assertEqual(
            evaluator.evaluate(0.1),
            PoseQuality.INVALID,
        )

    def test_pose_creation(self) -> None:
        estimate = self.service.create_estimate(
            access=self.access,
            robot_id="ROBOT-001",
            sensor_id="CAMERA-001",
            target_id="OBJECT-001",
            target_type=PoseTargetType.OBJECT,
            source=PoseSource.SIMULATION,
            pose=Pose3D.identity(),
            confidence=0.95,
        )

        self.assertEqual(
            estimate.quality,
            PoseQuality.EXCELLENT,
        )

    def test_temporal_filter(self) -> None:
        pose_filter = TemporalPoseFilter(
            alpha=0.5
        )

        first = Pose3D.identity()

        second = Pose3D(
            position=Vector3(2, 0, 0),
            orientation=Quaternion.identity(),
        )

        pose_filter.update(first)

        filtered = pose_filter.update(
            second
        )

        self.assertAlmostEqual(
            filtered.position.x,
            1.0,
        )

    def test_tracking(self) -> None:
        from datetime import timedelta

        timestamp = utc_now()

        for index in range(3):
            estimate = self.service.create_estimate(
                access=self.access,
                robot_id="ROBOT-001",
                sensor_id="CAMERA-001",
                target_id="OBJECT-001",
                target_type=PoseTargetType.OBJECT,
                source=PoseSource.SIMULATION,
                pose=Pose3D(
                    position=Vector3(
                        index * 0.1,
                        0.0,
                        0.0,
                    ),
                    orientation=Quaternion.identity(),
                ),
                confidence=0.95,
                timestamp=(
                    timestamp
                    + timedelta(
                        milliseconds=100 * index
                    )
                ),
            )

            self.service.track(
                estimate,
                access=self.access,
            )

        history = self.service.history(
            robot_id="ROBOT-001",
            target_id="OBJECT-001",
            access=self.access,
        )

        self.assertEqual(
            len(history),
            3,
        )

    def test_human_pose(self) -> None:
        keypoints = (
            HumanKeypoint(
                "left_shoulder",
                Vector3(-0.2, 0.0, 1.5),
                0.95,
            ),
            HumanKeypoint(
                "right_shoulder",
                Vector3(0.2, 0.0, 1.5),
                0.95,
            ),
            HumanKeypoint(
                "left_hip",
                Vector3(-0.15, 0.0, 1.0),
                0.95,
            ),
            HumanKeypoint(
                "right_hip",
                Vector3(0.15, 0.0, 1.0),
                0.95,
            ),
        )

        skeleton = HumanSkeleton(
            person_id="PERSON-001",
            frame_id="camera_optical",
            timestamp=utc_now(),
            keypoints=keypoints,
        )

        pose, confidence = (
            HumanPoseEstimator().estimate_body_pose(
                skeleton
            )
        )

        self.assertGreater(
            confidence,
            0.9,
        )

        self.assertAlmostEqual(
            pose.position.z,
            1.25,
        )

    def test_tenant_isolation(self) -> None:
        with self.assertRaises(
            PoseAuthorizationError
        ):
            self.service._authorize(
                self.access,
                "tenant_beta",
                "ROBOT-001",
                "can_view",
            )

    def test_json_export(self) -> None:
        estimate = self.service.create_estimate(
            access=self.access,
            robot_id="ROBOT-001",
            sensor_id="CAMERA-001",
            target_id="OBJECT-001",
            target_type=PoseTargetType.OBJECT,
            source=PoseSource.SIMULATION,
            pose=Pose3D.identity(),
            confidence=0.9,
        )

        payload = self.service.export_json(
            estimate
        )

        self.assertIn(
            "OBJECT-001",
            payload,
        )

    @unittest.skipIf(
        cv2 is None,
        "OpenCV is not installed",
    )
    def test_pnp_estimation(self) -> None:
        calibration = PoseCameraCalibration(
            fx=600.0,
            fy=600.0,
            cx=320.0,
            cy=240.0,
            width=640,
            height=480,
        )

        object_points = np.array(
            [
                [-0.5, -0.5, 0.0],
                [0.5, -0.5, 0.0],
                [0.5, 0.5, 0.0],
                [-0.5, 0.5, 0.0],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ],
            dtype=np.float64,
        )

        rotation_vector = np.array(
            [[0.1], [0.2], [0.05]],
            dtype=np.float64,
        )

        translation_vector = np.array(
            [[0.0], [0.0], [3.0]],
            dtype=np.float64,
        )

        image_points, _ = cv2.projectPoints(
            object_points,
            rotation_vector,
            translation_vector,
            calibration.camera_matrix(),
            calibration.distortion_array(),
        )

        image_points = image_points.reshape(
            -1,
            2,
        )

        pose, error = PnPObjectPoseEstimator().estimate(
            object_points_3d=object_points,
            image_points_2d=image_points,
            calibration=calibration,
        )

        self.assertLess(
            error,
            1e-3,
        )

        self.assertAlmostEqual(
            pose.position.z,
            3.0,
            places=2,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    from datetime import timedelta

    print(
        "\nPHYSICAL AI & AUTONOMOUS ROBOTICS PLATFORM"
    )

    print(
        "3D Pose Estimation and Tracking\n"
    )

    access = PoseAccessContext(
        tenant_id="tenant_alpha",
        user_id="robotics_engineer",
        allowed_robot_ids=frozenset({
            "ROBOT-001",
        }),
        can_estimate=True,
        can_view=True,
        can_manage=True,
    )

    service = RoboticsPoseEstimationService()

    poses = SyntheticPoseGenerator.moving_robot(
        frame_count=10,
        step_m=0.1,
    )

    start_time = utc_now()

    for index, pose in enumerate(poses):
        estimate = service.create_estimate(
            access=access,
            robot_id="ROBOT-001",
            sensor_id="SIMULATED-ODOMETRY",
            target_id="ROBOT-001",
            target_type=PoseTargetType.ROBOT,
            source=PoseSource.SIMULATION,
            pose=pose,
            confidence=0.98,
            timestamp=(
                start_time
                + timedelta(
                    milliseconds=index * 100
                )
            ),
        )

        service.track(
            estimate,
            access=access,
        )

    latest = service.latest(
        robot_id="ROBOT-001",
        target_id="ROBOT-001",
        access=access,
    )

    print("LATEST ROBOT POSE")

    print(
        service.export_json(
            latest
        )
    )

    history = service.history(
        robot_id="ROBOT-001",
        target_id="ROBOT-001",
        access=access,
    )

    print("\nPOSE TRACKING ANALYTICS")

    print(
        json.dumps(
            PoseAnalytics.summarize(
                history
            ),
            indent=2,
        )
    )

    predicted = service.predict(
        robot_id="ROBOT-001",
        target_id="ROBOT-001",
        timestamp=(
            history[-1].timestamp
            + timedelta(
                milliseconds=100
            )
        ),
        access=access,
    )

    print("\nPREDICTED POSE")

    print(
        service.export_json(
            predicted
        )
    )

    print("\nRUNNING UNIT TESTS\n")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestPoseEstimation
    )

    result = unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)

    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
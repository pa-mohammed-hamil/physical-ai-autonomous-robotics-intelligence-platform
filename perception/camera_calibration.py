"""Camera Calibration module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/perception/camera_calibration.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================

Module: camera_calibration.py

Camera calibration, intrinsic parameter management, lens distortion,
extrinsic transformations, reprojection analysis, calibration quality
evaluation, and multi-camera calibration registry.

Features:
    - Pinhole camera model
    - Camera intrinsic calibration
    - Brown-Conrady radial/tangential distortion
    - 3D-to-2D projection
    - Iterative pixel undistortion
    - Pixel-to-camera ray conversion
    - Camera extrinsic calibration
    - Rotation matrix validation
    - Quaternion and Euler rotation conversion
    - Homogeneous transformation matrices
    - Calibration target generation
    - Checkerboard object-point generation
    - Reprojection error calculation
    - Calibration quality assessment
    - Resolution-aware intrinsic scaling
    - Multi-camera calibration profiles
    - Camera-to-camera coordinate transforms
    - Calibration versioning
    - Multi-tenant camera isolation
    - Thread-safe calibration registry
    - JSON import/export
    - Optional OpenCV checkerboard calibration
    - Optional NumPy support through OpenCV
    - Audit logging
    - Unit tests
    - Demonstration

Dependencies:
    Core: Python 3.11+ standard library
    Optional: opencv-python, numpy

Installation:
    pip install opencv-python numpy

Safety:
    Calibration results are estimates, not certified safety measurements.
    Reprojection error alone does not establish physical measurement
    accuracy or safe robotic motion.

    This module never commands physical robots or actuators.
"""

from __future__ import annotations

import copy
import json
import math
import statistics
import threading
import unittest
import uuid

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# OPTIONAL DEPENDENCIES
# ============================================================

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None


# ============================================================
# EXCEPTIONS
# ============================================================

class CameraCalibrationError(Exception):
    """Base camera calibration exception."""


class CalibrationValidationError(CameraCalibrationError):
    """Invalid calibration parameters or observations."""


class CalibrationNotFoundError(CameraCalibrationError):
    """Requested calibration profile does not exist."""


class CalibrationAuthorizationError(CameraCalibrationError):
    """Camera calibration access denied."""


class CalibrationDependencyError(CameraCalibrationError):
    """Optional calibration dependency unavailable."""


class CalibrationQualityError(CameraCalibrationError):
    """Calibration quality does not meet configured limits."""


class CalibrationConflictError(CameraCalibrationError):
    """Calibration registry version conflict."""


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
        raise CalibrationValidationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > 128:
        raise CalibrationValidationError(
            f"{name} must contain 1-128 characters"
        )

    return value


def finite_float(
    value: Any,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise CalibrationValidationError(
            f"{name} must be numeric"
        )

    result = float(value)

    if not math.isfinite(result):
        raise CalibrationValidationError(
            f"{name} must be finite"
        )

    return result


def positive_float(
    value: Any,
    name: str,
) -> float:
    result = finite_float(value, name)

    if result <= 0:
        raise CalibrationValidationError(
            f"{name} must be positive"
        )

    return result


def aware_datetime(
    value: datetime,
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise CalibrationValidationError(
            "Timestamp must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

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

    if isinstance(value, (tuple, list, set, frozenset)):
        return [json_safe(item) for item in value]

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, (str, int, bool)) or value is None:
        return value

    return repr(value)


# ============================================================
# ENUMERATIONS
# ============================================================

class CameraModel(str, Enum):
    PINHOLE = "pinhole"
    BROWN_CONRADY = "brown_conrady"


class CalibrationTargetType(str, Enum):
    CHECKERBOARD = "checkerboard"
    CIRCLES_GRID = "circles_grid"
    CHARUCO = "charuco"


class CalibrationQuality(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    INVALID = "invalid"


class CalibrationStatus(str, Enum):
    DRAFT = "draft"
    VALIDATED = "validated"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class CameraCalibrationConfig:
    minimum_observation_frames: int = 8
    minimum_points_per_frame: int = 12

    excellent_rms_threshold: float = 0.25
    good_rms_threshold: float = 0.5
    acceptable_rms_threshold: float = 1.0

    maximum_acceptable_rms: float = 1.0
    maximum_undistortion_iterations: int = 30

    undistortion_tolerance: float = 1e-10

    maximum_registry_profiles: int = 1000
    maximum_profile_versions: int = 100

    require_quality_validation: bool = True

    def __post_init__(self) -> None:
        for name in (
            "minimum_observation_frames",
            "minimum_points_per_frame",
            "maximum_undistortion_iterations",
            "maximum_registry_profiles",
            "maximum_profile_versions",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise CalibrationValidationError(
                    f"{name} must be a positive integer"
                )

        for name in (
            "excellent_rms_threshold",
            "good_rms_threshold",
            "acceptable_rms_threshold",
            "maximum_acceptable_rms",
            "undistortion_tolerance",
        ):
            positive_float(
                getattr(self, name),
                name,
            )

        if not (
            self.excellent_rms_threshold
            <= self.good_rms_threshold
            <= self.acceptable_rms_threshold
        ):
            raise CalibrationValidationError(
                "Quality thresholds must be ascending"
            )


# ============================================================
# GEOMETRY
# ============================================================

@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

    def __post_init__(self) -> None:
        finite_float(self.x, "x")
        finite_float(self.y, "y")


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        finite_float(self.x, "x")
        finite_float(self.y, "y")
        finite_float(self.z, "z")


@dataclass(frozen=True)
class Quaternion:
    x: float
    y: float
    z: float
    w: float

    def __post_init__(self) -> None:
        for name in ("x", "y", "z", "w"):
            finite_float(getattr(self, name), name)

        norm = math.sqrt(
            self.x**2
            + self.y**2
            + self.z**2
            + self.w**2
        )

        if norm < 1e-12:
            raise CalibrationValidationError(
                "Quaternion norm cannot be zero"
            )

    def normalized(self) -> Quaternion:
        norm = math.sqrt(
            self.x**2
            + self.y**2
            + self.z**2
            + self.w**2
        )

        return Quaternion(
            x=self.x / norm,
            y=self.y / norm,
            z=self.z / norm,
            w=self.w / norm,
        )


# ============================================================
# MATRIX UTILITIES
# ============================================================

class MatrixOperations:

    @staticmethod
    def multiply(
        a: Sequence[Sequence[float]],
        b: Sequence[Sequence[float]],
    ) -> tuple[tuple[float, ...], ...]:
        if not a or not b:
            raise CalibrationValidationError(
                "Matrices cannot be empty"
            )

        columns_a = len(a[0])
        columns_b = len(b[0])

        if (
            any(len(row) != columns_a for row in a)
            or any(len(row) != columns_b for row in b)
        ):
            raise CalibrationValidationError(
                "Ragged matrices are unsupported"
            )

        if columns_a != len(b):
            raise CalibrationValidationError(
                "Incompatible matrix dimensions"
            )

        return tuple(
            tuple(
                sum(
                    a[i][k] * b[k][j]
                    for k in range(columns_a)
                )
                for j in range(columns_b)
            )
            for i in range(len(a))
        )

    @staticmethod
    def transpose(
        matrix: Sequence[Sequence[float]],
    ) -> tuple[tuple[float, ...], ...]:
        if not matrix:
            raise CalibrationValidationError(
                "Matrix cannot be empty"
            )

        width = len(matrix[0])

        if any(len(row) != width for row in matrix):
            raise CalibrationValidationError(
                "Ragged matrix"
            )

        return tuple(
            tuple(row[i] for row in matrix)
            for i in range(width)
        )

    @staticmethod
    def determinant_3x3(
        m: Sequence[Sequence[float]],
    ) -> float:
        if len(m) != 3 or any(len(row) != 3 for row in m):
            raise CalibrationValidationError(
                "Expected 3x3 matrix"
            )

        return (
            m[0][0] * (
                m[1][1] * m[2][2]
                - m[1][2] * m[2][1]
            )
            - m[0][1] * (
                m[1][0] * m[2][2]
                - m[1][2] * m[2][0]
            )
            + m[0][2] * (
                m[1][0] * m[2][1]
                - m[1][1] * m[2][0]
            )
        )


# ============================================================
# ROTATION MATRIX
# ============================================================

@dataclass(frozen=True)
class RotationMatrix:
    values: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]

    def __post_init__(self) -> None:
        matrix = self.values

        if (
            len(matrix) != 3
            or any(len(row) != 3 for row in matrix)
        ):
            raise CalibrationValidationError(
                "Rotation must be 3x3"
            )

        for row in matrix:
            for value in row:
                finite_float(value, "rotation element")

        product = MatrixOperations.multiply(
            matrix,
            MatrixOperations.transpose(matrix),
        )

        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0

                if abs(product[i][j] - expected) > 1e-5:
                    raise CalibrationValidationError(
                        "Rotation matrix is not orthonormal"
                    )

        determinant = MatrixOperations.determinant_3x3(
            matrix
        )

        if abs(determinant - 1.0) > 1e-5:
            raise CalibrationValidationError(
                "Rotation determinant must equal +1"
            )

    @staticmethod
    def identity() -> RotationMatrix:
        return RotationMatrix(
            (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        )

    @staticmethod
    def from_euler(
        roll: float,
        pitch: float,
        yaw: float,
    ) -> RotationMatrix:
        """
        Intrinsic ZYX convention:
            R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        """
        roll = finite_float(roll, "roll")
        pitch = finite_float(pitch, "pitch")
        yaw = finite_float(yaw, "yaw")

        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        return RotationMatrix(
            (
                (
                    cy * cp,
                    cy * sp * sr - sy * cr,
                    cy * sp * cr + sy * sr,
                ),
                (
                    sy * cp,
                    sy * sp * sr + cy * cr,
                    sy * sp * cr - cy * sr,
                ),
                (
                    -sp,
                    cp * sr,
                    cp * cr,
                ),
            )
        )

    @staticmethod
    def from_quaternion(
        quaternion: Quaternion,
    ) -> RotationMatrix:
        q = quaternion.normalized()

        x, y, z, w = q.x, q.y, q.z, q.w

        return RotationMatrix(
            (
                (
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                ),
                (
                    2 * (x * y + z * w),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - x * w),
                ),
                (
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x * x + y * y),
                ),
            )
        )

    def apply(self, point: Point3D) -> Point3D:
        m = self.values

        return Point3D(
            x=(
                m[0][0] * point.x
                + m[0][1] * point.y
                + m[0][2] * point.z
            ),
            y=(
                m[1][0] * point.x
                + m[1][1] * point.y
                + m[1][2] * point.z
            ),
            z=(
                m[2][0] * point.x
                + m[2][1] * point.y
                + m[2][2] * point.z
            ),
        )

    def inverse(self) -> RotationMatrix:
        return RotationMatrix(
            MatrixOperations.transpose(self.values)
        )


# ============================================================
# EXTRINSIC PARAMETERS
# ============================================================

@dataclass(frozen=True)
class CameraExtrinsics:
    """
    Transformation convention:

        p_target = R @ p_source + t

    The source and target frame names are mandatory.
    """

    source_frame: str
    target_frame: str

    rotation: RotationMatrix
    translation: Point3D

    def __post_init__(self) -> None:
        validate_identifier(
            self.source_frame,
            "source_frame",
        )

        validate_identifier(
            self.target_frame,
            "target_frame",
        )

    def transform_point(
        self,
        point: Point3D,
    ) -> Point3D:
        rotated = self.rotation.apply(point)

        return Point3D(
            rotated.x + self.translation.x,
            rotated.y + self.translation.y,
            rotated.z + self.translation.z,
        )

    def inverse(self) -> CameraExtrinsics:
        inverse_rotation = self.rotation.inverse()

        inverse_translation = inverse_rotation.apply(
            Point3D(
                -self.translation.x,
                -self.translation.y,
                -self.translation.z,
            )
        )

        return CameraExtrinsics(
            source_frame=self.target_frame,
            target_frame=self.source_frame,
            rotation=inverse_rotation,
            translation=inverse_translation,
        )

    def compose(
        self,
        next_transform: CameraExtrinsics,
    ) -> CameraExtrinsics:
        if self.target_frame != next_transform.source_frame:
            raise CalibrationValidationError(
                "Extrinsic frame mismatch"
            )

        rotation_values = MatrixOperations.multiply(
            next_transform.rotation.values,
            self.rotation.values,
        )

        rotated_translation = (
            next_transform.rotation.apply(
                self.translation
            )
        )

        translation = Point3D(
            rotated_translation.x
            + next_transform.translation.x,
            rotated_translation.y
            + next_transform.translation.y,
            rotated_translation.z
            + next_transform.translation.z,
        )

        return CameraExtrinsics(
            source_frame=self.source_frame,
            target_frame=next_transform.target_frame,
            rotation=RotationMatrix(rotation_values),
            translation=translation,
        )

    def homogeneous_matrix(
        self,
    ) -> tuple[tuple[float, ...], ...]:
        r = self.rotation.values
        t = self.translation

        return (
            (r[0][0], r[0][1], r[0][2], t.x),
            (r[1][0], r[1][1], r[1][2], t.y),
            (r[2][0], r[2][1], r[2][2], t.z),
            (0.0, 0.0, 0.0, 1.0),
        )


# ============================================================
# INTRINSIC PARAMETERS
# ============================================================

@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int

    fx: float
    fy: float

    cx: float
    cy: float

    skew: float = 0.0

    def __post_init__(self) -> None:
        for name in ("width", "height"):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise CalibrationValidationError(
                    f"{name} must be a positive integer"
                )

        positive_float(self.fx, "fx")
        positive_float(self.fy, "fy")

        finite_float(self.cx, "cx")
        finite_float(self.cy, "cy")
        finite_float(self.skew, "skew")

    def matrix(
        self,
    ) -> tuple[tuple[float, ...], ...]:
        return (
            (self.fx, self.skew, self.cx),
            (0.0, self.fy, self.cy),
            (0.0, 0.0, 1.0),
        )

    def scale(
        self,
        new_width: int,
        new_height: int,
    ) -> CameraIntrinsics:
        if new_width <= 0 or new_height <= 0:
            raise CalibrationValidationError(
                "Invalid target resolution"
            )

        sx = new_width / self.width
        sy = new_height / self.height

        return CameraIntrinsics(
            width=new_width,
            height=new_height,
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=(self.cx + 0.5) * sx - 0.5,
            cy=(self.cy + 0.5) * sy - 0.5,
            skew=self.skew * sx,
        )


# ============================================================
# DISTORTION PARAMETERS
# ============================================================

@dataclass(frozen=True)
class CameraDistortion:
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    def __post_init__(self) -> None:
        for name in ("k1", "k2", "p1", "p2", "k3"):
            finite_float(
                getattr(self, name),
                name,
            )

    def coefficients(
        self,
    ) -> tuple[float, ...]:
        return (
            self.k1,
            self.k2,
            self.p1,
            self.p2,
            self.k3,
        )

    def is_zero(self) -> bool:
        return all(
            value == 0.0
            for value in self.coefficients()
        )


# ============================================================
# CAMERA MODEL
# ============================================================

class PinholeCameraModel:

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        distortion: CameraDistortion | None = None,
        config: CameraCalibrationConfig | None = None,
    ) -> None:
        self.intrinsics = intrinsics
        self.distortion = (
            distortion or CameraDistortion()
        )
        self.config = (
            config or CameraCalibrationConfig()
        )

    def distort_normalized(
        self,
        x: float,
        y: float,
    ) -> Point2D:
        x = finite_float(x, "x")
        y = finite_float(y, "y")

        d = self.distortion

        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2

        radial = (
            1.0
            + d.k1 * r2
            + d.k2 * r4
            + d.k3 * r6
        )

        x_tangential = (
            2.0 * d.p1 * x * y
            + d.p2 * (r2 + 2.0 * x * x)
        )

        y_tangential = (
            d.p1 * (r2 + 2.0 * y * y)
            + 2.0 * d.p2 * x * y
        )

        return Point2D(
            x=x * radial + x_tangential,
            y=y * radial + y_tangential,
        )

    def normalized_to_pixel(
        self,
        point: Point2D,
    ) -> Point2D:
        k = self.intrinsics

        return Point2D(
            x=(
                k.fx * point.x
                + k.skew * point.y
                + k.cx
            ),
            y=k.fy * point.y + k.cy,
        )

    def pixel_to_normalized(
        self,
        pixel: Point2D,
    ) -> Point2D:
        k = self.intrinsics

        y = (pixel.y - k.cy) / k.fy

        x = (
            pixel.x
            - k.cx
            - k.skew * y
        ) / k.fx

        return Point2D(x, y)

    def project(
        self,
        point_camera: Point3D,
        apply_distortion: bool = True,
    ) -> Point2D:
        if point_camera.z <= 0:
            raise CalibrationValidationError(
                "Point must have positive camera depth"
            )

        x = point_camera.x / point_camera.z
        y = point_camera.y / point_camera.z

        normalized = (
            self.distort_normalized(x, y)
            if apply_distortion
            else Point2D(x, y)
        )

        return self.normalized_to_pixel(
            normalized
        )

    def project_points(
        self,
        points: Sequence[Point3D],
        apply_distortion: bool = True,
    ) -> tuple[Point2D, ...]:
        return tuple(
            self.project(
                point,
                apply_distortion,
            )
            for point in points
        )

    def undistort_normalized(
        self,
        distorted: Point2D,
    ) -> Point2D:
        if self.distortion.is_zero():
            return distorted

        x = distorted.x
        y = distorted.y

        d = self.distortion

        for _ in range(
            self.config.maximum_undistortion_iterations
        ):
            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r4 * r2

            radial = (
                1.0
                + d.k1 * r2
                + d.k2 * r4
                + d.k3 * r6
            )

            if (
                not math.isfinite(radial)
                or abs(radial) < 1e-12
            ):
                raise CalibrationValidationError(
                    "Undistortion became singular"
                )

            dx = (
                2.0 * d.p1 * x * y
                + d.p2 * (r2 + 2.0 * x * x)
            )

            dy = (
                d.p1 * (r2 + 2.0 * y * y)
                + 2.0 * d.p2 * x * y
            )

            next_x = (
                distorted.x - dx
            ) / radial

            next_y = (
                distorted.y - dy
            ) / radial

            if not (
                math.isfinite(next_x)
                and math.isfinite(next_y)
            ):
                raise CalibrationValidationError(
                    "Undistortion diverged"
                )

            delta = math.hypot(
                next_x - x,
                next_y - y,
            )

            x, y = next_x, next_y

            if delta <= self.config.undistortion_tolerance:
                return Point2D(x, y)

        residual = self.distort_normalized(x, y)

        error = math.hypot(
            residual.x - distorted.x,
            residual.y - distorted.y,
        )

        if error > 1e-6:
            raise CalibrationValidationError(
                "Undistortion did not converge"
            )

        return Point2D(x, y)

    def undistort_pixel(
        self,
        pixel: Point2D,
    ) -> Point2D:
        distorted = self.pixel_to_normalized(
            pixel
        )

        corrected = self.undistort_normalized(
            distorted
        )

        return self.normalized_to_pixel(
            corrected
        )

    def pixel_to_ray(
        self,
        pixel: Point2D,
        undistort: bool = True,
    ) -> Point3D:
        normalized = self.pixel_to_normalized(
            pixel
        )

        if undistort:
            normalized = self.undistort_normalized(
                normalized
            )

        length = math.sqrt(
            normalized.x**2
            + normalized.y**2
            + 1.0
        )

        return Point3D(
            normalized.x / length,
            normalized.y / length,
            1.0 / length,
        )

    def back_project(
        self,
        pixel: Point2D,
        depth_z: float,
    ) -> Point3D:
        depth_z = positive_float(
            depth_z,
            "depth_z",
        )

        normalized = self.undistort_normalized(
            self.pixel_to_normalized(pixel)
        )

        return Point3D(
            x=normalized.x * depth_z,
            y=normalized.y * depth_z,
            z=depth_z,
        )

    def is_inside_image(
        self,
        pixel: Point2D,
    ) -> bool:
        return (
            0 <= pixel.x < self.intrinsics.width
            and 0 <= pixel.y < self.intrinsics.height
        )


# ============================================================
# CHECKERBOARD TARGET
# ============================================================

@dataclass(frozen=True)
class CheckerboardTarget:
    columns: int
    rows: int
    square_size_m: float

    def __post_init__(self) -> None:
        if (
            self.columns < 2
            or self.rows < 2
        ):
            raise CalibrationValidationError(
                "Checkerboard requires at least 2x2 inner corners"
            )

        positive_float(
            self.square_size_m,
            "square_size_m",
        )

    @property
    def corner_count(self) -> int:
        return self.columns * self.rows

    def object_points(
        self,
    ) -> tuple[Point3D, ...]:
        return tuple(
            Point3D(
                x=column * self.square_size_m,
                y=row * self.square_size_m,
                z=0.0,
            )
            for row in range(self.rows)
            for column in range(self.columns)
        )


# ============================================================
# CALIBRATION OBSERVATIONS
# ============================================================

@dataclass(frozen=True)
class CalibrationObservation:
    observation_id: str

    object_points: tuple[Point3D, ...]
    image_points: tuple[Point2D, ...]

    image_width: int
    image_height: int

    timestamp: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_identifier(
            self.observation_id,
            "observation_id",
        )

        if len(self.object_points) != len(
            self.image_points
        ):
            raise CalibrationValidationError(
                "Object/image point counts differ"
            )

        if len(self.object_points) < 4:
            raise CalibrationValidationError(
                "At least four correspondences required"
            )

        if self.image_width <= 0 or self.image_height <= 0:
            raise CalibrationValidationError(
                "Invalid observation resolution"
            )

        aware_datetime(self.timestamp)


# ============================================================
# REPROJECTION ERROR
# ============================================================

@dataclass(frozen=True)
class ReprojectionMetrics:
    point_count: int

    rms_error_px: float
    mean_error_px: float
    median_error_px: float
    maximum_error_px: float

    per_point_errors_px: tuple[float, ...]


class ReprojectionErrorEvaluator:

    @staticmethod
    def evaluate(
        observed: Sequence[Point2D],
        predicted: Sequence[Point2D],
    ) -> ReprojectionMetrics:
        if not observed or len(observed) != len(predicted):
            raise CalibrationValidationError(
                "Reprojection arrays must be nonempty "
                "and equal length"
            )

        errors = tuple(
            math.hypot(
                actual.x - estimated.x,
                actual.y - estimated.y,
            )
            for actual, estimated
            in zip(observed, predicted)
        )

        rms = math.sqrt(
            sum(error**2 for error in errors)
            / len(errors)
        )

        return ReprojectionMetrics(
            point_count=len(errors),
            rms_error_px=rms,
            mean_error_px=statistics.mean(errors),
            median_error_px=statistics.median(errors),
            maximum_error_px=max(errors),
            per_point_errors_px=errors,
        )

    @staticmethod
    def evaluate_observation(
        observation: CalibrationObservation,
        camera: PinholeCameraModel,
        target_to_camera: CameraExtrinsics,
    ) -> ReprojectionMetrics:
        if target_to_camera.target_frame != "camera":
            raise CalibrationValidationError(
                "Target transform must terminate "
                "in the camera frame"
            )

        projected = []

        for point in observation.object_points:
            camera_point = (
                target_to_camera.transform_point(point)
            )

            projected.append(
                camera.project(camera_point)
            )

        return ReprojectionErrorEvaluator.evaluate(
            observation.image_points,
            projected,
        )


# ============================================================
# QUALITY EVALUATION
# ============================================================

@dataclass(frozen=True)
class CalibrationQualityReport:
    quality: CalibrationQuality

    rms_error_px: float

    accepted: bool

    warnings: tuple[str, ...]

    evaluated_at: datetime = field(
        default_factory=utc_now
    )


class CalibrationQualityEvaluator:

    def __init__(
        self,
        config: CameraCalibrationConfig | None = None,
    ) -> None:
        self.config = (
            config or CameraCalibrationConfig()
        )

    def evaluate(
        self,
        rms_error_px: float,
        observation_count: int,
        point_count: int,
    ) -> CalibrationQualityReport:
        rms = finite_float(
            rms_error_px,
            "rms_error_px",
        )

        if rms < 0:
            raise CalibrationValidationError(
                "RMS error cannot be negative"
            )

        warnings = []

        if (
            observation_count
            < self.config.minimum_observation_frames
        ):
            warnings.append(
                "Insufficient calibration frames"
            )

        if (
            point_count
            < observation_count
            * self.config.minimum_points_per_frame
        ):
            warnings.append(
                "Insufficient calibration point coverage"
            )

        if rms <= self.config.excellent_rms_threshold:
            quality = CalibrationQuality.EXCELLENT

        elif rms <= self.config.good_rms_threshold:
            quality = CalibrationQuality.GOOD

        elif rms <= self.config.acceptable_rms_threshold:
            quality = CalibrationQuality.ACCEPTABLE

        else:
            quality = CalibrationQuality.POOR

        if rms > self.config.maximum_acceptable_rms:
            warnings.append(
                "Reprojection RMS exceeds configured limit"
            )

        accepted = (
            not warnings
            and rms <= self.config.maximum_acceptable_rms
        )

        return CalibrationQualityReport(
            quality=quality,
            rms_error_px=rms,
            accepted=accepted,
            warnings=tuple(warnings),
        )


# ============================================================
# CALIBRATION PROFILE
# ============================================================

@dataclass(frozen=True)
class CameraCalibrationProfile:
    calibration_id: str

    tenant_id: str
    camera_id: str

    version: int

    model: CameraModel

    intrinsics: CameraIntrinsics
    distortion: CameraDistortion

    camera_frame: str = "camera"

    extrinsics: CameraExtrinsics | None = None

    rms_reprojection_error_px: float | None = None

    observation_count: int = 0
    calibration_point_count: int = 0

    status: CalibrationStatus = CalibrationStatus.DRAFT

    created_at: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_identifier(
            self.calibration_id,
            "calibration_id",
        )

        validate_identifier(
            self.tenant_id,
            "tenant_id",
        )

        validate_identifier(
            self.camera_id,
            "camera_id",
        )

        validate_identifier(
            self.camera_frame,
            "camera_frame",
        )

        if self.version <= 0:
            raise CalibrationValidationError(
                "Version must be positive"
            )

        if self.rms_reprojection_error_px is not None:
            rms = finite_float(
                self.rms_reprojection_error_px,
                "rms_reprojection_error_px",
            )

            if rms < 0:
                raise CalibrationValidationError(
                    "RMS cannot be negative"
                )

        if (
            self.model == CameraModel.PINHOLE
            and not self.distortion.is_zero()
        ):
            raise CalibrationValidationError(
                "PINHOLE profile must have zero distortion"
            )

        if self.extrinsics is not None:
            if (
                self.extrinsics.source_frame
                != self.camera_frame
            ):
                raise CalibrationValidationError(
                    "Extrinsic source frame must equal "
                    "camera frame"
                )

        aware_datetime(self.created_at)


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class CalibrationAccessContext:
    tenant_id: str
    user_id: str

    allowed_camera_ids: frozenset[str] = frozenset()

    can_read: bool = False
    can_write: bool = False
    can_validate: bool = False

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id, "tenant_id")
        validate_identifier(self.user_id, "user_id")


# ============================================================
# CALIBRATION REGISTRY
# ============================================================

class CameraCalibrationRegistry:

    def __init__(
        self,
        config: CameraCalibrationConfig | None = None,
    ) -> None:
        self.config = (
            config or CameraCalibrationConfig()
        )

        self._profiles: dict[
            tuple[str, str],
            list[CameraCalibrationProfile],
        ] = {}

        self._audit: list[dict[str, Any]] = []

        self._lock = threading.RLock()

        self._quality = CalibrationQualityEvaluator(
            self.config
        )

    @staticmethod
    def _authorize(
        access: CalibrationAccessContext,
        camera_id: str,
        permission: str,
    ) -> None:
        if not getattr(access, permission, False):
            raise CalibrationAuthorizationError(
                f"Missing permission: {permission}"
            )

        if camera_id not in access.allowed_camera_ids:
            raise CalibrationAuthorizationError(
                "Camera access denied"
            )

    def _audit_event(
        self,
        access: CalibrationAccessContext,
        action: str,
        details: Mapping[str, Any],
    ) -> None:
        self._audit.append({
            "event_id": generate_id(),
            "tenant_id": access.tenant_id,
            "user_id": access.user_id,
            "action": action,
            "details": json_safe(details),
            "timestamp": utc_now().isoformat(),
        })

    def register(
        self,
        profile: CameraCalibrationProfile,
        *,
        access: CalibrationAccessContext,
    ) -> CameraCalibrationProfile:
        self._authorize(
            access,
            profile.camera_id,
            "can_write",
        )

        if profile.tenant_id != access.tenant_id:
            raise CalibrationAuthorizationError(
                "Cross-tenant registration denied"
            )

        key = (
            profile.tenant_id,
            profile.camera_id,
        )

        with self._lock:
            if (
                key not in self._profiles
                and len(self._profiles)
                >= self.config.maximum_registry_profiles
            ):
                raise CalibrationConflictError(
                    "Registry capacity reached"
                )

            history = self._profiles.setdefault(
                key,
                [],
            )

            if (
                len(history)
                >= self.config.maximum_profile_versions
            ):
                raise CalibrationConflictError(
                    "Maximum calibration versions reached"
                )

            expected_version = len(history) + 1

            if profile.version != expected_version:
                raise CalibrationConflictError(
                    f"Expected version {expected_version}"
                )

            if history:
                history[-1] = replace(
                    history[-1],
                    status=CalibrationStatus.SUPERSEDED,
                )

            stored = copy.deepcopy(profile)

            history.append(stored)

            self._audit_event(
                access,
                "calibration_registered",
                {
                    "camera_id": profile.camera_id,
                    "version": profile.version,
                },
            )

            return copy.deepcopy(stored)

    def latest(
        self,
        camera_id: str,
        *,
        access: CalibrationAccessContext,
    ) -> CameraCalibrationProfile:
        self._authorize(
            access,
            camera_id,
            "can_read",
        )

        key = (
            access.tenant_id,
            camera_id,
        )

        with self._lock:
            history = self._profiles.get(key)

            if not history:
                raise CalibrationNotFoundError(
                    "Camera calibration not found"
                )

            return copy.deepcopy(history[-1])

    def get_version(
        self,
        camera_id: str,
        version: int,
        *,
        access: CalibrationAccessContext,
    ) -> CameraCalibrationProfile:
        self._authorize(
            access,
            camera_id,
            "can_read",
        )

        key = (
            access.tenant_id,
            camera_id,
        )

        with self._lock:
            history = self._profiles.get(key, [])

            for profile in history:
                if profile.version == version:
                    return copy.deepcopy(profile)

        raise CalibrationNotFoundError(
            "Calibration version not found"
        )

    def validate_latest(
        self,
        camera_id: str,
        *,
        access: CalibrationAccessContext,
    ) -> CalibrationQualityReport:
        self._authorize(
            access,
            camera_id,
            "can_validate",
        )

        key = (
            access.tenant_id,
            camera_id,
        )

        with self._lock:
            history = self._profiles.get(key)

            if not history:
                raise CalibrationNotFoundError(
                    "Camera calibration not found"
                )

            profile = history[-1]

            if profile.rms_reprojection_error_px is None:
                raise CalibrationQualityError(
                    "RMS reprojection error unavailable"
                )

            report = self._quality.evaluate(
                rms_error_px=(
                    profile.rms_reprojection_error_px
                ),
                observation_count=(
                    profile.observation_count
                ),
                point_count=(
                    profile.calibration_point_count
                ),
            )

            history[-1] = replace(
                profile,
                status=(
                    CalibrationStatus.VALIDATED
                    if report.accepted
                    else CalibrationStatus.REJECTED
                ),
            )

            self._audit_event(
                access,
                "calibration_validated",
                {
                    "camera_id": camera_id,
                    "version": profile.version,
                    "accepted": report.accepted,
                    "quality": report.quality.value,
                },
            )

            return report

    def history(
        self,
        camera_id: str,
        *,
        access: CalibrationAccessContext,
    ) -> tuple[CameraCalibrationProfile, ...]:
        self._authorize(
            access,
            camera_id,
            "can_read",
        )

        key = (
            access.tenant_id,
            camera_id,
        )

        with self._lock:
            return tuple(
                copy.deepcopy(
                    self._profiles.get(key, [])
                )
            )

    def audit_history(
        self,
        *,
        access: CalibrationAccessContext,
    ) -> list[dict[str, Any]]:
        if not access.can_read:
            raise CalibrationAuthorizationError(
                "Read permission required"
            )

        with self._lock:
            return copy.deepcopy([
                event
                for event in self._audit
                if event["tenant_id"] == access.tenant_id
                and event["details"].get("camera_id")
                in access.allowed_camera_ids
            ])


# ============================================================
# OPTIONAL OPENCV CALIBRATION
# ============================================================

@dataclass(frozen=True)
class OpenCVCalibrationResult:
    intrinsics: CameraIntrinsics
    distortion: CameraDistortion

    rms_error_px: float

    observation_count: int
    point_count: int

    target_poses: tuple[CameraExtrinsics, ...]

    quality_report: CalibrationQualityReport


class OpenCVCheckerboardCalibrator:

    def __init__(
        self,
        config: CameraCalibrationConfig | None = None,
    ) -> None:
        self.config = (
            config or CameraCalibrationConfig()
        )

    @staticmethod
    def _require_dependencies() -> None:
        if cv2 is None or np is None:
            raise CalibrationDependencyError(
                "OpenCV calibration requires "
                "'opencv-python' and 'numpy'"
            )

    def detect_checkerboard(
        self,
        image: Any,
        target: CheckerboardTarget,
    ) -> tuple[Point2D, ...] | None:
        self._require_dependencies()

        if image is None:
            raise CalibrationValidationError(
                "Image cannot be None"
            )

        if len(image.shape) == 3:
            grayscale = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )
        elif len(image.shape) == 2:
            grayscale = image
        else:
            raise CalibrationValidationError(
                "Unsupported image dimensions"
            )

        found, corners = cv2.findChessboardCorners(
            grayscale,
            (target.columns, target.rows),
            flags=(
                cv2.CALIB_CB_ADAPTIVE_THRESH
                | cv2.CALIB_CB_NORMALIZE_IMAGE
            ),
        )

        if not found:
            return None

        criteria = (
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )

        refined = cv2.cornerSubPix(
            grayscale,
            corners,
            (11, 11),
            (-1, -1),
            criteria,
        )

        return tuple(
            Point2D(
                float(point[0][0]),
                float(point[0][1]),
            )
            for point in refined
        )

    def observations_from_images(
        self,
        images: Sequence[Any],
        target: CheckerboardTarget,
    ) -> tuple[CalibrationObservation, ...]:
        self._require_dependencies()

        observations = []

        for image in images:
            if image is None:
                continue

            height, width = image.shape[:2]

            corners = self.detect_checkerboard(
                image,
                target,
            )

            if corners is None:
                continue

            observations.append(
                CalibrationObservation(
                    observation_id=generate_id(),
                    object_points=target.object_points(),
                    image_points=corners,
                    image_width=width,
                    image_height=height,
                )
            )

        return tuple(observations)

    def calibrate(
        self,
        observations: Sequence[CalibrationObservation],
    ) -> OpenCVCalibrationResult:
        self._require_dependencies()

        if (
            len(observations)
            < self.config.minimum_observation_frames
        ):
            raise CalibrationValidationError(
                "Insufficient calibration observations"
            )

        width = observations[0].image_width
        height = observations[0].image_height

        object_points = []
        image_points = []

        for observation in observations:
            if (
                observation.image_width != width
                or observation.image_height != height
            ):
                raise CalibrationValidationError(
                    "All observations must share resolution"
                )

            if (
                len(observation.object_points)
                < self.config.minimum_points_per_frame
            ):
                raise CalibrationValidationError(
                    "Insufficient points in observation"
                )

            object_points.append(
                np.asarray(
                    [
                        (p.x, p.y, p.z)
                        for p in observation.object_points
                    ],
                    dtype=np.float32,
                )
            )

            image_points.append(
                np.asarray(
                    [
                        (p.x, p.y)
                        for p in observation.image_points
                    ],
                    dtype=np.float32,
                ).reshape(-1, 1, 2)
            )

        try:
            rms, camera_matrix, coefficients, rvecs, tvecs = (
                cv2.calibrateCamera(
                    object_points,
                    image_points,
                    (width, height),
                    None,
                    None,
                )
            )
        except cv2.error as exc:
            raise CameraCalibrationError(
                "OpenCV camera calibration failed"
            ) from exc

        coefficient_vector = (
            coefficients.reshape(-1).tolist()
        )

        if len(coefficient_vector) < 5:
            raise CameraCalibrationError(
                "OpenCV returned incomplete distortion data"
            )

        intrinsics = CameraIntrinsics(
            width=width,
            height=height,
            fx=float(camera_matrix[0, 0]),
            fy=float(camera_matrix[1, 1]),
            cx=float(camera_matrix[0, 2]),
            cy=float(camera_matrix[1, 2]),
            skew=float(camera_matrix[0, 1]),
        )

        distortion = CameraDistortion(
            k1=float(coefficient_vector[0]),
            k2=float(coefficient_vector[1]),
            p1=float(coefficient_vector[2]),
            p2=float(coefficient_vector[3]),
            k3=float(coefficient_vector[4]),
        )

        poses = []

        for index, (rvec, tvec) in enumerate(
            zip(rvecs, tvecs)
        ):
            rotation_matrix, _ = cv2.Rodrigues(
                rvec
            )

            rotation = RotationMatrix(
                tuple(
                    tuple(
                        float(rotation_matrix[i, j])
                        for j in range(3)
                    )
                    for i in range(3)
                )
            )

            translation = Point3D(
                float(tvec[0][0]),
                float(tvec[1][0]),
                float(tvec[2][0]),
            )

            poses.append(
                CameraExtrinsics(
                    source_frame=f"target_{index}",
                    target_frame="camera",
                    rotation=rotation,
                    translation=translation,
                )
            )

        total_points = sum(
            len(observation.image_points)
            for observation in observations
        )

        report = CalibrationQualityEvaluator(
            self.config
        ).evaluate(
            rms_error_px=float(rms),
            observation_count=len(observations),
            point_count=total_points,
        )

        return OpenCVCalibrationResult(
            intrinsics=intrinsics,
            distortion=distortion,
            rms_error_px=float(rms),
            observation_count=len(observations),
            point_count=total_points,
            target_poses=tuple(poses),
            quality_report=report,
        )


# ============================================================
# MULTI-CAMERA TRANSFORM UTILITIES
# ============================================================

class MultiCameraCalibration:

    @staticmethod
    def camera_to_camera(
        camera_a_to_world: CameraExtrinsics,
        camera_b_to_world: CameraExtrinsics,
    ) -> CameraExtrinsics:
        if (
            camera_a_to_world.target_frame
            != camera_b_to_world.target_frame
        ):
            raise CalibrationValidationError(
                "Cameras must share the same world frame"
            )

        world_to_camera_b = (
            camera_b_to_world.inverse()
        )

        return camera_a_to_world.compose(
            world_to_camera_b
        )

    @staticmethod
    def transform_point_between_cameras(
        point_in_camera_a: Point3D,
        camera_a_to_world: CameraExtrinsics,
        camera_b_to_world: CameraExtrinsics,
    ) -> Point3D:
        transform = MultiCameraCalibration.camera_to_camera(
            camera_a_to_world,
            camera_b_to_world,
        )

        return transform.transform_point(
            point_in_camera_a
        )


# ============================================================
# CAMERA CALIBRATION SERVICE
# ============================================================

class RoboticsCameraCalibrationService:

    def __init__(
        self,
        config: CameraCalibrationConfig | None = None,
    ) -> None:
        self.config = (
            config or CameraCalibrationConfig()
        )

        self.registry = CameraCalibrationRegistry(
            self.config
        )

        self.opencv_calibrator = (
            OpenCVCheckerboardCalibrator(
                self.config
            )
        )

    def register_manual_calibration(
        self,
        *,
        camera_id: str,
        intrinsics: CameraIntrinsics,
        distortion: CameraDistortion,
        access: CalibrationAccessContext,
        version: int = 1,
        model: CameraModel = CameraModel.BROWN_CONRADY,
        extrinsics: CameraExtrinsics | None = None,
        rms_error_px: float | None = None,
        observation_count: int = 0,
        calibration_point_count: int = 0,
    ) -> CameraCalibrationProfile:
        profile = CameraCalibrationProfile(
            calibration_id=generate_id(),
            tenant_id=access.tenant_id,
            camera_id=camera_id,
            version=version,
            model=model,
            intrinsics=intrinsics,
            distortion=distortion,
            extrinsics=extrinsics,
            rms_reprojection_error_px=rms_error_px,
            observation_count=observation_count,
            calibration_point_count=calibration_point_count,
        )

        return self.registry.register(
            profile,
            access=access,
        )

    def calibrate_checkerboard_images(
        self,
        *,
        camera_id: str,
        images: Sequence[Any],
        target: CheckerboardTarget,
        access: CalibrationAccessContext,
        version: int = 1,
    ) -> CameraCalibrationProfile:
        observations = (
            self.opencv_calibrator.observations_from_images(
                images,
                target,
            )
        )

        result = self.opencv_calibrator.calibrate(
            observations
        )

        return self.register_manual_calibration(
            camera_id=camera_id,
            intrinsics=result.intrinsics,
            distortion=result.distortion,
            access=access,
            version=version,
            rms_error_px=result.rms_error_px,
            observation_count=result.observation_count,
            calibration_point_count=result.point_count,
        )

    def project_point(
        self,
        camera_id: str,
        point_camera: Point3D,
        *,
        access: CalibrationAccessContext,
    ) -> Point2D:
        profile = self.registry.latest(
            camera_id,
            access=access,
        )

        model = PinholeCameraModel(
            profile.intrinsics,
            profile.distortion,
            self.config,
        )

        return model.project(
            point_camera
        )

    def undistort_pixel(
        self,
        camera_id: str,
        pixel: Point2D,
        *,
        access: CalibrationAccessContext,
    ) -> Point2D:
        profile = self.registry.latest(
            camera_id,
            access=access,
        )

        model = PinholeCameraModel(
            profile.intrinsics,
            profile.distortion,
            self.config,
        )

        return model.undistort_pixel(
            pixel
        )

    def pixel_to_ray(
        self,
        camera_id: str,
        pixel: Point2D,
        *,
        access: CalibrationAccessContext,
    ) -> Point3D:
        profile = self.registry.latest(
            camera_id,
            access=access,
        )

        model = PinholeCameraModel(
            profile.intrinsics,
            profile.distortion,
            self.config,
        )

        return model.pixel_to_ray(
            pixel
        )

    @staticmethod
    def export_profile_json(
        profile: CameraCalibrationProfile,
    ) -> str:
        return json.dumps(
            json_safe(profile),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestCameraCalibration(unittest.TestCase):

    def setUp(self) -> None:
        self.intrinsics = CameraIntrinsics(
            width=1280,
            height=720,
            fx=900.0,
            fy=900.0,
            cx=640.0,
            cy=360.0,
        )

        self.distortion = CameraDistortion(
            k1=-0.1,
            k2=0.02,
            p1=0.001,
            p2=-0.001,
            k3=0.0,
        )

        self.camera = PinholeCameraModel(
            self.intrinsics,
            self.distortion,
        )

    def test_principal_point_projection(self) -> None:
        pixel = self.camera.project(
            Point3D(0.0, 0.0, 2.0)
        )

        self.assertAlmostEqual(
            pixel.x,
            640.0,
        )

        self.assertAlmostEqual(
            pixel.y,
            360.0,
        )

    def test_projection_without_distortion(self) -> None:
        pixel = self.camera.project(
            Point3D(1.0, 0.0, 2.0),
            apply_distortion=False,
        )

        self.assertAlmostEqual(
            pixel.x,
            1090.0,
        )

    def test_projection_rejects_negative_depth(self) -> None:
        with self.assertRaises(
            CalibrationValidationError
        ):
            self.camera.project(
                Point3D(0, 0, -1)
            )

    def test_distortion_roundtrip(self) -> None:
        original = Point2D(
            800.0,
            450.0,
        )

        normalized = (
            self.camera.pixel_to_normalized(
                original
            )
        )

        distorted = (
            self.camera.distort_normalized(
                normalized.x,
                normalized.y,
            )
        )

        distorted_pixel = (
            self.camera.normalized_to_pixel(
                distorted
            )
        )

        recovered = self.camera.undistort_pixel(
            distorted_pixel
        )

        self.assertAlmostEqual(
            recovered.x,
            original.x,
            places=5,
        )

        self.assertAlmostEqual(
            recovered.y,
            original.y,
            places=5,
        )

    def test_back_projection(self) -> None:
        point = self.camera.back_project(
            Point2D(640, 360),
            depth_z=3.0,
        )

        self.assertAlmostEqual(point.x, 0.0)
        self.assertAlmostEqual(point.y, 0.0)
        self.assertAlmostEqual(point.z, 3.0)

    def test_intrinsic_scaling(self) -> None:
        scaled = self.intrinsics.scale(
            640,
            360,
        )

        self.assertAlmostEqual(
            scaled.fx,
            450.0,
        )

        self.assertAlmostEqual(
            scaled.fy,
            450.0,
        )

    def test_rotation_identity(self) -> None:
        rotation = RotationMatrix.identity()

        point = Point3D(
            1.0,
            2.0,
            3.0,
        )

        self.assertEqual(
            rotation.apply(point),
            point,
        )

    def test_extrinsic_inverse(self) -> None:
        transform = CameraExtrinsics(
            source_frame="camera",
            target_frame="world",
            rotation=RotationMatrix.from_euler(
                0.1,
                0.2,
                0.3,
            ),
            translation=Point3D(
                1.0,
                2.0,
                3.0,
            ),
        )

        original = Point3D(
            0.5,
            -0.2,
            2.0,
        )

        world = transform.transform_point(
            original
        )

        recovered = transform.inverse().transform_point(
            world
        )

        self.assertAlmostEqual(
            recovered.x,
            original.x,
            places=6,
        )

        self.assertAlmostEqual(
            recovered.y,
            original.y,
            places=6,
        )

        self.assertAlmostEqual(
            recovered.z,
            original.z,
            places=6,
        )

    def test_checkerboard_generation(self) -> None:
        target = CheckerboardTarget(
            columns=9,
            rows=6,
            square_size_m=0.025,
        )

        self.assertEqual(
            len(target.object_points()),
            54,
        )

    def test_reprojection_error(self) -> None:
        observed = (
            Point2D(10, 10),
            Point2D(20, 20),
        )

        predicted = (
            Point2D(11, 10),
            Point2D(19, 20),
        )

        metrics = ReprojectionErrorEvaluator.evaluate(
            observed,
            predicted,
        )

        self.assertAlmostEqual(
            metrics.rms_error_px,
            1.0,
        )

    def test_quality_report(self) -> None:
        evaluator = CalibrationQualityEvaluator()

        report = evaluator.evaluate(
            rms_error_px=0.3,
            observation_count=12,
            point_count=600,
        )

        self.assertTrue(
            report.accepted
        )

        self.assertEqual(
            report.quality,
            CalibrationQuality.GOOD,
        )

    def test_registry_tenant_isolation(self) -> None:
        registry = CameraCalibrationRegistry()

        owner = CalibrationAccessContext(
            tenant_id="tenant_alpha",
            user_id="engineer",
            allowed_camera_ids=frozenset({
                "CAM-001",
            }),
            can_read=True,
            can_write=True,
        )

        other = CalibrationAccessContext(
            tenant_id="tenant_beta",
            user_id="other",
            allowed_camera_ids=frozenset({
                "CAM-001",
            }),
            can_read=True,
        )

        profile = CameraCalibrationProfile(
            calibration_id=generate_id(),
            tenant_id="tenant_alpha",
            camera_id="CAM-001",
            version=1,
            model=CameraModel.BROWN_CONRADY,
            intrinsics=self.intrinsics,
            distortion=self.distortion,
        )

        registry.register(
            profile,
            access=owner,
        )

        with self.assertRaises(
            CalibrationNotFoundError
        ):
            registry.latest(
                "CAM-001",
                access=other,
            )

    def test_json_serialization(self) -> None:
        profile = CameraCalibrationProfile(
            calibration_id=generate_id(),
            tenant_id="tenant_alpha",
            camera_id="CAM-001",
            version=1,
            model=CameraModel.BROWN_CONRADY,
            intrinsics=self.intrinsics,
            distortion=self.distortion,
        )

        payload = (
            RoboticsCameraCalibrationService
            .export_profile_json(profile)
        )

        self.assertIn(
            "CAM-001",
            payload,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print(
        "\nPHYSICAL AI & AUTONOMOUS ROBOTICS PLATFORM"
    )

    print(
        "Camera Calibration & Geometry Engine\n"
    )

    service = RoboticsCameraCalibrationService()

    access = CalibrationAccessContext(
        tenant_id="tenant_alpha",
        user_id="robotics_engineer",
        allowed_camera_ids=frozenset({
            "CAM-FRONT-001",
        }),
        can_read=True,
        can_write=True,
        can_validate=True,
    )

    intrinsics = CameraIntrinsics(
        width=1920,
        height=1080,
        fx=1400.0,
        fy=1400.0,
        cx=960.0,
        cy=540.0,
    )

    distortion = CameraDistortion(
        k1=-0.12,
        k2=0.035,
        p1=0.001,
        p2=-0.0008,
        k3=-0.005,
    )

    extrinsics = CameraExtrinsics(
        source_frame="camera",
        target_frame="robot_base",
        rotation=RotationMatrix.identity(),
        translation=Point3D(
            0.15,
            0.0,
            0.8,
        ),
    )

    profile = service.register_manual_calibration(
        camera_id="CAM-FRONT-001",
        intrinsics=intrinsics,
        distortion=distortion,
        extrinsics=extrinsics,
        rms_error_px=0.35,
        observation_count=15,
        calibration_point_count=810,
        access=access,
    )

    print("CAMERA CALIBRATION PROFILE\n")

    print(
        service.export_profile_json(
            profile
        )
    )

    quality = service.registry.validate_latest(
        "CAM-FRONT-001",
        access=access,
    )

    print("\nCALIBRATION QUALITY\n")

    print(
        json.dumps(
            json_safe(quality),
            indent=2,
        )
    )

    point_camera = Point3D(
        x=0.2,
        y=0.1,
        z=2.0,
    )

    pixel = service.project_point(
        "CAM-FRONT-001",
        point_camera,
        access=access,
    )

    print("\n3D POINT TO PIXEL\n")

    print(
        json.dumps(
            json_safe(pixel),
            indent=2,
        )
    )

    corrected = service.undistort_pixel(
        "CAM-FRONT-001",
        pixel,
        access=access,
    )

    print("\nUNDISTORTED PIXEL\n")

    print(
        json.dumps(
            json_safe(corrected),
            indent=2,
        )
    )

    ray = service.pixel_to_ray(
        "CAM-FRONT-001",
        pixel,
        access=access,
    )

    print("\nCAMERA RAY\n")

    print(
        json.dumps(
            json_safe(ray),
            indent=2,
        )
    )

    point_robot_base = extrinsics.transform_point(
        point_camera
    )

    print("\nCAMERA POINT IN ROBOT BASE FRAME\n")

    print(
        json.dumps(
            json_safe(point_robot_base),
            indent=2,
        )
    )

    print("\nRUNNING UNIT TESTS\n")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestCameraCalibration
    )

    unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)


if __name__ == "__main__":
    main()
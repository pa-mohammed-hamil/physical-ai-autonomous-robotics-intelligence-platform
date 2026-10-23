"""Forward Kinematics module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/forward_kinematics.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================
Module: Forward Kinematics

Features:
    - Revolute and prismatic joints
    - Standard Denavit-Hartenberg transformations
    - Modified Denavit-Hartenberg transformations
    - Forward kinematics for serial robotic manipulators
    - Joint position validation
    - Homogeneous transformation matrices
    - End-effector position and orientation
    - Rotation matrices
    - Euler angles
    - Quaternions
    - Geometric Jacobian
    - Linear and angular velocity estimation
    - Singularity analysis
    - Manipulability analysis
    - Joint frame transformations
    - Tool-center-point transformations
    - Base-frame transformations
    - Cartesian workspace validation
    - Batch forward kinematics
    - Pose comparison
    - Multi-tenant robot isolation
    - Thread-safe model registry
    - JSON export
    - Synthetic robotic arm models
    - Unit tests

Requirements:
    Python >= 3.11
    numpy >= 1.24

Install:
    pip install numpy

Usage:
    python forward_kinematics.py

Safety:
    This module performs mathematical calculations only.
    It does not send commands to physical robotic hardware.

    Standard and modified DH conventions must match the
    robot manufacturer's actual kinematic definitions.

    Physical deployment requires independently verified
    calibration, collision detection, safety monitoring,
    and hardware-specific validation.
"""

from __future__ import annotations

import json
import math
import threading
import unittest
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Sequence

import numpy as np


# ============================================================
# EXCEPTIONS
# ============================================================

class ForwardKinematicsError(Exception):
    """Base exception for forward kinematics."""


class KinematicsValidationError(ForwardKinematicsError):
    """Invalid kinematic configuration or input."""


class JointLimitError(ForwardKinematicsError):
    """Joint position exceeds configured limits."""


class RobotNotFoundError(ForwardKinematicsError):
    """Requested robot model does not exist."""


class KinematicsAuthorizationError(ForwardKinematicsError):
    """Access to a robot model is not authorized."""


class KinematicsConflictError(ForwardKinematicsError):
    """Robot model already exists."""


class WorkspaceError(ForwardKinematicsError):
    """End-effector pose violates workspace constraints."""


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
        raise KinematicsValidationError(
            f"Invalid {name}"
        )

    return value.strip()


def finite_number(
    value: Any,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise KinematicsValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise KinematicsValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise KinematicsValidationError(
            f"{name} must be finite"
        )

    return result


def validate_vector(
    values: Sequence[float],
    size: int,
    name: str,
) -> np.ndarray:
    try:
        result = np.asarray(
            values,
            dtype=float,
        )
    except (TypeError, ValueError) as exc:
        raise KinematicsValidationError(
            f"{name} must contain numeric values"
        ) from exc

    if result.shape != (size,):
        raise KinematicsValidationError(
            f"{name} must contain exactly {size} values"
        )

    if not np.all(np.isfinite(result)):
        raise KinematicsValidationError(
            f"{name} contains non-finite values"
        )

    return result


def validate_rotation(
    rotation: Sequence[Sequence[float]],
    name: str = "rotation",
) -> np.ndarray:
    try:
        matrix = np.asarray(
            rotation,
            dtype=float,
        )
    except (TypeError, ValueError) as exc:
        raise KinematicsValidationError(
            f"{name} must be numeric"
        ) from exc

    if matrix.shape != (3, 3):
        raise KinematicsValidationError(
            f"{name} must be a 3x3 matrix"
        )

    if not np.all(np.isfinite(matrix)):
        raise KinematicsValidationError(
            f"{name} contains non-finite values"
        )

    if not np.allclose(
        matrix.T @ matrix,
        np.eye(3),
        atol=1e-6,
    ):
        raise KinematicsValidationError(
            f"{name} must be orthonormal"
        )

    if not np.isclose(
        np.linalg.det(matrix),
        1.0,
        atol=1e-6,
    ):
        raise KinematicsValidationError(
            f"{name} must have determinant +1"
        )

    return matrix


def validate_transform(
    transform: Sequence[Sequence[float]],
    name: str = "transform",
) -> np.ndarray:
    try:
        matrix = np.asarray(
            transform,
            dtype=float,
        )
    except (TypeError, ValueError) as exc:
        raise KinematicsValidationError(
            f"{name} must be numeric"
        ) from exc

    if matrix.shape != (4, 4):
        raise KinematicsValidationError(
            f"{name} must be a 4x4 matrix"
        )

    if not np.all(np.isfinite(matrix)):
        raise KinematicsValidationError(
            f"{name} contains non-finite values"
        )

    if not np.allclose(
        matrix[3],
        [0.0, 0.0, 0.0, 1.0],
        atol=1e-8,
    ):
        raise KinematicsValidationError(
            f"{name} has invalid homogeneous row"
        )

    validate_rotation(
        matrix[:3, :3],
        f"{name} rotation",
    )

    return matrix


def matrix_tuple(
    matrix: np.ndarray,
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            float(value)
            for value in row
        )
        for row in matrix
    )


def vector_tuple(
    vector: np.ndarray,
) -> tuple[float, ...]:
    return tuple(
        float(value)
        for value in vector
    )


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
            raise KinematicsValidationError(
                "Cannot serialize non-finite value"
            )

    if isinstance(
        value,
        (str, int, float, bool),
    ) or value is None:
        return value

    raise KinematicsValidationError(
        f"Unsupported serialization type: "
        f"{type(value).__name__}"
    )


# ============================================================
# ENUMERATIONS
# ============================================================

class JointType(str, Enum):
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"


class DHConvention(str, Enum):
    STANDARD = "standard"
    MODIFIED = "modified"


class KinematicsStatus(str, Enum):
    VALID = "valid"
    JOINT_LIMIT_VIOLATION = "joint_limit_violation"
    WORKSPACE_VIOLATION = "workspace_violation"


class SingularityStatus(str, Enum):
    REGULAR = "regular"
    NEAR_SINGULAR = "near_singular"
    SINGULAR = "singular"
    UNDERACTUATED = "underactuated"


class OrientationFormat(str, Enum):
    ROTATION_MATRIX = "rotation_matrix"
    EULER_XYZ = "euler_xyz"
    QUATERNION_XYZW = "quaternion_xyzw"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class JointConfig:
    name: str

    joint_type: JointType

    minimum_position: float
    maximum_position: float

    dh_a: float = 0.0
    dh_alpha: float = 0.0
    dh_d: float = 0.0
    dh_theta: float = 0.0

    home_position: float = 0.0

    maximum_velocity: float = 1.0

    def __post_init__(self) -> None:
        validate_identifier(
            self.name,
            "joint name",
        )

        if not isinstance(
            self.joint_type,
            JointType,
        ):
            raise KinematicsValidationError(
                "Invalid joint type"
            )

        for name in (
            "minimum_position",
            "maximum_position",
            "dh_a",
            "dh_alpha",
            "dh_d",
            "dh_theta",
            "home_position",
            "maximum_velocity",
        ):
            finite_number(
                getattr(self, name),
                name,
            )

        if self.minimum_position >= self.maximum_position:
            raise KinematicsValidationError(
                f"Invalid joint limits: {self.name}"
            )

        if self.maximum_velocity <= 0:
            raise KinematicsValidationError(
                "Maximum velocity must be positive"
            )

        if not (
            self.minimum_position
            <= self.home_position
            <= self.maximum_position
        ):
            raise KinematicsValidationError(
                f"Home position outside limits: {self.name}"
            )


@dataclass(frozen=True)
class WorkspaceBounds:
    minimum_x: float = -3.0
    maximum_x: float = 3.0

    minimum_y: float = -3.0
    maximum_y: float = 3.0

    minimum_z: float = -1.0
    maximum_z: float = 3.0

    def __post_init__(self) -> None:
        for axis in ("x", "y", "z"):
            lower = finite_number(
                getattr(self, f"minimum_{axis}"),
                f"minimum_{axis}",
            )

            upper = finite_number(
                getattr(self, f"maximum_{axis}"),
                f"maximum_{axis}",
            )

            if lower >= upper:
                raise KinematicsValidationError(
                    f"Invalid {axis}-axis workspace limits"
                )

    def contains(
        self,
        position: Sequence[float],
    ) -> bool:
        x, y, z = validate_vector(
            position,
            3,
            "position",
        )

        return bool(
            self.minimum_x <= x <= self.maximum_x
            and self.minimum_y <= y <= self.maximum_y
            and self.minimum_z <= z <= self.maximum_z
        )


@dataclass(frozen=True)
class RobotKinematicsConfig:
    robot_id: str
    tenant_id: str

    joints: tuple[JointConfig, ...]

    dh_convention: DHConvention = DHConvention.STANDARD

    base_transform: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ] = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )

    tool_transform: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ] = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )

    workspace: WorkspaceBounds = field(
        default_factory=WorkspaceBounds
    )

    singularity_threshold: float = 1e-5
    near_singularity_threshold: float = 1e-3

    def __post_init__(self) -> None:
        validate_identifier(
            self.robot_id,
            "robot_id",
        )

        validate_identifier(
            self.tenant_id,
            "tenant_id",
        )

        if not self.joints:
            raise KinematicsValidationError(
                "At least one joint is required"
            )

        if len(self.joints) > 32:
            raise KinematicsValidationError(
                "Maximum supported joint count is 32"
            )

        names = [
            joint.name
            for joint in self.joints
        ]

        if len(names) != len(set(names)):
            raise KinematicsValidationError(
                "Duplicate joint names"
            )

        if not isinstance(
            self.dh_convention,
            DHConvention,
        ):
            raise KinematicsValidationError(
                "Invalid DH convention"
            )

        validate_transform(
            self.base_transform,
            "base_transform",
        )

        validate_transform(
            self.tool_transform,
            "tool_transform",
        )

        if self.singularity_threshold <= 0:
            raise KinematicsValidationError(
                "Singularity threshold must be positive"
            )

        if (
            self.near_singularity_threshold
            <= self.singularity_threshold
        ):
            raise KinematicsValidationError(
                "Near-singularity threshold must exceed "
                "singularity threshold"
            )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class KinematicsAccessContext:
    tenant_id: str
    user_id: str

    can_read: bool = False
    can_calculate: bool = False
    can_manage: bool = False


@dataclass(frozen=True)
class CartesianPose:
    position: tuple[float, float, float]

    rotation_matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]

    euler_xyz: tuple[float, float, float]

    quaternion_xyzw: tuple[
        float,
        float,
        float,
        float,
    ]

    transform_matrix: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]


@dataclass(frozen=True)
class JointFrame:
    frame_index: int
    joint_name: str

    origin: tuple[float, float, float]

    z_axis: tuple[float, float, float]

    transform_matrix: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]


@dataclass(frozen=True)
class JacobianResult:
    geometric_jacobian: tuple[tuple[float, ...], ...]
    linear_jacobian: tuple[tuple[float, ...], ...]
    angular_jacobian: tuple[tuple[float, ...], ...]

    rank: int
    singular_values: tuple[float, ...]


@dataclass(frozen=True)
class SingularityReport:
    status: SingularityStatus

    jacobian_rank: int
    expected_rank: int

    minimum_singular_value: float

    condition_number: float | None

    manipulability: float

    evaluated_at: datetime


@dataclass(frozen=True)
class CartesianVelocity:
    linear_velocity: tuple[float, float, float]

    angular_velocity: tuple[float, float, float]

    twist: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
    ]


@dataclass(frozen=True)
class PoseComparison:
    position_error: float
    orientation_error_radians: float

    translation_difference: tuple[float, float, float]


@dataclass(frozen=True)
class ForwardKinematicsResult:
    calculation_id: str

    robot_id: str
    tenant_id: str

    joint_positions: tuple[float, ...]

    end_effector_pose: CartesianPose

    joint_frames: tuple[JointFrame, ...]

    status: KinematicsStatus

    computed_at: datetime


# ============================================================
# ROTATION UTILITIES
# ============================================================

class RotationUtils:

    @staticmethod
    def rotation_x(
        angle: float,
    ) -> np.ndarray:
        angle = finite_number(
            angle,
            "angle",
        )

        c = math.cos(angle)
        s = math.sin(angle)

        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, c, -s],
                [0.0, s, c],
            ],
            dtype=float,
        )

    @staticmethod
    def rotation_y(
        angle: float,
    ) -> np.ndarray:
        angle = finite_number(
            angle,
            "angle",
        )

        c = math.cos(angle)
        s = math.sin(angle)

        return np.array(
            [
                [c, 0.0, s],
                [0.0, 1.0, 0.0],
                [-s, 0.0, c],
            ],
            dtype=float,
        )

    @staticmethod
    def rotation_z(
        angle: float,
    ) -> np.ndarray:
        angle = finite_number(
            angle,
            "angle",
        )

        c = math.cos(angle)
        s = math.sin(angle)

        return np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    @staticmethod
    def euler_xyz_to_matrix(
        roll: float,
        pitch: float,
        yaw: float,
    ) -> np.ndarray:
        """
        Fixed-axis XYZ roll/pitch/yaw convention.

        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        """
        return (
            RotationUtils.rotation_z(yaw)
            @ RotationUtils.rotation_y(pitch)
            @ RotationUtils.rotation_x(roll)
        )

    @staticmethod
    def matrix_to_euler_xyz(
        rotation: Sequence[Sequence[float]],
    ) -> tuple[float, float, float]:
        matrix = validate_rotation(
            rotation
        )

        pitch = math.asin(
            clamp(
                -float(matrix[2, 0]),
                -1.0,
                1.0,
            )
        )

        cosine_pitch = math.cos(
            pitch
        )

        if abs(cosine_pitch) > 1e-8:
            roll = math.atan2(
                matrix[2, 1],
                matrix[2, 2],
            )

            yaw = math.atan2(
                matrix[1, 0],
                matrix[0, 0],
            )

        else:
            # Gimbal-lock convention: yaw = 0.
            roll = math.atan2(
                -matrix[1, 2],
                matrix[1, 1],
            )

            yaw = 0.0

        return (
            float(roll),
            float(pitch),
            float(yaw),
        )

    @staticmethod
    def matrix_to_quaternion_xyzw(
        rotation: Sequence[Sequence[float]],
    ) -> tuple[float, float, float, float]:
        matrix = validate_rotation(
            rotation
        )

        trace = float(
            np.trace(matrix)
        )

        if trace > 0:
            scale = math.sqrt(
                trace + 1.0
            ) * 2.0

            w = 0.25 * scale
            x = (
                matrix[2, 1] - matrix[1, 2]
            ) / scale

            y = (
                matrix[0, 2] - matrix[2, 0]
            ) / scale

            z = (
                matrix[1, 0] - matrix[0, 1]
            ) / scale

        elif (
            matrix[0, 0] > matrix[1, 1]
            and matrix[0, 0] > matrix[2, 2]
        ):
            scale = math.sqrt(
                1.0
                + matrix[0, 0]
                - matrix[1, 1]
                - matrix[2, 2]
            ) * 2.0

            w = (
                matrix[2, 1] - matrix[1, 2]
            ) / scale

            x = 0.25 * scale

            y = (
                matrix[0, 1] + matrix[1, 0]
            ) / scale

            z = (
                matrix[0, 2] + matrix[2, 0]
            ) / scale

        elif matrix[1, 1] > matrix[2, 2]:
            scale = math.sqrt(
                1.0
                + matrix[1, 1]
                - matrix[0, 0]
                - matrix[2, 2]
            ) * 2.0

            w = (
                matrix[0, 2] - matrix[2, 0]
            ) / scale

            x = (
                matrix[0, 1] + matrix[1, 0]
            ) / scale

            y = 0.25 * scale

            z = (
                matrix[1, 2] + matrix[2, 1]
            ) / scale

        else:
            scale = math.sqrt(
                1.0
                + matrix[2, 2]
                - matrix[0, 0]
                - matrix[1, 1]
            ) * 2.0

            w = (
                matrix[1, 0] - matrix[0, 1]
            ) / scale

            x = (
                matrix[0, 2] + matrix[2, 0]
            ) / scale

            y = (
                matrix[1, 2] + matrix[2, 1]
            ) / scale

            z = 0.25 * scale

        quaternion = np.array(
            [x, y, z, w],
            dtype=float,
        )

        quaternion /= np.linalg.norm(
            quaternion
        )

        if quaternion[3] < 0:
            quaternion *= -1.0

        return tuple(
            float(value)
            for value in quaternion
        )

    @staticmethod
    def quaternion_xyzw_to_matrix(
        quaternion: Sequence[float],
    ) -> np.ndarray:
        q = validate_vector(
            quaternion,
            4,
            "quaternion",
        )

        norm = float(
            np.linalg.norm(q)
        )

        if norm <= 1e-12:
            raise KinematicsValidationError(
                "Quaternion magnitude must be positive"
            )

        x, y, z, w = q / norm

        return np.array(
            [
                [
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                ],
                [
                    2 * (x * y + z * w),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - x * w),
                ],
                [
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x * x + y * y),
                ],
            ],
            dtype=float,
        )

    @staticmethod
    def orientation_error(
        rotation_a: Sequence[Sequence[float]],
        rotation_b: Sequence[Sequence[float]],
    ) -> float:
        a = validate_rotation(
            rotation_a,
            "rotation_a",
        )

        b = validate_rotation(
            rotation_b,
            "rotation_b",
        )

        relative = a.T @ b

        cosine = (
            np.trace(relative) - 1.0
        ) / 2.0

        return float(
            math.acos(
                clamp(
                    float(cosine),
                    -1.0,
                    1.0,
                )
            )
        )


# ============================================================
# HOMOGENEOUS TRANSFORMATIONS
# ============================================================

class TransformUtils:

    @staticmethod
    def from_rotation_translation(
        rotation: Sequence[Sequence[float]],
        translation: Sequence[float],
    ) -> np.ndarray:
        rotation_matrix = validate_rotation(
            rotation
        )

        position = validate_vector(
            translation,
            3,
            "translation",
        )

        transform = np.eye(
            4,
            dtype=float,
        )

        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = position

        return transform

    @staticmethod
    def translation(
        x: float,
        y: float,
        z: float,
    ) -> np.ndarray:
        transform = np.eye(
            4,
            dtype=float,
        )

        transform[:3, 3] = [
            finite_number(x, "x"),
            finite_number(y, "y"),
            finite_number(z, "z"),
        ]

        return transform

    @staticmethod
    def rotation_x(
        angle: float,
    ) -> np.ndarray:
        return TransformUtils.from_rotation_translation(
            RotationUtils.rotation_x(angle),
            (0.0, 0.0, 0.0),
        )

    @staticmethod
    def rotation_z(
        angle: float,
    ) -> np.ndarray:
        return TransformUtils.from_rotation_translation(
            RotationUtils.rotation_z(angle),
            (0.0, 0.0, 0.0),
        )

    @staticmethod
    def inverse(
        transform: Sequence[Sequence[float]],
    ) -> np.ndarray:
        matrix = validate_transform(
            transform
        )

        rotation = matrix[:3, :3]
        position = matrix[:3, 3]

        inverse = np.eye(
            4,
            dtype=float,
        )

        inverse[:3, :3] = rotation.T
        inverse[:3, 3] = -rotation.T @ position

        return inverse

    @staticmethod
    def transform_point(
        transform: Sequence[Sequence[float]],
        point: Sequence[float],
    ) -> np.ndarray:
        matrix = validate_transform(
            transform
        )

        position = validate_vector(
            point,
            3,
            "point",
        )

        homogeneous = np.append(
            position,
            1.0,
        )

        result = matrix @ homogeneous

        return result[:3]

    @staticmethod
    def compose(
        *transforms: Sequence[Sequence[float]],
    ) -> np.ndarray:
        result = np.eye(
            4,
            dtype=float,
        )

        for transform in transforms:
            result = (
                result
                @ validate_transform(transform)
            )

        return result


# ============================================================
# DENAVIT-HARTENBERG TRANSFORMS
# ============================================================

class DHTransform:

    @staticmethod
    def standard(
        a: float,
        alpha: float,
        d: float,
        theta: float,
    ) -> np.ndarray:
        """
        Standard DH:

        T = RotZ(theta) @ TransZ(d)
            @ TransX(a) @ RotX(alpha)
        """
        ct = math.cos(theta)
        st = math.sin(theta)

        ca = math.cos(alpha)
        sa = math.sin(alpha)

        return np.array(
            [
                [
                    ct,
                    -st * ca,
                    st * sa,
                    a * ct,
                ],
                [
                    st,
                    ct * ca,
                    -ct * sa,
                    a * st,
                ],
                [
                    0.0,
                    sa,
                    ca,
                    d,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
            ],
            dtype=float,
        )

    @staticmethod
    def modified(
        a: float,
        alpha: float,
        d: float,
        theta: float,
    ) -> np.ndarray:
        """
        Modified DH convention used here:

        T = RotX(alpha) @ TransX(a)
            @ RotZ(theta) @ TransZ(d)

        Parameters a and alpha describe the preceding
        X-axis transformation for this convention.
        """
        return TransformUtils.compose(
            TransformUtils.rotation_x(alpha),
            TransformUtils.translation(a, 0.0, 0.0),
            TransformUtils.rotation_z(theta),
            TransformUtils.translation(0.0, 0.0, d),
        )

    @staticmethod
    def calculate(
        a: float,
        alpha: float,
        d: float,
        theta: float,
        convention: DHConvention,
    ) -> np.ndarray:
        for name, value in (
            ("a", a),
            ("alpha", alpha),
            ("d", d),
            ("theta", theta),
        ):
            finite_number(
                value,
                name,
            )

        if convention == DHConvention.STANDARD:
            return DHTransform.standard(
                a,
                alpha,
                d,
                theta,
            )

        if convention == DHConvention.MODIFIED:
            return DHTransform.modified(
                a,
                alpha,
                d,
                theta,
            )

        raise KinematicsValidationError(
            "Unsupported DH convention"
        )


# ============================================================
# JOINT VALIDATOR
# ============================================================

class JointPositionValidator:

    def __init__(
        self,
        joints: Sequence[JointConfig],
        *,
        tolerance: float = 1e-9,
    ) -> None:
        self.joints = tuple(joints)
        self.tolerance = tolerance

    def validate(
        self,
        positions: Sequence[float],
    ) -> np.ndarray:
        q = validate_vector(
            positions,
            len(self.joints),
            "joint positions",
        )

        for index, joint in enumerate(self.joints):
            if (
                q[index]
                < joint.minimum_position - self.tolerance
                or q[index]
                > joint.maximum_position + self.tolerance
            ):
                raise JointLimitError(
                    f"Joint '{joint.name}' position "
                    f"{q[index]:.6f} exceeds limits "
                    f"[{joint.minimum_position:.6f}, "
                    f"{joint.maximum_position:.6f}]"
                )

        return q

    def validate_velocities(
        self,
        velocities: Sequence[float],
    ) -> np.ndarray:
        qd = validate_vector(
            velocities,
            len(self.joints),
            "joint velocities",
        )

        for index, joint in enumerate(self.joints):
            if (
                abs(qd[index])
                > joint.maximum_velocity + self.tolerance
            ):
                raise JointLimitError(
                    f"Joint '{joint.name}' velocity "
                    "exceeds configured limit"
                )

        return qd


# ============================================================
# FORWARD KINEMATICS ENGINE
# ============================================================

class ForwardKinematicsEngine:

    def __init__(
        self,
        config: RobotKinematicsConfig,
    ) -> None:
        self.config = config
        self.joints = config.joints
        self.dof = len(config.joints)

        self.validator = JointPositionValidator(
            config.joints
        )

        self._base = validate_transform(
            config.base_transform,
            "base_transform",
        )

        self._tool = validate_transform(
            config.tool_transform,
            "tool_transform",
        )

    # --------------------------------------------------------
    # JOINT TRANSFORM
    # --------------------------------------------------------

    def joint_transform(
        self,
        joint_index: int,
        position: float,
    ) -> np.ndarray:
        if not 0 <= joint_index < self.dof:
            raise KinematicsValidationError(
                "Invalid joint index"
            )

        joint = self.joints[
            joint_index
        ]

        q = finite_number(
            position,
            "joint position",
        )

        if joint.joint_type == JointType.REVOLUTE:
            theta = joint.dh_theta + q
            d = joint.dh_d

        else:
            theta = joint.dh_theta
            d = joint.dh_d + q

        return DHTransform.calculate(
            a=joint.dh_a,
            alpha=joint.dh_alpha,
            d=d,
            theta=theta,
            convention=self.config.dh_convention,
        )

    # --------------------------------------------------------
    # TRANSFORM CHAIN
    # --------------------------------------------------------

    def transform_chain(
        self,
        joint_positions: Sequence[float],
    ) -> tuple[np.ndarray, ...]:
        q = self.validator.validate(
            joint_positions
        )

        current = self._base.copy()

        transforms = [
            current.copy()
        ]

        for index in range(self.dof):
            current = (
                current
                @ self.joint_transform(
                    index,
                    q[index],
                )
            )

            transforms.append(
                current.copy()
            )

        return tuple(
            transforms
        )

    # --------------------------------------------------------
    # END-EFFECTOR TRANSFORM
    # --------------------------------------------------------

    def end_effector_transform(
        self,
        joint_positions: Sequence[float],
    ) -> np.ndarray:
        chain = self.transform_chain(
            joint_positions
        )

        return (
            chain[-1] @ self._tool
        )

    # --------------------------------------------------------
    # POSE FROM TRANSFORM
    # --------------------------------------------------------

    @staticmethod
    def pose_from_transform(
        transform: Sequence[Sequence[float]],
    ) -> CartesianPose:
        matrix = validate_transform(
            transform
        )

        rotation = matrix[:3, :3]

        return CartesianPose(
            position=vector_tuple(
                matrix[:3, 3]
            ),
            rotation_matrix=matrix_tuple(
                rotation
            ),
            euler_xyz=RotationUtils.matrix_to_euler_xyz(
                rotation
            ),
            quaternion_xyzw=(
                RotationUtils.matrix_to_quaternion_xyzw(
                    rotation
                )
            ),
            transform_matrix=matrix_tuple(
                matrix
            ),
        )

    # --------------------------------------------------------
    # JOINT FRAMES
    # --------------------------------------------------------

    def joint_frames(
        self,
        joint_positions: Sequence[float],
    ) -> tuple[JointFrame, ...]:
        transforms = self.transform_chain(
            joint_positions
        )

        frames = []

        for index, transform in enumerate(
            transforms
        ):
            name = (
                "base"
                if index == 0
                else self.joints[index - 1].name
            )

            frames.append(
                JointFrame(
                    frame_index=index,
                    joint_name=name,
                    origin=vector_tuple(
                        transform[:3, 3]
                    ),
                    z_axis=vector_tuple(
                        transform[:3, 2]
                    ),
                    transform_matrix=matrix_tuple(
                        transform
                    ),
                )
            )

        return tuple(
            frames
        )

    # --------------------------------------------------------
    # COMPLETE FORWARD KINEMATICS
    # --------------------------------------------------------

    def calculate(
        self,
        joint_positions: Sequence[float],
        *,
        enforce_workspace: bool = True,
    ) -> ForwardKinematicsResult:
        q = self.validator.validate(
            joint_positions
        )

        transform = self.end_effector_transform(
            q
        )

        pose = self.pose_from_transform(
            transform
        )

        workspace_valid = (
            self.config.workspace.contains(
                pose.position
            )
        )

        if not workspace_valid and enforce_workspace:
            raise WorkspaceError(
                "End effector outside configured workspace"
            )

        return ForwardKinematicsResult(
            calculation_id=generate_id(),
            robot_id=self.config.robot_id,
            tenant_id=self.config.tenant_id,
            joint_positions=vector_tuple(q),
            end_effector_pose=pose,
            joint_frames=self.joint_frames(q),
            status=(
                KinematicsStatus.VALID
                if workspace_valid
                else KinematicsStatus.WORKSPACE_VIOLATION
            ),
            computed_at=utc_now(),
        )

    # --------------------------------------------------------
    # BATCH FORWARD KINEMATICS
    # --------------------------------------------------------

    def calculate_batch(
        self,
        joint_position_batches: Sequence[
            Sequence[float]
        ],
        *,
        enforce_workspace: bool = True,
    ) -> tuple[ForwardKinematicsResult, ...]:
        return tuple(
            self.calculate(
                positions,
                enforce_workspace=enforce_workspace,
            )
            for positions in joint_position_batches
        )

    # --------------------------------------------------------
    # GEOMETRIC JACOBIAN
    # --------------------------------------------------------

    def jacobian(
        self,
        joint_positions: Sequence[float],
    ) -> JacobianResult:
        q = self.validator.validate(
            joint_positions
        )

        transforms = self.transform_chain(
            q
        )

        end_position = (
            transforms[-1] @ self._tool
        )[:3, 3]

        jacobian = np.zeros(
            (6, self.dof),
            dtype=float,
        )

        for index, joint in enumerate(
            self.joints
        ):
            if (
                self.config.dh_convention
                == DHConvention.STANDARD
            ):
                # Standard DH: the variable joint axis
                # is Z of the preceding frame.
                axis_transform = transforms[index]

            else:
                # Modified DH:
                # RotX(alpha) and TransX(a) precede
                # the variable Z-axis transformation.
                axis_transform = (
                    transforms[index]
                    @ TransformUtils.rotation_x(
                        joint.dh_alpha
                    )
                    @ TransformUtils.translation(
                        joint.dh_a,
                        0.0,
                        0.0,
                    )
                )

            axis = axis_transform[:3, 2]
            origin = axis_transform[:3, 3]

            if joint.joint_type == JointType.REVOLUTE:
                jacobian[:3, index] = np.cross(
                    axis,
                    end_position - origin,
                )

                jacobian[3:, index] = axis

            else:
                jacobian[:3, index] = axis
                jacobian[3:, index] = 0.0

        singular_values = np.linalg.svd(
            jacobian,
            compute_uv=False,
        )

        rank = int(
            np.linalg.matrix_rank(
                jacobian,
                tol=self.config.singularity_threshold,
            )
        )

        return JacobianResult(
            geometric_jacobian=matrix_tuple(
                jacobian
            ),
            linear_jacobian=matrix_tuple(
                jacobian[:3, :]
            ),
            angular_jacobian=matrix_tuple(
                jacobian[3:, :]
            ),
            rank=rank,
            singular_values=vector_tuple(
                singular_values
            ),
        )

    # --------------------------------------------------------
    # CARTESIAN VELOCITY
    # --------------------------------------------------------

    def cartesian_velocity(
        self,
        joint_positions: Sequence[float],
        joint_velocities: Sequence[float],
    ) -> CartesianVelocity:
        qd = self.validator.validate_velocities(
            joint_velocities
        )

        result = self.jacobian(
            joint_positions
        )

        jacobian = np.asarray(
            result.geometric_jacobian,
            dtype=float,
        )

        twist = jacobian @ qd

        return CartesianVelocity(
            linear_velocity=vector_tuple(
                twist[:3]
            ),
            angular_velocity=vector_tuple(
                twist[3:]
            ),
            twist=vector_tuple(
                twist
            ),
        )

    # --------------------------------------------------------
    # SINGULARITY ANALYSIS
    # --------------------------------------------------------

    def analyze_singularity(
        self,
        joint_positions: Sequence[float],
    ) -> SingularityReport:
        result = self.jacobian(
            joint_positions
        )

        jacobian = np.asarray(
            result.geometric_jacobian,
            dtype=float,
        )

        expected_rank = min(
            6,
            self.dof,
        )

        singular_values = np.linalg.svd(
            jacobian,
            compute_uv=False,
        )

        minimum_value = float(
            min(singular_values)
        )

        maximum_value = float(
            max(singular_values)
        )

        if (
            minimum_value
            <= self.config.singularity_threshold
        ):
            condition_number = None

        else:
            condition_number = (
                maximum_value
                / minimum_value
            )

        # Product of singular values gives a generalized
        # manipulability measure for the represented task.
        manipulability = float(
            np.prod(singular_values)
        )

        if self.dof < 6:
            status = SingularityStatus.UNDERACTUATED

        elif (
            result.rank < expected_rank
            or minimum_value
            <= self.config.singularity_threshold
        ):
            status = SingularityStatus.SINGULAR

        elif (
            minimum_value
            <= self.config.near_singularity_threshold
        ):
            status = SingularityStatus.NEAR_SINGULAR

        else:
            status = SingularityStatus.REGULAR

        return SingularityReport(
            status=status,
            jacobian_rank=result.rank,
            expected_rank=expected_rank,
            minimum_singular_value=minimum_value,
            condition_number=condition_number,
            manipulability=manipulability,
            evaluated_at=utc_now(),
        )

    # --------------------------------------------------------
    # POSE COMPARISON
    # --------------------------------------------------------

    @staticmethod
    def compare_poses(
        pose_a: CartesianPose,
        pose_b: CartesianPose,
    ) -> PoseComparison:
        position_a = validate_vector(
            pose_a.position,
            3,
            "pose_a position",
        )

        position_b = validate_vector(
            pose_b.position,
            3,
            "pose_b position",
        )

        difference = (
            position_b - position_a
        )

        orientation_error = (
            RotationUtils.orientation_error(
                pose_a.rotation_matrix,
                pose_b.rotation_matrix,
            )
        )

        return PoseComparison(
            position_error=float(
                np.linalg.norm(difference)
            ),
            orientation_error_radians=orientation_error,
            translation_difference=vector_tuple(
                difference
            ),
        )


# ============================================================
# MULTI-TENANT FORWARD KINEMATICS SERVICE
# ============================================================

class ForwardKinematicsService:

    def __init__(self) -> None:
        self._engines: dict[
            tuple[str, str],
            ForwardKinematicsEngine,
        ] = {}

        self._lock = threading.RLock()

    @staticmethod
    def _authorize(
        access: KinematicsAccessContext,
        permission: str,
    ) -> None:
        if not getattr(
            access,
            permission,
            False,
        ):
            raise KinematicsAuthorizationError(
                f"Missing permission: {permission}"
            )

    def register_robot(
        self,
        config: RobotKinematicsConfig,
        access: KinematicsAccessContext,
    ) -> ForwardKinematicsEngine:
        self._authorize(
            access,
            "can_manage",
        )

        if access.tenant_id != config.tenant_id:
            raise KinematicsAuthorizationError(
                "Cross-tenant registration forbidden"
            )

        key = (
            config.tenant_id,
            config.robot_id,
        )

        with self._lock:
            if key in self._engines:
                raise KinematicsConflictError(
                    "Robot already registered"
                )

            engine = ForwardKinematicsEngine(
                config
            )

            self._engines[key] = engine

            return engine

    def get_robot(
        self,
        robot_id: str,
        access: KinematicsAccessContext,
    ) -> ForwardKinematicsEngine:
        self._authorize(
            access,
            "can_read",
        )

        key = (
            access.tenant_id,
            robot_id,
        )

        with self._lock:
            engine = self._engines.get(
                key
            )

        if engine is None:
            raise RobotNotFoundError(
                "Robot model not found"
            )

        return engine

    def calculate(
        self,
        robot_id: str,
        joint_positions: Sequence[float],
        access: KinematicsAccessContext,
        *,
        enforce_workspace: bool = True,
    ) -> ForwardKinematicsResult:
        self._authorize(
            access,
            "can_calculate",
        )

        engine = self.get_robot(
            robot_id,
            access,
        )

        return engine.calculate(
            joint_positions,
            enforce_workspace=enforce_workspace,
        )

    def calculate_batch(
        self,
        robot_id: str,
        batches: Sequence[Sequence[float]],
        access: KinematicsAccessContext,
    ) -> tuple[ForwardKinematicsResult, ...]:
        self._authorize(
            access,
            "can_calculate",
        )

        engine = self.get_robot(
            robot_id,
            access,
        )

        return engine.calculate_batch(
            batches
        )

    def list_robots(
        self,
        access: KinematicsAccessContext,
    ) -> tuple[str, ...]:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            return tuple(
                robot_id
                for tenant_id, robot_id
                in self._engines
                if tenant_id == access.tenant_id
            )

    def remove_robot(
        self,
        robot_id: str,
        access: KinematicsAccessContext,
    ) -> None:
        self._authorize(
            access,
            "can_manage",
        )

        key = (
            access.tenant_id,
            robot_id,
        )

        with self._lock:
            if key not in self._engines:
                raise RobotNotFoundError(
                    "Robot model not found"
                )

            del self._engines[key]

    def export_result_json(
        self,
        robot_id: str,
        joint_positions: Sequence[float],
        access: KinematicsAccessContext,
    ) -> str:
        result = self.calculate(
            robot_id,
            joint_positions,
            access,
        )

        return json.dumps(
            json_safe(result),
            indent=2,
            allow_nan=False,
        )


# ============================================================
# SYNTHETIC ROBOT FACTORY
# ============================================================

class SyntheticRobotFactory:

    @staticmethod
    def two_link_planar(
        *,
        robot_id: str = "planar-arm-001",
        tenant_id: str = "physical_ai_lab",
        link_1: float = 0.5,
        link_2: float = 0.4,
    ) -> RobotKinematicsConfig:
        link_1 = finite_number(
            link_1,
            "link_1",
        )

        link_2 = finite_number(
            link_2,
            "link_2",
        )

        if link_1 <= 0 or link_2 <= 0:
            raise KinematicsValidationError(
                "Link lengths must be positive"
            )

        return RobotKinematicsConfig(
            robot_id=robot_id,
            tenant_id=tenant_id,
            joints=(
                JointConfig(
                    name="shoulder",
                    joint_type=JointType.REVOLUTE,
                    minimum_position=-math.pi,
                    maximum_position=math.pi,
                    dh_a=link_1,
                    dh_alpha=0.0,
                    home_position=0.0,
                ),
                JointConfig(
                    name="elbow",
                    joint_type=JointType.REVOLUTE,
                    minimum_position=-math.pi,
                    maximum_position=math.pi,
                    dh_a=link_2,
                    dh_alpha=0.0,
                    home_position=0.0,
                ),
            ),
            workspace=WorkspaceBounds(
                minimum_x=-2.0,
                maximum_x=2.0,
                minimum_y=-2.0,
                maximum_y=2.0,
                minimum_z=-1.0,
                maximum_z=1.0,
            ),
        )

    @staticmethod
    def three_dof_arm(
        *,
        robot_id: str = "robot-arm-001",
        tenant_id: str = "physical_ai_lab",
    ) -> RobotKinematicsConfig:
        return RobotKinematicsConfig(
            robot_id=robot_id,
            tenant_id=tenant_id,
            joints=(
                JointConfig(
                    name="base_rotation",
                    joint_type=JointType.REVOLUTE,
                    minimum_position=-math.pi,
                    maximum_position=math.pi,
                    dh_a=0.0,
                    dh_alpha=math.pi / 2,
                    dh_d=0.3,
                ),
                JointConfig(
                    name="shoulder",
                    joint_type=JointType.REVOLUTE,
                    minimum_position=-2.5,
                    maximum_position=2.5,
                    dh_a=0.5,
                    dh_alpha=0.0,
                ),
                JointConfig(
                    name="elbow",
                    joint_type=JointType.REVOLUTE,
                    minimum_position=-2.5,
                    maximum_position=2.5,
                    dh_a=0.4,
                    dh_alpha=0.0,
                ),
            ),
        )

    @staticmethod
    def prismatic_arm(
        *,
        robot_id: str = "linear-arm-001",
        tenant_id: str = "physical_ai_lab",
    ) -> RobotKinematicsConfig:
        return RobotKinematicsConfig(
            robot_id=robot_id,
            tenant_id=tenant_id,
            joints=(
                JointConfig(
                    name="linear_z",
                    joint_type=JointType.PRISMATIC,
                    minimum_position=0.0,
                    maximum_position=1.0,
                    dh_d=0.2,
                    home_position=0.0,
                ),
            ),
        )

    @staticmethod
    def access(
        tenant_id: str = "physical_ai_lab",
    ) -> KinematicsAccessContext:
        return KinematicsAccessContext(
            tenant_id=tenant_id,
            user_id="robotics_engineer",
            can_read=True,
            can_calculate=True,
            can_manage=True,
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestForwardKinematics(unittest.TestCase):

    def setUp(self) -> None:
        self.config = (
            SyntheticRobotFactory.two_link_planar()
        )

        self.engine = ForwardKinematicsEngine(
            self.config
        )

        self.access = (
            SyntheticRobotFactory.access()
        )

    def test_identity_rotation(self) -> None:
        rotation = RotationUtils.rotation_z(
            0.0
        )

        self.assertTrue(
            np.allclose(
                rotation,
                np.eye(3),
            )
        )

    def test_rotation_z_90_degrees(self) -> None:
        rotation = RotationUtils.rotation_z(
            math.pi / 2
        )

        point = rotation @ np.array(
            [1.0, 0.0, 0.0]
        )

        self.assertTrue(
            np.allclose(
                point,
                [0.0, 1.0, 0.0],
                atol=1e-8,
            )
        )

    def test_standard_dh_identity(self) -> None:
        transform = DHTransform.standard(
            0.0,
            0.0,
            0.0,
            0.0,
        )

        self.assertTrue(
            np.allclose(
                transform,
                np.eye(4),
            )
        )

    def test_modified_dh_identity(self) -> None:
        transform = DHTransform.modified(
            0.0,
            0.0,
            0.0,
            0.0,
        )

        self.assertTrue(
            np.allclose(
                transform,
                np.eye(4),
            )
        )

    def test_planar_home_position(self) -> None:
        result = self.engine.calculate(
            (0.0, 0.0)
        )

        self.assertAlmostEqual(
            result.end_effector_pose.position[0],
            0.9,
            places=6,
        )

        self.assertAlmostEqual(
            result.end_effector_pose.position[1],
            0.0,
            places=6,
        )

    def test_planar_right_angle(self) -> None:
        result = self.engine.calculate(
            (math.pi / 2, 0.0)
        )

        self.assertAlmostEqual(
            result.end_effector_pose.position[0],
            0.0,
            places=6,
        )

        self.assertAlmostEqual(
            result.end_effector_pose.position[1],
            0.9,
            places=6,
        )

    def test_planar_elbow_rotation(self) -> None:
        result = self.engine.calculate(
            (0.0, math.pi / 2)
        )

        self.assertAlmostEqual(
            result.end_effector_pose.position[0],
            0.5,
            places=6,
        )

        self.assertAlmostEqual(
            result.end_effector_pose.position[1],
            0.4,
            places=6,
        )

    def test_joint_limit_violation(self) -> None:
        with self.assertRaises(
            JointLimitError
        ):
            self.engine.calculate(
                (10.0, 0.0)
            )

    def test_joint_frame_count(self) -> None:
        result = self.engine.calculate(
            (0.1, 0.2)
        )

        self.assertEqual(
            len(result.joint_frames),
            3,
        )

    def test_transform_inverse(self) -> None:
        transform = TransformUtils.compose(
            TransformUtils.translation(
                1.0,
                2.0,
                3.0,
            ),
            TransformUtils.rotation_z(
                0.5
            ),
        )

        inverse = TransformUtils.inverse(
            transform
        )

        self.assertTrue(
            np.allclose(
                transform @ inverse,
                np.eye(4),
                atol=1e-8,
            )
        )

    def test_quaternion_conversion(self) -> None:
        rotation = RotationUtils.euler_xyz_to_matrix(
            0.2,
            0.3,
            0.4,
        )

        quaternion = (
            RotationUtils.matrix_to_quaternion_xyzw(
                rotation
            )
        )

        recovered = (
            RotationUtils.quaternion_xyzw_to_matrix(
                quaternion
            )
        )

        self.assertTrue(
            np.allclose(
                rotation,
                recovered,
                atol=1e-8,
            )
        )

    def test_euler_conversion(self) -> None:
        original = (
            0.2,
            0.3,
            0.4,
        )

        rotation = RotationUtils.euler_xyz_to_matrix(
            *original
        )

        recovered = RotationUtils.matrix_to_euler_xyz(
            rotation
        )

        self.assertTrue(
            np.allclose(
                original,
                recovered,
                atol=1e-8,
            )
        )

    def test_jacobian_shape(self) -> None:
        result = self.engine.jacobian(
            (0.1, 0.2)
        )

        jacobian = np.asarray(
            result.geometric_jacobian
        )

        self.assertEqual(
            jacobian.shape,
            (6, 2),
        )

    def test_planar_jacobian(self) -> None:
        result = self.engine.jacobian(
            (0.0, 0.0)
        )

        jacobian = np.asarray(
            result.geometric_jacobian
        )

        self.assertAlmostEqual(
            jacobian[1, 0],
            0.9,
            places=6,
        )

        self.assertAlmostEqual(
            jacobian[1, 1],
            0.4,
            places=6,
        )

    def test_cartesian_velocity(self) -> None:
        result = self.engine.cartesian_velocity(
            (0.0, 0.0),
            (1.0, 0.0),
        )

        self.assertAlmostEqual(
            result.linear_velocity[1],
            0.9,
            places=6,
        )

    def test_prismatic_joint(self) -> None:
        config = SyntheticRobotFactory.prismatic_arm()

        engine = ForwardKinematicsEngine(
            config
        )

        result = engine.calculate(
            (0.5,)
        )

        self.assertAlmostEqual(
            result.end_effector_pose.position[2],
            0.7,
            places=6,
        )

    def test_prismatic_jacobian(self) -> None:
        config = SyntheticRobotFactory.prismatic_arm()

        engine = ForwardKinematicsEngine(
            config
        )

        result = engine.jacobian(
            (0.5,)
        )

        jacobian = np.asarray(
            result.geometric_jacobian
        )

        self.assertTrue(
            np.allclose(
                jacobian[:3, 0],
                [0.0, 0.0, 1.0],
            )
        )

    def test_tool_transform(self) -> None:
        tool = TransformUtils.translation(
            0.1,
            0.0,
            0.0,
        )

        config = RobotKinematicsConfig(
            robot_id="tool-test",
            tenant_id="physical_ai_lab",
            joints=self.config.joints,
            tool_transform=matrix_tuple(tool),
        )

        engine = ForwardKinematicsEngine(
            config
        )

        result = engine.calculate(
            (0.0, 0.0)
        )

        self.assertAlmostEqual(
            result.end_effector_pose.position[0],
            1.0,
            places=6,
        )

    def test_batch_calculation(self) -> None:
        results = self.engine.calculate_batch(
            [
                (0.0, 0.0),
                (0.1, 0.2),
                (0.3, 0.4),
            ]
        )

        self.assertEqual(
            len(results),
            3,
        )

    def test_pose_comparison(self) -> None:
        pose_a = self.engine.calculate(
            (0.0, 0.0)
        ).end_effector_pose

        pose_b = self.engine.calculate(
            (0.1, 0.2)
        ).end_effector_pose

        comparison = (
            ForwardKinematicsEngine.compare_poses(
                pose_a,
                pose_b,
            )
        )

        self.assertGreater(
            comparison.position_error,
            0.0,
        )

    def test_singularity_analysis(self) -> None:
        report = self.engine.analyze_singularity(
            (0.0, 0.0)
        )

        self.assertEqual(
            report.status,
            SingularityStatus.UNDERACTUATED,
        )

    def test_workspace_violation(self) -> None:
        config = RobotKinematicsConfig(
            robot_id="restricted-arm",
            tenant_id="physical_ai_lab",
            joints=self.config.joints,
            workspace=WorkspaceBounds(
                minimum_x=-0.2,
                maximum_x=0.2,
                minimum_y=-0.2,
                maximum_y=0.2,
                minimum_z=-0.2,
                maximum_z=0.2,
            ),
        )

        engine = ForwardKinematicsEngine(
            config
        )

        with self.assertRaises(
            WorkspaceError
        ):
            engine.calculate(
                (0.0, 0.0)
            )

    def test_service_registration(self) -> None:
        service = ForwardKinematicsService()

        service.register_robot(
            self.config,
            self.access,
        )

        self.assertEqual(
            service.list_robots(
                self.access
            ),
            ("planar-arm-001",),
        )

    def test_tenant_isolation(self) -> None:
        service = ForwardKinematicsService()

        service.register_robot(
            self.config,
            self.access,
        )

        other = KinematicsAccessContext(
            tenant_id="other_tenant",
            user_id="other_user",
            can_read=True,
            can_calculate=True,
        )

        with self.assertRaises(
            RobotNotFoundError
        ):
            service.get_robot(
                "planar-arm-001",
                other,
            )

    def test_permission_denied(self) -> None:
        service = ForwardKinematicsService()

        service.register_robot(
            self.config,
            self.access,
        )

        viewer = KinematicsAccessContext(
            tenant_id="physical_ai_lab",
            user_id="viewer",
            can_read=True,
        )

        with self.assertRaises(
            KinematicsAuthorizationError
        ):
            service.calculate(
                "planar-arm-001",
                (0.0, 0.0),
                viewer,
            )

    def test_json_export(self) -> None:
        service = ForwardKinematicsService()

        service.register_robot(
            self.config,
            self.access,
        )

        exported = json.loads(
            service.export_result_json(
                "planar-arm-001",
                (0.1, 0.2),
                self.access,
            )
        )

        self.assertEqual(
            exported["robot_id"],
            "planar-arm-001",
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    config = SyntheticRobotFactory.three_dof_arm(
        robot_id="physical-ai-arm-001",
        tenant_id="physical_ai_lab",
    )

    access = SyntheticRobotFactory.access()

    service = ForwardKinematicsService()

    engine = service.register_robot(
        config,
        access,
    )

    joint_positions = (
        0.3,
        0.4,
        -0.2,
    )

    print("\nPHYSICAL AI — FORWARD KINEMATICS")
    print("=" * 70)

    result = service.calculate(
        config.robot_id,
        joint_positions,
        access,
    )

    print(
        "Robot:",
        result.robot_id,
    )

    print(
        "Joint positions:",
        result.joint_positions,
    )

    print(
        "End-effector position:",
        result.end_effector_pose.position,
    )

    print(
        "Euler XYZ:",
        result.end_effector_pose.euler_xyz,
    )

    print(
        "Quaternion XYZW:",
        result.end_effector_pose.quaternion_xyzw,
    )

    print("\nEND-EFFECTOR TRANSFORM")
    print(
        np.array(
            result.end_effector_pose.transform_matrix
        )
    )

    jacobian = engine.jacobian(
        joint_positions
    )

    print("\nGEOMETRIC JACOBIAN")
    print(
        np.array(
            jacobian.geometric_jacobian
        )
    )

    velocity = engine.cartesian_velocity(
        joint_positions,
        (0.1, 0.2, -0.1),
    )

    print(
        "\nCartesian linear velocity:",
        velocity.linear_velocity,
    )

    print(
        "Cartesian angular velocity:",
        velocity.angular_velocity,
    )

    singularity = engine.analyze_singularity(
        joint_positions
    )

    print(
        "\nSingularity status:",
        singularity.status.value,
    )

    print(
        "Manipulability:",
        singularity.manipulability,
    )

    print("\nFORWARD KINEMATICS REPORT")
    print("=" * 70)

    print(
        json.dumps(
            json_safe(result),
            indent=2,
            allow_nan=False,
        )
    )

    print("\nRUNNING UNIT TESTS")
    print("=" * 70)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestForwardKinematics
    )

    test_result = unittest.TextTestRunner(
        verbosity=2
    ).run(suite)

    if not test_result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
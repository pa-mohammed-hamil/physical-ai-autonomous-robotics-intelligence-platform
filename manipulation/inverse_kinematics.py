"""Inverse Kinematics module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/manipulation/inverse_kinematics.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================
Module: Inverse Kinematics

Features:
    - Serial robot-arm inverse kinematics
    - Revolute and prismatic joints
    - Standard Denavit-Hartenberg kinematics
    - Forward kinematics
    - Geometric Jacobian
    - Position-only inverse kinematics
    - Full 6D pose inverse kinematics
    - Damped least-squares optimization
    - Adaptive damping
    - Joint-limit enforcement
    - Joint-center regularization
    - Multi-start numerical optimization
    - Singularity diagnostics
    - Workspace checks
    - Position and orientation error metrics
    - Joint configuration scoring
    - Candidate solution ranking
    - Batch inverse kinematics
    - Thread-safe multi-tenant service
    - JSON serialization
    - Synthetic robot models
    - Unit tests

Dependencies:
    Python >= 3.11
    numpy >= 1.24

Install:
    pip install numpy

Safety:
    Simulation and planning only.

    This module never commands physical robot actuators.

    Numerical convergence does not guarantee collision
    freedom, dynamic feasibility, or safe execution.

    Physical deployment requires independent collision
    checking, robot-specific calibration, safety-rated
    limits, and trajectory validation.
"""

from __future__ import annotations

import json
import math
import threading
import unittest
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Sequence

import numpy as np


# ============================================================
# EXCEPTIONS
# ============================================================

class InverseKinematicsError(Exception):
    pass


class IKValidationError(InverseKinematicsError):
    pass


class IKGeometryError(InverseKinematicsError):
    pass


class IKAuthorizationError(InverseKinematicsError):
    pass


class IKRobotNotFoundError(InverseKinematicsError):
    pass


class IKConflictError(InverseKinematicsError):
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
        raise IKValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise IKValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise IKValidationError(
            f"{name} must be finite"
        )

    return result


def identifier(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 256
    ):
        raise IKValidationError(
            f"Invalid {name}"
        )

    return value.strip()


def vector(
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
        raise IKValidationError(
            f"{name} must contain numeric values"
        ) from exc

    if result.shape != (size,):
        raise IKValidationError(
            f"{name} must have shape ({size},)"
        )

    if not np.all(np.isfinite(result)):
        raise IKValidationError(
            f"{name} contains non-finite values"
        )

    return result


def matrix(
    values: Sequence[Sequence[float]],
    shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    try:
        result = np.asarray(
            values,
            dtype=float,
        )
    except (TypeError, ValueError) as exc:
        raise IKValidationError(
            f"{name} must be numeric"
        ) from exc

    if result.shape != shape:
        raise IKValidationError(
            f"{name} must have shape {shape}"
        )

    if not np.all(np.isfinite(result)):
        raise IKValidationError(
            f"{name} contains non-finite values"
        )

    return result


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(
        minimum,
        min(maximum, value),
    )


def rotation_x(angle: float) -> np.ndarray:
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


def rotation_y(angle: float) -> np.ndarray:
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


def rotation_z(angle: float) -> np.ndarray:
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


def rpy_to_matrix(
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    return (
        rotation_z(yaw)
        @ rotation_y(pitch)
        @ rotation_x(roll)
    )


def matrix_to_rpy(
    rotation: np.ndarray,
) -> tuple[float, float, float]:
    r = matrix(
        rotation,
        (3, 3),
        "rotation",
    )

    pitch = math.atan2(
        -r[2, 0],
        math.hypot(
            r[0, 0],
            r[1, 0],
        ),
    )

    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(
            r[2, 1],
            r[2, 2],
        )

        yaw = math.atan2(
            r[1, 0],
            r[0, 0],
        )
    else:
        roll = 0.0

        yaw = math.atan2(
            -r[0, 1],
            r[1, 1],
        )

    return (
        float(roll),
        float(pitch),
        float(yaw),
    )


def skew(
    values: Sequence[float],
) -> np.ndarray:
    x, y, z = vector(
        values,
        3,
        "skew vector",
    )

    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def rotation_log(
    rotation: np.ndarray,
) -> np.ndarray:
    """
    SO(3) logarithm expressed as a rotation vector.

    The result is in the coordinate frame of the supplied
    relative rotation matrix.
    """
    r = matrix(
        rotation,
        (3, 3),
        "relative rotation",
    )

    cosine = clamp(
        (float(np.trace(r)) - 1.0) / 2.0,
        -1.0,
        1.0,
    )

    angle = math.acos(cosine)

    vee = np.array(
        [
            r[2, 1] - r[1, 2],
            r[0, 2] - r[2, 0],
            r[1, 0] - r[0, 1],
        ],
        dtype=float,
    )

    if angle < 1e-7:
        return 0.5 * vee

    if math.pi - angle < 1e-5:
        # Stable axis recovery near a 180-degree rotation.
        diagonal = np.maximum(
            (np.diag(r) + 1.0) / 2.0,
            0.0,
        )

        axis = np.sqrt(diagonal)

        index = int(
            np.argmax(axis)
        )

        if axis[index] > 1e-8:
            if index == 0:
                axis[1] = (
                    r[0, 1] + r[1, 0]
                ) / (4.0 * axis[0])

                axis[2] = (
                    r[0, 2] + r[2, 0]
                ) / (4.0 * axis[0])

            elif index == 1:
                axis[0] = (
                    r[0, 1] + r[1, 0]
                ) / (4.0 * axis[1])

                axis[2] = (
                    r[1, 2] + r[2, 1]
                ) / (4.0 * axis[1])

            else:
                axis[0] = (
                    r[0, 2] + r[2, 0]
                ) / (4.0 * axis[2])

                axis[1] = (
                    r[1, 2] + r[2, 1]
                ) / (4.0 * axis[2])

        norm = float(
            np.linalg.norm(axis)
        )

        if norm < 1e-8:
            raise IKGeometryError(
                "Unable to recover rotation axis"
            )

        return angle * axis / norm

    return (
        angle
        / (2.0 * math.sin(angle))
        * vee
    )


def orientation_error(
    current_rotation: np.ndarray,
    target_rotation: np.ndarray,
) -> np.ndarray:
    """
    Spatial orientation error in the base frame.

    For small errors:
        error ~= desired_angular_velocity * dt
    """
    current = matrix(
        current_rotation,
        (3, 3),
        "current rotation",
    )

    target = matrix(
        target_rotation,
        (3, 3),
        "target rotation",
    )

    relative = target @ current.T

    return rotation_log(relative)


def homogeneous_transform(
    rotation: np.ndarray,
    position: Sequence[float],
) -> np.ndarray:
    result = np.eye(
        4,
        dtype=float,
    )

    result[:3, :3] = matrix(
        rotation,
        (3, 3),
        "rotation",
    )

    result[:3, 3] = vector(
        position,
        3,
        "position",
    )

    return result


def validate_rotation(
    rotation: np.ndarray,
    tolerance: float = 1e-5,
) -> None:
    r = matrix(
        rotation,
        (3, 3),
        "rotation",
    )

    if not np.allclose(
        r.T @ r,
        np.eye(3),
        atol=tolerance,
    ):
        raise IKGeometryError(
            "Rotation matrix is not orthonormal"
        )

    if not math.isclose(
        float(np.linalg.det(r)),
        1.0,
        abs_tol=tolerance,
    ):
        raise IKGeometryError(
            "Rotation determinant must equal +1"
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
            key: json_safe(item)
            for key, item in asdict(value).items()
        }

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
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

class JointType(str, Enum):
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"


class IKTaskType(str, Enum):
    POSITION = "position"
    POSE = "pose"


class IKAlgorithm(str, Enum):
    DAMPED_LEAST_SQUARES = "damped_least_squares"
    ADAPTIVE_DAMPED_LEAST_SQUARES = (
        "adaptive_damped_least_squares"
    )


class IKStatus(str, Enum):
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    STAGNATED = "stagnated"
    OUT_OF_WORKSPACE = "out_of_workspace"
    NUMERICAL_FAILURE = "numerical_failure"


class IKSingularityLevel(str, Enum):
    REGULAR = "regular"
    NEAR_SINGULAR = "near_singular"
    SINGULAR = "singular"


# ============================================================
# ROBOT CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class IKJoint:
    name: str
    joint_type: JointType

    minimum: float
    maximum: float

    dh_a_m: float = 0.0
    dh_alpha_rad: float = 0.0
    dh_d_m: float = 0.0
    dh_theta_rad: float = 0.0

    home_position: float = 0.0

    def __post_init__(self) -> None:
        identifier(
            self.name,
            "joint name",
        )

        for name in (
            "minimum",
            "maximum",
            "dh_a_m",
            "dh_alpha_rad",
            "dh_d_m",
            "dh_theta_rad",
            "home_position",
        ):
            finite(
                getattr(self, name),
                name,
            )

        if self.minimum >= self.maximum:
            raise IKValidationError(
                f"Invalid limits for {self.name}"
            )

        if not (
            self.minimum
            <= self.home_position
            <= self.maximum
        ):
            raise IKValidationError(
                f"Home position outside limits: {self.name}"
            )


@dataclass(frozen=True)
class IKWorkspace:
    minimum_xyz: tuple[
        float,
        float,
        float,
    ]

    maximum_xyz: tuple[
        float,
        float,
        float,
    ]

    def __post_init__(self) -> None:
        minimum = vector(
            self.minimum_xyz,
            3,
            "workspace minimum",
        )

        maximum = vector(
            self.maximum_xyz,
            3,
            "workspace maximum",
        )

        if np.any(minimum >= maximum):
            raise IKValidationError(
                "Invalid workspace bounds"
            )

    def contains(
        self,
        position: Sequence[float],
    ) -> bool:
        p = vector(
            position,
            3,
            "position",
        )

        minimum = np.asarray(
            self.minimum_xyz,
            dtype=float,
        )

        maximum = np.asarray(
            self.maximum_xyz,
            dtype=float,
        )

        return bool(
            np.all(p >= minimum)
            and np.all(p <= maximum)
        )


@dataclass(frozen=True)
class IKRobotModel:
    robot_id: str
    tenant_id: str

    joints: tuple[IKJoint, ...]

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

    workspace: IKWorkspace | None = None

    def __post_init__(self) -> None:
        identifier(
            self.robot_id,
            "robot_id",
        )

        identifier(
            self.tenant_id,
            "tenant_id",
        )

        if not self.joints:
            raise IKValidationError(
                "Robot must contain at least one joint"
            )

        names = [
            joint.name
            for joint in self.joints
        ]

        if len(names) != len(set(names)):
            raise IKValidationError(
                "Joint names must be unique"
            )

        for name in (
            "base_transform",
            "tool_transform",
        ):
            transform = matrix(
                getattr(self, name),
                (4, 4),
                name,
            )

            if not np.allclose(
                transform[3],
                (0.0, 0.0, 0.0, 1.0),
                atol=1e-8,
            ):
                raise IKGeometryError(
                    f"Invalid homogeneous {name}"
                )

            validate_rotation(
                transform[:3, :3]
            )

    @property
    def degrees_of_freedom(self) -> int:
        return len(self.joints)

    def home_configuration(self) -> np.ndarray:
        return np.array(
            [
                joint.home_position
                for joint in self.joints
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class IKSolverConfig:
    algorithm: IKAlgorithm = (
        IKAlgorithm.ADAPTIVE_DAMPED_LEAST_SQUARES
    )

    maximum_iterations: int = 300

    position_tolerance_m: float = 1e-4
    orientation_tolerance_rad: float = 1e-3

    damping: float = 0.05
    minimum_damping: float = 0.005
    maximum_damping: float = 0.50

    step_size: float = 0.8
    maximum_joint_step: float = 0.15

    position_weight: float = 1.0
    orientation_weight: float = 0.3

    joint_center_weight: float = 0.002

    stagnation_tolerance: float = 1e-10
    stagnation_iterations: int = 30

    random_restarts: int = 8
    random_seed: int = 42

    singularity_threshold: float = 1e-4

    maximum_ranked_solutions: int = 10

    def __post_init__(self) -> None:
        if self.maximum_iterations < 1:
            raise IKValidationError(
                "maximum_iterations must be positive"
            )

        if self.random_restarts < 0:
            raise IKValidationError(
                "random_restarts cannot be negative"
            )

        if self.stagnation_iterations < 1:
            raise IKValidationError(
                "stagnation_iterations must be positive"
            )

        if self.maximum_ranked_solutions < 1:
            raise IKValidationError(
                "maximum_ranked_solutions must be positive"
            )

        for name in (
            "position_tolerance_m",
            "orientation_tolerance_rad",
            "damping",
            "minimum_damping",
            "maximum_damping",
            "step_size",
            "maximum_joint_step",
            "position_weight",
            "orientation_weight",
            "stagnation_tolerance",
            "singularity_threshold",
        ):
            if finite(
                getattr(self, name),
                name,
            ) <= 0:
                raise IKValidationError(
                    f"{name} must be positive"
                )

        if self.joint_center_weight < 0:
            raise IKValidationError(
                "joint_center_weight cannot be negative"
            )

        if self.minimum_damping > self.maximum_damping:
            raise IKValidationError(
                "Invalid damping bounds"
            )


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

@dataclass(frozen=True)
class IKPose:
    position_xyz: tuple[
        float,
        float,
        float,
    ]

    rotation_matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]

    def __post_init__(self) -> None:
        vector(
            self.position_xyz,
            3,
            "pose position",
        )

        validate_rotation(
            np.asarray(
                self.rotation_matrix,
                dtype=float,
            )
        )

    @classmethod
    def from_rpy(
        cls,
        position_xyz: Sequence[float],
        roll: float,
        pitch: float,
        yaw: float,
    ) -> IKPose:
        rotation = rpy_to_matrix(
            roll,
            pitch,
            yaw,
        )

        return cls(
            position_xyz=tuple(
                float(value)
                for value in vector(
                    position_xyz,
                    3,
                    "position",
                )
            ),
            rotation_matrix=tuple(
                tuple(
                    float(value)
                    for value in row
                )
                for row in rotation
            ),
        )

    def transform(self) -> np.ndarray:
        return homogeneous_transform(
            np.asarray(
                self.rotation_matrix,
                dtype=float,
            ),
            self.position_xyz,
        )


@dataclass(frozen=True)
class IKRequest:
    request_id: str

    target: IKPose

    task_type: IKTaskType = IKTaskType.POSITION

    initial_configuration: tuple[
        float,
        ...
    ] | None = None

    use_random_restarts: bool = True

    def __post_init__(self) -> None:
        identifier(
            self.request_id,
            "request_id",
        )


@dataclass(frozen=True)
class IKSingularityReport:
    minimum_singular_value: float

    maximum_singular_value: float

    condition_number: float | None

    effective_rank: int

    task_dimension: int

    level: IKSingularityLevel


@dataclass(frozen=True)
class IKSolution:
    solution_id: str

    status: IKStatus

    joint_positions: tuple[float, ...]

    position_error_m: float
    orientation_error_rad: float

    iterations: int

    objective_value: float

    joint_center_score: float

    singularity: IKSingularityReport

    message: str


@dataclass(frozen=True)
class IKResult:
    result_id: str

    request_id: str
    robot_id: str
    tenant_id: str

    task_type: IKTaskType

    status: IKStatus

    best_solution: IKSolution | None

    solutions: tuple[IKSolution, ...]

    attempted_starts: int

    created_at: datetime


@dataclass(frozen=True)
class IKAccessContext:
    tenant_id: str
    user_id: str

    can_read: bool = False
    can_solve: bool = False
    can_manage: bool = False


# ============================================================
# DENAVIT-HARTENBERG KINEMATICS
# ============================================================

class DHKinematics:

    @staticmethod
    def dh_transform(
        a: float,
        alpha: float,
        d: float,
        theta: float,
    ) -> np.ndarray:
        ct = math.cos(theta)
        st = math.sin(theta)

        ca = math.cos(alpha)
        sa = math.sin(alpha)

        return np.array(
            [
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0.0, sa, ca, d],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def __init__(
        self,
        robot: IKRobotModel,
    ) -> None:
        self.robot = robot

    def validate_configuration(
        self,
        configuration: Sequence[float],
    ) -> np.ndarray:
        q = vector(
            configuration,
            self.robot.degrees_of_freedom,
            "joint configuration",
        )

        return q

    def within_joint_limits(
        self,
        configuration: Sequence[float],
    ) -> bool:
        q = self.validate_configuration(
            configuration
        )

        return all(
            joint.minimum - 1e-10
            <= float(value)
            <= joint.maximum + 1e-10
            for joint, value
            in zip(
                self.robot.joints,
                q,
            )
        )

    def clamp_configuration(
        self,
        configuration: Sequence[float],
    ) -> np.ndarray:
        q = self.validate_configuration(
            configuration
        )

        return np.array(
            [
                clamp(
                    float(value),
                    joint.minimum,
                    joint.maximum,
                )
                for joint, value
                in zip(
                    self.robot.joints,
                    q,
                )
            ],
            dtype=float,
        )

    def transforms(
        self,
        configuration: Sequence[float],
    ) -> list[np.ndarray]:
        q = self.validate_configuration(
            configuration
        )

        transform = matrix(
            self.robot.base_transform,
            (4, 4),
            "base transform",
        ).copy()

        transforms = [
            transform.copy()
        ]

        for joint, value in zip(
            self.robot.joints,
            q,
        ):
            theta = joint.dh_theta_rad
            d = joint.dh_d_m

            if joint.joint_type == JointType.REVOLUTE:
                theta += float(value)

            else:
                d += float(value)

            transform = (
                transform
                @ self.dh_transform(
                    joint.dh_a_m,
                    joint.dh_alpha_rad,
                    d,
                    theta,
                )
            )

            transforms.append(
                transform.copy()
            )

        return transforms

    def forward_transform(
        self,
        configuration: Sequence[float],
    ) -> np.ndarray:
        frames = self.transforms(
            configuration
        )

        tool = matrix(
            self.robot.tool_transform,
            (4, 4),
            "tool transform",
        )

        return frames[-1] @ tool

    def forward_pose(
        self,
        configuration: Sequence[float],
    ) -> IKPose:
        transform = self.forward_transform(
            configuration
        )

        return IKPose(
            position_xyz=tuple(
                float(value)
                for value in transform[:3, 3]
            ),
            rotation_matrix=tuple(
                tuple(
                    float(value)
                    for value in row
                )
                for row in transform[:3, :3]
            ),
        )

    def jacobian(
        self,
        configuration: Sequence[float],
    ) -> np.ndarray:
        frames = self.transforms(
            configuration
        )

        end_effector = self.forward_transform(
            configuration
        )

        end_position = end_effector[:3, 3]

        dof = self.robot.degrees_of_freedom

        jacobian = np.zeros(
            (6, dof),
            dtype=float,
        )

        for index, joint in enumerate(
            self.robot.joints
        ):
            frame = frames[index]

            origin = frame[:3, 3]
            axis = frame[:3, 2]

            if joint.joint_type == JointType.REVOLUTE:
                jacobian[:3, index] = np.cross(
                    axis,
                    end_position - origin,
                )

                jacobian[3:, index] = axis

            else:
                jacobian[:3, index] = axis
                jacobian[3:, index] = 0.0

        return jacobian


# ============================================================
# JOINT LIMIT MANAGEMENT
# ============================================================

class IKJointLimitManager:

    def __init__(
        self,
        robot: IKRobotModel,
    ) -> None:
        self.robot = robot

        self.minimum = np.array(
            [
                joint.minimum
                for joint in robot.joints
            ],
            dtype=float,
        )

        self.maximum = np.array(
            [
                joint.maximum
                for joint in robot.joints
            ],
            dtype=float,
        )

        self.center = (
            self.minimum + self.maximum
        ) / 2.0

        self.range = (
            self.maximum - self.minimum
        )

    def clamp(
        self,
        configuration: Sequence[float],
    ) -> np.ndarray:
        q = vector(
            configuration,
            self.robot.degrees_of_freedom,
            "configuration",
        )

        return np.clip(
            q,
            self.minimum,
            self.maximum,
        )

    def center_gradient(
        self,
        configuration: Sequence[float],
    ) -> np.ndarray:
        q = vector(
            configuration,
            self.robot.degrees_of_freedom,
            "configuration",
        )

        normalized_offset = (
            q - self.center
        ) / self.range

        return (
            -2.0
            * normalized_offset
            / self.range
        )

    def center_score(
        self,
        configuration: Sequence[float],
    ) -> float:
        q = vector(
            configuration,
            self.robot.degrees_of_freedom,
            "configuration",
        )

        normalized_distance = (
            2.0
            * np.abs(
                q - self.center
            )
            / self.range
        )

        return float(
            np.clip(
                1.0
                - np.mean(
                    normalized_distance
                ),
                0.0,
                1.0,
            )
        )

    def random_configuration(
        self,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return rng.uniform(
            self.minimum,
            self.maximum,
        )


# ============================================================
# SINGULARITY ANALYSIS
# ============================================================

class IKSingularityAnalyzer:

    def __init__(
        self,
        threshold: float = 1e-4,
    ) -> None:
        self.threshold = threshold

    def analyze(
        self,
        jacobian: np.ndarray,
    ) -> IKSingularityReport:
        values = np.linalg.svd(
            jacobian,
            compute_uv=False,
        )

        if len(values) == 0:
            return IKSingularityReport(
                minimum_singular_value=0.0,
                maximum_singular_value=0.0,
                condition_number=None,
                effective_rank=0,
                task_dimension=jacobian.shape[0],
                level=IKSingularityLevel.SINGULAR,
            )

        minimum = float(
            np.min(values)
        )

        maximum = float(
            np.max(values)
        )

        rank = int(
            np.sum(
                values > self.threshold
            )
        )

        required_rank = min(
            jacobian.shape
        )

        if minimum <= 1e-12:
            condition = None
        else:
            condition = maximum / minimum

        if rank < required_rank:
            level = IKSingularityLevel.SINGULAR

        elif minimum < 10.0 * self.threshold:
            level = (
                IKSingularityLevel.NEAR_SINGULAR
            )

        else:
            level = IKSingularityLevel.REGULAR

        return IKSingularityReport(
            minimum_singular_value=minimum,
            maximum_singular_value=maximum,
            condition_number=condition,
            effective_rank=rank,
            task_dimension=jacobian.shape[0],
            level=level,
        )


# ============================================================
# INVERSE KINEMATICS SOLVER
# ============================================================

class NumericalIKSolver:

    def __init__(
        self,
        robot: IKRobotModel,
        config: IKSolverConfig | None = None,
    ) -> None:
        self.robot = robot

        self.config = (
            config
            or IKSolverConfig()
        )

        self.kinematics = DHKinematics(
            robot
        )

        self.limits = IKJointLimitManager(
            robot
        )

        self.singularity_analyzer = (
            IKSingularityAnalyzer(
                self.config.singularity_threshold
            )
        )

        self._lock = threading.RLock()

    def _task_error(
        self,
        configuration: np.ndarray,
        target: IKPose,
        task_type: IKTaskType,
    ) -> tuple[
        np.ndarray,
        float,
        float,
    ]:
        transform = (
            self.kinematics.forward_transform(
                configuration
            )
        )

        current_position = transform[:3, 3]
        current_rotation = transform[:3, :3]

        target_position = vector(
            target.position_xyz,
            3,
            "target position",
        )

        target_rotation = matrix(
            target.rotation_matrix,
            (3, 3),
            "target rotation",
        )

        position_error = (
            target_position
            - current_position
        )

        rotation_error = orientation_error(
            current_rotation,
            target_rotation,
        )

        position_norm = float(
            np.linalg.norm(
                position_error
            )
        )

        orientation_norm = float(
            np.linalg.norm(
                rotation_error
            )
        )

        if task_type == IKTaskType.POSITION:
            task_error = (
                self.config.position_weight
                * position_error
            )

        else:
            task_error = np.concatenate(
                [
                    self.config.position_weight
                    * position_error,

                    self.config.orientation_weight
                    * rotation_error,
                ]
            )

        return (
            task_error,
            position_norm,
            orientation_norm,
        )

    def _task_jacobian(
        self,
        configuration: np.ndarray,
        task_type: IKTaskType,
    ) -> np.ndarray:
        full_jacobian = (
            self.kinematics.jacobian(
                configuration
            )
        )

        if task_type == IKTaskType.POSITION:
            return (
                self.config.position_weight
                * full_jacobian[:3]
            )

        return np.vstack(
            [
                self.config.position_weight
                * full_jacobian[:3],

                self.config.orientation_weight
                * full_jacobian[3:],
            ]
        )

    def _converged(
        self,
        position_error: float,
        orientation_error_value: float,
        task_type: IKTaskType,
    ) -> bool:
        if (
            position_error
            > self.config.position_tolerance_m
        ):
            return False

        if (
            task_type == IKTaskType.POSE
            and orientation_error_value
            > self.config.orientation_tolerance_rad
        ):
            return False

        return True

    def _adaptive_damping(
        self,
        jacobian: np.ndarray,
    ) -> float:
        if (
            self.config.algorithm
            == IKAlgorithm.DAMPED_LEAST_SQUARES
        ):
            return self.config.damping

        singular_values = np.linalg.svd(
            jacobian,
            compute_uv=False,
        )

        if len(singular_values) == 0:
            return self.config.maximum_damping

        minimum = float(
            np.min(
                singular_values
            )
        )

        threshold = (
            10.0
            * self.config.singularity_threshold
        )

        if minimum >= threshold:
            return self.config.minimum_damping

        ratio = clamp(
            1.0 - minimum / threshold,
            0.0,
            1.0,
        )

        return (
            self.config.minimum_damping
            + ratio
            * (
                self.config.maximum_damping
                - self.config.minimum_damping
            )
        )

    def _damped_step(
        self,
        jacobian: np.ndarray,
        error: np.ndarray,
        damping: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        task_dimension = jacobian.shape[0]

        regularized = (
            jacobian @ jacobian.T
            + damping * damping
            * np.eye(task_dimension)
        )

        try:
            solved_error = np.linalg.solve(
                regularized,
                error,
            )

            solved_jacobian = np.linalg.solve(
                regularized,
                jacobian,
            )

        except np.linalg.LinAlgError:
            solved_error = np.linalg.lstsq(
                regularized,
                error,
                rcond=None,
            )[0]

            solved_jacobian = np.linalg.lstsq(
                regularized,
                jacobian,
                rcond=None,
            )[0]

        step = (
            jacobian.T
            @ solved_error
        )

        damped_pseudoinverse = (
            jacobian.T
            @ np.linalg.pinv(
                regularized
            )
        )

        return (
            step,
            damped_pseudoinverse,
        )

    def _objective(
        self,
        configuration: np.ndarray,
        target: IKPose,
        task_type: IKTaskType,
    ) -> float:
        error, _, _ = self._task_error(
            configuration,
            target,
            task_type,
        )

        return float(
            np.dot(
                error,
                error,
            )
        )

    def _make_solution(
        self,
        *,
        status: IKStatus,
        configuration: np.ndarray,
        target: IKPose,
        task_type: IKTaskType,
        iterations: int,
        message: str,
    ) -> IKSolution:
        _, position_error, orientation_error_value = (
            self._task_error(
                configuration,
                target,
                task_type,
            )
        )

        jacobian = self._task_jacobian(
            configuration,
            task_type,
        )

        singularity = (
            self.singularity_analyzer.analyze(
                jacobian
            )
        )

        return IKSolution(
            solution_id=new_id(),
            status=status,
            joint_positions=tuple(
                float(value)
                for value in configuration
            ),
            position_error_m=position_error,
            orientation_error_rad=orientation_error_value,
            iterations=iterations,
            objective_value=self._objective(
                configuration,
                target,
                task_type,
            ),
            joint_center_score=self.limits.center_score(
                configuration
            ),
            singularity=singularity,
            message=message,
        )

    def solve_from_seed(
        self,
        request: IKRequest,
        seed: Sequence[float],
    ) -> IKSolution:
        with self._lock:
            configuration = self.limits.clamp(
                seed
            )

            best_configuration = (
                configuration.copy()
            )

            best_objective = self._objective(
                configuration,
                request.target,
                request.task_type,
            )

            stagnation_count = 0

            for iteration in range(
                self.config.maximum_iterations + 1
            ):
                (
                    error,
                    position_error,
                    orientation_error_value,
                ) = self._task_error(
                    configuration,
                    request.target,
                    request.task_type,
                )

                if self._converged(
                    position_error,
                    orientation_error_value,
                    request.task_type,
                ):
                    return self._make_solution(
                        status=IKStatus.CONVERGED,
                        configuration=configuration,
                        target=request.target,
                        task_type=request.task_type,
                        iterations=iteration,
                        message="Target reached within configured tolerances",
                    )

                if (
                    iteration
                    == self.config.maximum_iterations
                ):
                    break

                jacobian = self._task_jacobian(
                    configuration,
                    request.task_type,
                )

                damping = self._adaptive_damping(
                    jacobian
                )

                try:
                    step, pseudoinverse = (
                        self._damped_step(
                            jacobian,
                            error,
                            damping,
                        )
                    )

                except np.linalg.LinAlgError:
                    return self._make_solution(
                        status=IKStatus.NUMERICAL_FAILURE,
                        configuration=best_configuration,
                        target=request.target,
                        task_type=request.task_type,
                        iterations=iteration,
                        message="Numerical linear algebra failure",
                    )

                if self.config.joint_center_weight > 0:
                    gradient = (
                        self.limits.center_gradient(
                            configuration
                        )
                    )

                    nullspace = (
                        np.eye(
                            self.robot.degrees_of_freedom
                        )
                        - pseudoinverse @ jacobian
                    )

                    step += (
                        self.config.joint_center_weight
                        * nullspace
                        @ gradient
                    )

                if not np.all(
                    np.isfinite(step)
                ):
                    return self._make_solution(
                        status=IKStatus.NUMERICAL_FAILURE,
                        configuration=best_configuration,
                        target=request.target,
                        task_type=request.task_type,
                        iterations=iteration,
                        message="Non-finite optimization step",
                    )

                step = np.clip(
                    step,
                    -self.config.maximum_joint_step,
                    self.config.maximum_joint_step,
                )

                current_objective = (
                    self._objective(
                        configuration,
                        request.target,
                        request.task_type,
                    )
                )

                accepted = False

                # Backtracking prevents unnecessarily
                # large updates from increasing task error.
                scale = self.config.step_size

                for _ in range(12):
                    trial = self.limits.clamp(
                        configuration
                        + scale * step
                    )

                    trial_objective = (
                        self._objective(
                            trial,
                            request.target,
                            request.task_type,
                        )
                    )

                    if (
                        trial_objective
                        < current_objective
                    ):
                        configuration = trial
                        accepted = True
                        break

                    scale *= 0.5

                objective = self._objective(
                    configuration,
                    request.target,
                    request.task_type,
                )

                if objective < best_objective:
                    best_objective = objective
                    best_configuration = (
                        configuration.copy()
                    )

                improvement = (
                    current_objective - objective
                )

                if (
                    not accepted
                    or improvement
                    < self.config.stagnation_tolerance
                ):
                    stagnation_count += 1

                else:
                    stagnation_count = 0

                if (
                    stagnation_count
                    >= self.config.stagnation_iterations
                ):
                    return self._make_solution(
                        status=IKStatus.STAGNATED,
                        configuration=best_configuration,
                        target=request.target,
                        task_type=request.task_type,
                        iterations=iteration + 1,
                        message="Optimization stagnated",
                    )

            return self._make_solution(
                status=IKStatus.MAX_ITERATIONS,
                configuration=best_configuration,
                target=request.target,
                task_type=request.task_type,
                iterations=self.config.maximum_iterations,
                message="Maximum iterations reached",
            )

    def solve(
        self,
        request: IKRequest,
    ) -> IKResult:
        with self._lock:
            target_position = vector(
                request.target.position_xyz,
                3,
                "target position",
            )

            if (
                self.robot.workspace is not None
                and not self.robot.workspace.contains(
                    target_position
                )
            ):
                return IKResult(
                    result_id=new_id(),
                    request_id=request.request_id,
                    robot_id=self.robot.robot_id,
                    tenant_id=self.robot.tenant_id,
                    task_type=request.task_type,
                    status=IKStatus.OUT_OF_WORKSPACE,
                    best_solution=None,
                    solutions=(),
                    attempted_starts=0,
                    created_at=utc_now(),
                )

            if request.initial_configuration is None:
                initial = (
                    self.robot.home_configuration()
                )
            else:
                initial = vector(
                    request.initial_configuration,
                    self.robot.degrees_of_freedom,
                    "initial configuration",
                )

            seeds = [
                self.limits.clamp(
                    initial
                )
            ]

            if request.use_random_restarts:
                rng = np.random.default_rng(
                    self.config.random_seed
                )

                for _ in range(
                    self.config.random_restarts
                ):
                    seeds.append(
                        self.limits.random_configuration(
                            rng
                        )
                    )

            solutions = [
                self.solve_from_seed(
                    request,
                    seed,
                )
                for seed in seeds
            ]

            # Prefer converged solutions, then lower
            # residual error, then joint-center score.
            solutions.sort(
                key=lambda solution: (
                    solution.status
                    != IKStatus.CONVERGED,
                    solution.objective_value,
                    -solution.joint_center_score,
                )
            )

            unique: list[IKSolution] = []

            for solution in solutions:
                configuration = np.asarray(
                    solution.joint_positions,
                    dtype=float,
                )

                duplicate = any(
                    np.linalg.norm(
                        configuration
                        - np.asarray(
                            existing.joint_positions,
                            dtype=float,
                        )
                    ) < 1e-3
                    for existing in unique
                )

                if not duplicate:
                    unique.append(
                        solution
                    )

            unique = unique[
                :self.config.maximum_ranked_solutions
            ]

            best = (
                unique[0]
                if unique
                else None
            )

            return IKResult(
                result_id=new_id(),
                request_id=request.request_id,
                robot_id=self.robot.robot_id,
                tenant_id=self.robot.tenant_id,
                task_type=request.task_type,
                status=(
                    best.status
                    if best is not None
                    else IKStatus.NUMERICAL_FAILURE
                ),
                best_solution=best,
                solutions=tuple(unique),
                attempted_starts=len(seeds),
                created_at=utc_now(),
            )

    def solve_batch(
        self,
        requests: Sequence[IKRequest],
    ) -> tuple[IKResult, ...]:
        return tuple(
            self.solve(request)
            for request in requests
        )


# ============================================================
# MULTI-TENANT SERVICE
# ============================================================

class InverseKinematicsService:

    def __init__(self) -> None:
        self._solvers: dict[
            tuple[str, str],
            NumericalIKSolver,
        ] = {}

        self._lock = threading.RLock()

    @staticmethod
    def _authorize(
        access: IKAccessContext,
        permission: str,
    ) -> None:
        if not getattr(
            access,
            permission,
            False,
        ):
            raise IKAuthorizationError(
                f"Missing permission: {permission}"
            )

    def register_robot(
        self,
        robot: IKRobotModel,
        access: IKAccessContext,
        config: IKSolverConfig | None = None,
    ) -> None:
        self._authorize(
            access,
            "can_manage",
        )

        if robot.tenant_id != access.tenant_id:
            raise IKAuthorizationError(
                "Cross-tenant registration forbidden"
            )

        key = (
            robot.tenant_id,
            robot.robot_id,
        )

        with self._lock:
            if key in self._solvers:
                raise IKConflictError(
                    "Robot already registered"
                )

            self._solvers[key] = (
                NumericalIKSolver(
                    robot,
                    config,
                )
            )

    def get_solver(
        self,
        robot_id: str,
        access: IKAccessContext,
    ) -> NumericalIKSolver:
        self._authorize(
            access,
            "can_read",
        )

        key = (
            access.tenant_id,
            robot_id,
        )

        with self._lock:
            solver = self._solvers.get(
                key
            )

        if solver is None:
            raise IKRobotNotFoundError(
                "Robot IK solver not found"
            )

        return solver

    def solve(
        self,
        robot_id: str,
        request: IKRequest,
        access: IKAccessContext,
    ) -> IKResult:
        self._authorize(
            access,
            "can_solve",
        )

        solver = self.get_solver(
            robot_id,
            access,
        )

        return solver.solve(
            request
        )

    def forward_kinematics(
        self,
        robot_id: str,
        configuration: Sequence[float],
        access: IKAccessContext,
    ) -> IKPose:
        solver = self.get_solver(
            robot_id,
            access,
        )

        return solver.kinematics.forward_pose(
            configuration
        )

    def jacobian(
        self,
        robot_id: str,
        configuration: Sequence[float],
        access: IKAccessContext,
    ) -> np.ndarray:
        solver = self.get_solver(
            robot_id,
            access,
        )

        return solver.kinematics.jacobian(
            configuration
        )

    def list_robots(
        self,
        access: IKAccessContext,
    ) -> tuple[str, ...]:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            return tuple(
                robot_id
                for tenant_id, robot_id
                in self._solvers
                if tenant_id == access.tenant_id
            )

    def export_json(
        self,
        result: IKResult,
        access: IKAccessContext,
    ) -> str:
        self._authorize(
            access,
            "can_read",
        )

        if result.tenant_id != access.tenant_id:
            raise IKAuthorizationError(
                "Cross-tenant result access forbidden"
            )

        return json.dumps(
            json_safe(
                {
                    "result": result,
                    "simulation_only": True,
                    "robot_actuation": False,
                    "collision_checked": False,
                    "dynamics_validated": False,
                    "safety_certified": False,
                }
            ),
            indent=2,
            allow_nan=False,
        )


# ============================================================
# SYNTHETIC ROBOT FACTORY
# ============================================================

class SyntheticIKFactory:

    @staticmethod
    def planar_3dof_robot(
        robot_id: str = "planar-arm-001",
        tenant_id: str = "physical_ai_lab",
    ) -> IKRobotModel:
        joints = (
            IKJoint(
                name="shoulder",
                joint_type=JointType.REVOLUTE,
                minimum=-math.pi,
                maximum=math.pi,
                dh_a_m=0.40,
                home_position=0.0,
            ),
            IKJoint(
                name="elbow",
                joint_type=JointType.REVOLUTE,
                minimum=-math.pi,
                maximum=math.pi,
                dh_a_m=0.30,
                home_position=0.0,
            ),
            IKJoint(
                name="wrist",
                joint_type=JointType.REVOLUTE,
                minimum=-math.pi,
                maximum=math.pi,
                dh_a_m=0.20,
                home_position=0.0,
            ),
        )

        return IKRobotModel(
            robot_id=robot_id,
            tenant_id=tenant_id,
            joints=joints,
            workspace=IKWorkspace(
                minimum_xyz=(-0.95, -0.95, -0.01),
                maximum_xyz=(0.95, 0.95, 0.01),
            ),
        )

    @staticmethod
    def spatial_6dof_robot(
        robot_id: str = "spatial-arm-001",
        tenant_id: str = "physical_ai_lab",
    ) -> IKRobotModel:
        joints = (
            IKJoint(
                name="joint_1",
                joint_type=JointType.REVOLUTE,
                minimum=-math.pi,
                maximum=math.pi,
                dh_d_m=0.30,
            ),
            IKJoint(
                name="joint_2",
                joint_type=JointType.REVOLUTE,
                minimum=-2.5,
                maximum=2.5,
                dh_a_m=0.30,
                dh_alpha_rad=-math.pi / 2,
            ),
            IKJoint(
                name="joint_3",
                joint_type=JointType.REVOLUTE,
                minimum=-2.5,
                maximum=2.5,
                dh_a_m=0.25,
            ),
            IKJoint(
                name="joint_4",
                joint_type=JointType.REVOLUTE,
                minimum=-math.pi,
                maximum=math.pi,
                dh_alpha_rad=-math.pi / 2,
            ),
            IKJoint(
                name="joint_5",
                joint_type=JointType.REVOLUTE,
                minimum=-2.5,
                maximum=2.5,
                dh_alpha_rad=math.pi / 2,
            ),
            IKJoint(
                name="joint_6",
                joint_type=JointType.REVOLUTE,
                minimum=-math.pi,
                maximum=math.pi,
                dh_d_m=0.12,
            ),
        )

        return IKRobotModel(
            robot_id=robot_id,
            tenant_id=tenant_id,
            joints=joints,
        )

    @staticmethod
    def access(
        tenant_id: str = "physical_ai_lab",
    ) -> IKAccessContext:
        return IKAccessContext(
            tenant_id=tenant_id,
            user_id="robotics_engineer",
            can_read=True,
            can_solve=True,
            can_manage=True,
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestInverseKinematics(unittest.TestCase):

    def setUp(self) -> None:
        self.robot = (
            SyntheticIKFactory.planar_3dof_robot()
        )

        self.config = IKSolverConfig(
            maximum_iterations=400,
            random_restarts=6,
            random_seed=42,
        )

        self.solver = NumericalIKSolver(
            self.robot,
            self.config,
        )

        self.access = (
            SyntheticIKFactory.access()
        )

    def test_rotation_matrix(self) -> None:
        rotation = rpy_to_matrix(
            0.0,
            0.0,
            math.pi / 2,
        )

        result = rotation @ np.array(
            [1.0, 0.0, 0.0]
        )

        self.assertTrue(
            np.allclose(
                result,
                (0.0, 1.0, 0.0),
                atol=1e-8,
            )
        )

    def test_rotation_log_identity(self) -> None:
        result = rotation_log(
            np.eye(3)
        )

        self.assertTrue(
            np.allclose(
                result,
                np.zeros(3),
            )
        )

    def test_orientation_error(self) -> None:
        result = orientation_error(
            np.eye(3),
            rotation_z(
                math.pi / 2
            ),
        )

        self.assertAlmostEqual(
            float(
                np.linalg.norm(result)
            ),
            math.pi / 2,
            places=6,
        )

    def test_dh_transform_identity(self) -> None:
        result = DHKinematics.dh_transform(
            0.0,
            0.0,
            0.0,
            0.0,
        )

        self.assertTrue(
            np.allclose(
                result,
                np.eye(4),
            )
        )

    def test_forward_kinematics_home(self) -> None:
        pose = self.solver.kinematics.forward_pose(
            (0.0, 0.0, 0.0)
        )

        self.assertAlmostEqual(
            pose.position_xyz[0],
            0.9,
        )

        self.assertAlmostEqual(
            pose.position_xyz[1],
            0.0,
        )

    def test_forward_kinematics_rotation(self) -> None:
        pose = self.solver.kinematics.forward_pose(
            (math.pi / 2, 0.0, 0.0)
        )

        self.assertAlmostEqual(
            pose.position_xyz[0],
            0.0,
            places=6,
        )

        self.assertAlmostEqual(
            pose.position_xyz[1],
            0.9,
            places=6,
        )

    def test_jacobian_shape(self) -> None:
        jacobian = self.solver.kinematics.jacobian(
            (0.1, 0.2, -0.3)
        )

        self.assertEqual(
            jacobian.shape,
            (6, 3),
        )

    def test_joint_limits(self) -> None:
        self.assertTrue(
            self.solver.kinematics.within_joint_limits(
                (0.0, 0.0, 0.0)
            )
        )

        self.assertFalse(
            self.solver.kinematics.within_joint_limits(
                (10.0, 0.0, 0.0)
            )
        )

    def test_joint_limit_clamping(self) -> None:
        result = self.solver.limits.clamp(
            (10.0, 0.0, 0.0)
        )

        self.assertAlmostEqual(
            result[0],
            math.pi,
        )

    def test_joint_center_score(self) -> None:
        score = self.solver.limits.center_score(
            (0.0, 0.0, 0.0)
        )

        self.assertAlmostEqual(
            score,
            1.0,
        )

    def test_position_ik_home_target(self) -> None:
        target = self.solver.kinematics.forward_pose(
            (0.0, 0.0, 0.0)
        )

        result = self.solver.solve(
            IKRequest(
                request_id=new_id(),
                target=target,
                task_type=IKTaskType.POSITION,
                use_random_restarts=False,
            )
        )

        self.assertEqual(
            result.status,
            IKStatus.CONVERGED,
        )

    def test_position_ik_nontrivial_target(self) -> None:
        reference = (
            0.4,
            -0.7,
            0.5,
        )

        target = self.solver.kinematics.forward_pose(
            reference
        )

        result = self.solver.solve(
            IKRequest(
                request_id=new_id(),
                target=target,
                task_type=IKTaskType.POSITION,
            )
        )

        self.assertIsNotNone(
            result.best_solution
        )

        self.assertEqual(
            result.status,
            IKStatus.CONVERGED,
        )

        self.assertLessEqual(
            result.best_solution.position_error_m,
            self.config.position_tolerance_m,
        )

    def test_pose_ik_home_target(self) -> None:
        target = self.solver.kinematics.forward_pose(
            (0.0, 0.0, 0.0)
        )

        result = self.solver.solve(
            IKRequest(
                request_id=new_id(),
                target=target,
                task_type=IKTaskType.POSE,
                use_random_restarts=False,
            )
        )

        self.assertEqual(
            result.status,
            IKStatus.CONVERGED,
        )

    def test_out_of_workspace(self) -> None:
        target = IKPose.from_rpy(
            (5.0, 5.0, 0.0),
            0.0,
            0.0,
            0.0,
        )

        result = self.solver.solve(
            IKRequest(
                request_id=new_id(),
                target=target,
            )
        )

        self.assertEqual(
            result.status,
            IKStatus.OUT_OF_WORKSPACE,
        )

    def test_random_restarts(self) -> None:
        target = self.solver.kinematics.forward_pose(
            (0.2, -0.5, 0.3)
        )

        result = self.solver.solve(
            IKRequest(
                request_id=new_id(),
                target=target,
                use_random_restarts=True,
            )
        )

        self.assertEqual(
            result.attempted_starts,
            7,
        )

    def test_singularity_analysis(self) -> None:
        jacobian = np.zeros(
            (3, 3)
        )

        report = (
            self.solver.singularity_analyzer.analyze(
                jacobian
            )
        )

        self.assertEqual(
            report.level,
            IKSingularityLevel.SINGULAR,
        )

    def test_workspace_contains(self) -> None:
        self.assertTrue(
            self.robot.workspace.contains(
                (0.3, 0.2, 0.0)
            )
        )

    def test_batch_ik(self) -> None:
        target = self.solver.kinematics.forward_pose(
            (0.0, 0.0, 0.0)
        )

        requests = [
            IKRequest(
                request_id=new_id(),
                target=target,
            )
            for _ in range(3)
        ]

        results = self.solver.solve_batch(
            requests
        )

        self.assertEqual(
            len(results),
            3,
        )

    def test_service_registration(self) -> None:
        service = InverseKinematicsService()

        service.register_robot(
            self.robot,
            self.access,
            self.config,
        )

        self.assertEqual(
            service.list_robots(
                self.access
            ),
            ("planar-arm-001",),
        )

    def test_service_solve(self) -> None:
        service = InverseKinematicsService()

        service.register_robot(
            self.robot,
            self.access,
            self.config,
        )

        target = self.solver.kinematics.forward_pose(
            (0.0, 0.0, 0.0)
        )

        result = service.solve(
            self.robot.robot_id,
            IKRequest(
                request_id=new_id(),
                target=target,
            ),
            self.access,
        )

        self.assertEqual(
            result.status,
            IKStatus.CONVERGED,
        )

    def test_tenant_isolation(self) -> None:
        service = InverseKinematicsService()

        service.register_robot(
            self.robot,
            self.access,
        )

        other = IKAccessContext(
            tenant_id="other_tenant",
            user_id="other_user",
            can_read=True,
            can_solve=True,
        )

        with self.assertRaises(
            IKRobotNotFoundError
        ):
            service.get_solver(
                self.robot.robot_id,
                other,
            )

    def test_permission_denied(self) -> None:
        service = InverseKinematicsService()

        service.register_robot(
            self.robot,
            self.access,
        )

        viewer = IKAccessContext(
            tenant_id="physical_ai_lab",
            user_id="viewer",
            can_read=True,
        )

        target = self.solver.kinematics.forward_pose(
            (0.0, 0.0, 0.0)
        )

        with self.assertRaises(
            IKAuthorizationError
        ):
            service.solve(
                self.robot.robot_id,
                IKRequest(
                    request_id=new_id(),
                    target=target,
                ),
                viewer,
            )

    def test_json_export(self) -> None:
        service = InverseKinematicsService()

        service.register_robot(
            self.robot,
            self.access,
        )

        target = self.solver.kinematics.forward_pose(
            (0.0, 0.0, 0.0)
        )

        result = service.solve(
            self.robot.robot_id,
            IKRequest(
                request_id=new_id(),
                target=target,
            ),
            self.access,
        )

        exported = json.loads(
            service.export_json(
                result,
                self.access,
            )
        )

        self.assertTrue(
            exported["simulation_only"]
        )

        self.assertFalse(
            exported["robot_actuation"]
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("\nPHYSICAL AI — INVERSE KINEMATICS")
    print("=" * 72)

    robot = SyntheticIKFactory.planar_3dof_robot(
        robot_id="physical-ai-arm-001"
    )

    access = SyntheticIKFactory.access()

    service = InverseKinematicsService()

    service.register_robot(
        robot,
        access,
        IKSolverConfig(
            maximum_iterations=400,
            random_restarts=8,
        ),
    )

    reference_configuration = (
        0.5,
        -0.8,
        0.4,
    )

    target_pose = service.forward_kinematics(
        robot.robot_id,
        reference_configuration,
        access,
    )

    print(
        "Robot:",
        robot.robot_id,
    )

    print(
        "Degrees of freedom:",
        robot.degrees_of_freedom,
    )

    print(
        "Target position:",
        target_pose.position_xyz,
    )

    request = IKRequest(
        request_id=new_id(),
        target=target_pose,
        task_type=IKTaskType.POSITION,
        initial_configuration=(
            0.0,
            0.0,
            0.0,
        ),
        use_random_restarts=True,
    )

    result = service.solve(
        robot.robot_id,
        request,
        access,
    )

    print(
        "Solver status:",
        result.status.value,
    )

    print(
        "Attempted starts:",
        result.attempted_starts,
    )

    if result.best_solution is not None:
        best = result.best_solution

        print(
            "Joint solution:",
            tuple(
                round(value, 6)
                for value in best.joint_positions
            ),
        )

        print(
            "Position error:",
            best.position_error_m,
            "m",
        )

        print(
            "Iterations:",
            best.iterations,
        )

        print(
            "Singularity:",
            best.singularity.level.value,
        )

    print("\nIK RESULT JSON")
    print("=" * 72)

    print(
        service.export_json(
            result,
            access,
        )
    )

    print("\nRUNNING UNIT TESTS")
    print("=" * 72)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestInverseKinematics
    )

    test_result = unittest.TextTestRunner(
        verbosity=2
    ).run(suite)

    if not test_result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
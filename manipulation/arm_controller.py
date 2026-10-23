"""Arm Controller module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/arm_controller.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================
Robotics Control — Arm Controller

Features:
    - Multi-joint robotic arm configuration
    - Revolute and prismatic joints
    - Joint position, velocity and acceleration limits
    - Forward kinematics using Denavit-Hartenberg parameters
    - Numerical inverse kinematics
    - Joint-space trajectory generation
    - Quintic polynomial motion profiles
    - Cartesian position planning
    - Joint limit validation
    - Workspace boundary validation
    - Singularity detection
    - Jacobian calculation
    - End-effector pose estimation
    - Robot state monitoring
    - Joint health monitoring
    - Advisory collision checking
    - Emergency-stop state management
    - Simulation-only execution
    - Multi-tenant robot isolation
    - Structured control reports
    - Synthetic robotic arm demonstration
    - Unit tests

Requirements:
    Python >= 3.11
    numpy >= 1.24

Install:
    pip install numpy

IMPORTANT SAFETY NOTICE:
    This controller is intended for simulation, education,
    offline trajectory planning and robotics research.

    It does not communicate with real robot hardware.

    Collision checking uses simplified spherical obstacles
    and joint-point approximations. It does not guarantee
    physical collision avoidance.

    Numerical inverse kinematics does not guarantee a
    collision-free or dynamically feasible trajectory.

    Real-world deployment requires certified safety
    systems, hardware-specific control integration,
    independent collision monitoring, emergency-stop
    hardware, and authorized engineering validation.
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

class ArmControllerError(Exception):
    pass


class ArmValidationError(ArmControllerError):
    pass


class JointLimitError(ArmControllerError):
    pass


class WorkspaceLimitError(ArmControllerError):
    pass


class InverseKinematicsError(ArmControllerError):
    pass


class TrajectoryPlanningError(ArmControllerError):
    pass


class CollisionRiskError(ArmControllerError):
    pass


class EmergencyStopError(ArmControllerError):
    pass


class ArmAuthorizationError(ArmControllerError):
    pass


class ArmNotFoundError(ArmControllerError):
    pass


class ArmStateError(ArmControllerError):
    pass


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


def finite_number(
    value: Any,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise ArmValidationError(
            f"{name} must be numeric"
        )

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ArmValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(number):
        raise ArmValidationError(
            f"{name} must be finite"
        )

    return number


def validate_identifier(
    value: str,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 256
    ):
        raise ArmValidationError(
            f"Invalid {name}"
        )

    return value.strip()


def validate_vector(
    values: Sequence[float],
    expected_size: int,
    name: str,
) -> np.ndarray:
    try:
        vector = np.asarray(
            values,
            dtype=float,
        )
    except (TypeError, ValueError) as exc:
        raise ArmValidationError(
            f"{name} must contain numeric values"
        ) from exc

    if vector.shape != (expected_size,):
        raise ArmValidationError(
            f"{name} must contain {expected_size} values"
        )

    if not np.all(np.isfinite(vector)):
        raise ArmValidationError(
            f"{name} contains non-finite values"
        )

    return vector


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

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArmValidationError(
                "Cannot serialize non-finite value"
            )

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    raise ArmValidationError(
        f"Unsupported serialization type: {type(value).__name__}"
    )


# ============================================================
# ENUMERATIONS
# ============================================================

class JointType(str, Enum):
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"


class ArmMode(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    SIMULATING = "simulating"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"
    FAULT = "fault"


class ArmHealth(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class TrajectoryStatus(str, Enum):
    PLANNED = "planned"
    VALIDATED = "validated"
    REJECTED = "rejected"
    SIMULATED = "simulated"


class CollisionStatus(str, Enum):
    CLEAR = "clear"
    NEAR_OBSTACLE = "near_obstacle"
    COLLISION_RISK = "collision_risk"


class IKStatus(str, Enum):
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    SINGULAR = "singular"
    OUT_OF_WORKSPACE = "out_of_workspace"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class JointConfig:
    name: str
    joint_type: JointType

    minimum_position: float
    maximum_position: float

    maximum_velocity: float
    maximum_acceleration: float

    dh_a: float = 0.0
    dh_alpha: float = 0.0
    dh_d: float = 0.0
    dh_theta: float = 0.0

    home_position: float = 0.0

    def __post_init__(self) -> None:
        validate_identifier(self.name, "joint name")

        numeric_fields = (
            "minimum_position",
            "maximum_position",
            "maximum_velocity",
            "maximum_acceleration",
            "dh_a",
            "dh_alpha",
            "dh_d",
            "dh_theta",
            "home_position",
        )

        for name in numeric_fields:
            finite_number(
                getattr(self, name),
                name,
            )

        if self.minimum_position >= self.maximum_position:
            raise ArmValidationError(
                f"Invalid position limits for {self.name}"
            )

        if self.maximum_velocity <= 0:
            raise ArmValidationError(
                "Maximum velocity must be positive"
            )

        if self.maximum_acceleration <= 0:
            raise ArmValidationError(
                "Maximum acceleration must be positive"
            )

        if not (
            self.minimum_position
            <= self.home_position
            <= self.maximum_position
        ):
            raise ArmValidationError(
                f"Home position outside limits: {self.name}"
            )


@dataclass(frozen=True)
class WorkspaceBounds:
    minimum_x: float = -2.0
    maximum_x: float = 2.0

    minimum_y: float = -2.0
    maximum_y: float = 2.0

    minimum_z: float = -0.5
    maximum_z: float = 2.5

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
                raise ArmValidationError(
                    f"Invalid workspace {axis} bounds"
                )

    def contains(
        self,
        position: Sequence[float],
    ) -> bool:
        x, y, z = validate_vector(
            position,
            3,
            "workspace position",
        )

        return (
            self.minimum_x <= x <= self.maximum_x
            and self.minimum_y <= y <= self.maximum_y
            and self.minimum_z <= z <= self.maximum_z
        )


@dataclass(frozen=True)
class ArmControllerConfig:
    robot_id: str
    tenant_id: str

    joints: tuple[JointConfig, ...]

    workspace: WorkspaceBounds = field(
        default_factory=WorkspaceBounds
    )

    control_frequency_hz: float = 100.0

    trajectory_sample_period: float = 0.02

    minimum_obstacle_clearance: float = 0.10

    ik_position_tolerance: float = 1e-4
    ik_max_iterations: int = 250
    ik_damping: float = 0.05
    ik_step_size: float = 0.5

    singularity_threshold: float = 1e-5

    def __post_init__(self) -> None:
        validate_identifier(self.robot_id, "robot_id")
        validate_identifier(self.tenant_id, "tenant_id")

        if not self.joints:
            raise ArmValidationError(
                "Robotic arm must have at least one joint"
            )

        if len(self.joints) > 32:
            raise ArmValidationError(
                "Maximum supported joint count is 32"
            )

        names = [joint.name for joint in self.joints]

        if len(names) != len(set(names)):
            raise ArmValidationError(
                "Duplicate joint names"
            )

        for name in (
            "control_frequency_hz",
            "trajectory_sample_period",
            "ik_position_tolerance",
            "ik_damping",
            "ik_step_size",
            "singularity_threshold",
        ):
            if finite_number(
                getattr(self, name),
                name,
            ) <= 0:
                raise ArmValidationError(
                    f"{name} must be positive"
                )

        if self.minimum_obstacle_clearance < 0:
            raise ArmValidationError(
                "Obstacle clearance cannot be negative"
            )

        if self.ik_max_iterations <= 0:
            raise ArmValidationError(
                "IK iteration limit must be positive"
            )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class ArmAccessContext:
    tenant_id: str
    user_id: str

    can_read: bool = False
    can_plan: bool = False
    can_simulate: bool = False
    can_manage: bool = False


@dataclass(frozen=True)
class EndEffectorPose:
    position: tuple[float, float, float]

    rotation_matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]

    transform_matrix: tuple[
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
    ]


@dataclass(frozen=True)
class JointState:
    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    accelerations: tuple[float, ...]

    timestamp: datetime = field(
        default_factory=utc_now
    )


@dataclass(frozen=True)
class ArmState:
    robot_id: str
    tenant_id: str

    mode: ArmMode
    health: ArmHealth

    joint_state: JointState
    end_effector_pose: EndEffectorPose

    emergency_stop_active: bool

    timestamp: datetime


@dataclass(frozen=True)
class TrajectoryPoint:
    time_from_start: float

    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    accelerations: tuple[float, ...]


@dataclass(frozen=True)
class ArmTrajectory:
    trajectory_id: str
    robot_id: str
    tenant_id: str

    points: tuple[TrajectoryPoint, ...]

    duration_seconds: float

    status: TrajectoryStatus

    created_at: datetime


@dataclass(frozen=True)
class SphericalObstacle:
    obstacle_id: str

    center: tuple[float, float, float]

    radius: float

    def __post_init__(self) -> None:
        validate_identifier(
            self.obstacle_id,
            "obstacle_id",
        )

        validate_vector(
            self.center,
            3,
            "obstacle center",
        )

        if finite_number(self.radius, "radius") <= 0:
            raise ArmValidationError(
                "Obstacle radius must be positive"
            )


@dataclass(frozen=True)
class CollisionReport:
    status: CollisionStatus

    minimum_clearance: float

    nearest_obstacle_id: str | None

    checked_points: int


@dataclass(frozen=True)
class IKResult:
    status: IKStatus

    joint_positions: tuple[float, ...]

    position_error: float

    iterations: int

    converged: bool


@dataclass(frozen=True)
class JointHealthReport:
    joint_name: str

    health: ArmHealth

    position_limit_margin: float

    velocity_utilization: float

    acceleration_utilization: float


@dataclass(frozen=True)
class ArmHealthReport:
    robot_id: str

    overall_health: ArmHealth

    joint_reports: tuple[JointHealthReport, ...]

    emergency_stop_active: bool

    evaluated_at: datetime


@dataclass(frozen=True)
class SimulationResult:
    simulation_id: str

    trajectory_id: str

    robot_id: str

    status: TrajectoryStatus

    final_joint_state: JointState

    final_end_effector_pose: EndEffectorPose

    sampled_states: int

    completed_at: datetime


# ============================================================
# DENAVIT-HARTENBERG KINEMATICS
# ============================================================

class ArmKinematics:

    def __init__(
        self,
        joints: Sequence[JointConfig],
    ) -> None:
        self.joints = tuple(joints)
        self.dof = len(self.joints)

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

    def forward_transforms(
        self,
        positions: Sequence[float],
    ) -> tuple[np.ndarray, ...]:
        q = validate_vector(
            positions,
            self.dof,
            "joint positions",
        )

        transform = np.eye(
            4,
            dtype=float,
        )

        transforms = [
            transform.copy()
        ]

        for index, joint in enumerate(self.joints):
            if joint.joint_type == JointType.REVOLUTE:
                theta = joint.dh_theta + q[index]
                d = joint.dh_d

            else:
                theta = joint.dh_theta
                d = joint.dh_d + q[index]

            transform = (
                transform
                @ self.dh_transform(
                    joint.dh_a,
                    joint.dh_alpha,
                    d,
                    theta,
                )
            )

            transforms.append(
                transform.copy()
            )

        return tuple(transforms)

    def forward_kinematics(
        self,
        positions: Sequence[float],
    ) -> EndEffectorPose:
        transform = self.forward_transforms(
            positions
        )[-1]

        rotation = transform[:3, :3]

        return EndEffectorPose(
            position=tuple(
                float(value)
                for value in transform[:3, 3]
            ),
            rotation_matrix=tuple(
                tuple(float(value) for value in row)
                for row in rotation
            ),
            transform_matrix=tuple(
                tuple(float(value) for value in row)
                for row in transform
            ),
        )

    def joint_positions_cartesian(
        self,
        positions: Sequence[float],
    ) -> tuple[tuple[float, float, float], ...]:
        transforms = self.forward_transforms(
            positions
        )

        return tuple(
            tuple(
                float(value)
                for value in transform[:3, 3]
            )
            for transform in transforms
        )

    def jacobian(
        self,
        positions: Sequence[float],
    ) -> np.ndarray:
        transforms = self.forward_transforms(
            positions
        )

        end_position = transforms[-1][:3, 3]

        jacobian = np.zeros(
            (6, self.dof),
            dtype=float,
        )

        for index, joint in enumerate(self.joints):
            frame = transforms[index]

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

    def position_jacobian(
        self,
        positions: Sequence[float],
    ) -> np.ndarray:
        return self.jacobian(
            positions
        )[:3, :]

    def singularity_measure(
        self,
        positions: Sequence[float],
    ) -> float:
        jacobian = self.position_jacobian(
            positions
        )

        singular_values = np.linalg.svd(
            jacobian,
            compute_uv=False,
        )

        if len(singular_values) == 0:
            return 0.0

        if np.linalg.matrix_rank(
            jacobian,
            tol=1e-10,
        ) < min(3, self.dof):
            return 0.0

        return float(
            min(singular_values)
        )


# ============================================================
# JOINT LIMIT VALIDATOR
# ============================================================

class JointLimitValidator:

    def __init__(
        self,
        joints: Sequence[JointConfig],
        *,
        tolerance: float = 1e-8,
    ) -> None:
        self.joints = tuple(joints)
        self.tolerance = tolerance

    def validate_positions(
        self,
        positions: Sequence[float],
    ) -> None:
        q = validate_vector(
            positions,
            len(self.joints),
            "joint positions",
        )

        for index, joint in enumerate(self.joints):
            if (
                q[index] < joint.minimum_position - self.tolerance
                or q[index] > joint.maximum_position + self.tolerance
            ):
                raise JointLimitError(
                    f"Joint {joint.name} position outside limits"
                )

    def validate_velocities(
        self,
        velocities: Sequence[float],
    ) -> None:
        values = validate_vector(
            velocities,
            len(self.joints),
            "joint velocities",
        )

        for index, joint in enumerate(self.joints):
            if (
                abs(values[index])
                > joint.maximum_velocity + self.tolerance
            ):
                raise JointLimitError(
                    f"Joint {joint.name} velocity exceeds limit"
                )

    def validate_accelerations(
        self,
        accelerations: Sequence[float],
    ) -> None:
        values = validate_vector(
            accelerations,
            len(self.joints),
            "joint accelerations",
        )

        for index, joint in enumerate(self.joints):
            if (
                abs(values[index])
                > joint.maximum_acceleration + self.tolerance
            ):
                raise JointLimitError(
                    f"Joint {joint.name} acceleration exceeds limit"
                )

    def validate_state(
        self,
        state: JointState,
    ) -> None:
        self.validate_positions(
            state.positions
        )

        self.validate_velocities(
            state.velocities
        )

        self.validate_accelerations(
            state.accelerations
        )


# ============================================================
# INVERSE KINEMATICS
# ============================================================

class ArmInverseKinematics:

    def __init__(
        self,
        kinematics: ArmKinematics,
        joints: Sequence[JointConfig],
        config: ArmControllerConfig,
    ) -> None:
        self.kinematics = kinematics
        self.joints = tuple(joints)
        self.config = config

    def solve_position(
        self,
        target_position: Sequence[float],
        initial_positions: Sequence[float],
    ) -> IKResult:
        target = validate_vector(
            target_position,
            3,
            "target position",
        )

        if not self.config.workspace.contains(
            target
        ):
            return IKResult(
                status=IKStatus.OUT_OF_WORKSPACE,
                joint_positions=tuple(
                    float(value)
                    for value in initial_positions
                ),
                position_error=float("inf"),
                iterations=0,
                converged=False,
            )

        q = validate_vector(
            initial_positions,
            len(self.joints),
            "initial joint positions",
        ).copy()

        lower = np.asarray(
            [
                joint.minimum_position
                for joint in self.joints
            ],
            dtype=float,
        )

        upper = np.asarray(
            [
                joint.maximum_position
                for joint in self.joints
            ],
            dtype=float,
        )

        q = np.clip(
            q,
            lower,
            upper,
        )

        for iteration in range(
            self.config.ik_max_iterations
        ):
            pose = self.kinematics.forward_kinematics(
                q
            )

            current = np.asarray(
                pose.position,
                dtype=float,
            )

            error = target - current

            error_norm = float(
                np.linalg.norm(error)
            )

            if error_norm <= self.config.ik_position_tolerance:
                return IKResult(
                    status=IKStatus.CONVERGED,
                    joint_positions=tuple(
                        float(value)
                        for value in q
                    ),
                    position_error=error_norm,
                    iterations=iteration,
                    converged=True,
                )

            jacobian = self.kinematics.position_jacobian(
                q
            )

            damping = self.config.ik_damping

            matrix = (
                jacobian @ jacobian.T
                + damping**2 * np.eye(3)
            )

            try:
                delta = (
                    jacobian.T
                    @ np.linalg.solve(
                        matrix,
                        error,
                    )
                )

            except np.linalg.LinAlgError:
                return IKResult(
                    status=IKStatus.SINGULAR,
                    joint_positions=tuple(
                        float(value)
                        for value in q
                    ),
                    position_error=error_norm,
                    iterations=iteration,
                    converged=False,
                )

            q = np.clip(
                q + self.config.ik_step_size * delta,
                lower,
                upper,
            )

        final_position = np.asarray(
            self.kinematics.forward_kinematics(
                q
            ).position
        )

        final_error = float(
            np.linalg.norm(
                target - final_position
            )
        )

        return IKResult(
            status=IKStatus.MAX_ITERATIONS,
            joint_positions=tuple(
                float(value)
                for value in q
            ),
            position_error=final_error,
            iterations=self.config.ik_max_iterations,
            converged=False,
        )


# ============================================================
# QUINTIC TRAJECTORY PLANNER
# ============================================================

class ArmTrajectoryPlanner:

    def __init__(
        self,
        config: ArmControllerConfig,
    ) -> None:
        self.config = config

        self.validator = JointLimitValidator(
            config.joints
        )

    @staticmethod
    def quintic_profile(
        normalized_time: float,
    ) -> tuple[float, float, float]:
        s = min(
            1.0,
            max(0.0, normalized_time),
        )

        position = (
            10 * s**3
            - 15 * s**4
            + 6 * s**5
        )

        velocity = (
            30 * s**2
            - 60 * s**3
            + 30 * s**4
        )

        acceleration = (
            60 * s
            - 180 * s**2
            + 120 * s**3
        )

        return (
            position,
            velocity,
            acceleration,
        )

    def minimum_duration(
        self,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> float:
        displacement = np.abs(
            goal - start
        )

        duration = self.config.trajectory_sample_period

        # Quintic normalized velocity maximum = 1.875.
        # Quintic normalized acceleration maximum ≈ 5.7735.
        maximum_velocity_factor = 1.875
        maximum_acceleration_factor = 5.773503

        for index, joint in enumerate(
            self.config.joints
        ):
            distance = displacement[index]

            velocity_duration = (
                maximum_velocity_factor
                * distance
                / joint.maximum_velocity
            )

            acceleration_duration = math.sqrt(
                maximum_acceleration_factor
                * distance
                / joint.maximum_acceleration
            )

            duration = max(
                duration,
                velocity_duration,
                acceleration_duration,
            )

        # Conservative margin for numerical sampling.
        return duration * 1.05

    def plan_joint_trajectory(
        self,
        start_positions: Sequence[float],
        goal_positions: Sequence[float],
        *,
        duration_seconds: float | None = None,
    ) -> ArmTrajectory:
        dof = len(
            self.config.joints
        )

        start = validate_vector(
            start_positions,
            dof,
            "start positions",
        )

        goal = validate_vector(
            goal_positions,
            dof,
            "goal positions",
        )

        self.validator.validate_positions(
            start
        )

        self.validator.validate_positions(
            goal
        )

        minimum_duration = self.minimum_duration(
            start,
            goal,
        )

        if duration_seconds is None:
            duration = minimum_duration

        else:
            duration = finite_number(
                duration_seconds,
                "duration_seconds",
            )

            if duration < minimum_duration:
                raise TrajectoryPlanningError(
                    "Requested duration violates conservative "
                    "joint velocity/acceleration constraints"
                )

        sample_period = (
            self.config.trajectory_sample_period
        )

        sample_count = max(
            2,
            math.ceil(
                duration / sample_period
            ) + 1,
        )

        points = []

        displacement = goal - start

        for index in range(sample_count):
            t = (
                duration * index
                / (sample_count - 1)
            )

            normalized = (
                t / duration
            )

            (
                profile_position,
                profile_velocity,
                profile_acceleration,
            ) = self.quintic_profile(
                normalized
            )

            positions = (
                start
                + displacement * profile_position
            )

            velocities = (
                displacement
                * profile_velocity
                / duration
            )

            accelerations = (
                displacement
                * profile_acceleration
                / duration**2
            )

            point = TrajectoryPoint(
                time_from_start=float(t),
                positions=tuple(
                    float(value)
                    for value in positions
                ),
                velocities=tuple(
                    float(value)
                    for value in velocities
                ),
                accelerations=tuple(
                    float(value)
                    for value in accelerations
                ),
            )

            self.validator.validate_positions(
                point.positions
            )

            self.validator.validate_velocities(
                point.velocities
            )

            self.validator.validate_accelerations(
                point.accelerations
            )

            points.append(
                point
            )

        return ArmTrajectory(
            trajectory_id=generate_id(),
            robot_id=self.config.robot_id,
            tenant_id=self.config.tenant_id,
            points=tuple(points),
            duration_seconds=float(duration),
            status=TrajectoryStatus.PLANNED,
            created_at=utc_now(),
        )


# ============================================================
# COLLISION RISK CHECKER
# ============================================================

class ArmCollisionChecker:

    def __init__(
        self,
        kinematics: ArmKinematics,
        minimum_clearance: float,
    ) -> None:
        self.kinematics = kinematics
        self.minimum_clearance = minimum_clearance

    @staticmethod
    def point_to_segment_distance(
        point: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
    ) -> float:
        segment = end - start

        denominator = float(
            np.dot(segment, segment)
        )

        if denominator <= 1e-12:
            return float(
                np.linalg.norm(
                    point - start
                )
            )

        projection = float(
            np.dot(
                point - start,
                segment,
            ) / denominator
        )

        projection = min(
            1.0,
            max(0.0, projection),
        )

        closest = (
            start + projection * segment
        )

        return float(
            np.linalg.norm(
                point - closest
            )
        )

    def check(
        self,
        positions: Sequence[float],
        obstacles: Sequence[SphericalObstacle],
    ) -> CollisionReport:
        if not obstacles:
            return CollisionReport(
                status=CollisionStatus.CLEAR,
                minimum_clearance=1e12,
                nearest_obstacle_id=None,
                checked_points=0,
            )

        joint_points = [
            np.asarray(point, dtype=float)
            for point in self.kinematics.joint_positions_cartesian(
                positions
            )
        ]

        minimum_clearance = float("inf")
        nearest_obstacle = None

        checks = 0

        for obstacle in obstacles:
            center = validate_vector(
                obstacle.center,
                3,
                "obstacle center",
            )

            for index in range(
                len(joint_points) - 1
            ):
                distance = self.point_to_segment_distance(
                    center,
                    joint_points[index],
                    joint_points[index + 1],
                )

                clearance = (
                    distance - obstacle.radius
                )

                checks += 1

                if clearance < minimum_clearance:
                    minimum_clearance = clearance
                    nearest_obstacle = obstacle.obstacle_id

        if minimum_clearance <= 0:
            status = CollisionStatus.COLLISION_RISK

        elif minimum_clearance < self.minimum_clearance:
            status = CollisionStatus.NEAR_OBSTACLE

        else:
            status = CollisionStatus.CLEAR

        return CollisionReport(
            status=status,
            minimum_clearance=minimum_clearance,
            nearest_obstacle_id=nearest_obstacle,
            checked_points=checks,
        )

    def validate_trajectory(
        self,
        trajectory: ArmTrajectory,
        obstacles: Sequence[SphericalObstacle],
    ) -> CollisionReport:
        worst_report = CollisionReport(
            status=CollisionStatus.CLEAR,
            minimum_clearance=1e12,
            nearest_obstacle_id=None,
            checked_points=0,
        )

        total_checks = 0

        for point in trajectory.points:
            report = self.check(
                point.positions,
                obstacles,
            )

            total_checks += report.checked_points

            if (
                report.minimum_clearance
                < worst_report.minimum_clearance
            ):
                worst_report = report

        return CollisionReport(
            status=worst_report.status,
            minimum_clearance=worst_report.minimum_clearance,
            nearest_obstacle_id=worst_report.nearest_obstacle_id,
            checked_points=total_checks,
        )


# ============================================================
# ARM HEALTH MONITOR
# ============================================================

class ArmHealthMonitor:

    def __init__(
        self,
        joints: Sequence[JointConfig],
    ) -> None:
        self.joints = tuple(joints)

    def evaluate(
        self,
        state: JointState,
        *,
        emergency_stop_active: bool = False,
        robot_id: str = "",
    ) -> ArmHealthReport:
        reports = []

        overall = ArmHealth.HEALTHY

        for index, joint in enumerate(self.joints):
            position = state.positions[index]
            velocity = state.velocities[index]
            acceleration = state.accelerations[index]

            lower_margin = (
                position - joint.minimum_position
            )

            upper_margin = (
                joint.maximum_position - position
            )

            margin = min(
                lower_margin,
                upper_margin,
            )

            velocity_utilization = (
                abs(velocity)
                / joint.maximum_velocity
            )

            acceleration_utilization = (
                abs(acceleration)
                / joint.maximum_acceleration
            )

            if (
                margin < 0
                or velocity_utilization > 1.0
                or acceleration_utilization > 1.0
            ):
                health = ArmHealth.CRITICAL

            elif (
                margin < 0.05 * (
                    joint.maximum_position
                    - joint.minimum_position
                )
                or velocity_utilization > 0.90
                or acceleration_utilization > 0.90
            ):
                health = ArmHealth.WARNING

            else:
                health = ArmHealth.HEALTHY

            if health == ArmHealth.CRITICAL:
                overall = ArmHealth.CRITICAL

            elif (
                health == ArmHealth.WARNING
                and overall == ArmHealth.HEALTHY
            ):
                overall = ArmHealth.WARNING

            reports.append(
                JointHealthReport(
                    joint_name=joint.name,
                    health=health,
                    position_limit_margin=float(margin),
                    velocity_utilization=float(
                        velocity_utilization
                    ),
                    acceleration_utilization=float(
                        acceleration_utilization
                    ),
                )
            )

        if emergency_stop_active:
            overall = ArmHealth.CRITICAL

        return ArmHealthReport(
            robot_id=robot_id,
            overall_health=overall,
            joint_reports=tuple(reports),
            emergency_stop_active=emergency_stop_active,
            evaluated_at=utc_now(),
        )


# ============================================================
# ROBOT ARM CONTROLLER
# ============================================================

class ArmController:

    def __init__(
        self,
        config: ArmControllerConfig,
    ) -> None:
        self.config = config

        self.kinematics = ArmKinematics(
            config.joints
        )

        self.joint_validator = JointLimitValidator(
            config.joints
        )

        self.inverse_kinematics = ArmInverseKinematics(
            self.kinematics,
            config.joints,
            config,
        )

        self.trajectory_planner = ArmTrajectoryPlanner(
            config
        )

        self.collision_checker = ArmCollisionChecker(
            self.kinematics,
            config.minimum_obstacle_clearance,
        )

        self.health_monitor = ArmHealthMonitor(
            config.joints
        )

        home_positions = tuple(
            joint.home_position
            for joint in config.joints
        )

        zeros = tuple(
            0.0
            for _ in config.joints
        )

        self._joint_state = JointState(
            positions=home_positions,
            velocities=zeros,
            accelerations=zeros,
        )

        self._mode = ArmMode.IDLE
        self._emergency_stop = False

        self._obstacles: dict[
            str,
            SphericalObstacle,
        ] = {}

        self._trajectories: dict[
            str,
            ArmTrajectory,
        ] = {}

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # ACCESS CONTROL
    # --------------------------------------------------------

    def _authorize(
        self,
        access: ArmAccessContext,
        permission: str,
    ) -> None:
        if access.tenant_id != self.config.tenant_id:
            raise ArmAuthorizationError(
                "Cross-tenant access forbidden"
            )

        if not getattr(
            access,
            permission,
            False,
        ):
            raise ArmAuthorizationError(
                f"Missing permission: {permission}"
            )

    def _require_operational(
        self,
    ) -> None:
        if self._emergency_stop:
            raise EmergencyStopError(
                "Emergency stop is active"
            )

        if self._mode == ArmMode.FAULT:
            raise ArmStateError(
                "Robot arm is in fault state"
            )

    # --------------------------------------------------------
    # GET STATE
    # --------------------------------------------------------

    def get_state(
        self,
        access: ArmAccessContext,
    ) -> ArmState:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            joint_state = self._joint_state

            pose = self.kinematics.forward_kinematics(
                joint_state.positions
            )

            health_report = self.health_monitor.evaluate(
                joint_state,
                emergency_stop_active=self._emergency_stop,
                robot_id=self.config.robot_id,
            )

            return ArmState(
                robot_id=self.config.robot_id,
                tenant_id=self.config.tenant_id,
                mode=self._mode,
                health=health_report.overall_health,
                joint_state=joint_state,
                end_effector_pose=pose,
                emergency_stop_active=self._emergency_stop,
                timestamp=utc_now(),
            )

    # --------------------------------------------------------
    # SET SIMULATED JOINT STATE
    # --------------------------------------------------------

    def set_simulated_joint_state(
        self,
        positions: Sequence[float],
        access: ArmAccessContext,
        *,
        velocities: Sequence[float] | None = None,
        accelerations: Sequence[float] | None = None,
    ) -> JointState:
        self._authorize(
            access,
            "can_simulate",
        )

        with self._lock:
            self._require_operational()

            dof = len(
                self.config.joints
            )

            position_vector = validate_vector(
                positions,
                dof,
                "positions",
            )

            velocity_vector = (
                validate_vector(
                    velocities,
                    dof,
                    "velocities",
                )
                if velocities is not None
                else np.zeros(dof)
            )

            acceleration_vector = (
                validate_vector(
                    accelerations,
                    dof,
                    "accelerations",
                )
                if accelerations is not None
                else np.zeros(dof)
            )

            state = JointState(
                positions=tuple(
                    float(value)
                    for value in position_vector
                ),
                velocities=tuple(
                    float(value)
                    for value in velocity_vector
                ),
                accelerations=tuple(
                    float(value)
                    for value in acceleration_vector
                ),
            )

            self.joint_validator.validate_state(
                state
            )

            pose = self.kinematics.forward_kinematics(
                state.positions
            )

            if not self.config.workspace.contains(
                pose.position
            ):
                raise WorkspaceLimitError(
                    "End effector outside configured workspace"
                )

            report = self.collision_checker.check(
                state.positions,
                tuple(self._obstacles.values()),
            )

            if report.status != CollisionStatus.CLEAR:
                raise CollisionRiskError(
                    "Simulated joint state violates "
                    "obstacle clearance policy"
                )

            self._joint_state = state

            return state

    # --------------------------------------------------------
    # FORWARD KINEMATICS
    # --------------------------------------------------------

    def forward_kinematics(
        self,
        positions: Sequence[float],
        access: ArmAccessContext,
    ) -> EndEffectorPose:
        self._authorize(
            access,
            "can_read",
        )

        self.joint_validator.validate_positions(
            positions
        )

        return self.kinematics.forward_kinematics(
            positions
        )

    # --------------------------------------------------------
    # INVERSE KINEMATICS
    # --------------------------------------------------------

    def solve_inverse_kinematics(
        self,
        target_position: Sequence[float],
        access: ArmAccessContext,
        *,
        initial_positions: Sequence[float] | None = None,
    ) -> IKResult:
        self._authorize(
            access,
            "can_plan",
        )

        with self._lock:
            self._require_operational()

            initial = (
                initial_positions
                if initial_positions is not None
                else self._joint_state.positions
            )

            return self.inverse_kinematics.solve_position(
                target_position,
                initial,
            )

    # --------------------------------------------------------
    # JOINT TRAJECTORY
    # --------------------------------------------------------

    def plan_joint_motion(
        self,
        target_positions: Sequence[float],
        access: ArmAccessContext,
        *,
        duration_seconds: float | None = None,
    ) -> ArmTrajectory:
        self._authorize(
            access,
            "can_plan",
        )

        with self._lock:
            self._require_operational()

            self._mode = ArmMode.PLANNING

            try:
                trajectory = (
                    self.trajectory_planner.plan_joint_trajectory(
                        self._joint_state.positions,
                        target_positions,
                        duration_seconds=duration_seconds,
                    )
                )

                obstacles = tuple(
                    self._obstacles.values()
                )

                # Offline sampled collision checking.
                # This is not continuous collision detection.
                collision = (
                    self.collision_checker.validate_trajectory(
                        trajectory,
                        obstacles,
                    )
                )

                if collision.status != CollisionStatus.CLEAR:
                    raise CollisionRiskError(
                        "Planned trajectory violates "
                        "obstacle clearance policy"
                    )

                for point in trajectory.points:
                    pose = (
                        self.kinematics.forward_kinematics(
                            point.positions
                        )
                    )

                    if not self.config.workspace.contains(
                        pose.position
                    ):
                        raise WorkspaceLimitError(
                            "Trajectory leaves configured workspace"
                        )

                validated = ArmTrajectory(
                    trajectory_id=trajectory.trajectory_id,
                    robot_id=trajectory.robot_id,
                    tenant_id=trajectory.tenant_id,
                    points=trajectory.points,
                    duration_seconds=trajectory.duration_seconds,
                    status=TrajectoryStatus.VALIDATED,
                    created_at=trajectory.created_at,
                )

                self._trajectories[
                    validated.trajectory_id
                ] = validated

                return validated

            finally:
                self._mode = ArmMode.IDLE

    # --------------------------------------------------------
    # CARTESIAN POSITION TRAJECTORY
    # --------------------------------------------------------

    def plan_cartesian_position(
        self,
        target_position: Sequence[float],
        access: ArmAccessContext,
        *,
        duration_seconds: float | None = None,
    ) -> tuple[IKResult, ArmTrajectory]:
        self._authorize(
            access,
            "can_plan",
        )

        result = self.solve_inverse_kinematics(
            target_position,
            access,
        )

        if not result.converged:
            raise InverseKinematicsError(
                f"Inverse kinematics failed: {result.status.value}"
            )

        trajectory = self.plan_joint_motion(
            result.joint_positions,
            access,
            duration_seconds=duration_seconds,
        )

        return (
            result,
            trajectory,
        )

    # --------------------------------------------------------
    # SIMULATE TRAJECTORY
    # --------------------------------------------------------

    def simulate_trajectory(
        self,
        trajectory_id: str,
        access: ArmAccessContext,
    ) -> SimulationResult:
        self._authorize(
            access,
            "can_simulate",
        )

        with self._lock:
            self._require_operational()

            trajectory = self._trajectories.get(
                trajectory_id
            )

            if trajectory is None:
                raise ArmNotFoundError(
                    "Trajectory not found"
                )

            if trajectory.status != TrajectoryStatus.VALIDATED:
                raise ArmStateError(
                    "Trajectory has not been validated"
                )

            if not trajectory.points:
                raise TrajectoryPlanningError(
                    "Trajectory contains no points"
                )

            current_positions = np.asarray(
                self._joint_state.positions
            )

            planned_start = np.asarray(
                trajectory.points[0].positions
            )

            if not np.allclose(
                current_positions,
                planned_start,
                atol=1e-6,
            ):
                raise ArmStateError(
                    "Trajectory start does not match current state"
                )

            self._mode = ArmMode.SIMULATING

            sampled = 0

            try:
                for point in trajectory.points:
                    self._require_operational()

                    state = JointState(
                        positions=point.positions,
                        velocities=point.velocities,
                        accelerations=point.accelerations,
                    )

                    self.joint_validator.validate_state(
                        state
                    )

                    pose = self.kinematics.forward_kinematics(
                        state.positions
                    )

                    if not self.config.workspace.contains(
                        pose.position
                    ):
                        raise WorkspaceLimitError(
                            "Simulation workspace violation"
                        )

                    collision = self.collision_checker.check(
                        state.positions,
                        tuple(self._obstacles.values()),
                    )

                    if collision.status != CollisionStatus.CLEAR:
                        raise CollisionRiskError(
                            "Simulation obstacle clearance violation"
                        )

                    self._joint_state = state
                    sampled += 1

                final_state = self._joint_state

                final_pose = self.kinematics.forward_kinematics(
                    final_state.positions
                )

                return SimulationResult(
                    simulation_id=generate_id(),
                    trajectory_id=trajectory_id,
                    robot_id=self.config.robot_id,
                    status=TrajectoryStatus.SIMULATED,
                    final_joint_state=final_state,
                    final_end_effector_pose=final_pose,
                    sampled_states=sampled,
                    completed_at=utc_now(),
                )

            finally:
                if not self._emergency_stop:
                    self._mode = ArmMode.IDLE

    # --------------------------------------------------------
    # OBSTACLE MANAGEMENT
    # --------------------------------------------------------

    def add_obstacle(
        self,
        obstacle: SphericalObstacle,
        access: ArmAccessContext,
    ) -> None:
        self._authorize(
            access,
            "can_manage",
        )

        with self._lock:
            self._obstacles[
                obstacle.obstacle_id
            ] = obstacle

    def remove_obstacle(
        self,
        obstacle_id: str,
        access: ArmAccessContext,
    ) -> None:
        self._authorize(
            access,
            "can_manage",
        )

        with self._lock:
            if obstacle_id not in self._obstacles:
                raise ArmNotFoundError(
                    "Obstacle not found"
                )

            del self._obstacles[
                obstacle_id
            ]

    def check_collision(
        self,
        positions: Sequence[float],
        access: ArmAccessContext,
    ) -> CollisionReport:
        self._authorize(
            access,
            "can_read",
        )

        self.joint_validator.validate_positions(
            positions
        )

        with self._lock:
            return self.collision_checker.check(
                positions,
                tuple(self._obstacles.values()),
            )

    # --------------------------------------------------------
    # EMERGENCY STOP
    # --------------------------------------------------------

    def emergency_stop(
        self,
        access: ArmAccessContext,
    ) -> None:
        self._authorize(
            access,
            "can_manage",
        )

        with self._lock:
            self._emergency_stop = True
            self._mode = ArmMode.EMERGENCY_STOP

            zeros = tuple(
                0.0
                for _ in self.config.joints
            )

            self._joint_state = JointState(
                positions=self._joint_state.positions,
                velocities=zeros,
                accelerations=zeros,
            )

    def reset_emergency_stop(
        self,
        access: ArmAccessContext,
    ) -> None:
        self._authorize(
            access,
            "can_manage",
        )

        with self._lock:
            self._emergency_stop = False
            self._mode = ArmMode.IDLE

    # --------------------------------------------------------
    # HEALTH REPORT
    # --------------------------------------------------------

    def health_report(
        self,
        access: ArmAccessContext,
    ) -> ArmHealthReport:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            return self.health_monitor.evaluate(
                self._joint_state,
                emergency_stop_active=self._emergency_stop,
                robot_id=self.config.robot_id,
            )

    # --------------------------------------------------------
    # SINGULARITY CHECK
    # --------------------------------------------------------

    def singularity_measure(
        self,
        positions: Sequence[float],
        access: ArmAccessContext,
    ) -> float:
        self._authorize(
            access,
            "can_read",
        )

        self.joint_validator.validate_positions(
            positions
        )

        return self.kinematics.singularity_measure(
            positions
        )

    # --------------------------------------------------------
    # EXPORT STATE
    # --------------------------------------------------------

    def export_state_json(
        self,
        access: ArmAccessContext,
    ) -> str:
        state = self.get_state(
            access
        )

        return json.dumps(
            json_safe(state),
            indent=2,
            allow_nan=False,
        )


# ============================================================
# MULTI-TENANT ARM CONTROLLER SERVICE
# ============================================================

class ArmControllerService:

    def __init__(self) -> None:
        self._controllers: dict[
            tuple[str, str],
            ArmController,
        ] = {}

        self._lock = threading.RLock()

    def register_arm(
        self,
        config: ArmControllerConfig,
        access: ArmAccessContext,
    ) -> ArmController:
        if (
            not access.can_manage
            or access.tenant_id != config.tenant_id
        ):
            raise ArmAuthorizationError(
                "Not authorized to register arm"
            )

        key = (
            config.tenant_id,
            config.robot_id,
        )

        with self._lock:
            if key in self._controllers:
                raise ArmStateError(
                    "Robot arm already registered"
                )

            controller = ArmController(
                config
            )

            self._controllers[key] = controller

            return controller

    def get_arm(
        self,
        robot_id: str,
        access: ArmAccessContext,
    ) -> ArmController:
        if not access.can_read:
            raise ArmAuthorizationError(
                "Read permission required"
            )

        with self._lock:
            controller = self._controllers.get(
                (
                    access.tenant_id,
                    robot_id,
                )
            )

        if controller is None:
            raise ArmNotFoundError(
                "Robot arm not found"
            )

        return controller

    def list_arms(
        self,
        access: ArmAccessContext,
    ) -> tuple[str, ...]:
        if not access.can_read:
            raise ArmAuthorizationError(
                "Read permission required"
            )

        with self._lock:
            return tuple(
                robot_id
                for tenant_id, robot_id
                in self._controllers
                if tenant_id == access.tenant_id
            )


# ============================================================
# SYNTHETIC ROBOTIC ARM FACTORY
# ============================================================

class SyntheticArmFactory:

    @staticmethod
    def create_three_dof_arm(
        *,
        robot_id: str = "robot-arm-001",
        tenant_id: str = "physical_ai_lab",
    ) -> ArmControllerConfig:
        joints = (
            JointConfig(
                name="shoulder",
                joint_type=JointType.REVOLUTE,
                minimum_position=-math.pi,
                maximum_position=math.pi,
                maximum_velocity=1.5,
                maximum_acceleration=3.0,
                dh_a=0.5,
                dh_alpha=0.0,
                dh_d=0.0,
                home_position=0.0,
            ),
            JointConfig(
                name="elbow",
                joint_type=JointType.REVOLUTE,
                minimum_position=-2.5,
                maximum_position=2.5,
                maximum_velocity=1.5,
                maximum_acceleration=3.0,
                dh_a=0.4,
                dh_alpha=0.0,
                dh_d=0.0,
                home_position=0.0,
            ),
            JointConfig(
                name="wrist",
                joint_type=JointType.REVOLUTE,
                minimum_position=-2.5,
                maximum_position=2.5,
                maximum_velocity=2.0,
                maximum_acceleration=4.0,
                dh_a=0.2,
                dh_alpha=0.0,
                dh_d=0.0,
                home_position=0.0,
            ),
        )

        return ArmControllerConfig(
            robot_id=robot_id,
            tenant_id=tenant_id,
            joints=joints,
            workspace=WorkspaceBounds(
                minimum_x=-1.5,
                maximum_x=1.5,
                minimum_y=-1.5,
                maximum_y=1.5,
                minimum_z=-0.5,
                maximum_z=0.5,
            ),
            trajectory_sample_period=0.02,
            minimum_obstacle_clearance=0.05,
        )

    @staticmethod
    def access(
        tenant_id: str = "physical_ai_lab",
    ) -> ArmAccessContext:
        return ArmAccessContext(
            tenant_id=tenant_id,
            user_id="robotics_engineer",
            can_read=True,
            can_plan=True,
            can_simulate=True,
            can_manage=True,
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestArmController(unittest.TestCase):

    def setUp(self) -> None:
        self.config = (
            SyntheticArmFactory.create_three_dof_arm()
        )

        self.access = SyntheticArmFactory.access()

        self.controller = ArmController(
            self.config
        )

    def test_home_state(self) -> None:
        state = self.controller.get_state(
            self.access
        )

        self.assertEqual(
            state.mode,
            ArmMode.IDLE,
        )

        self.assertEqual(
            len(state.joint_state.positions),
            3,
        )

    def test_forward_kinematics(self) -> None:
        pose = self.controller.forward_kinematics(
            (0.0, 0.0, 0.0),
            self.access,
        )

        self.assertAlmostEqual(
            pose.position[0],
            1.1,
            places=5,
        )

        self.assertAlmostEqual(
            pose.position[1],
            0.0,
            places=5,
        )

    def test_joint_limit_validation(self) -> None:
        with self.assertRaises(
            JointLimitError
        ):
            self.controller.forward_kinematics(
                (10.0, 0.0, 0.0),
                self.access,
            )

    def test_jacobian_shape(self) -> None:
        jacobian = self.controller.kinematics.jacobian(
            (0.1, 0.2, 0.3)
        )

        self.assertEqual(
            jacobian.shape,
            (6, 3),
        )

    def test_joint_trajectory(self) -> None:
        trajectory = self.controller.plan_joint_motion(
            (0.2, 0.3, -0.1),
            self.access,
        )

        self.assertEqual(
            trajectory.status,
            TrajectoryStatus.VALIDATED,
        )

        self.assertGreater(
            len(trajectory.points),
            2,
        )

    def test_trajectory_endpoints(self) -> None:
        target = (
            0.2,
            0.3,
            -0.1,
        )

        trajectory = self.controller.plan_joint_motion(
            target,
            self.access,
        )

        self.assertTrue(
            np.allclose(
                trajectory.points[-1].positions,
                target,
            )
        )

    def test_trajectory_velocity_limits(self) -> None:
        trajectory = self.controller.plan_joint_motion(
            (0.5, 0.4, -0.3),
            self.access,
        )

        for point in trajectory.points:
            self.controller.joint_validator.validate_velocities(
                point.velocities
            )

    def test_simulation(self) -> None:
        trajectory = self.controller.plan_joint_motion(
            (0.2, 0.3, -0.1),
            self.access,
        )

        result = self.controller.simulate_trajectory(
            trajectory.trajectory_id,
            self.access,
        )

        self.assertEqual(
            result.status,
            TrajectoryStatus.SIMULATED,
        )

    def test_simulated_state(self) -> None:
        state = self.controller.set_simulated_joint_state(
            (0.1, 0.2, 0.3),
            self.access,
        )

        self.assertEqual(
            len(state.positions),
            3,
        )

    def test_collision_checker(self) -> None:
        obstacle = SphericalObstacle(
            obstacle_id="obstacle-001",
            center=(0.5, 0.0, 0.0),
            radius=0.1,
        )

        self.controller.add_obstacle(
            obstacle,
            self.access,
        )

        report = self.controller.check_collision(
            (0.0, 0.0, 0.0),
            self.access,
        )

        self.assertEqual(
            report.status,
            CollisionStatus.COLLISION_RISK,
        )

    def test_collision_rejection(self) -> None:
        self.controller.add_obstacle(
            SphericalObstacle(
                obstacle_id="blocking-obstacle",
                center=(0.5, 0.0, 0.0),
                radius=0.1,
            ),
            self.access,
        )

        with self.assertRaises(
            CollisionRiskError
        ):
            self.controller.plan_joint_motion(
                (0.1, 0.1, 0.1),
                self.access,
            )

    def test_emergency_stop(self) -> None:
        self.controller.emergency_stop(
            self.access
        )

        state = self.controller.get_state(
            self.access
        )

        self.assertTrue(
            state.emergency_stop_active
        )

    def test_emergency_stop_blocks_planning(self) -> None:
        self.controller.emergency_stop(
            self.access
        )

        with self.assertRaises(
            EmergencyStopError
        ):
            self.controller.plan_joint_motion(
                (0.1, 0.2, 0.3),
                self.access,
            )

    def test_emergency_stop_reset(self) -> None:
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
            len(report.joint_reports),
            3,
        )

    def test_inverse_kinematics(self) -> None:
        target = self.controller.forward_kinematics(
            (0.2, 0.3, -0.1),
            self.access,
        ).position

        result = self.controller.solve_inverse_kinematics(
            target,
            self.access,
            initial_positions=(0.2, 0.3, -0.1),
        )

        self.assertTrue(
            result.converged
        )

    def test_workspace_bounds(self) -> None:
        self.assertTrue(
            self.config.workspace.contains(
                (0.5, 0.5, 0.0)
            )
        )

        self.assertFalse(
            self.config.workspace.contains(
                (5.0, 0.0, 0.0)
            )
        )

    def test_singularity_measure(self) -> None:
        measure = self.controller.singularity_measure(
            (0.1, 0.2, 0.3),
            self.access,
        )

        self.assertGreaterEqual(
            measure,
            0.0,
        )

    def test_tenant_isolation(self) -> None:
        other = ArmAccessContext(
            tenant_id="other_tenant",
            user_id="other_user",
            can_read=True,
        )

        with self.assertRaises(
            ArmAuthorizationError
        ):
            self.controller.get_state(
                other
            )

    def test_permission_denied(self) -> None:
        viewer = ArmAccessContext(
            tenant_id="physical_ai_lab",
            user_id="viewer",
            can_read=True,
        )

        with self.assertRaises(
            ArmAuthorizationError
        ):
            self.controller.plan_joint_motion(
                (0.1, 0.2, 0.3),
                viewer,
            )

    def test_service_registration(self) -> None:
        service = ArmControllerService()

        service.register_arm(
            self.config,
            self.access,
        )

        self.assertEqual(
            service.list_arms(
                self.access
            ),
            ("robot-arm-001",),
        )

    def test_export_state(self) -> None:
        exported = json.loads(
            self.controller.export_state_json(
                self.access
            )
        )

        self.assertEqual(
            exported["robot_id"],
            "robot-arm-001",
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    config = SyntheticArmFactory.create_three_dof_arm(
        robot_id="physical-ai-arm-001",
        tenant_id="physical_ai_lab",
    )

    access = SyntheticArmFactory.access(
        tenant_id="physical_ai_lab"
    )

    service = ArmControllerService()

    controller = service.register_arm(
        config,
        access,
    )

    print("\nPHYSICAL AI ROBOT ARM CONTROLLER")
    print("=" * 70)

    initial_state = controller.get_state(
        access
    )

    print(
        "Initial end-effector position:",
        initial_state.end_effector_pose.position,
    )

    trajectory = controller.plan_joint_motion(
        target_positions=(
            0.3,
            0.4,
            -0.2,
        ),
        access=access,
    )

    print(
        "Trajectory ID:",
        trajectory.trajectory_id,
    )

    print(
        "Trajectory duration:",
        round(
            trajectory.duration_seconds,
            3,
        ),
        "seconds",
    )

    print(
        "Trajectory samples:",
        len(trajectory.points),
    )

    simulation = controller.simulate_trajectory(
        trajectory.trajectory_id,
        access,
    )

    print(
        "Final joint positions:",
        simulation.final_joint_state.positions,
    )

    print(
        "Final end-effector position:",
        simulation.final_end_effector_pose.position,
    )

    health = controller.health_report(
        access
    )

    print(
        "Robot health:",
        health.overall_health.value,
    )

    print("\nSTATE REPORT")
    print("=" * 70)

    print(
        controller.export_state_json(
            access
        )
    )

    print("\nRUNNING UNIT TESTS")
    print("=" * 70)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestArmController
    )

    result = unittest.TextTestRunner(
        verbosity=2
    ).run(suite)

    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
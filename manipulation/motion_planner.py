"""Motion Planner module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/manipulation/motion_planner.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================
Module: Motion Planner

Simulation-only joint-space motion planning for serial robot arms.

Features:
    - Joint-space RRT-Connect planning
    - Direct-path planning
    - Joint position, velocity, and acceleration limits
    - Forward kinematics using standard DH parameters
    - Spherical workspace obstacles
    - Capsule-based robot link collision checking
    - Conservative sampled self-collision checking
    - Continuous path sampling at configurable joint resolution
    - Joint-space path shortcutting
    - Quintic time-parameterized trajectories
    - Trajectory validation
    - Deterministic random seeds
    - Planning diagnostics
    - Multi-tenant robot registry
    - Thread-safe planning services
    - JSON export
    - Synthetic robot examples
    - Unit tests

Dependencies:
    Python >= 3.11
    numpy >= 1.24

Install:
    pip install numpy

IMPORTANT:
    This is a planning and simulation component only.
    It does not send commands to physical robots.

    The collision model is approximate and uses spherical
    obstacles and capsule-shaped robot links. Collision
    sampling does not constitute formal continuous collision
    detection or safety certification.

    Real-world execution requires independent hardware
    safety systems, robot-specific dynamics, calibrated
    geometry, collision validation, and certified controls.
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

class MotionPlannerError(Exception):
    pass


class MotionValidationError(MotionPlannerError):
    pass


class MotionCollisionError(MotionPlannerError):
    pass


class MotionPlanningError(MotionPlannerError):
    pass


class MotionAuthorizationError(MotionPlannerError):
    pass


class MotionRobotNotFoundError(MotionPlannerError):
    pass


class MotionConflictError(MotionPlannerError):
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
        raise MotionValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MotionValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise MotionValidationError(
            f"{name} must be finite"
        )

    return result


def identifier(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 256
    ):
        raise MotionValidationError(
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
        raise MotionValidationError(
            f"{name} must be numeric"
        ) from exc

    if result.shape != (size,):
        raise MotionValidationError(
            f"{name} must have shape ({size},)"
        )

    if not np.all(np.isfinite(result)):
        raise MotionValidationError(
            f"{name} contains non-finite values"
        )

    return result


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(minimum, min(maximum, value))


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
        return [json_safe(item) for item in value]

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


# ============================================================
# ENUMERATIONS
# ============================================================

class JointType(str, Enum):
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"


class PlannerAlgorithm(str, Enum):
    DIRECT = "direct"
    RRT_CONNECT = "rrt_connect"


class PlanningStatus(str, Enum):
    SUCCESS = "success"
    INVALID_START = "invalid_start"
    INVALID_GOAL = "invalid_goal"
    COLLISION = "collision"
    MAX_ITERATIONS = "max_iterations"
    FAILED = "failed"


class CollisionType(str, Enum):
    NONE = "none"
    ENVIRONMENT = "environment"
    SELF_COLLISION = "self_collision"
    JOINT_LIMIT = "joint_limit"


class TrajectoryStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"


# ============================================================
# CONFIGURATION MODELS
# ============================================================

@dataclass(frozen=True)
class MotionJoint:
    name: str
    joint_type: JointType

    minimum: float
    maximum: float

    maximum_velocity: float
    maximum_acceleration: float

    dh_a_m: float = 0.0
    dh_alpha_rad: float = 0.0
    dh_d_m: float = 0.0
    dh_theta_rad: float = 0.0

    link_radius_m: float = 0.025

    home_position: float = 0.0

    def __post_init__(self) -> None:
        identifier(self.name, "joint name")

        for name in (
            "minimum",
            "maximum",
            "maximum_velocity",
            "maximum_acceleration",
            "dh_a_m",
            "dh_alpha_rad",
            "dh_d_m",
            "dh_theta_rad",
            "link_radius_m",
            "home_position",
        ):
            finite(getattr(self, name), name)

        if self.minimum >= self.maximum:
            raise MotionValidationError(
                f"Invalid joint limits: {self.name}"
            )

        if self.maximum_velocity <= 0:
            raise MotionValidationError(
                "Maximum velocity must be positive"
            )

        if self.maximum_acceleration <= 0:
            raise MotionValidationError(
                "Maximum acceleration must be positive"
            )

        if self.link_radius_m < 0:
            raise MotionValidationError(
                "Link radius cannot be negative"
            )

        if not (
            self.minimum
            <= self.home_position
            <= self.maximum
        ):
            raise MotionValidationError(
                "Home position outside joint limits"
            )


@dataclass(frozen=True)
class MotionRobot:
    robot_id: str
    tenant_id: str
    joints: tuple[MotionJoint, ...]

    def __post_init__(self) -> None:
        identifier(self.robot_id, "robot_id")
        identifier(self.tenant_id, "tenant_id")

        if not self.joints:
            raise MotionValidationError(
                "Robot requires at least one joint"
            )

        names = [joint.name for joint in self.joints]

        if len(names) != len(set(names)):
            raise MotionValidationError(
                "Duplicate joint names"
            )

    @property
    def dof(self) -> int:
        return len(self.joints)

    def home_configuration(self) -> np.ndarray:
        return np.array(
            [joint.home_position for joint in self.joints],
            dtype=float,
        )


@dataclass(frozen=True)
class SphericalObstacle:
    obstacle_id: str
    center_xyz: tuple[float, float, float]
    radius_m: float

    def __post_init__(self) -> None:
        identifier(self.obstacle_id, "obstacle_id")
        vector(self.center_xyz, 3, "obstacle center")

        if finite(self.radius_m, "radius_m") <= 0:
            raise MotionValidationError(
                "Obstacle radius must be positive"
            )


@dataclass(frozen=True)
class MotionPlannerConfig:
    algorithm: PlannerAlgorithm = PlannerAlgorithm.RRT_CONNECT

    maximum_iterations: int = 3000
    extension_step: float = 0.15
    collision_resolution: float = 0.04

    goal_bias: float = 0.10
    goal_tolerance: float = 0.05

    collision_margin_m: float = 0.01

    shortcut_iterations: int = 100

    trajectory_sample_period_s: float = 0.02

    minimum_segment_duration_s: float = 0.10

    velocity_utilization: float = 0.80
    acceleration_utilization: float = 0.80

    random_seed: int = 42

    check_self_collision: bool = True

    def __post_init__(self) -> None:
        if self.maximum_iterations < 1:
            raise MotionValidationError(
                "maximum_iterations must be positive"
            )

        if self.shortcut_iterations < 0:
            raise MotionValidationError(
                "shortcut_iterations cannot be negative"
            )

        for name in (
            "extension_step",
            "collision_resolution",
            "goal_tolerance",
            "trajectory_sample_period_s",
            "minimum_segment_duration_s",
        ):
            if finite(getattr(self, name), name) <= 0:
                raise MotionValidationError(
                    f"{name} must be positive"
                )

        if not 0 <= self.goal_bias <= 1:
            raise MotionValidationError(
                "goal_bias must be between 0 and 1"
            )

        if self.collision_margin_m < 0:
            raise MotionValidationError(
                "collision_margin_m cannot be negative"
            )

        for name in (
            "velocity_utilization",
            "acceleration_utilization",
        ):
            value = finite(getattr(self, name), name)

            if not 0 < value <= 1:
                raise MotionValidationError(
                    f"{name} must be in (0,1]"
                )


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

@dataclass(frozen=True)
class MotionPlanningRequest:
    request_id: str

    start_configuration: tuple[float, ...]
    goal_configuration: tuple[float, ...]

    obstacles: tuple[SphericalObstacle, ...] = ()

    optimize_path: bool = True

    def __post_init__(self) -> None:
        identifier(self.request_id, "request_id")


@dataclass(frozen=True)
class CollisionReport:
    collision: bool
    collision_type: CollisionType

    obstacle_id: str | None

    minimum_clearance_m: float

    message: str


@dataclass(frozen=True)
class TrajectoryPoint:
    time_from_start_s: float

    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    accelerations: tuple[float, ...]


@dataclass(frozen=True)
class MotionTrajectory:
    trajectory_id: str

    robot_id: str
    tenant_id: str

    points: tuple[TrajectoryPoint, ...]

    duration_s: float

    status: TrajectoryStatus


@dataclass(frozen=True)
class MotionPlanningResult:
    result_id: str
    request_id: str

    robot_id: str
    tenant_id: str

    status: PlanningStatus

    algorithm: PlannerAlgorithm

    path: tuple[tuple[float, ...], ...]

    trajectory: MotionTrajectory | None

    iterations: int
    nodes_explored: int

    path_length: float

    message: str

    created_at: datetime


@dataclass(frozen=True)
class MotionAccessContext:
    tenant_id: str
    user_id: str

    can_read: bool = False
    can_plan: bool = False
    can_manage: bool = False


# ============================================================
# FORWARD KINEMATICS
# ============================================================

class MotionKinematics:

    def __init__(self, robot: MotionRobot) -> None:
        self.robot = robot

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

    def transforms(
        self,
        configuration: Sequence[float],
    ) -> list[np.ndarray]:
        q = vector(
            configuration,
            self.robot.dof,
            "configuration",
        )

        transform = np.eye(4)

        result = [transform.copy()]

        for joint, position in zip(
            self.robot.joints,
            q,
        ):
            theta = joint.dh_theta_rad
            d = joint.dh_d_m

            if joint.joint_type == JointType.REVOLUTE:
                theta += float(position)
            else:
                d += float(position)

            transform = (
                transform
                @ self.dh_transform(
                    joint.dh_a_m,
                    joint.dh_alpha_rad,
                    d,
                    theta,
                )
            )

            result.append(transform.copy())

        return result

    def joint_positions(
        self,
        configuration: Sequence[float],
    ) -> np.ndarray:
        transforms = self.transforms(configuration)

        return np.array(
            [transform[:3, 3] for transform in transforms],
            dtype=float,
        )

    def end_effector_position(
        self,
        configuration: Sequence[float],
    ) -> np.ndarray:
        return self.transforms(configuration)[-1][:3, 3]


# ============================================================
# COLLISION GEOMETRY
# ============================================================

class CollisionGeometry:

    @staticmethod
    def point_segment_distance(
        point: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
    ) -> float:
        direction = end - start

        denominator = float(
            np.dot(direction, direction)
        )

        if denominator <= 1e-14:
            return float(np.linalg.norm(point - start))

        parameter = float(
            np.dot(point - start, direction)
            / denominator
        )

        parameter = clamp(parameter, 0.0, 1.0)

        closest = start + parameter * direction

        return float(np.linalg.norm(point - closest))

    @staticmethod
    def segment_segment_distance(
        p1: np.ndarray,
        q1: np.ndarray,
        p2: np.ndarray,
        q2: np.ndarray,
    ) -> float:
        """
        Minimum distance between two finite 3D segments.
        """
        d1 = q1 - p1
        d2 = q2 - p2
        r = p1 - p2

        a = float(np.dot(d1, d1))
        e = float(np.dot(d2, d2))
        f = float(np.dot(d2, r))

        epsilon = 1e-12

        if a <= epsilon and e <= epsilon:
            return float(np.linalg.norm(p1 - p2))

        if a <= epsilon:
            s = 0.0
            t = clamp(f / e, 0.0, 1.0)

        else:
            c = float(np.dot(d1, r))

            if e <= epsilon:
                t = 0.0
                s = clamp(-c / a, 0.0, 1.0)

            else:
                b = float(np.dot(d1, d2))

                denominator = a * e - b * b

                if abs(denominator) > epsilon:
                    s = clamp(
                        (b * f - c * e) / denominator,
                        0.0,
                        1.0,
                    )
                else:
                    s = 0.0

                t = (b * s + f) / e

                if t < 0.0:
                    t = 0.0
                    s = clamp(-c / a, 0.0, 1.0)

                elif t > 1.0:
                    t = 1.0
                    s = clamp((b - c) / a, 0.0, 1.0)

        closest_1 = p1 + s * d1
        closest_2 = p2 + t * d2

        return float(
            np.linalg.norm(closest_1 - closest_2)
        )


# ============================================================
# COLLISION CHECKER
# ============================================================

class MotionCollisionChecker:

    def __init__(
        self,
        robot: MotionRobot,
        config: MotionPlannerConfig,
    ) -> None:
        self.robot = robot
        self.config = config

        self.kinematics = MotionKinematics(robot)

    def check_configuration(
        self,
        configuration: Sequence[float],
        obstacles: Sequence[SphericalObstacle] = (),
    ) -> CollisionReport:
        q = vector(
            configuration,
            self.robot.dof,
            "configuration",
        )

        for joint, position in zip(self.robot.joints, q):
            if not (
                joint.minimum <= position <= joint.maximum
            ):
                return CollisionReport(
                    collision=True,
                    collision_type=CollisionType.JOINT_LIMIT,
                    obstacle_id=None,
                    minimum_clearance_m=0.0,
                    message=f"Joint limit exceeded: {joint.name}",
                )

        positions = self.kinematics.joint_positions(q)

        minimum_clearance = math.inf

        for index, joint in enumerate(self.robot.joints):
            start = positions[index]
            end = positions[index + 1]

            for obstacle in obstacles:
                center = vector(
                    obstacle.center_xyz,
                    3,
                    "obstacle center",
                )

                distance = (
                    CollisionGeometry.point_segment_distance(
                        center,
                        start,
                        end,
                    )
                )

                clearance = (
                    distance
                    - joint.link_radius_m
                    - obstacle.radius_m
                    - self.config.collision_margin_m
                )

                minimum_clearance = min(
                    minimum_clearance,
                    clearance,
                )

                if clearance <= 0:
                    return CollisionReport(
                        collision=True,
                        collision_type=CollisionType.ENVIRONMENT,
                        obstacle_id=obstacle.obstacle_id,
                        minimum_clearance_m=clearance,
                        message=(
                            f"Link {index} intersects "
                            f"obstacle {obstacle.obstacle_id}"
                        ),
                    )

        if self.config.check_self_collision:
            # Adjacent links share a joint and are excluded.
            # This approximation does not represent the full
            # shape of a real robot's joints or tool.
            for first in range(self.robot.dof):
                for second in range(
                    first + 2,
                    self.robot.dof,
                ):
                    distance = (
                        CollisionGeometry.segment_segment_distance(
                            positions[first],
                            positions[first + 1],
                            positions[second],
                            positions[second + 1],
                        )
                    )

                    clearance = (
                        distance
                        - self.robot.joints[first].link_radius_m
                        - self.robot.joints[second].link_radius_m
                        - self.config.collision_margin_m
                    )

                    minimum_clearance = min(
                        minimum_clearance,
                        clearance,
                    )

                    if clearance <= 0:
                        return CollisionReport(
                            collision=True,
                            collision_type=CollisionType.SELF_COLLISION,
                            obstacle_id=None,
                            minimum_clearance_m=clearance,
                            message=(
                                f"Possible collision between "
                                f"links {first} and {second}"
                            ),
                        )

        return CollisionReport(
            collision=False,
            collision_type=CollisionType.NONE,
            obstacle_id=None,
            minimum_clearance_m=minimum_clearance,
            message="No collision detected by approximate model",
        )

    def check_edge(
        self,
        start: Sequence[float],
        end: Sequence[float],
        obstacles: Sequence[SphericalObstacle] = (),
    ) -> CollisionReport:
        q_start = vector(
            start,
            self.robot.dof,
            "start",
        )

        q_end = vector(
            end,
            self.robot.dof,
            "end",
        )

        distance = float(
            np.linalg.norm(q_end - q_start)
        )

        samples = max(
            1,
            int(
                math.ceil(
                    distance / self.config.collision_resolution
                )
            ),
        )

        minimum_clearance = math.inf

        for index in range(samples + 1):
            alpha = index / samples

            q = (
                (1.0 - alpha) * q_start
                + alpha * q_end
            )

            report = self.check_configuration(
                q,
                obstacles,
            )

            minimum_clearance = min(
                minimum_clearance,
                report.minimum_clearance_m,
            )

            if report.collision:
                return report

        return CollisionReport(
            collision=False,
            collision_type=CollisionType.NONE,
            obstacle_id=None,
            minimum_clearance_m=minimum_clearance,
            message="Sampled edge is collision-free",
        )


# ============================================================
# RRT TREE
# ============================================================

@dataclass
class RRTNode:
    configuration: np.ndarray
    parent_index: int | None


class RRTTree:

    def __init__(
        self,
        root: np.ndarray,
    ) -> None:
        self.nodes: list[RRTNode] = [
            RRTNode(
                configuration=root.copy(),
                parent_index=None,
            )
        ]

    def nearest(
        self,
        target: np.ndarray,
    ) -> int:
        distances = [
            float(
                np.linalg.norm(
                    node.configuration - target
                )
            )
            for node in self.nodes
        ]

        return int(np.argmin(distances))

    def add(
        self,
        configuration: np.ndarray,
        parent_index: int,
    ) -> int:
        self.nodes.append(
            RRTNode(
                configuration=configuration.copy(),
                parent_index=parent_index,
            )
        )

        return len(self.nodes) - 1

    def trace(
        self,
        index: int,
    ) -> list[np.ndarray]:
        result: list[np.ndarray] = []

        while index is not None:
            node = self.nodes[index]

            result.append(
                node.configuration.copy()
            )

            index = node.parent_index

        result.reverse()

        return result


# ============================================================
# RRT-CONNECT PLANNER
# ============================================================

class RRTConnectPlanner:

    def __init__(
        self,
        robot: MotionRobot,
        checker: MotionCollisionChecker,
        config: MotionPlannerConfig,
    ) -> None:
        self.robot = robot
        self.checker = checker
        self.config = config

        self.minimum = np.array(
            [joint.minimum for joint in robot.joints],
            dtype=float,
        )

        self.maximum = np.array(
            [joint.maximum for joint in robot.joints],
            dtype=float,
        )

    def _steer(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        direction = target - source

        distance = float(
            np.linalg.norm(direction)
        )

        if distance <= self.config.extension_step:
            return target.copy()

        return (
            source
            + direction / distance
            * self.config.extension_step
        )

    def _extend(
        self,
        tree: RRTTree,
        target: np.ndarray,
        obstacles: Sequence[SphericalObstacle],
    ) -> tuple[int | None, bool]:
        nearest_index = tree.nearest(target)

        nearest = (
            tree.nodes[nearest_index].configuration
        )

        candidate = self._steer(
            nearest,
            target,
        )

        if np.allclose(
            nearest,
            candidate,
            atol=1e-12,
        ):
            return nearest_index, True

        report = self.checker.check_edge(
            nearest,
            candidate,
            obstacles,
        )

        if report.collision:
            return None, False

        index = tree.add(
            candidate,
            nearest_index,
        )

        reached = bool(
            np.linalg.norm(candidate - target)
            <= 1e-9
        )

        return index, reached

    def _connect(
        self,
        tree: RRTTree,
        target: np.ndarray,
        obstacles: Sequence[SphericalObstacle],
    ) -> tuple[int | None, bool]:
        last_index: int | None = None

        while True:
            index, reached = self._extend(
                tree,
                target,
                obstacles,
            )

            if index is None:
                return last_index, False

            last_index = index

            if reached:
                return index, True

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: Sequence[SphericalObstacle],
    ) -> tuple[
        list[np.ndarray] | None,
        int,
        int,
    ]:
        direct = self.checker.check_edge(
            start,
            goal,
            obstacles,
        )

        if not direct.collision:
            return (
                [start.copy(), goal.copy()],
                0,
                2,
            )

        rng = np.random.default_rng(
            self.config.random_seed
        )

        start_tree = RRTTree(start)
        goal_tree = RRTTree(goal)

        for iteration in range(
            1,
            self.config.maximum_iterations + 1,
        ):
            if rng.random() < self.config.goal_bias:
                sample = goal.copy()
            else:
                sample = rng.uniform(
                    self.minimum,
                    self.maximum,
                )

            # Alternate which tree is extended first.
            if iteration % 2 == 1:
                active = start_tree
                passive = goal_tree
                active_is_start = True
            else:
                active = goal_tree
                passive = start_tree
                active_is_start = False

            new_index, _ = self._extend(
                active,
                sample,
                obstacles,
            )

            if new_index is None:
                continue

            new_configuration = (
                active.nodes[new_index].configuration
            )

            connected_index, connected = self._connect(
                passive,
                new_configuration,
                obstacles,
            )

            if connected and connected_index is not None:
                active_path = active.trace(new_index)
                passive_path = passive.trace(
                    connected_index
                )

                if active_is_start:
                    start_path = active_path
                    goal_path = passive_path
                else:
                    start_path = passive_path
                    goal_path = active_path

                path = (
                    start_path
                    + list(reversed(goal_path[:-1]))
                )

                # Recheck the assembled path before
                # returning a candidate.
                if all(
                    not self.checker.check_edge(
                        path[index],
                        path[index + 1],
                        obstacles,
                    ).collision
                    for index in range(len(path) - 1)
                ):
                    return (
                        path,
                        iteration,
                        len(start_tree.nodes)
                        + len(goal_tree.nodes),
                    )

        return (
            None,
            self.config.maximum_iterations,
            len(start_tree.nodes)
            + len(goal_tree.nodes),
        )


# ============================================================
# PATH OPTIMIZATION
# ============================================================

class MotionPathOptimizer:

    def __init__(
        self,
        checker: MotionCollisionChecker,
        config: MotionPlannerConfig,
    ) -> None:
        self.checker = checker
        self.config = config

    @staticmethod
    def path_length(
        path: Sequence[np.ndarray],
    ) -> float:
        return sum(
            float(
                np.linalg.norm(
                    path[index + 1] - path[index]
                )
            )
            for index in range(len(path) - 1)
        )

    def shortcut(
        self,
        path: Sequence[np.ndarray],
        obstacles: Sequence[SphericalObstacle],
    ) -> list[np.ndarray]:
        optimized = [
            point.copy()
            for point in path
        ]

        if len(optimized) <= 2:
            return optimized

        rng = np.random.default_rng(
            self.config.random_seed + 1
        )

        for _ in range(
            self.config.shortcut_iterations
        ):
            if len(optimized) <= 2:
                break

            first, second = sorted(
                rng.choice(
                    len(optimized),
                    size=2,
                    replace=False,
                )
            )

            if second - first < 2:
                continue

            report = self.checker.check_edge(
                optimized[first],
                optimized[second],
                obstacles,
            )

            if not report.collision:
                optimized = (
                    optimized[:first + 1]
                    + optimized[second:]
                )

        return optimized


# ============================================================
# QUINTIC TRAJECTORY GENERATOR
# ============================================================

class QuinticTrajectoryGenerator:
    """
    Each path segment uses a quintic interpolation:

        s(u) = 10u^3 - 15u^4 + 6u^5

    Position, velocity, and acceleration are zero at
    every waypoint boundary.

    This produces stop-and-go motion, not a globally
    optimized minimum-time trajectory.
    """

    MAX_NORMALIZED_VELOCITY = 1.875

    MAX_NORMALIZED_ACCELERATION = (
        10.0 * math.sqrt(3.0) / 3.0
    )

    def __init__(
        self,
        robot: MotionRobot,
        config: MotionPlannerConfig,
    ) -> None:
        self.robot = robot
        self.config = config

    @staticmethod
    def profile(
        u: float,
    ) -> tuple[float, float, float]:
        u = clamp(u, 0.0, 1.0)

        position = (
            10.0 * u**3
            - 15.0 * u**4
            + 6.0 * u**5
        )

        velocity = (
            30.0 * u**2
            - 60.0 * u**3
            + 30.0 * u**4
        )

        acceleration = (
            60.0 * u
            - 180.0 * u**2
            + 120.0 * u**3
        )

        return position, velocity, acceleration

    def segment_duration(
        self,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> float:
        displacement = np.abs(
            goal - start
        )

        durations = [
            self.config.minimum_segment_duration_s
        ]

        for index, joint in enumerate(
            self.robot.joints
        ):
            distance = float(
                displacement[index]
            )

            if distance <= 1e-12:
                continue

            velocity_limit = (
                joint.maximum_velocity
                * self.config.velocity_utilization
            )

            acceleration_limit = (
                joint.maximum_acceleration
                * self.config.acceleration_utilization
            )

            velocity_duration = (
                self.MAX_NORMALIZED_VELOCITY
                * distance
                / velocity_limit
            )

            acceleration_duration = math.sqrt(
                self.MAX_NORMALIZED_ACCELERATION
                * distance
                / acceleration_limit
            )

            durations.extend(
                [
                    velocity_duration,
                    acceleration_duration,
                ]
            )

        return max(durations)

    def generate(
        self,
        path: Sequence[np.ndarray],
    ) -> MotionTrajectory:
        if not path:
            raise MotionValidationError(
                "Trajectory path cannot be empty"
            )

        for configuration in path:
            vector(
                configuration,
                self.robot.dof,
                "path configuration",
            )

        if len(path) == 1:
            zeros = tuple(
                0.0 for _ in range(self.robot.dof)
            )

            return MotionTrajectory(
                trajectory_id=new_id(),
                robot_id=self.robot.robot_id,
                tenant_id=self.robot.tenant_id,
                points=(
                    TrajectoryPoint(
                        time_from_start_s=0.0,
                        positions=tuple(
                            float(v) for v in path[0]
                        ),
                        velocities=zeros,
                        accelerations=zeros,
                    ),
                ),
                duration_s=0.0,
                status=TrajectoryStatus.VALID,
            )

        points: list[TrajectoryPoint] = []

        elapsed = 0.0

        for segment_index in range(
            len(path) - 1
        ):
            start = path[segment_index]
            goal = path[segment_index + 1]

            duration = self.segment_duration(
                start,
                goal,
            )

            samples = max(
                2,
                int(
                    math.ceil(
                        duration
                        / self.config.trajectory_sample_period_s
                    )
                ),
            )

            displacement = goal - start

            for sample_index in range(samples + 1):
                if (
                    segment_index > 0
                    and sample_index == 0
                ):
                    continue

                u = sample_index / samples

                (
                    position_factor,
                    velocity_factor,
                    acceleration_factor,
                ) = self.profile(u)

                positions = (
                    start
                    + position_factor * displacement
                )

                velocities = (
                    velocity_factor
                    * displacement
                    / duration
                )

                accelerations = (
                    acceleration_factor
                    * displacement
                    / (duration * duration)
                )

                points.append(
                    TrajectoryPoint(
                        time_from_start_s=(
                            elapsed + u * duration
                        ),
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
                )

            elapsed += duration

        return MotionTrajectory(
            trajectory_id=new_id(),
            robot_id=self.robot.robot_id,
            tenant_id=self.robot.tenant_id,
            points=tuple(points),
            duration_s=elapsed,
            status=TrajectoryStatus.VALID,
        )


# ============================================================
# TRAJECTORY VALIDATOR
# ============================================================

class MotionTrajectoryValidator:

    def __init__(
        self,
        robot: MotionRobot,
        checker: MotionCollisionChecker,
    ) -> None:
        self.robot = robot
        self.checker = checker

    def validate(
        self,
        trajectory: MotionTrajectory,
        obstacles: Sequence[SphericalObstacle],
    ) -> tuple[bool, str]:
        if not trajectory.points:
            return False, "Trajectory has no points"

        previous_time = -math.inf
        previous_configuration: np.ndarray | None = None

        for point in trajectory.points:
            if (
                not math.isfinite(point.time_from_start_s)
                or point.time_from_start_s < 0
            ):
                return False, "Invalid trajectory timestamp"

            if point.time_from_start_s <= previous_time:
                return False, "Trajectory times must increase"

            previous_time = point.time_from_start_s

            positions = vector(
                point.positions,
                self.robot.dof,
                "trajectory positions",
            )

            velocities = vector(
                point.velocities,
                self.robot.dof,
                "trajectory velocities",
            )

            accelerations = vector(
                point.accelerations,
                self.robot.dof,
                "trajectory accelerations",
            )

            for index, joint in enumerate(
                self.robot.joints
            ):
                if not (
                    joint.minimum - 1e-8
                    <= positions[index]
                    <= joint.maximum + 1e-8
                ):
                    return False, "Joint position limit exceeded"

                if (
                    abs(velocities[index])
                    > joint.maximum_velocity + 1e-8
                ):
                    return False, "Joint velocity limit exceeded"

                if (
                    abs(accelerations[index])
                    > joint.maximum_acceleration + 1e-8
                ):
                    return False, "Joint acceleration limit exceeded"

            report = self.checker.check_configuration(
                positions,
                obstacles,
            )

            if report.collision:
                return False, report.message

            if previous_configuration is not None:
                edge_report = self.checker.check_edge(
                    previous_configuration,
                    positions,
                    obstacles,
                )

                if edge_report.collision:
                    return False, edge_report.message

            previous_configuration = positions

        if not math.isclose(
            trajectory.points[-1].time_from_start_s,
            trajectory.duration_s,
            abs_tol=1e-7,
        ):
            return False, "Trajectory duration mismatch"

        return True, "Trajectory passed sampled validation"


# ============================================================
# MOTION PLANNER
# ============================================================

class MotionPlanner:

    def __init__(
        self,
        robot: MotionRobot,
        config: MotionPlannerConfig | None = None,
    ) -> None:
        self.robot = robot

        self.config = (
            config or MotionPlannerConfig()
        )

        self.kinematics = MotionKinematics(robot)

        self.collision_checker = MotionCollisionChecker(
            robot,
            self.config,
        )

        self.rrt = RRTConnectPlanner(
            robot,
            self.collision_checker,
            self.config,
        )

        self.optimizer = MotionPathOptimizer(
            self.collision_checker,
            self.config,
        )

        self.trajectory_generator = (
            QuinticTrajectoryGenerator(
                robot,
                self.config,
            )
        )

        self.trajectory_validator = (
            MotionTrajectoryValidator(
                robot,
                self.collision_checker,
            )
        )

        self._lock = threading.RLock()

    def _failure(
        self,
        request: MotionPlanningRequest,
        status: PlanningStatus,
        message: str,
        iterations: int = 0,
        nodes_explored: int = 0,
    ) -> MotionPlanningResult:
        return MotionPlanningResult(
            result_id=new_id(),
            request_id=request.request_id,
            robot_id=self.robot.robot_id,
            tenant_id=self.robot.tenant_id,
            status=status,
            algorithm=self.config.algorithm,
            path=(),
            trajectory=None,
            iterations=iterations,
            nodes_explored=nodes_explored,
            path_length=0.0,
            message=message,
            created_at=utc_now(),
        )

    def plan(
        self,
        request: MotionPlanningRequest,
    ) -> MotionPlanningResult:
        with self._lock:
            start = vector(
                request.start_configuration,
                self.robot.dof,
                "start configuration",
            )

            goal = vector(
                request.goal_configuration,
                self.robot.dof,
                "goal configuration",
            )

            obstacles = request.obstacles

            start_report = (
                self.collision_checker.check_configuration(
                    start,
                    obstacles,
                )
            )

            if start_report.collision:
                return self._failure(
                    request,
                    PlanningStatus.INVALID_START,
                    start_report.message,
                )

            goal_report = (
                self.collision_checker.check_configuration(
                    goal,
                    obstacles,
                )
            )

            if goal_report.collision:
                return self._failure(
                    request,
                    PlanningStatus.INVALID_GOAL,
                    goal_report.message,
                )

            if np.allclose(
                start,
                goal,
                atol=1e-10,
            ):
                path = [start.copy()]
                iterations = 0
                nodes_explored = 1

            elif self.config.algorithm == PlannerAlgorithm.DIRECT:
                report = self.collision_checker.check_edge(
                    start,
                    goal,
                    obstacles,
                )

                if report.collision:
                    return self._failure(
                        request,
                        PlanningStatus.COLLISION,
                        report.message,
                    )

                path = [
                    start.copy(),
                    goal.copy(),
                ]

                iterations = 0
                nodes_explored = 2

            else:
                (
                    path,
                    iterations,
                    nodes_explored,
                ) = self.rrt.plan(
                    start,
                    goal,
                    obstacles,
                )

                if path is None:
                    return self._failure(
                        request,
                        PlanningStatus.MAX_ITERATIONS,
                        "No collision-free path found",
                        iterations,
                        nodes_explored,
                    )

            if (
                request.optimize_path
                and len(path) > 2
            ):
                path = self.optimizer.shortcut(
                    path,
                    obstacles,
                )

            for index in range(len(path) - 1):
                report = self.collision_checker.check_edge(
                    path[index],
                    path[index + 1],
                    obstacles,
                )

                if report.collision:
                    return self._failure(
                        request,
                        PlanningStatus.COLLISION,
                        "Final path failed collision validation",
                        iterations,
                        nodes_explored,
                    )

            trajectory = (
                self.trajectory_generator.generate(
                    path
                )
            )

            valid, message = (
                self.trajectory_validator.validate(
                    trajectory,
                    obstacles,
                )
            )

            if not valid:
                return self._failure(
                    request,
                    PlanningStatus.FAILED,
                    f"Trajectory validation failed: {message}",
                    iterations,
                    nodes_explored,
                )

            return MotionPlanningResult(
                result_id=new_id(),
                request_id=request.request_id,
                robot_id=self.robot.robot_id,
                tenant_id=self.robot.tenant_id,
                status=PlanningStatus.SUCCESS,
                algorithm=self.config.algorithm,
                path=tuple(
                    tuple(float(value) for value in point)
                    for point in path
                ),
                trajectory=trajectory,
                iterations=iterations,
                nodes_explored=nodes_explored,
                path_length=self.optimizer.path_length(path),
                message=(
                    "Simulation-only motion plan generated "
                    "and validated using sampled collision checks"
                ),
                created_at=utc_now(),
            )

    def plan_batch(
        self,
        requests: Sequence[MotionPlanningRequest],
    ) -> tuple[MotionPlanningResult, ...]:
        return tuple(
            self.plan(request)
            for request in requests
        )


# ============================================================
# MULTI-TENANT SERVICE
# ============================================================

class MotionPlannerService:

    def __init__(self) -> None:
        self._planners: dict[
            tuple[str, str],
            MotionPlanner,
        ] = {}

        self._lock = threading.RLock()

    @staticmethod
    def _authorize(
        access: MotionAccessContext,
        permission: str,
    ) -> None:
        if not getattr(access, permission, False):
            raise MotionAuthorizationError(
                f"Missing permission: {permission}"
            )

    def register_robot(
        self,
        robot: MotionRobot,
        access: MotionAccessContext,
        config: MotionPlannerConfig | None = None,
    ) -> None:
        self._authorize(access, "can_manage")

        if robot.tenant_id != access.tenant_id:
            raise MotionAuthorizationError(
                "Cross-tenant registration forbidden"
            )

        key = (
            robot.tenant_id,
            robot.robot_id,
        )

        with self._lock:
            if key in self._planners:
                raise MotionConflictError(
                    "Robot already registered"
                )

            self._planners[key] = MotionPlanner(
                robot,
                config,
            )

    def get_planner(
        self,
        robot_id: str,
        access: MotionAccessContext,
    ) -> MotionPlanner:
        self._authorize(access, "can_read")

        key = (
            access.tenant_id,
            robot_id,
        )

        with self._lock:
            planner = self._planners.get(key)

        if planner is None:
            raise MotionRobotNotFoundError(
                "Robot motion planner not found"
            )

        return planner

    def plan(
        self,
        robot_id: str,
        request: MotionPlanningRequest,
        access: MotionAccessContext,
    ) -> MotionPlanningResult:
        self._authorize(access, "can_plan")

        planner = self.get_planner(
            robot_id,
            access,
        )

        return planner.plan(request)

    def list_robots(
        self,
        access: MotionAccessContext,
    ) -> tuple[str, ...]:
        self._authorize(access, "can_read")

        with self._lock:
            return tuple(
                robot_id
                for tenant_id, robot_id in self._planners
                if tenant_id == access.tenant_id
            )

    def export_json(
        self,
        result: MotionPlanningResult,
        access: MotionAccessContext,
    ) -> str:
        self._authorize(access, "can_read")

        if result.tenant_id != access.tenant_id:
            raise MotionAuthorizationError(
                "Cross-tenant result access forbidden"
            )

        return json.dumps(
            json_safe(
                {
                    "result": result,
                    "simulation_only": True,
                    "hardware_actuation": False,
                    "collision_model": "spheres_and_link_capsules",
                    "continuous_collision_certified": False,
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

class SyntheticMotionFactory:

    @staticmethod
    def planar_robot(
        robot_id: str = "planar-robot-001",
        tenant_id: str = "physical_ai_lab",
    ) -> MotionRobot:
        joints = (
            MotionJoint(
                name="shoulder",
                joint_type=JointType.REVOLUTE,
                minimum=-math.pi,
                maximum=math.pi,
                maximum_velocity=1.5,
                maximum_acceleration=3.0,
                dh_a_m=0.40,
                link_radius_m=0.015,
            ),
            MotionJoint(
                name="elbow",
                joint_type=JointType.REVOLUTE,
                minimum=-math.pi,
                maximum=math.pi,
                maximum_velocity=1.5,
                maximum_acceleration=3.0,
                dh_a_m=0.30,
                link_radius_m=0.015,
            ),
            MotionJoint(
                name="wrist",
                joint_type=JointType.REVOLUTE,
                minimum=-math.pi,
                maximum=math.pi,
                maximum_velocity=2.0,
                maximum_acceleration=4.0,
                dh_a_m=0.20,
                link_radius_m=0.01,
            ),
        )

        return MotionRobot(
            robot_id=robot_id,
            tenant_id=tenant_id,
            joints=joints,
        )

    @staticmethod
    def access(
        tenant_id: str = "physical_ai_lab",
    ) -> MotionAccessContext:
        return MotionAccessContext(
            tenant_id=tenant_id,
            user_id="robotics_engineer",
            can_read=True,
            can_plan=True,
            can_manage=True,
        )

    @staticmethod
    def sample_obstacles() -> tuple[SphericalObstacle, ...]:
        return (
            SphericalObstacle(
                obstacle_id="workspace-obstacle-001",
                center_xyz=(0.55, 0.10, 0.0),
                radius_m=0.07,
            ),
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestMotionPlanner(unittest.TestCase):

    def setUp(self) -> None:
        self.robot = SyntheticMotionFactory.planar_robot()

        self.config = MotionPlannerConfig(
            maximum_iterations=1500,
            shortcut_iterations=30,
            random_seed=42,
            check_self_collision=False,
        )

        self.planner = MotionPlanner(
            self.robot,
            self.config,
        )

        self.access = SyntheticMotionFactory.access()

    def test_robot_dof(self) -> None:
        self.assertEqual(self.robot.dof, 3)

    def test_home_configuration(self) -> None:
        self.assertTrue(
            np.allclose(
                self.robot.home_configuration(),
                (0.0, 0.0, 0.0),
            )
        )

    def test_forward_kinematics(self) -> None:
        position = (
            self.planner.kinematics.end_effector_position(
                (0.0, 0.0, 0.0)
            )
        )

        self.assertAlmostEqual(
            position[0],
            0.9,
        )

    def test_collision_geometry(self) -> None:
        distance = (
            CollisionGeometry.point_segment_distance(
                np.array([0.5, 1.0, 0.0]),
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
            )
        )

        self.assertAlmostEqual(distance, 1.0)

    def test_segment_distance(self) -> None:
        distance = (
            CollisionGeometry.segment_segment_distance(
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
                np.array([0.5, -1.0, 0.0]),
                np.array([0.5, 1.0, 0.0]),
            )
        )

        self.assertAlmostEqual(
            distance,
            0.0,
            places=6,
        )

    def test_clear_configuration(self) -> None:
        report = (
            self.planner.collision_checker.check_configuration(
                (0.0, 0.0, 0.0)
            )
        )

        self.assertFalse(report.collision)

    def test_obstacle_collision(self) -> None:
        obstacle = SphericalObstacle(
            obstacle_id="test",
            center_xyz=(0.20, 0.0, 0.0),
            radius_m=0.05,
        )

        report = (
            self.planner.collision_checker.check_configuration(
                (0.0, 0.0, 0.0),
                (obstacle,),
            )
        )

        self.assertTrue(report.collision)

        self.assertEqual(
            report.collision_type,
            CollisionType.ENVIRONMENT,
        )

    def test_joint_limit_collision(self) -> None:
        report = (
            self.planner.collision_checker.check_configuration(
                (10.0, 0.0, 0.0)
            )
        )

        self.assertTrue(report.collision)

    def test_direct_motion(self) -> None:
        planner = MotionPlanner(
            self.robot,
            MotionPlannerConfig(
                algorithm=PlannerAlgorithm.DIRECT,
                check_self_collision=False,
            ),
        )

        result = planner.plan(
            MotionPlanningRequest(
                request_id=new_id(),
                start_configuration=(0.0, 0.0, 0.0),
                goal_configuration=(0.2, 0.2, -0.1),
            )
        )

        self.assertEqual(
            result.status,
            PlanningStatus.SUCCESS,
        )

    def test_rrt_motion(self) -> None:
        result = self.planner.plan(
            MotionPlanningRequest(
                request_id=new_id(),
                start_configuration=(0.0, 0.0, 0.0),
                goal_configuration=(0.4, -0.5, 0.2),
            )
        )

        self.assertEqual(
            result.status,
            PlanningStatus.SUCCESS,
        )

    def test_same_start_and_goal(self) -> None:
        result = self.planner.plan(
            MotionPlanningRequest(
                request_id=new_id(),
                start_configuration=(0.0, 0.0, 0.0),
                goal_configuration=(0.0, 0.0, 0.0),
            )
        )

        self.assertEqual(
            result.status,
            PlanningStatus.SUCCESS,
        )

        self.assertEqual(
            len(result.path),
            1,
        )

    def test_invalid_start(self) -> None:
        result = self.planner.plan(
            MotionPlanningRequest(
                request_id=new_id(),
                start_configuration=(10.0, 0.0, 0.0),
                goal_configuration=(0.0, 0.0, 0.0),
            )
        )

        self.assertEqual(
            result.status,
            PlanningStatus.INVALID_START,
        )

    def test_invalid_goal(self) -> None:
        result = self.planner.plan(
            MotionPlanningRequest(
                request_id=new_id(),
                start_configuration=(0.0, 0.0, 0.0),
                goal_configuration=(10.0, 0.0, 0.0),
            )
        )

        self.assertEqual(
            result.status,
            PlanningStatus.INVALID_GOAL,
        )

    def test_quintic_profile(self) -> None:
        generator = self.planner.trajectory_generator

        start = generator.profile(0.0)
        end = generator.profile(1.0)

        self.assertAlmostEqual(start[0], 0.0)
        self.assertAlmostEqual(end[0], 1.0)

        self.assertAlmostEqual(start[1], 0.0)
        self.assertAlmostEqual(end[1], 0.0)

    def test_trajectory_generation(self) -> None:
        path = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.3, -0.2, 0.1]),
        ]

        trajectory = (
            self.planner.trajectory_generator.generate(
                path
            )
        )

        self.assertGreater(
            len(trajectory.points),
            2,
        )

        self.assertGreater(
            trajectory.duration_s,
            0.0,
        )

    def test_trajectory_validation(self) -> None:
        path = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.2, -0.2, 0.1]),
        ]

        trajectory = (
            self.planner.trajectory_generator.generate(
                path
            )
        )

        valid, _ = (
            self.planner.trajectory_validator.validate(
                trajectory,
                (),
            )
        )

        self.assertTrue(valid)

    def test_path_length(self) -> None:
        length = MotionPathOptimizer.path_length(
            [
                np.array([0.0, 0.0]),
                np.array([3.0, 4.0]),
            ]
        )

        self.assertAlmostEqual(
            length,
            5.0,
        )

    def test_service_registration(self) -> None:
        service = MotionPlannerService()

        service.register_robot(
            self.robot,
            self.access,
            self.config,
        )

        self.assertEqual(
            service.list_robots(self.access),
            ("planar-robot-001",),
        )

    def test_service_plan(self) -> None:
        service = MotionPlannerService()

        service.register_robot(
            self.robot,
            self.access,
            self.config,
        )

        result = service.plan(
            self.robot.robot_id,
            MotionPlanningRequest(
                request_id=new_id(),
                start_configuration=(0.0, 0.0, 0.0),
                goal_configuration=(0.2, 0.1, -0.1),
            ),
            self.access,
        )

        self.assertEqual(
            result.status,
            PlanningStatus.SUCCESS,
        )

    def test_tenant_isolation(self) -> None:
        service = MotionPlannerService()

        service.register_robot(
            self.robot,
            self.access,
        )

        other = MotionAccessContext(
            tenant_id="other_tenant",
            user_id="other_user",
            can_read=True,
            can_plan=True,
        )

        with self.assertRaises(
            MotionRobotNotFoundError
        ):
            service.get_planner(
                self.robot.robot_id,
                other,
            )

    def test_permission_denied(self) -> None:
        service = MotionPlannerService()

        service.register_robot(
            self.robot,
            self.access,
        )

        viewer = MotionAccessContext(
            tenant_id="physical_ai_lab",
            user_id="viewer",
            can_read=True,
        )

        with self.assertRaises(
            MotionAuthorizationError
        ):
            service.plan(
                self.robot.robot_id,
                MotionPlanningRequest(
                    request_id=new_id(),
                    start_configuration=(0.0, 0.0, 0.0),
                    goal_configuration=(0.1, 0.1, 0.0),
                ),
                viewer,
            )

    def test_json_export(self) -> None:
        service = MotionPlannerService()

        service.register_robot(
            self.robot,
            self.access,
            self.config,
        )

        result = service.plan(
            self.robot.robot_id,
            MotionPlanningRequest(
                request_id=new_id(),
                start_configuration=(0.0, 0.0, 0.0),
                goal_configuration=(0.1, 0.1, 0.0),
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
            exported["hardware_actuation"]
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("\nPHYSICAL AI — MOTION PLANNER")
    print("=" * 72)

    robot = SyntheticMotionFactory.planar_robot(
        robot_id="physical-ai-arm-001"
    )

    access = SyntheticMotionFactory.access()

    service = MotionPlannerService()

    service.register_robot(
        robot,
        access,
        MotionPlannerConfig(
            algorithm=PlannerAlgorithm.RRT_CONNECT,
            maximum_iterations=3000,
            shortcut_iterations=100,
            check_self_collision=False,
        ),
    )

    request = MotionPlanningRequest(
        request_id=new_id(),
        start_configuration=(0.0, 0.0, 0.0),
        goal_configuration=(0.6, -0.8, 0.4),
        obstacles=(),
        optimize_path=True,
    )

    result = service.plan(
        robot.robot_id,
        request,
        access,
    )

    print("Robot:", robot.robot_id)
    print("Degrees of freedom:", robot.dof)
    print("Status:", result.status.value)
    print("Algorithm:", result.algorithm.value)
    print("Iterations:", result.iterations)
    print("Nodes explored:", result.nodes_explored)
    print("Waypoints:", len(result.path))
    print("Path length:", round(result.path_length, 5))

    if result.trajectory is not None:
        print(
            "Trajectory duration:",
            round(result.trajectory.duration_s, 3),
            "seconds",
        )

        print(
            "Trajectory samples:",
            len(result.trajectory.points),
        )

    print("\nPLANNING RESULT JSON")
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
        TestMotionPlanner
    )

    test_result = unittest.TextTestRunner(
        verbosity=2
    ).run(suite)

    if not test_result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
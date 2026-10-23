"""Manipulation Service module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform
File: backend/app/services/manipulation_service.py

Robotic Manipulation Intelligence Service

Features:
- Robot arm registration
- Forward and inverse kinematics (planar 2-link arms)
- Joint limit validation
- Workspace reachability checks
- Pick-and-place task planning
- Gripper state simulation
- Object grasp validation
- Joint-space trajectory generation
- Joint velocity limit checks
- Obstacle collision checks
- Manipulation task state management
- Simulation execution
- Safety validation
- Evaluation metrics
- JSON reporting

Python 3.11+
Standard library only.

IMPORTANT:
This is a simplified 2D simulation model.
It does not command physical actuators.
It is not a certified robot safety controller.
"""

from __future__ import annotations

import json
import math
import threading
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class ManipulationError(Exception):
    pass


class ManipulationValidationError(ManipulationError):
    pass


class ManipulationNotFoundError(ManipulationError):
    pass


class ManipulationStateError(ManipulationError):
    pass


# ============================================================
# ENUMS
# ============================================================

class GripperState(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    HOLDING = "holding"


class ManipulationTaskType(str, Enum):
    PICK = "pick"
    PLACE = "place"
    PICK_AND_PLACE = "pick_and_place"
    MOVE = "move"


class ManipulationTaskStatus(str, Enum):
    PENDING = "pending"
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ManipulationFailure(str, Enum):
    NONE = "none"
    UNREACHABLE = "unreachable"
    JOINT_LIMIT = "joint_limit"
    COLLISION = "collision"
    GRASP_FAILED = "grasp_failed"
    PAYLOAD_EXCEEDED = "payload_exceeded"
    INVALID_STATE = "invalid_state"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ManipulationValidationError(
            f"{name} must be a finite number"
        )
    return float(value)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def distance(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    return math.hypot(x2 - x1, y2 - y1)


# ============================================================
# GEOMETRY
# ============================================================

@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

    def __post_init__(self) -> None:
        finite(self.x, "x")
        finite(self.y, "y")

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y}


@dataclass(frozen=True)
class CircularObstacle:
    obstacle_id: str
    center: Point2D
    radius_m: float

    def __post_init__(self) -> None:
        if self.radius_m <= 0:
            raise ManipulationValidationError(
                "Obstacle radius must be positive"
            )


@dataclass(frozen=True)
class JointLimits:
    minimum_rad: float
    maximum_rad: float
    maximum_velocity_rad_s: float = 1.0

    def __post_init__(self) -> None:
        finite(self.minimum_rad, "minimum_rad")
        finite(self.maximum_rad, "maximum_rad")
        finite(
            self.maximum_velocity_rad_s,
            "maximum_velocity_rad_s",
        )

        if self.minimum_rad >= self.maximum_rad:
            raise ManipulationValidationError(
                "Invalid joint angle limits"
            )

        if self.maximum_velocity_rad_s <= 0:
            raise ManipulationValidationError(
                "Joint velocity must be positive"
            )

    def contains(self, angle: float) -> bool:
        return self.minimum_rad <= angle <= self.maximum_rad


# ============================================================
# ROBOT ARM CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RobotArmConfig:
    robot_id: str

    link_1_length_m: float = 1.0
    link_2_length_m: float = 1.0

    joint_1_limits: JointLimits = field(
        default_factory=lambda: JointLimits(
            -math.pi,
            math.pi,
        )
    )

    joint_2_limits: JointLimits = field(
        default_factory=lambda: JointLimits(
            -math.pi,
            math.pi,
        )
    )

    maximum_payload_kg: float = 5.0

    gripper_reach_tolerance_m: float = 0.03

    collision_clearance_m: float = 0.03

    allow_hardware_control: bool = False

    def __post_init__(self) -> None:
        if not self.robot_id.strip():
            raise ManipulationValidationError(
                "robot_id is required"
            )

        if (
            self.link_1_length_m <= 0
            or self.link_2_length_m <= 0
        ):
            raise ManipulationValidationError(
                "Link lengths must be positive"
            )

        if self.maximum_payload_kg < 0:
            raise ManipulationValidationError(
                "Payload cannot be negative"
            )

        if self.gripper_reach_tolerance_m < 0:
            raise ManipulationValidationError(
                "Gripper tolerance cannot be negative"
            )

        if self.collision_clearance_m < 0:
            raise ManipulationValidationError(
                "Collision clearance cannot be negative"
            )

        if self.allow_hardware_control:
            raise ManipulationValidationError(
                "Physical hardware control is disabled"
            )


# ============================================================
# ROBOT ARM STATE
# ============================================================

@dataclass
class RobotArmState:
    robot_id: str

    joint_1_rad: float = 0.0
    joint_2_rad: float = 0.0

    gripper_state: GripperState = GripperState.OPEN

    held_object_id: str | None = None

    simulation_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "joint_1_rad": self.joint_1_rad,
            "joint_2_rad": self.joint_2_rad,
            "gripper_state": self.gripper_state.value,
            "held_object_id": self.held_object_id,
            "simulation_time_seconds": (
                self.simulation_time_seconds
            ),
        }


# ============================================================
# MANIPULATION OBJECT
# ============================================================

@dataclass
class ManipulationObject:
    object_id: str
    name: str
    position: Point2D
    mass_kg: float

    held_by_robot_id: str | None = None

    def __post_init__(self) -> None:
        if not self.object_id.strip():
            raise ManipulationValidationError(
                "object_id is required"
            )

        finite(self.mass_kg, "mass_kg")

        if self.mass_kg < 0:
            raise ManipulationValidationError(
                "Object mass cannot be negative"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "name": self.name,
            "position": self.position.to_dict(),
            "mass_kg": self.mass_kg,
            "held_by_robot_id": self.held_by_robot_id,
        }


# ============================================================
# FORWARD KINEMATICS
# ============================================================

class ForwardKinematics2D:

    @staticmethod
    def joint_positions(
        config: RobotArmConfig,
        theta1: float,
        theta2: float,
    ) -> tuple[Point2D, Point2D, Point2D]:
        base = Point2D(0.0, 0.0)

        elbow = Point2D(
            config.link_1_length_m * math.cos(theta1),
            config.link_1_length_m * math.sin(theta1),
        )

        end_effector = Point2D(
            elbow.x
            + config.link_2_length_m
            * math.cos(theta1 + theta2),
            elbow.y
            + config.link_2_length_m
            * math.sin(theta1 + theta2),
        )

        return base, elbow, end_effector

    @staticmethod
    def end_effector(
        config: RobotArmConfig,
        theta1: float,
        theta2: float,
    ) -> Point2D:
        return ForwardKinematics2D.joint_positions(
            config,
            theta1,
            theta2,
        )[2]


# ============================================================
# INVERSE KINEMATICS
# ============================================================

class InverseKinematics2D:

    @staticmethod
    def solve(
        config: RobotArmConfig,
        target: Point2D,
    ) -> list[tuple[float, float]]:
        l1 = config.link_1_length_m
        l2 = config.link_2_length_m

        r2 = target.x**2 + target.y**2

        cos_theta2 = (
            r2 - l1**2 - l2**2
        ) / (2.0 * l1 * l2)

        if cos_theta2 < -1.0 - 1e-9 or cos_theta2 > 1.0 + 1e-9:
            return []

        cos_theta2 = clamp(cos_theta2, -1.0, 1.0)

        solutions = []

        for theta2 in (
            math.acos(cos_theta2),
            -math.acos(cos_theta2),
        ):
            theta1 = (
                math.atan2(target.y, target.x)
                - math.atan2(
                    l2 * math.sin(theta2),
                    l1 + l2 * math.cos(theta2),
                )
            )

            if (
                config.joint_1_limits.contains(theta1)
                and config.joint_2_limits.contains(theta2)
            ):
                candidate = (theta1, theta2)

                if not any(
                    math.isclose(theta1, old[0], abs_tol=1e-9)
                    and math.isclose(theta2, old[1], abs_tol=1e-9)
                    for old in solutions
                ):
                    solutions.append(candidate)

        return solutions


# ============================================================
# COLLISION CHECKING
# ============================================================

class ManipulationCollisionChecker:

    @staticmethod
    def point_segment_distance(
        point: Point2D,
        start: Point2D,
        end: Point2D,
    ) -> float:
        dx = end.x - start.x
        dy = end.y - start.y

        length_squared = dx * dx + dy * dy

        if length_squared <= 1e-12:
            return distance(
                point.x, point.y,
                start.x, start.y,
            )

        t = clamp(
            (
                (point.x - start.x) * dx
                + (point.y - start.y) * dy
            ) / length_squared,
            0.0,
            1.0,
        )

        closest_x = start.x + t * dx
        closest_y = start.y + t * dy

        return distance(
            point.x,
            point.y,
            closest_x,
            closest_y,
        )

    @classmethod
    def configuration_collides(
        cls,
        config: RobotArmConfig,
        theta1: float,
        theta2: float,
        obstacles: Sequence[CircularObstacle],
    ) -> bool:
        base, elbow, tip = (
            ForwardKinematics2D.joint_positions(
                config,
                theta1,
                theta2,
            )
        )

        for obstacle in obstacles:
            minimum_distance = (
                obstacle.radius_m
                + config.collision_clearance_m
            )

            if cls.point_segment_distance(
                obstacle.center,
                base,
                elbow,
            ) <= minimum_distance:
                return True

            if cls.point_segment_distance(
                obstacle.center,
                elbow,
                tip,
            ) <= minimum_distance:
                return True

        return False


# ============================================================
# TRAJECTORY
# ============================================================

@dataclass(frozen=True)
class JointWaypoint:
    time_seconds: float
    joint_1_rad: float
    joint_2_rad: float

    def to_dict(self) -> dict[str, float]:
        return {
            "time_seconds": self.time_seconds,
            "joint_1_rad": self.joint_1_rad,
            "joint_2_rad": self.joint_2_rad,
        }


class JointTrajectoryPlanner:

    @staticmethod
    def interpolate(
        config: RobotArmConfig,
        start: tuple[float, float],
        goal: tuple[float, float],
        *,
        steps: int = 50,
    ) -> list[JointWaypoint]:
        if steps < 2:
            raise ManipulationValidationError(
                "At least two trajectory steps required"
            )

        for index, limits in enumerate(
            (config.joint_1_limits, config.joint_2_limits)
        ):
            if not limits.contains(start[index]):
                raise ManipulationValidationError(
                    "Start joint violates limits"
                )

            if not limits.contains(goal[index]):
                raise ManipulationValidationError(
                    "Goal joint violates limits"
                )

        duration = max(
            abs(goal[0] - start[0])
            / config.joint_1_limits.maximum_velocity_rad_s,
            abs(goal[1] - start[1])
            / config.joint_2_limits.maximum_velocity_rad_s,
            0.1,
        )

        trajectory = []

        for index in range(steps):
            alpha = index / (steps - 1)

            trajectory.append(
                JointWaypoint(
                    time_seconds=alpha * duration,
                    joint_1_rad=(
                        start[0]
                        + alpha * (goal[0] - start[0])
                    ),
                    joint_2_rad=(
                        start[1]
                        + alpha * (goal[1] - start[1])
                    ),
                )
            )

        return trajectory


# ============================================================
# TASK MODEL
# ============================================================

@dataclass
class ManipulationTask:
    task_id: str
    robot_id: str
    task_type: ManipulationTaskType

    target: Point2D
    object_id: str | None = None

    destination: Point2D | None = None

    status: ManipulationTaskStatus = (
        ManipulationTaskStatus.PENDING
    )

    failure_reason: ManipulationFailure = (
        ManipulationFailure.NONE
    )

    trajectories: list[list[JointWaypoint]] = field(
        default_factory=list
    )

    created_at: str = field(default_factory=utc_now)
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "robot_id": self.robot_id,
            "task_type": self.task_type.value,
            "target": self.target.to_dict(),
            "object_id": self.object_id,
            "destination": (
                self.destination.to_dict()
                if self.destination
                else None
            ),
            "status": self.status.value,
            "failure_reason": self.failure_reason.value,
            "trajectories": [
                [
                    waypoint.to_dict()
                    for waypoint in segment
                ]
                for segment in self.trajectories
            ],
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


# ============================================================
# MANIPULATION SERVICE
# ============================================================

class RoboticsManipulationService:

    def __init__(self) -> None:
        self.configurations: dict[
            str, RobotArmConfig
        ] = {}

        self.states: dict[
            str, RobotArmState
        ] = {}

        self.objects: dict[
            str, ManipulationObject
        ] = {}

        self.obstacles: dict[
            str, CircularObstacle
        ] = {}

        self.tasks: dict[
            str, ManipulationTask
        ] = {}

        self._lock = threading.RLock()

    def register_arm(
        self,
        config: RobotArmConfig,
    ) -> RobotArmState:
        with self._lock:
            if config.robot_id in self.configurations:
                raise ManipulationStateError(
                    "Robot arm already registered"
                )

            self.configurations[config.robot_id] = config

            state = RobotArmState(
                robot_id=config.robot_id
            )

            self.states[config.robot_id] = state

            return state

    def register_object(
        self,
        obj: ManipulationObject,
    ) -> None:
        with self._lock:
            if obj.object_id in self.objects:
                raise ManipulationStateError(
                    "Object already registered"
                )

            self.objects[obj.object_id] = obj

    def add_obstacle(
        self,
        obstacle: CircularObstacle,
    ) -> None:
        with self._lock:
            self.obstacles[
                obstacle.obstacle_id
            ] = obstacle

    def get_arm(
        self,
        robot_id: str,
    ) -> tuple[RobotArmConfig, RobotArmState]:
        try:
            return (
                self.configurations[robot_id],
                self.states[robot_id],
            )
        except KeyError as exc:
            raise ManipulationNotFoundError(
                f"Robot arm not found: {robot_id}"
            ) from exc

    def get_end_effector(
        self,
        robot_id: str,
    ) -> Point2D:
        config, state = self.get_arm(robot_id)

        return ForwardKinematics2D.end_effector(
            config,
            state.joint_1_rad,
            state.joint_2_rad,
        )

    def _plan_segment(
        self,
        config: RobotArmConfig,
        start: tuple[float, float],
        target: Point2D,
    ) -> list[JointWaypoint]:
        solutions = InverseKinematics2D.solve(
            config,
            target,
        )

        if not solutions:
            raise ManipulationValidationError(
                "Target is unreachable or violates joint limits"
            )

        obstacles = list(self.obstacles.values())

        candidates = sorted(
            solutions,
            key=lambda solution: (
                abs(solution[0] - start[0])
                + abs(solution[1] - start[1])
            ),
        )

        for solution in candidates:
            trajectory = JointTrajectoryPlanner.interpolate(
                config,
                start,
                solution,
            )

            if all(
                not ManipulationCollisionChecker.configuration_collides(
                    config,
                    waypoint.joint_1_rad,
                    waypoint.joint_2_rad,
                    obstacles,
                )
                for waypoint in trajectory
            ):
                return trajectory

        raise ManipulationValidationError(
            "No collision-free interpolated trajectory found"
        )

    def create_task(
        self,
        *,
        robot_id: str,
        task_type: ManipulationTaskType,
        target: Point2D,
        object_id: str | None = None,
        destination: Point2D | None = None,
    ) -> ManipulationTask:
        with self._lock:
            self.get_arm(robot_id)

            task_type = ManipulationTaskType(task_type)

            if task_type in (
                ManipulationTaskType.PICK,
                ManipulationTaskType.PLACE,
                ManipulationTaskType.PICK_AND_PLACE,
            ) and not object_id:
                raise ManipulationValidationError(
                    "Object ID is required"
                )

            if (
                task_type == ManipulationTaskType.PICK_AND_PLACE
                and destination is None
            ):
                raise ManipulationValidationError(
                    "Destination is required"
                )

            if object_id and object_id not in self.objects:
                raise ManipulationNotFoundError(
                    f"Object not found: {object_id}"
                )

            task = ManipulationTask(
                task_id=str(uuid.uuid4()),
                robot_id=robot_id,
                task_type=task_type,
                target=target,
                object_id=object_id,
                destination=destination,
            )

            self.tasks[task.task_id] = task

            return task

    def plan_task(
        self,
        task_id: str,
    ) -> ManipulationTask:
        with self._lock:
            task = self.tasks.get(task_id)

            if task is None:
                raise ManipulationNotFoundError(
                    "Task not found"
                )

            if task.status != ManipulationTaskStatus.PENDING:
                raise ManipulationStateError(
                    "Only pending tasks can be planned"
                )

            config, state = self.get_arm(task.robot_id)

            current = (
                state.joint_1_rad,
                state.joint_2_rad,
            )

            first_segment = self._plan_segment(
                config,
                current,
                task.target,
            )

            trajectories = [first_segment]

            if task.task_type == ManipulationTaskType.PICK_AND_PLACE:
                assert task.destination is not None

                last = first_segment[-1]

                second_segment = self._plan_segment(
                    config,
                    (
                        last.joint_1_rad,
                        last.joint_2_rad,
                    ),
                    task.destination,
                )

                trajectories.append(second_segment)

            task.trajectories = trajectories
            task.status = ManipulationTaskStatus.PLANNED

            return task

    def _simulate_grasp(
        self,
        task: ManipulationTask,
        config: RobotArmConfig,
        state: RobotArmState,
    ) -> None:
        if task.object_id is None:
            raise ManipulationStateError(
                "No object specified"
            )

        obj = self.objects[task.object_id]

        if obj.held_by_robot_id is not None:
            raise ManipulationStateError(
                "Object is already held"
            )

        if state.held_object_id is not None:
            raise ManipulationStateError(
                "Gripper is already holding an object"
            )

        if obj.mass_kg > config.maximum_payload_kg:
            raise ManipulationValidationError(
                "Object exceeds maximum payload"
            )

        tip = ForwardKinematics2D.end_effector(
            config,
            state.joint_1_rad,
            state.joint_2_rad,
        )

        if distance(
            tip.x,
            tip.y,
            obj.position.x,
            obj.position.y,
        ) > config.gripper_reach_tolerance_m:
            raise ManipulationValidationError(
                "Object is outside grasp tolerance"
            )

        state.gripper_state = GripperState.HOLDING
        state.held_object_id = obj.object_id
        obj.held_by_robot_id = task.robot_id

    def _simulate_release(
        self,
        task: ManipulationTask,
        config: RobotArmConfig,
        state: RobotArmState,
    ) -> None:
        if (
            task.object_id is None
            or state.held_object_id != task.object_id
        ):
            raise ManipulationStateError(
                "Requested object is not held"
            )

        obj = self.objects[task.object_id]

        tip = ForwardKinematics2D.end_effector(
            config,
            state.joint_1_rad,
            state.joint_2_rad,
        )

        obj.position = tip
        obj.held_by_robot_id = None

        state.held_object_id = None
        state.gripper_state = GripperState.OPEN

    def execute_simulation(
        self,
        task_id: str,
    ) -> ManipulationTask:
        with self._lock:
            task = self.tasks.get(task_id)

            if task is None:
                raise ManipulationNotFoundError(
                    "Task not found"
                )

            if task.status != ManipulationTaskStatus.PLANNED:
                raise ManipulationStateError(
                    "Task must be planned first"
                )

            config, state = self.get_arm(task.robot_id)

            task.status = ManipulationTaskStatus.RUNNING

            try:
                for segment_index, segment in enumerate(
                    task.trajectories
                ):
                    for waypoint in segment:
                        if ManipulationCollisionChecker.configuration_collides(
                            config,
                            waypoint.joint_1_rad,
                            waypoint.joint_2_rad,
                            list(self.obstacles.values()),
                        ):
                            raise ManipulationValidationError(
                                "Collision detected during simulation"
                            )

                        state.joint_1_rad = waypoint.joint_1_rad
                        state.joint_2_rad = waypoint.joint_2_rad

                        if state.held_object_id is not None:
                            held = self.objects[
                                state.held_object_id
                            ]

                            held.position = (
                                ForwardKinematics2D.end_effector(
                                    config,
                                    state.joint_1_rad,
                                    state.joint_2_rad,
                                )
                            )

                    state.simulation_time_seconds += (
                        segment[-1].time_seconds
                    )

                    if (
                        segment_index == 0
                        and task.task_type in (
                            ManipulationTaskType.PICK,
                            ManipulationTaskType.PICK_AND_PLACE,
                        )
                    ):
                        self._simulate_grasp(
                            task,
                            config,
                            state,
                        )

                if task.task_type in (
                    ManipulationTaskType.PLACE,
                    ManipulationTaskType.PICK_AND_PLACE,
                ):
                    self._simulate_release(
                        task,
                        config,
                        state,
                    )

                task.status = (
                    ManipulationTaskStatus.COMPLETED
                )

            except ManipulationError as exc:
                task.status = ManipulationTaskStatus.FAILED

                if "payload" in str(exc).lower():
                    task.failure_reason = (
                        ManipulationFailure.PAYLOAD_EXCEEDED
                    )
                elif "collision" in str(exc).lower():
                    task.failure_reason = (
                        ManipulationFailure.COLLISION
                    )
                elif "grasp" in str(exc).lower():
                    task.failure_reason = (
                        ManipulationFailure.GRASP_FAILED
                    )
                else:
                    task.failure_reason = (
                        ManipulationFailure.INVALID_STATE
                    )

            task.completed_at = utc_now()

            return task

    def statistics(self) -> dict[str, Any]:
        tasks = list(self.tasks.values())

        completed = sum(
            task.status == ManipulationTaskStatus.COMPLETED
            for task in tasks
        )

        failed = sum(
            task.status == ManipulationTaskStatus.FAILED
            for task in tasks
        )

        return {
            "registered_arms": len(self.configurations),
            "registered_objects": len(self.objects),
            "registered_obstacles": len(self.obstacles),
            "total_tasks": len(tasks),
            "completed_tasks": completed,
            "failed_tasks": failed,
            "task_success_rate": (
                completed / (completed + failed)
                if completed + failed
                else None
            ),
            "simulation_only": True,
        }

    def export_json(
        self,
        path: str | Path,
    ) -> Path:
        output = Path(path)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "generated_at": utc_now(),
            "statistics": self.statistics(),
            "robots": {
                robot_id: state.to_dict()
                for robot_id, state in self.states.items()
            },
            "objects": [
                obj.to_dict()
                for obj in self.objects.values()
            ],
            "tasks": [
                task.to_dict()
                for task in self.tasks.values()
            ],
        }

        output.write_text(
            json.dumps(
                payload,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        return output


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    service = RoboticsManipulationService()

    config = RobotArmConfig(
        robot_id="robot_arm_001",
        link_1_length_m=1.0,
        link_2_length_m=1.0,
        maximum_payload_kg=3.0,
    )

    service.register_arm(config)

    object_position = Point2D(1.2, 0.6)

    service.register_object(
        ManipulationObject(
            object_id="part_001",
            name="Motor Housing",
            position=object_position,
            mass_kg=1.5,
        )
    )

    task = service.create_task(
        robot_id="robot_arm_001",
        task_type=ManipulationTaskType.PICK_AND_PLACE,
        target=object_position,
        object_id="part_001",
        destination=Point2D(0.5, 1.4),
    )

    service.plan_task(task.task_id)

    result = service.execute_simulation(
        task.task_id
    )

    print(
        json.dumps(
            {
                "task": result.to_dict(),
                "statistics": service.statistics(),
            },
            indent=2,
        )
    )

    service.export_json(
        "data/manipulation/manipulation_report.json"
    )


if __name__ == "__main__":
    main()

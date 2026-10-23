"""Task Execution module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/manipulation/task_execution.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================
Module: Task Execution

Simulation-only robotic manipulation task execution engine.

Features:
    - Task creation and validation
    - Sequential manipulation workflows
    - Pick-and-place task execution
    - Grasp and release simulation
    - Cartesian movement simulation
    - Object attachment and tracking
    - Task dependencies
    - Execution state machine
    - Step-by-step execution
    - Pause, resume, and cancellation
    - Retry policies
    - Failure handling and recovery
    - Execution timeouts
    - Safety precondition checks
    - Execution monitoring
    - Task history and audit events
    - Multi-robot task coordination
    - Multi-tenant authorization
    - JSON reporting
    - Synthetic demonstrations
    - Unit tests

Requirements:
    Python >= 3.11
    numpy >= 1.24

IMPORTANT:
    Simulation and advisory only.
    No physical robot actuation is performed.

    Cartesian reach checks are approximate.
    This module does not replace robot-specific inverse
    kinematics, trajectory planning, collision detection,
    dynamics validation, or certified safety systems.
"""

from __future__ import annotations

import json
import math
import threading
import unittest
import uuid

from collections import deque
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import numpy as np


# ============================================================
# EXCEPTIONS
# ============================================================

class TaskExecutionError(Exception):
    pass


class TaskValidationError(TaskExecutionError):
    pass


class TaskAuthorizationError(TaskExecutionError):
    pass


class TaskNotFoundError(TaskExecutionError):
    pass


class TaskConflictError(TaskExecutionError):
    pass


class TaskStateError(TaskExecutionError):
    pass


class TaskPlanningError(TaskExecutionError):
    pass


class TaskTimeoutError(TaskExecutionError):
    pass


class TaskSafetyError(TaskExecutionError):
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
        raise TaskValidationError(
            f"Invalid {name}"
        )

    return value.strip()


def finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TaskValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TaskValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise TaskValidationError(
            f"{name} must be finite"
        )

    return result


def positive(value: Any, name: str) -> float:
    result = finite(value, name)

    if result <= 0:
        raise TaskValidationError(
            f"{name} must be positive"
        )

    return result


def nonnegative(value: Any, name: str) -> float:
    result = finite(value, name)

    if result < 0:
        raise TaskValidationError(
            f"{name} cannot be negative"
        )

    return result


def vector3(
    values: Sequence[float],
    name: str,
) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TaskValidationError(
            f"{name} must contain numeric values"
        ) from exc

    if result.shape != (3,):
        raise TaskValidationError(
            f"{name} must contain three values"
        )

    if not np.all(np.isfinite(result)):
        raise TaskValidationError(
            f"{name} contains non-finite values"
        )

    return result


def tuple3(
    values: Sequence[float],
) -> tuple[float, float, float]:
    result = vector3(values, "vector")

    return (
        float(result[0]),
        float(result[1]),
        float(result[2]),
    )


def distance3(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    return float(
        np.linalg.norm(
            vector3(first, "first")
            - vector3(second, "second")
        )
    )


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
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

    if isinstance(value, (tuple, list, deque)):
        return [json_safe(item) for item in value]

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


# ============================================================
# ENUMERATIONS
# ============================================================

class TaskType(str, Enum):
    PICK_AND_PLACE = "pick_and_place"
    OBJECT_TRANSFER = "object_transfer"
    GRASP_OBJECT = "grasp_object"
    RELEASE_OBJECT = "release_object"
    MOVE_TO_POSE = "move_to_pose"
    INSPECTION = "inspection"
    CUSTOM_WORKFLOW = "custom_workflow"


class TaskStatus(str, Enum):
    CREATED = "created"
    VALIDATED = "validated"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ActionType(str, Enum):
    MOVE = "move"
    OPEN_GRIPPER = "open_gripper"
    CLOSE_GRIPPER = "close_gripper"
    ATTACH_OBJECT = "attach_object"
    DETACH_OBJECT = "detach_object"
    VERIFY_GRASP = "verify_grasp"
    VERIFY_PLACEMENT = "verify_placement"
    INSPECT_OBJECT = "inspect_object"
    WAIT = "wait"


class GripperState(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    HOLDING = "holding"


class ObjectState(str, Enum):
    AVAILABLE = "available"
    ATTACHED = "attached"
    PLACED = "placed"


class FailureReason(str, Enum):
    NONE = "none"
    INVALID_TASK = "invalid_task"
    INVALID_TRANSITION = "invalid_transition"
    ROBOT_UNAVAILABLE = "robot_unavailable"
    OBJECT_NOT_FOUND = "object_not_found"
    OBJECT_UNAVAILABLE = "object_unavailable"
    OBJECT_NOT_ATTACHED = "object_not_attached"
    PAYLOAD_EXCEEDED = "payload_exceeded"
    UNREACHABLE_TARGET = "unreachable_target"
    COLLISION_RISK = "collision_risk"
    GRASP_FAILURE = "grasp_failure"
    PLACEMENT_FAILURE = "placement_failure"
    DEPENDENCY_FAILURE = "dependency_failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    SIMULATION_FAILURE = "simulation_failure"


class EventSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# ============================================================
# GEOMETRY
# ============================================================

@dataclass(frozen=True)
class Pose3D:
    position_xyz: tuple[float, float, float]

    orientation_rpy: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )

    def __post_init__(self) -> None:
        vector3(
            self.position_xyz,
            "position_xyz",
        )

        vector3(
            self.orientation_rpy,
            "orientation_rpy",
        )

    def translated(
        self,
        offset_xyz: Sequence[float],
    ) -> Pose3D:
        result = (
            vector3(self.position_xyz, "position")
            + vector3(offset_xyz, "offset")
        )

        return Pose3D(
            position_xyz=tuple3(result),
            orientation_rpy=self.orientation_rpy,
        )


@dataclass(frozen=True)
class WorkspaceBounds:
    minimum_xyz: tuple[float, float, float]
    maximum_xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        minimum = vector3(
            self.minimum_xyz,
            "minimum_xyz",
        )

        maximum = vector3(
            self.maximum_xyz,
            "maximum_xyz",
        )

        if np.any(minimum >= maximum):
            raise TaskValidationError(
                "Invalid workspace bounds"
            )

    def contains(
        self,
        position: Sequence[float],
    ) -> bool:
        point = vector3(position, "position")

        minimum = vector3(
            self.minimum_xyz,
            "minimum_xyz",
        )

        maximum = vector3(
            self.maximum_xyz,
            "maximum_xyz",
        )

        return bool(
            np.all(point >= minimum)
            and np.all(point <= maximum)
        )


@dataclass(frozen=True)
class SphericalObstacle:
    obstacle_id: str

    center_xyz: tuple[float, float, float]

    radius_m: float

    def __post_init__(self) -> None:
        identifier(
            self.obstacle_id,
            "obstacle_id",
        )

        vector3(
            self.center_xyz,
            "center_xyz",
        )

        positive(
            self.radius_m,
            "radius_m",
        )


# ============================================================
# ROBOT AND OBJECT MODELS
# ============================================================

@dataclass(frozen=True)
class RobotSpecification:
    robot_id: str
    tenant_id: str

    home_pose: Pose3D

    workspace: WorkspaceBounds

    maximum_reach_m: float = 1.5
    maximum_payload_kg: float = 5.0

    maximum_speed_mps: float = 0.25

    gripper_maximum_force_n: float = 100.0

    def __post_init__(self) -> None:
        identifier(self.robot_id, "robot_id")
        identifier(self.tenant_id, "tenant_id")

        positive(
            self.maximum_reach_m,
            "maximum_reach_m",
        )

        positive(
            self.maximum_payload_kg,
            "maximum_payload_kg",
        )

        positive(
            self.maximum_speed_mps,
            "maximum_speed_mps",
        )

        positive(
            self.gripper_maximum_force_n,
            "gripper_maximum_force_n",
        )

        if not self.workspace.contains(
            self.home_pose.position_xyz
        ):
            raise TaskValidationError(
                "Robot home pose outside workspace"
            )


@dataclass(frozen=True)
class ManipulationObject:
    object_id: str
    tenant_id: str

    name: str

    pose: Pose3D

    mass_kg: float

    grasp_confidence: float = 0.95

    state: ObjectState = ObjectState.AVAILABLE

    def __post_init__(self) -> None:
        identifier(self.object_id, "object_id")
        identifier(self.tenant_id, "tenant_id")
        identifier(self.name, "name")

        positive(
            self.mass_kg,
            "mass_kg",
        )

        confidence = finite(
            self.grasp_confidence,
            "grasp_confidence",
        )

        if not 0 <= confidence <= 1:
            raise TaskValidationError(
                "Grasp confidence must be in [0,1]"
            )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class TaskExecutionConfig:
    default_timeout_s: float = 120.0

    maximum_retries: int = 2

    retry_delay_s: float = 0.5

    minimum_grasp_confidence: float = 0.65

    payload_safety_factor: float = 1.5

    collision_margin_m: float = 0.03

    collision_resolution_m: float = 0.02

    placement_tolerance_m: float = 0.015

    approach_height_m: float = 0.12

    lift_height_m: float = 0.18

    maximum_history_events: int = 1000

    def __post_init__(self) -> None:
        for name in (
            "default_timeout_s",
            "payload_safety_factor",
            "collision_resolution_m",
            "placement_tolerance_m",
            "approach_height_m",
            "lift_height_m",
        ):
            positive(
                getattr(self, name),
                name,
            )

        nonnegative(
            self.retry_delay_s,
            "retry_delay_s",
        )

        nonnegative(
            self.collision_margin_m,
            "collision_margin_m",
        )

        if self.maximum_retries < 0:
            raise TaskValidationError(
                "maximum_retries cannot be negative"
            )

        if self.maximum_history_events < 1:
            raise TaskValidationError(
                "maximum_history_events must be positive"
            )

        if not 0 <= self.minimum_grasp_confidence <= 1:
            raise TaskValidationError(
                "Invalid grasp confidence threshold"
            )


# ============================================================
# TASK DEFINITIONS
# ============================================================

@dataclass(frozen=True)
class TaskStep:
    step_id: str

    action: ActionType

    target_pose: Pose3D | None = None

    object_id: str | None = None

    duration_s: float = 0.0

    depends_on: tuple[str, ...] = ()

    retryable: bool = True

    def __post_init__(self) -> None:
        identifier(self.step_id, "step_id")

        nonnegative(
            self.duration_s,
            "duration_s",
        )

        if self.object_id is not None:
            identifier(
                self.object_id,
                "object_id",
            )

        for dependency in self.depends_on:
            identifier(
                dependency,
                "dependency",
            )

        if self.step_id in self.depends_on:
            raise TaskValidationError(
                "Step cannot depend on itself"
            )

        if self.action == ActionType.MOVE:
            if self.target_pose is None:
                raise TaskValidationError(
                    "MOVE requires target_pose"
                )

        if self.action in (
            ActionType.ATTACH_OBJECT,
            ActionType.DETACH_OBJECT,
            ActionType.VERIFY_GRASP,
            ActionType.VERIFY_PLACEMENT,
            ActionType.INSPECT_OBJECT,
        ):
            if self.object_id is None:
                raise TaskValidationError(
                    f"{self.action.value} requires object_id"
                )


@dataclass(frozen=True)
class ManipulationTask:
    task_id: str
    tenant_id: str

    robot_id: str

    task_type: TaskType

    name: str

    steps: tuple[TaskStep, ...]

    timeout_s: float = 120.0

    priority: int = 5

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        identifier(self.task_id, "task_id")
        identifier(self.tenant_id, "tenant_id")
        identifier(self.robot_id, "robot_id")
        identifier(self.name, "name")

        positive(
            self.timeout_s,
            "timeout_s",
        )

        if not 0 <= self.priority <= 10:
            raise TaskValidationError(
                "Priority must be in [0,10]"
            )

        if not self.steps:
            raise TaskValidationError(
                "Task must contain at least one step"
            )


@dataclass(frozen=True)
class StepExecutionResult:
    step_id: str

    action: ActionType

    status: StepStatus

    attempts: int

    duration_s: float

    failure_reason: FailureReason

    message: str


@dataclass(frozen=True)
class TaskExecutionEvent:
    event_id: str
    task_id: str

    step_id: str | None

    status: TaskStatus

    severity: EventSeverity

    message: str

    timestamp: datetime


@dataclass(frozen=True)
class TaskExecutionReport:
    report_id: str

    task_id: str
    tenant_id: str
    robot_id: str

    status: TaskStatus

    failure_reason: FailureReason

    completed_steps: int
    total_steps: int

    progress_percent: float

    simulated_duration_s: float

    final_robot_pose: Pose3D

    final_gripper_state: GripperState

    attached_object_id: str | None

    step_results: tuple[StepExecutionResult, ...]

    events: tuple[TaskExecutionEvent, ...]

    created_at: datetime


@dataclass(frozen=True)
class TaskAccessContext:
    tenant_id: str
    user_id: str

    can_read: bool = False
    can_create: bool = False
    can_execute: bool = False
    can_manage: bool = False


# ============================================================
# TASK VALIDATOR
# ============================================================

class TaskValidator:

    @staticmethod
    def validate(
        task: ManipulationTask,
    ) -> tuple[TaskStep, ...]:
        steps_by_id: dict[str, TaskStep] = {}

        for step in task.steps:
            if step.step_id in steps_by_id:
                raise TaskValidationError(
                    f"Duplicate step ID: {step.step_id}"
                )

            steps_by_id[step.step_id] = step

        for step in task.steps:
            for dependency in step.depends_on:
                if dependency not in steps_by_id:
                    raise TaskValidationError(
                        f"Unknown dependency: {dependency}"
                    )

        indegree = {
            step.step_id: len(step.depends_on)
            for step in task.steps
        }

        children: dict[str, list[str]] = {
            step.step_id: []
            for step in task.steps
        }

        for step in task.steps:
            for dependency in step.depends_on:
                children[dependency].append(
                    step.step_id
                )

        queue = deque(
            step.step_id
            for step in task.steps
            if indegree[step.step_id] == 0
        )

        ordered: list[TaskStep] = []

        while queue:
            step_id = queue.popleft()

            ordered.append(
                steps_by_id[step_id]
            )

            for child in children[step_id]:
                indegree[child] -= 1

                if indegree[child] == 0:
                    queue.append(child)

        if len(ordered) != len(task.steps):
            raise TaskValidationError(
                "Task dependencies contain a cycle"
            )

        return tuple(ordered)


# ============================================================
# COLLISION CHECKING
# ============================================================

class TaskCollisionChecker:

    def __init__(
        self,
        config: TaskExecutionConfig,
    ) -> None:
        self.config = config

    def segment_collision(
        self,
        start: Sequence[float],
        end: Sequence[float],
        obstacles: Sequence[SphericalObstacle],
    ) -> bool:
        first = vector3(start, "start")
        second = vector3(end, "end")

        distance = float(
            np.linalg.norm(second - first)
        )

        samples = max(
            1,
            math.ceil(
                distance
                / self.config.collision_resolution_m
            ),
        )

        for index in range(samples + 1):
            alpha = index / samples

            point = (
                first * (1.0 - alpha)
                + second * alpha
            )

            for obstacle in obstacles:
                center = vector3(
                    obstacle.center_xyz,
                    "obstacle center",
                )

                minimum_distance = (
                    obstacle.radius_m
                    + self.config.collision_margin_m
                )

                if (
                    np.linalg.norm(point - center)
                    <= minimum_distance
                ):
                    return True

        return False


# ============================================================
# SIMULATION WORLD
# ============================================================

@dataclass
class RobotRuntimeState:
    robot_id: str

    pose: Pose3D

    gripper_state: GripperState = GripperState.OPEN

    attached_object_id: str | None = None

    simulated_time_s: float = 0.0


@dataclass
class SimulationWorld:
    robot: RobotSpecification

    objects: dict[str, ManipulationObject]

    obstacles: tuple[SphericalObstacle, ...] = ()

    runtime: RobotRuntimeState = field(init=False)

    def __post_init__(self) -> None:
        self.runtime = RobotRuntimeState(
            robot_id=self.robot.robot_id,
            pose=self.robot.home_pose,
        )

        for object_id, obj in self.objects.items():
            if object_id != obj.object_id:
                raise TaskValidationError(
                    "Object dictionary key mismatch"
                )

            if obj.tenant_id != self.robot.tenant_id:
                raise TaskAuthorizationError(
                    "Cross-tenant object in simulation world"
                )


# ============================================================
# SAFETY VALIDATOR
# ============================================================

class TaskSafetyValidator:

    def __init__(
        self,
        config: TaskExecutionConfig,
    ) -> None:
        self.config = config

        self.collision_checker = (
            TaskCollisionChecker(config)
        )

    def validate_move(
        self,
        world: SimulationWorld,
        target_pose: Pose3D,
    ) -> None:
        robot = world.robot

        if not robot.workspace.contains(
            target_pose.position_xyz
        ):
            raise TaskSafetyError(
                "Target outside robot workspace"
            )

        if (
            distance3(
                robot.home_pose.position_xyz,
                target_pose.position_xyz,
            )
            > robot.maximum_reach_m
        ):
            raise TaskSafetyError(
                "Target exceeds approximate robot reach"
            )

        if self.collision_checker.segment_collision(
            world.runtime.pose.position_xyz,
            target_pose.position_xyz,
            world.obstacles,
        ):
            raise TaskSafetyError(
                "Approximate Cartesian collision detected"
            )

    def validate_grasp(
        self,
        world: SimulationWorld,
        obj: ManipulationObject,
    ) -> None:
        if obj.state != ObjectState.AVAILABLE:
            raise TaskSafetyError(
                "Object is not available"
            )

        if (
            obj.grasp_confidence
            < self.config.minimum_grasp_confidence
        ):
            raise TaskSafetyError(
                "Object grasp confidence below threshold"
            )

        if (
            obj.mass_kg
            * self.config.payload_safety_factor
            > world.robot.maximum_payload_kg
        ):
            raise TaskSafetyError(
                "Object exceeds payload safety margin"
            )


# ============================================================
# ACTION EXECUTOR
# ============================================================

class SimulatedActionExecutor:

    def __init__(
        self,
        config: TaskExecutionConfig,
    ) -> None:
        self.config = config

        self.safety = TaskSafetyValidator(
            config
        )

    def execute(
        self,
        step: TaskStep,
        world: SimulationWorld,
    ) -> float:
        action = step.action

        runtime = world.runtime

        if action == ActionType.MOVE:
            assert step.target_pose is not None

            self.safety.validate_move(
                world,
                step.target_pose,
            )

            distance = distance3(
                runtime.pose.position_xyz,
                step.target_pose.position_xyz,
            )

            duration = (
                distance
                / world.robot.maximum_speed_mps
            )

            previous_pose = runtime.pose

            runtime.pose = step.target_pose

            if runtime.attached_object_id is not None:
                object_id = runtime.attached_object_id

                obj = world.objects[object_id]

                displacement = (
                    vector3(
                        runtime.pose.position_xyz,
                        "new position",
                    )
                    - vector3(
                        previous_pose.position_xyz,
                        "previous position",
                    )
                )

                world.objects[object_id] = replace(
                    obj,
                    pose=obj.pose.translated(
                        displacement
                    ),
                )

        elif action == ActionType.OPEN_GRIPPER:
            if runtime.attached_object_id is not None:
                raise TaskStateError(
                    "Cannot open gripper while object attached; "
                    "detach object first"
                )

            runtime.gripper_state = GripperState.OPEN
            duration = 0.2

        elif action == ActionType.CLOSE_GRIPPER:
            if runtime.attached_object_id is not None:
                raise TaskStateError(
                    "Gripper already holds an object"
                )

            runtime.gripper_state = GripperState.CLOSED
            duration = 0.2

        elif action == ActionType.ATTACH_OBJECT:
            assert step.object_id is not None

            obj = world.objects.get(
                step.object_id
            )

            if obj is None:
                raise TaskNotFoundError(
                    "Object not found"
                )

            if runtime.attached_object_id is not None:
                raise TaskStateError(
                    "Another object is already attached"
                )

            if runtime.gripper_state != GripperState.CLOSED:
                raise TaskStateError(
                    "Gripper must be closed before attachment"
                )

            self.safety.validate_grasp(
                world,
                obj,
            )

            if (
                distance3(
                    runtime.pose.position_xyz,
                    obj.pose.position_xyz,
                )
                > 0.15
            ):
                raise TaskSafetyError(
                    "Gripper too far from object"
                )

            runtime.attached_object_id = obj.object_id

            runtime.gripper_state = GripperState.HOLDING

            world.objects[obj.object_id] = replace(
                obj,
                state=ObjectState.ATTACHED,
            )

            duration = 0.1

        elif action == ActionType.DETACH_OBJECT:
            assert step.object_id is not None

            if (
                runtime.attached_object_id
                != step.object_id
            ):
                raise TaskStateError(
                    "Requested object is not attached"
                )

            obj = world.objects[step.object_id]

            world.objects[step.object_id] = replace(
                obj,
                state=ObjectState.PLACED,
            )

            runtime.attached_object_id = None

            runtime.gripper_state = GripperState.OPEN

            duration = 0.2

        elif action == ActionType.VERIFY_GRASP:
            assert step.object_id is not None

            if (
                runtime.attached_object_id
                != step.object_id
            ):
                raise TaskStateError(
                    "Grasp verification failed"
                )

            if runtime.gripper_state != GripperState.HOLDING:
                raise TaskStateError(
                    "Gripper is not holding the object"
                )

            duration = 0.1

        elif action == ActionType.VERIFY_PLACEMENT:
            assert step.object_id is not None

            obj = world.objects.get(
                step.object_id
            )

            if obj is None:
                raise TaskNotFoundError(
                    "Object not found"
                )

            if obj.state != ObjectState.PLACED:
                raise TaskStateError(
                    "Object has not been placed"
                )

            if step.target_pose is not None:
                error = distance3(
                    obj.pose.position_xyz,
                    step.target_pose.position_xyz,
                )

                if (
                    error
                    > self.config.placement_tolerance_m
                ):
                    raise TaskStateError(
                        "Placement error exceeds tolerance"
                    )

            duration = 0.1

        elif action == ActionType.INSPECT_OBJECT:
            assert step.object_id is not None

            if step.object_id not in world.objects:
                raise TaskNotFoundError(
                    "Object not found"
                )

            duration = max(
                step.duration_s,
                0.2,
            )

        elif action == ActionType.WAIT:
            duration = step.duration_s

        else:
            raise TaskValidationError(
                f"Unsupported action: {action}"
            )

        runtime.simulated_time_s += duration

        return duration


# ============================================================
# EXECUTION STATE MACHINE
# ============================================================

class TaskExecutionStateMachine:

    TRANSITIONS = {
        TaskStatus.CREATED: {
            TaskStatus.VALIDATED,
            TaskStatus.CANCELLED,
        },
        TaskStatus.VALIDATED: {
            TaskStatus.READY,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        },
        TaskStatus.READY: {
            TaskStatus.RUNNING,
            TaskStatus.CANCELLED,
        },
        TaskStatus.RUNNING: {
            TaskStatus.PAUSED,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMED_OUT,
        },
        TaskStatus.PAUSED: {
            TaskStatus.RUNNING,
            TaskStatus.CANCELLED,
            TaskStatus.FAILED,
            TaskStatus.TIMED_OUT,
        },
        TaskStatus.COMPLETED: set(),
        TaskStatus.FAILED: set(),
        TaskStatus.CANCELLED: set(),
        TaskStatus.TIMED_OUT: set(),
    }

    def __init__(self) -> None:
        self.status = TaskStatus.CREATED

    def transition(
        self,
        new_status: TaskStatus,
    ) -> None:
        if (
            new_status
            not in self.TRANSITIONS[self.status]
        ):
            raise TaskStateError(
                f"Invalid task transition: "
                f"{self.status.value} -> "
                f"{new_status.value}"
            )

        self.status = new_status


# ============================================================
# EXECUTION SESSION
# ============================================================

class TaskExecutionSession:

    def __init__(
        self,
        task: ManipulationTask,
        world: SimulationWorld,
        config: TaskExecutionConfig,
    ) -> None:
        if task.robot_id != world.robot.robot_id:
            raise TaskValidationError(
                "Task robot mismatch"
            )

        if task.tenant_id != world.robot.tenant_id:
            raise TaskAuthorizationError(
                "Task tenant mismatch"
            )

        self.task = task
        self.world = world
        self.config = config

        self.ordered_steps = TaskValidator.validate(
            task
        )

        self.machine = TaskExecutionStateMachine()

        self.executor = SimulatedActionExecutor(
            config
        )

        self.step_results: list[
            StepExecutionResult
        ] = []

        self.events: deque[
            TaskExecutionEvent
        ] = deque(
            maxlen=config.maximum_history_events
        )

        self.next_step_index = 0

        self.failure_reason = FailureReason.NONE

        self._lock = threading.RLock()

        self.machine.transition(
            TaskStatus.VALIDATED
        )

        self.machine.transition(
            TaskStatus.READY
        )

        self._event(
            None,
            EventSeverity.INFO,
            "Task validated and ready",
        )

    def _event(
        self,
        step_id: str | None,
        severity: EventSeverity,
        message: str,
    ) -> None:
        self.events.append(
            TaskExecutionEvent(
                event_id=new_id(),
                task_id=self.task.task_id,
                step_id=step_id,
                status=self.machine.status,
                severity=severity,
                message=message,
                timestamp=utc_now(),
            )
        )

    def start(self) -> None:
        with self._lock:
            if self.machine.status != TaskStatus.READY:
                raise TaskStateError(
                    "Task is not ready"
                )

            self.machine.transition(
                TaskStatus.RUNNING
            )

            self._event(
                None,
                EventSeverity.INFO,
                "Task execution started",
            )

    def pause(self) -> None:
        with self._lock:
            self.machine.transition(
                TaskStatus.PAUSED
            )

            self._event(
                None,
                EventSeverity.INFO,
                "Task execution paused",
            )

    def resume(self) -> None:
        with self._lock:
            self.machine.transition(
                TaskStatus.RUNNING
            )

            self._event(
                None,
                EventSeverity.INFO,
                "Task execution resumed",
            )

    def cancel(self) -> None:
        with self._lock:
            self.machine.transition(
                TaskStatus.CANCELLED
            )

            self.failure_reason = FailureReason.CANCELLED

            self._event(
                None,
                EventSeverity.WARNING,
                "Task execution cancelled",
            )

    def _failure_reason(
        self,
        exc: Exception,
    ) -> FailureReason:
        if isinstance(exc, TaskNotFoundError):
            return FailureReason.OBJECT_NOT_FOUND

        if isinstance(exc, TaskSafetyError):
            message = str(exc).lower()

            if "payload" in message:
                return FailureReason.PAYLOAD_EXCEEDED

            if "collision" in message:
                return FailureReason.COLLISION_RISK

            if (
                "reach" in message
                or "workspace" in message
            ):
                return FailureReason.UNREACHABLE_TARGET

            return FailureReason.GRASP_FAILURE

        if isinstance(exc, TaskStateError):
            return FailureReason.INVALID_TRANSITION

        return FailureReason.SIMULATION_FAILURE

    def execute_next(
        self,
        *,
        inject_failure_step_id: str | None = None,
    ) -> StepExecutionResult | None:
        with self._lock:
            if self.machine.status != TaskStatus.RUNNING:
                raise TaskStateError(
                    "Task must be running"
                )

            if self.next_step_index >= len(
                self.ordered_steps
            ):
                self.machine.transition(
                    TaskStatus.COMPLETED
                )

                return None

            step = self.ordered_steps[
                self.next_step_index
            ]

            completed_ids = {
                result.step_id
                for result in self.step_results
                if result.status == StepStatus.COMPLETED
            }

            if not set(step.depends_on).issubset(
                completed_ids
            ):
                self.machine.transition(
                    TaskStatus.FAILED
                )

                self.failure_reason = (
                    FailureReason.DEPENDENCY_FAILURE
                )

                raise TaskStateError(
                    "Step dependencies are incomplete"
                )

            attempts = 0
            duration = 0.0

            maximum_attempts = (
                1 + self.config.maximum_retries
                if step.retryable
                else 1
            )

            for attempt in range(maximum_attempts):
                attempts += 1

                try:
                    if (
                        inject_failure_step_id
                        == step.step_id
                    ):
                        raise TaskExecutionError(
                            "Injected simulation failure"
                        )

                    duration = self.executor.execute(
                        step,
                        self.world,
                    )

                    result = StepExecutionResult(
                        step_id=step.step_id,
                        action=step.action,
                        status=StepStatus.COMPLETED,
                        attempts=attempts,
                        duration_s=duration,
                        failure_reason=FailureReason.NONE,
                        message="Step completed",
                    )

                    self.step_results.append(
                        result
                    )

                    self.next_step_index += 1

                    self._event(
                        step.step_id,
                        EventSeverity.INFO,
                        f"Completed: {step.action.value}",
                    )

                    if (
                        self.next_step_index
                        >= len(self.ordered_steps)
                    ):
                        self.machine.transition(
                            TaskStatus.COMPLETED
                        )

                        self._event(
                            None,
                            EventSeverity.INFO,
                            "All task steps completed",
                        )

                    return result

                except (
                    TaskExecutionError,
                    ValueError,
                ) as exc:
                    self._event(
                        step.step_id,
                        EventSeverity.WARNING,
                        f"Attempt {attempts} failed: {exc}",
                    )

                    if attempt + 1 < maximum_attempts:
                        self.world.runtime.simulated_time_s += (
                            self.config.retry_delay_s
                        )

                        continue

                    reason = self._failure_reason(
                        exc
                    )

                    self.failure_reason = reason

                    self.machine.transition(
                        TaskStatus.FAILED
                    )

                    result = StepExecutionResult(
                        step_id=step.step_id,
                        action=step.action,
                        status=StepStatus.FAILED,
                        attempts=attempts,
                        duration_s=duration,
                        failure_reason=reason,
                        message=str(exc),
                    )

                    self.step_results.append(
                        result
                    )

                    self._event(
                        step.step_id,
                        EventSeverity.ERROR,
                        f"Task failed: {exc}",
                    )

                    return result

            raise TaskExecutionError(
                "Unexpected execution state"
            )

    def run(
        self,
        *,
        inject_failure_step_id: str | None = None,
    ) -> TaskExecutionReport:
        with self._lock:
            if self.machine.status == TaskStatus.READY:
                self.start()

            if self.machine.status != TaskStatus.RUNNING:
                raise TaskStateError(
                    "Task cannot run in its current state"
                )

            while self.machine.status == TaskStatus.RUNNING:
                if (
                    self.world.runtime.simulated_time_s
                    >= self.task.timeout_s
                ):
                    self.machine.transition(
                        TaskStatus.TIMED_OUT
                    )

                    self.failure_reason = (
                        FailureReason.TIMEOUT
                    )

                    self._event(
                        None,
                        EventSeverity.ERROR,
                        "Simulated execution timeout",
                    )

                    break

                self.execute_next(
                    inject_failure_step_id=(
                        inject_failure_step_id
                    )
                )

            return self.report()

    def report(self) -> TaskExecutionReport:
        with self._lock:
            completed = sum(
                result.status == StepStatus.COMPLETED
                for result in self.step_results
            )

            total = len(self.ordered_steps)

            progress = (
                100.0 * completed / total
            )

            return TaskExecutionReport(
                report_id=new_id(),
                task_id=self.task.task_id,
                tenant_id=self.task.tenant_id,
                robot_id=self.task.robot_id,
                status=self.machine.status,
                failure_reason=self.failure_reason,
                completed_steps=completed,
                total_steps=total,
                progress_percent=progress,
                simulated_duration_s=(
                    self.world.runtime.simulated_time_s
                ),
                final_robot_pose=(
                    self.world.runtime.pose
                ),
                final_gripper_state=(
                    self.world.runtime.gripper_state
                ),
                attached_object_id=(
                    self.world.runtime.attached_object_id
                ),
                step_results=tuple(
                    self.step_results
                ),
                events=tuple(
                    self.events
                ),
                created_at=utc_now(),
            )


# ============================================================
# PICK-AND-PLACE TASK FACTORY
# ============================================================

class PickPlaceTaskFactory:

    @staticmethod
    def create(
        tenant_id: str,
        robot_id: str,
        object_id: str,
        source_pose: Pose3D,
        destination_pose: Pose3D,
        *,
        approach_height_m: float = 0.12,
        lift_height_m: float = 0.18,
        timeout_s: float = 120.0,
    ) -> ManipulationTask:
        positive(
            approach_height_m,
            "approach_height_m",
        )

        positive(
            lift_height_m,
            "lift_height_m",
        )

        pregrasp = source_pose.translated(
            (0.0, 0.0, approach_height_m)
        )

        lifted = source_pose.translated(
            (0.0, 0.0, lift_height_m)
        )

        preplace = destination_pose.translated(
            (0.0, 0.0, lift_height_m)
        )

        steps = (
            TaskStep(
                step_id="move_pregrasp",
                action=ActionType.MOVE,
                target_pose=pregrasp,
            ),
            TaskStep(
                step_id="approach",
                action=ActionType.MOVE,
                target_pose=source_pose,
                depends_on=("move_pregrasp",),
            ),
            TaskStep(
                step_id="close_gripper",
                action=ActionType.CLOSE_GRIPPER,
                depends_on=("approach",),
            ),
            TaskStep(
                step_id="attach",
                action=ActionType.ATTACH_OBJECT,
                object_id=object_id,
                depends_on=("close_gripper",),
            ),
            TaskStep(
                step_id="verify_grasp",
                action=ActionType.VERIFY_GRASP,
                object_id=object_id,
                depends_on=("attach",),
            ),
            TaskStep(
                step_id="lift",
                action=ActionType.MOVE,
                target_pose=lifted,
                depends_on=("verify_grasp",),
            ),
            TaskStep(
                step_id="move_preplace",
                action=ActionType.MOVE,
                target_pose=preplace,
                depends_on=("lift",),
            ),
            TaskStep(
                step_id="lower",
                action=ActionType.MOVE,
                target_pose=destination_pose,
                depends_on=("move_preplace",),
            ),
            TaskStep(
                step_id="release",
                action=ActionType.DETACH_OBJECT,
                object_id=object_id,
                depends_on=("lower",),
            ),
            TaskStep(
                step_id="verify_placement",
                action=ActionType.VERIFY_PLACEMENT,
                object_id=object_id,
                target_pose=destination_pose,
                depends_on=("release",),
            ),
            TaskStep(
                step_id="retreat",
                action=ActionType.MOVE,
                target_pose=preplace,
                depends_on=("verify_placement",),
            ),
        )

        return ManipulationTask(
            task_id=new_id(),
            tenant_id=tenant_id,
            robot_id=robot_id,
            task_type=TaskType.PICK_AND_PLACE,
            name=f"Pick and Place {object_id}",
            steps=steps,
            timeout_s=timeout_s,
        )


# ============================================================
# TASK RECOVERY ADVISOR
# ============================================================

class TaskRecoveryAdvisor:

    RECOMMENDATIONS = {
        FailureReason.NONE: (
            "No recovery required."
        ),
        FailureReason.INVALID_TASK: (
            "Validate task definitions and dependencies."
        ),
        FailureReason.INVALID_TRANSITION: (
            "Inspect the execution state machine."
        ),
        FailureReason.ROBOT_UNAVAILABLE: (
            "Verify robot availability before rescheduling."
        ),
        FailureReason.OBJECT_NOT_FOUND: (
            "Refresh object perception and tracking."
        ),
        FailureReason.OBJECT_UNAVAILABLE: (
            "Confirm object ownership and availability."
        ),
        FailureReason.OBJECT_NOT_ATTACHED: (
            "Reevaluate grasp attachment state."
        ),
        FailureReason.PAYLOAD_EXCEEDED: (
            "Use a robot rated for the object payload."
        ),
        FailureReason.UNREACHABLE_TARGET: (
            "Replan using verified inverse kinematics."
        ),
        FailureReason.COLLISION_RISK: (
            "Replan using full robot collision geometry."
        ),
        FailureReason.GRASP_FAILURE: (
            "Re-detect the object and select another grasp."
        ),
        FailureReason.PLACEMENT_FAILURE: (
            "Inspect placement alignment and target clearance."
        ),
        FailureReason.DEPENDENCY_FAILURE: (
            "Repair task dependency ordering."
        ),
        FailureReason.TIMEOUT: (
            "Review task timing and motion feasibility."
        ),
        FailureReason.CANCELLED: (
            "Task was cancelled; inspect state before restart."
        ),
        FailureReason.SIMULATION_FAILURE: (
            "Inspect simulation logs and retry after correction."
        ),
    }

    @classmethod
    def recommend(
        cls,
        reason: FailureReason,
    ) -> str:
        return cls.RECOMMENDATIONS[reason]


# ============================================================
# MULTI-TENANT EXECUTION SERVICE
# ============================================================

class TaskExecutionService:

    def __init__(
        self,
        config: TaskExecutionConfig | None = None,
    ) -> None:
        self.config = config or TaskExecutionConfig()

        self._robots: dict[
            tuple[str, str],
            RobotSpecification,
        ] = {}

        self._objects: dict[
            tuple[str, str],
            ManipulationObject,
        ] = {}

        self._tasks: dict[
            tuple[str, str],
            ManipulationTask,
        ] = {}

        self._sessions: dict[
            tuple[str, str],
            TaskExecutionSession,
        ] = {}

        self._reports: dict[
            tuple[str, str],
            TaskExecutionReport,
        ] = {}

        self._robot_locks: dict[
            tuple[str, str],
            threading.RLock,
        ] = {}

        self._lock = threading.RLock()

    @staticmethod
    def _authorize(
        access: TaskAccessContext,
        permission: str,
    ) -> None:
        if not getattr(access, permission, False):
            raise TaskAuthorizationError(
                f"Missing permission: {permission}"
            )

    def register_robot(
        self,
        robot: RobotSpecification,
        access: TaskAccessContext,
    ) -> None:
        self._authorize(access, "can_manage")

        if robot.tenant_id != access.tenant_id:
            raise TaskAuthorizationError(
                "Cross-tenant robot registration forbidden"
            )

        key = (
            robot.tenant_id,
            robot.robot_id,
        )

        with self._lock:
            if key in self._robots:
                raise TaskConflictError(
                    "Robot already registered"
                )

            self._robots[key] = robot

            self._robot_locks[key] = threading.RLock()

    def register_object(
        self,
        obj: ManipulationObject,
        access: TaskAccessContext,
    ) -> None:
        self._authorize(access, "can_manage")

        if obj.tenant_id != access.tenant_id:
            raise TaskAuthorizationError(
                "Cross-tenant object registration forbidden"
            )

        key = (
            obj.tenant_id,
            obj.object_id,
        )

        with self._lock:
            if key in self._objects:
                raise TaskConflictError(
                    "Object already registered"
                )

            self._objects[key] = obj

    def create_task(
        self,
        task: ManipulationTask,
        access: TaskAccessContext,
    ) -> None:
        self._authorize(access, "can_create")

        if task.tenant_id != access.tenant_id:
            raise TaskAuthorizationError(
                "Cross-tenant task creation forbidden"
            )

        TaskValidator.validate(task)

        key = (
            task.tenant_id,
            task.task_id,
        )

        with self._lock:
            if key in self._tasks:
                raise TaskConflictError(
                    "Task already exists"
                )

            if (
                task.tenant_id,
                task.robot_id,
            ) not in self._robots:
                raise TaskNotFoundError(
                    "Robot not registered"
                )

            self._tasks[key] = task

    def get_task(
        self,
        task_id: str,
        access: TaskAccessContext,
    ) -> ManipulationTask:
        self._authorize(access, "can_read")

        with self._lock:
            task = self._tasks.get(
                (access.tenant_id, task_id)
            )

        if task is None:
            raise TaskNotFoundError(
                "Task not found"
            )

        return task

    def create_session(
        self,
        task_id: str,
        access: TaskAccessContext,
        obstacles: Sequence[SphericalObstacle] = (),
    ) -> TaskExecutionSession:
        self._authorize(access, "can_execute")

        with self._lock:
            task = self.get_task(
                task_id,
                access,
            )

            robot = self._robots.get(
                (access.tenant_id, task.robot_id)
            )

            if robot is None:
                raise TaskNotFoundError(
                    "Robot not found"
                )

            objects = {
                object_id: obj
                for (tenant_id, object_id), obj
                in self._objects.items()
                if tenant_id == access.tenant_id
            }

            world = SimulationWorld(
                robot=robot,
                objects=objects,
                obstacles=tuple(obstacles),
            )

            session = TaskExecutionSession(
                task,
                world,
                self.config,
            )

            self._sessions[
                (access.tenant_id, task_id)
            ] = session

            return session

    def execute_task(
        self,
        task_id: str,
        access: TaskAccessContext,
        *,
        obstacles: Sequence[SphericalObstacle] = (),
        inject_failure_step_id: str | None = None,
    ) -> TaskExecutionReport:
        self._authorize(access, "can_execute")

        task = self.get_task(
            task_id,
            access,
        )

        robot_key = (
            access.tenant_id,
            task.robot_id,
        )

        with self._lock:
            robot_lock = self._robot_locks.get(
                robot_key
            )

            if robot_lock is None:
                raise TaskNotFoundError(
                    "Robot not found"
                )

        with robot_lock:
            session = self.create_session(
                task_id,
                access,
                obstacles,
            )

            report = session.run(
                inject_failure_step_id=(
                    inject_failure_step_id
                )
            )

            with self._lock:
                self._reports[
                    (access.tenant_id, report.report_id)
                ] = report

                if report.status == TaskStatus.COMPLETED:
                    for object_id, obj in (
                        session.world.objects.items()
                    ):
                        key = (
                            access.tenant_id,
                            object_id,
                        )

                        self._objects[key] = obj

            return report

    def get_report(
        self,
        report_id: str,
        access: TaskAccessContext,
    ) -> TaskExecutionReport:
        self._authorize(access, "can_read")

        with self._lock:
            report = self._reports.get(
                (access.tenant_id, report_id)
            )

        if report is None:
            raise TaskNotFoundError(
                "Execution report not found"
            )

        return report

    def list_tasks(
        self,
        access: TaskAccessContext,
    ) -> tuple[ManipulationTask, ...]:
        self._authorize(access, "can_read")

        with self._lock:
            return tuple(
                task
                for (tenant_id, _), task
                in self._tasks.items()
                if tenant_id == access.tenant_id
            )

    def export_json(
        self,
        report: TaskExecutionReport,
        access: TaskAccessContext,
    ) -> str:
        self._authorize(access, "can_read")

        if report.tenant_id != access.tenant_id:
            raise TaskAuthorizationError(
                "Cross-tenant export forbidden"
            )

        return json.dumps(
            json_safe(
                {
                    "report": report,
                    "simulation_only": True,
                    "physical_actuation": False,
                    "inverse_kinematics_verified": False,
                    "full_arm_collision_verified": False,
                    "dynamics_verified": False,
                    "safety_certified": False,
                }
            ),
            indent=2,
            allow_nan=False,
        )


# ============================================================
# SYNTHETIC FACTORY
# ============================================================

class SyntheticTaskFactory:

    @staticmethod
    def robot(
        tenant_id: str = "physical_ai_lab",
    ) -> RobotSpecification:
        return RobotSpecification(
            robot_id="robot-arm-001",
            tenant_id=tenant_id,
            home_pose=Pose3D(
                position_xyz=(0.3, 0.0, 0.5)
            ),
            workspace=WorkspaceBounds(
                minimum_xyz=(-1.0, -1.0, 0.0),
                maximum_xyz=(1.0, 1.0, 1.5),
            ),
            maximum_reach_m=1.5,
            maximum_payload_kg=5.0,
            maximum_speed_mps=0.25,
        )

    @staticmethod
    def object(
        tenant_id: str = "physical_ai_lab",
    ) -> ManipulationObject:
        return ManipulationObject(
            object_id="component-001",
            tenant_id=tenant_id,
            name="Industrial Component",
            pose=Pose3D(
                position_xyz=(0.45, 0.0, 0.15)
            ),
            mass_kg=0.8,
            grasp_confidence=0.98,
        )

    @staticmethod
    def task(
        tenant_id: str = "physical_ai_lab",
    ) -> ManipulationTask:
        return PickPlaceTaskFactory.create(
            tenant_id=tenant_id,
            robot_id="robot-arm-001",
            object_id="component-001",
            source_pose=Pose3D(
                position_xyz=(0.45, 0.0, 0.15)
            ),
            destination_pose=Pose3D(
                position_xyz=(0.55, 0.25, 0.15)
            ),
        )

    @staticmethod
    def access(
        tenant_id: str = "physical_ai_lab",
    ) -> TaskAccessContext:
        return TaskAccessContext(
            tenant_id=tenant_id,
            user_id="robotics_engineer",
            can_read=True,
            can_create=True,
            can_execute=True,
            can_manage=True,
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestTaskExecution(unittest.TestCase):

    def setUp(self) -> None:
        self.robot = SyntheticTaskFactory.robot()
        self.obj = SyntheticTaskFactory.object()
        self.task = SyntheticTaskFactory.task()
        self.access = SyntheticTaskFactory.access()

        self.config = TaskExecutionConfig()

    def test_pose_translation(self) -> None:
        translated = self.obj.pose.translated(
            (0.0, 0.0, 0.1)
        )

        self.assertAlmostEqual(
            translated.position_xyz[2],
            0.25,
        )

    def test_workspace_contains(self) -> None:
        self.assertTrue(
            self.robot.workspace.contains(
                (0.5, 0.0, 0.5)
            )
        )

    def test_workspace_rejects(self) -> None:
        self.assertFalse(
            self.robot.workspace.contains(
                (10.0, 0.0, 0.5)
            )
        )

    def test_task_validation(self) -> None:
        ordered = TaskValidator.validate(
            self.task
        )

        self.assertEqual(
            len(ordered),
            11,
        )

    def test_task_dependency_order(self) -> None:
        ordered = TaskValidator.validate(
            self.task
        )

        ids = [
            step.step_id
            for step in ordered
        ]

        self.assertLess(
            ids.index("attach"),
            ids.index("verify_grasp"),
        )

    def test_duplicate_step_detection(self) -> None:
        invalid = replace(
            self.task,
            steps=(
                self.task.steps[0],
                self.task.steps[0],
            ),
        )

        with self.assertRaises(
            TaskValidationError
        ):
            TaskValidator.validate(
                invalid
            )

    def test_missing_dependency(self) -> None:
        invalid = replace(
            self.task,
            steps=(
                TaskStep(
                    step_id="step-a",
                    action=ActionType.WAIT,
                    depends_on=("missing",),
                ),
            ),
        )

        with self.assertRaises(
            TaskValidationError
        ):
            TaskValidator.validate(
                invalid
            )

    def test_dependency_cycle(self) -> None:
        invalid = replace(
            self.task,
            steps=(
                TaskStep(
                    step_id="a",
                    action=ActionType.WAIT,
                    depends_on=("b",),
                ),
                TaskStep(
                    step_id="b",
                    action=ActionType.WAIT,
                    depends_on=("a",),
                ),
            ),
        )

        with self.assertRaises(
            TaskValidationError
        ):
            TaskValidator.validate(
                invalid
            )

    def test_collision_checker(self) -> None:
        checker = TaskCollisionChecker(
            self.config
        )

        obstacle = SphericalObstacle(
            obstacle_id="obstacle",
            center_xyz=(0.5, 0.0, 0.0),
            radius_m=0.1,
        )

        self.assertTrue(
            checker.segment_collision(
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (obstacle,),
            )
        )

    def test_clear_path(self) -> None:
        checker = TaskCollisionChecker(
            self.config
        )

        self.assertFalse(
            checker.segment_collision(
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (),
            )
        )

    def test_world_initialization(self) -> None:
        world = SimulationWorld(
            robot=self.robot,
            objects={
                self.obj.object_id: self.obj
            },
        )

        self.assertEqual(
            world.runtime.gripper_state,
            GripperState.OPEN,
        )

    def test_move_action(self) -> None:
        world = SimulationWorld(
            robot=self.robot,
            objects={
                self.obj.object_id: self.obj
            },
        )

        executor = SimulatedActionExecutor(
            self.config
        )

        target = Pose3D(
            position_xyz=(0.4, 0.0, 0.4)
        )

        executor.execute(
            TaskStep(
                step_id="move",
                action=ActionType.MOVE,
                target_pose=target,
            ),
            world,
        )

        self.assertEqual(
            world.runtime.pose,
            target,
        )

    def test_gripper_close(self) -> None:
        world = SimulationWorld(
            robot=self.robot,
            objects={},
        )

        executor = SimulatedActionExecutor(
            self.config
        )

        executor.execute(
            TaskStep(
                step_id="close",
                action=ActionType.CLOSE_GRIPPER,
            ),
            world,
        )

        self.assertEqual(
            world.runtime.gripper_state,
            GripperState.CLOSED,
        )

    def test_grasp_payload_validation(self) -> None:
        heavy = replace(
            self.obj,
            mass_kg=100.0,
        )

        world = SimulationWorld(
            robot=self.robot,
            objects={
                heavy.object_id: heavy
            },
        )

        with self.assertRaises(
            TaskSafetyError
        ):
            TaskSafetyValidator(
                self.config
            ).validate_grasp(
                world,
                heavy,
            )

    def test_state_machine(self) -> None:
        machine = TaskExecutionStateMachine()

        machine.transition(
            TaskStatus.VALIDATED
        )

        machine.transition(
            TaskStatus.READY
        )

        machine.transition(
            TaskStatus.RUNNING
        )

        machine.transition(
            TaskStatus.COMPLETED
        )

        self.assertEqual(
            machine.status,
            TaskStatus.COMPLETED,
        )

    def test_invalid_state_transition(self) -> None:
        machine = TaskExecutionStateMachine()

        with self.assertRaises(
            TaskStateError
        ):
            machine.transition(
                TaskStatus.COMPLETED
            )

    def test_session_start(self) -> None:
        world = SimulationWorld(
            robot=self.robot,
            objects={
                self.obj.object_id: self.obj
            },
        )

        session = TaskExecutionSession(
            self.task,
            world,
            self.config,
        )

        session.start()

        self.assertEqual(
            session.machine.status,
            TaskStatus.RUNNING,
        )

    def test_session_pause_resume(self) -> None:
        world = SimulationWorld(
            robot=self.robot,
            objects={
                self.obj.object_id: self.obj
            },
        )

        session = TaskExecutionSession(
            self.task,
            world,
            self.config,
        )

        session.start()
        session.pause()

        self.assertEqual(
            session.machine.status,
            TaskStatus.PAUSED,
        )

        session.resume()

        self.assertEqual(
            session.machine.status,
            TaskStatus.RUNNING,
        )

    def test_session_cancel(self) -> None:
        world = SimulationWorld(
            robot=self.robot,
            objects={
                self.obj.object_id: self.obj
            },
        )

        session = TaskExecutionSession(
            self.task,
            world,
            self.config,
        )

        session.cancel()

        self.assertEqual(
            session.machine.status,
            TaskStatus.CANCELLED,
        )

    def test_full_task_execution(self) -> None:
        world = SimulationWorld(
            robot=self.robot,
            objects={
                self.obj.object_id: self.obj
            },
        )

        session = TaskExecutionSession(
            self.task,
            world,
            self.config,
        )

        report = session.run()

        self.assertEqual(
            report.status,
            TaskStatus.COMPLETED,
        )

    def test_task_progress(self) -> None:
        world = SimulationWorld(
            robot=self.robot,
            objects={
                self.obj.object_id: self.obj
            },
        )

        session = TaskExecutionSession(
            self.task,
            world,
            self.config,
        )

        report = session.run()

        self.assertAlmostEqual(
            report.progress_percent,
            100.0,
        )

    def test_object_placement(self) -> None:
        world = SimulationWorld(
            robot=self.robot,
            objects={
                self.obj.object_id: self.obj
            },
        )

        session = TaskExecutionSession(
            self.task,
            world,
            self.config,
        )

        session.run()

        placed = world.objects[
            self.obj.object_id
        ]

        self.assertEqual(
            placed.state,
            ObjectState.PLACED,
        )

    def test_failure_injection(self) -> None:
        world = SimulationWorld(
            robot=self.robot,
            objects={
                self.obj.object_id: self.obj
            },
        )

        session = TaskExecutionSession(
            self.task,
            world,
            self.config,
        )

        report = session.run(
            inject_failure_step_id="attach"
        )

        self.assertEqual(
            report.status,
            TaskStatus.FAILED,
        )

    def test_recovery_advisor(self) -> None:
        recommendation = TaskRecoveryAdvisor.recommend(
            FailureReason.GRASP_FAILURE
        )

        self.assertTrue(
            recommendation
        )

    def test_service_registration(self) -> None:
        service = TaskExecutionService()

        service.register_robot(
            self.robot,
            self.access,
        )

        service.register_object(
            self.obj,
            self.access,
        )

        service.create_task(
            self.task,
            self.access,
        )

        self.assertEqual(
            len(service.list_tasks(self.access)),
            1,
        )

    def test_service_execution(self) -> None:
        service = TaskExecutionService()

        service.register_robot(
            self.robot,
            self.access,
        )

        service.register_object(
            self.obj,
            self.access,
        )

        service.create_task(
            self.task,
            self.access,
        )

        report = service.execute_task(
            self.task.task_id,
            self.access,
        )

        self.assertEqual(
            report.status,
            TaskStatus.COMPLETED,
        )

    def test_tenant_isolation(self) -> None:
        service = TaskExecutionService()

        service.register_robot(
            self.robot,
            self.access,
        )

        service.create_task(
            self.task,
            self.access,
        )

        other = TaskAccessContext(
            tenant_id="other_tenant",
            user_id="other_user",
            can_read=True,
        )

        with self.assertRaises(
            TaskNotFoundError
        ):
            service.get_task(
                self.task.task_id,
                other,
            )

    def test_permission_denied(self) -> None:
        service = TaskExecutionService()

        viewer = TaskAccessContext(
            tenant_id=self.robot.tenant_id,
            user_id="viewer",
            can_read=True,
        )

        with self.assertRaises(
            TaskAuthorizationError
        ):
            service.register_robot(
                self.robot,
                viewer,
            )

    def test_json_export(self) -> None:
        service = TaskExecutionService()

        service.register_robot(
            self.robot,
            self.access,
        )

        service.register_object(
            self.obj,
            self.access,
        )

        service.create_task(
            self.task,
            self.access,
        )

        report = service.execute_task(
            self.task.task_id,
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
    print("\nPHYSICAL AI — TASK EXECUTION")
    print("=" * 72)

    robot = SyntheticTaskFactory.robot()
    obj = SyntheticTaskFactory.object()
    task = SyntheticTaskFactory.task()
    access = SyntheticTaskFactory.access()

    service = TaskExecutionService()

    service.register_robot(
        robot,
        access,
    )

    service.register_object(
        obj,
        access,
    )

    service.create_task(
        task,
        access,
    )

    print("Robot:", robot.robot_id)
    print("Task:", task.name)
    print("Task type:", task.task_type.value)
    print("Object:", obj.name)
    print("Steps:", len(task.steps))

    report = service.execute_task(
        task.task_id,
        access,
    )

    print(
        "Execution status:",
        report.status.value,
    )

    print(
        "Completed steps:",
        report.completed_steps,
    )

    print(
        "Progress:",
        round(report.progress_percent, 2),
        "%",
    )

    print(
        "Simulated duration:",
        round(report.simulated_duration_s, 3),
        "s",
    )

    print(
        "Final gripper state:",
        report.final_gripper_state.value,
    )

    print(
        "Final robot position:",
        report.final_robot_pose.position_xyz,
    )

    print("\nEXECUTION REPORT JSON")
    print("=" * 72)

    print(
        service.export_json(
            report,
            access,
        )
    )

    print("\nUNIT TESTS")
    print("=" * 72)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestTaskExecution
    )

    result = unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)

    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
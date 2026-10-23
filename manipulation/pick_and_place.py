"""Pick And Place module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/manipulation/pick_and_place.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================
Module: Pick and Place

Simulation-only pick-and-place task planning, validation,
execution, monitoring, and recovery.

Features:
    - Pick-and-place task creation
    - Object and robot registration
    - Object detection confidence validation
    - Parallel-jaw and suction gripper support
    - Grasp selection and feasibility evaluation
    - Payload and holding-force checks
    - Pre-grasp, approach, lift, transport, and placement planning
    - Object-relative grasp transformations
    - Simplified collision checking
    - Cartesian waypoint generation
    - Joint-space motion planning integration hooks
    - Gripper state simulation
    - Object attachment and detachment
    - Pick-and-place state machine
    - Step-by-step simulation
    - Failure injection and recovery recommendations
    - Multi-tenant access control
    - Task history and audit logging
    - JSON serialization
    - Synthetic robot demonstrations
    - Unit tests

Requirements:
    Python >= 3.11
    numpy >= 1.24

Install:
    pip install numpy

SAFETY:
    This module does not control physical hardware.

    Collision checking is approximate. Cartesian waypoints
    are not proof of inverse-kinematics feasibility or
    full-arm collision avoidance.

    Real robot deployment requires robot-specific IK,
    dynamics, safety-rated controllers, force monitoring,
    collision validation, and independent emergency stops.
"""

from __future__ import annotations

import json
import math
import threading
import unittest
import uuid

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Sequence

import numpy as np


# ============================================================
# EXCEPTIONS
# ============================================================

class PickPlaceError(Exception):
    pass


class PickPlaceValidationError(PickPlaceError):
    pass


class PickPlaceAuthorizationError(PickPlaceError):
    pass


class PickPlaceNotFoundError(PickPlaceError):
    pass


class PickPlaceConflictError(PickPlaceError):
    pass


class PickPlacePlanningError(PickPlaceError):
    pass


class PickPlaceStateError(PickPlaceError):
    pass


class PickPlaceSimulationError(PickPlaceError):
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
        raise PickPlaceValidationError(
            f"Invalid {name}"
        )

    return value.strip()


def finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise PickPlaceValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PickPlaceValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise PickPlaceValidationError(
            f"{name} must be finite"
        )

    return result


def positive(value: Any, name: str) -> float:
    result = finite(value, name)

    if result <= 0:
        raise PickPlaceValidationError(
            f"{name} must be positive"
        )

    return result


def nonnegative(value: Any, name: str) -> float:
    result = finite(value, name)

    if result < 0:
        raise PickPlaceValidationError(
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
        raise PickPlaceValidationError(
            f"{name} must contain numeric values"
        ) from exc

    if result.shape != (3,):
        raise PickPlaceValidationError(
            f"{name} must contain three values"
        )

    if not np.all(np.isfinite(result)):
        raise PickPlaceValidationError(
            f"{name} contains non-finite values"
        )

    return result


def unit_vector(
    values: Sequence[float],
    name: str = "direction",
) -> np.ndarray:
    result = vector3(values, name)

    norm = float(np.linalg.norm(result))

    if norm <= 1e-12:
        raise PickPlaceValidationError(
            f"{name} cannot be zero"
        )

    return result / norm


def tuple3(values: Sequence[float]) -> tuple[float, float, float]:
    array = vector3(values, "vector")

    return (
        float(array[0]),
        float(array[1]),
        float(array[2]),
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

    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


# ============================================================
# ENUMERATIONS
# ============================================================

class GripperType(str, Enum):
    PARALLEL_JAW = "parallel_jaw"
    SUCTION = "suction"


class GripperState(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    HOLDING = "holding"
    ERROR = "error"


class ObjectState(str, Enum):
    AVAILABLE = "available"
    RESERVED = "reserved"
    ATTACHED = "attached"
    PLACED = "placed"
    LOST = "lost"


class PickPlaceStatus(str, Enum):
    CREATED = "created"
    PLANNED = "planned"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PickPlaceStep(str, Enum):
    MOVE_TO_PREGRASP = "move_to_pregrasp"
    APPROACH_OBJECT = "approach_object"
    CLOSE_GRIPPER = "close_gripper"
    VERIFY_GRASP = "verify_grasp"
    LIFT_OBJECT = "lift_object"
    MOVE_TO_PREPLACE = "move_to_preplace"
    LOWER_OBJECT = "lower_object"
    RELEASE_OBJECT = "release_object"
    RETREAT = "retreat"
    RETURN_HOME = "return_home"


class FailureReason(str, Enum):
    NONE = "none"
    OBJECT_UNAVAILABLE = "object_unavailable"
    LOW_DETECTION_CONFIDENCE = "low_detection_confidence"
    PAYLOAD_EXCEEDED = "payload_exceeded"
    GRIPPER_INCOMPATIBLE = "gripper_incompatible"
    INSUFFICIENT_GRIP_FORCE = "insufficient_grip_force"
    TARGET_OCCUPIED = "target_occupied"
    COLLISION_RISK = "collision_risk"
    UNREACHABLE_POSE = "unreachable_pose"
    GRASP_FAILED = "grasp_failed"
    OBJECT_DROPPED = "object_dropped"
    SIMULATION_ERROR = "simulation_error"


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
        vector3(self.position_xyz, "position_xyz")
        vector3(self.orientation_rpy, "orientation_rpy")

    def translated(
        self,
        offset: Sequence[float],
    ) -> Pose3D:
        result = (
            vector3(self.position_xyz, "position")
            + vector3(offset, "offset")
        )

        return Pose3D(
            position_xyz=tuple3(result),
            orientation_rpy=self.orientation_rpy,
        )


@dataclass(frozen=True)
class ObjectDimensions:
    length_m: float
    width_m: float
    height_m: float

    def __post_init__(self) -> None:
        positive(self.length_m, "length_m")
        positive(self.width_m, "width_m")
        positive(self.height_m, "height_m")

    @property
    def bounding_radius_m(self) -> float:
        return 0.5 * math.sqrt(
            self.length_m**2
            + self.width_m**2
            + self.height_m**2
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
            raise PickPlaceValidationError(
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


# ============================================================
# ROBOT AND OBJECT MODELS
# ============================================================

@dataclass(frozen=True)
class GripperSpecification:
    gripper_id: str
    gripper_type: GripperType

    minimum_opening_m: float = 0.005
    maximum_opening_m: float = 0.12

    maximum_payload_kg: float = 5.0
    maximum_normal_force_n: float = 80.0
    maximum_suction_force_n: float = 60.0

    safety_factor: float = 2.0

    def __post_init__(self) -> None:
        identifier(self.gripper_id, "gripper_id")

        nonnegative(
            self.minimum_opening_m,
            "minimum_opening_m",
        )

        positive(
            self.maximum_opening_m,
            "maximum_opening_m",
        )

        if (
            self.minimum_opening_m
            >= self.maximum_opening_m
        ):
            raise PickPlaceValidationError(
                "Invalid gripper opening range"
            )

        positive(
            self.maximum_payload_kg,
            "maximum_payload_kg",
        )

        positive(
            self.maximum_normal_force_n,
            "maximum_normal_force_n",
        )

        positive(
            self.maximum_suction_force_n,
            "maximum_suction_force_n",
        )

        if finite(
            self.safety_factor,
            "safety_factor",
        ) < 1:
            raise PickPlaceValidationError(
                "Safety factor must be >= 1"
            )


@dataclass(frozen=True)
class PickPlaceRobot:
    robot_id: str
    tenant_id: str

    gripper: GripperSpecification

    maximum_payload_kg: float
    maximum_reach_m: float

    home_pose: Pose3D

    workspace: WorkspaceBounds

    def __post_init__(self) -> None:
        identifier(self.robot_id, "robot_id")
        identifier(self.tenant_id, "tenant_id")

        positive(
            self.maximum_payload_kg,
            "maximum_payload_kg",
        )

        positive(
            self.maximum_reach_m,
            "maximum_reach_m",
        )

        if not self.workspace.contains(
            self.home_pose.position_xyz
        ):
            raise PickPlaceValidationError(
                "Home pose outside workspace"
            )


@dataclass(frozen=True)
class PickPlaceObject:
    object_id: str
    tenant_id: str

    name: str
    pose: Pose3D

    dimensions: ObjectDimensions

    mass_kg: float
    friction_coefficient: float

    detection_confidence: float = 1.0

    state: ObjectState = ObjectState.AVAILABLE

    def __post_init__(self) -> None:
        identifier(self.object_id, "object_id")
        identifier(self.tenant_id, "tenant_id")
        identifier(self.name, "name")

        positive(self.mass_kg, "mass_kg")

        positive(
            self.friction_coefficient,
            "friction_coefficient",
        )

        confidence = finite(
            self.detection_confidence,
            "detection_confidence",
        )

        if not 0 <= confidence <= 1:
            raise PickPlaceValidationError(
                "Detection confidence must be in [0,1]"
            )


@dataclass(frozen=True)
class SphericalObstacle:
    obstacle_id: str
    center_xyz: tuple[float, float, float]
    radius_m: float

    def __post_init__(self) -> None:
        identifier(self.obstacle_id, "obstacle_id")
        vector3(self.center_xyz, "center_xyz")
        positive(self.radius_m, "radius_m")


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PickPlaceConfig:
    minimum_detection_confidence: float = 0.60

    approach_distance_m: float = 0.12
    lift_height_m: float = 0.18
    retreat_distance_m: float = 0.12

    transport_clearance_m: float = 0.08

    collision_margin_m: float = 0.02
    collision_resolution_m: float = 0.02

    placement_tolerance_m: float = 0.01

    maximum_grasp_attempts: int = 3

    simulation_speed_mps: float = 0.20

    return_home: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_detection_confidence <= 1:
            raise PickPlaceValidationError(
                "Invalid detection confidence threshold"
            )

        for name in (
            "approach_distance_m",
            "lift_height_m",
            "retreat_distance_m",
            "collision_resolution_m",
            "simulation_speed_mps",
        ):
            positive(getattr(self, name), name)

        for name in (
            "transport_clearance_m",
            "collision_margin_m",
            "placement_tolerance_m",
        ):
            nonnegative(getattr(self, name), name)

        if self.maximum_grasp_attempts < 1:
            raise PickPlaceValidationError(
                "maximum_grasp_attempts must be positive"
            )


# ============================================================
# TASK MODELS
# ============================================================

@dataclass(frozen=True)
class PickPlaceRequest:
    request_id: str
    robot_id: str
    object_id: str

    destination_pose: Pose3D

    obstacles: tuple[SphericalObstacle, ...] = ()

    preferred_grasp_width_m: float | None = None

    def __post_init__(self) -> None:
        identifier(self.request_id, "request_id")
        identifier(self.robot_id, "robot_id")
        identifier(self.object_id, "object_id")

        if self.preferred_grasp_width_m is not None:
            positive(
                self.preferred_grasp_width_m,
                "preferred_grasp_width_m",
            )


@dataclass(frozen=True)
class GraspAssessment:
    feasible: bool

    grasp_width_m: float

    payload_margin_kg: float

    available_holding_force_n: float
    required_holding_force_n: float

    quality_score: float

    failure_reason: FailureReason

    message: str


@dataclass(frozen=True)
class PickPlaceWaypoint:
    waypoint_id: str

    step: PickPlaceStep

    pose: Pose3D

    gripper_state: GripperState

    object_attached: bool

    estimated_duration_s: float = 0.0


@dataclass(frozen=True)
class PickPlacePlan:
    plan_id: str
    request_id: str

    robot_id: str
    object_id: str
    tenant_id: str

    grasp_assessment: GraspAssessment

    waypoints: tuple[PickPlaceWaypoint, ...]

    total_distance_m: float
    estimated_duration_s: float

    created_at: datetime


@dataclass(frozen=True)
class PickPlaceEvent:
    event_id: str
    task_id: str

    step: PickPlaceStep | None

    status: PickPlaceStatus

    message: str

    timestamp: datetime


@dataclass(frozen=True)
class PickPlaceResult:
    result_id: str
    task_id: str

    tenant_id: str
    robot_id: str
    object_id: str

    status: PickPlaceStatus
    failure_reason: FailureReason

    completed_steps: int

    final_object_pose: Pose3D

    final_gripper_state: GripperState

    plan: PickPlacePlan

    events: tuple[PickPlaceEvent, ...]

    created_at: datetime


@dataclass(frozen=True)
class PickPlaceAccessContext:
    tenant_id: str
    user_id: str

    can_read: bool = False
    can_plan: bool = False
    can_simulate: bool = False
    can_manage: bool = False


# ============================================================
# GRASP FEASIBILITY
# ============================================================

class PickPlaceGraspEvaluator:

    GRAVITY = 9.81

    def evaluate(
        self,
        robot: PickPlaceRobot,
        obj: PickPlaceObject,
        requested_width_m: float | None = None,
    ) -> GraspAssessment:
        gripper = robot.gripper

        effective_payload = min(
            robot.maximum_payload_kg,
            gripper.maximum_payload_kg,
        )

        payload_margin = (
            effective_payload
            - obj.mass_kg * gripper.safety_factor
        )

        payload_feasible = payload_margin >= 0

        required_force = (
            obj.mass_kg
            * self.GRAVITY
            * gripper.safety_factor
        )

        if gripper.gripper_type == GripperType.PARALLEL_JAW:
            candidate_widths = (
                [requested_width_m]
                if requested_width_m is not None
                else [
                    obj.dimensions.width_m,
                    obj.dimensions.length_m,
                ]
            )

            valid_widths = [
                width
                for width in candidate_widths
                if (
                    width is not None
                    and gripper.minimum_opening_m
                    <= width
                    <= gripper.maximum_opening_m
                )
            ]

            geometry_feasible = bool(valid_widths)

            selected_width = (
                min(valid_widths)
                if valid_widths
                else float(candidate_widths[0])
            )

            # Simplified two-finger friction estimate.
            # Assumes each finger can supply the specified
            # normal force and contacts are symmetric.
            available_force = (
                2.0
                * obj.friction_coefficient
                * gripper.maximum_normal_force_n
            )

        else:
            geometry_feasible = True
            selected_width = 0.0

            # Assumes a suitable airtight surface.
            # Surface porosity and cup geometry are not modeled.
            available_force = (
                gripper.maximum_suction_force_n
            )

        force_feasible = (
            available_force >= required_force
        )

        if not payload_feasible:
            failure = FailureReason.PAYLOAD_EXCEEDED
            message = "Payload exceeds configured safety margin"

        elif not geometry_feasible:
            failure = FailureReason.GRIPPER_INCOMPATIBLE
            message = "Object width incompatible with gripper"

        elif not force_feasible:
            failure = FailureReason.INSUFFICIENT_GRIP_FORCE
            message = "Estimated holding force is insufficient"

        else:
            failure = FailureReason.NONE
            message = "Grasp passed approximate feasibility checks"

        payload_score = clamp(
            effective_payload
            / max(
                obj.mass_kg * gripper.safety_factor,
                1e-9,
            ),
            0.0,
            1.0,
        )

        force_score = clamp(
            available_force / max(required_force, 1e-9),
            0.0,
            1.0,
        )

        geometry_score = (
            1.0 if geometry_feasible else 0.0
        )

        quality_score = (
            0.25 * payload_score
            + 0.30 * force_score
            + 0.25 * geometry_score
            + 0.20 * obj.detection_confidence
        )

        return GraspAssessment(
            feasible=(
                payload_feasible
                and geometry_feasible
                and force_feasible
            ),
            grasp_width_m=selected_width,
            payload_margin_kg=payload_margin,
            available_holding_force_n=available_force,
            required_holding_force_n=required_force,
            quality_score=quality_score,
            failure_reason=failure,
            message=message,
        )


# ============================================================
# COLLISION CHECKER
# ============================================================

class PickPlaceCollisionChecker:

    def __init__(
        self,
        config: PickPlaceConfig,
    ) -> None:
        self.config = config

    def point_collision(
        self,
        position: Sequence[float],
        obstacles: Sequence[SphericalObstacle],
        payload_radius_m: float = 0.0,
    ) -> bool:
        point = vector3(position, "position")

        for obstacle in obstacles:
            center = vector3(
                obstacle.center_xyz,
                "obstacle center",
            )

            required_distance = (
                obstacle.radius_m
                + payload_radius_m
                + self.config.collision_margin_m
            )

            if (
                np.linalg.norm(point - center)
                <= required_distance
            ):
                return True

        return False

    def segment_collision(
        self,
        start: Sequence[float],
        end: Sequence[float],
        obstacles: Sequence[SphericalObstacle],
        payload_radius_m: float = 0.0,
    ) -> bool:
        first = vector3(start, "start")
        second = vector3(end, "end")

        length = float(
            np.linalg.norm(second - first)
        )

        samples = max(
            1,
            math.ceil(
                length
                / self.config.collision_resolution_m
            ),
        )

        for index in range(samples + 1):
            alpha = index / samples

            point = (
                first * (1.0 - alpha)
                + second * alpha
            )

            if self.point_collision(
                point,
                obstacles,
                payload_radius_m,
            ):
                return True

        return False


# ============================================================
# WAYPOINT GENERATOR
# ============================================================

class PickPlaceWaypointGenerator:

    def __init__(
        self,
        config: PickPlaceConfig,
    ) -> None:
        self.config = config

        self.collision_checker = (
            PickPlaceCollisionChecker(config)
        )

    @staticmethod
    def _waypoint(
        step: PickPlaceStep,
        pose: Pose3D,
        gripper_state: GripperState,
        attached: bool,
    ) -> PickPlaceWaypoint:
        return PickPlaceWaypoint(
            waypoint_id=new_id(),
            step=step,
            pose=pose,
            gripper_state=gripper_state,
            object_attached=attached,
        )

    def generate(
        self,
        robot: PickPlaceRobot,
        obj: PickPlaceObject,
        request: PickPlaceRequest,
    ) -> tuple[PickPlaceWaypoint, ...]:
        source = vector3(
            obj.pose.position_xyz,
            "source",
        )

        target = vector3(
            request.destination_pose.position_xyz,
            "target",
        )

        top_offset = np.array(
            [
                0.0,
                0.0,
                obj.dimensions.height_m / 2.0,
            ]
        )

        grasp = source + top_offset
        place = target + top_offset

        pregrasp = grasp + np.array(
            [
                0.0,
                0.0,
                self.config.approach_distance_m,
            ]
        )

        lift = grasp + np.array(
            [
                0.0,
                0.0,
                self.config.lift_height_m,
            ]
        )

        # Raise the transport height enough to clear
        # the source and destination approach positions.
        transport_z = max(
            lift[2],
            place[2] + self.config.lift_height_m,
        )

        if request.obstacles:
            transport_z = max(
                transport_z,
                max(
                    obstacle.center_xyz[2]
                    + obstacle.radius_m
                    + obj.dimensions.bounding_radius_m
                    + self.config.transport_clearance_m
                    for obstacle in request.obstacles
                ),
            )

        preplace = np.array(
            [
                place[0],
                place[1],
                transport_z,
            ]
        )

        retreat = place + np.array(
            [
                0.0,
                0.0,
                self.config.retreat_distance_m,
            ]
        )

        def pose(position: np.ndarray) -> Pose3D:
            return Pose3D(
                position_xyz=tuple3(position),
                orientation_rpy=obj.pose.orientation_rpy,
            )

        waypoints = [
            self._waypoint(
                PickPlaceStep.MOVE_TO_PREGRASP,
                pose(pregrasp),
                GripperState.OPEN,
                False,
            ),
            self._waypoint(
                PickPlaceStep.APPROACH_OBJECT,
                pose(grasp),
                GripperState.OPEN,
                False,
            ),
            self._waypoint(
                PickPlaceStep.CLOSE_GRIPPER,
                pose(grasp),
                GripperState.CLOSED,
                False,
            ),
            self._waypoint(
                PickPlaceStep.VERIFY_GRASP,
                pose(grasp),
                GripperState.HOLDING,
                True,
            ),
            self._waypoint(
                PickPlaceStep.LIFT_OBJECT,
                pose(lift),
                GripperState.HOLDING,
                True,
            ),
            self._waypoint(
                PickPlaceStep.MOVE_TO_PREPLACE,
                pose(preplace),
                GripperState.HOLDING,
                True,
            ),
            self._waypoint(
                PickPlaceStep.LOWER_OBJECT,
                pose(place),
                GripperState.HOLDING,
                True,
            ),
            self._waypoint(
                PickPlaceStep.RELEASE_OBJECT,
                pose(place),
                GripperState.OPEN,
                False,
            ),
            self._waypoint(
                PickPlaceStep.RETREAT,
                pose(retreat),
                GripperState.OPEN,
                False,
            ),
        ]

        if self.config.return_home:
            waypoints.append(
                self._waypoint(
                    PickPlaceStep.RETURN_HOME,
                    robot.home_pose,
                    GripperState.OPEN,
                    False,
                )
            )

        return tuple(waypoints)

    def validate(
        self,
        robot: PickPlaceRobot,
        obj: PickPlaceObject,
        request: PickPlaceRequest,
        waypoints: Sequence[PickPlaceWaypoint],
    ) -> None:
        if not waypoints:
            raise PickPlacePlanningError(
                "No waypoints generated"
            )

        home = vector3(
            robot.home_pose.position_xyz,
            "home position",
        )

        for waypoint in waypoints:
            position = waypoint.pose.position_xyz

            if not robot.workspace.contains(position):
                raise PickPlacePlanningError(
                    f"Waypoint outside workspace: "
                    f"{waypoint.step.value}"
                )

            # Approximate reach check only.
            if (
                distance3(home, position)
                > robot.maximum_reach_m
            ):
                raise PickPlacePlanningError(
                    f"Waypoint exceeds approximate reach: "
                    f"{waypoint.step.value}"
                )

        # The first waypoint is assumed to be reached by
        # a separately validated robot motion planner.
        # Check the Cartesian segments between waypoints.
        for index in range(1, len(waypoints)):
            previous = waypoints[index - 1]
            current = waypoints[index]

            if (
                previous.pose.position_xyz
                == current.pose.position_xyz
            ):
                continue

            payload_radius = (
                obj.dimensions.bounding_radius_m
                if (
                    previous.object_attached
                    and current.object_attached
                )
                else 0.0
            )

            if self.collision_checker.segment_collision(
                previous.pose.position_xyz,
                current.pose.position_xyz,
                request.obstacles,
                payload_radius,
            ):
                raise PickPlacePlanningError(
                    f"Approximate collision on segment: "
                    f"{previous.step.value} -> "
                    f"{current.step.value}"
                )

        # Check the destination object volume.
        if self.collision_checker.point_collision(
            request.destination_pose.position_xyz,
            request.obstacles,
            obj.dimensions.bounding_radius_m,
        ):
            raise PickPlacePlanningError(
                "Destination occupied by an obstacle"
            )


# ============================================================
# PICK AND PLACE PLANNER
# ============================================================

class PickPlacePlanner:

    def __init__(
        self,
        config: PickPlaceConfig | None = None,
    ) -> None:
        self.config = config or PickPlaceConfig()

        self.grasp_evaluator = (
            PickPlaceGraspEvaluator()
        )

        self.waypoint_generator = (
            PickPlaceWaypointGenerator(self.config)
        )

    def plan(
        self,
        robot: PickPlaceRobot,
        obj: PickPlaceObject,
        request: PickPlaceRequest,
    ) -> PickPlacePlan:
        if robot.robot_id != request.robot_id:
            raise PickPlaceValidationError(
                "Request robot mismatch"
            )

        if obj.object_id != request.object_id:
            raise PickPlaceValidationError(
                "Request object mismatch"
            )

        if robot.tenant_id != obj.tenant_id:
            raise PickPlaceAuthorizationError(
                "Robot and object tenants differ"
            )

        if obj.state != ObjectState.AVAILABLE:
            raise PickPlacePlanningError(
                "Object is not available"
            )

        if (
            obj.detection_confidence
            < self.config.minimum_detection_confidence
        ):
            raise PickPlacePlanningError(
                "Object detection confidence below threshold"
            )

        grasp = self.grasp_evaluator.evaluate(
            robot,
            obj,
            request.preferred_grasp_width_m,
        )

        if not grasp.feasible:
            raise PickPlacePlanningError(
                grasp.message
            )

        waypoints = self.waypoint_generator.generate(
            robot,
            obj,
            request,
        )

        self.waypoint_generator.validate(
            robot,
            obj,
            request,
            waypoints,
        )

        total_distance = 0.0
        estimated_duration = 0.0

        updated_waypoints: list[PickPlaceWaypoint] = []

        previous_position = vector3(
            robot.home_pose.position_xyz,
            "home position",
        )

        for waypoint in waypoints:
            position = vector3(
                waypoint.pose.position_xyz,
                "waypoint position",
            )

            segment_distance = float(
                np.linalg.norm(
                    position - previous_position
                )
            )

            duration = (
                segment_distance
                / self.config.simulation_speed_mps
            )

            if waypoint.step in (
                PickPlaceStep.CLOSE_GRIPPER,
                PickPlaceStep.VERIFY_GRASP,
                PickPlaceStep.RELEASE_OBJECT,
            ):
                duration += 0.25

            total_distance += segment_distance
            estimated_duration += duration

            updated_waypoints.append(
                replace(
                    waypoint,
                    estimated_duration_s=duration,
                )
            )

            previous_position = position

        return PickPlacePlan(
            plan_id=new_id(),
            request_id=request.request_id,
            robot_id=robot.robot_id,
            object_id=obj.object_id,
            tenant_id=robot.tenant_id,
            grasp_assessment=grasp,
            waypoints=tuple(updated_waypoints),
            total_distance_m=total_distance,
            estimated_duration_s=estimated_duration,
            created_at=utc_now(),
        )


# ============================================================
# SIMULATED GRIPPER
# ============================================================

class SimulatedGripper:

    def __init__(
        self,
        specification: GripperSpecification,
    ) -> None:
        self.specification = specification

        self.state = GripperState.OPEN
        self.attached_object_id: str | None = None

    def close(self) -> None:
        if self.attached_object_id is not None:
            raise PickPlaceStateError(
                "Cannot close gripper while holding an object"
            )

        self.state = GripperState.CLOSED

    def attach(self, object_id: str) -> None:
        if self.state != GripperState.CLOSED:
            raise PickPlaceStateError(
                "Gripper must be closed before attachment"
            )

        if self.attached_object_id is not None:
            raise PickPlaceStateError(
                "Gripper already holds an object"
            )

        self.attached_object_id = object_id
        self.state = GripperState.HOLDING

    def release(self) -> str | None:
        released = self.attached_object_id

        self.attached_object_id = None
        self.state = GripperState.OPEN

        return released


# ============================================================
# PICK AND PLACE STATE MACHINE
# ============================================================

class PickPlaceStateMachine:

    TRANSITIONS = {
        PickPlaceStatus.CREATED: {
            PickPlaceStatus.PLANNED,
            PickPlaceStatus.CANCELLED,
            PickPlaceStatus.FAILED,
        },
        PickPlaceStatus.PLANNED: {
            PickPlaceStatus.RUNNING,
            PickPlaceStatus.CANCELLED,
            PickPlaceStatus.FAILED,
        },
        PickPlaceStatus.RUNNING: {
            PickPlaceStatus.PAUSED,
            PickPlaceStatus.COMPLETED,
            PickPlaceStatus.FAILED,
            PickPlaceStatus.CANCELLED,
        },
        PickPlaceStatus.PAUSED: {
            PickPlaceStatus.RUNNING,
            PickPlaceStatus.CANCELLED,
            PickPlaceStatus.FAILED,
        },
        PickPlaceStatus.COMPLETED: set(),
        PickPlaceStatus.FAILED: set(),
        PickPlaceStatus.CANCELLED: set(),
    }

    def __init__(self) -> None:
        self.status = PickPlaceStatus.CREATED

    def transition(
        self,
        new_status: PickPlaceStatus,
    ) -> None:
        allowed = self.TRANSITIONS[self.status]

        if new_status not in allowed:
            raise PickPlaceStateError(
                f"Invalid transition: "
                f"{self.status.value} -> {new_status.value}"
            )

        self.status = new_status


# ============================================================
# SIMULATION ENGINE
# ============================================================

class PickPlaceSimulator:

    def __init__(
        self,
        config: PickPlaceConfig | None = None,
    ) -> None:
        self.config = config or PickPlaceConfig()

    def simulate(
        self,
        robot: PickPlaceRobot,
        obj: PickPlaceObject,
        request: PickPlaceRequest,
        plan: PickPlacePlan,
        *,
        fail_at_step: PickPlaceStep | None = None,
    ) -> PickPlaceResult:
        if (
            plan.robot_id != robot.robot_id
            or plan.object_id != obj.object_id
            or plan.request_id != request.request_id
            or plan.tenant_id != robot.tenant_id
            or obj.tenant_id != robot.tenant_id
        ):
            raise PickPlaceValidationError(
                "Plan, request, robot, or object mismatch"
            )

        task_id = new_id()

        machine = PickPlaceStateMachine()

        machine.transition(
            PickPlaceStatus.PLANNED
        )

        machine.transition(
            PickPlaceStatus.RUNNING
        )

        gripper = SimulatedGripper(
            robot.gripper
        )

        events: list[PickPlaceEvent] = []

        object_pose = obj.pose

        completed_steps = 0

        failure_reason = FailureReason.NONE

        grasp_position = vector3(
            obj.pose.position_xyz,
            "object center",
        ) + np.array(
            [
                0.0,
                0.0,
                obj.dimensions.height_m / 2.0,
            ]
        )

        object_offset = (
            vector3(
                obj.pose.position_xyz,
                "object position",
            )
            - grasp_position
        )

        def add_event(
            step: PickPlaceStep | None,
            message: str,
        ) -> None:
            events.append(
                PickPlaceEvent(
                    event_id=new_id(),
                    task_id=task_id,
                    step=step,
                    status=machine.status,
                    message=message,
                    timestamp=utc_now(),
                )
            )

        add_event(
            None,
            "Pick-and-place simulation started",
        )

        for waypoint in plan.waypoints:
            step = waypoint.step

            if step == fail_at_step:
                machine.transition(
                    PickPlaceStatus.FAILED
                )

                failure_reason = (
                    FailureReason.SIMULATION_ERROR
                )

                add_event(
                    step,
                    "Injected simulation failure",
                )

                break

            if step == PickPlaceStep.CLOSE_GRIPPER:
                gripper.close()

            elif step == PickPlaceStep.VERIFY_GRASP:
                if not plan.grasp_assessment.feasible:
                    machine.transition(
                        PickPlaceStatus.FAILED
                    )

                    failure_reason = (
                        FailureReason.GRASP_FAILED
                    )

                    add_event(
                        step,
                        "Grasp verification failed",
                    )

                    break

                gripper.attach(obj.object_id)

            elif step == PickPlaceStep.RELEASE_OBJECT:
                if (
                    gripper.attached_object_id
                    != obj.object_id
                ):
                    machine.transition(
                        PickPlaceStatus.FAILED
                    )

                    failure_reason = (
                        FailureReason.OBJECT_DROPPED
                    )

                    add_event(
                        step,
                        "Object was not attached at release",
                    )

                    break

                gripper.release()

                object_pose = request.destination_pose

            elif (
                gripper.attached_object_id
                == obj.object_id
            ):
                tool_position = vector3(
                    waypoint.pose.position_xyz,
                    "tool position",
                )

                object_position = (
                    tool_position + object_offset
                )

                object_pose = Pose3D(
                    position_xyz=tuple3(
                        object_position
                    ),
                    orientation_rpy=(
                        obj.pose.orientation_rpy
                    ),
                )

            completed_steps += 1

            add_event(
                step,
                f"Simulated step completed: {step.value}",
            )

        if machine.status == PickPlaceStatus.RUNNING:
            machine.transition(
                PickPlaceStatus.COMPLETED
            )

            add_event(
                None,
                "Pick-and-place simulation completed",
            )

        return PickPlaceResult(
            result_id=new_id(),
            task_id=task_id,
            tenant_id=robot.tenant_id,
            robot_id=robot.robot_id,
            object_id=obj.object_id,
            status=machine.status,
            failure_reason=failure_reason,
            completed_steps=completed_steps,
            final_object_pose=object_pose,
            final_gripper_state=gripper.state,
            plan=plan,
            events=tuple(events),
            created_at=utc_now(),
        )


# ============================================================
# FAILURE RECOVERY ADVISOR
# ============================================================

class PickPlaceRecoveryAdvisor:

    RECOMMENDATIONS = {
        FailureReason.OBJECT_UNAVAILABLE: (
            "Refresh object tracking and verify availability."
        ),
        FailureReason.LOW_DETECTION_CONFIDENCE: (
            "Reacquire the object using improved camera views."
        ),
        FailureReason.PAYLOAD_EXCEEDED: (
            "Use a robot and gripper rated for the payload."
        ),
        FailureReason.GRIPPER_INCOMPATIBLE: (
            "Select a compatible gripper or grasp orientation."
        ),
        FailureReason.INSUFFICIENT_GRIP_FORCE: (
            "Review contact geometry and holding-force limits."
        ),
        FailureReason.TARGET_OCCUPIED: (
            "Select an unobstructed placement location."
        ),
        FailureReason.COLLISION_RISK: (
            "Replan using verified robot geometry and obstacles."
        ),
        FailureReason.UNREACHABLE_POSE: (
            "Select a reachable pose and validate inverse kinematics."
        ),
        FailureReason.GRASP_FAILED: (
            "Re-detect the object and generate alternative grasps."
        ),
        FailureReason.OBJECT_DROPPED: (
            "Stop the task and inspect the object and workspace."
        ),
        FailureReason.SIMULATION_ERROR: (
            "Inspect simulation logs and validate task state."
        ),
        FailureReason.NONE: (
            "No recovery action required."
        ),
    }

    @classmethod
    def recommend(
        cls,
        failure: FailureReason,
    ) -> str:
        return cls.RECOMMENDATIONS[failure]


# ============================================================
# MULTI-TENANT SERVICE
# ============================================================

class PickPlaceService:

    def __init__(
        self,
        config: PickPlaceConfig | None = None,
    ) -> None:
        self.config = config or PickPlaceConfig()

        self.planner = PickPlacePlanner(
            self.config
        )

        self.simulator = PickPlaceSimulator(
            self.config
        )

        self._robots: dict[
            tuple[str, str],
            PickPlaceRobot,
        ] = {}

        self._objects: dict[
            tuple[str, str],
            PickPlaceObject,
        ] = {}

        self._results: dict[
            tuple[str, str],
            PickPlaceResult,
        ] = {}

        self._lock = threading.RLock()

    @staticmethod
    def _authorize(
        access: PickPlaceAccessContext,
        permission: str,
    ) -> None:
        if not getattr(access, permission, False):
            raise PickPlaceAuthorizationError(
                f"Missing permission: {permission}"
            )

    def register_robot(
        self,
        robot: PickPlaceRobot,
        access: PickPlaceAccessContext,
    ) -> None:
        self._authorize(access, "can_manage")

        if robot.tenant_id != access.tenant_id:
            raise PickPlaceAuthorizationError(
                "Cross-tenant robot registration forbidden"
            )

        key = (
            robot.tenant_id,
            robot.robot_id,
        )

        with self._lock:
            if key in self._robots:
                raise PickPlaceConflictError(
                    "Robot already registered"
                )

            self._robots[key] = robot

    def register_object(
        self,
        obj: PickPlaceObject,
        access: PickPlaceAccessContext,
    ) -> None:
        self._authorize(access, "can_manage")

        if obj.tenant_id != access.tenant_id:
            raise PickPlaceAuthorizationError(
                "Cross-tenant object registration forbidden"
            )

        key = (
            obj.tenant_id,
            obj.object_id,
        )

        with self._lock:
            if key in self._objects:
                raise PickPlaceConflictError(
                    "Object already registered"
                )

            self._objects[key] = obj

    def get_robot(
        self,
        robot_id: str,
        access: PickPlaceAccessContext,
    ) -> PickPlaceRobot:
        self._authorize(access, "can_read")

        with self._lock:
            robot = self._robots.get(
                (access.tenant_id, robot_id)
            )

        if robot is None:
            raise PickPlaceNotFoundError(
                "Robot not found"
            )

        return robot

    def get_object(
        self,
        object_id: str,
        access: PickPlaceAccessContext,
    ) -> PickPlaceObject:
        self._authorize(access, "can_read")

        with self._lock:
            obj = self._objects.get(
                (access.tenant_id, object_id)
            )

        if obj is None:
            raise PickPlaceNotFoundError(
                "Object not found"
            )

        return obj

    def plan(
        self,
        request: PickPlaceRequest,
        access: PickPlaceAccessContext,
    ) -> PickPlacePlan:
        self._authorize(access, "can_plan")

        robot = self.get_robot(
            request.robot_id,
            access,
        )

        obj = self.get_object(
            request.object_id,
            access,
        )

        return self.planner.plan(
            robot,
            obj,
            request,
        )

    def simulate(
        self,
        request: PickPlaceRequest,
        access: PickPlaceAccessContext,
        *,
        fail_at_step: PickPlaceStep | None = None,
    ) -> PickPlaceResult:
        self._authorize(access, "can_simulate")
        self._authorize(access, "can_plan")

        with self._lock:
            robot = self.get_robot(
                request.robot_id,
                access,
            )

            obj = self.get_object(
                request.object_id,
                access,
            )

            plan = self.planner.plan(
                robot,
                obj,
                request,
            )

            result = self.simulator.simulate(
                robot,
                obj,
                request,
                plan,
                fail_at_step=fail_at_step,
            )

            if result.status == PickPlaceStatus.COMPLETED:
                self._objects[
                    (access.tenant_id, obj.object_id)
                ] = replace(
                    obj,
                    pose=result.final_object_pose,
                    state=ObjectState.PLACED,
                )

            self._results[
                (access.tenant_id, result.result_id)
            ] = result

            return result

    def get_result(
        self,
        result_id: str,
        access: PickPlaceAccessContext,
    ) -> PickPlaceResult:
        self._authorize(access, "can_read")

        with self._lock:
            result = self._results.get(
                (access.tenant_id, result_id)
            )

        if result is None:
            raise PickPlaceNotFoundError(
                "Result not found"
            )

        return result

    def list_objects(
        self,
        access: PickPlaceAccessContext,
    ) -> tuple[PickPlaceObject, ...]:
        self._authorize(access, "can_read")

        with self._lock:
            return tuple(
                obj
                for (tenant_id, _), obj
                in self._objects.items()
                if tenant_id == access.tenant_id
            )

    def export_json(
        self,
        result: PickPlaceResult,
        access: PickPlaceAccessContext,
    ) -> str:
        self._authorize(access, "can_read")

        if result.tenant_id != access.tenant_id:
            raise PickPlaceAuthorizationError(
                "Cross-tenant result access forbidden"
            )

        return json.dumps(
            json_safe(
                {
                    "result": result,
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

class SyntheticPickPlaceFactory:

    @staticmethod
    def robot(
        tenant_id: str = "physical_ai_lab",
    ) -> PickPlaceRobot:
        return PickPlaceRobot(
            robot_id="robot-arm-001",
            tenant_id=tenant_id,
            gripper=GripperSpecification(
                gripper_id="gripper-001",
                gripper_type=GripperType.PARALLEL_JAW,
                minimum_opening_m=0.005,
                maximum_opening_m=0.12,
                maximum_payload_kg=5.0,
                maximum_normal_force_n=80.0,
                safety_factor=2.0,
            ),
            maximum_payload_kg=5.0,
            maximum_reach_m=1.5,
            home_pose=Pose3D(
                position_xyz=(0.30, 0.0, 0.50)
            ),
            workspace=WorkspaceBounds(
                minimum_xyz=(-1.0, -1.0, 0.0),
                maximum_xyz=(1.0, 1.0, 1.5),
            ),
        )

    @staticmethod
    def object(
        tenant_id: str = "physical_ai_lab",
    ) -> PickPlaceObject:
        return PickPlaceObject(
            object_id="component-001",
            tenant_id=tenant_id,
            name="Industrial Component",
            pose=Pose3D(
                position_xyz=(0.45, 0.0, 0.15)
            ),
            dimensions=ObjectDimensions(
                length_m=0.10,
                width_m=0.07,
                height_m=0.08,
            ),
            mass_kg=0.8,
            friction_coefficient=0.6,
            detection_confidence=0.98,
        )

    @staticmethod
    def request() -> PickPlaceRequest:
        return PickPlaceRequest(
            request_id=new_id(),
            robot_id="robot-arm-001",
            object_id="component-001",
            destination_pose=Pose3D(
                position_xyz=(0.55, 0.25, 0.15)
            ),
        )

    @staticmethod
    def access(
        tenant_id: str = "physical_ai_lab",
    ) -> PickPlaceAccessContext:
        return PickPlaceAccessContext(
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

class TestPickAndPlace(unittest.TestCase):

    def setUp(self) -> None:
        self.robot = SyntheticPickPlaceFactory.robot()
        self.obj = SyntheticPickPlaceFactory.object()
        self.request = SyntheticPickPlaceFactory.request()
        self.access = SyntheticPickPlaceFactory.access()

        self.planner = PickPlacePlanner()

    def test_pose_translation(self) -> None:
        translated = self.obj.pose.translated(
            (0.0, 0.0, 0.1)
        )

        self.assertAlmostEqual(
            translated.position_xyz[2],
            0.25,
        )

    def test_object_dimensions(self) -> None:
        self.assertGreater(
            self.obj.dimensions.bounding_radius_m,
            0,
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

    def test_grasp_feasible(self) -> None:
        result = PickPlaceGraspEvaluator().evaluate(
            self.robot,
            self.obj,
        )

        self.assertTrue(result.feasible)

    def test_grasp_payload_failure(self) -> None:
        heavy = replace(
            self.obj,
            mass_kg=100.0,
        )

        result = PickPlaceGraspEvaluator().evaluate(
            self.robot,
            heavy,
        )

        self.assertFalse(result.feasible)

        self.assertEqual(
            result.failure_reason,
            FailureReason.PAYLOAD_EXCEEDED,
        )

    def test_gripper_width_failure(self) -> None:
        result = PickPlaceGraspEvaluator().evaluate(
            self.robot,
            self.obj,
            requested_width_m=0.5,
        )

        self.assertFalse(result.feasible)

    def test_collision_checker(self) -> None:
        checker = PickPlaceCollisionChecker(
            PickPlaceConfig()
        )

        obstacle = SphericalObstacle(
            obstacle_id="obstacle",
            center_xyz=(0.0, 0.0, 0.0),
            radius_m=0.1,
        )

        self.assertTrue(
            checker.point_collision(
                (0.0, 0.0, 0.0),
                (obstacle,),
            )
        )

    def test_clear_collision_checker(self) -> None:
        checker = PickPlaceCollisionChecker(
            PickPlaceConfig()
        )

        self.assertFalse(
            checker.point_collision(
                (0.0, 0.0, 0.0),
                (),
            )
        )

    def test_waypoint_generation(self) -> None:
        waypoints = PickPlaceWaypointGenerator(
            PickPlaceConfig()
        ).generate(
            self.robot,
            self.obj,
            self.request,
        )

        self.assertEqual(
            len(waypoints),
            10,
        )

    def test_planning(self) -> None:
        plan = self.planner.plan(
            self.robot,
            self.obj,
            self.request,
        )

        self.assertGreater(
            plan.total_distance_m,
            0.0,
        )

    def test_plan_duration(self) -> None:
        plan = self.planner.plan(
            self.robot,
            self.obj,
            self.request,
        )

        self.assertGreater(
            plan.estimated_duration_s,
            0.0,
        )

    def test_low_detection_confidence(self) -> None:
        low_confidence = replace(
            self.obj,
            detection_confidence=0.1,
        )

        with self.assertRaises(
            PickPlacePlanningError
        ):
            self.planner.plan(
                self.robot,
                low_confidence,
                self.request,
            )

    def test_unavailable_object(self) -> None:
        unavailable = replace(
            self.obj,
            state=ObjectState.ATTACHED,
        )

        with self.assertRaises(
            PickPlacePlanningError
        ):
            self.planner.plan(
                self.robot,
                unavailable,
                self.request,
            )

    def test_gripper_close(self) -> None:
        gripper = SimulatedGripper(
            self.robot.gripper
        )

        gripper.close()

        self.assertEqual(
            gripper.state,
            GripperState.CLOSED,
        )

    def test_gripper_attach(self) -> None:
        gripper = SimulatedGripper(
            self.robot.gripper
        )

        gripper.close()
        gripper.attach(self.obj.object_id)

        self.assertEqual(
            gripper.state,
            GripperState.HOLDING,
        )

    def test_gripper_release(self) -> None:
        gripper = SimulatedGripper(
            self.robot.gripper
        )

        gripper.close()
        gripper.attach(self.obj.object_id)
        gripper.release()

        self.assertEqual(
            gripper.state,
            GripperState.OPEN,
        )

    def test_state_machine(self) -> None:
        machine = PickPlaceStateMachine()

        machine.transition(
            PickPlaceStatus.PLANNED
        )

        machine.transition(
            PickPlaceStatus.RUNNING
        )

        machine.transition(
            PickPlaceStatus.COMPLETED
        )

        self.assertEqual(
            machine.status,
            PickPlaceStatus.COMPLETED,
        )

    def test_invalid_state_transition(self) -> None:
        machine = PickPlaceStateMachine()

        with self.assertRaises(
            PickPlaceStateError
        ):
            machine.transition(
                PickPlaceStatus.COMPLETED
            )

    def test_simulation(self) -> None:
        plan = self.planner.plan(
            self.robot,
            self.obj,
            self.request,
        )

        result = PickPlaceSimulator().simulate(
            self.robot,
            self.obj,
            self.request,
            plan,
        )

        self.assertEqual(
            result.status,
            PickPlaceStatus.COMPLETED,
        )

    def test_final_object_pose(self) -> None:
        plan = self.planner.plan(
            self.robot,
            self.obj,
            self.request,
        )

        result = PickPlaceSimulator().simulate(
            self.robot,
            self.obj,
            self.request,
            plan,
        )

        self.assertEqual(
            result.final_object_pose,
            self.request.destination_pose,
        )

    def test_failure_injection(self) -> None:
        plan = self.planner.plan(
            self.robot,
            self.obj,
            self.request,
        )

        result = PickPlaceSimulator().simulate(
            self.robot,
            self.obj,
            self.request,
            plan,
            fail_at_step=PickPlaceStep.VERIFY_GRASP,
        )

        self.assertEqual(
            result.status,
            PickPlaceStatus.FAILED,
        )

    def test_recovery_advice(self) -> None:
        advice = PickPlaceRecoveryAdvisor.recommend(
            FailureReason.GRASP_FAILED
        )

        self.assertTrue(advice)

    def test_service_registration(self) -> None:
        service = PickPlaceService()

        service.register_robot(
            self.robot,
            self.access,
        )

        service.register_object(
            self.obj,
            self.access,
        )

        self.assertEqual(
            service.get_robot(
                self.robot.robot_id,
                self.access,
            ).robot_id,
            self.robot.robot_id,
        )

    def test_service_plan(self) -> None:
        service = PickPlaceService()

        service.register_robot(
            self.robot,
            self.access,
        )

        service.register_object(
            self.obj,
            self.access,
        )

        plan = service.plan(
            self.request,
            self.access,
        )

        self.assertEqual(
            plan.object_id,
            self.obj.object_id,
        )

    def test_service_simulate(self) -> None:
        service = PickPlaceService()

        service.register_robot(
            self.robot,
            self.access,
        )

        service.register_object(
            self.obj,
            self.access,
        )

        result = service.simulate(
            self.request,
            self.access,
        )

        self.assertEqual(
            result.status,
            PickPlaceStatus.COMPLETED,
        )

    def test_service_updates_object(self) -> None:
        service = PickPlaceService()

        service.register_robot(
            self.robot,
            self.access,
        )

        service.register_object(
            self.obj,
            self.access,
        )

        service.simulate(
            self.request,
            self.access,
        )

        updated = service.get_object(
            self.obj.object_id,
            self.access,
        )

        self.assertEqual(
            updated.state,
            ObjectState.PLACED,
        )

    def test_tenant_isolation(self) -> None:
        service = PickPlaceService()

        service.register_robot(
            self.robot,
            self.access,
        )

        other = PickPlaceAccessContext(
            tenant_id="other_tenant",
            user_id="other_user",
            can_read=True,
        )

        with self.assertRaises(
            PickPlaceNotFoundError
        ):
            service.get_robot(
                self.robot.robot_id,
                other,
            )

    def test_permission_denied(self) -> None:
        service = PickPlaceService()

        service.register_robot(
            self.robot,
            self.access,
        )

        viewer = PickPlaceAccessContext(
            tenant_id=self.robot.tenant_id,
            user_id="viewer",
            can_read=True,
        )

        with self.assertRaises(
            PickPlaceAuthorizationError
        ):
            service.plan(
                self.request,
                viewer,
            )

    def test_json_export(self) -> None:
        service = PickPlaceService()

        service.register_robot(
            self.robot,
            self.access,
        )

        service.register_object(
            self.obj,
            self.access,
        )

        result = service.simulate(
            self.request,
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
            exported["physical_actuation"]
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("\nPHYSICAL AI — PICK AND PLACE")
    print("=" * 72)

    robot = SyntheticPickPlaceFactory.robot()
    obj = SyntheticPickPlaceFactory.object()
    request = SyntheticPickPlaceFactory.request()
    access = SyntheticPickPlaceFactory.access()

    service = PickPlaceService()

    service.register_robot(
        robot,
        access,
    )

    service.register_object(
        obj,
        access,
    )

    plan = service.plan(
        request,
        access,
    )

    print("Robot:", robot.robot_id)
    print("Object:", obj.name)
    print("Object mass:", obj.mass_kg, "kg")

    print(
        "Grasp feasible:",
        plan.grasp_assessment.feasible,
    )

    print(
        "Grasp quality:",
        round(
            plan.grasp_assessment.quality_score,
            4,
        ),
    )

    print(
        "Waypoints:",
        len(plan.waypoints),
    )

    print(
        "Estimated travel:",
        round(plan.total_distance_m, 4),
        "m",
    )

    print(
        "Estimated duration:",
        round(plan.estimated_duration_s, 3),
        "s",
    )

    result = service.simulate(
        request,
        access,
    )

    print(
        "Simulation status:",
        result.status.value,
    )

    print(
        "Completed steps:",
        result.completed_steps,
    )

    print(
        "Final object position:",
        result.final_object_pose.position_xyz,
    )

    print("\nRESULT JSON")
    print("=" * 72)

    print(
        service.export_json(
            result,
            access,
        )
    )

    print("\nUNIT TESTS")
    print("=" * 72)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestPickAndPlace
    )

    test_result = unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)

    if not test_result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
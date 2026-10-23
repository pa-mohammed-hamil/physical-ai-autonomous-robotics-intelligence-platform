"""Object Manipulation module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/manipulation/object_manipulation.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================
Module: Object Manipulation

Simulation-only object manipulation planning and evaluation.

Features:
    - Object detection input validation
    - Object geometry and physical properties
    - Pick-and-place task planning
    - Grasp candidate evaluation
    - Gripper compatibility checks
    - Payload and friction feasibility estimates
    - Approach and retreat waypoint generation
    - Joint-space manipulation trajectory planning
    - Simulated grasp attachment and release
    - Object pose tracking
    - Manipulation state machine
    - Collision-aware object placement
    - Manipulation quality scoring
    - Task execution simulation
    - Failure handling and recovery planning
    - Multi-tenant robot management
    - Audit events and task history
    - JSON serialization
    - Synthetic demonstrations
    - Unit tests

Dependencies:
    Python >= 3.11
    numpy >= 1.24

Install:
    pip install numpy

SAFETY:
    Planning and simulation only.

    No physical robot actuation is implemented.
    Grasp feasibility is approximate.
    Collision checks use simplified spherical geometry.
    No safety certification is implied.
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
from typing import Any, Sequence

import numpy as np


# ============================================================
# EXCEPTIONS
# ============================================================

class ObjectManipulationError(Exception):
    pass


class ManipulationValidationError(ObjectManipulationError):
    pass


class ManipulationAuthorizationError(ObjectManipulationError):
    pass


class ManipulationNotFoundError(ObjectManipulationError):
    pass


class ManipulationConflictError(ObjectManipulationError):
    pass


class ManipulationStateError(ObjectManipulationError):
    pass


class ManipulationPlanningError(ObjectManipulationError):
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
        raise ManipulationValidationError(
            f"Invalid {name}"
        )

    return value.strip()


def finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ManipulationValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ManipulationValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise ManipulationValidationError(
            f"{name} must be finite"
        )

    return result


def positive(value: Any, name: str) -> float:
    result = finite(value, name)

    if result <= 0:
        raise ManipulationValidationError(
            f"{name} must be positive"
        )

    return result


def vector3(
    values: Sequence[float],
    name: str,
) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ManipulationValidationError(
            f"{name} must be numeric"
        ) from exc

    if result.shape != (3,):
        raise ManipulationValidationError(
            f"{name} must have three components"
        )

    if not np.all(np.isfinite(result)):
        raise ManipulationValidationError(
            f"{name} contains non-finite values"
        )

    return result


def normalize(
    values: Sequence[float],
    name: str = "direction",
) -> np.ndarray:
    result = vector3(values, name)

    norm = float(np.linalg.norm(result))

    if norm <= 1e-12:
        raise ManipulationValidationError(
            f"{name} cannot be zero"
        )

    return result / norm


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(minimum, min(maximum, value))


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

class ObjectShape(str, Enum):
    BOX = "box"
    CYLINDER = "cylinder"
    SPHERE = "sphere"
    IRREGULAR = "irregular"


class ObjectState(str, Enum):
    DETECTED = "detected"
    AVAILABLE = "available"
    RESERVED = "reserved"
    ATTACHED = "attached"
    PLACED = "placed"
    LOST = "lost"


class GripperType(str, Enum):
    PARALLEL_JAW = "parallel_jaw"
    SUCTION = "suction"


class GraspType(str, Enum):
    TOP = "top"
    SIDE = "side"
    PINCH = "pinch"
    SUCTION = "suction"


class ManipulationAction(str, Enum):
    MOVE_TO_PREGRASP = "move_to_pregrasp"
    APPROACH = "approach"
    CLOSE_GRIPPER = "close_gripper"
    VERIFY_GRASP = "verify_grasp"
    LIFT = "lift"
    TRANSPORT = "transport"
    LOWER = "lower"
    OPEN_GRIPPER = "open_gripper"
    RETREAT = "retreat"


class ManipulationStatus(str, Enum):
    CREATED = "created"
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ManipulationFailure(str, Enum):
    NONE = "none"
    OBJECT_NOT_FOUND = "object_not_found"
    GRASP_NOT_FEASIBLE = "grasp_not_feasible"
    PAYLOAD_EXCEEDED = "payload_exceeded"
    COLLISION_RISK = "collision_risk"
    TARGET_OCCUPIED = "target_occupied"
    SIMULATION_FAILURE = "simulation_failure"


# ============================================================
# GEOMETRY AND OBJECT MODELS
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
        position = (
            vector3(self.position_xyz, "position")
            + vector3(offset, "offset")
        )

        return Pose3D(
            position_xyz=tuple(
                float(value) for value in position
            ),
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
    def volume_m3(self) -> float:
        return (
            self.length_m
            * self.width_m
            * self.height_m
        )

    @property
    def bounding_radius_m(self) -> float:
        return 0.5 * math.sqrt(
            self.length_m**2
            + self.width_m**2
            + self.height_m**2
        )


@dataclass(frozen=True)
class ManipulationObject:
    object_id: str
    tenant_id: str

    name: str
    shape: ObjectShape

    pose: Pose3D
    dimensions: ObjectDimensions

    mass_kg: float

    friction_coefficient: float = 0.5
    detection_confidence: float = 1.0

    state: ObjectState = ObjectState.AVAILABLE

    def __post_init__(self) -> None:
        identifier(self.object_id, "object_id")
        identifier(self.tenant_id, "tenant_id")
        identifier(self.name, "name")

        positive(self.mass_kg, "mass_kg")

        if finite(
            self.friction_coefficient,
            "friction_coefficient",
        ) <= 0:
            raise ManipulationValidationError(
                "Friction coefficient must be positive"
            )

        if not 0 <= self.detection_confidence <= 1:
            raise ManipulationValidationError(
                "Detection confidence must be in [0,1]"
            )


@dataclass(frozen=True)
class GripperConfiguration:
    gripper_id: str
    gripper_type: GripperType

    minimum_opening_m: float = 0.005
    maximum_opening_m: float = 0.12

    maximum_payload_kg: float = 5.0
    maximum_normal_force_n: float = 80.0

    suction_force_n: float = 50.0

    safety_factor: float = 2.0

    def __post_init__(self) -> None:
        identifier(self.gripper_id, "gripper_id")

        if self.minimum_opening_m < 0:
            raise ManipulationValidationError(
                "Minimum opening cannot be negative"
            )

        if (
            self.maximum_opening_m
            <= self.minimum_opening_m
        ):
            raise ManipulationValidationError(
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
            self.suction_force_n,
            "suction_force_n",
        )

        if self.safety_factor < 1:
            raise ManipulationValidationError(
                "Safety factor must be at least 1"
            )


@dataclass(frozen=True)
class ManipulationRobot:
    robot_id: str
    tenant_id: str

    gripper: GripperConfiguration

    maximum_reach_m: float = 1.5
    maximum_payload_kg: float = 5.0

    home_pose: Pose3D = field(
        default_factory=lambda: Pose3D(
            (0.3, 0.0, 0.5)
        )
    )

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


@dataclass(frozen=True)
class ManipulationObstacle:
    obstacle_id: str
    center_xyz: tuple[float, float, float]
    radius_m: float

    def __post_init__(self) -> None:
        identifier(self.obstacle_id, "obstacle_id")
        vector3(self.center_xyz, "center_xyz")
        positive(self.radius_m, "radius_m")


# ============================================================
# GRASP MODELS
# ============================================================

@dataclass(frozen=True)
class GraspCandidate:
    candidate_id: str
    object_id: str

    grasp_type: GraspType

    grasp_pose: Pose3D

    opening_width_m: float

    approach_direction: tuple[float, float, float]

    confidence: float

    def __post_init__(self) -> None:
        identifier(self.candidate_id, "candidate_id")
        identifier(self.object_id, "object_id")

        if self.opening_width_m < 0:
            raise ManipulationValidationError(
                "Grasp opening cannot be negative"
            )

        normalize(
            self.approach_direction,
            "approach_direction",
        )

        if not 0 <= self.confidence <= 1:
            raise ManipulationValidationError(
                "Grasp confidence must be in [0,1]"
            )


@dataclass(frozen=True)
class GraspEvaluation:
    candidate_id: str

    feasible: bool

    payload_score: float
    geometry_score: float
    friction_score: float
    confidence_score: float

    overall_score: float

    failure: ManipulationFailure

    message: str


# ============================================================
# MANIPULATION TASK MODELS
# ============================================================

@dataclass(frozen=True)
class ManipulationRequest:
    request_id: str
    object_id: str

    destination_pose: Pose3D

    obstacles: tuple[ManipulationObstacle, ...] = ()

    preferred_grasp_type: GraspType | None = None

    approach_distance_m: float = 0.12
    lift_height_m: float = 0.15
    retreat_distance_m: float = 0.12

    def __post_init__(self) -> None:
        identifier(self.request_id, "request_id")
        identifier(self.object_id, "object_id")

        positive(
            self.approach_distance_m,
            "approach_distance_m",
        )

        positive(
            self.lift_height_m,
            "lift_height_m",
        )

        positive(
            self.retreat_distance_m,
            "retreat_distance_m",
        )


@dataclass(frozen=True)
class ManipulationWaypoint:
    waypoint_id: str

    action: ManipulationAction

    pose: Pose3D

    gripper_closed: bool

    object_attached: bool


@dataclass(frozen=True)
class ManipulationPlan:
    plan_id: str
    request_id: str

    robot_id: str
    object_id: str

    selected_grasp: GraspCandidate

    grasp_evaluation: GraspEvaluation

    waypoints: tuple[ManipulationWaypoint, ...]

    estimated_distance_m: float

    created_at: datetime


@dataclass(frozen=True)
class ManipulationEvent:
    event_id: str
    task_id: str

    action: ManipulationAction | None

    status: ManipulationStatus

    message: str

    timestamp: datetime


@dataclass(frozen=True)
class ManipulationResult:
    result_id: str
    task_id: str

    robot_id: str
    object_id: str
    tenant_id: str

    status: ManipulationStatus

    failure: ManipulationFailure

    plan: ManipulationPlan | None

    final_object_pose: Pose3D | None

    completed_actions: int

    events: tuple[ManipulationEvent, ...]

    created_at: datetime


@dataclass(frozen=True)
class ManipulationAccessContext:
    tenant_id: str
    user_id: str

    can_read: bool = False
    can_plan: bool = False
    can_simulate: bool = False
    can_manage: bool = False


# ============================================================
# OBJECT GEOMETRY ANALYZER
# ============================================================

class ObjectGeometryAnalyzer:

    @staticmethod
    def approximate_grasp_widths(
        obj: ManipulationObject,
    ) -> tuple[float, ...]:
        dimensions = obj.dimensions

        if obj.shape == ObjectShape.SPHERE:
            diameter = max(
                dimensions.length_m,
                dimensions.width_m,
                dimensions.height_m,
            )

            return (diameter,)

        if obj.shape == ObjectShape.CYLINDER:
            return (
                max(
                    dimensions.length_m,
                    dimensions.width_m,
                ),
            )

        return (
            dimensions.width_m,
            dimensions.length_m,
        )

    @staticmethod
    def top_surface_pose(
        obj: ManipulationObject,
    ) -> Pose3D:
        return obj.pose.translated(
            (
                0.0,
                0.0,
                obj.dimensions.height_m / 2.0,
            )
        )

    @staticmethod
    def side_surface_pose(
        obj: ManipulationObject,
    ) -> Pose3D:
        return obj.pose.translated(
            (
                0.0,
                -obj.dimensions.width_m / 2.0,
                0.0,
            )
        )


# ============================================================
# GRASP CANDIDATE GENERATOR
# ============================================================

class ObjectGraspGenerator:

    def generate(
        self,
        obj: ManipulationObject,
        gripper: GripperConfiguration,
    ) -> tuple[GraspCandidate, ...]:
        candidates: list[GraspCandidate] = []

        widths = (
            ObjectGeometryAnalyzer.approximate_grasp_widths(
                obj
            )
        )

        if gripper.gripper_type == GripperType.SUCTION:
            candidates.append(
                GraspCandidate(
                    candidate_id=new_id(),
                    object_id=obj.object_id,
                    grasp_type=GraspType.SUCTION,
                    grasp_pose=(
                        ObjectGeometryAnalyzer.top_surface_pose(
                            obj
                        )
                    ),
                    opening_width_m=0.0,
                    approach_direction=(0.0, 0.0, -1.0),
                    confidence=(
                        0.85
                        * obj.detection_confidence
                    ),
                )
            )

            return tuple(candidates)

        for width in widths:
            candidates.append(
                GraspCandidate(
                    candidate_id=new_id(),
                    object_id=obj.object_id,
                    grasp_type=GraspType.TOP,
                    grasp_pose=(
                        ObjectGeometryAnalyzer.top_surface_pose(
                            obj
                        )
                    ),
                    opening_width_m=width,
                    approach_direction=(0.0, 0.0, -1.0),
                    confidence=(
                        0.90
                        * obj.detection_confidence
                    ),
                )
            )

        candidates.append(
            GraspCandidate(
                candidate_id=new_id(),
                object_id=obj.object_id,
                grasp_type=GraspType.SIDE,
                grasp_pose=(
                    ObjectGeometryAnalyzer.side_surface_pose(
                        obj
                    )
                ),
                opening_width_m=obj.dimensions.width_m,
                approach_direction=(0.0, 1.0, 0.0),
                confidence=(
                    0.75
                    * obj.detection_confidence
                ),
            )
        )

        return tuple(candidates)


# ============================================================
# GRASP FEASIBILITY EVALUATOR
# ============================================================

class ObjectGraspEvaluator:

    GRAVITY = 9.81

    def evaluate(
        self,
        obj: ManipulationObject,
        robot: ManipulationRobot,
        candidate: GraspCandidate,
    ) -> GraspEvaluation:
        gripper = robot.gripper

        maximum_payload = min(
            robot.maximum_payload_kg,
            gripper.maximum_payload_kg,
        )

        payload_feasible = (
            obj.mass_kg * gripper.safety_factor
            <= maximum_payload
        )

        payload_score = clamp(
            maximum_payload
            / max(
                obj.mass_kg * gripper.safety_factor,
                1e-9,
            ),
            0.0,
            1.0,
        )

        if gripper.gripper_type == GripperType.PARALLEL_JAW:
            geometry_feasible = (
                gripper.minimum_opening_m
                <= candidate.opening_width_m
                <= gripper.maximum_opening_m
            )

            geometry_score = (
                1.0 if geometry_feasible else 0.0
            )

            # Simplified static two-contact friction estimate.
            # Assumes the force is distributed equally across
            # two opposing fingers.
            available_friction_force = (
                2.0
                * obj.friction_coefficient
                * gripper.maximum_normal_force_n
            )

            required_force = (
                obj.mass_kg
                * self.GRAVITY
                * gripper.safety_factor
            )

            friction_score = clamp(
                available_friction_force
                / max(required_force, 1e-9),
                0.0,
                1.0,
            )

            friction_feasible = (
                available_friction_force
                >= required_force
            )

            compatible_type = (
                candidate.grasp_type
                != GraspType.SUCTION
            )

        else:
            geometry_feasible = (
                candidate.grasp_type
                == GraspType.SUCTION
            )

            geometry_score = (
                1.0 if geometry_feasible else 0.0
            )

            required_force = (
                obj.mass_kg
                * self.GRAVITY
                * gripper.safety_factor
            )

            friction_score = clamp(
                gripper.suction_force_n
                / max(required_force, 1e-9),
                0.0,
                1.0,
            )

            friction_feasible = (
                gripper.suction_force_n
                >= required_force
            )

            compatible_type = geometry_feasible

        feasible = (
            payload_feasible
            and geometry_feasible
            and friction_feasible
            and compatible_type
        )

        overall_score = (
            0.25 * payload_score
            + 0.25 * geometry_score
            + 0.25 * friction_score
            + 0.25 * candidate.confidence
        )

        if not payload_feasible:
            failure = ManipulationFailure.PAYLOAD_EXCEEDED
            message = "Robot or gripper payload exceeded"

        elif not geometry_feasible or not compatible_type:
            failure = ManipulationFailure.GRASP_NOT_FEASIBLE
            message = "Grasp geometry incompatible with gripper"

        elif not friction_feasible:
            failure = ManipulationFailure.GRASP_NOT_FEASIBLE
            message = "Insufficient estimated holding force"

        else:
            failure = ManipulationFailure.NONE
            message = "Grasp passed approximate feasibility checks"

        return GraspEvaluation(
            candidate_id=candidate.candidate_id,
            feasible=feasible,
            payload_score=payload_score,
            geometry_score=geometry_score,
            friction_score=friction_score,
            confidence_score=candidate.confidence,
            overall_score=overall_score,
            failure=failure,
            message=message,
        )


# ============================================================
# GRASP SELECTOR
# ============================================================

class ObjectGraspSelector:

    def __init__(self) -> None:
        self.generator = ObjectGraspGenerator()
        self.evaluator = ObjectGraspEvaluator()

    def select(
        self,
        obj: ManipulationObject,
        robot: ManipulationRobot,
        preferred_type: GraspType | None = None,
    ) -> tuple[GraspCandidate, GraspEvaluation]:
        candidates = self.generator.generate(
            obj,
            robot.gripper,
        )

        evaluations: list[
            tuple[GraspCandidate, GraspEvaluation]
        ] = []

        for candidate in candidates:
            if (
                preferred_type is not None
                and candidate.grasp_type != preferred_type
            ):
                continue

            evaluation = self.evaluator.evaluate(
                obj,
                robot,
                candidate,
            )

            if evaluation.feasible:
                evaluations.append(
                    (candidate, evaluation)
                )

        if not evaluations:
            raise ManipulationPlanningError(
                "No feasible grasp candidate found"
            )

        evaluations.sort(
            key=lambda item: item[1].overall_score,
            reverse=True,
        )

        return evaluations[0]


# ============================================================
# COLLISION CHECKER
# ============================================================

class ObjectManipulationCollisionChecker:

    def __init__(
        self,
        clearance_m: float = 0.02,
        sampling_resolution_m: float = 0.025,
    ) -> None:
        if clearance_m < 0:
            raise ManipulationValidationError(
                "Clearance cannot be negative"
            )

        positive(
            sampling_resolution_m,
            "sampling_resolution_m",
        )

        self.clearance_m = clearance_m
        self.sampling_resolution_m = sampling_resolution_m

    def point_collision(
        self,
        position: Sequence[float],
        obstacles: Sequence[ManipulationObstacle],
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
                + self.clearance_m
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
        obstacles: Sequence[ManipulationObstacle],
        payload_radius_m: float = 0.0,
    ) -> bool:
        first = vector3(start, "start")
        second = vector3(end, "end")

        length = float(
            np.linalg.norm(second - first)
        )

        samples = max(
            1,
            int(
                math.ceil(
                    length / self.sampling_resolution_m
                )
            ),
        )

        for index in range(samples + 1):
            alpha = index / samples

            position = (
                (1.0 - alpha) * first
                + alpha * second
            )

            if self.point_collision(
                position,
                obstacles,
                payload_radius_m,
            ):
                return True

        return False


# ============================================================
# WAYPOINT PLANNER
# ============================================================

class ObjectWaypointPlanner:

    def __init__(
        self,
        collision_checker: (
            ObjectManipulationCollisionChecker
        ) | None = None,
    ) -> None:
        self.collision_checker = (
            collision_checker
            or ObjectManipulationCollisionChecker()
        )

    @staticmethod
    def _waypoint(
        action: ManipulationAction,
        pose: Pose3D,
        gripper_closed: bool,
        object_attached: bool,
    ) -> ManipulationWaypoint:
        return ManipulationWaypoint(
            waypoint_id=new_id(),
            action=action,
            pose=pose,
            gripper_closed=gripper_closed,
            object_attached=object_attached,
        )

    def generate(
        self,
        request: ManipulationRequest,
        obj: ManipulationObject,
        robot: ManipulationRobot,
        grasp: GraspCandidate,
    ) -> tuple[ManipulationWaypoint, ...]:
        approach_direction = normalize(
            grasp.approach_direction,
            "approach_direction",
        )

        grasp_position = vector3(
            grasp.grasp_pose.position_xyz,
            "grasp position",
        )

        pregrasp_position = (
            grasp_position
            - approach_direction
            * request.approach_distance_m
        )

        lift_position = (
            grasp_position
            + np.array(
                [0.0, 0.0, request.lift_height_m]
            )
        )

        # Preserve the relative translation between the
        # object center and the grasp frame.
        object_center = vector3(
            obj.pose.position_xyz,
            "object center",
        )

        grasp_offset = (
            grasp_position - object_center
        )

        destination_center = vector3(
            request.destination_pose.position_xyz,
            "destination",
        )

        destination_grasp_position = (
            destination_center + grasp_offset
        )

        transport_position = (
            destination_grasp_position
            + np.array(
                [0.0, 0.0, request.lift_height_m]
            )
        )

        retreat_position = (
            destination_grasp_position
            - approach_direction
            * request.retreat_distance_m
        )

        def pose_at(position: np.ndarray) -> Pose3D:
            return Pose3D(
                position_xyz=tuple(
                    float(value) for value in position
                ),
                orientation_rpy=(
                    grasp.grasp_pose.orientation_rpy
                ),
            )

        waypoints = (
            self._waypoint(
                ManipulationAction.MOVE_TO_PREGRASP,
                pose_at(pregrasp_position),
                False,
                False,
            ),
            self._waypoint(
                ManipulationAction.APPROACH,
                grasp.grasp_pose,
                False,
                False,
            ),
            self._waypoint(
                ManipulationAction.CLOSE_GRIPPER,
                grasp.grasp_pose,
                True,
                False,
            ),
            self._waypoint(
                ManipulationAction.VERIFY_GRASP,
                grasp.grasp_pose,
                True,
                True,
            ),
            self._waypoint(
                ManipulationAction.LIFT,
                pose_at(lift_position),
                True,
                True,
            ),
            self._waypoint(
                ManipulationAction.TRANSPORT,
                pose_at(transport_position),
                True,
                True,
            ),
            self._waypoint(
                ManipulationAction.LOWER,
                pose_at(destination_grasp_position),
                True,
                True,
            ),
            self._waypoint(
                ManipulationAction.OPEN_GRIPPER,
                pose_at(destination_grasp_position),
                False,
                False,
            ),
            self._waypoint(
                ManipulationAction.RETREAT,
                pose_at(retreat_position),
                False,
                False,
            ),
        )

        for waypoint in waypoints:
            if (
                distance3(
                    robot.home_pose.position_xyz,
                    waypoint.pose.position_xyz,
                )
                > robot.maximum_reach_m
            ):
                raise ManipulationPlanningError(
                    f"Waypoint exceeds approximate reach: "
                    f"{waypoint.action.value}"
                )

        # Check the tool path using a simplified point or
        # payload sphere. This is not full-arm collision
        # checking and does not replace motion_planner.py.
        for index, waypoint in enumerate(waypoints):
            if index == 0:
                continue

            previous = waypoints[index - 1]

            if (
                waypoint.pose.position_xyz
                == previous.pose.position_xyz
            ):
                continue

            payload_radius = (
                obj.dimensions.bounding_radius_m
                if previous.object_attached
                and waypoint.object_attached
                else 0.0
            )

            if self.collision_checker.segment_collision(
                previous.pose.position_xyz,
                waypoint.pose.position_xyz,
                request.obstacles,
                payload_radius,
            ):
                raise ManipulationPlanningError(
                    "Approximate collision detected during "
                    f"{waypoint.action.value}"
                )

        return waypoints


# ============================================================
# MANIPULATION PLAN BUILDER
# ============================================================

class ObjectManipulationPlanner:

    def __init__(self) -> None:
        self.grasp_selector = ObjectGraspSelector()

        self.waypoint_planner = ObjectWaypointPlanner()

    def plan(
        self,
        request: ManipulationRequest,
        obj: ManipulationObject,
        robot: ManipulationRobot,
    ) -> ManipulationPlan:
        if obj.object_id != request.object_id:
            raise ManipulationValidationError(
                "Request object mismatch"
            )

        if obj.tenant_id != robot.tenant_id:
            raise ManipulationAuthorizationError(
                "Object and robot tenants differ"
            )

        if obj.state not in (
            ObjectState.AVAILABLE,
            ObjectState.DETECTED,
            ObjectState.RESERVED,
        ):
            raise ManipulationStateError(
                "Object is not available for manipulation"
            )

        grasp, evaluation = (
            self.grasp_selector.select(
                obj,
                robot,
                request.preferred_grasp_type,
            )
        )

        waypoints = self.waypoint_planner.generate(
            request,
            obj,
            robot,
            grasp,
        )

        estimated_distance = sum(
            distance3(
                waypoints[index].pose.position_xyz,
                waypoints[index + 1].pose.position_xyz,
            )
            for index in range(len(waypoints) - 1)
        )

        return ManipulationPlan(
            plan_id=new_id(),
            request_id=request.request_id,
            robot_id=robot.robot_id,
            object_id=obj.object_id,
            selected_grasp=grasp,
            grasp_evaluation=evaluation,
            waypoints=waypoints,
            estimated_distance_m=estimated_distance,
            created_at=utc_now(),
        )


# ============================================================
# MANIPULATION SIMULATOR
# ============================================================

class ObjectManipulationSimulator:

    def simulate(
        self,
        plan: ManipulationPlan,
        obj: ManipulationObject,
        robot: ManipulationRobot,
        destination_pose: Pose3D,
    ) -> ManipulationResult:
        task_id = new_id()

        events: list[ManipulationEvent] = []

        attached = False
        gripper_closed = False

        completed_actions = 0

        current_pose = obj.pose

        status = ManipulationStatus.RUNNING

        events.append(
            ManipulationEvent(
                event_id=new_id(),
                task_id=task_id,
                action=None,
                status=status,
                message="Manipulation simulation started",
                timestamp=utc_now(),
            )
        )

        grasp_position = vector3(
            plan.selected_grasp.grasp_pose.position_xyz,
            "grasp position",
        )

        object_position = vector3(
            obj.pose.position_xyz,
            "object position",
        )

        object_offset = (
            object_position - grasp_position
        )

        for waypoint in plan.waypoints:
            action = waypoint.action

            if action == ManipulationAction.CLOSE_GRIPPER:
                gripper_closed = True

            elif action == ManipulationAction.VERIFY_GRASP:
                if (
                    not gripper_closed
                    or not plan.grasp_evaluation.feasible
                ):
                    status = ManipulationStatus.FAILED

                    events.append(
                        ManipulationEvent(
                            event_id=new_id(),
                            task_id=task_id,
                            action=action,
                            status=status,
                            message="Simulated grasp verification failed",
                            timestamp=utc_now(),
                        )
                    )

                    break

                attached = True

            elif action == ManipulationAction.OPEN_GRIPPER:
                attached = False
                gripper_closed = False

                current_pose = destination_pose

            elif attached:
                tool_position = vector3(
                    waypoint.pose.position_xyz,
                    "tool position",
                )

                new_object_position = (
                    tool_position + object_offset
                )

                current_pose = Pose3D(
                    position_xyz=tuple(
                        float(value)
                        for value in new_object_position
                    ),
                    orientation_rpy=(
                        current_pose.orientation_rpy
                    ),
                )

            completed_actions += 1

            events.append(
                ManipulationEvent(
                    event_id=new_id(),
                    task_id=task_id,
                    action=action,
                    status=ManipulationStatus.RUNNING,
                    message=(
                        f"Simulated action completed: "
                        f"{action.value}"
                    ),
                    timestamp=utc_now(),
                )
            )

        if status != ManipulationStatus.FAILED:
            status = ManipulationStatus.COMPLETED

            events.append(
                ManipulationEvent(
                    event_id=new_id(),
                    task_id=task_id,
                    action=None,
                    status=status,
                    message=(
                        "Pick-and-place simulation completed"
                    ),
                    timestamp=utc_now(),
                )
            )

        return ManipulationResult(
            result_id=new_id(),
            task_id=task_id,
            robot_id=robot.robot_id,
            object_id=obj.object_id,
            tenant_id=robot.tenant_id,
            status=status,
            failure=(
                ManipulationFailure.NONE
                if status == ManipulationStatus.COMPLETED
                else ManipulationFailure.SIMULATION_FAILURE
            ),
            plan=plan,
            final_object_pose=current_pose,
            completed_actions=completed_actions,
            events=tuple(events),
            created_at=utc_now(),
        )


# ============================================================
# OBJECT STATE TRACKER
# ============================================================

class ManipulationObjectTracker:

    def __init__(self) -> None:
        self._objects: dict[
            tuple[str, str],
            ManipulationObject,
        ] = {}

        self._lock = threading.RLock()

    def register(
        self,
        obj: ManipulationObject,
    ) -> None:
        key = (
            obj.tenant_id,
            obj.object_id,
        )

        with self._lock:
            if key in self._objects:
                raise ManipulationConflictError(
                    "Object already registered"
                )

            self._objects[key] = obj

    def get(
        self,
        tenant_id: str,
        object_id: str,
    ) -> ManipulationObject:
        key = (
            tenant_id,
            object_id,
        )

        with self._lock:
            obj = self._objects.get(key)

        if obj is None:
            raise ManipulationNotFoundError(
                "Object not found"
            )

        return obj

    def update_state(
        self,
        tenant_id: str,
        object_id: str,
        state: ObjectState,
        pose: Pose3D | None = None,
    ) -> ManipulationObject:
        key = (
            tenant_id,
            object_id,
        )

        with self._lock:
            obj = self._objects.get(key)

            if obj is None:
                raise ManipulationNotFoundError(
                    "Object not found"
                )

            updated = replace(
                obj,
                state=state,
                pose=pose if pose is not None else obj.pose,
            )

            self._objects[key] = updated

            return updated

    def list_objects(
        self,
        tenant_id: str,
    ) -> tuple[ManipulationObject, ...]:
        with self._lock:
            return tuple(
                obj
                for (tenant, _), obj
                in self._objects.items()
                if tenant == tenant_id
            )


# ============================================================
# MULTI-TENANT MANIPULATION SERVICE
# ============================================================

class ObjectManipulationService:

    def __init__(self) -> None:
        self._robots: dict[
            tuple[str, str],
            ManipulationRobot,
        ] = {}

        self._results: dict[
            tuple[str, str],
            ManipulationResult,
        ] = {}

        self.objects = ManipulationObjectTracker()

        self.planner = ObjectManipulationPlanner()
        self.simulator = ObjectManipulationSimulator()

        self._lock = threading.RLock()

    @staticmethod
    def _authorize(
        access: ManipulationAccessContext,
        permission: str,
    ) -> None:
        if not getattr(access, permission, False):
            raise ManipulationAuthorizationError(
                f"Missing permission: {permission}"
            )

    def register_robot(
        self,
        robot: ManipulationRobot,
        access: ManipulationAccessContext,
    ) -> None:
        self._authorize(access, "can_manage")

        if robot.tenant_id != access.tenant_id:
            raise ManipulationAuthorizationError(
                "Cross-tenant registration forbidden"
            )

        key = (
            robot.tenant_id,
            robot.robot_id,
        )

        with self._lock:
            if key in self._robots:
                raise ManipulationConflictError(
                    "Robot already registered"
                )

            self._robots[key] = robot

    def register_object(
        self,
        obj: ManipulationObject,
        access: ManipulationAccessContext,
    ) -> None:
        self._authorize(access, "can_manage")

        if obj.tenant_id != access.tenant_id:
            raise ManipulationAuthorizationError(
                "Cross-tenant object registration forbidden"
            )

        self.objects.register(obj)

    def get_robot(
        self,
        robot_id: str,
        access: ManipulationAccessContext,
    ) -> ManipulationRobot:
        self._authorize(access, "can_read")

        with self._lock:
            robot = self._robots.get(
                (access.tenant_id, robot_id)
            )

        if robot is None:
            raise ManipulationNotFoundError(
                "Robot not found"
            )

        return robot

    def plan(
        self,
        robot_id: str,
        request: ManipulationRequest,
        access: ManipulationAccessContext,
    ) -> ManipulationPlan:
        self._authorize(access, "can_plan")

        robot = self.get_robot(
            robot_id,
            access,
        )

        obj = self.objects.get(
            access.tenant_id,
            request.object_id,
        )

        return self.planner.plan(
            request,
            obj,
            robot,
        )

    def simulate(
        self,
        robot_id: str,
        request: ManipulationRequest,
        access: ManipulationAccessContext,
    ) -> ManipulationResult:
        self._authorize(access, "can_simulate")
        self._authorize(access, "can_plan")

        with self._lock:
            robot = self.get_robot(
                robot_id,
                access,
            )

            obj = self.objects.get(
                access.tenant_id,
                request.object_id,
            )

            plan = self.planner.plan(
                request,
                obj,
                robot,
            )

            result = self.simulator.simulate(
                plan,
                obj,
                robot,
                request.destination_pose,
            )

            if result.status == ManipulationStatus.COMPLETED:
                self.objects.update_state(
                    access.tenant_id,
                    obj.object_id,
                    ObjectState.PLACED,
                    result.final_object_pose,
                )

            self._results[
                (access.tenant_id, result.result_id)
            ] = result

            return result

    def get_result(
        self,
        result_id: str,
        access: ManipulationAccessContext,
    ) -> ManipulationResult:
        self._authorize(access, "can_read")

        with self._lock:
            result = self._results.get(
                (access.tenant_id, result_id)
            )

        if result is None:
            raise ManipulationNotFoundError(
                "Manipulation result not found"
            )

        return result

    def export_json(
        self,
        result: ManipulationResult,
        access: ManipulationAccessContext,
    ) -> str:
        self._authorize(access, "can_read")

        if result.tenant_id != access.tenant_id:
            raise ManipulationAuthorizationError(
                "Cross-tenant result access forbidden"
            )

        return json.dumps(
            json_safe(
                {
                    "result": result,
                    "simulation_only": True,
                    "physical_actuation": False,
                    "collision_model": "simplified_spheres",
                    "inverse_kinematics_verified": False,
                    "full_arm_motion_verified": False,
                    "grasp_force_certified": False,
                    "safety_certified": False,
                }
            ),
            indent=2,
            allow_nan=False,
        )


# ============================================================
# SYNTHETIC DATA FACTORY
# ============================================================

class SyntheticManipulationFactory:

    @staticmethod
    def robot(
        tenant_id: str = "physical_ai_lab",
    ) -> ManipulationRobot:
        return ManipulationRobot(
            robot_id="robot-arm-001",
            tenant_id=tenant_id,
            gripper=GripperConfiguration(
                gripper_id="parallel-gripper-001",
                gripper_type=GripperType.PARALLEL_JAW,
                minimum_opening_m=0.005,
                maximum_opening_m=0.12,
                maximum_payload_kg=5.0,
                maximum_normal_force_n=80.0,
                safety_factor=2.0,
            ),
            maximum_reach_m=1.5,
            maximum_payload_kg=5.0,
        )

    @staticmethod
    def object(
        tenant_id: str = "physical_ai_lab",
    ) -> ManipulationObject:
        return ManipulationObject(
            object_id="object-001",
            tenant_id=tenant_id,
            name="Industrial Component",
            shape=ObjectShape.BOX,
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
    def request() -> ManipulationRequest:
        return ManipulationRequest(
            request_id=new_id(),
            object_id="object-001",
            destination_pose=Pose3D(
                position_xyz=(0.55, 0.25, 0.15)
            ),
            approach_distance_m=0.12,
            lift_height_m=0.15,
            retreat_distance_m=0.12,
        )

    @staticmethod
    def access(
        tenant_id: str = "physical_ai_lab",
    ) -> ManipulationAccessContext:
        return ManipulationAccessContext(
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

class TestObjectManipulation(unittest.TestCase):

    def setUp(self) -> None:
        self.robot = SyntheticManipulationFactory.robot()
        self.obj = SyntheticManipulationFactory.object()
        self.request = SyntheticManipulationFactory.request()
        self.access = SyntheticManipulationFactory.access()

    def test_object_dimensions(self) -> None:
        self.assertAlmostEqual(
            self.obj.dimensions.volume_m3,
            0.10 * 0.07 * 0.08,
        )

    def test_object_mass(self) -> None:
        self.assertGreater(
            self.obj.mass_kg,
            0.0,
        )

    def test_pose_translation(self) -> None:
        pose = self.obj.pose.translated(
            (0.0, 0.0, 0.1)
        )

        self.assertAlmostEqual(
            pose.position_xyz[2],
            0.25,
        )

    def test_grasp_generation(self) -> None:
        candidates = ObjectGraspGenerator().generate(
            self.obj,
            self.robot.gripper,
        )

        self.assertGreater(
            len(candidates),
            0,
        )

    def test_grasp_evaluation(self) -> None:
        candidate = (
            ObjectGraspGenerator().generate(
                self.obj,
                self.robot.gripper,
            )[0]
        )

        evaluation = ObjectGraspEvaluator().evaluate(
            self.obj,
            self.robot,
            candidate,
        )

        self.assertTrue(
            evaluation.feasible
        )

    def test_grasp_selection(self) -> None:
        candidate, evaluation = (
            ObjectGraspSelector().select(
                self.obj,
                self.robot,
            )
        )

        self.assertEqual(
            candidate.object_id,
            self.obj.object_id,
        )

        self.assertTrue(
            evaluation.feasible
        )

    def test_payload_exceeded(self) -> None:
        heavy = replace(
            self.obj,
            mass_kg=100.0,
        )

        candidate = (
            ObjectGraspGenerator().generate(
                heavy,
                self.robot.gripper,
            )[0]
        )

        evaluation = ObjectGraspEvaluator().evaluate(
            heavy,
            self.robot,
            candidate,
        )

        self.assertFalse(
            evaluation.feasible
        )

        self.assertEqual(
            evaluation.failure,
            ManipulationFailure.PAYLOAD_EXCEEDED,
        )

    def test_collision_checker_clear(self) -> None:
        checker = ObjectManipulationCollisionChecker()

        self.assertFalse(
            checker.point_collision(
                (0.0, 0.0, 0.0),
                (),
            )
        )

    def test_collision_checker_obstacle(self) -> None:
        checker = ObjectManipulationCollisionChecker()

        obstacle = ManipulationObstacle(
            obstacle_id="obstacle-001",
            center_xyz=(0.0, 0.0, 0.0),
            radius_m=0.1,
        )

        self.assertTrue(
            checker.point_collision(
                (0.0, 0.0, 0.0),
                (obstacle,),
            )
        )

    def test_waypoint_generation(self) -> None:
        grasp, _ = ObjectGraspSelector().select(
            self.obj,
            self.robot,
        )

        waypoints = ObjectWaypointPlanner().generate(
            self.request,
            self.obj,
            self.robot,
            grasp,
        )

        self.assertEqual(
            len(waypoints),
            9,
        )

    def test_manipulation_plan(self) -> None:
        plan = ObjectManipulationPlanner().plan(
            self.request,
            self.obj,
            self.robot,
        )

        self.assertGreater(
            plan.estimated_distance_m,
            0.0,
        )

    def test_manipulation_simulation(self) -> None:
        plan = ObjectManipulationPlanner().plan(
            self.request,
            self.obj,
            self.robot,
        )

        result = ObjectManipulationSimulator().simulate(
            plan,
            self.obj,
            self.robot,
            self.request.destination_pose,
        )

        self.assertEqual(
            result.status,
            ManipulationStatus.COMPLETED,
        )

        self.assertEqual(
            result.completed_actions,
            9,
        )

    def test_simulated_final_pose(self) -> None:
        plan = ObjectManipulationPlanner().plan(
            self.request,
            self.obj,
            self.robot,
        )

        result = ObjectManipulationSimulator().simulate(
            plan,
            self.obj,
            self.robot,
            self.request.destination_pose,
        )

        self.assertEqual(
            result.final_object_pose,
            self.request.destination_pose,
        )

    def test_object_tracker(self) -> None:
        tracker = ManipulationObjectTracker()

        tracker.register(self.obj)

        found = tracker.get(
            self.obj.tenant_id,
            self.obj.object_id,
        )

        self.assertEqual(
            found.object_id,
            self.obj.object_id,
        )

    def test_object_state_update(self) -> None:
        tracker = ManipulationObjectTracker()

        tracker.register(self.obj)

        updated = tracker.update_state(
            self.obj.tenant_id,
            self.obj.object_id,
            ObjectState.RESERVED,
        )

        self.assertEqual(
            updated.state,
            ObjectState.RESERVED,
        )

    def test_service_registration(self) -> None:
        service = ObjectManipulationService()

        service.register_robot(
            self.robot,
            self.access,
        )

        service.register_object(
            self.obj,
            self.access,
        )

        found = service.get_robot(
            self.robot.robot_id,
            self.access,
        )

        self.assertEqual(
            found.robot_id,
            self.robot.robot_id,
        )

    def test_service_plan(self) -> None:
        service = ObjectManipulationService()

        service.register_robot(
            self.robot,
            self.access,
        )

        service.register_object(
            self.obj,
            self.access,
        )

        plan = service.plan(
            self.robot.robot_id,
            self.request,
            self.access,
        )

        self.assertEqual(
            plan.object_id,
            self.obj.object_id,
        )

    def test_service_simulate(self) -> None:
        service = ObjectManipulationService()

        service.register_robot(
            self.robot,
            self.access,
        )

        service.register_object(
            self.obj,
            self.access,
        )

        result = service.simulate(
            self.robot.robot_id,
            self.request,
            self.access,
        )

        self.assertEqual(
            result.status,
            ManipulationStatus.COMPLETED,
        )

    def test_service_updates_object(self) -> None:
        service = ObjectManipulationService()

        service.register_robot(
            self.robot,
            self.access,
        )

        service.register_object(
            self.obj,
            self.access,
        )

        service.simulate(
            self.robot.robot_id,
            self.request,
            self.access,
        )

        updated = service.objects.get(
            self.access.tenant_id,
            self.obj.object_id,
        )

        self.assertEqual(
            updated.state,
            ObjectState.PLACED,
        )

    def test_tenant_isolation(self) -> None:
        service = ObjectManipulationService()

        service.register_robot(
            self.robot,
            self.access,
        )

        other = ManipulationAccessContext(
            tenant_id="other_tenant",
            user_id="other_user",
            can_read=True,
            can_plan=True,
        )

        with self.assertRaises(
            ManipulationNotFoundError
        ):
            service.get_robot(
                self.robot.robot_id,
                other,
            )

    def test_permission_denied(self) -> None:
        service = ObjectManipulationService()

        service.register_robot(
            self.robot,
            self.access,
        )

        viewer = ManipulationAccessContext(
            tenant_id="physical_ai_lab",
            user_id="viewer",
            can_read=True,
        )

        with self.assertRaises(
            ManipulationAuthorizationError
        ):
            service.plan(
                self.robot.robot_id,
                self.request,
                viewer,
            )

    def test_json_export(self) -> None:
        service = ObjectManipulationService()

        service.register_robot(
            self.robot,
            self.access,
        )

        service.register_object(
            self.obj,
            self.access,
        )

        result = service.simulate(
            self.robot.robot_id,
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
    print("\nPHYSICAL AI — OBJECT MANIPULATION")
    print("=" * 72)

    robot = SyntheticManipulationFactory.robot()
    obj = SyntheticManipulationFactory.object()
    request = SyntheticManipulationFactory.request()
    access = SyntheticManipulationFactory.access()

    service = ObjectManipulationService()

    service.register_robot(
        robot,
        access,
    )

    service.register_object(
        obj,
        access,
    )

    plan = service.plan(
        robot.robot_id,
        request,
        access,
    )

    print("Robot:", robot.robot_id)
    print("Object:", obj.name)
    print("Object mass:", obj.mass_kg, "kg")
    print(
        "Selected grasp:",
        plan.selected_grasp.grasp_type.value,
    )
    print(
        "Grasp quality:",
        round(
            plan.grasp_evaluation.overall_score,
            4,
        ),
    )
    print(
        "Planned waypoints:",
        len(plan.waypoints),
    )
    print(
        "Estimated tool travel:",
        round(
            plan.estimated_distance_m,
            4,
        ),
        "m",
    )

    result = service.simulate(
        robot.robot_id,
        request,
        access,
    )

    print(
        "Simulation status:",
        result.status.value,
    )
    print(
        "Completed actions:",
        result.completed_actions,
    )
    print(
        "Final object pose:",
        result.final_object_pose,
    )

    print("\nMANIPULATION RESULT JSON")
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
        TestObjectManipulation
    )

    test_result = unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)

    if not test_result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
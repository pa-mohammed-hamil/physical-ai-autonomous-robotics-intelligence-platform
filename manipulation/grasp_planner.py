"""Grasp Planner module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/manipulation/grasp_planner.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================
Module: Grasp Planner

Features:
    - Parallel-jaw grasp planning
    - Grasp candidate validation and ranking
    - Camera-to-robot coordinate transformation
    - Grasp approach and retreat planning
    - Pre-grasp and contact poses
    - Gripper opening estimation
    - Object geometry and mass constraints
    - Surface-normal alignment
    - Friction-aware grasp scoring
    - Contact force estimation
    - Grasp stability estimation
    - Workspace validation
    - Joint-independent Cartesian waypoint planning
    - Conservative obstacle clearance checks
    - Grasp feasibility analysis
    - Multi-object grasp planning
    - Multi-tenant robot isolation
    - Thread-safe service
    - JSON export
    - Synthetic demonstrations
    - Unit tests

Dependencies:
    numpy >= 1.24

Install:
    pip install numpy

Safety:
    Planning and scoring only.
    No physical robot commands are issued.

    This is not a certified motion planner or collision checker.
    Actual execution requires validated inverse kinematics,
    full robot geometry, swept-volume collision checking,
    force control, calibrated transforms, and safety interlocks.
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

class GraspPlannerError(Exception):
    pass


class GraspValidationError(GraspPlannerError):
    pass


class GraspNotFoundError(GraspPlannerError):
    pass


class GraspAuthorizationError(GraspPlannerError):
    pass


class GraspConflictError(GraspPlannerError):
    pass


class GraspGeometryError(GraspPlannerError):
    pass


class GraspPlanningError(GraspPlannerError):
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
        raise GraspValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise GraspValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise GraspValidationError(
            f"{name} must be finite"
        )

    return result


def identifier(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 256
    ):
        raise GraspValidationError(
            f"Invalid {name}"
        )

    return value.strip()


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(
        minimum,
        min(maximum, value),
    )


def vector3(
    values: Sequence[float],
    name: str,
) -> np.ndarray:
    try:
        result = np.asarray(
            values,
            dtype=float,
        )
    except (TypeError, ValueError) as exc:
        raise GraspValidationError(
            f"{name} must contain numeric values"
        ) from exc

    if result.shape != (3,):
        raise GraspValidationError(
            f"{name} must contain three values"
        )

    if not np.all(np.isfinite(result)):
        raise GraspValidationError(
            f"{name} contains non-finite values"
        )

    return result


def normalize(
    values: Sequence[float],
    name: str = "vector",
) -> np.ndarray:
    result = vector3(
        values,
        name,
    )

    magnitude = float(
        np.linalg.norm(result)
    )

    if magnitude < 1e-12:
        raise GraspGeometryError(
            f"{name} has zero magnitude"
        )

    return result / magnitude


def tuple_vector(
    values: Sequence[float],
) -> tuple[float, ...]:
    return tuple(
        float(value)
        for value in values
    )


def tuple_matrix(
    matrix: np.ndarray,
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple_vector(row)
        for row in matrix
    )


def validate_rotation(
    rotation: Sequence[Sequence[float]],
) -> np.ndarray:
    matrix = np.asarray(
        rotation,
        dtype=float,
    )

    if matrix.shape != (3, 3):
        raise GraspGeometryError(
            "Rotation must be a 3x3 matrix"
        )

    if not np.all(np.isfinite(matrix)):
        raise GraspGeometryError(
            "Rotation contains non-finite values"
        )

    if not np.allclose(
        matrix.T @ matrix,
        np.eye(3),
        atol=1e-6,
    ):
        raise GraspGeometryError(
            "Rotation must be orthonormal"
        )

    if not np.isclose(
        np.linalg.det(matrix),
        1.0,
        atol=1e-6,
    ):
        raise GraspGeometryError(
            "Rotation determinant must be +1"
        )

    return matrix


def validate_transform(
    transform: Sequence[Sequence[float]],
) -> np.ndarray:
    matrix = np.asarray(
        transform,
        dtype=float,
    )

    if matrix.shape != (4, 4):
        raise GraspGeometryError(
            "Transform must be a 4x4 matrix"
        )

    if not np.all(np.isfinite(matrix)):
        raise GraspGeometryError(
            "Transform contains non-finite values"
        )

    if not np.allclose(
        matrix[3],
        [0.0, 0.0, 0.0, 1.0],
        atol=1e-8,
    ):
        raise GraspGeometryError(
            "Invalid homogeneous transform"
        )

    validate_rotation(
        matrix[:3, :3]
    )

    return matrix


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

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    return value


# ============================================================
# ENUMERATIONS
# ============================================================

class GraspType(str, Enum):
    PARALLEL_JAW = "parallel_jaw"
    TOP_DOWN = "top_down"
    SIDE_GRASP = "side_grasp"


class GraspPlanStatus(str, Enum):
    FEASIBLE = "feasible"
    REJECTED = "rejected"
    INSUFFICIENT_INFORMATION = "insufficient_information"


class GraspRejectionReason(str, Enum):
    NONE = "none"
    LOW_DETECTION_CONFIDENCE = "low_detection_confidence"
    INVALID_GRIPPER_WIDTH = "invalid_gripper_width"
    EXCESSIVE_OBJECT_MASS = "excessive_object_mass"
    INSUFFICIENT_FRICTION = "insufficient_friction"
    INSUFFICIENT_CONTACT_FORCE = "insufficient_contact_force"
    WORKSPACE_VIOLATION = "workspace_violation"
    OBSTACLE_COLLISION_RISK = "obstacle_collision_risk"
    INVALID_GEOMETRY = "invalid_geometry"


class WaypointType(str, Enum):
    PRE_GRASP = "pre_grasp"
    APPROACH = "approach"
    CONTACT = "contact"
    RETREAT = "retreat"


class GraspQualityLevel(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class GripperSpecification:
    minimum_opening_m: float = 0.005
    maximum_opening_m: float = 0.120

    finger_length_m: float = 0.080
    finger_thickness_m: float = 0.010

    maximum_normal_force_n: float = 40.0

    maximum_payload_kg: float = 2.0

    safety_margin_m: float = 0.005

    default_friction_coefficient: float = 0.6

    def __post_init__(self) -> None:
        for name in (
            "minimum_opening_m",
            "maximum_opening_m",
            "finger_length_m",
            "finger_thickness_m",
            "maximum_normal_force_n",
            "maximum_payload_kg",
            "safety_margin_m",
            "default_friction_coefficient",
        ):
            value = finite(
                getattr(self, name),
                name,
            )

            if value < 0:
                raise GraspValidationError(
                    f"{name} cannot be negative"
                )

        if (
            self.maximum_opening_m
            <= self.minimum_opening_m
        ):
            raise GraspValidationError(
                "Invalid gripper opening range"
            )

        if self.maximum_normal_force_n <= 0:
            raise GraspValidationError(
                "Maximum force must be positive"
            )

        if self.maximum_payload_kg <= 0:
            raise GraspValidationError(
                "Maximum payload must be positive"
            )

        if self.default_friction_coefficient <= 0:
            raise GraspValidationError(
                "Friction coefficient must be positive"
            )


@dataclass(frozen=True)
class WorkspaceBounds:
    minimum_xyz: tuple[float, float, float] = (
        -2.0,
        -2.0,
        0.0,
    )

    maximum_xyz: tuple[float, float, float] = (
        2.0,
        2.0,
        2.5,
    )

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
            raise GraspValidationError(
                "Invalid workspace bounds"
            )

    def contains(
        self,
        point: Sequence[float],
    ) -> bool:
        position = vector3(
            point,
            "point",
        )

        return bool(
            np.all(
                position >= np.asarray(
                    self.minimum_xyz
                )
            )
            and np.all(
                position <= np.asarray(
                    self.maximum_xyz
                )
            )
        )


@dataclass(frozen=True)
class GraspPlannerConfig:
    minimum_detection_confidence: float = 0.40

    pre_grasp_distance_m: float = 0.12
    approach_step_m: float = 0.02
    retreat_distance_m: float = 0.15

    contact_clearance_m: float = 0.005

    obstacle_safety_margin_m: float = 0.03

    minimum_friction_coefficient: float = 0.15

    grasp_force_safety_factor: float = 2.0

    maximum_candidates: int = 30

    gravity_m_s2: float = 9.81

    def __post_init__(self) -> None:
        if not (
            0.0
            <= self.minimum_detection_confidence
            <= 1.0
        ):
            raise GraspValidationError(
                "Invalid confidence threshold"
            )

        for name in (
            "pre_grasp_distance_m",
            "approach_step_m",
            "retreat_distance_m",
            "contact_clearance_m",
            "obstacle_safety_margin_m",
            "minimum_friction_coefficient",
            "grasp_force_safety_factor",
            "gravity_m_s2",
        ):
            value = finite(
                getattr(self, name),
                name,
            )

            if value <= 0 and name in (
                "approach_step_m",
                "minimum_friction_coefficient",
                "grasp_force_safety_factor",
                "gravity_m_s2",
            ):
                raise GraspValidationError(
                    f"{name} must be positive"
                )

            if value < 0:
                raise GraspValidationError(
                    f"{name} cannot be negative"
                )

        if self.maximum_candidates <= 0:
            raise GraspValidationError(
                "Maximum candidates must be positive"
            )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class GraspAccessContext:
    tenant_id: str
    user_id: str

    can_read: bool = False
    can_plan: bool = False
    can_manage: bool = False


@dataclass(frozen=True)
class CartesianPose:
    position: tuple[float, float, float]

    rotation_matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]

    def __post_init__(self) -> None:
        vector3(
            self.position,
            "pose position",
        )

        validate_rotation(
            self.rotation_matrix
        )


@dataclass(frozen=True)
class ObjectProperties:
    object_id: str

    mass_kg: float

    friction_coefficient: float

    dimensions_xyz_m: tuple[
        float,
        float,
        float,
    ]

    def __post_init__(self) -> None:
        identifier(
            self.object_id,
            "object_id",
        )

        if self.mass_kg <= 0:
            raise GraspValidationError(
                "Object mass must be positive"
            )

        if self.friction_coefficient <= 0:
            raise GraspValidationError(
                "Friction coefficient must be positive"
            )

        dimensions = vector3(
            self.dimensions_xyz_m,
            "object dimensions",
        )

        if np.any(dimensions <= 0):
            raise GraspValidationError(
                "Object dimensions must be positive"
            )


@dataclass(frozen=True)
class GraspCandidate:
    candidate_id: str
    object_id: str

    grasp_type: GraspType

    pose_camera: CartesianPose

    grasp_width_m: float

    detection_confidence: float

    surface_normal_camera: tuple[
        float,
        float,
        float,
    ]

    closing_direction_camera: tuple[
        float,
        float,
        float,
    ]

    def __post_init__(self) -> None:
        identifier(
            self.candidate_id,
            "candidate_id",
        )

        identifier(
            self.object_id,
            "object_id",
        )

        if self.grasp_width_m <= 0:
            raise GraspValidationError(
                "Grasp width must be positive"
            )

        if not 0 <= self.detection_confidence <= 1:
            raise GraspValidationError(
                "Detection confidence must be in [0,1]"
            )

        normalize(
            self.surface_normal_camera,
            "surface normal",
        )

        normalize(
            self.closing_direction_camera,
            "closing direction",
        )


@dataclass(frozen=True)
class SphereObstacle:
    obstacle_id: str

    center_xyz: tuple[
        float,
        float,
        float,
    ]

    radius_m: float

    def __post_init__(self) -> None:
        identifier(
            self.obstacle_id,
            "obstacle_id",
        )

        vector3(
            self.center_xyz,
            "obstacle center",
        )

        if self.radius_m <= 0:
            raise GraspValidationError(
                "Obstacle radius must be positive"
            )


@dataclass(frozen=True)
class GraspWaypoint:
    waypoint_id: str

    waypoint_type: WaypointType

    pose: CartesianPose

    gripper_opening_m: float

    sequence_index: int


@dataclass(frozen=True)
class GraspForceEstimate:
    object_weight_n: float

    required_total_normal_force_n: float

    required_force_per_finger_n: float

    available_force_per_finger_n: float

    force_margin_n: float

    feasible: bool


@dataclass(frozen=True)
class GraspQuality:
    total_score: float

    detection_score: float
    geometry_score: float
    stability_score: float
    force_score: float
    clearance_score: float

    quality_level: GraspQualityLevel


@dataclass(frozen=True)
class GraspFeasibility:
    feasible: bool

    status: GraspPlanStatus

    rejection_reasons: tuple[
        GraspRejectionReason,
        ...
    ]

    minimum_obstacle_clearance_m: float | None


@dataclass(frozen=True)
class GraspPlan:
    plan_id: str

    robot_id: str
    tenant_id: str

    object_id: str
    candidate_id: str

    grasp_type: GraspType

    grasp_pose_base: CartesianPose

    waypoints: tuple[
        GraspWaypoint,
        ...
    ]

    force_estimate: GraspForceEstimate

    quality: GraspQuality

    feasibility: GraspFeasibility

    created_at: datetime


@dataclass(frozen=True)
class GraspPlanningResult:
    planning_id: str

    robot_id: str
    tenant_id: str

    object_id: str

    plans: tuple[
        GraspPlan,
        ...
    ]

    best_plan: GraspPlan | None

    evaluated_candidates: int

    feasible_candidates: int

    created_at: datetime


# ============================================================
# TRANSFORM UTILITIES
# ============================================================

class GraspTransformUtils:

    @staticmethod
    def identity() -> np.ndarray:
        return np.eye(
            4,
            dtype=float,
        )

    @staticmethod
    def from_pose(
        pose: CartesianPose,
    ) -> np.ndarray:
        transform = np.eye(
            4,
            dtype=float,
        )

        transform[:3, :3] = validate_rotation(
            pose.rotation_matrix
        )

        transform[:3, 3] = vector3(
            pose.position,
            "position",
        )

        return transform

    @staticmethod
    def to_pose(
        transform: Sequence[Sequence[float]],
    ) -> CartesianPose:
        matrix = validate_transform(
            transform
        )

        return CartesianPose(
            position=tuple_vector(
                matrix[:3, 3]
            ),
            rotation_matrix=tuple_matrix(
                matrix[:3, :3]
            ),
        )

    @staticmethod
    def transform_pose(
        base_from_camera: Sequence[
            Sequence[float]
        ],
        camera_pose: CartesianPose,
    ) -> CartesianPose:
        transform = (
            validate_transform(base_from_camera)
            @ GraspTransformUtils.from_pose(
                camera_pose
            )
        )

        return GraspTransformUtils.to_pose(
            transform
        )

    @staticmethod
    def transform_direction(
        transform: Sequence[
            Sequence[float]
        ],
        direction: Sequence[float],
    ) -> np.ndarray:
        matrix = validate_transform(
            transform
        )

        return normalize(
            matrix[:3, :3]
            @ normalize(direction)
        )

    @staticmethod
    def offset_pose(
        pose: CartesianPose,
        offset: Sequence[float],
    ) -> CartesianPose:
        position = (
            vector3(
                pose.position,
                "pose position",
            )
            + vector3(
                offset,
                "offset",
            )
        )

        return CartesianPose(
            position=tuple_vector(
                position
            ),
            rotation_matrix=pose.rotation_matrix,
        )


# ============================================================
# GRIPPER WIDTH VALIDATOR
# ============================================================

class GripperWidthValidator:

    def __init__(
        self,
        specification: GripperSpecification,
    ) -> None:
        self.specification = specification

    def is_feasible(
        self,
        width_m: float,
    ) -> bool:
        width = finite(
            width_m,
            "grasp width",
        )

        maximum = (
            self.specification.maximum_opening_m
            - 2.0
            * self.specification.safety_margin_m
        )

        return bool(
            self.specification.minimum_opening_m
            <= width
            <= maximum
        )

    def required_opening(
        self,
        width_m: float,
    ) -> float:
        width = finite(
            width_m,
            "grasp width",
        )

        return (
            width
            + 2.0
            * self.specification.safety_margin_m
        )


# ============================================================
# FORCE ESTIMATION
# ============================================================

class GraspForceEstimator:
    """
    Simplified symmetric two-finger friction model.

    Assumptions:
        - Equal normal force at both contacts
        - Vertical gravity load
        - Coulomb friction
        - No significant external torque
        - No dynamic acceleration

    This is not a full wrench-closure analysis.
    """

    def __init__(
        self,
        gripper: GripperSpecification,
        config: GraspPlannerConfig,
    ) -> None:
        self.gripper = gripper
        self.config = config

    def estimate(
        self,
        object_properties: ObjectProperties,
    ) -> GraspForceEstimate:
        weight = (
            object_properties.mass_kg
            * self.config.gravity_m_s2
        )

        friction = (
            object_properties.friction_coefficient
        )

        required_total = (
            weight
            * self.config.grasp_force_safety_factor
            / friction
        )

        per_finger = (
            required_total / 2.0
        )

        available_per_finger = (
            self.gripper.maximum_normal_force_n
        )

        margin = (
            available_per_finger
            - per_finger
        )

        feasible = bool(
            margin >= 0
            and object_properties.mass_kg
            <= self.gripper.maximum_payload_kg
            and friction
            >= self.config.minimum_friction_coefficient
        )

        return GraspForceEstimate(
            object_weight_n=float(weight),
            required_total_normal_force_n=float(
                required_total
            ),
            required_force_per_finger_n=float(
                per_finger
            ),
            available_force_per_finger_n=float(
                available_per_finger
            ),
            force_margin_n=float(margin),
            feasible=feasible,
        )


# ============================================================
# CARTESIAN WAYPOINT GENERATOR
# ============================================================

class GraspWaypointGenerator:

    def __init__(
        self,
        config: GraspPlannerConfig,
        gripper: GripperSpecification,
    ) -> None:
        self.config = config
        self.gripper = gripper

    def generate(
        self,
        grasp_pose: CartesianPose,
        approach_direction: Sequence[float],
        grasp_width_m: float,
    ) -> tuple[GraspWaypoint, ...]:
        approach = normalize(
            approach_direction,
            "approach direction",
        )

        pre_grasp_position = (
            vector3(
                grasp_pose.position,
                "grasp position",
            )
            - approach
            * self.config.pre_grasp_distance_m
        )

        contact_position = vector3(
            grasp_pose.position,
            "grasp position",
        )

        retreat_position = (
            contact_position
            - approach
            * self.config.retreat_distance_m
        )

        opening = (
            grasp_width_m
            + 2.0
            * self.gripper.safety_margin_m
        )

        opening = min(
            opening,
            self.gripper.maximum_opening_m,
        )

        waypoints: list[GraspWaypoint] = []

        def add_waypoint(
            waypoint_type: WaypointType,
            position: np.ndarray,
            opening_m: float,
        ) -> None:
            waypoints.append(
                GraspWaypoint(
                    waypoint_id=new_id(),
                    waypoint_type=waypoint_type,
                    pose=CartesianPose(
                        position=tuple_vector(
                            position
                        ),
                        rotation_matrix=(
                            grasp_pose.rotation_matrix
                        ),
                    ),
                    gripper_opening_m=float(
                        opening_m
                    ),
                    sequence_index=len(waypoints),
                )
            )

        add_waypoint(
            WaypointType.PRE_GRASP,
            pre_grasp_position,
            opening,
        )

        distance = float(
            np.linalg.norm(
                contact_position
                - pre_grasp_position
            )
        )

        steps = max(
            1,
            math.ceil(
                distance
                / self.config.approach_step_m
            ),
        )

        for index in range(1, steps):
            alpha = index / steps

            position = (
                pre_grasp_position
                + alpha
                * (
                    contact_position
                    - pre_grasp_position
                )
            )

            add_waypoint(
                WaypointType.APPROACH,
                position,
                opening,
            )

        add_waypoint(
            WaypointType.CONTACT,
            contact_position,
            grasp_width_m,
        )

        # Retreat waypoints are hypothetical and must
        # be separately verified before execution.
        retreat_steps = max(
            1,
            math.ceil(
                self.config.retreat_distance_m
                / self.config.approach_step_m
            ),
        )

        for index in range(1, retreat_steps + 1):
            alpha = (
                index / retreat_steps
            )

            position = (
                contact_position
                + alpha
                * (
                    retreat_position
                    - contact_position
                )
            )

            add_waypoint(
                WaypointType.RETREAT,
                position,
                grasp_width_m,
            )

        return tuple(
            waypoints
        )


# ============================================================
# OBSTACLE CLEARANCE
# ============================================================

class GraspObstacleChecker:
    """
    Checks waypoint-center clearance against spherical
    obstacles with a configurable conservative margin.

    This does not account for the complete robot arm,
    gripper finger geometry, or swept volumes.
    """

    def __init__(
        self,
        config: GraspPlannerConfig,
    ) -> None:
        self.config = config

    def minimum_clearance(
        self,
        waypoints: Sequence[GraspWaypoint],
        obstacles: Sequence[SphereObstacle],
    ) -> float | None:
        if not obstacles:
            return None

        minimum = float("inf")

        for waypoint in waypoints:
            position = vector3(
                waypoint.pose.position,
                "waypoint position",
            )

            for obstacle in obstacles:
                center = vector3(
                    obstacle.center_xyz,
                    "obstacle center",
                )

                clearance = (
                    float(
                        np.linalg.norm(
                            position - center
                        )
                    )
                    - obstacle.radius_m
                )

                minimum = min(
                    minimum,
                    clearance,
                )

        return minimum

    def is_clear(
        self,
        waypoints: Sequence[GraspWaypoint],
        obstacles: Sequence[SphereObstacle],
    ) -> bool:
        clearance = self.minimum_clearance(
            waypoints,
            obstacles,
        )

        if clearance is None:
            return True

        return (
            clearance
            >= self.config.obstacle_safety_margin_m
        )


# ============================================================
# GRASP QUALITY SCORER
# ============================================================

class GraspQualityScorer:

    @staticmethod
    def quality_level(
        score: float,
    ) -> GraspQualityLevel:
        if score >= 0.85:
            return GraspQualityLevel.EXCELLENT

        if score >= 0.70:
            return GraspQualityLevel.GOOD

        if score >= 0.45:
            return GraspQualityLevel.MODERATE

        return GraspQualityLevel.POOR

    @staticmethod
    def score(
        candidate: GraspCandidate,
        force: GraspForceEstimate,
        *,
        width_feasible: bool,
        clearance: float | None,
        config: GraspPlannerConfig,
    ) -> GraspQuality:
        detection_score = clamp(
            candidate.detection_confidence
        )

        geometry_score = (
            1.0 if width_feasible else 0.0
        )

        if force.feasible:
            stability_score = clamp(
                0.5
                + 0.5
                * force.force_margin_n
                / max(
                    force.available_force_per_finger_n,
                    1e-9,
                )
            )
        else:
            stability_score = 0.0

        force_score = (
            1.0
            if force.feasible
            else 0.0
        )

        if clearance is None:
            # No obstacles supplied does not prove
            # the environment is collision-free.
            clearance_score = 0.5

        else:
            clearance_score = clamp(
                clearance
                / max(
                    config.obstacle_safety_margin_m
                    * 4.0,
                    1e-9,
                )
            )

        total = (
            0.30 * detection_score
            + 0.20 * geometry_score
            + 0.20 * stability_score
            + 0.20 * force_score
            + 0.10 * clearance_score
        )

        total = clamp(
            total
        )

        return GraspQuality(
            total_score=total,
            detection_score=detection_score,
            geometry_score=geometry_score,
            stability_score=stability_score,
            force_score=force_score,
            clearance_score=clearance_score,
            quality_level=(
                GraspQualityScorer.quality_level(
                    total
                )
            ),
        )


# ============================================================
# GRASP FEASIBILITY EVALUATOR
# ============================================================

class GraspFeasibilityEvaluator:

    def __init__(
        self,
        config: GraspPlannerConfig,
        gripper: GripperSpecification,
        workspace: WorkspaceBounds,
    ) -> None:
        self.config = config
        self.gripper = gripper
        self.workspace = workspace

        self.width_validator = (
            GripperWidthValidator(
                gripper
            )
        )

        self.obstacle_checker = (
            GraspObstacleChecker(
                config
            )
        )

    def evaluate(
        self,
        candidate: GraspCandidate,
        object_properties: ObjectProperties,
        waypoints: Sequence[GraspWaypoint],
        force: GraspForceEstimate,
        obstacles: Sequence[SphereObstacle],
    ) -> GraspFeasibility:
        reasons: list[
            GraspRejectionReason
        ] = []

        if (
            candidate.detection_confidence
            < self.config.minimum_detection_confidence
        ):
            reasons.append(
                GraspRejectionReason.LOW_DETECTION_CONFIDENCE
            )

        if not self.width_validator.is_feasible(
            candidate.grasp_width_m
        ):
            reasons.append(
                GraspRejectionReason.INVALID_GRIPPER_WIDTH
            )

        if (
            object_properties.mass_kg
            > self.gripper.maximum_payload_kg
        ):
            reasons.append(
                GraspRejectionReason.EXCESSIVE_OBJECT_MASS
            )

        if (
            object_properties.friction_coefficient
            < self.config.minimum_friction_coefficient
        ):
            reasons.append(
                GraspRejectionReason.INSUFFICIENT_FRICTION
            )

        if not force.feasible:
            reasons.append(
                GraspRejectionReason.INSUFFICIENT_CONTACT_FORCE
            )

        if any(
            not self.workspace.contains(
                waypoint.pose.position
            )
            for waypoint in waypoints
        ):
            reasons.append(
                GraspRejectionReason.WORKSPACE_VIOLATION
            )

        clearance = (
            self.obstacle_checker.minimum_clearance(
                waypoints,
                obstacles,
            )
        )

        if (
            clearance is not None
            and clearance
            < self.config.obstacle_safety_margin_m
        ):
            reasons.append(
                GraspRejectionReason.OBSTACLE_COLLISION_RISK
            )

        unique_reasons = tuple(
            dict.fromkeys(reasons)
        )

        feasible = (
            len(unique_reasons) == 0
        )

        return GraspFeasibility(
            feasible=feasible,
            status=(
                GraspPlanStatus.FEASIBLE
                if feasible
                else GraspPlanStatus.REJECTED
            ),
            rejection_reasons=unique_reasons,
            minimum_obstacle_clearance_m=clearance,
        )


# ============================================================
# GRASP PLANNER
# ============================================================

class GraspPlanner:

    def __init__(
        self,
        *,
        robot_id: str,
        tenant_id: str,
        gripper: GripperSpecification | None = None,
        workspace: WorkspaceBounds | None = None,
        config: GraspPlannerConfig | None = None,
    ) -> None:
        self.robot_id = identifier(
            robot_id,
            "robot_id",
        )

        self.tenant_id = identifier(
            tenant_id,
            "tenant_id",
        )

        self.gripper = (
            gripper
            or GripperSpecification()
        )

        self.workspace = (
            workspace
            or WorkspaceBounds()
        )

        self.config = (
            config
            or GraspPlannerConfig()
        )

        self.force_estimator = (
            GraspForceEstimator(
                self.gripper,
                self.config,
            )
        )

        self.waypoint_generator = (
            GraspWaypointGenerator(
                self.config,
                self.gripper,
            )
        )

        self.feasibility_evaluator = (
            GraspFeasibilityEvaluator(
                self.config,
                self.gripper,
                self.workspace,
            )
        )

        self.obstacle_checker = (
            GraspObstacleChecker(
                self.config
            )
        )

        self.width_validator = (
            GripperWidthValidator(
                self.gripper
            )
        )

        self._lock = threading.RLock()

    def plan_candidate(
        self,
        candidate: GraspCandidate,
        object_properties: ObjectProperties,
        base_from_camera: Sequence[
            Sequence[float]
        ],
        obstacles: Sequence[
            SphereObstacle
        ] = (),
    ) -> GraspPlan:
        if (
            candidate.object_id
            != object_properties.object_id
        ):
            raise GraspValidationError(
                "Candidate and object identifiers mismatch"
            )

        transform = validate_transform(
            base_from_camera
        )

        pose_base = (
            GraspTransformUtils.transform_pose(
                transform,
                candidate.pose_camera,
            )
        )

        surface_normal_base = (
            GraspTransformUtils.transform_direction(
                transform,
                candidate.surface_normal_camera,
            )
        )

        # The approach points toward the object,
        # opposite its outward surface normal.
        approach = (
            -surface_normal_base
        )

        waypoints = (
            self.waypoint_generator.generate(
                pose_base,
                approach,
                candidate.grasp_width_m,
            )
        )

        force = (
            self.force_estimator.estimate(
                object_properties
            )
        )

        feasibility = (
            self.feasibility_evaluator.evaluate(
                candidate,
                object_properties,
                waypoints,
                force,
                obstacles,
            )
        )

        quality = GraspQualityScorer.score(
            candidate,
            force,
            width_feasible=(
                self.width_validator.is_feasible(
                    candidate.grasp_width_m
                )
            ),
            clearance=(
                feasibility.minimum_obstacle_clearance_m
            ),
            config=self.config,
        )

        return GraspPlan(
            plan_id=new_id(),
            robot_id=self.robot_id,
            tenant_id=self.tenant_id,
            object_id=candidate.object_id,
            candidate_id=candidate.candidate_id,
            grasp_type=candidate.grasp_type,
            grasp_pose_base=pose_base,
            waypoints=waypoints,
            force_estimate=force,
            quality=quality,
            feasibility=feasibility,
            created_at=utc_now(),
        )

    def plan(
        self,
        object_properties: ObjectProperties,
        candidates: Sequence[GraspCandidate],
        base_from_camera: Sequence[
            Sequence[float]
        ],
        obstacles: Sequence[
            SphereObstacle
        ] = (),
    ) -> GraspPlanningResult:
        with self._lock:
            selected_candidates = sorted(
                candidates,
                key=lambda candidate: (
                    candidate.detection_confidence
                ),
                reverse=True,
            )[: self.config.maximum_candidates]

            plans: list[GraspPlan] = []

            for candidate in selected_candidates:
                if (
                    candidate.object_id
                    != object_properties.object_id
                ):
                    continue

                try:
                    plan = self.plan_candidate(
                        candidate,
                        object_properties,
                        base_from_camera,
                        obstacles,
                    )

                    plans.append(
                        plan
                    )

                except (
                    GraspValidationError,
                    GraspGeometryError,
                    GraspPlanningError,
                ):
                    continue

            plans.sort(
                key=lambda plan: (
                    plan.feasibility.feasible,
                    plan.quality.total_score,
                ),
                reverse=True,
            )

            feasible_plans = [
                plan
                for plan in plans
                if plan.feasibility.feasible
            ]

            best = (
                feasible_plans[0]
                if feasible_plans
                else None
            )

            return GraspPlanningResult(
                planning_id=new_id(),
                robot_id=self.robot_id,
                tenant_id=self.tenant_id,
                object_id=object_properties.object_id,
                plans=tuple(plans),
                best_plan=best,
                evaluated_candidates=len(
                    selected_candidates
                ),
                feasible_candidates=len(
                    feasible_plans
                ),
                created_at=utc_now(),
            )

    def plan_multiple_objects(
        self,
        objects: Sequence[ObjectProperties],
        candidates: Sequence[GraspCandidate],
        base_from_camera: Sequence[
            Sequence[float]
        ],
        obstacles: Sequence[
            SphereObstacle
        ] = (),
    ) -> tuple[
        GraspPlanningResult,
        ...
    ]:
        results = []

        for object_properties in objects:
            object_candidates = [
                candidate
                for candidate in candidates
                if candidate.object_id
                == object_properties.object_id
            ]

            results.append(
                self.plan(
                    object_properties,
                    object_candidates,
                    base_from_camera,
                    obstacles,
                )
            )

        return tuple(
            results
        )


# ============================================================
# MULTI-TENANT SERVICE
# ============================================================

class GraspPlanningService:

    def __init__(self) -> None:
        self._planners: dict[
            tuple[str, str],
            GraspPlanner,
        ] = {}

        self._lock = threading.RLock()

    @staticmethod
    def _authorize(
        access: GraspAccessContext,
        permission: str,
    ) -> None:
        if not getattr(
            access,
            permission,
            False,
        ):
            raise GraspAuthorizationError(
                f"Missing permission: {permission}"
            )

    def register_planner(
        self,
        planner: GraspPlanner,
        access: GraspAccessContext,
    ) -> None:
        self._authorize(
            access,
            "can_manage",
        )

        if (
            planner.tenant_id
            != access.tenant_id
        ):
            raise GraspAuthorizationError(
                "Cross-tenant registration forbidden"
            )

        key = (
            planner.tenant_id,
            planner.robot_id,
        )

        with self._lock:
            if key in self._planners:
                raise GraspConflictError(
                    "Planner already registered"
                )

            self._planners[key] = planner

    def get_planner(
        self,
        robot_id: str,
        access: GraspAccessContext,
    ) -> GraspPlanner:
        self._authorize(
            access,
            "can_read",
        )

        key = (
            access.tenant_id,
            robot_id,
        )

        with self._lock:
            planner = self._planners.get(
                key
            )

        if planner is None:
            raise GraspNotFoundError(
                "Planner not found"
            )

        return planner

    def plan(
        self,
        robot_id: str,
        object_properties: ObjectProperties,
        candidates: Sequence[GraspCandidate],
        base_from_camera: Sequence[
            Sequence[float]
        ],
        access: GraspAccessContext,
        obstacles: Sequence[
            SphereObstacle
        ] = (),
    ) -> GraspPlanningResult:
        self._authorize(
            access,
            "can_plan",
        )

        planner = self.get_planner(
            robot_id,
            access,
        )

        return planner.plan(
            object_properties,
            candidates,
            base_from_camera,
            obstacles,
        )

    def list_planners(
        self,
        access: GraspAccessContext,
    ) -> tuple[str, ...]:
        self._authorize(
            access,
            "can_read",
        )

        with self._lock:
            return tuple(
                robot_id
                for tenant_id, robot_id
                in self._planners
                if tenant_id == access.tenant_id
            )

    def remove_planner(
        self,
        robot_id: str,
        access: GraspAccessContext,
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
            if key not in self._planners:
                raise GraspNotFoundError(
                    "Planner not found"
                )

            del self._planners[key]

    def export_json(
        self,
        result: GraspPlanningResult,
        access: GraspAccessContext,
    ) -> str:
        self._authorize(
            access,
            "can_read",
        )

        if (
            result.tenant_id
            != access.tenant_id
        ):
            raise GraspAuthorizationError(
                "Cross-tenant result access forbidden"
            )

        return json.dumps(
            json_safe(
                {
                    "result": result,
                    "advisory_only": True,
                    "physical_execution": False,
                    "full_collision_checking": False,
                    "inverse_kinematics_verified": False,
                }
            ),
            indent=2,
            allow_nan=False,
        )


# ============================================================
# SYNTHETIC DATA FACTORY
# ============================================================

class SyntheticGraspPlanningFactory:

    @staticmethod
    def identity_transform() -> np.ndarray:
        return np.eye(
            4,
            dtype=float,
        )

    @staticmethod
    def camera_transform() -> np.ndarray:
        transform = np.eye(
            4,
            dtype=float,
        )

        transform[:3, 3] = [
            0.2,
            0.0,
            0.3,
        ]

        return transform

    @staticmethod
    def object_properties(
        object_id: str = "box-001",
        mass_kg: float = 0.4,
    ) -> ObjectProperties:
        return ObjectProperties(
            object_id=object_id,
            mass_kg=mass_kg,
            friction_coefficient=0.7,
            dimensions_xyz_m=(
                0.06,
                0.05,
                0.08,
            ),
        )

    @staticmethod
    def candidate(
        *,
        candidate_id: str = "grasp-001",
        object_id: str = "box-001",
        confidence: float = 0.92,
        width_m: float = 0.06,
    ) -> GraspCandidate:
        return GraspCandidate(
            candidate_id=candidate_id,
            object_id=object_id,
            grasp_type=GraspType.PARALLEL_JAW,
            pose_camera=CartesianPose(
                position=(
                    0.0,
                    0.0,
                    0.6,
                ),
                rotation_matrix=(
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
            ),
            grasp_width_m=width_m,
            detection_confidence=confidence,
            surface_normal_camera=(
                0.0,
                0.0,
                -1.0,
            ),
            closing_direction_camera=(
                1.0,
                0.0,
                0.0,
            ),
        )

    @staticmethod
    def candidates() -> list[GraspCandidate]:
        return [
            SyntheticGraspPlanningFactory.candidate(
                candidate_id="grasp-001",
                confidence=0.95,
                width_m=0.06,
            ),
            SyntheticGraspPlanningFactory.candidate(
                candidate_id="grasp-002",
                confidence=0.82,
                width_m=0.05,
            ),
            SyntheticGraspPlanningFactory.candidate(
                candidate_id="grasp-003",
                confidence=0.70,
                width_m=0.15,
            ),
            SyntheticGraspPlanningFactory.candidate(
                candidate_id="grasp-004",
                confidence=0.25,
                width_m=0.05,
            ),
        ]

    @staticmethod
    def planner(
        robot_id: str = "robot-arm-001",
        tenant_id: str = "physical_ai_lab",
    ) -> GraspPlanner:
        return GraspPlanner(
            robot_id=robot_id,
            tenant_id=tenant_id,
            gripper=GripperSpecification(
                minimum_opening_m=0.005,
                maximum_opening_m=0.12,
                maximum_normal_force_n=40.0,
                maximum_payload_kg=2.0,
            ),
            workspace=WorkspaceBounds(
                minimum_xyz=(
                    -1.0,
                    -1.0,
                    0.0,
                ),
                maximum_xyz=(
                    1.0,
                    1.0,
                    2.0,
                ),
            ),
        )

    @staticmethod
    def access(
        tenant_id: str = "physical_ai_lab",
    ) -> GraspAccessContext:
        return GraspAccessContext(
            tenant_id=tenant_id,
            user_id="robotics_engineer",
            can_read=True,
            can_plan=True,
            can_manage=True,
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestGraspPlanner(unittest.TestCase):

    def setUp(self) -> None:
        self.planner = (
            SyntheticGraspPlanningFactory.planner()
        )

        self.object_properties = (
            SyntheticGraspPlanningFactory.object_properties()
        )

        self.candidate = (
            SyntheticGraspPlanningFactory.candidate()
        )

        self.transform = (
            SyntheticGraspPlanningFactory.camera_transform()
        )

        self.access = (
            SyntheticGraspPlanningFactory.access()
        )

    def test_identity_transform(self) -> None:
        matrix = (
            GraspTransformUtils.identity()
        )

        self.assertTrue(
            np.allclose(
                matrix,
                np.eye(4),
            )
        )

    def test_camera_to_base_transform(self) -> None:
        pose = (
            GraspTransformUtils.transform_pose(
                self.transform,
                self.candidate.pose_camera,
            )
        )

        self.assertAlmostEqual(
            pose.position[0],
            0.2,
        )

        self.assertAlmostEqual(
            pose.position[2],
            0.9,
        )

    def test_direction_transform(self) -> None:
        direction = (
            GraspTransformUtils.transform_direction(
                self.transform,
                (1.0, 0.0, 0.0),
            )
        )

        self.assertTrue(
            np.allclose(
                direction,
                [1.0, 0.0, 0.0],
            )
        )

    def test_valid_gripper_width(self) -> None:
        validator = (
            GripperWidthValidator(
                self.planner.gripper
            )
        )

        self.assertTrue(
            validator.is_feasible(
                0.06
            )
        )

    def test_invalid_gripper_width(self) -> None:
        validator = (
            GripperWidthValidator(
                self.planner.gripper
            )
        )

        self.assertFalse(
            validator.is_feasible(
                0.20
            )
        )

    def test_force_estimation(self) -> None:
        force = (
            self.planner.force_estimator.estimate(
                self.object_properties
            )
        )

        self.assertGreater(
            force.required_total_normal_force_n,
            0.0,
        )

        self.assertTrue(
            force.feasible
        )

    def test_heavy_object_rejected(self) -> None:
        heavy = (
            SyntheticGraspPlanningFactory.object_properties(
                mass_kg=10.0
            )
        )

        force = (
            self.planner.force_estimator.estimate(
                heavy
            )
        )

        self.assertFalse(
            force.feasible
        )

    def test_waypoint_generation(self) -> None:
        plan = self.planner.plan_candidate(
            self.candidate,
            self.object_properties,
            self.transform,
        )

        self.assertGreater(
            len(plan.waypoints),
            3,
        )

    def test_waypoint_sequence(self) -> None:
        plan = self.planner.plan_candidate(
            self.candidate,
            self.object_properties,
            self.transform,
        )

        self.assertEqual(
            plan.waypoints[0].waypoint_type,
            WaypointType.PRE_GRASP,
        )

        self.assertIn(
            WaypointType.CONTACT,
            [
                waypoint.waypoint_type
                for waypoint in plan.waypoints
            ],
        )

        self.assertEqual(
            plan.waypoints[-1].waypoint_type,
            WaypointType.RETREAT,
        )

    def test_feasible_grasp(self) -> None:
        plan = self.planner.plan_candidate(
            self.candidate,
            self.object_properties,
            self.transform,
        )

        self.assertTrue(
            plan.feasibility.feasible
        )

    def test_low_confidence_rejected(self) -> None:
        candidate = (
            SyntheticGraspPlanningFactory.candidate(
                confidence=0.1
            )
        )

        plan = self.planner.plan_candidate(
            candidate,
            self.object_properties,
            self.transform,
        )

        self.assertFalse(
            plan.feasibility.feasible
        )

        self.assertIn(
            GraspRejectionReason.LOW_DETECTION_CONFIDENCE,
            plan.feasibility.rejection_reasons,
        )

    def test_width_rejected(self) -> None:
        candidate = (
            SyntheticGraspPlanningFactory.candidate(
                width_m=0.20
            )
        )

        plan = self.planner.plan_candidate(
            candidate,
            self.object_properties,
            self.transform,
        )

        self.assertIn(
            GraspRejectionReason.INVALID_GRIPPER_WIDTH,
            plan.feasibility.rejection_reasons,
        )

    def test_obstacle_collision_risk(self) -> None:
        obstacle = SphereObstacle(
            obstacle_id="obstacle-001",
            center_xyz=(
                0.2,
                0.0,
                0.9,
            ),
            radius_m=0.10,
        )

        plan = self.planner.plan_candidate(
            self.candidate,
            self.object_properties,
            self.transform,
            [obstacle],
        )

        self.assertFalse(
            plan.feasibility.feasible
        )

    def test_workspace_violation(self) -> None:
        planner = GraspPlanner(
            robot_id="restricted-robot",
            tenant_id="physical_ai_lab",
            workspace=WorkspaceBounds(
                minimum_xyz=(
                    -0.1,
                    -0.1,
                    0.0,
                ),
                maximum_xyz=(
                    0.1,
                    0.1,
                    0.5,
                ),
            ),
        )

        plan = planner.plan_candidate(
            self.candidate,
            self.object_properties,
            self.transform,
        )

        self.assertIn(
            GraspRejectionReason.WORKSPACE_VIOLATION,
            plan.feasibility.rejection_reasons,
        )

    def test_candidate_ranking(self) -> None:
        result = self.planner.plan(
            self.object_properties,
            SyntheticGraspPlanningFactory.candidates(),
            self.transform,
        )

        self.assertIsNotNone(
            result.best_plan
        )

        self.assertTrue(
            result.best_plan.feasibility.feasible
        )

    def test_best_plan_selection(self) -> None:
        result = self.planner.plan(
            self.object_properties,
            SyntheticGraspPlanningFactory.candidates(),
            self.transform,
        )

        self.assertGreater(
            result.feasible_candidates,
            0,
        )

    def test_empty_candidates(self) -> None:
        result = self.planner.plan(
            self.object_properties,
            [],
            self.transform,
        )

        self.assertIsNone(
            result.best_plan
        )

    def test_multiple_objects(self) -> None:
        objects = [
            self.object_properties,
            SyntheticGraspPlanningFactory.object_properties(
                object_id="box-002"
            ),
        ]

        candidates = [
            self.candidate,
            SyntheticGraspPlanningFactory.candidate(
                candidate_id="grasp-box-002",
                object_id="box-002",
            ),
        ]

        results = self.planner.plan_multiple_objects(
            objects,
            candidates,
            self.transform,
        )

        self.assertEqual(
            len(results),
            2,
        )

    def test_service_registration(self) -> None:
        service = GraspPlanningService()

        service.register_planner(
            self.planner,
            self.access,
        )

        self.assertEqual(
            service.list_planners(
                self.access
            ),
            ("robot-arm-001",),
        )

    def test_service_planning(self) -> None:
        service = GraspPlanningService()

        service.register_planner(
            self.planner,
            self.access,
        )

        result = service.plan(
            "robot-arm-001",
            self.object_properties,
            [self.candidate],
            self.transform,
            self.access,
        )

        self.assertIsNotNone(
            result.best_plan
        )

    def test_tenant_isolation(self) -> None:
        service = GraspPlanningService()

        service.register_planner(
            self.planner,
            self.access,
        )

        other_access = GraspAccessContext(
            tenant_id="other_tenant",
            user_id="other_user",
            can_read=True,
            can_plan=True,
        )

        with self.assertRaises(
            GraspNotFoundError
        ):
            service.get_planner(
                "robot-arm-001",
                other_access,
            )

    def test_permission_denied(self) -> None:
        service = GraspPlanningService()

        service.register_planner(
            self.planner,
            self.access,
        )

        viewer = GraspAccessContext(
            tenant_id="physical_ai_lab",
            user_id="viewer",
            can_read=True,
        )

        with self.assertRaises(
            GraspAuthorizationError
        ):
            service.plan(
                "robot-arm-001",
                self.object_properties,
                [self.candidate],
                self.transform,
                viewer,
            )

    def test_json_export(self) -> None:
        service = GraspPlanningService()

        service.register_planner(
            self.planner,
            self.access,
        )

        result = service.plan(
            "robot-arm-001",
            self.object_properties,
            [self.candidate],
            self.transform,
            self.access,
        )

        exported = json.loads(
            service.export_json(
                result,
                self.access,
            )
        )

        self.assertTrue(
            exported["advisory_only"]
        )

        self.assertFalse(
            exported["physical_execution"]
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("\nPHYSICAL AI — GRASP PLANNER")
    print("=" * 70)

    planner = (
        SyntheticGraspPlanningFactory.planner(
            robot_id="physical-ai-robot-arm-001"
        )
    )

    access = (
        SyntheticGraspPlanningFactory.access()
    )

    service = GraspPlanningService()

    service.register_planner(
        planner,
        access,
    )

    object_properties = (
        SyntheticGraspPlanningFactory.object_properties()
    )

    candidates = (
        SyntheticGraspPlanningFactory.candidates()
    )

    transform = (
        SyntheticGraspPlanningFactory.camera_transform()
    )

    result = service.plan(
        planner.robot_id,
        object_properties,
        candidates,
        transform,
        access,
    )

    print(
        "Robot:",
        result.robot_id,
    )

    print(
        "Object:",
        result.object_id,
    )

    print(
        "Evaluated candidates:",
        result.evaluated_candidates,
    )

    print(
        "Feasible candidates:",
        result.feasible_candidates,
    )

    if result.best_plan is not None:
        best = result.best_plan

        print(
            "\nBest candidate:",
            best.candidate_id,
        )

        print(
            "Grasp quality:",
            round(
                best.quality.total_score,
                4,
            ),
        )

        print(
            "Grasp pose:",
            best.grasp_pose_base.position,
        )

        print(
            "Required normal force per finger:",
            round(
                best.force_estimate.required_force_per_finger_n,
                3,
            ),
            "N",
        )

        print(
            "Waypoint count:",
            len(best.waypoints),
        )

        print(
            "Feasible:",
            best.feasibility.feasible,
        )

    print("\nGRASP PLANNING REPORT")
    print("=" * 70)

    print(
        service.export_json(
            result,
            access,
        )
    )

    print("\nRUNNING UNIT TESTS")
    print("=" * 70)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestGraspPlanner
    )

    test_result = unittest.TextTestRunner(
        verbosity=2
    ).run(suite)

    if not test_result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
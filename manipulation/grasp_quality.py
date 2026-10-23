"""Grasp Quality module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/manipulation/grasp_quality.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================
Module: Grasp Quality Evaluation

Features:
    - Parallel-jaw grasp quality evaluation
    - Contact geometry validation
    - Surface-normal alignment
    - Friction cone analysis
    - Antipodal grasp scoring
    - Contact separation scoring
    - Gripper aperture validation
    - Force feasibility estimation
    - Payload capacity validation
    - Center-of-mass alignment
    - Torque resistance estimation
    - Contact confidence scoring
    - Collision-clearance scoring
    - Uncertainty-aware evaluation
    - Robustness under friction uncertainty
    - Multi-criteria grasp ranking
    - Weighted quality scoring
    - Rejection reasons and recommendations
    - Multi-object evaluation
    - Thread-safe multi-tenant service
    - JSON export
    - Synthetic test scenarios
    - Unit tests

Dependencies:
    Python >= 3.11
    numpy >= 1.24

Install:
    pip install numpy

Safety:
    This module performs advisory grasp-quality analysis only.

    It does not command a robot, apply forces, or execute
    grasp motions.

    Its friction, torque, and force models are simplified.
    A high quality score does not establish force closure,
    collision freedom, reachability, or safe execution.

    Physical deployment requires independent validation,
    calibrated sensors, complete robot dynamics, and
    certified safety mechanisms where applicable.
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

class GraspQualityError(Exception):
    pass


class GraspValidationError(GraspQualityError):
    pass


class GraspGeometryError(GraspQualityError):
    pass


class GraspNotFoundError(GraspQualityError):
    pass


class GraspAuthorizationError(GraspQualityError):
    pass


class GraspConflictError(GraspQualityError):
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
        raise GraspGeometryError(
            f"{name} must contain numeric values"
        ) from exc

    if result.shape != (3,):
        raise GraspGeometryError(
            f"{name} must contain three values"
        )

    if not np.all(np.isfinite(result)):
        raise GraspGeometryError(
            f"{name} contains non-finite values"
        )

    return result


def normalize(
    values: Sequence[float],
    name: str = "vector",
) -> np.ndarray:
    result = vector3(values, name)

    magnitude = float(
        np.linalg.norm(result)
    )

    if magnitude <= 1e-12:
        raise GraspGeometryError(
            f"{name} has zero magnitude"
        )

    return result / magnitude


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(
        minimum,
        min(maximum, value),
    )


def angle_between(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    a = normalize(first, "first vector")
    b = normalize(second, "second vector")

    cosine = clamp(
        float(np.dot(a, b)),
        -1.0,
        1.0,
    )

    return float(
        math.acos(cosine)
    )


def distance3(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    return float(
        np.linalg.norm(
            vector3(first, "first point")
            - vector3(second, "second point")
        )
    )


def tuple_vector(
    values: Sequence[float],
) -> tuple[float, float, float]:
    result = vector3(values, "vector")

    return (
        float(result[0]),
        float(result[1]),
        float(result[2]),
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

class GraspType(str, Enum):
    PARALLEL_JAW = "parallel_jaw"
    TOP_DOWN = "top_down"
    SIDE_GRASP = "side_grasp"


class GraspQualityLevel(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"


class GraspEvaluationStatus(str, Enum):
    ACCEPTABLE = "acceptable"
    REVIEW_REQUIRED = "review_required"
    REJECTED = "rejected"


class GraspRejectionReason(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    INVALID_APERTURE = "invalid_aperture"
    INSUFFICIENT_FRICTION = "insufficient_friction"
    POOR_ANTIPODAL_ALIGNMENT = "poor_antipodal_alignment"
    INSUFFICIENT_CONTACT_FORCE = "insufficient_contact_force"
    EXCESSIVE_PAYLOAD = "excessive_payload"
    EXCESSIVE_TORQUE = "excessive_torque"
    LOW_COLLISION_CLEARANCE = "low_collision_clearance"
    HIGH_UNCERTAINTY = "high_uncertainty"
    LOW_QUALITY_SCORE = "low_quality_score"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class GripperSpecification:
    minimum_opening_m: float = 0.005
    maximum_opening_m: float = 0.120

    maximum_normal_force_per_finger_n: float = 40.0
    maximum_payload_kg: float = 2.0

    maximum_resisting_torque_nm: float = 2.0

    aperture_safety_margin_m: float = 0.005

    def __post_init__(self) -> None:
        values = (
            "minimum_opening_m",
            "maximum_opening_m",
            "maximum_normal_force_per_finger_n",
            "maximum_payload_kg",
            "maximum_resisting_torque_nm",
            "aperture_safety_margin_m",
        )

        for name in values:
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
                "Invalid gripper aperture range"
            )

        if self.maximum_normal_force_per_finger_n <= 0:
            raise GraspValidationError(
                "Maximum normal force must be positive"
            )

        if self.maximum_payload_kg <= 0:
            raise GraspValidationError(
                "Maximum payload must be positive"
            )

        if self.maximum_resisting_torque_nm <= 0:
            raise GraspValidationError(
                "Maximum torque must be positive"
            )


@dataclass(frozen=True)
class GraspQualityWeights:
    antipodal: float = 0.20
    friction: float = 0.15
    force: float = 0.15
    geometry: float = 0.15
    torque: float = 0.10
    clearance: float = 0.10
    confidence: float = 0.10
    robustness: float = 0.05

    def __post_init__(self) -> None:
        values = [
            finite(
                getattr(self, name),
                name,
            )
            for name in (
                "antipodal",
                "friction",
                "force",
                "geometry",
                "torque",
                "clearance",
                "confidence",
                "robustness",
            )
        ]

        if any(value < 0 for value in values):
            raise GraspValidationError(
                "Quality weights cannot be negative"
            )

        if sum(values) <= 0:
            raise GraspValidationError(
                "At least one quality weight must be positive"
            )

    def normalized(self) -> dict[str, float]:
        names = (
            "antipodal",
            "friction",
            "force",
            "geometry",
            "torque",
            "clearance",
            "confidence",
            "robustness",
        )

        total = sum(
            getattr(self, name)
            for name in names
        )

        return {
            name: getattr(self, name) / total
            for name in names
        }


@dataclass(frozen=True)
class GraspQualityConfig:
    minimum_quality_score: float = 0.60
    minimum_detection_confidence: float = 0.40

    minimum_friction_coefficient: float = 0.15

    force_safety_factor: float = 2.0

    gravity_m_s2: float = 9.81

    minimum_collision_clearance_m: float = 0.02

    maximum_uncertainty: float = 0.40

    maximum_antipodal_angle_degrees: float = 30.0

    robustness_friction_reduction: float = 0.20

    center_offset_scale_m: float = 0.05

    maximum_results: int = 50

    def __post_init__(self) -> None:
        for name in (
            "minimum_quality_score",
            "minimum_detection_confidence",
            "maximum_uncertainty",
            "robustness_friction_reduction",
        ):
            value = finite(
                getattr(self, name),
                name,
            )

            if not 0 <= value <= 1:
                raise GraspValidationError(
                    f"{name} must be in [0,1]"
                )

        for name in (
            "minimum_friction_coefficient",
            "force_safety_factor",
            "gravity_m_s2",
            "center_offset_scale_m",
        ):
            if finite(
                getattr(self, name),
                name,
            ) <= 0:
                raise GraspValidationError(
                    f"{name} must be positive"
                )

        if self.minimum_collision_clearance_m < 0:
            raise GraspValidationError(
                "Minimum clearance cannot be negative"
            )

        if not (
            0 < self.maximum_antipodal_angle_degrees <= 90
        ):
            raise GraspValidationError(
                "Invalid antipodal angle"
            )

        if self.maximum_results < 1:
            raise GraspValidationError(
                "maximum_results must be positive"
            )


# ============================================================
# INPUT MODELS
# ============================================================

@dataclass(frozen=True)
class GraspContact:
    position_xyz: tuple[float, float, float]

    outward_normal_xyz: tuple[float, float, float]

    confidence: float = 1.0

    def __post_init__(self) -> None:
        vector3(
            self.position_xyz,
            "contact position",
        )

        normalize(
            self.outward_normal_xyz,
            "contact normal",
        )

        if not 0 <= self.confidence <= 1:
            raise GraspValidationError(
                "Contact confidence must be in [0,1]"
            )


@dataclass(frozen=True)
class ObjectProperties:
    object_id: str

    mass_kg: float

    friction_coefficient: float

    center_of_mass_xyz: tuple[
        float,
        float,
        float,
    ]

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

        if finite(self.mass_kg, "mass_kg") <= 0:
            raise GraspValidationError(
                "Object mass must be positive"
            )

        if finite(
            self.friction_coefficient,
            "friction_coefficient",
        ) <= 0:
            raise GraspValidationError(
                "Friction coefficient must be positive"
            )

        vector3(
            self.center_of_mass_xyz,
            "center of mass",
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

    left_contact: GraspContact
    right_contact: GraspContact

    detection_confidence: float

    collision_clearance_m: float | None = None

    uncertainty: float = 0.10

    def __post_init__(self) -> None:
        identifier(
            self.candidate_id,
            "candidate_id",
        )

        identifier(
            self.object_id,
            "object_id",
        )

        if not 0 <= self.detection_confidence <= 1:
            raise GraspValidationError(
                "Detection confidence must be in [0,1]"
            )

        if not 0 <= self.uncertainty <= 1:
            raise GraspValidationError(
                "Uncertainty must be in [0,1]"
            )

        if self.collision_clearance_m is not None:
            finite(
                self.collision_clearance_m,
                "collision_clearance_m",
            )


@dataclass(frozen=True)
class GraspAccessContext:
    tenant_id: str
    user_id: str

    can_read: bool = False
    can_evaluate: bool = False
    can_manage: bool = False


# ============================================================
# OUTPUT MODELS
# ============================================================

@dataclass(frozen=True)
class AntipodalAnalysis:
    left_alignment_degrees: float
    right_alignment_degrees: float

    maximum_alignment_degrees: float

    friction_cone_half_angle_degrees: float

    antipodal_score: float

    within_friction_cones: bool


@dataclass(frozen=True)
class ForceAnalysis:
    object_weight_n: float

    required_force_per_finger_n: float

    available_force_per_finger_n: float

    force_margin_n: float

    force_score: float

    feasible: bool


@dataclass(frozen=True)
class TorqueAnalysis:
    estimated_gravity_torque_nm: float

    maximum_resisting_torque_nm: float

    torque_margin_nm: float

    torque_score: float

    feasible: bool


@dataclass(frozen=True)
class GeometryAnalysis:
    contact_distance_m: float

    grasp_center_xyz: tuple[
        float,
        float,
        float,
    ]

    center_of_mass_offset_m: float

    aperture_score: float

    centering_score: float

    geometry_score: float

    aperture_feasible: bool


@dataclass(frozen=True)
class RobustnessAnalysis:
    nominal_friction_coefficient: float

    reduced_friction_coefficient: float

    nominal_required_force_n: float

    robust_required_force_n: float

    robustness_score: float

    robust_force_feasible: bool


@dataclass(frozen=True)
class GraspQualityBreakdown:
    antipodal_score: float
    friction_score: float
    force_score: float
    geometry_score: float
    torque_score: float
    clearance_score: float
    confidence_score: float
    robustness_score: float

    total_score: float

    quality_level: GraspQualityLevel


@dataclass(frozen=True)
class GraspQualityReport:
    report_id: str

    tenant_id: str
    robot_id: str

    candidate_id: str
    object_id: str

    status: GraspEvaluationStatus

    antipodal: AntipodalAnalysis
    force: ForceAnalysis
    torque: TorqueAnalysis
    geometry: GeometryAnalysis
    robustness: RobustnessAnalysis

    quality: GraspQualityBreakdown

    rejection_reasons: tuple[
        GraspRejectionReason,
        ...
    ]

    recommendations: tuple[str, ...]

    created_at: datetime


@dataclass(frozen=True)
class GraspRankingResult:
    ranking_id: str

    tenant_id: str
    robot_id: str

    object_id: str

    reports: tuple[GraspQualityReport, ...]

    best_grasp: GraspQualityReport | None

    evaluated_candidates: int
    acceptable_candidates: int

    created_at: datetime


# ============================================================
# CONTACT GEOMETRY
# ============================================================

class ContactGeometryAnalyzer:

    @staticmethod
    def contact_axis(
        candidate: GraspCandidate,
    ) -> np.ndarray:
        left = vector3(
            candidate.left_contact.position_xyz,
            "left contact",
        )

        right = vector3(
            candidate.right_contact.position_xyz,
            "right contact",
        )

        return normalize(
            right - left,
            "contact axis",
        )

    @staticmethod
    def contact_distance(
        candidate: GraspCandidate,
    ) -> float:
        return distance3(
            candidate.left_contact.position_xyz,
            candidate.right_contact.position_xyz,
        )

    @staticmethod
    def grasp_center(
        candidate: GraspCandidate,
    ) -> np.ndarray:
        left = vector3(
            candidate.left_contact.position_xyz,
            "left contact",
        )

        right = vector3(
            candidate.right_contact.position_xyz,
            "right contact",
        )

        return (
            left + right
        ) / 2.0

    @staticmethod
    def analyze(
        candidate: GraspCandidate,
        object_properties: ObjectProperties,
        gripper: GripperSpecification,
        config: GraspQualityConfig,
    ) -> GeometryAnalysis:
        distance = (
            ContactGeometryAnalyzer.contact_distance(
                candidate
            )
        )

        center = (
            ContactGeometryAnalyzer.grasp_center(
                candidate
            )
        )

        center_of_mass = vector3(
            object_properties.center_of_mass_xyz,
            "center of mass",
        )

        offset = float(
            np.linalg.norm(
                center - center_of_mass
            )
        )

        maximum_aperture = (
            gripper.maximum_opening_m
            - 2.0
            * gripper.aperture_safety_margin_m
        )

        aperture_feasible = bool(
            gripper.minimum_opening_m
            <= distance
            <= maximum_aperture
        )

        if aperture_feasible:
            aperture_score = 1.0
        else:
            aperture_score = 0.0

        centering_score = clamp(
            math.exp(
                -offset
                / config.center_offset_scale_m
            )
        )

        geometry_score = (
            0.6 * aperture_score
            + 0.4 * centering_score
        )

        return GeometryAnalysis(
            contact_distance_m=distance,
            grasp_center_xyz=tuple_vector(
                center
            ),
            center_of_mass_offset_m=offset,
            aperture_score=aperture_score,
            centering_score=centering_score,
            geometry_score=clamp(
                geometry_score
            ),
            aperture_feasible=aperture_feasible,
        )


# ============================================================
# FRICTION CONE AND ANTIPODAL ANALYSIS
# ============================================================

class AntipodalGraspAnalyzer:

    @staticmethod
    def analyze(
        candidate: GraspCandidate,
        friction_coefficient: float,
        config: GraspQualityConfig,
    ) -> AntipodalAnalysis:
        axis = (
            ContactGeometryAnalyzer.contact_axis(
                candidate
            )
        )

        left_normal = normalize(
            candidate.left_contact.outward_normal_xyz,
            "left outward normal",
        )

        right_normal = normalize(
            candidate.right_contact.outward_normal_xyz,
            "right outward normal",
        )

        # For a valid opposing contact pair:
        # left outward normal points toward the left,
        # right outward normal points toward the right.
        left_expected = -axis
        right_expected = axis

        left_angle = math.degrees(
            angle_between(
                left_normal,
                left_expected,
            )
        )

        right_angle = math.degrees(
            angle_between(
                right_normal,
                right_expected,
            )
        )

        maximum_angle = max(
            left_angle,
            right_angle,
        )

        friction_half_angle = math.degrees(
            math.atan(
                friction_coefficient
            )
        )

        allowed_angle = min(
            friction_half_angle,
            config.maximum_antipodal_angle_degrees,
        )

        within_cones = bool(
            left_angle <= allowed_angle
            and right_angle <= allowed_angle
        )

        score = clamp(
            1.0
            - maximum_angle
            / max(
                allowed_angle,
                1e-9,
            )
        )

        return AntipodalAnalysis(
            left_alignment_degrees=float(
                left_angle
            ),
            right_alignment_degrees=float(
                right_angle
            ),
            maximum_alignment_degrees=float(
                maximum_angle
            ),
            friction_cone_half_angle_degrees=float(
                friction_half_angle
            ),
            antipodal_score=score,
            within_friction_cones=within_cones,
        )


# ============================================================
# FORCE ANALYSIS
# ============================================================

class GraspForceAnalyzer:

    @staticmethod
    def analyze(
        object_properties: ObjectProperties,
        gripper: GripperSpecification,
        config: GraspQualityConfig,
    ) -> ForceAnalysis:
        weight = (
            object_properties.mass_kg
            * config.gravity_m_s2
        )

        friction = (
            object_properties.friction_coefficient
        )

        required_per_finger = (
            weight
            * config.force_safety_factor
            / (
                2.0 * friction
            )
        )

        available = (
            gripper.maximum_normal_force_per_finger_n
        )

        margin = (
            available
            - required_per_finger
        )

        feasible = bool(
            required_per_finger <= available
            and object_properties.mass_kg
            <= gripper.maximum_payload_kg
        )

        force_score = clamp(
            available
            / max(
                required_per_finger,
                1e-9,
            )
            - 1.0
        )

        return ForceAnalysis(
            object_weight_n=float(
                weight
            ),
            required_force_per_finger_n=float(
                required_per_finger
            ),
            available_force_per_finger_n=float(
                available
            ),
            force_margin_n=float(
                margin
            ),
            force_score=force_score,
            feasible=feasible,
        )


# ============================================================
# TORQUE ANALYSIS
# ============================================================

class GraspTorqueAnalyzer:

    @staticmethod
    def analyze(
        object_properties: ObjectProperties,
        geometry: GeometryAnalysis,
        gripper: GripperSpecification,
        config: GraspQualityConfig,
    ) -> TorqueAnalysis:
        grasp_center = vector3(
            geometry.grasp_center_xyz,
            "grasp center",
        )

        center_of_mass = vector3(
            object_properties.center_of_mass_xyz,
            "center of mass",
        )

        lever_arm = (
            center_of_mass
            - grasp_center
        )

        gravity_force = np.array(
            [
                0.0,
                0.0,
                -object_properties.mass_kg
                * config.gravity_m_s2,
            ]
        )

        torque_vector = np.cross(
            lever_arm,
            gravity_force,
        )

        estimated_torque = float(
            np.linalg.norm(
                torque_vector
            )
        )

        maximum_torque = (
            gripper.maximum_resisting_torque_nm
        )

        margin = (
            maximum_torque
            - estimated_torque
        )

        feasible = bool(
            estimated_torque <= maximum_torque
        )

        score = clamp(
            1.0
            - estimated_torque
            / maximum_torque
        )

        return TorqueAnalysis(
            estimated_gravity_torque_nm=estimated_torque,
            maximum_resisting_torque_nm=maximum_torque,
            torque_margin_nm=float(
                margin
            ),
            torque_score=score,
            feasible=feasible,
        )


# ============================================================
# ROBUSTNESS ANALYSIS
# ============================================================

class GraspRobustnessAnalyzer:

    @staticmethod
    def analyze(
        object_properties: ObjectProperties,
        gripper: GripperSpecification,
        config: GraspQualityConfig,
    ) -> RobustnessAnalysis:
        nominal_friction = (
            object_properties.friction_coefficient
        )

        reduced_friction = max(
            1e-6,
            nominal_friction
            * (
                1.0
                - config.robustness_friction_reduction
            ),
        )

        weight = (
            object_properties.mass_kg
            * config.gravity_m_s2
        )

        nominal_required = (
            weight
            * config.force_safety_factor
            / (
                2.0 * nominal_friction
            )
        )

        robust_required = (
            weight
            * config.force_safety_factor
            / (
                2.0 * reduced_friction
            )
        )

        available = (
            gripper.maximum_normal_force_per_finger_n
        )

        feasible = bool(
            robust_required <= available
        )

        score = clamp(
            available
            / max(
                robust_required,
                1e-9,
            )
            - 1.0
        )

        return RobustnessAnalysis(
            nominal_friction_coefficient=float(
                nominal_friction
            ),
            reduced_friction_coefficient=float(
                reduced_friction
            ),
            nominal_required_force_n=float(
                nominal_required
            ),
            robust_required_force_n=float(
                robust_required
            ),
            robustness_score=score,
            robust_force_feasible=feasible,
        )


# ============================================================
# QUALITY SCORING
# ============================================================

class WeightedGraspQualityScorer:

    def __init__(
        self,
        weights: GraspQualityWeights,
        config: GraspQualityConfig,
    ) -> None:
        self.weights = weights.normalized()
        self.config = config

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

    def score(
        self,
        candidate: GraspCandidate,
        object_properties: ObjectProperties,
        antipodal: AntipodalAnalysis,
        force: ForceAnalysis,
        torque: TorqueAnalysis,
        geometry: GeometryAnalysis,
        robustness: RobustnessAnalysis,
    ) -> GraspQualityBreakdown:
        antipodal_score = (
            antipodal.antipodal_score
        )

        friction_score = clamp(
            object_properties.friction_coefficient
            / max(
                self.config.minimum_friction_coefficient
                * 4.0,
                1e-9,
            )
        )

        force_score = (
            force.force_score
        )

        geometry_score = (
            geometry.geometry_score
        )

        torque_score = (
            torque.torque_score
        )

        if candidate.collision_clearance_m is None:
            # Unknown clearance is not equivalent to
            # verified collision freedom.
            clearance_score = 0.5
        else:
            clearance_score = clamp(
                candidate.collision_clearance_m
                / max(
                    self.config.minimum_collision_clearance_m
                    * 4.0,
                    1e-9,
                )
            )

        confidence_score = clamp(
            candidate.detection_confidence
            * candidate.left_contact.confidence
            * candidate.right_contact.confidence
            * (1.0 - candidate.uncertainty)
        )

        robustness_score = (
            robustness.robustness_score
        )

        metrics = {
            "antipodal": antipodal_score,
            "friction": friction_score,
            "force": force_score,
            "geometry": geometry_score,
            "torque": torque_score,
            "clearance": clearance_score,
            "confidence": confidence_score,
            "robustness": robustness_score,
        }

        total = sum(
            self.weights[name]
            * clamp(value)
            for name, value in metrics.items()
        )

        total = clamp(
            total
        )

        return GraspQualityBreakdown(
            antipodal_score=antipodal_score,
            friction_score=friction_score,
            force_score=force_score,
            geometry_score=geometry_score,
            torque_score=torque_score,
            clearance_score=clearance_score,
            confidence_score=confidence_score,
            robustness_score=robustness_score,
            total_score=total,
            quality_level=self.quality_level(
                total
            ),
        )


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

class GraspRecommendationEngine:

    @staticmethod
    def generate(
        reasons: Sequence[GraspRejectionReason],
    ) -> tuple[str, ...]:
        mapping = {
            GraspRejectionReason.LOW_CONFIDENCE:
                "Improve perception confidence or collect another view.",

            GraspRejectionReason.INVALID_APERTURE:
                "Choose a different contact pair or compatible gripper.",

            GraspRejectionReason.INSUFFICIENT_FRICTION:
                "Use verified higher-friction contact surfaces.",

            GraspRejectionReason.POOR_ANTIPODAL_ALIGNMENT:
                "Select more opposing surface contacts.",

            GraspRejectionReason.INSUFFICIENT_CONTACT_FORCE:
                "Reduce the load or select an appropriately rated gripper.",

            GraspRejectionReason.EXCESSIVE_PAYLOAD:
                "Use a gripper and robot rated for the object mass.",

            GraspRejectionReason.EXCESSIVE_TORQUE:
                "Move the grasp center closer to the center of mass.",

            GraspRejectionReason.LOW_COLLISION_CLEARANCE:
                "Perform full 3D collision checking and revise the approach.",

            GraspRejectionReason.HIGH_UNCERTAINTY:
                "Improve depth, contact, and material estimates.",

            GraspRejectionReason.LOW_QUALITY_SCORE:
                "Evaluate alternative grasp candidates.",
        }

        return tuple(
            mapping[reason]
            for reason in dict.fromkeys(reasons)
            if reason in mapping
        )


# ============================================================
# GRASP QUALITY EVALUATOR
# ============================================================

class GraspQualityEvaluator:

    def __init__(
        self,
        *,
        robot_id: str,
        tenant_id: str,
        gripper: GripperSpecification | None = None,
        config: GraspQualityConfig | None = None,
        weights: GraspQualityWeights | None = None,
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

        self.config = (
            config
            or GraspQualityConfig()
        )

        self.weights = (
            weights
            or GraspQualityWeights()
        )

        self.scorer = (
            WeightedGraspQualityScorer(
                self.weights,
                self.config,
            )
        )

        self._lock = threading.RLock()

    def evaluate(
        self,
        candidate: GraspCandidate,
        object_properties: ObjectProperties,
    ) -> GraspQualityReport:
        if (
            candidate.object_id
            != object_properties.object_id
        ):
            raise GraspValidationError(
                "Candidate and object identifiers do not match"
            )

        with self._lock:
            geometry = ContactGeometryAnalyzer.analyze(
                candidate,
                object_properties,
                self.gripper,
                self.config,
            )

            antipodal = AntipodalGraspAnalyzer.analyze(
                candidate,
                object_properties.friction_coefficient,
                self.config,
            )

            force = GraspForceAnalyzer.analyze(
                object_properties,
                self.gripper,
                self.config,
            )

            torque = GraspTorqueAnalyzer.analyze(
                object_properties,
                geometry,
                self.gripper,
                self.config,
            )

            robustness = GraspRobustnessAnalyzer.analyze(
                object_properties,
                self.gripper,
                self.config,
            )

            quality = self.scorer.score(
                candidate,
                object_properties,
                antipodal,
                force,
                torque,
                geometry,
                robustness,
            )

            reasons: list[
                GraspRejectionReason
            ] = []

            if (
                candidate.detection_confidence
                < self.config.minimum_detection_confidence
            ):
                reasons.append(
                    GraspRejectionReason.LOW_CONFIDENCE
                )

            if not geometry.aperture_feasible:
                reasons.append(
                    GraspRejectionReason.INVALID_APERTURE
                )

            if (
                object_properties.friction_coefficient
                < self.config.minimum_friction_coefficient
            ):
                reasons.append(
                    GraspRejectionReason.INSUFFICIENT_FRICTION
                )

            if not antipodal.within_friction_cones:
                reasons.append(
                    GraspRejectionReason.POOR_ANTIPODAL_ALIGNMENT
                )

            if not force.feasible:
                reasons.append(
                    GraspRejectionReason.INSUFFICIENT_CONTACT_FORCE
                )

            if (
                object_properties.mass_kg
                > self.gripper.maximum_payload_kg
            ):
                reasons.append(
                    GraspRejectionReason.EXCESSIVE_PAYLOAD
                )

            if not torque.feasible:
                reasons.append(
                    GraspRejectionReason.EXCESSIVE_TORQUE
                )

            if (
                candidate.collision_clearance_m is not None
                and candidate.collision_clearance_m
                < self.config.minimum_collision_clearance_m
            ):
                reasons.append(
                    GraspRejectionReason.LOW_COLLISION_CLEARANCE
                )

            if (
                candidate.uncertainty
                > self.config.maximum_uncertainty
            ):
                reasons.append(
                    GraspRejectionReason.HIGH_UNCERTAINTY
                )

            if (
                quality.total_score
                < self.config.minimum_quality_score
            ):
                reasons.append(
                    GraspRejectionReason.LOW_QUALITY_SCORE
                )

            reasons = list(
                dict.fromkeys(reasons)
            )

            if reasons:
                status = GraspEvaluationStatus.REJECTED

            elif (
                candidate.collision_clearance_m is None
                or not robustness.robust_force_feasible
            ):
                status = (
                    GraspEvaluationStatus.REVIEW_REQUIRED
                )

            else:
                status = (
                    GraspEvaluationStatus.ACCEPTABLE
                )

            recommendations = (
                GraspRecommendationEngine.generate(
                    reasons
                )
            )

            if (
                candidate.collision_clearance_m is None
            ):
                recommendations += (
                    "Collision clearance is unknown; perform full 3D collision checking.",
                )

            if not robustness.robust_force_feasible:
                recommendations += (
                    "Verify force margin under reduced-friction conditions.",
                )

            return GraspQualityReport(
                report_id=new_id(),
                tenant_id=self.tenant_id,
                robot_id=self.robot_id,
                candidate_id=candidate.candidate_id,
                object_id=candidate.object_id,
                status=status,
                antipodal=antipodal,
                force=force,
                torque=torque,
                geometry=geometry,
                robustness=robustness,
                quality=quality,
                rejection_reasons=tuple(
                    reasons
                ),
                recommendations=recommendations,
                created_at=utc_now(),
            )

    def rank(
        self,
        object_properties: ObjectProperties,
        candidates: Sequence[GraspCandidate],
    ) -> GraspRankingResult:
        with self._lock:
            reports: list[
                GraspQualityReport
            ] = []

            for candidate in candidates:
                if (
                    candidate.object_id
                    != object_properties.object_id
                ):
                    continue

                try:
                    report = self.evaluate(
                        candidate,
                        object_properties,
                    )

                    reports.append(
                        report
                    )

                except (
                    GraspValidationError,
                    GraspGeometryError,
                ):
                    continue

            status_priority = {
                GraspEvaluationStatus.ACCEPTABLE: 2,
                GraspEvaluationStatus.REVIEW_REQUIRED: 1,
                GraspEvaluationStatus.REJECTED: 0,
            }

            reports.sort(
                key=lambda report: (
                    status_priority[report.status],
                    report.quality.total_score,
                ),
                reverse=True,
            )

            reports = reports[
                :self.config.maximum_results
            ]

            acceptable = [
                report
                for report in reports
                if report.status
                == GraspEvaluationStatus.ACCEPTABLE
            ]

            best = (
                acceptable[0]
                if acceptable
                else None
            )

            return GraspRankingResult(
                ranking_id=new_id(),
                tenant_id=self.tenant_id,
                robot_id=self.robot_id,
                object_id=object_properties.object_id,
                reports=tuple(reports),
                best_grasp=best,
                evaluated_candidates=len(
                    reports
                ),
                acceptable_candidates=len(
                    acceptable
                ),
                created_at=utc_now(),
            )


# ============================================================
# MULTI-TENANT SERVICE
# ============================================================

class GraspQualityService:

    def __init__(self) -> None:
        self._evaluators: dict[
            tuple[str, str],
            GraspQualityEvaluator,
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

    def register_evaluator(
        self,
        evaluator: GraspQualityEvaluator,
        access: GraspAccessContext,
    ) -> None:
        self._authorize(
            access,
            "can_manage",
        )

        if (
            evaluator.tenant_id
            != access.tenant_id
        ):
            raise GraspAuthorizationError(
                "Cross-tenant registration forbidden"
            )

        key = (
            evaluator.tenant_id,
            evaluator.robot_id,
        )

        with self._lock:
            if key in self._evaluators:
                raise GraspConflictError(
                    "Evaluator already registered"
                )

            self._evaluators[key] = evaluator

    def get_evaluator(
        self,
        robot_id: str,
        access: GraspAccessContext,
    ) -> GraspQualityEvaluator:
        self._authorize(
            access,
            "can_read",
        )

        key = (
            access.tenant_id,
            robot_id,
        )

        with self._lock:
            evaluator = self._evaluators.get(
                key
            )

        if evaluator is None:
            raise GraspNotFoundError(
                "Grasp evaluator not found"
            )

        return evaluator

    def evaluate(
        self,
        robot_id: str,
        candidate: GraspCandidate,
        object_properties: ObjectProperties,
        access: GraspAccessContext,
    ) -> GraspQualityReport:
        self._authorize(
            access,
            "can_evaluate",
        )

        evaluator = self.get_evaluator(
            robot_id,
            access,
        )

        return evaluator.evaluate(
            candidate,
            object_properties,
        )

    def rank(
        self,
        robot_id: str,
        object_properties: ObjectProperties,
        candidates: Sequence[GraspCandidate],
        access: GraspAccessContext,
    ) -> GraspRankingResult:
        self._authorize(
            access,
            "can_evaluate",
        )

        evaluator = self.get_evaluator(
            robot_id,
            access,
        )

        return evaluator.rank(
            object_properties,
            candidates,
        )

    def list_evaluators(
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
                in self._evaluators
                if tenant_id == access.tenant_id
            )

    def export_json(
        self,
        report: GraspQualityReport | GraspRankingResult,
        access: GraspAccessContext,
    ) -> str:
        self._authorize(
            access,
            "can_read",
        )

        if (
            report.tenant_id
            != access.tenant_id
        ):
            raise GraspAuthorizationError(
                "Cross-tenant report access forbidden"
            )

        return json.dumps(
            json_safe(
                {
                    "report": report,
                    "advisory_only": True,
                    "robot_actuation": False,
                    "certified_force_closure": False,
                    "certified_collision_check": False,
                }
            ),
            indent=2,
            allow_nan=False,
        )


# ============================================================
# SYNTHETIC DATA
# ============================================================

class SyntheticGraspQualityFactory:

    @staticmethod
    def object_properties(
        object_id: str = "box-001",
        mass_kg: float = 0.4,
        friction_coefficient: float = 0.7,
    ) -> ObjectProperties:
        return ObjectProperties(
            object_id=object_id,
            mass_kg=mass_kg,
            friction_coefficient=friction_coefficient,
            center_of_mass_xyz=(
                0.0,
                0.0,
                0.6,
            ),
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
        width_m: float = 0.06,
        confidence: float = 0.95,
        clearance_m: float | None = 0.10,
        uncertainty: float = 0.05,
        aligned: bool = True,
    ) -> GraspCandidate:
        half_width = width_m / 2.0

        left_normal = (
            (-1.0, 0.0, 0.0)
            if aligned
            else (0.0, 1.0, 0.0)
        )

        right_normal = (
            (1.0, 0.0, 0.0)
            if aligned
            else (0.0, -1.0, 0.0)
        )

        return GraspCandidate(
            candidate_id=candidate_id,
            object_id=object_id,
            grasp_type=GraspType.PARALLEL_JAW,
            left_contact=GraspContact(
                position_xyz=(
                    -half_width,
                    0.0,
                    0.6,
                ),
                outward_normal_xyz=left_normal,
                confidence=0.98,
            ),
            right_contact=GraspContact(
                position_xyz=(
                    half_width,
                    0.0,
                    0.6,
                ),
                outward_normal_xyz=right_normal,
                confidence=0.98,
            ),
            detection_confidence=confidence,
            collision_clearance_m=clearance_m,
            uncertainty=uncertainty,
        )

    @staticmethod
    def candidates() -> list[GraspCandidate]:
        return [
            SyntheticGraspQualityFactory.candidate(
                candidate_id="grasp-001",
                confidence=0.98,
                width_m=0.06,
            ),
            SyntheticGraspQualityFactory.candidate(
                candidate_id="grasp-002",
                confidence=0.80,
                width_m=0.05,
                uncertainty=0.15,
            ),
            SyntheticGraspQualityFactory.candidate(
                candidate_id="grasp-003",
                confidence=0.90,
                width_m=0.20,
            ),
            SyntheticGraspQualityFactory.candidate(
                candidate_id="grasp-004",
                confidence=0.90,
                width_m=0.06,
                aligned=False,
            ),
            SyntheticGraspQualityFactory.candidate(
                candidate_id="grasp-005",
                confidence=0.25,
                width_m=0.06,
            ),
        ]

    @staticmethod
    def evaluator(
        robot_id: str = "robot-arm-001",
        tenant_id: str = "physical_ai_lab",
    ) -> GraspQualityEvaluator:
        return GraspQualityEvaluator(
            robot_id=robot_id,
            tenant_id=tenant_id,
            gripper=GripperSpecification(
                minimum_opening_m=0.005,
                maximum_opening_m=0.12,
                maximum_normal_force_per_finger_n=40.0,
                maximum_payload_kg=2.0,
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
            can_evaluate=True,
            can_manage=True,
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestGraspQuality(unittest.TestCase):

    def setUp(self) -> None:
        self.evaluator = (
            SyntheticGraspQualityFactory.evaluator()
        )

        self.object_properties = (
            SyntheticGraspQualityFactory.object_properties()
        )

        self.candidate = (
            SyntheticGraspQualityFactory.candidate()
        )

        self.access = (
            SyntheticGraspQualityFactory.access()
        )

    def test_vector_normalization(self) -> None:
        result = normalize(
            (3.0, 0.0, 0.0)
        )

        self.assertTrue(
            np.allclose(
                result,
                (1.0, 0.0, 0.0),
            )
        )

    def test_angle_between_vectors(self) -> None:
        angle = angle_between(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )

        self.assertAlmostEqual(
            math.degrees(angle),
            90.0,
        )

    def test_contact_distance(self) -> None:
        distance = (
            ContactGeometryAnalyzer.contact_distance(
                self.candidate
            )
        )

        self.assertAlmostEqual(
            distance,
            0.06,
        )

    def test_grasp_center(self) -> None:
        center = (
            ContactGeometryAnalyzer.grasp_center(
                self.candidate
            )
        )

        self.assertTrue(
            np.allclose(
                center,
                (0.0, 0.0, 0.6),
            )
        )

    def test_antipodal_alignment(self) -> None:
        analysis = (
            AntipodalGraspAnalyzer.analyze(
                self.candidate,
                0.7,
                self.evaluator.config,
            )
        )

        self.assertTrue(
            analysis.within_friction_cones
        )

        self.assertAlmostEqual(
            analysis.antipodal_score,
            1.0,
        )

    def test_bad_antipodal_alignment(self) -> None:
        candidate = (
            SyntheticGraspQualityFactory.candidate(
                aligned=False
            )
        )

        analysis = (
            AntipodalGraspAnalyzer.analyze(
                candidate,
                0.7,
                self.evaluator.config,
            )
        )

        self.assertFalse(
            analysis.within_friction_cones
        )

    def test_friction_cone_angle(self) -> None:
        analysis = (
            AntipodalGraspAnalyzer.analyze(
                self.candidate,
                1.0,
                self.evaluator.config,
            )
        )

        self.assertAlmostEqual(
            analysis.friction_cone_half_angle_degrees,
            45.0,
        )

    def test_force_feasible(self) -> None:
        analysis = GraspForceAnalyzer.analyze(
            self.object_properties,
            self.evaluator.gripper,
            self.evaluator.config,
        )

        self.assertTrue(
            analysis.feasible
        )

    def test_heavy_object_rejected(self) -> None:
        heavy = (
            SyntheticGraspQualityFactory.object_properties(
                mass_kg=10.0
            )
        )

        report = self.evaluator.evaluate(
            self.candidate,
            heavy,
        )

        self.assertEqual(
            report.status,
            GraspEvaluationStatus.REJECTED,
        )

    def test_torque_at_center_of_mass(self) -> None:
        geometry = (
            ContactGeometryAnalyzer.analyze(
                self.candidate,
                self.object_properties,
                self.evaluator.gripper,
                self.evaluator.config,
            )
        )

        torque = GraspTorqueAnalyzer.analyze(
            self.object_properties,
            geometry,
            self.evaluator.gripper,
            self.evaluator.config,
        )

        self.assertAlmostEqual(
            torque.estimated_gravity_torque_nm,
            0.0,
        )

    def test_aperture_feasible(self) -> None:
        geometry = (
            ContactGeometryAnalyzer.analyze(
                self.candidate,
                self.object_properties,
                self.evaluator.gripper,
                self.evaluator.config,
            )
        )

        self.assertTrue(
            geometry.aperture_feasible
        )

    def test_aperture_rejected(self) -> None:
        candidate = (
            SyntheticGraspQualityFactory.candidate(
                width_m=0.20
            )
        )

        report = self.evaluator.evaluate(
            candidate,
            self.object_properties,
        )

        self.assertIn(
            GraspRejectionReason.INVALID_APERTURE,
            report.rejection_reasons,
        )

    def test_quality_score_range(self) -> None:
        report = self.evaluator.evaluate(
            self.candidate,
            self.object_properties,
        )

        self.assertGreaterEqual(
            report.quality.total_score,
            0.0,
        )

        self.assertLessEqual(
            report.quality.total_score,
            1.0,
        )

    def test_good_grasp(self) -> None:
        report = self.evaluator.evaluate(
            self.candidate,
            self.object_properties,
        )

        self.assertEqual(
            report.status,
            GraspEvaluationStatus.ACCEPTABLE,
        )

    def test_low_confidence_rejected(self) -> None:
        candidate = (
            SyntheticGraspQualityFactory.candidate(
                confidence=0.10
            )
        )

        report = self.evaluator.evaluate(
            candidate,
            self.object_properties,
        )

        self.assertIn(
            GraspRejectionReason.LOW_CONFIDENCE,
            report.rejection_reasons,
        )

    def test_collision_clearance_rejected(self) -> None:
        candidate = (
            SyntheticGraspQualityFactory.candidate(
                clearance_m=0.001
            )
        )

        report = self.evaluator.evaluate(
            candidate,
            self.object_properties,
        )

        self.assertIn(
            GraspRejectionReason.LOW_COLLISION_CLEARANCE,
            report.rejection_reasons,
        )

    def test_unknown_clearance_requires_review(self) -> None:
        candidate = (
            SyntheticGraspQualityFactory.candidate(
                clearance_m=None
            )
        )

        report = self.evaluator.evaluate(
            candidate,
            self.object_properties,
        )

        self.assertEqual(
            report.status,
            GraspEvaluationStatus.REVIEW_REQUIRED,
        )

    def test_high_uncertainty_rejected(self) -> None:
        candidate = (
            SyntheticGraspQualityFactory.candidate(
                uncertainty=0.80
            )
        )

        report = self.evaluator.evaluate(
            candidate,
            self.object_properties,
        )

        self.assertIn(
            GraspRejectionReason.HIGH_UNCERTAINTY,
            report.rejection_reasons,
        )

    def test_robustness_analysis(self) -> None:
        robustness = (
            GraspRobustnessAnalyzer.analyze(
                self.object_properties,
                self.evaluator.gripper,
                self.evaluator.config,
            )
        )

        self.assertLess(
            robustness.reduced_friction_coefficient,
            robustness.nominal_friction_coefficient,
        )

        self.assertGreater(
            robustness.robust_required_force_n,
            robustness.nominal_required_force_n,
        )

    def test_candidate_ranking(self) -> None:
        result = self.evaluator.rank(
            self.object_properties,
            SyntheticGraspQualityFactory.candidates(),
        )

        self.assertIsNotNone(
            result.best_grasp
        )

        self.assertEqual(
            result.best_grasp.status,
            GraspEvaluationStatus.ACCEPTABLE,
        )

    def test_empty_candidate_ranking(self) -> None:
        result = self.evaluator.rank(
            self.object_properties,
            [],
        )

        self.assertIsNone(
            result.best_grasp
        )

    def test_service_registration(self) -> None:
        service = GraspQualityService()

        service.register_evaluator(
            self.evaluator,
            self.access,
        )

        self.assertEqual(
            service.list_evaluators(
                self.access
            ),
            ("robot-arm-001",),
        )

    def test_service_evaluation(self) -> None:
        service = GraspQualityService()

        service.register_evaluator(
            self.evaluator,
            self.access,
        )

        report = service.evaluate(
            "robot-arm-001",
            self.candidate,
            self.object_properties,
            self.access,
        )

        self.assertEqual(
            report.robot_id,
            "robot-arm-001",
        )

    def test_tenant_isolation(self) -> None:
        service = GraspQualityService()

        service.register_evaluator(
            self.evaluator,
            self.access,
        )

        other = GraspAccessContext(
            tenant_id="other_tenant",
            user_id="other_user",
            can_read=True,
            can_evaluate=True,
        )

        with self.assertRaises(
            GraspNotFoundError
        ):
            service.get_evaluator(
                "robot-arm-001",
                other,
            )

    def test_permission_denied(self) -> None:
        service = GraspQualityService()

        service.register_evaluator(
            self.evaluator,
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
            service.evaluate(
                "robot-arm-001",
                self.candidate,
                self.object_properties,
                viewer,
            )

    def test_json_export(self) -> None:
        service = GraspQualityService()

        service.register_evaluator(
            self.evaluator,
            self.access,
        )

        report = service.evaluate(
            "robot-arm-001",
            self.candidate,
            self.object_properties,
            self.access,
        )

        exported = json.loads(
            service.export_json(
                report,
                self.access,
            )
        )

        self.assertTrue(
            exported["advisory_only"]
        )

        self.assertFalse(
            exported["robot_actuation"]
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print("\nPHYSICAL AI — GRASP QUALITY EVALUATION")
    print("=" * 72)

    evaluator = (
        SyntheticGraspQualityFactory.evaluator(
            robot_id="physical-ai-robot-arm-001"
        )
    )

    access = (
        SyntheticGraspQualityFactory.access()
    )

    service = GraspQualityService()

    service.register_evaluator(
        evaluator,
        access,
    )

    object_properties = (
        SyntheticGraspQualityFactory.object_properties()
    )

    candidates = (
        SyntheticGraspQualityFactory.candidates()
    )

    result = service.rank(
        evaluator.robot_id,
        object_properties,
        candidates,
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
        "Acceptable candidates:",
        result.acceptable_candidates,
    )

    if result.best_grasp is not None:
        best = result.best_grasp

        print(
            "\nBest candidate:",
            best.candidate_id,
        )

        print(
            "Quality score:",
            round(
                best.quality.total_score,
                4,
            ),
        )

        print(
            "Quality level:",
            best.quality.quality_level.value,
        )

        print(
            "Antipodal score:",
            round(
                best.antipodal.antipodal_score,
                4,
            ),
        )

        print(
            "Required force per finger:",
            round(
                best.force.required_force_per_finger_n,
                3,
            ),
            "N",
        )

        print(
            "Estimated gravity torque:",
            round(
                best.torque.estimated_gravity_torque_nm,
                4,
            ),
            "Nm",
        )

        print(
            "Evaluation status:",
            best.status.value,
        )

    print("\nGRASP QUALITY REPORT")
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
        TestGraspQuality
    )

    test_result = unittest.TextTestRunner(
        verbosity=2
    ).run(suite)

    if not test_result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
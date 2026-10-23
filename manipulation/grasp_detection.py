"""Grasp Detection module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/manipulation/grasp_detection.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================
Module: Grasp Detection

Features:
    - RGB-D grasp candidate detection
    - Depth-image preprocessing
    - Camera intrinsic calibration
    - Pixel-to-3D point projection
    - Object-mask-based grasp detection
    - Parallel-jaw grasp generation
    - Surface-normal estimation
    - Principal component analysis (PCA)
    - Grasp width estimation
    - Grasp quality scoring
    - Gripper width validation
    - Depth discontinuity detection
    - Surface flatness estimation
    - Contact point estimation
    - Grasp approach direction
    - Grasp pose generation
    - Collision clearance approximation
    - Grasp candidate ranking
    - Non-maximum suppression
    - Confidence filtering
    - Object-specific grasp results
    - Multi-object grasp detection
    - Thread-safe multi-tenant service
    - Synthetic RGB-D test scenes
    - Unit tests
    - JSON export

Requirements:
    Python >= 3.11
    numpy >= 1.24

Install:
    pip install numpy

Safety:
    This module generates advisory grasp candidates only.
    It does not command a robot or close a gripper.

    Collision checking is approximate and image-based.
    Real-world grasp execution requires calibrated
    camera-to-robot transforms, 3D collision checking,
    gripper geometry, force limits, and independent
    safety validation.
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

class GraspDetectionError(Exception):
    pass


class GraspValidationError(GraspDetectionError):
    pass


class GraspNotFoundError(GraspDetectionError):
    pass


class GraspAuthorizationError(GraspDetectionError):
    pass


class GraspConfigurationError(GraspDetectionError):
    pass


class GraspProcessingError(GraspDetectionError):
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
        raise GraspValidationError(
            f"{name} must be numeric"
        )

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GraspValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(number):
        raise GraspValidationError(
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
        raise GraspValidationError(
            f"Invalid {name}"
        )

    return value.strip()


def normalize_vector(
    vector: Sequence[float],
    *,
    fallback: Sequence[float] | None = None,
) -> np.ndarray:
    values = np.asarray(
        vector,
        dtype=float,
    )

    if values.shape != (3,):
        raise GraspValidationError(
            "Vector must contain three elements"
        )

    if not np.all(np.isfinite(values)):
        raise GraspValidationError(
            "Vector contains non-finite values"
        )

    norm = float(
        np.linalg.norm(values)
    )

    if norm <= 1e-12:
        if fallback is None:
            raise GraspValidationError(
                "Cannot normalize zero vector"
            )

        return normalize_vector(
            fallback
        )

    return values / norm


def vector_tuple(
    vector: Sequence[float],
) -> tuple[float, ...]:
    return tuple(
        float(value)
        for value in vector
    )


def matrix_tuple(
    matrix: np.ndarray,
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        vector_tuple(row)
        for row in matrix
    )


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
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

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, float):
        if not math.isfinite(value):
            raise GraspValidationError(
                "Cannot serialize non-finite number"
            )

    if isinstance(
        value,
        (str, int, float, bool),
    ) or value is None:
        return value

    raise GraspValidationError(
        f"Unsupported serialization type: "
        f"{type(value).__name__}"
    )


# ============================================================
# ENUMERATIONS
# ============================================================

class GraspType(str, Enum):
    PARALLEL_JAW = "parallel_jaw"
    TOP_DOWN = "top_down"
    SIDE_GRASP = "side_grasp"


class GraspStatus(str, Enum):
    VALID = "valid"
    LOW_CONFIDENCE = "low_confidence"
    WIDTH_EXCEEDED = "width_exceeded"
    INSUFFICIENT_DEPTH = "insufficient_depth"
    COLLISION_RISK = "collision_risk"
    INVALID_GEOMETRY = "invalid_geometry"


class GraspQualityLevel(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"


class DetectionStatus(str, Enum):
    SUCCESS = "success"
    NO_OBJECT = "no_object"
    NO_VALID_GRASP = "no_valid_grasp"
    INSUFFICIENT_DEPTH = "insufficient_depth"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    width: int
    height: int

    depth_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "fx",
            "fy",
            "cx",
            "cy",
            "depth_scale",
        ):
            finite_number(
                getattr(self, name),
                name,
            )

        if self.fx <= 0 or self.fy <= 0:
            raise GraspConfigurationError(
                "Focal lengths must be positive"
            )

        if self.depth_scale <= 0:
            raise GraspConfigurationError(
                "Depth scale must be positive"
            )

        if self.width <= 0 or self.height <= 0:
            raise GraspConfigurationError(
                "Image dimensions must be positive"
            )


@dataclass(frozen=True)
class GripperConfig:
    minimum_width_m: float = 0.005
    maximum_width_m: float = 0.100

    finger_length_m: float = 0.060
    finger_thickness_m: float = 0.010

    grasp_depth_m: float = 0.025

    safety_margin_m: float = 0.005

    def __post_init__(self) -> None:
        fields = (
            "minimum_width_m",
            "maximum_width_m",
            "finger_length_m",
            "finger_thickness_m",
            "grasp_depth_m",
            "safety_margin_m",
        )

        for name in fields:
            value = finite_number(
                getattr(self, name),
                name,
            )

            if value < 0:
                raise GraspConfigurationError(
                    f"{name} cannot be negative"
                )

        if self.maximum_width_m <= self.minimum_width_m:
            raise GraspConfigurationError(
                "Maximum gripper width must exceed minimum"
            )

        if self.finger_length_m <= 0:
            raise GraspConfigurationError(
                "Finger length must be positive"
            )


@dataclass(frozen=True)
class GraspDetectionConfig:
    minimum_depth_m: float = 0.05
    maximum_depth_m: float = 3.0

    minimum_object_pixels: int = 25

    maximum_candidates: int = 50

    minimum_confidence: float = 0.35

    depth_discontinuity_threshold_m: float = 0.04

    normal_window_radius: int = 3

    candidate_stride_pixels: int = 8

    suppression_distance_pixels: float = 12.0

    suppression_angle_radians: float = 0.35

    surface_flatness_scale_m: float = 0.02

    random_seed: int = 42

    def __post_init__(self) -> None:
        if not (
            0 < self.minimum_depth_m
            < self.maximum_depth_m
        ):
            raise GraspConfigurationError(
                "Invalid depth range"
            )

        if self.minimum_object_pixels < 3:
            raise GraspConfigurationError(
                "Minimum object pixels must be >= 3"
            )

        if self.maximum_candidates < 1:
            raise GraspConfigurationError(
                "Maximum candidates must be positive"
            )

        if not 0 <= self.minimum_confidence <= 1:
            raise GraspConfigurationError(
                "Confidence threshold must be in [0, 1]"
            )

        if self.normal_window_radius < 1:
            raise GraspConfigurationError(
                "Normal window radius must be positive"
            )

        if self.candidate_stride_pixels < 1:
            raise GraspConfigurationError(
                "Candidate stride must be positive"
            )

        if self.suppression_distance_pixels < 0:
            raise GraspConfigurationError(
                "Suppression distance cannot be negative"
            )

        if self.suppression_angle_radians < 0:
            raise GraspConfigurationError(
                "Suppression angle cannot be negative"
            )

        if self.surface_flatness_scale_m <= 0:
            raise GraspConfigurationError(
                "Flatness scale must be positive"
            )


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class GraspAccessContext:
    tenant_id: str
    user_id: str

    can_read: bool = False
    can_detect: bool = False
    can_manage: bool = False


@dataclass(frozen=True)
class GraspPose:
    position: tuple[float, float, float]

    rotation_matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]

    approach_direction: tuple[float, float, float]

    closing_direction: tuple[float, float, float]


@dataclass(frozen=True)
class GraspQuality:
    confidence: float

    geometry_score: float
    surface_score: float
    width_score: float
    clearance_score: float
    centering_score: float

    quality_level: GraspQualityLevel


@dataclass(frozen=True)
class GraspCandidate:
    candidate_id: str
    object_id: str

    grasp_type: GraspType

    center_pixel: tuple[int, int]

    angle_radians: float

    grasp_width_m: float

    contact_point_left: tuple[float, float, float]
    contact_point_right: tuple[float, float, float]

    pose: GraspPose

    quality: GraspQuality

    status: GraspStatus

    timestamp: datetime


@dataclass(frozen=True)
class ObjectGraspResult:
    object_id: str

    detection_status: DetectionStatus

    candidates: tuple[GraspCandidate, ...]

    best_grasp: GraspCandidate | None

    valid_depth_pixels: int

    processing_timestamp: datetime


@dataclass(frozen=True)
class GraspDetectionResult:
    detection_id: str

    tenant_id: str
    robot_id: str

    objects: tuple[ObjectGraspResult, ...]

    total_candidates: int

    successful_objects: int

    timestamp: datetime


# ============================================================
# RGB-D FRAME
# ============================================================

@dataclass
class RGBDFrame:
    depth: np.ndarray

    intrinsics: CameraIntrinsics

    rgb: np.ndarray | None = None

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        self.depth = np.asarray(
            self.depth,
            dtype=float,
        ).copy()

        expected = (
            self.intrinsics.height,
            self.intrinsics.width,
        )

        if self.depth.shape != expected:
            raise GraspValidationError(
                "Depth image dimensions do not match "
                "camera intrinsics"
            )

        if self.rgb is not None:
            self.rgb = np.asarray(
                self.rgb
            ).copy()

            if self.rgb.shape != (
                self.intrinsics.height,
                self.intrinsics.width,
                3,
            ):
                raise GraspValidationError(
                    "RGB image must have shape HxWx3"
                )

        if (
            not isinstance(self.timestamp, datetime)
            or self.timestamp.tzinfo is None
        ):
            raise GraspValidationError(
                "Frame timestamp must be timezone-aware"
            )


@dataclass
class ObjectMask:
    object_id: str

    mask: np.ndarray

    def __post_init__(self) -> None:
        validate_identifier(
            self.object_id,
            "object_id",
        )

        self.mask = np.asarray(
            self.mask,
            dtype=bool,
        ).copy()

        if self.mask.ndim != 2:
            raise GraspValidationError(
                "Object mask must be two-dimensional"
            )


# ============================================================
# DEPTH PROCESSING
# ============================================================

class DepthProcessor:

    @staticmethod
    def metric_depth(
        frame: RGBDFrame,
    ) -> np.ndarray:
        return (
            frame.depth
            * frame.intrinsics.depth_scale
        )

    @staticmethod
    def valid_depth_mask(
        depth_m: np.ndarray,
        minimum_depth: float,
        maximum_depth: float,
    ) -> np.ndarray:
        return (
            np.isfinite(depth_m)
            & (depth_m >= minimum_depth)
            & (depth_m <= maximum_depth)
        )

    @staticmethod
    def median_filter(
        depth: np.ndarray,
        window_size: int = 3,
    ) -> np.ndarray:
        if (
            window_size < 1
            or window_size % 2 == 0
        ):
            raise GraspValidationError(
                "Median window must be positive and odd"
            )

        image = np.asarray(
            depth,
            dtype=float,
        )

        height, width = image.shape

        radius = window_size // 2

        result = image.copy()

        for y in range(height):
            y0 = max(
                0,
                y - radius,
            )

            y1 = min(
                height,
                y + radius + 1,
            )

            for x in range(width):
                x0 = max(
                    0,
                    x - radius,
                )

                x1 = min(
                    width,
                    x + radius + 1,
                )

                region = image[
                    y0:y1,
                    x0:x1,
                ]

                valid = region[
                    np.isfinite(region)
                    & (region > 0)
                ]

                if valid.size:
                    result[y, x] = np.median(
                        valid
                    )

        return result

    @staticmethod
    def depth_discontinuity(
        depth: np.ndarray,
        x: int,
        y: int,
        radius: int = 3,
    ) -> float:
        height, width = depth.shape

        x0 = max(
            0,
            x - radius,
        )

        x1 = min(
            width,
            x + radius + 1,
        )

        y0 = max(
            0,
            y - radius,
        )

        y1 = min(
            height,
            y + radius + 1,
        )

        region = depth[
            y0:y1,
            x0:x1,
        ]

        valid = region[
            np.isfinite(region)
            & (region > 0)
        ]

        if valid.size < 3:
            return float("inf")

        return float(
            np.percentile(valid, 90)
            - np.percentile(valid, 10)
        )


# ============================================================
# CAMERA GEOMETRY
# ============================================================

class CameraGeometry:

    @staticmethod
    def pixel_to_point(
        x: float,
        y: float,
        depth_m: float,
        intrinsics: CameraIntrinsics,
    ) -> np.ndarray:
        z = finite_number(
            depth_m,
            "depth_m",
        )

        if z <= 0:
            raise GraspValidationError(
                "Depth must be positive"
            )

        x_camera = (
            (x - intrinsics.cx)
            * z
            / intrinsics.fx
        )

        y_camera = (
            (y - intrinsics.cy)
            * z
            / intrinsics.fy
        )

        return np.array(
            [
                x_camera,
                y_camera,
                z,
            ],
            dtype=float,
        )

    @staticmethod
    def point_to_pixel(
        point: Sequence[float],
        intrinsics: CameraIntrinsics,
    ) -> tuple[float, float]:
        point = np.asarray(
            point,
            dtype=float,
        )

        if point.shape != (3,):
            raise GraspValidationError(
                "Point must be three-dimensional"
            )

        if (
            not np.all(np.isfinite(point))
            or point[2] <= 0
        ):
            raise GraspValidationError(
                "Invalid camera-space point"
            )

        x = (
            intrinsics.fx
            * point[0]
            / point[2]
            + intrinsics.cx
        )

        y = (
            intrinsics.fy
            * point[1]
            / point[2]
            + intrinsics.cy
        )

        return (
            float(x),
            float(y),
        )

    @staticmethod
    def depth_to_point_cloud(
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        height, width = depth_m.shape

        if mask is None:
            valid = (
                np.isfinite(depth_m)
                & (depth_m > 0)
            )

        else:
            valid = (
                np.asarray(
                    mask,
                    dtype=bool,
                )
                & np.isfinite(depth_m)
                & (depth_m > 0)
            )

        ys, xs = np.nonzero(
            valid
        )

        z = depth_m[
            ys,
            xs,
        ]

        x = (
            (xs - intrinsics.cx)
            * z
            / intrinsics.fx
        )

        y = (
            (ys - intrinsics.cy)
            * z
            / intrinsics.fy
        )

        return np.column_stack(
            [
                x,
                y,
                z,
            ]
        )


# ============================================================
# SURFACE NORMAL ESTIMATION
# ============================================================

class SurfaceNormalEstimator:

    @staticmethod
    def estimate(
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        x: int,
        y: int,
        radius: int = 3,
        object_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        height, width = depth_m.shape

        x0 = max(
            0,
            x - radius,
        )

        x1 = min(
            width,
            x + radius + 1,
        )

        y0 = max(
            0,
            y - radius,
        )

        y1 = min(
            height,
            y + radius + 1,
        )

        patch = depth_m[
            y0:y1,
            x0:x1,
        ]

        valid = (
            np.isfinite(patch)
            & (patch > 0)
        )

        if object_mask is not None:
            valid &= object_mask[
                y0:y1,
                x0:x1,
            ]

        local_ys, local_xs = np.nonzero(
            valid
        )

        if len(local_xs) < 6:
            return (
                np.array(
                    [0.0, 0.0, -1.0]
                ),
                float("inf"),
            )

        xs = local_xs + x0
        ys = local_ys + y0

        depths = depth_m[
            ys,
            xs,
        ]

        points = np.column_stack(
            [
                (
                    (xs - intrinsics.cx)
                    * depths
                    / intrinsics.fx
                ),
                (
                    (ys - intrinsics.cy)
                    * depths
                    / intrinsics.fy
                ),
                depths,
            ]
        )

        centroid = np.mean(
            points,
            axis=0,
        )

        centered = (
            points - centroid
        )

        _, _, vh = np.linalg.svd(
            centered,
            full_matrices=False,
        )

        normal = vh[-1]

        normal = normalize_vector(
            normal
        )

        # Orient the normal toward the camera.
        if normal[2] > 0:
            normal = -normal

        distances = (
            centered @ normal
        )

        flatness = float(
            np.sqrt(
                np.mean(
                    distances ** 2
                )
            )
        )

        return (
            normal,
            flatness,
        )


# ============================================================
# OBJECT GEOMETRY ANALYZER
# ============================================================

class ObjectGeometryAnalyzer:

    @staticmethod
    def centroid_pixel(
        mask: np.ndarray,
    ) -> tuple[int, int]:
        ys, xs = np.nonzero(
            mask
        )

        if len(xs) == 0:
            raise GraspProcessingError(
                "Object mask is empty"
            )

        return (
            int(round(np.mean(xs))),
            int(round(np.mean(ys))),
        )

    @staticmethod
    def principal_axes(
        points: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or len(points) < 3
        ):
            raise GraspProcessingError(
                "Insufficient points for PCA"
            )

        centroid = np.mean(
            points,
            axis=0,
        )

        centered = (
            points - centroid
        )

        covariance = (
            centered.T @ centered
        ) / max(
            len(points) - 1,
            1,
        )

        eigenvalues, eigenvectors = np.linalg.eigh(
            covariance
        )

        order = np.argsort(
            eigenvalues
        )[::-1]

        eigenvalues = eigenvalues[
            order
        ]

        eigenvectors = eigenvectors[
            :,
            order,
        ]

        return (
            centroid,
            eigenvalues,
            eigenvectors,
        )

    @staticmethod
    def projected_width(
        points: np.ndarray,
        direction: np.ndarray,
    ) -> float:
        direction = normalize_vector(
            direction
        )

        projections = (
            points @ direction
        )

        return float(
            np.max(projections)
            - np.min(projections)
        )


# ============================================================
# GRASP POSE BUILDER
# ============================================================

class GraspPoseBuilder:

    @staticmethod
    def build(
        center: Sequence[float],
        closing_direction: Sequence[float],
        surface_normal: Sequence[float],
    ) -> GraspPose:
        position = np.asarray(
            center,
            dtype=float,
        )

        if position.shape != (3,):
            raise GraspValidationError(
                "Grasp center must contain three values"
            )

        approach = -normalize_vector(
            surface_normal
        )

        closing = normalize_vector(
            closing_direction
        )

        # Project closing direction into the plane
        # perpendicular to the approach direction.
        closing = (
            closing
            - np.dot(
                closing,
                approach,
            ) * approach
        )

        if np.linalg.norm(closing) < 1e-8:
            reference = np.array(
                [1.0, 0.0, 0.0]
            )

            if abs(
                np.dot(reference, approach)
            ) > 0.9:
                reference = np.array(
                    [0.0, 1.0, 0.0]
                )

            closing = (
                reference
                - np.dot(
                    reference,
                    approach,
                ) * approach
            )

        closing = normalize_vector(
            closing
        )

        lateral = normalize_vector(
            np.cross(
                approach,
                closing,
            )
        )

        closing = normalize_vector(
            np.cross(
                lateral,
                approach,
            )
        )

        # Columns form a right-handed gripper frame:
        # X = closing direction
        # Y = lateral direction
        # Z = approach direction
        rotation = np.column_stack(
            [
                closing,
                lateral,
                approach,
            ]
        )

        return GraspPose(
            position=vector_tuple(
                position
            ),
            rotation_matrix=matrix_tuple(
                rotation
            ),
            approach_direction=vector_tuple(
                approach
            ),
            closing_direction=vector_tuple(
                closing
            ),
        )


# ============================================================
# GRASP QUALITY SCORER
# ============================================================

class GraspQualityScorer:

    @staticmethod
    def quality_level(
        confidence: float,
    ) -> GraspQualityLevel:
        if confidence >= 0.85:
            return GraspQualityLevel.EXCELLENT

        if confidence >= 0.70:
            return GraspQualityLevel.GOOD

        if confidence >= 0.45:
            return GraspQualityLevel.MODERATE

        return GraspQualityLevel.POOR

    @staticmethod
    def score(
        *,
        geometry_score: float,
        surface_score: float,
        width_score: float,
        clearance_score: float,
        centering_score: float,
    ) -> GraspQuality:
        geometry = clamp(
            geometry_score
        )

        surface = clamp(
            surface_score
        )

        width = clamp(
            width_score
        )

        clearance = clamp(
            clearance_score
        )

        centering = clamp(
            centering_score
        )

        confidence = (
            0.25 * geometry
            + 0.25 * surface
            + 0.20 * width
            + 0.20 * clearance
            + 0.10 * centering
        )

        confidence = clamp(
            confidence
        )

        return GraspQuality(
            confidence=confidence,
            geometry_score=geometry,
            surface_score=surface,
            width_score=width,
            clearance_score=clearance,
            centering_score=centering,
            quality_level=(
                GraspQualityScorer.quality_level(
                    confidence
                )
            ),
        )


# ============================================================
# COLLISION CLEARANCE ESTIMATOR
# ============================================================

class GraspClearanceEstimator:
    """
    Estimates image-space clearance around a grasp.

    This is not a substitute for a 3D collision checker.
    """

    @staticmethod
    def estimate(
        object_mask: np.ndarray,
        x: int,
        y: int,
        *,
        radius: int = 8,
    ) -> float:
        height, width = object_mask.shape

        x0 = max(
            0,
            x - radius,
        )

        x1 = min(
            width,
            x + radius + 1,
        )

        y0 = max(
            0,
            y - radius,
        )

        y1 = min(
            height,
            y + radius + 1,
        )

        region = object_mask[
            y0:y1,
            x0:x1,
        ]

        if region.size == 0:
            return 0.0

        occupancy = float(
            np.mean(region)
        )

        return clamp(
            occupancy
        )


# ============================================================
# GRASP CANDIDATE GENERATOR
# ============================================================

class GraspCandidateGenerator:

    def __init__(
        self,
        gripper: GripperConfig,
        config: GraspDetectionConfig,
    ) -> None:
        self.gripper = gripper
        self.config = config

    def _candidate_pixels(
        self,
        mask: np.ndarray,
    ) -> list[tuple[int, int]]:
        ys, xs = np.nonzero(
            mask
        )

        if len(xs) == 0:
            return []

        centroid_x = int(
            round(np.mean(xs))
        )

        centroid_y = int(
            round(np.mean(ys))
        )

        pixels = [
            (
                centroid_x,
                centroid_y,
            )
        ]

        stride = (
            self.config.candidate_stride_pixels
        )

        for index in range(
            0,
            len(xs),
            stride,
        ):
            pixel = (
                int(xs[index]),
                int(ys[index]),
            )

            if pixel not in pixels:
                pixels.append(
                    pixel
                )

        return pixels[
            : self.config.maximum_candidates * 4
        ]

    def _estimate_closing_direction(
        self,
        points: np.ndarray,
        normal: np.ndarray,
    ) -> np.ndarray:
        _, _, eigenvectors = (
            ObjectGeometryAnalyzer.principal_axes(
                points
            )
        )

        # Prefer the second principal axis, typically
        # associated with a narrower object dimension.
        direction = eigenvectors[
            :,
            1,
        ]

        direction = (
            direction
            - np.dot(
                direction,
                normal,
            ) * normal
        )

        if np.linalg.norm(direction) < 1e-8:
            direction = np.array(
                [1.0, 0.0, 0.0]
            )

        return normalize_vector(
            direction
        )

    def _estimate_grasp_width(
        self,
        points: np.ndarray,
        direction: np.ndarray,
    ) -> float:
        width = ObjectGeometryAnalyzer.projected_width(
            points,
            direction,
        )

        return max(
            0.0,
            width,
        )

    def _width_score(
        self,
        width: float,
    ) -> float:
        minimum = (
            self.gripper.minimum_width_m
        )

        maximum = (
            self.gripper.maximum_width_m
            - 2.0
            * self.gripper.safety_margin_m
        )

        if maximum <= minimum:
            return 0.0

        if width < minimum or width > maximum:
            return 0.0

        center = (
            minimum + maximum
        ) / 2.0

        half_range = (
            maximum - minimum
        ) / 2.0

        return clamp(
            1.0
            - 0.5
            * abs(width - center)
            / max(half_range, 1e-9)
        )

    def _surface_score(
        self,
        flatness: float,
    ) -> float:
        if not math.isfinite(flatness):
            return 0.0

        return clamp(
            math.exp(
                -flatness
                / self.config.surface_flatness_scale_m
            )
        )

    def _centering_score(
        self,
        x: int,
        y: int,
        centroid_x: int,
        centroid_y: int,
        mask: np.ndarray,
    ) -> float:
        height, width = mask.shape

        distance = math.hypot(
            x - centroid_x,
            y - centroid_y,
        )

        scale = math.hypot(
            width,
            height,
        )

        return clamp(
            1.0
            - distance
            / max(scale * 0.25, 1.0)
        )

    def generate(
        self,
        frame: RGBDFrame,
        object_mask: ObjectMask,
    ) -> ObjectGraspResult:
        depth_m = DepthProcessor.metric_depth(
            frame
        )

        if object_mask.mask.shape != depth_m.shape:
            raise GraspValidationError(
                "Object mask dimensions mismatch"
            )

        valid_depth = DepthProcessor.valid_depth_mask(
            depth_m,
            self.config.minimum_depth_m,
            self.config.maximum_depth_m,
        )

        valid_mask = (
            object_mask.mask
            & valid_depth
        )

        valid_count = int(
            np.count_nonzero(
                valid_mask
            )
        )

        if valid_count < self.config.minimum_object_pixels:
            return ObjectGraspResult(
                object_id=object_mask.object_id,
                detection_status=(
                    DetectionStatus.INSUFFICIENT_DEPTH
                ),
                candidates=(),
                best_grasp=None,
                valid_depth_pixels=valid_count,
                processing_timestamp=utc_now(),
            )

        points = CameraGeometry.depth_to_point_cloud(
            depth_m,
            frame.intrinsics,
            valid_mask,
        )

        centroid_x, centroid_y = (
            ObjectGeometryAnalyzer.centroid_pixel(
                valid_mask
            )
        )

        candidate_pixels = self._candidate_pixels(
            valid_mask
        )

        candidates: list[GraspCandidate] = []

        for x, y in candidate_pixels:
            if not valid_mask[y, x]:
                continue

            depth = float(
                depth_m[y, x]
            )

            try:
                center = CameraGeometry.pixel_to_point(
                    x,
                    y,
                    depth,
                    frame.intrinsics,
                )

                normal, flatness = (
                    SurfaceNormalEstimator.estimate(
                        depth_m,
                        frame.intrinsics,
                        x,
                        y,
                        radius=(
                            self.config.normal_window_radius
                        ),
                        object_mask=valid_mask,
                    )
                )

                closing = self._estimate_closing_direction(
                    points,
                    normal,
                )

                grasp_width = self._estimate_grasp_width(
                    points,
                    closing,
                )

                pose = GraspPoseBuilder.build(
                    center,
                    closing,
                    normal,
                )

            except (
                GraspValidationError,
                GraspProcessingError,
                np.linalg.LinAlgError,
            ):
                continue

            discontinuity = (
                DepthProcessor.depth_discontinuity(
                    depth_m,
                    x,
                    y,
                    radius=(
                        self.config.normal_window_radius
                    ),
                )
            )

            if math.isfinite(discontinuity):
                geometry_score = clamp(
                    1.0
                    - discontinuity
                    / max(
                        self.config.depth_discontinuity_threshold_m,
                        1e-9,
                    )
                )
            else:
                geometry_score = 0.0

            surface_score = self._surface_score(
                flatness
            )

            width_score = self._width_score(
                grasp_width
            )

            clearance_score = (
                GraspClearanceEstimator.estimate(
                    valid_mask,
                    x,
                    y,
                    radius=(
                        self.config.normal_window_radius * 2
                    ),
                )
            )

            centering_score = self._centering_score(
                x,
                y,
                centroid_x,
                centroid_y,
                valid_mask,
            )

            quality = GraspQualityScorer.score(
                geometry_score=geometry_score,
                surface_score=surface_score,
                width_score=width_score,
                clearance_score=clearance_score,
                centering_score=centering_score,
            )

            maximum_allowed_width = (
                self.gripper.maximum_width_m
                - 2.0
                * self.gripper.safety_margin_m
            )

            if (
                grasp_width
                < self.gripper.minimum_width_m
                or grasp_width > maximum_allowed_width
            ):
                status = GraspStatus.WIDTH_EXCEEDED

            elif (
                quality.confidence
                < self.config.minimum_confidence
            ):
                status = GraspStatus.LOW_CONFIDENCE

            elif clearance_score < 0.15:
                status = GraspStatus.COLLISION_RISK

            else:
                status = GraspStatus.VALID

            half_width = (
                grasp_width / 2.0
            )

            left_contact = (
                center
                - closing * half_width
            )

            right_contact = (
                center
                + closing * half_width
            )

            angle = math.atan2(
                closing[1],
                closing[0],
            )

            candidate = GraspCandidate(
                candidate_id=generate_id(),
                object_id=object_mask.object_id,
                grasp_type=GraspType.PARALLEL_JAW,
                center_pixel=(x, y),
                angle_radians=float(angle),
                grasp_width_m=float(grasp_width),
                contact_point_left=vector_tuple(
                    left_contact
                ),
                contact_point_right=vector_tuple(
                    right_contact
                ),
                pose=pose,
                quality=quality,
                status=status,
                timestamp=utc_now(),
            )

            candidates.append(
                candidate
            )

        candidates.sort(
            key=lambda candidate: (
                candidate.status == GraspStatus.VALID,
                candidate.quality.confidence,
            ),
            reverse=True,
        )

        candidates = GraspNonMaximumSuppression.suppress(
            candidates,
            distance_threshold=(
                self.config.suppression_distance_pixels
            ),
            angle_threshold=(
                self.config.suppression_angle_radians
            ),
            maximum_candidates=(
                self.config.maximum_candidates
            ),
        )

        valid_candidates = [
            candidate
            for candidate in candidates
            if candidate.status == GraspStatus.VALID
        ]

        best = (
            valid_candidates[0]
            if valid_candidates
            else None
        )

        return ObjectGraspResult(
            object_id=object_mask.object_id,
            detection_status=(
                DetectionStatus.SUCCESS
                if best is not None
                else DetectionStatus.NO_VALID_GRASP
            ),
            candidates=tuple(candidates),
            best_grasp=best,
            valid_depth_pixels=valid_count,
            processing_timestamp=utc_now(),
        )


# ============================================================
# NON-MAXIMUM SUPPRESSION
# ============================================================

class GraspNonMaximumSuppression:

    @staticmethod
    def angular_distance(
        angle_a: float,
        angle_b: float,
    ) -> float:
        # Parallel-jaw grasps are equivalent under 180°
        # rotation around their approach axis.
        difference = abs(
            angle_a - angle_b
        ) % math.pi

        return min(
            difference,
            math.pi - difference,
        )

    @staticmethod
    def suppress(
        candidates: Sequence[GraspCandidate],
        *,
        distance_threshold: float,
        angle_threshold: float,
        maximum_candidates: int,
    ) -> list[GraspCandidate]:
        sorted_candidates = sorted(
            candidates,
            key=lambda candidate: (
                candidate.status == GraspStatus.VALID,
                candidate.quality.confidence,
            ),
            reverse=True,
        )

        selected: list[GraspCandidate] = []

        for candidate in sorted_candidates:
            keep = True

            for existing in selected:
                pixel_distance = math.hypot(
                    candidate.center_pixel[0]
                    - existing.center_pixel[0],
                    candidate.center_pixel[1]
                    - existing.center_pixel[1],
                )

                angle_distance = (
                    GraspNonMaximumSuppression.angular_distance(
                        candidate.angle_radians,
                        existing.angle_radians,
                    )
                )

                if (
                    pixel_distance < distance_threshold
                    and angle_distance < angle_threshold
                ):
                    keep = False
                    break

            if keep:
                selected.append(
                    candidate
                )

            if len(selected) >= maximum_candidates:
                break

        return selected


# ============================================================
# GRASP DETECTOR
# ============================================================

class GraspDetector:

    def __init__(
        self,
        *,
        robot_id: str,
        tenant_id: str,
        gripper: GripperConfig | None = None,
        config: GraspDetectionConfig | None = None,
    ) -> None:
        self.robot_id = validate_identifier(
            robot_id,
            "robot_id",
        )

        self.tenant_id = validate_identifier(
            tenant_id,
            "tenant_id",
        )

        self.gripper = (
            gripper or GripperConfig()
        )

        self.config = (
            config or GraspDetectionConfig()
        )

        self.generator = GraspCandidateGenerator(
            self.gripper,
            self.config,
        )

        self._lock = threading.RLock()

    def detect(
        self,
        frame: RGBDFrame,
        objects: Sequence[ObjectMask],
    ) -> GraspDetectionResult:
        with self._lock:
            object_results = []

            for object_mask in objects:
                result = self.generator.generate(
                    frame,
                    object_mask,
                )

                object_results.append(
                    result
                )

            total_candidates = sum(
                len(result.candidates)
                for result in object_results
            )

            successful_objects = sum(
                result.best_grasp is not None
                for result in object_results
            )

            return GraspDetectionResult(
                detection_id=generate_id(),
                tenant_id=self.tenant_id,
                robot_id=self.robot_id,
                objects=tuple(
                    object_results
                ),
                total_candidates=total_candidates,
                successful_objects=successful_objects,
                timestamp=utc_now(),
            )


# ============================================================
# MULTI-TENANT GRASP DETECTION SERVICE
# ============================================================

class GraspDetectionService:

    def __init__(self) -> None:
        self._detectors: dict[
            tuple[str, str],
            GraspDetector,
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

    def register_detector(
        self,
        detector: GraspDetector,
        access: GraspAccessContext,
    ) -> None:
        self._authorize(
            access,
            "can_manage",
        )

        if access.tenant_id != detector.tenant_id:
            raise GraspAuthorizationError(
                "Cross-tenant registration forbidden"
            )

        key = (
            detector.tenant_id,
            detector.robot_id,
        )

        with self._lock:
            if key in self._detectors:
                raise GraspConfigurationError(
                    "Detector already registered"
                )

            self._detectors[key] = detector

    def get_detector(
        self,
        robot_id: str,
        access: GraspAccessContext,
    ) -> GraspDetector:
        self._authorize(
            access,
            "can_read",
        )

        key = (
            access.tenant_id,
            robot_id,
        )

        with self._lock:
            detector = self._detectors.get(
                key
            )

        if detector is None:
            raise GraspNotFoundError(
                "Grasp detector not found"
            )

        return detector

    def detect(
        self,
        robot_id: str,
        frame: RGBDFrame,
        objects: Sequence[ObjectMask],
        access: GraspAccessContext,
    ) -> GraspDetectionResult:
        self._authorize(
            access,
            "can_detect",
        )

        detector = self.get_detector(
            robot_id,
            access,
        )

        return detector.detect(
            frame,
            objects,
        )

    def list_detectors(
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
                in self._detectors
                if tenant_id == access.tenant_id
            )

    def export_result_json(
        self,
        result: GraspDetectionResult,
        access: GraspAccessContext,
    ) -> str:
        self._authorize(
            access,
            "can_read",
        )

        if result.tenant_id != access.tenant_id:
            raise GraspAuthorizationError(
                "Cross-tenant result access forbidden"
            )

        return json.dumps(
            json_safe(
                {
                    "result": result,
                    "simulation_only": True,
                    "robot_actuation": False,
                    "collision_checking": "approximate",
                }
            ),
            indent=2,
            allow_nan=False,
        )


# ============================================================
# SYNTHETIC RGB-D FACTORY
# ============================================================

class SyntheticGraspFactory:

    @staticmethod
    def camera(
        width: int = 128,
        height: int = 128,
    ) -> CameraIntrinsics:
        return CameraIntrinsics(
            fx=180.0,
            fy=180.0,
            cx=(width - 1) / 2.0,
            cy=(height - 1) / 2.0,
            width=width,
            height=height,
        )

    @staticmethod
    def rectangular_object_scene(
        *,
        width: int = 128,
        height: int = 128,
        object_depth_m: float = 0.6,
        background_depth_m: float = 1.0,
        object_width_pixels: int = 20,
        object_height_pixels: int = 30,
    ) -> tuple[RGBDFrame, ObjectMask]:
        camera = SyntheticGraspFactory.camera(
            width,
            height,
        )

        depth = np.full(
            (height, width),
            background_depth_m,
            dtype=float,
        )

        mask = np.zeros(
            (height, width),
            dtype=bool,
        )

        center_x = width // 2
        center_y = height // 2

        x0 = max(
            0,
            center_x - object_width_pixels // 2,
        )

        x1 = min(
            width,
            x0 + object_width_pixels,
        )

        y0 = max(
            0,
            center_y - object_height_pixels // 2,
        )

        y1 = min(
            height,
            y0 + object_height_pixels,
        )

        depth[
            y0:y1,
            x0:x1,
        ] = object_depth_m

        mask[
            y0:y1,
            x0:x1,
        ] = True

        rgb = np.zeros(
            (height, width, 3),
            dtype=np.uint8,
        )

        rgb[mask] = [
            80,
            160,
            240,
        ]

        return (
            RGBDFrame(
                depth=depth,
                intrinsics=camera,
                rgb=rgb,
            ),
            ObjectMask(
                object_id="synthetic-box-001",
                mask=mask,
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
            can_detect=True,
            can_manage=True,
        )

    @staticmethod
    def detector(
        tenant_id: str = "physical_ai_lab",
        robot_id: str = "robot-arm-001",
    ) -> GraspDetector:
        return GraspDetector(
            robot_id=robot_id,
            tenant_id=tenant_id,
            gripper=GripperConfig(
                minimum_width_m=0.005,
                maximum_width_m=0.12,
                safety_margin_m=0.005,
            ),
            config=GraspDetectionConfig(
                minimum_confidence=0.30,
                maximum_candidates=20,
                candidate_stride_pixels=10,
            ),
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestGraspDetection(unittest.TestCase):

    def setUp(self) -> None:
        self.frame, self.object_mask = (
            SyntheticGraspFactory.rectangular_object_scene()
        )

        self.detector = (
            SyntheticGraspFactory.detector()
        )

        self.access = (
            SyntheticGraspFactory.access()
        )

    def test_camera_projection(self) -> None:
        point = CameraGeometry.pixel_to_point(
            self.frame.intrinsics.cx,
            self.frame.intrinsics.cy,
            1.0,
            self.frame.intrinsics,
        )

        self.assertAlmostEqual(
            point[0],
            0.0,
        )

        self.assertAlmostEqual(
            point[1],
            0.0,
        )

        self.assertAlmostEqual(
            point[2],
            1.0,
        )

    def test_inverse_camera_projection(self) -> None:
        point = (
            0.1,
            0.2,
            1.0,
        )

        pixel = CameraGeometry.point_to_pixel(
            point,
            self.frame.intrinsics,
        )

        recovered = CameraGeometry.pixel_to_point(
            pixel[0],
            pixel[1],
            point[2],
            self.frame.intrinsics,
        )

        self.assertTrue(
            np.allclose(
                point,
                recovered,
            )
        )

    def test_valid_depth_mask(self) -> None:
        depth = np.array(
            [
                [0.0, 0.5],
                [1.0, np.nan],
            ]
        )

        valid = DepthProcessor.valid_depth_mask(
            depth,
            0.1,
            2.0,
        )

        self.assertEqual(
            int(np.count_nonzero(valid)),
            2,
        )

    def test_point_cloud_generation(self) -> None:
        depth = DepthProcessor.metric_depth(
            self.frame
        )

        points = CameraGeometry.depth_to_point_cloud(
            depth,
            self.frame.intrinsics,
            self.object_mask.mask,
        )

        self.assertEqual(
            points.shape[1],
            3,
        )

        self.assertGreater(
            len(points),
            0,
        )

    def test_object_centroid(self) -> None:
        x, y = ObjectGeometryAnalyzer.centroid_pixel(
            self.object_mask.mask
        )

        self.assertAlmostEqual(
            x,
            64,
            delta=2,
        )

        self.assertAlmostEqual(
            y,
            64,
            delta=2,
        )

    def test_surface_normal(self) -> None:
        depth = DepthProcessor.metric_depth(
            self.frame
        )

        normal, flatness = (
            SurfaceNormalEstimator.estimate(
                depth,
                self.frame.intrinsics,
                64,
                64,
                object_mask=self.object_mask.mask,
            )
        )

        self.assertAlmostEqual(
            np.linalg.norm(normal),
            1.0,
            places=6,
        )

        self.assertLess(
            flatness,
            1e-5,
        )

    def test_grasp_pose_orthonormal(self) -> None:
        pose = GraspPoseBuilder.build(
            center=(0.0, 0.0, 0.6),
            closing_direction=(1.0, 0.0, 0.0),
            surface_normal=(0.0, 0.0, -1.0),
        )

        rotation = np.asarray(
            pose.rotation_matrix
        )

        self.assertTrue(
            np.allclose(
                rotation.T @ rotation,
                np.eye(3),
                atol=1e-8,
            )
        )

    def test_grasp_quality_score(self) -> None:
        quality = GraspQualityScorer.score(
            geometry_score=1.0,
            surface_score=1.0,
            width_score=1.0,
            clearance_score=1.0,
            centering_score=1.0,
        )

        self.assertAlmostEqual(
            quality.confidence,
            1.0,
        )

    def test_grasp_detection(self) -> None:
        result = self.detector.detect(
            self.frame,
            [self.object_mask],
        )

        self.assertEqual(
            len(result.objects),
            1,
        )

        self.assertGreater(
            result.total_candidates,
            0,
        )

    def test_valid_grasp_candidate(self) -> None:
        result = self.detector.detect(
            self.frame,
            [self.object_mask],
        )

        object_result = result.objects[0]

        self.assertIsNotNone(
            object_result.best_grasp
        )

        self.assertEqual(
            object_result.best_grasp.status,
            GraspStatus.VALID,
        )

    def test_grasp_width(self) -> None:
        result = self.detector.detect(
            self.frame,
            [self.object_mask],
        )

        grasp = result.objects[0].best_grasp

        self.assertIsNotNone(
            grasp
        )

        self.assertGreater(
            grasp.grasp_width_m,
            0.0,
        )

    def test_insufficient_depth(self) -> None:
        empty_depth = np.zeros_like(
            self.frame.depth
        )

        frame = RGBDFrame(
            depth=empty_depth,
            intrinsics=self.frame.intrinsics,
        )

        result = self.detector.detect(
            frame,
            [self.object_mask],
        )

        self.assertEqual(
            result.objects[0].detection_status,
            DetectionStatus.INSUFFICIENT_DEPTH,
        )

    def test_empty_object_mask(self) -> None:
        empty_mask = ObjectMask(
            object_id="empty-object",
            mask=np.zeros_like(
                self.object_mask.mask
            ),
        )

        result = self.detector.detect(
            self.frame,
            [empty_mask],
        )

        self.assertIsNone(
            result.objects[0].best_grasp
        )

    def test_nms(self) -> None:
        result = self.detector.detect(
            self.frame,
            [self.object_mask],
        )

        self.assertLessEqual(
            len(result.objects[0].candidates),
            self.detector.config.maximum_candidates,
        )

    def test_multi_object_detection(self) -> None:
        second_mask = ObjectMask(
            object_id="second-object",
            mask=self.object_mask.mask.copy(),
        )

        result = self.detector.detect(
            self.frame,
            [
                self.object_mask,
                second_mask,
            ],
        )

        self.assertEqual(
            len(result.objects),
            2,
        )

    def test_service_registration(self) -> None:
        service = GraspDetectionService()

        service.register_detector(
            self.detector,
            self.access,
        )

        self.assertEqual(
            service.list_detectors(
                self.access
            ),
            ("robot-arm-001",),
        )

    def test_service_detection(self) -> None:
        service = GraspDetectionService()

        service.register_detector(
            self.detector,
            self.access,
        )

        result = service.detect(
            "robot-arm-001",
            self.frame,
            [self.object_mask],
            self.access,
        )

        self.assertEqual(
            result.robot_id,
            "robot-arm-001",
        )

    def test_tenant_isolation(self) -> None:
        service = GraspDetectionService()

        service.register_detector(
            self.detector,
            self.access,
        )

        other = GraspAccessContext(
            tenant_id="other_tenant",
            user_id="other_user",
            can_read=True,
            can_detect=True,
        )

        with self.assertRaises(
            GraspNotFoundError
        ):
            service.get_detector(
                "robot-arm-001",
                other,
            )

    def test_permission_denied(self) -> None:
        service = GraspDetectionService()

        service.register_detector(
            self.detector,
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
            service.detect(
                "robot-arm-001",
                self.frame,
                [self.object_mask],
                viewer,
            )

    def test_json_export(self) -> None:
        service = GraspDetectionService()

        service.register_detector(
            self.detector,
            self.access,
        )

        result = service.detect(
            "robot-arm-001",
            self.frame,
            [self.object_mask],
            self.access,
        )

        exported = json.loads(
            service.export_result_json(
                result,
                self.access,
            )
        )

        self.assertTrue(
            exported["simulation_only"]
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    frame, object_mask = (
        SyntheticGraspFactory.rectangular_object_scene(
            width=128,
            height=128,
            object_depth_m=0.6,
            object_width_pixels=20,
            object_height_pixels=30,
        )
    )

    access = SyntheticGraspFactory.access()

    detector = SyntheticGraspFactory.detector(
        robot_id="physical-ai-robot-arm-001"
    )

    service = GraspDetectionService()

    service.register_detector(
        detector,
        access,
    )

    print("\nPHYSICAL AI — GRASP DETECTION")
    print("=" * 70)

    result = service.detect(
        detector.robot_id,
        frame,
        [object_mask],
        access,
    )

    print(
        "Robot:",
        result.robot_id,
    )

    print(
        "Detected objects:",
        len(result.objects),
    )

    print(
        "Total grasp candidates:",
        result.total_candidates,
    )

    print(
        "Objects with valid grasps:",
        result.successful_objects,
    )

    for object_result in result.objects:
        print(
            "\nObject:",
            object_result.object_id,
        )

        print(
            "Status:",
            object_result.detection_status.value,
        )

        print(
            "Candidates:",
            len(object_result.candidates),
        )

        grasp = object_result.best_grasp

        if grasp is not None:
            print(
                "Best grasp confidence:",
                round(
                    grasp.quality.confidence,
                    4,
                ),
            )

            print(
                "Grasp width:",
                round(
                    grasp.grasp_width_m,
                    4,
                ),
                "m",
            )

            print(
                "Grasp center:",
                grasp.pose.position,
            )

            print(
                "Approach direction:",
                grasp.pose.approach_direction,
            )

            print(
                "Closing direction:",
                grasp.pose.closing_direction,
            )

            print(
                "Quality:",
                grasp.quality.quality_level.value,
            )

    print("\nGRASP DETECTION REPORT")
    print("=" * 70)

    print(
        service.export_result_json(
            result,
            access,
        )
    )

    print("\nRUNNING UNIT TESTS")
    print("=" * 70)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestGraspDetection
    )

    test_result = unittest.TextTestRunner(
        verbosity=2
    ).run(suite)

    if not test_result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
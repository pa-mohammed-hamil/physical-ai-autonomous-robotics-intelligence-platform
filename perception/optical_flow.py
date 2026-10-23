"""Optical Flow module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/perception/optical_flow.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================

Module: optical_flow.py

Real-Time Optical Flow & Motion Intelligence Engine

Features:
    - Sparse optical flow (Lucas-Kanade)
    - Dense optical flow (Farneback)
    - Feature detection (Shi-Tomasi)
    - Feature point tracking
    - Forward-backward consistency checking
    - Global camera-motion estimation
    - Motion compensation
    - Moving-object region detection
    - Optical flow visualization
    - Motion vector analysis
    - Motion magnitude and direction
    - Frame-to-frame motion estimation
    - Optical flow quality evaluation
    - Camera-specific processing
    - Tenant-aware authorization
    - Frame history
    - Audit logging
    - JSON export
    - Synthetic motion generation
    - Unit tests

Dependencies:
    numpy
    opencv-python

Installation:
    pip install numpy opencv-python

Coordinate System:
    x increases toward image right.
    y increases toward image bottom.

Units:
    Optical flow: pixels per frame
    Velocity: pixels per second when timestamps are supplied

Safety:
    Optical flow provides perception estimates only.
    It must not independently authorize robot movement,
    braking, emergency stops, or actuator commands.
"""

from __future__ import annotations

import json
import math
import threading
import time
import unittest
import uuid

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


# ============================================================
# EXCEPTIONS
# ============================================================

class OpticalFlowError(Exception):
    """Base optical flow exception."""


class OpticalFlowValidationError(OpticalFlowError):
    """Invalid image, configuration, or motion data."""


class OpticalFlowAuthorizationError(OpticalFlowError):
    """Optical flow access denied."""


class OpticalFlowDependencyError(OpticalFlowError):
    """Required dependency unavailable."""


class OpticalFlowProcessingError(OpticalFlowError):
    """Optical flow computation failed."""


class OpticalFlowCapacityError(OpticalFlowError):
    """Configured processing limit exceeded."""


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


def require_opencv() -> None:
    if cv2 is None:
        raise OpticalFlowDependencyError(
            "OpenCV is required: pip install opencv-python"
        )


def finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise OpticalFlowValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise OpticalFlowValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise OpticalFlowValidationError(
            f"{name} must be finite"
        )

    return result


def validate_identifier(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise OpticalFlowValidationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > 128:
        raise OpticalFlowValidationError(
            f"{name} must contain 1-128 characters"
        )

    return value


def validate_timestamp(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise OpticalFlowValidationError(
            "timestamp must be datetime"
        )

    if value.tzinfo is None or value.utcoffset() is None:
        raise OpticalFlowValidationError(
            "timestamp must be timezone-aware"
        )

    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (tuple, list, deque)):
        return [json_safe(item) for item in value]

    if hasattr(value, "__dataclass_fields__"):
        return {
            key: json_safe(getattr(value, key))
            for key in value.__dataclass_fields__
        }

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


# ============================================================
# ENUMERATIONS
# ============================================================

class OpticalFlowMethod(str, Enum):
    SPARSE_LUCAS_KANADE = "sparse_lucas_kanade"
    DENSE_FARNEBACK = "dense_farneback"


class OpticalFlowStatus(str, Enum):
    INITIALIZED = "initialized"
    SUCCESS = "success"
    INSUFFICIENT_FEATURES = "insufficient_features"
    DEGRADED = "degraded"


class MotionDirection(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    UP_LEFT = "up_left"
    UP_RIGHT = "up_right"
    DOWN_LEFT = "down_left"
    DOWN_RIGHT = "down_right"
    STATIONARY = "stationary"


class FlowQuality(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNKNOWN = "unknown"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class OpticalFlowConfig:
    method: OpticalFlowMethod = (
        OpticalFlowMethod.SPARSE_LUCAS_KANADE
    )

    maximum_image_width: int = 4096
    maximum_image_height: int = 4096
    maximum_image_pixels: int = 16_777_216

    maximum_features: int = 500
    minimum_features: int = 8

    feature_quality_level: float = 0.01
    minimum_feature_distance: float = 7.0
    feature_block_size: int = 7

    lk_window_size: int = 21
    lk_max_level: int = 3
    lk_max_iterations: int = 30
    lk_epsilon: float = 0.01

    forward_backward_threshold_px: float = 1.5

    farneback_pyr_scale: float = 0.5
    farneback_levels: int = 3
    farneback_window_size: int = 15
    farneback_iterations: int = 3
    farneback_poly_n: int = 5
    farneback_poly_sigma: float = 1.2

    stationary_threshold_px: float = 0.5
    moving_region_threshold_px: float = 2.0

    minimum_motion_region_area_px: int = 100

    maximum_frame_gap_s: float = 2.0

    maximum_history_frames: int = 50
    maximum_audit_events: int = 1000

    enable_camera_motion_compensation: bool = True
    enable_forward_backward_check: bool = True

    def __post_init__(self) -> None:
        positive_ints = (
            "maximum_image_width",
            "maximum_image_height",
            "maximum_image_pixels",
            "maximum_features",
            "minimum_features",
            "feature_block_size",
            "lk_window_size",
            "lk_max_iterations",
            "farneback_levels",
            "farneback_window_size",
            "farneback_iterations",
            "farneback_poly_n",
            "minimum_motion_region_area_px",
            "maximum_history_frames",
            "maximum_audit_events",
        )

        for name in positive_ints:
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise OpticalFlowValidationError(
                    f"{name} must be a positive integer"
                )

        if self.minimum_features > self.maximum_features:
            raise OpticalFlowValidationError(
                "minimum_features cannot exceed maximum_features"
            )

        if self.lk_max_level < 0:
            raise OpticalFlowValidationError(
                "lk_max_level cannot be negative"
            )

        if self.farneback_poly_n not in (5, 7):
            raise OpticalFlowValidationError(
                "farneback_poly_n must be 5 or 7"
            )

        if self.lk_window_size % 2 == 0:
            raise OpticalFlowValidationError(
                "lk_window_size must be odd"
            )

        for name in (
            "feature_quality_level",
            "minimum_feature_distance",
            "lk_epsilon",
            "forward_backward_threshold_px",
            "farneback_pyr_scale",
            "farneback_poly_sigma",
            "stationary_threshold_px",
            "moving_region_threshold_px",
            "maximum_frame_gap_s",
        ):
            if finite_float(getattr(self, name), name) <= 0:
                raise OpticalFlowValidationError(
                    f"{name} must be positive"
                )

        if self.feature_quality_level > 1:
            raise OpticalFlowValidationError(
                "feature_quality_level must be <= 1"
            )

        if self.farneback_pyr_scale >= 1:
            raise OpticalFlowValidationError(
                "farneback_pyr_scale must be < 1"
            )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class OpticalFlowAccessContext:
    tenant_id: str
    user_id: str

    allowed_robot_ids: frozenset[str] = frozenset()
    allowed_camera_ids: frozenset[str] = frozenset()

    can_process: bool = False
    can_view: bool = False

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id, "tenant_id")
        validate_identifier(self.user_id, "user_id")


# ============================================================
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class MotionVector:
    start_x: float
    start_y: float

    end_x: float
    end_y: float

    dx: float
    dy: float

    magnitude_px: float
    angle_degrees: float

    velocity_x_px_s: float | None = None
    velocity_y_px_s: float | None = None

    @classmethod
    def from_points(
        cls,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        dt_s: float | None = None,
    ) -> MotionVector:
        dx = end_x - start_x
        dy = end_y - start_y

        magnitude = math.hypot(dx, dy)

        angle = math.degrees(
            math.atan2(dy, dx)
        )

        velocity_x = None
        velocity_y = None

        if dt_s is not None and dt_s > 0:
            velocity_x = dx / dt_s
            velocity_y = dy / dt_s

        return cls(
            start_x=float(start_x),
            start_y=float(start_y),
            end_x=float(end_x),
            end_y=float(end_y),
            dx=float(dx),
            dy=float(dy),
            magnitude_px=float(magnitude),
            angle_degrees=float(angle),
            velocity_x_px_s=velocity_x,
            velocity_y_px_s=velocity_y,
        )


@dataclass(frozen=True)
class MotionRegion:
    x1: int
    y1: int
    x2: int
    y2: int

    area_px: int

    mean_motion_px: float
    maximum_motion_px: float

    centroid_x: float
    centroid_y: float

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass(frozen=True)
class GlobalMotionEstimate:
    translation_x_px: float
    translation_y_px: float

    magnitude_px: float

    rotation_degrees: float = 0.0
    scale: float = 1.0

    inlier_count: int = 0
    inlier_ratio: float = 0.0

    reliable: bool = False


@dataclass(frozen=True)
class OpticalFlowQualityReport:
    quality: FlowQuality

    tracked_features: int
    valid_feature_ratio: float

    mean_motion_px: float
    median_motion_px: float
    maximum_motion_px: float

    finite_flow_ratio: float

    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class OpticalFlowFrame:
    frame_id: str

    tenant_id: str
    robot_id: str
    camera_id: str

    method: OpticalFlowMethod
    status: OpticalFlowStatus

    image_width: int
    image_height: int

    timestamp: datetime
    frame_interval_s: float | None

    motion_vectors: tuple[MotionVector, ...]

    global_motion: GlobalMotionEstimate | None

    motion_regions: tuple[MotionRegion, ...]

    quality: OpticalFlowQualityReport

    dominant_direction: MotionDirection

    processing_time_ms: float

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


@dataclass(frozen=True)
class OpticalFlowAuditEvent:
    event_id: str

    tenant_id: str
    user_id: str

    robot_id: str
    camera_id: str

    frame_id: str
    action: str

    timestamp: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# IMAGE VALIDATION
# ============================================================

class OpticalFlowImageValidator:

    def __init__(
        self,
        config: OpticalFlowConfig,
    ) -> None:
        self.config = config

    def validate(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise OpticalFlowValidationError(
                "Image must be a NumPy array"
            )

        if image.dtype != np.uint8:
            raise OpticalFlowValidationError(
                "Image must use uint8 pixels"
            )

        if image.ndim == 2:
            height, width = image.shape

        elif image.ndim == 3 and image.shape[2] == 3:
            height, width = image.shape[:2]

        else:
            raise OpticalFlowValidationError(
                "Expected grayscale or BGR image"
            )

        if (
            height <= 0
            or width <= 0
            or width > self.config.maximum_image_width
            or height > self.config.maximum_image_height
            or width * height > self.config.maximum_image_pixels
        ):
            raise OpticalFlowCapacityError(
                "Image exceeds configured dimensions"
            )

        return image

    def grayscale(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        self.validate(image)

        if image.ndim == 2:
            return np.ascontiguousarray(image)

        require_opencv()

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )


# ============================================================
# FEATURE DETECTOR
# ============================================================

class ShiTomasiFeatureDetector:

    def __init__(
        self,
        config: OpticalFlowConfig,
    ) -> None:
        require_opencv()

        self.config = config

    def detect(
        self,
        grayscale: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        features = cv2.goodFeaturesToTrack(
            grayscale,
            maxCorners=self.config.maximum_features,
            qualityLevel=self.config.feature_quality_level,
            minDistance=self.config.minimum_feature_distance,
            blockSize=self.config.feature_block_size,
            mask=mask,
        )

        if features is None:
            return np.empty(
                (0, 1, 2),
                dtype=np.float32,
            )

        return np.ascontiguousarray(
            features,
            dtype=np.float32,
        )


# ============================================================
# SPARSE LUCAS-KANADE OPTICAL FLOW
# ============================================================

@dataclass(frozen=True)
class SparseFlowResult:
    previous_points: np.ndarray
    current_points: np.ndarray

    motion_vectors: tuple[MotionVector, ...]

    original_feature_count: int
    tracked_feature_count: int

    valid_feature_ratio: float


class SparseLucasKanadeFlow:

    def __init__(
        self,
        config: OpticalFlowConfig,
    ) -> None:
        require_opencv()

        self.config = config

        self.feature_detector = ShiTomasiFeatureDetector(
            config
        )

    def calculate(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        *,
        previous_points: np.ndarray | None = None,
        dt_s: float | None = None,
    ) -> SparseFlowResult:
        if previous_gray.shape != current_gray.shape:
            raise OpticalFlowValidationError(
                "Frame dimensions must match"
            )

        if previous_points is None:
            previous_points = self.feature_detector.detect(
                previous_gray
            )

        previous_points = np.asarray(
            previous_points,
            dtype=np.float32,
        ).reshape(-1, 1, 2)

        original_count = len(previous_points)

        if original_count == 0:
            empty = np.empty(
                (0, 1, 2),
                dtype=np.float32,
            )

            return SparseFlowResult(
                previous_points=empty,
                current_points=empty.copy(),
                motion_vectors=(),
                original_feature_count=0,
                tracked_feature_count=0,
                valid_feature_ratio=0.0,
            )

        criteria = (
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_COUNT,
            self.config.lk_max_iterations,
            self.config.lk_epsilon,
        )

        next_points, status, _ = (
            cv2.calcOpticalFlowPyrLK(
                previous_gray,
                current_gray,
                previous_points,
                None,
                winSize=(
                    self.config.lk_window_size,
                    self.config.lk_window_size,
                ),
                maxLevel=self.config.lk_max_level,
                criteria=criteria,
            )
        )

        if next_points is None or status is None:
            return SparseFlowResult(
                previous_points=np.empty(
                    (0, 1, 2),
                    dtype=np.float32,
                ),
                current_points=np.empty(
                    (0, 1, 2),
                    dtype=np.float32,
                ),
                motion_vectors=(),
                original_feature_count=original_count,
                tracked_feature_count=0,
                valid_feature_ratio=0.0,
            )

        valid = status.reshape(-1) == 1

        valid &= np.all(
            np.isfinite(next_points.reshape(-1, 2)),
            axis=1,
        )

        if self.config.enable_forward_backward_check:
            backward_points, backward_status, _ = (
                cv2.calcOpticalFlowPyrLK(
                    current_gray,
                    previous_gray,
                    next_points,
                    None,
                    winSize=(
                        self.config.lk_window_size,
                        self.config.lk_window_size,
                    ),
                    maxLevel=self.config.lk_max_level,
                    criteria=criteria,
                )
            )

            if (
                backward_points is None
                or backward_status is None
            ):
                valid[:] = False

            else:
                errors = np.linalg.norm(
                    previous_points.reshape(-1, 2)
                    - backward_points.reshape(-1, 2),
                    axis=1,
                )

                valid &= (
                    backward_status.reshape(-1) == 1
                )

                valid &= np.isfinite(errors)

                valid &= (
                    errors
                    <= self.config.forward_backward_threshold_px
                )

        good_previous = previous_points[valid]
        good_current = next_points[valid]

        vectors = []

        for old_point, new_point in zip(
            good_previous.reshape(-1, 2),
            good_current.reshape(-1, 2),
        ):
            vectors.append(
                MotionVector.from_points(
                    float(old_point[0]),
                    float(old_point[1]),
                    float(new_point[0]),
                    float(new_point[1]),
                    dt_s,
                )
            )

        return SparseFlowResult(
            previous_points=good_previous,
            current_points=good_current,
            motion_vectors=tuple(vectors),
            original_feature_count=original_count,
            tracked_feature_count=len(vectors),
            valid_feature_ratio=(
                len(vectors) / original_count
            ),
        )


# ============================================================
# DENSE FARNEBACK OPTICAL FLOW
# ============================================================

@dataclass(frozen=True)
class DenseFlowResult:
    flow: np.ndarray
    magnitude: np.ndarray
    angle_degrees: np.ndarray

    mean_magnitude_px: float
    median_magnitude_px: float
    maximum_magnitude_px: float


class DenseFarnebackFlow:

    def __init__(
        self,
        config: OpticalFlowConfig,
    ) -> None:
        require_opencv()

        self.config = config

    def calculate(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
    ) -> DenseFlowResult:
        if previous_gray.shape != current_gray.shape:
            raise OpticalFlowValidationError(
                "Frame dimensions must match"
            )

        try:
            flow = cv2.calcOpticalFlowFarneback(
                previous_gray,
                current_gray,
                None,
                pyr_scale=self.config.farneback_pyr_scale,
                levels=self.config.farneback_levels,
                winsize=self.config.farneback_window_size,
                iterations=self.config.farneback_iterations,
                poly_n=self.config.farneback_poly_n,
                poly_sigma=self.config.farneback_poly_sigma,
                flags=0,
            )
        except cv2.error as exc:
            raise OpticalFlowProcessingError(
                "Farneback optical flow failed"
            ) from exc

        if not np.all(np.isfinite(flow)):
            raise OpticalFlowProcessingError(
                "Dense optical flow contains invalid values"
            )

        magnitude = np.linalg.norm(
            flow,
            axis=2,
        )

        angle = np.degrees(
            np.arctan2(
                flow[:, :, 1],
                flow[:, :, 0],
            )
        )

        return DenseFlowResult(
            flow=flow,
            magnitude=magnitude,
            angle_degrees=angle,
            mean_magnitude_px=float(
                np.mean(magnitude)
            ),
            median_magnitude_px=float(
                np.median(magnitude)
            ),
            maximum_magnitude_px=float(
                np.max(magnitude)
            ),
        )


# ============================================================
# GLOBAL CAMERA MOTION ESTIMATION
# ============================================================

class GlobalMotionEstimator:

    @staticmethod
    def from_sparse(
        result: SparseFlowResult,
    ) -> GlobalMotionEstimate:
        if result.tracked_feature_count < 3:
            return GlobalMotionEstimate(
                translation_x_px=0.0,
                translation_y_px=0.0,
                magnitude_px=0.0,
                reliable=False,
            )

        previous = result.previous_points.reshape(
            -1,
            2,
        )

        current = result.current_points.reshape(
            -1,
            2,
        )

        if cv2 is not None:
            matrix, inliers = cv2.estimateAffinePartial2D(
                previous,
                current,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
            )

            if matrix is not None:
                dx = float(matrix[0, 2])
                dy = float(matrix[1, 2])

                rotation = math.degrees(
                    math.atan2(
                        matrix[1, 0],
                        matrix[0, 0],
                    )
                )

                scale = math.hypot(
                    matrix[0, 0],
                    matrix[1, 0],
                )

                inlier_count = (
                    int(np.sum(inliers))
                    if inliers is not None
                    else 0
                )

                ratio = (
                    inlier_count / len(previous)
                )

                return GlobalMotionEstimate(
                    translation_x_px=dx,
                    translation_y_px=dy,
                    magnitude_px=math.hypot(dx, dy),
                    rotation_degrees=rotation,
                    scale=scale,
                    inlier_count=inlier_count,
                    inlier_ratio=ratio,
                    reliable=(
                        inlier_count >= 3
                        and ratio >= 0.5
                    ),
                )

        displacement = current - previous

        median_displacement = np.median(
            displacement,
            axis=0,
        )

        dx = float(median_displacement[0])
        dy = float(median_displacement[1])

        return GlobalMotionEstimate(
            translation_x_px=dx,
            translation_y_px=dy,
            magnitude_px=math.hypot(dx, dy),
            reliable=False,
        )

    @staticmethod
    def from_dense(
        result: DenseFlowResult,
    ) -> GlobalMotionEstimate:
        dx = float(
            np.median(result.flow[:, :, 0])
        )

        dy = float(
            np.median(result.flow[:, :, 1])
        )

        return GlobalMotionEstimate(
            translation_x_px=dx,
            translation_y_px=dy,
            magnitude_px=math.hypot(dx, dy),
            reliable=False,
        )


# ============================================================
# CAMERA MOTION COMPENSATION
# ============================================================

class OpticalFlowMotionCompensator:

    @staticmethod
    def compensate_dense(
        flow: np.ndarray,
        global_motion: GlobalMotionEstimate,
    ) -> np.ndarray:
        if flow.ndim != 3 or flow.shape[2] != 2:
            raise OpticalFlowValidationError(
                "Dense flow must have shape H x W x 2"
            )

        compensated = flow.astype(
            np.float32,
            copy=True,
        )

        compensated[:, :, 0] -= (
            global_motion.translation_x_px
        )

        compensated[:, :, 1] -= (
            global_motion.translation_y_px
        )

        return compensated

    @staticmethod
    def compensate_sparse(
        vectors: Sequence[MotionVector],
        global_motion: GlobalMotionEstimate,
        dt_s: float | None = None,
    ) -> tuple[MotionVector, ...]:
        compensated = []

        for vector in vectors:
            new_x = (
                vector.end_x
                - global_motion.translation_x_px
            )

            new_y = (
                vector.end_y
                - global_motion.translation_y_px
            )

            compensated.append(
                MotionVector.from_points(
                    vector.start_x,
                    vector.start_y,
                    new_x,
                    new_y,
                    dt_s,
                )
            )

        return tuple(compensated)


# ============================================================
# MOTION DIRECTION ANALYZER
# ============================================================

class MotionDirectionAnalyzer:

    @staticmethod
    def classify(
        dx: float,
        dy: float,
        *,
        stationary_threshold: float = 0.5,
    ) -> MotionDirection:
        magnitude = math.hypot(dx, dy)

        if magnitude < stationary_threshold:
            return MotionDirection.STATIONARY

        angle = math.degrees(
            math.atan2(dy, dx)
        )

        if -22.5 <= angle < 22.5:
            return MotionDirection.RIGHT

        if 22.5 <= angle < 67.5:
            return MotionDirection.DOWN_RIGHT

        if 67.5 <= angle < 112.5:
            return MotionDirection.DOWN

        if 112.5 <= angle < 157.5:
            return MotionDirection.DOWN_LEFT

        if angle >= 157.5 or angle < -157.5:
            return MotionDirection.LEFT

        if -157.5 <= angle < -112.5:
            return MotionDirection.UP_LEFT

        if -112.5 <= angle < -67.5:
            return MotionDirection.UP

        return MotionDirection.UP_RIGHT

    @classmethod
    def dominant_sparse_direction(
        cls,
        vectors: Sequence[MotionVector],
        threshold: float,
    ) -> MotionDirection:
        if not vectors:
            return MotionDirection.STATIONARY

        dx = float(
            np.median([vector.dx for vector in vectors])
        )

        dy = float(
            np.median([vector.dy for vector in vectors])
        )

        return cls.classify(
            dx,
            dy,
            stationary_threshold=threshold,
        )

    @classmethod
    def dominant_dense_direction(
        cls,
        flow: np.ndarray,
        threshold: float,
    ) -> MotionDirection:
        dx = float(
            np.median(flow[:, :, 0])
        )

        dy = float(
            np.median(flow[:, :, 1])
        )

        return cls.classify(
            dx,
            dy,
            stationary_threshold=threshold,
        )


# ============================================================
# MOVING OBJECT REGION DETECTOR
# ============================================================

class MotionRegionDetector:

    def __init__(
        self,
        config: OpticalFlowConfig,
    ) -> None:
        require_opencv()

        self.config = config

    def detect(
        self,
        flow: np.ndarray,
    ) -> tuple[MotionRegion, ...]:
        if flow.ndim != 3 or flow.shape[2] != 2:
            raise OpticalFlowValidationError(
                "Expected dense flow H x W x 2"
            )

        magnitude = np.linalg.norm(
            flow,
            axis=2,
        )

        motion_mask = (
            magnitude
            >= self.config.moving_region_threshold_px
        ).astype(np.uint8) * 255

        kernel = np.ones(
            (3, 3),
            dtype=np.uint8,
        )

        motion_mask = cv2.morphologyEx(
            motion_mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        motion_mask = cv2.morphologyEx(
            motion_mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        contours, _ = cv2.findContours(
            motion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        regions = []

        for contour in contours:
            contour_area = cv2.contourArea(
                contour
            )

            if (
                contour_area
                < self.config.minimum_motion_region_area_px
            ):
                continue

            x, y, width, height = cv2.boundingRect(
                contour
            )

            region_magnitude = magnitude[
                y:y + height,
                x:x + width,
            ]

            regions.append(
                MotionRegion(
                    x1=int(x),
                    y1=int(y),
                    x2=int(x + width),
                    y2=int(y + height),
                    area_px=int(contour_area),
                    mean_motion_px=float(
                        np.mean(region_magnitude)
                    ),
                    maximum_motion_px=float(
                        np.max(region_magnitude)
                    ),
                    centroid_x=float(
                        x + width / 2
                    ),
                    centroid_y=float(
                        y + height / 2
                    ),
                )
            )

        regions.sort(
            key=lambda region: region.area_px,
            reverse=True,
        )

        return tuple(regions)


# ============================================================
# FLOW QUALITY EVALUATOR
# ============================================================

class OpticalFlowQualityEvaluator:

    def __init__(
        self,
        config: OpticalFlowConfig,
    ) -> None:
        self.config = config

    def evaluate_sparse(
        self,
        result: SparseFlowResult,
    ) -> OpticalFlowQualityReport:
        magnitudes = np.array(
            [
                vector.magnitude_px
                for vector in result.motion_vectors
            ],
            dtype=np.float64,
        )

        notes = []

        if (
            result.tracked_feature_count
            < self.config.minimum_features
        ):
            notes.append(
                "Insufficient reliable tracked features"
            )

        if result.tracked_feature_count == 0:
            quality = FlowQuality.POOR

        elif (
            result.tracked_feature_count
            >= self.config.minimum_features
            and result.valid_feature_ratio >= 0.85
        ):
            quality = FlowQuality.EXCELLENT

        elif result.valid_feature_ratio >= 0.65:
            quality = FlowQuality.GOOD

        elif result.valid_feature_ratio >= 0.35:
            quality = FlowQuality.FAIR

        else:
            quality = FlowQuality.POOR

        return OpticalFlowQualityReport(
            quality=quality,
            tracked_features=result.tracked_feature_count,
            valid_feature_ratio=result.valid_feature_ratio,
            mean_motion_px=(
                float(np.mean(magnitudes))
                if magnitudes.size
                else 0.0
            ),
            median_motion_px=(
                float(np.median(magnitudes))
                if magnitudes.size
                else 0.0
            ),
            maximum_motion_px=(
                float(np.max(magnitudes))
                if magnitudes.size
                else 0.0
            ),
            finite_flow_ratio=1.0,
            notes=tuple(notes),
        )

    @staticmethod
    def evaluate_dense(
        result: DenseFlowResult,
    ) -> OpticalFlowQualityReport:
        finite_mask = np.isfinite(
            result.flow
        ).all(axis=2)

        finite_ratio = float(
            np.mean(finite_mask)
        )

        if finite_ratio >= 0.999:
            quality = FlowQuality.GOOD

        elif finite_ratio >= 0.95:
            quality = FlowQuality.FAIR

        else:
            quality = FlowQuality.POOR

        return OpticalFlowQualityReport(
            quality=quality,
            tracked_features=0,
            valid_feature_ratio=0.0,
            mean_motion_px=result.mean_magnitude_px,
            median_motion_px=result.median_magnitude_px,
            maximum_motion_px=result.maximum_magnitude_px,
            finite_flow_ratio=finite_ratio,
            notes=(
                "Dense flow quality requires external "
                "ground-truth or consistency validation",
            ),
        )


# ============================================================
# OPTICAL FLOW VISUALIZER
# ============================================================

class OpticalFlowVisualizer:

    @staticmethod
    def draw_sparse(
        image: np.ndarray,
        vectors: Sequence[MotionVector],
        *,
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        require_opencv()

        output = image.copy()

        if output.ndim == 2:
            output = cv2.cvtColor(
                output,
                cv2.COLOR_GRAY2BGR,
            )

        for vector in vectors:
            start = (
                int(round(vector.start_x)),
                int(round(vector.start_y)),
            )

            end = (
                int(round(vector.end_x)),
                int(round(vector.end_y)),
            )

            cv2.arrowedLine(
                output,
                start,
                end,
                color,
                1,
                tipLength=0.25,
            )

            cv2.circle(
                output,
                end,
                2,
                (0, 0, 255),
                -1,
            )

        return output

    @staticmethod
    def dense_hsv(
        flow: np.ndarray,
    ) -> np.ndarray:
        require_opencv()

        if flow.ndim != 3 or flow.shape[2] != 2:
            raise OpticalFlowValidationError(
                "Expected H x W x 2 flow"
            )

        magnitude, angle = cv2.cartToPolar(
            flow[:, :, 0],
            flow[:, :, 1],
            angleInDegrees=True,
        )

        hsv = np.zeros(
            (
                flow.shape[0],
                flow.shape[1],
                3,
            ),
            dtype=np.uint8,
        )

        hsv[:, :, 0] = (
            angle / 2
        ).astype(np.uint8)

        hsv[:, :, 1] = 255

        maximum = float(
            np.max(magnitude)
        )

        if maximum > 0:
            hsv[:, :, 2] = np.clip(
                magnitude / maximum * 255,
                0,
                255,
            ).astype(np.uint8)

        return cv2.cvtColor(
            hsv,
            cv2.COLOR_HSV2BGR,
        )

    @staticmethod
    def draw_motion_regions(
        image: np.ndarray,
        regions: Sequence[MotionRegion],
    ) -> np.ndarray:
        require_opencv()

        output = image.copy()

        if output.ndim == 2:
            output = cv2.cvtColor(
                output,
                cv2.COLOR_GRAY2BGR,
            )

        for region in regions:
            cv2.rectangle(
                output,
                (region.x1, region.y1),
                (region.x2, region.y2),
                (0, 165, 255),
                2,
            )

            label = (
                f"Motion {region.mean_motion_px:.2f}px"
            )

            cv2.putText(
                output,
                label,
                (
                    region.x1,
                    max(15, region.y1 - 8),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                1,
                cv2.LINE_AA,
            )

        return output


# ============================================================
# CAMERA STREAM STATE
# ============================================================

@dataclass
class CameraOpticalFlowState:
    previous_gray: np.ndarray | None = None

    previous_timestamp: datetime | None = None

    previous_points: np.ndarray | None = None

    frame_count: int = 0


# ============================================================
# ROBOTICS OPTICAL FLOW SERVICE
# ============================================================

class RoboticsOpticalFlowService:

    def __init__(
        self,
        config: OpticalFlowConfig | None = None,
    ) -> None:
        self.config = (
            config or OpticalFlowConfig()
        )

        self.validator = OpticalFlowImageValidator(
            self.config
        )

        self.quality_evaluator = (
            OpticalFlowQualityEvaluator(
                self.config
            )
        )

        self._states: dict[
            tuple[str, str, str],
            CameraOpticalFlowState,
        ] = {}

        self._history: dict[
            tuple[str, str, str],
            deque[OpticalFlowFrame],
        ] = {}

        self._audit: deque[
            OpticalFlowAuditEvent
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    @staticmethod
    def _authorize(
        access: OpticalFlowAccessContext,
        tenant_id: str,
        robot_id: str,
        camera_id: str,
        permission: str,
    ) -> None:
        if access.tenant_id != tenant_id:
            raise OpticalFlowAuthorizationError(
                "Cross-tenant access denied"
            )

        if robot_id not in access.allowed_robot_ids:
            raise OpticalFlowAuthorizationError(
                "Robot access denied"
            )

        if camera_id not in access.allowed_camera_ids:
            raise OpticalFlowAuthorizationError(
                "Camera access denied"
            )

        if not getattr(access, permission, False):
            raise OpticalFlowAuthorizationError(
                f"Missing permission: {permission}"
            )

    def _empty_quality(
        self,
    ) -> OpticalFlowQualityReport:
        return OpticalFlowQualityReport(
            quality=FlowQuality.UNKNOWN,
            tracked_features=0,
            valid_feature_ratio=0.0,
            mean_motion_px=0.0,
            median_motion_px=0.0,
            maximum_motion_px=0.0,
            finite_flow_ratio=1.0,
            notes=("Initial frame; no motion baseline",),
        )

    def process(
        self,
        image: np.ndarray,
        *,
        robot_id: str,
        camera_id: str,
        access: OpticalFlowAccessContext,
        timestamp: datetime | None = None,
    ) -> OpticalFlowFrame:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            camera_id,
            "can_process",
        )

        require_opencv()

        timestamp = validate_timestamp(
            timestamp or utc_now()
        )

        gray = self.validator.grayscale(
            image
        )

        started = time.perf_counter()

        key = (
            access.tenant_id,
            robot_id,
            camera_id,
        )

        with self._lock:
            state = self._states.setdefault(
                key,
                CameraOpticalFlowState(),
            )

            vectors: tuple[MotionVector, ...] = ()
            regions: tuple[MotionRegion, ...] = ()

            global_motion = None
            dt_s = None

            direction = MotionDirection.STATIONARY

            status = OpticalFlowStatus.INITIALIZED

            quality = self._empty_quality()

            previous = state.previous_gray

            if previous is not None:
                if previous.shape != gray.shape:
                    state.previous_points = None
                    previous = None

                else:
                    dt_s = (
                        timestamp - state.previous_timestamp
                    ).total_seconds()

                    if dt_s <= 0:
                        raise OpticalFlowValidationError(
                            "Frame timestamps must increase"
                        )

                    if dt_s > self.config.maximum_frame_gap_s:
                        state.previous_points = None
                        previous = None

            if previous is not None:
                if (
                    self.config.method
                    == OpticalFlowMethod.SPARSE_LUCAS_KANADE
                ):
                    engine = SparseLucasKanadeFlow(
                        self.config
                    )

                    sparse = engine.calculate(
                        previous,
                        gray,
                        previous_points=state.previous_points,
                        dt_s=dt_s,
                    )

                    vectors = sparse.motion_vectors

                    global_motion = (
                        GlobalMotionEstimator.from_sparse(
                            sparse
                        )
                    )

                    quality = (
                        self.quality_evaluator.evaluate_sparse(
                            sparse
                        )
                    )

                    if (
                        sparse.tracked_feature_count
                        < self.config.minimum_features
                    ):
                        status = (
                            OpticalFlowStatus.INSUFFICIENT_FEATURES
                        )

                    else:
                        status = OpticalFlowStatus.SUCCESS

                    if (
                        self.config.enable_camera_motion_compensation
                        and global_motion.reliable
                    ):
                        analysis_vectors = (
                            OpticalFlowMotionCompensator
                            .compensate_sparse(
                                vectors,
                                global_motion,
                                dt_s,
                            )
                        )

                    else:
                        analysis_vectors = vectors

                    direction = (
                        MotionDirectionAnalyzer
                        .dominant_sparse_direction(
                            analysis_vectors,
                            self.config.stationary_threshold_px,
                        )
                    )

                    state.previous_points = (
                        sparse.current_points
                        if (
                            sparse.tracked_feature_count
                            >= self.config.minimum_features
                        )
                        else None
                    )

                elif (
                    self.config.method
                    == OpticalFlowMethod.DENSE_FARNEBACK
                ):
                    engine = DenseFarnebackFlow(
                        self.config
                    )

                    dense = engine.calculate(
                        previous,
                        gray,
                    )

                    global_motion = (
                        GlobalMotionEstimator.from_dense(
                            dense
                        )
                    )

                    quality = (
                        self.quality_evaluator.evaluate_dense(
                            dense
                        )
                    )

                    analysis_flow = dense.flow

                    if (
                        self.config.enable_camera_motion_compensation
                        and global_motion.reliable
                    ):
                        analysis_flow = (
                            OpticalFlowMotionCompensator
                            .compensate_dense(
                                dense.flow,
                                global_motion,
                            )
                        )

                    region_detector = MotionRegionDetector(
                        self.config
                    )

                    regions = region_detector.detect(
                        analysis_flow
                    )

                    direction = (
                        MotionDirectionAnalyzer
                        .dominant_dense_direction(
                            analysis_flow,
                            self.config.stationary_threshold_px,
                        )
                    )

                    status = OpticalFlowStatus.SUCCESS

                    state.previous_points = None

                else:
                    raise OpticalFlowValidationError(
                        "Unsupported optical flow method"
                    )

            state.previous_gray = gray.copy()
            state.previous_timestamp = timestamp
            state.frame_count += 1

            elapsed_ms = (
                time.perf_counter() - started
            ) * 1000.0

            frame = OpticalFlowFrame(
                frame_id=generate_id(),
                tenant_id=access.tenant_id,
                robot_id=robot_id,
                camera_id=camera_id,
                method=self.config.method,
                status=status,
                image_width=gray.shape[1],
                image_height=gray.shape[0],
                timestamp=timestamp,
                frame_interval_s=dt_s,
                motion_vectors=vectors,
                global_motion=global_motion,
                motion_regions=regions,
                quality=quality,
                dominant_direction=direction,
                processing_time_ms=elapsed_ms,
                metadata={
                    "camera_motion_compensation_requested": (
                        self.config.enable_camera_motion_compensation
                    ),
                    "advisory_only": True,
                    "motion_authorization": False,
                    "flow_units": "pixels_per_frame",
                    "velocity_units": "pixels_per_second",
                    "stream_frame_count": state.frame_count,
                },
            )

            history = self._history.setdefault(
                key,
                deque(
                    maxlen=self.config.maximum_history_frames
                ),
            )

            history.append(frame)

            self._audit.append(
                OpticalFlowAuditEvent(
                    event_id=generate_id(),
                    tenant_id=access.tenant_id,
                    user_id=access.user_id,
                    robot_id=robot_id,
                    camera_id=camera_id,
                    frame_id=frame.frame_id,
                    action="optical_flow.process",
                )
            )

            return frame

    def latest(
        self,
        *,
        robot_id: str,
        camera_id: str,
        access: OpticalFlowAccessContext,
    ) -> OpticalFlowFrame | None:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            camera_id,
            "can_view",
        )

        key = (
            access.tenant_id,
            robot_id,
            camera_id,
        )

        with self._lock:
            history = self._history.get(key)

            return history[-1] if history else None

    def history(
        self,
        *,
        robot_id: str,
        camera_id: str,
        access: OpticalFlowAccessContext,
    ) -> tuple[OpticalFlowFrame, ...]:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            camera_id,
            "can_view",
        )

        key = (
            access.tenant_id,
            robot_id,
            camera_id,
        )

        with self._lock:
            return tuple(
                self._history.get(key, ())
            )

    def reset_camera(
        self,
        *,
        robot_id: str,
        camera_id: str,
        access: OpticalFlowAccessContext,
    ) -> None:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            camera_id,
            "can_process",
        )

        key = (
            access.tenant_id,
            robot_id,
            camera_id,
        )

        with self._lock:
            self._states.pop(key, None)
            self._history.pop(key, None)

    def audit_events(
        self,
        *,
        access: OpticalFlowAccessContext,
    ) -> tuple[OpticalFlowAuditEvent, ...]:
        if not access.can_view:
            raise OpticalFlowAuthorizationError(
                "Audit viewing permission denied"
            )

        with self._lock:
            return tuple(
                event
                for event in self._audit
                if (
                    event.tenant_id == access.tenant_id
                    and event.robot_id in access.allowed_robot_ids
                    and event.camera_id in access.allowed_camera_ids
                )
            )

    @staticmethod
    def export_json(
        frame: OpticalFlowFrame,
    ) -> str:
        return json.dumps(
            frame.to_dict(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


# ============================================================
# SYNTHETIC MOTION GENERATOR
# ============================================================

class SyntheticOpticalFlowGenerator:

    @staticmethod
    def generate_frame(
        frame_number: int,
        *,
        width: int = 640,
        height: int = 480,
        displacement_x: int = 5,
        displacement_y: int = 2,
    ) -> np.ndarray:
        require_opencv()

        image = np.zeros(
            (height, width, 3),
            dtype=np.uint8,
        )

        x = (
            80
            + frame_number * displacement_x
        )

        y = (
            100
            + frame_number * displacement_y
        )

        cv2.rectangle(
            image,
            (x, y),
            (x + 100, y + 80),
            (255, 255, 255),
            -1,
        )

        cv2.circle(
            image,
            (x + 50, y + 40),
            15,
            (0, 0, 0),
            -1,
        )

        cv2.line(
            image,
            (x + 10, y + 10),
            (x + 90, y + 70),
            (0, 0, 0),
            3,
        )

        return image


# ============================================================
# UNIT TESTS
# ============================================================

@unittest.skipIf(
    cv2 is None,
    "OpenCV is unavailable",
)
class TestOpticalFlow(unittest.TestCase):

    def setUp(self) -> None:
        self.config = OpticalFlowConfig(
            minimum_features=3,
        )

        self.access = OpticalFlowAccessContext(
            tenant_id="tenant_alpha",
            user_id="robotics_engineer",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
            }),
            allowed_camera_ids=frozenset({
                "CAM-001",
            }),
            can_process=True,
            can_view=True,
        )

        self.first = (
            SyntheticOpticalFlowGenerator.generate_frame(
                0
            )
        )

        self.second = (
            SyntheticOpticalFlowGenerator.generate_frame(
                1
            )
        )

    def test_motion_vector(self) -> None:
        vector = MotionVector.from_points(
            10,
            10,
            20,
            10,
            0.1,
        )

        self.assertAlmostEqual(
            vector.magnitude_px,
            10.0,
        )

        self.assertAlmostEqual(
            vector.velocity_x_px_s,
            100.0,
        )

    def test_direction_right(self) -> None:
        direction = MotionDirectionAnalyzer.classify(
            10,
            0,
        )

        self.assertEqual(
            direction,
            MotionDirection.RIGHT,
        )

    def test_direction_stationary(self) -> None:
        direction = MotionDirectionAnalyzer.classify(
            0.1,
            0.1,
        )

        self.assertEqual(
            direction,
            MotionDirection.STATIONARY,
        )

    def test_feature_detection(self) -> None:
        gray = cv2.cvtColor(
            self.first,
            cv2.COLOR_BGR2GRAY,
        )

        detector = ShiTomasiFeatureDetector(
            self.config
        )

        features = detector.detect(
            gray
        )

        self.assertGreater(
            len(features),
            0,
        )

    def test_sparse_optical_flow(self) -> None:
        previous = cv2.cvtColor(
            self.first,
            cv2.COLOR_BGR2GRAY,
        )

        current = cv2.cvtColor(
            self.second,
            cv2.COLOR_BGR2GRAY,
        )

        engine = SparseLucasKanadeFlow(
            self.config
        )

        result = engine.calculate(
            previous,
            current,
            dt_s=0.1,
        )

        self.assertGreater(
            result.tracked_feature_count,
            0,
        )

    def test_dense_optical_flow(self) -> None:
        previous = cv2.cvtColor(
            self.first,
            cv2.COLOR_BGR2GRAY,
        )

        current = cv2.cvtColor(
            self.second,
            cv2.COLOR_BGR2GRAY,
        )

        engine = DenseFarnebackFlow(
            self.config
        )

        result = engine.calculate(
            previous,
            current,
        )

        self.assertEqual(
            result.flow.shape,
            (480, 640, 2),
        )

    def test_global_motion(self) -> None:
        previous = cv2.cvtColor(
            self.first,
            cv2.COLOR_BGR2GRAY,
        )

        current = cv2.cvtColor(
            self.second,
            cv2.COLOR_BGR2GRAY,
        )

        result = SparseLucasKanadeFlow(
            self.config
        ).calculate(
            previous,
            current,
        )

        motion = GlobalMotionEstimator.from_sparse(
            result
        )

        self.assertTrue(
            math.isfinite(
                motion.magnitude_px
            )
        )

    def test_motion_compensation(self) -> None:
        flow = np.ones(
            (10, 10, 2),
            dtype=np.float32,
        )

        motion = GlobalMotionEstimate(
            translation_x_px=1.0,
            translation_y_px=1.0,
            magnitude_px=math.sqrt(2),
        )

        compensated = (
            OpticalFlowMotionCompensator
            .compensate_dense(
                flow,
                motion,
            )
        )

        self.assertTrue(
            np.allclose(
                compensated,
                0.0,
            )
        )

    def test_service_initialization(self) -> None:
        service = RoboticsOpticalFlowService(
            self.config
        )

        frame = service.process(
            self.first,
            robot_id="ROBOT-001",
            camera_id="CAM-001",
            access=self.access,
        )

        self.assertEqual(
            frame.status,
            OpticalFlowStatus.INITIALIZED,
        )

    def test_service_processing(self) -> None:
        service = RoboticsOpticalFlowService(
            self.config
        )

        timestamp = utc_now()

        service.process(
            self.first,
            robot_id="ROBOT-001",
            camera_id="CAM-001",
            access=self.access,
            timestamp=timestamp,
        )

        frame = service.process(
            self.second,
            robot_id="ROBOT-001",
            camera_id="CAM-001",
            access=self.access,
            timestamp=(
                timestamp
                + timedelta(seconds=0.1)
            ),
        )

        self.assertIn(
            frame.status,
            (
                OpticalFlowStatus.SUCCESS,
                OpticalFlowStatus.INSUFFICIENT_FEATURES,
            ),
        )

    def test_dense_service(self) -> None:
        config = OpticalFlowConfig(
            method=OpticalFlowMethod.DENSE_FARNEBACK,
        )

        service = RoboticsOpticalFlowService(
            config
        )

        timestamp = utc_now()

        service.process(
            self.first,
            robot_id="ROBOT-001",
            camera_id="CAM-001",
            access=self.access,
            timestamp=timestamp,
        )

        frame = service.process(
            self.second,
            robot_id="ROBOT-001",
            camera_id="CAM-001",
            access=self.access,
            timestamp=(
                timestamp
                + timedelta(seconds=0.1)
            ),
        )

        self.assertEqual(
            frame.status,
            OpticalFlowStatus.SUCCESS,
        )

    def test_history(self) -> None:
        service = RoboticsOpticalFlowService(
            self.config
        )

        service.process(
            self.first,
            robot_id="ROBOT-001",
            camera_id="CAM-001",
            access=self.access,
        )

        history = service.history(
            robot_id="ROBOT-001",
            camera_id="CAM-001",
            access=self.access,
        )

        self.assertEqual(
            len(history),
            1,
        )

    def test_tenant_isolation(self) -> None:
        with self.assertRaises(
            OpticalFlowAuthorizationError
        ):
            RoboticsOpticalFlowService._authorize(
                self.access,
                "tenant_beta",
                "ROBOT-001",
                "CAM-001",
                "can_process",
            )

    def test_json_export(self) -> None:
        service = RoboticsOpticalFlowService(
            self.config
        )

        frame = service.process(
            self.first,
            robot_id="ROBOT-001",
            camera_id="CAM-001",
            access=self.access,
        )

        payload = service.export_json(
            frame
        )

        self.assertIn(
            "ROBOT-001",
            payload,
        )

    def test_visualization(self) -> None:
        vector = MotionVector.from_points(
            10,
            10,
            20,
            20,
        )

        output = OpticalFlowVisualizer.draw_sparse(
            self.first,
            (vector,),
        )

        self.assertEqual(
            output.shape,
            self.first.shape,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    require_opencv()

    print(
        "\nPHYSICAL AI & AUTONOMOUS ROBOTICS PLATFORM"
    )

    print(
        "Real-Time Optical Flow & Motion Intelligence\n"
    )

    access = OpticalFlowAccessContext(
        tenant_id="tenant_alpha",
        user_id="robotics_engineer",
        allowed_robot_ids=frozenset({
            "ROBOT-001",
        }),
        allowed_camera_ids=frozenset({
            "CAM-FRONT-001",
        }),
        can_process=True,
        can_view=True,
    )

    service = RoboticsOpticalFlowService(
        OpticalFlowConfig(
            method=OpticalFlowMethod.SPARSE_LUCAS_KANADE,
            minimum_features=3,
        )
    )

    timestamp = utc_now()

    for index in range(10):
        image = (
            SyntheticOpticalFlowGenerator.generate_frame(
                index
            )
        )

        frame = service.process(
            image,
            robot_id="ROBOT-001",
            camera_id="CAM-FRONT-001",
            access=access,
            timestamp=(
                timestamp
                + timedelta(
                    seconds=index * 0.1
                )
            ),
        )

        print(
            f"Frame {index + 1}: "
            f"status={frame.status.value}, "
            f"vectors={len(frame.motion_vectors)}, "
            f"direction={frame.dominant_direction.value}, "
            f"quality={frame.quality.quality.value}"
        )

    print("\nLATEST OPTICAL FLOW RESULT\n")

    print(
        service.export_json(frame)
    )

    print("\nRUNNING UNIT TESTS\n")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestOpticalFlow
    )

    unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)


if __name__ == "__main__":
    main()
"""Perception Pipeline module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/perception/perception_pipeline.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================

Module: perception_pipeline.py

Unified Real-Time Robotics Perception Pipeline

Capabilities
------------
- Camera image ingestion and validation
- Image preprocessing
- Object detection
- Instance segmentation
- Object tracking
- Depth estimation
- LiDAR point-cloud processing
- Optical flow estimation
- Multi-sensor fusion
- Obstacle perception
- Human proximity observations
- Perception quality evaluation
- Pipeline health monitoring
- Configurable processing stages
- Dependency-aware execution
- Stage timeouts and failure isolation
- Per-stage latency measurements
- Sensor timestamp synchronization checks
- Tenant and robot access control
- Bounded processing history
- Audit logging
- JSON reporting
- Synthetic demonstration
- Unit tests

Architecture
------------
Camera / LiDAR / Depth
          |
          v
   Input Validation
          |
          v
    Preprocessing
          |
          +-------------------+
          |                   |
          v                   v
   Object Detection     Depth Estimation
          |                   |
          v                   |
   Instance Segmentation      |
          |                   |
          v                   |
   Object Tracking            |
          |                   |
          +---------+---------+
                    |
                    v
             Sensor Fusion
                    |
                    v
          Obstacle Observations
                    |
                    v
          Quality / Health Monitor
                    |
                    v
           Perception Results

Integration
-----------
Existing perception implementations can be registered through
PerceptionStageAdapter. This module deliberately avoids assuming
undocumented APIs in neighboring modules.

Dependencies
------------
Python >= 3.11
numpy

Optional:
opencv-python

Safety
------
Perception results are advisory observations.

This pipeline does not issue robot movement commands, authorize
actuator operations, override protective interlocks, or implement
a certified functional-safety system.

Stage timeouts are detected after synchronous execution. For hard
execution deadlines, isolate stages in supervised worker processes.
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
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


# ============================================================
# EXCEPTIONS
# ============================================================

class PerceptionPipelineError(Exception):
    """Base perception pipeline exception."""


class PerceptionValidationError(PerceptionPipelineError):
    """Invalid perception input or configuration."""


class PerceptionAuthorizationError(PerceptionPipelineError):
    """Perception pipeline access denied."""


class PerceptionDependencyError(PerceptionPipelineError):
    """A required dependency is unavailable."""


class PerceptionStageError(PerceptionPipelineError):
    """A perception processing stage failed."""


class PerceptionCapacityError(PerceptionPipelineError):
    """Configured processing capacity exceeded."""


class PerceptionSynchronizationError(PerceptionPipelineError):
    """Sensor timestamps are inconsistent."""


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


def validate_identifier(
    value: str,
    name: str,
) -> str:
    if not isinstance(value, str):
        raise PerceptionValidationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > 128:
        raise PerceptionValidationError(
            f"{name} must contain 1-128 characters"
        )

    return value


def validate_timestamp(
    timestamp: datetime,
) -> datetime:
    if not isinstance(timestamp, datetime):
        raise PerceptionValidationError(
            "Timestamp must be datetime"
        )

    if (
        timestamp.tzinfo is None
        or timestamp.utcoffset() is None
    ):
        raise PerceptionValidationError(
            "Timestamp must be timezone-aware"
        )

    return timestamp.astimezone(timezone.utc)


def finite_float(
    value: Any,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise PerceptionValidationError(
            f"{name} must be numeric"
        )

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PerceptionValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(number):
        raise PerceptionValidationError(
            f"{name} must be finite"
        )

    return number


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
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    if isinstance(value, np.generic):
        return value.item()

    if hasattr(value, "__dataclass_fields__"):
        return {
            name: json_safe(getattr(value, name))
            for name in value.__dataclass_fields__
        }

    if isinstance(value, Mapping):
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

class PerceptionStage(str, Enum):
    VALIDATION = "validation"
    PREPROCESSING = "preprocessing"
    OBJECT_DETECTION = "object_detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    OBJECT_TRACKING = "object_tracking"
    DEPTH_ESTIMATION = "depth_estimation"
    LIDAR_PROCESSING = "lidar_processing"
    OPTICAL_FLOW = "optical_flow"
    SENSOR_FUSION = "sensor_fusion"
    OBSTACLE_ANALYSIS = "obstacle_analysis"
    QUALITY_EVALUATION = "quality_evaluation"


class StageStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"


class PipelineStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"


class PerceptionQuality(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    DEGRADED = "degraded"
    POOR = "poor"
    UNKNOWN = "unknown"


class SensorType(str, Enum):
    CAMERA = "camera"
    LIDAR = "lidar"
    DEPTH_CAMERA = "depth_camera"
    IMU = "imu"


class ObstacleRiskLevel(str, Enum):
    UNKNOWN = "unknown"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PerceptionPipelineConfig:
    enable_preprocessing: bool = True
    enable_object_detection: bool = True
    enable_instance_segmentation: bool = True
    enable_object_tracking: bool = True
    enable_depth_estimation: bool = True
    enable_lidar_processing: bool = True
    enable_optical_flow: bool = True
    enable_sensor_fusion: bool = True
    enable_obstacle_analysis: bool = True
    enable_quality_evaluation: bool = True

    maximum_image_width: int = 4096
    maximum_image_height: int = 4096
    maximum_image_pixels: int = 16_777_216

    maximum_lidar_points: int = 2_000_000

    maximum_sensor_skew_ms: float = 100.0

    warning_pipeline_latency_ms: float = 150.0
    critical_pipeline_latency_ms: float = 500.0

    stage_timeout_ms: float = 1000.0

    minimum_detection_confidence: float = 0.40

    minimum_quality_score: float = 60.0

    obstacle_warning_distance_m: float = 3.0
    obstacle_high_risk_distance_m: float = 1.0

    maximum_history: int = 100
    maximum_audit_events: int = 1000

    continue_on_stage_failure: bool = True
    require_registered_stages: bool = False

    def __post_init__(self) -> None:
        for name in (
            "maximum_image_width",
            "maximum_image_height",
            "maximum_image_pixels",
            "maximum_lidar_points",
            "maximum_history",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise PerceptionValidationError(
                    f"{name} must be a positive integer"
                )

        for name in (
            "maximum_sensor_skew_ms",
            "warning_pipeline_latency_ms",
            "critical_pipeline_latency_ms",
            "stage_timeout_ms",
            "obstacle_warning_distance_m",
            "obstacle_high_risk_distance_m",
        ):
            if finite_float(getattr(self, name), name) <= 0:
                raise PerceptionValidationError(
                    f"{name} must be positive"
                )

        for name in (
            "minimum_detection_confidence",
        ):
            value = finite_float(getattr(self, name), name)

            if not 0 <= value <= 1:
                raise PerceptionValidationError(
                    f"{name} must be between 0 and 1"
                )

        if not 0 <= self.minimum_quality_score <= 100:
            raise PerceptionValidationError(
                "minimum_quality_score must be between 0 and 100"
            )

        if (
            self.warning_pipeline_latency_ms
            >= self.critical_pipeline_latency_ms
        ):
            raise PerceptionValidationError(
                "Critical latency must exceed warning latency"
            )

        if (
            self.obstacle_high_risk_distance_m
            >= self.obstacle_warning_distance_m
        ):
            raise PerceptionValidationError(
                "High-risk distance must be below warning distance"
            )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class PerceptionAccessContext:
    tenant_id: str
    user_id: str

    allowed_robot_ids: frozenset[str] = frozenset()

    can_process: bool = False
    can_view: bool = False
    can_manage: bool = False

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id, "tenant_id")
        validate_identifier(self.user_id, "user_id")


# ============================================================
# SENSOR INPUT MODELS
# ============================================================

@dataclass(frozen=True)
class CameraInput:
    camera_id: str
    image: np.ndarray
    timestamp: datetime

    encoding: str = "bgr8"

    def __post_init__(self) -> None:
        validate_identifier(self.camera_id, "camera_id")
        validate_timestamp(self.timestamp)

        if not isinstance(self.image, np.ndarray):
            raise PerceptionValidationError(
                "Camera image must be a NumPy array"
            )

        if self.image.dtype != np.uint8:
            raise PerceptionValidationError(
                "Camera image must use uint8 pixels"
            )

        if (
            self.image.ndim != 2
            and not (
                self.image.ndim == 3
                and self.image.shape[2] == 3
            )
        ):
            raise PerceptionValidationError(
                "Expected grayscale or 3-channel image"
            )

        if self.encoding not in ("bgr8", "rgb8", "mono8"):
            raise PerceptionValidationError(
                "Unsupported camera encoding"
            )

        if self.encoding == "mono8" and self.image.ndim != 2:
            raise PerceptionValidationError(
                "mono8 requires a grayscale image"
            )

        if self.encoding in ("bgr8", "rgb8"):
            if self.image.ndim != 3:
                raise PerceptionValidationError(
                    "Color encoding requires a 3-channel image"
                )


@dataclass(frozen=True)
class LidarInput:
    lidar_id: str
    points: np.ndarray
    timestamp: datetime

    frame_id: str = "lidar"

    def __post_init__(self) -> None:
        validate_identifier(self.lidar_id, "lidar_id")
        validate_identifier(self.frame_id, "frame_id")
        validate_timestamp(self.timestamp)

        if not isinstance(self.points, np.ndarray):
            raise PerceptionValidationError(
                "LiDAR points must be a NumPy array"
            )

        if (
            self.points.ndim != 2
            or self.points.shape[1] not in (3, 4)
        ):
            raise PerceptionValidationError(
                "LiDAR points must have shape N x 3 or N x 4"
            )

        if not np.issubdtype(
            self.points.dtype,
            np.number,
        ):
            raise PerceptionValidationError(
                "LiDAR points must be numeric"
            )

        if not np.all(np.isfinite(self.points)):
            raise PerceptionValidationError(
                "LiDAR points contain nonfinite values"
            )


@dataclass(frozen=True)
class DepthInput:
    sensor_id: str
    depth_map: np.ndarray
    timestamp: datetime

    depth_scale_m: float = 1.0

    def __post_init__(self) -> None:
        validate_identifier(self.sensor_id, "sensor_id")
        validate_timestamp(self.timestamp)

        if not isinstance(self.depth_map, np.ndarray):
            raise PerceptionValidationError(
                "Depth map must be a NumPy array"
            )

        if self.depth_map.ndim != 2:
            raise PerceptionValidationError(
                "Depth map must be two-dimensional"
            )

        if not np.issubdtype(
            self.depth_map.dtype,
            np.number,
        ):
            raise PerceptionValidationError(
                "Depth map must be numeric"
            )

        if finite_float(
            self.depth_scale_m,
            "depth_scale_m",
        ) <= 0:
            raise PerceptionValidationError(
                "Depth scale must be positive"
            )


@dataclass(frozen=True)
class PerceptionInput:
    request_id: str

    tenant_id: str
    robot_id: str

    timestamp: datetime

    camera: CameraInput | None = None
    lidar: LidarInput | None = None
    depth: DepthInput | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_identifier(self.request_id, "request_id")
        validate_identifier(self.tenant_id, "tenant_id")
        validate_identifier(self.robot_id, "robot_id")
        validate_timestamp(self.timestamp)

        if (
            self.camera is None
            and self.lidar is None
            and self.depth is None
        ):
            raise PerceptionValidationError(
                "At least one sensor input is required"
            )


# ============================================================
# PERCEPTION OUTPUT MODELS
# ============================================================

@dataclass(frozen=True)
class Detection:
    detection_id: str
    class_name: str
    confidence: float

    x1: float
    y1: float
    x2: float
    y2: float

    track_id: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.detection_id, "detection_id")
        validate_identifier(self.class_name, "class_name")

        if not 0 <= finite_float(
            self.confidence,
            "confidence",
        ) <= 1:
            raise PerceptionValidationError(
                "Detection confidence must be between 0 and 1"
            )

        for name in ("x1", "y1", "x2", "y2"):
            finite_float(getattr(self, name), name)

        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise PerceptionValidationError(
                "Invalid detection bounding box"
            )


@dataclass(frozen=True)
class TrackedObject:
    track_id: str
    class_name: str

    confidence: float

    x1: float
    y1: float
    x2: float
    y2: float

    velocity_x_px_s: float = 0.0
    velocity_y_px_s: float = 0.0

    age_frames: int = 1


@dataclass(frozen=True)
class ObstacleObservation:
    obstacle_id: str

    distance_m: float
    risk_level: ObstacleRiskLevel

    source: str

    class_name: str = "unknown"

    confidence: float = 1.0


@dataclass(frozen=True)
class StageExecutionResult:
    stage: PerceptionStage
    status: StageStatus

    latency_ms: float

    output_available: bool

    error: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class PerceptionQualityReport:
    quality: PerceptionQuality

    quality_score: float

    successful_stages: int
    failed_stages: int
    unavailable_stages: int

    average_stage_latency_ms: float
    total_latency_ms: float

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class PerceptionPipelineResult:
    result_id: str
    request_id: str

    tenant_id: str
    robot_id: str

    timestamp: datetime

    status: PipelineStatus

    detections: tuple[Detection, ...]
    tracked_objects: tuple[TrackedObject, ...]
    obstacles: tuple[ObstacleObservation, ...]

    stage_results: tuple[StageExecutionResult, ...]

    quality: PerceptionQualityReport

    total_processing_time_ms: float

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


@dataclass(frozen=True)
class PerceptionAuditEvent:
    event_id: str

    tenant_id: str
    user_id: str
    robot_id: str

    request_id: str
    action: str

    timestamp: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# INPUT VALIDATOR
# ============================================================

class PerceptionInputValidator:

    def __init__(
        self,
        config: PerceptionPipelineConfig,
    ) -> None:
        self.config = config

    def validate(
        self,
        request: PerceptionInput,
    ) -> None:
        if request.camera is not None:
            image = request.camera.image

            height, width = image.shape[:2]

            if (
                height <= 0
                or width <= 0
                or width > self.config.maximum_image_width
                or height > self.config.maximum_image_height
                or width * height > self.config.maximum_image_pixels
            ):
                raise PerceptionCapacityError(
                    "Camera image exceeds configured capacity"
                )

        if request.lidar is not None:
            if (
                len(request.lidar.points)
                > self.config.maximum_lidar_points
            ):
                raise PerceptionCapacityError(
                    "LiDAR point count exceeds configured capacity"
                )

        if request.depth is not None:
            height, width = request.depth.depth_map.shape

            if (
                height <= 0
                or width <= 0
                or height * width > self.config.maximum_image_pixels
            ):
                raise PerceptionCapacityError(
                    "Depth map exceeds configured capacity"
                )

    def validate_synchronization(
        self,
        request: PerceptionInput,
    ) -> float:
        timestamps = []

        for sensor in (
            request.camera,
            request.lidar,
            request.depth,
        ):
            if sensor is not None:
                timestamps.append(
                    validate_timestamp(sensor.timestamp)
                )

        if len(timestamps) <= 1:
            return 0.0

        skew_ms = (
            max(timestamps) - min(timestamps)
        ).total_seconds() * 1000.0

        if skew_ms > self.config.maximum_sensor_skew_ms:
            raise PerceptionSynchronizationError(
                "Sensor timestamp skew exceeds configured limit: "
                f"{skew_ms:.2f}ms"
            )

        return skew_ms


# ============================================================
# STAGE INTERFACE
# ============================================================

class PerceptionStageAdapter(Protocol):
    """
    Adapter contract for an existing perception implementation.

    Implementations receive a stage context and return a value.
    """

    def process(
        self,
        context: PerceptionStageContext,
    ) -> Any:
        ...


@dataclass
class PerceptionStageContext:
    request: PerceptionInput

    config: PerceptionPipelineConfig

    outputs: dict[PerceptionStage, Any] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def get(
        self,
        stage: PerceptionStage,
        default: Any = None,
    ) -> Any:
        return self.outputs.get(stage, default)


class FunctionStageAdapter:
    """Wrap a callable as a perception stage."""

    def __init__(
        self,
        function: Callable[
            [PerceptionStageContext],
            Any,
        ],
    ) -> None:
        if not callable(function):
            raise PerceptionValidationError(
                "Stage function must be callable"
            )

        self.function = function

    def process(
        self,
        context: PerceptionStageContext,
    ) -> Any:
        return self.function(context)


# ============================================================
# BUILT-IN PREPROCESSING
# ============================================================

class BasicImagePreprocessor:

    def process(
        self,
        context: PerceptionStageContext,
    ) -> np.ndarray | None:
        camera = context.request.camera

        if camera is None:
            return None

        image = camera.image

        if image.ndim == 2:
            return np.ascontiguousarray(
                image.copy()
            )

        if cv2 is None:
            if camera.encoding == "rgb8":
                return np.ascontiguousarray(
                    image[:, :, ::-1]
                )

            return np.ascontiguousarray(
                image.copy()
            )

        if camera.encoding == "rgb8":
            return cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR,
            )

        return np.ascontiguousarray(
            image.copy()
        )


# ============================================================
# BUILT-IN DEPTH PROCESSING
# ============================================================

class BasicDepthProcessor:

    def process(
        self,
        context: PerceptionStageContext,
    ) -> dict[str, Any] | None:
        depth = context.request.depth

        if depth is None:
            return None

        depth_map = depth.depth_map.astype(
            np.float32,
            copy=False,
        ) * depth.depth_scale_m

        valid = (
            np.isfinite(depth_map)
            & (depth_map > 0)
        )

        valid_depth = depth_map[valid]

        return {
            "depth_map_m": depth_map,
            "valid_ratio": float(
                np.mean(valid)
            ),
            "minimum_depth_m": (
                float(np.min(valid_depth))
                if valid_depth.size
                else None
            ),
            "median_depth_m": (
                float(np.median(valid_depth))
                if valid_depth.size
                else None
            ),
        }


# ============================================================
# BUILT-IN LIDAR PROCESSING
# ============================================================

class BasicLidarProcessor:

    def process(
        self,
        context: PerceptionStageContext,
    ) -> dict[str, Any] | None:
        lidar = context.request.lidar

        if lidar is None:
            return None

        points = lidar.points

        if len(points) == 0:
            return {
                "point_count": 0,
                "minimum_range_m": None,
                "median_range_m": None,
            }

        ranges = np.linalg.norm(
            points[:, :3].astype(np.float64),
            axis=1,
        )

        valid = ranges > 0

        valid_ranges = ranges[valid]

        return {
            "point_count": int(len(points)),
            "minimum_range_m": (
                float(np.min(valid_ranges))
                if valid_ranges.size
                else None
            ),
            "median_range_m": (
                float(np.median(valid_ranges))
                if valid_ranges.size
                else None
            ),
            "frame_id": lidar.frame_id,
        }


# ============================================================
# OBSTACLE OBSERVATION ANALYZER
# ============================================================

class BasicObstacleAnalyzer:

    def __init__(
        self,
        config: PerceptionPipelineConfig,
    ) -> None:
        self.config = config

    def classify_risk(
        self,
        distance_m: float,
    ) -> ObstacleRiskLevel:
        if distance_m <= self.config.obstacle_high_risk_distance_m:
            return ObstacleRiskLevel.HIGH

        if distance_m <= self.config.obstacle_warning_distance_m:
            return ObstacleRiskLevel.MODERATE

        return ObstacleRiskLevel.LOW

    def process(
        self,
        context: PerceptionStageContext,
    ) -> tuple[ObstacleObservation, ...]:
        observations = []

        depth_result = context.get(
            PerceptionStage.DEPTH_ESTIMATION
        )

        if isinstance(depth_result, Mapping):
            distance = depth_result.get(
                "minimum_depth_m"
            )

            if (
                distance is not None
                and math.isfinite(float(distance))
                and float(distance) > 0
            ):
                observations.append(
                    ObstacleObservation(
                        obstacle_id=generate_id(),
                        distance_m=float(distance),
                        risk_level=self.classify_risk(
                            float(distance)
                        ),
                        source="depth",
                    )
                )

        lidar_result = context.get(
            PerceptionStage.LIDAR_PROCESSING
        )

        if isinstance(lidar_result, Mapping):
            distance = lidar_result.get(
                "minimum_range_m"
            )

            if (
                distance is not None
                and math.isfinite(float(distance))
                and float(distance) > 0
            ):
                observations.append(
                    ObstacleObservation(
                        obstacle_id=generate_id(),
                        distance_m=float(distance),
                        risk_level=self.classify_risk(
                            float(distance)
                        ),
                        source="lidar",
                    )
                )

        return tuple(observations)


# ============================================================
# QUALITY EVALUATOR
# ============================================================

class PerceptionPipelineQualityEvaluator:

    def __init__(
        self,
        config: PerceptionPipelineConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        stages: Sequence[StageExecutionResult],
        total_latency_ms: float,
    ) -> PerceptionQualityReport:
        successful = sum(
            stage.status == StageStatus.SUCCESS
            for stage in stages
        )

        failed = sum(
            stage.status in (
                StageStatus.FAILED,
                StageStatus.TIMEOUT,
            )
            for stage in stages
        )

        unavailable = sum(
            stage.status == StageStatus.UNAVAILABLE
            for stage in stages
        )

        attempted = sum(
            stage.status != StageStatus.SKIPPED
            for stage in stages
        )

        score = (
            100.0 * successful / attempted
            if attempted
            else 0.0
        )

        warnings = []

        if failed:
            warnings.append(
                f"{failed} perception stage(s) failed"
            )

        if unavailable:
            warnings.append(
                f"{unavailable} perception stage(s) unavailable"
            )

        if (
            total_latency_ms
            >= self.config.critical_pipeline_latency_ms
        ):
            score -= 30.0
            warnings.append(
                "Critical end-to-end perception latency"
            )

        elif (
            total_latency_ms
            >= self.config.warning_pipeline_latency_ms
        ):
            score -= 10.0
            warnings.append(
                "Elevated end-to-end perception latency"
            )

        score = clamp(
            score,
            0.0,
            100.0,
        )

        if attempted == 0:
            quality = PerceptionQuality.UNKNOWN

        elif score >= 90:
            quality = PerceptionQuality.EXCELLENT

        elif score >= 75:
            quality = PerceptionQuality.GOOD

        elif score >= self.config.minimum_quality_score:
            quality = PerceptionQuality.DEGRADED

        else:
            quality = PerceptionQuality.POOR

        latencies = [
            stage.latency_ms
            for stage in stages
            if stage.status != StageStatus.SKIPPED
        ]

        return PerceptionQualityReport(
            quality=quality,
            quality_score=score,
            successful_stages=successful,
            failed_stages=failed,
            unavailable_stages=unavailable,
            average_stage_latency_ms=(
                float(np.mean(latencies))
                if latencies
                else 0.0
            ),
            total_latency_ms=total_latency_ms,
            warnings=tuple(warnings),
        )


# ============================================================
# STAGE EXECUTOR
# ============================================================

class PerceptionStageExecutor:

    def __init__(
        self,
        config: PerceptionPipelineConfig,
    ) -> None:
        self.config = config

    def execute(
        self,
        stage: PerceptionStage,
        adapter: PerceptionStageAdapter | None,
        context: PerceptionStageContext,
    ) -> StageExecutionResult:
        if adapter is None:
            return StageExecutionResult(
                stage=stage,
                status=StageStatus.UNAVAILABLE,
                latency_ms=0.0,
                output_available=False,
                error="No adapter registered",
            )

        started = time.perf_counter()

        try:
            output = adapter.process(context)

            elapsed_ms = (
                time.perf_counter() - started
            ) * 1000.0

            if elapsed_ms > self.config.stage_timeout_ms:
                return StageExecutionResult(
                    stage=stage,
                    status=StageStatus.TIMEOUT,
                    latency_ms=elapsed_ms,
                    output_available=False,
                    error=(
                        "Stage exceeded execution deadline; "
                        "output discarded"
                    ),
                )

            context.outputs[stage] = output

            return StageExecutionResult(
                stage=stage,
                status=StageStatus.SUCCESS,
                latency_ms=elapsed_ms,
                output_available=output is not None,
            )

        except Exception as exc:
            elapsed_ms = (
                time.perf_counter() - started
            ) * 1000.0

            return StageExecutionResult(
                stage=stage,
                status=StageStatus.FAILED,
                latency_ms=elapsed_ms,
                output_available=False,
                error=(
                    f"{type(exc).__name__}: {str(exc)[:300]}"
                ),
            )


# ============================================================
# UNIFIED PERCEPTION PIPELINE
# ============================================================

class RoboticsPerceptionPipeline:

    STAGE_ORDER = (
        PerceptionStage.PREPROCESSING,
        PerceptionStage.OBJECT_DETECTION,
        PerceptionStage.INSTANCE_SEGMENTATION,
        PerceptionStage.OBJECT_TRACKING,
        PerceptionStage.DEPTH_ESTIMATION,
        PerceptionStage.LIDAR_PROCESSING,
        PerceptionStage.OPTICAL_FLOW,
        PerceptionStage.SENSOR_FUSION,
        PerceptionStage.OBSTACLE_ANALYSIS,
    )

    STAGE_DEPENDENCIES = {
        PerceptionStage.INSTANCE_SEGMENTATION: (
            PerceptionStage.OBJECT_DETECTION,
        ),
        PerceptionStage.OBJECT_TRACKING: (
            PerceptionStage.OBJECT_DETECTION,
        ),
    }

    def __init__(
        self,
        config: PerceptionPipelineConfig | None = None,
    ) -> None:
        self.config = (
            config or PerceptionPipelineConfig()
        )

        self.validator = PerceptionInputValidator(
            self.config
        )

        self.executor = PerceptionStageExecutor(
            self.config
        )

        self.quality_evaluator = (
            PerceptionPipelineQualityEvaluator(
                self.config
            )
        )

        self._adapters: dict[
            PerceptionStage,
            PerceptionStageAdapter,
        ] = {
            PerceptionStage.PREPROCESSING: (
                BasicImagePreprocessor()
            ),
            PerceptionStage.DEPTH_ESTIMATION: (
                BasicDepthProcessor()
            ),
            PerceptionStage.LIDAR_PROCESSING: (
                BasicLidarProcessor()
            ),
            PerceptionStage.OBSTACLE_ANALYSIS: (
                BasicObstacleAnalyzer(self.config)
            ),
        }

        self._history: dict[
            tuple[str, str],
            deque[PerceptionPipelineResult],
        ] = {}

        self._audit: deque[
            PerceptionAuditEvent
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    @staticmethod
    def _authorize(
        access: PerceptionAccessContext,
        tenant_id: str,
        robot_id: str,
        permission: str,
    ) -> None:
        if access.tenant_id != tenant_id:
            raise PerceptionAuthorizationError(
                "Cross-tenant access denied"
            )

        if robot_id not in access.allowed_robot_ids:
            raise PerceptionAuthorizationError(
                "Robot access denied"
            )

        if not getattr(access, permission, False):
            raise PerceptionAuthorizationError(
                f"Missing permission: {permission}"
            )

    def register_stage(
        self,
        stage: PerceptionStage,
        adapter: PerceptionStageAdapter,
        *,
        access: PerceptionAccessContext,
        robot_id: str,
    ) -> None:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_manage",
        )

        if stage not in self.STAGE_ORDER:
            raise PerceptionValidationError(
                "Stage cannot be registered"
            )

        if not callable(getattr(adapter, "process", None)):
            raise PerceptionValidationError(
                "Adapter must implement process(context)"
            )

        with self._lock:
            self._adapters[stage] = adapter

    def unregister_stage(
        self,
        stage: PerceptionStage,
        *,
        access: PerceptionAccessContext,
        robot_id: str,
    ) -> None:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_manage",
        )

        with self._lock:
            self._adapters.pop(stage, None)

    def _stage_enabled(
        self,
        stage: PerceptionStage,
    ) -> bool:
        mapping = {
            PerceptionStage.PREPROCESSING:
                self.config.enable_preprocessing,
            PerceptionStage.OBJECT_DETECTION:
                self.config.enable_object_detection,
            PerceptionStage.INSTANCE_SEGMENTATION:
                self.config.enable_instance_segmentation,
            PerceptionStage.OBJECT_TRACKING:
                self.config.enable_object_tracking,
            PerceptionStage.DEPTH_ESTIMATION:
                self.config.enable_depth_estimation,
            PerceptionStage.LIDAR_PROCESSING:
                self.config.enable_lidar_processing,
            PerceptionStage.OPTICAL_FLOW:
                self.config.enable_optical_flow,
            PerceptionStage.SENSOR_FUSION:
                self.config.enable_sensor_fusion,
            PerceptionStage.OBSTACLE_ANALYSIS:
                self.config.enable_obstacle_analysis,
        }

        return mapping.get(stage, False)

    @staticmethod
    def _sensor_available(
        stage: PerceptionStage,
        request: PerceptionInput,
    ) -> bool:
        camera_stages = {
            PerceptionStage.PREPROCESSING,
            PerceptionStage.OBJECT_DETECTION,
            PerceptionStage.INSTANCE_SEGMENTATION,
            PerceptionStage.OBJECT_TRACKING,
            PerceptionStage.OPTICAL_FLOW,
        }

        if stage in camera_stages:
            return request.camera is not None

        if stage == PerceptionStage.DEPTH_ESTIMATION:
            return (
                request.camera is not None
                or request.depth is not None
            )

        if stage == PerceptionStage.LIDAR_PROCESSING:
            return request.lidar is not None

        return True

    def process(
        self,
        request: PerceptionInput,
        *,
        access: PerceptionAccessContext,
    ) -> PerceptionPipelineResult:
        self._authorize(
            access,
            request.tenant_id,
            request.robot_id,
            "can_process",
        )

        self.validator.validate(request)

        sensor_skew_ms = (
            self.validator.validate_synchronization(
                request
            )
        )

        started = time.perf_counter()

        context = PerceptionStageContext(
            request=request,
            config=self.config,
            metadata={
                "sensor_skew_ms": sensor_skew_ms,
            },
        )

        stage_results: list[
            StageExecutionResult
        ] = []

        with self._lock:
            adapters = dict(self._adapters)

        for stage in self.STAGE_ORDER:
            if not self._stage_enabled(stage):
                stage_results.append(
                    StageExecutionResult(
                        stage=stage,
                        status=StageStatus.SKIPPED,
                        latency_ms=0.0,
                        output_available=False,
                        metadata={"reason": "disabled"},
                    )
                )
                continue

            if not self._sensor_available(stage, request):
                stage_results.append(
                    StageExecutionResult(
                        stage=stage,
                        status=StageStatus.SKIPPED,
                        latency_ms=0.0,
                        output_available=False,
                        metadata={
                            "reason": "sensor_unavailable"
                        },
                    )
                )
                continue

            dependencies = self.STAGE_DEPENDENCIES.get(
                stage,
                (),
            )

            if any(
                dependency not in context.outputs
                or context.outputs[dependency] is None
                for dependency in dependencies
            ):
                stage_results.append(
                    StageExecutionResult(
                        stage=stage,
                        status=StageStatus.SKIPPED,
                        latency_ms=0.0,
                        output_available=False,
                        metadata={
                            "reason": "dependency_unavailable",
                            "dependencies": [
                                dependency.value
                                for dependency in dependencies
                            ],
                        },
                    )
                )
                continue

            adapter = adapters.get(stage)

            result = self.executor.execute(
                stage,
                adapter,
                context,
            )

            stage_results.append(result)

            if (
                result.status == StageStatus.UNAVAILABLE
                and self.config.require_registered_stages
            ):
                break

            if (
                result.status in (
                    StageStatus.FAILED,
                    StageStatus.TIMEOUT,
                )
                and not self.config.continue_on_stage_failure
            ):
                break

        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000.0

        quality = self.quality_evaluator.evaluate(
            stage_results,
            elapsed_ms,
        )

        failures = sum(
            result.status in (
                StageStatus.FAILED,
                StageStatus.TIMEOUT,
            )
            for result in stage_results
        )

        successes = sum(
            result.status == StageStatus.SUCCESS
            for result in stage_results
        )

        unavailable = sum(
            result.status == StageStatus.UNAVAILABLE
            for result in stage_results
        )

        if successes == 0:
            status = PipelineStatus.FAILED

        elif failures or unavailable:
            status = PipelineStatus.PARTIAL_SUCCESS

        else:
            status = PipelineStatus.SUCCESS

        raw_detections = context.get(
            PerceptionStage.OBJECT_DETECTION,
            (),
        )

        detections = tuple(
            item
            for item in (
                raw_detections
                if isinstance(raw_detections, (list, tuple))
                else ()
            )
            if (
                isinstance(item, Detection)
                and item.confidence
                >= self.config.minimum_detection_confidence
            )
        )

        raw_tracks = context.get(
            PerceptionStage.OBJECT_TRACKING,
            (),
        )

        tracked_objects = tuple(
            item
            for item in (
                raw_tracks
                if isinstance(raw_tracks, (list, tuple))
                else ()
            )
            if isinstance(item, TrackedObject)
        )

        raw_obstacles = context.get(
            PerceptionStage.OBSTACLE_ANALYSIS,
            (),
        )

        obstacles = tuple(
            item
            for item in (
                raw_obstacles
                if isinstance(raw_obstacles, (list, tuple))
                else ()
            )
            if isinstance(item, ObstacleObservation)
        )

        pipeline_result = PerceptionPipelineResult(
            result_id=generate_id(),
            request_id=request.request_id,
            tenant_id=request.tenant_id,
            robot_id=request.robot_id,
            timestamp=request.timestamp,
            status=status,
            detections=detections,
            tracked_objects=tracked_objects,
            obstacles=obstacles,
            stage_results=tuple(stage_results),
            quality=quality,
            total_processing_time_ms=elapsed_ms,
            metadata={
                "sensor_skew_ms": sensor_skew_ms,
                "camera_available": request.camera is not None,
                "lidar_available": request.lidar is not None,
                "depth_available": request.depth is not None,
                "advisory_only": True,
                "motion_authorization": False,
                "protective_control": False,
            },
        )

        key = (
            request.tenant_id,
            request.robot_id,
        )

        with self._lock:
            history = self._history.setdefault(
                key,
                deque(maxlen=self.config.maximum_history),
            )

            history.append(pipeline_result)

            self._audit.append(
                PerceptionAuditEvent(
                    event_id=generate_id(),
                    tenant_id=request.tenant_id,
                    user_id=access.user_id,
                    robot_id=request.robot_id,
                    request_id=request.request_id,
                    action="perception_pipeline.process",
                )
            )

        return pipeline_result

    def latest(
        self,
        *,
        robot_id: str,
        access: PerceptionAccessContext,
    ) -> PerceptionPipelineResult | None:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_view",
        )

        key = (
            access.tenant_id,
            robot_id,
        )

        with self._lock:
            history = self._history.get(key)

            return history[-1] if history else None

    def history(
        self,
        *,
        robot_id: str,
        access: PerceptionAccessContext,
    ) -> tuple[PerceptionPipelineResult, ...]:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_view",
        )

        key = (
            access.tenant_id,
            robot_id,
        )

        with self._lock:
            return tuple(
                self._history.get(key, ())
            )

    def audit_events(
        self,
        *,
        robot_id: str,
        access: PerceptionAccessContext,
    ) -> tuple[PerceptionAuditEvent, ...]:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_view",
        )

        with self._lock:
            return tuple(
                event
                for event in self._audit
                if (
                    event.tenant_id == access.tenant_id
                    and event.robot_id == robot_id
                )
            )

    def reset_robot(
        self,
        *,
        robot_id: str,
        access: PerceptionAccessContext,
    ) -> None:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_manage",
        )

        key = (
            access.tenant_id,
            robot_id,
        )

        with self._lock:
            self._history.pop(key, None)

    @staticmethod
    def export_json(
        result: PerceptionPipelineResult,
    ) -> str:
        return json.dumps(
            result.to_dict(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


# ============================================================
# DEMONSTRATION ADAPTERS
# ============================================================

class DemoObjectDetector:

    def process(
        self,
        context: PerceptionStageContext,
    ) -> tuple[Detection, ...]:
        camera = context.request.camera

        if camera is None:
            return ()

        height, width = camera.image.shape[:2]

        return (
            Detection(
                detection_id=generate_id(),
                class_name="person",
                confidence=0.94,
                x1=width * 0.25,
                y1=height * 0.20,
                x2=width * 0.45,
                y2=height * 0.85,
            ),
        )


class DemoObjectTracker:

    def process(
        self,
        context: PerceptionStageContext,
    ) -> tuple[TrackedObject, ...]:
        detections = context.get(
            PerceptionStage.OBJECT_DETECTION,
            (),
        )

        tracks = []

        for index, detection in enumerate(detections):
            if not isinstance(detection, Detection):
                continue

            tracks.append(
                TrackedObject(
                    track_id=f"TRACK-{index + 1:03d}",
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    x1=detection.x1,
                    y1=detection.y1,
                    x2=detection.x2,
                    y2=detection.y2,
                )
            )

        return tuple(tracks)


class DemoInstanceSegmenter:

    def process(
        self,
        context: PerceptionStageContext,
    ) -> dict[str, Any]:
        detections = context.get(
            PerceptionStage.OBJECT_DETECTION,
            (),
        )

        return {
            "instance_count": len(detections),
            "method": "demonstration_only",
            "masks_generated": False,
        }


class DemoOpticalFlow:

    def process(
        self,
        context: PerceptionStageContext,
    ) -> dict[str, Any]:
        return {
            "mean_motion_px": 0.0,
            "tracked_features": 0,
            "status": "demonstration_only",
        }


class DemoSensorFusion:

    def process(
        self,
        context: PerceptionStageContext,
    ) -> dict[str, Any]:
        return {
            "detections_available": (
                PerceptionStage.OBJECT_DETECTION
                in context.outputs
            ),
            "depth_available": (
                PerceptionStage.DEPTH_ESTIMATION
                in context.outputs
            ),
            "lidar_available": (
                PerceptionStage.LIDAR_PROCESSING
                in context.outputs
            ),
            "fusion_status": "demonstration_only",
        }


# ============================================================
# UNIT TESTS
# ============================================================

class TestPerceptionPipeline(unittest.TestCase):

    def setUp(self) -> None:
        self.config = PerceptionPipelineConfig()

        self.pipeline = RoboticsPerceptionPipeline(
            self.config
        )

        self.access = PerceptionAccessContext(
            tenant_id="tenant_alpha",
            user_id="robotics_engineer",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
            }),
            can_process=True,
            can_view=True,
            can_manage=True,
        )

        self.timestamp = utc_now()

        self.image = np.zeros(
            (240, 320, 3),
            dtype=np.uint8,
        )

    def make_request(
        self,
        **overrides: Any,
    ) -> PerceptionInput:
        values = {
            "request_id": generate_id(),
            "tenant_id": "tenant_alpha",
            "robot_id": "ROBOT-001",
            "timestamp": self.timestamp,
            "camera": CameraInput(
                camera_id="CAM-001",
                image=self.image,
                timestamp=self.timestamp,
            ),
        }

        values.update(overrides)

        return PerceptionInput(**values)

    def test_camera_input(self) -> None:
        request = self.make_request()

        self.assertEqual(
            request.camera.image.shape,
            (240, 320, 3),
        )

    def test_invalid_image(self) -> None:
        with self.assertRaises(
            PerceptionValidationError
        ):
            CameraInput(
                camera_id="CAM-001",
                image=np.zeros(
                    (10, 10),
                    dtype=np.float32,
                ),
                timestamp=self.timestamp,
            )

    def test_preprocessing(self) -> None:
        context = PerceptionStageContext(
            request=self.make_request(),
            config=self.config,
        )

        output = BasicImagePreprocessor().process(
            context
        )

        self.assertEqual(
            output.shape,
            self.image.shape,
        )

    def test_lidar_processing(self) -> None:
        points = np.array(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        request = self.make_request(
            camera=None,
            lidar=LidarInput(
                lidar_id="LIDAR-001",
                points=points,
                timestamp=self.timestamp,
            ),
        )

        result = BasicLidarProcessor().process(
            PerceptionStageContext(
                request=request,
                config=self.config,
            )
        )

        self.assertAlmostEqual(
            result["minimum_range_m"],
            1.0,
        )

    def test_depth_processing(self) -> None:
        request = self.make_request(
            camera=None,
            depth=DepthInput(
                sensor_id="DEPTH-001",
                depth_map=np.array(
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                    ],
                    dtype=np.float32,
                ),
                timestamp=self.timestamp,
            ),
        )

        result = BasicDepthProcessor().process(
            PerceptionStageContext(
                request=request,
                config=self.config,
            )
        )

        self.assertAlmostEqual(
            result["minimum_depth_m"],
            1.0,
        )

    def test_sensor_synchronization(self) -> None:
        request = self.make_request(
            lidar=LidarInput(
                lidar_id="LIDAR-001",
                points=np.zeros(
                    (10, 3),
                    dtype=np.float32,
                ),
                timestamp=(
                    self.timestamp
                    + timedelta(milliseconds=200)
                ),
            ),
        )

        with self.assertRaises(
            PerceptionSynchronizationError
        ):
            self.pipeline.validator.validate_synchronization(
                request
            )

    def test_register_detector(self) -> None:
        self.pipeline.register_stage(
            PerceptionStage.OBJECT_DETECTION,
            DemoObjectDetector(),
            access=self.access,
            robot_id="ROBOT-001",
        )

        self.assertIn(
            PerceptionStage.OBJECT_DETECTION,
            self.pipeline._adapters,
        )

    def test_pipeline_processing(self) -> None:
        result = self.pipeline.process(
            self.make_request(),
            access=self.access,
        )

        self.assertIn(
            result.status,
            (
                PipelineStatus.SUCCESS,
                PipelineStatus.PARTIAL_SUCCESS,
            ),
        )

    def test_object_detection(self) -> None:
        self.pipeline.register_stage(
            PerceptionStage.OBJECT_DETECTION,
            DemoObjectDetector(),
            access=self.access,
            robot_id="ROBOT-001",
        )

        result = self.pipeline.process(
            self.make_request(),
            access=self.access,
        )

        self.assertEqual(
            len(result.detections),
            1,
        )

    def test_object_tracking(self) -> None:
        self.pipeline.register_stage(
            PerceptionStage.OBJECT_DETECTION,
            DemoObjectDetector(),
            access=self.access,
            robot_id="ROBOT-001",
        )

        self.pipeline.register_stage(
            PerceptionStage.OBJECT_TRACKING,
            DemoObjectTracker(),
            access=self.access,
            robot_id="ROBOT-001",
        )

        result = self.pipeline.process(
            self.make_request(),
            access=self.access,
        )

        self.assertEqual(
            len(result.tracked_objects),
            1,
        )

    def test_obstacle_analysis(self) -> None:
        request = self.make_request(
            camera=None,
            lidar=LidarInput(
                lidar_id="LIDAR-001",
                points=np.array(
                    [
                        [0.5, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
                timestamp=self.timestamp,
            ),
        )

        result = self.pipeline.process(
            request,
            access=self.access,
        )

        self.assertTrue(
            any(
                obstacle.risk_level
                == ObstacleRiskLevel.HIGH
                for obstacle in result.obstacles
            )
        )

    def test_tenant_isolation(self) -> None:
        with self.assertRaises(
            PerceptionAuthorizationError
        ):
            self.pipeline._authorize(
                self.access,
                "tenant_beta",
                "ROBOT-001",
                "can_process",
            )

    def test_history(self) -> None:
        self.pipeline.process(
            self.make_request(),
            access=self.access,
        )

        history = self.pipeline.history(
            robot_id="ROBOT-001",
            access=self.access,
        )

        self.assertEqual(
            len(history),
            1,
        )

    def test_latest(self) -> None:
        self.pipeline.process(
            self.make_request(),
            access=self.access,
        )

        latest = self.pipeline.latest(
            robot_id="ROBOT-001",
            access=self.access,
        )

        self.assertIsNotNone(latest)

    def test_json_export(self) -> None:
        result = self.pipeline.process(
            self.make_request(),
            access=self.access,
        )

        payload = self.pipeline.export_json(
            result
        )

        self.assertIn(
            "ROBOT-001",
            payload,
        )

    def test_stage_failure_isolation(self) -> None:
        def failing_stage(
            context: PerceptionStageContext,
        ) -> Any:
            raise RuntimeError(
                "Synthetic inference failure"
            )

        self.pipeline.register_stage(
            PerceptionStage.OBJECT_DETECTION,
            FunctionStageAdapter(failing_stage),
            access=self.access,
            robot_id="ROBOT-001",
        )

        result = self.pipeline.process(
            self.make_request(),
            access=self.access,
        )

        self.assertTrue(
            any(
                stage.status == StageStatus.FAILED
                for stage in result.stage_results
            )
        )

    def test_quality_score(self) -> None:
        evaluator = PerceptionPipelineQualityEvaluator(
            self.config
        )

        report = evaluator.evaluate(
            [
                StageExecutionResult(
                    stage=PerceptionStage.PREPROCESSING,
                    status=StageStatus.SUCCESS,
                    latency_ms=10.0,
                    output_available=True,
                )
            ],
            10.0,
        )

        self.assertEqual(
            report.quality_score,
            100.0,
        )

    def test_audit(self) -> None:
        self.pipeline.process(
            self.make_request(),
            access=self.access,
        )

        events = self.pipeline.audit_events(
            robot_id="ROBOT-001",
            access=self.access,
        )

        self.assertEqual(
            len(events),
            1,
        )

    def test_reset(self) -> None:
        self.pipeline.process(
            self.make_request(),
            access=self.access,
        )

        self.pipeline.reset_robot(
            robot_id="ROBOT-001",
            access=self.access,
        )

        self.assertEqual(
            len(
                self.pipeline.history(
                    robot_id="ROBOT-001",
                    access=self.access,
                )
            ),
            0,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print(
        "\nPHYSICAL AI & AUTONOMOUS ROBOTICS PLATFORM"
    )

    print(
        "Unified Real-Time Perception Pipeline\n"
    )

    access = PerceptionAccessContext(
        tenant_id="tenant_alpha",
        user_id="robotics_engineer",
        allowed_robot_ids=frozenset({
            "ROBOT-001",
        }),
        can_process=True,
        can_view=True,
        can_manage=True,
    )

    pipeline = RoboticsPerceptionPipeline()

    demo_adapters = {
        PerceptionStage.OBJECT_DETECTION:
            DemoObjectDetector(),
        PerceptionStage.INSTANCE_SEGMENTATION:
            DemoInstanceSegmenter(),
        PerceptionStage.OBJECT_TRACKING:
            DemoObjectTracker(),
        PerceptionStage.OPTICAL_FLOW:
            DemoOpticalFlow(),
        PerceptionStage.SENSOR_FUSION:
            DemoSensorFusion(),
    }

    for stage, adapter in demo_adapters.items():
        pipeline.register_stage(
            stage,
            adapter,
            access=access,
            robot_id="ROBOT-001",
        )

    timestamp = utc_now()

    image = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    lidar_points = np.array(
        [
            [0.8, 0.0, 0.0],
            [1.5, 0.2, 0.0],
            [3.0, 0.0, 0.0],
            [5.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    depth_map = np.full(
        (480, 640),
        2.5,
        dtype=np.float32,
    )

    request = PerceptionInput(
        request_id=generate_id(),
        tenant_id="tenant_alpha",
        robot_id="ROBOT-001",
        timestamp=timestamp,
        camera=CameraInput(
            camera_id="CAM-FRONT-001",
            image=image,
            timestamp=timestamp,
        ),
        lidar=LidarInput(
            lidar_id="LIDAR-001",
            points=lidar_points,
            timestamp=timestamp,
        ),
        depth=DepthInput(
            sensor_id="DEPTH-001",
            depth_map=depth_map,
            timestamp=timestamp,
        ),
    )

    result = pipeline.process(
        request,
        access=access,
    )

    print("PIPELINE STATUS")
    print(result.status.value)

    print("\nSTAGE EXECUTION")

    for stage in result.stage_results:
        print(
            f"{stage.stage.value:25s} "
            f"{stage.status.value:18s} "
            f"{stage.latency_ms:.2f}ms"
        )

    print("\nDETECTIONS")
    print(len(result.detections))

    print("\nTRACKED OBJECTS")
    print(len(result.tracked_objects))

    print("\nOBSTACLE OBSERVATIONS")

    for obstacle in result.obstacles:
        print(
            f"{obstacle.source}: "
            f"{obstacle.distance_m:.2f}m "
            f"({obstacle.risk_level.value})"
        )

    print("\nPERCEPTION QUALITY")
    print(result.quality.quality.value)
    print(
        f"Score: {result.quality.quality_score:.1f}/100"
    )

    print("\nJSON REPORT\n")

    print(
        pipeline.export_json(result)
    )

    print("\nRUNNING UNIT TESTS\n")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestPerceptionPipeline
    )

    test_result = unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)

    if not test_result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
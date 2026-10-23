"""Perception Agent module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/ai/agents/perception_agent.py

Physical AI & Autonomous Robotics Intelligence Platform
Multimodal Robotic Perception Agent

Python 3.11+

Features:
- Camera-based object detection
- Object classification and localization
- Object tracking
- Image segmentation integration
- Depth camera processing
- LiDAR point-cloud processing
- Sensor fusion
- 3D object position estimation
- Obstacle detection
- Human and worker detection
- PPE detection integration
- Robot workspace monitoring
- Sensor health monitoring
- Perception confidence assessment
- Anomaly detection integration
- Scene understanding
- Perception result aggregation
- Safety-aware perception reports
- Tenant-aware authorization
- Audit logging

Important:
- AI models must be supplied through provider interfaces.
- Detection and segmentation outputs are not fabricated.
- No physical robot actuation.
- No emergency-stop override.
- Perception uncertainty must not be treated as proof
  that the robot workspace is safe.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import re
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

MAX_DETECTIONS = 2000
MAX_TRACKS = 2000
MAX_POINT_CLOUD_POINTS = 2_000_000
MAX_SENSOR_FRAMES = 32
MAX_BATCH_REQUESTS = 25
MAX_METADATA_FIELDS = 50

DEFAULT_DETECTION_THRESHOLD = 0.50
DEFAULT_TRACKING_THRESHOLD = 0.40
DEFAULT_HUMAN_SAFETY_DISTANCE_M = 1.50
DEFAULT_SENSOR_STALE_SECONDS = 2.0

IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z0-9_.:-]{1,128}$"
)


# ============================================================
# Exceptions
# ============================================================

class PerceptionAgentError(Exception):
    """Base perception exception."""


class PerceptionValidationError(PerceptionAgentError):
    """Invalid perception input."""


class PerceptionNotFoundError(PerceptionAgentError):
    """Required sensor data was not found."""


class PerceptionPermissionError(PerceptionAgentError):
    """Perception access was denied."""


class PerceptionModelError(PerceptionAgentError):
    """An AI perception model failed."""


class PerceptionSensorError(PerceptionAgentError):
    """A sensor reported invalid or unavailable data."""


class PerceptionFusionError(PerceptionAgentError):
    """Sensor fusion failed."""


class PerceptionServiceUnavailableError(
    PerceptionAgentError
):
    """Required perception provider is unavailable."""


# ============================================================
# Utilities
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return str(uuid.uuid4())


def validate_identifier(value: str) -> str:
    if (
        not isinstance(value, str)
        or not IDENTIFIER_PATTERN.fullmatch(value)
    ):
        raise PerceptionValidationError(
            f"Invalid identifier: {value!r}"
        )

    return value


def validate_finite(value: float) -> float:
    if not math.isfinite(value):
        raise PerceptionValidationError(
            "Value must be finite"
        )

    return value


def validate_nonnegative(value: float) -> float:
    validate_finite(value)

    if value < 0:
        raise PerceptionValidationError(
            "Value must be nonnegative"
        )

    return value


def validate_positive(value: float) -> float:
    validate_finite(value)

    if value <= 0:
        raise PerceptionValidationError(
            "Value must be positive"
        )

    return value


def validate_probability(value: float) -> float:
    validate_finite(value)

    if not 0.0 <= value <= 1.0:
        raise PerceptionValidationError(
            "Probability must be between 0 and 1"
        )

    return value


def validate_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise PerceptionValidationError(
            "Datetime must be timezone-aware"
        )

    return value


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(minimum, min(maximum, value))


# ============================================================
# Enumerations
# ============================================================

class SensorType(str, Enum):
    RGB_CAMERA = "rgb_camera"
    DEPTH_CAMERA = "depth_camera"
    STEREO_CAMERA = "stereo_camera"
    THERMAL_CAMERA = "thermal_camera"
    LIDAR = "lidar"
    RADAR = "radar"
    ULTRASONIC = "ultrasonic"
    IMU = "imu"
    FORCE_TORQUE = "force_torque"
    TACTILE = "tactile"
    CUSTOM = "custom"


class SensorHealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAULT = "fault"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class PerceptionTaskType(str, Enum):
    OBJECT_DETECTION = "object_detection"
    OBJECT_TRACKING = "object_tracking"
    IMAGE_SEGMENTATION = "image_segmentation"
    DEPTH_ESTIMATION = "depth_estimation"
    POINT_CLOUD_ANALYSIS = "point_cloud_analysis"
    OBSTACLE_DETECTION = "obstacle_detection"
    HUMAN_DETECTION = "human_detection"
    PPE_DETECTION = "ppe_detection"
    SENSOR_FUSION = "sensor_fusion"
    SCENE_UNDERSTANDING = "scene_understanding"
    ANOMALY_DETECTION = "anomaly_detection"
    FULL_PERCEPTION = "full_perception"


class ObjectCategory(str, Enum):
    HUMAN = "human"
    ROBOT = "robot"
    VEHICLE = "vehicle"
    TOOL = "tool"
    MACHINE = "machine"
    ROBOT_PART = "robot_part"
    PACKAGE = "package"
    PALLET = "pallet"
    BOX = "box"
    OBSTACLE = "obstacle"
    SAFETY_EQUIPMENT = "safety_equipment"
    UNKNOWN = "unknown"


class TrackingStatus(str, Enum):
    NEW = "new"
    TRACKED = "tracked"
    OCCLUDED = "occluded"
    LOST = "lost"


class PerceptionRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class PerceptionStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    REVIEW_REQUIRED = "review_required"


class SceneSafetyStatus(str, Enum):
    NO_HAZARD_DETECTED = "no_hazard_detected"
    POTENTIAL_HAZARD = "potential_hazard"
    HAZARD_DETECTED = "hazard_detected"
    UNKNOWN = "unknown"


# ============================================================
# Security Scope
# ============================================================

@dataclass(frozen=True, slots=True)
class PerceptionScope:
    tenant_id: str
    user_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_identifier(self.user_id)


# ============================================================
# Geometry Models
# ============================================================

@dataclass(frozen=True, slots=True)
class Vector3:
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        validate_finite(self.x)
        validate_finite(self.y)
        validate_finite(self.z)

    def distance_to(self, other: Vector3) -> float:
        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )


@dataclass(frozen=True, slots=True)
class BoundingBox2D:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        for value in (
            self.x_min,
            self.y_min,
            self.x_max,
            self.y_max,
        ):
            validate_finite(value)

        if (
            self.x_min >= self.x_max
            or self.y_min >= self.y_max
        ):
            raise PerceptionValidationError(
                "Invalid bounding box"
            )

    @property
    def area(self) -> float:
        return (
            (self.x_max - self.x_min)
            * (self.y_max - self.y_min)
        )

    def intersection_over_union(
        self,
        other: BoundingBox2D,
    ) -> float:

        x1 = max(self.x_min, other.x_min)
        y1 = max(self.y_min, other.y_min)

        x2 = min(self.x_max, other.x_max)
        y2 = min(self.y_max, other.y_max)

        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)

        intersection = width * height

        union = (
            self.area + other.area - intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union


@dataclass(frozen=True, slots=True)
class BoundingBox3D:
    center: Vector3
    dimensions: Vector3

    def __post_init__(self) -> None:
        validate_positive(self.dimensions.x)
        validate_positive(self.dimensions.y)
        validate_positive(self.dimensions.z)


# ============================================================
# Sensor Models
# ============================================================

@dataclass(slots=True)
class SensorMetadata:
    sensor_id: str
    robot_id: str
    tenant_id: str

    sensor_type: SensorType

    frame_id: str

    timestamp: datetime

    health_status: SensorHealthStatus = (
        SensorHealthStatus.UNKNOWN
    )

    confidence: float = 1.0

    def __post_init__(self) -> None:
        validate_identifier(self.sensor_id)
        validate_identifier(self.robot_id)
        validate_identifier(self.tenant_id)
        validate_identifier(self.frame_id)

        validate_timestamp(self.timestamp)
        validate_probability(self.confidence)


@dataclass(slots=True)
class ImageFrame:
    metadata: SensorMetadata

    width: int
    height: int

    encoding: str

    image_uri: str

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise PerceptionValidationError(
                "Invalid image dimensions"
            )

        if not self.image_uri.strip():
            raise PerceptionValidationError(
                "Image URI is required"
            )


@dataclass(slots=True)
class DepthFrame:
    metadata: SensorMetadata

    width: int
    height: int

    depth_uri: str

    minimum_depth_m: float
    maximum_depth_m: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise PerceptionValidationError(
                "Invalid depth frame dimensions"
            )

        validate_nonnegative(
            self.minimum_depth_m
        )

        validate_positive(
            self.maximum_depth_m
        )

        if (
            self.minimum_depth_m
            >= self.maximum_depth_m
        ):
            raise PerceptionValidationError(
                "Invalid depth range"
            )


@dataclass(slots=True)
class PointCloudFrame:
    metadata: SensorMetadata

    point_cloud_uri: str

    point_count: int

    coordinate_frame: str

    def __post_init__(self) -> None:
        if not (
            0 <= self.point_count
            <= MAX_POINT_CLOUD_POINTS
        ):
            raise PerceptionValidationError(
                "Invalid point cloud size"
            )

        validate_identifier(
            self.coordinate_frame
        )

        if not self.point_cloud_uri.strip():
            raise PerceptionValidationError(
                "Point cloud URI is required"
            )


# ============================================================
# Detection Models
# ============================================================

@dataclass(slots=True)
class ObjectDetection:
    detection_id: str

    label: str
    category: ObjectCategory

    confidence: float

    bounding_box: BoundingBox2D

    source_sensor_id: str

    position_3d: Vector3 | None = None
    position_frame_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_identifier(self.detection_id)
        validate_identifier(self.source_sensor_id)

        validate_probability(self.confidence)

        if self.position_3d is not None:
            if self.position_frame_id is None:
                raise PerceptionValidationError(
                    "3D position requires a frame identifier"
                )

            validate_identifier(
                self.position_frame_id
            )


@dataclass(slots=True)
class ObjectTrack:
    track_id: str

    category: ObjectCategory
    label: str

    status: TrackingStatus

    confidence: float

    position: Vector3 | None
    velocity: Vector3 | None

    frame_id: str | None

    last_seen: datetime

    detection_ids: list[str] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        validate_identifier(self.track_id)
        validate_probability(self.confidence)
        validate_timestamp(self.last_seen)

        if self.position is not None:
            if self.frame_id is None:
                raise PerceptionValidationError(
                    "Tracked position requires frame ID"
                )

            validate_identifier(self.frame_id)


@dataclass(slots=True)
class SegmentationResult:
    segmentation_id: str

    source_sensor_id: str

    mask_uri: str

    class_labels: list[str]

    confidence: float

    def __post_init__(self) -> None:
        validate_identifier(self.segmentation_id)
        validate_identifier(self.source_sensor_id)
        validate_probability(self.confidence)

        if not self.mask_uri.strip():
            raise PerceptionValidationError(
                "Segmentation mask URI is required"
            )


@dataclass(slots=True)
class PerceptionAnomaly:
    anomaly_id: str

    anomaly_type: str
    description: str

    confidence: float

    risk_level: PerceptionRiskLevel

    sensor_id: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.anomaly_id)
        validate_probability(self.confidence)


# ============================================================
# Obstacle Models
# ============================================================

@dataclass(slots=True)
class PerceivedObstacle:
    obstacle_id: str

    category: ObjectCategory

    position: Vector3

    frame_id: str

    radius_m: float

    confidence: float

    dynamic: bool = False

    def __post_init__(self) -> None:
        validate_identifier(self.obstacle_id)
        validate_identifier(self.frame_id)

        validate_nonnegative(self.radius_m)
        validate_probability(self.confidence)


# ============================================================
# Perception Request
# ============================================================

@dataclass(slots=True)
class PerceptionRequest:
    tenant_id: str
    robot_id: str

    task_type: PerceptionTaskType

    image_frames: list[ImageFrame] = field(
        default_factory=list
    )

    depth_frames: list[DepthFrame] = field(
        default_factory=list
    )

    point_cloud_frames: list[PointCloudFrame] = field(
        default_factory=list
    )

    detection_threshold: float = (
        DEFAULT_DETECTION_THRESHOLD
    )

    enable_tracking: bool = False
    enable_segmentation: bool = False
    enable_sensor_fusion: bool = False
    enable_anomaly_detection: bool = False

    reference_position: Vector3 | None = None
    reference_frame_id: str | None = None

    human_safety_distance_m: float = (
        DEFAULT_HUMAN_SAFETY_DISTANCE_M
    )

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_identifier(self.robot_id)

        validate_probability(
            self.detection_threshold
        )

        validate_positive(
            self.human_safety_distance_m
        )

        total_frames = (
            len(self.image_frames)
            + len(self.depth_frames)
            + len(self.point_cloud_frames)
        )

        if total_frames > MAX_SENSOR_FRAMES:
            raise PerceptionValidationError(
                "Too many sensor frames"
            )

        if total_frames == 0:
            raise PerceptionValidationError(
                "At least one sensor frame is required"
            )

        if self.reference_position is not None:
            if self.reference_frame_id is None:
                raise PerceptionValidationError(
                    "Reference position requires frame ID"
                )

            validate_identifier(
                self.reference_frame_id
            )

        for frame in (
            *self.image_frames,
            *self.depth_frames,
            *self.point_cloud_frames,
        ):
            metadata = frame.metadata

            if (
                metadata.tenant_id != self.tenant_id
                or metadata.robot_id != self.robot_id
            ):
                raise PerceptionPermissionError(
                    "Sensor frame scope mismatch"
                )


# ============================================================
# Safety Assessment
# ============================================================

@dataclass(frozen=True, slots=True)
class PerceptionSafetyAssessment:
    safety_status: SceneSafetyStatus

    risk_level: PerceptionRiskLevel

    human_detected: bool

    human_within_safety_distance: bool | None

    critical_anomaly_detected: bool

    sensor_data_reliable: bool

    requires_human_review: bool

    findings: tuple[str, ...]


# ============================================================
# Perception Result
# ============================================================

@dataclass(slots=True)
class PerceptionResult:
    result_id: str

    tenant_id: str
    robot_id: str

    task_type: PerceptionTaskType

    status: PerceptionStatus

    detections: list[ObjectDetection]

    tracks: list[ObjectTrack]

    segmentations: list[SegmentationResult]

    obstacles: list[PerceivedObstacle]

    anomalies: list[PerceptionAnomaly]

    safety_assessment: PerceptionSafetyAssessment

    sensor_count: int

    processing_duration_seconds: float

    reasoning: list[str]

    created_at: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# AI Provider Protocols
# ============================================================

class ObjectDetectionProvider(Protocol):

    async def detect_objects(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        frame: ImageFrame,
        confidence_threshold: float,
    ) -> list[ObjectDetection]:
        ...


class ObjectTrackingProvider(Protocol):

    async def update_tracks(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        detections: Sequence[ObjectDetection],
    ) -> list[ObjectTrack]:
        ...


class ImageSegmentationProvider(Protocol):

    async def segment_image(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        frame: ImageFrame,
    ) -> list[SegmentationResult]:
        ...


class DepthEstimationProvider(Protocol):

    async def estimate_positions(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        detections: Sequence[ObjectDetection],
        depth_frames: Sequence[DepthFrame],
    ) -> list[ObjectDetection]:
        ...


class PointCloudAnalysisProvider(Protocol):

    async def detect_obstacles(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        frames: Sequence[PointCloudFrame],
    ) -> list[PerceivedObstacle]:
        ...


class SensorFusionProvider(Protocol):

    async def fuse(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        detections: Sequence[ObjectDetection],
        obstacles: Sequence[PerceivedObstacle],
    ) -> tuple[
        list[ObjectDetection],
        list[PerceivedObstacle],
    ]:
        ...


class PerceptionAnomalyProvider(Protocol):

    async def detect_anomalies(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        detections: Sequence[ObjectDetection],
        obstacles: Sequence[PerceivedObstacle],
    ) -> list[PerceptionAnomaly]:
        ...


# ============================================================
# Infrastructure Protocols
# ============================================================

class PerceptionAuthorizer(Protocol):

    async def require_permission(
        self,
        *,
        tenant_id: str,
        user_id: str,
        robot_id: str,
        permission: str,
    ) -> None:
        ...


class PerceptionRepository(Protocol):

    async def save_result(
        self,
        *,
        result: PerceptionResult,
    ) -> None:
        ...


class PerceptionAuditSink(Protocol):

    async def record(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        action: str,
        resource_id: str,
        metadata: Mapping[str, Any],
    ) -> None:
        ...


# ============================================================
# Perception Agent
# ============================================================

class PerceptionAgent:

    def __init__(
        self,
        *,
        detection_provider: ObjectDetectionProvider,
        authorizer: PerceptionAuthorizer,
        tracking_provider: (
            ObjectTrackingProvider | None
        ) = None,
        segmentation_provider: (
            ImageSegmentationProvider | None
        ) = None,
        depth_provider: (
            DepthEstimationProvider | None
        ) = None,
        point_cloud_provider: (
            PointCloudAnalysisProvider | None
        ) = None,
        fusion_provider: (
            SensorFusionProvider | None
        ) = None,
        anomaly_provider: (
            PerceptionAnomalyProvider | None
        ) = None,
        repository: PerceptionRepository | None = None,
        audit_sink: PerceptionAuditSink | None = None,
        stale_sensor_seconds: float = (
            DEFAULT_SENSOR_STALE_SECONDS
        ),
    ) -> None:

        validate_positive(stale_sensor_seconds)

        self.detection_provider = detection_provider
        self.tracking_provider = tracking_provider
        self.segmentation_provider = (
            segmentation_provider
        )
        self.depth_provider = depth_provider
        self.point_cloud_provider = (
            point_cloud_provider
        )
        self.fusion_provider = fusion_provider
        self.anomaly_provider = anomaly_provider

        self.authorizer = authorizer
        self.repository = repository
        self.audit_sink = audit_sink

        self.stale_sensor_seconds = (
            stale_sensor_seconds
        )

    # --------------------------------------------------------
    # Authorization
    # --------------------------------------------------------

    async def _authorize(
        self,
        *,
        scope: PerceptionScope,
        robot_id: str,
    ) -> None:

        await self.authorizer.require_permission(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            robot_id=robot_id,
            permission="perception:analyze",
        )

    # --------------------------------------------------------
    # Sensor Health
    # --------------------------------------------------------

    def evaluate_sensor_health(
        self,
        *,
        request: PerceptionRequest,
    ) -> tuple[bool, list[str]]:

        findings: list[str] = []

        now = utc_now()

        frames = [
            *request.image_frames,
            *request.depth_frames,
            *request.point_cloud_frames,
        ]

        reliable = True

        for frame in frames:
            metadata = frame.metadata

            age_seconds = (
                now - metadata.timestamp
            ).total_seconds()

            if age_seconds < -self.stale_sensor_seconds:
                reliable = False
                findings.append(
                    f"Sensor {metadata.sensor_id} "
                    "has a future timestamp."
                )

            if age_seconds > self.stale_sensor_seconds:
                reliable = False
                findings.append(
                    f"Sensor {metadata.sensor_id} "
                    "data is stale."
                )

            if metadata.health_status != (
                SensorHealthStatus.HEALTHY
            ):
                reliable = False
                findings.append(
                    f"Sensor {metadata.sensor_id} "
                    f"health is {metadata.health_status.value}."
                )

            if metadata.confidence < 0.80:
                reliable = False
                findings.append(
                    f"Sensor {metadata.sensor_id} "
                    "reports low confidence."
                )

        return reliable, findings

    # --------------------------------------------------------
    # Object Detection
    # --------------------------------------------------------

    async def detect_objects(
        self,
        *,
        scope: PerceptionScope,
        request: PerceptionRequest,
    ) -> list[ObjectDetection]:

        detections: list[ObjectDetection] = []

        for frame in request.image_frames:
            frame_detections = (
                await self.detection_provider.detect_objects(
                    tenant_id=scope.tenant_id,
                    robot_id=request.robot_id,
                    frame=frame,
                    confidence_threshold=(
                        request.detection_threshold
                    ),
                )
            )

            for detection in frame_detections:
                if (
                    detection.source_sensor_id
                    != frame.metadata.sensor_id
                ):
                    raise PerceptionValidationError(
                        "Detection source sensor mismatch"
                    )

                if (
                    detection.confidence
                    >= request.detection_threshold
                ):
                    detections.append(detection)

        if len(detections) > MAX_DETECTIONS:
            raise PerceptionValidationError(
                "Detection limit exceeded"
            )

        return detections

    # --------------------------------------------------------
    # Object Tracking
    # --------------------------------------------------------

    async def track_objects(
        self,
        *,
        scope: PerceptionScope,
        robot_id: str,
        detections: Sequence[ObjectDetection],
    ) -> list[ObjectTrack]:

        if self.tracking_provider is None:
            raise PerceptionServiceUnavailableError(
                "Tracking provider is unavailable"
            )

        tracks = (
            await self.tracking_provider.update_tracks(
                tenant_id=scope.tenant_id,
                robot_id=robot_id,
                detections=detections,
            )
        )

        if len(tracks) > MAX_TRACKS:
            raise PerceptionValidationError(
                "Tracking limit exceeded"
            )

        return tracks

    # --------------------------------------------------------
    # Image Segmentation
    # --------------------------------------------------------

    async def segment_images(
        self,
        *,
        scope: PerceptionScope,
        request: PerceptionRequest,
    ) -> list[SegmentationResult]:

        if self.segmentation_provider is None:
            raise PerceptionServiceUnavailableError(
                "Segmentation provider is unavailable"
            )

        results: list[SegmentationResult] = []

        for frame in request.image_frames:
            segmentations = (
                await self.segmentation_provider.segment_image(
                    tenant_id=scope.tenant_id,
                    robot_id=request.robot_id,
                    frame=frame,
                )
            )

            for segmentation in segmentations:
                if (
                    segmentation.source_sensor_id
                    != frame.metadata.sensor_id
                ):
                    raise PerceptionValidationError(
                        "Segmentation sensor mismatch"
                    )

            results.extend(segmentations)

        return results

    # --------------------------------------------------------
    # Point Cloud Analysis
    # --------------------------------------------------------

    async def analyze_point_clouds(
        self,
        *,
        scope: PerceptionScope,
        request: PerceptionRequest,
    ) -> list[PerceivedObstacle]:

        if not request.point_cloud_frames:
            return []

        if self.point_cloud_provider is None:
            raise PerceptionServiceUnavailableError(
                "Point cloud provider is unavailable"
            )

        return (
            await self.point_cloud_provider.detect_obstacles(
                tenant_id=scope.tenant_id,
                robot_id=request.robot_id,
                frames=request.point_cloud_frames,
            )
        )

    # --------------------------------------------------------
    # Safety Assessment
    # --------------------------------------------------------

    @staticmethod
    def assess_scene_safety(
        *,
        request: PerceptionRequest,
        detections: Sequence[ObjectDetection],
        obstacles: Sequence[PerceivedObstacle],
        anomalies: Sequence[PerceptionAnomaly],
        sensor_data_reliable: bool,
        sensor_findings: Sequence[str],
    ) -> PerceptionSafetyAssessment:

        findings = list(sensor_findings)

        human_detections = [
            detection
            for detection in detections
            if detection.category == ObjectCategory.HUMAN
        ]

        human_obstacles = [
            obstacle
            for obstacle in obstacles
            if obstacle.category == ObjectCategory.HUMAN
        ]

        human_detected = bool(
            human_detections or human_obstacles
        )

        human_within_distance: bool | None = None

        if (
            request.reference_position is not None
            and request.reference_frame_id is not None
        ):
            distances: list[float] = []

            for detection in human_detections:
                if (
                    detection.position_3d is not None
                    and detection.position_frame_id
                    == request.reference_frame_id
                ):
                    distances.append(
                        request.reference_position.distance_to(
                            detection.position_3d
                        )
                    )

            for obstacle in human_obstacles:
                if (
                    obstacle.frame_id
                    == request.reference_frame_id
                ):
                    distances.append(
                        max(
                            0.0,
                            request.reference_position.distance_to(
                                obstacle.position
                            ) - obstacle.radius_m,
                        )
                    )

            if distances:
                human_within_distance = (
                    min(distances)
                    <= request.human_safety_distance_m
                )

        if human_detected:
            findings.append(
                "Human presence detected in the scene."
            )

        if human_within_distance is True:
            findings.append(
                "Human detected within configured "
                "safety distance."
            )

        if (
            human_detected
            and human_within_distance is None
        ):
            findings.append(
                "Human distance could not be verified."
            )

        critical_anomaly = any(
            anomaly.risk_level
            == PerceptionRiskLevel.CRITICAL
            for anomaly in anomalies
        )

        high_anomaly = any(
            anomaly.risk_level
            in (
                PerceptionRiskLevel.HIGH,
                PerceptionRiskLevel.CRITICAL,
            )
            for anomaly in anomalies
        )

        if critical_anomaly:
            findings.append(
                "Critical perception anomaly detected."
            )

        if not sensor_data_reliable:
            findings.append(
                "Sensor data cannot be considered reliable."
            )

        if (
            critical_anomaly
            or human_within_distance is True
        ):
            return PerceptionSafetyAssessment(
                safety_status=(
                    SceneSafetyStatus.HAZARD_DETECTED
                ),
                risk_level=PerceptionRiskLevel.CRITICAL,
                human_detected=human_detected,
                human_within_safety_distance=(
                    human_within_distance
                ),
                critical_anomaly_detected=(
                    critical_anomaly
                ),
                sensor_data_reliable=(
                    sensor_data_reliable
                ),
                requires_human_review=True,
                findings=tuple(findings),
            )

        if (
            high_anomaly
            or human_detected
        ):
            return PerceptionSafetyAssessment(
                safety_status=(
                    SceneSafetyStatus.POTENTIAL_HAZARD
                ),
                risk_level=PerceptionRiskLevel.HIGH,
                human_detected=human_detected,
                human_within_safety_distance=(
                    human_within_distance
                ),
                critical_anomaly_detected=(
                    critical_anomaly
                ),
                sensor_data_reliable=(
                    sensor_data_reliable
                ),
                requires_human_review=True,
                findings=tuple(findings),
            )

        if not sensor_data_reliable:
            return PerceptionSafetyAssessment(
                safety_status=SceneSafetyStatus.UNKNOWN,
                risk_level=PerceptionRiskLevel.UNKNOWN,
                human_detected=human_detected,
                human_within_safety_distance=(
                    human_within_distance
                ),
                critical_anomaly_detected=False,
                sensor_data_reliable=False,
                requires_human_review=True,
                findings=tuple(findings),
            )

        return PerceptionSafetyAssessment(
            safety_status=(
                SceneSafetyStatus.NO_HAZARD_DETECTED
            ),
            risk_level=PerceptionRiskLevel.LOW,
            human_detected=False,
            human_within_safety_distance=False,
            critical_anomaly_detected=False,
            sensor_data_reliable=True,
            requires_human_review=False,
            findings=tuple(findings),
        )

    # --------------------------------------------------------
    # Full Perception Pipeline
    # --------------------------------------------------------

    async def analyze_scene(
        self,
        *,
        scope: PerceptionScope,
        request: PerceptionRequest,
    ) -> PerceptionResult:

        if request.tenant_id != scope.tenant_id:
            raise PerceptionPermissionError(
                "Tenant mismatch"
            )

        await self._authorize(
            scope=scope,
            robot_id=request.robot_id,
        )

        started_at = utc_now()

        reasoning: list[str] = []

        sensor_reliable, sensor_findings = (
            self.evaluate_sensor_health(
                request=request
            )
        )

        detections = await self.detect_objects(
            scope=scope,
            request=request,
        )

        reasoning.append(
            f"Detected {len(detections)} objects."
        )

        if (
            request.depth_frames
            and self.depth_provider is not None
            and detections
        ):
            detections = (
                await self.depth_provider.estimate_positions(
                    tenant_id=scope.tenant_id,
                    robot_id=request.robot_id,
                    detections=detections,
                    depth_frames=request.depth_frames,
                )
            )

            reasoning.append(
                "Depth-based object localization completed."
            )

        tracks: list[ObjectTrack] = []

        if request.enable_tracking:
            tracks = await self.track_objects(
                scope=scope,
                robot_id=request.robot_id,
                detections=detections,
            )

            reasoning.append(
                f"Generated {len(tracks)} object tracks."
            )

        segmentations: list[SegmentationResult] = []

        if request.enable_segmentation:
            segmentations = await self.segment_images(
                scope=scope,
                request=request,
            )

            reasoning.append(
                f"Generated {len(segmentations)} "
                "segmentation results."
            )

        obstacles = await self.analyze_point_clouds(
            scope=scope,
            request=request,
        )

        reasoning.append(
            f"Detected {len(obstacles)} "
            "point-cloud obstacles."
        )

        if request.enable_sensor_fusion:
            if self.fusion_provider is None:
                raise PerceptionServiceUnavailableError(
                    "Sensor fusion provider is unavailable"
                )

            detections, obstacles = (
                await self.fusion_provider.fuse(
                    tenant_id=scope.tenant_id,
                    robot_id=request.robot_id,
                    detections=detections,
                    obstacles=obstacles,
                )
            )

            reasoning.append(
                "Multimodal sensor fusion completed."
            )

        anomalies: list[PerceptionAnomaly] = []

        if request.enable_anomaly_detection:
            if self.anomaly_provider is None:
                raise PerceptionServiceUnavailableError(
                    "Anomaly provider is unavailable"
                )

            anomalies = (
                await self.anomaly_provider.detect_anomalies(
                    tenant_id=scope.tenant_id,
                    robot_id=request.robot_id,
                    detections=detections,
                    obstacles=obstacles,
                )
            )

            reasoning.append(
                f"Detected {len(anomalies)} anomalies."
            )

        if len(detections) > MAX_DETECTIONS:
            raise PerceptionValidationError(
                "Detection limit exceeded after fusion"
            )

        if len(obstacles) > MAX_DETECTIONS:
            raise PerceptionValidationError(
                "Obstacle limit exceeded"
            )

        safety = self.assess_scene_safety(
            request=request,
            detections=detections,
            obstacles=obstacles,
            anomalies=anomalies,
            sensor_data_reliable=sensor_reliable,
            sensor_findings=sensor_findings,
        )

        if safety.requires_human_review:
            status = PerceptionStatus.REVIEW_REQUIRED
        else:
            status = PerceptionStatus.COMPLETED

        finished_at = utc_now()

        duration = (
            finished_at - started_at
        ).total_seconds()

        result = PerceptionResult(
            result_id=generate_id(),
            tenant_id=scope.tenant_id,
            robot_id=request.robot_id,
            task_type=request.task_type,
            status=status,
            detections=detections,
            tracks=tracks,
            segmentations=segmentations,
            obstacles=obstacles,
            anomalies=anomalies,
            safety_assessment=safety,
            sensor_count=(
                len(request.image_frames)
                + len(request.depth_frames)
                + len(request.point_cloud_frames)
            ),
            processing_duration_seconds=duration,
            reasoning=reasoning + list(
                safety.findings
            ),
        )

        if self.repository is not None:
            await self.repository.save_result(
                result=result
            )

        if self.audit_sink is not None:
            await self.audit_sink.record(
                tenant_id=scope.tenant_id,
                actor_id=scope.user_id,
                action="perception_agent.scene_analyzed",
                resource_id=result.result_id,
                metadata={
                    "robot_id": request.robot_id,
                    "status": result.status.value,
                    "detection_count": len(detections),
                    "obstacle_count": len(obstacles),
                    "anomaly_count": len(anomalies),
                    "risk_level": (
                        safety.risk_level.value
                    ),
                },
            )

        return result

    # --------------------------------------------------------
    # Batch Perception
    # --------------------------------------------------------

    async def analyze_batch(
        self,
        *,
        scope: PerceptionScope,
        requests: Sequence[PerceptionRequest],
    ) -> list[PerceptionResult]:

        if len(requests) > MAX_BATCH_REQUESTS:
            raise PerceptionValidationError(
                "Batch request limit exceeded"
            )

        semaphore = asyncio.Semaphore(4)

        async def process(
            request: PerceptionRequest,
        ) -> PerceptionResult:

            async with semaphore:
                return await self.analyze_scene(
                    scope=scope,
                    request=request,
                )

        return list(
            await asyncio.gather(
                *(process(request) for request in requests)
            )
        )

    # --------------------------------------------------------
    # Perception Report
    # --------------------------------------------------------

    @staticmethod
    def generate_report(
        result: PerceptionResult,
    ) -> dict[str, Any]:

        return {
            "result_id": result.result_id,
            "tenant_id": result.tenant_id,
            "robot_id": result.robot_id,
            "task_type": result.task_type.value,
            "status": result.status.value,
            "created_at": result.created_at.isoformat(),
            "processing_duration_seconds": (
                result.processing_duration_seconds
            ),
            "sensor_count": result.sensor_count,
            "detection_count": len(result.detections),
            "tracking_count": len(result.tracks),
            "segmentation_count": len(
                result.segmentations
            ),
            "obstacle_count": len(result.obstacles),
            "anomaly_count": len(result.anomalies),
            "detections": [
                {
                    "detection_id": detection.detection_id,
                    "label": detection.label,
                    "category": detection.category.value,
                    "confidence": detection.confidence,
                    "source_sensor_id": (
                        detection.source_sensor_id
                    ),
                    "bounding_box": {
                        "x_min": detection.bounding_box.x_min,
                        "y_min": detection.bounding_box.y_min,
                        "x_max": detection.bounding_box.x_max,
                        "y_max": detection.bounding_box.y_max,
                    },
                    "position_3d": (
                        {
                            "x": detection.position_3d.x,
                            "y": detection.position_3d.y,
                            "z": detection.position_3d.z,
                            "frame_id": (
                                detection.position_frame_id
                            ),
                        }
                        if detection.position_3d is not None
                        else None
                    ),
                }
                for detection in result.detections
            ],
            "tracks": [
                {
                    "track_id": track.track_id,
                    "label": track.label,
                    "category": track.category.value,
                    "status": track.status.value,
                    "confidence": track.confidence,
                    "last_seen": track.last_seen.isoformat(),
                }
                for track in result.tracks
            ],
            "obstacles": [
                {
                    "obstacle_id": obstacle.obstacle_id,
                    "category": obstacle.category.value,
                    "position": {
                        "x": obstacle.position.x,
                        "y": obstacle.position.y,
                        "z": obstacle.position.z,
                    },
                    "frame_id": obstacle.frame_id,
                    "radius_m": obstacle.radius_m,
                    "confidence": obstacle.confidence,
                    "dynamic": obstacle.dynamic,
                }
                for obstacle in result.obstacles
            ],
            "anomalies": [
                {
                    "anomaly_id": anomaly.anomaly_id,
                    "anomaly_type": anomaly.anomaly_type,
                    "description": anomaly.description,
                    "confidence": anomaly.confidence,
                    "risk_level": anomaly.risk_level.value,
                }
                for anomaly in result.anomalies
            ],
            "safety": {
                "safety_status": (
                    result.safety_assessment
                    .safety_status.value
                ),
                "risk_level": (
                    result.safety_assessment
                    .risk_level.value
                ),
                "human_detected": (
                    result.safety_assessment
                    .human_detected
                ),
                "human_within_safety_distance": (
                    result.safety_assessment
                    .human_within_safety_distance
                ),
                "critical_anomaly_detected": (
                    result.safety_assessment
                    .critical_anomaly_detected
                ),
                "sensor_data_reliable": (
                    result.safety_assessment
                    .sensor_data_reliable
                ),
                "requires_human_review": (
                    result.safety_assessment
                    .requires_human_review
                ),
                "findings": list(
                    result.safety_assessment.findings
                ),
            },
            "reasoning": result.reasoning,
            "physical_actuation_enabled": False,
            "safety_certification": False,
        }


# ============================================================
# Factory
# ============================================================

def create_perception_agent(
    *,
    detection_provider: ObjectDetectionProvider,
    authorizer: PerceptionAuthorizer,
    tracking_provider: (
        ObjectTrackingProvider | None
    ) = None,
    segmentation_provider: (
        ImageSegmentationProvider | None
    ) = None,
    depth_provider: (
        DepthEstimationProvider | None
    ) = None,
    point_cloud_provider: (
        PointCloudAnalysisProvider | None
    ) = None,
    fusion_provider: (
        SensorFusionProvider | None
    ) = None,
    anomaly_provider: (
        PerceptionAnomalyProvider | None
    ) = None,
    repository: PerceptionRepository | None = None,
    audit_sink: PerceptionAuditSink | None = None,
    stale_sensor_seconds: float = (
        DEFAULT_SENSOR_STALE_SECONDS
    ),
) -> PerceptionAgent:

    return PerceptionAgent(
        detection_provider=detection_provider,
        authorizer=authorizer,
        tracking_provider=tracking_provider,
        segmentation_provider=segmentation_provider,
        depth_provider=depth_provider,
        point_cloud_provider=point_cloud_provider,
        fusion_provider=fusion_provider,
        anomaly_provider=anomaly_provider,
        repository=repository,
        audit_sink=audit_sink,
        stale_sensor_seconds=stale_sensor_seconds,
    )


# ============================================================
# Exports
# ============================================================

__all__ = [
    "PerceptionAgentError",
    "PerceptionValidationError",
    "PerceptionNotFoundError",
    "PerceptionPermissionError",
    "PerceptionModelError",
    "PerceptionSensorError",
    "PerceptionFusionError",
    "PerceptionServiceUnavailableError",
    "SensorType",
    "SensorHealthStatus",
    "PerceptionTaskType",
    "ObjectCategory",
    "TrackingStatus",
    "PerceptionRiskLevel",
    "PerceptionStatus",
    "SceneSafetyStatus",
    "PerceptionScope",
    "Vector3",
    "BoundingBox2D",
    "BoundingBox3D",
    "SensorMetadata",
    "ImageFrame",
    "DepthFrame",
    "PointCloudFrame",
    "ObjectDetection",
    "ObjectTrack",
    "SegmentationResult",
    "PerceptionAnomaly",
    "PerceivedObstacle",
    "PerceptionRequest",
    "PerceptionSafetyAssessment",
    "PerceptionResult",
    "ObjectDetectionProvider",
    "ObjectTrackingProvider",
    "ImageSegmentationProvider",
    "DepthEstimationProvider",
    "PointCloudAnalysisProvider",
    "SensorFusionProvider",
    "PerceptionAnomalyProvider",
    "PerceptionAuthorizer",
    "PerceptionRepository",
    "PerceptionAuditSink",
    "PerceptionAgent",
    "create_perception_agent",
]
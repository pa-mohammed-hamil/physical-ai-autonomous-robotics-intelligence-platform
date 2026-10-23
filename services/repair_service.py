"""Repair Service module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
    backend/app/services/perception_service.py

Robotics Perception Intelligence Service

Python 3.11+
Standard library only.

Features:
    - Multi-robot perception
    - Camera frame ingestion
    - Detection adapter interface
    - Bounding-box validation
    - Multi-object tracking
    - Depth-assisted obstacle localization
    - LiDAR obstacle ingestion
    - Camera/LiDAR observation fusion
    - Robot-frame obstacle coordinates
    - Collision proximity warnings
    - Perception quality metrics
    - Detection confidence filtering
    - Object tracking history
    - Camera frame metadata
    - Fleet perception dashboard
    - JSON export

IMPORTANT:
    This is a perception orchestration service.

    Actual image decoding, neural-network object
    detection, segmentation, and visual embeddings
    require separately installed model adapters.

    All coordinates are in the robot's local 2D frame.
    This module does not command robot hardware.
"""

from __future__ import annotations

import json
import math
import threading
import uuid

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class PerceptionServiceError(Exception):
    pass


class PerceptionValidationError(PerceptionServiceError):
    pass


class PerceptionNotFoundError(PerceptionServiceError):
    pass


class PerceptionStateError(PerceptionServiceError):
    pass


class PerceptionInferenceError(PerceptionServiceError):
    pass


# ============================================================
# ENUMS
# ============================================================

class SensorType(str, Enum):
    RGB_CAMERA = "rgb_camera"
    DEPTH_CAMERA = "depth_camera"
    STEREO_CAMERA = "stereo_camera"
    LIDAR = "lidar"
    RADAR = "radar"
    ULTRASONIC = "ultrasonic"


class ObjectCategory(str, Enum):
    PERSON = "person"
    ROBOT = "robot"
    VEHICLE = "vehicle"
    PALLET = "pallet"
    BOX = "box"
    MACHINE = "machine"
    TOOL = "tool"
    PART = "part"
    ANIMAL = "animal"
    OBSTACLE = "obstacle"
    UNKNOWN = "unknown"


class TrackStatus(str, Enum):
    ACTIVE = "active"
    LOST = "lost"


class PerceptionSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite(
    value: float,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise PerceptionValidationError(
            f"{name} must be finite"
        )

    return float(value)


def bounded(
    value: float,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    value = finite(value, name)

    if not minimum <= value <= maximum:
        raise PerceptionValidationError(
            f"{name} must be in [{minimum}, {maximum}]"
        )

    return value


def require_text(
    value: str,
    name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PerceptionValidationError(
            f"{name} is required"
        )

    return value.strip()


def euclidean_distance(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    return math.hypot(
        x2 - x1,
        y2 - y1,
    )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PerceptionServiceConfig:
    minimum_detection_confidence: float = 0.4

    tracking_match_distance_m: float = 1.0

    maximum_missed_frames: int = 5

    fusion_distance_threshold_m: float = 0.5

    warning_distance_m: float = 2.0

    critical_distance_m: float = 0.8

    maximum_detection_distance_m: float = 100.0

    maximum_frame_history: int = 1000

    maximum_event_history: int = 10000

    allow_hardware_control: bool = False

    def __post_init__(self) -> None:
        bounded(
            self.minimum_detection_confidence,
            "minimum_detection_confidence",
            0.0,
            1.0,
        )

        for name in (
            "tracking_match_distance_m",
            "fusion_distance_threshold_m",
            "warning_distance_m",
            "critical_distance_m",
            "maximum_detection_distance_m",
        ):
            if finite(getattr(self, name), name) <= 0:
                raise PerceptionValidationError(
                    f"{name} must be positive"
                )

        if (
            self.critical_distance_m
            >= self.warning_distance_m
        ):
            raise PerceptionValidationError(
                "Critical distance must be below warning distance"
            )

        if self.maximum_missed_frames < 0:
            raise PerceptionValidationError(
                "maximum_missed_frames cannot be negative"
            )

        if (
            self.maximum_frame_history < 1
            or self.maximum_event_history < 1
        ):
            raise PerceptionValidationError(
                "History limits must be positive"
            )

        if self.allow_hardware_control:
            raise PerceptionValidationError(
                "Hardware control is disabled"
            )


# ============================================================
# GEOMETRY
# ============================================================

@dataclass(frozen=True)
class Point2D:
    x_m: float
    y_m: float

    def __post_init__(self) -> None:
        finite(self.x_m, "x_m")
        finite(self.y_m, "y_m")

    def to_dict(self) -> dict[str, float]:
        return {
            "x_m": self.x_m,
            "y_m": self.y_m,
        }


@dataclass(frozen=True)
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        for name in (
            "x_min",
            "y_min",
            "x_max",
            "y_max",
        ):
            bounded(
                getattr(self, name),
                name,
                0.0,
                1.0,
            )

        if (
            self.x_min >= self.x_max
            or self.y_min >= self.y_max
        ):
            raise PerceptionValidationError(
                "Invalid normalized bounding box"
            )

    @property
    def center_x(self) -> float:
        return (
            self.x_min + self.x_max
        ) / 2.0

    @property
    def center_y(self) -> float:
        return (
            self.y_min + self.y_max
        ) / 2.0

    def to_dict(self) -> dict[str, float]:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
        }


# ============================================================
# CAMERA CALIBRATION
# ============================================================

@dataclass(frozen=True)
class CameraCalibration:
    horizontal_fov_deg: float = 70.0

    camera_offset_x_m: float = 0.0
    camera_offset_y_m: float = 0.0

    camera_yaw_rad: float = 0.0

    def __post_init__(self) -> None:
        if not (
            0 < self.horizontal_fov_deg < 180
        ):
            raise PerceptionValidationError(
                "Horizontal FOV must be between 0 and 180 degrees"
            )

        for name in (
            "camera_offset_x_m",
            "camera_offset_y_m",
            "camera_yaw_rad",
        ):
            finite(getattr(self, name), name)


# ============================================================
# CAMERA FRAME
# ============================================================

@dataclass(frozen=True)
class CameraFrame:
    frame_id: str
    robot_id: str
    sensor_id: str

    width: int
    height: int

    timestamp_seconds: float

    image_reference: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        require_text(self.frame_id, "frame_id")
        require_text(self.robot_id, "robot_id")
        require_text(self.sensor_id, "sensor_id")

        if (
            isinstance(self.width, bool)
            or isinstance(self.height, bool)
            or not isinstance(self.width, int)
            or not isinstance(self.height, int)
            or self.width <= 0
            or self.height <= 0
        ):
            raise PerceptionValidationError(
                "Frame dimensions must be positive integers"
            )

        finite(
            self.timestamp_seconds,
            "timestamp_seconds",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "robot_id": self.robot_id,
            "sensor_id": self.sensor_id,
            "width": self.width,
            "height": self.height,
            "timestamp_seconds": (
                self.timestamp_seconds
            ),
            "image_reference": self.image_reference,
            "metadata": self.metadata,
        }


# ============================================================
# DETECTION
# ============================================================

@dataclass(frozen=True)
class ObjectDetection:
    detection_id: str
    category: ObjectCategory
    confidence: float

    bounding_box: BoundingBox

    distance_m: float | None = None

    label: str | None = None

    def __post_init__(self) -> None:
        require_text(
            self.detection_id,
            "detection_id",
        )

        bounded(
            self.confidence,
            "confidence",
            0.0,
            1.0,
        )

        if self.distance_m is not None:
            if finite(
                self.distance_m,
                "distance_m",
            ) <= 0:
                raise PerceptionValidationError(
                    "Detection distance must be positive"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "category": self.category.value,
            "confidence": self.confidence,
            "bounding_box": (
                self.bounding_box.to_dict()
            ),
            "distance_m": self.distance_m,
            "label": self.label,
        }


# ============================================================
# DETECTOR INTERFACE
# ============================================================

class ObjectDetector(Protocol):

    def detect(
        self,
        frame: CameraFrame,
    ) -> Sequence[ObjectDetection]:
        ...


class CallbackObjectDetector:
    """
    Wrap an external inference callback.

    Example:
        YOLO / ONNX / TensorFlow / PyTorch
        adapter can implement this interface.

    The callback receives frame metadata and
    returns validated ObjectDetection objects.
    """

    def __init__(
        self,
        callback: Callable[
            [CameraFrame],
            Sequence[ObjectDetection],
        ],
    ) -> None:
        if not callable(callback):
            raise PerceptionValidationError(
                "Detector callback must be callable"
            )

        self.callback = callback

    def detect(
        self,
        frame: CameraFrame,
    ) -> Sequence[ObjectDetection]:
        return self.callback(frame)


# ============================================================
# CAMERA PROJECTION
# ============================================================

class CameraObstacleProjector:

    @staticmethod
    def project(
        detection: ObjectDetection,
        calibration: CameraCalibration,
    ) -> Point2D | None:
        """
        Approximate 2D position from horizontal
        image location and supplied range.

        Robot frame:
            +X = forward
            +Y = left

        Assumes camera optical axis points forward.
        This is not a calibrated 3D reconstruction.
        """

        if detection.distance_m is None:
            return None

        half_fov = math.radians(
            calibration.horizontal_fov_deg
        ) / 2.0

        relative_angle = (
            (0.5 - detection.bounding_box.center_x)
            * 2.0
            * half_fov
        )

        angle = (
            calibration.camera_yaw_rad
            + relative_angle
        )

        return Point2D(
            x_m=(
                calibration.camera_offset_x_m
                + detection.distance_m
                * math.cos(angle)
            ),
            y_m=(
                calibration.camera_offset_y_m
                + detection.distance_m
                * math.sin(angle)
            ),
        )


# ============================================================
# SPATIAL OBSERVATION
# ============================================================

@dataclass(frozen=True)
class SpatialObservation:
    observation_id: str
    robot_id: str
    category: ObjectCategory

    position: Point2D

    confidence: float

    source: SensorType

    timestamp_seconds: float

    detection_id: str | None = None

    def __post_init__(self) -> None:
        bounded(
            self.confidence,
            "confidence",
            0.0,
            1.0,
        )

        finite(
            self.timestamp_seconds,
            "timestamp_seconds",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "robot_id": self.robot_id,
            "category": self.category.value,
            "position": self.position.to_dict(),
            "confidence": self.confidence,
            "source": self.source.value,
            "timestamp_seconds": (
                self.timestamp_seconds
            ),
            "detection_id": self.detection_id,
        }


# ============================================================
# LIDAR OBSERVATION
# ============================================================

@dataclass(frozen=True)
class LidarObservation:
    robot_id: str
    sensor_id: str

    angle_rad: float
    distance_m: float

    timestamp_seconds: float

    confidence: float = 1.0

    def __post_init__(self) -> None:
        require_text(self.robot_id, "robot_id")
        require_text(self.sensor_id, "sensor_id")

        finite(self.angle_rad, "angle_rad")

        if finite(
            self.distance_m,
            "distance_m",
        ) <= 0:
            raise PerceptionValidationError(
                "LiDAR distance must be positive"
            )

        bounded(
            self.confidence,
            "confidence",
            0.0,
            1.0,
        )

        finite(
            self.timestamp_seconds,
            "timestamp_seconds",
        )


# ============================================================
# SENSOR FUSION
# ============================================================

class PerceptionSensorFusion:

    def __init__(
        self,
        config: PerceptionServiceConfig,
    ) -> None:
        self.config = config

    def fuse(
        self,
        observations: Sequence[SpatialObservation],
    ) -> list[SpatialObservation]:
        """
        Greedy proximity-based fusion.

        Merges spatial observations of compatible
        categories within a configured radius.

        UNKNOWN and OBSTACLE can match a
        semantic camera observation.

        This is a simplified fusion heuristic,
        not a probabilistic sensor fusion filter.
        """

        ordered = sorted(
            observations,
            key=lambda item: item.confidence,
            reverse=True,
        )

        clusters: list[list[SpatialObservation]] = []

        for observation in ordered:
            matching_cluster = None

            for cluster in clusters:
                anchor = cluster[0]

                compatible = (
                    anchor.category == observation.category
                    or anchor.category in (
                        ObjectCategory.UNKNOWN,
                        ObjectCategory.OBSTACLE,
                    )
                    or observation.category in (
                        ObjectCategory.UNKNOWN,
                        ObjectCategory.OBSTACLE,
                    )
                )

                if not compatible:
                    continue

                center_x = sum(
                    item.position.x_m
                    for item in cluster
                ) / len(cluster)

                center_y = sum(
                    item.position.y_m
                    for item in cluster
                ) / len(cluster)

                separation = euclidean_distance(
                    observation.position.x_m,
                    observation.position.y_m,
                    center_x,
                    center_y,
                )

                if (
                    separation
                    <= self.config.fusion_distance_threshold_m
                ):
                    matching_cluster = cluster
                    break

            if matching_cluster is None:
                clusters.append([observation])
            else:
                matching_cluster.append(observation)

        fused = []

        for cluster in clusters:
            weight_sum = sum(
                max(item.confidence, 1e-6)
                for item in cluster
            )

            x = sum(
                item.position.x_m
                * max(item.confidence, 1e-6)
                for item in cluster
            ) / weight_sum

            y = sum(
                item.position.y_m
                * max(item.confidence, 1e-6)
                for item in cluster
            ) / weight_sum

            semantic = next(
                (
                    item
                    for item in cluster
                    if item.category not in (
                        ObjectCategory.UNKNOWN,
                        ObjectCategory.OBSTACLE,
                    )
                ),
                cluster[0],
            )

            fused.append(
                SpatialObservation(
                    observation_id=str(uuid.uuid4()),
                    robot_id=semantic.robot_id,
                    category=semantic.category,
                    position=Point2D(x, y),
                    confidence=max(
                        item.confidence
                        for item in cluster
                    ),
                    source=semantic.source,
                    timestamp_seconds=max(
                        item.timestamp_seconds
                        for item in cluster
                    ),
                    detection_id=semantic.detection_id,
                )
            )

        return fused


# ============================================================
# OBJECT TRACK
# ============================================================

@dataclass
class TrackedObject:
    track_id: str
    robot_id: str

    category: ObjectCategory

    position: Point2D

    confidence: float

    first_seen_seconds: float
    last_seen_seconds: float

    observation_count: int = 1

    missed_frames: int = 0

    status: TrackStatus = TrackStatus.ACTIVE

    velocity_x_mps: float = 0.0
    velocity_y_mps: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "robot_id": self.robot_id,
            "category": self.category.value,
            "position": self.position.to_dict(),
            "confidence": self.confidence,
            "first_seen_seconds": (
                self.first_seen_seconds
            ),
            "last_seen_seconds": (
                self.last_seen_seconds
            ),
            "observation_count": (
                self.observation_count
            ),
            "missed_frames": self.missed_frames,
            "status": self.status.value,
            "velocity_x_mps": (
                self.velocity_x_mps
            ),
            "velocity_y_mps": (
                self.velocity_y_mps
            ),
        }


# ============================================================
# MULTI-OBJECT TRACKER
# ============================================================

class MultiObjectTracker:

    def __init__(
        self,
        config: PerceptionServiceConfig,
    ) -> None:
        self.config = config

        self.tracks: dict[
            str,
            TrackedObject
        ] = {}

    def update(
        self,
        robot_id: str,
        observations: Sequence[SpatialObservation],
        timestamp_seconds: float,
    ) -> list[TrackedObject]:
        finite(
            timestamp_seconds,
            "timestamp_seconds",
        )

        active_tracks = [
            track
            for track in self.tracks.values()
            if (
                track.robot_id == robot_id
                and track.status == TrackStatus.ACTIVE
            )
        ]

        pairs = []

        for track in active_tracks:
            for index, observation in enumerate(
                observations
            ):
                if observation.robot_id != robot_id:
                    continue

                if track.category != observation.category:
                    continue

                gap = euclidean_distance(
                    track.position.x_m,
                    track.position.y_m,
                    observation.position.x_m,
                    observation.position.y_m,
                )

                if gap <= self.config.tracking_match_distance_m:
                    pairs.append(
                        (gap, track.track_id, index)
                    )

        pairs.sort()

        matched_tracks = set()
        matched_observations = set()

        for _, track_id, index in pairs:
            if (
                track_id in matched_tracks
                or index in matched_observations
            ):
                continue

            track = self.tracks[track_id]
            observation = observations[index]

            dt = (
                timestamp_seconds
                - track.last_seen_seconds
            )

            if dt > 0:
                track.velocity_x_mps = (
                    observation.position.x_m
                    - track.position.x_m
                ) / dt

                track.velocity_y_mps = (
                    observation.position.y_m
                    - track.position.y_m
                ) / dt

            track.position = observation.position
            track.confidence = observation.confidence
            track.last_seen_seconds = timestamp_seconds

            track.observation_count += 1
            track.missed_frames = 0

            matched_tracks.add(track_id)
            matched_observations.add(index)

        for track in active_tracks:
            if track.track_id in matched_tracks:
                continue

            track.missed_frames += 1

            if (
                track.missed_frames
                > self.config.maximum_missed_frames
            ):
                track.status = TrackStatus.LOST

        for index, observation in enumerate(observations):
            if index in matched_observations:
                continue

            if observation.robot_id != robot_id:
                continue

            track = TrackedObject(
                track_id=str(uuid.uuid4()),
                robot_id=robot_id,
                category=observation.category,
                position=observation.position,
                confidence=observation.confidence,
                first_seen_seconds=timestamp_seconds,
                last_seen_seconds=timestamp_seconds,
            )

            self.tracks[track.track_id] = track

        return [
            track
            for track in self.tracks.values()
            if (
                track.robot_id == robot_id
                and track.status == TrackStatus.ACTIVE
            )
        ]


# ============================================================
# PROXIMITY ALERT
# ============================================================

@dataclass(frozen=True)
class PerceptionAlert:
    alert_id: str
    robot_id: str

    severity: PerceptionSeverity

    category: ObjectCategory

    distance_m: float

    track_id: str

    message: str

    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "robot_id": self.robot_id,
            "severity": self.severity.value,
            "category": self.category.value,
            "distance_m": self.distance_m,
            "track_id": self.track_id,
            "message": self.message,
            "created_at": self.created_at,
        }


class PerceptionSafetyAnalyzer:

    def __init__(
        self,
        config: PerceptionServiceConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        tracks: Sequence[TrackedObject],
    ) -> list[PerceptionAlert]:
        alerts = []

        for track in tracks:
            if track.status != TrackStatus.ACTIVE:
                continue

            distance_m = math.hypot(
                track.position.x_m,
                track.position.y_m,
            )

            if distance_m <= self.config.critical_distance_m:
                severity = PerceptionSeverity.CRITICAL

            elif distance_m <= self.config.warning_distance_m:
                severity = PerceptionSeverity.WARNING

            else:
                continue

            alerts.append(
                PerceptionAlert(
                    alert_id=str(uuid.uuid4()),
                    robot_id=track.robot_id,
                    severity=severity,
                    category=track.category,
                    distance_m=distance_m,
                    track_id=track.track_id,
                    message=(
                        f"{track.category.value} detected "
                        f"at {distance_m:.2f} m"
                    ),
                    created_at=utc_now(),
                )
            )

        return alerts


# ============================================================
# PERCEPTION RESULT
# ============================================================

@dataclass
class PerceptionResult:
    result_id: str
    robot_id: str

    timestamp_seconds: float

    detections: list[ObjectDetection]

    observations: list[SpatialObservation]

    tracks: list[TrackedObject]

    alerts: list[PerceptionAlert]

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "robot_id": self.robot_id,
            "timestamp_seconds": (
                self.timestamp_seconds
            ),
            "detections": [
                item.to_dict()
                for item in self.detections
            ],
            "observations": [
                item.to_dict()
                for item in self.observations
            ],
            "tracks": [
                item.to_dict()
                for item in self.tracks
            ],
            "alerts": [
                item.to_dict()
                for item in self.alerts
            ],
        }


# ============================================================
# PERCEPTION SERVICE
# ============================================================

class RoboticsPerceptionService:

    def __init__(
        self,
        config: PerceptionServiceConfig | None = None,
    ) -> None:
        self.config = (
            config or PerceptionServiceConfig()
        )

        self.detectors: dict[
            str,
            ObjectDetector
        ] = {}

        self.calibrations: dict[
            str,
            CameraCalibration
        ] = {}

        self.fusion = PerceptionSensorFusion(
            self.config
        )

        self.tracker = MultiObjectTracker(
            self.config
        )

        self.safety = PerceptionSafetyAnalyzer(
            self.config
        )

        self.results: deque[
            PerceptionResult
        ] = deque(
            maxlen=self.config.maximum_frame_history
        )

        self.alert_history: deque[
            PerceptionAlert
        ] = deque(
            maxlen=self.config.maximum_event_history
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # REGISTER DETECTOR
    # --------------------------------------------------------

    def register_detector(
        self,
        sensor_id: str,
        detector: ObjectDetector,
        calibration: CameraCalibration | None = None,
    ) -> None:
        require_text(
            sensor_id,
            "sensor_id",
        )

        if not callable(
            getattr(detector, "detect", None)
        ):
            raise PerceptionValidationError(
                "Detector must implement detect(frame)"
            )

        with self._lock:
            self.detectors[sensor_id] = detector

            self.calibrations[sensor_id] = (
                calibration or CameraCalibration()
            )

    # --------------------------------------------------------
    # CAMERA DETECTIONS
    # --------------------------------------------------------

    def detect(
        self,
        frame: CameraFrame,
    ) -> list[ObjectDetection]:
        detector = self.detectors.get(
            frame.sensor_id
        )

        if detector is None:
            raise PerceptionNotFoundError(
                f"No detector for sensor {frame.sensor_id}"
            )

        try:
            detections = list(
                detector.detect(frame)
            )
        except Exception as exc:
            raise PerceptionInferenceError(
                f"Detection failed: {exc}"
            ) from exc

        for item in detections:
            if not isinstance(item, ObjectDetection):
                raise PerceptionInferenceError(
                    "Detector returned an invalid detection"
                )

        return [
            item
            for item in detections
            if (
                item.confidence
                >= self.config.minimum_detection_confidence
            )
        ]

    # --------------------------------------------------------
    # CAMERA TO SPATIAL OBSERVATIONS
    # --------------------------------------------------------

    def camera_observations(
        self,
        frame: CameraFrame,
        detections: Sequence[ObjectDetection],
    ) -> list[SpatialObservation]:
        calibration = self.calibrations.get(
            frame.sensor_id,
            CameraCalibration(),
        )

        observations = []

        for detection in detections:
            if detection.distance_m is None:
                continue

            if (
                detection.distance_m
                > self.config.maximum_detection_distance_m
            ):
                continue

            position = CameraObstacleProjector.project(
                detection,
                calibration,
            )

            if position is None:
                continue

            observations.append(
                SpatialObservation(
                    observation_id=str(uuid.uuid4()),
                    robot_id=frame.robot_id,
                    category=detection.category,
                    position=position,
                    confidence=detection.confidence,
                    source=SensorType.DEPTH_CAMERA,
                    timestamp_seconds=(
                        frame.timestamp_seconds
                    ),
                    detection_id=detection.detection_id,
                )
            )

        return observations

    # --------------------------------------------------------
    # LIDAR TO SPATIAL OBSERVATIONS
    # --------------------------------------------------------

    def lidar_observations(
        self,
        readings: Sequence[LidarObservation],
    ) -> list[SpatialObservation]:
        observations = []

        for reading in readings:
            if (
                reading.distance_m
                > self.config.maximum_detection_distance_m
            ):
                continue

            observations.append(
                SpatialObservation(
                    observation_id=str(uuid.uuid4()),
                    robot_id=reading.robot_id,
                    category=ObjectCategory.OBSTACLE,
                    position=Point2D(
                        x_m=(
                            reading.distance_m
                            * math.cos(reading.angle_rad)
                        ),
                        y_m=(
                            reading.distance_m
                            * math.sin(reading.angle_rad)
                        ),
                    ),
                    confidence=reading.confidence,
                    source=SensorType.LIDAR,
                    timestamp_seconds=(
                        reading.timestamp_seconds
                    ),
                )
            )

        return observations

    # --------------------------------------------------------
    # PROCESS SENSOR FRAME
    # --------------------------------------------------------

    def process_frame(
        self,
        frame: CameraFrame,
        *,
        lidar_readings: Sequence[
            LidarObservation
        ] = (),
        supplied_detections: Sequence[
            ObjectDetection
        ] | None = None,
    ) -> PerceptionResult:
        with self._lock:
            if supplied_detections is None:
                detections = self.detect(frame)
            else:
                detections = list(supplied_detections)

                if not all(
                    isinstance(item, ObjectDetection)
                    for item in detections
                ):
                    raise PerceptionValidationError(
                        "Invalid supplied detections"
                    )

                detections = [
                    item
                    for item in detections
                    if item.confidence
                    >= self.config.minimum_detection_confidence
                ]

            for reading in lidar_readings:
                if reading.robot_id != frame.robot_id:
                    raise PerceptionValidationError(
                        "LiDAR reading belongs to another robot"
                    )

            camera_observations = (
                self.camera_observations(
                    frame,
                    detections,
                )
            )

            lidar_observations = (
                self.lidar_observations(
                    lidar_readings
                )
            )

            fused = self.fusion.fuse(
                [
                    *camera_observations,
                    *lidar_observations,
                ]
            )

            tracks = self.tracker.update(
                frame.robot_id,
                fused,
                frame.timestamp_seconds,
            )

            alerts = self.safety.evaluate(
                tracks
            )

            result = PerceptionResult(
                result_id=str(uuid.uuid4()),
                robot_id=frame.robot_id,
                timestamp_seconds=frame.timestamp_seconds,
                detections=detections,
                observations=fused,
                tracks=tracks,
                alerts=alerts,
            )

            self.results.append(
                result
            )

            self.alert_history.extend(
                alerts
            )

            return result

    # --------------------------------------------------------
    # ACTIVE TRACKS
    # --------------------------------------------------------

    def active_tracks(
        self,
        robot_id: str | None = None,
    ) -> list[TrackedObject]:
        with self._lock:
            return [
                track
                for track in self.tracker.tracks.values()
                if (
                    track.status == TrackStatus.ACTIVE
                    and (
                        robot_id is None
                        or track.robot_id == robot_id
                    )
                )
            ]

    # --------------------------------------------------------
    # LATEST RESULT
    # --------------------------------------------------------

    def latest_result(
        self,
        robot_id: str,
    ) -> PerceptionResult | None:
        with self._lock:
            for result in reversed(self.results):
                if result.robot_id == robot_id:
                    return result

        return None

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
    ) -> dict[str, Any]:
        with self._lock:
            results = list(
                self.results
            )

            tracks = list(
                self.tracker.tracks.values()
            )

            alerts = list(
                self.alert_history
            )

        detections = [
            detection
            for result in results
            for detection in result.detections
        ]

        average_confidence = (
            sum(
                detection.confidence
                for detection in detections
            ) / len(detections)
            if detections
            else None
        )

        return {
            "processed_frames": len(results),
            "total_detections": len(detections),
            "average_detection_confidence": (
                average_confidence
            ),
            "active_tracks": sum(
                track.status == TrackStatus.ACTIVE
                for track in tracks
            ),
            "lost_tracks": sum(
                track.status == TrackStatus.LOST
                for track in tracks
            ),
            "detection_categories": dict(
                Counter(
                    detection.category.value
                    for detection in detections
                )
            ),
            "alert_severity_distribution": dict(
                Counter(
                    alert.severity.value
                    for alert in alerts
                )
            ),
            "registered_detectors": len(
                self.detectors
            ),
            "simulation_only": True,
        }

    # --------------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------------

    def dashboard(
        self,
    ) -> dict[str, Any]:
        return {
            "generated_at": utc_now(),
            "statistics": self.statistics(),
            "active_tracks": [
                track.to_dict()
                for track in self.active_tracks()
            ],
            "recent_alerts": [
                alert.to_dict()
                for alert in list(
                    self.alert_history
                )[-100:]
            ],
            "latest_results": [
                result.to_dict()
                for result in list(
                    self.results
                )[-20:]
            ],
        }

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        path: str | Path,
    ) -> Path:
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.write_text(
            json.dumps(
                self.dashboard(),
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        return output


# ============================================================
# EXAMPLE DETECTOR
# ============================================================

class DemoObjectDetector:
    """
    Demonstration detections.

    This is not a trained computer vision model.
    """

    def detect(
        self,
        frame: CameraFrame,
    ) -> Sequence[ObjectDetection]:
        return [
            ObjectDetection(
                detection_id=str(uuid.uuid4()),
                category=ObjectCategory.PERSON,
                confidence=0.96,
                bounding_box=BoundingBox(
                    x_min=0.40,
                    y_min=0.20,
                    x_max=0.58,
                    y_max=0.90,
                ),
                distance_m=1.5,
                label="person",
            ),
            ObjectDetection(
                detection_id=str(uuid.uuid4()),
                category=ObjectCategory.BOX,
                confidence=0.91,
                bounding_box=BoundingBox(
                    x_min=0.70,
                    y_min=0.45,
                    x_max=0.85,
                    y_max=0.75,
                ),
                distance_m=3.0,
                label="box",
            ),
        ]


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    service = RoboticsPerceptionService()

    service.register_detector(
        sensor_id="front_camera",
        detector=DemoObjectDetector(),
        calibration=CameraCalibration(
            horizontal_fov_deg=70.0,
        ),
    )

    frame = CameraFrame(
        frame_id="frame_001",
        robot_id="robot_001",
        sensor_id="front_camera",
        width=1280,
        height=720,
        timestamp_seconds=1.0,
        image_reference="demo://frame_001",
    )

    lidar_readings = [
        LidarObservation(
            robot_id="robot_001",
            sensor_id="front_lidar",
            angle_rad=0.0,
            distance_m=1.45,
            timestamp_seconds=1.0,
        ),
        LidarObservation(
            robot_id="robot_001",
            sensor_id="front_lidar",
            angle_rad=0.4,
            distance_m=3.1,
            timestamp_seconds=1.0,
        ),
    ]

    result = service.process_frame(
        frame,
        lidar_readings=lidar_readings,
    )

    print(
        json.dumps(
            {
                "result": result.to_dict(),
                "statistics": service.statistics(),
            },
            indent=2,
        )
    )

    service.export_json(
        "data/perception/perception_report.json"
    )


if __name__ == "__main__":
    main()

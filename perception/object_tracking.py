"""Object Tracking module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/perception/object_tracking.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================

Module: object_tracking.py

Real-Time Multi-Object Tracking Engine

Features
--------
- Multi-object detection association
- Persistent object tracking IDs
- Bounding-box IoU matching
- Hungarian assignment (optional SciPy)
- Greedy assignment fallback
- Constant-velocity Kalman filtering
- Object trajectory history
- Velocity and acceleration estimation
- Track lifecycle management
- Lost-object detection
- Track re-identification within a configurable window
- Stationary and moving object classification
- Human, robot, vehicle, and obstacle tracking
- Camera-based multi-object tracking
- Multi-camera tracker isolation
- Advisory proximity-risk estimation
- Tenant-aware access control
- Processing statistics
- Audit logging
- JSON export
- Synthetic tracking demonstration
- Unit tests

Dependencies
------------
Required:
    numpy

Optional:
    scipy

Installation:
    pip install numpy scipy

Coordinate conventions
----------------------
Image coordinates:
    x increases right
    y increases down

Tracking state:
    [cx, cy, vx, vy]

Velocity:
    pixels per second

Acceleration:
    pixels per second squared

Important
---------
Tracking results are advisory perception outputs.
They must not directly control physical robot actuators.

Actual collision avoidance requires independently validated
robot safety controllers and calibrated sensing systems.
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
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


# ============================================================
# EXCEPTIONS
# ============================================================

class ObjectTrackingError(Exception):
    """Base tracking exception."""


class TrackingValidationError(ObjectTrackingError):
    """Invalid tracking data or configuration."""


class TrackingAuthorizationError(ObjectTrackingError):
    """Tracking access denied."""


class TrackingCapacityError(ObjectTrackingError):
    """Tracking resource limit exceeded."""


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
        raise TrackingValidationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > 128:
        raise TrackingValidationError(
            f"{name} must contain 1-128 characters"
        )

    return value


def finite_float(
    value: Any,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise TrackingValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TrackingValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise TrackingValidationError(
            f"{name} must be finite"
        )

    return result


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (tuple, list, deque)):
        return [json_safe(item) for item in value]

    if hasattr(value, "__dataclass_fields__"):
        return {
            name: json_safe(getattr(value, name))
            for name in value.__dataclass_fields__
        }

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


# ============================================================
# ENUMERATIONS
# ============================================================

class TrackStatus(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    REMOVED = "removed"


class MotionStatus(str, Enum):
    STATIONARY = "stationary"
    MOVING = "moving"
    UNKNOWN = "unknown"


class TrackingRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class TrackingAlgorithm(str, Enum):
    IOU_KALMAN = "iou_kalman"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ObjectTrackingConfig:
    minimum_confidence: float = 0.30

    iou_match_threshold: float = 0.20

    minimum_confirmed_hits: int = 3
    maximum_missed_frames: int = 8

    maximum_tracks: int = 200
    maximum_detections_per_frame: int = 300

    trajectory_history_size: int = 100
    frame_history_size: int = 50

    process_noise: float = 5.0
    measurement_noise: float = 10.0

    maximum_prediction_dt_s: float = 2.0

    stationary_speed_threshold_px_s: float = 5.0

    enable_class_aware_matching: bool = True
    enable_reidentification: bool = True

    reidentification_window_frames: int = 15
    reidentification_iou_threshold: float = 0.30

    critical_distance_m: float = 0.5
    high_risk_distance_m: float = 1.5
    moderate_risk_distance_m: float = 3.0

    def __post_init__(self) -> None:
        for name in (
            "minimum_confidence",
            "iou_match_threshold",
            "reidentification_iou_threshold",
        ):
            value = finite_float(
                getattr(self, name),
                name,
            )

            if not 0 <= value <= 1:
                raise TrackingValidationError(
                    f"{name} must be between 0 and 1"
                )

        for name in (
            "minimum_confirmed_hits",
            "maximum_tracks",
            "maximum_detections_per_frame",
            "trajectory_history_size",
            "frame_history_size",
            "reidentification_window_frames",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise TrackingValidationError(
                    f"{name} must be a positive integer"
                )

        if self.maximum_missed_frames < 0:
            raise TrackingValidationError(
                "maximum_missed_frames cannot be negative"
            )

        for name in (
            "process_noise",
            "measurement_noise",
            "maximum_prediction_dt_s",
            "stationary_speed_threshold_px_s",
        ):
            if finite_float(
                getattr(self, name),
                name,
            ) <= 0:
                raise TrackingValidationError(
                    f"{name} must be positive"
                )

        if not (
            0 < self.critical_distance_m
            < self.high_risk_distance_m
            < self.moderate_risk_distance_m
        ):
            raise TrackingValidationError(
                "Risk thresholds must be strictly increasing"
            )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class TrackingAccessContext:
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
# BOUNDING BOX
# ============================================================

@dataclass(frozen=True)
class TrackingBoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        for name in ("x1", "y1", "x2", "y2"):
            finite_float(
                getattr(self, name),
                name,
            )

        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise TrackingValidationError(
                "Bounding box has invalid dimensions"
            )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.x1 + self.x2) / 2,
            (self.y1 + self.y2) / 2,
        )

    @classmethod
    def from_center(
        cls,
        cx: float,
        cy: float,
        width: float,
        height: float,
    ) -> TrackingBoundingBox:
        return cls(
            cx - width / 2,
            cy - height / 2,
            cx + width / 2,
            cy + height / 2,
        )

    def iou(
        self,
        other: TrackingBoundingBox,
    ) -> float:
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)

        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)

        intersection_width = max(
            0.0,
            x2 - x1,
        )

        intersection_height = max(
            0.0,
            y2 - y1,
        )

        intersection = (
            intersection_width * intersection_height
        )

        union = (
            self.area
            + other.area
            - intersection
        )

        return (
            intersection / union
            if union > 0
            else 0.0
        )


# ============================================================
# DETECTION INPUT
# ============================================================

@dataclass(frozen=True)
class TrackingDetection:
    detection_id: str

    class_id: int
    class_name: str

    confidence: float

    bounding_box: TrackingBoundingBox

    distance_m: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_identifier(
            self.detection_id,
            "detection_id",
        )

        if self.class_id < 0:
            raise TrackingValidationError(
                "class_id must be nonnegative"
            )

        if not 0 <= self.confidence <= 1:
            raise TrackingValidationError(
                "Confidence must be in [0, 1]"
            )

        if self.distance_m is not None:
            if finite_float(
                self.distance_m,
                "distance_m",
            ) <= 0:
                raise TrackingValidationError(
                    "Distance must be positive"
                )


# ============================================================
# TRAJECTORY POINT
# ============================================================

@dataclass(frozen=True)
class TrajectoryPoint:
    x: float
    y: float

    timestamp: datetime

    velocity_x: float
    velocity_y: float

    speed_px_s: float


# ============================================================
# CONSTANT-VELOCITY KALMAN FILTER
# ============================================================

class KalmanBoxFilter:
    """
    State:
        [center_x, center_y, velocity_x, velocity_y]

    Measurement:
        [center_x, center_y]

    Bounding-box width and height are maintained separately.
    """

    def __init__(
        self,
        bounding_box: TrackingBoundingBox,
        *,
        process_noise: float = 5.0,
        measurement_noise: float = 10.0,
    ) -> None:
        cx, cy = bounding_box.center

        self.state = np.array(
            [
                [cx],
                [cy],
                [0.0],
                [0.0],
            ],
            dtype=np.float64,
        )

        self.covariance = np.diag(
            [25.0, 25.0, 100.0, 100.0]
        )

        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        self.width = bounding_box.width
        self.height = bounding_box.height

        self.measurement_matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        self.identity = np.eye(
            4,
            dtype=np.float64,
        )

    def predict(
        self,
        dt: float,
    ) -> TrackingBoundingBox:
        dt = finite_float(dt, "dt")

        if dt < 0:
            raise TrackingValidationError(
                "Prediction time cannot be negative"
            )

        transition = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2

        q = self.process_noise

        process_covariance = q * np.array(
            [
                [dt4 / 4, 0, dt3 / 2, 0],
                [0, dt4 / 4, 0, dt3 / 2],
                [dt3 / 2, 0, dt2, 0],
                [0, dt3 / 2, 0, dt2],
            ],
            dtype=np.float64,
        )

        self.state = transition @ self.state

        self.covariance = (
            transition
            @ self.covariance
            @ transition.T
            + process_covariance
        )

        return self.bounding_box

    def update(
        self,
        bounding_box: TrackingBoundingBox,
    ) -> TrackingBoundingBox:
        cx, cy = bounding_box.center

        measurement = np.array(
            [
                [cx],
                [cy],
            ],
            dtype=np.float64,
        )

        observation_covariance = (
            np.eye(2)
            * self.measurement_noise
        )

        residual = (
            measurement
            - self.measurement_matrix @ self.state
        )

        innovation_covariance = (
            self.measurement_matrix
            @ self.covariance
            @ self.measurement_matrix.T
            + observation_covariance
        )

        gain = (
            self.covariance
            @ self.measurement_matrix.T
            @ np.linalg.inv(innovation_covariance)
        )

        self.state = (
            self.state + gain @ residual
        )

        correction = (
            self.identity
            - gain @ self.measurement_matrix
        )

        # Joseph form improves numerical stability.
        self.covariance = (
            correction
            @ self.covariance
            @ correction.T
            + gain
            @ observation_covariance
            @ gain.T
        )

        self.width = bounding_box.width
        self.height = bounding_box.height

        return self.bounding_box

    @property
    def bounding_box(self) -> TrackingBoundingBox:
        return TrackingBoundingBox.from_center(
            cx=float(self.state[0, 0]),
            cy=float(self.state[1, 0]),
            width=self.width,
            height=self.height,
        )

    @property
    def position(self) -> tuple[float, float]:
        return (
            float(self.state[0, 0]),
            float(self.state[1, 0]),
        )

    @property
    def velocity(self) -> tuple[float, float]:
        return (
            float(self.state[2, 0]),
            float(self.state[3, 0]),
        )

    @property
    def speed(self) -> float:
        vx, vy = self.velocity

        return math.hypot(vx, vy)


# ============================================================
# TRACK STATE
# ============================================================

@dataclass
class TrackedObject:
    track_id: str

    class_id: int
    class_name: str

    bounding_box: TrackingBoundingBox

    confidence: float

    status: TrackStatus = TrackStatus.TENTATIVE

    hits: int = 1
    age_frames: int = 1
    missed_frames: int = 0

    velocity_x: float = 0.0
    velocity_y: float = 0.0

    acceleration_x: float = 0.0
    acceleration_y: float = 0.0

    distance_m: float | None = None

    motion_status: MotionStatus = MotionStatus.UNKNOWN

    risk_level: TrackingRiskLevel = (
        TrackingRiskLevel.UNKNOWN
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    last_seen: datetime = field(
        default_factory=utc_now
    )

    trajectory: deque[TrajectoryPoint] = field(
        default_factory=deque,
        repr=False,
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def center(self) -> tuple[float, float]:
        return self.bounding_box.center

    @property
    def speed_px_s(self) -> float:
        return math.hypot(
            self.velocity_x,
            self.velocity_y,
        )

    @property
    def acceleration_px_s2(self) -> float:
        return math.hypot(
            self.acceleration_x,
            self.acceleration_y,
        )

    def snapshot(self) -> dict[str, Any]:
        return json_safe(self)


# ============================================================
# TRACK ASSOCIATION
# ============================================================

@dataclass(frozen=True)
class AssociationResult:
    matches: tuple[tuple[int, int], ...]

    unmatched_tracks: tuple[int, ...]
    unmatched_detections: tuple[int, ...]


class TrackAssociationEngine:

    def __init__(
        self,
        config: ObjectTrackingConfig,
    ) -> None:
        self.config = config

    def associate(
        self,
        tracks: Sequence[TrackedObject],
        detections: Sequence[TrackingDetection],
    ) -> AssociationResult:
        if not tracks:
            return AssociationResult(
                matches=(),
                unmatched_tracks=(),
                unmatched_detections=tuple(
                    range(len(detections))
                ),
            )

        if not detections:
            return AssociationResult(
                matches=(),
                unmatched_tracks=tuple(
                    range(len(tracks))
                ),
                unmatched_detections=(),
            )

        cost_matrix = np.full(
            (len(tracks), len(detections)),
            1e6,
            dtype=np.float64,
        )

        for track_index, track in enumerate(tracks):
            for detection_index, detection in enumerate(
                detections
            ):
                if (
                    self.config.enable_class_aware_matching
                    and track.class_id != detection.class_id
                ):
                    continue

                overlap = track.bounding_box.iou(
                    detection.bounding_box
                )

                if overlap < self.config.iou_match_threshold:
                    continue

                cost_matrix[
                    track_index,
                    detection_index,
                ] = 1.0 - overlap

        if linear_sum_assignment is not None:
            row_indices, column_indices = (
                linear_sum_assignment(cost_matrix)
            )

            candidates = list(
                zip(
                    row_indices.tolist(),
                    column_indices.tolist(),
                )
            )

        else:
            candidates = self._greedy_assignment(
                cost_matrix
            )

        matches = []

        matched_tracks = set()
        matched_detections = set()

        for track_index, detection_index in candidates:
            if (
                cost_matrix[
                    track_index,
                    detection_index,
                ] >= 1e6
            ):
                continue

            matches.append(
                (track_index, detection_index)
            )

            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

        return AssociationResult(
            matches=tuple(matches),
            unmatched_tracks=tuple(
                index
                for index in range(len(tracks))
                if index not in matched_tracks
            ),
            unmatched_detections=tuple(
                index
                for index in range(len(detections))
                if index not in matched_detections
            ),
        )

    @staticmethod
    def _greedy_assignment(
        cost_matrix: np.ndarray,
    ) -> list[tuple[int, int]]:
        candidates = []

        for row in range(cost_matrix.shape[0]):
            for column in range(cost_matrix.shape[1]):
                if cost_matrix[row, column] < 1e6:
                    candidates.append(
                        (
                            float(cost_matrix[row, column]),
                            row,
                            column,
                        )
                    )

        candidates.sort(
            key=lambda item: item[0]
        )

        used_rows = set()
        used_columns = set()

        selected = []

        for _, row, column in candidates:
            if row in used_rows or column in used_columns:
                continue

            selected.append(
                (row, column)
            )

            used_rows.add(row)
            used_columns.add(column)

        return selected


# ============================================================
# MOTION ANALYSIS
# ============================================================

class ObjectMotionAnalyzer:

    def __init__(
        self,
        config: ObjectTrackingConfig,
    ) -> None:
        self.config = config

    def classify(
        self,
        track: TrackedObject,
    ) -> MotionStatus:
        if track.hits < 2:
            return MotionStatus.UNKNOWN

        if (
            track.speed_px_s
            < self.config.stationary_speed_threshold_px_s
        ):
            return MotionStatus.STATIONARY

        return MotionStatus.MOVING


# ============================================================
# PROXIMITY RISK ASSESSMENT
# ============================================================

class TrackingRiskAssessor:

    def __init__(
        self,
        config: ObjectTrackingConfig,
    ) -> None:
        self.config = config

    def assess(
        self,
        distance_m: float | None,
    ) -> TrackingRiskLevel:
        if distance_m is None:
            return TrackingRiskLevel.UNKNOWN

        if distance_m < self.config.critical_distance_m:
            return TrackingRiskLevel.CRITICAL

        if distance_m < self.config.high_risk_distance_m:
            return TrackingRiskLevel.HIGH

        if distance_m < self.config.moderate_risk_distance_m:
            return TrackingRiskLevel.MODERATE

        return TrackingRiskLevel.LOW


# ============================================================
# TRACK MANAGER
# ============================================================

class MultiObjectTracker:

    def __init__(
        self,
        config: ObjectTrackingConfig | None = None,
    ) -> None:
        self.config = (
            config or ObjectTrackingConfig()
        )

        self.association = TrackAssociationEngine(
            self.config
        )

        self.motion_analyzer = ObjectMotionAnalyzer(
            self.config
        )

        self.risk_assessor = TrackingRiskAssessor(
            self.config
        )

        self._tracks: dict[str, TrackedObject] = {}

        self._filters: dict[str, KalmanBoxFilter] = {}

        self._lost_tracks: dict[
            str,
            tuple[TrackedObject, KalmanBoxFilter, int],
        ] = {}

        self._frame_number = 0

        self._last_timestamp: datetime | None = None

        self._lock = threading.RLock()

    def _create_track(
        self,
        detection: TrackingDetection,
        timestamp: datetime,
    ) -> TrackedObject:
        track_id = generate_id()

        status = (
            TrackStatus.CONFIRMED
            if self.config.minimum_confirmed_hits <= 1
            else TrackStatus.TENTATIVE
        )

        track = TrackedObject(
            track_id=track_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            bounding_box=detection.bounding_box,
            confidence=detection.confidence,
            status=status,
            distance_m=detection.distance_m,
            created_at=timestamp,
            last_seen=timestamp,
            trajectory=deque(
                maxlen=self.config.trajectory_history_size
            ),
            metadata={
                **detection.metadata,
                "last_detection_id": detection.detection_id,
            },
        )

        track.risk_level = self.risk_assessor.assess(
            detection.distance_m
        )

        self._tracks[track_id] = track

        self._filters[track_id] = KalmanBoxFilter(
            detection.bounding_box,
            process_noise=self.config.process_noise,
            measurement_noise=self.config.measurement_noise,
        )

        self._append_trajectory(
            track,
            timestamp,
        )

        return track

    @staticmethod
    def _append_trajectory(
        track: TrackedObject,
        timestamp: datetime,
    ) -> None:
        cx, cy = track.center

        track.trajectory.append(
            TrajectoryPoint(
                x=cx,
                y=cy,
                timestamp=timestamp,
                velocity_x=track.velocity_x,
                velocity_y=track.velocity_y,
                speed_px_s=track.speed_px_s,
            )
        )

    def _update_matched_track(
        self,
        track: TrackedObject,
        detection: TrackingDetection,
        timestamp: datetime,
        dt: float,
    ) -> None:
        kalman_filter = self._filters[
            track.track_id
        ]

        previous_vx = track.velocity_x
        previous_vy = track.velocity_y

        updated_box = kalman_filter.update(
            detection.bounding_box
        )

        vx, vy = kalman_filter.velocity

        track.bounding_box = updated_box

        track.velocity_x = vx
        track.velocity_y = vy

        if dt > 0:
            track.acceleration_x = (
                vx - previous_vx
            ) / dt

            track.acceleration_y = (
                vy - previous_vy
            ) / dt

        track.confidence = detection.confidence
        track.distance_m = detection.distance_m

        track.hits += 1
        track.missed_frames = 0

        track.last_seen = timestamp

        if (
            track.hits
            >= self.config.minimum_confirmed_hits
        ):
            track.status = TrackStatus.CONFIRMED

        track.motion_status = (
            self.motion_analyzer.classify(track)
        )

        track.risk_level = (
            self.risk_assessor.assess(
                detection.distance_m
            )
        )

        track.metadata.update(
            detection.metadata
        )

        track.metadata["last_detection_id"] = (
            detection.detection_id
        )

        self._append_trajectory(
            track,
            timestamp,
        )

    def _mark_missed(
        self,
        track: TrackedObject,
    ) -> None:
        track.missed_frames += 1

        if (
            track.missed_frames
            > self.config.maximum_missed_frames
        ):
            track.status = TrackStatus.LOST

            track_id = track.track_id

            self._lost_tracks[track_id] = (
                track,
                self._filters[track_id],
                self._frame_number,
            )

            del self._tracks[track_id]
            del self._filters[track_id]

    def _reidentify(
        self,
        detection: TrackingDetection,
        timestamp: datetime,
        dt: float,
    ) -> bool:
        if not self.config.enable_reidentification:
            return False

        best_track_id = None
        best_iou = (
            self.config.reidentification_iou_threshold
        )

        for track_id, (
            track,
            kalman_filter,
            lost_frame,
        ) in self._lost_tracks.items():
            if (
                self._frame_number - lost_frame
                > self.config.reidentification_window_frames
            ):
                continue

            if (
                self.config.enable_class_aware_matching
                and track.class_id != detection.class_id
            ):
                continue

            overlap = kalman_filter.bounding_box.iou(
                detection.bounding_box
            )

            if overlap >= best_iou:
                best_iou = overlap
                best_track_id = track_id

        if best_track_id is None:
            return False

        track, kalman_filter, _ = (
            self._lost_tracks.pop(best_track_id)
        )

        self._tracks[best_track_id] = track
        self._filters[best_track_id] = kalman_filter

        track.status = TrackStatus.CONFIRMED
        track.missed_frames = 0

        self._update_matched_track(
            track,
            detection,
            timestamp,
            dt,
        )

        return True

    def _expire_lost_tracks(self) -> None:
        expired = []

        for track_id, (
            track,
            _,
            lost_frame,
        ) in self._lost_tracks.items():
            if (
                self._frame_number - lost_frame
                > self.config.reidentification_window_frames
            ):
                track.status = TrackStatus.REMOVED
                expired.append(track_id)

        for track_id in expired:
            del self._lost_tracks[track_id]

    def update(
        self,
        detections: Sequence[TrackingDetection],
        *,
        timestamp: datetime | None = None,
    ) -> tuple[TrackedObject, ...]:
        timestamp = timestamp or utc_now()

        if timestamp.tzinfo is None:
            raise TrackingValidationError(
                "Timestamp must be timezone-aware"
            )

        if (
            len(detections)
            > self.config.maximum_detections_per_frame
        ):
            raise TrackingCapacityError(
                "Too many detections in frame"
            )

        filtered = [
            detection
            for detection in detections
            if (
                detection.confidence
                >= self.config.minimum_confidence
            )
        ]

        with self._lock:
            if self._last_timestamp is not None:
                raw_dt = (
                    timestamp - self._last_timestamp
                ).total_seconds()

                if raw_dt < 0:
                    raise TrackingValidationError(
                        "Out-of-order frame timestamp"
                    )

                dt = min(
                    raw_dt,
                    self.config.maximum_prediction_dt_s,
                )

            else:
                dt = 0.0

            self._frame_number += 1
            self._last_timestamp = timestamp

            for track_id, track in self._tracks.items():
                kalman_filter = self._filters[
                    track_id
                ]

                track.bounding_box = (
                    kalman_filter.predict(dt)
                )

                track.age_frames += 1

            active_tracks = list(
                self._tracks.values()
            )

            association = self.association.associate(
                active_tracks,
                filtered,
            )

            for track_index, detection_index in (
                association.matches
            ):
                track = active_tracks[
                    track_index
                ]

                detection = filtered[
                    detection_index
                ]

                self._update_matched_track(
                    track,
                    detection,
                    timestamp,
                    dt,
                )

            for track_index in (
                association.unmatched_tracks
            ):
                self._mark_missed(
                    active_tracks[track_index]
                )

            for detection_index in (
                association.unmatched_detections
            ):
                detection = filtered[
                    detection_index
                ]

                recovered = self._reidentify(
                    detection,
                    timestamp,
                    dt,
                )

                if recovered:
                    continue

                if (
                    len(self._tracks)
                    >= self.config.maximum_tracks
                ):
                    continue

                self._create_track(
                    detection,
                    timestamp,
                )

            self._expire_lost_tracks()

            return tuple(
                self._tracks.values()
            )

    def active_tracks(
        self,
        *,
        confirmed_only: bool = False,
    ) -> tuple[TrackedObject, ...]:
        with self._lock:
            return tuple(
                track
                for track in self._tracks.values()
                if (
                    not confirmed_only
                    or track.status == TrackStatus.CONFIRMED
                )
            )

    def lost_tracks(
        self,
    ) -> tuple[TrackedObject, ...]:
        with self._lock:
            return tuple(
                item[0]
                for item in self._lost_tracks.values()
            )

    def get_track(
        self,
        track_id: str,
    ) -> TrackedObject | None:
        with self._lock:
            return self._tracks.get(track_id)

    def reset(self) -> None:
        with self._lock:
            self._tracks.clear()
            self._filters.clear()
            self._lost_tracks.clear()

            self._frame_number = 0
            self._last_timestamp = None


# ============================================================
# TRACKING FRAME RESULT
# ============================================================

@dataclass(frozen=True)
class ObjectTrackingFrame:
    frame_id: str

    tenant_id: str
    robot_id: str
    camera_id: str

    algorithm: TrackingAlgorithm

    detection_count: int

    active_track_count: int
    confirmed_track_count: int
    lost_track_count: int

    tracks: tuple[dict[str, Any], ...]

    processing_time_ms: float

    timestamp: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# ============================================================
# TRACKING STATISTICS
# ============================================================

@dataclass(frozen=True)
class TrackingStatistics:
    active_tracks: int
    confirmed_tracks: int
    tentative_tracks: int
    lost_tracks: int

    moving_objects: int
    stationary_objects: int

    human_tracks: int
    robot_tracks: int
    vehicle_tracks: int

    critical_risk_tracks: int
    high_risk_tracks: int

    average_speed_px_s: float


class TrackingAnalytics:

    @staticmethod
    def analyze(
        tracks: Sequence[TrackedObject],
        lost_tracks: Sequence[TrackedObject] = (),
    ) -> TrackingStatistics:
        confirmed = 0
        tentative = 0

        moving = 0
        stationary = 0

        humans = 0
        robots = 0
        vehicles = 0

        critical = 0
        high = 0

        speeds = []

        for track in tracks:
            if track.status == TrackStatus.CONFIRMED:
                confirmed += 1

            if track.status == TrackStatus.TENTATIVE:
                tentative += 1

            if track.motion_status == MotionStatus.MOVING:
                moving += 1

            if track.motion_status == MotionStatus.STATIONARY:
                stationary += 1

            name = track.class_name.lower()

            if name in ("person", "human", "worker"):
                humans += 1

            elif name in ("robot", "mobile robot", "robot arm"):
                robots += 1

            elif name in (
                "car",
                "truck",
                "bus",
                "forklift",
                "vehicle",
            ):
                vehicles += 1

            if track.risk_level == TrackingRiskLevel.CRITICAL:
                critical += 1

            if track.risk_level == TrackingRiskLevel.HIGH:
                high += 1

            speeds.append(
                track.speed_px_s
            )

        return TrackingStatistics(
            active_tracks=len(tracks),
            confirmed_tracks=confirmed,
            tentative_tracks=tentative,
            lost_tracks=len(lost_tracks),
            moving_objects=moving,
            stationary_objects=stationary,
            human_tracks=humans,
            robot_tracks=robots,
            vehicle_tracks=vehicles,
            critical_risk_tracks=critical,
            high_risk_tracks=high,
            average_speed_px_s=(
                float(np.mean(speeds))
                if speeds
                else 0.0
            ),
        )


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class TrackingAuditEvent:
    event_id: str

    tenant_id: str
    user_id: str

    robot_id: str
    camera_id: str

    action: str
    frame_id: str

    timestamp: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# MULTI-ROBOT TRACKING SERVICE
# ============================================================

class RoboticsObjectTrackingService:

    def __init__(
        self,
        config: ObjectTrackingConfig | None = None,
    ) -> None:
        self.config = (
            config or ObjectTrackingConfig()
        )

        self._trackers: dict[
            tuple[str, str, str],
            MultiObjectTracker,
        ] = {}

        self._history: dict[
            tuple[str, str, str],
            deque[ObjectTrackingFrame],
        ] = {}

        self._audit: deque[
            TrackingAuditEvent
        ] = deque(maxlen=1000)

        self._lock = threading.RLock()

    @staticmethod
    def _authorize(
        access: TrackingAccessContext,
        tenant_id: str,
        robot_id: str,
        camera_id: str,
        permission: str,
    ) -> None:
        if access.tenant_id != tenant_id:
            raise TrackingAuthorizationError(
                "Cross-tenant access denied"
            )

        if robot_id not in access.allowed_robot_ids:
            raise TrackingAuthorizationError(
                "Robot access denied"
            )

        if camera_id not in access.allowed_camera_ids:
            raise TrackingAuthorizationError(
                "Camera access denied"
            )

        if not getattr(access, permission, False):
            raise TrackingAuthorizationError(
                f"Missing permission: {permission}"
            )

    def process(
        self,
        detections: Sequence[TrackingDetection],
        *,
        robot_id: str,
        camera_id: str,
        access: TrackingAccessContext,
        timestamp: datetime | None = None,
    ) -> ObjectTrackingFrame:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            camera_id,
            "can_process",
        )

        started = time.perf_counter()

        key = (
            access.tenant_id,
            robot_id,
            camera_id,
        )

        with self._lock:
            tracker = self._trackers.setdefault(
                key,
                MultiObjectTracker(
                    self.config
                ),
            )

            tracks = tracker.update(
                detections,
                timestamp=timestamp,
            )

            lost = tracker.lost_tracks()

            frame = ObjectTrackingFrame(
                frame_id=generate_id(),
                tenant_id=access.tenant_id,
                robot_id=robot_id,
                camera_id=camera_id,
                algorithm=TrackingAlgorithm.IOU_KALMAN,
                detection_count=len(detections),
                active_track_count=len(tracks),
                confirmed_track_count=sum(
                    track.status == TrackStatus.CONFIRMED
                    for track in tracks
                ),
                lost_track_count=len(lost),
                tracks=tuple(
                    track.snapshot()
                    for track in tracks
                ),
                processing_time_ms=(
                    time.perf_counter() - started
                ) * 1000.0,
                timestamp=timestamp or utc_now(),
                metadata={
                    "advisory_only": True,
                    "coordinate_system": "image_pixels",
                    "motion_authorization": False,
                },
            )

            history = self._history.setdefault(
                key,
                deque(
                    maxlen=self.config.frame_history_size
                ),
            )

            history.append(frame)

            self._audit.append(
                TrackingAuditEvent(
                    event_id=generate_id(),
                    tenant_id=access.tenant_id,
                    user_id=access.user_id,
                    robot_id=robot_id,
                    camera_id=camera_id,
                    action="object_tracking.process",
                    frame_id=frame.frame_id,
                )
            )

            return frame

    def latest(
        self,
        *,
        robot_id: str,
        camera_id: str,
        access: TrackingAccessContext,
    ) -> ObjectTrackingFrame | None:
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
        access: TrackingAccessContext,
    ) -> tuple[ObjectTrackingFrame, ...]:
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

    def statistics(
        self,
        *,
        robot_id: str,
        camera_id: str,
        access: TrackingAccessContext,
    ) -> TrackingStatistics:
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
            tracker = self._trackers.get(key)

            if tracker is None:
                return TrackingAnalytics.analyze(())

            return TrackingAnalytics.analyze(
                tracker.active_tracks(),
                tracker.lost_tracks(),
            )

    def audit_events(
        self,
        *,
        access: TrackingAccessContext,
    ) -> tuple[TrackingAuditEvent, ...]:
        if not access.can_view:
            raise TrackingAuthorizationError(
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
        frame: ObjectTrackingFrame,
    ) -> str:
        return json.dumps(
            frame.to_dict(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


# ============================================================
# SYNTHETIC DETECTION GENERATOR
# ============================================================

class SyntheticTrackingGenerator:

    @staticmethod
    def generate_frame(
        frame_number: int,
    ) -> tuple[TrackingDetection, ...]:
        person_x = 50 + frame_number * 8

        robot_x = 350 - frame_number * 4

        return (
            TrackingDetection(
                detection_id=generate_id(),
                class_id=0,
                class_name="person",
                confidence=0.96,
                bounding_box=TrackingBoundingBox(
                    person_x,
                    100,
                    person_x + 80,
                    300,
                ),
                distance_m=max(
                    0.3,
                    4.0 - frame_number * 0.25,
                ),
            ),
            TrackingDetection(
                detection_id=generate_id(),
                class_id=1,
                class_name="robot",
                confidence=0.94,
                bounding_box=TrackingBoundingBox(
                    robot_x,
                    180,
                    robot_x + 100,
                    350,
                ),
                distance_m=2.5,
            ),
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestObjectTracking(unittest.TestCase):

    def setUp(self) -> None:
        self.config = ObjectTrackingConfig(
            minimum_confirmed_hits=2,
        )

        self.tracker = MultiObjectTracker(
            self.config
        )

        self.access = TrackingAccessContext(
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

    def test_bounding_box_iou(self) -> None:
        box = TrackingBoundingBox(
            10,
            10,
            100,
            100,
        )

        self.assertAlmostEqual(
            box.iou(box),
            1.0,
        )

    def test_kalman_filter(self) -> None:
        box = TrackingBoundingBox(
            10,
            10,
            100,
            100,
        )

        kalman = KalmanBoxFilter(box)

        kalman.predict(0.1)

        updated = kalman.update(
            TrackingBoundingBox(
                15,
                10,
                105,
                100,
            )
        )

        self.assertGreater(
            updated.center[0],
            box.center[0],
        )

    def test_track_creation(self) -> None:
        detections = (
            SyntheticTrackingGenerator.generate_frame(0)
        )

        tracks = self.tracker.update(
            detections
        )

        self.assertEqual(
            len(tracks),
            2,
        )

    def test_persistent_track_id(self) -> None:
        from datetime import timedelta

        timestamp = utc_now()

        first = self.tracker.update(
            SyntheticTrackingGenerator.generate_frame(0),
            timestamp=timestamp,
        )

        second = self.tracker.update(
            SyntheticTrackingGenerator.generate_frame(1),
            timestamp=timestamp + timedelta(seconds=0.1),
        )

        first_ids = {
            track.class_id: track.track_id
            for track in first
        }

        second_ids = {
            track.class_id: track.track_id
            for track in second
        }

        self.assertEqual(
            first_ids,
            second_ids,
        )

    def test_track_confirmation(self) -> None:
        from datetime import timedelta

        timestamp = utc_now()

        self.tracker.update(
            SyntheticTrackingGenerator.generate_frame(0),
            timestamp=timestamp,
        )

        tracks = self.tracker.update(
            SyntheticTrackingGenerator.generate_frame(1),
            timestamp=timestamp + timedelta(seconds=0.1),
        )

        self.assertTrue(
            all(
                track.status == TrackStatus.CONFIRMED
                for track in tracks
            )
        )

    def test_trajectory_history(self) -> None:
        from datetime import timedelta

        timestamp = utc_now()

        for index in range(3):
            self.tracker.update(
                SyntheticTrackingGenerator.generate_frame(index),
                timestamp=(
                    timestamp
                    + timedelta(seconds=index * 0.1)
                ),
            )

        tracks = self.tracker.active_tracks()

        self.assertTrue(
            all(
                len(track.trajectory) >= 2
                for track in tracks
            )
        )

    def test_risk_assessment(self) -> None:
        assessor = TrackingRiskAssessor(
            self.config
        )

        self.assertEqual(
            assessor.assess(0.3),
            TrackingRiskLevel.CRITICAL,
        )

        self.assertEqual(
            assessor.assess(None),
            TrackingRiskLevel.UNKNOWN,
        )

    def test_track_loss(self) -> None:
        tracker = MultiObjectTracker(
            ObjectTrackingConfig(
                maximum_missed_frames=1,
            )
        )

        tracker.update(
            SyntheticTrackingGenerator.generate_frame(0)
        )

        tracker.update(())
        tracker.update(())

        self.assertEqual(
            len(tracker.active_tracks()),
            0,
        )

        self.assertEqual(
            len(tracker.lost_tracks()),
            2,
        )

    def test_statistics(self) -> None:
        tracks = self.tracker.update(
            SyntheticTrackingGenerator.generate_frame(0)
        )

        stats = TrackingAnalytics.analyze(
            tracks
        )

        self.assertEqual(
            stats.active_tracks,
            2,
        )

    def test_service_history(self) -> None:
        service = RoboticsObjectTrackingService(
            self.config
        )

        service.process(
            SyntheticTrackingGenerator.generate_frame(0),
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
            TrackingAuthorizationError
        ):
            RoboticsObjectTrackingService._authorize(
                self.access,
                "tenant_beta",
                "ROBOT-001",
                "CAM-001",
                "can_process",
            )

    def test_json_export(self) -> None:
        service = RoboticsObjectTrackingService(
            self.config
        )

        frame = service.process(
            SyntheticTrackingGenerator.generate_frame(0),
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


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    from datetime import timedelta

    print(
        "\nPHYSICAL AI & AUTONOMOUS ROBOTICS PLATFORM"
    )

    print(
        "Real-Time Multi-Object Tracking Engine\n"
    )

    access = TrackingAccessContext(
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

    service = RoboticsObjectTrackingService(
        ObjectTrackingConfig(
            minimum_confirmed_hits=2,
        )
    )

    start_timestamp = utc_now()

    for frame_number in range(10):
        detections = (
            SyntheticTrackingGenerator.generate_frame(
                frame_number
            )
        )

        frame = service.process(
            detections,
            robot_id="ROBOT-001",
            camera_id="CAM-FRONT-001",
            access=access,
            timestamp=(
                start_timestamp
                + timedelta(
                    seconds=frame_number * 0.1
                )
            ),
        )

        print(
            f"Frame {frame_number + 1}: "
            f"{frame.active_track_count} active tracks, "
            f"{frame.confirmed_track_count} confirmed"
        )

    print("\nLATEST TRACKING RESULT\n")

    print(
        service.export_json(
            frame
        )
    )

    statistics = service.statistics(
        robot_id="ROBOT-001",
        camera_id="CAM-FRONT-001",
        access=access,
    )

    print("\nTRACKING STATISTICS\n")

    print(
        json.dumps(
            json_safe(statistics),
            indent=2,
        )
    )

    print("\nRUNNING UNIT TESTS\n")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestObjectTracking
    )

    unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)


if __name__ == "__main__":
    main()
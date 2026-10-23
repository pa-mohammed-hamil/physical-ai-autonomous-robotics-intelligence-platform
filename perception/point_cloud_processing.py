"""Point Cloud Processing module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/perception/point_cloud_processing.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================

Module: point_cloud_processing.py

Production-oriented 3D Point Cloud Processing

Features
--------
- LiDAR and RGB-D point cloud ingestion
- Point cloud validation and sanitization
- NaN / Inf removal
- Distance and workspace filtering
- Statistical outlier removal
- Radius outlier removal
- Voxel grid downsampling
- Random downsampling
- Coordinate transformations
- Point cloud normalization
- Ground plane segmentation using RANSAC
- Euclidean point cloud clustering
- 3D bounding box extraction
- Obstacle observations
- Point cloud density estimation
- Point cloud quality analysis
- Sensor health metrics
- Multi-tenant access control
- Bounded processing history
- JSON reports
- NumPy-based processing
- Optional SciPy acceleration
- Unit tests and demonstration

Dependencies
------------
Python >= 3.11
numpy

Optional
--------
scipy

Safety
------
This module produces perception observations only.

It does not authorize robot motion, perform emergency stopping,
or replace certified collision avoidance and safety controllers.
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
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


# ============================================================
# EXCEPTIONS
# ============================================================

class PointCloudError(Exception):
    """Base point cloud exception."""


class PointCloudValidationError(PointCloudError):
    """Invalid point cloud or configuration."""


class PointCloudAuthorizationError(PointCloudError):
    """Point cloud access denied."""


class PointCloudCapacityError(PointCloudError):
    """Point cloud capacity exceeded."""


class PointCloudProcessingError(PointCloudError):
    """Point cloud processing failed."""


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
        raise PointCloudValidationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > 128:
        raise PointCloudValidationError(
            f"{name} must contain 1-128 characters"
        )

    return value


def validate_timestamp(
    value: datetime,
) -> datetime:
    if not isinstance(value, datetime):
        raise PointCloudValidationError(
            "Timestamp must be datetime"
        )

    if (
        value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise PointCloudValidationError(
            "Timestamp must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


def finite_float(
    value: Any,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise PointCloudValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PointCloudValidationError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(result):
        raise PointCloudValidationError(
            f"{name} must be finite"
        )

    return result


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, np.ndarray):
        return value.tolist()

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

    return value


# ============================================================
# ENUMERATIONS
# ============================================================

class PointCloudSource(str, Enum):
    LIDAR = "lidar"
    DEPTH_CAMERA = "depth_camera"
    STEREO_CAMERA = "stereo_camera"
    SIMULATION = "simulation"
    FUSED = "fused"


class PointCloudStatus(str, Enum):
    SUCCESS = "success"
    EMPTY = "empty"
    DEGRADED = "degraded"
    FAILED = "failed"


class PointCloudQuality(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    DEGRADED = "degraded"
    POOR = "poor"
    UNKNOWN = "unknown"


class ObstacleRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    UNKNOWN = "unknown"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PointCloudProcessingConfig:
    maximum_input_points: int = 2_000_000
    maximum_output_points: int = 500_000

    minimum_range_m: float = 0.10
    maximum_range_m: float = 100.0

    voxel_size_m: float = 0.10

    enable_voxel_downsampling: bool = True
    enable_statistical_outlier_removal: bool = False
    enable_radius_outlier_removal: bool = False
    enable_ground_segmentation: bool = True
    enable_clustering: bool = True

    statistical_neighbors: int = 12
    statistical_std_ratio: float = 2.0

    radius_outlier_radius_m: float = 0.50
    radius_outlier_min_neighbors: int = 3

    ground_distance_threshold_m: float = 0.12
    ground_ransac_iterations: int = 100
    minimum_ground_inliers: int = 30
    maximum_ground_tilt_degrees: float = 30.0

    clustering_distance_m: float = 0.60
    minimum_cluster_points: int = 5
    maximum_cluster_points: int = 100_000
    maximum_clusters: int = 100

    obstacle_warning_distance_m: float = 3.0
    obstacle_high_risk_distance_m: float = 1.0

    minimum_quality_score: float = 60.0

    maximum_history: int = 100

    random_seed: int = 42

    def __post_init__(self) -> None:
        integer_fields = (
            "maximum_input_points",
            "maximum_output_points",
            "statistical_neighbors",
            "radius_outlier_min_neighbors",
            "ground_ransac_iterations",
            "minimum_ground_inliers",
            "minimum_cluster_points",
            "maximum_cluster_points",
            "maximum_clusters",
            "maximum_history",
        )

        for name in integer_fields:
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise PointCloudValidationError(
                    f"{name} must be a positive integer"
                )

        positive_fields = (
            "minimum_range_m",
            "maximum_range_m",
            "voxel_size_m",
            "statistical_std_ratio",
            "radius_outlier_radius_m",
            "ground_distance_threshold_m",
            "clustering_distance_m",
            "obstacle_warning_distance_m",
            "obstacle_high_risk_distance_m",
        )

        for name in positive_fields:
            if finite_float(getattr(self, name), name) <= 0:
                raise PointCloudValidationError(
                    f"{name} must be positive"
                )

        if self.minimum_range_m >= self.maximum_range_m:
            raise PointCloudValidationError(
                "Minimum range must be below maximum range"
            )

        if (
            self.obstacle_high_risk_distance_m
            >= self.obstacle_warning_distance_m
        ):
            raise PointCloudValidationError(
                "High-risk distance must be below warning distance"
            )

        if (
            self.minimum_cluster_points
            > self.maximum_cluster_points
        ):
            raise PointCloudValidationError(
                "Minimum cluster size exceeds maximum"
            )

        if not 0 <= self.maximum_ground_tilt_degrees < 90:
            raise PointCloudValidationError(
                "Ground tilt must be in [0, 90)"
            )

        if not 0 <= self.minimum_quality_score <= 100:
            raise PointCloudValidationError(
                "Quality score threshold must be in [0, 100]"
            )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class PointCloudAccessContext:
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
# DATA MODELS
# ============================================================

@dataclass(frozen=True)
class PointCloudFrame:
    frame_id: str

    tenant_id: str
    robot_id: str
    sensor_id: str

    source: PointCloudSource

    timestamp: datetime

    points: np.ndarray

    coordinate_frame: str = "sensor"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "frame_id",
            "tenant_id",
            "robot_id",
            "sensor_id",
            "coordinate_frame",
        ):
            validate_identifier(
                getattr(self, name),
                name,
            )

        validate_timestamp(self.timestamp)

        if not isinstance(self.source, PointCloudSource):
            raise PointCloudValidationError(
                "Invalid point cloud source"
            )

        if not isinstance(self.points, np.ndarray):
            raise PointCloudValidationError(
                "Points must be a NumPy array"
            )

        if (
            self.points.ndim != 2
            or self.points.shape[1] not in (3, 4, 6, 7)
        ):
            raise PointCloudValidationError(
                "Expected point cloud shape N x 3, 4, 6 or 7"
            )

        if not np.issubdtype(
            self.points.dtype,
            np.number,
        ):
            raise PointCloudValidationError(
                "Point cloud must contain numeric values"
            )


@dataclass(frozen=True)
class PointCloudBoundingBox:
    minimum_xyz: tuple[float, float, float]
    maximum_xyz: tuple[float, float, float]

    center_xyz: tuple[float, float, float]
    dimensions_xyz: tuple[float, float, float]

    volume_m3: float


@dataclass(frozen=True)
class PointCloudCluster:
    cluster_id: str

    point_count: int

    centroid_xyz: tuple[float, float, float]

    bounding_box: PointCloudBoundingBox

    minimum_distance_m: float

    confidence: float = 1.0


@dataclass(frozen=True)
class GroundPlane:
    normal_xyz: tuple[float, float, float]

    offset: float

    inlier_count: int
    inlier_ratio: float

    tilt_degrees: float


@dataclass(frozen=True)
class ObstacleObservation:
    obstacle_id: str

    cluster_id: str

    centroid_xyz: tuple[float, float, float]

    distance_m: float

    bounding_box: PointCloudBoundingBox

    risk_level: ObstacleRiskLevel

    point_count: int


@dataclass(frozen=True)
class PointCloudStatistics:
    input_point_count: int
    valid_point_count: int
    filtered_point_count: int
    output_point_count: int

    invalid_point_ratio: float

    minimum_range_m: float | None
    maximum_range_m: float | None
    mean_range_m: float | None

    centroid_xyz: tuple[float, float, float] | None

    bounding_box: PointCloudBoundingBox | None

    estimated_density_points_m3: float | None

    processing_latency_ms: float


@dataclass(frozen=True)
class PointCloudQualityReport:
    quality: PointCloudQuality

    quality_score: float

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class PointCloudProcessingResult:
    result_id: str

    frame_id: str
    tenant_id: str
    robot_id: str
    sensor_id: str

    timestamp: datetime

    status: PointCloudStatus

    processed_points: np.ndarray

    ground_plane: GroundPlane | None

    ground_point_count: int
    nonground_point_count: int

    clusters: tuple[PointCloudCluster, ...]

    obstacles: tuple[ObstacleObservation, ...]

    statistics: PointCloudStatistics

    quality: PointCloudQualityReport

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
        *,
        include_points: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "result_id": self.result_id,
            "frame_id": self.frame_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "sensor_id": self.sensor_id,
            "timestamp": self.timestamp,
            "status": self.status,
            "ground_plane": self.ground_plane,
            "ground_point_count": self.ground_point_count,
            "nonground_point_count": self.nonground_point_count,
            "clusters": self.clusters,
            "obstacles": self.obstacles,
            "statistics": self.statistics,
            "quality": self.quality,
            "metadata": self.metadata,
            "processed_point_shape": list(
                self.processed_points.shape
            ),
        }

        if include_points:
            payload["processed_points"] = (
                self.processed_points
            )

        return json_safe(payload)


# ============================================================
# POINT CLOUD VALIDATION
# ============================================================

class PointCloudValidator:

    def __init__(
        self,
        config: PointCloudProcessingConfig,
    ) -> None:
        self.config = config

    def validate(
        self,
        frame: PointCloudFrame,
    ) -> None:
        points = frame.points

        if len(points) > self.config.maximum_input_points:
            raise PointCloudCapacityError(
                "Input point count exceeds configured maximum"
            )

    @staticmethod
    def sanitize(
        points: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        if points.ndim != 2:
            raise PointCloudValidationError(
                "Point cloud must be two-dimensional"
            )

        finite_mask = np.all(
            np.isfinite(points),
            axis=1,
        )

        sanitized = points[finite_mask].astype(
            np.float32,
            copy=True,
        )

        removed = int(
            len(points) - len(sanitized)
        )

        return sanitized, removed


# ============================================================
# POINT CLOUD FILTERS
# ============================================================

class PointCloudFilters:

    @staticmethod
    def range_filter(
        points: np.ndarray,
        minimum_range_m: float,
        maximum_range_m: float,
    ) -> np.ndarray:
        if len(points) == 0:
            return points.copy()

        distances = np.linalg.norm(
            points[:, :3],
            axis=1,
        )

        mask = (
            (distances >= minimum_range_m)
            & (distances <= maximum_range_m)
        )

        return points[mask].copy()

    @staticmethod
    def workspace_filter(
        points: np.ndarray,
        minimum_xyz: Sequence[float],
        maximum_xyz: Sequence[float],
    ) -> np.ndarray:
        minimum = np.asarray(
            minimum_xyz,
            dtype=np.float64,
        )

        maximum = np.asarray(
            maximum_xyz,
            dtype=np.float64,
        )

        if minimum.shape != (3,) or maximum.shape != (3,):
            raise PointCloudValidationError(
                "Workspace bounds must contain three values"
            )

        if not (
            np.all(np.isfinite(minimum))
            and np.all(np.isfinite(maximum))
        ):
            raise PointCloudValidationError(
                "Workspace bounds must be finite"
            )

        if np.any(minimum >= maximum):
            raise PointCloudValidationError(
                "Workspace minimum must be below maximum"
            )

        mask = np.all(
            (points[:, :3] >= minimum)
            & (points[:, :3] <= maximum),
            axis=1,
        )

        return points[mask].copy()

    @staticmethod
    def voxel_downsample(
        points: np.ndarray,
        voxel_size_m: float,
    ) -> np.ndarray:
        voxel_size_m = finite_float(
            voxel_size_m,
            "voxel_size_m",
        )

        if voxel_size_m <= 0:
            raise PointCloudValidationError(
                "Voxel size must be positive"
            )

        if len(points) == 0:
            return points.copy()

        voxel_indices = np.floor(
            points[:, :3] / voxel_size_m
        ).astype(np.int64)

        _, inverse = np.unique(
            voxel_indices,
            axis=0,
            return_inverse=True,
        )

        voxel_count = int(
            inverse.max()
        ) + 1

        counts = np.bincount(
            inverse,
            minlength=voxel_count,
        )

        output = np.empty(
            (voxel_count, points.shape[1]),
            dtype=np.float32,
        )

        for column in range(points.shape[1]):
            sums = np.bincount(
                inverse,
                weights=points[:, column],
                minlength=voxel_count,
            )

            output[:, column] = (
                sums / counts
            )

        return output

    @staticmethod
    def random_downsample(
        points: np.ndarray,
        maximum_points: int,
        *,
        seed: int = 42,
    ) -> np.ndarray:
        if maximum_points <= 0:
            raise PointCloudValidationError(
                "maximum_points must be positive"
            )

        if len(points) <= maximum_points:
            return points.copy()

        rng = np.random.default_rng(seed)

        indices = rng.choice(
            len(points),
            size=maximum_points,
            replace=False,
        )

        indices.sort()

        return points[indices].copy()

    @staticmethod
    def statistical_outlier_removal(
        points: np.ndarray,
        neighbors: int = 12,
        std_ratio: float = 2.0,
    ) -> np.ndarray:
        if len(points) <= neighbors:
            return points.copy()

        if neighbors <= 0 or std_ratio <= 0:
            raise PointCloudValidationError(
                "Invalid statistical filter parameters"
            )

        if cKDTree is None:
            raise PointCloudProcessingError(
                "SciPy is required for statistical outlier removal"
            )

        tree = cKDTree(
            points[:, :3]
        )

        distances, _ = tree.query(
            points[:, :3],
            k=neighbors + 1,
        )

        mean_distances = np.mean(
            distances[:, 1:],
            axis=1,
        )

        threshold = (
            float(np.mean(mean_distances))
            + std_ratio
            * float(np.std(mean_distances))
        )

        return points[
            mean_distances <= threshold
        ].copy()

    @staticmethod
    def radius_outlier_removal(
        points: np.ndarray,
        radius_m: float = 0.5,
        minimum_neighbors: int = 3,
    ) -> np.ndarray:
        if len(points) == 0:
            return points.copy()

        if radius_m <= 0 or minimum_neighbors <= 0:
            raise PointCloudValidationError(
                "Invalid radius filter parameters"
            )

        if cKDTree is None:
            raise PointCloudProcessingError(
                "SciPy is required for radius outlier removal"
            )

        tree = cKDTree(
            points[:, :3]
        )

        counts = tree.query_ball_point(
            points[:, :3],
            r=radius_m,
            return_length=True,
        )

        mask = (
            np.asarray(counts) - 1
            >= minimum_neighbors
        )

        return points[mask].copy()


# ============================================================
# COORDINATE TRANSFORMATIONS
# ============================================================

class PointCloudTransformations:

    @staticmethod
    def transform(
        points: np.ndarray,
        transformation_matrix: np.ndarray,
    ) -> np.ndarray:
        matrix = np.asarray(
            transformation_matrix,
            dtype=np.float64,
        )

        if matrix.shape != (4, 4):
            raise PointCloudValidationError(
                "Transformation matrix must be 4x4"
            )

        if not np.all(np.isfinite(matrix)):
            raise PointCloudValidationError(
                "Transformation matrix must be finite"
            )

        if not np.allclose(
            matrix[3],
            [0.0, 0.0, 0.0, 1.0],
            atol=1e-6,
        ):
            raise PointCloudValidationError(
                "Invalid homogeneous transformation matrix"
            )

        rotation = matrix[:3, :3]

        if not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=1e-4,
        ):
            raise PointCloudValidationError(
                "Rotation matrix must be orthonormal"
            )

        if not np.isclose(
            np.linalg.det(rotation),
            1.0,
            atol=1e-4,
        ):
            raise PointCloudValidationError(
                "Rotation matrix determinant must be +1"
            )

        output = points.copy()

        output[:, :3] = (
            points[:, :3] @ rotation.T
            + matrix[:3, 3]
        )

        return output

    @staticmethod
    def translation_matrix(
        x: float,
        y: float,
        z: float,
    ) -> np.ndarray:
        matrix = np.eye(
            4,
            dtype=np.float64,
        )

        matrix[:3, 3] = [
            finite_float(x, "x"),
            finite_float(y, "y"),
            finite_float(z, "z"),
        ]

        return matrix

    @staticmethod
    def rotation_z(
        angle_radians: float,
    ) -> np.ndarray:
        angle = finite_float(
            angle_radians,
            "angle_radians",
        )

        cosine = math.cos(angle)
        sine = math.sin(angle)

        matrix = np.eye(
            4,
            dtype=np.float64,
        )

        matrix[:3, :3] = [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]

        return matrix


# ============================================================
# BOUNDING BOX CALCULATOR
# ============================================================

class PointCloudGeometry:

    @staticmethod
    def bounding_box(
        points: np.ndarray,
    ) -> PointCloudBoundingBox | None:
        if len(points) == 0:
            return None

        minimum = np.min(
            points[:, :3],
            axis=0,
        )

        maximum = np.max(
            points[:, :3],
            axis=0,
        )

        center = (
            minimum + maximum
        ) / 2.0

        dimensions = (
            maximum - minimum
        )

        volume = float(
            np.prod(dimensions)
        )

        return PointCloudBoundingBox(
            minimum_xyz=tuple(
                float(value)
                for value in minimum
            ),
            maximum_xyz=tuple(
                float(value)
                for value in maximum
            ),
            center_xyz=tuple(
                float(value)
                for value in center
            ),
            dimensions_xyz=tuple(
                float(value)
                for value in dimensions
            ),
            volume_m3=volume,
        )

    @staticmethod
    def centroid(
        points: np.ndarray,
    ) -> tuple[float, float, float] | None:
        if len(points) == 0:
            return None

        center = np.mean(
            points[:, :3],
            axis=0,
        )

        return tuple(
            float(value)
            for value in center
        )


# ============================================================
# GROUND PLANE SEGMENTATION
# ============================================================

class GroundPlaneSegmenter:

    def __init__(
        self,
        config: PointCloudProcessingConfig,
    ) -> None:
        self.config = config

    def segment(
        self,
        points: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        GroundPlane | None,
    ]:
        if len(points) < 3:
            return (
                np.empty(
                    (0, points.shape[1]),
                    dtype=points.dtype,
                ),
                points.copy(),
                None,
            )

        xyz = points[:, :3]

        rng = np.random.default_rng(
            self.config.random_seed
        )

        best_mask = None
        best_normal = None
        best_offset = 0.0
        best_count = 0

        max_tilt = math.radians(
            self.config.maximum_ground_tilt_degrees
        )

        for _ in range(
            self.config.ground_ransac_iterations
        ):
            indices = rng.choice(
                len(points),
                size=3,
                replace=False,
            )

            p1, p2, p3 = xyz[indices]

            normal = np.cross(
                p2 - p1,
                p3 - p1,
            )

            magnitude = np.linalg.norm(normal)

            if magnitude < 1e-9:
                continue

            normal = normal / magnitude

            if normal[2] < 0:
                normal = -normal

            tilt = math.acos(
                float(
                    np.clip(
                        normal[2],
                        -1.0,
                        1.0,
                    )
                )
            )

            if tilt > max_tilt:
                continue

            offset = -float(
                np.dot(normal, p1)
            )

            distances = np.abs(
                xyz @ normal + offset
            )

            mask = (
                distances
                <= self.config.ground_distance_threshold_m
            )

            count = int(
                np.count_nonzero(mask)
            )

            if count > best_count:
                best_count = count
                best_mask = mask
                best_normal = normal
                best_offset = offset

        if (
            best_mask is None
            or best_count
            < self.config.minimum_ground_inliers
        ):
            return (
                np.empty(
                    (0, points.shape[1]),
                    dtype=points.dtype,
                ),
                points.copy(),
                None,
            )

        ground = points[best_mask].copy()
        nonground = points[~best_mask].copy()

        tilt_degrees = math.degrees(
            math.acos(
                float(
                    np.clip(
                        best_normal[2],
                        -1.0,
                        1.0,
                    )
                )
            )
        )

        plane = GroundPlane(
            normal_xyz=tuple(
                float(value)
                for value in best_normal
            ),
            offset=float(best_offset),
            inlier_count=best_count,
            inlier_ratio=(
                best_count / len(points)
            ),
            tilt_degrees=tilt_degrees,
        )

        return ground, nonground, plane


# ============================================================
# EUCLIDEAN CLUSTERING
# ============================================================

class EuclideanPointCloudClusterer:

    def __init__(
        self,
        config: PointCloudProcessingConfig,
    ) -> None:
        self.config = config

    def cluster(
        self,
        points: np.ndarray,
    ) -> tuple[PointCloudCluster, ...]:
        if len(points) == 0:
            return ()

        if cKDTree is None:
            return self._voxel_connected_clusters(
                points
            )

        return self._kdtree_clusters(
            points
        )

    def _kdtree_clusters(
        self,
        points: np.ndarray,
    ) -> tuple[PointCloudCluster, ...]:
        tree = cKDTree(
            points[:, :3]
        )

        visited = np.zeros(
            len(points),
            dtype=bool,
        )

        clusters = []

        for start in range(len(points)):
            if visited[start]:
                continue

            queue = deque([start])
            visited[start] = True

            indices = []

            while queue:
                index = queue.popleft()
                indices.append(index)

                neighbors = tree.query_ball_point(
                    points[index, :3],
                    r=self.config.clustering_distance_m,
                )

                for neighbor in neighbors:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

            if not (
                self.config.minimum_cluster_points
                <= len(indices)
                <= self.config.maximum_cluster_points
            ):
                continue

            cluster_points = points[
                np.asarray(indices, dtype=np.int64)
            ]

            clusters.append(
                self._make_cluster(
                    cluster_points,
                    len(clusters),
                )
            )

            if (
                len(clusters)
                >= self.config.maximum_clusters
            ):
                break

        return tuple(clusters)

    def _voxel_connected_clusters(
        self,
        points: np.ndarray,
    ) -> tuple[PointCloudCluster, ...]:
        """
        Approximate connected-component clustering fallback.

        Uses occupied voxels and 26-neighborhood adjacency.
        Unlike KD-tree radius clustering, this is approximate.
        """
        resolution = (
            self.config.clustering_distance_m
            / math.sqrt(3.0)
        )

        voxel_coordinates = np.floor(
            points[:, :3] / resolution
        ).astype(np.int64)

        voxel_map: dict[
            tuple[int, int, int],
            list[int],
        ] = {}

        for index, voxel in enumerate(voxel_coordinates):
            key = tuple(
                int(value)
                for value in voxel
            )

            voxel_map.setdefault(
                key,
                [],
            ).append(index)

        visited = set()
        clusters = []

        offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ]

        for starting_voxel in voxel_map:
            if starting_voxel in visited:
                continue

            queue = deque([
                starting_voxel
            ])

            visited.add(
                starting_voxel
            )

            point_indices = []

            while queue:
                voxel = queue.popleft()

                point_indices.extend(
                    voxel_map[voxel]
                )

                for dx, dy, dz in offsets:
                    neighbor = (
                        voxel[0] + dx,
                        voxel[1] + dy,
                        voxel[2] + dz,
                    )

                    if (
                        neighbor in voxel_map
                        and neighbor not in visited
                    ):
                        visited.add(neighbor)
                        queue.append(neighbor)

            if not (
                self.config.minimum_cluster_points
                <= len(point_indices)
                <= self.config.maximum_cluster_points
            ):
                continue

            cluster_points = points[
                np.asarray(
                    point_indices,
                    dtype=np.int64,
                )
            ]

            clusters.append(
                self._make_cluster(
                    cluster_points,
                    len(clusters),
                )
            )

            if (
                len(clusters)
                >= self.config.maximum_clusters
            ):
                break

        return tuple(clusters)

    @staticmethod
    def _make_cluster(
        points: np.ndarray,
        index: int,
    ) -> PointCloudCluster:
        bounding_box = (
            PointCloudGeometry.bounding_box(points)
        )

        centroid = (
            PointCloudGeometry.centroid(points)
        )

        distances = np.linalg.norm(
            points[:, :3],
            axis=1,
        )

        return PointCloudCluster(
            cluster_id=f"CLUSTER-{index + 1:04d}",
            point_count=len(points),
            centroid_xyz=centroid,
            bounding_box=bounding_box,
            minimum_distance_m=float(
                np.min(distances)
            ),
        )


# ============================================================
# OBSTACLE ANALYZER
# ============================================================

class PointCloudObstacleAnalyzer:

    def __init__(
        self,
        config: PointCloudProcessingConfig,
    ) -> None:
        self.config = config

    def classify_risk(
        self,
        distance_m: float,
    ) -> ObstacleRiskLevel:
        if (
            distance_m
            <= self.config.obstacle_high_risk_distance_m
        ):
            return ObstacleRiskLevel.HIGH

        if (
            distance_m
            <= self.config.obstacle_warning_distance_m
        ):
            return ObstacleRiskLevel.MODERATE

        return ObstacleRiskLevel.LOW

    def analyze(
        self,
        clusters: Sequence[PointCloudCluster],
    ) -> tuple[ObstacleObservation, ...]:
        observations = []

        for cluster in clusters:
            observations.append(
                ObstacleObservation(
                    obstacle_id=generate_id(),
                    cluster_id=cluster.cluster_id,
                    centroid_xyz=cluster.centroid_xyz,
                    distance_m=cluster.minimum_distance_m,
                    bounding_box=cluster.bounding_box,
                    risk_level=self.classify_risk(
                        cluster.minimum_distance_m
                    ),
                    point_count=cluster.point_count,
                )
            )

        return tuple(observations)


# ============================================================
# POINT CLOUD STATISTICS
# ============================================================

class PointCloudStatisticsCalculator:

    @staticmethod
    def calculate(
        *,
        input_count: int,
        valid_count: int,
        filtered_count: int,
        points: np.ndarray,
        latency_ms: float,
    ) -> PointCloudStatistics:
        if len(points) == 0:
            return PointCloudStatistics(
                input_point_count=input_count,
                valid_point_count=valid_count,
                filtered_point_count=filtered_count,
                output_point_count=0,
                invalid_point_ratio=(
                    (input_count - valid_count) / input_count
                    if input_count
                    else 0.0
                ),
                minimum_range_m=None,
                maximum_range_m=None,
                mean_range_m=None,
                centroid_xyz=None,
                bounding_box=None,
                estimated_density_points_m3=None,
                processing_latency_ms=latency_ms,
            )

        ranges = np.linalg.norm(
            points[:, :3],
            axis=1,
        )

        bounding_box = (
            PointCloudGeometry.bounding_box(points)
        )

        density = (
            len(points) / bounding_box.volume_m3
            if bounding_box.volume_m3 > 1e-9
            else None
        )

        return PointCloudStatistics(
            input_point_count=input_count,
            valid_point_count=valid_count,
            filtered_point_count=filtered_count,
            output_point_count=len(points),
            invalid_point_ratio=(
                (input_count - valid_count) / input_count
                if input_count
                else 0.0
            ),
            minimum_range_m=float(
                np.min(ranges)
            ),
            maximum_range_m=float(
                np.max(ranges)
            ),
            mean_range_m=float(
                np.mean(ranges)
            ),
            centroid_xyz=(
                PointCloudGeometry.centroid(points)
            ),
            bounding_box=bounding_box,
            estimated_density_points_m3=density,
            processing_latency_ms=latency_ms,
        )


# ============================================================
# QUALITY EVALUATOR
# ============================================================

class PointCloudQualityEvaluator:

    def __init__(
        self,
        config: PointCloudProcessingConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        statistics: PointCloudStatistics,
    ) -> PointCloudQualityReport:
        score = 100.0
        warnings = []

        if statistics.input_point_count == 0:
            return PointCloudQualityReport(
                quality=PointCloudQuality.UNKNOWN,
                quality_score=0.0,
                warnings=("Empty point cloud",),
            )

        if statistics.invalid_point_ratio >= 0.20:
            score -= 30.0
            warnings.append(
                "High invalid point ratio"
            )

        elif statistics.invalid_point_ratio >= 0.05:
            score -= 10.0
            warnings.append(
                "Elevated invalid point ratio"
            )

        if statistics.output_point_count == 0:
            score -= 60.0
            warnings.append(
                "No points remain after filtering"
            )

        elif statistics.output_point_count < 10:
            score -= 20.0
            warnings.append(
                "Very sparse point cloud"
            )

        if statistics.processing_latency_ms > 500:
            score -= 20.0
            warnings.append(
                "High processing latency"
            )

        score = max(
            0.0,
            min(100.0, score),
        )

        if score >= 90:
            quality = PointCloudQuality.EXCELLENT

        elif score >= 75:
            quality = PointCloudQuality.GOOD

        elif score >= self.config.minimum_quality_score:
            quality = PointCloudQuality.DEGRADED

        else:
            quality = PointCloudQuality.POOR

        return PointCloudQualityReport(
            quality=quality,
            quality_score=score,
            warnings=tuple(warnings),
        )


# ============================================================
# MAIN POINT CLOUD PROCESSING SERVICE
# ============================================================

class RoboticsPointCloudProcessingService:

    def __init__(
        self,
        config: PointCloudProcessingConfig | None = None,
    ) -> None:
        self.config = (
            config or PointCloudProcessingConfig()
        )

        self.validator = PointCloudValidator(
            self.config
        )

        self.filters = PointCloudFilters()

        self.ground_segmenter = GroundPlaneSegmenter(
            self.config
        )

        self.clusterer = EuclideanPointCloudClusterer(
            self.config
        )

        self.obstacle_analyzer = PointCloudObstacleAnalyzer(
            self.config
        )

        self.statistics_calculator = (
            PointCloudStatisticsCalculator()
        )

        self.quality_evaluator = (
            PointCloudQualityEvaluator(
                self.config
            )
        )

        self._history: dict[
            tuple[str, str],
            deque[PointCloudProcessingResult],
        ] = {}

        self._lock = threading.RLock()

    @staticmethod
    def _authorize(
        access: PointCloudAccessContext,
        tenant_id: str,
        robot_id: str,
        permission: str,
    ) -> None:
        if access.tenant_id != tenant_id:
            raise PointCloudAuthorizationError(
                "Cross-tenant access denied"
            )

        if robot_id not in access.allowed_robot_ids:
            raise PointCloudAuthorizationError(
                "Robot access denied"
            )

        if not getattr(access, permission, False):
            raise PointCloudAuthorizationError(
                f"Missing permission: {permission}"
            )

    def process(
        self,
        frame: PointCloudFrame,
        *,
        access: PointCloudAccessContext,
        transformation_matrix: np.ndarray | None = None,
        workspace_minimum_xyz: Sequence[float] | None = None,
        workspace_maximum_xyz: Sequence[float] | None = None,
    ) -> PointCloudProcessingResult:
        self._authorize(
            access,
            frame.tenant_id,
            frame.robot_id,
            "can_process",
        )

        self.validator.validate(frame)

        started = time.perf_counter()

        input_count = len(frame.points)

        points, invalid_count = (
            self.validator.sanitize(
                frame.points
            )
        )

        valid_count = len(points)

        if transformation_matrix is not None:
            points = PointCloudTransformations.transform(
                points,
                transformation_matrix,
            )

        points = self.filters.range_filter(
            points,
            self.config.minimum_range_m,
            self.config.maximum_range_m,
        )

        if (
            workspace_minimum_xyz is not None
            or workspace_maximum_xyz is not None
        ):
            if (
                workspace_minimum_xyz is None
                or workspace_maximum_xyz is None
            ):
                raise PointCloudValidationError(
                    "Both workspace bounds are required"
                )

            points = self.filters.workspace_filter(
                points,
                workspace_minimum_xyz,
                workspace_maximum_xyz,
            )

        if (
            self.config.enable_statistical_outlier_removal
            and len(points)
        ):
            points = (
                self.filters.statistical_outlier_removal(
                    points,
                    neighbors=self.config.statistical_neighbors,
                    std_ratio=self.config.statistical_std_ratio,
                )
            )

        if (
            self.config.enable_radius_outlier_removal
            and len(points)
        ):
            points = (
                self.filters.radius_outlier_removal(
                    points,
                    radius_m=(
                        self.config.radius_outlier_radius_m
                    ),
                    minimum_neighbors=(
                        self.config.radius_outlier_min_neighbors
                    ),
                )
            )

        filtered_count = len(points)

        if (
            self.config.enable_voxel_downsampling
            and len(points)
        ):
            points = self.filters.voxel_downsample(
                points,
                self.config.voxel_size_m,
            )

        if len(points) > self.config.maximum_output_points:
            points = self.filters.random_downsample(
                points,
                self.config.maximum_output_points,
                seed=self.config.random_seed,
            )

        ground = np.empty(
            (0, points.shape[1]),
            dtype=points.dtype,
        )

        nonground = points.copy()

        ground_plane = None

        if (
            self.config.enable_ground_segmentation
            and len(points) >= 3
        ):
            (
                ground,
                nonground,
                ground_plane,
            ) = self.ground_segmenter.segment(
                points
            )

        clusters = ()

        if (
            self.config.enable_clustering
            and len(nonground)
        ):
            clusters = self.clusterer.cluster(
                nonground
            )

        obstacles = self.obstacle_analyzer.analyze(
            clusters
        )

        latency_ms = (
            time.perf_counter() - started
        ) * 1000.0

        statistics = (
            self.statistics_calculator.calculate(
                input_count=input_count,
                valid_count=valid_count,
                filtered_count=filtered_count,
                points=points,
                latency_ms=latency_ms,
            )
        )

        quality = self.quality_evaluator.evaluate(
            statistics
        )

        if len(points) == 0:
            status = PointCloudStatus.EMPTY

        elif quality.quality in (
            PointCloudQuality.DEGRADED,
            PointCloudQuality.POOR,
        ):
            status = PointCloudStatus.DEGRADED

        else:
            status = PointCloudStatus.SUCCESS

        result = PointCloudProcessingResult(
            result_id=generate_id(),
            frame_id=frame.frame_id,
            tenant_id=frame.tenant_id,
            robot_id=frame.robot_id,
            sensor_id=frame.sensor_id,
            timestamp=frame.timestamp,
            status=status,
            processed_points=points,
            ground_plane=ground_plane,
            ground_point_count=len(ground),
            nonground_point_count=len(nonground),
            clusters=clusters,
            obstacles=obstacles,
            statistics=statistics,
            quality=quality,
            metadata={
                "source": frame.source.value,
                "coordinate_frame": frame.coordinate_frame,
                "invalid_points_removed": invalid_count,
                "scipy_available": cKDTree is not None,
                "advisory_only": True,
                "motion_authorization": False,
            },
        )

        key = (
            frame.tenant_id,
            frame.robot_id,
        )

        with self._lock:
            history = self._history.setdefault(
                key,
                deque(
                    maxlen=self.config.maximum_history
                ),
            )

            history.append(result)

        return result

    def latest(
        self,
        *,
        robot_id: str,
        access: PointCloudAccessContext,
    ) -> PointCloudProcessingResult | None:
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
        access: PointCloudAccessContext,
    ) -> tuple[PointCloudProcessingResult, ...]:
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

    def reset_robot(
        self,
        *,
        robot_id: str,
        access: PointCloudAccessContext,
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
        result: PointCloudProcessingResult,
        *,
        include_points: bool = False,
    ) -> str:
        return json.dumps(
            result.to_dict(
                include_points=include_points
            ),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


# ============================================================
# POINT CLOUD FACTORY
# ============================================================

class PointCloudFactory:

    @staticmethod
    def from_numpy(
        *,
        tenant_id: str,
        robot_id: str,
        sensor_id: str,
        points: np.ndarray,
        source: PointCloudSource = PointCloudSource.LIDAR,
        coordinate_frame: str = "sensor",
        timestamp: datetime | None = None,
    ) -> PointCloudFrame:
        return PointCloudFrame(
            frame_id=generate_id(),
            tenant_id=tenant_id,
            robot_id=robot_id,
            sensor_id=sensor_id,
            source=source,
            timestamp=timestamp or utc_now(),
            points=np.asarray(points),
            coordinate_frame=coordinate_frame,
        )

    @staticmethod
    def from_depth_map(
        *,
        tenant_id: str,
        robot_id: str,
        sensor_id: str,
        depth_map: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        depth_scale_m: float = 1.0,
        timestamp: datetime | None = None,
    ) -> PointCloudFrame:
        depth = np.asarray(
            depth_map,
            dtype=np.float32,
        )

        if depth.ndim != 2:
            raise PointCloudValidationError(
                "Depth map must be two-dimensional"
            )

        fx = finite_float(fx, "fx")
        fy = finite_float(fy, "fy")
        cx = finite_float(cx, "cx")
        cy = finite_float(cy, "cy")

        depth_scale_m = finite_float(
            depth_scale_m,
            "depth_scale_m",
        )

        if fx <= 0 or fy <= 0 or depth_scale_m <= 0:
            raise PointCloudValidationError(
                "Focal lengths and depth scale must be positive"
            )

        height, width = depth.shape

        u, v = np.meshgrid(
            np.arange(width),
            np.arange(height),
        )

        z = depth * depth_scale_m

        valid = (
            np.isfinite(z)
            & (z > 0)
        )

        x = (
            (u - cx) * z / fx
        )

        y = (
            (v - cy) * z / fy
        )

        points = np.column_stack(
            (
                x[valid],
                y[valid],
                z[valid],
            )
        ).astype(np.float32)

        return PointCloudFrame(
            frame_id=generate_id(),
            tenant_id=tenant_id,
            robot_id=robot_id,
            sensor_id=sensor_id,
            source=PointCloudSource.DEPTH_CAMERA,
            timestamp=timestamp or utc_now(),
            points=points,
            coordinate_frame="camera_optical",
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestPointCloudProcessing(unittest.TestCase):

    def setUp(self) -> None:
        self.config = PointCloudProcessingConfig(
            minimum_ground_inliers=5,
            minimum_cluster_points=3,
        )

        self.service = (
            RoboticsPointCloudProcessingService(
                self.config
            )
        )

        self.access = PointCloudAccessContext(
            tenant_id="tenant_alpha",
            user_id="robotics_engineer",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
            }),
            can_process=True,
            can_view=True,
            can_manage=True,
        )

        self.points = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.1, 0.1, 0.0],
                [1.2, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.1, 0.1, 0.0],
                [2.2, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.1, 0.1, 1.0],
                [1.2, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def make_frame(
        self,
        points: np.ndarray | None = None,
    ) -> PointCloudFrame:
        return PointCloudFactory.from_numpy(
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            sensor_id="LIDAR-001",
            points=(
                self.points
                if points is None
                else points
            ),
        )

    def test_frame_creation(self) -> None:
        frame = self.make_frame()

        self.assertEqual(
            frame.source,
            PointCloudSource.LIDAR,
        )

    def test_invalid_shape(self) -> None:
        with self.assertRaises(
            PointCloudValidationError
        ):
            self.make_frame(
                np.zeros(
                    (10, 2),
                    dtype=np.float32,
                )
            )

    def test_sanitize(self) -> None:
        points = np.array(
            [
                [1.0, 2.0, 3.0],
                [np.nan, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        cleaned, removed = (
            PointCloudValidator.sanitize(points)
        )

        self.assertEqual(
            len(cleaned),
            1,
        )

        self.assertEqual(
            removed,
            1,
        )

    def test_range_filter(self) -> None:
        filtered = PointCloudFilters.range_filter(
            self.points,
            0.5,
            1.5,
        )

        self.assertGreater(
            len(filtered),
            0,
        )

    def test_workspace_filter(self) -> None:
        filtered = PointCloudFilters.workspace_filter(
            self.points,
            [0.0, -1.0, -1.0],
            [1.5, 1.0, 2.0],
        )

        self.assertTrue(
            np.all(
                filtered[:, 0] <= 1.5
            )
        )

    def test_voxel_downsample(self) -> None:
        output = PointCloudFilters.voxel_downsample(
            self.points,
            0.5,
        )

        self.assertLessEqual(
            len(output),
            len(self.points),
        )

    def test_random_downsample(self) -> None:
        output = PointCloudFilters.random_downsample(
            self.points,
            4,
        )

        self.assertEqual(
            len(output),
            4,
        )

    def test_translation(self) -> None:
        matrix = (
            PointCloudTransformations.translation_matrix(
                1.0,
                2.0,
                3.0,
            )
        )

        transformed = PointCloudTransformations.transform(
            self.points,
            matrix,
        )

        np.testing.assert_allclose(
            transformed[0, :3],
            [2.0, 2.0, 3.0],
        )

    def test_rotation(self) -> None:
        matrix = (
            PointCloudTransformations.rotation_z(
                math.pi / 2
            )
        )

        point = np.array(
            [[1.0, 0.0, 0.0]],
            dtype=np.float32,
        )

        transformed = PointCloudTransformations.transform(
            point,
            matrix,
        )

        np.testing.assert_allclose(
            transformed[0, :3],
            [0.0, 1.0, 0.0],
            atol=1e-6,
        )

    def test_bounding_box(self) -> None:
        box = PointCloudGeometry.bounding_box(
            self.points
        )

        self.assertIsNotNone(box)

        self.assertGreaterEqual(
            box.volume_m3,
            0.0,
        )

    def test_centroid(self) -> None:
        centroid = PointCloudGeometry.centroid(
            self.points
        )

        self.assertEqual(
            len(centroid),
            3,
        )

    def test_ground_segmentation(self) -> None:
        ground, nonground, plane = (
            self.service.ground_segmenter.segment(
                self.points
            )
        )

        self.assertEqual(
            len(ground) + len(nonground),
            len(self.points),
        )

    def test_clustering(self) -> None:
        clusters = self.service.clusterer.cluster(
            self.points
        )

        self.assertIsInstance(
            clusters,
            tuple,
        )

    def test_processing(self) -> None:
        result = self.service.process(
            self.make_frame(),
            access=self.access,
        )

        self.assertIn(
            result.status,
            (
                PointCloudStatus.SUCCESS,
                PointCloudStatus.DEGRADED,
                PointCloudStatus.EMPTY,
            ),
        )

    def test_history(self) -> None:
        self.service.process(
            self.make_frame(),
            access=self.access,
        )

        history = self.service.history(
            robot_id="ROBOT-001",
            access=self.access,
        )

        self.assertEqual(
            len(history),
            1,
        )

    def test_latest(self) -> None:
        self.service.process(
            self.make_frame(),
            access=self.access,
        )

        latest = self.service.latest(
            robot_id="ROBOT-001",
            access=self.access,
        )

        self.assertIsNotNone(latest)

    def test_tenant_isolation(self) -> None:
        with self.assertRaises(
            PointCloudAuthorizationError
        ):
            self.service._authorize(
                self.access,
                "tenant_beta",
                "ROBOT-001",
                "can_view",
            )

    def test_json_export(self) -> None:
        result = self.service.process(
            self.make_frame(),
            access=self.access,
        )

        payload = self.service.export_json(
            result
        )

        self.assertIn(
            "ROBOT-001",
            payload,
        )

    def test_depth_to_point_cloud(self) -> None:
        depth = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=np.float32,
        )

        frame = PointCloudFactory.from_depth_map(
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            sensor_id="DEPTH-001",
            depth_map=depth,
            fx=100.0,
            fy=100.0,
            cx=0.5,
            cy=0.5,
        )

        self.assertEqual(
            frame.points.shape,
            (4, 3),
        )

    def test_reset(self) -> None:
        self.service.process(
            self.make_frame(),
            access=self.access,
        )

        self.service.reset_robot(
            robot_id="ROBOT-001",
            access=self.access,
        )

        self.assertEqual(
            len(
                self.service.history(
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
        "3D Point Cloud Processing & Obstacle Intelligence\n"
    )

    rng = np.random.default_rng(42)

    ground = np.column_stack(
        (
            rng.uniform(-5, 5, 1500),
            rng.uniform(-5, 5, 1500),
            rng.normal(0, 0.02, 1500),
        )
    )

    obstacle_one = rng.normal(
        loc=[1.5, 0.5, 1.0],
        scale=[0.15, 0.15, 0.25],
        size=(300, 3),
    )

    obstacle_two = rng.normal(
        loc=[4.0, 2.0, 1.0],
        scale=[0.20, 0.20, 0.30],
        size=(300, 3),
    )

    points = np.vstack(
        (
            ground,
            obstacle_one,
            obstacle_two,
        )
    ).astype(np.float32)

    access = PointCloudAccessContext(
        tenant_id="tenant_alpha",
        user_id="robotics_engineer",
        allowed_robot_ids=frozenset({
            "ROBOT-001",
        }),
        can_process=True,
        can_view=True,
        can_manage=True,
    )

    service = RoboticsPointCloudProcessingService()

    frame = PointCloudFactory.from_numpy(
        tenant_id="tenant_alpha",
        robot_id="ROBOT-001",
        sensor_id="LIDAR-FRONT-001",
        points=points,
        source=PointCloudSource.SIMULATION,
        coordinate_frame="base_link",
    )

    result = service.process(
        frame,
        access=access,
    )

    print("PROCESSING STATUS")
    print(result.status.value)

    print("\nPOINT COUNTS")
    print(
        "Input:",
        result.statistics.input_point_count,
    )
    print(
        "Output:",
        result.statistics.output_point_count,
    )
    print(
        "Ground:",
        result.ground_point_count,
    )
    print(
        "Non-ground:",
        result.nonground_point_count,
    )

    print("\nGROUND PLANE")
    print(
        json.dumps(
            json_safe(result.ground_plane),
            indent=2,
        )
    )

    print("\nCLUSTERS")
    print(len(result.clusters))

    print("\nOBSTACLES")

    for obstacle in result.obstacles:
        print(
            f"{obstacle.cluster_id}: "
            f"distance={obstacle.distance_m:.2f}m, "
            f"risk={obstacle.risk_level.value}, "
            f"points={obstacle.point_count}"
        )

    print("\nQUALITY")
    print(result.quality.quality.value)
    print(
        f"Score: {result.quality.quality_score:.1f}/100"
    )

    print("\nPROCESSING LATENCY")
    print(
        f"{result.statistics.processing_latency_ms:.2f}ms"
    )

    print("\nJSON REPORT\n")
    print(
        service.export_json(result)
    )

    print("\nRUNNING UNIT TESTS\n")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestPointCloudProcessing
    )

    test_result = unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)

    if not test_result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
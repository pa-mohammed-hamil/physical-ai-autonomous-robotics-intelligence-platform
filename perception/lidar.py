"""Lidar module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/perception/lidar.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================

Module: lidar.py

LiDAR perception and point-cloud processing engine.

Features
--------
- 2D and 3D LiDAR point-cloud ingestion
- XYZ / XYZI point formats
- Point-cloud validation
- Range and field-of-view filtering
- Voxel-grid downsampling
- Statistical outlier removal
- Radius-based outlier removal
- Ground-plane segmentation using RANSAC
- Euclidean point-cloud clustering
- Obstacle detection and 3D bounding boxes
- Obstacle distance estimation
- Point-cloud coordinate transformations
- 2D occupancy-grid generation
- Simulated collision-risk assessment
- Multi-robot and multi-tenant isolation
- LiDAR frame history
- Processing metrics and audit logging
- JSON metadata export
- Synthetic LiDAR scan generation
- Unit tests and demonstration

Dependencies
------------
Required:
    numpy

Optional:
    scipy

Installation:
    pip install numpy scipy

Safety
------
This module provides advisory perception results only.

It does not issue motion commands, actuate robots, or certify
collision avoidance. Real-world operation requires independent
validated safety systems, calibrated sensors, synchronization,
sensor-fusion checks, and fail-safe robot controllers.
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

class LiDARError(Exception):
    """Base LiDAR processing exception."""


class LiDARValidationError(LiDARError):
    """Invalid LiDAR input or configuration."""


class LiDARAuthorizationError(LiDARError):
    """LiDAR access denied."""


class LiDARCapacityError(LiDARError):
    """LiDAR processing capacity exceeded."""


class LiDARProcessingError(LiDARError):
    """LiDAR processing failed."""


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return uuid.uuid4().hex


def validate_identifier(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise LiDARValidationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > 128:
        raise LiDARValidationError(
            f"{name} must contain 1-128 characters"
        )

    return value


def finite_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise LiDARValidationError(
            f"{name} must be numeric"
        )

    result = float(value)

    if not math.isfinite(result):
        raise LiDARValidationError(
            f"{name} must be finite"
        )

    return result


def positive_float(value: Any, name: str) -> float:
    result = finite_float(value, name)

    if result <= 0:
        raise LiDARValidationError(
            f"{name} must be positive"
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
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
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

class LiDARType(str, Enum):
    PLANAR_2D = "planar_2d"
    SPINNING_3D = "spinning_3d"
    SOLID_STATE_3D = "solid_state_3d"
    SYNTHETIC = "synthetic"


class PointCloudFormat(str, Enum):
    XYZ = "xyz"
    XYZI = "xyzi"


class LiDARFrameStatus(str, Enum):
    VALID = "valid"
    EMPTY = "empty"
    DEGRADED = "degraded"


class ObstacleRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class OccupancyCell(str, Enum):
    UNKNOWN = "unknown"
    FREE = "free"
    OCCUPIED = "occupied"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class LiDARConfig:
    minimum_range_m: float = 0.1
    maximum_range_m: float = 100.0

    voxel_size_m: float = 0.1

    minimum_cluster_points: int = 5
    maximum_cluster_points: int = 100_000

    clustering_tolerance_m: float = 0.6

    ground_distance_threshold_m: float = 0.15
    ground_ransac_iterations: int = 100

    statistical_neighbors: int = 12
    statistical_std_ratio: float = 2.0

    radius_outlier_radius_m: float = 0.5
    radius_outlier_min_neighbors: int = 3

    maximum_input_points: int = 500_000
    maximum_processing_points: int = 100_000

    maximum_history_frames: int = 50

    critical_distance_m: float = 0.5
    high_risk_distance_m: float = 1.5
    moderate_risk_distance_m: float = 3.0

    def __post_init__(self) -> None:
        for name in (
            "minimum_range_m",
            "maximum_range_m",
            "voxel_size_m",
            "clustering_tolerance_m",
            "ground_distance_threshold_m",
            "statistical_std_ratio",
            "radius_outlier_radius_m",
            "critical_distance_m",
            "high_risk_distance_m",
            "moderate_risk_distance_m",
        ):
            positive_float(getattr(self, name), name)

        if self.minimum_range_m >= self.maximum_range_m:
            raise LiDARValidationError(
                "minimum_range_m must be less than maximum_range_m"
            )

        if not (
            self.critical_distance_m
            < self.high_risk_distance_m
            < self.moderate_risk_distance_m
        ):
            raise LiDARValidationError(
                "Risk thresholds must be strictly increasing"
            )

        for name in (
            "minimum_cluster_points",
            "maximum_cluster_points",
            "ground_ransac_iterations",
            "statistical_neighbors",
            "radius_outlier_min_neighbors",
            "maximum_input_points",
            "maximum_processing_points",
            "maximum_history_frames",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise LiDARValidationError(
                    f"{name} must be a positive integer"
                )

        if (
            self.minimum_cluster_points
            > self.maximum_cluster_points
        ):
            raise LiDARValidationError(
                "Invalid cluster size limits"
            )

        if (
            self.maximum_processing_points
            > self.maximum_input_points
        ):
            raise LiDARValidationError(
                "Processing limit cannot exceed input limit"
            )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class LiDARAccessContext:
    tenant_id: str
    user_id: str

    allowed_robot_ids: frozenset[str] = frozenset()
    allowed_sensor_ids: frozenset[str] = frozenset()

    can_process: bool = False
    can_view: bool = False

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id, "tenant_id")
        validate_identifier(self.user_id, "user_id")


# ============================================================
# POINT CLOUD DATA
# ============================================================

@dataclass(frozen=True)
class PointCloud:
    points: np.ndarray = field(repr=False)

    frame_id: str = "lidar"
    timestamp: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not isinstance(self.points, np.ndarray):
            raise LiDARValidationError(
                "Point cloud must be a NumPy array"
            )

        if (
            self.points.ndim != 2
            or self.points.shape[1] not in (3, 4)
        ):
            raise LiDARValidationError(
                "Point cloud must have shape (N, 3) or (N, 4)"
            )

        if not np.issubdtype(
            self.points.dtype,
            np.number,
        ):
            raise LiDARValidationError(
                "Point cloud must contain numeric values"
            )

        validate_identifier(self.frame_id, "frame_id")

    @property
    def count(self) -> int:
        return int(self.points.shape[0])

    @property
    def xyz(self) -> np.ndarray:
        return self.points[:, :3]

    @property
    def has_intensity(self) -> bool:
        return self.points.shape[1] == 4

    def copy(self) -> PointCloud:
        return PointCloud(
            points=self.points.copy(),
            frame_id=self.frame_id,
            timestamp=self.timestamp,
        )


# ============================================================
# LiDAR SCAN DATA
# ============================================================

@dataclass(frozen=True)
class LiDARScan2D:
    ranges_m: np.ndarray = field(repr=False)

    angle_min_rad: float = -math.pi
    angle_increment_rad: float = math.pi / 180.0

    range_min_m: float = 0.1
    range_max_m: float = 100.0

    frame_id: str = "lidar"

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.ranges_m, np.ndarray)
            or self.ranges_m.ndim != 1
        ):
            raise LiDARValidationError(
                "2D scan ranges must be a 1D NumPy array"
            )

        finite_float(
            self.angle_min_rad,
            "angle_min_rad",
        )

        positive_float(
            self.angle_increment_rad,
            "angle_increment_rad",
        )

        positive_float(
            self.range_min_m,
            "range_min_m",
        )

        positive_float(
            self.range_max_m,
            "range_max_m",
        )

        if self.range_min_m >= self.range_max_m:
            raise LiDARValidationError(
                "Invalid scan range limits"
            )


# ============================================================
# POINT CLOUD VALIDATOR
# ============================================================

class PointCloudValidator:

    def __init__(
        self,
        config: LiDARConfig,
    ) -> None:
        self.config = config

    def validate(
        self,
        cloud: PointCloud,
    ) -> PointCloud:
        if cloud.count > self.config.maximum_input_points:
            raise LiDARCapacityError(
                "Point cloud exceeds maximum input points"
            )

        if not np.all(np.isfinite(cloud.xyz)):
            raise LiDARValidationError(
                "Point cloud contains invalid XYZ coordinates"
            )

        if (
            cloud.has_intensity
            and not np.all(np.isfinite(cloud.points[:, 3]))
        ):
            raise LiDARValidationError(
                "Point cloud contains invalid intensity values"
            )

        return cloud


# ============================================================
# 2D SCAN CONVERSION
# ============================================================

class LiDARScanConverter:

    @staticmethod
    def scan_to_point_cloud(
        scan: LiDARScan2D,
    ) -> PointCloud:
        ranges = np.asarray(
            scan.ranges_m,
            dtype=np.float64,
        )

        angles = (
            scan.angle_min_rad
            + np.arange(ranges.size)
            * scan.angle_increment_rad
        )

        valid = (
            np.isfinite(ranges)
            & (ranges >= scan.range_min_m)
            & (ranges <= scan.range_max_m)
        )

        valid_ranges = ranges[valid]
        valid_angles = angles[valid]

        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)

        z = np.zeros_like(x)

        points = np.column_stack(
            (x, y, z)
        ).astype(np.float32)

        return PointCloud(
            points=points,
            frame_id=scan.frame_id,
            timestamp=scan.timestamp,
        )


# ============================================================
# POINT CLOUD FILTERING
# ============================================================

class PointCloudFilter:

    @staticmethod
    def range_filter(
        cloud: PointCloud,
        minimum_range_m: float,
        maximum_range_m: float,
    ) -> PointCloud:
        minimum_range_m = positive_float(
            minimum_range_m,
            "minimum_range_m",
        )

        maximum_range_m = positive_float(
            maximum_range_m,
            "maximum_range_m",
        )

        if minimum_range_m >= maximum_range_m:
            raise LiDARValidationError(
                "Invalid range filter limits"
            )

        distances = np.linalg.norm(
            cloud.xyz,
            axis=1,
        )

        valid = (
            np.isfinite(distances)
            & (distances >= minimum_range_m)
            & (distances <= maximum_range_m)
        )

        return PointCloud(
            points=cloud.points[valid].copy(),
            frame_id=cloud.frame_id,
            timestamp=cloud.timestamp,
        )

    @staticmethod
    def height_filter(
        cloud: PointCloud,
        minimum_z: float,
        maximum_z: float,
    ) -> PointCloud:
        minimum_z = finite_float(
            minimum_z,
            "minimum_z",
        )

        maximum_z = finite_float(
            maximum_z,
            "maximum_z",
        )

        if minimum_z >= maximum_z:
            raise LiDARValidationError(
                "Invalid height filter limits"
            )

        z = cloud.xyz[:, 2]

        valid = (
            (z >= minimum_z)
            & (z <= maximum_z)
        )

        return PointCloud(
            points=cloud.points[valid].copy(),
            frame_id=cloud.frame_id,
            timestamp=cloud.timestamp,
        )

    @staticmethod
    def voxel_downsample(
        cloud: PointCloud,
        voxel_size_m: float,
    ) -> PointCloud:
        voxel_size_m = positive_float(
            voxel_size_m,
            "voxel_size_m",
        )

        if cloud.count == 0:
            return cloud.copy()

        voxel_indices = np.floor(
            cloud.xyz / voxel_size_m
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
        ).astype(np.float64)

        output = np.empty(
            (voxel_count, cloud.points.shape[1]),
            dtype=np.float64,
        )

        for column in range(cloud.points.shape[1]):
            sums = np.bincount(
                inverse,
                weights=cloud.points[:, column],
                minlength=voxel_count,
            )

            output[:, column] = sums / counts

        return PointCloud(
            points=output.astype(np.float32),
            frame_id=cloud.frame_id,
            timestamp=cloud.timestamp,
        )

    @staticmethod
    def statistical_outlier_filter(
        cloud: PointCloud,
        *,
        neighbors: int = 12,
        std_ratio: float = 2.0,
    ) -> PointCloud:
        if cKDTree is None:
            raise LiDARProcessingError(
                "SciPy required for statistical outlier filtering"
            )

        if cloud.count <= neighbors:
            return cloud.copy()

        tree = cKDTree(cloud.xyz)

        distances, _ = tree.query(
            cloud.xyz,
            k=neighbors + 1,
        )

        mean_distances = np.mean(
            distances[:, 1:],
            axis=1,
        )

        threshold = (
            float(np.mean(mean_distances))
            + std_ratio * float(np.std(mean_distances))
        )

        valid = mean_distances <= threshold

        return PointCloud(
            points=cloud.points[valid].copy(),
            frame_id=cloud.frame_id,
            timestamp=cloud.timestamp,
        )

    @staticmethod
    def radius_outlier_filter(
        cloud: PointCloud,
        *,
        radius_m: float = 0.5,
        minimum_neighbors: int = 3,
    ) -> PointCloud:
        if cKDTree is None:
            raise LiDARProcessingError(
                "SciPy required for radius filtering"
            )

        radius_m = positive_float(
            radius_m,
            "radius_m",
        )

        if cloud.count == 0:
            return cloud.copy()

        tree = cKDTree(cloud.xyz)

        neighbors = tree.query_ball_point(
            cloud.xyz,
            r=radius_m,
        )

        valid = np.array(
            [
                len(indices) - 1 >= minimum_neighbors
                for indices in neighbors
            ],
            dtype=bool,
        )

        return PointCloud(
            points=cloud.points[valid].copy(),
            frame_id=cloud.frame_id,
            timestamp=cloud.timestamp,
        )


# ============================================================
# POINT CLOUD TRANSFORMATIONS
# ============================================================

@dataclass(frozen=True)
class RigidTransform:
    rotation: np.ndarray = field(repr=False)
    translation: np.ndarray = field(repr=False)

    source_frame: str = "lidar"
    target_frame: str = "base_link"

    def __post_init__(self) -> None:
        rotation = np.asarray(
            self.rotation,
            dtype=np.float64,
        )

        translation = np.asarray(
            self.translation,
            dtype=np.float64,
        )

        if rotation.shape != (3, 3):
            raise LiDARValidationError(
                "Rotation must be 3x3"
            )

        if translation.shape != (3,):
            raise LiDARValidationError(
                "Translation must have shape (3,)"
            )

        if not (
            np.all(np.isfinite(rotation))
            and np.all(np.isfinite(translation))
        ):
            raise LiDARValidationError(
                "Transform contains invalid values"
            )

        if not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=1e-5,
        ):
            raise LiDARValidationError(
                "Rotation matrix is not orthonormal"
            )

        if not np.isclose(
            np.linalg.det(rotation),
            1.0,
            atol=1e-5,
        ):
            raise LiDARValidationError(
                "Rotation matrix determinant must be +1"
            )

        object.__setattr__(
            self,
            "rotation",
            rotation.copy(),
        )

        object.__setattr__(
            self,
            "translation",
            translation.copy(),
        )

    def apply(
        self,
        cloud: PointCloud,
    ) -> PointCloud:
        if cloud.frame_id != self.source_frame:
            raise LiDARValidationError(
                "Point cloud frame does not match transform"
            )

        xyz = (
            cloud.xyz @ self.rotation.T
            + self.translation
        )

        if cloud.has_intensity:
            points = np.column_stack(
                (
                    xyz,
                    cloud.points[:, 3],
                )
            )
        else:
            points = xyz

        return PointCloud(
            points=points.astype(np.float32),
            frame_id=self.target_frame,
            timestamp=cloud.timestamp,
        )


# ============================================================
# GROUND PLANE SEGMENTATION
# ============================================================

@dataclass(frozen=True)
class GroundSegmentationResult:
    ground: PointCloud
    nonground: PointCloud

    plane_normal: tuple[float, float, float] | None
    plane_offset: float | None

    ground_ratio: float


class GroundPlaneSegmenter:

    @staticmethod
    def segment(
        cloud: PointCloud,
        *,
        distance_threshold_m: float = 0.15,
        iterations: int = 100,
        seed: int = 42,
        minimum_normal_z: float = 0.85,
    ) -> GroundSegmentationResult:
        distance_threshold_m = positive_float(
            distance_threshold_m,
            "distance_threshold_m",
        )

        if iterations <= 0:
            raise LiDARValidationError(
                "iterations must be positive"
            )

        if not 0 <= minimum_normal_z <= 1:
            raise LiDARValidationError(
                "minimum_normal_z must be in [0, 1]"
            )

        if cloud.count < 3:
            return GroundSegmentationResult(
                ground=PointCloud(
                    np.empty(
                        (0, cloud.points.shape[1]),
                        dtype=np.float32,
                    ),
                    frame_id=cloud.frame_id,
                ),
                nonground=cloud.copy(),
                plane_normal=None,
                plane_offset=None,
                ground_ratio=0.0,
            )

        rng = np.random.default_rng(seed)

        points = cloud.xyz

        best_mask = np.zeros(
            cloud.count,
            dtype=bool,
        )

        best_normal = None
        best_offset = None

        for _ in range(iterations):
            indices = rng.choice(
                cloud.count,
                size=3,
                replace=False,
            )

            p1, p2, p3 = points[indices]

            normal = np.cross(
                p2 - p1,
                p3 - p1,
            )

            norm = float(
                np.linalg.norm(normal)
            )

            if norm < 1e-8:
                continue

            normal /= norm

            if abs(normal[2]) < minimum_normal_z:
                continue

            offset = -float(
                np.dot(normal, p1)
            )

            distances = np.abs(
                points @ normal + offset
            )

            mask = distances <= distance_threshold_m

            if np.count_nonzero(mask) > np.count_nonzero(
                best_mask
            ):
                best_mask = mask
                best_normal = normal.copy()
                best_offset = offset

        ground = PointCloud(
            points=cloud.points[best_mask].copy(),
            frame_id=cloud.frame_id,
            timestamp=cloud.timestamp,
        )

        nonground = PointCloud(
            points=cloud.points[~best_mask].copy(),
            frame_id=cloud.frame_id,
            timestamp=cloud.timestamp,
        )

        return GroundSegmentationResult(
            ground=ground,
            nonground=nonground,
            plane_normal=(
                tuple(float(v) for v in best_normal)
                if best_normal is not None
                else None
            ),
            plane_offset=best_offset,
            ground_ratio=(
                ground.count / cloud.count
                if cloud.count
                else 0.0
            ),
        )


# ============================================================
# EUCLIDEAN CLUSTERING
# ============================================================

@dataclass(frozen=True)
class PointCluster:
    cluster_id: str
    point_indices: np.ndarray = field(repr=False)
    points: np.ndarray = field(repr=False)

    @property
    def point_count(self) -> int:
        return int(self.points.shape[0])

    @property
    def centroid(self) -> tuple[float, float, float]:
        center = np.mean(
            self.points[:, :3],
            axis=0,
        )

        return tuple(
            float(value)
            for value in center
        )


class EuclideanClusterExtractor:

    @staticmethod
    def extract(
        cloud: PointCloud,
        *,
        tolerance_m: float = 0.6,
        minimum_points: int = 5,
        maximum_points: int = 100_000,
    ) -> tuple[PointCluster, ...]:
        if cKDTree is None:
            raise LiDARProcessingError(
                "SciPy required for Euclidean clustering"
            )

        tolerance_m = positive_float(
            tolerance_m,
            "tolerance_m",
        )

        if cloud.count == 0:
            return ()

        tree = cKDTree(cloud.xyz)

        visited = np.zeros(
            cloud.count,
            dtype=bool,
        )

        clusters: list[PointCluster] = []

        for start in range(cloud.count):
            if visited[start]:
                continue

            visited[start] = True

            queue = deque([start])
            indices = []

            while queue:
                current = queue.popleft()
                indices.append(current)

                neighbors = tree.query_ball_point(
                    cloud.xyz[current],
                    r=tolerance_m,
                )

                for neighbor in neighbors:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

            if not (
                minimum_points
                <= len(indices)
                <= maximum_points
            ):
                continue

            index_array = np.asarray(
                indices,
                dtype=np.int64,
            )

            clusters.append(
                PointCluster(
                    cluster_id=generate_id(),
                    point_indices=index_array,
                    points=cloud.points[index_array].copy(),
                )
            )

        return tuple(clusters)


# ============================================================
# OBSTACLE DETECTION
# ============================================================

@dataclass(frozen=True)
class LiDARObstacle:
    obstacle_id: str

    centroid: tuple[float, float, float]

    minimum_xyz: tuple[float, float, float]
    maximum_xyz: tuple[float, float, float]

    dimensions_xyz: tuple[float, float, float]

    point_count: int

    minimum_distance_m: float
    centroid_distance_m: float

    risk_level: ObstacleRiskLevel

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class LiDARObstacleDetector:

    def __init__(
        self,
        config: LiDARConfig,
    ) -> None:
        self.config = config

    def risk_level(
        self,
        distance_m: float,
    ) -> ObstacleRiskLevel:
        if distance_m < self.config.critical_distance_m:
            return ObstacleRiskLevel.CRITICAL

        if distance_m < self.config.high_risk_distance_m:
            return ObstacleRiskLevel.HIGH

        if distance_m < self.config.moderate_risk_distance_m:
            return ObstacleRiskLevel.MODERATE

        return ObstacleRiskLevel.LOW

    def detect(
        self,
        clusters: Sequence[PointCluster],
    ) -> tuple[LiDARObstacle, ...]:
        obstacles = []

        for cluster in clusters:
            xyz = cluster.points[:, :3]

            minimum = np.min(
                xyz,
                axis=0,
            )

            maximum = np.max(
                xyz,
                axis=0,
            )

            centroid = np.mean(
                xyz,
                axis=0,
            )

            dimensions = maximum - minimum

            distances = np.linalg.norm(
                xyz,
                axis=1,
            )

            minimum_distance = float(
                np.min(distances)
            )

            centroid_distance = float(
                np.linalg.norm(centroid)
            )

            obstacles.append(
                LiDARObstacle(
                    obstacle_id=generate_id(),
                    centroid=tuple(
                        float(v)
                        for v in centroid
                    ),
                    minimum_xyz=tuple(
                        float(v)
                        for v in minimum
                    ),
                    maximum_xyz=tuple(
                        float(v)
                        for v in maximum
                    ),
                    dimensions_xyz=tuple(
                        float(v)
                        for v in dimensions
                    ),
                    point_count=cluster.point_count,
                    minimum_distance_m=minimum_distance,
                    centroid_distance_m=centroid_distance,
                    risk_level=self.risk_level(
                        minimum_distance
                    ),
                    metadata={
                        "cluster_id": cluster.cluster_id,
                        "distance_reference": (
                            "sensor_origin"
                        ),
                    },
                )
            )

        obstacles.sort(
            key=lambda obstacle: (
                obstacle.minimum_distance_m
            )
        )

        return tuple(obstacles)


# ============================================================
# OCCUPANCY GRID
# ============================================================

@dataclass(frozen=True)
class OccupancyGridConfig:
    width: int = 200
    height: int = 200

    resolution_m: float = 0.1

    origin_x_m: float = -10.0
    origin_y_m: float = -10.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise LiDARValidationError(
                "Invalid occupancy grid dimensions"
            )

        positive_float(
            self.resolution_m,
            "resolution_m",
        )


@dataclass(frozen=True)
class OccupancyGrid:
    grid: np.ndarray = field(repr=False)
    config: OccupancyGridConfig = field(
        default_factory=OccupancyGridConfig
    )

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "width": self.config.width,
            "height": self.config.height,
            "resolution_m": self.config.resolution_m,
            "origin_x_m": self.config.origin_x_m,
            "origin_y_m": self.config.origin_y_m,
            "occupied_cells": int(
                np.count_nonzero(self.grid == 100)
            ),
            "free_cells": int(
                np.count_nonzero(self.grid == 0)
            ),
            "unknown_cells": int(
                np.count_nonzero(self.grid == -1)
            ),
        }


class LiDAROccupancyGridBuilder:
    """
    Endpoint-only occupancy map.

    Cells containing LiDAR returns are marked occupied.
    Unobserved cells remain unknown.

    Free-space ray tracing is intentionally not inferred
    without validated scan origin and visibility information.
    """

    @staticmethod
    def build(
        cloud: PointCloud,
        config: OccupancyGridConfig,
    ) -> OccupancyGrid:
        grid = np.full(
            (
                config.height,
                config.width,
            ),
            -1,
            dtype=np.int8,
        )

        if cloud.count == 0:
            return OccupancyGrid(
                grid=grid,
                config=config,
            )

        x = cloud.xyz[:, 0]
        y = cloud.xyz[:, 1]

        col = np.floor(
            (x - config.origin_x_m)
            / config.resolution_m
        ).astype(np.int64)

        row = np.floor(
            (y - config.origin_y_m)
            / config.resolution_m
        ).astype(np.int64)

        valid = (
            (col >= 0)
            & (col < config.width)
            & (row >= 0)
            & (row < config.height)
        )

        grid[
            row[valid],
            col[valid],
        ] = 100

        return OccupancyGrid(
            grid=grid,
            config=config,
        )


# ============================================================
# LIDAR PROCESSING RESULT
# ============================================================

@dataclass(frozen=True)
class LiDARProcessingResult:
    processing_id: str

    tenant_id: str
    robot_id: str
    sensor_id: str

    status: LiDARFrameStatus

    original_point_count: int
    filtered_point_count: int

    ground_point_count: int
    nonground_point_count: int

    obstacles: tuple[LiDARObstacle, ...]

    closest_obstacle_distance_m: float | None

    processing_time_ms: float

    timestamp: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "processing_id": self.processing_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "sensor_id": self.sensor_id,
            "status": self.status.value,
            "original_point_count": (
                self.original_point_count
            ),
            "filtered_point_count": (
                self.filtered_point_count
            ),
            "ground_point_count": (
                self.ground_point_count
            ),
            "nonground_point_count": (
                self.nonground_point_count
            ),
            "obstacle_count": len(self.obstacles),
            "closest_obstacle_distance_m": (
                self.closest_obstacle_distance_m
            ),
            "obstacles": [
                json_safe(obstacle)
                for obstacle in self.obstacles
            ],
            "processing_time_ms": (
                self.processing_time_ms
            ),
            "timestamp": self.timestamp.isoformat(),
            "metadata": json_safe(self.metadata),
        }


# ============================================================
# LIDAR PROCESSING PIPELINE
# ============================================================

class LiDARProcessingPipeline:

    def __init__(
        self,
        config: LiDARConfig | None = None,
    ) -> None:
        self.config = config or LiDARConfig()

        self.validator = PointCloudValidator(
            self.config
        )

        self.obstacle_detector = LiDARObstacleDetector(
            self.config
        )

    def process(
        self,
        cloud: PointCloud,
        *,
        tenant_id: str,
        robot_id: str,
        sensor_id: str,
        remove_ground: bool = True,
        voxel_downsample: bool = True,
    ) -> LiDARProcessingResult:
        validate_identifier(
            tenant_id,
            "tenant_id",
        )

        validate_identifier(
            robot_id,
            "robot_id",
        )

        validate_identifier(
            sensor_id,
            "sensor_id",
        )

        started = time.perf_counter()

        self.validator.validate(cloud)

        original_count = cloud.count

        filtered = PointCloudFilter.range_filter(
            cloud,
            self.config.minimum_range_m,
            self.config.maximum_range_m,
        )

        if voxel_downsample:
            filtered = PointCloudFilter.voxel_downsample(
                filtered,
                self.config.voxel_size_m,
            )

        if (
            filtered.count
            > self.config.maximum_processing_points
        ):
            raise LiDARCapacityError(
                "Filtered cloud exceeds processing limit"
            )

        ground_count = 0

        if remove_ground:
            ground_result = GroundPlaneSegmenter.segment(
                filtered,
                distance_threshold_m=(
                    self.config.ground_distance_threshold_m
                ),
                iterations=(
                    self.config.ground_ransac_iterations
                ),
            )

            ground_count = ground_result.ground.count
            nonground = ground_result.nonground

        else:
            nonground = filtered

        clusters = EuclideanClusterExtractor.extract(
            nonground,
            tolerance_m=(
                self.config.clustering_tolerance_m
            ),
            minimum_points=(
                self.config.minimum_cluster_points
            ),
            maximum_points=(
                self.config.maximum_cluster_points
            ),
        )

        obstacles = self.obstacle_detector.detect(
            clusters
        )

        closest_distance = (
            obstacles[0].minimum_distance_m
            if obstacles
            else None
        )

        if filtered.count == 0:
            status = LiDARFrameStatus.EMPTY

        elif not obstacles:
            status = LiDARFrameStatus.DEGRADED

        else:
            status = LiDARFrameStatus.VALID

        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000.0

        return LiDARProcessingResult(
            processing_id=generate_id(),
            tenant_id=tenant_id,
            robot_id=robot_id,
            sensor_id=sensor_id,
            status=status,
            original_point_count=original_count,
            filtered_point_count=filtered.count,
            ground_point_count=ground_count,
            nonground_point_count=nonground.count,
            obstacles=obstacles,
            closest_obstacle_distance_m=closest_distance,
            processing_time_ms=elapsed_ms,
            metadata={
                "frame_id": cloud.frame_id,
                "remove_ground": remove_ground,
                "voxel_downsample": voxel_downsample,
                "advisory_only": True,
                "distance_reference": "sensor_origin",
                "no_obstacles_does_not_mean_clear": True,
            },
        )


# ============================================================
# MULTI-ROBOT LIDAR SERVICE
# ============================================================

class RoboticsLiDARService:

    def __init__(
        self,
        config: LiDARConfig | None = None,
    ) -> None:
        self.config = config or LiDARConfig()

        self.pipeline = LiDARProcessingPipeline(
            self.config
        )

        self._history: dict[
            tuple[str, str, str],
            deque[LiDARProcessingResult],
        ] = {}

        self._lock = threading.RLock()

    @staticmethod
    def _authorize(
        access: LiDARAccessContext,
        tenant_id: str,
        robot_id: str,
        sensor_id: str,
        permission: str,
    ) -> None:
        if access.tenant_id != tenant_id:
            raise LiDARAuthorizationError(
                "Cross-tenant access denied"
            )

        if robot_id not in access.allowed_robot_ids:
            raise LiDARAuthorizationError(
                "Robot access denied"
            )

        if sensor_id not in access.allowed_sensor_ids:
            raise LiDARAuthorizationError(
                "Sensor access denied"
            )

        if not getattr(access, permission, False):
            raise LiDARAuthorizationError(
                f"Missing permission: {permission}"
            )

    def process(
        self,
        cloud: PointCloud,
        *,
        robot_id: str,
        sensor_id: str,
        access: LiDARAccessContext,
        remove_ground: bool = True,
    ) -> LiDARProcessingResult:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            sensor_id,
            "can_process",
        )

        result = self.pipeline.process(
            cloud,
            tenant_id=access.tenant_id,
            robot_id=robot_id,
            sensor_id=sensor_id,
            remove_ground=remove_ground,
        )

        key = (
            access.tenant_id,
            robot_id,
            sensor_id,
        )

        with self._lock:
            history = self._history.setdefault(
                key,
                deque(
                    maxlen=self.config.maximum_history_frames
                ),
            )

            history.append(result)

        return result

    def latest(
        self,
        *,
        robot_id: str,
        sensor_id: str,
        access: LiDARAccessContext,
    ) -> LiDARProcessingResult | None:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            sensor_id,
            "can_view",
        )

        key = (
            access.tenant_id,
            robot_id,
            sensor_id,
        )

        with self._lock:
            history = self._history.get(key)

            return (
                history[-1]
                if history
                else None
            )

    def history(
        self,
        *,
        robot_id: str,
        sensor_id: str,
        access: LiDARAccessContext,
    ) -> tuple[LiDARProcessingResult, ...]:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            sensor_id,
            "can_view",
        )

        key = (
            access.tenant_id,
            robot_id,
            sensor_id,
        )

        with self._lock:
            return tuple(
                self._history.get(
                    key,
                    (),
                )
            )

    @staticmethod
    def export_json(
        result: LiDARProcessingResult,
    ) -> str:
        return json.dumps(
            result.metadata_dict(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


# ============================================================
# SYNTHETIC LIDAR GENERATOR
# ============================================================

class SyntheticLiDARGenerator:

    @staticmethod
    def generate(
        *,
        ground_points: int = 500,
        obstacle_points: int = 200,
        seed: int = 42,
    ) -> PointCloud:
        rng = np.random.default_rng(seed)

        ground_x = rng.uniform(
            -10,
            10,
            ground_points,
        )

        ground_y = rng.uniform(
            -10,
            10,
            ground_points,
        )

        ground_z = rng.normal(
            0,
            0.015,
            ground_points,
        )

        ground = np.column_stack(
            (
                ground_x,
                ground_y,
                ground_z,
            )
        )

        obstacle_1 = rng.normal(
            loc=(3.0, 2.0, 1.0),
            scale=(0.2, 0.2, 0.3),
            size=(obstacle_points, 3),
        )

        obstacle_2 = rng.normal(
            loc=(6.0, -2.0, 1.2),
            scale=(0.25, 0.25, 0.35),
            size=(obstacle_points, 3),
        )

        points = np.vstack(
            (
                ground,
                obstacle_1,
                obstacle_2,
            )
        ).astype(np.float32)

        return PointCloud(
            points=points,
            frame_id="lidar_front",
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestLiDAR(unittest.TestCase):

    def setUp(self) -> None:
        self.config = LiDARConfig()

        self.cloud = SyntheticLiDARGenerator.generate()

        self.access = LiDARAccessContext(
            tenant_id="tenant_alpha",
            user_id="robotics_engineer",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
            }),
            allowed_sensor_ids=frozenset({
                "LIDAR-001",
            }),
            can_process=True,
            can_view=True,
        )

    def test_point_cloud_validation(self) -> None:
        validated = PointCloudValidator(
            self.config
        ).validate(self.cloud)

        self.assertGreater(
            validated.count,
            0,
        )

    def test_invalid_point_cloud(self) -> None:
        with self.assertRaises(
            LiDARValidationError
        ):
            PointCloud(
                points=np.zeros(
                    (10, 2),
                    dtype=np.float32,
                )
            )

    def test_range_filter(self) -> None:
        filtered = PointCloudFilter.range_filter(
            self.cloud,
            0.1,
            20.0,
        )

        self.assertLessEqual(
            filtered.count,
            self.cloud.count,
        )

    def test_voxel_downsampling(self) -> None:
        downsampled = PointCloudFilter.voxel_downsample(
            self.cloud,
            0.2,
        )

        self.assertLessEqual(
            downsampled.count,
            self.cloud.count,
        )

    def test_height_filter(self) -> None:
        filtered = PointCloudFilter.height_filter(
            self.cloud,
            0.5,
            3.0,
        )

        self.assertTrue(
            np.all(
                filtered.xyz[:, 2] >= 0.5
            )
        )

    def test_scan_conversion(self) -> None:
        scan = LiDARScan2D(
            ranges_m=np.array(
                [1.0, 2.0, 3.0],
                dtype=np.float32,
            ),
            angle_min_rad=0.0,
            angle_increment_rad=math.pi / 2,
        )

        cloud = LiDARScanConverter.scan_to_point_cloud(
            scan
        )

        self.assertEqual(
            cloud.count,
            3,
        )

    def test_rigid_transform(self) -> None:
        transform = RigidTransform(
            rotation=np.eye(3),
            translation=np.array(
                [1.0, 0.0, 0.0]
            ),
            source_frame="lidar_front",
            target_frame="base_link",
        )

        transformed = transform.apply(
            self.cloud
        )

        np.testing.assert_allclose(
            transformed.xyz[:, 0],
            self.cloud.xyz[:, 0] + 1.0,
            atol=1e-5,
        )

    def test_ground_segmentation(self) -> None:
        result = GroundPlaneSegmenter.segment(
            self.cloud,
            iterations=100,
        )

        self.assertEqual(
            result.ground.count
            + result.nonground.count,
            self.cloud.count,
        )

    @unittest.skipIf(
        cKDTree is None,
        "SciPy unavailable",
    )
    def test_clustering(self) -> None:
        cloud = PointCloudFilter.height_filter(
            self.cloud,
            0.5,
            3.0,
        )

        clusters = EuclideanClusterExtractor.extract(
            cloud,
            tolerance_m=0.6,
            minimum_points=5,
        )

        self.assertGreaterEqual(
            len(clusters),
            2,
        )

    def test_occupancy_grid(self) -> None:
        grid = LiDAROccupancyGridBuilder.build(
            self.cloud,
            OccupancyGridConfig(),
        )

        self.assertEqual(
            grid.grid.shape,
            (200, 200),
        )

    @unittest.skipIf(
        cKDTree is None,
        "SciPy unavailable",
    )
    def test_pipeline(self) -> None:
        pipeline = LiDARProcessingPipeline(
            self.config
        )

        result = pipeline.process(
            self.cloud,
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            sensor_id="LIDAR-001",
        )

        self.assertGreater(
            result.filtered_point_count,
            0,
        )

    def test_tenant_isolation(self) -> None:
        with self.assertRaises(
            LiDARAuthorizationError
        ):
            RoboticsLiDARService._authorize(
                self.access,
                "tenant_beta",
                "ROBOT-001",
                "LIDAR-001",
                "can_process",
            )

    @unittest.skipIf(
        cKDTree is None,
        "SciPy unavailable",
    )
    def test_service_history(self) -> None:
        service = RoboticsLiDARService(
            self.config
        )

        service.process(
            self.cloud,
            robot_id="ROBOT-001",
            sensor_id="LIDAR-001",
            access=self.access,
        )

        history = service.history(
            robot_id="ROBOT-001",
            sensor_id="LIDAR-001",
            access=self.access,
        )

        self.assertEqual(
            len(history),
            1,
        )

    @unittest.skipIf(
        cKDTree is None,
        "SciPy unavailable",
    )
    def test_json_export(self) -> None:
        service = RoboticsLiDARService(
            self.config
        )

        result = service.process(
            self.cloud,
            robot_id="ROBOT-001",
            sensor_id="LIDAR-001",
            access=self.access,
        )

        payload = service.export_json(
            result
        )

        self.assertIn(
            "ROBOT-001",
            payload,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print(
        "\nPHYSICAL AI & AUTONOMOUS ROBOTICS PLATFORM"
    )

    print(
        "LiDAR Perception and Obstacle Detection\n"
    )

    cloud = SyntheticLiDARGenerator.generate(
        ground_points=1000,
        obstacle_points=300,
    )

    print(
        f"Generated LiDAR points: {cloud.count}"
    )

    access = LiDARAccessContext(
        tenant_id="tenant_alpha",
        user_id="robotics_engineer",
        allowed_robot_ids=frozenset({
            "ROBOT-001",
        }),
        allowed_sensor_ids=frozenset({
            "LIDAR-001",
        }),
        can_process=True,
        can_view=True,
    )

    service = RoboticsLiDARService()

    if cKDTree is None:
        print(
            "SciPy is required for clustering. "
            "Install it with: pip install scipy"
        )
        return

    result = service.process(
        cloud,
        robot_id="ROBOT-001",
        sensor_id="LIDAR-001",
        access=access,
    )

    print("\nLIDAR PROCESSING RESULT\n")

    print(
        service.export_json(
            result
        )
    )

    grid = LiDAROccupancyGridBuilder.build(
        cloud,
        OccupancyGridConfig(),
    )

    print("\nOCCUPANCY GRID\n")

    print(
        json.dumps(
            grid.metadata_dict(),
            indent=2,
        )
    )

    print("\nRUNNING UNIT TESTS\n")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestLiDAR
    )

    unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)


if __name__ == "__main__":
    main()
"""Point Cloud module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/perception/point_cloud.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================

Core 3D Point Cloud Data Model and Utilities

Features:
    - XYZ, XYZI, XYZRGB and XYZRGBI point clouds
    - LiDAR, RGB-D, stereo and simulation sources
    - Immutable metadata and validated sensor timestamps
    - Point cloud construction and serialization
    - Coordinate frame transformations
    - Point cloud merging
    - Point filtering and cropping
    - Voxel downsampling
    - Random sampling
    - Point cloud normalization
    - Centroid and bounding-box calculation
    - Distance and nearest-neighbor queries
    - RGB-D to 3D point cloud conversion
    - NumPy NPZ and ASCII PLY export/import
    - Multi-tenant access control
    - Bounded in-memory point cloud repository
    - Unit tests and demonstration

Dependencies:
    Python >= 3.11
    numpy

Optional:
    scipy

Safety:
    Point cloud data is observational and does not authorize
    robot movement or replace certified safety systems.
"""

from __future__ import annotations

import json
import math
import os
import threading
import unittest
import uuid

from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
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
    pass


class PointCloudValidationError(PointCloudError):
    pass


class PointCloudAuthorizationError(PointCloudError):
    pass


class PointCloudCapacityError(PointCloudError):
    pass


class PointCloudSerializationError(PointCloudError):
    pass


class PointCloudNotFoundError(PointCloudError):
    pass


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
    timestamp: datetime,
) -> datetime:
    if not isinstance(timestamp, datetime):
        raise PointCloudValidationError(
            "Timestamp must be datetime"
        )

    if (
        timestamp.tzinfo is None
        or timestamp.utcoffset() is None
    ):
        raise PointCloudValidationError(
            "Timestamp must be timezone-aware"
        )

    return timestamp.astimezone(timezone.utc)


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

    if isinstance(value, (list, tuple, deque)):
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
    FILE = "file"


class PointCloudFormat(str, Enum):
    XYZ = "xyz"
    XYZI = "xyzi"
    XYZRGB = "xyzrgb"
    XYZRGBI = "xyzrgbi"


class PointCloudCoordinateFrame(str, Enum):
    SENSOR = "sensor"
    CAMERA_OPTICAL = "camera_optical"
    BASE_LINK = "base_link"
    ODOM = "odom"
    MAP = "map"
    WORLD = "world"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class PointCloudConfig:
    maximum_points: int = 2_000_000
    maximum_repository_frames: int = 100
    maximum_export_points: int = 500_000

    minimum_valid_range_m: float = 0.0
    maximum_valid_range_m: float = 200.0

    default_voxel_size_m: float = 0.10
    default_random_seed: int = 42

    maximum_merge_timestamp_skew_ms: float = 100.0

    allow_empty_clouds: bool = True
    reject_nonfinite_points: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_points",
            "maximum_repository_frames",
            "maximum_export_points",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise PointCloudValidationError(
                    f"{name} must be a positive integer"
                )

        if self.minimum_valid_range_m < 0:
            raise PointCloudValidationError(
                "Minimum range cannot be negative"
            )

        if (
            self.maximum_valid_range_m
            <= self.minimum_valid_range_m
        ):
            raise PointCloudValidationError(
                "Maximum range must exceed minimum range"
            )

        if self.default_voxel_size_m <= 0:
            raise PointCloudValidationError(
                "Voxel size must be positive"
            )

        if self.maximum_merge_timestamp_skew_ms < 0:
            raise PointCloudValidationError(
                "Maximum timestamp skew cannot be negative"
            )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class PointCloudAccessContext:
    tenant_id: str
    user_id: str

    allowed_robot_ids: frozenset[str] = frozenset()

    can_read: bool = False
    can_write: bool = False
    can_delete: bool = False

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id, "tenant_id")
        validate_identifier(self.user_id, "user_id")


# ============================================================
# GEOMETRY MODELS
# ============================================================

@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        finite_float(self.x, "x")
        finite_float(self.y, "y")
        finite_float(self.z, "z")

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.x, self.y, self.z],
            dtype=np.float64,
        )

    def distance_to(
        self,
        other: Point3D,
    ) -> float:
        return float(
            np.linalg.norm(
                self.as_array() - other.as_array()
            )
        )


@dataclass(frozen=True)
class BoundingBox3D:
    minimum: Point3D
    maximum: Point3D

    center: Point3D
    dimensions: Point3D

    volume_m3: float

    def contains(
        self,
        point: Point3D,
    ) -> bool:
        return (
            self.minimum.x <= point.x <= self.maximum.x
            and self.minimum.y <= point.y <= self.maximum.y
            and self.minimum.z <= point.z <= self.maximum.z
        )


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    width: int
    height: int

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise PointCloudValidationError(
                "Camera focal lengths must be positive"
            )

        if self.width <= 0 or self.height <= 0:
            raise PointCloudValidationError(
                "Camera dimensions must be positive"
            )

        for name in ("fx", "fy", "cx", "cy"):
            finite_float(getattr(self, name), name)


@dataclass(frozen=True)
class RigidTransform3D:
    rotation: np.ndarray
    translation: np.ndarray

    source_frame: str
    target_frame: str

    def __post_init__(self) -> None:
        validate_identifier(
            self.source_frame,
            "source_frame",
        )

        validate_identifier(
            self.target_frame,
            "target_frame",
        )

        rotation = np.asarray(
            self.rotation,
            dtype=np.float64,
        )

        translation = np.asarray(
            self.translation,
            dtype=np.float64,
        )

        if rotation.shape != (3, 3):
            raise PointCloudValidationError(
                "Rotation must be a 3x3 matrix"
            )

        if translation.shape != (3,):
            raise PointCloudValidationError(
                "Translation must contain three values"
            )

        if not (
            np.all(np.isfinite(rotation))
            and np.all(np.isfinite(translation))
        ):
            raise PointCloudValidationError(
                "Transform values must be finite"
            )

        if not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=1e-5,
        ):
            raise PointCloudValidationError(
                "Rotation matrix must be orthonormal"
            )

        if not np.isclose(
            np.linalg.det(rotation),
            1.0,
            atol=1e-5,
        ):
            raise PointCloudValidationError(
                "Rotation determinant must be +1"
            )

        rotation = rotation.copy()
        translation = translation.copy()

        rotation.setflags(write=False)
        translation.setflags(write=False)

        object.__setattr__(
            self,
            "rotation",
            rotation,
        )

        object.__setattr__(
            self,
            "translation",
            translation,
        )

    @classmethod
    def identity(
        cls,
        frame: str = "sensor",
    ) -> RigidTransform3D:
        return cls(
            rotation=np.eye(3),
            translation=np.zeros(3),
            source_frame=frame,
            target_frame=frame,
        )

    def as_matrix(self) -> np.ndarray:
        matrix = np.eye(
            4,
            dtype=np.float64,
        )

        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation

        return matrix

    def inverse(self) -> RigidTransform3D:
        inverse_rotation = self.rotation.T

        inverse_translation = (
            -inverse_rotation @ self.translation
        )

        return RigidTransform3D(
            rotation=inverse_rotation,
            translation=inverse_translation,
            source_frame=self.target_frame,
            target_frame=self.source_frame,
        )

    def apply(
        self,
        xyz: np.ndarray,
    ) -> np.ndarray:
        xyz = np.asarray(
            xyz,
            dtype=np.float64,
        )

        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise PointCloudValidationError(
                "XYZ points must have shape N x 3"
            )

        return (
            xyz @ self.rotation.T
            + self.translation
        )


# ============================================================
# POINT CLOUD DATA MODEL
# ============================================================

@dataclass(frozen=True)
class PointCloud:
    cloud_id: str

    tenant_id: str
    robot_id: str
    sensor_id: str

    source: PointCloudSource
    point_format: PointCloudFormat

    timestamp: datetime

    coordinate_frame: str

    points: np.ndarray

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "cloud_id",
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

        if not isinstance(
            self.point_format,
            PointCloudFormat,
        ):
            raise PointCloudValidationError(
                "Invalid point cloud format"
            )

        if not isinstance(self.points, np.ndarray):
            raise PointCloudValidationError(
                "Points must be a NumPy array"
            )

        expected_columns = {
            PointCloudFormat.XYZ: 3,
            PointCloudFormat.XYZI: 4,
            PointCloudFormat.XYZRGB: 6,
            PointCloudFormat.XYZRGBI: 7,
        }[self.point_format]

        if (
            self.points.ndim != 2
            or self.points.shape[1] != expected_columns
        ):
            raise PointCloudValidationError(
                f"{self.point_format.value} requires "
                f"N x {expected_columns} points"
            )

        if not np.issubdtype(
            self.points.dtype,
            np.number,
        ):
            raise PointCloudValidationError(
                "Point values must be numeric"
            )

        if not np.all(np.isfinite(self.points)):
            raise PointCloudValidationError(
                "Point cloud contains NaN or Inf"
            )

        normalized = np.array(
            self.points,
            dtype=np.float32,
            copy=True,
        )

        normalized.setflags(write=False)

        object.__setattr__(
            self,
            "points",
            normalized,
        )

        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )

    @property
    def point_count(self) -> int:
        return int(len(self.points))

    @property
    def is_empty(self) -> bool:
        return self.point_count == 0

    @property
    def xyz(self) -> np.ndarray:
        return self.points[:, :3].copy()

    @property
    def intensity(self) -> np.ndarray | None:
        if self.point_format == PointCloudFormat.XYZI:
            return self.points[:, 3].copy()

        if self.point_format == PointCloudFormat.XYZRGBI:
            return self.points[:, 6].copy()

        return None

    @property
    def rgb(self) -> np.ndarray | None:
        if self.point_format in (
            PointCloudFormat.XYZRGB,
            PointCloudFormat.XYZRGBI,
        ):
            return self.points[:, 3:6].copy()

        return None

    def copy(
        self,
        *,
        new_id: bool = True,
    ) -> PointCloud:
        return replace(
            self,
            cloud_id=(
                generate_id()
                if new_id
                else self.cloud_id
            ),
            points=self.points.copy(),
            metadata=dict(self.metadata),
        )

    def with_points(
        self,
        points: np.ndarray,
        *,
        operation: str,
    ) -> PointCloud:
        metadata = dict(self.metadata)

        metadata["last_operation"] = operation
        metadata["parent_cloud_id"] = self.cloud_id

        return replace(
            self,
            cloud_id=generate_id(),
            points=np.asarray(
                points,
                dtype=np.float32,
            ),
            metadata=metadata,
        )

    def to_dict(
        self,
        *,
        include_points: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "cloud_id": self.cloud_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "sensor_id": self.sensor_id,
            "source": self.source.value,
            "point_format": self.point_format.value,
            "timestamp": self.timestamp.isoformat(),
            "coordinate_frame": self.coordinate_frame,
            "point_count": self.point_count,
            "metadata": self.metadata,
        }

        if include_points:
            payload["points"] = self.points.tolist()

        return json_safe(payload)


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
        point_format: PointCloudFormat = PointCloudFormat.XYZ,
        coordinate_frame: str = "sensor",
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PointCloud:
        return PointCloud(
            cloud_id=generate_id(),
            tenant_id=tenant_id,
            robot_id=robot_id,
            sensor_id=sensor_id,
            source=source,
            point_format=point_format,
            timestamp=timestamp or utc_now(),
            coordinate_frame=coordinate_frame,
            points=np.asarray(points),
            metadata=metadata or {},
        )

    @staticmethod
    def empty(
        *,
        tenant_id: str,
        robot_id: str,
        sensor_id: str,
        point_format: PointCloudFormat = PointCloudFormat.XYZ,
    ) -> PointCloud:
        columns = {
            PointCloudFormat.XYZ: 3,
            PointCloudFormat.XYZI: 4,
            PointCloudFormat.XYZRGB: 6,
            PointCloudFormat.XYZRGBI: 7,
        }[point_format]

        return PointCloudFactory.from_numpy(
            tenant_id=tenant_id,
            robot_id=robot_id,
            sensor_id=sensor_id,
            points=np.empty(
                (0, columns),
                dtype=np.float32,
            ),
            point_format=point_format,
        )

    @staticmethod
    def from_depth(
        *,
        tenant_id: str,
        robot_id: str,
        sensor_id: str,
        depth_map: np.ndarray,
        intrinsics: CameraIntrinsics,
        depth_scale_m: float = 1.0,
        rgb_image: np.ndarray | None = None,
        timestamp: datetime | None = None,
    ) -> PointCloud:
        depth = np.asarray(
            depth_map,
            dtype=np.float32,
        )

        if depth.ndim != 2:
            raise PointCloudValidationError(
                "Depth map must be two-dimensional"
            )

        if depth.shape != (
            intrinsics.height,
            intrinsics.width,
        ):
            raise PointCloudValidationError(
                "Depth dimensions do not match intrinsics"
            )

        scale = finite_float(
            depth_scale_m,
            "depth_scale_m",
        )

        if scale <= 0:
            raise PointCloudValidationError(
                "Depth scale must be positive"
            )

        height, width = depth.shape

        u, v = np.meshgrid(
            np.arange(width),
            np.arange(height),
        )

        z = depth * scale

        valid = (
            np.isfinite(z)
            & (z > 0)
        )

        x = (
            (u - intrinsics.cx)
            * z / intrinsics.fx
        )

        y = (
            (v - intrinsics.cy)
            * z / intrinsics.fy
        )

        xyz = np.column_stack(
            (
                x[valid],
                y[valid],
                z[valid],
            )
        )

        point_format = PointCloudFormat.XYZ

        if rgb_image is not None:
            rgb = np.asarray(rgb_image)

            if rgb.shape != (
                height,
                width,
                3,
            ):
                raise PointCloudValidationError(
                    "RGB image shape does not match depth"
                )

            if rgb.dtype != np.uint8:
                raise PointCloudValidationError(
                    "RGB image must be uint8"
                )

            colors = rgb[valid].astype(
                np.float32
            )

            xyz = np.column_stack(
                (xyz, colors)
            )

            point_format = PointCloudFormat.XYZRGB

        return PointCloudFactory.from_numpy(
            tenant_id=tenant_id,
            robot_id=robot_id,
            sensor_id=sensor_id,
            points=xyz.astype(np.float32),
            source=PointCloudSource.DEPTH_CAMERA,
            point_format=point_format,
            coordinate_frame="camera_optical",
            timestamp=timestamp,
        )


# ============================================================
# POINT CLOUD GEOMETRY
# ============================================================

class PointCloudGeometry:

    @staticmethod
    def centroid(
        cloud: PointCloud,
    ) -> Point3D | None:
        if cloud.is_empty:
            return None

        center = np.mean(
            cloud.points[:, :3],
            axis=0,
        )

        return Point3D(
            x=float(center[0]),
            y=float(center[1]),
            z=float(center[2]),
        )

    @staticmethod
    def bounding_box(
        cloud: PointCloud,
    ) -> BoundingBox3D | None:
        if cloud.is_empty:
            return None

        xyz = cloud.points[:, :3]

        minimum = np.min(
            xyz,
            axis=0,
        )

        maximum = np.max(
            xyz,
            axis=0,
        )

        center = (
            minimum + maximum
        ) / 2.0

        dimensions = (
            maximum - minimum
        )

        return BoundingBox3D(
            minimum=Point3D(*map(float, minimum)),
            maximum=Point3D(*map(float, maximum)),
            center=Point3D(*map(float, center)),
            dimensions=Point3D(
                *map(float, dimensions)
            ),
            volume_m3=float(
                np.prod(dimensions)
            ),
        )

    @staticmethod
    def ranges(
        cloud: PointCloud,
        *,
        origin: Point3D | None = None,
    ) -> np.ndarray:
        reference = (
            origin.as_array()
            if origin is not None
            else np.zeros(3)
        )

        return np.linalg.norm(
            cloud.points[:, :3] - reference,
            axis=1,
        )

    @staticmethod
    def nearest_point(
        cloud: PointCloud,
        query: Point3D,
    ) -> tuple[int, Point3D, float] | None:
        if cloud.is_empty:
            return None

        xyz = cloud.points[:, :3]

        query_array = query.as_array()

        if cKDTree is not None:
            tree = cKDTree(xyz)

            distance, index = tree.query(
                query_array,
                k=1,
            )

            index = int(index)
            distance = float(distance)

        else:
            distances = np.linalg.norm(
                xyz - query_array,
                axis=1,
            )

            index = int(
                np.argmin(distances)
            )

            distance = float(
                distances[index]
            )

        point = Point3D(
            *map(float, xyz[index])
        )

        return index, point, distance

    @staticmethod
    def statistics(
        cloud: PointCloud,
    ) -> dict[str, Any]:
        if cloud.is_empty:
            return {
                "point_count": 0,
                "centroid": None,
                "bounding_box": None,
                "minimum_range_m": None,
                "maximum_range_m": None,
                "mean_range_m": None,
            }

        ranges = PointCloudGeometry.ranges(
            cloud
        )

        return {
            "point_count": cloud.point_count,
            "centroid": json_safe(
                PointCloudGeometry.centroid(cloud)
            ),
            "bounding_box": json_safe(
                PointCloudGeometry.bounding_box(cloud)
            ),
            "minimum_range_m": float(
                np.min(ranges)
            ),
            "maximum_range_m": float(
                np.max(ranges)
            ),
            "mean_range_m": float(
                np.mean(ranges)
            ),
            "standard_deviation_range_m": float(
                np.std(ranges)
            ),
        }


# ============================================================
# POINT CLOUD OPERATIONS
# ============================================================

class PointCloudOperations:

    @staticmethod
    def remove_nonfinite(
        points: np.ndarray,
    ) -> np.ndarray:
        array = np.asarray(points)

        if array.ndim != 2:
            raise PointCloudValidationError(
                "Points must be two-dimensional"
            )

        mask = np.all(
            np.isfinite(array),
            axis=1,
        )

        return array[mask].copy()

    @staticmethod
    def range_filter(
        cloud: PointCloud,
        minimum_range_m: float,
        maximum_range_m: float,
    ) -> PointCloud:
        minimum = finite_float(
            minimum_range_m,
            "minimum_range_m",
        )

        maximum = finite_float(
            maximum_range_m,
            "maximum_range_m",
        )

        if minimum < 0 or maximum <= minimum:
            raise PointCloudValidationError(
                "Invalid range filter limits"
            )

        distances = PointCloudGeometry.ranges(
            cloud
        )

        mask = (
            (distances >= minimum)
            & (distances <= maximum)
        )

        return cloud.with_points(
            cloud.points[mask],
            operation="range_filter",
        )

    @staticmethod
    def crop(
        cloud: PointCloud,
        minimum: Point3D,
        maximum: Point3D,
    ) -> PointCloud:
        lower = minimum.as_array()
        upper = maximum.as_array()

        if np.any(lower >= upper):
            raise PointCloudValidationError(
                "Invalid crop bounds"
            )

        xyz = cloud.points[:, :3]

        mask = np.all(
            (xyz >= lower)
            & (xyz <= upper),
            axis=1,
        )

        return cloud.with_points(
            cloud.points[mask],
            operation="crop",
        )

    @staticmethod
    def voxel_downsample(
        cloud: PointCloud,
        voxel_size_m: float,
    ) -> PointCloud:
        voxel_size = finite_float(
            voxel_size_m,
            "voxel_size_m",
        )

        if voxel_size <= 0:
            raise PointCloudValidationError(
                "Voxel size must be positive"
            )

        if cloud.is_empty:
            return cloud.with_points(
                cloud.points,
                operation="voxel_downsample",
            )

        voxel_indices = np.floor(
            cloud.points[:, :3] / voxel_size
        ).astype(np.int64)

        _, inverse = np.unique(
            voxel_indices,
            axis=0,
            return_inverse=True,
        )

        voxel_count = (
            int(np.max(inverse)) + 1
        )

        counts = np.bincount(
            inverse,
            minlength=voxel_count,
        )

        output = np.empty(
            (
                voxel_count,
                cloud.points.shape[1],
            ),
            dtype=np.float32,
        )

        for column in range(
            cloud.points.shape[1]
        ):
            sums = np.bincount(
                inverse,
                weights=cloud.points[:, column],
                minlength=voxel_count,
            )

            output[:, column] = (
                sums / counts
            )

        if cloud.rgb is not None:
            output[:, 3:6] = np.clip(
                output[:, 3:6],
                0,
                255,
            )

        return cloud.with_points(
            output,
            operation="voxel_downsample",
        )

    @staticmethod
    def random_sample(
        cloud: PointCloud,
        maximum_points: int,
        *,
        seed: int = 42,
    ) -> PointCloud:
        if maximum_points <= 0:
            raise PointCloudValidationError(
                "maximum_points must be positive"
            )

        if cloud.point_count <= maximum_points:
            return cloud.copy()

        rng = np.random.default_rng(
            seed
        )

        indices = rng.choice(
            cloud.point_count,
            size=maximum_points,
            replace=False,
        )

        indices.sort()

        return cloud.with_points(
            cloud.points[indices],
            operation="random_sample",
        )

    @staticmethod
    def transform(
        cloud: PointCloud,
        transformation: RigidTransform3D,
    ) -> PointCloud:
        if (
            cloud.coordinate_frame
            != transformation.source_frame
        ):
            raise PointCloudValidationError(
                "Transform source frame does not match cloud frame"
            )

        output = cloud.points.copy()

        output[:, :3] = transformation.apply(
            cloud.points[:, :3]
        )

        transformed = cloud.with_points(
            output,
            operation="transform",
        )

        return replace(
            transformed,
            coordinate_frame=(
                transformation.target_frame
            ),
        )

    @staticmethod
    def merge(
        clouds: Sequence[PointCloud],
        *,
        maximum_timestamp_skew_ms: float = 100.0,
    ) -> PointCloud:
        if not clouds:
            raise PointCloudValidationError(
                "At least one point cloud is required"
            )

        reference = clouds[0]

        for cloud in clouds[1:]:
            if cloud.tenant_id != reference.tenant_id:
                raise PointCloudValidationError(
                    "Cannot merge clouds across tenants"
                )

            if cloud.robot_id != reference.robot_id:
                raise PointCloudValidationError(
                    "Cannot merge clouds from different robots"
                )

            if cloud.coordinate_frame != reference.coordinate_frame:
                raise PointCloudValidationError(
                    "Clouds must share a coordinate frame"
                )

            if cloud.point_format != reference.point_format:
                raise PointCloudValidationError(
                    "Cloud formats must match"
                )

        timestamps = [
            validate_timestamp(cloud.timestamp)
            for cloud in clouds
        ]

        skew_ms = (
            max(timestamps) - min(timestamps)
        ).total_seconds() * 1000.0

        if skew_ms > maximum_timestamp_skew_ms:
            raise PointCloudValidationError(
                "Cloud timestamps exceed merge skew limit"
            )

        merged = np.concatenate(
            [cloud.points for cloud in clouds],
            axis=0,
        )

        return PointCloud(
            cloud_id=generate_id(),
            tenant_id=reference.tenant_id,
            robot_id=reference.robot_id,
            sensor_id="fused_sensors",
            source=PointCloudSource.FUSED,
            point_format=reference.point_format,
            timestamp=max(timestamps),
            coordinate_frame=reference.coordinate_frame,
            points=merged,
            metadata={
                "source_cloud_ids": [
                    cloud.cloud_id
                    for cloud in clouds
                ],
                "source_count": len(clouds),
                "timestamp_skew_ms": skew_ms,
            },
        )

    @staticmethod
    def normalize_xyz(
        cloud: PointCloud,
    ) -> tuple[PointCloud, dict[str, Any]]:
        if cloud.is_empty:
            return cloud.copy(), {
                "center": [0.0, 0.0, 0.0],
                "scale": 1.0,
            }

        xyz = cloud.points[:, :3]

        center = np.mean(
            xyz,
            axis=0,
        )

        centered = xyz - center

        scale = float(
            np.max(
                np.linalg.norm(
                    centered,
                    axis=1,
                )
            )
        )

        if scale < 1e-12:
            scale = 1.0

        output = cloud.points.copy()

        output[:, :3] = (
            centered / scale
        )

        normalized = cloud.with_points(
            output,
            operation="normalize_xyz",
        )

        return normalized, {
            "center": center.tolist(),
            "scale": scale,
            "units": "normalized",
            "metric_distances_preserved": False,
        }


# ============================================================
# SERIALIZATION
# ============================================================

class PointCloudSerializer:

    @staticmethod
    def to_json(
        cloud: PointCloud,
        *,
        include_points: bool = False,
    ) -> str:
        return json.dumps(
            cloud.to_dict(
                include_points=include_points
            ),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    @staticmethod
    def save_npz(
        cloud: PointCloud,
        path: str | Path,
    ) -> None:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata = cloud.to_dict(
            include_points=False
        )

        try:
            np.savez_compressed(
                destination,
                points=cloud.points,
                metadata=np.array(
                    json.dumps(metadata)
                ),
            )

        except Exception as exc:
            raise PointCloudSerializationError(
                f"Failed to save NPZ: {exc}"
            ) from exc

    @staticmethod
    def load_npz(
        path: str | Path,
    ) -> PointCloud:
        source = Path(path)

        try:
            with np.load(
                source,
                allow_pickle=False,
            ) as archive:
                points = archive["points"].copy()

                metadata = json.loads(
                    str(
                        archive["metadata"].item()
                    )
                )

        except Exception as exc:
            raise PointCloudSerializationError(
                f"Failed to load NPZ: {exc}"
            ) from exc

        return PointCloud(
            cloud_id=metadata["cloud_id"],
            tenant_id=metadata["tenant_id"],
            robot_id=metadata["robot_id"],
            sensor_id=metadata["sensor_id"],
            source=PointCloudSource(
                metadata["source"]
            ),
            point_format=PointCloudFormat(
                metadata["point_format"]
            ),
            timestamp=datetime.fromisoformat(
                metadata["timestamp"]
            ),
            coordinate_frame=metadata[
                "coordinate_frame"
            ],
            points=points,
            metadata=metadata.get(
                "metadata",
                {},
            ),
        )

    @staticmethod
    def save_ply_ascii(
        cloud: PointCloud,
        path: str | Path,
    ) -> None:
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        has_rgb = cloud.rgb is not None

        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {cloud.point_count}",
            "property float x",
            "property float y",
            "property float z",
        ]

        if has_rgb:
            header.extend(
                [
                    "property uchar red",
                    "property uchar green",
                    "property uchar blue",
                ]
            )

        header.append("end_header")

        try:
            with destination.open(
                "w",
                encoding="utf-8",
            ) as file:
                file.write(
                    "\n".join(header) + "\n"
                )

                for point in cloud.points:
                    xyz = (
                        f"{point[0]:.8f} "
                        f"{point[1]:.8f} "
                        f"{point[2]:.8f}"
                    )

                    if has_rgb:
                        rgb = np.clip(
                            np.rint(point[3:6]),
                            0,
                            255,
                        ).astype(np.uint8)

                        xyz += (
                            f" {int(rgb[0])}"
                            f" {int(rgb[1])}"
                            f" {int(rgb[2])}"
                        )

                    file.write(
                        xyz + "\n"
                    )

        except OSError as exc:
            raise PointCloudSerializationError(
                f"Failed to write PLY: {exc}"
            ) from exc


# ============================================================
# MULTI-TENANT POINT CLOUD REPOSITORY
# ============================================================

class PointCloudRepository:

    def __init__(
        self,
        config: PointCloudConfig | None = None,
    ) -> None:
        self.config = (
            config or PointCloudConfig()
        )

        self._clouds: dict[
            tuple[str, str],
            deque[PointCloud],
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

    def save(
        self,
        cloud: PointCloud,
        *,
        access: PointCloudAccessContext,
    ) -> None:
        self._authorize(
            access,
            cloud.tenant_id,
            cloud.robot_id,
            "can_write",
        )

        if cloud.point_count > self.config.maximum_points:
            raise PointCloudCapacityError(
                "Cloud exceeds repository point limit"
            )

        if (
            cloud.is_empty
            and not self.config.allow_empty_clouds
        ):
            raise PointCloudValidationError(
                "Empty point clouds are disabled"
            )

        key = (
            cloud.tenant_id,
            cloud.robot_id,
        )

        with self._lock:
            history = self._clouds.setdefault(
                key,
                deque(
                    maxlen=self.config.maximum_repository_frames
                ),
            )

            if any(
                item.cloud_id == cloud.cloud_id
                for item in history
            ):
                raise PointCloudValidationError(
                    "Duplicate cloud ID"
                )

            history.append(
                cloud.copy(new_id=False)
            )

    def get(
        self,
        cloud_id: str,
        *,
        robot_id: str,
        access: PointCloudAccessContext,
    ) -> PointCloud:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_read",
        )

        key = (
            access.tenant_id,
            robot_id,
        )

        with self._lock:
            for cloud in self._clouds.get(
                key,
                (),
            ):
                if cloud.cloud_id == cloud_id:
                    return cloud.copy(
                        new_id=False
                    )

        raise PointCloudNotFoundError(
            f"Point cloud not found: {cloud_id}"
        )

    def latest(
        self,
        *,
        robot_id: str,
        access: PointCloudAccessContext,
    ) -> PointCloud | None:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_read",
        )

        key = (
            access.tenant_id,
            robot_id,
        )

        with self._lock:
            history = self._clouds.get(
                key
            )

            if not history:
                return None

            return history[-1].copy(
                new_id=False
            )

    def history(
        self,
        *,
        robot_id: str,
        access: PointCloudAccessContext,
    ) -> tuple[PointCloud, ...]:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_read",
        )

        key = (
            access.tenant_id,
            robot_id,
        )

        with self._lock:
            return tuple(
                cloud.copy(new_id=False)
                for cloud in self._clouds.get(
                    key,
                    (),
                )
            )

    def delete(
        self,
        cloud_id: str,
        *,
        robot_id: str,
        access: PointCloudAccessContext,
    ) -> bool:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_delete",
        )

        key = (
            access.tenant_id,
            robot_id,
        )

        with self._lock:
            history = self._clouds.get(
                key
            )

            if not history:
                return False

            remaining = [
                cloud
                for cloud in history
                if cloud.cloud_id != cloud_id
            ]

            if len(remaining) == len(history):
                return False

            self._clouds[key] = deque(
                remaining,
                maxlen=self.config.maximum_repository_frames,
            )

            return True

    def clear_robot(
        self,
        *,
        robot_id: str,
        access: PointCloudAccessContext,
    ) -> None:
        self._authorize(
            access,
            access.tenant_id,
            robot_id,
            "can_delete",
        )

        key = (
            access.tenant_id,
            robot_id,
        )

        with self._lock:
            self._clouds.pop(
                key,
                None,
            )


# ============================================================
# UNIT TESTS
# ============================================================

class TestPointCloud(unittest.TestCase):

    def setUp(self) -> None:
        self.points = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.1, 0.1, 0.0],
                [2.0, 0.0, 1.0],
                [2.1, 0.1, 1.0],
                [3.0, 0.0, 2.0],
            ],
            dtype=np.float32,
        )

        self.cloud = PointCloudFactory.from_numpy(
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            sensor_id="LIDAR-001",
            points=self.points,
        )

        self.access = PointCloudAccessContext(
            tenant_id="tenant_alpha",
            user_id="engineer",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
            }),
            can_read=True,
            can_write=True,
            can_delete=True,
        )

        self.repository = PointCloudRepository()

    def test_point_count(self) -> None:
        self.assertEqual(
            self.cloud.point_count,
            5,
        )

    def test_xyz(self) -> None:
        self.assertEqual(
            self.cloud.xyz.shape,
            (5, 3),
        )

    def test_invalid_shape(self) -> None:
        with self.assertRaises(
            PointCloudValidationError
        ):
            PointCloudFactory.from_numpy(
                tenant_id="tenant_alpha",
                robot_id="ROBOT-001",
                sensor_id="LIDAR-001",
                points=np.zeros(
                    (10, 2),
                    dtype=np.float32,
                ),
            )

    def test_nonfinite_rejection(self) -> None:
        with self.assertRaises(
            PointCloudValidationError
        ):
            PointCloudFactory.from_numpy(
                tenant_id="tenant_alpha",
                robot_id="ROBOT-001",
                sensor_id="LIDAR-001",
                points=np.array(
                    [[np.nan, 0, 0]]
                ),
            )

    def test_immutable_points(self) -> None:
        with self.assertRaises(
            ValueError
        ):
            self.cloud.points[0, 0] = 99.0

    def test_centroid(self) -> None:
        center = PointCloudGeometry.centroid(
            self.cloud
        )

        self.assertIsNotNone(center)

    def test_bounding_box(self) -> None:
        box = PointCloudGeometry.bounding_box(
            self.cloud
        )

        self.assertAlmostEqual(
            box.minimum.x,
            1.0,
        )

    def test_nearest_point(self) -> None:
        result = PointCloudGeometry.nearest_point(
            self.cloud,
            Point3D(
                1.05,
                0.0,
                0.0,
            ),
        )

        self.assertIsNotNone(result)

        self.assertLess(
            result[2],
            0.2,
        )

    def test_range_filter(self) -> None:
        filtered = PointCloudOperations.range_filter(
            self.cloud,
            0.5,
            2.5,
        )

        self.assertLess(
            filtered.point_count,
            self.cloud.point_count,
        )

    def test_crop(self) -> None:
        cropped = PointCloudOperations.crop(
            self.cloud,
            Point3D(
                0.0,
                -1.0,
                -1.0,
            ),
            Point3D(
                1.5,
                1.0,
                1.0,
            ),
        )

        self.assertEqual(
            cropped.point_count,
            2,
        )

    def test_voxel_downsample(self) -> None:
        output = PointCloudOperations.voxel_downsample(
            self.cloud,
            0.5,
        )

        self.assertLessEqual(
            output.point_count,
            self.cloud.point_count,
        )

    def test_random_sample(self) -> None:
        output = PointCloudOperations.random_sample(
            self.cloud,
            3,
        )

        self.assertEqual(
            output.point_count,
            3,
        )

    def test_transform(self) -> None:
        transform = RigidTransform3D(
            rotation=np.eye(3),
            translation=np.array(
                [1.0, 2.0, 3.0]
            ),
            source_frame="sensor",
            target_frame="base_link",
        )

        output = PointCloudOperations.transform(
            self.cloud,
            transform,
        )

        self.assertEqual(
            output.coordinate_frame,
            "base_link",
        )

        np.testing.assert_allclose(
            output.points[0, :3],
            [2.0, 2.0, 3.0],
        )

    def test_transform_inverse(self) -> None:
        transform = RigidTransform3D(
            rotation=np.eye(3),
            translation=np.array(
                [1.0, 2.0, 3.0]
            ),
            source_frame="sensor",
            target_frame="base_link",
        )

        transformed = transform.apply(
            self.points
        )

        restored = transform.inverse().apply(
            transformed
        )

        np.testing.assert_allclose(
            restored,
            self.points,
            atol=1e-6,
        )

    def test_merge(self) -> None:
        merged = PointCloudOperations.merge(
            [
                self.cloud,
                self.cloud.copy(),
            ]
        )

        self.assertEqual(
            merged.point_count,
            10,
        )

    def test_normalize(self) -> None:
        normalized, metadata = (
            PointCloudOperations.normalize_xyz(
                self.cloud
            )
        )

        self.assertEqual(
            normalized.point_count,
            5,
        )

        self.assertIn(
            "scale",
            metadata,
        )

    def test_depth_conversion(self) -> None:
        depth = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=np.float32,
        )

        intrinsics = CameraIntrinsics(
            fx=100.0,
            fy=100.0,
            cx=0.5,
            cy=0.5,
            width=2,
            height=2,
        )

        cloud = PointCloudFactory.from_depth(
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            sensor_id="DEPTH-001",
            depth_map=depth,
            intrinsics=intrinsics,
        )

        self.assertEqual(
            cloud.point_count,
            4,
        )

    def test_repository_save(self) -> None:
        self.repository.save(
            self.cloud,
            access=self.access,
        )

        latest = self.repository.latest(
            robot_id="ROBOT-001",
            access=self.access,
        )

        self.assertIsNotNone(latest)

    def test_repository_get(self) -> None:
        self.repository.save(
            self.cloud,
            access=self.access,
        )

        result = self.repository.get(
            self.cloud.cloud_id,
            robot_id="ROBOT-001",
            access=self.access,
        )

        self.assertEqual(
            result.cloud_id,
            self.cloud.cloud_id,
        )

    def test_repository_delete(self) -> None:
        self.repository.save(
            self.cloud,
            access=self.access,
        )

        deleted = self.repository.delete(
            self.cloud.cloud_id,
            robot_id="ROBOT-001",
            access=self.access,
        )

        self.assertTrue(deleted)

    def test_tenant_isolation(self) -> None:
        with self.assertRaises(
            PointCloudAuthorizationError
        ):
            self.repository._authorize(
                self.access,
                "tenant_beta",
                "ROBOT-001",
                "can_read",
            )

    def test_json_serialization(self) -> None:
        payload = PointCloudSerializer.to_json(
            self.cloud
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
        "Core 3D Point Cloud Module\n"
    )

    rng = np.random.default_rng(42)

    points = rng.normal(
        loc=[2.0, 0.0, 1.0],
        scale=[0.5, 0.5, 0.3],
        size=(1000, 3),
    ).astype(np.float32)

    cloud = PointCloudFactory.from_numpy(
        tenant_id="tenant_alpha",
        robot_id="ROBOT-001",
        sensor_id="LIDAR-FRONT-001",
        points=points,
        source=PointCloudSource.SIMULATION,
        coordinate_frame="sensor",
    )

    print("CLOUD ID")
    print(cloud.cloud_id)

    print("\nPOINT COUNT")
    print(cloud.point_count)

    print("\nGEOMETRY STATISTICS")

    print(
        json.dumps(
            PointCloudGeometry.statistics(
                cloud
            ),
            indent=2,
        )
    )

    downsampled = (
        PointCloudOperations.voxel_downsample(
            cloud,
            voxel_size_m=0.15,
        )
    )

    print("\nVOXEL DOWNSAMPLING")
    print(
        f"{cloud.point_count} -> "
        f"{downsampled.point_count}"
    )

    transform = RigidTransform3D(
        rotation=np.eye(3),
        translation=np.array(
            [1.0, 0.0, 0.5]
        ),
        source_frame="sensor",
        target_frame="base_link",
    )

    transformed = PointCloudOperations.transform(
        downsampled,
        transform,
    )

    print("\nCOORDINATE TRANSFORMATION")
    print(transformed.coordinate_frame)

    access = PointCloudAccessContext(
        tenant_id="tenant_alpha",
        user_id="robotics_engineer",
        allowed_robot_ids=frozenset({
            "ROBOT-001",
        }),
        can_read=True,
        can_write=True,
        can_delete=True,
    )

    repository = PointCloudRepository()

    repository.save(
        transformed,
        access=access,
    )

    print("\nREPOSITORY HISTORY")
    print(
        len(
            repository.history(
                robot_id="ROBOT-001",
                access=access,
            )
        )
    )

    print("\nPOINT CLOUD JSON METADATA")

    print(
        PointCloudSerializer.to_json(
            transformed,
            include_points=False,
        )
    )

    print("\nRUNNING UNIT TESTS\n")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestPointCloud
    )

    result = unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)

    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
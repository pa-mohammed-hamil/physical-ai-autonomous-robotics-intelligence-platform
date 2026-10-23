"""Coordinate Frames module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
backend/app/robotics/control/coordinate_frames.py

Python 3.11+
Standard library only.

ROBOT COORDINATE FRAME MANAGER — SIMULATION ONLY

Features:
- 3D vectors and quaternions
- Rotation and translation transforms
- Transform composition and inversion
- Coordinate frame registration
- Parent-child frame hierarchy
- Static and dynamic frames
- Frame cycle prevention
- Frame-to-frame transform lookup
- Point and pose transformation
- Dynamic transform freshness checks
- Tenant and robot isolation
- Thread-safe operations
- JSON report export

Coordinate convention:
    T_parent_child maps a point expressed
    in child coordinates into parent coordinates.

    T_A_B * T_B_C = T_A_C

Limitations:
    No real ROS 2 tf2 integration.
    No historical interpolation.
    No physical robot control.
"""

from __future__ import annotations

import json
import math
import threading
import time

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


# ============================================================
# EXCEPTIONS
# ============================================================

class CoordinateFrameError(Exception):
    pass


class FrameNotFound(CoordinateFrameError):
    pass


class FrameConflict(CoordinateFrameError):
    pass


class TransformUnavailable(CoordinateFrameError):
    pass


class TransformStale(CoordinateFrameError):
    pass


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_name(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not all(
            ch.isascii()
            and (ch.isalnum() or ch in "_-/")
            for ch in value
        )
    ):
        raise CoordinateFrameError(
            f"Invalid {label}"
        )

    return value


def finite(value: float, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise CoordinateFrameError(
            f"{label} must be finite"
        )

    return float(value)


# ============================================================
# VECTOR 3D
# ============================================================

@dataclass(frozen=True)
class Vector3:
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        for label in ("x", "y", "z"):
            finite(
                getattr(self, label),
                label,
            )

    def __add__(self, other: Vector3) -> Vector3:
        return Vector3(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __sub__(self, other: Vector3) -> Vector3:
        return Vector3(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )

    def __neg__(self) -> Vector3:
        return Vector3(
            -self.x,
            -self.y,
            -self.z,
        )

    def scale(self, factor: float) -> Vector3:
        factor = finite(factor, "factor")

        return Vector3(
            self.x * factor,
            self.y * factor,
            self.z * factor,
        )

    def norm(self) -> float:
        return math.sqrt(
            self.x**2 + self.y**2 + self.z**2
        )

    def distance_to(self, other: Vector3) -> float:
        return (self - other).norm()

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }


# ============================================================
# QUATERNION
# ============================================================

@dataclass(frozen=True)
class Quaternion:
    x: float
    y: float
    z: float
    w: float

    def __post_init__(self) -> None:
        values = (
            self.x,
            self.y,
            self.z,
            self.w,
        )

        for value in values:
            finite(value, "quaternion_component")

        if math.sqrt(
            sum(value * value for value in values)
        ) < 1e-12:
            raise CoordinateFrameError(
                "Zero quaternion is invalid"
            )

    @staticmethod
    def identity() -> Quaternion:
        return Quaternion(0, 0, 0, 1)

    @staticmethod
    def from_euler(
        roll: float,
        pitch: float,
        yaw: float,
    ) -> Quaternion:
        """
        Roll, pitch, yaw in radians.
        Intrinsic XYZ / equivalent extrinsic ZYX.
        """

        roll = finite(roll, "roll")
        pitch = finite(pitch, "pitch")
        yaw = finite(yaw, "yaw")

        cr = math.cos(roll / 2)
        sr = math.sin(roll / 2)

        cp = math.cos(pitch / 2)
        sp = math.sin(pitch / 2)

        cy = math.cos(yaw / 2)
        sy = math.sin(yaw / 2)

        return Quaternion(
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ).normalized()

    def normalized(self) -> Quaternion:
        norm = math.sqrt(
            self.x**2
            + self.y**2
            + self.z**2
            + self.w**2
        )

        return Quaternion(
            self.x / norm,
            self.y / norm,
            self.z / norm,
            self.w / norm,
        )

    def conjugate(self) -> Quaternion:
        return Quaternion(
            -self.x,
            -self.y,
            -self.z,
            self.w,
        )

    def inverse(self) -> Quaternion:
        return self.normalized().conjugate()

    def multiply(
        self,
        other: Quaternion,
    ) -> Quaternion:
        a = self.normalized()
        b = other.normalized()

        return Quaternion(
            a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
            a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
            a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
            a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        ).normalized()

    def rotate(self, point: Vector3) -> Vector3:
        q = self.normalized()

        x, y, z, w = (
            q.x,
            q.y,
            q.z,
            q.w,
        )

        return Vector3(
            (1 - 2*(y*y + z*z))*point.x
            + 2*(x*y - z*w)*point.y
            + 2*(x*z + y*w)*point.z,

            2*(x*y + z*w)*point.x
            + (1 - 2*(x*x + z*z))*point.y
            + 2*(y*z - x*w)*point.z,

            2*(x*z - y*w)*point.x
            + 2*(y*z + x*w)*point.y
            + (1 - 2*(x*x + y*y))*point.z,
        )

    def to_dict(self) -> dict[str, float]:
        q = self.normalized()

        return {
            "x": q.x,
            "y": q.y,
            "z": q.z,
            "w": q.w,
        }


# ============================================================
# TRANSFORM 3D
# ============================================================

@dataclass(frozen=True)
class Transform3D:
    translation: Vector3
    rotation: Quaternion

    @staticmethod
    def identity() -> Transform3D:
        return Transform3D(
            translation=Vector3(0, 0, 0),
            rotation=Quaternion.identity(),
        )

    def compose(
        self,
        other: Transform3D,
    ) -> Transform3D:
        return Transform3D(
            translation=(
                self.translation
                + self.rotation.rotate(
                    other.translation
                )
            ),
            rotation=self.rotation.multiply(
                other.rotation
            ),
        )

    def inverse(self) -> Transform3D:
        inverse_rotation = (
            self.rotation.inverse()
        )

        return Transform3D(
            translation=inverse_rotation.rotate(
                -self.translation
            ),
            rotation=inverse_rotation,
        )

    def apply(self, point: Vector3) -> Vector3:
        return (
            self.rotation.rotate(point)
            + self.translation
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "translation": (
                self.translation.to_dict()
            ),
            "rotation": (
                self.rotation.to_dict()
            ),
        }


# ============================================================
# POSE 3D
# ============================================================

@dataclass(frozen=True)
class Pose3D:
    position: Vector3
    orientation: Quaternion

    def to_transform(self) -> Transform3D:
        return Transform3D(
            translation=self.position,
            rotation=self.orientation,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position.to_dict(),
            "orientation": (
                self.orientation.to_dict()
            ),
        }


# ============================================================
# FRAME TYPES
# ============================================================

class FrameType(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"


# ============================================================
# CONFIG
# ============================================================

@dataclass(frozen=True)
class CoordinateFrameConfig:
    maximum_frames: int = 10000
    dynamic_timeout_seconds: float = 2.0
    simulation_only: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.maximum_frames, bool)
            or not isinstance(self.maximum_frames, int)
            or self.maximum_frames <= 0
        ):
            raise CoordinateFrameError(
                "maximum_frames must be positive"
            )

        if (
            finite(
                self.dynamic_timeout_seconds,
                "dynamic_timeout_seconds",
            ) <= 0
        ):
            raise CoordinateFrameError(
                "dynamic_timeout_seconds must be positive"
            )

        if not self.simulation_only:
            raise CoordinateFrameError(
                "Physical ROS 2 integration unsupported"
            )


# ============================================================
# FRAME RECORD
# ============================================================

@dataclass
class CoordinateFrame:
    tenant_id: str
    robot_id: str
    frame_id: str

    parent_frame: str | None

    frame_type: FrameType

    transform: Transform3D | None = None

    updated_at: datetime | None = None

    updated_monotonic: float | None = None

    update_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "frame_id": self.frame_id,
            "parent_frame": self.parent_frame,
            "frame_type": self.frame_type.value,
            "transform": (
                self.transform.to_dict()
                if self.transform
                else None
            ),
            "updated_at": (
                self.updated_at.isoformat()
                if self.updated_at
                else None
            ),
            "update_count": self.update_count,
        }


# ============================================================
# COORDINATE FRAME MANAGER
# ============================================================

class RoboticsCoordinateFrameManager:

    def __init__(
        self,
        config: CoordinateFrameConfig | None = None,
    ) -> None:
        self.config = (
            config or CoordinateFrameConfig()
        )

        self._frames: dict[
            tuple[str, str, str],
            CoordinateFrame
        ] = {}

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # KEY
    # --------------------------------------------------------

    def _key(
        self,
        tenant_id: str,
        robot_id: str,
        frame_id: str,
    ) -> tuple[str, str, str]:
        return (
            validate_name(
                tenant_id,
                "tenant_id",
            ),
            validate_name(
                robot_id,
                "robot_id",
            ),
            validate_name(
                frame_id,
                "frame_id",
            ),
        )

    def _get(
        self,
        tenant_id: str,
        robot_id: str,
        frame_id: str,
    ) -> CoordinateFrame:
        key = self._key(
            tenant_id,
            robot_id,
            frame_id,
        )

        record = self._frames.get(key)

        if record is None:
            raise FrameNotFound(
                f"Frame not found: {frame_id}"
            )

        return record

    # --------------------------------------------------------
    # REGISTER FRAME
    # --------------------------------------------------------

    def register_frame(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        frame_id: str,
        parent_frame: str | None,
        frame_type: FrameType,
        initial_transform: Transform3D | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            key = self._key(
                tenant_id,
                robot_id,
                frame_id,
            )

            if key in self._frames:
                raise FrameConflict(
                    "Frame already registered"
                )

            if (
                len(self._frames)
                >= self.config.maximum_frames
            ):
                raise CoordinateFrameError(
                    "Maximum frame capacity reached"
                )

            if not isinstance(
                frame_type,
                FrameType,
            ):
                raise CoordinateFrameError(
                    "Invalid frame type"
                )

            if (
                initial_transform is not None
                and not isinstance(
                    initial_transform,
                    Transform3D,
                )
            ):
                raise CoordinateFrameError(
                    "Invalid initial transform"
                )

            if parent_frame is not None:
                validate_name(
                    parent_frame,
                    "parent_frame",
                )

                if parent_frame == frame_id:
                    raise FrameConflict(
                        "Frame cannot parent itself"
                    )

                self._get(
                    tenant_id,
                    robot_id,
                    parent_frame,
                )

            elif initial_transform is not None:
                raise CoordinateFrameError(
                    "Root frame cannot have a parent transform"
                )

            record = CoordinateFrame(
                tenant_id=tenant_id,
                robot_id=robot_id,
                frame_id=frame_id,
                parent_frame=parent_frame,
                frame_type=frame_type,
            )

            if initial_transform is not None:
                record.transform = initial_transform
                record.updated_at = utc_now()
                record.updated_monotonic = (
                    time.monotonic()
                )
                record.update_count = 1

            self._frames[key] = record

            return record.to_dict()

    # --------------------------------------------------------
    # UPDATE TRANSFORM
    # --------------------------------------------------------

    def update_transform(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        frame_id: str,
        transform: Transform3D,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                frame_id,
            )

            if record.parent_frame is None:
                raise CoordinateFrameError(
                    "Root frame has no parent transform"
                )

            if not isinstance(
                transform,
                Transform3D,
            ):
                raise CoordinateFrameError(
                    "Invalid transform"
                )

            if (
                record.frame_type == FrameType.STATIC
                and record.update_count > 0
            ):
                raise CoordinateFrameError(
                    "Static frame transform is immutable"
                )

            record.transform = transform
            record.updated_at = utc_now()
            record.updated_monotonic = (
                time.monotonic()
            )
            record.update_count += 1

            return record.to_dict()

    # --------------------------------------------------------
    # GET FRESH TRANSFORM
    # --------------------------------------------------------

    def _require_transform(
        self,
        record: CoordinateFrame,
    ) -> Transform3D:
        if record.transform is None:
            raise TransformUnavailable(
                f"Transform missing: {record.frame_id}"
            )

        if record.frame_type == FrameType.DYNAMIC:
            last = record.updated_monotonic

            if (
                last is None
                or (
                    time.monotonic() - last
                )
                > self.config.dynamic_timeout_seconds
            ):
                raise TransformStale(
                    f"Stale transform: {record.frame_id}"
                )

        return record.transform

    # --------------------------------------------------------
    # BUILD ANCESTOR CHAIN
    # --------------------------------------------------------

    def _ancestors(
        self,
        tenant_id: str,
        robot_id: str,
        frame_id: str,
    ) -> dict[str, Transform3D]:
        chain = {
            frame_id: Transform3D.identity()
        }

        current = frame_id

        accumulated = (
            Transform3D.identity()
        )

        visited: set[str] = set()

        while True:
            if current in visited:
                raise CoordinateFrameError(
                    "Frame cycle detected"
                )

            visited.add(current)

            record = self._get(
                tenant_id,
                robot_id,
                current,
            )

            if record.parent_frame is None:
                break

            parent_to_child = (
                self._require_transform(record)
            )

            accumulated = (
                parent_to_child.compose(
                    accumulated
                )
            )

            current = record.parent_frame

            chain[current] = accumulated

        return chain

    # --------------------------------------------------------
    # LOOKUP TRANSFORM
    # --------------------------------------------------------

    def lookup_transform(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        source_frame: str,
        target_frame: str,
    ) -> dict[str, Any]:
        with self._lock:
            source_chain = self._ancestors(
                tenant_id,
                robot_id,
                source_frame,
            )

            target_chain = self._ancestors(
                tenant_id,
                robot_id,
                target_frame,
            )

            common = next(
                (
                    frame
                    for frame in source_chain
                    if frame in target_chain
                ),
                None,
            )

            if common is None:
                raise TransformUnavailable(
                    "Frames are disconnected"
                )

            transform = (
                target_chain[common]
                .inverse()
                .compose(
                    source_chain[common]
                )
            )

            source_path = list(
                source_chain
            )

            target_path = list(
                target_chain
            )

            source_to_common = (
                source_path[
                    :source_path.index(common) + 1
                ]
            )

            target_to_common = (
                target_path[
                    :target_path.index(common)
                ]
            )

            frame_path = (
                source_to_common
                + list(reversed(target_to_common))
            )

            return {
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "source_frame": source_frame,
                "target_frame": target_frame,
                "frame_path": frame_path,
                "transform": transform.to_dict(),
                "timestamp": utc_now().isoformat(),
            }

    # --------------------------------------------------------
    # TRANSFORM POINT
    # --------------------------------------------------------

    def transform_point(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        source_frame: str,
        target_frame: str,
        point: Vector3,
    ) -> Vector3:
        result = self.lookup_transform(
            tenant_id=tenant_id,
            robot_id=robot_id,
            source_frame=source_frame,
            target_frame=target_frame,
        )

        data = result["transform"]

        transform = Transform3D(
            translation=Vector3(
                **data["translation"]
            ),
            rotation=Quaternion(
                **data["rotation"]
            ),
        )

        return transform.apply(point)

    # --------------------------------------------------------
    # TRANSFORM POSE
    # --------------------------------------------------------

    def transform_pose(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        source_frame: str,
        target_frame: str,
        pose: Pose3D,
    ) -> Pose3D:
        result = self.lookup_transform(
            tenant_id=tenant_id,
            robot_id=robot_id,
            source_frame=source_frame,
            target_frame=target_frame,
        )

        data = result["transform"]

        transform = Transform3D(
            translation=Vector3(
                **data["translation"]
            ),
            rotation=Quaternion(
                **data["rotation"]
            ),
        )

        transformed = transform.compose(
            pose.to_transform()
        )

        return Pose3D(
            position=transformed.translation,
            orientation=transformed.rotation,
        )

    # --------------------------------------------------------
    # LIST FRAMES
    # --------------------------------------------------------

    def list_frames(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> list[dict[str, Any]]:
        with self._lock:
            return sorted(
                [
                    record.to_dict()
                    for record in self._frames.values()
                    if (
                        record.tenant_id == tenant_id
                        and record.robot_id == robot_id
                    )
                ],
                key=lambda item: item["frame_id"],
            )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            records = [
                record
                for record in self._frames.values()
                if (
                    record.tenant_id == tenant_id
                    and record.robot_id == robot_id
                )
            ]

            return {
                "total_frames": len(records),
                "static_frames": sum(
                    record.frame_type == FrameType.STATIC
                    for record in records
                ),
                "dynamic_frames": sum(
                    record.frame_type == FrameType.DYNAMIC
                    for record in records
                ),
                "root_frames": sum(
                    record.parent_frame is None
                    for record in records
                ),
                "frames_without_transform": sum(
                    record.parent_frame is not None
                    and record.transform is None
                    for record in records
                ),
                "simulation_only": True,
            }

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        path: str | Path,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> Path:
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            payload = {
                "generated_at": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "statistics": self.statistics(
                    tenant_id=tenant_id,
                    robot_id=robot_id,
                ),
                "frames": self.list_frames(
                    tenant_id=tenant_id,
                    robot_id=robot_id,
                ),
            }

        output.write_text(
            json.dumps(
                payload,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        return output


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    manager = RoboticsCoordinateFrameManager()

    scope = {
        "tenant_id": "tenant_alpha",
        "robot_id": "robot_001",
    }

    manager.register_frame(
        **scope,
        frame_id="map",
        parent_frame=None,
        frame_type=FrameType.STATIC,
    )

    manager.register_frame(
        **scope,
        frame_id="base_link",
        parent_frame="map",
        frame_type=FrameType.DYNAMIC,
        initial_transform=Transform3D(
            translation=Vector3(2, 3, 0),
            rotation=Quaternion.identity(),
        ),
    )

    manager.register_frame(
        **scope,
        frame_id="camera_link",
        parent_frame="base_link",
        frame_type=FrameType.STATIC,
        initial_transform=Transform3D(
            translation=Vector3(0.2, 0, 1),
            rotation=Quaternion.identity(),
        ),
    )

    result = manager.lookup_transform(
        **scope,
        source_frame="camera_link",
        target_frame="map",
    )

    point = manager.transform_point(
        **scope,
        source_frame="camera_link",
        target_frame="map",
        point=Vector3(1, 0, 0),
    )

    print(
        "Transform:",
        json.dumps(
            result,
            indent=2,
        ),
    )

    print(
        "Map coordinates:",
        point.to_dict(),
    )

    report = manager.export_json(
        "data/robotics/coordinate_frames_report.json",
        **scope,
    )

    print("Report:", report)


if __name__ == "__main__":
    main()

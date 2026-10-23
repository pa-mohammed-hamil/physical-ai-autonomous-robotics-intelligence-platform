
"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
backend/app/robotics/control/transformations.py

Python 3.11+
Standard library only.

3D ROBOTICS TRANSFORMATION ENGINE

Features:
    - 3D vector operations
    - Quaternion rotation
    - Quaternion normalization
    - Euler angle conversion
    - Rotation matrix conversion
    - Rigid-body SE(3) transformations
    - Transform composition and inversion
    - Pose transformation
    - Point-cloud transformation
    - Quaternion SLERP interpolation
    - Transform interpolation
    - Coordinate-frame graph
    - Multi-hop frame lookup
    - Frame-cycle detection
    - JSON export
    - Simulation-only demonstration

Coordinate conventions:
    - Right-handed Cartesian coordinates
    - Distances in meters
    - Angles in radians
    - Quaternion order: x, y, z, w
    - Euler convention: intrinsic XYZ,
      equivalent to Rz(yaw) Ry(pitch) Rx(roll)

Transform convention:
    T_parent_child maps coordinates
    expressed in child into parent.

    p_parent = R_parent_child * p_child + t

This module performs mathematics only.
It does not control physical robots.
"""

from __future__ import annotations

import json
import math
import re
import threading

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================
# EXCEPTIONS
# ============================================================

class TransformationError(Exception):
    pass


class TransformationValidationError(TransformationError):
    pass


class FrameNotFound(TransformationError):
    pass


class FrameConflict(TransformationError):
    pass


# ============================================================
# UTILITIES
# ============================================================

EPSILON = 1e-12


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def finite(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise TransformationValidationError(
            f"{name} must be finite"
        )
    return float(value)


def validate_frame_id(value: str) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_./-]{0,127}",
            value,
        )
        or value.endswith("/")
        or "//" in value
    ):
        raise TransformationValidationError(
            "Invalid frame ID"
        )
    return value


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(minimum, min(maximum, value))


# ============================================================
# VECTOR3
# ============================================================

@dataclass(frozen=True)
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __post_init__(self) -> None:
        finite(self.x, "x")
        finite(self.y, "y")
        finite(self.z, "z")

    def __add__(self, other: Vector3) -> Vector3:
        if not isinstance(other, Vector3):
            return NotImplemented

        return Vector3(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __sub__(self, other: Vector3) -> Vector3:
        if not isinstance(other, Vector3):
            return NotImplemented

        return Vector3(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )

    def __mul__(self, scalar: float) -> Vector3:
        value = finite(scalar, "scalar")

        return Vector3(
            self.x * value,
            self.y * value,
            self.z * value,
        )

    def __rmul__(self, scalar: float) -> Vector3:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Vector3:
        value = finite(scalar, "scalar")

        if abs(value) < EPSILON:
            raise TransformationValidationError(
                "Division by zero"
            )

        return self * (1.0 / value)

    def dot(self, other: Vector3) -> float:
        return (
            self.x * other.x
            + self.y * other.y
            + self.z * other.z
        )

    def cross(self, other: Vector3) -> Vector3:
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def norm(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self) -> Vector3:
        magnitude = self.norm()

        if magnitude < EPSILON:
            raise TransformationValidationError(
                "Cannot normalize zero vector"
            )

        return self / magnitude

    def distance_to(self, other: Vector3) -> float:
        return (self - other).norm()

    def lerp(
        self,
        other: Vector3,
        t: float,
    ) -> Vector3:
        alpha = finite(t, "t")

        if not 0 <= alpha <= 1:
            raise TransformationValidationError(
                "Interpolation factor must be 0-1"
            )

        return self * (1 - alpha) + other * alpha

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
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def __post_init__(self) -> None:
        for name in ("x", "y", "z", "w"):
            finite(getattr(self, name), name)

        if self.norm() < EPSILON:
            raise TransformationValidationError(
                "Zero quaternion is invalid"
            )

    def norm(self) -> float:
        return math.sqrt(
            self.x ** 2
            + self.y ** 2
            + self.z ** 2
            + self.w ** 2
        )

    def normalized(self) -> Quaternion:
        n = self.norm()

        return Quaternion(
            self.x / n,
            self.y / n,
            self.z / n,
            self.w / n,
        )

    def conjugate(self) -> Quaternion:
        return Quaternion(
            -self.x,
            -self.y,
            -self.z,
            self.w,
        )

    def inverse(self) -> Quaternion:
        n2 = self.norm() ** 2

        return Quaternion(
            -self.x / n2,
            -self.y / n2,
            -self.z / n2,
            self.w / n2,
        )

    def __mul__(
        self,
        other: Quaternion,
    ) -> Quaternion:
        if not isinstance(other, Quaternion):
            return NotImplemented

        ax, ay, az, aw = (
            self.x,
            self.y,
            self.z,
            self.w,
        )

        bx, by, bz, bw = (
            other.x,
            other.y,
            other.z,
            other.w,
        )

        return Quaternion(
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )

    def rotate(self, vector: Vector3) -> Vector3:
        """
        Rotate a vector by this quaternion.

        Uses the cross-product formula, avoiding
        construction of a zero pure-vector quaternion.
        """
        q = self.normalized()

        u = Vector3(q.x, q.y, q.z)

        uv = u.cross(vector)
        uuv = u.cross(uv)

        return (
            vector
            + uv * (2.0 * q.w)
            + uuv * 2.0
        )

    @staticmethod
    def from_euler(
        roll: float,
        pitch: float,
        yaw: float,
    ) -> Quaternion:
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
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy,
            w=cr * cp * cy + sr * sp * sy,
        ).normalized()

    def to_euler(self) -> tuple[float, float, float]:
        q = self.normalized()

        sinr = 2 * (
            q.w * q.x + q.y * q.z
        )

        cosr = 1 - 2 * (
            q.x ** 2 + q.y ** 2
        )

        roll = math.atan2(sinr, cosr)

        sinp = 2 * (
            q.w * q.y - q.z * q.x
        )

        pitch = math.asin(
            clamp(sinp, -1.0, 1.0)
        )

        siny = 2 * (
            q.w * q.z + q.x * q.y
        )

        cosy = 1 - 2 * (
            q.y ** 2 + q.z ** 2
        )

        yaw = math.atan2(siny, cosy)

        return roll, pitch, yaw

    def to_rotation_matrix(
        self,
    ) -> tuple[tuple[float, ...], ...]:
        q = self.normalized()

        x, y, z, w = q.x, q.y, q.z, q.w

        return (
            (
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ),
            (
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ),
            (
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ),
        )

    @staticmethod
    def slerp(
        start: Quaternion,
        end: Quaternion,
        t: float,
    ) -> Quaternion:
        alpha = finite(t, "t")

        if not 0 <= alpha <= 1:
            raise TransformationValidationError(
                "Interpolation factor must be 0-1"
            )

        a = start.normalized()
        b = end.normalized()

        dot = (
            a.x * b.x
            + a.y * b.y
            + a.z * b.z
            + a.w * b.w
        )

        # Choose shortest rotational path.
        if dot < 0:
            b = Quaternion(
                -b.x, -b.y, -b.z, -b.w
            )
            dot = -dot

        dot = clamp(dot, -1.0, 1.0)

        if dot > 0.9995:
            result = Quaternion(
                a.x + alpha * (b.x - a.x),
                a.y + alpha * (b.y - a.y),
                a.z + alpha * (b.z - a.z),
                a.w + alpha * (b.w - a.w),
            )
            return result.normalized()

        theta = math.acos(dot)
        sin_theta = math.sin(theta)

        weight_a = math.sin(
            (1 - alpha) * theta
        ) / sin_theta

        weight_b = math.sin(
            alpha * theta
        ) / sin_theta

        return Quaternion(
            a.x * weight_a + b.x * weight_b,
            a.y * weight_a + b.y * weight_b,
            a.z * weight_a + b.z * weight_b,
            a.w * weight_a + b.w * weight_b,
        ).normalized()

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "w": self.w,
        }


# ============================================================
# RIGID TRANSFORM
# ============================================================

@dataclass(frozen=True)
class Transform3D:
    translation: Vector3 = field(
        default_factory=Vector3
    )

    rotation: Quaternion = field(
        default_factory=Quaternion
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.translation,
            Vector3,
        ):
            raise TransformationValidationError(
                "Invalid translation"
            )

        if not isinstance(
            self.rotation,
            Quaternion,
        ):
            raise TransformationValidationError(
                "Invalid rotation"
            )

        object.__setattr__(
            self,
            "rotation",
            self.rotation.normalized(),
        )

    @staticmethod
    def identity() -> Transform3D:
        return Transform3D()

    def apply_point(
        self,
        point: Vector3,
    ) -> Vector3:
        return (
            self.rotation.rotate(point)
            + self.translation
        )

    def apply_vector(
        self,
        vector: Vector3,
    ) -> Vector3:
        return self.rotation.rotate(vector)

    def inverse(self) -> Transform3D:
        inverse_rotation = (
            self.rotation.inverse().normalized()
        )

        inverse_translation = (
            inverse_rotation.rotate(
                self.translation * -1.0
            )
        )

        return Transform3D(
            translation=inverse_translation,
            rotation=inverse_rotation,
        )

    def compose(
        self,
        other: Transform3D,
    ) -> Transform3D:
        """
        Compose self with other:

            T_a_c = T_a_b * T_b_c
        """

        if not isinstance(other, Transform3D):
            raise TransformationValidationError(
                "Invalid transform"
            )

        translation = (
            self.rotation.rotate(
                other.translation
            )
            + self.translation
        )

        rotation = (
            self.rotation * other.rotation
        ).normalized()

        return Transform3D(
            translation=translation,
            rotation=rotation,
        )

    def __matmul__(
        self,
        other: Transform3D,
    ) -> Transform3D:
        return self.compose(other)

    def interpolate(
        self,
        other: Transform3D,
        t: float,
    ) -> Transform3D:
        return Transform3D(
            translation=self.translation.lerp(
                other.translation,
                t,
            ),
            rotation=Quaternion.slerp(
                self.rotation,
                other.rotation,
                t,
            ),
        )

    def to_matrix4(
        self,
    ) -> tuple[tuple[float, ...], ...]:
        r = self.rotation.to_rotation_matrix()
        t = self.translation

        return (
            (r[0][0], r[0][1], r[0][2], t.x),
            (r[1][0], r[1][1], r[1][2], t.y),
            (r[2][0], r[2][1], r[2][2], t.z),
            (0.0, 0.0, 0.0, 1.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "translation": (
                self.translation.to_dict()
            ),
            "rotation": (
                self.rotation.to_dict()
            ),
            "matrix4": [
                list(row)
                for row in self.to_matrix4()
            ],
        }


# ============================================================
# POSE
# ============================================================

@dataclass(frozen=True)
class Pose3D:
    position: Vector3
    orientation: Quaternion
    frame_id: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.position,
            Vector3,
        ):
            raise TransformationValidationError(
                "Invalid pose position"
            )

        if not isinstance(
            self.orientation,
            Quaternion,
        ):
            raise TransformationValidationError(
                "Invalid pose orientation"
            )

        validate_frame_id(self.frame_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position.to_dict(),
            "orientation": (
                self.orientation.to_dict()
            ),
            "frame_id": self.frame_id,
        }


def transform_pose(
    pose: Pose3D,
    transform: Transform3D,
    target_frame: str,
) -> Pose3D:
    """
    Transform must map from pose.frame_id
    into target_frame.
    """

    validate_frame_id(target_frame)

    return Pose3D(
        position=transform.apply_point(
            pose.position
        ),
        orientation=(
            transform.rotation
            * pose.orientation
        ).normalized(),
        frame_id=target_frame,
    )


def transform_point_cloud(
    points: list[Vector3],
    transform: Transform3D,
) -> list[Vector3]:
    if not isinstance(points, list):
        raise TransformationValidationError(
            "Point cloud must be a list"
        )

    return [
        transform.apply_point(point)
        for point in points
    ]


# ============================================================
# FRAME GRAPH
# ============================================================

@dataclass(frozen=True)
class FrameTransform:
    parent_frame: str
    child_frame: str
    transform: Transform3D

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_frame_id(
            self.parent_frame
        )

        validate_frame_id(
            self.child_frame
        )

        if self.parent_frame == self.child_frame:
            raise TransformationValidationError(
                "Frame cannot parent itself"
            )

        if not isinstance(
            self.transform,
            Transform3D,
        ):
            raise TransformationValidationError(
                "Invalid frame transform"
            )

        if (
            not isinstance(self.timestamp, datetime)
            or self.timestamp.tzinfo is None
            or self.timestamp.utcoffset() is None
        ):
            raise TransformationValidationError(
                "Timezone-aware timestamp required"
            )


class RoboticsTransformationManager:

    """
    In-memory frame graph.

    Each child has one parent.

    Lookup:
        lookup_transform(target, source)

    Returns T_target_source.

    Frame transforms are latest-value snapshots,
    not time-indexed TF buffers.
    """

    def __init__(
        self,
        maximum_frames: int = 10000,
    ) -> None:
        if (
            isinstance(maximum_frames, bool)
            or not isinstance(maximum_frames, int)
            or maximum_frames <= 0
        ):
            raise TransformationValidationError(
                "Invalid maximum_frames"
            )

        self.maximum_frames = maximum_frames

        self._parents: dict[str, FrameTransform] = {}

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # CYCLE CHECK
    # --------------------------------------------------------

    def _would_create_cycle(
        self,
        parent: str,
        child: str,
    ) -> bool:
        current = parent
        visited = set()

        while current in self._parents:
            if current == child:
                return True

            if current in visited:
                return True

            visited.add(current)

            current = (
                self._parents[current].parent_frame
            )

        return current == child

    # --------------------------------------------------------
    # REGISTER / UPDATE
    # --------------------------------------------------------

    def set_transform(
        self,
        frame: FrameTransform,
    ) -> None:
        if not isinstance(
            frame,
            FrameTransform,
        ):
            raise TransformationValidationError(
                "Invalid frame transform"
            )

        with self._lock:
            existing = self._parents.get(
                frame.child_frame
            )

            if (
                existing is None
                and len(self._parents)
                >= self.maximum_frames
            ):
                raise FrameConflict(
                    "Frame capacity reached"
                )

            if self._would_create_cycle(
                frame.parent_frame,
                frame.child_frame,
            ):
                raise FrameConflict(
                    "Frame cycle detected"
                )

            self._parents[
                frame.child_frame
            ] = frame

    # --------------------------------------------------------
    # LOOKUP
    # --------------------------------------------------------

    def lookup_transform(
        self,
        target_frame: str,
        source_frame: str,
    ) -> Transform3D:
        """
        Return T_target_source.

        Uses bidirectional traversal over the
        registered parent-child transforms.
        """

        validate_frame_id(target_frame)
        validate_frame_id(source_frame)

        if target_frame == source_frame:
            return Transform3D.identity()

        with self._lock:
            graph: dict[
                str,
                list[tuple[str, Transform3D]],
            ] = {}

            for frame in self._parents.values():
                parent = frame.parent_frame
                child = frame.child_frame

                # T_parent_child
                graph.setdefault(
                    child,
                    [],
                ).append(
                    (
                        parent,
                        frame.transform,
                    )
                )

                # T_child_parent
                graph.setdefault(
                    parent,
                    [],
                ).append(
                    (
                        child,
                        frame.transform.inverse(),
                    )
                )

            if (
                source_frame not in graph
                or target_frame not in graph
            ):
                raise FrameNotFound(
                    "Source or target frame unavailable"
                )

            queue = deque(
                [
                    (
                        source_frame,
                        Transform3D.identity(),
                    )
                ]
            )

            visited = {
                source_frame
            }

            while queue:
                current, transform = queue.popleft()

                if current == target_frame:
                    return transform

                for neighbor, edge in graph.get(
                    current,
                    [],
                ):
                    if neighbor in visited:
                        continue

                    visited.add(neighbor)

                    # transform = T_current_source
                    # edge = T_neighbor_current
                    # result = T_neighbor_source
                    next_transform = (
                        edge @ transform
                    )

                    queue.append(
                        (
                            neighbor,
                            next_transform,
                        )
                    )

        raise FrameNotFound(
            "No transform path between frames"
        )

    # --------------------------------------------------------
    # TRANSFORM POSE
    # --------------------------------------------------------

    def convert_pose(
        self,
        pose: Pose3D,
        target_frame: str,
    ) -> Pose3D:
        transform = self.lookup_transform(
            target_frame,
            pose.frame_id,
        )

        return transform_pose(
            pose,
            transform,
            target_frame,
        )

    # --------------------------------------------------------
    # LIST FRAMES
    # --------------------------------------------------------

    def list_frames(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "parent_frame": frame.parent_frame,
                    "child_frame": frame.child_frame,
                    "transform": (
                        frame.transform.to_dict()
                    ),
                    "timestamp": (
                        frame.timestamp.isoformat()
                    ),
                }
                for frame in sorted(
                    self._parents.values(),
                    key=lambda item: (
                        item.parent_frame,
                        item.child_frame,
                    ),
                )
            ]

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

        payload = {
            "generated_at": utc_now().isoformat(),
            "frame_count": len(
                self.list_frames()
            ),
            "frames": self.list_frames(),
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
    manager = RoboticsTransformationManager()

    manager.set_transform(
        FrameTransform(
            parent_frame="map",
            child_frame="base_link",
            transform=Transform3D(
                translation=Vector3(
                    2.0,
                    3.0,
                    0.0,
                ),
                rotation=Quaternion.from_euler(
                    0.0,
                    0.0,
                    math.radians(45),
                ),
            ),
        )
    )

    manager.set_transform(
        FrameTransform(
            parent_frame="base_link",
            child_frame="camera_link",
            transform=Transform3D(
                translation=Vector3(
                    0.3,
                    0.0,
                    0.5,
                ),
                rotation=Quaternion(),
            ),
        )
    )

    point_in_camera = Vector3(
        1.0,
        0.0,
        0.0,
    )

    transform = manager.lookup_transform(
        "map",
        "camera_link",
    )

    point_in_map = transform.apply_point(
        point_in_camera
    )

    print(
        "Camera point in map:",
        json.dumps(
            point_in_map.to_dict(),
            indent=2,
        ),
    )

    pose = Pose3D(
        position=Vector3(
            1.0,
            0.0,
            0.0,
        ),
        orientation=Quaternion(),
        frame_id="camera_link",
    )

    converted = manager.convert_pose(
        pose,
        "map",
    )

    print(
        "\nConverted pose:",
        json.dumps(
            converted.to_dict(),
            indent=2,
        ),
    )

    report = manager.export_json(
        "data/robotics/transformations_report.json"
    )

    print("\nReport:", report)


if __name__ == "__main__":
    main()

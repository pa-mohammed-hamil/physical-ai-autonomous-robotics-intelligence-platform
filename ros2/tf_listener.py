
"""
Physical AI & Autonomous Robotics Intelligence Platform

File: backend/app/robotics/control/tf_listener.py

Python 3.11+
Standard library only.

Robotics TF Listener — Simulation Adapter

Features:
- Coordinate frame registration
- 3D translation and quaternion rotation
- Parent-child transform relationships
- Static and dynamic transforms
- Transform composition
- Inverse transformations
- Multi-frame transform lookup
- Point coordinate conversion
- Transform freshness validation
- Cycle detection
- Tenant and robot isolation
- Thread-safe operations
- Audit logging
- JSON export

SAFETY:
    Simulation-only transform listener.

    This module does not connect directly to
    ROS 2 tf2 or physical robot hardware.

    Transform data must come from trusted
    and validated simulation sources.

    This module does not perform collision
    avoidance or authorize robot movement.
"""

from __future__ import annotations

import json
import math
import threading
import time
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


# ============================================================
# EXCEPTIONS
# ============================================================

class TFListenerError(Exception):
    pass


class TFValidationError(TFListenerError):
    pass


class TFFrameNotFound(TFListenerError):
    pass


class TFTransformUnavailable(TFListenerError):
    pass


class TFStaleTransform(TFListenerError):
    pass


class TFFrameCycleError(TFListenerError):
    pass


# ============================================================
# ENUMS
# ============================================================

class TransformType(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"


class TransformStatus(str, Enum):
    AVAILABLE = "available"
    STALE = "stale"
    UNAVAILABLE = "unavailable"


class TFEventType(str, Enum):
    FRAME_REGISTERED = "frame_registered"
    TRANSFORM_UPDATED = "transform_updated"
    TRANSFORM_LOOKUP = "transform_lookup"
    TRANSFORM_STALE = "transform_stale"
    TRANSFORM_REJECTED = "transform_rejected"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_id(
    value: str,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not all(
            char.isalnum()
            or char in "_-"
            for char in value
        )
    ):
        raise TFValidationError(
            f"Invalid {name}"
        )

    return value


def validate_frame_id(
    value: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or value.startswith("/")
        or value.endswith("/")
        or "//" in value
        or not all(
            char.isalnum()
            or char in "_/"
            for char in value
        )
    ):
        raise TFValidationError(
            "Invalid coordinate frame ID"
        )

    return value


def finite(
    value: float,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise TFValidationError(
            f"{name} must be finite"
        )

    return float(value)


def validate_timestamp(
    value: datetime,
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise TFValidationError(
            "Timezone-aware timestamp required"
        )

    return value.astimezone(timezone.utc)


# ============================================================
# VECTOR 3D
# ============================================================

@dataclass(frozen=True)
class Vector3D:
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        for name in ("x", "y", "z"):
            finite(
                getattr(self, name),
                name,
            )

    def __add__(
        self,
        other: Vector3D,
    ) -> Vector3D:
        return Vector3D(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __sub__(
        self,
        other: Vector3D,
    ) -> Vector3D:
        return Vector3D(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )

    def __neg__(self) -> Vector3D:
        return Vector3D(
            -self.x,
            -self.y,
            -self.z,
        )

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

        if not all(
            math.isfinite(value)
            and not isinstance(value, bool)
            for value in values
        ):
            raise TFValidationError(
                "Quaternion values must be finite"
            )

        magnitude = math.sqrt(
            sum(value * value for value in values)
        )

        if magnitude < 1e-12:
            raise TFValidationError(
                "Zero quaternion not allowed"
            )

    def normalized(self) -> Quaternion:
        magnitude = math.sqrt(
            self.x * self.x
            + self.y * self.y
            + self.z * self.z
            + self.w * self.w
        )

        return Quaternion(
            self.x / magnitude,
            self.y / magnitude,
            self.z / magnitude,
            self.w / magnitude,
        )

    def conjugate(self) -> Quaternion:
        return Quaternion(
            -self.x,
            -self.y,
            -self.z,
            self.w,
        )

    def multiply(
        self,
        other: Quaternion,
    ) -> Quaternion:
        a = self
        b = other

        return Quaternion(
            a.w * b.x
            + a.x * b.w
            + a.y * b.z
            - a.z * b.y,

            a.w * b.y
            - a.x * b.z
            + a.y * b.w
            + a.z * b.x,

            a.w * b.z
            + a.x * b.y
            - a.y * b.x
            + a.z * b.w,

            a.w * b.w
            - a.x * b.x
            - a.y * b.y
            - a.z * b.z,
        )

    def rotate(
        self,
        vector: Vector3D,
    ) -> Vector3D:
        q = self.normalized()

        # Rotate using the equivalent 3x3
        # rotation matrix.
        xx = q.x * q.x
        yy = q.y * q.y
        zz = q.z * q.z

        xy = q.x * q.y
        xz = q.x * q.z
        yz = q.y * q.z

        wx = q.w * q.x
        wy = q.w * q.y
        wz = q.w * q.z

        return Vector3D(
            (
                (1 - 2 * (yy + zz)) * vector.x
                + 2 * (xy - wz) * vector.y
                + 2 * (xz + wy) * vector.z
            ),
            (
                2 * (xy + wz) * vector.x
                + (1 - 2 * (xx + zz)) * vector.y
                + 2 * (yz - wx) * vector.z
            ),
            (
                2 * (xz - wy) * vector.x
                + 2 * (yz + wx) * vector.y
                + (1 - 2 * (xx + yy)) * vector.z
            ),
        )

    @staticmethod
    def identity() -> Quaternion:
        return Quaternion(
            0.0,
            0.0,
            0.0,
            1.0,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "w": self.w,
        }


# ============================================================
# TRANSFORM
# ============================================================

@dataclass(frozen=True)
class Transform3D:
    translation: Vector3D
    rotation: Quaternion

    @staticmethod
    def identity() -> Transform3D:
        return Transform3D(
            translation=Vector3D(
                0.0,
                0.0,
                0.0,
            ),
            rotation=Quaternion.identity(),
        )

    def compose(
        self,
        other: Transform3D,
    ) -> Transform3D:
        """
        If self is T_A_B and other is T_B_C,
        returns T_A_C.

        Coordinate convention:
            p_A = R_A_B * p_B + t_A_B
        """

        rotated_translation = (
            self.rotation.rotate(
                other.translation
            )
        )

        return Transform3D(
            translation=(
                self.translation
                + rotated_translation
            ),
            rotation=(
                self.rotation.multiply(
                    other.rotation
                ).normalized()
            ),
        )

    def inverse(self) -> Transform3D:
        inverse_rotation = (
            self.rotation.normalized().conjugate()
        )

        inverse_translation = (
            inverse_rotation.rotate(
                -self.translation
            )
        )

        return Transform3D(
            translation=inverse_translation,
            rotation=inverse_rotation,
        )

    def apply(
        self,
        point: Vector3D,
    ) -> Vector3D:
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
                self.rotation.normalized().to_dict()
            ),
        }


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class TFListenerConfig:
    maximum_frames: int = 10000

    dynamic_transform_timeout_seconds: float = 2.0

    maximum_audit_events: int = 20000

    simulation_only: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.maximum_frames, bool)
            or not isinstance(self.maximum_frames, int)
            or self.maximum_frames <= 0
        ):
            raise TFValidationError(
                "maximum_frames must be positive"
            )

        if (
            isinstance(self.maximum_audit_events, bool)
            or not isinstance(
                self.maximum_audit_events,
                int,
            )
            or self.maximum_audit_events <= 0
        ):
            raise TFValidationError(
                "maximum_audit_events must be positive"
            )

        timeout = finite(
            self.dynamic_transform_timeout_seconds,
            "dynamic_transform_timeout_seconds",
        )

        if timeout <= 0:
            raise TFValidationError(
                "Transform timeout must be positive"
            )

        if not self.simulation_only:
            raise TFValidationError(
                "Physical ROS 2 transform listener unsupported"
            )


# ============================================================
# FRAME DEFINITION
# ============================================================

@dataclass(frozen=True)
class CoordinateFrame:
    tenant_id: str
    robot_id: str

    frame_id: str
    parent_frame_id: str | None

    transform_type: TransformType = (
        TransformType.DYNAMIC
    )

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")

        validate_frame_id(
            self.frame_id
        )

        if self.parent_frame_id is not None:
            validate_frame_id(
                self.parent_frame_id
            )

        if (
            self.parent_frame_id
            == self.frame_id
        ):
            raise TFFrameCycleError(
                "Frame cannot be its own parent"
            )

        if not isinstance(
            self.transform_type,
            TransformType,
        ):
            raise TFValidationError(
                "Invalid transform type"
            )


# ============================================================
# TRANSFORM OBSERVATION
# ============================================================

@dataclass(frozen=True)
class TransformObservation:
    tenant_id: str
    robot_id: str
    frame_id: str

    transform: Transform3D

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")
        validate_frame_id(self.frame_id)

        if not isinstance(
            self.transform,
            Transform3D,
        ):
            raise TFValidationError(
                "Invalid transform"
            )

        validate_timestamp(
            self.timestamp
        )


# ============================================================
# FRAME RECORD
# ============================================================

@dataclass
class FrameRecord:
    definition: CoordinateFrame

    transform: Transform3D | None = None

    updated_at: datetime | None = None
    updated_monotonic: float | None = None

    update_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.definition.tenant_id,
            "robot_id": self.definition.robot_id,
            "frame_id": self.definition.frame_id,
            "parent_frame_id": (
                self.definition.parent_frame_id
            ),
            "transform_type": (
                self.definition.transform_type.value
            ),
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
# LOOKUP RESULT
# ============================================================

@dataclass(frozen=True)
class TransformLookupResult:
    tenant_id: str
    robot_id: str

    target_frame: str
    source_frame: str

    transform: Transform3D

    frame_path: tuple[str, ...]

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "target_frame": self.target_frame,
            "source_frame": self.source_frame,
            "transform": self.transform.to_dict(),
            "frame_path": list(self.frame_path),
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class TFAuditEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    robot_id: str
    frame_id: str

    event_type: TFEventType
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "frame_id": self.frame_id,
            "event_type": self.event_type.value,
            "details": self.details,
        }


# ============================================================
# TF LISTENER
# ============================================================

class RoboticsTFListener:

    def __init__(
        self,
        config: TFListenerConfig | None = None,
    ) -> None:
        self.config = (
            config or TFListenerConfig()
        )

        self.frames: dict[
            tuple[str, str, str],
            FrameRecord
        ] = {}

        self.audit_events: list[
            TFAuditEvent
        ] = []

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------

    def _key(
        self,
        tenant_id: str,
        robot_id: str,
        frame_id: str,
    ) -> tuple[str, str, str]:
        return (
            validate_id(tenant_id, "tenant_id"),
            validate_id(robot_id, "robot_id"),
            validate_frame_id(frame_id),
        )

    def _get(
        self,
        tenant_id: str,
        robot_id: str,
        frame_id: str,
    ) -> FrameRecord:
        record = self.frames.get(
            self._key(
                tenant_id,
                robot_id,
                frame_id,
            )
        )

        if record is None:
            raise TFFrameNotFound(
                f"Frame not found: {frame_id}"
            )

        return record

    def _audit(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        frame_id: str,
        event_type: TFEventType,
        details: str,
    ) -> None:
        self.audit_events.append(
            TFAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                tenant_id=tenant_id,
                robot_id=robot_id,
                frame_id=frame_id,
                event_type=event_type,
                details=details,
            )
        )

        if (
            len(self.audit_events)
            > self.config.maximum_audit_events
        ):
            self.audit_events.pop(0)

    # --------------------------------------------------------
    # REGISTER FRAME
    # --------------------------------------------------------

    def register_frame(
        self,
        definition: CoordinateFrame,
    ) -> dict[str, Any]:
        with self._lock:
            key = self._key(
                definition.tenant_id,
                definition.robot_id,
                definition.frame_id,
            )

            if key in self.frames:
                raise TFValidationError(
                    "Frame already registered"
                )

            if (
                len(self.frames)
                >= self.config.maximum_frames
            ):
                raise TFListenerError(
                    "Maximum frame capacity reached"
                )

            if definition.parent_frame_id is not None:
                self._get(
                    definition.tenant_id,
                    definition.robot_id,
                    definition.parent_frame_id,
                )

            record = FrameRecord(
                definition=definition
            )

            self.frames[key] = record

            self._audit(
                tenant_id=definition.tenant_id,
                robot_id=definition.robot_id,
                frame_id=definition.frame_id,
                event_type=(
                    TFEventType.FRAME_REGISTERED
                ),
                details="Coordinate frame registered",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # UPDATE TRANSFORM
    # --------------------------------------------------------

    def update_transform(
        self,
        observation: TransformObservation,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                observation.tenant_id,
                observation.robot_id,
                observation.frame_id,
            )

            if (
                record.definition.parent_frame_id
                is None
            ):
                raise TFValidationError(
                    "Root frame cannot have a parent transform"
                )

            if (
                record.definition.transform_type
                == TransformType.STATIC
                and record.transform is not None
            ):
                raise TFValidationError(
                    "Static transform cannot be overwritten"
                )

            timestamp = validate_timestamp(
                observation.timestamp
            )

            if (
                record.updated_at is not None
                and timestamp <= record.updated_at
            ):
                raise TFValidationError(
                    "Out-of-order transform observation"
                )

            if (
                timestamp
                > utc_now()
            ):
                raise TFValidationError(
                    "Future transform timestamp rejected"
                )

            record.transform = (
                observation.transform
            )

            record.updated_at = timestamp
            record.updated_monotonic = (
                time.monotonic()
            )

            record.update_count += 1

            self._audit(
                tenant_id=observation.tenant_id,
                robot_id=observation.robot_id,
                frame_id=observation.frame_id,
                event_type=(
                    TFEventType.TRANSFORM_UPDATED
                ),
                details="Transform updated",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # VALIDATE TRANSFORM FRESHNESS
    # --------------------------------------------------------

    def _require_fresh(
        self,
        record: FrameRecord,
    ) -> Transform3D:
        if record.transform is None:
            raise TFTransformUnavailable(
                "Transform not initialized"
            )

        if (
            record.definition.transform_type
            == TransformType.DYNAMIC
        ):
            if record.updated_at is None:
                raise TFTransformUnavailable(
                    "Dynamic transform has no timestamp"
                )

            age = (
                utc_now()
                - record.updated_at
            ).total_seconds()

            if (
                age
                > self.config.dynamic_transform_timeout_seconds
            ):
                raise TFStaleTransform(
                    f"Stale transform: "
                    f"{record.definition.frame_id}"
                )

        return record.transform

    # --------------------------------------------------------
    # BUILD ANCESTOR TRANSFORMS
    # --------------------------------------------------------

    def _ancestor_transforms(
        self,
        tenant_id: str,
        robot_id: str,
        frame_id: str,
    ) -> dict[str, Transform3D]:
        """
        Returns transforms T_ancestor_frame.

        T_A_B converts a point expressed
        in B into coordinates of A.
        """

        result = {
            frame_id: Transform3D.identity()
        }

        current_frame = frame_id
        current_transform = (
            Transform3D.identity()
        )

        visited: set[str] = set()

        while True:
            if current_frame in visited:
                raise TFFrameCycleError(
                    "Cycle detected in TF tree"
                )

            visited.add(current_frame)

            record = self._get(
                tenant_id,
                robot_id,
                current_frame,
            )

            parent = (
                record.definition.parent_frame_id
            )

            if parent is None:
                break

            parent_to_child = (
                self._require_fresh(record)
            )

            current_transform = (
                parent_to_child.compose(
                    current_transform
                )
            )

            result[parent] = (
                current_transform
            )

            current_frame = parent

        return result

    # --------------------------------------------------------
    # LOOKUP TRANSFORM
    # --------------------------------------------------------

    def lookup_transform(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        target_frame: str,
        source_frame: str,
    ) -> TransformLookupResult:
        with self._lock:
            self._get(
                tenant_id,
                robot_id,
                target_frame,
            )

            self._get(
                tenant_id,
                robot_id,
                source_frame,
            )

            source_ancestors = (
                self._ancestor_transforms(
                    tenant_id,
                    robot_id,
                    source_frame,
                )
            )

            target_ancestors = (
                self._ancestor_transforms(
                    tenant_id,
                    robot_id,
                    target_frame,
                )
            )

            common = next(
                (
                    ancestor
                    for ancestor in source_ancestors
                    if ancestor in target_ancestors
                ),
                None,
            )

            if common is None:
                raise TFTransformUnavailable(
                    "Frames are disconnected"
                )

            common_to_source = (
                source_ancestors[common]
            )

            common_to_target = (
                target_ancestors[common]
            )

            target_to_source = (
                common_to_target.inverse().compose(
                    common_to_source
                )
            )

            path = (
                tuple(source_ancestors)
                + tuple(
                    frame
                    for frame in reversed(
                        tuple(target_ancestors)
                    )
                    if frame not in source_ancestors
                )
            )

            self._audit(
                tenant_id=tenant_id,
                robot_id=robot_id,
                frame_id=source_frame,
                event_type=(
                    TFEventType.TRANSFORM_LOOKUP
                ),
                details=(
                    f"{source_frame} -> {target_frame}"
                ),
            )

            return TransformLookupResult(
                tenant_id=tenant_id,
                robot_id=robot_id,
                target_frame=target_frame,
                source_frame=source_frame,
                transform=target_to_source,
                frame_path=path,
            )

    # --------------------------------------------------------
    # TRANSFORM POINT
    # --------------------------------------------------------

    def transform_point(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        target_frame: str,
        source_frame: str,
        point: Vector3D,
    ) -> Vector3D:
        lookup = self.lookup_transform(
            tenant_id=tenant_id,
            robot_id=robot_id,
            target_frame=target_frame,
            source_frame=source_frame,
        )

        return lookup.transform.apply(
            point
        )

    # --------------------------------------------------------
    # GET FRAME
    # --------------------------------------------------------

    def get_frame(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        frame_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            return self._get(
                tenant_id,
                robot_id,
                frame_id,
            ).to_dict()

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
            return [
                record.to_dict()
                for record in self.frames.values()
                if (
                    record.definition.tenant_id
                    == tenant_id
                    and record.definition.robot_id
                    == robot_id
                )
            ]

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            records = [
                record
                for record in self.frames.values()
                if (
                    tenant_id is None
                    or record.definition.tenant_id
                    == tenant_id
                )
            ]

            return {
                "total_frames": len(records),
                "static_frames": sum(
                    record.definition.transform_type
                    == TransformType.STATIC
                    for record in records
                ),
                "dynamic_frames": sum(
                    record.definition.transform_type
                    == TransformType.DYNAMIC
                    for record in records
                ),
                "initialized_transforms": sum(
                    record.transform is not None
                    for record in records
                ),
            }

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        path: str | Path,
        *,
        tenant_id: str,
    ) -> Path:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            payload = {
                "generated_at": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "statistics": self.statistics(
                    tenant_id=tenant_id
                ),
                "frames": [
                    record.to_dict()
                    for record in self.frames.values()
                    if record.definition.tenant_id
                    == tenant_id
                ],
                "audit_events": [
                    event.to_dict()
                    for event in self.audit_events
                    if event.tenant_id == tenant_id
                ],
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
    listener = RoboticsTFListener()

    tenant_id = "tenant_alpha"
    robot_id = "robot_001"

    # World root
    listener.register_frame(
        CoordinateFrame(
            tenant_id=tenant_id,
            robot_id=robot_id,
            frame_id="map",
            parent_frame_id=None,
            transform_type=TransformType.STATIC,
        )
    )

    # Robot base relative to map
    listener.register_frame(
        CoordinateFrame(
            tenant_id=tenant_id,
            robot_id=robot_id,
            frame_id="base_link",
            parent_frame_id="map",
            transform_type=TransformType.DYNAMIC,
        )
    )

    # Camera relative to robot base
    listener.register_frame(
        CoordinateFrame(
            tenant_id=tenant_id,
            robot_id=robot_id,
            frame_id="camera_link",
            parent_frame_id="base_link",
            transform_type=TransformType.STATIC,
        )
    )

    listener.update_transform(
        TransformObservation(
            tenant_id=tenant_id,
            robot_id=robot_id,
            frame_id="base_link",
            transform=Transform3D(
                translation=Vector3D(
                    2.0,
                    3.0,
                    0.0,
                ),
                rotation=Quaternion.identity(),
            ),
        )
    )

    listener.update_transform(
        TransformObservation(
            tenant_id=tenant_id,
            robot_id=robot_id,
            frame_id="camera_link",
            transform=Transform3D(
                translation=Vector3D(
                    0.2,
                    0.0,
                    1.0,
                ),
                rotation=Quaternion.identity(),
            ),
        )
    )

    camera_point = Vector3D(
        1.0,
        0.0,
        0.0,
    )

    map_point = listener.transform_point(
        tenant_id=tenant_id,
        robot_id=robot_id,
        target_frame="map",
        source_frame="camera_link",
        point=camera_point,
    )

    print(
        "Camera point:",
        camera_point.to_dict(),
    )

    print(
        "Map point:",
        map_point.to_dict(),
    )

    print(
        "\nTF lookup:"
    )

    result = listener.lookup_transform(
        tenant_id=tenant_id,
        robot_id=robot_id,
        target_frame="map",
        source_frame="camera_link",
    )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
        )
    )

    listener.export_json(
        "data/robotics/tf_listener_report.json",
        tenant_id=tenant_id,
    )


if __name__ == "__main__":
    main()

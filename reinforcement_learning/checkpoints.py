# backend/app/robotics/control/checkpoints.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Robotics Checkpoint Management Engine

Python 3.11+
Dependencies: Standard library only

Features:
    - Robot navigation checkpoints
    - Waypoint and pose checkpoint management
    - Task execution checkpoints
    - Simulation and reinforcement learning checkpoints
    - Robot state snapshots
    - Checkpoint creation, validation and retrieval
    - Checkpoint versioning
    - Mission progress tracking
    - Checkpoint dependencies
    - Checkpoint completion and failure tracking
    - Checkpoint recovery recommendations
    - Geofencing and workspace bounds validation
    - Checkpoint expiration
    - Multi-tenant isolation
    - Thread-safe in-memory storage
    - Audit logging
    - JSON export and import
    - Human approval requirements

Integrations:
    action_space.py
    robot_state.py
    robot_controller.py
    navigation_service.py
    planning.py
    task_service.py
    simulation_adapter.py
    safety_monitor.py
    geofencing.py
    workspace_limits.py
    robot_lifecycle.py
    command_validator.py

Safety:
    This module manages logical checkpoints and snapshots.

    It does not:
        - Move robots
        - Restore hardware state
        - Dispatch physical commands
        - Bypass safety interlocks
        - Guarantee collision-free navigation

    Checkpoint restoration produces advisory recovery
    information only. Physical execution requires separate
    safety checks and authorization.
"""

from __future__ import annotations

import copy
import json
import math
import threading
import uuid

from collections import Counter, deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class CheckpointError(Exception):
    pass


class CheckpointValidationError(CheckpointError):
    pass


class CheckpointNotFound(CheckpointError):
    pass


class CheckpointConflict(CheckpointError):
    pass


class CheckpointPermissionDenied(CheckpointError):
    pass


# ============================================================
# ENUMS
# ============================================================

class CheckpointType(str, Enum):
    NAVIGATION = "navigation"
    TASK = "task"
    ROBOT_STATE = "robot_state"
    SIMULATION = "simulation"
    TRAINING = "training"
    MAINTENANCE = "maintenance"
    INSPECTION = "inspection"
    SAFETY = "safety"
    CUSTOM = "custom"


class CheckpointStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class CheckpointPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStatus(str, Enum):
    AVAILABLE = "available"
    REQUIRES_REVIEW = "requires_review"
    UNAVAILABLE = "unavailable"


class CoordinateFrame(str, Enum):
    MAP = "map"
    ODOM = "odom"
    BASE_LINK = "base_link"
    WORLD = "world"
    CUSTOM = "custom"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


def validate_id(
    value: str,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not all(
            char.isalnum() or char in "._-"
            for char in value
        )
    ):
        raise CheckpointValidationError(
            f"Invalid {name}"
        )

    return value


def validate_text(
    value: str,
    name: str,
    *,
    allow_empty: bool = False,
    maximum: int = 4000,
) -> str:
    if not isinstance(value, str):
        raise CheckpointValidationError(
            f"{name} must be a string"
        )

    result = value.strip()

    if len(result) > maximum:
        raise CheckpointValidationError(
            f"{name} exceeds maximum length"
        )

    if not allow_empty and not result:
        raise CheckpointValidationError(
            f"{name} cannot be empty"
        )

    return result


def validate_integer(
    value: int,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = 10000000,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise CheckpointValidationError(
            f"Invalid {name}"
        )

    return value


def finite_number(
    value: Any,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise CheckpointValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise CheckpointValidationError(
            f"Invalid {name}"
        ) from exc

    if not math.isfinite(result):
        raise CheckpointValidationError(
            f"{name} must be finite"
        )

    return result


def validate_datetime(
    value: datetime,
    name: str,
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise CheckpointValidationError(
            f"{name} must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, float):
        if not math.isfinite(value):
            raise CheckpointValidationError(
                "Non-finite JSON value"
            )

    return value


def validate_json_payload(
    value: Any,
    name: str,
    *,
    max_bytes: int = 1048576,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointValidationError(
            f"{name} must be a mapping"
        )

    try:
        cloned = copy.deepcopy(dict(value))

        serialized = json.dumps(
            cloned,
            allow_nan=False,
        )

    except (
        TypeError,
        ValueError,
        RecursionError,
    ) as exc:
        raise CheckpointValidationError(
            f"{name} must contain JSON-compatible data"
        ) from exc

    if len(serialized.encode("utf-8")) > max_bytes:
        raise CheckpointValidationError(
            f"{name} exceeds maximum size"
        )

    return cloned


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class CheckpointConfig:
    maximum_checkpoints: int = 100000

    maximum_snapshots_per_robot: int = 1000

    maximum_audit_events: int = 20000

    maximum_payload_bytes: int = 1048576

    default_position_tolerance: float = 0.25

    default_orientation_tolerance: float = 0.15

    maximum_checkpoint_dependencies: int = 100

    advisory_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_checkpoints",
            "maximum_snapshots_per_robot",
            "maximum_audit_events",
            "maximum_payload_bytes",
            "maximum_checkpoint_dependencies",
        ):
            validate_integer(
                getattr(self, name),
                name,
                minimum=1,
            )

        for name in (
            "default_position_tolerance",
            "default_orientation_tolerance",
        ):
            value = finite_number(
                getattr(self, name),
                name,
            )

            if value <= 0:
                raise CheckpointValidationError(
                    f"{name} must be positive"
                )

        if self.advisory_only is not True:
            raise CheckpointValidationError(
                "Only advisory mode is supported"
            )


# ============================================================
# ROBOT POSE
# ============================================================

@dataclass(frozen=True)
class RobotPose:
    x: float
    y: float
    z: float = 0.0

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    frame: CoordinateFrame = CoordinateFrame.MAP

    def __post_init__(self) -> None:
        for name in (
            "x",
            "y",
            "z",
            "roll",
            "pitch",
            "yaw",
        ):
            finite_number(
                getattr(self, name),
                name,
            )

        if not isinstance(
            self.frame,
            CoordinateFrame,
        ):
            raise CheckpointValidationError(
                "Invalid coordinate frame"
            )

    def distance_to(
        self,
        other: RobotPose,
    ) -> float:
        if not isinstance(other, RobotPose):
            raise CheckpointValidationError(
                "Invalid target pose"
            )

        if self.frame != other.frame:
            raise CheckpointValidationError(
                "Cannot compare poses in different frames"
            )

        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )

    def angular_difference(
        self,
        other: RobotPose,
    ) -> float:
        if not isinstance(other, RobotPose):
            raise CheckpointValidationError(
                "Invalid target pose"
            )

        if self.frame != other.frame:
            raise CheckpointValidationError(
                "Coordinate frame mismatch"
            )

        difference = (
            self.yaw - other.yaw + math.pi
        ) % (2 * math.pi) - math.pi

        return abs(difference)

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "frame": self.frame.value,
        }


# ============================================================
# WORKSPACE BOUNDS
# ============================================================

@dataclass(frozen=True)
class WorkspaceBounds:
    minimum_x: float
    maximum_x: float

    minimum_y: float
    maximum_y: float

    minimum_z: float = -1000.0
    maximum_z: float = 1000.0

    frame: CoordinateFrame = CoordinateFrame.MAP

    def __post_init__(self) -> None:
        for axis in ("x", "y", "z"):
            minimum = finite_number(
                getattr(self, f"minimum_{axis}"),
                f"minimum_{axis}",
            )

            maximum = finite_number(
                getattr(self, f"maximum_{axis}"),
                f"maximum_{axis}",
            )

            if minimum > maximum:
                raise CheckpointValidationError(
                    f"Invalid {axis}-axis bounds"
                )

        if not isinstance(
            self.frame,
            CoordinateFrame,
        ):
            raise CheckpointValidationError(
                "Invalid workspace coordinate frame"
            )

    def contains(
        self,
        pose: RobotPose,
    ) -> bool:
        if pose.frame != self.frame:
            return False

        return (
            self.minimum_x <= pose.x <= self.maximum_x
            and self.minimum_y <= pose.y <= self.maximum_y
            and self.minimum_z <= pose.z <= self.maximum_z
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_x": self.minimum_x,
            "maximum_x": self.maximum_x,
            "minimum_y": self.minimum_y,
            "maximum_y": self.maximum_y,
            "minimum_z": self.minimum_z,
            "maximum_z": self.maximum_z,
            "frame": self.frame.value,
        }


# ============================================================
# CHECKPOINT DEFINITION
# ============================================================

@dataclass(frozen=True)
class RobotCheckpoint:
    checkpoint_id: str

    tenant_id: str
    robot_id: str
    mission_id: str

    name: str
    checkpoint_type: CheckpointType

    pose: RobotPose | None = None

    priority: CheckpointPriority = (
        CheckpointPriority.NORMAL
    )

    position_tolerance: float = 0.25
    orientation_tolerance: float = 0.15

    dependencies: tuple[str, ...] = ()

    requires_human_approval: bool = False

    expires_at: datetime | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    description: str = ""

    version: int = 1

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_id",
            "tenant_id",
            "robot_id",
            "mission_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        validate_text(
            self.name,
            "name",
        )

        if not isinstance(
            self.checkpoint_type,
            CheckpointType,
        ):
            raise CheckpointValidationError(
                "Invalid checkpoint type"
            )

        if (
            self.pose is not None
            and not isinstance(self.pose, RobotPose)
        ):
            raise CheckpointValidationError(
                "Invalid checkpoint pose"
            )

        for name in (
            "position_tolerance",
            "orientation_tolerance",
        ):
            value = finite_number(
                getattr(self, name),
                name,
            )

            if value <= 0:
                raise CheckpointValidationError(
                    f"{name} must be positive"
                )

        if not isinstance(
            self.priority,
            CheckpointPriority,
        ):
            raise CheckpointValidationError(
                "Invalid checkpoint priority"
            )

        if not isinstance(
            self.dependencies,
            tuple,
        ):
            raise CheckpointValidationError(
                "dependencies must be a tuple"
            )

        if len(set(self.dependencies)) != len(
            self.dependencies
        ):
            raise CheckpointValidationError(
                "Duplicate checkpoint dependencies"
            )

        for dependency in self.dependencies:
            validate_id(
                dependency,
                "dependency",
            )

            if dependency == self.checkpoint_id:
                raise CheckpointValidationError(
                    "Checkpoint cannot depend on itself"
                )

        if not isinstance(
            self.requires_human_approval,
            bool,
        ):
            raise CheckpointValidationError(
                "requires_human_approval must be boolean"
            )

        if (
            self.priority == CheckpointPriority.CRITICAL
            and not self.requires_human_approval
        ):
            raise CheckpointValidationError(
                "Critical checkpoints require human approval"
            )

        if self.expires_at is not None:
            validate_datetime(
                self.expires_at,
                "expires_at",
            )

        validate_json_payload(
            self.metadata,
            "metadata",
        )

        validate_text(
            self.description,
            "description",
            allow_empty=True,
        )

        validate_integer(
            self.version,
            "version",
            minimum=1,
        )

        validate_datetime(
            self.created_at,
            "created_at",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "mission_id": self.mission_id,
            "name": self.name,
            "checkpoint_type": self.checkpoint_type.value,
            "pose": (
                self.pose.to_dict()
                if self.pose is not None
                else None
            ),
            "priority": self.priority.value,
            "position_tolerance": (
                self.position_tolerance
            ),
            "orientation_tolerance": (
                self.orientation_tolerance
            ),
            "dependencies": list(
                self.dependencies
            ),
            "requires_human_approval": (
                self.requires_human_approval
            ),
            "expires_at": (
                self.expires_at.isoformat()
                if self.expires_at is not None
                else None
            ),
            "metadata": copy.deepcopy(
                dict(self.metadata)
            ),
            "description": self.description,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
        }


# ============================================================
# CHECKPOINT STATE
# ============================================================

@dataclass
class CheckpointRecord:
    checkpoint: RobotCheckpoint

    status: CheckpointStatus = (
        CheckpointStatus.PENDING
    )

    actual_pose: RobotPose | None = None

    started_at: datetime | None = None
    completed_at: datetime | None = None

    failure_reason: str = ""

    reviewed_by: str | None = None

    updated_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        result = self.checkpoint.to_dict()

        result.update(
            {
                "status": self.status.value,
                "actual_pose": (
                    self.actual_pose.to_dict()
                    if self.actual_pose is not None
                    else None
                ),
                "started_at": (
                    self.started_at.isoformat()
                    if self.started_at is not None
                    else None
                ),
                "completed_at": (
                    self.completed_at.isoformat()
                    if self.completed_at is not None
                    else None
                ),
                "failure_reason": self.failure_reason,
                "reviewed_by": self.reviewed_by,
                "updated_at": self.updated_at.isoformat(),
            }
        )

        return result


# ============================================================
# ROBOT STATE SNAPSHOT
# ============================================================

@dataclass(frozen=True)
class RobotStateSnapshot:
    snapshot_id: str

    tenant_id: str
    robot_id: str

    checkpoint_id: str | None

    pose: RobotPose | None

    joint_positions: Mapping[str, float]

    battery_percentage: float | None

    robot_mode: str

    task_state: Mapping[str, Any]

    sensor_summary: Mapping[str, Any]

    model_state_reference: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for name in (
            "snapshot_id",
            "tenant_id",
            "robot_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        if self.checkpoint_id is not None:
            validate_id(
                self.checkpoint_id,
                "checkpoint_id",
            )

        if (
            self.pose is not None
            and not isinstance(self.pose, RobotPose)
        ):
            raise CheckpointValidationError(
                "Invalid snapshot pose"
            )

        if not isinstance(
            self.joint_positions,
            Mapping,
        ):
            raise CheckpointValidationError(
                "joint_positions must be a mapping"
            )

        for joint, value in self.joint_positions.items():
            validate_id(
                joint,
                "joint name",
            )

            finite_number(
                value,
                f"joint position {joint}",
            )

        if self.battery_percentage is not None:
            battery = finite_number(
                self.battery_percentage,
                "battery_percentage",
            )

            if not 0 <= battery <= 100:
                raise CheckpointValidationError(
                    "Battery percentage outside [0, 100]"
                )

        validate_text(
            self.robot_mode,
            "robot_mode",
        )

        validate_json_payload(
            self.task_state,
            "task_state",
        )

        validate_json_payload(
            self.sensor_summary,
            "sensor_summary",
        )

        validate_text(
            self.model_state_reference,
            "model_state_reference",
            allow_empty=True,
            maximum=512,
        )

        validate_datetime(
            self.created_at,
            "created_at",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "checkpoint_id": self.checkpoint_id,
            "pose": (
                self.pose.to_dict()
                if self.pose is not None
                else None
            ),
            "joint_positions": dict(
                self.joint_positions
            ),
            "battery_percentage": (
                self.battery_percentage
            ),
            "robot_mode": self.robot_mode,
            "task_state": copy.deepcopy(
                dict(self.task_state)
            ),
            "sensor_summary": copy.deepcopy(
                dict(self.sensor_summary)
            ),
            "model_state_reference": (
                self.model_state_reference
            ),
            "created_at": self.created_at.isoformat(),
            "physical_state_restored": False,
        }


# ============================================================
# CHECKPOINT ENGINE
# ============================================================

class RoboticsCheckpointEngine:

    def __init__(
        self,
        config: CheckpointConfig | None = None,
    ) -> None:
        self.config = config or CheckpointConfig()

        self._checkpoints: dict[
            tuple[str, str],
            CheckpointRecord,
        ] = {}

        self._snapshots: dict[
            tuple[str, str],
            RobotStateSnapshot,
        ] = {}

        self._workspace_bounds: dict[
            tuple[str, str],
            WorkspaceBounds,
        ] = {}

        self._audit: deque[
            dict[str, Any]
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _log(
        self,
        *,
        tenant_id: str,
        action: str,
        entity_id: str,
        actor_id: str = "system",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._audit.append(
            {
                "event_id": new_id(),
                "timestamp": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "action": action,
                "entity_id": entity_id,
                "actor_id": actor_id,
                "details": json_safe(
                    dict(details or {})
                ),
            }
        )

    # --------------------------------------------------------
    # SET WORKSPACE BOUNDS
    # --------------------------------------------------------

    def set_workspace_bounds(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        bounds: WorkspaceBounds,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise CheckpointPermissionDenied(
                "Authorized workspace update required"
            )

        validate_id(tenant_id, "tenant_id")
        validate_id(robot_id, "robot_id")
        validate_id(actor_id, "actor_id")

        if not isinstance(
            bounds,
            WorkspaceBounds,
        ):
            raise CheckpointValidationError(
                "Invalid workspace bounds"
            )

        with self._lock:
            self._workspace_bounds[
                (tenant_id, robot_id)
            ] = bounds

            self._log(
                tenant_id=tenant_id,
                action="workspace_bounds_updated",
                entity_id=robot_id,
                actor_id=actor_id,
            )

            return bounds.to_dict()

    # --------------------------------------------------------
    # REGISTER CHECKPOINT
    # --------------------------------------------------------

    def register_checkpoint(
        self,
        checkpoint: RobotCheckpoint,
        *,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise CheckpointPermissionDenied(
                "Authorized checkpoint registration required"
            )

        if not isinstance(
            checkpoint,
            RobotCheckpoint,
        ):
            raise CheckpointValidationError(
                "Invalid checkpoint"
            )

        validate_id(actor_id, "actor_id")

        if (
            len(checkpoint.dependencies)
            > self.config.maximum_checkpoint_dependencies
        ):
            raise CheckpointValidationError(
                "Too many checkpoint dependencies"
            )

        key = (
            checkpoint.tenant_id,
            checkpoint.checkpoint_id,
        )

        with self._lock:
            if key in self._checkpoints:
                raise CheckpointConflict(
                    "Checkpoint already exists"
                )

            if (
                len(self._checkpoints)
                >= self.config.maximum_checkpoints
            ):
                raise CheckpointConflict(
                    "Checkpoint capacity reached"
                )

            self._validate_workspace(checkpoint)

            self._validate_dependencies(checkpoint)

            record = CheckpointRecord(
                checkpoint=checkpoint
            )

            self._checkpoints[key] = record

            self._log(
                tenant_id=checkpoint.tenant_id,
                action="checkpoint_registered",
                entity_id=checkpoint.checkpoint_id,
                actor_id=actor_id,
                details={
                    "robot_id": checkpoint.robot_id,
                    "mission_id": checkpoint.mission_id,
                    "type": checkpoint.checkpoint_type.value,
                },
            )

            return record.to_dict()

    # --------------------------------------------------------
    # WORKSPACE VALIDATION
    # --------------------------------------------------------

    def _validate_workspace(
        self,
        checkpoint: RobotCheckpoint,
    ) -> None:
        if checkpoint.pose is None:
            return

        bounds = self._workspace_bounds.get(
            (
                checkpoint.tenant_id,
                checkpoint.robot_id,
            )
        )

        if bounds is None:
            return

        if not bounds.contains(checkpoint.pose):
            raise CheckpointValidationError(
                "Checkpoint pose outside configured "
                "workspace or in a mismatched frame"
            )

    # --------------------------------------------------------
    # DEPENDENCY VALIDATION
    # --------------------------------------------------------

    def _validate_dependencies(
        self,
        checkpoint: RobotCheckpoint,
    ) -> None:
        for dependency_id in checkpoint.dependencies:
            record = self._checkpoints.get(
                (
                    checkpoint.tenant_id,
                    dependency_id,
                )
            )

            if record is None:
                raise CheckpointValidationError(
                    f"Dependency not found: {dependency_id}"
                )

            dependency = record.checkpoint

            if (
                dependency.robot_id != checkpoint.robot_id
                or dependency.mission_id
                != checkpoint.mission_id
            ):
                raise CheckpointValidationError(
                    "Checkpoint dependencies must belong "
                    "to the same robot and mission"
                )

        # Existing checkpoints cannot be edited to add
        # dependencies, so registration against earlier
        # checkpoints cannot create a dependency cycle.

    # --------------------------------------------------------
    # GET CHECKPOINT
    # --------------------------------------------------------

    def get_checkpoint(
        self,
        *,
        tenant_id: str,
        checkpoint_id: str,
    ) -> dict[str, Any]:
        validate_id(tenant_id, "tenant_id")
        validate_id(checkpoint_id, "checkpoint_id")

        with self._lock:
            record = self._checkpoints.get(
                (
                    tenant_id,
                    checkpoint_id,
                )
            )

            if record is None:
                raise CheckpointNotFound(
                    "Checkpoint not found"
                )

            return record.to_dict()

    # --------------------------------------------------------
    # LIST CHECKPOINTS
    # --------------------------------------------------------

    def list_checkpoints(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
        mission_id: str | None = None,
        status: CheckpointStatus | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        if robot_id is not None:
            validate_id(robot_id, "robot_id")

        if mission_id is not None:
            validate_id(mission_id, "mission_id")

        if status is not None and not isinstance(
            status,
            CheckpointStatus,
        ):
            raise CheckpointValidationError(
                "Invalid checkpoint status"
            )

        validate_integer(
            limit,
            "limit",
            minimum=1,
            maximum=100000,
        )

        with self._lock:
            records = [
                record
                for (tenant, _), record
                in self._checkpoints.items()
                if (
                    tenant == tenant_id
                    and (
                        robot_id is None
                        or record.checkpoint.robot_id
                        == robot_id
                    )
                    and (
                        mission_id is None
                        or record.checkpoint.mission_id
                        == mission_id
                    )
                    and (
                        status is None
                        or record.status == status
                    )
                )
            ]

            records.sort(
                key=lambda item: (
                    item.checkpoint.created_at,
                    item.checkpoint.checkpoint_id,
                )
            )

            return [
                item.to_dict()
                for item in records[:limit]
            ]

    # --------------------------------------------------------
    # DEPENDENCY READINESS
    # --------------------------------------------------------

    def dependency_readiness(
        self,
        *,
        tenant_id: str,
        checkpoint_id: str,
    ) -> dict[str, Any]:
        validate_id(tenant_id, "tenant_id")
        validate_id(checkpoint_id, "checkpoint_id")

        with self._lock:
            record = self._checkpoints.get(
                (tenant_id, checkpoint_id)
            )

            if record is None:
                raise CheckpointNotFound(
                    "Checkpoint not found"
                )

            incomplete = []

            for dependency_id in (
                record.checkpoint.dependencies
            ):
                dependency = self._checkpoints.get(
                    (
                        tenant_id,
                        dependency_id,
                    )
                )

                if (
                    dependency is None
                    or dependency.status
                    != CheckpointStatus.COMPLETED
                ):
                    incomplete.append(dependency_id)

            return {
                "checkpoint_id": checkpoint_id,
                "ready": not incomplete,
                "incomplete_dependencies": incomplete,
            }

    # --------------------------------------------------------
    # ACTIVATE CHECKPOINT
    # --------------------------------------------------------

    def activate_checkpoint(
        self,
        *,
        tenant_id: str,
        checkpoint_id: str,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise CheckpointPermissionDenied(
                "Authorized checkpoint activation required"
            )

        validate_id(tenant_id, "tenant_id")
        validate_id(checkpoint_id, "checkpoint_id")
        validate_id(actor_id, "actor_id")

        with self._lock:
            record = self._checkpoints.get(
                (tenant_id, checkpoint_id)
            )

            if record is None:
                raise CheckpointNotFound(
                    "Checkpoint not found"
                )

            if record.status != CheckpointStatus.PENDING:
                raise CheckpointConflict(
                    "Only pending checkpoints can activate"
                )

            if (
                record.checkpoint.expires_at is not None
                and utc_now()
                > record.checkpoint.expires_at
            ):
                record.status = CheckpointStatus.EXPIRED
                record.updated_at = utc_now()

                raise CheckpointConflict(
                    "Checkpoint has expired"
                )

            readiness = self.dependency_readiness(
                tenant_id=tenant_id,
                checkpoint_id=checkpoint_id,
            )

            if not readiness["ready"]:
                raise CheckpointConflict(
                    "Checkpoint dependencies are incomplete"
                )

            record.status = CheckpointStatus.ACTIVE
            record.started_at = utc_now()
            record.updated_at = utc_now()

            self._log(
                tenant_id=tenant_id,
                action="checkpoint_activated",
                entity_id=checkpoint_id,
                actor_id=actor_id,
            )

            return record.to_dict()

    # --------------------------------------------------------
    # CHECK POSITION REACHED
    # --------------------------------------------------------

    def check_pose_reached(
        self,
        *,
        tenant_id: str,
        checkpoint_id: str,
        actual_pose: RobotPose,
    ) -> dict[str, Any]:
        validate_id(tenant_id, "tenant_id")
        validate_id(checkpoint_id, "checkpoint_id")

        if not isinstance(actual_pose, RobotPose):
            raise CheckpointValidationError(
                "Invalid actual pose"
            )

        with self._lock:
            record = self._checkpoints.get(
                (tenant_id, checkpoint_id)
            )

            if record is None:
                raise CheckpointNotFound(
                    "Checkpoint not found"
                )

            target = record.checkpoint.pose

            if target is None:
                raise CheckpointValidationError(
                    "Checkpoint has no target pose"
                )

            distance = target.distance_to(
                actual_pose
            )

            yaw_difference = (
                target.angular_difference(
                    actual_pose
                )
            )

            position_reached = (
                distance
                <= record.checkpoint.position_tolerance
            )

            orientation_reached = (
                yaw_difference
                <= record.checkpoint.orientation_tolerance
            )

            return {
                "checkpoint_id": checkpoint_id,
                "position_distance_m": distance,
                "yaw_difference_rad": yaw_difference,
                "position_reached": position_reached,
                "orientation_reached": orientation_reached,
                "pose_reached": (
                    position_reached
                    and orientation_reached
                ),
                "target_pose": target.to_dict(),
                "actual_pose": actual_pose.to_dict(),
                "sensor_pose_trusted": False,
                "advisory_only": True,
            }

    # --------------------------------------------------------
    # COMPLETE CHECKPOINT
    # --------------------------------------------------------

    def complete_checkpoint(
        self,
        *,
        tenant_id: str,
        checkpoint_id: str,
        actual_pose: RobotPose | None = None,
        reviewer_id: str | None = None,
        authorized: bool = False,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise CheckpointPermissionDenied(
                "Authorized checkpoint completion required"
            )

        validate_id(tenant_id, "tenant_id")
        validate_id(checkpoint_id, "checkpoint_id")

        if reviewer_id is not None:
            validate_id(reviewer_id, "reviewer_id")

        with self._lock:
            record = self._checkpoints.get(
                (tenant_id, checkpoint_id)
            )

            if record is None:
                raise CheckpointNotFound(
                    "Checkpoint not found"
                )

            if record.status != CheckpointStatus.ACTIVE:
                raise CheckpointConflict(
                    "Only active checkpoints can complete"
                )

            if (
                record.checkpoint.requires_human_approval
                and reviewer_id is None
            ):
                raise CheckpointPermissionDenied(
                    "Human reviewer ID required"
                )

            if record.checkpoint.pose is not None:
                if actual_pose is None:
                    raise CheckpointValidationError(
                        "Actual pose required"
                    )

                pose_result = self.check_pose_reached(
                    tenant_id=tenant_id,
                    checkpoint_id=checkpoint_id,
                    actual_pose=actual_pose,
                )

                if not pose_result["pose_reached"]:
                    raise CheckpointConflict(
                        "Checkpoint pose tolerance not met"
                    )

            record.status = CheckpointStatus.COMPLETED
            record.actual_pose = actual_pose
            record.completed_at = utc_now()
            record.updated_at = utc_now()
            record.reviewed_by = reviewer_id

            self._log(
                tenant_id=tenant_id,
                action="checkpoint_completed",
                entity_id=checkpoint_id,
                actor_id=reviewer_id or "system",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # FAIL CHECKPOINT
    # --------------------------------------------------------

    def fail_checkpoint(
        self,
        *,
        tenant_id: str,
        checkpoint_id: str,
        reason: str,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise CheckpointPermissionDenied(
                "Authorized checkpoint failure update required"
            )

        validate_id(tenant_id, "tenant_id")
        validate_id(checkpoint_id, "checkpoint_id")
        validate_id(actor_id, "actor_id")

        validate_text(
            reason,
            "reason",
        )

        with self._lock:
            record = self._checkpoints.get(
                (tenant_id, checkpoint_id)
            )

            if record is None:
                raise CheckpointNotFound(
                    "Checkpoint not found"
                )

            if record.status not in (
                CheckpointStatus.PENDING,
                CheckpointStatus.ACTIVE,
            ):
                raise CheckpointConflict(
                    "Checkpoint cannot fail in current status"
                )

            record.status = CheckpointStatus.FAILED
            record.failure_reason = reason
            record.updated_at = utc_now()

            self._log(
                tenant_id=tenant_id,
                action="checkpoint_failed",
                entity_id=checkpoint_id,
                actor_id=actor_id,
                details={
                    "reason": reason,
                },
            )

            return record.to_dict()

    # --------------------------------------------------------
    # CREATE ROBOT SNAPSHOT
    # --------------------------------------------------------

    def create_snapshot(
        self,
        snapshot: RobotStateSnapshot,
        *,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise CheckpointPermissionDenied(
                "Authorized snapshot creation required"
            )

        if not isinstance(
            snapshot,
            RobotStateSnapshot,
        ):
            raise CheckpointValidationError(
                "Invalid robot snapshot"
            )

        validate_id(actor_id, "actor_id")

        key = (
            snapshot.tenant_id,
            snapshot.snapshot_id,
        )

        with self._lock:
            if key in self._snapshots:
                raise CheckpointConflict(
                    "Snapshot already exists"
                )

            robot_snapshot_count = sum(
                1
                for (tenant, _), existing
                in self._snapshots.items()
                if (
                    tenant == snapshot.tenant_id
                    and existing.robot_id
                    == snapshot.robot_id
                )
            )

            if (
                robot_snapshot_count
                >= self.config.maximum_snapshots_per_robot
            ):
                raise CheckpointConflict(
                    "Robot snapshot capacity reached"
                )

            if snapshot.checkpoint_id is not None:
                checkpoint = self._checkpoints.get(
                    (
                        snapshot.tenant_id,
                        snapshot.checkpoint_id,
                    )
                )

                if checkpoint is None:
                    raise CheckpointNotFound(
                        "Referenced checkpoint not found"
                    )

                if (
                    checkpoint.checkpoint.robot_id
                    != snapshot.robot_id
                ):
                    raise CheckpointValidationError(
                        "Snapshot robot mismatch"
                    )

            self._snapshots[key] = snapshot

            self._log(
                tenant_id=snapshot.tenant_id,
                action="robot_snapshot_created",
                entity_id=snapshot.snapshot_id,
                actor_id=actor_id,
                details={
                    "robot_id": snapshot.robot_id,
                    "checkpoint_id": snapshot.checkpoint_id,
                },
            )

            return snapshot.to_dict()

    # --------------------------------------------------------
    # GET SNAPSHOT
    # --------------------------------------------------------

    def get_snapshot(
        self,
        *,
        tenant_id: str,
        snapshot_id: str,
    ) -> dict[str, Any]:
        validate_id(tenant_id, "tenant_id")
        validate_id(snapshot_id, "snapshot_id")

        with self._lock:
            snapshot = self._snapshots.get(
                (tenant_id, snapshot_id)
            )

            if snapshot is None:
                raise CheckpointNotFound(
                    "Snapshot not found"
                )

            return snapshot.to_dict()

    # --------------------------------------------------------
    # LIST SNAPSHOTS
    # --------------------------------------------------------

    def list_snapshots(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")
        validate_id(robot_id, "robot_id")

        validate_integer(
            limit,
            "limit",
            minimum=1,
            maximum=100000,
        )

        with self._lock:
            snapshots = [
                snapshot
                for (tenant, _), snapshot
                in self._snapshots.items()
                if (
                    tenant == tenant_id
                    and snapshot.robot_id == robot_id
                )
            ]

            snapshots.sort(
                key=lambda item: item.created_at,
                reverse=True,
            )

            return [
                item.to_dict()
                for item in snapshots[:limit]
            ]

    # --------------------------------------------------------
    # RECOVERY ADVISOR
    # --------------------------------------------------------

    def recovery_advice(
        self,
        *,
        tenant_id: str,
        snapshot_id: str,
        current_robot_id: str,
    ) -> dict[str, Any]:
        validate_id(tenant_id, "tenant_id")
        validate_id(snapshot_id, "snapshot_id")
        validate_id(current_robot_id, "current_robot_id")

        with self._lock:
            snapshot = self._snapshots.get(
                (tenant_id, snapshot_id)
            )

            if snapshot is None:
                raise CheckpointNotFound(
                    "Snapshot not found"
                )

            warnings = []

            if snapshot.robot_id != current_robot_id:
                status = RecoveryStatus.UNAVAILABLE

                warnings.append(
                    "Snapshot belongs to another robot."
                )

            else:
                status = RecoveryStatus.REQUIRES_REVIEW

                warnings.append(
                    "Robot state snapshots cannot be "
                    "directly restored to physical hardware."
                )

                warnings.append(
                    "Verify current robot configuration, "
                    "joint limits, workspace, sensors, "
                    "obstacles and safety interlocks."
                )

            return {
                "snapshot_id": snapshot_id,
                "tenant_id": tenant_id,
                "robot_id": current_robot_id,
                "recovery_status": status.value,
                "snapshot": snapshot.to_dict(),
                "warnings": warnings,
                "recommended_actions": [
                    "Inspect current robot state",
                    "Validate robot hardware configuration",
                    "Compare current and saved joint states",
                    "Validate workspace and obstacle clearance",
                    "Run recovery in simulation",
                    "Obtain independent safety approval",
                ],
                "physical_state_restored": False,
                "execution_authorized": False,
                "advisory_only": True,
            }

    # --------------------------------------------------------
    # MISSION PROGRESS
    # --------------------------------------------------------

    def mission_progress(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        mission_id: str,
    ) -> dict[str, Any]:
        validate_id(tenant_id, "tenant_id")
        validate_id(robot_id, "robot_id")
        validate_id(mission_id, "mission_id")

        with self._lock:
            records = [
                record
                for (tenant, _), record
                in self._checkpoints.items()
                if (
                    tenant == tenant_id
                    and record.checkpoint.robot_id
                    == robot_id
                    and record.checkpoint.mission_id
                    == mission_id
                )
            ]

            counts = Counter(
                record.status.value
                for record in records
            )

            total = len(records)

            completed = counts.get(
                CheckpointStatus.COMPLETED.value,
                0,
            )

            progress = (
                completed / total * 100
                if total
                else 0.0
            )

            return {
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "mission_id": mission_id,
                "total_checkpoints": total,
                "completed_checkpoints": completed,
                "progress_percentage": round(
                    progress,
                    2,
                ),
                "status_counts": dict(counts),
                "all_completed": (
                    total > 0
                    and completed == total
                ),
                "mission_execution_authorized": False,
            }

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    def audit_history(
        self,
        *,
        tenant_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        validate_integer(
            limit,
            "limit",
            minimum=1,
            maximum=100000,
        )

        with self._lock:
            events = [
                copy.deepcopy(event)
                for event in self._audit
                if event["tenant_id"] == tenant_id
            ]

            return events[-limit:]

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        path: str | Path,
        *,
        tenant_id: str,
    ) -> Path:
        validate_id(tenant_id, "tenant_id")

        output = Path(path)

        with self._lock:
            checkpoints = [
                record.to_dict()
                for (tenant, _), record
                in self._checkpoints.items()
                if tenant == tenant_id
            ]

            snapshots = [
                snapshot.to_dict()
                for (tenant, _), snapshot
                in self._snapshots.items()
                if tenant == tenant_id
            ]

            bounds = [
                {
                    "robot_id": robot_id,
                    "bounds": workspace.to_dict(),
                }
                for (tenant, robot_id), workspace
                in self._workspace_bounds.items()
                if tenant == tenant_id
            ]

            audit = [
                copy.deepcopy(event)
                for event in self._audit
                if event["tenant_id"] == tenant_id
            ]

        payload = {
            "generated_at": utc_now().isoformat(),
            "tenant_id": tenant_id,
            "checkpoints": checkpoints,
            "snapshots": snapshots,
            "workspace_bounds": bounds,
            "audit_events": audit,
            "advisory_only": True,
        }

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.write_text(
            json.dumps(
                json_safe(payload),
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
    engine = RoboticsCheckpointEngine()

    engine.set_workspace_bounds(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        bounds=WorkspaceBounds(
            minimum_x=-20.0,
            maximum_x=20.0,
            minimum_y=-20.0,
            maximum_y=20.0,
            minimum_z=0.0,
            maximum_z=5.0,
        ),
        authorized=True,
        actor_id="robotics_admin",
    )

    checkpoint_1 = RobotCheckpoint(
        checkpoint_id="checkpoint_001",
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        mission_id="mission_001",
        name="Warehouse Entry",
        checkpoint_type=CheckpointType.NAVIGATION,
        pose=RobotPose(
            x=2.0,
            y=3.0,
            yaw=0.0,
        ),
        priority=CheckpointPriority.NORMAL,
        position_tolerance=0.3,
        orientation_tolerance=0.2,
        description="First warehouse navigation checkpoint",
    )

    checkpoint_2 = RobotCheckpoint(
        checkpoint_id="checkpoint_002",
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        mission_id="mission_001",
        name="Inspection Station",
        checkpoint_type=CheckpointType.INSPECTION,
        pose=RobotPose(
            x=8.0,
            y=6.0,
            yaw=1.57,
        ),
        priority=CheckpointPriority.HIGH,
        dependencies=(
            "checkpoint_001",
        ),
        description="Robot inspection station",
    )

    engine.register_checkpoint(
        checkpoint_1,
        authorized=True,
        actor_id="robotics_admin",
    )

    engine.register_checkpoint(
        checkpoint_2,
        authorized=True,
        actor_id="robotics_admin",
    )

    engine.activate_checkpoint(
        tenant_id="tenant_alpha",
        checkpoint_id="checkpoint_001",
        authorized=True,
        actor_id="mission_supervisor",
    )

    pose_result = engine.check_pose_reached(
        tenant_id="tenant_alpha",
        checkpoint_id="checkpoint_001",
        actual_pose=RobotPose(
            x=2.1,
            y=3.1,
            yaw=0.05,
        ),
    )

    print("\nPOSE VALIDATION")

    print(
        json.dumps(
            pose_result,
            indent=2,
        )
    )

    engine.complete_checkpoint(
        tenant_id="tenant_alpha",
        checkpoint_id="checkpoint_001",
        actual_pose=RobotPose(
            x=2.1,
            y=3.1,
            yaw=0.05,
        ),
        authorized=True,
    )

    engine.activate_checkpoint(
        tenant_id="tenant_alpha",
        checkpoint_id="checkpoint_002",
        authorized=True,
        actor_id="mission_supervisor",
    )

    snapshot = RobotStateSnapshot(
        snapshot_id="snapshot_001",
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        checkpoint_id="checkpoint_002",
        pose=RobotPose(
            x=8.0,
            y=6.0,
            yaw=1.57,
        ),
        joint_positions={
            "joint_1": 0.1,
            "joint_2": -0.2,
            "joint_3": 0.4,
        },
        battery_percentage=78.5,
        robot_mode="simulation",
        task_state={
            "current_task": "inspection",
            "task_progress": 0.4,
        },
        sensor_summary={
            "lidar_available": True,
            "camera_available": True,
        },
        model_state_reference="model_checkpoint_001",
    )

    engine.create_snapshot(
        snapshot,
        authorized=True,
        actor_id="mission_supervisor",
    )

    print("\nMISSION PROGRESS")

    print(
        json.dumps(
            engine.mission_progress(
                tenant_id="tenant_alpha",
                robot_id="robot_001",
                mission_id="mission_001",
            ),
            indent=2,
        )
    )

    print("\nRECOVERY ADVICE")

    print(
        json.dumps(
            engine.recovery_advice(
                tenant_id="tenant_alpha",
                snapshot_id="snapshot_001",
                current_robot_id="robot_001",
            ),
            indent=2,
        )
    )

    output = engine.export_json(
        "data/robotics/checkpoints_report.json",
        tenant_id="tenant_alpha",
    )

    print("\nExported:", output)


if __name__ == "__main__":
    main()
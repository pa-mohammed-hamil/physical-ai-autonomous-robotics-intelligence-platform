"""Observation Space module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/observation_space.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Robotics Observation Space Engine

Python 3.11+
Dependencies: Standard library only

Features:
    - Continuous and discrete observation spaces
    - Vector, scalar, boolean, categorical and image metadata
    - Robot position, velocity and orientation observations
    - Joint position, velocity and torque observations
    - Camera, LiDAR, IMU and force sensor observations
    - Battery, temperature and robot health observations
    - Observation schema registration and versioning
    - Sensor value validation
    - Missing and stale sensor detection
    - Observation normalization and clipping
    - Observation masks
    - Observation flattening for ML/RL models
    - Observation history
    - Batch observation processing
    - Simulation episode observation snapshots
    - Multi-robot and multi-tenant isolation
    - Thread-safe in-memory registry
    - Audit logging
    - JSON export

Integrations:
    environment.py
    action_space.py
    imitation_learning.py
    curriculum.py
    domain_randomization.py
    sensor_manager.py
    robot_state.py
    simulator.py
    simulation_adapter.py
    evaluation_service.py

Safety:
    Observation processing only.

    This module does not actuate physical robots.
    Sensor values are not independently authenticated.
    Missing or stale safety-critical observations must
    be handled by independent safety systems.
"""

from __future__ import annotations

import copy
import json
import math
import threading
import uuid

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class ObservationSpaceError(Exception):
    pass


class ObservationValidationError(ObservationSpaceError):
    pass


class ObservationSpaceNotFound(ObservationSpaceError):
    pass


class ObservationConflict(ObservationSpaceError):
    pass


class ObservationPermissionDenied(ObservationSpaceError):
    pass


# ============================================================
# ENUMS
# ============================================================

class ObservationType(str, Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    DISCRETE = "discrete"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    IMAGE_METADATA = "image_metadata"


class ObservationCategory(str, Enum):
    POSITION = "position"
    ORIENTATION = "orientation"
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"
    JOINT = "joint"
    CAMERA = "camera"
    LIDAR = "lidar"
    IMU = "imu"
    FORCE = "force"
    BATTERY = "battery"
    TEMPERATURE = "temperature"
    HEALTH = "health"
    NAVIGATION = "navigation"
    ENVIRONMENT = "environment"
    CUSTOM = "custom"


class ObservationQuality(str, Enum):
    VALID = "valid"
    MISSING = "missing"
    STALE = "stale"
    INVALID = "invalid"
    MASKED = "masked"


class NormalizationType(str, Enum):
    NONE = "none"
    MIN_MAX = "min_max"
    NEGATIVE_ONE_TO_ONE = "negative_one_to_one"
    Z_SCORE = "z_score"


class ObservationSource(str, Enum):
    SIMULATION = "simulation"
    SENSOR_ADAPTER = "sensor_adapter"
    STATE_ESTIMATOR = "state_estimator"
    REPLAY = "replay"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


def validate_id(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not all(
            character.isalnum() or character in "._-"
            for character in value
        )
    ):
        raise ObservationValidationError(
            f"Invalid {name}"
        )

    return value


def finite_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ObservationValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ObservationValidationError(
            f"Invalid {name}"
        ) from exc

    if not math.isfinite(result):
        raise ObservationValidationError(
            f"{name} must be finite"
        )

    return result


def validate_integer(
    value: int,
    name: str,
    minimum: int = 0,
    maximum: int = 10_000_000,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ObservationValidationError(
            f"Invalid {name}"
        )

    return value


def validate_datetime(
    value: datetime,
    name: str,
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ObservationValidationError(
            f"{name} must be timezone-aware"
        )

    return value


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

    if isinstance(value, (tuple, list)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, float) and not math.isfinite(value):
        raise ObservationValidationError(
            "Non-finite JSON value"
        )

    return value


# ============================================================
# ENGINE CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ObservationSpaceConfig:
    maximum_spaces: int = 10_000
    maximum_fields_per_space: int = 256
    maximum_vector_dimension: int = 100_000
    maximum_history_per_robot: int = 1_000
    maximum_audit_events: int = 20_000
    maximum_batch_size: int = 1_000

    default_stale_after_seconds: float = 2.0

    reject_future_timestamps: bool = True
    future_timestamp_tolerance_seconds: float = 1.0

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_spaces",
            "maximum_fields_per_space",
            "maximum_vector_dimension",
            "maximum_history_per_robot",
            "maximum_audit_events",
            "maximum_batch_size",
        ):
            validate_integer(
                getattr(self, name),
                name,
                minimum=1,
            )

        if finite_number(
            self.default_stale_after_seconds,
            "default_stale_after_seconds",
        ) <= 0:
            raise ObservationValidationError(
                "Stale threshold must be positive"
            )

        if finite_number(
            self.future_timestamp_tolerance_seconds,
            "future_timestamp_tolerance_seconds",
        ) < 0:
            raise ObservationValidationError(
                "Future timestamp tolerance cannot be negative"
            )

        if self.simulation_only is not True:
            raise ObservationValidationError(
                "Only simulation observation processing "
                "is supported"
            )


# ============================================================
# OBSERVATION FIELD
# ============================================================

@dataclass(frozen=True)
class ObservationField:
    name: str

    observation_type: ObservationType
    category: ObservationCategory

    dimension: int = 1

    minimum: float | None = None
    maximum: float | None = None

    allowed_values: tuple[str, ...] = ()

    required: bool = True
    safety_critical: bool = False

    stale_after_seconds: float | None = None

    normalization: NormalizationType = (
        NormalizationType.NONE
    )

    mean: float | tuple[float, ...] | None = None
    standard_deviation: float | tuple[float, ...] | None = None

    unit: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        validate_id(self.name, "field name")

        if not isinstance(
            self.observation_type,
            ObservationType,
        ):
            raise ObservationValidationError(
                "Invalid observation type"
            )

        if not isinstance(
            self.category,
            ObservationCategory,
        ):
            raise ObservationValidationError(
                "Invalid observation category"
            )

        validate_integer(
            self.dimension,
            "dimension",
            minimum=1,
        )

        if (
            self.observation_type
            != ObservationType.VECTOR
            and self.dimension != 1
        ):
            raise ObservationValidationError(
                "Only vector fields support dimension > 1"
            )

        if self.minimum is not None:
            finite_number(self.minimum, "minimum")

        if self.maximum is not None:
            finite_number(self.maximum, "maximum")

        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ObservationValidationError(
                "Minimum exceeds maximum"
            )

        if (
            self.observation_type
            == ObservationType.CATEGORICAL
        ):
            if not self.allowed_values:
                raise ObservationValidationError(
                    "Categorical field requires allowed values"
                )

            if (
                len(set(self.allowed_values))
                != len(self.allowed_values)
            ):
                raise ObservationValidationError(
                    "Duplicate categorical values"
                )

            for value in self.allowed_values:
                if not isinstance(value, str) or not value:
                    raise ObservationValidationError(
                        "Invalid categorical value"
                    )

        if (
            self.observation_type
            == ObservationType.DISCRETE
        ):
            if (
                self.minimum is not None
                and not float(self.minimum).is_integer()
            ):
                raise ObservationValidationError(
                    "Discrete minimum must be integral"
                )

            if (
                self.maximum is not None
                and not float(self.maximum).is_integer()
            ):
                raise ObservationValidationError(
                    "Discrete maximum must be integral"
                )

        if self.stale_after_seconds is not None:
            if finite_number(
                self.stale_after_seconds,
                "stale_after_seconds",
            ) <= 0:
                raise ObservationValidationError(
                    "Stale threshold must be positive"
                )

        if not isinstance(
            self.normalization,
            NormalizationType,
        ):
            raise ObservationValidationError(
                "Invalid normalization type"
            )

        if self.normalization in (
            NormalizationType.MIN_MAX,
            NormalizationType.NEGATIVE_ONE_TO_ONE,
        ):
            if (
                self.minimum is None
                or self.maximum is None
                or self.minimum >= self.maximum
            ):
                raise ObservationValidationError(
                    "Min-max normalization requires "
                    "distinct finite bounds"
                )

        if self.normalization == NormalizationType.Z_SCORE:
            if (
                self.mean is None
                or self.standard_deviation is None
            ):
                raise ObservationValidationError(
                    "Z-score requires mean and standard deviation"
                )

            self._validate_statistics(
                self.mean,
                "mean",
                positive=False,
            )

            self._validate_statistics(
                self.standard_deviation,
                "standard_deviation",
                positive=True,
            )

        if self.observation_type not in (
            ObservationType.SCALAR,
            ObservationType.VECTOR,
            ObservationType.DISCRETE,
        ) and self.normalization != NormalizationType.NONE:
            raise ObservationValidationError(
                "Normalization requires a numeric field"
            )

        if not isinstance(self.required, bool):
            raise ObservationValidationError(
                "required must be boolean"
            )

        if not isinstance(self.safety_critical, bool):
            raise ObservationValidationError(
                "safety_critical must be boolean"
            )

    def _validate_statistics(
        self,
        values: float | tuple[float, ...],
        name: str,
        *,
        positive: bool,
    ) -> None:
        if isinstance(values, tuple):
            if len(values) != self.dimension:
                raise ObservationValidationError(
                    f"{name} dimension mismatch"
                )

            items = values

        else:
            items = (values,)

        for item in items:
            number = finite_number(item, name)

            if positive and number <= 0:
                raise ObservationValidationError(
                    f"{name} must be positive"
                )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "name": self.name,
                "observation_type": self.observation_type,
                "category": self.category,
                "dimension": self.dimension,
                "minimum": self.minimum,
                "maximum": self.maximum,
                "allowed_values": self.allowed_values,
                "required": self.required,
                "safety_critical": self.safety_critical,
                "stale_after_seconds": (
                    self.stale_after_seconds
                ),
                "normalization": self.normalization,
                "mean": self.mean,
                "standard_deviation": (
                    self.standard_deviation
                ),
                "unit": self.unit,
                "description": self.description,
            }
        )


# ============================================================
# OBSERVATION SPACE
# ============================================================

@dataclass(frozen=True)
class RobotObservationSpace:
    space_id: str
    tenant_id: str
    robot_id: str
    name: str

    fields: tuple[ObservationField, ...]

    version: int = 1

    description: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for name in (
            "space_id",
            "tenant_id",
            "robot_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        if not isinstance(self.name, str) or not self.name.strip():
            raise ObservationValidationError(
                "Observation space name cannot be empty"
            )

        if not self.fields:
            raise ObservationValidationError(
                "Observation space requires fields"
            )

        names = set()

        for item in self.fields:
            if not isinstance(item, ObservationField):
                raise ObservationValidationError(
                    "Invalid observation field"
                )

            if item.name in names:
                raise ObservationValidationError(
                    "Duplicate observation field"
                )

            names.add(item.name)

        validate_integer(
            self.version,
            "version",
            minimum=1,
        )

        validate_datetime(
            self.created_at,
            "created_at",
        )

    def get_field(
        self,
        name: str,
    ) -> ObservationField:
        for item in self.fields:
            if item.name == name:
                return item

        raise ObservationSpaceNotFound(
            f"Observation field not found: {name}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "space_id": self.space_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "name": self.name,
            "fields": [
                item.to_dict()
                for item in self.fields
            ],
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "simulation_only": True,
        }


# ============================================================
# OBSERVATION SAMPLE
# ============================================================

@dataclass(frozen=True)
class ObservationSample:
    field_name: str
    value: Any

    timestamp: datetime = field(
        default_factory=utc_now
    )

    source: ObservationSource = (
        ObservationSource.SIMULATION
    )

    def __post_init__(self) -> None:
        validate_id(
            self.field_name,
            "field_name",
        )

        validate_datetime(
            self.timestamp,
            "timestamp",
        )

        if not isinstance(
            self.source,
            ObservationSource,
        ):
            raise ObservationValidationError(
                "Invalid observation source"
            )


# ============================================================
# OBSERVATION MASK
# ============================================================

@dataclass(frozen=True)
class ObservationMask:
    tenant_id: str
    robot_id: str
    space_id: str

    enabled_fields: frozenset[str]

    reason: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for name in (
            "tenant_id",
            "robot_id",
            "space_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        for name in self.enabled_fields:
            validate_id(name, "enabled field")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "space_id": self.space_id,
            "enabled_fields": sorted(
                self.enabled_fields
            ),
            "reason": self.reason,
            "created_at": self.created_at.isoformat(),
        }


# ============================================================
# VALIDATION RESULT
# ============================================================

@dataclass(frozen=True)
class ObservationValidationResult:
    field_name: str
    quality: ObservationQuality
    valid: bool

    raw_value: Any = None
    normalized_value: Any = None

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    timestamp: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "field_name": self.field_name,
                "quality": self.quality,
                "valid": self.valid,
                "raw_value": self.raw_value,
                "normalized_value": (
                    self.normalized_value
                ),
                "errors": self.errors,
                "warnings": self.warnings,
                "timestamp": self.timestamp,
            }
        )


# ============================================================
# OBSERVATION FRAME
# ============================================================

@dataclass(frozen=True)
class ObservationFrame:
    frame_id: str
    tenant_id: str
    robot_id: str
    space_id: str
    space_version: int

    values: Mapping[str, Any]
    normalized_values: Mapping[str, Any]

    quality: Mapping[str, str]

    valid: bool

    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    captured_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "frame_id": self.frame_id,
                "tenant_id": self.tenant_id,
                "robot_id": self.robot_id,
                "space_id": self.space_id,
                "space_version": self.space_version,
                "values": self.values,
                "normalized_values": (
                    self.normalized_values
                ),
                "quality": self.quality,
                "valid": self.valid,
                "errors": self.errors,
                "warnings": self.warnings,
                "captured_at": self.captured_at,
                "simulation_only": True,
                "physical_execution_authorized": False,
            }
        )


# ============================================================
# OBSERVATION PROCESSOR
# ============================================================

class ObservationProcessor:

    @staticmethod
    def validate_value(
        observation_field: ObservationField,
        value: Any,
    ) -> Any:
        observation_type = (
            observation_field.observation_type
        )

        if observation_type == ObservationType.BOOLEAN:
            if not isinstance(value, bool):
                raise ObservationValidationError(
                    "Expected boolean"
                )

            return value

        if observation_type == ObservationType.CATEGORICAL:
            if (
                not isinstance(value, str)
                or value not in observation_field.allowed_values
            ):
                raise ObservationValidationError(
                    "Invalid categorical value"
                )

            return value

        if observation_type == ObservationType.IMAGE_METADATA:
            if not isinstance(value, Mapping):
                raise ObservationValidationError(
                    "Image metadata must be a mapping"
                )

            required_keys = (
                "width",
                "height",
                "channels",
            )

            for key in required_keys:
                if key not in value:
                    raise ObservationValidationError(
                        f"Missing image metadata: {key}"
                    )

                validate_integer(
                    value[key],
                    key,
                    minimum=1,
                )

            if (
                value["width"] > 16_384
                or value["height"] > 16_384
                or value["channels"] > 16
            ):
                raise ObservationValidationError(
                    "Image metadata dimensions exceed limits"
                )

            result = copy.deepcopy(dict(value))

            try:
                json.dumps(
                    result,
                    allow_nan=False,
                )
            except (TypeError, ValueError) as exc:
                raise ObservationValidationError(
                    "Image metadata is not JSON-compatible"
                ) from exc

            return result

        if observation_type == ObservationType.DISCRETE:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
            ):
                raise ObservationValidationError(
                    "Expected integer"
                )

            ObservationProcessor._check_bounds(
                observation_field,
                value,
            )

            return value

        if observation_type == ObservationType.SCALAR:
            result = finite_number(
                value,
                observation_field.name,
            )

            ObservationProcessor._check_bounds(
                observation_field,
                result,
            )

            return result

        if observation_type == ObservationType.VECTOR:
            if (
                not isinstance(value, (list, tuple))
                or len(value) != observation_field.dimension
            ):
                raise ObservationValidationError(
                    "Vector dimension mismatch"
                )

            result = []

            for component in value:
                numeric = finite_number(
                    component,
                    observation_field.name,
                )

                ObservationProcessor._check_bounds(
                    observation_field,
                    numeric,
                )

                result.append(numeric)

            return result

        raise ObservationValidationError(
            "Unsupported observation type"
        )

    @staticmethod
    def _check_bounds(
        observation_field: ObservationField,
        value: float,
    ) -> None:
        if (
            observation_field.minimum is not None
            and value < observation_field.minimum
        ):
            raise ObservationValidationError(
                f"{observation_field.name} below minimum"
            )

        if (
            observation_field.maximum is not None
            and value > observation_field.maximum
        ):
            raise ObservationValidationError(
                f"{observation_field.name} above maximum"
            )

    @staticmethod
    def normalize(
        observation_field: ObservationField,
        value: Any,
    ) -> Any:
        mode = observation_field.normalization

        if mode == NormalizationType.NONE:
            return copy.deepcopy(value)

        if isinstance(value, list):
            return [
                ObservationProcessor._normalize_number(
                    observation_field,
                    item,
                    index,
                )
                for index, item in enumerate(value)
            ]

        return ObservationProcessor._normalize_number(
            observation_field,
            value,
            0,
        )

    @staticmethod
    def _normalize_number(
        observation_field: ObservationField,
        value: float,
        index: int,
    ) -> float:
        mode = observation_field.normalization

        numeric = finite_number(
            value,
            "normalization value",
        )

        if mode == NormalizationType.MIN_MAX:
            result = (
                (numeric - observation_field.minimum)
                / (
                    observation_field.maximum
                    - observation_field.minimum
                )
            )

            return max(0.0, min(1.0, result))

        if mode == NormalizationType.NEGATIVE_ONE_TO_ONE:
            result = (
                2.0
                * (
                    numeric - observation_field.minimum
                )
                / (
                    observation_field.maximum
                    - observation_field.minimum
                )
                - 1.0
            )

            return max(-1.0, min(1.0, result))

        if mode == NormalizationType.Z_SCORE:
            mean = observation_field.mean
            std = observation_field.standard_deviation

            if isinstance(mean, tuple):
                mean = mean[index]

            if isinstance(std, tuple):
                std = std[index]

            return (
                numeric - float(mean)
            ) / float(std)

        return numeric


# ============================================================
# OBSERVATION SPACE ENGINE
# ============================================================

class RoboticsObservationSpaceEngine:

    def __init__(
        self,
        config: ObservationSpaceConfig | None = None,
    ) -> None:
        self.config = (
            config or ObservationSpaceConfig()
        )

        self._spaces: dict[
            tuple[str, str],
            RobotObservationSpace,
        ] = {}

        self._masks: dict[
            tuple[str, str],
            ObservationMask,
        ] = {}

        self._history: dict[
            tuple[str, str, str],
            deque[ObservationFrame],
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
    # REGISTER OBSERVATION SPACE
    # --------------------------------------------------------

    def register_space(
        self,
        space: RobotObservationSpace,
        *,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise ObservationPermissionDenied(
                "Authorized registration required"
            )

        if not isinstance(
            space,
            RobotObservationSpace,
        ):
            raise ObservationValidationError(
                "Invalid observation space"
            )

        validate_id(actor_id, "actor_id")

        if (
            len(space.fields)
            > self.config.maximum_fields_per_space
        ):
            raise ObservationValidationError(
                "Too many observation fields"
            )

        for item in space.fields:
            if (
                item.dimension
                > self.config.maximum_vector_dimension
            ):
                raise ObservationValidationError(
                    "Observation dimension exceeds limit"
                )

        key = (space.tenant_id, space.space_id)

        with self._lock:
            if key in self._spaces:
                raise ObservationConflict(
                    "Observation space already exists"
                )

            if (
                len(self._spaces)
                >= self.config.maximum_spaces
            ):
                raise ObservationConflict(
                    "Observation space capacity reached"
                )

            self._spaces[key] = space

            self._log(
                tenant_id=space.tenant_id,
                action="observation_space_registered",
                entity_id=space.space_id,
                actor_id=actor_id,
                details={
                    "robot_id": space.robot_id,
                    "field_count": len(space.fields),
                },
            )

            return space.to_dict()

    # --------------------------------------------------------
    # UPDATE OBSERVATION SPACE
    # --------------------------------------------------------

    def update_space(
        self,
        space: RobotObservationSpace,
        *,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise ObservationPermissionDenied(
                "Authorized update required"
            )

        with self._lock:
            current = self._get_space(
                space.tenant_id,
                space.space_id,
            )

            if current.robot_id != space.robot_id:
                raise ObservationConflict(
                    "Cannot change robot identity"
                )

            if space.version != current.version + 1:
                raise ObservationConflict(
                    "Version must increase by one"
                )

            if (
                len(space.fields)
                > self.config.maximum_fields_per_space
                or any(
                    item.dimension
                    > self.config.maximum_vector_dimension
                    for item in space.fields
                )
            ):
                raise ObservationValidationError(
                    "Observation space exceeds limits"
                )

            key = (
                space.tenant_id,
                space.space_id,
            )

            self._spaces[key] = space

            self._masks.pop(key, None)

            self._log(
                tenant_id=space.tenant_id,
                action="observation_space_updated",
                entity_id=space.space_id,
                actor_id=actor_id,
                details={
                    "version": space.version,
                },
            )

            return space.to_dict()

    # --------------------------------------------------------
    # GET OBSERVATION SPACE
    # --------------------------------------------------------

    def _get_space(
        self,
        tenant_id: str,
        space_id: str,
    ) -> RobotObservationSpace:
        validate_id(tenant_id, "tenant_id")
        validate_id(space_id, "space_id")

        space = self._spaces.get(
            (tenant_id, space_id)
        )

        if space is None:
            raise ObservationSpaceNotFound(
                "Observation space not found"
            )

        return space

    def get_space(
        self,
        *,
        tenant_id: str,
        space_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            return self._get_space(
                tenant_id,
                space_id,
            ).to_dict()

    # --------------------------------------------------------
    # LIST OBSERVATION SPACES
    # --------------------------------------------------------

    def list_spaces(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        if robot_id is not None:
            validate_id(robot_id, "robot_id")

        validate_integer(
            limit,
            "limit",
            minimum=1,
        )

        with self._lock:
            spaces = [
                space.to_dict()
                for (tenant, _), space
                in self._spaces.items()
                if (
                    tenant == tenant_id
                    and (
                        robot_id is None
                        or space.robot_id == robot_id
                    )
                )
            ]

            return spaces[:limit]

    # --------------------------------------------------------
    # SET OBSERVATION MASK
    # --------------------------------------------------------

    def set_observation_mask(
        self,
        mask: ObservationMask,
        *,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise ObservationPermissionDenied(
                "Authorized mask update required"
            )

        with self._lock:
            space = self._get_space(
                mask.tenant_id,
                mask.space_id,
            )

            if space.robot_id != mask.robot_id:
                raise ObservationConflict(
                    "Observation mask robot mismatch"
                )

            known_fields = {
                item.name
                for item in space.fields
            }

            unknown = (
                mask.enabled_fields - known_fields
            )

            if unknown:
                raise ObservationValidationError(
                    f"Unknown mask fields: {sorted(unknown)}"
                )

            self._masks[
                (mask.tenant_id, mask.space_id)
            ] = mask

            self._log(
                tenant_id=mask.tenant_id,
                action="observation_mask_updated",
                entity_id=mask.space_id,
                actor_id=actor_id,
            )

            return mask.to_dict()

    # --------------------------------------------------------
    # PROCESS OBSERVATION
    # --------------------------------------------------------

    def process_observation(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        space_id: str,
        samples: Mapping[str, ObservationSample],
        captured_at: datetime | None = None,
        store_history: bool = True,
    ) -> dict[str, Any]:
        if not isinstance(samples, Mapping):
            raise ObservationValidationError(
                "samples must be a mapping"
            )

        timestamp = captured_at or utc_now()

        validate_datetime(
            timestamp,
            "captured_at",
        )

        with self._lock:
            space = self._get_space(
                tenant_id,
                space_id,
            )

            if space.robot_id != robot_id:
                raise ObservationPermissionDenied(
                    "Robot identity mismatch"
                )

            mask = self._masks.get(
                (tenant_id, space_id)
            )

            known_fields = {
                item.name
                for item in space.fields
            }

            unknown_fields = (
                set(samples) - known_fields
            )

            errors = []
            warnings = []

            if unknown_fields:
                errors.append(
                    f"Unknown fields: {sorted(unknown_fields)}"
                )

            values = {}
            normalized_values = {}
            quality = {}

            for observation_field in space.fields:
                field_name = observation_field.name

                if (
                    mask is not None
                    and field_name
                    not in mask.enabled_fields
                ):
                    quality[field_name] = (
                        ObservationQuality.MASKED.value
                    )

                    if observation_field.safety_critical:
                        errors.append(
                            f"Safety-critical field masked: "
                            f"{field_name}"
                        )

                    continue

                sample = samples.get(field_name)

                if sample is None:
                    quality[field_name] = (
                        ObservationQuality.MISSING.value
                    )

                    if (
                        observation_field.required
                        or observation_field.safety_critical
                    ):
                        errors.append(
                            f"Missing required observation: "
                            f"{field_name}"
                        )

                    continue

                if not isinstance(
                    sample,
                    ObservationSample,
                ):
                    quality[field_name] = (
                        ObservationQuality.INVALID.value
                    )

                    errors.append(
                        f"Invalid sample object: {field_name}"
                    )

                    continue

                if sample.field_name != field_name:
                    quality[field_name] = (
                        ObservationQuality.INVALID.value
                    )

                    errors.append(
                        f"Field name mismatch: {field_name}"
                    )

                    continue

                age_seconds = (
                    timestamp - sample.timestamp
                ).total_seconds()

                if (
                    self.config.reject_future_timestamps
                    and age_seconds
                    < -self.config.future_timestamp_tolerance_seconds
                ):
                    quality[field_name] = (
                        ObservationQuality.INVALID.value
                    )

                    errors.append(
                        f"Future timestamp: {field_name}"
                    )

                    continue

                stale_threshold = (
                    observation_field.stale_after_seconds
                    or self.config.default_stale_after_seconds
                )

                if age_seconds > stale_threshold:
                    quality[field_name] = (
                        ObservationQuality.STALE.value
                    )

                    if (
                        observation_field.required
                        or observation_field.safety_critical
                    ):
                        errors.append(
                            f"Stale observation: {field_name}"
                        )

                    else:
                        warnings.append(
                            f"Optional observation stale: "
                            f"{field_name}"
                        )

                    continue

                try:
                    validated_value = (
                        ObservationProcessor.validate_value(
                            observation_field,
                            sample.value,
                        )
                    )

                    normalized_value = (
                        ObservationProcessor.normalize(
                            observation_field,
                            validated_value,
                        )
                    )

                except ObservationValidationError as exc:
                    quality[field_name] = (
                        ObservationQuality.INVALID.value
                    )

                    errors.append(
                        f"{field_name}: {exc}"
                    )

                    continue

                values[field_name] = validated_value

                normalized_values[
                    field_name
                ] = normalized_value

                quality[field_name] = (
                    ObservationQuality.VALID.value
                )

            frame = ObservationFrame(
                frame_id=new_id(),
                tenant_id=tenant_id,
                robot_id=robot_id,
                space_id=space_id,
                space_version=space.version,
                values=values,
                normalized_values=normalized_values,
                quality=quality,
                valid=not errors,
                errors=tuple(errors),
                warnings=tuple(warnings),
                captured_at=timestamp,
            )

            if store_history:
                history_key = (
                    tenant_id,
                    robot_id,
                    space_id,
                )

                if history_key not in self._history:
                    self._history[
                        history_key
                    ] = deque(
                        maxlen=(
                            self.config.maximum_history_per_robot
                        )
                    )

                self._history[
                    history_key
                ].append(frame)

            if errors:
                self._log(
                    tenant_id=tenant_id,
                    action="observation_validation_failed",
                    entity_id=frame.frame_id,
                    details={
                        "robot_id": robot_id,
                        "space_id": space_id,
                        "error_count": len(errors),
                    },
                )

            return frame.to_dict()

    # --------------------------------------------------------
    # BATCH PROCESSING
    # --------------------------------------------------------

    def process_batch(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        space_id: str,
        batches: Sequence[
            Mapping[str, ObservationSample]
        ],
    ) -> list[dict[str, Any]]:
        if (
            not isinstance(batches, (list, tuple))
            or not 1 <= len(batches)
            <= self.config.maximum_batch_size
        ):
            raise ObservationValidationError(
                "Invalid observation batch"
            )

        return [
            self.process_observation(
                tenant_id=tenant_id,
                robot_id=robot_id,
                space_id=space_id,
                samples=batch,
            )
            for batch in batches
        ]

    # --------------------------------------------------------
    # FLATTEN OBSERVATION
    # --------------------------------------------------------

    def flatten_observation(
        self,
        *,
        tenant_id: str,
        space_id: str,
        frame: Mapping[str, Any],
        normalized: bool = True,
    ) -> dict[str, Any]:
        with self._lock:
            space = self._get_space(
                tenant_id,
                space_id,
            )

            if (
                frame.get("tenant_id") != tenant_id
                or frame.get("space_id") != space_id
                or frame.get("robot_id") != space.robot_id
                or frame.get("space_version")
                != space.version
            ):
                raise ObservationPermissionDenied(
                    "Frame identity or version mismatch"
                )

            if frame.get("valid") is not True:
                raise ObservationValidationError(
                    "Cannot flatten invalid observation frame"
                )

            source = (
                frame.get("normalized_values")
                if normalized
                else frame.get("values")
            )

            if not isinstance(source, Mapping):
                raise ObservationValidationError(
                    "Invalid frame values"
                )

            vector = []
            feature_names = []

            for item in space.fields:
                if item.name not in source:
                    continue

                value = source[item.name]

                if item.observation_type == ObservationType.VECTOR:
                    for index, component in enumerate(value):
                        vector.append(
                            finite_number(
                                component,
                                item.name,
                            )
                        )

                        feature_names.append(
                            f"{item.name}[{index}]"
                        )

                elif item.observation_type in (
                    ObservationType.SCALAR,
                    ObservationType.DISCRETE,
                ):
                    vector.append(
                        finite_number(
                            value,
                            item.name,
                        )
                    )

                    feature_names.append(
                        item.name
                    )

                elif item.observation_type == ObservationType.BOOLEAN:
                    vector.append(
                        1.0 if value else 0.0
                    )

                    feature_names.append(
                        item.name
                    )

                elif item.observation_type == ObservationType.CATEGORICAL:
                    for category in item.allowed_values:
                        vector.append(
                            1.0 if value == category else 0.0
                        )

                        feature_names.append(
                            f"{item.name}={category}"
                        )

                elif item.observation_type == ObservationType.IMAGE_METADATA:
                    for key in (
                        "width",
                        "height",
                        "channels",
                    ):
                        vector.append(
                            float(value[key])
                        )

                        feature_names.append(
                            f"{item.name}.{key}"
                        )

            return {
                "frame_id": frame["frame_id"],
                "space_id": space_id,
                "space_version": space.version,
                "feature_names": feature_names,
                "vector": vector,
                "dimension": len(vector),
                "normalized": normalized,
                "simulation_only": True,
            }

    # --------------------------------------------------------
    # OBSERVATION HISTORY
    # --------------------------------------------------------

    def observation_history(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        space_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_integer(
            limit,
            "limit",
            minimum=1,
        )

        with self._lock:
            space = self._get_space(
                tenant_id,
                space_id,
            )

            if space.robot_id != robot_id:
                raise ObservationPermissionDenied(
                    "Robot identity mismatch"
                )

            history = self._history.get(
                (tenant_id, robot_id, space_id),
                (),
            )

            return [
                frame.to_dict()
                for frame in list(history)[-limit:]
            ]

    # --------------------------------------------------------
    # LATEST OBSERVATION
    # --------------------------------------------------------

    def latest_observation(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        space_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            history = self.observation_history(
                tenant_id=tenant_id,
                robot_id=robot_id,
                space_id=space_id,
                limit=1,
            )

            if not history:
                raise ObservationSpaceNotFound(
                    "No observation frames available"
                )

            return history[-1]

    # --------------------------------------------------------
    # OBSERVATION SUMMARY
    # --------------------------------------------------------

    def observation_summary(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        space_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            space = self._get_space(
                tenant_id,
                space_id,
            )

            if space.robot_id != robot_id:
                raise ObservationPermissionDenied(
                    "Robot identity mismatch"
                )

            history = list(
                self._history.get(
                    (tenant_id, robot_id, space_id),
                    (),
                )
            )

            valid_frames = sum(
                frame.valid
                for frame in history
            )

            quality_counts = {
                quality.value: 0
                for quality in ObservationQuality
            }

            for frame in history:
                for value in frame.quality.values():
                    quality_counts[value] += 1

            return {
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "space_id": space_id,
                "space_version": space.version,
                "field_count": len(space.fields),
                "history_count": len(history),
                "valid_frames": valid_frames,
                "invalid_frames": (
                    len(history) - valid_frames
                ),
                "quality_counts": quality_counts,
                "latest_frame_id": (
                    history[-1].frame_id
                    if history else None
                ),
                "simulation_only": True,
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
        )

        with self._lock:
            return [
                copy.deepcopy(event)
                for event in self._audit
                if event["tenant_id"] == tenant_id
            ][-limit:]

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
            spaces = [
                space.to_dict()
                for (tenant, _), space
                in self._spaces.items()
                if tenant == tenant_id
            ]

            history = [
                frame.to_dict()
                for (tenant, _, _), frames
                in self._history.items()
                if tenant == tenant_id
                for frame in frames
            ]

            audit = [
                copy.deepcopy(event)
                for event in self._audit
                if event["tenant_id"] == tenant_id
            ]

        payload = {
            "generated_at": utc_now().isoformat(),
            "tenant_id": tenant_id,
            "observation_spaces": spaces,
            "observation_history": history,
            "audit_events": audit,
            "simulation_only": True,
            "physical_execution_authorized": False,
        }

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

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
# DEFAULT MOBILE ROBOT OBSERVATION SPACE
# ============================================================

def create_mobile_robot_observation_space(
    *,
    tenant_id: str,
    robot_id: str,
    space_id: str = "mobile_robot_observations",
) -> RobotObservationSpace:

    fields = (
        ObservationField(
            name="position",
            observation_type=ObservationType.VECTOR,
            category=ObservationCategory.POSITION,
            dimension=3,
            minimum=-1000.0,
            maximum=1000.0,
            required=True,
            normalization=(
                NormalizationType.NEGATIVE_ONE_TO_ONE
            ),
            unit="m",
        ),
        ObservationField(
            name="orientation",
            observation_type=ObservationType.VECTOR,
            category=ObservationCategory.ORIENTATION,
            dimension=3,
            minimum=-math.pi,
            maximum=math.pi,
            required=True,
            unit="rad",
        ),
        ObservationField(
            name="linear_velocity",
            observation_type=ObservationType.VECTOR,
            category=ObservationCategory.VELOCITY,
            dimension=3,
            minimum=-5.0,
            maximum=5.0,
            required=True,
            normalization=(
                NormalizationType.NEGATIVE_ONE_TO_ONE
            ),
            unit="m/s",
        ),
        ObservationField(
            name="angular_velocity",
            observation_type=ObservationType.VECTOR,
            category=ObservationCategory.VELOCITY,
            dimension=3,
            minimum=-10.0,
            maximum=10.0,
            required=True,
            unit="rad/s",
        ),
        ObservationField(
            name="lidar_distance",
            observation_type=ObservationType.VECTOR,
            category=ObservationCategory.LIDAR,
            dimension=8,
            minimum=0.0,
            maximum=30.0,
            required=True,
            safety_critical=True,
            stale_after_seconds=0.5,
            normalization=NormalizationType.MIN_MAX,
            unit="m",
        ),
        ObservationField(
            name="imu_acceleration",
            observation_type=ObservationType.VECTOR,
            category=ObservationCategory.IMU,
            dimension=3,
            minimum=-100.0,
            maximum=100.0,
            required=True,
            stale_after_seconds=0.5,
            unit="m/s²",
        ),
        ObservationField(
            name="battery_soc",
            observation_type=ObservationType.SCALAR,
            category=ObservationCategory.BATTERY,
            minimum=0.0,
            maximum=100.0,
            required=True,
            normalization=NormalizationType.MIN_MAX,
            unit="%",
        ),
        ObservationField(
            name="motor_temperature",
            observation_type=ObservationType.SCALAR,
            category=ObservationCategory.TEMPERATURE,
            minimum=-40.0,
            maximum=150.0,
            required=True,
            safety_critical=True,
            unit="°C",
        ),
        ObservationField(
            name="obstacle_detected",
            observation_type=ObservationType.BOOLEAN,
            category=ObservationCategory.NAVIGATION,
            required=True,
            safety_critical=True,
        ),
        ObservationField(
            name="robot_health",
            observation_type=ObservationType.CATEGORICAL,
            category=ObservationCategory.HEALTH,
            allowed_values=(
                "healthy",
                "warning",
                "critical",
                "unknown",
            ),
            required=True,
        ),
        ObservationField(
            name="camera_metadata",
            observation_type=ObservationType.IMAGE_METADATA,
            category=ObservationCategory.CAMERA,
            required=False,
        ),
    )

    return RobotObservationSpace(
        space_id=space_id,
        tenant_id=tenant_id,
        robot_id=robot_id,
        name="Mobile Robot Observation Space",
        fields=fields,
        description=(
            "Simulation observations for mobile robot "
            "navigation, perception, battery, telemetry "
            "and health monitoring."
        ),
    )


# ============================================================
# DEFAULT MANIPULATOR OBSERVATION SPACE
# ============================================================

def create_manipulator_observation_space(
    *,
    tenant_id: str,
    robot_id: str,
    joint_count: int = 6,
    space_id: str = "manipulator_observations",
) -> RobotObservationSpace:

    validate_integer(
        joint_count,
        "joint_count",
        minimum=1,
        maximum=128,
    )

    fields = (
        ObservationField(
            name="joint_positions",
            observation_type=ObservationType.VECTOR,
            category=ObservationCategory.JOINT,
            dimension=joint_count,
            minimum=-math.pi,
            maximum=math.pi,
            required=True,
            unit="rad",
        ),
        ObservationField(
            name="joint_velocities",
            observation_type=ObservationType.VECTOR,
            category=ObservationCategory.JOINT,
            dimension=joint_count,
            minimum=-20.0,
            maximum=20.0,
            required=True,
            unit="rad/s",
        ),
        ObservationField(
            name="joint_torques",
            observation_type=ObservationType.VECTOR,
            category=ObservationCategory.JOINT,
            dimension=joint_count,
            minimum=-500.0,
            maximum=500.0,
            required=True,
            safety_critical=True,
            unit="Nm",
        ),
        ObservationField(
            name="end_effector_position",
            observation_type=ObservationType.VECTOR,
            category=ObservationCategory.POSITION,
            dimension=3,
            minimum=-10.0,
            maximum=10.0,
            required=True,
            unit="m",
        ),
        ObservationField(
            name="gripper_force",
            observation_type=ObservationType.SCALAR,
            category=ObservationCategory.FORCE,
            minimum=0.0,
            maximum=1000.0,
            required=True,
            safety_critical=True,
            unit="N",
        ),
        ObservationField(
            name="object_detected",
            observation_type=ObservationType.BOOLEAN,
            category=ObservationCategory.CAMERA,
            required=True,
        ),
        ObservationField(
            name="robot_health",
            observation_type=ObservationType.CATEGORICAL,
            category=ObservationCategory.HEALTH,
            allowed_values=(
                "healthy",
                "warning",
                "critical",
                "unknown",
            ),
            required=True,
        ),
    )

    return RobotObservationSpace(
        space_id=space_id,
        tenant_id=tenant_id,
        robot_id=robot_id,
        name="Manipulator Observation Space",
        fields=fields,
        description=(
            "Simulation observation schema for robotic "
            "arm joints, end-effector, gripper and health."
        ),
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    engine = RoboticsObservationSpaceEngine()

    tenant_id = "tenant_alpha"
    robot_id = "robot_001"
    space_id = "mobile_robot_observations"

    space = create_mobile_robot_observation_space(
        tenant_id=tenant_id,
        robot_id=robot_id,
        space_id=space_id,
    )

    engine.register_space(
        space,
        authorized=True,
        actor_id="simulation_admin",
    )

    timestamp = utc_now()

    samples = {
        "position": ObservationSample(
            field_name="position",
            value=[2.0, 3.0, 0.0],
            timestamp=timestamp,
        ),
        "orientation": ObservationSample(
            field_name="orientation",
            value=[0.0, 0.0, 1.57],
            timestamp=timestamp,
        ),
        "linear_velocity": ObservationSample(
            field_name="linear_velocity",
            value=[0.5, 0.0, 0.0],
            timestamp=timestamp,
        ),
        "angular_velocity": ObservationSample(
            field_name="angular_velocity",
            value=[0.0, 0.0, 0.1],
            timestamp=timestamp,
        ),
        "lidar_distance": ObservationSample(
            field_name="lidar_distance",
            value=[
                3.2, 4.1, 5.0, 6.0,
                2.5, 1.8, 4.4, 5.6,
            ],
            timestamp=timestamp,
        ),
        "imu_acceleration": ObservationSample(
            field_name="imu_acceleration",
            value=[0.1, 0.0, 9.81],
            timestamp=timestamp,
        ),
        "battery_soc": ObservationSample(
            field_name="battery_soc",
            value=82.5,
            timestamp=timestamp,
        ),
        "motor_temperature": ObservationSample(
            field_name="motor_temperature",
            value=42.0,
            timestamp=timestamp,
        ),
        "obstacle_detected": ObservationSample(
            field_name="obstacle_detected",
            value=True,
            timestamp=timestamp,
        ),
        "robot_health": ObservationSample(
            field_name="robot_health",
            value="healthy",
            timestamp=timestamp,
        ),
        "camera_metadata": ObservationSample(
            field_name="camera_metadata",
            value={
                "width": 1280,
                "height": 720,
                "channels": 3,
            },
            timestamp=timestamp,
        ),
    }

    print("\nPROCESS OBSERVATION")

    frame = engine.process_observation(
        tenant_id=tenant_id,
        robot_id=robot_id,
        space_id=space_id,
        samples=samples,
    )

    print(
        json.dumps(
            frame,
            indent=2,
        )
    )

    print("\nFLATTEN OBSERVATION FOR ML / RL")

    flattened = engine.flatten_observation(
        tenant_id=tenant_id,
        space_id=space_id,
        frame=frame,
        normalized=True,
    )

    print(
        json.dumps(
            flattened,
            indent=2,
        )
    )

    print("\nOBSERVATION SUMMARY")

    print(
        json.dumps(
            engine.observation_summary(
                tenant_id=tenant_id,
                robot_id=robot_id,
                space_id=space_id,
            ),
            indent=2,
        )
    )

    print("\nLATEST OBSERVATION")

    print(
        json.dumps(
            engine.latest_observation(
                tenant_id=tenant_id,
                robot_id=robot_id,
                space_id=space_id,
            ),
            indent=2,
        )
    )

    print("\nEXPORT OBSERVATION REPORT")

    output = engine.export_json(
        "data/robotics/observation_space_report.json",
        tenant_id=tenant_id,
    )

    print("Exported:", output)


if __name__ == "__main__":
    main()
"""Action Space module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/action_space.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Robotics Action Space Management Engine

Python 3.11+
Dependencies: Standard library only

Features:
    - Discrete and continuous robot action spaces
    - Multi-dimensional and hybrid action spaces
    - Joint position, velocity and torque actions
    - Mobile robot navigation actions
    - Manipulator and gripper actions
    - Action parameter validation
    - Action bounds and normalization
    - Action masking
    - Robot capability constraints
    - Workspace and speed limits
    - Safety-critical action identification
    - Human approval requirements
    - Action sequence validation
    - Action space versioning
    - Multi-tenant robot isolation
    - Thread-safe action registry
    - Audit logging
    - JSON export

Integrations:
    robot_capabilities.py
    robot_controller.py
    command_validator.py
    command_dispatcher.py
    robot_state.py
    safety_monitor.py
    workspace_limits.py
    speed_limits.py
    command_interlock.py
    human_approval.py
    simulation_adapter.py
    reinforcement_learning.py
    vla.py
    planning.py

Safety:
    This module defines and validates action spaces.
    It does not execute commands or control physical robots.

    All action validation is advisory. A successful
    validation does not imply that an action is safe
    to execute on physical hardware.

    Safety-critical commands require independent
    safety validation and human authorization.
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

class ActionSpaceError(Exception):
    pass


class ActionValidationError(ActionSpaceError):
    pass


class ActionSpaceNotFound(ActionSpaceError):
    pass


class ActionNotFound(ActionSpaceError):
    pass


class ActionSpaceConflict(ActionSpaceError):
    pass


class ActionPermissionDenied(ActionSpaceError):
    pass


# ============================================================
# ENUMS
# ============================================================

class ActionSpaceType(str, Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    MULTI_DISCRETE = "multi_discrete"
    MULTI_CONTINUOUS = "multi_continuous"
    HYBRID = "hybrid"


class ActionCategory(str, Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    LOCOMOTION = "locomotion"
    JOINT_CONTROL = "joint_control"
    GRIPPER_CONTROL = "gripper_control"
    SENSOR_CONTROL = "sensor_control"
    CAMERA_CONTROL = "camera_control"
    TASK_CONTROL = "task_control"
    SIMULATION = "simulation"
    OTHER = "other"


class ActionParameterType(str, Enum):
    FLOAT = "float"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    STRING = "string"
    ENUM = "enum"


class ActionSafetyLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ActionValidationStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    MASKED = "masked"
    REQUIRES_APPROVAL = "requires_approval"


class ActionExecutionMode(str, Enum):
    SIMULATION = "simulation"
    ADVISORY = "advisory"


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
        raise ActionValidationError(
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
        raise ActionValidationError(
            f"{name} must be a string"
        )

    result = value.strip()

    if len(result) > maximum:
        raise ActionValidationError(
            f"{name} exceeds maximum length"
        )

    if not allow_empty and not result:
        raise ActionValidationError(
            f"{name} cannot be empty"
        )

    return result


def validate_integer(
    value: int,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = 1000000,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ActionValidationError(
            f"Invalid {name}"
        )

    return value


def finite_number(
    value: Any,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise ActionValidationError(
            f"{name} must be numeric"
        )

    if not isinstance(value, (int, float)):
        raise ActionValidationError(
            f"{name} must be numeric"
        )

    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ActionValidationError(
            f"Invalid {name}"
        ) from exc

    if not math.isfinite(result):
        raise ActionValidationError(
            f"{name} must be finite"
        )

    return result


def validate_boolean(
    value: Any,
    name: str,
) -> bool:
    if not isinstance(value, bool):
        raise ActionValidationError(
            f"{name} must be boolean"
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
        raise ActionValidationError(
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

    if isinstance(value, (tuple, list)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ActionValidationError(
                "Non-finite value cannot be serialized"
            )

    return value


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ActionSpaceConfig:
    maximum_action_spaces: int = 10000
    maximum_actions_per_space: int = 1000
    maximum_parameters_per_action: int = 128
    maximum_sequence_length: int = 10000
    maximum_audit_events: int = 20000

    execution_mode: ActionExecutionMode = (
        ActionExecutionMode.SIMULATION
    )

    human_approval_for_high_risk: bool = True
    require_action_mask_check: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_action_spaces",
            "maximum_actions_per_space",
            "maximum_parameters_per_action",
            "maximum_sequence_length",
            "maximum_audit_events",
        ):
            validate_integer(
                getattr(self, name),
                name,
                minimum=1,
            )

        if not isinstance(
            self.execution_mode,
            ActionExecutionMode,
        ):
            raise ActionValidationError(
                "Invalid execution mode"
            )

        if self.human_approval_for_high_risk is not True:
            raise ActionValidationError(
                "High-risk actions must require approval"
            )

        validate_boolean(
            self.require_action_mask_check,
            "require_action_mask_check",
        )


# ============================================================
# ACTION PARAMETER DEFINITION
# ============================================================

@dataclass(frozen=True)
class ActionParameter:
    name: str

    parameter_type: ActionParameterType

    required: bool = True

    minimum: float | None = None
    maximum: float | None = None

    allowed_values: tuple[Any, ...] = ()

    unit: str = ""

    description: str = ""

    default: Any = None

    def __post_init__(self) -> None:
        validate_id(self.name, "parameter name")

        if not isinstance(
            self.parameter_type,
            ActionParameterType,
        ):
            raise ActionValidationError(
                "Invalid parameter type"
            )

        validate_boolean(
            self.required,
            "required",
        )

        if self.minimum is not None:
            finite_number(
                self.minimum,
                "minimum",
            )

        if self.maximum is not None:
            finite_number(
                self.maximum,
                "maximum",
            )

        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ActionValidationError(
                "Parameter minimum exceeds maximum"
            )

        if not isinstance(
            self.allowed_values,
            tuple,
        ):
            raise ActionValidationError(
                "allowed_values must be a tuple"
            )

        if self.parameter_type == ActionParameterType.ENUM:
            if not self.allowed_values:
                raise ActionValidationError(
                    "ENUM parameter requires allowed values"
                )

        for value in self.allowed_values:
            if isinstance(value, (dict, list, set, tuple)):
                raise ActionValidationError(
                    "Allowed values must be scalar"
                )

            self._validate_scalar(value)

        validate_text(
            self.unit,
            "unit",
            allow_empty=True,
            maximum=64,
        )

        validate_text(
            self.description,
            "description",
            allow_empty=True,
        )

        if self.default is not None:
            self.validate_value(self.default)

    def _validate_scalar(
        self,
        value: Any,
    ) -> None:
        if self.parameter_type == ActionParameterType.FLOAT:
            finite_number(
                value,
                self.name,
            )

        elif self.parameter_type == ActionParameterType.INTEGER:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
            ):
                raise ActionValidationError(
                    f"{self.name} must be integer"
                )

        elif self.parameter_type == ActionParameterType.BOOLEAN:
            validate_boolean(
                value,
                self.name,
            )

        elif self.parameter_type in (
            ActionParameterType.STRING,
            ActionParameterType.ENUM,
        ):
            if not isinstance(value, str):
                raise ActionValidationError(
                    f"{self.name} must be string"
                )

            validate_text(
                value,
                self.name,
                maximum=1024,
            )

    def validate_value(
        self,
        value: Any,
    ) -> Any:
        self._validate_scalar(value)

        if self.parameter_type in (
            ActionParameterType.FLOAT,
            ActionParameterType.INTEGER,
        ):
            number = finite_number(
                value,
                self.name,
            )

            if (
                self.minimum is not None
                and number < self.minimum
            ):
                raise ActionValidationError(
                    f"{self.name} below minimum "
                    f"{self.minimum}"
                )

            if (
                self.maximum is not None
                and number > self.maximum
            ):
                raise ActionValidationError(
                    f"{self.name} above maximum "
                    f"{self.maximum}"
                )

        if (
            self.allowed_values
            and value not in self.allowed_values
        ):
            raise ActionValidationError(
                f"{self.name} not in allowed values"
            )

        return value

    def normalize(
        self,
        value: Any,
    ) -> float:
        if self.parameter_type not in (
            ActionParameterType.FLOAT,
            ActionParameterType.INTEGER,
        ):
            raise ActionValidationError(
                "Only numeric parameters can be normalized"
            )

        if (
            self.minimum is None
            or self.maximum is None
        ):
            raise ActionValidationError(
                "Normalization requires finite bounds"
            )

        self.validate_value(value)

        if self.maximum == self.minimum:
            return 0.0

        return (
            2.0
            * (
                float(value) - self.minimum
            )
            / (
                self.maximum - self.minimum
            )
            - 1.0
        )

    def denormalize(
        self,
        value: float,
    ) -> float | int:
        if self.parameter_type not in (
            ActionParameterType.FLOAT,
            ActionParameterType.INTEGER,
        ):
            raise ActionValidationError(
                "Only numeric parameters can be denormalized"
            )

        if (
            self.minimum is None
            or self.maximum is None
        ):
            raise ActionValidationError(
                "Denormalization requires finite bounds"
            )

        normalized = finite_number(
            value,
            "normalized value",
        )

        if not -1.0 <= normalized <= 1.0:
            raise ActionValidationError(
                "Normalized value must be in [-1, 1]"
            )

        result = (
            self.minimum
            + (
                normalized + 1.0
            )
            * (
                self.maximum - self.minimum
            )
            / 2.0
        )

        if self.parameter_type == ActionParameterType.INTEGER:
            return int(round(result))

        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameter_type": self.parameter_type.value,
            "required": self.required,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "allowed_values": list(
                self.allowed_values
            ),
            "unit": self.unit,
            "description": self.description,
            "default": self.default,
        }


# ============================================================
# ROBOT ACTION DEFINITION
# ============================================================

@dataclass(frozen=True)
class RobotAction:
    action_id: str

    name: str
    category: ActionCategory

    parameters: tuple[ActionParameter, ...] = ()

    safety_level: ActionSafetyLevel = (
        ActionSafetyLevel.MODERATE
    )

    required_capabilities: tuple[str, ...] = ()

    enabled: bool = True

    human_approval_required: bool = False

    description: str = ""

    def __post_init__(self) -> None:
        validate_id(self.action_id, "action_id")

        validate_text(
            self.name,
            "action name",
        )

        if not isinstance(
            self.category,
            ActionCategory,
        ):
            raise ActionValidationError(
                "Invalid action category"
            )

        if not isinstance(
            self.safety_level,
            ActionSafetyLevel,
        ):
            raise ActionValidationError(
                "Invalid safety level"
            )

        if not isinstance(
            self.parameters,
            tuple,
        ):
            raise ActionValidationError(
                "parameters must be a tuple"
            )

        names = set()

        for parameter in self.parameters:
            if not isinstance(
                parameter,
                ActionParameter,
            ):
                raise ActionValidationError(
                    "Invalid action parameter"
                )

            if parameter.name in names:
                raise ActionValidationError(
                    "Duplicate parameter name"
                )

            names.add(parameter.name)

        if not isinstance(
            self.required_capabilities,
            tuple,
        ):
            raise ActionValidationError(
                "required_capabilities must be a tuple"
            )

        for capability in self.required_capabilities:
            validate_id(
                capability,
                "capability",
            )

        validate_boolean(
            self.enabled,
            "enabled",
        )

        validate_boolean(
            self.human_approval_required,
            "human_approval_required",
        )

        if (
            self.safety_level
            in (
                ActionSafetyLevel.HIGH,
                ActionSafetyLevel.CRITICAL,
            )
            and not self.human_approval_required
        ):
            raise ActionValidationError(
                "High-risk actions require human approval"
            )

        validate_text(
            self.description,
            "description",
            allow_empty=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "name": self.name,
            "category": self.category.value,
            "parameters": [
                item.to_dict()
                for item in self.parameters
            ],
            "safety_level": self.safety_level.value,
            "required_capabilities": list(
                self.required_capabilities
            ),
            "enabled": self.enabled,
            "human_approval_required": (
                self.human_approval_required
            ),
            "description": self.description,
        }


# ============================================================
# ACTION SPACE
# ============================================================

@dataclass(frozen=True)
class RobotActionSpace:
    space_id: str

    tenant_id: str
    robot_id: str

    name: str
    space_type: ActionSpaceType

    actions: tuple[RobotAction, ...]

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

        validate_text(
            self.name,
            "name",
        )

        if not isinstance(
            self.space_type,
            ActionSpaceType,
        ):
            raise ActionValidationError(
                "Invalid action space type"
            )

        if not isinstance(self.actions, tuple):
            raise ActionValidationError(
                "actions must be a tuple"
            )

        if not self.actions:
            raise ActionValidationError(
                "Action space cannot be empty"
            )

        action_ids = set()

        for action in self.actions:
            if not isinstance(
                action,
                RobotAction,
            ):
                raise ActionValidationError(
                    "Invalid robot action"
                )

            if action.action_id in action_ids:
                raise ActionValidationError(
                    "Duplicate action ID"
                )

            action_ids.add(action.action_id)

        validate_integer(
            self.version,
            "version",
            minimum=1,
        )

        validate_text(
            self.description,
            "description",
            allow_empty=True,
        )

        validate_datetime(
            self.created_at,
            "created_at",
        )

    def get_action(
        self,
        action_id: str,
    ) -> RobotAction:
        validate_id(action_id, "action_id")

        for action in self.actions:
            if action.action_id == action_id:
                return action

        raise ActionNotFound(
            f"Action not found: {action_id}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "space_id": self.space_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "name": self.name,
            "space_type": self.space_type.value,
            "version": self.version,
            "description": self.description,
            "actions": [
                action.to_dict()
                for action in self.actions
            ],
            "created_at": self.created_at.isoformat(),
            "execution_authorized": False,
        }


# ============================================================
# ACTION MASK
# ============================================================

@dataclass(frozen=True)
class ActionMask:
    tenant_id: str
    robot_id: str
    space_id: str

    allowed_action_ids: frozenset[str]

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

        if not isinstance(
            self.allowed_action_ids,
            frozenset,
        ):
            raise ActionValidationError(
                "allowed_action_ids must be frozenset"
            )

        for action_id in self.allowed_action_ids:
            validate_id(
                action_id,
                "action_id",
            )

        validate_text(
            self.reason,
            "reason",
            allow_empty=True,
        )

        validate_datetime(
            self.created_at,
            "created_at",
        )

    def is_allowed(
        self,
        action_id: str,
    ) -> bool:
        return action_id in self.allowed_action_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "space_id": self.space_id,
            "allowed_action_ids": sorted(
                self.allowed_action_ids
            ),
            "reason": self.reason,
            "created_at": self.created_at.isoformat(),
        }


# ============================================================
# ACTION REQUEST
# ============================================================

@dataclass(frozen=True)
class ActionRequest:
    tenant_id: str
    robot_id: str
    space_id: str
    action_id: str

    parameters: Mapping[str, Any] = field(
        default_factory=dict
    )

    available_capabilities: frozenset[str] = (
        frozenset()
    )

    approved_by: str | None = None

    request_id: str = field(
        default_factory=new_id
    )

    def __post_init__(self) -> None:
        for name in (
            "tenant_id",
            "robot_id",
            "space_id",
            "action_id",
            "request_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        if not isinstance(
            self.parameters,
            Mapping,
        ):
            raise ActionValidationError(
                "parameters must be a mapping"
            )

        for name in self.parameters:
            validate_id(
                name,
                "parameter name",
            )

        if not isinstance(
            self.available_capabilities,
            frozenset,
        ):
            raise ActionValidationError(
                "available_capabilities must be frozenset"
            )

        for capability in self.available_capabilities:
            validate_id(
                capability,
                "capability",
            )

        if self.approved_by is not None:
            validate_id(
                self.approved_by,
                "approved_by",
            )


# ============================================================
# ACTION VALIDATION RESULT
# ============================================================

@dataclass(frozen=True)
class ActionValidationResult:
    validation_id: str

    request_id: str

    tenant_id: str
    robot_id: str
    space_id: str
    action_id: str

    status: ActionValidationStatus

    valid: bool

    normalized_parameters: Mapping[str, float]

    resolved_parameters: Mapping[str, Any]

    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    human_approval_required: bool

    checked_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "validation_id": self.validation_id,
            "request_id": self.request_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "space_id": self.space_id,
            "action_id": self.action_id,
            "status": self.status.value,
            "valid": self.valid,
            "normalized_parameters": dict(
                self.normalized_parameters
            ),
            "resolved_parameters": dict(
                self.resolved_parameters
            ),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "human_approval_required": (
                self.human_approval_required
            ),
            "checked_at": self.checked_at.isoformat(),
            "physical_execution_authorized": False,
            "independent_safety_check_required": True,
        }


# ============================================================
# ACTION SPACE ENGINE
# ============================================================

class RoboticsActionSpaceEngine:

    def __init__(
        self,
        config: ActionSpaceConfig | None = None,
    ) -> None:
        self.config = config or ActionSpaceConfig()

        self._spaces: dict[
            tuple[str, str],
            RobotActionSpace,
        ] = {}

        self._masks: dict[
            tuple[str, str],
            ActionMask,
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
        space_id: str,
        action: str,
        actor_id: str = "system",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._audit.append(
            {
                "event_id": new_id(),
                "timestamp": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "space_id": space_id,
                "action": action,
                "actor_id": actor_id,
                "details": json_safe(
                    dict(details or {})
                ),
            }
        )

    # --------------------------------------------------------
    # REGISTER ACTION SPACE
    # --------------------------------------------------------

    def register_space(
        self,
        space: RobotActionSpace,
        *,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise ActionPermissionDenied(
                "Authorized action space registration required"
            )

        if not isinstance(
            space,
            RobotActionSpace,
        ):
            raise ActionValidationError(
                "Invalid action space"
            )

        validate_id(actor_id, "actor_id")

        if (
            len(space.actions)
            > self.config.maximum_actions_per_space
        ):
            raise ActionValidationError(
                "Too many actions in action space"
            )

        for action in space.actions:
            if (
                len(action.parameters)
                > self.config.maximum_parameters_per_action
            ):
                raise ActionValidationError(
                    "Too many action parameters"
                )

        key = (
            space.tenant_id,
            space.space_id,
        )

        with self._lock:
            if key in self._spaces:
                raise ActionSpaceConflict(
                    "Action space already exists"
                )

            if (
                len(self._spaces)
                >= self.config.maximum_action_spaces
            ):
                raise ActionSpaceConflict(
                    "Action space capacity reached"
                )

            self._spaces[key] = space

            self._log(
                tenant_id=space.tenant_id,
                space_id=space.space_id,
                action="action_space_registered",
                actor_id=actor_id,
                details={
                    "robot_id": space.robot_id,
                    "action_count": len(space.actions),
                },
            )

            return space.to_dict()

    # --------------------------------------------------------
    # UPDATE ACTION SPACE
    # --------------------------------------------------------

    def update_space(
        self,
        space: RobotActionSpace,
        *,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise ActionPermissionDenied(
                "Authorized action space update required"
            )

        if not isinstance(
            space,
            RobotActionSpace,
        ):
            raise ActionValidationError(
                "Invalid action space"
            )

        validate_id(actor_id, "actor_id")

        if (
            len(space.actions)
            > self.config.maximum_actions_per_space
        ):
            raise ActionValidationError(
                "Too many actions"
            )

        for action in space.actions:
            if (
                len(action.parameters)
                > self.config.maximum_parameters_per_action
            ):
                raise ActionValidationError(
                    "Too many action parameters"
                )

        key = (
            space.tenant_id,
            space.space_id,
        )

        with self._lock:
            existing = self._spaces.get(key)

            if existing is None:
                raise ActionSpaceNotFound(
                    "Action space not found"
                )

            if existing.robot_id != space.robot_id:
                raise ActionSpaceConflict(
                    "Robot identity cannot change"
                )

            if space.version != existing.version + 1:
                raise ActionSpaceConflict(
                    "Action space version must increment by one"
                )

            self._spaces[key] = space

            # Fail closed: previous masks may no longer
            # represent the updated action space.
            self._masks.pop(key, None)

            self._log(
                tenant_id=space.tenant_id,
                space_id=space.space_id,
                action="action_space_updated",
                actor_id=actor_id,
                details={
                    "old_version": existing.version,
                    "new_version": space.version,
                    "mask_invalidated": True,
                },
            )

            return space.to_dict()

    # --------------------------------------------------------
    # GET ACTION SPACE
    # --------------------------------------------------------

    def get_space(
        self,
        *,
        tenant_id: str,
        space_id: str,
    ) -> dict[str, Any]:
        validate_id(tenant_id, "tenant_id")
        validate_id(space_id, "space_id")

        with self._lock:
            space = self._spaces.get(
                (tenant_id, space_id)
            )

            if space is None:
                raise ActionSpaceNotFound(
                    "Action space not found"
                )

            return space.to_dict()

    # --------------------------------------------------------
    # LIST ACTION SPACES
    # --------------------------------------------------------

    def list_spaces(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        if robot_id is not None:
            validate_id(robot_id, "robot_id")

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

            return sorted(
                spaces,
                key=lambda item: item["space_id"],
            )

    # --------------------------------------------------------
    # GET ACTION
    # --------------------------------------------------------

    def get_action(
        self,
        *,
        tenant_id: str,
        space_id: str,
        action_id: str,
    ) -> dict[str, Any]:
        validate_id(tenant_id, "tenant_id")
        validate_id(space_id, "space_id")
        validate_id(action_id, "action_id")

        with self._lock:
            space = self._spaces.get(
                (tenant_id, space_id)
            )

            if space is None:
                raise ActionSpaceNotFound(
                    "Action space not found"
                )

            return space.get_action(
                action_id
            ).to_dict()

    # --------------------------------------------------------
    # SET ACTION MASK
    # --------------------------------------------------------

    def set_action_mask(
        self,
        mask: ActionMask,
        *,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise ActionPermissionDenied(
                "Authorized action mask update required"
            )

        if not isinstance(mask, ActionMask):
            raise ActionValidationError(
                "Invalid action mask"
            )

        validate_id(actor_id, "actor_id")

        key = (
            mask.tenant_id,
            mask.space_id,
        )

        with self._lock:
            space = self._spaces.get(key)

            if space is None:
                raise ActionSpaceNotFound(
                    "Action space not found"
                )

            if space.robot_id != mask.robot_id:
                raise ActionValidationError(
                    "Action mask robot mismatch"
                )

            registered_ids = {
                action.action_id
                for action in space.actions
            }

            unknown = (
                mask.allowed_action_ids
                - registered_ids
            )

            if unknown:
                raise ActionValidationError(
                    "Mask contains unknown action IDs"
                )

            self._masks[key] = mask

            self._log(
                tenant_id=mask.tenant_id,
                space_id=mask.space_id,
                action="action_mask_updated",
                actor_id=actor_id,
                details={
                    "allowed_actions": sorted(
                        mask.allowed_action_ids
                    ),
                },
            )

            return mask.to_dict()

    # --------------------------------------------------------
    # GET ACTION MASK
    # --------------------------------------------------------

    def get_action_mask(
        self,
        *,
        tenant_id: str,
        space_id: str,
    ) -> dict[str, Any] | None:
        validate_id(tenant_id, "tenant_id")
        validate_id(space_id, "space_id")

        with self._lock:
            mask = self._masks.get(
                (tenant_id, space_id)
            )

            return (
                mask.to_dict()
                if mask is not None
                else None
            )

    # --------------------------------------------------------
    # VALIDATE ACTION
    # --------------------------------------------------------

    def validate_action(
        self,
        request: ActionRequest,
    ) -> dict[str, Any]:
        if not isinstance(
            request,
            ActionRequest,
        ):
            raise ActionValidationError(
                "Invalid action request"
            )

        with self._lock:
            space = self._spaces.get(
                (
                    request.tenant_id,
                    request.space_id,
                )
            )

            if space is None:
                raise ActionSpaceNotFound(
                    "Action space not found"
                )

            if space.robot_id != request.robot_id:
                raise ActionValidationError(
                    "Action space robot mismatch"
                )

            action = space.get_action(
                request.action_id
            )

            mask = self._masks.get(
                (
                    request.tenant_id,
                    request.space_id,
                )
            )

            result = self._validate_against_snapshot(
                request,
                action,
                mask,
            )

            self._log(
                tenant_id=request.tenant_id,
                space_id=request.space_id,
                action="action_validated",
                details={
                    "request_id": request.request_id,
                    "action_id": request.action_id,
                    "status": result.status.value,
                },
            )

            return result.to_dict()

    # --------------------------------------------------------
    # INTERNAL VALIDATION
    # --------------------------------------------------------

    def _validate_against_snapshot(
        self,
        request: ActionRequest,
        action: RobotAction,
        mask: ActionMask | None,
    ) -> ActionValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        resolved: dict[str, Any] = {}
        normalized: dict[str, float] = {}

        masked = False

        if not action.enabled:
            errors.append(
                "Action is disabled"
            )

        if self.config.require_action_mask_check:
            if mask is None:
                errors.append(
                    "No action mask configured; "
                    "validation fails closed"
                )

                masked = True

            elif not mask.is_allowed(
                action.action_id
            ):
                errors.append(
                    "Action is blocked by the action mask"
                )

                masked = True

        missing_capabilities = (
            set(action.required_capabilities)
            - set(request.available_capabilities)
        )

        if missing_capabilities:
            errors.append(
                "Missing robot capabilities: "
                + ", ".join(
                    sorted(missing_capabilities)
                )
            )

        parameter_map = {
            parameter.name: parameter
            for parameter in action.parameters
        }

        unknown_parameters = (
            set(request.parameters)
            - set(parameter_map)
        )

        if unknown_parameters:
            errors.append(
                "Unknown parameters: "
                + ", ".join(
                    sorted(unknown_parameters)
                )
            )

        for parameter in action.parameters:
            if parameter.name in request.parameters:
                value = request.parameters[
                    parameter.name
                ]

            elif parameter.default is not None:
                value = parameter.default

            elif parameter.required:
                errors.append(
                    f"Missing required parameter: "
                    f"{parameter.name}"
                )

                continue

            else:
                continue

            try:
                parameter.validate_value(value)

                resolved[
                    parameter.name
                ] = value

                if (
                    parameter.parameter_type
                    in (
                        ActionParameterType.FLOAT,
                        ActionParameterType.INTEGER,
                    )
                    and parameter.minimum is not None
                    and parameter.maximum is not None
                ):
                    normalized[
                        parameter.name
                    ] = parameter.normalize(value)

            except ActionValidationError as exc:
                errors.append(str(exc))

        approval_required = (
            action.human_approval_required
            or action.safety_level
            in (
                ActionSafetyLevel.HIGH,
                ActionSafetyLevel.CRITICAL,
            )
        )

        if approval_required:
            warnings.append(
                "Independent human approval is required "
                "before any physical execution."
            )

        if request.approved_by is not None:
            warnings.append(
                "An approval reference was supplied, "
                "but this module does not authenticate "
                "or verify that approval."
            )

        warnings.append(
            "Action-space validation does not replace "
            "collision checking, interlocks, "
            "geofencing or hardware safety validation."
        )

        if errors:
            status = (
                ActionValidationStatus.MASKED
                if masked
                else ActionValidationStatus.INVALID
            )

        elif approval_required:
            status = (
                ActionValidationStatus.REQUIRES_APPROVAL
            )

        else:
            status = ActionValidationStatus.VALID

        return ActionValidationResult(
            validation_id=new_id(),
            request_id=request.request_id,
            tenant_id=request.tenant_id,
            robot_id=request.robot_id,
            space_id=request.space_id,
            action_id=request.action_id,
            status=status,
            valid=not errors,
            normalized_parameters=normalized,
            resolved_parameters=resolved,
            errors=tuple(errors),
            warnings=tuple(warnings),
            human_approval_required=approval_required,
        )

    # --------------------------------------------------------
    # VALIDATE ACTION SEQUENCE
    # --------------------------------------------------------

    def validate_sequence(
        self,
        requests: Sequence[ActionRequest],
    ) -> dict[str, Any]:
        if not isinstance(
            requests,
            (tuple, list),
        ):
            raise ActionValidationError(
                "requests must be a list or tuple"
            )

        if not requests:
            raise ActionValidationError(
                "Action sequence cannot be empty"
            )

        if (
            len(requests)
            > self.config.maximum_sequence_length
        ):
            raise ActionValidationError(
                "Action sequence exceeds maximum length"
            )

        results = [
            self.validate_action(request)
            for request in requests
        ]

        all_valid = all(
            item["valid"]
            for item in results
        )

        requires_approval = any(
            item["human_approval_required"]
            for item in results
        )

        return {
            "sequence_id": new_id(),
            "action_count": len(results),
            "all_valid": all_valid,
            "requires_human_approval": (
                requires_approval
            ),
            "results": results,
            "execution_authorized": False,
            "sequence_dynamics_validated": False,
            "collision_free": None,
            "advisory_only": True,
        }

    # --------------------------------------------------------
    # ACTION SPACE SUMMARY
    # --------------------------------------------------------

    def action_space_summary(
        self,
        *,
        tenant_id: str,
        space_id: str,
    ) -> dict[str, Any]:
        validate_id(tenant_id, "tenant_id")
        validate_id(space_id, "space_id")

        with self._lock:
            space = self._spaces.get(
                (tenant_id, space_id)
            )

            if space is None:
                raise ActionSpaceNotFound(
                    "Action space not found"
                )

            mask = self._masks.get(
                (tenant_id, space_id)
            )

            enabled = sum(
                1
                for action in space.actions
                if action.enabled
            )

            high_risk = sum(
                1
                for action in space.actions
                if action.safety_level
                in (
                    ActionSafetyLevel.HIGH,
                    ActionSafetyLevel.CRITICAL,
                )
            )

            return {
                "tenant_id": tenant_id,
                "robot_id": space.robot_id,
                "space_id": space_id,
                "space_type": space.space_type.value,
                "version": space.version,
                "total_actions": len(space.actions),
                "enabled_actions": enabled,
                "disabled_actions": (
                    len(space.actions) - enabled
                ),
                "high_risk_actions": high_risk,
                "mask_configured": mask is not None,
                "allowed_action_count": (
                    len(mask.allowed_action_ids)
                    if mask is not None
                    else 0
                ),
                "execution_authorized": False,
            }

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    def audit_history(
        self,
        *,
        tenant_id: str,
        space_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        if space_id is not None:
            validate_id(space_id, "space_id")

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
                if (
                    event["tenant_id"] == tenant_id
                    and (
                        space_id is None
                        or event["space_id"] == space_id
                    )
                )
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
            spaces = [
                space.to_dict()
                for (tenant, _), space
                in self._spaces.items()
                if tenant == tenant_id
            ]

            masks = [
                mask.to_dict()
                for (tenant, _), mask
                in self._masks.items()
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
            "action_spaces": spaces,
            "action_masks": masks,
            "audit_events": audit,
            "execution_authorized": False,
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
# STANDARD ROBOT ACTION SPACE FACTORY
# ============================================================

def create_mobile_robot_action_space(
    *,
    tenant_id: str,
    robot_id: str,
    space_id: str = "mobile_robot_actions",
    max_linear_speed: float = 1.0,
    max_angular_speed: float = 1.0,
) -> RobotActionSpace:
    max_linear_speed = finite_number(
        max_linear_speed,
        "max_linear_speed",
    )

    max_angular_speed = finite_number(
        max_angular_speed,
        "max_angular_speed",
    )

    if (
        max_linear_speed <= 0
        or max_angular_speed <= 0
    ):
        raise ActionValidationError(
            "Speed limits must be positive"
        )

    actions = (
        RobotAction(
            action_id="stop",
            name="Stop Motion Request",
            category=ActionCategory.NAVIGATION,
            parameters=(),
            safety_level=ActionSafetyLevel.MODERATE,
            required_capabilities=(
                "mobile_base",
            ),
            description=(
                "Request zero commanded velocity. "
                "Not a certified emergency stop."
            ),
        ),
        RobotAction(
            action_id="move_velocity",
            name="Move With Velocity",
            category=ActionCategory.LOCOMOTION,
            parameters=(
                ActionParameter(
                    name="linear_velocity",
                    parameter_type=ActionParameterType.FLOAT,
                    minimum=-max_linear_speed,
                    maximum=max_linear_speed,
                    unit="m/s",
                    description="Linear velocity command",
                ),
                ActionParameter(
                    name="angular_velocity",
                    parameter_type=ActionParameterType.FLOAT,
                    minimum=-max_angular_speed,
                    maximum=max_angular_speed,
                    unit="rad/s",
                    description="Angular velocity command",
                ),
                ActionParameter(
                    name="duration_seconds",
                    parameter_type=ActionParameterType.FLOAT,
                    minimum=0.01,
                    maximum=10.0,
                    unit="s",
                    description="Command duration",
                ),
            ),
            safety_level=ActionSafetyLevel.HIGH,
            required_capabilities=(
                "mobile_base",
                "velocity_control",
            ),
            human_approval_required=True,
            description=(
                "Bounded mobile robot velocity action."
            ),
        ),
        RobotAction(
            action_id="rotate",
            name="Rotate Robot",
            category=ActionCategory.NAVIGATION,
            parameters=(
                ActionParameter(
                    name="angle",
                    parameter_type=ActionParameterType.FLOAT,
                    minimum=-math.pi,
                    maximum=math.pi,
                    unit="rad",
                ),
                ActionParameter(
                    name="angular_speed",
                    parameter_type=ActionParameterType.FLOAT,
                    minimum=0.01,
                    maximum=max_angular_speed,
                    unit="rad/s",
                ),
            ),
            safety_level=ActionSafetyLevel.HIGH,
            required_capabilities=(
                "mobile_base",
                "rotation_control",
            ),
            human_approval_required=True,
        ),
    )

    return RobotActionSpace(
        space_id=space_id,
        tenant_id=tenant_id,
        robot_id=robot_id,
        name="Mobile Robot Action Space",
        space_type=ActionSpaceType.HYBRID,
        actions=actions,
        description=(
            "Discrete navigation commands with "
            "bounded continuous parameters."
        ),
    )


def create_manipulator_action_space(
    *,
    tenant_id: str,
    robot_id: str,
    joint_count: int = 6,
    space_id: str = "manipulator_actions",
    joint_position_limit: float = math.pi,
) -> RobotActionSpace:
    validate_integer(
        joint_count,
        "joint_count",
        minimum=1,
        maximum=128,
    )

    joint_position_limit = finite_number(
        joint_position_limit,
        "joint_position_limit",
    )

    if joint_position_limit <= 0:
        raise ActionValidationError(
            "joint_position_limit must be positive"
        )

    joint_parameters = tuple(
        ActionParameter(
            name=f"joint_{index + 1}",
            parameter_type=ActionParameterType.FLOAT,
            minimum=-joint_position_limit,
            maximum=joint_position_limit,
            unit="rad",
        )
        for index in range(joint_count)
    )

    actions = (
        RobotAction(
            action_id="joint_position",
            name="Joint Position Command",
            category=ActionCategory.JOINT_CONTROL,
            parameters=joint_parameters,
            safety_level=ActionSafetyLevel.HIGH,
            required_capabilities=(
                "manipulator",
                "joint_position_control",
            ),
            human_approval_required=True,
        ),
        RobotAction(
            action_id="gripper_position",
            name="Gripper Position Command",
            category=ActionCategory.GRIPPER_CONTROL,
            parameters=(
                ActionParameter(
                    name="opening",
                    parameter_type=ActionParameterType.FLOAT,
                    minimum=0.0,
                    maximum=1.0,
                    unit="normalized",
                ),
                ActionParameter(
                    name="force_limit",
                    parameter_type=ActionParameterType.FLOAT,
                    minimum=0.0,
                    maximum=100.0,
                    unit="N",
                ),
            ),
            safety_level=ActionSafetyLevel.HIGH,
            required_capabilities=(
                "gripper",
            ),
            human_approval_required=True,
        ),
    )

    return RobotActionSpace(
        space_id=space_id,
        tenant_id=tenant_id,
        robot_id=robot_id,
        name="Manipulator Action Space",
        space_type=ActionSpaceType.HYBRID,
        actions=actions,
        description=(
            "Joint position and gripper actions "
            "for a robotic manipulator."
        ),
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    engine = RoboticsActionSpaceEngine()

    space = create_mobile_robot_action_space(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        max_linear_speed=1.5,
        max_angular_speed=1.2,
    )

    engine.register_space(
        space,
        authorized=True,
        actor_id="robotics_admin",
    )

    mask = ActionMask(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        space_id="mobile_robot_actions",
        allowed_action_ids=frozenset(
            {
                "stop",
                "move_velocity",
                "rotate",
            }
        ),
        reason="Simulation action policy",
    )

    engine.set_action_mask(
        mask,
        authorized=True,
        actor_id="robotics_admin",
    )

    request = ActionRequest(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        space_id="mobile_robot_actions",
        action_id="move_velocity",
        parameters={
            "linear_velocity": 0.5,
            "angular_velocity": 0.2,
            "duration_seconds": 2.0,
        },
        available_capabilities=frozenset(
            {
                "mobile_base",
                "velocity_control",
                "rotation_control",
            }
        ),
    )

    result = engine.validate_action(request)

    print("\nACTION VALIDATION RESULT")

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    print("\nACTION SPACE SUMMARY")

    print(
        json.dumps(
            engine.action_space_summary(
                tenant_id="tenant_alpha",
                space_id="mobile_robot_actions",
            ),
            indent=2,
        )
    )

    output = engine.export_json(
        "data/robotics/action_space_report.json",
        tenant_id="tenant_alpha",
    )

    print("\nExported:", output)


if __name__ == "__main__":
    main()
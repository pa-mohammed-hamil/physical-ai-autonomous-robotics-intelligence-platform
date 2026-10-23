"""Goal Manager module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/planning/goal_manager.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================

Module: goal_manager.py

Hierarchical goal management for autonomous robotics,
multi-robot planning, behavior trees, task scheduling,
simulation, inspection, diagnostics, and maintenance.

Python 3.11+
Dependencies: Standard library only.

Features
--------
- Goal creation, validation, and registration
- Hierarchical goals and subgoals
- Goal dependency management
- Priority-based goal selection
- Goal lifecycle state machine
- Goal assignment to robots
- Robot capability and zone matching
- Human approval requirements
- Goal preconditions and postconditions
- Goal deadlines and timeouts
- Progress tracking
- Goal failure and cancellation
- Goal retry recommendations
- Goal decomposition
- Goal conflict detection
- Resource reservation recommendations
- Behavior tree integration payloads
- Action scheduler integration payloads
- Execution monitor integration
- Multi-tenant access control
- Audit logging
- Goal statistics and JSON export
- Unit tests

SAFETY
------
This module manages planning goals and simulated execution
states only.

It does not directly command motors, manipulators, navigation,
actuators, emergency-stop devices, or physical robots.

Hardware execution requires independent authorization,
fresh safety verification, command interlocks, and certified
safety controls appropriate to the robot and environment.

Goal completion is not proof of physical task completion.
"""

from __future__ import annotations

import json
import math
import threading
import unittest
import uuid

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class GoalManagerError(Exception):
    """Base goal management exception."""


class GoalValidationError(GoalManagerError):
    """Invalid goal configuration or input."""


class GoalAuthorizationError(GoalManagerError):
    """Unauthorized goal operation."""


class GoalNotFoundError(GoalManagerError):
    """Requested goal was not found."""


class GoalConflictError(GoalManagerError):
    """Conflicting goal or duplicate identifier."""


class GoalStateError(GoalManagerError):
    """Invalid goal lifecycle transition."""


class GoalSafetyError(GoalManagerError):
    """Goal safety requirements are not satisfied."""


class GoalCapacityError(GoalManagerError):
    """Goal manager capacity was exceeded."""


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


def validate_id(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise GoalValidationError(
            f"{field_name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > 128:
        raise GoalValidationError(
            f"{field_name} must contain 1–128 characters"
        )

    return value


def validate_timestamp(
    value: datetime,
    field_name: str,
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise GoalValidationError(
            f"{field_name} must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


def validate_number(
    value: Any,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise GoalValidationError(
            f"{field_name} must be finite"
        )

    number = float(value)

    if minimum is not None and number < minimum:
        raise GoalValidationError(
            f"{field_name} must be >= {minimum}"
        )

    if maximum is not None and number > maximum:
        raise GoalValidationError(
            f"{field_name} must be <= {maximum}"
        )

    return number


def to_json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return {
            name: to_json_safe(getattr(value, name))
            for name in value.__dataclass_fields__
            if name != "lock"
        }

    if isinstance(value, Mapping):
        return {
            str(key): to_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (tuple, list, set, frozenset, deque)):
        return [to_json_safe(item) for item in value]

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, (str, int, bool)) or value is None:
        return value

    return repr(value)


# ============================================================
# ENUMS
# ============================================================

class GoalStatus(str, Enum):
    CREATED = "created"
    VALIDATED = "validated"
    BLOCKED = "blocked"
    READY = "ready"
    ACTIVE = "active"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class GoalType(str, Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    INSPECTION = "inspection"
    DIAGNOSTICS = "diagnostics"
    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    CALIBRATION = "calibration"
    CHARGING = "charging"
    MONITORING = "monitoring"
    PERCEPTION = "perception"
    DATA_COLLECTION = "data_collection"
    SIMULATION = "simulation"
    SAFETY_CHECK = "safety_check"
    CUSTOM = "custom"


class GoalPriority(int, Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class GoalConditionOperator(str, Enum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    EXISTS = "exists"


class GoalConditionCategory(str, Enum):
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    SAFETY = "safety"


class GoalEventType(str, Enum):
    CREATED = "created"
    VALIDATED = "validated"
    READY = "ready"
    BLOCKED = "blocked"
    ACTIVATED = "activated"
    PAUSED = "paused"
    RESUMED = "resumed"
    PROGRESS_UPDATED = "progress_updated"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    ASSIGNED = "assigned"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_REVOKED = "approval_revoked"
    RETRY_PREPARED = "retry_prepared"
    CHILD_ADDED = "child_added"
    CONFLICT_DETECTED = "conflict_detected"


class GoalDecision(str, Enum):
    READY = "ready"
    WAIT_FOR_DEPENDENCIES = "wait_for_dependencies"
    WAIT_FOR_APPROVAL = "wait_for_approval"
    WAIT_FOR_ROBOT = "wait_for_robot"
    WAIT_FOR_CONDITIONS = "wait_for_conditions"
    SAFETY_BLOCKED = "safety_blocked"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    INVALID = "invalid"


class RobotGoalSafetyState(str, Enum):
    SAFE = "safe"
    CAUTION = "caution"
    UNSAFE = "unsafe"
    EMERGENCY_STOP = "emergency_stop"
    UNKNOWN = "unknown"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class GoalManagerConfig:
    maximum_goals: int = 10000
    maximum_goals_per_tenant: int = 2000
    maximum_children_per_goal: int = 100
    maximum_dependencies_per_goal: int = 100

    maximum_active_goals_per_robot: int = 1

    maximum_events_per_goal: int = 1000

    default_timeout_seconds: float = 1800.0

    maximum_retry_attempts: int = 3

    minimum_robot_battery_percentage: float = 10.0
    robot_state_max_age_seconds: float = 30.0

    require_fresh_robot_state: bool = True
    require_safe_robot_state: bool = True
    require_human_approval: bool = True

    allow_goal_preemption: bool = False

    def __post_init__(self) -> None:
        for name in (
            "maximum_goals",
            "maximum_goals_per_tenant",
            "maximum_children_per_goal",
            "maximum_dependencies_per_goal",
            "maximum_active_goals_per_robot",
            "maximum_events_per_goal",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise GoalValidationError(
                    f"{name} must be a positive integer"
                )

        if self.maximum_retry_attempts < 0:
            raise GoalValidationError(
                "maximum_retry_attempts cannot be negative"
            )

        validate_number(
            self.default_timeout_seconds,
            "default_timeout_seconds",
            minimum=0.001,
        )

        validate_number(
            self.minimum_robot_battery_percentage,
            "minimum_robot_battery_percentage",
            minimum=0,
            maximum=100,
        )

        validate_number(
            self.robot_state_max_age_seconds,
            "robot_state_max_age_seconds",
            minimum=0.001,
        )


# ============================================================
# ACCESS CONTEXT
# ============================================================

@dataclass(frozen=True)
class GoalAccessContext:
    tenant_id: str
    user_id: str

    allowed_robot_ids: frozenset[str] = frozenset()

    roles: frozenset[str] = frozenset()

    can_create: bool = False
    can_view: bool = False
    can_modify: bool = False
    can_assign: bool = False
    can_manage_approvals: bool = False
    can_update_progress: bool = False
    can_cancel: bool = False
    can_delete: bool = False

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.user_id, "user_id")


# ============================================================
# GOAL CONDITIONS
# ============================================================

@dataclass(frozen=True)
class GoalCondition:
    condition_id: str
    name: str

    field_name: str

    operator: GoalConditionOperator

    expected_value: Any = None

    category: GoalConditionCategory = (
        GoalConditionCategory.PRECONDITION
    )

    required: bool = True

    description: str = ""

    def __post_init__(self) -> None:
        validate_id(self.condition_id, "condition_id")
        validate_id(self.name, "condition name")
        validate_id(self.field_name, "field_name")


@dataclass(frozen=True)
class GoalConditionResult:
    condition_id: str
    name: str

    satisfied: bool

    actual_value: Any

    expected_value: Any

    reason: str


class GoalConditionEvaluator:
    """
    Evaluates declarative conditions against trusted,
    validated state snapshots.

    No eval(), exec(), or dynamic Python expressions.
    """

    @staticmethod
    def evaluate(
        condition: GoalCondition,
        state: Mapping[str, Any],
    ) -> GoalConditionResult:
        exists = condition.field_name in state

        actual = state.get(
            condition.field_name
        )

        expected = condition.expected_value

        operator = condition.operator

        satisfied = False

        try:
            if operator == GoalConditionOperator.EXISTS:
                satisfied = exists

            elif not exists:
                satisfied = False

            elif operator == GoalConditionOperator.EQ:
                satisfied = actual == expected

            elif operator == GoalConditionOperator.NE:
                satisfied = actual != expected

            elif operator == GoalConditionOperator.GT:
                satisfied = actual > expected

            elif operator == GoalConditionOperator.GTE:
                satisfied = actual >= expected

            elif operator == GoalConditionOperator.LT:
                satisfied = actual < expected

            elif operator == GoalConditionOperator.LTE:
                satisfied = actual <= expected

            elif operator == GoalConditionOperator.IN:
                satisfied = actual in expected

            elif operator == GoalConditionOperator.NOT_IN:
                satisfied = actual not in expected

        except (TypeError, ValueError):
            satisfied = False

        return GoalConditionResult(
            condition_id=condition.condition_id,
            name=condition.name,
            satisfied=bool(satisfied),
            actual_value=actual,
            expected_value=expected,
            reason=(
                "Condition satisfied"
                if satisfied
                else "Condition not satisfied"
            ),
        )

    @classmethod
    def evaluate_all(
        cls,
        conditions: Sequence[GoalCondition],
        state: Mapping[str, Any],
    ) -> tuple[GoalConditionResult, ...]:
        return tuple(
            cls.evaluate(condition, state)
            for condition in conditions
        )


# ============================================================
# ROBOT STATE
# ============================================================

@dataclass(frozen=True)
class GoalRobotState:
    tenant_id: str
    robot_id: str

    capabilities: frozenset[str]

    battery_percentage: float

    safety_state: RobotGoalSafetyState = (
        RobotGoalSafetyState.UNKNOWN
    )

    emergency_stop_active: bool = False

    active_faults: tuple[str, ...] = ()

    available: bool = False

    current_zone: str | None = None

    allowed_zones: frozenset[str] = frozenset()

    maximum_payload_kg: float = 0.0

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")

        validate_number(
            self.battery_percentage,
            "battery_percentage",
            minimum=0,
            maximum=100,
        )

        validate_number(
            self.maximum_payload_kg,
            "maximum_payload_kg",
            minimum=0,
        )

        validate_timestamp(
            self.timestamp,
            "timestamp",
        )


# ============================================================
# GOAL SPECIFICATION
# ============================================================

@dataclass(frozen=True)
class GoalSpecification:
    goal_id: str
    tenant_id: str

    name: str

    goal_type: GoalType

    description: str = ""

    priority: GoalPriority = GoalPriority.NORMAL

    robot_id: str | None = None

    parent_goal_id: str | None = None

    dependencies: tuple[str, ...] = ()

    required_capabilities: frozenset[str] = frozenset()

    required_zone: str | None = None

    required_payload_kg: float = 0.0

    requires_human_approval: bool = False

    approval_id: str | None = None

    preconditions: tuple[GoalCondition, ...] = ()

    postconditions: tuple[GoalCondition, ...] = ()

    safety_conditions: tuple[GoalCondition, ...] = ()

    timeout_seconds: float | None = None

    deadline: datetime | None = None

    estimated_duration_seconds: float | None = None

    estimated_energy_wh: float | None = None

    retry_limit: int = 0

    created_by: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_id(self.goal_id, "goal_id")
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.name, "name")

        if self.robot_id is not None:
            validate_id(self.robot_id, "robot_id")

        if self.parent_goal_id is not None:
            validate_id(
                self.parent_goal_id,
                "parent_goal_id",
            )

            if self.parent_goal_id == self.goal_id:
                raise GoalValidationError(
                    "Goal cannot be its own parent"
                )

        if self.goal_id in self.dependencies:
            raise GoalValidationError(
                "Goal cannot depend on itself"
            )

        if len(set(self.dependencies)) != len(
            self.dependencies
        ):
            raise GoalValidationError(
                "Duplicate dependencies"
            )

        validate_number(
            self.required_payload_kg,
            "required_payload_kg",
            minimum=0,
        )

        for name in (
            "timeout_seconds",
            "estimated_duration_seconds",
            "estimated_energy_wh",
        ):
            value = getattr(self, name)

            if value is not None:
                validate_number(
                    value,
                    name,
                    minimum=(
                        0
                        if name == "estimated_energy_wh"
                        else 0.001
                    ),
                )

        if self.retry_limit < 0:
            raise GoalValidationError(
                "retry_limit cannot be negative"
            )

        if (
            self.requires_human_approval
            and not self.approval_id
        ):
            raise GoalValidationError(
                "Approval ID required"
            )

        validate_timestamp(
            self.created_at,
            "created_at",
        )

        if self.deadline is not None:
            validate_timestamp(
                self.deadline,
                "deadline",
            )


# ============================================================
# GOAL EVENT
# ============================================================

@dataclass(frozen=True)
class GoalEvent:
    event_id: str

    goal_id: str
    tenant_id: str

    event_type: GoalEventType

    actor: str

    timestamp: datetime = field(
        default_factory=utc_now
    )

    details: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# GOAL READINESS RESULT
# ============================================================

@dataclass(frozen=True)
class GoalReadinessResult:
    goal_id: str

    ready: bool

    decision: GoalDecision

    reasons: tuple[str, ...]

    condition_results: tuple[
        GoalConditionResult, ...
    ] = ()

    checked_at: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# GOAL RUNTIME
# ============================================================

@dataclass
class GoalRuntime:
    specification: GoalSpecification

    status: GoalStatus = GoalStatus.CREATED

    assigned_robot_id: str | None = None

    progress_percentage: float = 0.0

    children: set[str] = field(
        default_factory=set
    )

    approvals: set[str] = field(
        default_factory=set
    )

    retry_count: int = 0

    started_at: datetime | None = None
    completed_at: datetime | None = None

    last_updated: datetime = field(
        default_factory=utc_now
    )

    failure_reason: str | None = None

    events: deque[GoalEvent] = field(
        default_factory=deque
    )

    lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )

    def snapshot(self) -> dict[str, Any]:
        return {
            "specification": to_json_safe(
                self.specification
            ),
            "status": self.status.value,
            "assigned_robot_id": (
                self.assigned_robot_id
            ),
            "progress_percentage": (
                self.progress_percentage
            ),
            "children": sorted(self.children),
            "retry_count": self.retry_count,
            "started_at": to_json_safe(
                self.started_at
            ),
            "completed_at": to_json_safe(
                self.completed_at
            ),
            "last_updated": to_json_safe(
                self.last_updated
            ),
            "failure_reason": self.failure_reason,
            "event_count": len(self.events),
            "simulation_only": True,
            "hardware_execution_authorized": False,
        }


# ============================================================
# ROBOT ELIGIBILITY
# ============================================================

class GoalRobotEligibilityEvaluator:

    def __init__(
        self,
        config: GoalManagerConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        goal: GoalSpecification,
        robot: GoalRobotState,
    ) -> tuple[bool, tuple[str, ...]]:
        reasons = []

        if goal.tenant_id != robot.tenant_id:
            reasons.append(
                "Tenant mismatch"
            )

        if not robot.available:
            reasons.append(
                "Robot unavailable"
            )

        if self.config.require_safe_robot_state:
            if robot.safety_state != RobotGoalSafetyState.SAFE:
                reasons.append(
                    "Robot safety state is not SAFE"
                )

        if robot.emergency_stop_active:
            reasons.append(
                "Emergency stop active"
            )

        if robot.active_faults:
            reasons.append(
                "Active robot faults"
            )

        if self.config.require_fresh_robot_state:
            age = (
                utc_now() - robot.timestamp
            ).total_seconds()

            if (
                age < 0
                or age
                > self.config.robot_state_max_age_seconds
            ):
                reasons.append(
                    "Robot state stale"
                )

        if (
            robot.battery_percentage
            < self.config.minimum_robot_battery_percentage
        ):
            reasons.append(
                "Insufficient battery percentage"
            )

        if not goal.required_capabilities.issubset(
            robot.capabilities
        ):
            reasons.append(
                "Required capabilities unavailable"
            )

        if (
            goal.required_zone is not None
            and goal.required_zone
            not in robot.allowed_zones
        ):
            reasons.append(
                "Robot not authorized for required zone"
            )

        if (
            goal.required_payload_kg
            > robot.maximum_payload_kg
        ):
            reasons.append(
                "Payload capacity exceeded"
            )

        return (
            not reasons,
            tuple(reasons),
        )


# ============================================================
# GOAL CONFLICT DETECTOR
# ============================================================

@dataclass(frozen=True)
class GoalConflict:
    first_goal_id: str
    second_goal_id: str

    robot_id: str

    reason: str

    severity: str = "warning"


class GoalConflictDetector:

    @staticmethod
    def detect(
        goals: Sequence[GoalRuntime],
    ) -> tuple[GoalConflict, ...]:
        conflicts = []

        active = [
            goal
            for goal in goals
            if goal.status == GoalStatus.ACTIVE
            and goal.assigned_robot_id is not None
        ]

        for index, first in enumerate(active):
            for second in active[index + 1:]:
                if (
                    first.assigned_robot_id
                    == second.assigned_robot_id
                ):
                    conflicts.append(
                        GoalConflict(
                            first_goal_id=(
                                first.specification.goal_id
                            ),
                            second_goal_id=(
                                second.specification.goal_id
                            ),
                            robot_id=(
                                first.assigned_robot_id
                            ),
                            reason=(
                                "Multiple active goals "
                                "share the same robot"
                            ),
                        )
                    )

        return tuple(conflicts)


# ============================================================
# GOAL DECOMPOSITION
# ============================================================

class GoalDecomposer:
    """
    Generates suggested subgoal templates.

    Templates are planning suggestions only and must be
    registered and validated separately.
    """

    TEMPLATES: dict[
        GoalType,
        tuple[tuple[str, GoalType], ...],
    ] = {
        GoalType.INSPECTION: (
            ("Verify inspection readiness", GoalType.SAFETY_CHECK),
            ("Collect inspection data", GoalType.DATA_COLLECTION),
            ("Analyze inspection results", GoalType.PERCEPTION),
        ),
        GoalType.MAINTENANCE: (
            ("Inspect equipment", GoalType.INSPECTION),
            ("Run diagnostics", GoalType.DIAGNOSTICS),
            ("Review maintenance findings", GoalType.MONITORING),
        ),
        GoalType.NAVIGATION: (
            ("Verify navigation readiness", GoalType.SAFETY_CHECK),
            ("Plan simulated route", GoalType.SIMULATION),
            ("Review navigation outcome", GoalType.MONITORING),
        ),
        GoalType.REPAIR: (
            ("Inspect component", GoalType.INSPECTION),
            ("Diagnose fault", GoalType.DIAGNOSTICS),
            ("Review repair recommendation", GoalType.MONITORING),
        ),
        GoalType.CALIBRATION: (
            ("Verify calibration readiness", GoalType.SAFETY_CHECK),
            ("Collect calibration measurements", GoalType.DATA_COLLECTION),
            ("Evaluate calibration quality", GoalType.DIAGNOSTICS),
        ),
    }

    @classmethod
    def suggest(
        cls,
        parent: GoalSpecification,
    ) -> tuple[GoalSpecification, ...]:
        templates = cls.TEMPLATES.get(
            parent.goal_type,
            (),
        )

        results = []

        previous_id = None

        for name, goal_type in templates:
            goal_id = new_id()

            dependencies = (
                (previous_id,)
                if previous_id is not None
                else ()
            )

            results.append(
                GoalSpecification(
                    goal_id=goal_id,
                    tenant_id=parent.tenant_id,
                    name=name,
                    goal_type=goal_type,
                    priority=parent.priority,
                    robot_id=parent.robot_id,
                    parent_goal_id=parent.goal_id,
                    dependencies=dependencies,
                    required_zone=parent.required_zone,
                    created_by=parent.created_by,
                    metadata={
                        "generated_template": True,
                        "parent_goal_type": (
                            parent.goal_type.value
                        ),
                    },
                )
            )

            previous_id = goal_id

        return tuple(results)


# ============================================================
# GOAL MANAGER
# ============================================================

class RoboticsGoalManager:
    """
    Thread-safe hierarchical goal management engine.

    Access contexts must be produced by trusted
    authentication and authorization middleware.

    This manager does not execute hardware commands.
    """

    TERMINAL_STATES = frozenset({
        GoalStatus.SUCCEEDED,
        GoalStatus.FAILED,
        GoalStatus.CANCELLED,
        GoalStatus.EXPIRED,
    })

    def __init__(
        self,
        config: GoalManagerConfig | None = None,
    ) -> None:
        self.config = (
            config
            or GoalManagerConfig()
        )

        self._goals: dict[
            str, GoalRuntime
        ] = {}

        self._robot_states: dict[
            tuple[str, str], GoalRobotState
        ] = {}

        self._lock = threading.RLock()

        self._eligibility = GoalRobotEligibilityEvaluator(
            self.config
        )

        self._metrics = Counter()

    # --------------------------------------------------------
    # ACCESS CONTROL
    # --------------------------------------------------------

    @staticmethod
    def _require(
        access: GoalAccessContext,
        permission: str,
    ) -> None:
        if not getattr(access, permission, False):
            raise GoalAuthorizationError(
                f"Permission required: {permission}"
            )

    @staticmethod
    def _authorize_robot(
        access: GoalAccessContext,
        robot_id: str,
    ) -> None:
        if robot_id not in access.allowed_robot_ids:
            raise GoalAuthorizationError(
                "Robot access denied"
            )

    def _get(
        self,
        goal_id: str,
        access: GoalAccessContext,
    ) -> GoalRuntime:
        with self._lock:
            runtime = self._goals.get(
                goal_id
            )

        if runtime is None:
            raise GoalNotFoundError(
                f"Goal not found: {goal_id}"
            )

        if (
            runtime.specification.tenant_id
            != access.tenant_id
        ):
            raise GoalAuthorizationError(
                "Cross-tenant goal access denied"
            )

        if runtime.assigned_robot_id is not None:
            self._authorize_robot(
                access,
                runtime.assigned_robot_id,
            )

        return runtime

    # --------------------------------------------------------
    # EVENTS
    # --------------------------------------------------------

    def _record_event(
        self,
        runtime: GoalRuntime,
        event_type: GoalEventType,
        *,
        actor: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        runtime.events.append(
            GoalEvent(
                event_id=new_id(),
                goal_id=runtime.specification.goal_id,
                tenant_id=runtime.specification.tenant_id,
                event_type=event_type,
                actor=actor,
                details=to_json_safe(
                    details or {}
                ),
            )
        )

        runtime.last_updated = utc_now()

    # --------------------------------------------------------
    # REGISTER GOAL
    # --------------------------------------------------------

    def register_goal(
        self,
        specification: GoalSpecification,
        *,
        access: GoalAccessContext,
    ) -> dict[str, Any]:
        self._require(
            access,
            "can_create",
        )

        if specification.tenant_id != access.tenant_id:
            raise GoalAuthorizationError(
                "Tenant mismatch"
            )

        if specification.robot_id is not None:
            self._authorize_robot(
                access,
                specification.robot_id,
            )

        if (
            len(specification.dependencies)
            > self.config.maximum_dependencies_per_goal
        ):
            raise GoalCapacityError(
                "Too many goal dependencies"
            )

        with self._lock:
            if specification.goal_id in self._goals:
                raise GoalConflictError(
                    "Goal ID already exists"
                )

            if len(self._goals) >= self.config.maximum_goals:
                raise GoalCapacityError(
                    "Global goal capacity exceeded"
                )

            tenant_count = sum(
                runtime.specification.tenant_id
                == specification.tenant_id
                for runtime in self._goals.values()
            )

            if (
                tenant_count
                >= self.config.maximum_goals_per_tenant
            ):
                raise GoalCapacityError(
                    "Tenant goal capacity exceeded"
                )

            references = list(
                specification.dependencies
            )

            if specification.parent_goal_id is not None:
                references.append(
                    specification.parent_goal_id
                )

            for reference_id in references:
                referenced = self._goals.get(
                    reference_id
                )

                if referenced is None:
                    raise GoalValidationError(
                        f"Referenced goal not found: "
                        f"{reference_id}"
                    )

                if (
                    referenced.specification.tenant_id
                    != specification.tenant_id
                ):
                    raise GoalAuthorizationError(
                        "Cross-tenant goal reference denied"
                    )

                if referenced.assigned_robot_id is not None:
                    self._authorize_robot(
                        access,
                        referenced.assigned_robot_id,
                    )

            if specification.parent_goal_id is not None:
                parent = self._goals[
                    specification.parent_goal_id
                ]

                if (
                    len(parent.children)
                    >= self.config.maximum_children_per_goal
                ):
                    raise GoalCapacityError(
                        "Parent has too many children"
                    )

            runtime = GoalRuntime(
                specification=specification,
                assigned_robot_id=specification.robot_id,
                events=deque(
                    maxlen=self.config.maximum_events_per_goal
                ),
            )

            self._goals[
                specification.goal_id
            ] = runtime

            if specification.parent_goal_id is not None:
                parent = self._goals[
                    specification.parent_goal_id
                ]

                parent.children.add(
                    specification.goal_id
                )

                self._record_event(
                    parent,
                    GoalEventType.CHILD_ADDED,
                    actor=access.user_id,
                    details={
                        "child_goal_id": (
                            specification.goal_id
                        ),
                    },
                )

            self._record_event(
                runtime,
                GoalEventType.CREATED,
                actor=access.user_id,
            )

            self._metrics["created"] += 1

            return runtime.snapshot()

    # --------------------------------------------------------
    # REGISTER ROBOT STATE
    # --------------------------------------------------------

    def update_robot_state(
        self,
        robot_state: GoalRobotState,
        *,
        access: GoalAccessContext,
    ) -> None:
        self._require(
            access,
            "can_modify",
        )

        if robot_state.tenant_id != access.tenant_id:
            raise GoalAuthorizationError(
                "Robot tenant mismatch"
            )

        self._authorize_robot(
            access,
            robot_state.robot_id,
        )

        key = (
            robot_state.tenant_id,
            robot_state.robot_id,
        )

        with self._lock:
            previous = self._robot_states.get(
                key
            )

            if (
                previous is not None
                and robot_state.timestamp <= previous.timestamp
            ):
                raise GoalValidationError(
                    "Robot state timestamp must increase"
                )

            self._robot_states[key] = robot_state

    # --------------------------------------------------------
    # ASSIGN GOAL
    # --------------------------------------------------------

    def assign_goal(
        self,
        goal_id: str,
        robot_id: str,
        *,
        access: GoalAccessContext,
    ) -> dict[str, Any]:
        self._require(
            access,
            "can_assign",
        )

        self._authorize_robot(
            access,
            robot_id,
        )

        runtime = self._get(
            goal_id,
            access,
        )

        with self._lock, runtime.lock:
            if runtime.status in self.TERMINAL_STATES:
                raise GoalStateError(
                    "Cannot assign terminal goal"
                )

            if runtime.status == GoalStatus.ACTIVE:
                raise GoalStateError(
                    "Cannot reassign active goal"
                )

            robot_state = self._robot_states.get(
                (access.tenant_id, robot_id)
            )

            if robot_state is None:
                raise GoalSafetyError(
                    "Robot state unavailable"
                )

            eligible, reasons = self._eligibility.evaluate(
                runtime.specification,
                robot_state,
            )

            if not eligible:
                raise GoalSafetyError(
                    "; ".join(reasons)
                )

            runtime.assigned_robot_id = robot_id

            self._record_event(
                runtime,
                GoalEventType.ASSIGNED,
                actor=access.user_id,
                details={
                    "robot_id": robot_id,
                },
            )

            return runtime.snapshot()

    # --------------------------------------------------------
    # APPROVAL MANAGEMENT
    # --------------------------------------------------------

    def grant_approval(
        self,
        goal_id: str,
        approval_id: str,
        *,
        access: GoalAccessContext,
    ) -> None:
        self._require(
            access,
            "can_manage_approvals",
        )

        runtime = self._get(
            goal_id,
            access,
        )

        with runtime.lock:
            if (
                not runtime.specification.requires_human_approval
                or approval_id
                != runtime.specification.approval_id
            ):
                raise GoalValidationError(
                    "Approval ID does not match goal"
                )

            runtime.approvals.add(
                approval_id
            )

            self._record_event(
                runtime,
                GoalEventType.APPROVAL_GRANTED,
                actor=access.user_id,
                details={
                    "approval_id": approval_id,
                    "hardware_authorization": False,
                },
            )

    def revoke_approval(
        self,
        goal_id: str,
        approval_id: str,
        *,
        access: GoalAccessContext,
    ) -> None:
        self._require(
            access,
            "can_manage_approvals",
        )

        runtime = self._get(
            goal_id,
            access,
        )

        with runtime.lock:
            runtime.approvals.discard(
                approval_id
            )

            if runtime.status == GoalStatus.ACTIVE:
                runtime.status = GoalStatus.BLOCKED
                runtime.failure_reason = (
                    "Required approval revoked"
                )

            self._record_event(
                runtime,
                GoalEventType.APPROVAL_REVOKED,
                actor=access.user_id,
                details={
                    "approval_id": approval_id,
                },
            )

    # --------------------------------------------------------
    # READINESS EVALUATION
    # --------------------------------------------------------

    def evaluate_readiness(
        self,
        goal_id: str,
        *,
        access: GoalAccessContext,
        world_state: Mapping[str, Any] | None = None,
    ) -> GoalReadinessResult:
        self._require(
            access,
            "can_view",
        )

        runtime = self._get(
            goal_id,
            access,
        )

        specification = runtime.specification

        reasons = []
        condition_results = []

        now = utc_now()

        if (
            specification.deadline is not None
            and now >= specification.deadline
        ):
            return GoalReadinessResult(
                goal_id=goal_id,
                ready=False,
                decision=GoalDecision.DEADLINE_EXCEEDED,
                reasons=("Goal deadline exceeded",),
            )

        with self._lock:
            for dependency_id in specification.dependencies:
                dependency = self._goals.get(
                    dependency_id
                )

                if (
                    dependency is None
                    or dependency.status
                    != GoalStatus.SUCCEEDED
                ):
                    reasons.append(
                        f"Dependency incomplete: {dependency_id}"
                    )

            if reasons:
                return GoalReadinessResult(
                    goal_id=goal_id,
                    ready=False,
                    decision=GoalDecision.WAIT_FOR_DEPENDENCIES,
                    reasons=tuple(reasons),
                )

            for child_id in runtime.children:
                child = self._goals.get(
                    child_id
                )

                if (
                    child is None
                    or child.status != GoalStatus.SUCCEEDED
                ):
                    reasons.append(
                        f"Child goal incomplete: {child_id}"
                    )

            if reasons:
                return GoalReadinessResult(
                    goal_id=goal_id,
                    ready=False,
                    decision=GoalDecision.WAIT_FOR_DEPENDENCIES,
                    reasons=tuple(reasons),
                )

            if (
                specification.requires_human_approval
                and specification.approval_id
                not in runtime.approvals
            ):
                return GoalReadinessResult(
                    goal_id=goal_id,
                    ready=False,
                    decision=GoalDecision.WAIT_FOR_APPROVAL,
                    reasons=("Human approval required",),
                )

            robot_id = runtime.assigned_robot_id

            if robot_id is None:
                return GoalReadinessResult(
                    goal_id=goal_id,
                    ready=False,
                    decision=GoalDecision.WAIT_FOR_ROBOT,
                    reasons=("No robot assigned",),
                )

            robot_state = self._robot_states.get(
                (access.tenant_id, robot_id)
            )

        if robot_state is None:
            return GoalReadinessResult(
                goal_id=goal_id,
                ready=False,
                decision=GoalDecision.WAIT_FOR_ROBOT,
                reasons=("Robot state unavailable",),
            )

        eligible, robot_reasons = self._eligibility.evaluate(
            specification,
            robot_state,
        )

        if not eligible:
            return GoalReadinessResult(
                goal_id=goal_id,
                ready=False,
                decision=GoalDecision.SAFETY_BLOCKED,
                reasons=robot_reasons,
            )

        state = world_state or {}

        condition_results.extend(
            GoalConditionEvaluator.evaluate_all(
                specification.preconditions,
                state,
            )
        )

        condition_results.extend(
            GoalConditionEvaluator.evaluate_all(
                specification.safety_conditions,
                state,
            )
        )

        failed_conditions = [
            result
            for result in condition_results
            if not result.satisfied
        ]

        if failed_conditions:
            return GoalReadinessResult(
                goal_id=goal_id,
                ready=False,
                decision=GoalDecision.WAIT_FOR_CONDITIONS,
                reasons=tuple(
                    f"Condition not satisfied: {result.name}"
                    for result in failed_conditions
                ),
                condition_results=tuple(
                    condition_results
                ),
            )

        return GoalReadinessResult(
            goal_id=goal_id,
            ready=True,
            decision=GoalDecision.READY,
            reasons=(),
            condition_results=tuple(
                condition_results
            ),
        )

    # --------------------------------------------------------
    # ACTIVATE GOAL
    # --------------------------------------------------------

    def activate_goal(
        self,
        goal_id: str,
        *,
        access: GoalAccessContext,
        world_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._require(
            access,
            "can_modify",
        )

        runtime = self._get(
            goal_id,
            access,
        )

        with self._lock, runtime.lock:
            if runtime.status not in (
                GoalStatus.CREATED,
                GoalStatus.VALIDATED,
                GoalStatus.BLOCKED,
                GoalStatus.READY,
            ):
                raise GoalStateError(
                    "Goal cannot be activated"
                )

            readiness = self.evaluate_readiness(
                goal_id,
                access=access,
                world_state=world_state,
            )

            if not readiness.ready:
                runtime.status = GoalStatus.BLOCKED

                self._record_event(
                    runtime,
                    GoalEventType.BLOCKED,
                    actor=access.user_id,
                    details={
                        "decision": readiness.decision.value,
                        "reasons": readiness.reasons,
                    },
                )

                raise GoalSafetyError(
                    "; ".join(readiness.reasons)
                )

            robot_id = runtime.assigned_robot_id

            active_count = sum(
                goal.status == GoalStatus.ACTIVE
                and goal.assigned_robot_id == robot_id
                and goal.specification.tenant_id
                == access.tenant_id
                for goal in self._goals.values()
            )

            if (
                active_count
                >= self.config.maximum_active_goals_per_robot
            ):
                raise GoalConflictError(
                    "Robot active-goal capacity exceeded"
                )

            runtime.status = GoalStatus.ACTIVE

            if runtime.started_at is None:
                runtime.started_at = utc_now()

            self._record_event(
                runtime,
                GoalEventType.ACTIVATED,
                actor=access.user_id,
                details={
                    "robot_id": robot_id,
                    "simulation_only": True,
                },
            )

            self._metrics["activated"] += 1

            return runtime.snapshot()

    # --------------------------------------------------------
    # UPDATE PROGRESS
    # --------------------------------------------------------

    def update_progress(
        self,
        goal_id: str,
        progress_percentage: float,
        *,
        access: GoalAccessContext,
    ) -> dict[str, Any]:
        self._require(
            access,
            "can_update_progress",
        )

        progress = validate_number(
            progress_percentage,
            "progress_percentage",
            minimum=0,
            maximum=100,
        )

        runtime = self._get(
            goal_id,
            access,
        )

        with runtime.lock:
            if runtime.status != GoalStatus.ACTIVE:
                raise GoalStateError(
                    "Only ACTIVE goals can update progress"
                )

            if progress < runtime.progress_percentage:
                raise GoalValidationError(
                    "Goal progress cannot decrease"
                )

            runtime.progress_percentage = progress

            self._record_event(
                runtime,
                GoalEventType.PROGRESS_UPDATED,
                actor=access.user_id,
                details={
                    "progress_percentage": progress,
                },
            )

            return runtime.snapshot()

    # --------------------------------------------------------
    # COMPLETE GOAL
    # --------------------------------------------------------

    def complete_goal(
        self,
        goal_id: str,
        *,
        access: GoalAccessContext,
        world_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._require(
            access,
            "can_modify",
        )

        runtime = self._get(
            goal_id,
            access,
        )

        with runtime.lock:
            if runtime.status != GoalStatus.ACTIVE:
                raise GoalStateError(
                    "Goal must be ACTIVE"
                )

            results = GoalConditionEvaluator.evaluate_all(
                runtime.specification.postconditions,
                world_state,
            )

            if any(
                not result.satisfied
                for result in results
            ):
                raise GoalStateError(
                    "Goal postconditions not satisfied"
                )

            runtime.status = GoalStatus.SUCCEEDED
            runtime.progress_percentage = 100.0
            runtime.completed_at = utc_now()

            self._record_event(
                runtime,
                GoalEventType.SUCCEEDED,
                actor=access.user_id,
                details={
                    "postconditions": to_json_safe(results),
                    "simulation_only": True,
                },
            )

            self._metrics["succeeded"] += 1

            return runtime.snapshot()

    # --------------------------------------------------------
    # FAIL GOAL
    # --------------------------------------------------------

    def fail_goal(
        self,
        goal_id: str,
        *,
        reason: str,
        access: GoalAccessContext,
    ) -> dict[str, Any]:
        self._require(
            access,
            "can_modify",
        )

        validate_id(reason, "reason")

        runtime = self._get(
            goal_id,
            access,
        )

        with runtime.lock:
            if runtime.status in self.TERMINAL_STATES:
                raise GoalStateError(
                    "Goal already terminal"
                )

            runtime.status = GoalStatus.FAILED
            runtime.failure_reason = reason
            runtime.completed_at = utc_now()

            self._record_event(
                runtime,
                GoalEventType.FAILED,
                actor=access.user_id,
                details={
                    "reason": reason,
                },
            )

            self._metrics["failed"] += 1

            return runtime.snapshot()

    # --------------------------------------------------------
    # PAUSE GOAL
    # --------------------------------------------------------

    def pause_goal(
        self,
        goal_id: str,
        *,
        access: GoalAccessContext,
    ) -> dict[str, Any]:
        self._require(
            access,
            "can_modify",
        )

        runtime = self._get(
            goal_id,
            access,
        )

        with runtime.lock:
            if runtime.status != GoalStatus.ACTIVE:
                raise GoalStateError(
                    "Goal must be ACTIVE"
                )

            runtime.status = GoalStatus.PAUSED

            self._record_event(
                runtime,
                GoalEventType.PAUSED,
                actor=access.user_id,
            )

            return runtime.snapshot()

    # --------------------------------------------------------
    # RESUME GOAL
    # --------------------------------------------------------

    def resume_goal(
        self,
        goal_id: str,
        *,
        access: GoalAccessContext,
        world_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._require(
            access,
            "can_modify",
        )

        runtime = self._get(
            goal_id,
            access,
        )

        with self._lock, runtime.lock:
            if runtime.status != GoalStatus.PAUSED:
                raise GoalStateError(
                    "Goal must be PAUSED"
                )

            readiness = self.evaluate_readiness(
                goal_id,
                access=access,
                world_state=world_state,
            )

            if not readiness.ready:
                raise GoalSafetyError(
                    "; ".join(readiness.reasons)
                )

            active_count = sum(
                goal.status == GoalStatus.ACTIVE
                and goal.assigned_robot_id
                == runtime.assigned_robot_id
                and goal.specification.tenant_id
                == access.tenant_id
                for goal in self._goals.values()
            )

            if (
                active_count
                >= self.config.maximum_active_goals_per_robot
            ):
                raise GoalConflictError(
                    "Robot already has active goals"
                )

            runtime.status = GoalStatus.ACTIVE

            self._record_event(
                runtime,
                GoalEventType.RESUMED,
                actor=access.user_id,
            )

            return runtime.snapshot()

    # --------------------------------------------------------
    # CANCEL GOAL
    # --------------------------------------------------------

    def cancel_goal(
        self,
        goal_id: str,
        *,
        access: GoalAccessContext,
        reason: str = "Cancelled by operator",
    ) -> dict[str, Any]:
        self._require(
            access,
            "can_cancel",
        )

        runtime = self._get(
            goal_id,
            access,
        )

        with runtime.lock:
            if runtime.status in self.TERMINAL_STATES:
                return runtime.snapshot()

            runtime.status = GoalStatus.CANCELLED
            runtime.completed_at = utc_now()
            runtime.failure_reason = reason

            self._record_event(
                runtime,
                GoalEventType.CANCELLED,
                actor=access.user_id,
                details={
                    "reason": reason,
                },
            )

            self._metrics["cancelled"] += 1

            return runtime.snapshot()

    # --------------------------------------------------------
    # RETRY GOAL
    # --------------------------------------------------------

    def prepare_retry(
        self,
        goal_id: str,
        *,
        access: GoalAccessContext,
    ) -> dict[str, Any]:
        self._require(
            access,
            "can_modify",
        )

        runtime = self._get(
            goal_id,
            access,
        )

        with runtime.lock:
            if runtime.status != GoalStatus.FAILED:
                raise GoalStateError(
                    "Only FAILED goals can retry"
                )

            maximum_retries = min(
                runtime.specification.retry_limit,
                self.config.maximum_retry_attempts,
            )

            if runtime.retry_count >= maximum_retries:
                raise GoalStateError(
                    "Goal retry limit exceeded"
                )

            runtime.retry_count += 1

            runtime.status = GoalStatus.CREATED
            runtime.progress_percentage = 0.0

            runtime.started_at = None
            runtime.completed_at = None
            runtime.failure_reason = None

            self._record_event(
                runtime,
                GoalEventType.RETRY_PREPARED,
                actor=access.user_id,
                details={
                    "retry_count": runtime.retry_count,
                },
            )

            return runtime.snapshot()

    # --------------------------------------------------------
    # GOAL TIMEOUT MONITOR
    # --------------------------------------------------------

    def monitor_timeouts(
        self,
        *,
        access: GoalAccessContext,
    ) -> list[dict[str, Any]]:
        self._require(
            access,
            "can_modify",
        )

        now = utc_now()

        expired = []

        with self._lock:
            runtimes = list(
                self._goals.values()
            )

        for runtime in runtimes:
            with runtime.lock:
                specification = runtime.specification

                if specification.tenant_id != access.tenant_id:
                    continue

                if runtime.status in self.TERMINAL_STATES:
                    continue

                if (
                    runtime.assigned_robot_id is not None
                    and runtime.assigned_robot_id
                    not in access.allowed_robot_ids
                ):
                    continue

                deadline_exceeded = (
                    specification.deadline is not None
                    and now >= specification.deadline
                )

                timeout = (
                    specification.timeout_seconds
                    or self.config.default_timeout_seconds
                )

                duration_exceeded = (
                    runtime.started_at is not None
                    and runtime.status == GoalStatus.ACTIVE
                    and (
                        now - runtime.started_at
                    ).total_seconds() >= timeout
                )

                if deadline_exceeded or duration_exceeded:
                    runtime.status = GoalStatus.EXPIRED
                    runtime.completed_at = now

                    runtime.failure_reason = (
                        "Goal deadline exceeded"
                        if deadline_exceeded
                        else "Goal execution timeout"
                    )

                    self._record_event(
                        runtime,
                        GoalEventType.EXPIRED,
                        actor=access.user_id,
                        details={
                            "reason": runtime.failure_reason,
                        },
                    )

                    expired.append(
                        runtime.snapshot()
                    )

                    self._metrics["expired"] += 1

        return expired

    # --------------------------------------------------------
    # READY GOAL SELECTION
    # --------------------------------------------------------

    def select_ready_goals(
        self,
        *,
        access: GoalAccessContext,
        world_state: Mapping[str, Any] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        self._require(
            access,
            "can_view",
        )

        if limit < 1:
            raise GoalValidationError(
                "Limit must be positive"
            )

        with self._lock:
            goals = [
                runtime
                for runtime in self._goals.values()
                if (
                    runtime.specification.tenant_id
                    == access.tenant_id
                    and runtime.status in (
                        GoalStatus.CREATED,
                        GoalStatus.VALIDATED,
                        GoalStatus.BLOCKED,
                        GoalStatus.READY,
                    )
                    and (
                        runtime.assigned_robot_id is None
                        or runtime.assigned_robot_id
                        in access.allowed_robot_ids
                    )
                )
            ]

        goals.sort(
            key=lambda runtime: (
                -runtime.specification.priority.value,
                runtime.specification.deadline
                or datetime.max.replace(
                    tzinfo=timezone.utc
                ),
                runtime.specification.created_at,
            )
        )

        ready = []

        for runtime in goals:
            readiness = self.evaluate_readiness(
                runtime.specification.goal_id,
                access=access,
                world_state=world_state,
            )

            if readiness.ready:
                ready.append(
                    runtime.snapshot()
                )

            if len(ready) >= limit:
                break

        return ready

    # --------------------------------------------------------
    # GOAL DECOMPOSITION
    # --------------------------------------------------------

    def suggest_subgoals(
        self,
        goal_id: str,
        *,
        access: GoalAccessContext,
    ) -> list[dict[str, Any]]:
        self._require(
            access,
            "can_view",
        )

        runtime = self._get(
            goal_id,
            access,
        )

        suggestions = GoalDecomposer.suggest(
            runtime.specification
        )

        return to_json_safe(
            suggestions
        )

    # --------------------------------------------------------
    # GOAL CONFLICTS
    # --------------------------------------------------------

    def detect_conflicts(
        self,
        *,
        access: GoalAccessContext,
    ) -> list[dict[str, Any]]:
        self._require(
            access,
            "can_view",
        )

        with self._lock:
            goals = [
                runtime
                for runtime in self._goals.values()
                if (
                    runtime.specification.tenant_id
                    == access.tenant_id
                    and (
                        runtime.assigned_robot_id is None
                        or runtime.assigned_robot_id
                        in access.allowed_robot_ids
                    )
                )
            ]

        return to_json_safe(
            GoalConflictDetector.detect(goals)
        )

    # --------------------------------------------------------
    # BEHAVIOR TREE ADAPTER
    # --------------------------------------------------------

    def behavior_tree_payload(
        self,
        goal_id: str,
        *,
        access: GoalAccessContext,
    ) -> dict[str, Any]:
        self._require(
            access,
            "can_view",
        )

        runtime = self._get(
            goal_id,
            access,
        )

        specification = runtime.specification

        return {
            "goal_id": goal_id,
            "tenant_id": specification.tenant_id,
            "robot_id": runtime.assigned_robot_id,
            "goal_type": specification.goal_type.value,
            "priority": specification.priority.value,
            "status": runtime.status.value,
            "preconditions": to_json_safe(
                specification.preconditions
            ),
            "postconditions": to_json_safe(
                specification.postconditions
            ),
            "safety_conditions": to_json_safe(
                specification.safety_conditions
            ),
            "dependencies": list(
                specification.dependencies
            ),
            "simulation_only": True,
            "hardware_execution_authorized": False,
        }

    # --------------------------------------------------------
    # ACTION SCHEDULER ADAPTER
    # --------------------------------------------------------

    def action_scheduler_payload(
        self,
        goal_id: str,
        *,
        access: GoalAccessContext,
    ) -> dict[str, Any]:
        self._require(
            access,
            "can_view",
        )

        runtime = self._get(
            goal_id,
            access,
        )

        specification = runtime.specification

        return {
            "goal_id": goal_id,
            "tenant_id": specification.tenant_id,
            "robot_id": runtime.assigned_robot_id,
            "priority": specification.priority.value,
            "estimated_duration_seconds": (
                specification.estimated_duration_seconds
            ),
            "estimated_energy_wh": (
                specification.estimated_energy_wh
            ),
            "deadline": to_json_safe(
                specification.deadline
            ),
            "dependencies": list(
                specification.dependencies
            ),
            "required_capabilities": sorted(
                specification.required_capabilities
            ),
            "required_zone": specification.required_zone,
            "requires_human_approval": (
                specification.requires_human_approval
            ),
            "simulation_only": True,
            "hardware_execution_authorized": False,
        }

    # --------------------------------------------------------
    # EXECUTION MONITOR ADAPTER
    # --------------------------------------------------------

    def execution_monitor_payload(
        self,
        goal_id: str,
        *,
        access: GoalAccessContext,
    ) -> dict[str, Any]:
        self._require(
            access,
            "can_view",
        )

        runtime = self._get(
            goal_id,
            access,
        )

        return {
            "goal_id": goal_id,
            "tenant_id": runtime.specification.tenant_id,
            "robot_id": runtime.assigned_robot_id,
            "status": runtime.status.value,
            "progress_percentage": (
                runtime.progress_percentage
            ),
            "started_at": to_json_safe(
                runtime.started_at
            ),
            "completed_at": to_json_safe(
                runtime.completed_at
            ),
            "failure_reason": runtime.failure_reason,
            "retry_count": runtime.retry_count,
            "simulation_only": True,
        }

    # --------------------------------------------------------
    # GET GOAL
    # --------------------------------------------------------

    def get_goal(
        self,
        goal_id: str,
        *,
        access: GoalAccessContext,
    ) -> dict[str, Any]:
        self._require(
            access,
            "can_view",
        )

        runtime = self._get(
            goal_id,
            access,
        )

        with runtime.lock:
            return runtime.snapshot()

    # --------------------------------------------------------
    # LIST GOALS
    # --------------------------------------------------------

    def list_goals(
        self,
        *,
        access: GoalAccessContext,
        status: GoalStatus | None = None,
        robot_id: str | None = None,
        goal_type: GoalType | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        self._require(
            access,
            "can_view",
        )

        if limit < 1:
            raise GoalValidationError(
                "Limit must be positive"
            )

        if robot_id is not None:
            self._authorize_robot(
                access,
                robot_id,
            )

        with self._lock:
            goals = list(
                self._goals.values()
            )

        results = []

        for runtime in goals:
            specification = runtime.specification

            if specification.tenant_id != access.tenant_id:
                continue

            if (
                runtime.assigned_robot_id is not None
                and runtime.assigned_robot_id
                not in access.allowed_robot_ids
            ):
                continue

            if (
                status is not None
                and runtime.status != status
            ):
                continue

            if (
                robot_id is not None
                and runtime.assigned_robot_id != robot_id
            ):
                continue

            if (
                goal_type is not None
                and specification.goal_type != goal_type
            ):
                continue

            with runtime.lock:
                results.append(
                    runtime.snapshot()
                )

            if len(results) >= limit:
                break

        return results

    # --------------------------------------------------------
    # GOAL EVENTS
    # --------------------------------------------------------

    def goal_events(
        self,
        goal_id: str,
        *,
        access: GoalAccessContext,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        self._require(
            access,
            "can_view",
        )

        if limit < 1:
            raise GoalValidationError(
                "Limit must be positive"
            )

        runtime = self._get(
            goal_id,
            access,
        )

        with runtime.lock:
            return to_json_safe(
                list(runtime.events)[-limit:]
            )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        access: GoalAccessContext,
    ) -> dict[str, Any]:
        goals = self.list_goals(
            access=access,
            limit=self.config.maximum_goals,
        )

        status_counts = Counter(
            goal["status"]
            for goal in goals
        )

        type_counts = Counter(
            goal["specification"]["goal_type"]
            for goal in goals
        )

        return {
            "tenant_id": access.tenant_id,
            "total_goals": len(goals),
            "statuses": dict(status_counts),
            "goal_types": dict(type_counts),
            "active_goals": status_counts.get(
                GoalStatus.ACTIVE.value,
                0,
            ),
            "successful_goals": status_counts.get(
                GoalStatus.SUCCEEDED.value,
                0,
            ),
            "failed_goals": status_counts.get(
                GoalStatus.FAILED.value,
                0,
            ),
            "average_progress_percentage": (
                sum(
                    goal["progress_percentage"]
                    for goal in goals
                ) / len(goals)
                if goals
                else 0.0
            ),
            "simulation_only": True,
            "hardware_execution": False,
            "timestamp": utc_now().isoformat(),
        }

    # --------------------------------------------------------
    # JSON EXPORT
    # --------------------------------------------------------

    def export_goal_json(
        self,
        goal_id: str,
        *,
        access: GoalAccessContext,
    ) -> str:
        payload = {
            "platform": (
                "Physical AI & Autonomous Robotics "
                "Intelligence Platform"
            ),
            "module": "goal_manager",
            "goal": self.get_goal(
                goal_id,
                access=access,
            ),
            "events": self.goal_events(
                goal_id,
                access=access,
            ),
            "simulation_only": True,
            "hardware_execution": False,
            "exported_at": utc_now().isoformat(),
        }

        return json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestRoboticsGoalManager(unittest.TestCase):

    def setUp(self) -> None:
        self.manager = RoboticsGoalManager(
            GoalManagerConfig(
                robot_state_max_age_seconds=60,
            )
        )

        self.access = GoalAccessContext(
            tenant_id="tenant_alpha",
            user_id="robotics_engineer",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
                "ROBOT-002",
            }),
            can_create=True,
            can_view=True,
            can_modify=True,
            can_assign=True,
            can_manage_approvals=True,
            can_update_progress=True,
            can_cancel=True,
            can_delete=True,
        )

        self.robot_state = GoalRobotState(
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            capabilities=frozenset({
                "inspection",
                "diagnostics",
            }),
            battery_percentage=90,
            safety_state=RobotGoalSafetyState.SAFE,
            available=True,
            current_zone="ZONE-A",
            allowed_zones=frozenset({
                "ZONE-A",
            }),
            maximum_payload_kg=20,
        )

        self.goal = GoalSpecification(
            goal_id="GOAL-001",
            tenant_id="tenant_alpha",
            name="Inspect Industrial Robot",
            goal_type=GoalType.INSPECTION,
            priority=GoalPriority.HIGH,
            robot_id="ROBOT-001",
            required_capabilities=frozenset({
                "inspection",
            }),
            required_zone="ZONE-A",
            postconditions=(
                GoalCondition(
                    condition_id="COND-001",
                    name="Inspection completed",
                    field_name="inspection_completed",
                    operator=GoalConditionOperator.EQ,
                    expected_value=True,
                    category=GoalConditionCategory.POSTCONDITION,
                ),
            ),
        )

    def prepare(self) -> None:
        self.manager.register_goal(
            self.goal,
            access=self.access,
        )

        self.manager.update_robot_state(
            self.robot_state,
            access=self.access,
        )

    def test_register_goal(self) -> None:
        self.prepare()

        result = self.manager.get_goal(
            "GOAL-001",
            access=self.access,
        )

        self.assertEqual(
            result["status"],
            "created",
        )

    def test_activate_goal(self) -> None:
        self.prepare()

        result = self.manager.activate_goal(
            "GOAL-001",
            access=self.access,
        )

        self.assertEqual(
            result["status"],
            "active",
        )

    def test_complete_goal(self) -> None:
        self.prepare()

        self.manager.activate_goal(
            "GOAL-001",
            access=self.access,
        )

        self.manager.update_progress(
            "GOAL-001",
            80,
            access=self.access,
        )

        result = self.manager.complete_goal(
            "GOAL-001",
            access=self.access,
            world_state={
                "inspection_completed": True,
            },
        )

        self.assertEqual(
            result["status"],
            "succeeded",
        )

    def test_postcondition_failure(self) -> None:
        self.prepare()

        self.manager.activate_goal(
            "GOAL-001",
            access=self.access,
        )

        with self.assertRaises(
            GoalStateError
        ):
            self.manager.complete_goal(
                "GOAL-001",
                access=self.access,
                world_state={
                    "inspection_completed": False,
                },
            )

    def test_unsafe_robot(self) -> None:
        self.manager.register_goal(
            self.goal,
            access=self.access,
        )

        unsafe = GoalRobotState(
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            capabilities=frozenset({
                "inspection",
            }),
            battery_percentage=90,
            safety_state=RobotGoalSafetyState.UNSAFE,
            available=True,
            allowed_zones=frozenset({
                "ZONE-A",
            }),
        )

        self.manager.update_robot_state(
            unsafe,
            access=self.access,
        )

        with self.assertRaises(
            GoalSafetyError
        ):
            self.manager.activate_goal(
                "GOAL-001",
                access=self.access,
            )

    def test_cancel_goal(self) -> None:
        self.prepare()

        result = self.manager.cancel_goal(
            "GOAL-001",
            access=self.access,
        )

        self.assertEqual(
            result["status"],
            "cancelled",
        )

    def test_tenant_isolation(self) -> None:
        self.prepare()

        other_access = GoalAccessContext(
            tenant_id="tenant_beta",
            user_id="other_user",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
            }),
            can_view=True,
        )

        with self.assertRaises(
            GoalAuthorizationError
        ):
            self.manager.get_goal(
                "GOAL-001",
                access=other_access,
            )

    def test_goal_decomposition(self) -> None:
        self.prepare()

        suggestions = self.manager.suggest_subgoals(
            "GOAL-001",
            access=self.access,
        )

        self.assertGreater(
            len(suggestions),
            0,
        )

    def test_json_export(self) -> None:
        self.prepare()

        result = self.manager.export_goal_json(
            "GOAL-001",
            access=self.access,
        )

        self.assertIn(
            "GOAL-001",
            result,
        )

        self.assertIn(
            "simulation_only",
            result,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print(
        "\nPHYSICAL AI & AUTONOMOUS ROBOTICS PLATFORM"
    )

    print(
        "Goal Manager — Hierarchical Goal Planning\n"
    )

    manager = RoboticsGoalManager()

    access = GoalAccessContext(
        tenant_id="tenant_alpha",
        user_id="robotics_engineer",
        allowed_robot_ids=frozenset({
            "ROBOT-001",
            "ROBOT-002",
        }),
        roles=frozenset({
            "robotics_engineer",
        }),
        can_create=True,
        can_view=True,
        can_modify=True,
        can_assign=True,
        can_manage_approvals=True,
        can_update_progress=True,
        can_cancel=True,
        can_delete=True,
    )

    robot_state = GoalRobotState(
        tenant_id="tenant_alpha",
        robot_id="ROBOT-001",
        capabilities=frozenset({
            "inspection",
            "diagnostics",
            "navigation",
        }),
        battery_percentage=92,
        safety_state=RobotGoalSafetyState.SAFE,
        available=True,
        current_zone="ZONE-A",
        allowed_zones=frozenset({
            "ZONE-A",
            "ZONE-B",
        }),
        maximum_payload_kg=25,
    )

    manager.update_robot_state(
        robot_state,
        access=access,
    )

    goal = GoalSpecification(
        goal_id="GOAL-INSPECTION-001",
        tenant_id="tenant_alpha",
        name="Industrial Robot Health Inspection",
        goal_type=GoalType.INSPECTION,
        description=(
            "Inspect industrial robot health, "
            "collect diagnostic data, and review findings"
        ),
        priority=GoalPriority.HIGH,
        robot_id="ROBOT-001",
        required_capabilities=frozenset({
            "inspection",
        }),
        required_zone="ZONE-A",
        timeout_seconds=900,
        estimated_duration_seconds=300,
        estimated_energy_wh=80,
        retry_limit=2,
        postconditions=(
            GoalCondition(
                condition_id="POST-001",
                name="Inspection completed",
                field_name="inspection_completed",
                operator=GoalConditionOperator.EQ,
                expected_value=True,
                category=GoalConditionCategory.POSTCONDITION,
            ),
        ),
        created_by=access.user_id,
    )

    manager.register_goal(
        goal,
        access=access,
    )

    print("GOAL READINESS\n")

    readiness = manager.evaluate_readiness(
        goal.goal_id,
        access=access,
    )

    print(
        json.dumps(
            to_json_safe(readiness),
            indent=2,
        )
    )

    manager.activate_goal(
        goal.goal_id,
        access=access,
    )

    manager.update_progress(
        goal.goal_id,
        50,
        access=access,
    )

    manager.complete_goal(
        goal.goal_id,
        access=access,
        world_state={
            "inspection_completed": True,
        },
    )

    print("\nGOAL RESULT\n")

    print(
        json.dumps(
            manager.get_goal(
                goal.goal_id,
                access=access,
            ),
            indent=2,
        )
    )

    print("\nBEHAVIOR TREE INTEGRATION\n")

    print(
        json.dumps(
            manager.behavior_tree_payload(
                goal.goal_id,
                access=access,
            ),
            indent=2,
        )
    )

    print("\nACTION SCHEDULER INTEGRATION\n")

    print(
        json.dumps(
            manager.action_scheduler_payload(
                goal.goal_id,
                access=access,
            ),
            indent=2,
        )
    )

    print("\nGOAL STATISTICS\n")

    print(
        json.dumps(
            manager.statistics(
                access=access,
            ),
            indent=2,
        )
    )

    print("\nUNIT TESTS\n")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestRoboticsGoalManager
    )

    unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)


if __name__ == "__main__":
    main()
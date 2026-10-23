"""Hierarchical Planner module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/planning/hierarchical_planner.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================

Module: hierarchical_planner.py

Hierarchical Task Network (HTN) planner for robotics.

Features:
- High-level goal decomposition
- Recursive task planning
- Compound and primitive tasks
- HTN methods and operators
- Preconditions and effects
- Task dependencies
- Priority-aware planning
- Robot capability validation
- Multi-robot task assignment recommendations
- Resource conflict detection
- Safety constraints
- Human approval gates
- Planning depth and timeout limits
- Cycle detection
- Plan cost and duration estimation
- Alternative plan generation
- Plan validation
- Behavior tree integration
- Action scheduler integration
- Goal manager integration
- Execution monitor integration
- Multi-tenant isolation
- Audit events
- JSON export
- Unit tests

Safety:
This module produces planning recommendations only.
It never sends physical robot commands.

A generated plan does not authorize hardware execution.
Physical execution requires independent safety systems,
authorization, current robot state, and command interlocks.

Python 3.11+
Standard library only.
"""

from __future__ import annotations

import copy
import heapq
import json
import math
import threading
import time
import unittest
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class HierarchicalPlannerError(Exception):
    pass


class PlanningValidationError(HierarchicalPlannerError):
    pass


class PlanningAuthorizationError(HierarchicalPlannerError):
    pass


class PlanningFailureError(HierarchicalPlannerError):
    pass


class PlanningTimeoutError(HierarchicalPlannerError):
    pass


class PlanningCycleError(HierarchicalPlannerError):
    pass


class PlanningSafetyError(HierarchicalPlannerError):
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
        raise PlanningValidationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > 128:
        raise PlanningValidationError(
            f"{name} must contain 1–128 characters"
        )

    return value


def finite_number(
    value: Any,
    name: str,
    minimum: float = 0.0,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
    ):
        raise PlanningValidationError(
            f"{name} must be finite and >= {minimum}"
        )

    return float(value)


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

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

    if isinstance(value, (tuple, list, set, frozenset)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, (str, int, bool)) or value is None:
        return value

    return repr(value)


# ============================================================
# ENUMS
# ============================================================

class TaskKind(str, Enum):
    PRIMITIVE = "primitive"
    COMPOUND = "compound"


class PlanningStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"


class TaskCategory(str, Enum):
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
    SAFETY_CHECK = "safety_check"
    SIMULATION = "simulation"
    CUSTOM = "custom"


class ConditionOperator(str, Enum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    EXISTS = "exists"


class PlanningPriority(int, Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class RobotSafetyState(str, Enum):
    SAFE = "safe"
    CAUTION = "caution"
    UNSAFE = "unsafe"
    EMERGENCY_STOP = "emergency_stop"
    UNKNOWN = "unknown"


class PlanRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class HierarchicalPlannerConfig:
    max_depth: int = 12
    max_expansions: int = 10000
    max_plan_steps: int = 500

    max_alternatives: int = 5

    planning_timeout_seconds: float = 10.0

    max_tasks_per_method: int = 100

    minimum_battery_percentage: float = 15.0

    robot_state_max_age_seconds: float = 30.0

    require_safe_robot_state: bool = True
    require_fresh_robot_state: bool = True

    require_human_approval_for_high_risk: bool = True

    reject_unknown_operators: bool = True
    reject_unresolved_compound_tasks: bool = True

    cost_weight: float = 1.0
    duration_weight: float = 0.2
    risk_weight: float = 10.0

    def __post_init__(self) -> None:
        for name in (
            "max_depth",
            "max_expansions",
            "max_plan_steps",
            "max_alternatives",
            "max_tasks_per_method",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise PlanningValidationError(
                    f"{name} must be positive"
                )

        for name in (
            "planning_timeout_seconds",
            "robot_state_max_age_seconds",
        ):
            finite_number(
                getattr(self, name),
                name,
                minimum=0.001,
            )

        finite_number(
            self.minimum_battery_percentage,
            "minimum_battery_percentage",
        )

        if self.minimum_battery_percentage > 100:
            raise PlanningValidationError(
                "Battery threshold exceeds 100%"
            )

        for name in (
            "cost_weight",
            "duration_weight",
            "risk_weight",
        ):
            finite_number(
                getattr(self, name),
                name,
            )


# ============================================================
# ACCESS CONTEXT
# ============================================================

@dataclass(frozen=True)
class PlannerAccessContext:
    tenant_id: str
    user_id: str

    allowed_robot_ids: frozenset[str] = frozenset()

    roles: frozenset[str] = frozenset()

    can_plan: bool = False
    can_view: bool = False
    can_register_tasks: bool = False
    can_register_methods: bool = False
    can_register_operators: bool = False
    can_approve_plans: bool = False

    def __post_init__(self) -> None:
        validate_identifier(
            self.tenant_id,
            "tenant_id",
        )

        validate_identifier(
            self.user_id,
            "user_id",
        )


# ============================================================
# CONDITIONS
# ============================================================

@dataclass(frozen=True)
class PlanningCondition:
    key: str
    operator: ConditionOperator
    expected: Any = None

    def __post_init__(self) -> None:
        validate_identifier(
            self.key,
            "condition key",
        )


class ConditionEvaluator:

    @staticmethod
    def evaluate(
        condition: PlanningCondition,
        state: Mapping[str, Any],
    ) -> bool:
        exists = condition.key in state

        actual = state.get(
            condition.key
        )

        expected = condition.expected

        try:
            if condition.operator == ConditionOperator.EXISTS:
                return exists

            if not exists:
                return False

            if condition.operator == ConditionOperator.EQ:
                return actual == expected

            if condition.operator == ConditionOperator.NE:
                return actual != expected

            if condition.operator == ConditionOperator.GT:
                return actual > expected

            if condition.operator == ConditionOperator.GTE:
                return actual >= expected

            if condition.operator == ConditionOperator.LT:
                return actual < expected

            if condition.operator == ConditionOperator.LTE:
                return actual <= expected

            if condition.operator == ConditionOperator.IN:
                return actual in expected

            if condition.operator == ConditionOperator.NOT_IN:
                return actual not in expected

        except (TypeError, ValueError):
            return False

        return False

    @classmethod
    def all_satisfied(
        cls,
        conditions: Sequence[PlanningCondition],
        state: Mapping[str, Any],
    ) -> bool:
        return all(
            cls.evaluate(condition, state)
            for condition in conditions
        )


# ============================================================
# TASK DEFINITION
# ============================================================

@dataclass(frozen=True)
class PlanningTask:
    task_id: str
    name: str

    kind: TaskKind

    category: TaskCategory = TaskCategory.CUSTOM

    parameters: dict[str, Any] = field(
        default_factory=dict
    )

    required_capabilities: frozenset[str] = frozenset()

    required_resources: frozenset[str] = frozenset()

    required_zone: str | None = None

    priority: PlanningPriority = PlanningPriority.NORMAL

    risk_level: PlanRiskLevel = PlanRiskLevel.LOW

    requires_human_approval: bool = False

    approval_id: str | None = None

    robot_id: str | None = None

    preconditions: tuple[PlanningCondition, ...] = ()

    postconditions: tuple[PlanningCondition, ...] = ()

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_identifier(
            self.task_id,
            "task_id",
        )

        validate_identifier(
            self.name,
            "task name",
        )

        if self.robot_id is not None:
            validate_identifier(
                self.robot_id,
                "robot_id",
            )

        if self.required_zone is not None:
            validate_identifier(
                self.required_zone,
                "required_zone",
            )


# ============================================================
# HTN METHOD
# ============================================================

@dataclass(frozen=True)
class HTNMethod:
    method_id: str
    name: str

    compound_task_name: str

    subtasks: tuple[PlanningTask, ...]

    preconditions: tuple[PlanningCondition, ...] = ()

    priority: int = 0

    estimated_cost: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_identifier(
            self.method_id,
            "method_id",
        )

        validate_identifier(
            self.name,
            "method name",
        )

        validate_identifier(
            self.compound_task_name,
            "compound_task_name",
        )

        finite_number(
            self.estimated_cost,
            "estimated_cost",
        )


# ============================================================
# PRIMITIVE OPERATOR
# ============================================================

@dataclass(frozen=True)
class PlanningOperator:
    operator_id: str

    task_name: str

    preconditions: tuple[PlanningCondition, ...] = ()

    effects: dict[str, Any] = field(
        default_factory=dict
    )

    estimated_cost: float = 1.0

    estimated_duration_seconds: float = 1.0

    estimated_energy_wh: float = 0.0

    risk_level: PlanRiskLevel = PlanRiskLevel.LOW

    required_capabilities: frozenset[str] = frozenset()

    required_resources: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        validate_identifier(
            self.operator_id,
            "operator_id",
        )

        validate_identifier(
            self.task_name,
            "task_name",
        )

        for name in (
            "estimated_cost",
            "estimated_duration_seconds",
            "estimated_energy_wh",
        ):
            finite_number(
                getattr(self, name),
                name,
            )


# ============================================================
# ROBOT SNAPSHOT
# ============================================================

@dataclass(frozen=True)
class PlannerRobotState:
    tenant_id: str
    robot_id: str

    capabilities: frozenset[str]

    battery_percentage: float

    safety_state: RobotSafetyState = (
        RobotSafetyState.UNKNOWN
    )

    emergency_stop_active: bool = False

    available: bool = False

    active_faults: tuple[str, ...] = ()

    allowed_zones: frozenset[str] = frozenset()

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_identifier(
            self.tenant_id,
            "tenant_id",
        )

        validate_identifier(
            self.robot_id,
            "robot_id",
        )

        finite_number(
            self.battery_percentage,
            "battery_percentage",
        )

        if self.battery_percentage > 100:
            raise PlanningValidationError(
                "Battery percentage exceeds 100"
            )

        if (
            self.timestamp.tzinfo is None
            or self.timestamp.utcoffset() is None
        ):
            raise PlanningValidationError(
                "Robot timestamp must be timezone-aware"
            )


# ============================================================
# PLANNING REQUEST
# ============================================================

@dataclass(frozen=True)
class HierarchicalPlanningRequest:
    request_id: str

    tenant_id: str

    root_task: PlanningTask

    initial_state: dict[str, Any]

    robot_states: tuple[PlannerRobotState, ...] = ()

    approved_ids: frozenset[str] = frozenset()

    maximum_alternatives: int = 1

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_identifier(
            self.request_id,
            "request_id",
        )

        validate_identifier(
            self.tenant_id,
            "tenant_id",
        )

        if self.maximum_alternatives < 1:
            raise PlanningValidationError(
                "maximum_alternatives must be positive"
            )


# ============================================================
# PLAN STEP
# ============================================================

@dataclass(frozen=True)
class HierarchicalPlanStep:
    step_id: str

    task_id: str
    task_name: str

    operator_id: str

    robot_id: str

    category: TaskCategory

    priority: PlanningPriority

    risk_level: PlanRiskLevel

    required_resources: frozenset[str]

    required_capabilities: frozenset[str]

    required_zone: str | None

    requires_human_approval: bool

    estimated_cost: float
    estimated_duration_seconds: float
    estimated_energy_wh: float

    preconditions: tuple[PlanningCondition, ...]

    effects: dict[str, Any]

    sequence_index: int

    parent_task_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# PLAN RESULT
# ============================================================

@dataclass(frozen=True)
class HierarchicalPlan:
    plan_id: str

    request_id: str
    tenant_id: str

    root_task_id: str

    status: PlanningStatus

    steps: tuple[HierarchicalPlanStep, ...]

    total_cost: float
    total_duration_seconds: float
    total_energy_wh: float

    risk_level: PlanRiskLevel

    final_state: dict[str, Any]

    method_trace: tuple[str, ...]

    warnings: tuple[str, ...] = ()

    errors: tuple[str, ...] = ()

    created_at: datetime = field(
        default_factory=utc_now
    )

    simulation_only: bool = True

    hardware_execution_authorized: bool = False


@dataclass(frozen=True)
class HierarchicalPlanningResponse:
    request_id: str

    status: PlanningStatus

    plans: tuple[HierarchicalPlan, ...]

    explored_nodes: int

    planning_duration_ms: float

    errors: tuple[str, ...] = ()

    warnings: tuple[str, ...] = ()


# ============================================================
# ACCESS CONTROLLER
# ============================================================

class PlannerAccessController:

    @staticmethod
    def require(
        access: PlannerAccessContext,
        permission: str,
    ) -> None:
        if not getattr(access, permission, False):
            raise PlanningAuthorizationError(
                f"Missing permission: {permission}"
            )

    @staticmethod
    def require_tenant(
        access: PlannerAccessContext,
        tenant_id: str,
    ) -> None:
        if access.tenant_id != tenant_id:
            raise PlanningAuthorizationError(
                "Cross-tenant planning denied"
            )

    @staticmethod
    def require_robot(
        access: PlannerAccessContext,
        robot_id: str,
    ) -> None:
        if robot_id not in access.allowed_robot_ids:
            raise PlanningAuthorizationError(
                "Robot access denied"
            )


# ============================================================
# ROBOT ELIGIBILITY
# ============================================================

class RobotPlanningEligibility:

    def __init__(
        self,
        config: HierarchicalPlannerConfig,
    ) -> None:
        self.config = config

    def check(
        self,
        task: PlanningTask,
        operator: PlanningOperator,
        robot: PlannerRobotState,
    ) -> tuple[bool, tuple[str, ...]]:
        reasons = []

        if not robot.available:
            reasons.append(
                "Robot unavailable"
            )

        if self.config.require_safe_robot_state:
            if robot.safety_state != RobotSafetyState.SAFE:
                reasons.append(
                    "Robot safety state not SAFE"
                )

        if robot.emergency_stop_active:
            reasons.append(
                "Emergency stop active"
            )

        if robot.active_faults:
            reasons.append(
                "Active robot faults"
            )

        if (
            robot.battery_percentage
            < self.config.minimum_battery_percentage
        ):
            reasons.append(
                "Battery below planning threshold"
            )

        if self.config.require_fresh_robot_state:
            age = (
                utc_now()
                - robot.timestamp.astimezone(timezone.utc)
            ).total_seconds()

            if (
                age < 0
                or age
                > self.config.robot_state_max_age_seconds
            ):
                reasons.append(
                    "Robot state stale"
                )

        required_capabilities = (
            task.required_capabilities
            | operator.required_capabilities
        )

        if not required_capabilities.issubset(
            robot.capabilities
        ):
            reasons.append(
                "Required capabilities unavailable"
            )

        if (
            task.required_zone is not None
            and task.required_zone
            not in robot.allowed_zones
        ):
            reasons.append(
                "Required zone not authorized"
            )

        return (
            not reasons,
            tuple(reasons),
        )


# ============================================================
# RESOURCE CONFLICT CHECKER
# ============================================================

class PlanningResourceChecker:

    @staticmethod
    def find_conflicts(
        steps: Sequence[HierarchicalPlanStep],
    ) -> tuple[str, ...]:
        """
        Detects duplicate resource use by distinct steps.

        This is conservative reporting. Sequential steps may
        legitimately reuse the same resource.
        """
        resource_owners: dict[
            str, list[str]
        ] = {}

        for step in steps:
            for resource in step.required_resources:
                resource_owners.setdefault(
                    resource,
                    [],
                ).append(
                    step.step_id
                )

        return tuple(
            f"Resource {resource} is used by "
            f"{len(owners)} plan steps"
            for resource, owners in resource_owners.items()
            if len(owners) > 1
        )


# ============================================================
# PLAN SCORING
# ============================================================

class HierarchicalPlanScorer:

    RISK_SCORES = {
        PlanRiskLevel.LOW: 0,
        PlanRiskLevel.MEDIUM: 1,
        PlanRiskLevel.HIGH: 3,
        PlanRiskLevel.CRITICAL: 10,
    }

    def __init__(
        self,
        config: HierarchicalPlannerConfig,
    ) -> None:
        self.config = config

    def score(
        self,
        plan: HierarchicalPlan,
    ) -> float:
        return (
            self.config.cost_weight
            * plan.total_cost
            + self.config.duration_weight
            * plan.total_duration_seconds
            + self.config.risk_weight
            * self.RISK_SCORES[plan.risk_level]
        )


# ============================================================
# INTERNAL SEARCH NODE
# ============================================================

@dataclass
class _SearchNode:
    pending_tasks: tuple[
        tuple[PlanningTask, str | None], ...
    ]

    world_state: dict[str, Any]

    steps: list[HierarchicalPlanStep]

    method_trace: list[str]

    active_compound_path: tuple[str, ...]

    total_cost: float = 0.0
    total_duration: float = 0.0
    total_energy: float = 0.0

    warnings: list[str] = field(
        default_factory=list
    )


# ============================================================
# HIERARCHICAL PLANNER
# ============================================================

class RoboticsHierarchicalPlanner:
    """
    Deterministic, bounded HTN planner.

    Compound tasks are decomposed using registered methods.
    Primitive tasks are expanded using registered operators.

    Search simulates operator effects on a copied world state.

    No hardware execution occurs.
    """

    def __init__(
        self,
        config: HierarchicalPlannerConfig | None = None,
    ) -> None:
        self.config = (
            config
            or HierarchicalPlannerConfig()
        )

        self._methods: dict[
            tuple[str, str],
            list[HTNMethod],
        ] = {}

        self._operators: dict[
            tuple[str, str],
            PlanningOperator,
        ] = {}

        self._lock = threading.RLock()

        self._eligibility = RobotPlanningEligibility(
            self.config
        )

        self._scorer = HierarchicalPlanScorer(
            self.config
        )

        self._audit: list[dict[str, Any]] = []

        self._statistics = {
            "planning_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "generated_plans": 0,
        }

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit_event(
        self,
        *,
        tenant_id: str,
        actor: str,
        action: str,
        details: Mapping[str, Any],
    ) -> None:
        self._audit.append({
            "event_id": generate_id(),
            "tenant_id": tenant_id,
            "actor": actor,
            "action": action,
            "details": json_safe(details),
            "timestamp": utc_now().isoformat(),
        })

    # --------------------------------------------------------
    # REGISTER METHOD
    # --------------------------------------------------------

    def register_method(
        self,
        method: HTNMethod,
        *,
        access: PlannerAccessContext,
    ) -> None:
        PlannerAccessController.require(
            access,
            "can_register_methods",
        )

        if (
            len(method.subtasks)
            > self.config.max_tasks_per_method
        ):
            raise PlanningValidationError(
                "Method contains too many subtasks"
            )

        key = (
            access.tenant_id,
            method.compound_task_name,
        )

        with self._lock:
            methods = self._methods.setdefault(
                key,
                [],
            )

            if any(
                item.method_id == method.method_id
                for item in methods
            ):
                raise PlanningValidationError(
                    "Duplicate method ID"
                )

            methods.append(method)

            methods.sort(
                key=lambda item: (
                    -item.priority,
                    item.estimated_cost,
                    item.method_id,
                )
            )

            self._audit_event(
                tenant_id=access.tenant_id,
                actor=access.user_id,
                action="method_registered",
                details={
                    "method_id": method.method_id,
                },
            )

    # --------------------------------------------------------
    # REGISTER OPERATOR
    # --------------------------------------------------------

    def register_operator(
        self,
        operator: PlanningOperator,
        *,
        access: PlannerAccessContext,
    ) -> None:
        PlannerAccessController.require(
            access,
            "can_register_operators",
        )

        key = (
            access.tenant_id,
            operator.task_name,
        )

        with self._lock:
            if key in self._operators:
                raise PlanningValidationError(
                    "Operator already registered for task"
                )

            self._operators[key] = operator

            self._audit_event(
                tenant_id=access.tenant_id,
                actor=access.user_id,
                action="operator_registered",
                details={
                    "operator_id": operator.operator_id,
                },
            )

    # --------------------------------------------------------
    # GET METHODS
    # --------------------------------------------------------

    def get_methods(
        self,
        task_name: str,
        *,
        access: PlannerAccessContext,
    ) -> tuple[HTNMethod, ...]:
        PlannerAccessController.require(
            access,
            "can_view",
        )

        with self._lock:
            return tuple(
                self._methods.get(
                    (access.tenant_id, task_name),
                    (),
                )
            )

    # --------------------------------------------------------
    # GET OPERATOR
    # --------------------------------------------------------

    def get_operator(
        self,
        task_name: str,
        *,
        access: PlannerAccessContext,
    ) -> PlanningOperator | None:
        PlannerAccessController.require(
            access,
            "can_view",
        )

        with self._lock:
            return self._operators.get(
                (access.tenant_id, task_name)
            )

    # --------------------------------------------------------
    # ROBOT SELECTION
    # --------------------------------------------------------

    def _select_robot(
        self,
        task: PlanningTask,
        operator: PlanningOperator,
        request: HierarchicalPlanningRequest,
        access: PlannerAccessContext,
    ) -> PlannerRobotState | None:
        candidates = []

        for robot in request.robot_states:
            if robot.tenant_id != request.tenant_id:
                continue

            if robot.robot_id not in access.allowed_robot_ids:
                continue

            if (
                task.robot_id is not None
                and task.robot_id != robot.robot_id
            ):
                continue

            eligible, _ = self._eligibility.check(
                task,
                operator,
                robot,
            )

            if eligible:
                candidates.append(robot)

        if not candidates:
            return None

        candidates.sort(
            key=lambda robot: (
                -robot.battery_percentage,
                robot.robot_id,
            )
        )

        return candidates[0]

    # --------------------------------------------------------
    # SAFETY APPROVAL
    # --------------------------------------------------------

    def _approval_satisfied(
        self,
        task: PlanningTask,
        request: HierarchicalPlanningRequest,
    ) -> bool:
        high_risk = task.risk_level in (
            PlanRiskLevel.HIGH,
            PlanRiskLevel.CRITICAL,
        )

        requires_approval = (
            task.requires_human_approval
            or (
                high_risk
                and self.config.require_human_approval_for_high_risk
            )
        )

        if not requires_approval:
            return True

        if task.approval_id is None:
            return False

        return task.approval_id in request.approved_ids

    # --------------------------------------------------------
    # EXPAND PRIMITIVE TASK
    # --------------------------------------------------------

    def _expand_primitive(
        self,
        task: PlanningTask,
        parent_task_id: str | None,
        node: _SearchNode,
        request: HierarchicalPlanningRequest,
        access: PlannerAccessContext,
        operators: Mapping[str, PlanningOperator],
    ) -> _SearchNode | None:
        operator = operators.get(
            task.name
        )

        if operator is None:
            return None

        if not self._approval_satisfied(
            task,
            request,
        ):
            return None

        conditions = (
            task.preconditions
            + operator.preconditions
        )

        if not ConditionEvaluator.all_satisfied(
            conditions,
            node.world_state,
        ):
            return None

        robot = self._select_robot(
            task,
            operator,
            request,
            access,
        )

        if robot is None:
            return None

        if len(node.steps) >= self.config.max_plan_steps:
            return None

        new_state = copy.deepcopy(
            node.world_state
        )

        for key, value in operator.effects.items():
            new_state[key] = copy.deepcopy(
                value
            )

        if not ConditionEvaluator.all_satisfied(
            task.postconditions,
            new_state,
        ):
            return None

        risk = max(
            (
                task.risk_level,
                operator.risk_level,
            ),
            key=lambda item: (
                HierarchicalPlanScorer.RISK_SCORES[item]
            ),
        )

        step = HierarchicalPlanStep(
            step_id=generate_id(),
            task_id=task.task_id,
            task_name=task.name,
            operator_id=operator.operator_id,
            robot_id=robot.robot_id,
            category=task.category,
            priority=task.priority,
            risk_level=risk,
            required_resources=(
                task.required_resources
                | operator.required_resources
            ),
            required_capabilities=(
                task.required_capabilities
                | operator.required_capabilities
            ),
            required_zone=task.required_zone,
            requires_human_approval=(
                task.requires_human_approval
                or risk in (
                    PlanRiskLevel.HIGH,
                    PlanRiskLevel.CRITICAL,
                )
            ),
            estimated_cost=operator.estimated_cost,
            estimated_duration_seconds=(
                operator.estimated_duration_seconds
            ),
            estimated_energy_wh=(
                operator.estimated_energy_wh
            ),
            preconditions=conditions,
            effects=copy.deepcopy(
                operator.effects
            ),
            sequence_index=len(node.steps),
            parent_task_id=parent_task_id,
            metadata=copy.deepcopy(
                task.metadata
            ),
        )

        return _SearchNode(
            pending_tasks=node.pending_tasks[1:],
            world_state=new_state,
            steps=node.steps + [step],
            method_trace=list(
                node.method_trace
            ),
            active_compound_path=(
                node.active_compound_path
            ),
            total_cost=(
                node.total_cost
                + operator.estimated_cost
            ),
            total_duration=(
                node.total_duration
                + operator.estimated_duration_seconds
            ),
            total_energy=(
                node.total_energy
                + operator.estimated_energy_wh
            ),
            warnings=list(
                node.warnings
            ),
        )

    # --------------------------------------------------------
    # EXPAND COMPOUND TASK
    # --------------------------------------------------------

    def _expand_compound(
        self,
        task: PlanningTask,
        node: _SearchNode,
        methods: Mapping[
            str, tuple[HTNMethod, ...]
        ],
    ) -> list[_SearchNode]:
        if task.name in node.active_compound_path:
            return []

        if (
            len(node.active_compound_path)
            >= self.config.max_depth
        ):
            return []

        alternatives = []

        for method in methods.get(task.name, ()):
            if not ConditionEvaluator.all_satisfied(
                task.preconditions
                + method.preconditions,
                node.world_state,
            ):
                continue

            expanded_tasks = tuple(
                (
                    subtask,
                    task.task_id,
                )
                for subtask in method.subtasks
            )

            alternatives.append(
                _SearchNode(
                    pending_tasks=(
                        expanded_tasks
                        + node.pending_tasks[1:]
                    ),
                    world_state=copy.deepcopy(
                        node.world_state
                    ),
                    steps=list(
                        node.steps
                    ),
                    method_trace=(
                        node.method_trace
                        + [method.method_id]
                    ),
                    active_compound_path=(
                        node.active_compound_path
                        + (task.name,)
                    ),
                    total_cost=(
                        node.total_cost
                        + method.estimated_cost
                    ),
                    total_duration=node.total_duration,
                    total_energy=node.total_energy,
                    warnings=list(
                        node.warnings
                    ),
                )
            )

        return alternatives

    # --------------------------------------------------------
    # BUILD PLAN
    # --------------------------------------------------------

    def _build_plan(
        self,
        node: _SearchNode,
        request: HierarchicalPlanningRequest,
    ) -> HierarchicalPlan:
        risk_order = {
            PlanRiskLevel.LOW: 0,
            PlanRiskLevel.MEDIUM: 1,
            PlanRiskLevel.HIGH: 2,
            PlanRiskLevel.CRITICAL: 3,
        }

        maximum_risk = max(
            (
                step.risk_level
                for step in node.steps
            ),
            default=PlanRiskLevel.LOW,
            key=lambda risk: risk_order[risk],
        )

        warnings = list(
            node.warnings
        )

        warnings.extend(
            PlanningResourceChecker.find_conflicts(
                node.steps
            )
        )

        return HierarchicalPlan(
            plan_id=generate_id(),
            request_id=request.request_id,
            tenant_id=request.tenant_id,
            root_task_id=request.root_task.task_id,
            status=PlanningStatus.SUCCESS,
            steps=tuple(node.steps),
            total_cost=node.total_cost,
            total_duration_seconds=node.total_duration,
            total_energy_wh=node.total_energy,
            risk_level=maximum_risk,
            final_state=copy.deepcopy(
                node.world_state
            ),
            method_trace=tuple(
                node.method_trace
            ),
            warnings=tuple(warnings),
        )

    # --------------------------------------------------------
    # MAIN PLANNER
    # --------------------------------------------------------

    def plan(
        self,
        request: HierarchicalPlanningRequest,
        *,
        access: PlannerAccessContext,
    ) -> HierarchicalPlanningResponse:
        PlannerAccessController.require(
            access,
            "can_plan",
        )

        PlannerAccessController.require_tenant(
            access,
            request.tenant_id,
        )

        started = time.monotonic()

        with self._lock:
            methods = {
                task_name: tuple(items)
                for (tenant_id, task_name), items
                in self._methods.items()
                if tenant_id == request.tenant_id
            }

            operators = {
                task_name: operator
                for (tenant_id, task_name), operator
                in self._operators.items()
                if tenant_id == request.tenant_id
            }

            self._statistics["planning_requests"] += 1

        initial = _SearchNode(
            pending_tasks=(
                (
                    request.root_task,
                    None,
                ),
            ),
            world_state=copy.deepcopy(
                request.initial_state
            ),
            steps=[],
            method_trace=[],
            active_compound_path=(),
        )

        frontier = []
        sequence = 0

        heapq.heappush(
            frontier,
            (
                0.0,
                sequence,
                initial,
            ),
        )

        explored = 0
        completed = []

        errors = []
        timed_out = False

        maximum_alternatives = min(
            request.maximum_alternatives,
            self.config.max_alternatives,
        )

        while frontier:
            elapsed = (
                time.monotonic() - started
            )

            if elapsed > self.config.planning_timeout_seconds:
                timed_out = True
                errors.append(
                    "Planning timeout exceeded"
                )
                break

            if explored >= self.config.max_expansions:
                errors.append(
                    "Maximum planning expansions exceeded"
                )
                break

            _, _, node = heapq.heappop(
                frontier
            )

            explored += 1

            if not node.pending_tasks:
                completed.append(
                    self._build_plan(
                        node,
                        request,
                    )
                )

                if (
                    len(completed)
                    >= maximum_alternatives
                ):
                    break

                continue

            task, parent_id = node.pending_tasks[0]

            if task.kind == TaskKind.PRIMITIVE:
                next_node = self._expand_primitive(
                    task,
                    parent_id,
                    node,
                    request,
                    access,
                    operators,
                )

                if next_node is not None:
                    sequence += 1

                    heapq.heappush(
                        frontier,
                        (
                            next_node.total_cost,
                            sequence,
                            next_node,
                        ),
                    )

            else:
                alternatives = self._expand_compound(
                    task,
                    node,
                    methods,
                )

                for alternative in alternatives:
                    sequence += 1

                    heapq.heappush(
                        frontier,
                        (
                            alternative.total_cost,
                            sequence,
                            alternative,
                        ),
                    )

        completed.sort(
            key=self._scorer.score
        )

        duration_ms = (
            time.monotonic() - started
        ) * 1000

        if completed:
            status = PlanningStatus.SUCCESS

        elif timed_out:
            status = PlanningStatus.TIMEOUT

        else:
            status = PlanningStatus.FAILED

        with self._lock:
            if completed:
                self._statistics[
                    "successful_requests"
                ] += 1

                self._statistics[
                    "generated_plans"
                ] += len(completed)

            else:
                self._statistics[
                    "failed_requests"
                ] += 1

            self._audit_event(
                tenant_id=request.tenant_id,
                actor=access.user_id,
                action="planning_completed",
                details={
                    "request_id": request.request_id,
                    "status": status.value,
                    "explored_nodes": explored,
                    "generated_plans": len(completed),
                    "duration_ms": duration_ms,
                },
            )

        return HierarchicalPlanningResponse(
            request_id=request.request_id,
            status=status,
            plans=tuple(completed),
            explored_nodes=explored,
            planning_duration_ms=duration_ms,
            errors=tuple(errors),
        )

    # --------------------------------------------------------
    # PLAN VALIDATION
    # --------------------------------------------------------

    def validate_plan(
        self,
        plan: HierarchicalPlan,
        *,
        access: PlannerAccessContext,
    ) -> dict[str, Any]:
        PlannerAccessController.require(
            access,
            "can_view",
        )

        PlannerAccessController.require_tenant(
            access,
            plan.tenant_id,
        )

        issues = []

        for step in plan.steps:
            if step.robot_id not in access.allowed_robot_ids:
                issues.append(
                    f"Robot access denied: {step.robot_id}"
                )

            if (
                step.requires_human_approval
                and step.risk_level
                in (
                    PlanRiskLevel.HIGH,
                    PlanRiskLevel.CRITICAL,
                )
            ):
                issues.append(
                    f"Step {step.step_id} requires "
                    "independent execution authorization"
                )

        return {
            "plan_id": plan.plan_id,
            "structurally_valid": not any(
                "access denied" in issue
                for issue in issues
            ),
            "issues": issues,
            "simulation_only": True,
            "hardware_execution_authorized": False,
        }

    # --------------------------------------------------------
    # GOAL MANAGER ADAPTER
    # --------------------------------------------------------

    @staticmethod
    def goal_manager_payload(
        plan: HierarchicalPlan,
    ) -> dict[str, Any]:
        return {
            "goal_id": plan.root_task_id,
            "plan_id": plan.plan_id,
            "tenant_id": plan.tenant_id,
            "status": plan.status.value,
            "step_count": len(plan.steps),
            "estimated_duration_seconds": (
                plan.total_duration_seconds
            ),
            "estimated_cost": plan.total_cost,
            "risk_level": plan.risk_level.value,
            "simulation_only": True,
        }

    # --------------------------------------------------------
    # BEHAVIOR TREE ADAPTER
    # --------------------------------------------------------

    @staticmethod
    def behavior_tree_payload(
        plan: HierarchicalPlan,
    ) -> dict[str, Any]:
        return {
            "tree_id": generate_id(),
            "plan_id": plan.plan_id,
            "root": {
                "type": "sequence",
                "name": "HTN Generated Sequence",
                "children": [
                    {
                        "type": "action",
                        "name": step.task_name,
                        "step_id": step.step_id,
                        "robot_id": step.robot_id,
                        "operator_id": step.operator_id,
                        "simulation_only": True,
                    }
                    for step in plan.steps
                ],
            },
            "hardware_execution_authorized": False,
        }

    # --------------------------------------------------------
    # ACTION SCHEDULER ADAPTER
    # --------------------------------------------------------

    @staticmethod
    def action_scheduler_payload(
        plan: HierarchicalPlan,
    ) -> dict[str, Any]:
        return {
            "plan_id": plan.plan_id,
            "tenant_id": plan.tenant_id,
            "tasks": [
                {
                    "step_id": step.step_id,
                    "task_id": step.task_id,
                    "task_name": step.task_name,
                    "robot_id": step.robot_id,
                    "priority": step.priority.value,
                    "duration_seconds": (
                        step.estimated_duration_seconds
                    ),
                    "resources": sorted(
                        step.required_resources
                    ),
                    "simulation_only": True,
                }
                for step in plan.steps
            ],
            "hardware_execution_authorized": False,
        }

    # --------------------------------------------------------
    # EXECUTION MONITOR ADAPTER
    # --------------------------------------------------------

    @staticmethod
    def execution_monitor_payload(
        plan: HierarchicalPlan,
    ) -> dict[str, Any]:
        return {
            "plan_id": plan.plan_id,
            "tenant_id": plan.tenant_id,
            "expected_steps": len(plan.steps),
            "expected_duration_seconds": (
                plan.total_duration_seconds
            ),
            "expected_energy_wh": (
                plan.total_energy_wh
            ),
            "risk_level": plan.risk_level.value,
            "simulation_only": True,
        }

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    def audit_history(
        self,
        *,
        access: PlannerAccessContext,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        PlannerAccessController.require(
            access,
            "can_view",
        )

        if limit < 1:
            raise PlanningValidationError(
                "Limit must be positive"
            )

        with self._lock:
            events = [
                event
                for event in self._audit
                if event["tenant_id"]
                == access.tenant_id
            ]

        return events[-limit:]

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        access: PlannerAccessContext,
    ) -> dict[str, Any]:
        PlannerAccessController.require(
            access,
            "can_view",
        )

        with self._lock:
            method_count = sum(
                len(methods)
                for (tenant_id, _), methods
                in self._methods.items()
                if tenant_id == access.tenant_id
            )

            operator_count = sum(
                1
                for tenant_id, _
                in self._operators
                if tenant_id == access.tenant_id
            )

        return {
            "tenant_id": access.tenant_id,
            "registered_methods": method_count,
            "registered_operators": operator_count,
            "simulation_only": True,
            "hardware_execution_authorized": False,
            "timestamp": utc_now().isoformat(),
        }

    # --------------------------------------------------------
    # JSON EXPORT
    # --------------------------------------------------------

    @staticmethod
    def export_response_json(
        response: HierarchicalPlanningResponse,
    ) -> str:
        return json.dumps(
            json_safe(response),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


# ============================================================
# TESTS
# ============================================================

class TestRoboticsHierarchicalPlanner(unittest.TestCase):

    def setUp(self) -> None:
        self.planner = RoboticsHierarchicalPlanner()

        self.access = PlannerAccessContext(
            tenant_id="tenant_alpha",
            user_id="engineer",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
            }),
            can_plan=True,
            can_view=True,
            can_register_methods=True,
            can_register_operators=True,
        )

        self.robot = PlannerRobotState(
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            capabilities=frozenset({
                "inspection",
                "diagnostics",
            }),
            battery_percentage=90,
            safety_state=RobotSafetyState.SAFE,
            available=True,
            allowed_zones=frozenset({
                "ZONE-A",
            }),
        )

        self.inspect_task = PlanningTask(
            task_id="TASK-INSPECT",
            name="inspect_asset",
            kind=TaskKind.PRIMITIVE,
            category=TaskCategory.INSPECTION,
            robot_id="ROBOT-001",
            required_capabilities=frozenset({
                "inspection",
            }),
            required_zone="ZONE-A",
        )

        self.root_task = PlanningTask(
            task_id="TASK-ROOT",
            name="robot_health_assessment",
            kind=TaskKind.COMPOUND,
            category=TaskCategory.DIAGNOSTICS,
        )

        self.planner.register_operator(
            PlanningOperator(
                operator_id="OP-INSPECT",
                task_name="inspect_asset",
                effects={
                    "inspection_completed": True,
                },
                estimated_cost=5,
                estimated_duration_seconds=60,
                required_capabilities=frozenset({
                    "inspection",
                }),
            ),
            access=self.access,
        )

        self.planner.register_method(
            HTNMethod(
                method_id="METHOD-HEALTH",
                name="Health Assessment Method",
                compound_task_name=(
                    "robot_health_assessment"
                ),
                subtasks=(
                    self.inspect_task,
                ),
            ),
            access=self.access,
        )

    def request(self) -> HierarchicalPlanningRequest:
        return HierarchicalPlanningRequest(
            request_id=generate_id(),
            tenant_id="tenant_alpha",
            root_task=self.root_task,
            initial_state={},
            robot_states=(
                self.robot,
            ),
        )

    def test_successful_plan(self) -> None:
        result = self.planner.plan(
            self.request(),
            access=self.access,
        )

        self.assertEqual(
            result.status,
            PlanningStatus.SUCCESS,
        )

        self.assertEqual(
            len(result.plans[0].steps),
            1,
        )

    def test_operator_effect(self) -> None:
        result = self.planner.plan(
            self.request(),
            access=self.access,
        )

        self.assertTrue(
            result.plans[0].final_state[
                "inspection_completed"
            ]
        )

    def test_unsafe_robot(self) -> None:
        unsafe_robot = PlannerRobotState(
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            capabilities=frozenset({
                "inspection",
            }),
            battery_percentage=90,
            safety_state=RobotSafetyState.UNSAFE,
            available=True,
            allowed_zones=frozenset({
                "ZONE-A",
            }),
        )

        request = HierarchicalPlanningRequest(
            request_id=generate_id(),
            tenant_id="tenant_alpha",
            root_task=self.root_task,
            initial_state={},
            robot_states=(
                unsafe_robot,
            ),
        )

        result = self.planner.plan(
            request,
            access=self.access,
        )

        self.assertEqual(
            result.status,
            PlanningStatus.FAILED,
        )

    def test_tenant_isolation(self) -> None:
        other_access = PlannerAccessContext(
            tenant_id="tenant_beta",
            user_id="other",
            can_plan=True,
        )

        with self.assertRaises(
            PlanningAuthorizationError
        ):
            self.planner.plan(
                self.request(),
                access=other_access,
            )

    def test_behavior_tree_adapter(self) -> None:
        result = self.planner.plan(
            self.request(),
            access=self.access,
        )

        payload = self.planner.behavior_tree_payload(
            result.plans[0]
        )

        self.assertEqual(
            payload["root"]["type"],
            "sequence",
        )

    def test_json_export(self) -> None:
        result = self.planner.plan(
            self.request(),
            access=self.access,
        )

        exported = self.planner.export_response_json(
            result
        )

        self.assertIn(
            "robot_health_assessment",
            exported,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print(
        "\nPHYSICAL AI & AUTONOMOUS ROBOTICS INTELLIGENCE"
    )

    print(
        "Hierarchical Task Network Planner\n"
    )

    planner = RoboticsHierarchicalPlanner()

    access = PlannerAccessContext(
        tenant_id="tenant_alpha",
        user_id="robotics_engineer",
        allowed_robot_ids=frozenset({
            "ROBOT-001",
        }),
        can_plan=True,
        can_view=True,
        can_register_methods=True,
        can_register_operators=True,
    )

    robot = PlannerRobotState(
        tenant_id="tenant_alpha",
        robot_id="ROBOT-001",
        capabilities=frozenset({
            "inspection",
            "diagnostics",
        }),
        battery_percentage=95,
        safety_state=RobotSafetyState.SAFE,
        available=True,
        allowed_zones=frozenset({
            "ZONE-A",
        }),
    )

    inspect = PlanningTask(
        task_id="TASK-INSPECT",
        name="inspect_robot",
        kind=TaskKind.PRIMITIVE,
        category=TaskCategory.INSPECTION,
        robot_id="ROBOT-001",
        required_capabilities=frozenset({
            "inspection",
        }),
        required_zone="ZONE-A",
    )

    diagnose = PlanningTask(
        task_id="TASK-DIAGNOSE",
        name="diagnose_robot",
        kind=TaskKind.PRIMITIVE,
        category=TaskCategory.DIAGNOSTICS,
        robot_id="ROBOT-001",
        required_capabilities=frozenset({
            "diagnostics",
        }),
        preconditions=(
            PlanningCondition(
                key="inspection_completed",
                operator=ConditionOperator.EQ,
                expected=True,
            ),
        ),
    )

    root = PlanningTask(
        task_id="TASK-ROOT",
        name="robot_health_workflow",
        kind=TaskKind.COMPOUND,
        category=TaskCategory.MAINTENANCE,
        priority=PlanningPriority.HIGH,
    )

    planner.register_operator(
        PlanningOperator(
            operator_id="OP-INSPECT",
            task_name="inspect_robot",
            effects={
                "inspection_completed": True,
            },
            estimated_cost=5,
            estimated_duration_seconds=60,
        ),
        access=access,
    )

    planner.register_operator(
        PlanningOperator(
            operator_id="OP-DIAGNOSE",
            task_name="diagnose_robot",
            effects={
                "diagnostics_completed": True,
            },
            estimated_cost=8,
            estimated_duration_seconds=90,
        ),
        access=access,
    )

    planner.register_method(
        HTNMethod(
            method_id="METHOD-HEALTH",
            name="Robot Health Workflow",
            compound_task_name="robot_health_workflow",
            subtasks=(
                inspect,
                diagnose,
            ),
            priority=10,
        ),
        access=access,
    )

    request = HierarchicalPlanningRequest(
        request_id=generate_id(),
        tenant_id="tenant_alpha",
        root_task=root,
        initial_state={},
        robot_states=(
            robot,
        ),
    )

    response = planner.plan(
        request,
        access=access,
    )

    print(
        planner.export_response_json(
            response
        )
    )

    if response.plans:
        plan = response.plans[0]

        print(
            "\nBEHAVIOR TREE PAYLOAD\n"
        )

        print(
            json.dumps(
                planner.behavior_tree_payload(
                    plan
                ),
                indent=2,
            )
        )

        print(
            "\nACTION SCHEDULER PAYLOAD\n"
        )

        print(
            json.dumps(
                planner.action_scheduler_payload(
                    plan
                ),
                indent=2,
            )
        )

    print(
        "\nRUNNING UNIT TESTS\n"
    )

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestRoboticsHierarchicalPlanner
    )

    unittest.TextTestRunner(
        verbosity=2
    ).run(suite)


if __name__ == "__main__":
    main()
"""Replanning module.

TODO: implement and test; no real robot control is enabled.
"""

    # backend/app/robotics/planning/replanning.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================

Module: replanning.py

Adaptive, safety-aware robotic task replanning.

Features:
    - Goal-preserving replanning
    - Failed-task recovery
    - Robot fault and health monitoring
    - Low-battery replanning
    - Dynamic obstacle response
    - Resource availability changes
    - Robot capability changes
    - Geofence and workspace restrictions
    - Task dependency validation
    - Alternative robot selection
    - Alternative task strategy selection
    - Remaining-plan optimization
    - Cost, duration, and energy estimation
    - Human approval requirements
    - Risk-based plan selection
    - Plan versioning and revision history
    - Multi-tenant access control
    - Replanning audit events
    - Execution monitor integration
    - Goal manager integration
    - Hierarchical planner integration
    - Plan validator integration
    - Action scheduler integration
    - JSON export
    - Unit tests

IMPORTANT:
    Simulation and planning advisory only.

    This module NEVER sends commands to physical robots.
    A replanned action requires independent runtime safety
    checks and authenticated authorization before any
    physical execution.

Python: 3.11+
Dependencies: Standard library only
"""

from __future__ import annotations

import copy
import json
import math
import threading
import unittest
import uuid

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class ReplanningError(Exception):
    """Base replanning exception."""


class ReplanningValidationError(ReplanningError):
    """Invalid replanning input."""


class ReplanningAuthorizationError(ReplanningError):
    """Unauthorized operation."""


class ReplanningNotFoundError(ReplanningError):
    """Requested plan was not found."""


class ReplanningConflictError(ReplanningError):
    """Plan version conflict."""


class ReplanningSafetyError(ReplanningError):
    """Safety-related replanning rejection."""


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


def validate_id(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise ReplanningValidationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > 128:
        raise ReplanningValidationError(
            f"{name} must contain 1-128 characters"
        )

    return value


def validate_number(
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
        raise ReplanningValidationError(
            f"{name} must be finite and >= {minimum}"
        )

    return float(value)


def normalize_time(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ReplanningValidationError(
            "Timestamp must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return {
            key: to_jsonable(getattr(value, key))
            for key in value.__dataclass_fields__
        }

    if isinstance(value, Mapping):
        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in value]

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, (str, int, bool)) or value is None:
        return value

    return repr(value)


# ============================================================
# ENUMERATIONS
# ============================================================

class ReplanningTrigger(str, Enum):
    TASK_FAILURE = "task_failure"
    ROBOT_FAULT = "robot_fault"
    ROBOT_UNAVAILABLE = "robot_unavailable"
    LOW_BATTERY = "low_battery"
    OBSTACLE_DETECTED = "obstacle_detected"
    COLLISION_RISK = "collision_risk"
    HUMAN_PROXIMITY = "human_proximity"
    SAFETY_VIOLATION = "safety_violation"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    CAPABILITY_CHANGE = "capability_change"
    ENVIRONMENT_CHANGE = "environment_change"
    DEADLINE_RISK = "deadline_risk"
    COST_OVERRUN = "cost_overrun"
    GOAL_CHANGE = "goal_change"
    MANUAL_REQUEST = "manual_request"


class ReplanningStatus(str, Enum):
    SUCCESS = "success"
    NO_CHANGE = "no_change"
    BLOCKED = "blocked"
    FAILED = "failed"


class TaskExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class RobotSafetyState(str, Enum):
    SAFE = "safe"
    CAUTION = "caution"
    UNSAFE = "unsafe"
    EMERGENCY_STOP = "emergency_stop"
    UNKNOWN = "unknown"


class RobotOperationalState(str, Enum):
    IDLE = "idle"
    READY = "ready"
    BUSY = "busy"
    FAULT = "fault"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class ReplanningStrategy(str, Enum):
    RETRY = "retry"
    REASSIGN_ROBOT = "reassign_robot"
    SUBSTITUTE_TASK = "substitute_task"
    SKIP_OPTIONAL = "skip_optional"
    PRESERVE_REMAINING = "preserve_remaining"
    STOP_AND_REVIEW = "stop_and_review"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DecisionType(str, Enum):
    SIMULATION_READY = "simulation_ready"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    SAFE_STOP_RECOMMENDED = "safe_stop_recommended"
    NO_FEASIBLE_PLAN = "no_feasible_plan"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ReplanningConfig:
    maximum_tasks: int = 1000
    maximum_alternatives: int = 20
    maximum_plan_versions: int = 100

    minimum_battery_percentage: float = 20.0
    maximum_robot_state_age_seconds: float = 30.0

    maximum_plan_cost: float = 1_000_000.0
    maximum_plan_duration_seconds: float = 86400.0
    maximum_plan_energy_wh: float = 100_000.0

    cost_weight: float = 1.0
    duration_weight: float = 0.1
    energy_weight: float = 0.05
    risk_weight: float = 10.0

    require_safe_robot_state: bool = True
    require_fresh_robot_state: bool = True
    require_human_review_for_motion_changes: bool = True
    require_human_review_for_high_risk: bool = True

    preserve_completed_tasks: bool = True
    allow_optional_task_skipping: bool = True
    allow_robot_reassignment: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_tasks",
            "maximum_alternatives",
            "maximum_plan_versions",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ReplanningValidationError(
                    f"{name} must be a positive integer"
                )

        for name in (
            "minimum_battery_percentage",
            "maximum_robot_state_age_seconds",
            "maximum_plan_cost",
            "maximum_plan_duration_seconds",
            "maximum_plan_energy_wh",
            "cost_weight",
            "duration_weight",
            "energy_weight",
            "risk_weight",
        ):
            validate_number(
                getattr(self, name),
                name,
            )

        if self.minimum_battery_percentage > 100:
            raise ReplanningValidationError(
                "Battery threshold cannot exceed 100%"
            )


# ============================================================
# ACCESS CONTEXT
# ============================================================

@dataclass(frozen=True)
class ReplanningAccessContext:
    tenant_id: str
    user_id: str

    allowed_robot_ids: frozenset[str] = frozenset()

    can_replan: bool = False
    can_view: bool = False
    can_register_plan: bool = False

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.user_id, "user_id")


# ============================================================
# TASK MODEL
# ============================================================

@dataclass(frozen=True)
class ReplanningTask:
    task_id: str
    name: str
    robot_id: str

    category: str = "custom"

    status: TaskExecutionStatus = (
        TaskExecutionStatus.PENDING
    )

    dependencies: tuple[str, ...] = ()

    required_capabilities: frozenset[str] = frozenset()
    required_resources: frozenset[str] = frozenset()

    required_zone: str | None = None
    required_workspace: str | None = None

    estimated_cost: float = 0.0
    estimated_duration_seconds: float = 0.0
    estimated_energy_wh: float = 0.0

    risk_level: RiskLevel = RiskLevel.LOW

    optional: bool = False
    requires_human_approval: bool = False

    alternative_task_names: tuple[str, ...] = ()

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_id(self.task_id, "task_id")
        validate_id(self.name, "name")
        validate_id(self.robot_id, "robot_id")

        for name in (
            "estimated_cost",
            "estimated_duration_seconds",
            "estimated_energy_wh",
        ):
            validate_number(
                getattr(self, name),
                name,
            )

        if len(set(self.dependencies)) != len(
            self.dependencies
        ):
            raise ReplanningValidationError(
                "Duplicate task dependencies"
            )


# ============================================================
# PLAN MODEL
# ============================================================

@dataclass(frozen=True)
class ReplanningPlan:
    plan_id: str
    tenant_id: str
    goal_id: str

    version: int

    tasks: tuple[ReplanningTask, ...]

    parent_version: int | None = None

    created_at: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_id(self.plan_id, "plan_id")
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.goal_id, "goal_id")

        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, int)
            or self.version < 1
        ):
            raise ReplanningValidationError(
                "Plan version must be positive"
            )

        if not self.tasks:
            raise ReplanningValidationError(
                "Plan must contain tasks"
            )

        normalize_time(self.created_at)


# ============================================================
# ROBOT STATE
# ============================================================

@dataclass(frozen=True)
class ReplanningRobotState:
    tenant_id: str
    robot_id: str

    capabilities: frozenset[str]
    available_resources: frozenset[str]

    allowed_zones: frozenset[str]
    allowed_workspaces: frozenset[str]

    battery_percentage: float

    safety_state: RobotSafetyState
    operational_state: RobotOperationalState

    active_faults: tuple[str, ...] = ()
    emergency_stop_active: bool = False

    estimated_energy_budget_wh: float | None = None

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")

        validate_number(
            self.battery_percentage,
            "battery_percentage",
        )

        if self.battery_percentage > 100:
            raise ReplanningValidationError(
                "Battery percentage exceeds 100%"
            )

        if self.estimated_energy_budget_wh is not None:
            validate_number(
                self.estimated_energy_budget_wh,
                "estimated_energy_budget_wh",
            )

        normalize_time(self.timestamp)


# ============================================================
# ENVIRONMENT SNAPSHOT
# ============================================================

@dataclass(frozen=True)
class ReplanningEnvironment:
    blocked_zones: frozenset[str] = frozenset()
    blocked_workspaces: frozenset[str] = frozenset()
    unavailable_resources: frozenset[str] = frozenset()

    collision_risk_active: bool = False
    human_proximity_alert: bool = False

    emergency_active: bool = False

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        normalize_time(self.timestamp)


# ============================================================
# REPLANNING REQUEST
# ============================================================

@dataclass(frozen=True)
class ReplanningRequest:
    request_id: str

    tenant_id: str
    plan_id: str

    expected_version: int

    trigger: ReplanningTrigger

    robot_states: tuple[ReplanningRobotState, ...]

    environment: ReplanningEnvironment

    failed_task_id: str | None = None

    blocked_task_ids: frozenset[str] = frozenset()

    requested_at: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_id(self.request_id, "request_id")
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.plan_id, "plan_id")

        if self.expected_version < 1:
            raise ReplanningValidationError(
                "expected_version must be positive"
            )

        normalize_time(self.requested_at)


# ============================================================
# REPLANNING RESULT
# ============================================================

@dataclass(frozen=True)
class ReplanningCandidate:
    candidate_id: str

    strategy: ReplanningStrategy

    tasks: tuple[ReplanningTask, ...]

    total_cost: float
    total_duration_seconds: float
    total_energy_wh: float

    maximum_risk: RiskLevel

    score: float

    requires_human_review: bool

    explanation: str

    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReplanningResult:
    request_id: str

    plan_id: str
    tenant_id: str

    previous_version: int

    status: ReplanningStatus

    decision: DecisionType

    new_plan: ReplanningPlan | None

    selected_candidate: ReplanningCandidate | None

    alternatives: tuple[ReplanningCandidate, ...]

    reasons: tuple[str, ...]

    created_at: datetime = field(
        default_factory=utc_now
    )

    simulation_only: bool = True

    hardware_execution_authorized: bool = False


# ============================================================
# RISK SCORING
# ============================================================

class ReplanningRiskScorer:

    RISK_ORDER = {
        RiskLevel.LOW: 0,
        RiskLevel.MEDIUM: 1,
        RiskLevel.HIGH: 3,
        RiskLevel.CRITICAL: 10,
    }

    @classmethod
    def maximum_risk(
        cls,
        tasks: Sequence[ReplanningTask],
    ) -> RiskLevel:
        return max(
            (task.risk_level for task in tasks),
            default=RiskLevel.LOW,
            key=lambda risk: cls.RISK_ORDER[risk],
        )


# ============================================================
# TASK DEPENDENCY VALIDATOR
# ============================================================

class ReplanningDependencyValidator:

    @staticmethod
    def validate(
        tasks: Sequence[ReplanningTask],
    ) -> tuple[bool, tuple[str, ...]]:
        issues = []

        task_map = {
            task.task_id: task
            for task in tasks
        }

        if len(task_map) != len(tasks):
            issues.append(
                "Duplicate task IDs"
            )

        positions = {
            task.task_id: index
            for index, task in enumerate(tasks)
        }

        for task in tasks:
            for dependency in task.dependencies:
                if dependency not in task_map:
                    issues.append(
                        f"Unknown dependency {dependency}"
                    )

                elif positions[dependency] >= positions[task.task_id]:
                    issues.append(
                        f"Dependency ordering violation "
                        f"for {task.task_id}"
                    )

        return (
            not issues,
            tuple(issues),
        )


# ============================================================
# ROBOT ELIGIBILITY
# ============================================================

class ReplanningRobotEligibility:

    def __init__(
        self,
        config: ReplanningConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        task: ReplanningTask,
        robot: ReplanningRobotState,
        environment: ReplanningEnvironment,
        now: datetime,
    ) -> tuple[bool, tuple[str, ...]]:
        reasons = []

        if robot.operational_state not in (
            RobotOperationalState.READY,
            RobotOperationalState.IDLE,
        ):
            reasons.append(
                "Robot is not available for assignment"
            )

        if self.config.require_safe_robot_state:
            if robot.safety_state != RobotSafetyState.SAFE:
                reasons.append(
                    "Robot safety state is not SAFE"
                )

        if robot.emergency_stop_active:
            reasons.append(
                "Emergency stop active"
            )

        if robot.active_faults:
            reasons.append(
                "Robot has active faults"
            )

        if (
            robot.battery_percentage
            < self.config.minimum_battery_percentage
        ):
            reasons.append(
                "Battery below configured threshold"
            )

        if self.config.require_fresh_robot_state:
            age = (
                normalize_time(now)
                - normalize_time(robot.timestamp)
            ).total_seconds()

            if (
                age < 0
                or age > self.config.maximum_robot_state_age_seconds
            ):
                reasons.append(
                    "Robot telemetry is stale"
                )

        if not task.required_capabilities.issubset(
            robot.capabilities
        ):
            reasons.append(
                "Required robot capabilities unavailable"
            )

        if not task.required_resources.issubset(
            robot.available_resources
        ):
            reasons.append(
                "Robot resources unavailable"
            )

        if (
            task.required_resources
            & environment.unavailable_resources
        ):
            reasons.append(
                "Environment resources unavailable"
            )

        if task.required_zone is not None:
            if task.required_zone not in robot.allowed_zones:
                reasons.append(
                    "Robot not authorized for required zone"
                )

            if task.required_zone in environment.blocked_zones:
                reasons.append(
                    "Required zone is blocked"
                )

        if task.required_workspace is not None:
            if (
                task.required_workspace
                not in robot.allowed_workspaces
            ):
                reasons.append(
                    "Robot not authorized for workspace"
                )

            if (
                task.required_workspace
                in environment.blocked_workspaces
            ):
                reasons.append(
                    "Required workspace is blocked"
                )

        if robot.estimated_energy_budget_wh is not None:
            if (
                task.estimated_energy_wh
                > robot.estimated_energy_budget_wh
            ):
                reasons.append(
                    "Task energy exceeds robot budget"
                )

        if environment.emergency_active:
            reasons.append(
                "Environment emergency active"
            )

        if environment.collision_risk_active:
            reasons.append(
                "Collision risk active"
            )

        if environment.human_proximity_alert:
            reasons.append(
                "Human proximity alert active"
            )

        return (
            not reasons,
            tuple(reasons),
        )


# ============================================================
# CANDIDATE EVALUATOR
# ============================================================

class ReplanningCandidateEvaluator:

    def __init__(
        self,
        config: ReplanningConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        tasks: Sequence[ReplanningTask],
        strategy: ReplanningStrategy,
        explanation: str,
        warnings: Sequence[str] = (),
    ) -> ReplanningCandidate:
        remaining = [
            task
            for task in tasks
            if task.status not in (
                TaskExecutionStatus.COMPLETED,
                TaskExecutionStatus.CANCELLED,
            )
        ]

        total_cost = sum(
            task.estimated_cost
            for task in remaining
        )

        total_duration = sum(
            task.estimated_duration_seconds
            for task in remaining
        )

        total_energy = sum(
            task.estimated_energy_wh
            for task in remaining
        )

        risk = ReplanningRiskScorer.maximum_risk(
            remaining
        )

        score = (
            self.config.cost_weight * total_cost
            + self.config.duration_weight * total_duration
            + self.config.energy_weight * total_energy
            + self.config.risk_weight
            * ReplanningRiskScorer.RISK_ORDER[risk]
        )

        changed_motion = any(
            task.category in (
                "navigation",
                "manipulation",
                "pick",
                "place",
                "repair",
            )
            and task.status != TaskExecutionStatus.COMPLETED
            for task in remaining
        )

        requires_review = (
            (
                self.config.require_human_review_for_motion_changes
                and changed_motion
            )
            or (
                self.config.require_human_review_for_high_risk
                and risk in (
                    RiskLevel.HIGH,
                    RiskLevel.CRITICAL,
                )
            )
            or any(
                task.requires_human_approval
                for task in remaining
            )
        )

        return ReplanningCandidate(
            candidate_id=new_id(),
            strategy=strategy,
            tasks=tuple(tasks),
            total_cost=total_cost,
            total_duration_seconds=total_duration,
            total_energy_wh=total_energy,
            maximum_risk=risk,
            score=score,
            requires_human_review=requires_review,
            explanation=explanation,
            warnings=tuple(warnings),
        )


# ============================================================
# MAIN REPLANNING ENGINE
# ============================================================

class RoboticsReplanningEngine:
    """
    Stateful, multi-tenant replanning service.

    Replanning generates advisory plans only.

    Completed tasks are preserved.
    Running tasks cannot be silently reassigned.
    Safety incidents fail closed.
    """

    SAFETY_TRIGGERS = frozenset({
        ReplanningTrigger.COLLISION_RISK,
        ReplanningTrigger.HUMAN_PROXIMITY,
        ReplanningTrigger.SAFETY_VIOLATION,
        ReplanningTrigger.OBSTACLE_DETECTED,
    })

    def __init__(
        self,
        config: ReplanningConfig | None = None,
    ) -> None:
        self.config = config or ReplanningConfig()

        self._plans: dict[
            tuple[str, str],
            ReplanningPlan,
        ] = {}

        self._history: dict[
            tuple[str, str],
            list[ReplanningPlan],
        ] = {}

        self._audit: list[dict[str, Any]] = []

        self._lock = threading.RLock()

        self._eligibility = ReplanningRobotEligibility(
            self.config
        )

        self._evaluator = ReplanningCandidateEvaluator(
            self.config
        )

    # --------------------------------------------------------
    # AUTHORIZATION
    # --------------------------------------------------------

    @staticmethod
    def _require(
        access: ReplanningAccessContext,
        permission: str,
    ) -> None:
        if not getattr(access, permission, False):
            raise ReplanningAuthorizationError(
                f"Missing permission: {permission}"
            )

    @staticmethod
    def _check_tenant(
        tenant_id: str,
        access: ReplanningAccessContext,
    ) -> None:
        if tenant_id != access.tenant_id:
            raise ReplanningAuthorizationError(
                "Cross-tenant operation denied"
            )

    @staticmethod
    def _check_robot_access(
        tasks: Sequence[ReplanningTask],
        access: ReplanningAccessContext,
    ) -> None:
        for task in tasks:
            if task.robot_id not in access.allowed_robot_ids:
                raise ReplanningAuthorizationError(
                    f"Robot access denied: {task.robot_id}"
                )

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit_event(
        self,
        access: ReplanningAccessContext,
        action: str,
        details: Mapping[str, Any],
    ) -> None:
        self._audit.append({
            "event_id": new_id(),
            "tenant_id": access.tenant_id,
            "actor": access.user_id,
            "action": action,
            "details": to_jsonable(details),
            "timestamp": utc_now().isoformat(),
        })

    # --------------------------------------------------------
    # REGISTER PLAN
    # --------------------------------------------------------

    def register_plan(
        self,
        plan: ReplanningPlan,
        *,
        access: ReplanningAccessContext,
    ) -> None:
        self._require(
            access,
            "can_register_plan",
        )

        self._check_tenant(
            plan.tenant_id,
            access,
        )

        self._check_robot_access(
            plan.tasks,
            access,
        )

        if len(plan.tasks) > self.config.maximum_tasks:
            raise ReplanningValidationError(
                "Plan exceeds maximum task count"
            )

        valid, issues = ReplanningDependencyValidator.validate(
            plan.tasks
        )

        if not valid:
            raise ReplanningValidationError(
                "; ".join(issues)
            )

        key = (
            plan.tenant_id,
            plan.plan_id,
        )

        with self._lock:
            if key in self._plans:
                raise ReplanningConflictError(
                    "Plan already registered"
                )

            self._plans[key] = copy.deepcopy(plan)
            self._history[key] = [
                copy.deepcopy(plan)
            ]

            self._audit_event(
                access,
                "plan_registered",
                {
                    "plan_id": plan.plan_id,
                    "version": plan.version,
                },
            )

    # --------------------------------------------------------
    # GET PLAN
    # --------------------------------------------------------

    def get_plan(
        self,
        plan_id: str,
        *,
        access: ReplanningAccessContext,
    ) -> ReplanningPlan:
        self._require(
            access,
            "can_view",
        )

        key = (
            access.tenant_id,
            plan_id,
        )

        with self._lock:
            plan = self._plans.get(key)

            if plan is None:
                raise ReplanningNotFoundError(
                    "Plan not found"
                )

            self._check_robot_access(
                plan.tasks,
                access,
            )

            return copy.deepcopy(plan)

    # --------------------------------------------------------
    # SAFE STOP RESULT
    # --------------------------------------------------------

    @staticmethod
    def _blocked_result(
        request: ReplanningRequest,
        reasons: Sequence[str],
    ) -> ReplanningResult:
        return ReplanningResult(
            request_id=request.request_id,
            plan_id=request.plan_id,
            tenant_id=request.tenant_id,
            previous_version=request.expected_version,
            status=ReplanningStatus.BLOCKED,
            decision=DecisionType.SAFE_STOP_RECOMMENDED,
            new_plan=None,
            selected_candidate=None,
            alternatives=(),
            reasons=tuple(reasons),
        )

    # --------------------------------------------------------
    # SAFETY GATE
    # --------------------------------------------------------

    def _safety_gate(
        self,
        request: ReplanningRequest,
    ) -> tuple[bool, tuple[str, ...]]:
        reasons = []

        if request.environment.emergency_active:
            reasons.append(
                "Environment emergency active"
            )

        if request.environment.collision_risk_active:
            reasons.append(
                "Collision risk detected"
            )

        if request.environment.human_proximity_alert:
            reasons.append(
                "Human proximity safety alert"
            )

        if request.trigger in self.SAFETY_TRIGGERS:
            reasons.append(
                "Safety-triggered replanning requires "
                "independent review"
            )

        age = (
            normalize_time(request.requested_at)
            - normalize_time(request.environment.timestamp)
        ).total_seconds()

        if (
            age < 0
            or age > self.config.maximum_robot_state_age_seconds
        ):
            reasons.append(
                "Environment snapshot stale or future-dated"
            )

        return (
            not reasons,
            tuple(reasons),
        )

    # --------------------------------------------------------
    # ROBOT SELECTION
    # --------------------------------------------------------

    def _find_alternative_robots(
        self,
        task: ReplanningTask,
        request: ReplanningRequest,
        access: ReplanningAccessContext,
    ) -> list[ReplanningRobotState]:
        candidates = []

        for robot in request.robot_states:
            if robot.tenant_id != request.tenant_id:
                continue

            if robot.robot_id == task.robot_id:
                continue

            if robot.robot_id not in access.allowed_robot_ids:
                continue

            eligible, _ = self._eligibility.evaluate(
                task,
                robot,
                request.environment,
                request.requested_at,
            )

            if eligible:
                candidates.append(robot)

        candidates.sort(
            key=lambda robot: (
                -robot.battery_percentage,
                robot.robot_id,
            )
        )

        return candidates

    # --------------------------------------------------------
    # TASK FEASIBILITY
    # --------------------------------------------------------

    def _task_feasible(
        self,
        task: ReplanningTask,
        request: ReplanningRequest,
        robots: Mapping[str, ReplanningRobotState],
    ) -> bool:
        robot = robots.get(task.robot_id)

        if robot is None:
            return False

        feasible, _ = self._eligibility.evaluate(
            task,
            robot,
            request.environment,
            request.requested_at,
        )

        return feasible

    # --------------------------------------------------------
    # CANDIDATE VALIDATION
    # --------------------------------------------------------

    def _validate_candidate(
        self,
        candidate: ReplanningCandidate,
        original: ReplanningPlan,
        request: ReplanningRequest,
        access: ReplanningAccessContext,
        robots: Mapping[str, ReplanningRobotState],
    ) -> tuple[bool, tuple[str, ...]]:
        issues = []

        if len(candidate.tasks) > self.config.maximum_tasks:
            issues.append(
                "Candidate exceeds maximum task count"
            )

        valid_dependencies, dependency_issues = (
            ReplanningDependencyValidator.validate(
                candidate.tasks
            )
        )

        if not valid_dependencies:
            issues.extend(dependency_issues)

        original_tasks = {
            task.task_id: task
            for task in original.tasks
        }

        candidate_tasks = {
            task.task_id: task
            for task in candidate.tasks
        }

        if set(original_tasks) != set(candidate_tasks):
            issues.append(
                "Candidate changes task identity set"
            )

        for task_id, original_task in original_tasks.items():
            candidate_task = candidate_tasks.get(task_id)

            if candidate_task is None:
                continue

            if original_task.status == TaskExecutionStatus.COMPLETED:
                if candidate_task != original_task:
                    issues.append(
                        f"Completed task modified: {task_id}"
                    )

            if original_task.status == TaskExecutionStatus.RUNNING:
                if candidate_task != original_task:
                    issues.append(
                        f"Running task modified: {task_id}"
                    )

            if (
                original_task.status
                not in (
                    TaskExecutionStatus.COMPLETED,
                    TaskExecutionStatus.RUNNING,
                )
                and candidate_task.status
                not in (
                    TaskExecutionStatus.CANCELLED,
                    TaskExecutionStatus.COMPLETED,
                )
            ):
                if not self._task_feasible(
                    candidate_task,
                    request,
                    robots,
                ):
                    issues.append(
                        f"Task infeasible: {task_id}"
                    )

            if (
                candidate_task.robot_id
                not in access.allowed_robot_ids
            ):
                issues.append(
                    f"Unauthorized robot assignment: {task_id}"
                )

        if candidate.total_cost > self.config.maximum_plan_cost:
            issues.append(
                "Plan cost budget exceeded"
            )

        if (
            candidate.total_duration_seconds
            > self.config.maximum_plan_duration_seconds
        ):
            issues.append(
                "Plan duration budget exceeded"
            )

        if (
            candidate.total_energy_wh
            > self.config.maximum_plan_energy_wh
        ):
            issues.append(
                "Plan energy budget exceeded"
            )

        return (
            not issues,
            tuple(issues),
        )

    # --------------------------------------------------------
    # REPLAN
    # --------------------------------------------------------

    def replan(
        self,
        request: ReplanningRequest,
        *,
        access: ReplanningAccessContext,
    ) -> ReplanningResult:
        self._require(
            access,
            "can_replan",
        )

        self._check_tenant(
            request.tenant_id,
            access,
        )

        key = (
            request.tenant_id,
            request.plan_id,
        )

        with self._lock:
            original = self._plans.get(key)

            if original is None:
                raise ReplanningNotFoundError(
                    "Plan not found"
                )

            if original.version != request.expected_version:
                raise ReplanningConflictError(
                    "Plan version changed; refresh before replanning"
                )

            self._check_robot_access(
                original.tasks,
                access,
            )

            safety_ok, safety_reasons = self._safety_gate(
                request
            )

            if not safety_ok:
                result = self._blocked_result(
                    request,
                    safety_reasons,
                )

                self._audit_event(
                    access,
                    "replanning_blocked",
                    {
                        "plan_id": request.plan_id,
                        "reasons": safety_reasons,
                    },
                )

                return result

            robots = {
                robot.robot_id: robot
                for robot in request.robot_states
                if robot.tenant_id == request.tenant_id
            }

            if len(robots) != len([
                robot
                for robot in request.robot_states
                if robot.tenant_id == request.tenant_id
            ]):
                raise ReplanningValidationError(
                    "Duplicate robot snapshots"
                )

            candidates: list[ReplanningCandidate] = []

            base_tasks = list(original.tasks)

            affected_ids = set(
                request.blocked_task_ids
            )

            if request.failed_task_id is not None:
                affected_ids.add(
                    request.failed_task_id
                )

            known_task_ids = {
                task.task_id
                for task in original.tasks
            }

            if not affected_ids.issubset(known_task_ids):
                raise ReplanningValidationError(
                    "Unknown affected task ID"
                )

            for task in base_tasks:
                if task.status in (
                    TaskExecutionStatus.COMPLETED,
                    TaskExecutionStatus.CANCELLED,
                ):
                    continue

                if task.status == TaskExecutionStatus.RUNNING:
                    continue

                if not self._task_feasible(
                    task,
                    request,
                    robots,
                ):
                    affected_ids.add(task.task_id)

            if not affected_ids:
                result = ReplanningResult(
                    request_id=request.request_id,
                    plan_id=request.plan_id,
                    tenant_id=request.tenant_id,
                    previous_version=original.version,
                    status=ReplanningStatus.NO_CHANGE,
                    decision=DecisionType.SIMULATION_READY,
                    new_plan=None,
                    selected_candidate=None,
                    alternatives=(),
                    reasons=(
                        "No infeasible pending tasks detected",
                    ),
                )

                self._audit_event(
                    access,
                    "replanning_no_change",
                    {
                        "plan_id": request.plan_id,
                    },
                )

                return result

            for index, task in enumerate(base_tasks):
                if task.task_id not in affected_ids:
                    continue

                if task.status in (
                    TaskExecutionStatus.COMPLETED,
                    TaskExecutionStatus.RUNNING,
                ):
                    continue

                # --------------------------------------------
                # Strategy 1: Retry on the same robot
                # --------------------------------------------

                if self._task_feasible(
                    task,
                    request,
                    robots,
                ):
                    retry_tasks = list(base_tasks)

                    retry_tasks[index] = replace(
                        task,
                        status=TaskExecutionStatus.PENDING,
                    )

                    candidates.append(
                        self._evaluator.evaluate(
                            retry_tasks,
                            ReplanningStrategy.RETRY,
                            (
                                f"Retry task {task.task_id} "
                                "in simulation"
                            ),
                        )
                    )

                # --------------------------------------------
                # Strategy 2: Reassign to another robot
                # --------------------------------------------

                if self.config.allow_robot_reassignment:
                    alternative_robots = (
                        self._find_alternative_robots(
                            task,
                            request,
                            access,
                        )
                    )

                    for robot in alternative_robots:
                        reassigned = list(base_tasks)

                        reassigned[index] = replace(
                            task,
                            robot_id=robot.robot_id,
                            status=TaskExecutionStatus.PENDING,
                        )

                        candidates.append(
                            self._evaluator.evaluate(
                                reassigned,
                                ReplanningStrategy.REASSIGN_ROBOT,
                                (
                                    f"Reassign {task.task_id} "
                                    f"to {robot.robot_id}"
                                ),
                            )
                        )

                # --------------------------------------------
                # Strategy 3: Substitute task strategy
                # --------------------------------------------

                for alternative_name in task.alternative_task_names:
                    substituted = list(base_tasks)

                    substituted[index] = replace(
                        task,
                        name=alternative_name,
                        status=TaskExecutionStatus.PENDING,
                    )

                    candidates.append(
                        self._evaluator.evaluate(
                            substituted,
                            ReplanningStrategy.SUBSTITUTE_TASK,
                            (
                                f"Use alternative strategy "
                                f"{alternative_name}"
                            ),
                            warnings=(
                                "Alternative strategy requires "
                                "independent capability and "
                                "operator validation",
                            ),
                        )
                    )

                # --------------------------------------------
                # Strategy 4: Skip optional task
                # --------------------------------------------

                if (
                    task.optional
                    and self.config.allow_optional_task_skipping
                ):
                    skipped = list(base_tasks)

                    skipped[index] = replace(
                        task,
                        status=TaskExecutionStatus.CANCELLED,
                    )

                    candidates.append(
                        self._evaluator.evaluate(
                            skipped,
                            ReplanningStrategy.SKIP_OPTIONAL,
                            (
                                f"Skip optional task {task.task_id}"
                            ),
                        )
                    )

            valid_candidates = []

            for candidate in candidates:
                valid, _ = self._validate_candidate(
                    candidate,
                    original,
                    request,
                    access,
                    robots,
                )

                if valid:
                    valid_candidates.append(candidate)

            # Reject candidates that leave any affected
            # task unresolved.
            filtered_candidates = []

            for candidate in valid_candidates:
                candidate_map = {
                    task.task_id: task
                    for task in candidate.tasks
                }

                resolved = True

                for task_id in affected_ids:
                    task = candidate_map[task_id]

                    if task.status in (
                        TaskExecutionStatus.COMPLETED,
                        TaskExecutionStatus.CANCELLED,
                    ):
                        continue

                    if not self._task_feasible(
                        task,
                        request,
                        robots,
                    ):
                        resolved = False
                        break

                if resolved:
                    filtered_candidates.append(candidate)

            filtered_candidates.sort(
                key=lambda candidate: (
                    candidate.score,
                    candidate.strategy.value,
                    candidate.candidate_id,
                )
            )

            filtered_candidates = filtered_candidates[
                :self.config.maximum_alternatives
            ]

            if not filtered_candidates:
                result = ReplanningResult(
                    request_id=request.request_id,
                    plan_id=request.plan_id,
                    tenant_id=request.tenant_id,
                    previous_version=original.version,
                    status=ReplanningStatus.FAILED,
                    decision=DecisionType.NO_FEASIBLE_PLAN,
                    new_plan=None,
                    selected_candidate=None,
                    alternatives=(),
                    reasons=(
                        "No feasible replacement plan found",
                        "Keep affected actions blocked",
                        "Request human review",
                    ),
                )

                self._audit_event(
                    access,
                    "replanning_failed",
                    {
                        "plan_id": request.plan_id,
                        "affected_tasks": sorted(affected_ids),
                    },
                )

                return result

            selected = filtered_candidates[0]

            history = self._history[key]

            if len(history) >= self.config.maximum_plan_versions:
                raise ReplanningConflictError(
                    "Maximum plan version history reached"
                )

            new_plan = ReplanningPlan(
                plan_id=original.plan_id,
                tenant_id=original.tenant_id,
                goal_id=original.goal_id,
                version=original.version + 1,
                parent_version=original.version,
                tasks=selected.tasks,
                metadata={
                    **copy.deepcopy(original.metadata),
                    "replanning_request_id": request.request_id,
                    "replanning_trigger": request.trigger.value,
                    "selected_strategy": selected.strategy.value,
                    "simulation_only": True,
                },
            )

            self._plans[key] = copy.deepcopy(new_plan)

            self._history[key].append(
                copy.deepcopy(new_plan)
            )

            decision = (
                DecisionType.HUMAN_REVIEW_REQUIRED
                if selected.requires_human_review
                else DecisionType.SIMULATION_READY
            )

            result = ReplanningResult(
                request_id=request.request_id,
                plan_id=request.plan_id,
                tenant_id=request.tenant_id,
                previous_version=original.version,
                status=ReplanningStatus.SUCCESS,
                decision=decision,
                new_plan=new_plan,
                selected_candidate=selected,
                alternatives=tuple(filtered_candidates),
                reasons=(
                    selected.explanation,
                ),
            )

            self._audit_event(
                access,
                "replanning_completed",
                {
                    "plan_id": request.plan_id,
                    "old_version": original.version,
                    "new_version": new_plan.version,
                    "strategy": selected.strategy.value,
                    "decision": decision.value,
                },
            )

            return result

    # --------------------------------------------------------
    # PLAN HISTORY
    # --------------------------------------------------------

    def plan_history(
        self,
        plan_id: str,
        *,
        access: ReplanningAccessContext,
    ) -> tuple[ReplanningPlan, ...]:
        self._require(
            access,
            "can_view",
        )

        key = (
            access.tenant_id,
            plan_id,
        )

        with self._lock:
            history = self._history.get(key)

            if history is None:
                raise ReplanningNotFoundError(
                    "Plan history not found"
                )

            for plan in history:
                self._check_robot_access(
                    plan.tasks,
                    access,
                )

            return tuple(
                copy.deepcopy(history)
            )

    # --------------------------------------------------------
    # EXECUTION MONITOR ADAPTER
    # --------------------------------------------------------

    @staticmethod
    def execution_monitor_payload(
        result: ReplanningResult,
    ) -> dict[str, Any]:
        return {
            "plan_id": result.plan_id,
            "replanning_request_id": result.request_id,
            "status": result.status.value,
            "decision": result.decision.value,
            "previous_version": result.previous_version,
            "new_version": (
                result.new_plan.version
                if result.new_plan
                else None
            ),
            "simulation_only": True,
            "hardware_execution_authorized": False,
        }

    # --------------------------------------------------------
    # GOAL MANAGER ADAPTER
    # --------------------------------------------------------

    @staticmethod
    def goal_manager_payload(
        result: ReplanningResult,
    ) -> dict[str, Any]:
        return {
            "goal_id": (
                result.new_plan.goal_id
                if result.new_plan
                else None
            ),
            "plan_id": result.plan_id,
            "replanning_status": result.status.value,
            "human_review_required": (
                result.decision
                == DecisionType.HUMAN_REVIEW_REQUIRED
            ),
            "simulation_only": True,
        }

    # --------------------------------------------------------
    # HIERARCHICAL PLANNER ADAPTER
    # --------------------------------------------------------

    @staticmethod
    def hierarchical_planner_payload(
        result: ReplanningResult,
    ) -> dict[str, Any]:
        if result.new_plan is None:
            return {
                "plan_id": result.plan_id,
                "replanning_available": False,
                "simulation_only": True,
            }

        return {
            "plan_id": result.plan_id,
            "goal_id": result.new_plan.goal_id,
            "version": result.new_plan.version,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "robot_id": task.robot_id,
                    "dependencies": list(task.dependencies),
                    "status": task.status.value,
                }
                for task in result.new_plan.tasks
            ],
            "simulation_only": True,
        }

    # --------------------------------------------------------
    # PLAN VALIDATOR ADAPTER
    # --------------------------------------------------------

    @staticmethod
    def plan_validator_payload(
        result: ReplanningResult,
    ) -> dict[str, Any]:
        return {
            "plan_id": result.plan_id,
            "version": (
                result.new_plan.version
                if result.new_plan
                else None
            ),
            "requires_revalidation": (
                result.new_plan is not None
            ),
            "requires_human_review": (
                result.decision
                == DecisionType.HUMAN_REVIEW_REQUIRED
            ),
            "hardware_execution_authorized": False,
        }

    # --------------------------------------------------------
    # ACTION SCHEDULER ADAPTER
    # --------------------------------------------------------

    @staticmethod
    def action_scheduler_payload(
        result: ReplanningResult,
    ) -> dict[str, Any]:
        if result.new_plan is None:
            return {
                "plan_id": result.plan_id,
                "allow_simulation_scheduling": False,
                "actions": [],
            }

        return {
            "plan_id": result.plan_id,
            "version": result.new_plan.version,
            "allow_simulation_scheduling": False,
            "requires_plan_revalidation": True,
            "actions": [
                {
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "robot_id": task.robot_id,
                    "status": task.status.value,
                    "dependencies": list(task.dependencies),
                }
                for task in result.new_plan.tasks
                if task.status == TaskExecutionStatus.PENDING
            ],
            "simulation_only": True,
            "hardware_execution_authorized": False,
        }

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    def audit_history(
        self,
        *,
        access: ReplanningAccessContext,
    ) -> list[dict[str, Any]]:
        self._require(
            access,
            "can_view",
        )

        with self._lock:
            return copy.deepcopy([
                event
                for event in self._audit
                if event["tenant_id"] == access.tenant_id
            ])

    # --------------------------------------------------------
    # JSON EXPORT
    # --------------------------------------------------------

    @staticmethod
    def export_json(
        result: ReplanningResult,
    ) -> str:
        return json.dumps(
            to_jsonable(result),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestRoboticsReplanningEngine(unittest.TestCase):

    def setUp(self) -> None:
        self.engine = RoboticsReplanningEngine()

        self.access = ReplanningAccessContext(
            tenant_id="tenant_alpha",
            user_id="engineer",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
                "ROBOT-002",
            }),
            can_replan=True,
            can_view=True,
            can_register_plan=True,
        )

        self.task_1 = ReplanningTask(
            task_id="TASK-001",
            name="inspect_asset",
            robot_id="ROBOT-001",
            category="inspection",
            status=TaskExecutionStatus.COMPLETED,
            required_capabilities=frozenset({
                "inspection",
            }),
        )

        self.task_2 = ReplanningTask(
            task_id="TASK-002",
            name="diagnose_asset",
            robot_id="ROBOT-001",
            category="diagnostics",
            status=TaskExecutionStatus.FAILED,
            dependencies=("TASK-001",),
            required_capabilities=frozenset({
                "diagnostics",
            }),
            estimated_cost=10,
            estimated_duration_seconds=30,
            estimated_energy_wh=5,
        )

        self.plan = ReplanningPlan(
            plan_id="PLAN-001",
            tenant_id="tenant_alpha",
            goal_id="GOAL-001",
            version=1,
            tasks=(
                self.task_1,
                self.task_2,
            ),
        )

        self.engine.register_plan(
            self.plan,
            access=self.access,
        )

        self.robot_1 = ReplanningRobotState(
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            capabilities=frozenset({
                "inspection",
            }),
            available_resources=frozenset(),
            allowed_zones=frozenset(),
            allowed_workspaces=frozenset(),
            battery_percentage=90,
            safety_state=RobotSafetyState.SAFE,
            operational_state=RobotOperationalState.READY,
        )

        self.robot_2 = ReplanningRobotState(
            tenant_id="tenant_alpha",
            robot_id="ROBOT-002",
            capabilities=frozenset({
                "diagnostics",
            }),
            available_resources=frozenset(),
            allowed_zones=frozenset(),
            allowed_workspaces=frozenset(),
            battery_percentage=95,
            safety_state=RobotSafetyState.SAFE,
            operational_state=RobotOperationalState.READY,
        )

    def request(
        self,
        environment: ReplanningEnvironment | None = None,
    ) -> ReplanningRequest:
        return ReplanningRequest(
            request_id=new_id(),
            tenant_id="tenant_alpha",
            plan_id="PLAN-001",
            expected_version=1,
            trigger=ReplanningTrigger.TASK_FAILURE,
            robot_states=(
                self.robot_1,
                self.robot_2,
            ),
            environment=(
                environment
                or ReplanningEnvironment()
            ),
            failed_task_id="TASK-002",
        )

    def test_robot_reassignment(self) -> None:
        result = self.engine.replan(
            self.request(),
            access=self.access,
        )

        self.assertEqual(
            result.status,
            ReplanningStatus.SUCCESS,
        )

        self.assertEqual(
            result.new_plan.tasks[1].robot_id,
            "ROBOT-002",
        )

    def test_completed_task_preserved(self) -> None:
        result = self.engine.replan(
            self.request(),
            access=self.access,
        )

        self.assertEqual(
            result.new_plan.tasks[0],
            self.task_1,
        )

    def test_version_increment(self) -> None:
        result = self.engine.replan(
            self.request(),
            access=self.access,
        )

        self.assertEqual(
            result.new_plan.version,
            2,
        )

    def test_collision_risk_blocks(self) -> None:
        result = self.engine.replan(
            self.request(
                ReplanningEnvironment(
                    collision_risk_active=True,
                )
            ),
            access=self.access,
        )

        self.assertEqual(
            result.status,
            ReplanningStatus.BLOCKED,
        )

    def test_version_conflict(self) -> None:
        self.engine.replan(
            self.request(),
            access=self.access,
        )

        with self.assertRaises(
            ReplanningConflictError
        ):
            self.engine.replan(
                self.request(),
                access=self.access,
            )

    def test_tenant_isolation(self) -> None:
        other_access = ReplanningAccessContext(
            tenant_id="tenant_beta",
            user_id="other",
            can_replan=True,
        )

        with self.assertRaises(
            ReplanningAuthorizationError
        ):
            self.engine.replan(
                self.request(),
                access=other_access,
            )

    def test_json_export(self) -> None:
        result = self.engine.replan(
            self.request(),
            access=self.access,
        )

        exported = self.engine.export_json(
            result
        )

        self.assertIn(
            "ROBOT-002",
            exported,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print(
        "\nPHYSICAL AI & AUTONOMOUS ROBOTICS PLATFORM"
    )

    print(
        "Adaptive Replanning Engine\n"
    )

    engine = RoboticsReplanningEngine()

    access = ReplanningAccessContext(
        tenant_id="tenant_alpha",
        user_id="robotics_engineer",
        allowed_robot_ids=frozenset({
            "ROBOT-001",
            "ROBOT-002",
        }),
        can_replan=True,
        can_view=True,
        can_register_plan=True,
    )

    tasks = (
        ReplanningTask(
            task_id="TASK-001",
            name="inspect_robot",
            robot_id="ROBOT-001",
            category="inspection",
            status=TaskExecutionStatus.COMPLETED,
            required_capabilities=frozenset({
                "inspection",
            }),
        ),
        ReplanningTask(
            task_id="TASK-002",
            name="diagnose_robot",
            robot_id="ROBOT-001",
            category="diagnostics",
            status=TaskExecutionStatus.FAILED,
            dependencies=("TASK-001",),
            required_capabilities=frozenset({
                "diagnostics",
            }),
            estimated_cost=15,
            estimated_duration_seconds=90,
            estimated_energy_wh=25,
        ),
    )

    plan = ReplanningPlan(
        plan_id="PLAN-ROBOT-HEALTH",
        tenant_id="tenant_alpha",
        goal_id="GOAL-ROBOT-HEALTH",
        version=1,
        tasks=tasks,
    )

    engine.register_plan(
        plan,
        access=access,
    )

    robot_1 = ReplanningRobotState(
        tenant_id="tenant_alpha",
        robot_id="ROBOT-001",
        capabilities=frozenset({
            "inspection",
        }),
        available_resources=frozenset(),
        allowed_zones=frozenset({
            "ZONE-A",
        }),
        allowed_workspaces=frozenset({
            "WORKSPACE-1",
        }),
        battery_percentage=80,
        safety_state=RobotSafetyState.SAFE,
        operational_state=RobotOperationalState.READY,
    )

    robot_2 = ReplanningRobotState(
        tenant_id="tenant_alpha",
        robot_id="ROBOT-002",
        capabilities=frozenset({
            "diagnostics",
        }),
        available_resources=frozenset(),
        allowed_zones=frozenset({
            "ZONE-A",
        }),
        allowed_workspaces=frozenset({
            "WORKSPACE-1",
        }),
        battery_percentage=95,
        safety_state=RobotSafetyState.SAFE,
        operational_state=RobotOperationalState.READY,
    )

    request = ReplanningRequest(
        request_id=new_id(),
        tenant_id="tenant_alpha",
        plan_id="PLAN-ROBOT-HEALTH",
        expected_version=1,
        trigger=ReplanningTrigger.TASK_FAILURE,
        robot_states=(
            robot_1,
            robot_2,
        ),
        environment=ReplanningEnvironment(),
        failed_task_id="TASK-002",
    )

    result = engine.replan(
        request,
        access=access,
    )

    print("REPLANNING RESULT\n")

    print(
        engine.export_json(result)
    )

    print("\nPLAN VALIDATOR INTEGRATION\n")

    print(
        json.dumps(
            engine.plan_validator_payload(result),
            indent=2,
        )
    )

    print("\nACTION SCHEDULER INTEGRATION\n")

    print(
        json.dumps(
            engine.action_scheduler_payload(result),
            indent=2,
        )
    )

    print("\nRUNNING UNIT TESTS\n")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestRoboticsReplanningEngine
    )

    unittest.TextTestRunner(
        verbosity=2
    ).run(suite)


if __name__ == "__main__":
    main()
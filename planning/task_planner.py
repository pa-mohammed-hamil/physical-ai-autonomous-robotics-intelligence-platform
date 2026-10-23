"""Task Planner module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/planning/task_planner.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================

Module: task_planner.py

Goal-oriented, multi-robot task planning engine.

Features
--------
- Goal decomposition into executable task specifications
- Hierarchical task decomposition (HTN)
- Task dependency resolution
- Topological task ordering
- Robot capability matching
- Multi-robot task assignment
- Robot health and battery checks
- Geofence and workspace validation
- Resource availability validation
- Task preconditions and effects
- Simulated world-state tracking
- Task priority management
- Time, energy, and cost estimation
- Risk-based task planning
- Alternative robot selection
- Deadline validation
- Human-review requirements
- Multi-tenant access control
- Immutable plan versions
- Structured planning reports
- Replanning integration
- Plan validator integration
- Action scheduler integration
- Execution monitor integration
- JSON export
- Unit tests and demonstration

Safety
------
This module produces advisory plans for simulation or dry-run.

It does NOT:
    - Control physical robots
    - Send ROS2 commands
    - Move actuators
    - Bypass emergency stops
    - Authorize real-world execution
    - Certify collision avoidance

All physical execution requires independent, authenticated,
hardware-enforced safety controls.

Python 3.11+
Dependencies: Standard library only.
"""

from __future__ import annotations

import copy
import heapq
import json
import math
import threading
import unittest
import uuid

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class TaskPlannerError(Exception):
    """Base task planner exception."""


class TaskPlannerValidationError(TaskPlannerError):
    """Invalid planning input."""


class TaskPlannerAuthorizationError(TaskPlannerError):
    """Unauthorized planning operation."""


class TaskPlannerNotFoundError(TaskPlannerError):
    """Requested planning object does not exist."""


class TaskPlannerCapacityError(TaskPlannerError):
    """Planning capacity exceeded."""


class TaskPlannerDependencyError(TaskPlannerError):
    """Invalid task dependency graph."""


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
        raise TaskPlannerValidationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > 128:
        raise TaskPlannerValidationError(
            f"{name} must contain 1-128 characters"
        )

    return value


def validate_nonnegative(
    value: Any,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise TaskPlannerValidationError(
            f"{name} must be a finite nonnegative number"
        )

    return float(value)


def aware_datetime(
    value: datetime,
    name: str = "timestamp",
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise TaskPlannerValidationError(
            f"{name} must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return {
            key: json_safe(getattr(value, key))
            for key in value.__dataclass_fields__
        }

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (tuple, list, set, frozenset)):
        return [json_safe(item) for item in value]

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, (str, int, bool)) or value is None:
        return value

    return repr(value)


# ============================================================
# ENUMERATIONS
# ============================================================

class PlanningStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    BLOCKED_SAFETY = "blocked_safety"


class TaskCategory(str, Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PICK = "pick"
    PLACE = "place"
    INSPECTION = "inspection"
    DIAGNOSTICS = "diagnostics"
    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    CALIBRATION = "calibration"
    CHARGING = "charging"
    PERCEPTION = "perception"
    MONITORING = "monitoring"
    DATA_COLLECTION = "data_collection"
    SAFETY_CHECK = "safety_check"
    SIMULATION = "simulation"
    CUSTOM = "custom"


class TaskPriority(int, Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


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


class TaskPlanningMode(str, Enum):
    SIMULATION = "simulation"
    DRY_RUN = "dry_run"


class TaskPlanDecision(str, Enum):
    SIMULATION_ELIGIBLE = "simulation_eligible"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    REJECT = "reject"


class ConditionOperator(str, Enum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    EXISTS = "exists"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class TaskPlannerConfig:
    maximum_tasks: int = 1000
    maximum_hierarchy_depth: int = 20
    maximum_saved_plans: int = 1000

    minimum_battery_percentage: float = 20.0
    maximum_robot_state_age_seconds: float = 30.0
    maximum_environment_age_seconds: float = 30.0

    maximum_plan_duration_seconds: float = 86400.0
    maximum_plan_energy_wh: float = 100000.0
    maximum_plan_cost: float = 1000000.0

    cost_weight: float = 1.0
    duration_weight: float = 0.1
    energy_weight: float = 0.05
    risk_weight: float = 10.0

    require_safe_robot_state: bool = True
    require_fresh_robot_state: bool = True

    require_review_for_motion: bool = True
    require_review_for_high_risk: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_tasks",
            "maximum_hierarchy_depth",
            "maximum_saved_plans",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise TaskPlannerValidationError(
                    f"{name} must be a positive integer"
                )

        for name in (
            "minimum_battery_percentage",
            "maximum_robot_state_age_seconds",
            "maximum_environment_age_seconds",
            "maximum_plan_duration_seconds",
            "maximum_plan_energy_wh",
            "maximum_plan_cost",
            "cost_weight",
            "duration_weight",
            "energy_weight",
            "risk_weight",
        ):
            validate_nonnegative(
                getattr(self, name),
                name,
            )

        if self.minimum_battery_percentage > 100:
            raise TaskPlannerValidationError(
                "Battery threshold exceeds 100%"
            )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class TaskPlannerAccessContext:
    tenant_id: str
    user_id: str

    allowed_robot_ids: frozenset[str] = frozenset()

    can_plan: bool = False
    can_view_plans: bool = False
    can_manage_templates: bool = False

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id, "tenant_id")
        validate_identifier(self.user_id, "user_id")


# ============================================================
# CONDITIONS
# ============================================================

@dataclass(frozen=True)
class TaskCondition:
    key: str
    operator: ConditionOperator
    expected: Any = None

    def __post_init__(self) -> None:
        validate_identifier(self.key, "condition key")


class TaskConditionEvaluator:

    @staticmethod
    def evaluate(
        condition: TaskCondition,
        state: Mapping[str, Any],
    ) -> bool:
        exists = condition.key in state

        if condition.operator == ConditionOperator.EXISTS:
            return exists

        if not exists:
            return False

        actual = state[condition.key]
        expected = condition.expected

        try:
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

        except (TypeError, ValueError):
            return False

        return False


# ============================================================
# TASK SPECIFICATION
# ============================================================

@dataclass(frozen=True)
class TaskSpecification:
    task_id: str
    name: str

    category: TaskCategory

    priority: TaskPriority = TaskPriority.NORMAL
    risk_level: TaskRiskLevel = TaskRiskLevel.LOW

    dependencies: tuple[str, ...] = ()

    required_capabilities: frozenset[str] = frozenset()
    required_resources: frozenset[str] = frozenset()

    preferred_robot_id: str | None = None

    required_zone: str | None = None
    required_workspace: str | None = None

    preconditions: tuple[TaskCondition, ...] = ()
    postconditions: tuple[TaskCondition, ...] = ()

    simulated_effects: dict[str, Any] = field(
        default_factory=dict
    )

    estimated_duration_seconds: float = 0.0
    estimated_energy_wh: float = 0.0
    estimated_cost: float = 0.0

    optional: bool = False
    requires_human_review: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_identifier(self.task_id, "task_id")
        validate_identifier(self.name, "name")

        for name in (
            "estimated_duration_seconds",
            "estimated_energy_wh",
            "estimated_cost",
        ):
            validate_nonnegative(
                getattr(self, name),
                name,
            )

        if len(set(self.dependencies)) != len(
            self.dependencies
        ):
            raise TaskPlannerValidationError(
                "Duplicate dependencies"
            )


# ============================================================
# HIERARCHICAL TASK TEMPLATE
# ============================================================

@dataclass(frozen=True)
class HierarchicalTaskTemplate:
    template_id: str
    name: str

    subtasks: tuple[TaskSpecification, ...]

    description: str = ""

    def __post_init__(self) -> None:
        validate_identifier(
            self.template_id,
            "template_id",
        )

        validate_identifier(
            self.name,
            "name",
        )

        if not self.subtasks:
            raise TaskPlannerValidationError(
                "Template must contain subtasks"
            )


# ============================================================
# GOAL SPECIFICATION
# ============================================================

@dataclass(frozen=True)
class TaskPlanningGoal:
    goal_id: str
    tenant_id: str
    name: str

    tasks: tuple[TaskSpecification, ...]

    initial_world_state: dict[str, Any] = field(
        default_factory=dict
    )

    expected_world_state: dict[str, Any] = field(
        default_factory=dict
    )

    deadline: datetime | None = None

    maximum_cost: float | None = None
    maximum_energy_wh: float | None = None
    maximum_duration_seconds: float | None = None

    mode: TaskPlanningMode = (
        TaskPlanningMode.SIMULATION
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_identifier(self.goal_id, "goal_id")
        validate_identifier(self.tenant_id, "tenant_id")
        validate_identifier(self.name, "name")

        if not self.tasks:
            raise TaskPlannerValidationError(
                "Goal must contain tasks"
            )

        if self.deadline is not None:
            aware_datetime(self.deadline, "deadline")

        for name in (
            "maximum_cost",
            "maximum_energy_wh",
            "maximum_duration_seconds",
        ):
            value = getattr(self, name)

            if value is not None:
                validate_nonnegative(value, name)


# ============================================================
# ROBOT STATE
# ============================================================

@dataclass(frozen=True)
class TaskPlannerRobotState:
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

    energy_budget_wh: float | None = None

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id, "tenant_id")
        validate_identifier(self.robot_id, "robot_id")

        validate_nonnegative(
            self.battery_percentage,
            "battery_percentage",
        )

        if self.battery_percentage > 100:
            raise TaskPlannerValidationError(
                "Battery percentage exceeds 100%"
            )

        if self.energy_budget_wh is not None:
            validate_nonnegative(
                self.energy_budget_wh,
                "energy_budget_wh",
            )

        aware_datetime(self.timestamp)


# ============================================================
# ENVIRONMENT STATE
# ============================================================

@dataclass(frozen=True)
class TaskPlannerEnvironment:
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
        aware_datetime(self.timestamp)


# ============================================================
# PLANNING REQUEST
# ============================================================

@dataclass(frozen=True)
class TaskPlanningRequest:
    request_id: str

    goal: TaskPlanningGoal

    robots: tuple[TaskPlannerRobotState, ...]

    environment: TaskPlannerEnvironment

    requested_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_identifier(self.request_id, "request_id")
        aware_datetime(self.requested_at)


# ============================================================
# PLANNED TASK
# ============================================================

@dataclass(frozen=True)
class PlannedTask:
    task_id: str
    name: str

    robot_id: str

    category: TaskCategory
    priority: TaskPriority
    risk_level: TaskRiskLevel

    sequence_index: int

    dependencies: tuple[str, ...]

    required_capabilities: frozenset[str]
    required_resources: frozenset[str]

    required_zone: str | None
    required_workspace: str | None

    preconditions: tuple[TaskCondition, ...]
    postconditions: tuple[TaskCondition, ...]

    simulated_effects: dict[str, Any]

    estimated_start_seconds: float
    estimated_finish_seconds: float

    estimated_duration_seconds: float
    estimated_energy_wh: float
    estimated_cost: float

    requires_human_review: bool

    optional: bool

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# PLAN RESULT
# ============================================================

@dataclass(frozen=True)
class TaskPlan:
    plan_id: str

    tenant_id: str
    goal_id: str

    version: int

    tasks: tuple[PlannedTask, ...]

    status: PlanningStatus
    decision: TaskPlanDecision

    total_duration_seconds: float
    total_energy_wh: float
    total_cost: float

    simulated_final_state: dict[str, Any]

    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    created_at: datetime

    simulation_only: bool = True
    hardware_execution_authorized: bool = False


# ============================================================
# DEPENDENCY RESOLVER
# ============================================================

class TaskDependencyResolver:
    """
    Deterministic priority-aware topological ordering.
    """

    @staticmethod
    def resolve(
        tasks: Sequence[TaskSpecification],
    ) -> tuple[TaskSpecification, ...]:
        task_map = {
            task.task_id: task
            for task in tasks
        }

        if len(task_map) != len(tasks):
            raise TaskPlannerDependencyError(
                "Duplicate task IDs"
            )

        indegree = {
            task.task_id: 0
            for task in tasks
        }

        dependents: dict[
            str, list[str]
        ] = defaultdict(list)

        for task in tasks:
            for dependency in task.dependencies:
                if dependency not in task_map:
                    raise TaskPlannerDependencyError(
                        f"Unknown dependency: {dependency}"
                    )

                if dependency == task.task_id:
                    raise TaskPlannerDependencyError(
                        "Self dependency detected"
                    )

                indegree[task.task_id] += 1

                dependents[dependency].append(
                    task.task_id
                )

        queue = []

        for task in tasks:
            if indegree[task.task_id] == 0:
                heapq.heappush(
                    queue,
                    (
                        -task.priority.value,
                        task.task_id,
                    ),
                )

        ordered = []

        while queue:
            _, task_id = heapq.heappop(queue)

            ordered.append(
                task_map[task_id]
            )

            for dependent in dependents[task_id]:
                indegree[dependent] -= 1

                if indegree[dependent] == 0:
                    dependent_task = task_map[dependent]

                    heapq.heappush(
                        queue,
                        (
                            -dependent_task.priority.value,
                            dependent,
                        ),
                    )

        if len(ordered) != len(tasks):
            raise TaskPlannerDependencyError(
                "Task dependency cycle detected"
            )

        return tuple(ordered)


# ============================================================
# ROBOT ELIGIBILITY
# ============================================================

class TaskRobotEligibility:

    def __init__(
        self,
        config: TaskPlannerConfig,
    ) -> None:
        self.config = config

    def check(
        self,
        task: TaskSpecification,
        robot: TaskPlannerRobotState,
        environment: TaskPlannerEnvironment,
        timestamp: datetime,
    ) -> tuple[bool, tuple[str, ...]]:
        reasons = []

        if robot.operational_state not in (
            RobotOperationalState.IDLE,
            RobotOperationalState.READY,
        ):
            reasons.append(
                "Robot unavailable"
            )

        if self.config.require_safe_robot_state:
            if robot.safety_state != RobotSafetyState.SAFE:
                reasons.append(
                    "Unsafe robot state"
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
                "Insufficient battery percentage"
            )

        if self.config.require_fresh_robot_state:
            age = (
                aware_datetime(timestamp)
                - aware_datetime(robot.timestamp)
            ).total_seconds()

            if (
                age < 0
                or age
                > self.config.maximum_robot_state_age_seconds
            ):
                reasons.append(
                    "Stale robot telemetry"
                )

        if not task.required_capabilities.issubset(
            robot.capabilities
        ):
            reasons.append(
                "Missing required capabilities"
            )

        if not task.required_resources.issubset(
            robot.available_resources
        ):
            reasons.append(
                "Missing required resources"
            )

        if (
            task.required_resources
            & environment.unavailable_resources
        ):
            reasons.append(
                "Required resources unavailable globally"
            )

        if task.required_zone is not None:
            if task.required_zone not in robot.allowed_zones:
                reasons.append(
                    "Zone not authorized"
                )

            if task.required_zone in environment.blocked_zones:
                reasons.append(
                    "Zone blocked"
                )

        if task.required_workspace is not None:
            if (
                task.required_workspace
                not in robot.allowed_workspaces
            ):
                reasons.append(
                    "Workspace not authorized"
                )

            if (
                task.required_workspace
                in environment.blocked_workspaces
            ):
                reasons.append(
                    "Workspace blocked"
                )

        if robot.energy_budget_wh is not None:
            if (
                task.estimated_energy_wh
                > robot.energy_budget_wh
            ):
                reasons.append(
                    "Insufficient robot energy budget"
                )

        if environment.emergency_active:
            reasons.append(
                "Environment emergency"
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
# ROBOT ASSIGNMENT ENGINE
# ============================================================

class TaskRobotAssignmentEngine:

    def __init__(
        self,
        config: TaskPlannerConfig,
    ) -> None:
        self.config = config
        self.eligibility = TaskRobotEligibility(config)

    def select_robot(
        self,
        task: TaskSpecification,
        robots: Sequence[TaskPlannerRobotState],
        environment: TaskPlannerEnvironment,
        timestamp: datetime,
        assigned_energy: Mapping[str, float],
        allowed_robot_ids: frozenset[str],
    ) -> TaskPlannerRobotState | None:
        candidates = []

        for robot in robots:
            if robot.robot_id not in allowed_robot_ids:
                continue

            eligible, _ = self.eligibility.check(
                task,
                robot,
                environment,
                timestamp,
            )

            if not eligible:
                continue

            used_energy = assigned_energy.get(
                robot.robot_id,
                0.0,
            )

            if robot.energy_budget_wh is not None:
                if (
                    used_energy + task.estimated_energy_wh
                    > robot.energy_budget_wh
                ):
                    continue

            preference_penalty = (
                0
                if task.preferred_robot_id == robot.robot_id
                else 1
            )

            candidates.append(
                (
                    preference_penalty,
                    used_energy,
                    -robot.battery_percentage,
                    robot.robot_id,
                    robot,
                )
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item[:4]
        )

        return candidates[0][4]


# ============================================================
# TASK SCHEDULER
# ============================================================

class TaskPlanningScheduler:
    """
    Builds a conservative sequential simulation schedule.

    Sequential scheduling avoids claiming that resources,
    zones, or robot movements can safely overlap.
    """

    @staticmethod
    def schedule(
        tasks: Sequence[
            tuple[TaskSpecification, TaskPlannerRobotState]
        ],
    ) -> tuple[PlannedTask, ...]:
        planned = []

        current_time = 0.0

        for index, (task, robot) in enumerate(tasks):
            start_time = current_time

            finish_time = (
                start_time
                + task.estimated_duration_seconds
            )

            current_time = finish_time

            planned.append(
                PlannedTask(
                    task_id=task.task_id,
                    name=task.name,
                    robot_id=robot.robot_id,
                    category=task.category,
                    priority=task.priority,
                    risk_level=task.risk_level,
                    sequence_index=index,
                    dependencies=task.dependencies,
                    required_capabilities=(
                        task.required_capabilities
                    ),
                    required_resources=(
                        task.required_resources
                    ),
                    required_zone=task.required_zone,
                    required_workspace=(
                        task.required_workspace
                    ),
                    preconditions=task.preconditions,
                    postconditions=task.postconditions,
                    simulated_effects=copy.deepcopy(
                        task.simulated_effects
                    ),
                    estimated_start_seconds=start_time,
                    estimated_finish_seconds=finish_time,
                    estimated_duration_seconds=(
                        task.estimated_duration_seconds
                    ),
                    estimated_energy_wh=(
                        task.estimated_energy_wh
                    ),
                    estimated_cost=task.estimated_cost,
                    requires_human_review=(
                        task.requires_human_review
                    ),
                    optional=task.optional,
                    metadata=copy.deepcopy(task.metadata),
                )
            )

        return tuple(planned)


# ============================================================
# WORLD STATE SIMULATOR
# ============================================================

class TaskWorldStateSimulator:

    @staticmethod
    def simulate(
        tasks: Sequence[PlannedTask],
        initial_state: Mapping[str, Any],
    ) -> tuple[
        dict[str, Any],
        tuple[str, ...],
    ]:
        state = copy.deepcopy(dict(initial_state))

        errors = []

        for task in tasks:
            for condition in task.preconditions:
                if not TaskConditionEvaluator.evaluate(
                    condition,
                    state,
                ):
                    errors.append(
                        f"Precondition failed for "
                        f"{task.task_id}: {condition.key}"
                    )

            for key, value in task.simulated_effects.items():
                state[key] = copy.deepcopy(value)

            for condition in task.postconditions:
                if not TaskConditionEvaluator.evaluate(
                    condition,
                    state,
                ):
                    errors.append(
                        f"Postcondition failed for "
                        f"{task.task_id}: {condition.key}"
                    )

        return (
            state,
            tuple(errors),
        )


# ============================================================
# MAIN TASK PLANNER
# ============================================================

class RoboticsTaskPlanner:
    """
    Multi-robot, goal-oriented simulation task planner.

    Plans are advisory and must be independently validated
    before being passed to a simulation scheduler.
    """

    MOTION_CATEGORIES = frozenset({
        TaskCategory.NAVIGATION,
        TaskCategory.MANIPULATION,
        TaskCategory.PICK,
        TaskCategory.PLACE,
        TaskCategory.REPAIR,
    })

    def __init__(
        self,
        config: TaskPlannerConfig | None = None,
    ) -> None:
        self.config = config or TaskPlannerConfig()

        self._plans: dict[
            tuple[str, str],
            TaskPlan,
        ] = {}

        self._templates: dict[
            tuple[str, str],
            HierarchicalTaskTemplate,
        ] = {}

        self._audit: list[dict[str, Any]] = []

        self._lock = threading.RLock()

        self._assignment = TaskRobotAssignmentEngine(
            self.config
        )

    # --------------------------------------------------------
    # ACCESS CONTROL
    # --------------------------------------------------------

    @staticmethod
    def _require(
        access: TaskPlannerAccessContext,
        permission: str,
    ) -> None:
        if not getattr(access, permission, False):
            raise TaskPlannerAuthorizationError(
                f"Missing permission: {permission}"
            )

    @staticmethod
    def _authorize_tenant(
        tenant_id: str,
        access: TaskPlannerAccessContext,
    ) -> None:
        if tenant_id != access.tenant_id:
            raise TaskPlannerAuthorizationError(
                "Cross-tenant access denied"
            )

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit_event(
        self,
        access: TaskPlannerAccessContext,
        action: str,
        details: Mapping[str, Any],
    ) -> None:
        self._audit.append({
            "event_id": generate_id(),
            "tenant_id": access.tenant_id,
            "user_id": access.user_id,
            "action": action,
            "details": json_safe(details),
            "timestamp": utc_now().isoformat(),
        })

    # --------------------------------------------------------
    # REGISTER HIERARCHICAL TEMPLATE
    # --------------------------------------------------------

    def register_template(
        self,
        template: HierarchicalTaskTemplate,
        *,
        access: TaskPlannerAccessContext,
    ) -> None:
        self._require(
            access,
            "can_manage_templates",
        )

        TaskDependencyResolver.resolve(
            template.subtasks
        )

        key = (
            access.tenant_id,
            template.template_id,
        )

        with self._lock:
            if key in self._templates:
                raise TaskPlannerValidationError(
                    "Template already registered"
                )

            self._templates[key] = copy.deepcopy(
                template
            )

            self._audit_event(
                access,
                "template_registered",
                {
                    "template_id": template.template_id,
                },
            )

    # --------------------------------------------------------
    # GET TEMPLATE
    # --------------------------------------------------------

    def get_template(
        self,
        template_id: str,
        *,
        access: TaskPlannerAccessContext,
    ) -> HierarchicalTaskTemplate:
        self._require(
            access,
            "can_view_plans",
        )

        key = (
            access.tenant_id,
            template_id,
        )

        with self._lock:
            template = self._templates.get(key)

            if template is None:
                raise TaskPlannerNotFoundError(
                    "Template not found"
                )

            return copy.deepcopy(template)

    # --------------------------------------------------------
    # SAFETY GATE
    # --------------------------------------------------------

    def _check_environment(
        self,
        request: TaskPlanningRequest,
    ) -> tuple[bool, tuple[str, ...]]:
        reasons = []

        environment = request.environment

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

        age = (
            aware_datetime(request.requested_at)
            - aware_datetime(environment.timestamp)
        ).total_seconds()

        if (
            age < 0
            or age > self.config.maximum_environment_age_seconds
        ):
            reasons.append(
                "Environment snapshot stale or future-dated"
            )

        return (
            not reasons,
            tuple(reasons),
        )

    # --------------------------------------------------------
    # PLAN GOAL
    # --------------------------------------------------------

    def plan(
        self,
        request: TaskPlanningRequest,
        *,
        access: TaskPlannerAccessContext,
    ) -> TaskPlan:
        self._require(
            access,
            "can_plan",
        )

        goal = request.goal

        self._authorize_tenant(
            goal.tenant_id,
            access,
        )

        if len(goal.tasks) > self.config.maximum_tasks:
            raise TaskPlannerCapacityError(
                "Goal exceeds maximum task count"
            )

        if len({
            robot.robot_id
            for robot in request.robots
        }) != len(request.robots):
            raise TaskPlannerValidationError(
                "Duplicate robot snapshots"
            )

        for robot in request.robots:
            if robot.tenant_id != goal.tenant_id:
                raise TaskPlannerAuthorizationError(
                    "Cross-tenant robot snapshot"
                )

        ordered_tasks = TaskDependencyResolver.resolve(
            goal.tasks
        )

        environment_safe, safety_reasons = (
            self._check_environment(request)
        )

        if not environment_safe:
            result = TaskPlan(
                plan_id=generate_id(),
                tenant_id=goal.tenant_id,
                goal_id=goal.goal_id,
                version=1,
                tasks=(),
                status=PlanningStatus.BLOCKED_SAFETY,
                decision=TaskPlanDecision.REJECT,
                total_duration_seconds=0,
                total_energy_wh=0,
                total_cost=0,
                simulated_final_state=copy.deepcopy(
                    goal.initial_world_state
                ),
                warnings=(),
                errors=safety_reasons,
                created_at=utc_now(),
            )

            return self._save_plan(
                result,
                access,
            )

        assigned_energy: dict[
            str, float
        ] = defaultdict(float)

        assignments: list[
            tuple[TaskSpecification, TaskPlannerRobotState]
        ] = []

        warnings = []
        errors = []

        skipped_task_ids = set()

        for task in ordered_tasks:
            if any(
                dependency in skipped_task_ids
                for dependency in task.dependencies
            ):
                if task.optional:
                    skipped_task_ids.add(task.task_id)

                    warnings.append(
                        f"Skipped optional task {task.task_id} "
                        "because a dependency was skipped"
                    )

                    continue

                errors.append(
                    f"Required task {task.task_id} depends "
                    "on a skipped task"
                )

                continue

            robot = self._assignment.select_robot(
                task,
                request.robots,
                request.environment,
                request.requested_at,
                assigned_energy,
                access.allowed_robot_ids,
            )

            if robot is None:
                if task.optional:
                    skipped_task_ids.add(
                        task.task_id
                    )

                    warnings.append(
                        f"Optional task skipped: {task.task_id}"
                    )

                    continue

                errors.append(
                    f"No eligible robot for task: {task.task_id}"
                )

                continue

            assignments.append(
                (task, robot)
            )

            assigned_energy[robot.robot_id] += (
                task.estimated_energy_wh
            )

        planned_tasks = TaskPlanningScheduler.schedule(
            assignments
        )

        final_state, state_errors = (
            TaskWorldStateSimulator.simulate(
                planned_tasks,
                goal.initial_world_state,
            )
        )

        errors.extend(state_errors)

        for key, expected in goal.expected_world_state.items():
            if (
                key not in final_state
                or final_state[key] != expected
            ):
                errors.append(
                    f"Goal final state mismatch: {key}"
                )

        total_duration = sum(
            task.estimated_duration_seconds
            for task in planned_tasks
        )

        total_energy = sum(
            task.estimated_energy_wh
            for task in planned_tasks
        )

        total_cost = sum(
            task.estimated_cost
            for task in planned_tasks
        )

        max_duration = min(
            self.config.maximum_plan_duration_seconds,
            (
                goal.maximum_duration_seconds
                if goal.maximum_duration_seconds is not None
                else self.config.maximum_plan_duration_seconds
            ),
        )

        max_energy = min(
            self.config.maximum_plan_energy_wh,
            (
                goal.maximum_energy_wh
                if goal.maximum_energy_wh is not None
                else self.config.maximum_plan_energy_wh
            ),
        )

        max_cost = min(
            self.config.maximum_plan_cost,
            (
                goal.maximum_cost
                if goal.maximum_cost is not None
                else self.config.maximum_plan_cost
            ),
        )

        if total_duration > max_duration:
            errors.append(
                "Plan duration budget exceeded"
            )

        if total_energy > max_energy:
            errors.append(
                "Plan energy budget exceeded"
            )

        if total_cost > max_cost:
            errors.append(
                "Plan cost budget exceeded"
            )

        if goal.deadline is not None:
            estimated_finish = (
                aware_datetime(request.requested_at)
                + timedelta(seconds=total_duration)
            )

            if estimated_finish > aware_datetime(
                goal.deadline
            ):
                errors.append(
                    "Plan cannot meet requested deadline"
                )

        review_required = False

        for task in planned_tasks:
            if task.requires_human_review:
                review_required = True

            if (
                self.config.require_review_for_high_risk
                and task.risk_level in (
                    TaskRiskLevel.HIGH,
                    TaskRiskLevel.CRITICAL,
                )
            ):
                review_required = True

            if (
                self.config.require_review_for_motion
                and task.category in self.MOTION_CATEGORIES
            ):
                review_required = True

                warnings.append(
                    f"Task {task.task_id}: physical motion "
                    "safety has not been verified"
                )

        if errors:
            status = PlanningStatus.FAILED
            decision = TaskPlanDecision.REJECT

        elif skipped_task_ids:
            status = PlanningStatus.PARTIAL
            decision = (
                TaskPlanDecision.HUMAN_REVIEW_REQUIRED
            )

        else:
            status = PlanningStatus.SUCCESS

            decision = (
                TaskPlanDecision.HUMAN_REVIEW_REQUIRED
                if review_required
                else TaskPlanDecision.SIMULATION_ELIGIBLE
            )

        result = TaskPlan(
            plan_id=generate_id(),
            tenant_id=goal.tenant_id,
            goal_id=goal.goal_id,
            version=1,
            tasks=planned_tasks,
            status=status,
            decision=decision,
            total_duration_seconds=total_duration,
            total_energy_wh=total_energy,
            total_cost=total_cost,
            simulated_final_state=final_state,
            warnings=tuple(warnings),
            errors=tuple(errors),
            created_at=utc_now(),
        )

        return self._save_plan(
            result,
            access,
        )

    # --------------------------------------------------------
    # SAVE PLAN
    # --------------------------------------------------------

    def _save_plan(
        self,
        plan: TaskPlan,
        access: TaskPlannerAccessContext,
    ) -> TaskPlan:
        key = (
            plan.tenant_id,
            plan.plan_id,
        )

        with self._lock:
            if (
                len(self._plans)
                >= self.config.maximum_saved_plans
            ):
                raise TaskPlannerCapacityError(
                    "Maximum saved plan capacity reached"
                )

            self._plans[key] = copy.deepcopy(plan)

            self._audit_event(
                access,
                "task_plan_created",
                {
                    "plan_id": plan.plan_id,
                    "goal_id": plan.goal_id,
                    "status": plan.status.value,
                    "decision": plan.decision.value,
                },
            )

        return copy.deepcopy(plan)

    # --------------------------------------------------------
    # GET PLAN
    # --------------------------------------------------------

    def get_plan(
        self,
        plan_id: str,
        *,
        access: TaskPlannerAccessContext,
    ) -> TaskPlan:
        self._require(
            access,
            "can_view_plans",
        )

        key = (
            access.tenant_id,
            plan_id,
        )

        with self._lock:
            plan = self._plans.get(key)

            if plan is None:
                raise TaskPlannerNotFoundError(
                    "Task plan not found"
                )

            for task in plan.tasks:
                if (
                    task.robot_id
                    not in access.allowed_robot_ids
                ):
                    raise TaskPlannerAuthorizationError(
                        "Robot access denied"
                    )

            return copy.deepcopy(plan)

    # --------------------------------------------------------
    # LIST PLANS
    # --------------------------------------------------------

    def list_plans(
        self,
        *,
        access: TaskPlannerAccessContext,
    ) -> tuple[TaskPlan, ...]:
        self._require(
            access,
            "can_view_plans",
        )

        with self._lock:
            plans = [
                copy.deepcopy(plan)
                for (tenant_id, _), plan
                in self._plans.items()
                if tenant_id == access.tenant_id
                and all(
                    task.robot_id
                    in access.allowed_robot_ids
                    for task in plan.tasks
                )
            ]

        return tuple(
            sorted(
                plans,
                key=lambda plan: plan.created_at,
                reverse=True,
            )
        )

    # --------------------------------------------------------
    # PLAN VALIDATOR INTEGRATION
    # --------------------------------------------------------

    @staticmethod
    def plan_validator_payload(
        plan: TaskPlan,
    ) -> dict[str, Any]:
        return {
            "plan_id": plan.plan_id,
            "tenant_id": plan.tenant_id,
            "goal_id": plan.goal_id,
            "version": plan.version,
            "planning_status": plan.status.value,
            "requires_independent_validation": True,
            "tasks": [
                {
                    "step_id": task.task_id,
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "robot_id": task.robot_id,
                    "category": task.category.value,
                    "sequence_index": task.sequence_index,
                    "dependencies": list(
                        task.dependencies
                    ),
                    "required_capabilities": sorted(
                        task.required_capabilities
                    ),
                    "required_resources": sorted(
                        task.required_resources
                    ),
                    "required_zone": task.required_zone,
                    "required_workspace": (
                        task.required_workspace
                    ),
                    "estimated_cost": task.estimated_cost,
                    "estimated_duration_seconds": (
                        task.estimated_duration_seconds
                    ),
                    "estimated_energy_wh": (
                        task.estimated_energy_wh
                    ),
                    "risk_level": task.risk_level.value,
                    "requires_human_approval": (
                        task.requires_human_review
                    ),
                    "effects": json_safe(
                        task.simulated_effects
                    ),
                }
                for task in plan.tasks
            ],
            "simulation_only": True,
            "hardware_execution_authorized": False,
        }

    # --------------------------------------------------------
    # ACTION SCHEDULER INTEGRATION
    # --------------------------------------------------------

    @staticmethod
    def action_scheduler_payload(
        plan: TaskPlan,
    ) -> dict[str, Any]:
        return {
            "plan_id": plan.plan_id,
            "goal_id": plan.goal_id,
            "allow_simulation_scheduling": False,
            "requires_plan_validation": True,
            "requires_approval": (
                plan.decision
                == TaskPlanDecision.HUMAN_REVIEW_REQUIRED
            ),
            "actions": [
                {
                    "task_id": task.task_id,
                    "robot_id": task.robot_id,
                    "action_type": task.category.value,
                    "priority": task.priority.value,
                    "dependencies": list(
                        task.dependencies
                    ),
                    "estimated_start_seconds": (
                        task.estimated_start_seconds
                    ),
                    "estimated_duration_seconds": (
                        task.estimated_duration_seconds
                    ),
                }
                for task in plan.tasks
            ],
            "simulation_only": True,
            "hardware_execution_authorized": False,
        }

    # --------------------------------------------------------
    # EXECUTION MONITOR INTEGRATION
    # --------------------------------------------------------

    @staticmethod
    def execution_monitor_payload(
        plan: TaskPlan,
    ) -> dict[str, Any]:
        return {
            "plan_id": plan.plan_id,
            "goal_id": plan.goal_id,
            "plan_version": plan.version,
            "expected_task_count": len(plan.tasks),
            "expected_duration_seconds": (
                plan.total_duration_seconds
            ),
            "expected_energy_wh": plan.total_energy_wh,
            "expected_cost": plan.total_cost,
            "planning_status": plan.status.value,
            "simulation_only": True,
        }

    # --------------------------------------------------------
    # REPLANNING INTEGRATION
    # --------------------------------------------------------

    @staticmethod
    def replanning_payload(
        plan: TaskPlan,
    ) -> dict[str, Any]:
        return {
            "plan_id": plan.plan_id,
            "tenant_id": plan.tenant_id,
            "goal_id": plan.goal_id,
            "version": plan.version,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "name": task.name,
                    "robot_id": task.robot_id,
                    "category": task.category.value,
                    "status": "pending",
                    "dependencies": list(
                        task.dependencies
                    ),
                    "required_capabilities": sorted(
                        task.required_capabilities
                    ),
                    "required_resources": sorted(
                        task.required_resources
                    ),
                    "estimated_cost": task.estimated_cost,
                    "estimated_duration_seconds": (
                        task.estimated_duration_seconds
                    ),
                    "estimated_energy_wh": (
                        task.estimated_energy_wh
                    ),
                }
                for task in plan.tasks
            ],
            "simulation_only": True,
        }

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    def audit_history(
        self,
        *,
        access: TaskPlannerAccessContext,
    ) -> list[dict[str, Any]]:
        self._require(
            access,
            "can_view_plans",
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
        plan: TaskPlan,
    ) -> str:
        return json.dumps(
            json_safe(plan),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestRoboticsTaskPlanner(unittest.TestCase):

    def setUp(self) -> None:
        self.planner = RoboticsTaskPlanner()

        self.access = TaskPlannerAccessContext(
            tenant_id="tenant_alpha",
            user_id="engineer",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
                "ROBOT-002",
            }),
            can_plan=True,
            can_view_plans=True,
            can_manage_templates=True,
        )

        self.robot = TaskPlannerRobotState(
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            capabilities=frozenset({
                "inspection",
                "diagnostics",
            }),
            available_resources=frozenset({
                "camera",
                "diagnostic_tool",
            }),
            allowed_zones=frozenset({
                "ZONE-A",
            }),
            allowed_workspaces=frozenset({
                "WORKSPACE-1",
            }),
            battery_percentage=95,
            safety_state=RobotSafetyState.SAFE,
            operational_state=RobotOperationalState.READY,
            energy_budget_wh=500,
        )

        self.task_1 = TaskSpecification(
            task_id="TASK-001",
            name="inspect_asset",
            category=TaskCategory.INSPECTION,
            priority=TaskPriority.HIGH,
            required_capabilities=frozenset({
                "inspection",
            }),
            required_resources=frozenset({
                "camera",
            }),
            simulated_effects={
                "inspection_completed": True,
            },
            estimated_duration_seconds=30,
            estimated_energy_wh=10,
            estimated_cost=5,
        )

        self.task_2 = TaskSpecification(
            task_id="TASK-002",
            name="diagnose_asset",
            category=TaskCategory.DIAGNOSTICS,
            dependencies=("TASK-001",),
            required_capabilities=frozenset({
                "diagnostics",
            }),
            required_resources=frozenset({
                "diagnostic_tool",
            }),
            preconditions=(
                TaskCondition(
                    key="inspection_completed",
                    operator=ConditionOperator.EQ,
                    expected=True,
                ),
            ),
            simulated_effects={
                "diagnostics_completed": True,
            },
            estimated_duration_seconds=60,
            estimated_energy_wh=20,
            estimated_cost=10,
        )

        self.goal = TaskPlanningGoal(
            goal_id="GOAL-001",
            tenant_id="tenant_alpha",
            name="Inspect and Diagnose Asset",
            tasks=(
                self.task_2,
                self.task_1,
            ),
            expected_world_state={
                "inspection_completed": True,
                "diagnostics_completed": True,
            },
        )

    def request(
        self,
        environment: TaskPlannerEnvironment | None = None,
    ) -> TaskPlanningRequest:
        return TaskPlanningRequest(
            request_id=generate_id(),
            goal=self.goal,
            robots=(self.robot,),
            environment=(
                environment
                or TaskPlannerEnvironment()
            ),
        )

    def test_successful_planning(self) -> None:
        plan = self.planner.plan(
            self.request(),
            access=self.access,
        )

        self.assertEqual(
            plan.status,
            PlanningStatus.SUCCESS,
        )

        self.assertEqual(
            len(plan.tasks),
            2,
        )

    def test_dependency_ordering(self) -> None:
        plan = self.planner.plan(
            self.request(),
            access=self.access,
        )

        self.assertEqual(
            plan.tasks[0].task_id,
            "TASK-001",
        )

        self.assertEqual(
            plan.tasks[1].task_id,
            "TASK-002",
        )

    def test_world_state(self) -> None:
        plan = self.planner.plan(
            self.request(),
            access=self.access,
        )

        self.assertTrue(
            plan.simulated_final_state[
                "diagnostics_completed"
            ]
        )

    def test_duration(self) -> None:
        plan = self.planner.plan(
            self.request(),
            access=self.access,
        )

        self.assertEqual(
            plan.total_duration_seconds,
            90,
        )

    def test_collision_risk_blocks(self) -> None:
        plan = self.planner.plan(
            self.request(
                TaskPlannerEnvironment(
                    collision_risk_active=True,
                )
            ),
            access=self.access,
        )

        self.assertEqual(
            plan.status,
            PlanningStatus.BLOCKED_SAFETY,
        )

    def test_tenant_isolation(self) -> None:
        other_access = TaskPlannerAccessContext(
            tenant_id="tenant_beta",
            user_id="other",
            can_plan=True,
        )

        with self.assertRaises(
            TaskPlannerAuthorizationError
        ):
            self.planner.plan(
                self.request(),
                access=other_access,
            )

    def test_json_export(self) -> None:
        plan = self.planner.plan(
            self.request(),
            access=self.access,
        )

        exported = self.planner.export_json(plan)

        self.assertIn(
            "TASK-001",
            exported,
        )

        self.assertIn(
            "hardware_execution_authorized",
            exported,
        )

    def test_dependency_cycle(self) -> None:
        a = TaskSpecification(
            task_id="A",
            name="A",
            category=TaskCategory.CUSTOM,
            dependencies=("B",),
        )

        b = TaskSpecification(
            task_id="B",
            name="B",
            category=TaskCategory.CUSTOM,
            dependencies=("A",),
        )

        with self.assertRaises(
            TaskPlannerDependencyError
        ):
            TaskDependencyResolver.resolve(
                (a, b)
            )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print(
        "\nPHYSICAL AI & AUTONOMOUS ROBOTICS PLATFORM"
    )

    print(
        "Goal-Oriented Multi-Robot Task Planner\n"
    )

    planner = RoboticsTaskPlanner()

    access = TaskPlannerAccessContext(
        tenant_id="tenant_alpha",
        user_id="robotics_engineer",
        allowed_robot_ids=frozenset({
            "ROBOT-001",
            "ROBOT-002",
        }),
        can_plan=True,
        can_view_plans=True,
        can_manage_templates=True,
    )

    robot_1 = TaskPlannerRobotState(
        tenant_id="tenant_alpha",
        robot_id="ROBOT-001",
        capabilities=frozenset({
            "inspection",
        }),
        available_resources=frozenset({
            "camera",
        }),
        allowed_zones=frozenset({
            "ZONE-A",
        }),
        allowed_workspaces=frozenset({
            "WORKSPACE-1",
        }),
        battery_percentage=95,
        safety_state=RobotSafetyState.SAFE,
        operational_state=RobotOperationalState.READY,
        energy_budget_wh=500,
    )

    robot_2 = TaskPlannerRobotState(
        tenant_id="tenant_alpha",
        robot_id="ROBOT-002",
        capabilities=frozenset({
            "diagnostics",
        }),
        available_resources=frozenset({
            "diagnostic_tool",
        }),
        allowed_zones=frozenset({
            "ZONE-A",
        }),
        allowed_workspaces=frozenset({
            "WORKSPACE-1",
        }),
        battery_percentage=90,
        safety_state=RobotSafetyState.SAFE,
        operational_state=RobotOperationalState.READY,
        energy_budget_wh=500,
    )

    inspect = TaskSpecification(
        task_id="TASK-INSPECT",
        name="Inspect Industrial Robot",
        category=TaskCategory.INSPECTION,
        priority=TaskPriority.HIGH,
        required_capabilities=frozenset({
            "inspection",
        }),
        required_resources=frozenset({
            "camera",
        }),
        required_zone="ZONE-A",
        required_workspace="WORKSPACE-1",
        simulated_effects={
            "inspection_completed": True,
        },
        estimated_duration_seconds=60,
        estimated_energy_wh=15,
        estimated_cost=10,
    )

    diagnose = TaskSpecification(
        task_id="TASK-DIAGNOSE",
        name="Diagnose Robot Components",
        category=TaskCategory.DIAGNOSTICS,
        dependencies=("TASK-INSPECT",),
        required_capabilities=frozenset({
            "diagnostics",
        }),
        required_resources=frozenset({
            "diagnostic_tool",
        }),
        required_zone="ZONE-A",
        required_workspace="WORKSPACE-1",
        preconditions=(
            TaskCondition(
                key="inspection_completed",
                operator=ConditionOperator.EQ,
                expected=True,
            ),
        ),
        simulated_effects={
            "diagnostics_completed": True,
        },
        estimated_duration_seconds=90,
        estimated_energy_wh=25,
        estimated_cost=20,
    )

    goal = TaskPlanningGoal(
        goal_id="GOAL-ROBOT-HEALTH",
        tenant_id="tenant_alpha",
        name="Robot Health Inspection and Diagnostics",
        tasks=(
            diagnose,
            inspect,
        ),
        expected_world_state={
            "inspection_completed": True,
            "diagnostics_completed": True,
        },
        maximum_cost=100,
        maximum_energy_wh=200,
        maximum_duration_seconds=600,
    )

    request = TaskPlanningRequest(
        request_id=generate_id(),
        goal=goal,
        robots=(
            robot_1,
            robot_2,
        ),
        environment=TaskPlannerEnvironment(),
    )

    plan = planner.plan(
        request,
        access=access,
    )

    print("GENERATED TASK PLAN\n")

    print(
        planner.export_json(plan)
    )

    print("\nPLAN VALIDATOR PAYLOAD\n")

    print(
        json.dumps(
            planner.plan_validator_payload(plan),
            indent=2,
        )
    )

    print("\nACTION SCHEDULER PAYLOAD\n")

    print(
        json.dumps(
            planner.action_scheduler_payload(plan),
            indent=2,
        )
    )

    print("\nREPLANNING PAYLOAD\n")

    print(
        json.dumps(
            planner.replanning_payload(plan),
            indent=2,
        )
    )

    print("\nRUNNING UNIT TESTS\n")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestRoboticsTaskPlanner
    )

    unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)


if __name__ == "__main__":
    main()
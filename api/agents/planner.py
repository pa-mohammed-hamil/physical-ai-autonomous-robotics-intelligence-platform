"""Planner module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/ai/agents/planner.py

Physical AI & Autonomous Robotics Intelligence Platform
Autonomous Task Planning & Reasoning Agent

Python 3.11+

Features:
- Goal-driven task planning
- Hierarchical task decomposition
- Multi-agent planning
- Dependency-aware DAG generation
- Robot capability matching
- Resource and constraint validation
- Task priority and scheduling
- Risk and safety assessment
- Human-in-the-loop approval requirements
- Planning simulation integration
- Plan evaluation and optimization
- Task replanning
- Explainable planning decisions
- Tenant isolation and authorization
- Audit logging

Safety:
- Planning is advisory and simulation-only.
- No direct robot or actuator commands.
- No emergency-stop reset.
- No automatic authorization for physical execution.
- Independent safety validation is required before
  any real-world robot operation.
"""

from __future__ import annotations

import asyncio
import copy
import math
import re
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


# ============================================================
# Constants
# ============================================================

MAX_PLAN_TASKS = 200
MAX_TASK_DEPENDENCIES = 50
MAX_PLANNING_DEPTH = 20
MAX_CONSTRAINTS = 100
MAX_CANDIDATE_PLANS = 10
MAX_BATCH_PLANS = 25

DEFAULT_TASK_DURATION_SECONDS = 60.0
DEFAULT_MAX_PLANNING_TIME_SECONDS = 120.0

IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z0-9_.:-]{1,128}$"
)


# ============================================================
# Exceptions
# ============================================================

class PlannerError(Exception):
    """Base planning exception."""


class PlannerValidationError(PlannerError):
    """Invalid planning request or data."""


class PlannerPermissionError(PlannerError):
    """Planning authorization failure."""


class PlannerNotFoundError(PlannerError):
    """Required planning resource not found."""


class PlannerDependencyError(PlannerError):
    """Invalid task dependency graph."""


class PlannerCapabilityError(PlannerError):
    """Required robot or agent capability unavailable."""


class PlannerSafetyError(PlannerError):
    """Plan violates a safety restriction."""


class PlannerServiceUnavailableError(PlannerError):
    """Required planning provider unavailable."""


# ============================================================
# Utilities
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return str(uuid.uuid4())


def validate_identifier(value: str) -> str:
    if (
        not isinstance(value, str)
        or not IDENTIFIER_PATTERN.fullmatch(value)
    ):
        raise PlannerValidationError(
            f"Invalid identifier: {value!r}"
        )
    return value


def validate_finite(value: float) -> float:
    if isinstance(value, bool) or not math.isfinite(value):
        raise PlannerValidationError(
            "Value must be finite"
        )
    return value


def validate_nonnegative(value: float) -> float:
    validate_finite(value)
    if value < 0:
        raise PlannerValidationError(
            "Value must be nonnegative"
        )
    return value


def validate_positive(value: float) -> float:
    validate_finite(value)
    if value <= 0:
        raise PlannerValidationError(
            "Value must be positive"
        )
    return value


def validate_probability(value: float) -> float:
    validate_finite(value)
    if not 0.0 <= value <= 1.0:
        raise PlannerValidationError(
            "Probability must be between 0 and 1"
        )
    return value


def validate_timestamp(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise PlannerValidationError(
            "Datetime must be timezone-aware"
        )
    return value


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(minimum, min(maximum, value))


# ============================================================
# Enumerations
# ============================================================

class PlanningMode(str, Enum):
    GOAL_DRIVEN = "goal_driven"
    HIERARCHICAL = "hierarchical"
    MULTI_AGENT = "multi_agent"
    BEHAVIOR_TREE = "behavior_tree"
    WORKFLOW = "workflow"
    SIMULATION = "simulation"
    REPLANNING = "replanning"


class PlanningDomain(str, Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    INSPECTION = "inspection"
    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    FLEET = "fleet"
    PERCEPTION = "perception"
    HEALTH = "health"
    SAFETY = "safety"
    SIMULATION = "simulation"
    RESEARCH = "research"
    GENERAL = "general"


class PlanningTaskType(str, Enum):
    PERCEPTION = "perception"
    OBJECT_DETECTION = "object_detection"
    SCENE_ANALYSIS = "scene_analysis"
    NAVIGATION_PLANNING = "navigation_planning"
    PATH_VALIDATION = "path_validation"
    MANIPULATION_PLANNING = "manipulation_planning"
    GRASP_PLANNING = "grasp_planning"
    HEALTH_ANALYSIS = "health_analysis"
    FAULT_DIAGNOSIS = "fault_diagnosis"
    MAINTENANCE_PLANNING = "maintenance_planning"
    REPAIR_ESTIMATION = "repair_estimation"
    COST_ANALYSIS = "cost_analysis"
    SAFETY_ASSESSMENT = "safety_assessment"
    SIMULATION = "simulation"
    EVALUATION = "evaluation"
    HUMAN_REVIEW = "human_review"
    REPORT_GENERATION = "report_generation"
    CUSTOM = "custom"


class PlanningStatus(str, Enum):
    DRAFT = "draft"
    VALIDATING = "validating"
    VALIDATED = "validated"
    REVIEW_REQUIRED = "review_required"
    SIMULATION_PASSED = "simulation_passed"
    SIMULATION_FAILED = "simulation_failed"
    REJECTED = "rejected"
    FAILED = "failed"


class PlanningTaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    BLOCKED = "blocked"
    COMPLETED_IN_SIMULATION = "completed_in_simulation"
    FAILED_IN_SIMULATION = "failed_in_simulation"


class PlanningPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class PlanningRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class PlanningSafetyClass(str, Enum):
    READ_ONLY = "read_only"
    ADVISORY = "advisory"
    SIMULATION = "simulation"
    SAFETY_CRITICAL = "safety_critical"
    PHYSICAL_ACTUATION = "physical_actuation"


class ConstraintType(str, Enum):
    REQUIRED_CAPABILITY = "required_capability"
    MAXIMUM_DURATION = "maximum_duration"
    MAXIMUM_COST = "maximum_cost"
    MINIMUM_BATTERY = "minimum_battery"
    ALLOWED_ROBOT = "allowed_robot"
    FORBIDDEN_ROBOT = "forbidden_robot"
    HUMAN_APPROVAL = "human_approval"
    SIMULATION_REQUIRED = "simulation_required"
    CUSTOM = "custom"


class ConstraintSeverity(str, Enum):
    SOFT = "soft"
    HARD = "hard"


class PlanningSimulationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    NOT_RUN = "not_run"


# ============================================================
# Scope
# ============================================================

@dataclass(frozen=True, slots=True)
class PlannerScope:
    tenant_id: str
    user_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_identifier(self.user_id)


# ============================================================
# Goal and Constraint Models
# ============================================================

@dataclass(frozen=True, slots=True)
class PlanningGoal:
    goal_id: str
    description: str
    domain: PlanningDomain

    target_robot_id: str | None = None

    success_criteria: tuple[str, ...] = ()

    priority: PlanningPriority = PlanningPriority.NORMAL

    def __post_init__(self) -> None:
        validate_identifier(self.goal_id)

        if not self.description.strip():
            raise PlannerValidationError(
                "Goal description is required"
            )

        if self.target_robot_id is not None:
            validate_identifier(self.target_robot_id)


@dataclass(frozen=True, slots=True)
class PlanningConstraint:
    constraint_id: str
    constraint_type: ConstraintType

    value: Any

    severity: ConstraintSeverity = ConstraintSeverity.HARD

    description: str = ""

    def __post_init__(self) -> None:
        validate_identifier(self.constraint_id)


# ============================================================
# Robot Capability Models
# ============================================================

@dataclass(frozen=True, slots=True)
class PlanningRobot:
    robot_id: str
    tenant_id: str

    capabilities: frozenset[str]

    available: bool
    healthy: bool

    battery_soc: float | None = None

    simulation_supported: bool = True

    def __post_init__(self) -> None:
        validate_identifier(self.robot_id)
        validate_identifier(self.tenant_id)

        if self.battery_soc is not None:
            validate_probability(self.battery_soc)


# ============================================================
# Task Models
# ============================================================

@dataclass(slots=True)
class PlanningTask:
    task_id: str
    name: str
    description: str

    task_type: PlanningTaskType

    dependencies: list[str] = field(default_factory=list)

    required_capabilities: set[str] = field(
        default_factory=set
    )

    assigned_robot_id: str | None = None

    estimated_duration_seconds: float = (
        DEFAULT_TASK_DURATION_SECONDS
    )

    estimated_cost: float = 0.0

    priority: PlanningPriority = PlanningPriority.NORMAL

    safety_class: PlanningSafetyClass = (
        PlanningSafetyClass.ADVISORY
    )

    requires_human_review: bool = False

    status: PlanningTaskStatus = (
        PlanningTaskStatus.PENDING
    )

    parameters: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_identifier(self.task_id)

        if not self.name.strip():
            raise PlannerValidationError(
                "Task name is required"
            )

        if len(self.dependencies) > MAX_TASK_DEPENDENCIES:
            raise PlannerValidationError(
                "Too many dependencies"
            )

        if len(self.dependencies) != len(
            set(self.dependencies)
        ):
            raise PlannerDependencyError(
                "Duplicate task dependencies"
            )

        for dependency in self.dependencies:
            validate_identifier(dependency)

        if self.task_id in self.dependencies:
            raise PlannerDependencyError(
                "Task cannot depend on itself"
            )

        validate_positive(
            self.estimated_duration_seconds
        )

        validate_nonnegative(self.estimated_cost)

        if self.assigned_robot_id is not None:
            validate_identifier(self.assigned_robot_id)

        if (
            self.safety_class
            == PlanningSafetyClass.PHYSICAL_ACTUATION
        ):
            raise PlannerSafetyError(
                "Physical actuation is not supported "
                "by the advisory planner"
            )

        if (
            self.safety_class
            == PlanningSafetyClass.SAFETY_CRITICAL
        ):
            self.requires_human_review = True


# ============================================================
# Planning Request
# ============================================================

@dataclass(slots=True)
class PlanningRequest:
    tenant_id: str

    goal: PlanningGoal

    mode: PlanningMode = PlanningMode.HIERARCHICAL

    constraints: list[PlanningConstraint] = field(
        default_factory=list
    )

    available_robot_ids: list[str] = field(
        default_factory=list
    )

    maximum_tasks: int = MAX_PLAN_TASKS

    maximum_planning_depth: int = MAX_PLANNING_DEPTH

    require_simulation: bool = True

    simulation_only: bool = True

    context: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)

        if len(self.constraints) > MAX_CONSTRAINTS:
            raise PlannerValidationError(
                "Too many planning constraints"
            )

        if not 1 <= self.maximum_tasks <= MAX_PLAN_TASKS:
            raise PlannerValidationError(
                "Invalid maximum task count"
            )

        if not (
            1 <= self.maximum_planning_depth
            <= MAX_PLANNING_DEPTH
        ):
            raise PlannerValidationError(
                "Invalid planning depth"
            )

        if not self.simulation_only:
            raise PlannerSafetyError(
                "Planner supports simulation-only plans"
            )

        for robot_id in self.available_robot_ids:
            validate_identifier(robot_id)


# ============================================================
# Plan Evaluation Models
# ============================================================

@dataclass(frozen=True, slots=True)
class PlanningSafetyAssessment:
    risk_level: PlanningRiskLevel

    valid: bool

    requires_human_review: bool

    violations: tuple[str, ...]

    warnings: tuple[str, ...]


@dataclass(slots=True)
class PlanningSimulationResult:
    simulation_id: str

    status: PlanningSimulationStatus

    success_probability: float | None

    estimated_duration_seconds: float | None

    findings: list[str] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        validate_identifier(self.simulation_id)

        if self.success_probability is not None:
            validate_probability(self.success_probability)

        if self.estimated_duration_seconds is not None:
            validate_nonnegative(
                self.estimated_duration_seconds
            )


@dataclass(frozen=True, slots=True)
class PlanningMetrics:
    task_count: int
    dependency_count: int

    estimated_duration_seconds: float
    estimated_cost: float

    parallelism_factor: float

    capability_coverage: float

    safety_score: float

    overall_score: float


@dataclass(slots=True)
class TaskPlan:
    plan_id: str
    tenant_id: str

    goal: PlanningGoal

    mode: PlanningMode

    status: PlanningStatus

    tasks: list[PlanningTask]

    constraints: list[PlanningConstraint]

    safety_assessment: PlanningSafetyAssessment

    metrics: PlanningMetrics

    simulation_result: (
        PlanningSimulationResult | None
    ) = None

    reasoning: list[str] = field(
        default_factory=list
    )

    requires_human_review: bool = False

    created_at: datetime = field(
        default_factory=utc_now
    )

    simulation_only: bool = True


# ============================================================
# Provider Protocols
# ============================================================

class TaskDecompositionProvider(Protocol):

    async def decompose_goal(
        self,
        *,
        tenant_id: str,
        goal: PlanningGoal,
        context: Mapping[str, Any],
        maximum_tasks: int,
    ) -> list[PlanningTask]:
        ...


class PlanningRobotProvider(Protocol):

    async def get_robots(
        self,
        *,
        tenant_id: str,
        robot_ids: Sequence[str],
    ) -> list[PlanningRobot]:
        ...


class PlanningSimulator(Protocol):

    async def simulate_plan(
        self,
        *,
        tenant_id: str,
        plan: TaskPlan,
    ) -> PlanningSimulationResult:
        ...


class PlannerAuthorizer(Protocol):

    async def require_permission(
        self,
        *,
        tenant_id: str,
        user_id: str,
        permission: str,
    ) -> None:
        ...


class PlannerRepository(Protocol):

    async def save_plan(
        self,
        *,
        plan: TaskPlan,
    ) -> None:
        ...


class PlannerAuditSink(Protocol):

    async def record(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        action: str,
        resource_id: str,
        metadata: Mapping[str, Any],
    ) -> None:
        ...


# ============================================================
# Task Dependency Graph
# ============================================================

class PlanningGraph:

    def __init__(
        self,
        tasks: Sequence[PlanningTask],
    ) -> None:

        self.tasks = {
            task.task_id: task
            for task in tasks
        }

        if len(self.tasks) != len(tasks):
            raise PlannerDependencyError(
                "Duplicate task IDs"
            )

        self.dependencies: dict[
            str,
            set[str],
        ] = {}

        self.dependents: dict[
            str,
            set[str],
        ] = {
            task_id: set()
            for task_id in self.tasks
        }

        for task in tasks:
            dependencies = set(task.dependencies)

            for dependency in dependencies:
                if dependency not in self.tasks:
                    raise PlannerDependencyError(
                        f"Missing dependency: {dependency}"
                    )

                self.dependents[dependency].add(
                    task.task_id
                )

            self.dependencies[task.task_id] = dependencies

        self.topological_order()

    def topological_order(self) -> list[str]:

        indegree = {
            task_id: len(dependencies)
            for task_id, dependencies
            in self.dependencies.items()
        }

        queue = sorted(
            task_id
            for task_id, degree in indegree.items()
            if degree == 0
        )

        order: list[str] = []

        while queue:
            current = queue.pop(0)
            order.append(current)

            for dependent in sorted(
                self.dependents[current]
            ):
                indegree[dependent] -= 1

                if indegree[dependent] == 0:
                    queue.append(dependent)
                    queue.sort()

        if len(order) != len(self.tasks):
            raise PlannerDependencyError(
                "Task dependency cycle detected"
            )

        return order

    def calculate_depth(self) -> int:

        depths: dict[str, int] = {}

        for task_id in self.topological_order():
            dependencies = self.dependencies[task_id]

            depths[task_id] = (
                1 + max(
                    depths[dependency]
                    for dependency in dependencies
                )
                if dependencies
                else 1
            )

        return max(depths.values(), default=0)

    def critical_path_duration(self) -> float:

        finish_times: dict[str, float] = {}

        for task_id in self.topological_order():
            task = self.tasks[task_id]

            dependency_finish = max(
                (
                    finish_times[dependency]
                    for dependency
                    in self.dependencies[task_id]
                ),
                default=0.0,
            )

            finish_times[task_id] = (
                dependency_finish
                + task.estimated_duration_seconds
            )

        return max(
            finish_times.values(),
            default=0.0,
        )

    def ready_tasks(
        self,
        completed_task_ids: set[str],
    ) -> list[PlanningTask]:

        return [
            task
            for task in self.tasks.values()
            if task.task_id not in completed_task_ids
            and self.dependencies[
                task.task_id
            ].issubset(completed_task_ids)
        ]


# ============================================================
# Planning Agent
# ============================================================

class PlannerAgent:

    PRIORITY_RANK = {
        PlanningPriority.LOW: 1,
        PlanningPriority.NORMAL: 2,
        PlanningPriority.HIGH: 3,
        PlanningPriority.CRITICAL: 4,
    }

    def __init__(
        self,
        *,
        decomposition_provider: TaskDecompositionProvider,
        robot_provider: PlanningRobotProvider,
        authorizer: PlannerAuthorizer,
        simulator: PlanningSimulator | None = None,
        repository: PlannerRepository | None = None,
        audit_sink: PlannerAuditSink | None = None,
    ) -> None:

        self.decomposition_provider = (
            decomposition_provider
        )

        self.robot_provider = robot_provider
        self.authorizer = authorizer
        self.simulator = simulator
        self.repository = repository
        self.audit_sink = audit_sink

    # --------------------------------------------------------
    # Authorization
    # --------------------------------------------------------

    async def _authorize(
        self,
        *,
        scope: PlannerScope,
    ) -> None:

        await self.authorizer.require_permission(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            permission="planning:create",
        )

    # --------------------------------------------------------
    # Goal Decomposition
    # --------------------------------------------------------

    async def decompose_goal(
        self,
        *,
        request: PlanningRequest,
    ) -> list[PlanningTask]:

        tasks = (
            await self.decomposition_provider.decompose_goal(
                tenant_id=request.tenant_id,
                goal=request.goal,
                context=copy.deepcopy(
                    request.context
                ),
                maximum_tasks=request.maximum_tasks,
            )
        )

        if not tasks:
            raise PlannerValidationError(
                "Goal decomposition returned no tasks"
            )

        if len(tasks) > request.maximum_tasks:
            raise PlannerValidationError(
                "Decomposition exceeded task limit"
            )

        graph = PlanningGraph(tasks)

        if (
            graph.calculate_depth()
            > request.maximum_planning_depth
        ):
            raise PlannerValidationError(
                "Plan exceeds maximum dependency depth"
            )

        return tasks

    # --------------------------------------------------------
    # Robot Assignment
    # --------------------------------------------------------

    @staticmethod
    def assign_robots(
        *,
        tasks: Sequence[PlanningTask],
        robots: Sequence[PlanningRobot],
        tenant_id: str,
    ) -> list[str]:

        findings: list[str] = []

        eligible_robots = [
            robot
            for robot in robots
            if robot.tenant_id == tenant_id
            and robot.available
            and robot.healthy
            and robot.simulation_supported
        ]

        for task in tasks:
            if not task.required_capabilities:
                continue

            candidates = [
                robot
                for robot in eligible_robots
                if task.required_capabilities.issubset(
                    robot.capabilities
                )
            ]

            if task.assigned_robot_id is not None:
                candidates = [
                    robot
                    for robot in candidates
                    if robot.robot_id
                    == task.assigned_robot_id
                ]

            if not candidates:
                findings.append(
                    f"Task {task.task_id} has no "
                    "eligible robot with required capabilities."
                )
                continue

            candidates.sort(
                key=lambda robot: (
                    -(
                        robot.battery_soc
                        if robot.battery_soc is not None
                        else -1.0
                    ),
                    robot.robot_id,
                )
            )

            task.assigned_robot_id = (
                candidates[0].robot_id
            )

        return findings

    # --------------------------------------------------------
    # Constraint Validation
    # --------------------------------------------------------

    @staticmethod
    def evaluate_constraints(
        *,
        tasks: Sequence[PlanningTask],
        constraints: Sequence[PlanningConstraint],
        robots: Sequence[PlanningRobot],
    ) -> tuple[list[str], list[str]]:

        violations: list[str] = []
        warnings: list[str] = []

        robot_map = {
            robot.robot_id: robot
            for robot in robots
        }

        total_cost = sum(
            task.estimated_cost
            for task in tasks
        )

        graph = PlanningGraph(tasks)

        duration = (
            graph.critical_path_duration()
        )

        for constraint in constraints:
            message: str | None = None

            if (
                constraint.constraint_type
                == ConstraintType.MAXIMUM_DURATION
            ):
                limit = validate_nonnegative(
                    float(constraint.value)
                )

                if duration > limit:
                    message = (
                        f"Estimated duration {duration:.2f}s "
                        f"exceeds {limit:.2f}s."
                    )

            elif (
                constraint.constraint_type
                == ConstraintType.MAXIMUM_COST
            ):
                limit = validate_nonnegative(
                    float(constraint.value)
                )

                if total_cost > limit:
                    message = (
                        f"Estimated cost {total_cost:.2f} "
                        f"exceeds {limit:.2f}."
                    )

            elif (
                constraint.constraint_type
                == ConstraintType.MINIMUM_BATTERY
            ):
                minimum = validate_probability(
                    float(constraint.value)
                )

                for task in tasks:
                    if task.assigned_robot_id is None:
                        continue

                    robot = robot_map.get(
                        task.assigned_robot_id
                    )

                    if (
                        robot is None
                        or robot.battery_soc is None
                        or robot.battery_soc < minimum
                    ):
                        message = (
                            f"Robot battery requirement "
                            f"not verified for task {task.task_id}."
                        )
                        break

            elif (
                constraint.constraint_type
                == ConstraintType.ALLOWED_ROBOT
            ):
                allowed = {
                    str(value)
                    for value in constraint.value
                }

                for task in tasks:
                    if (
                        task.assigned_robot_id is not None
                        and task.assigned_robot_id
                        not in allowed
                    ):
                        message = (
                            f"Task {task.task_id} uses "
                            "a robot outside the allowed set."
                        )
                        break

            elif (
                constraint.constraint_type
                == ConstraintType.FORBIDDEN_ROBOT
            ):
                forbidden = {
                    str(value)
                    for value in constraint.value
                }

                for task in tasks:
                    if task.assigned_robot_id in forbidden:
                        message = (
                            f"Task {task.task_id} uses "
                            "a forbidden robot."
                        )
                        break

            elif (
                constraint.constraint_type
                == ConstraintType.REQUIRED_CAPABILITY
            ):
                required = str(
                    constraint.value
                )

                for task in tasks:
                    if required not in task.required_capabilities:
                        message = (
                            f"Task {task.task_id} does not "
                            f"declare capability {required}."
                        )
                        break

            elif (
                constraint.constraint_type
                == ConstraintType.HUMAN_APPROVAL
            ):
                if not all(
                    task.requires_human_review
                    for task in tasks
                ):
                    message = (
                        "One or more tasks do not "
                        "require human review."
                    )

            elif (
                constraint.constraint_type
                == ConstraintType.SIMULATION_REQUIRED
            ):
                if not bool(constraint.value):
                    message = (
                        "Simulation requirement cannot "
                        "be disabled by a constraint."
                    )

            elif (
                constraint.constraint_type
                == ConstraintType.CUSTOM
            ):
                message = (
                    "Custom constraint requires an "
                    "external validator."
                )

            if message is not None:
                if (
                    constraint.severity
                    == ConstraintSeverity.HARD
                ):
                    violations.append(message)
                else:
                    warnings.append(message)

        return violations, warnings

    # --------------------------------------------------------
    # Safety Assessment
    # --------------------------------------------------------

    @staticmethod
    def assess_safety(
        *,
        tasks: Sequence[PlanningTask],
        capability_findings: Sequence[str],
        constraint_violations: Sequence[str],
        constraint_warnings: Sequence[str],
    ) -> PlanningSafetyAssessment:

        violations = list(
            constraint_violations
        )

        violations.extend(
            capability_findings
        )

        warnings = list(
            constraint_warnings
        )

        requires_review = False

        for task in tasks:
            if (
                task.safety_class
                == PlanningSafetyClass.PHYSICAL_ACTUATION
            ):
                violations.append(
                    f"Task {task.task_id} requests "
                    "physical actuation."
                )

            if (
                task.safety_class
                == PlanningSafetyClass.SAFETY_CRITICAL
            ):
                requires_review = True
                warnings.append(
                    f"Task {task.task_id} is safety-critical."
                )

            if task.requires_human_review:
                requires_review = True

        if violations:
            return PlanningSafetyAssessment(
                risk_level=PlanningRiskLevel.CRITICAL,
                valid=False,
                requires_human_review=True,
                violations=tuple(violations),
                warnings=tuple(warnings),
            )

        if requires_review:
            return PlanningSafetyAssessment(
                risk_level=PlanningRiskLevel.HIGH,
                valid=True,
                requires_human_review=True,
                violations=(),
                warnings=tuple(warnings),
            )

        if warnings:
            return PlanningSafetyAssessment(
                risk_level=PlanningRiskLevel.MODERATE,
                valid=True,
                requires_human_review=False,
                violations=(),
                warnings=tuple(warnings),
            )

        return PlanningSafetyAssessment(
            risk_level=PlanningRiskLevel.LOW,
            valid=True,
            requires_human_review=False,
            violations=(),
            warnings=(),
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    @staticmethod
    def calculate_metrics(
        *,
        tasks: Sequence[PlanningTask],
        safety: PlanningSafetyAssessment,
    ) -> PlanningMetrics:

        graph = PlanningGraph(tasks)

        task_count = len(tasks)

        dependency_count = sum(
            len(task.dependencies)
            for task in tasks
        )

        serial_duration = sum(
            task.estimated_duration_seconds
            for task in tasks
        )

        critical_duration = (
            graph.critical_path_duration()
        )

        estimated_cost = sum(
            task.estimated_cost
            for task in tasks
        )

        parallelism = (
            serial_duration / critical_duration
            if critical_duration > 0
            else 1.0
        )

        tasks_requiring_robots = [
            task
            for task in tasks
            if task.required_capabilities
        ]

        if tasks_requiring_robots:
            assigned_count = sum(
                task.assigned_robot_id is not None
                for task in tasks_requiring_robots
            )

            capability_coverage = (
                assigned_count
                / len(tasks_requiring_robots)
            )
        else:
            capability_coverage = 1.0

        safety_score = {
            PlanningRiskLevel.LOW: 1.0,
            PlanningRiskLevel.MODERATE: 0.75,
            PlanningRiskLevel.HIGH: 0.40,
            PlanningRiskLevel.CRITICAL: 0.0,
            PlanningRiskLevel.UNKNOWN: 0.0,
        }[safety.risk_level]

        efficiency_score = clamp(
            parallelism / 4.0
        )

        overall_score = (
            0.45 * capability_coverage
            + 0.35 * safety_score
            + 0.20 * efficiency_score
        )

        if not safety.valid:
            overall_score = 0.0

        return PlanningMetrics(
            task_count=task_count,
            dependency_count=dependency_count,
            estimated_duration_seconds=critical_duration,
            estimated_cost=estimated_cost,
            parallelism_factor=parallelism,
            capability_coverage=capability_coverage,
            safety_score=safety_score,
            overall_score=overall_score,
        )

    # --------------------------------------------------------
    # Create Plan
    # --------------------------------------------------------

    async def create_plan(
        self,
        *,
        scope: PlannerScope,
        request: PlanningRequest,
    ) -> TaskPlan:

        if request.tenant_id != scope.tenant_id:
            raise PlannerPermissionError(
                "Tenant mismatch"
            )

        await self._authorize(
            scope=scope
        )

        tasks = await self.decompose_goal(
            request=request
        )

        robots = await self.robot_provider.get_robots(
            tenant_id=scope.tenant_id,
            robot_ids=request.available_robot_ids,
        )

        for robot in robots:
            if robot.tenant_id != scope.tenant_id:
                raise PlannerPermissionError(
                    "Robot provider returned cross-tenant data"
                )

        capability_findings = self.assign_robots(
            tasks=tasks,
            robots=robots,
            tenant_id=scope.tenant_id,
        )

        violations, warnings = (
            self.evaluate_constraints(
                tasks=tasks,
                constraints=request.constraints,
                robots=robots,
            )
        )

        safety = self.assess_safety(
            tasks=tasks,
            capability_findings=capability_findings,
            constraint_violations=violations,
            constraint_warnings=warnings,
        )

        metrics = self.calculate_metrics(
            tasks=tasks,
            safety=safety,
        )

        reasoning = [
            f"Goal decomposed into {len(tasks)} tasks.",
            (
                "Critical-path estimated duration: "
                f"{metrics.estimated_duration_seconds:.2f}s."
            ),
            (
                "Capability coverage: "
                f"{metrics.capability_coverage:.2%}."
            ),
            (
                "Planning safety risk: "
                f"{safety.risk_level.value}."
            ),
        ]

        reasoning.extend(safety.violations)
        reasoning.extend(safety.warnings)

        if not safety.valid:
            status = PlanningStatus.REJECTED

        elif safety.requires_human_review:
            status = PlanningStatus.REVIEW_REQUIRED

        else:
            status = PlanningStatus.VALIDATED

        plan = TaskPlan(
            plan_id=generate_id(),
            tenant_id=scope.tenant_id,
            goal=request.goal,
            mode=request.mode,
            status=status,
            tasks=tasks,
            constraints=copy.deepcopy(
                request.constraints
            ),
            safety_assessment=safety,
            metrics=metrics,
            reasoning=reasoning,
            requires_human_review=(
                safety.requires_human_review
            ),
            simulation_only=True,
        )

        if (
            request.require_simulation
            and safety.valid
        ):
            if self.simulator is None:
                plan.status = (
                    PlanningStatus.REVIEW_REQUIRED
                )

                plan.requires_human_review = True

                plan.reasoning.append(
                    "Simulation provider unavailable."
                )

            else:
                simulation = (
                    await self.simulator.simulate_plan(
                        tenant_id=scope.tenant_id,
                        plan=plan,
                    )
                )

                plan.simulation_result = simulation

                if (
                    simulation.status
                    == PlanningSimulationStatus.PASSED
                    and not safety.requires_human_review
                ):
                    plan.status = (
                        PlanningStatus.SIMULATION_PASSED
                    )

                elif (
                    simulation.status
                    == PlanningSimulationStatus.FAILED
                ):
                    plan.status = (
                        PlanningStatus.SIMULATION_FAILED
                    )

                    plan.requires_human_review = True

                else:
                    plan.status = (
                        PlanningStatus.REVIEW_REQUIRED
                    )

                    plan.requires_human_review = True

        if self.repository is not None:
            await self.repository.save_plan(
                plan=plan
            )

        if self.audit_sink is not None:
            await self.audit_sink.record(
                tenant_id=scope.tenant_id,
                actor_id=scope.user_id,
                action="planner.plan_created",
                resource_id=plan.plan_id,
                metadata={
                    "goal_id": request.goal.goal_id,
                    "status": plan.status.value,
                    "task_count": len(tasks),
                    "risk_level": (
                        safety.risk_level.value
                    ),
                    "simulation_only": True,
                },
            )

        return plan

    # --------------------------------------------------------
    # Replanning
    # --------------------------------------------------------

    async def replan(
        self,
        *,
        scope: PlannerScope,
        previous_plan: TaskPlan,
        updated_context: Mapping[str, Any],
        available_robot_ids: Sequence[str],
    ) -> TaskPlan:

        if previous_plan.tenant_id != scope.tenant_id:
            raise PlannerPermissionError(
                "Previous plan tenant mismatch"
            )

        request = PlanningRequest(
            tenant_id=scope.tenant_id,
            goal=previous_plan.goal,
            mode=PlanningMode.REPLANNING,
            constraints=copy.deepcopy(
                previous_plan.constraints
            ),
            available_robot_ids=list(
                available_robot_ids
            ),
            context=dict(updated_context),
            require_simulation=True,
            simulation_only=True,
        )

        return await self.create_plan(
            scope=scope,
            request=request,
        )

    # --------------------------------------------------------
    # Batch Planning
    # --------------------------------------------------------

    async def create_batch_plans(
        self,
        *,
        scope: PlannerScope,
        requests: Sequence[PlanningRequest],
    ) -> list[TaskPlan]:

        if len(requests) > MAX_BATCH_PLANS:
            raise PlannerValidationError(
                "Batch planning limit exceeded"
            )

        semaphore = asyncio.Semaphore(4)

        async def process(
            request: PlanningRequest,
        ) -> TaskPlan:

            async with semaphore:
                return await self.create_plan(
                    scope=scope,
                    request=request,
                )

        return list(
            await asyncio.gather(
                *(process(request) for request in requests)
            )
        )

    # --------------------------------------------------------
    # Explainable Planning Report
    # --------------------------------------------------------

    @staticmethod
    def generate_plan_report(
        plan: TaskPlan,
    ) -> dict[str, Any]:

        graph = PlanningGraph(plan.tasks)

        return {
            "plan_id": plan.plan_id,
            "tenant_id": plan.tenant_id,
            "goal": {
                "goal_id": plan.goal.goal_id,
                "description": plan.goal.description,
                "domain": plan.goal.domain.value,
                "target_robot_id": (
                    plan.goal.target_robot_id
                ),
                "success_criteria": list(
                    plan.goal.success_criteria
                ),
            },
            "mode": plan.mode.value,
            "status": plan.status.value,
            "created_at": plan.created_at.isoformat(),
            "task_execution_order": (
                graph.topological_order()
            ),
            "tasks": [
                {
                    "task_id": task.task_id,
                    "name": task.name,
                    "description": task.description,
                    "task_type": task.task_type.value,
                    "dependencies": task.dependencies,
                    "required_capabilities": sorted(
                        task.required_capabilities
                    ),
                    "assigned_robot_id": (
                        task.assigned_robot_id
                    ),
                    "estimated_duration_seconds": (
                        task.estimated_duration_seconds
                    ),
                    "estimated_cost": task.estimated_cost,
                    "priority": task.priority.value,
                    "safety_class": (
                        task.safety_class.value
                    ),
                    "requires_human_review": (
                        task.requires_human_review
                    ),
                    "status": task.status.value,
                }
                for task in plan.tasks
            ],
            "metrics": {
                "task_count": plan.metrics.task_count,
                "dependency_count": (
                    plan.metrics.dependency_count
                ),
                "estimated_duration_seconds": (
                    plan.metrics.estimated_duration_seconds
                ),
                "estimated_cost": (
                    plan.metrics.estimated_cost
                ),
                "parallelism_factor": (
                    plan.metrics.parallelism_factor
                ),
                "capability_coverage": (
                    plan.metrics.capability_coverage
                ),
                "safety_score": (
                    plan.metrics.safety_score
                ),
                "overall_score": (
                    plan.metrics.overall_score
                ),
            },
            "safety": {
                "risk_level": (
                    plan.safety_assessment.risk_level.value
                ),
                "valid": plan.safety_assessment.valid,
                "requires_human_review": (
                    plan.requires_human_review
                ),
                "violations": list(
                    plan.safety_assessment.violations
                ),
                "warnings": list(
                    plan.safety_assessment.warnings
                ),
            },
            "simulation": (
                {
                    "simulation_id": (
                        plan.simulation_result.simulation_id
                    ),
                    "status": (
                        plan.simulation_result.status.value
                    ),
                    "success_probability": (
                        plan.simulation_result.success_probability
                    ),
                    "estimated_duration_seconds": (
                        plan.simulation_result
                        .estimated_duration_seconds
                    ),
                    "findings": (
                        plan.simulation_result.findings
                    ),
                }
                if plan.simulation_result is not None
                else None
            ),
            "reasoning": plan.reasoning,
            "simulation_only": True,
            "physical_execution_enabled": False,
        }


# ============================================================
# Factory
# ============================================================

def create_planner_agent(
    *,
    decomposition_provider: TaskDecompositionProvider,
    robot_provider: PlanningRobotProvider,
    authorizer: PlannerAuthorizer,
    simulator: PlanningSimulator | None = None,
    repository: PlannerRepository | None = None,
    audit_sink: PlannerAuditSink | None = None,
) -> PlannerAgent:

    return PlannerAgent(
        decomposition_provider=decomposition_provider,
        robot_provider=robot_provider,
        authorizer=authorizer,
        simulator=simulator,
        repository=repository,
        audit_sink=audit_sink,
    )


# ============================================================
# Exports
# ============================================================

__all__ = [
    "PlannerError",
    "PlannerValidationError",
    "PlannerPermissionError",
    "PlannerNotFoundError",
    "PlannerDependencyError",
    "PlannerCapabilityError",
    "PlannerSafetyError",
    "PlannerServiceUnavailableError",
    "PlanningMode",
    "PlanningDomain",
    "PlanningTaskType",
    "PlanningStatus",
    "PlanningTaskStatus",
    "PlanningPriority",
    "PlanningRiskLevel",
    "PlanningSafetyClass",
    "ConstraintType",
    "ConstraintSeverity",
    "PlanningSimulationStatus",
    "PlannerScope",
    "PlanningGoal",
    "PlanningConstraint",
    "PlanningRobot",
    "PlanningTask",
    "PlanningRequest",
    "PlanningSafetyAssessment",
    "PlanningSimulationResult",
    "PlanningMetrics",
    "TaskPlan",
    "TaskDecompositionProvider",
    "PlanningRobotProvider",
    "PlanningSimulator",
    "PlannerAuthorizer",
    "PlannerRepository",
    "PlannerAuditSink",
    "PlanningGraph",
    "PlannerAgent",
    "create_planner_agent",
]
"""Orchestration module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/ai/agents/orchestration.py

Physical AI & Autonomous Robotics Intelligence Platform
Multi-Agent Orchestration Engine

Python 3.11+

Features:
- Multi-agent task orchestration
- Supervisor and specialized-agent coordination
- Agent registry integration
- DAG-based workflow planning
- Dependency-aware task scheduling
- Parallel agent execution
- Sequential workflow execution
- Task prioritization
- Agent capability matching
- Timeout and retry management
- Failure handling and recovery
- Human-in-the-loop approval gates
- Safety-critical task restrictions
- Tenant-aware authorization
- Workflow state management
- Agent execution monitoring
- Audit logging
- Execution history
- Structured workflow results

Safety:
- No direct robot actuation.
- No automatic emergency-stop reset.
- No bypass of physical safety controllers.
- Human approval does not replace independent
  hardware safety validation.
- Robot-control, navigation, and manipulation
  tasks remain advisory or simulation-only.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import logging
import math
import re
import uuid

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence


logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

MAX_WORKFLOW_TASKS = 200
MAX_TASK_DEPENDENCIES = 50
MAX_CONCURRENT_TASKS = 10
MAX_TASK_RETRIES = 3
MAX_TASK_TIMEOUT_SECONDS = 600.0
MAX_WORKFLOW_DEPTH = 100

DEFAULT_TASK_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_CONCURRENCY = 5

IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z0-9_.:-]{1,128}$"
)


# ============================================================
# Exceptions
# ============================================================

class OrchestrationError(Exception):
    """Base orchestration exception."""


class OrchestrationValidationError(OrchestrationError):
    """Invalid workflow or task configuration."""


class WorkflowNotFoundError(OrchestrationError):
    """Workflow does not exist."""


class AgentNotAvailableError(OrchestrationError):
    """No suitable agent is available."""


class WorkflowConflictError(OrchestrationError):
    """Workflow state conflict."""


class WorkflowPermissionError(OrchestrationError):
    """Workflow authorization failure."""


class WorkflowDependencyError(OrchestrationError):
    """Invalid workflow dependency graph."""


class WorkflowExecutionError(OrchestrationError):
    """Workflow execution failed."""


class HumanApprovalRequiredError(OrchestrationError):
    """Task requires human authorization."""


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
        raise OrchestrationValidationError(
            f"Invalid identifier: {value!r}"
        )

    return value


def validate_positive(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise OrchestrationValidationError(
            "Value must be finite and positive"
        )

    return value


def validate_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise OrchestrationValidationError(
            "Datetime must be timezone-aware"
        )

    return value


# ============================================================
# Enumerations
# ============================================================

class AgentRole(str, Enum):
    SUPERVISOR = "supervisor"
    PLANNER = "planner"
    RESEARCH = "research"
    PERCEPTION = "perception"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    ROBOT_CONTROL = "robot_control"
    WORLD_MODEL = "world_model"
    IMAGINATION = "imagination"
    VLA = "vla"
    SAFETY = "safety"
    HEALTH = "health"
    MAINTENANCE = "maintenance"
    DIAGNOSTICS = "diagnostics"
    REPAIR = "repair"
    COST = "cost"
    FLEET = "fleet"
    SIMULATION = "simulation"
    TRAINING = "training"
    EVALUATION = "evaluation"
    RAG = "rag"
    MEMORY = "memory"
    ANALYTICS = "analytics"
    REPORTING = "reporting"
    CUSTOM = "custom"


class WorkflowStatus(str, Enum):
    CREATED = "created"
    VALIDATED = "validated"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ExecutionMode(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DAG = "dag"


class FailurePolicy(str, Enum):
    FAIL_FAST = "fail_fast"
    CONTINUE = "continue"


class TaskSafetyClass(str, Enum):
    READ_ONLY = "read_only"
    ADVISORY = "advisory"
    SIMULATION = "simulation"
    SAFETY_CRITICAL = "safety_critical"
    PHYSICAL_ACTUATION = "physical_actuation"


class ApprovalStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


# ============================================================
# Security Scope
# ============================================================

@dataclass(frozen=True, slots=True)
class OrchestrationScope:
    tenant_id: str
    user_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_identifier(self.user_id)


# ============================================================
# Agent Metadata
# ============================================================

@dataclass(frozen=True, slots=True)
class AgentDescriptor:
    agent_id: str
    tenant_id: str
    role: AgentRole

    capabilities: frozenset[str]

    available: bool = True
    healthy: bool = True

    priority: int = 0
    active_tasks: int = 0
    maximum_concurrent_tasks: int = 1

    def __post_init__(self) -> None:
        validate_identifier(self.agent_id)
        validate_identifier(self.tenant_id)

        if self.maximum_concurrent_tasks < 1:
            raise OrchestrationValidationError(
                "Agent capacity must be positive"
            )

        if not (
            0 <= self.active_tasks
            <= self.maximum_concurrent_tasks
        ):
            raise OrchestrationValidationError(
                "Invalid active-task count"
            )


# ============================================================
# Task Models
# ============================================================

@dataclass(slots=True)
class OrchestrationTask:
    task_id: str
    name: str

    agent_role: AgentRole

    input_data: dict[str, Any]

    dependencies: list[str] = field(
        default_factory=list
    )

    required_capabilities: set[str] = field(
        default_factory=set
    )

    priority: TaskPriority = TaskPriority.NORMAL

    safety_class: TaskSafetyClass = (
        TaskSafetyClass.ADVISORY
    )

    requires_human_approval: bool = False

    timeout_seconds: float = (
        DEFAULT_TASK_TIMEOUT_SECONDS
    )

    max_retries: int = 0

    status: TaskStatus = TaskStatus.PENDING

    assigned_agent_id: str | None = None

    approval_id: str | None = None

    approval_status: ApprovalStatus = (
        ApprovalStatus.NOT_REQUIRED
    )

    result: dict[str, Any] | None = None

    error: str | None = None

    attempts: int = 0

    started_at: datetime | None = None
    completed_at: datetime | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.task_id)

        if not self.name.strip():
            raise OrchestrationValidationError(
                "Task name is required"
            )

        if len(self.dependencies) > MAX_TASK_DEPENDENCIES:
            raise OrchestrationValidationError(
                "Too many task dependencies"
            )

        for dependency in self.dependencies:
            validate_identifier(dependency)

        if len(self.dependencies) != len(
            set(self.dependencies)
        ):
            raise OrchestrationValidationError(
                "Duplicate task dependencies"
            )

        if self.task_id in self.dependencies:
            raise WorkflowDependencyError(
                "Task cannot depend on itself"
            )

        validate_positive(self.timeout_seconds)

        if self.timeout_seconds > MAX_TASK_TIMEOUT_SECONDS:
            raise OrchestrationValidationError(
                "Task timeout exceeds limit"
            )

        if not 0 <= self.max_retries <= MAX_TASK_RETRIES:
            raise OrchestrationValidationError(
                "Invalid retry count"
            )

        if (
            self.safety_class
            == TaskSafetyClass.PHYSICAL_ACTUATION
        ):
            raise OrchestrationValidationError(
                "Physical actuation cannot be executed "
                "through this orchestration engine"
            )

        if (
            self.safety_class
            == TaskSafetyClass.SAFETY_CRITICAL
        ):
            self.requires_human_approval = True

        if self.requires_human_approval:
            self.approval_status = ApprovalStatus.PENDING


@dataclass(slots=True)
class OrchestrationWorkflow:
    workflow_id: str
    tenant_id: str
    created_by: str

    name: str
    description: str

    tasks: list[OrchestrationTask]

    execution_mode: ExecutionMode = ExecutionMode.DAG

    failure_policy: FailurePolicy = (
        FailurePolicy.FAIL_FAST
    )

    max_concurrency: int = (
        DEFAULT_MAX_CONCURRENCY
    )

    status: WorkflowStatus = WorkflowStatus.CREATED

    created_at: datetime = field(
        default_factory=utc_now
    )

    started_at: datetime | None = None
    completed_at: datetime | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_identifier(self.workflow_id)
        validate_identifier(self.tenant_id)
        validate_identifier(self.created_by)

        if not self.name.strip():
            raise OrchestrationValidationError(
                "Workflow name is required"
            )

        if len(self.tasks) > MAX_WORKFLOW_TASKS:
            raise OrchestrationValidationError(
                "Too many workflow tasks"
            )

        if not (
            1 <= self.max_concurrency
            <= MAX_CONCURRENT_TASKS
        ):
            raise OrchestrationValidationError(
                "Invalid workflow concurrency"
            )


@dataclass(frozen=True, slots=True)
class AgentExecutionContext:
    workflow_id: str
    task_id: str

    tenant_id: str
    user_id: str

    agent_id: str

    input_data: Mapping[str, Any]

    dependency_results: Mapping[
        str,
        Mapping[str, Any],
    ]

    simulation_only: bool = True


@dataclass(slots=True)
class TaskExecutionResult:
    task_id: str
    agent_id: str | None

    status: TaskStatus

    output: dict[str, Any] | None

    attempts: int

    duration_seconds: float

    error: str | None = None


@dataclass(slots=True)
class WorkflowExecutionResult:
    workflow_id: str
    tenant_id: str

    status: WorkflowStatus

    completed_tasks: int
    failed_tasks: int
    skipped_tasks: int
    pending_tasks: int

    task_results: list[TaskExecutionResult]

    started_at: datetime | None
    completed_at: datetime | None


# ============================================================
# Integration Protocols
# ============================================================

class AgentRegistryProvider(Protocol):

    async def find_agents(
        self,
        *,
        tenant_id: str,
        role: AgentRole,
        required_capabilities: set[str],
    ) -> list[AgentDescriptor]:
        ...


class AgentExecutor(Protocol):

    async def execute(
        self,
        context: AgentExecutionContext,
    ) -> Mapping[str, Any]:
        ...


class OrchestrationRepository(Protocol):

    async def save_workflow(
        self,
        *,
        workflow: OrchestrationWorkflow,
    ) -> None:
        ...

    async def get_workflow(
        self,
        *,
        tenant_id: str,
        workflow_id: str,
    ) -> OrchestrationWorkflow | None:
        ...


class OrchestrationAuthorizer(Protocol):

    async def require_permission(
        self,
        *,
        tenant_id: str,
        user_id: str,
        permission: str,
    ) -> None:
        ...


class OrchestrationApprovalProvider(Protocol):

    async def request_approval(
        self,
        *,
        tenant_id: str,
        user_id: str,
        workflow_id: str,
        task_id: str,
        reason: str,
    ) -> str:
        ...

    async def get_approval_status(
        self,
        *,
        tenant_id: str,
        approval_id: str,
    ) -> ApprovalStatus:
        ...


class OrchestrationAuditSink(Protocol):

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
# Dependency Graph
# ============================================================

class WorkflowGraph:

    def __init__(
        self,
        tasks: Sequence[OrchestrationTask],
    ) -> None:

        self.tasks = {
            task.task_id: task
            for task in tasks
        }

        if len(self.tasks) != len(tasks):
            raise WorkflowDependencyError(
                "Duplicate task identifiers"
            )

        self.dependencies: dict[
            str,
            set[str],
        ] = {}

        self.dependents: dict[
            str,
            set[str],
        ] = defaultdict(set)

        for task in tasks:
            dependencies = set(task.dependencies)

            for dependency in dependencies:
                if dependency not in self.tasks:
                    raise WorkflowDependencyError(
                        f"Missing dependency: {dependency}"
                    )

                self.dependents[dependency].add(
                    task.task_id
                )

            self.dependencies[task.task_id] = dependencies

        self._validate_acyclic()

    def _validate_acyclic(self) -> None:

        indegree = {
            task_id: len(dependencies)
            for task_id, dependencies
            in self.dependencies.items()
        }

        queue = [
            task_id
            for task_id, degree in indegree.items()
            if degree == 0
        ]

        visited = 0

        while queue:
            current = queue.pop()
            visited += 1

            for dependent in self.dependents[current]:
                indegree[dependent] -= 1

                if indegree[dependent] == 0:
                    queue.append(dependent)

        if visited != len(self.tasks):
            raise WorkflowDependencyError(
                "Workflow contains a dependency cycle"
            )

    def ready_tasks(
        self,
        completed: set[str],
        terminal: set[str],
    ) -> list[OrchestrationTask]:

        ready: list[OrchestrationTask] = []

        for task_id, task in self.tasks.items():
            if task_id in terminal:
                continue

            if self.dependencies[task_id].issubset(
                completed
            ):
                ready.append(task)

        return ready

    def blocked_tasks(
        self,
        failed: set[str],
    ) -> set[str]:

        blocked: set[str] = set()
        stack = list(failed)

        while stack:
            current = stack.pop()

            for dependent in self.dependents[current]:
                if dependent not in blocked:
                    blocked.add(dependent)
                    stack.append(dependent)

        return blocked


# ============================================================
# Orchestration Engine
# ============================================================

class MultiAgentOrchestrator:

    PRIORITY_RANK = {
        TaskPriority.LOW: 1,
        TaskPriority.NORMAL: 2,
        TaskPriority.HIGH: 3,
        TaskPriority.CRITICAL: 4,
    }

    def __init__(
        self,
        *,
        agent_registry: AgentRegistryProvider,
        agent_executor: AgentExecutor,
        authorizer: OrchestrationAuthorizer,
        repository: OrchestrationRepository | None = None,
        approval_provider: (
            OrchestrationApprovalProvider | None
        ) = None,
        audit_sink: OrchestrationAuditSink | None = None,
    ) -> None:

        self.agent_registry = agent_registry
        self.agent_executor = agent_executor
        self.authorizer = authorizer

        self.repository = repository
        self.approval_provider = approval_provider
        self.audit_sink = audit_sink

        self._workflow_locks: dict[
            str,
            asyncio.Lock,
        ] = {}

        self._active_workflows: set[str] = set()

    # --------------------------------------------------------
    # Authorization
    # --------------------------------------------------------

    async def _authorize(
        self,
        *,
        scope: OrchestrationScope,
        permission: str,
    ) -> None:

        await self.authorizer.require_permission(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            permission=permission,
        )

    # --------------------------------------------------------
    # Audit
    # --------------------------------------------------------

    async def _audit(
        self,
        *,
        scope: OrchestrationScope,
        action: str,
        resource_id: str,
        metadata: Mapping[str, Any],
    ) -> None:

        if self.audit_sink is None:
            return

        await self.audit_sink.record(
            tenant_id=scope.tenant_id,
            actor_id=scope.user_id,
            action=action,
            resource_id=resource_id,
            metadata=metadata,
        )

    # --------------------------------------------------------
    # Workflow Validation
    # --------------------------------------------------------

    @staticmethod
    def validate_workflow(
        workflow: OrchestrationWorkflow,
    ) -> WorkflowGraph:

        graph = WorkflowGraph(workflow.tasks)

        if workflow.execution_mode == ExecutionMode.SEQUENTIAL:
            # Sequential mode uses the task list order,
            # while still respecting declared dependencies.
            positions = {
                task.task_id: index
                for index, task in enumerate(
                    workflow.tasks
                )
            }

            for task in workflow.tasks:
                for dependency in task.dependencies:
                    if positions[dependency] >= positions[
                        task.task_id
                    ]:
                        raise WorkflowDependencyError(
                            "Sequential dependencies must "
                            "refer to earlier tasks"
                        )

        return graph

    # --------------------------------------------------------
    # Agent Selection
    # --------------------------------------------------------

    async def select_agent(
        self,
        *,
        tenant_id: str,
        task: OrchestrationTask,
    ) -> AgentDescriptor:

        agents = await self.agent_registry.find_agents(
            tenant_id=tenant_id,
            role=task.agent_role,
            required_capabilities=(
                task.required_capabilities
            ),
        )

        eligible = [
            agent
            for agent in agents
            if agent.tenant_id == tenant_id
            and agent.available
            and agent.healthy
            and agent.active_tasks
            < agent.maximum_concurrent_tasks
            and task.required_capabilities.issubset(
                agent.capabilities
            )
        ]

        if not eligible:
            raise AgentNotAvailableError(
                f"No available agent for role "
                f"{task.agent_role.value}"
            )

        eligible.sort(
            key=lambda agent: (
                agent.active_tasks
                / agent.maximum_concurrent_tasks,
                -agent.priority,
                agent.agent_id,
            )
        )

        return eligible[0]

    # --------------------------------------------------------
    # Human Approval
    # --------------------------------------------------------

    async def ensure_approval(
        self,
        *,
        scope: OrchestrationScope,
        workflow: OrchestrationWorkflow,
        task: OrchestrationTask,
    ) -> bool:

        if not task.requires_human_approval:
            return True

        if task.approval_status == ApprovalStatus.APPROVED:
            return True

        if task.approval_status == ApprovalStatus.REJECTED:
            task.status = TaskStatus.FAILED
            task.error = "Human approval rejected"
            return False

        if self.approval_provider is None:
            task.status = TaskStatus.WAITING_APPROVAL
            task.error = (
                "Human approval provider is unavailable"
            )
            return False

        if task.approval_id is None:
            approval_id = (
                await self.approval_provider.request_approval(
                    tenant_id=scope.tenant_id,
                    user_id=scope.user_id,
                    workflow_id=workflow.workflow_id,
                    task_id=task.task_id,
                    reason=(
                        f"Approval required for "
                        f"{task.safety_class.value} task"
                    ),
                )
            )

            validate_identifier(approval_id)

            task.approval_id = approval_id
            task.approval_status = ApprovalStatus.PENDING
            task.status = TaskStatus.WAITING_APPROVAL

            return False

        status = (
            await self.approval_provider.get_approval_status(
                tenant_id=scope.tenant_id,
                approval_id=task.approval_id,
            )
        )

        task.approval_status = status

        if status == ApprovalStatus.APPROVED:
            return True

        if status == ApprovalStatus.REJECTED:
            task.status = TaskStatus.FAILED
            task.error = "Human approval rejected"
            return False

        task.status = TaskStatus.WAITING_APPROVAL
        return False

    # --------------------------------------------------------
    # Execute Single Task
    # --------------------------------------------------------

    async def execute_task(
        self,
        *,
        scope: OrchestrationScope,
        workflow: OrchestrationWorkflow,
        task: OrchestrationTask,
        dependency_results: Mapping[
            str,
            Mapping[str, Any],
        ],
    ) -> TaskExecutionResult:

        started = utc_now()

        if (
            task.safety_class
            == TaskSafetyClass.PHYSICAL_ACTUATION
        ):
            task.status = TaskStatus.FAILED
            task.error = (
                "Physical actuation is prohibited"
            )

            return TaskExecutionResult(
                task_id=task.task_id,
                agent_id=None,
                status=task.status,
                output=None,
                attempts=0,
                duration_seconds=0.0,
                error=task.error,
            )

        approved = await self.ensure_approval(
            scope=scope,
            workflow=workflow,
            task=task,
        )

        if not approved:
            return TaskExecutionResult(
                task_id=task.task_id,
                agent_id=None,
                status=task.status,
                output=None,
                attempts=task.attempts,
                duration_seconds=0.0,
                error=task.error,
            )

        task.status = TaskStatus.RUNNING
        task.started_at = started
        task.error = None

        last_error: str | None = None

        for attempt in range(
            1,
            task.max_retries + 2,
        ):
            task.attempts = attempt

            try:
                agent = await self.select_agent(
                    tenant_id=scope.tenant_id,
                    task=task,
                )

                task.assigned_agent_id = agent.agent_id

                context = AgentExecutionContext(
                    workflow_id=workflow.workflow_id,
                    task_id=task.task_id,
                    tenant_id=scope.tenant_id,
                    user_id=scope.user_id,
                    agent_id=agent.agent_id,
                    input_data=copy.deepcopy(
                        task.input_data
                    ),
                    dependency_results=copy.deepcopy(
                        dict(dependency_results)
                    ),
                    simulation_only=True,
                )

                output = await asyncio.wait_for(
                    self.agent_executor.execute(
                        context
                    ),
                    timeout=task.timeout_seconds,
                )

                if not isinstance(output, Mapping):
                    raise WorkflowExecutionError(
                        "Agent output must be a mapping"
                    )

                task.result = dict(output)
                task.status = TaskStatus.COMPLETED
                task.completed_at = utc_now()

                duration = (
                    task.completed_at - started
                ).total_seconds()

                await self._audit(
                    scope=scope,
                    action="orchestration.task_completed",
                    resource_id=task.task_id,
                    metadata={
                        "workflow_id": workflow.workflow_id,
                        "agent_id": agent.agent_id,
                        "attempts": attempt,
                        "duration_seconds": duration,
                    },
                )

                return TaskExecutionResult(
                    task_id=task.task_id,
                    agent_id=agent.agent_id,
                    status=TaskStatus.COMPLETED,
                    output=task.result,
                    attempts=attempt,
                    duration_seconds=duration,
                )

            except asyncio.CancelledError:
                task.status = TaskStatus.CANCELLED
                task.completed_at = utc_now()
                raise

            except (
                AgentNotAvailableError,
                asyncio.TimeoutError,
                WorkflowExecutionError,
            ) as exc:
                last_error = str(exc)

            except Exception as exc:
                logger.exception(
                    "Agent task failed: %s",
                    task.task_id,
                )
                last_error = (
                    f"{type(exc).__name__}: {exc}"
                )

            if attempt <= task.max_retries:
                await asyncio.sleep(
                    min(2 ** (attempt - 1), 8)
                )

        task.status = TaskStatus.FAILED
        task.error = last_error
        task.completed_at = utc_now()

        duration = (
            task.completed_at - started
        ).total_seconds()

        await self._audit(
            scope=scope,
            action="orchestration.task_failed",
            resource_id=task.task_id,
            metadata={
                "workflow_id": workflow.workflow_id,
                "attempts": task.attempts,
                "error": last_error,
            },
        )

        return TaskExecutionResult(
            task_id=task.task_id,
            agent_id=task.assigned_agent_id,
            status=TaskStatus.FAILED,
            output=None,
            attempts=task.attempts,
            duration_seconds=duration,
            error=last_error,
        )

    # --------------------------------------------------------
    # Workflow Execution
    # --------------------------------------------------------

    async def execute_workflow(
        self,
        *,
        scope: OrchestrationScope,
        workflow: OrchestrationWorkflow,
    ) -> WorkflowExecutionResult:

        if workflow.tenant_id != scope.tenant_id:
            raise WorkflowPermissionError(
                "Workflow tenant mismatch"
            )

        await self._authorize(
            scope=scope,
            permission="orchestration:execute",
        )

        graph = self.validate_workflow(workflow)

        if workflow.status in (
            WorkflowStatus.COMPLETED,
            WorkflowStatus.CANCELLED,
        ):
            raise WorkflowConflictError(
                "Workflow is already terminal"
            )

        lock_key = (
            f"{scope.tenant_id}:{workflow.workflow_id}"
        )

        lock = self._workflow_locks.setdefault(
            lock_key,
            asyncio.Lock(),
        )

        if lock.locked():
            raise WorkflowConflictError(
                "Workflow is already executing"
            )

        async with lock:
            return await self._run_workflow(
                scope=scope,
                workflow=workflow,
                graph=graph,
            )

    async def _run_workflow(
        self,
        *,
        scope: OrchestrationScope,
        workflow: OrchestrationWorkflow,
        graph: WorkflowGraph,
    ) -> WorkflowExecutionResult:

        workflow.status = WorkflowStatus.RUNNING

        if workflow.started_at is None:
            workflow.started_at = utc_now()

        completed = {
            task.task_id
            for task in workflow.tasks
            if task.status == TaskStatus.COMPLETED
        }

        failed = {
            task.task_id
            for task in workflow.tasks
            if task.status == TaskStatus.FAILED
        }

        skipped = {
            task.task_id
            for task in workflow.tasks
            if task.status == TaskStatus.SKIPPED
        }

        results: dict[str, TaskExecutionResult] = {}

        semaphore = asyncio.Semaphore(
            1
            if workflow.execution_mode
            == ExecutionMode.SEQUENTIAL
            else workflow.max_concurrency
        )

        while True:
            blocked = graph.blocked_tasks(
                failed | skipped
            )

            for task_id in blocked:
                task = graph.tasks[task_id]

                if task.status in (
                    TaskStatus.PENDING,
                    TaskStatus.READY,
                ):
                    task.status = TaskStatus.SKIPPED
                    task.error = (
                        "Dependency failed or was skipped"
                    )
                    skipped.add(task_id)

            terminal = completed | failed | skipped

            if len(terminal) == len(workflow.tasks):
                break

            ready = graph.ready_tasks(
                completed=completed,
                terminal=terminal,
            )

            ready = [
                task
                for task in ready
                if task.status not in (
                    TaskStatus.RUNNING,
                    TaskStatus.WAITING_APPROVAL,
                )
            ]

            if not ready:
                break

            ready.sort(
                key=lambda task: (
                    -self.PRIORITY_RANK[task.priority],
                    task.task_id,
                )
            )

            if (
                workflow.execution_mode
                == ExecutionMode.SEQUENTIAL
            ):
                ready = ready[:1]

            async def run_one(
                task: OrchestrationTask,
            ) -> TaskExecutionResult:

                async with semaphore:
                    dependency_results = {
                        dependency: copy.deepcopy(
                            graph.tasks[dependency].result
                            or {}
                        )
                        for dependency
                        in task.dependencies
                    }

                    return await self.execute_task(
                        scope=scope,
                        workflow=workflow,
                        task=task,
                        dependency_results=(
                            dependency_results
                        ),
                    )

            batch_results = await asyncio.gather(
                *(run_one(task) for task in ready)
            )

            progress = False

            for result in batch_results:
                results[result.task_id] = result

                if result.status == TaskStatus.COMPLETED:
                    completed.add(result.task_id)
                    progress = True

                elif result.status == TaskStatus.FAILED:
                    failed.add(result.task_id)
                    progress = True

            if (
                failed
                and workflow.failure_policy
                == FailurePolicy.FAIL_FAST
            ):
                break

            if not progress:
                break

        if failed:
            if completed:
                workflow.status = (
                    WorkflowStatus.PARTIALLY_COMPLETED
                )
            else:
                workflow.status = WorkflowStatus.FAILED

        elif any(
            task.status == TaskStatus.WAITING_APPROVAL
            for task in workflow.tasks
        ):
            workflow.status = (
                WorkflowStatus.WAITING_APPROVAL
            )

        elif len(completed) == len(workflow.tasks):
            workflow.status = WorkflowStatus.COMPLETED

        else:
            workflow.status = WorkflowStatus.FAILED

        if workflow.status in (
            WorkflowStatus.COMPLETED,
            WorkflowStatus.PARTIALLY_COMPLETED,
            WorkflowStatus.FAILED,
        ):
            workflow.completed_at = utc_now()

        if self.repository is not None:
            await self.repository.save_workflow(
                workflow=workflow
            )

        await self._audit(
            scope=scope,
            action="orchestration.workflow_finished",
            resource_id=workflow.workflow_id,
            metadata={
                "status": workflow.status.value,
                "completed_tasks": len(completed),
                "failed_tasks": len(failed),
                "skipped_tasks": len(skipped),
            },
        )

        return WorkflowExecutionResult(
            workflow_id=workflow.workflow_id,
            tenant_id=scope.tenant_id,
            status=workflow.status,
            completed_tasks=len(completed),
            failed_tasks=len(failed),
            skipped_tasks=len(skipped),
            pending_tasks=sum(
                task.status in (
                    TaskStatus.PENDING,
                    TaskStatus.READY,
                    TaskStatus.WAITING_APPROVAL,
                )
                for task in workflow.tasks
            ),
            task_results=[
                results[task.task_id]
                for task in workflow.tasks
                if task.task_id in results
            ],
            started_at=workflow.started_at,
            completed_at=workflow.completed_at,
        )

    # --------------------------------------------------------
    # Resume Workflow
    # --------------------------------------------------------

    async def resume_workflow(
        self,
        *,
        scope: OrchestrationScope,
        workflow_id: str,
    ) -> WorkflowExecutionResult:

        if self.repository is None:
            raise WorkflowNotFoundError(
                "Workflow repository is unavailable"
            )

        validate_identifier(workflow_id)

        workflow = await self.repository.get_workflow(
            tenant_id=scope.tenant_id,
            workflow_id=workflow_id,
        )

        if workflow is None:
            raise WorkflowNotFoundError(
                "Workflow not found"
            )

        if workflow.status not in (
            WorkflowStatus.WAITING_APPROVAL,
            WorkflowStatus.PARTIALLY_COMPLETED,
            WorkflowStatus.FAILED,
        ):
            raise WorkflowConflictError(
                "Workflow cannot be resumed "
                "from its current state"
            )

        for task in workflow.tasks:
            if task.status == TaskStatus.WAITING_APPROVAL:
                task.status = TaskStatus.PENDING

        return await self.execute_workflow(
            scope=scope,
            workflow=workflow,
        )

    # --------------------------------------------------------
    # Workflow Summary
    # --------------------------------------------------------

    @staticmethod
    def generate_workflow_report(
        workflow: OrchestrationWorkflow,
    ) -> dict[str, Any]:

        return {
            "workflow_id": workflow.workflow_id,
            "tenant_id": workflow.tenant_id,
            "name": workflow.name,
            "description": workflow.description,
            "status": workflow.status.value,
            "execution_mode": (
                workflow.execution_mode.value
            ),
            "failure_policy": (
                workflow.failure_policy.value
            ),
            "created_at": (
                workflow.created_at.isoformat()
            ),
            "started_at": (
                workflow.started_at.isoformat()
                if workflow.started_at
                else None
            ),
            "completed_at": (
                workflow.completed_at.isoformat()
                if workflow.completed_at
                else None
            ),
            "tasks": [
                {
                    "task_id": task.task_id,
                    "name": task.name,
                    "agent_role": task.agent_role.value,
                    "status": task.status.value,
                    "priority": task.priority.value,
                    "safety_class": (
                        task.safety_class.value
                    ),
                    "dependencies": task.dependencies,
                    "assigned_agent_id": (
                        task.assigned_agent_id
                    ),
                    "attempts": task.attempts,
                    "approval_status": (
                        task.approval_status.value
                    ),
                    "error": task.error,
                    "result": task.result,
                }
                for task in workflow.tasks
            ],
            "safety": {
                "physical_actuation_enabled": False,
                "emergency_stop_override": False,
                "simulation_only": True,
                "human_review_for_critical_tasks": True,
            },
        }


# ============================================================
# Workflow Builder
# ============================================================

class WorkflowBuilder:

    def __init__(
        self,
        *,
        tenant_id: str,
        created_by: str,
        name: str,
        description: str = "",
        execution_mode: ExecutionMode = ExecutionMode.DAG,
        failure_policy: FailurePolicy = (
            FailurePolicy.FAIL_FAST
        ),
    ) -> None:

        validate_identifier(tenant_id)
        validate_identifier(created_by)

        self.tenant_id = tenant_id
        self.created_by = created_by
        self.name = name
        self.description = description

        self.execution_mode = execution_mode
        self.failure_policy = failure_policy

        self.tasks: list[OrchestrationTask] = []

    def add_task(
        self,
        *,
        name: str,
        agent_role: AgentRole,
        input_data: Mapping[str, Any],
        dependencies: Sequence[str] = (),
        priority: TaskPriority = TaskPriority.NORMAL,
        safety_class: TaskSafetyClass = (
            TaskSafetyClass.ADVISORY
        ),
        requires_human_approval: bool = False,
        timeout_seconds: float = (
            DEFAULT_TASK_TIMEOUT_SECONDS
        ),
        max_retries: int = 0,
        task_id: str | None = None,
    ) -> str:

        identifier = task_id or generate_id()

        task = OrchestrationTask(
            task_id=identifier,
            name=name,
            agent_role=agent_role,
            input_data=dict(input_data),
            dependencies=list(dependencies),
            priority=priority,
            safety_class=safety_class,
            requires_human_approval=(
                requires_human_approval
            ),
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )

        self.tasks.append(task)

        return identifier

    def build(
        self,
        *,
        workflow_id: str | None = None,
        max_concurrency: int = (
            DEFAULT_MAX_CONCURRENCY
        ),
    ) -> OrchestrationWorkflow:

        workflow = OrchestrationWorkflow(
            workflow_id=workflow_id or generate_id(),
            tenant_id=self.tenant_id,
            created_by=self.created_by,
            name=self.name,
            description=self.description,
            tasks=copy.deepcopy(self.tasks),
            execution_mode=self.execution_mode,
            failure_policy=self.failure_policy,
            max_concurrency=max_concurrency,
        )

        WorkflowGraph(workflow.tasks)

        return workflow


# ============================================================
# Example Robotics Workflow Factory
# ============================================================

def build_robot_inspection_workflow(
    *,
    tenant_id: str,
    user_id: str,
    robot_id: str,
) -> OrchestrationWorkflow:

    validate_identifier(robot_id)

    builder = WorkflowBuilder(
        tenant_id=tenant_id,
        created_by=user_id,
        name="Robot Health and Maintenance Inspection",
        description=(
            "Multi-agent robot health, diagnostics, "
            "maintenance, cost and safety analysis"
        ),
        execution_mode=ExecutionMode.DAG,
        failure_policy=FailurePolicy.CONTINUE,
    )

    health_id = builder.add_task(
        name="Robot Health Analysis",
        agent_role=AgentRole.HEALTH,
        input_data={"robot_id": robot_id},
        priority=TaskPriority.HIGH,
        safety_class=TaskSafetyClass.READ_ONLY,
    )

    diagnostics_id = builder.add_task(
        name="Robot Fault Diagnostics",
        agent_role=AgentRole.DIAGNOSTICS,
        input_data={"robot_id": robot_id},
        dependencies=[health_id],
        priority=TaskPriority.HIGH,
    )

    maintenance_id = builder.add_task(
        name="Predictive Maintenance Planning",
        agent_role=AgentRole.MAINTENANCE,
        input_data={"robot_id": robot_id},
        dependencies=[
            health_id,
            diagnostics_id,
        ],
        priority=TaskPriority.HIGH,
    )

    cost_id = builder.add_task(
        name="Maintenance Cost Analysis",
        agent_role=AgentRole.COST,
        input_data={"robot_id": robot_id},
        dependencies=[maintenance_id],
        priority=TaskPriority.NORMAL,
    )

    safety_id = builder.add_task(
        name="Robot Safety Review",
        agent_role=AgentRole.SAFETY,
        input_data={"robot_id": robot_id},
        dependencies=[
            diagnostics_id,
            maintenance_id,
        ],
        priority=TaskPriority.CRITICAL,
        safety_class=TaskSafetyClass.SAFETY_CRITICAL,
        requires_human_approval=True,
    )

    builder.add_task(
        name="Generate Inspection Report",
        agent_role=AgentRole.REPORTING,
        input_data={"robot_id": robot_id},
        dependencies=[
            cost_id,
            safety_id,
        ],
        priority=TaskPriority.NORMAL,
    )

    return builder.build()


# ============================================================
# Factory
# ============================================================

def create_multi_agent_orchestrator(
    *,
    agent_registry: AgentRegistryProvider,
    agent_executor: AgentExecutor,
    authorizer: OrchestrationAuthorizer,
    repository: OrchestrationRepository | None = None,
    approval_provider: (
        OrchestrationApprovalProvider | None
    ) = None,
    audit_sink: OrchestrationAuditSink | None = None,
) -> MultiAgentOrchestrator:

    return MultiAgentOrchestrator(
        agent_registry=agent_registry,
        agent_executor=agent_executor,
        authorizer=authorizer,
        repository=repository,
        approval_provider=approval_provider,
        audit_sink=audit_sink,
    )


# ============================================================
# Exports
# ============================================================

__all__ = [
    "OrchestrationError",
    "OrchestrationValidationError",
    "WorkflowNotFoundError",
    "AgentNotAvailableError",
    "WorkflowConflictError",
    "WorkflowPermissionError",
    "WorkflowDependencyError",
    "WorkflowExecutionError",
    "HumanApprovalRequiredError",
    "AgentRole",
    "WorkflowStatus",
    "TaskStatus",
    "TaskPriority",
    "ExecutionMode",
    "FailurePolicy",
    "TaskSafetyClass",
    "ApprovalStatus",
    "OrchestrationScope",
    "AgentDescriptor",
    "OrchestrationTask",
    "OrchestrationWorkflow",
    "AgentExecutionContext",
    "TaskExecutionResult",
    "WorkflowExecutionResult",
    "AgentRegistryProvider",
    "AgentExecutor",
    "OrchestrationRepository",
    "OrchestrationAuthorizer",
    "OrchestrationApprovalProvider",
    "OrchestrationAuditSink",
    "WorkflowGraph",
    "MultiAgentOrchestrator",
    "WorkflowBuilder",
    "build_robot_inspection_workflow",
    "create_multi_agent_orchestrator",
]
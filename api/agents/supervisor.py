"""
backend/app/ai/agents/supervisor.py

Physical AI & Autonomous Robotics Intelligence Platform
Multi-Agent Supervisor & Task Coordination

Python 3.11+

Features:
- Multi-agent task supervision
- Agent selection and routing
- Workflow planning and delegation
- Dependency-aware task execution
- Parallel agent coordination
- Agent health and capability checks
- Retry and timeout management
- Human-in-the-loop approval integration
- Safety-first execution policies
- Task result verification
- Supervisor memory integration
- Tenant-aware authorization
- Audit logging
- Workflow monitoring and reporting

Safety:
- No direct physical robot actuation.
- Physical commands are never delegated.
- Emergency-stop resets and safety bypasses
  are prohibited.
- Safety-critical tasks require human review.
- Human approval does not override independent
  robot safety systems.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import logging
import math
import re
import time
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

MAX_SUPERVISOR_TASKS = 200
MAX_TASK_DEPENDENCIES = 50
MAX_TASK_RETRIES = 3
MAX_CONCURRENT_AGENTS = 10
MAX_TASK_TIMEOUT_SECONDS = 600.0

DEFAULT_TASK_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_CONCURRENCY = 5

IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z0-9_.:-]{1,128}$"
)


# ============================================================
# Exceptions
# ============================================================

class SupervisorError(Exception):
    """Base supervisor exception."""


class SupervisorValidationError(SupervisorError):
    """Invalid supervisor input."""


class SupervisorPermissionError(SupervisorError):
    """Supervisor authorization failure."""


class SupervisorAgentUnavailableError(SupervisorError):
    """No suitable agent is available."""


class SupervisorWorkflowError(SupervisorError):
    """Workflow execution or validation failure."""


class SupervisorDependencyError(SupervisorError):
    """Invalid or cyclic task dependencies."""


class SupervisorSafetyError(SupervisorError):
    """Unsafe task request."""


class SupervisorApprovalError(SupervisorError):
    """Human approval is missing or rejected."""


class SupervisorVerificationError(SupervisorError):
    """Agent output verification failed."""


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
        raise SupervisorValidationError(
            f"Invalid identifier: {value!r}"
        )
    return value


def validate_positive(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise SupervisorValidationError(
            "Value must be finite and positive"
        )
    return float(value)


# ============================================================
# Enumerations
# ============================================================

class SupervisorAgentRole(str, Enum):
    SUPERVISOR = "supervisor"
    PLANNER = "planner"
    RESEARCH = "research"
    PERCEPTION = "perception"
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
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
    EVALUATION = "evaluation"
    RAG = "rag"
    MEMORY = "memory"
    ANALYTICS = "analytics"
    REPORTING = "reporting"
    CUSTOM = "custom"


class SupervisorTaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class SupervisorWorkflowStatus(str, Enum):
    CREATED = "created"
    VALIDATED = "validated"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SupervisorTaskPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class SupervisorExecutionMode(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DAG = "dag"


class SupervisorFailurePolicy(str, Enum):
    FAIL_FAST = "fail_fast"
    CONTINUE = "continue"


class SupervisorSafetyClass(str, Enum):
    READ_ONLY = "read_only"
    ADVISORY = "advisory"
    SIMULATION = "simulation"
    SAFETY_CRITICAL = "safety_critical"
    PHYSICAL_ACTUATION = "physical_actuation"


class SupervisorApprovalStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class SupervisorVerificationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


# ============================================================
# Scope
# ============================================================

@dataclass(frozen=True, slots=True)
class SupervisorScope:
    tenant_id: str
    user_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_identifier(self.user_id)


# ============================================================
# Agent Descriptor
# ============================================================

@dataclass(frozen=True, slots=True)
class SupervisorAgentDescriptor:
    agent_id: str
    tenant_id: str

    role: SupervisorAgentRole

    capabilities: frozenset[str]

    available: bool = True
    healthy: bool = True

    priority: int = 0

    active_tasks: int = 0
    maximum_concurrent_tasks: int = 1

    def __post_init__(self) -> None:
        validate_identifier(self.agent_id)
        validate_identifier(self.tenant_id)

        if self.active_tasks < 0:
            raise SupervisorValidationError(
                "Active task count cannot be negative"
            )

        if self.maximum_concurrent_tasks < 1:
            raise SupervisorValidationError(
                "Agent concurrency must be positive"
            )


# ============================================================
# Task
# ============================================================

@dataclass(slots=True)
class SupervisorTask:
    task_id: str

    name: str
    agent_role: SupervisorAgentRole

    input_data: dict[str, Any] = field(
        default_factory=dict
    )

    dependencies: tuple[str, ...] = ()

    required_capabilities: frozenset[str] = frozenset()

    priority: SupervisorTaskPriority = (
        SupervisorTaskPriority.NORMAL
    )

    safety_class: SupervisorSafetyClass = (
        SupervisorSafetyClass.ADVISORY
    )

    requires_human_approval: bool = False

    timeout_seconds: float = (
        DEFAULT_TASK_TIMEOUT_SECONDS
    )

    max_retries: int = 1

    status: SupervisorTaskStatus = (
        SupervisorTaskStatus.PENDING
    )

    assigned_agent_id: str | None = None

    approval_id: str | None = None

    approval_status: SupervisorApprovalStatus = (
        SupervisorApprovalStatus.NOT_REQUIRED
    )

    result: dict[str, Any] | None = None
    error: str | None = None

    attempts: int = 0

    started_at: datetime | None = None
    completed_at: datetime | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.task_id)

        if not self.name.strip():
            raise SupervisorValidationError(
                "Task name is required"
            )

        if len(self.dependencies) > MAX_TASK_DEPENDENCIES:
            raise SupervisorValidationError(
                "Too many task dependencies"
            )

        for dependency in self.dependencies:
            validate_identifier(dependency)

        if len(set(self.dependencies)) != len(
            self.dependencies
        ):
            raise SupervisorDependencyError(
                "Duplicate task dependencies"
            )

        validate_positive(self.timeout_seconds)

        if self.timeout_seconds > MAX_TASK_TIMEOUT_SECONDS:
            raise SupervisorValidationError(
                "Task timeout exceeds maximum"
            )

        if not 0 <= self.max_retries <= MAX_TASK_RETRIES:
            raise SupervisorValidationError(
                "Invalid retry count"
            )

        if self.safety_class == (
            SupervisorSafetyClass.PHYSICAL_ACTUATION
        ):
            raise SupervisorSafetyError(
                "Physical actuation is prohibited "
                "in the supervisor"
            )

        if self.safety_class == (
            SupervisorSafetyClass.SAFETY_CRITICAL
        ):
            self.requires_human_approval = True


# ============================================================
# Workflow
# ============================================================

@dataclass(slots=True)
class SupervisorWorkflow:
    workflow_id: str

    tenant_id: str
    created_by: str

    name: str

    tasks: list[SupervisorTask]

    execution_mode: SupervisorExecutionMode = (
        SupervisorExecutionMode.DAG
    )

    failure_policy: SupervisorFailurePolicy = (
        SupervisorFailurePolicy.FAIL_FAST
    )

    max_concurrency: int = DEFAULT_MAX_CONCURRENCY

    status: SupervisorWorkflowStatus = (
        SupervisorWorkflowStatus.CREATED
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    started_at: datetime | None = None
    completed_at: datetime | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.workflow_id)
        validate_identifier(self.tenant_id)
        validate_identifier(self.created_by)

        if not self.name.strip():
            raise SupervisorValidationError(
                "Workflow name is required"
            )

        if not 1 <= len(self.tasks) <= MAX_SUPERVISOR_TASKS:
            raise SupervisorValidationError(
                "Invalid workflow task count"
            )

        if not 1 <= self.max_concurrency <= MAX_CONCURRENT_AGENTS:
            raise SupervisorValidationError(
                "Invalid workflow concurrency"
            )


# ============================================================
# Execution Context
# ============================================================

@dataclass(frozen=True, slots=True)
class SupervisorExecutionContext:
    workflow_id: str
    task_id: str

    tenant_id: str
    user_id: str

    agent_id: str

    input_data: Mapping[str, Any]
    dependency_results: Mapping[str, Mapping[str, Any]]

    simulation_only: bool = True

    def __post_init__(self) -> None:
        if not self.simulation_only:
            raise SupervisorSafetyError(
                "Supervisor execution must remain "
                "advisory or simulation-only"
            )


# ============================================================
# Verification
# ============================================================

@dataclass(frozen=True, slots=True)
class SupervisorVerificationResult:
    status: SupervisorVerificationStatus

    confidence: float

    findings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
        ):
            raise SupervisorValidationError(
                "Invalid verification confidence"
            )


# ============================================================
# Results
# ============================================================

@dataclass(frozen=True, slots=True)
class SupervisorTaskResult:
    task_id: str
    agent_id: str | None

    status: SupervisorTaskStatus

    output: Mapping[str, Any] | None

    attempts: int
    duration_seconds: float

    verification: (
        SupervisorVerificationResult | None
    ) = None

    error: str | None = None


@dataclass(frozen=True, slots=True)
class SupervisorWorkflowResult:
    workflow_id: str
    tenant_id: str

    status: SupervisorWorkflowStatus

    completed_tasks: int
    failed_tasks: int
    skipped_tasks: int
    pending_tasks: int

    task_results: tuple[SupervisorTaskResult, ...]

    started_at: datetime | None
    completed_at: datetime | None

    physical_execution_authorized: bool = False


# ============================================================
# Provider Protocols
# ============================================================

class SupervisorAgentRegistry(Protocol):

    async def find_agents(
        self,
        *,
        tenant_id: str,
        role: SupervisorAgentRole,
        required_capabilities: frozenset[str],
    ) -> list[SupervisorAgentDescriptor]:
        ...


class SupervisorAgentExecutor(Protocol):

    async def execute(
        self,
        *,
        agent: SupervisorAgentDescriptor,
        context: SupervisorExecutionContext,
    ) -> Mapping[str, Any]:
        ...


class SupervisorOutputVerifier(Protocol):

    async def verify(
        self,
        *,
        tenant_id: str,
        task: SupervisorTask,
        output: Mapping[str, Any],
    ) -> SupervisorVerificationResult:
        ...


class SupervisorApprovalProvider(Protocol):

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
    ) -> SupervisorApprovalStatus:
        ...


class SupervisorAuthorizer(Protocol):

    async def require_permission(
        self,
        *,
        tenant_id: str,
        user_id: str,
        permission: str,
    ) -> None:
        ...


class SupervisorRepository(Protocol):

    async def save_workflow(
        self,
        *,
        workflow: SupervisorWorkflow,
    ) -> None:
        ...

    async def get_workflow(
        self,
        *,
        tenant_id: str,
        workflow_id: str,
    ) -> SupervisorWorkflow | None:
        ...


class SupervisorAuditSink(Protocol):

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


class SupervisorMemoryProvider(Protocol):

    async def remember_task_result(
        self,
        *,
        tenant_id: str,
        workflow_id: str,
        task_id: str,
        result: Mapping[str, Any],
    ) -> None:
        ...


# ============================================================
# Workflow Graph
# ============================================================

class SupervisorWorkflowGraph:

    def __init__(
        self,
        tasks: Sequence[SupervisorTask],
    ) -> None:

        self.tasks = {
            task.task_id: task
            for task in tasks
        }

        if len(self.tasks) != len(tasks):
            raise SupervisorDependencyError(
                "Duplicate task identifiers"
            )

        self.validate()

    def validate(self) -> None:

        for task in self.tasks.values():
            for dependency in task.dependencies:
                if dependency not in self.tasks:
                    raise SupervisorDependencyError(
                        f"Unknown dependency: {dependency}"
                    )

                if dependency == task.task_id:
                    raise SupervisorDependencyError(
                        "Task cannot depend on itself"
                    )

        indegree = {
            task_id: len(task.dependencies)
            for task_id, task in self.tasks.items()
        }

        dependents: dict[str, list[str]] = {
            task_id: []
            for task_id in self.tasks
        }

        for task in self.tasks.values():
            for dependency in task.dependencies:
                dependents[dependency].append(
                    task.task_id
                )

        ready = [
            task_id
            for task_id, degree in indegree.items()
            if degree == 0
        ]

        visited = 0

        while ready:
            task_id = ready.pop()
            visited += 1

            for dependent in dependents[task_id]:
                indegree[dependent] -= 1

                if indegree[dependent] == 0:
                    ready.append(dependent)

        if visited != len(self.tasks):
            raise SupervisorDependencyError(
                "Workflow contains cyclic dependencies"
            )

    def ready_tasks(
        self,
        *,
        completed: set[str],
        terminal: set[str],
    ) -> list[SupervisorTask]:

        return [
            task
            for task in self.tasks.values()
            if task.task_id not in terminal
            and all(
                dependency in completed
                for dependency in task.dependencies
            )
        ]

    def blocked_tasks(
        self,
        *,
        failed: set[str],
    ) -> set[str]:

        blocked = set(failed)

        changed = True

        while changed:
            changed = False

            for task in self.tasks.values():
                if task.task_id in blocked:
                    continue

                if any(
                    dependency in blocked
                    for dependency in task.dependencies
                ):
                    blocked.add(task.task_id)
                    changed = True

        return blocked - failed


# ============================================================
# Supervisor Agent
# ============================================================

class SupervisorAgent:

    def __init__(
        self,
        *,
        registry: SupervisorAgentRegistry,
        executor: SupervisorAgentExecutor,
        authorizer: SupervisorAuthorizer,
        verifier: SupervisorOutputVerifier | None = None,
        approval_provider: SupervisorApprovalProvider | None = None,
        repository: SupervisorRepository | None = None,
        audit_sink: SupervisorAuditSink | None = None,
        memory_provider: SupervisorMemoryProvider | None = None,
    ) -> None:

        self.registry = registry
        self.executor = executor
        self.authorizer = authorizer

        self.verifier = verifier
        self.approval_provider = approval_provider

        self.repository = repository
        self.audit_sink = audit_sink
        self.memory_provider = memory_provider

        self._workflow_locks: dict[str, asyncio.Lock] = {}

    # --------------------------------------------------------
    # Authorization
    # --------------------------------------------------------

    async def _authorize(
        self,
        *,
        scope: SupervisorScope,
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
        scope: SupervisorScope,
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
    # Validate Workflow
    # --------------------------------------------------------

    @staticmethod
    def validate_workflow(
        workflow: SupervisorWorkflow,
    ) -> SupervisorWorkflowGraph:

        graph = SupervisorWorkflowGraph(
            workflow.tasks
        )

        if workflow.execution_mode == (
            SupervisorExecutionMode.SEQUENTIAL
        ):
            task_positions = {
                task.task_id: index
                for index, task in enumerate(
                    workflow.tasks
                )
            }

            for task in workflow.tasks:
                for dependency in task.dependencies:
                    if (
                        task_positions[dependency]
                        >= task_positions[task.task_id]
                    ):
                        raise SupervisorDependencyError(
                            "Sequential workflow dependency "
                            "must reference an earlier task"
                        )

        return graph

    # --------------------------------------------------------
    # Select Agent
    # --------------------------------------------------------

    async def select_agent(
        self,
        *,
        scope: SupervisorScope,
        task: SupervisorTask,
    ) -> SupervisorAgentDescriptor:

        candidates = await self.registry.find_agents(
            tenant_id=scope.tenant_id,
            role=task.agent_role,
            required_capabilities=(
                task.required_capabilities
            ),
        )

        eligible = [
            agent
            for agent in candidates
            if agent.tenant_id == scope.tenant_id
            and agent.role == task.agent_role
            and agent.available
            and agent.healthy
            and agent.active_tasks
            < agent.maximum_concurrent_tasks
            and task.required_capabilities.issubset(
                agent.capabilities
            )
        ]

        if not eligible:
            raise SupervisorAgentUnavailableError(
                f"No healthy agent available for "
                f"{task.agent_role.value}"
            )

        eligible.sort(
            key=lambda agent: (
                -agent.priority,
                agent.active_tasks
                / agent.maximum_concurrent_tasks,
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
        scope: SupervisorScope,
        workflow: SupervisorWorkflow,
        task: SupervisorTask,
    ) -> bool:

        if not task.requires_human_approval:
            task.approval_status = (
                SupervisorApprovalStatus.NOT_REQUIRED
            )
            return True

        if self.approval_provider is None:
            raise SupervisorApprovalError(
                "Human approval provider unavailable"
            )

        if task.approval_id is None:
            task.approval_id = (
                await self.approval_provider.request_approval(
                    tenant_id=scope.tenant_id,
                    user_id=scope.user_id,
                    workflow_id=workflow.workflow_id,
                    task_id=task.task_id,
                    reason=(
                        f"Human review required for "
                        f"{task.name}"
                    ),
                )
            )

            task.approval_status = (
                SupervisorApprovalStatus.PENDING
            )

            task.status = (
                SupervisorTaskStatus.WAITING_APPROVAL
            )

            return False

        approval_status = (
            await self.approval_provider.get_approval_status(
                tenant_id=scope.tenant_id,
                approval_id=task.approval_id,
            )
        )

        task.approval_status = approval_status

        if approval_status == (
            SupervisorApprovalStatus.APPROVED
        ):
            return True

        if approval_status == (
            SupervisorApprovalStatus.REJECTED
        ):
            task.status = SupervisorTaskStatus.FAILED
            task.error = "Human approval rejected"
            task.completed_at = utc_now()
            return False

        task.status = (
            SupervisorTaskStatus.WAITING_APPROVAL
        )

        return False

    # --------------------------------------------------------
    # Verify Output
    # --------------------------------------------------------

    async def verify_output(
        self,
        *,
        scope: SupervisorScope,
        task: SupervisorTask,
        output: Mapping[str, Any],
    ) -> SupervisorVerificationResult:

        if self.verifier is None:
            return SupervisorVerificationResult(
                status=(
                    SupervisorVerificationStatus.INCONCLUSIVE
                ),
                confidence=0.0,
                findings=(
                    "No independent output verifier configured.",
                ),
            )

        result = await self.verifier.verify(
            tenant_id=scope.tenant_id,
            task=task,
            output=output,
        )

        return result

    # --------------------------------------------------------
    # Execute Task
    # --------------------------------------------------------

    async def execute_task(
        self,
        *,
        scope: SupervisorScope,
        workflow: SupervisorWorkflow,
        task: SupervisorTask,
        dependency_results: Mapping[
            str,
            Mapping[str, Any],
        ],
    ) -> SupervisorTaskResult:

        if task.safety_class == (
            SupervisorSafetyClass.PHYSICAL_ACTUATION
        ):
            raise SupervisorSafetyError(
                "Physical actuation prohibited"
            )

        started = time.monotonic()

        task.status = SupervisorTaskStatus.RUNNING
        task.started_at = utc_now()
        task.error = None

        verification: (
            SupervisorVerificationResult | None
        ) = None

        for attempt in range(
            task.max_retries + 1
        ):
            task.attempts += 1

            try:
                agent = await self.select_agent(
                    scope=scope,
                    task=task,
                )

                task.assigned_agent_id = agent.agent_id

                context = SupervisorExecutionContext(
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
                    self.executor.execute(
                        agent=agent,
                        context=context,
                    ),
                    timeout=task.timeout_seconds,
                )

                if not isinstance(output, Mapping):
                    raise SupervisorVerificationError(
                        "Agent output must be a mapping"
                    )

                if output.get(
                    "physical_execution_authorized"
                ) is True:
                    raise SupervisorSafetyError(
                        "Agent attempted to authorize "
                        "physical execution"
                    )

                verification = await self.verify_output(
                    scope=scope,
                    task=task,
                    output=output,
                )

                if verification.status == (
                    SupervisorVerificationStatus.FAILED
                ):
                    raise SupervisorVerificationError(
                        "Agent output verification failed"
                    )

                if (
                    task.safety_class
                    == SupervisorSafetyClass.SAFETY_CRITICAL
                    and verification.status
                    != SupervisorVerificationStatus.PASSED
                ):
                    raise SupervisorVerificationError(
                        "Safety-critical output requires "
                        "successful independent verification"
                    )

                task.result = dict(output)
                task.status = (
                    SupervisorTaskStatus.COMPLETED
                )
                task.completed_at = utc_now()

                if self.memory_provider is not None:
                    await self.memory_provider.remember_task_result(
                        tenant_id=scope.tenant_id,
                        workflow_id=workflow.workflow_id,
                        task_id=task.task_id,
                        result=task.result,
                    )

                await self._audit(
                    scope=scope,
                    action="supervisor.task_completed",
                    resource_id=task.task_id,
                    metadata={
                        "workflow_id": workflow.workflow_id,
                        "agent_id": agent.agent_id,
                        "attempts": task.attempts,
                    },
                )

                return SupervisorTaskResult(
                    task_id=task.task_id,
                    agent_id=agent.agent_id,
                    status=task.status,
                    output=task.result,
                    attempts=task.attempts,
                    duration_seconds=(
                        time.monotonic() - started
                    ),
                    verification=verification,
                )

            except asyncio.CancelledError:
                task.status = (
                    SupervisorTaskStatus.CANCELLED
                )
                task.error = "Task cancelled"
                task.completed_at = utc_now()
                raise

            except SupervisorSafetyError:
                task.status = SupervisorTaskStatus.FAILED
                task.error = "Safety policy violation"
                task.completed_at = utc_now()
                break

            except Exception as exc:
                task.error = (
                    f"{type(exc).__name__}: {exc}"
                )

                logger.warning(
                    "Supervisor task %s attempt %s failed: %s",
                    task.task_id,
                    attempt + 1,
                    type(exc).__name__,
                )

                if attempt < task.max_retries:
                    await asyncio.sleep(
                        min(2 ** attempt, 8)
                    )

        task.status = SupervisorTaskStatus.FAILED
        task.completed_at = utc_now()

        await self._audit(
            scope=scope,
            action="supervisor.task_failed",
            resource_id=task.task_id,
            metadata={
                "workflow_id": workflow.workflow_id,
                "attempts": task.attempts,
                "error": task.error,
            },
        )

        return SupervisorTaskResult(
            task_id=task.task_id,
            agent_id=task.assigned_agent_id,
            status=task.status,
            output=None,
            attempts=task.attempts,
            duration_seconds=(
                time.monotonic() - started
            ),
            verification=verification,
            error=task.error,
        )

    # --------------------------------------------------------
    # Execute Workflow
    # --------------------------------------------------------

    async def execute_workflow(
        self,
        *,
        scope: SupervisorScope,
        workflow: SupervisorWorkflow,
    ) -> SupervisorWorkflowResult:

        if workflow.tenant_id != scope.tenant_id:
            raise SupervisorPermissionError(
                "Workflow tenant mismatch"
            )

        await self._authorize(
            scope=scope,
            permission="supervisor:execute",
        )

        graph = self.validate_workflow(
            workflow
        )

        lock = self._workflow_locks.setdefault(
            workflow.workflow_id,
            asyncio.Lock(),
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
        scope: SupervisorScope,
        workflow: SupervisorWorkflow,
        graph: SupervisorWorkflowGraph,
    ) -> SupervisorWorkflowResult:

        if workflow.started_at is None:
            workflow.started_at = utc_now()

        workflow.completed_at = None
        workflow.status = (
            SupervisorWorkflowStatus.RUNNING
        )

        task_results: dict[
            str,
            SupervisorTaskResult,
        ] = {}

        completed: set[str] = {
            task.task_id
            for task in workflow.tasks
            if task.status
            == SupervisorTaskStatus.COMPLETED
        }

        failed: set[str] = {
            task.task_id
            for task in workflow.tasks
            if task.status
            == SupervisorTaskStatus.FAILED
        }

        terminal: set[str] = {
            task.task_id
            for task in workflow.tasks
            if task.status in (
                SupervisorTaskStatus.COMPLETED,
                SupervisorTaskStatus.FAILED,
                SupervisorTaskStatus.SKIPPED,
                SupervisorTaskStatus.CANCELLED,
            )
        }

        semaphore = asyncio.Semaphore(
            workflow.max_concurrency
        )

        async def run_task(
            task: SupervisorTask,
        ) -> SupervisorTaskResult:

            async with semaphore:
                dependency_results = {
                    dependency: (
                        graph.tasks[dependency].result or {}
                    )
                    for dependency in task.dependencies
                }

                return await self.execute_task(
                    scope=scope,
                    workflow=workflow,
                    task=task,
                    dependency_results=dependency_results,
                )

        while True:
            blocked = graph.blocked_tasks(
                failed=failed
            )

            for task_id in blocked:
                task = graph.tasks[task_id]

                if task.status not in (
                    SupervisorTaskStatus.COMPLETED,
                    SupervisorTaskStatus.FAILED,
                    SupervisorTaskStatus.SKIPPED,
                ):
                    task.status = (
                        SupervisorTaskStatus.SKIPPED
                    )
                    task.error = (
                        "Required dependency failed"
                    )
                    task.completed_at = utc_now()
                    terminal.add(task_id)

            ready = [
                task
                for task in graph.ready_tasks(
                    completed=completed,
                    terminal=terminal,
                )
                if task.status in (
                    SupervisorTaskStatus.PENDING,
                    SupervisorTaskStatus.READY,
                    SupervisorTaskStatus.WAITING_APPROVAL,
                )
            ]

            if not ready:
                break

            approved_tasks: list[SupervisorTask] = []

            for task in ready:
                approved = await self.ensure_approval(
                    scope=scope,
                    workflow=workflow,
                    task=task,
                )

                if approved:
                    task.status = SupervisorTaskStatus.READY
                    approved_tasks.append(task)

                elif task.status == (
                    SupervisorTaskStatus.FAILED
                ):
                    failed.add(task.task_id)
                    terminal.add(task.task_id)

            if not approved_tasks:
                break

            if workflow.execution_mode == (
                SupervisorExecutionMode.SEQUENTIAL
            ):
                approved_tasks = approved_tasks[:1]

            results = await asyncio.gather(
                *(
                    run_task(task)
                    for task in approved_tasks
                )
            )

            for result in results:
                task_results[result.task_id] = result

                terminal.add(result.task_id)

                if result.status == (
                    SupervisorTaskStatus.COMPLETED
                ):
                    completed.add(result.task_id)

                elif result.status == (
                    SupervisorTaskStatus.FAILED
                ):
                    failed.add(result.task_id)

            if (
                failed
                and workflow.failure_policy
                == SupervisorFailurePolicy.FAIL_FAST
            ):
                break

        waiting = any(
            task.status == (
                SupervisorTaskStatus.WAITING_APPROVAL
            )
            for task in workflow.tasks
        )

        pending = any(
            task.status in (
                SupervisorTaskStatus.PENDING,
                SupervisorTaskStatus.READY,
            )
            for task in workflow.tasks
        )

        if waiting:
            workflow.status = (
                SupervisorWorkflowStatus.WAITING_APPROVAL
            )

        elif failed:
            workflow.status = (
                SupervisorWorkflowStatus.PARTIALLY_COMPLETED
                if completed
                else SupervisorWorkflowStatus.FAILED
            )

        elif pending:
            workflow.status = (
                SupervisorWorkflowStatus.PARTIALLY_COMPLETED
            )

        else:
            workflow.status = (
                SupervisorWorkflowStatus.COMPLETED
            )

        if workflow.status != (
            SupervisorWorkflowStatus.WAITING_APPROVAL
        ):
            workflow.completed_at = utc_now()

        if self.repository is not None:
            await self.repository.save_workflow(
                workflow=workflow
            )

        await self._audit(
            scope=scope,
            action="supervisor.workflow_updated",
            resource_id=workflow.workflow_id,
            metadata={
                "status": workflow.status.value,
                "completed_tasks": len(completed),
                "failed_tasks": len(failed),
            },
        )

        return SupervisorWorkflowResult(
            workflow_id=workflow.workflow_id,
            tenant_id=workflow.tenant_id,
            status=workflow.status,
            completed_tasks=sum(
                task.status == (
                    SupervisorTaskStatus.COMPLETED
                )
                for task in workflow.tasks
            ),
            failed_tasks=sum(
                task.status == (
                    SupervisorTaskStatus.FAILED
                )
                for task in workflow.tasks
            ),
            skipped_tasks=sum(
                task.status == (
                    SupervisorTaskStatus.SKIPPED
                )
                for task in workflow.tasks
            ),
            pending_tasks=sum(
                task.status in (
                    SupervisorTaskStatus.PENDING,
                    SupervisorTaskStatus.READY,
                    SupervisorTaskStatus.WAITING_APPROVAL,
                )
                for task in workflow.tasks
            ),
            task_results=tuple(
                task_results.values()
            ),
            started_at=workflow.started_at,
            completed_at=workflow.completed_at,
            physical_execution_authorized=False,
        )

    # --------------------------------------------------------
    # Resume Workflow
    # --------------------------------------------------------

    async def resume_workflow(
        self,
        *,
        scope: SupervisorScope,
        workflow_id: str,
    ) -> SupervisorWorkflowResult:

        validate_identifier(workflow_id)

        if self.repository is None:
            raise SupervisorWorkflowError(
                "Workflow repository unavailable"
            )

        workflow = await self.repository.get_workflow(
            tenant_id=scope.tenant_id,
            workflow_id=workflow_id,
        )

        if workflow is None:
            raise SupervisorWorkflowError(
                "Workflow not found"
            )

        if workflow.tenant_id != scope.tenant_id:
            raise SupervisorPermissionError(
                "Workflow tenant mismatch"
            )

        return await self.execute_workflow(
            scope=scope,
            workflow=workflow,
        )

    # --------------------------------------------------------
    # Workflow Report
    # --------------------------------------------------------

    @staticmethod
    def generate_workflow_report(
        workflow: SupervisorWorkflow,
    ) -> dict[str, Any]:

        return {
            "workflow_id": workflow.workflow_id,
            "tenant_id": workflow.tenant_id,
            "name": workflow.name,
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
                    "agent_role": (
                        task.agent_role.value
                    ),
                    "status": task.status.value,
                    "priority": task.priority.value,
                    "safety_class": (
                        task.safety_class.value
                    ),
                    "dependencies": list(
                        task.dependencies
                    ),
                    "assigned_agent_id": (
                        task.assigned_agent_id
                    ),
                    "approval_status": (
                        task.approval_status.value
                    ),
                    "attempts": task.attempts,
                    "error": task.error,
                    "result": task.result,
                }
                for task in workflow.tasks
            ],
            "safety": {
                "physical_execution_authorized": False,
                "automatic_actuation_enabled": False,
                "emergency_stop_reset_allowed": False,
                "safety_interlock_override_allowed": False,
                "human_review_required_for_safety_tasks": True,
            },
        }


# ============================================================
# Workflow Builder
# ============================================================

class SupervisorWorkflowBuilder:

    def __init__(
        self,
        *,
        tenant_id: str,
        created_by: str,
        name: str,
    ) -> None:

        self.tenant_id = validate_identifier(
            tenant_id
        )

        self.created_by = validate_identifier(
            created_by
        )

        self.name = name

        self.tasks: list[SupervisorTask] = []

    def add_task(
        self,
        *,
        name: str,
        agent_role: SupervisorAgentRole,
        input_data: Mapping[str, Any] | None = None,
        dependencies: Sequence[str] = (),
        priority: SupervisorTaskPriority = (
            SupervisorTaskPriority.NORMAL
        ),
        safety_class: SupervisorSafetyClass = (
            SupervisorSafetyClass.ADVISORY
        ),
        requires_human_approval: bool = False,
        timeout_seconds: float = (
            DEFAULT_TASK_TIMEOUT_SECONDS
        ),
    ) -> str:

        task_id = generate_id()

        task = SupervisorTask(
            task_id=task_id,
            name=name,
            agent_role=agent_role,
            input_data=dict(input_data or {}),
            dependencies=tuple(dependencies),
            priority=priority,
            safety_class=safety_class,
            requires_human_approval=(
                requires_human_approval
            ),
            timeout_seconds=timeout_seconds,
        )

        self.tasks.append(task)

        return task_id

    def build(
        self,
        *,
        execution_mode: SupervisorExecutionMode = (
            SupervisorExecutionMode.DAG
        ),
        failure_policy: SupervisorFailurePolicy = (
            SupervisorFailurePolicy.FAIL_FAST
        ),
        max_concurrency: int = (
            DEFAULT_MAX_CONCURRENCY
        ),
    ) -> SupervisorWorkflow:

        workflow = SupervisorWorkflow(
            workflow_id=generate_id(),
            tenant_id=self.tenant_id,
            created_by=self.created_by,
            name=self.name,
            tasks=copy.deepcopy(self.tasks),
            execution_mode=execution_mode,
            failure_policy=failure_policy,
            max_concurrency=max_concurrency,
        )

        SupervisorAgent.validate_workflow(
            workflow
        )

        return workflow


# ============================================================
# Example Robot Inspection Workflow
# ============================================================

def build_robot_inspection_workflow(
    *,
    tenant_id: str,
    user_id: str,
    robot_id: str,
) -> SupervisorWorkflow:

    validate_identifier(robot_id)

    builder = SupervisorWorkflowBuilder(
        tenant_id=tenant_id,
        created_by=user_id,
        name="Robot Inspection and Repair Intelligence",
    )

    health_task = builder.add_task(
        name="Robot health assessment",
        agent_role=SupervisorAgentRole.HEALTH,
        input_data={
            "robot_id": robot_id,
        },
        safety_class=(
            SupervisorSafetyClass.READ_ONLY
        ),
    )

    diagnostics_task = builder.add_task(
        name="Fault diagnostics",
        agent_role=SupervisorAgentRole.DIAGNOSTICS,
        input_data={
            "robot_id": robot_id,
        },
        dependencies=(health_task,),
    )

    safety_task = builder.add_task(
        name="Robot safety assessment",
        agent_role=SupervisorAgentRole.SAFETY,
        input_data={
            "robot_id": robot_id,
        },
        dependencies=(health_task,),
        priority=SupervisorTaskPriority.CRITICAL,
        safety_class=(
            SupervisorSafetyClass.SAFETY_CRITICAL
        ),
        requires_human_approval=True,
    )

    repair_task = builder.add_task(
        name="Repair recommendation",
        agent_role=SupervisorAgentRole.REPAIR,
        input_data={
            "robot_id": robot_id,
        },
        dependencies=(
            diagnostics_task,
            safety_task,
        ),
    )

    cost_task = builder.add_task(
        name="Repair cost estimation",
        agent_role=SupervisorAgentRole.COST,
        input_data={
            "robot_id": robot_id,
        },
        dependencies=(repair_task,),
    )

    builder.add_task(
        name="Inspection report",
        agent_role=SupervisorAgentRole.REPORTING,
        input_data={
            "robot_id": robot_id,
        },
        dependencies=(
            health_task,
            diagnostics_task,
            safety_task,
            repair_task,
            cost_task,
        ),
        safety_class=(
            SupervisorSafetyClass.READ_ONLY
        ),
    )

    return builder.build()


# ============================================================
# Factory
# ============================================================

def create_supervisor_agent(
    *,
    registry: SupervisorAgentRegistry,
    executor: SupervisorAgentExecutor,
    authorizer: SupervisorAuthorizer,
    verifier: SupervisorOutputVerifier | None = None,
    approval_provider: SupervisorApprovalProvider | None = None,
    repository: SupervisorRepository | None = None,
    audit_sink: SupervisorAuditSink | None = None,
    memory_provider: SupervisorMemoryProvider | None = None,
) -> SupervisorAgent:

    return SupervisorAgent(
        registry=registry,
        executor=executor,
        authorizer=authorizer,
        verifier=verifier,
        approval_provider=approval_provider,
        repository=repository,
        audit_sink=audit_sink,
        memory_provider=memory_provider,
    )


# ============================================================
# Exports
# ============================================================

__all__ = [
    "SupervisorError",
    "SupervisorValidationError",
    "SupervisorPermissionError",
    "SupervisorAgentUnavailableError",
    "SupervisorWorkflowError",
    "SupervisorDependencyError",
    "SupervisorSafetyError",
    "SupervisorApprovalError",
    "SupervisorVerificationError",
    "SupervisorAgentRole",
    "SupervisorTaskStatus",
    "SupervisorWorkflowStatus",
    "SupervisorTaskPriority",
    "SupervisorExecutionMode",
    "SupervisorFailurePolicy",
    "SupervisorSafetyClass",
    "SupervisorApprovalStatus",
    "SupervisorVerificationStatus",
    "SupervisorScope",
    "SupervisorAgentDescriptor",
    "SupervisorTask",
    "SupervisorWorkflow",
    "SupervisorExecutionContext",
    "SupervisorVerificationResult",
    "SupervisorTaskResult",
    "SupervisorWorkflowResult",
    "SupervisorAgentRegistry",
    "SupervisorAgentExecutor",
    "SupervisorOutputVerifier",
    "SupervisorApprovalProvider",
    "SupervisorAuthorizer",
    "SupervisorRepository",
    "SupervisorAuditSink",
    "SupervisorMemoryProvider",
    "SupervisorWorkflowGraph",
    "SupervisorAgent",
    "SupervisorWorkflowBuilder",
    "build_robot_inspection_workflow",
    "create_supervisor_agent",
]
"""Workflow State module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/ai/agents/workflow_state.py

Physical AI & Autonomous Robotics Intelligence Platform
Multi-Agent Workflow State Management

Python 3.11+

Features:
- Workflow lifecycle management
- Task state tracking
- Dependency-aware execution readiness
- Human approval state tracking
- Workflow pause, resume and cancellation
- Optimistic concurrency control
- State transition validation
- Workflow event history
- Retry tracking
- Tenant isolation
- Agent assignment tracking
- Workflow progress calculation
- Execution checkpoints
- State snapshots and restoration
- Audit logging
- In-memory repository
- Production repository integration

Safety:
- No physical robot actuation
- No emergency-stop reset
- No safety-interlock bypass
- Safety-critical tasks require human approval
- Approval does not authorize physical execution
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

MAX_WORKFLOW_TASKS = 500
MAX_TASK_DEPENDENCIES = 100
MAX_TASK_RETRIES = 5
MAX_EVENT_HISTORY = 10_000
MAX_METADATA_FIELDS = 100

IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z0-9_.:-]{1,128}$"
)


# ============================================================
# Exceptions
# ============================================================

class WorkflowStateError(Exception):
    """Base workflow state exception."""


class WorkflowValidationError(WorkflowStateError):
    """Invalid workflow state or input."""


class WorkflowNotFoundError(WorkflowStateError):
    """Workflow does not exist."""


class WorkflowPermissionError(WorkflowStateError):
    """Workflow access denied."""


class WorkflowConflictError(WorkflowStateError):
    """Workflow version conflict."""


class WorkflowTransitionError(WorkflowStateError):
    """Invalid workflow state transition."""


class WorkflowDependencyError(WorkflowStateError):
    """Invalid task dependency configuration."""


class WorkflowApprovalError(WorkflowStateError):
    """Human approval requirements not satisfied."""


class WorkflowSafetyError(WorkflowStateError):
    """Unsafe workflow operation."""


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
        raise WorkflowValidationError(
            f"Invalid identifier: {value!r}"
        )
    return value


def validate_probability(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise WorkflowValidationError(
            "Probability must be between 0 and 1"
        )
    return float(value)


# ============================================================
# Enumerations
# ============================================================

class WorkflowStatus(str, Enum):
    CREATED = "created"
    VALIDATED = "validated"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowTaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class WorkflowTaskPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class WorkflowExecutionMode(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DAG = "dag"


class WorkflowFailurePolicy(str, Enum):
    FAIL_FAST = "fail_fast"
    CONTINUE = "continue"


class WorkflowSafetyClass(str, Enum):
    READ_ONLY = "read_only"
    ADVISORY = "advisory"
    SIMULATION = "simulation"
    SAFETY_CRITICAL = "safety_critical"
    PHYSICAL_ACTUATION = "physical_actuation"


class WorkflowApprovalStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class WorkflowEventType(str, Enum):
    WORKFLOW_CREATED = "workflow_created"
    WORKFLOW_VALIDATED = "workflow_validated"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_PAUSED = "workflow_paused"
    WORKFLOW_RESUMED = "workflow_resumed"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    WORKFLOW_CANCELLED = "workflow_cancelled"
    TASK_READY = "task_ready"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_RETRIED = "task_retried"
    TASK_SKIPPED = "task_skipped"
    TASK_CANCELLED = "task_cancelled"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_APPROVED = "approval_approved"
    APPROVAL_REJECTED = "approval_rejected"
    CHECKPOINT_CREATED = "checkpoint_created"
    STATE_RESTORED = "state_restored"


# ============================================================
# Scope
# ============================================================

@dataclass(frozen=True, slots=True)
class WorkflowScope:
    tenant_id: str
    user_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_identifier(self.user_id)


# ============================================================
# Task State
# ============================================================

@dataclass(slots=True)
class WorkflowTaskState:
    task_id: str
    name: str

    agent_role: str

    dependencies: tuple[str, ...] = ()

    priority: WorkflowTaskPriority = (
        WorkflowTaskPriority.NORMAL
    )

    safety_class: WorkflowSafetyClass = (
        WorkflowSafetyClass.ADVISORY
    )

    status: WorkflowTaskStatus = (
        WorkflowTaskStatus.PENDING
    )

    requires_human_approval: bool = False

    approval_id: str | None = None

    approval_status: WorkflowApprovalStatus = (
        WorkflowApprovalStatus.NOT_REQUIRED
    )

    assigned_agent_id: str | None = None

    input_data: dict[str, Any] = field(
        default_factory=dict
    )

    output_data: dict[str, Any] | None = None

    error_message: str | None = None

    attempts: int = 0
    max_retries: int = 1

    progress: float = 0.0

    started_at: datetime | None = None
    completed_at: datetime | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_identifier(self.task_id)

        if not self.name.strip():
            raise WorkflowValidationError(
                "Task name is required"
            )

        if not self.agent_role.strip():
            raise WorkflowValidationError(
                "Agent role is required"
            )

        if len(self.dependencies) > MAX_TASK_DEPENDENCIES:
            raise WorkflowValidationError(
                "Too many task dependencies"
            )

        for dependency in self.dependencies:
            validate_identifier(dependency)

        if len(set(self.dependencies)) != len(
            self.dependencies
        ):
            raise WorkflowDependencyError(
                "Duplicate task dependencies"
            )

        if self.task_id in self.dependencies:
            raise WorkflowDependencyError(
                "Task cannot depend on itself"
            )

        if not 0 <= self.max_retries <= MAX_TASK_RETRIES:
            raise WorkflowValidationError(
                "Invalid retry limit"
            )

        if self.attempts < 0:
            raise WorkflowValidationError(
                "Attempts cannot be negative"
            )

        validate_probability(self.progress)

        if len(self.metadata) > MAX_METADATA_FIELDS:
            raise WorkflowValidationError(
                "Too many task metadata fields"
            )

        if self.safety_class == (
            WorkflowSafetyClass.PHYSICAL_ACTUATION
        ):
            raise WorkflowSafetyError(
                "Physical actuation is not supported"
            )

        if self.safety_class == (
            WorkflowSafetyClass.SAFETY_CRITICAL
        ):
            self.requires_human_approval = True


# ============================================================
# Workflow Event
# ============================================================

@dataclass(frozen=True, slots=True)
class WorkflowEvent:
    event_id: str
    workflow_id: str
    tenant_id: str

    event_type: WorkflowEventType

    actor_id: str

    task_id: str | None = None

    details: Mapping[str, Any] = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_identifier(self.event_id)
        validate_identifier(self.workflow_id)
        validate_identifier(self.tenant_id)
        validate_identifier(self.actor_id)

        if self.task_id is not None:
            validate_identifier(self.task_id)


# ============================================================
# Workflow State
# ============================================================

@dataclass(slots=True)
class WorkflowState:
    workflow_id: str
    tenant_id: str
    created_by: str

    name: str

    tasks: dict[str, WorkflowTaskState]

    execution_mode: WorkflowExecutionMode = (
        WorkflowExecutionMode.DAG
    )

    failure_policy: WorkflowFailurePolicy = (
        WorkflowFailurePolicy.FAIL_FAST
    )

    status: WorkflowStatus = WorkflowStatus.CREATED

    version: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    events: list[WorkflowEvent] = field(
        default_factory=list
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    started_at: datetime | None = None
    completed_at: datetime | None = None

    physical_execution_authorized: bool = False

    def __post_init__(self) -> None:
        validate_identifier(self.workflow_id)
        validate_identifier(self.tenant_id)
        validate_identifier(self.created_by)

        if not self.name.strip():
            raise WorkflowValidationError(
                "Workflow name is required"
            )

        if not 1 <= len(self.tasks) <= MAX_WORKFLOW_TASKS:
            raise WorkflowValidationError(
                "Invalid workflow task count"
            )

        for task_id, task in self.tasks.items():
            if task_id != task.task_id:
                raise WorkflowValidationError(
                    "Task dictionary key mismatch"
                )

        if self.version < 0:
            raise WorkflowValidationError(
                "Workflow version cannot be negative"
            )

        if len(self.metadata) > MAX_METADATA_FIELDS:
            raise WorkflowValidationError(
                "Too many workflow metadata fields"
            )

        if self.physical_execution_authorized:
            raise WorkflowSafetyError(
                "Workflow cannot authorize "
                "physical robot execution"
            )


# ============================================================
# Workflow Progress
# ============================================================

@dataclass(frozen=True, slots=True)
class WorkflowProgress:
    workflow_id: str
    tenant_id: str

    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    skipped_tasks: int
    running_tasks: int
    pending_tasks: int
    waiting_approval_tasks: int

    completion_percentage: float

    workflow_status: WorkflowStatus


# ============================================================
# Workflow Checkpoint
# ============================================================

@dataclass(frozen=True, slots=True)
class WorkflowCheckpoint:
    checkpoint_id: str
    workflow_id: str
    tenant_id: str

    version: int

    state: WorkflowState

    created_by: str
    created_at: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# Provider Protocols
# ============================================================

class WorkflowStateAuthorizer(Protocol):

    async def require_permission(
        self,
        *,
        tenant_id: str,
        user_id: str,
        permission: str,
    ) -> None:
        ...


class WorkflowStateRepository(Protocol):

    async def create(
        self,
        *,
        state: WorkflowState,
    ) -> None:
        ...

    async def get(
        self,
        *,
        tenant_id: str,
        workflow_id: str,
    ) -> WorkflowState | None:
        ...

    async def update(
        self,
        *,
        state: WorkflowState,
        expected_version: int,
    ) -> WorkflowState:
        ...


class WorkflowCheckpointRepository(Protocol):

    async def save_checkpoint(
        self,
        *,
        checkpoint: WorkflowCheckpoint,
    ) -> None:
        ...

    async def get_checkpoint(
        self,
        *,
        tenant_id: str,
        checkpoint_id: str,
    ) -> WorkflowCheckpoint | None:
        ...


class WorkflowStateAuditSink(Protocol):

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


class WorkflowApprovalVerifier(Protocol):

    async def verify_approval(
        self,
        *,
        tenant_id: str,
        approval_id: str,
        workflow_id: str,
        task_id: str,
    ) -> bool:
        ...


# ============================================================
# In-Memory Repository
# ============================================================

class InMemoryWorkflowStateRepository:
    """
    Development and testing repository.

    Uses optimistic concurrency and defensive copies.
    Replace with PostgreSQL for production.
    """

    def __init__(self) -> None:
        self._states: dict[
            tuple[str, str],
            WorkflowState,
        ] = {}

        self._lock = asyncio.Lock()

    async def create(
        self,
        *,
        state: WorkflowState,
    ) -> None:

        key = (
            state.tenant_id,
            state.workflow_id,
        )

        async with self._lock:
            if key in self._states:
                raise WorkflowConflictError(
                    "Workflow already exists"
                )

            self._states[key] = copy.deepcopy(
                state
            )

    async def get(
        self,
        *,
        tenant_id: str,
        workflow_id: str,
    ) -> WorkflowState | None:

        key = (
            tenant_id,
            workflow_id,
        )

        async with self._lock:
            state = self._states.get(key)

            return (
                copy.deepcopy(state)
                if state is not None
                else None
            )

    async def update(
        self,
        *,
        state: WorkflowState,
        expected_version: int,
    ) -> WorkflowState:

        key = (
            state.tenant_id,
            state.workflow_id,
        )

        async with self._lock:
            current = self._states.get(key)

            if current is None:
                raise WorkflowNotFoundError(
                    "Workflow not found"
                )

            if current.version != expected_version:
                raise WorkflowConflictError(
                    "Workflow version conflict"
                )

            updated = copy.deepcopy(state)

            updated.version = (
                expected_version + 1
            )

            updated.updated_at = utc_now()

            self._states[key] = updated

            return copy.deepcopy(updated)


# ============================================================
# In-Memory Checkpoint Repository
# ============================================================

class InMemoryWorkflowCheckpointRepository:

    def __init__(self) -> None:
        self._checkpoints: dict[
            tuple[str, str],
            WorkflowCheckpoint,
        ] = {}

        self._lock = asyncio.Lock()

    async def save_checkpoint(
        self,
        *,
        checkpoint: WorkflowCheckpoint,
    ) -> None:

        key = (
            checkpoint.tenant_id,
            checkpoint.checkpoint_id,
        )

        async with self._lock:
            if key in self._checkpoints:
                raise WorkflowConflictError(
                    "Checkpoint already exists"
                )

            self._checkpoints[key] = copy.deepcopy(
                checkpoint
            )

    async def get_checkpoint(
        self,
        *,
        tenant_id: str,
        checkpoint_id: str,
    ) -> WorkflowCheckpoint | None:

        key = (
            tenant_id,
            checkpoint_id,
        )

        async with self._lock:
            checkpoint = self._checkpoints.get(key)

            return (
                copy.deepcopy(checkpoint)
                if checkpoint is not None
                else None
            )


# ============================================================
# Dependency Graph
# ============================================================

class WorkflowDependencyGraph:

    def __init__(
        self,
        tasks: Mapping[str, WorkflowTaskState],
    ) -> None:

        self.tasks = dict(tasks)
        self.validate()

    def validate(self) -> None:

        for task in self.tasks.values():
            for dependency in task.dependencies:
                if dependency not in self.tasks:
                    raise WorkflowDependencyError(
                        f"Unknown dependency: {dependency}"
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
            raise WorkflowDependencyError(
                "Cyclic workflow dependencies detected"
            )

    def ready_tasks(self) -> list[WorkflowTaskState]:

        return [
            task
            for task in self.tasks.values()
            if task.status in (
                WorkflowTaskStatus.PENDING,
                WorkflowTaskStatus.READY,
            )
            and all(
                self.tasks[dependency].status
                == WorkflowTaskStatus.COMPLETED
                for dependency in task.dependencies
            )
        ]

    def blocked_tasks(self) -> list[WorkflowTaskState]:

        failed_or_blocked = {
            task.task_id
            for task in self.tasks.values()
            if task.status in (
                WorkflowTaskStatus.FAILED,
                WorkflowTaskStatus.SKIPPED,
                WorkflowTaskStatus.CANCELLED,
            )
        }

        changed = True

        while changed:
            changed = False

            for task in self.tasks.values():
                if task.task_id in failed_or_blocked:
                    continue

                if any(
                    dependency in failed_or_blocked
                    for dependency in task.dependencies
                ):
                    failed_or_blocked.add(
                        task.task_id
                    )
                    changed = True

        return [
            task
            for task in self.tasks.values()
            if task.task_id in failed_or_blocked
            and task.status in (
                WorkflowTaskStatus.PENDING,
                WorkflowTaskStatus.READY,
            )
        ]


# ============================================================
# Workflow State Manager
# ============================================================

class WorkflowStateManager:

    TERMINAL_WORKFLOW_STATUSES = frozenset(
        {
            WorkflowStatus.COMPLETED,
            WorkflowStatus.PARTIALLY_COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.CANCELLED,
        }
    )

    TERMINAL_TASK_STATUSES = frozenset(
        {
            WorkflowTaskStatus.COMPLETED,
            WorkflowTaskStatus.FAILED,
            WorkflowTaskStatus.SKIPPED,
            WorkflowTaskStatus.CANCELLED,
        }
    )

    def __init__(
        self,
        *,
        repository: WorkflowStateRepository,
        authorizer: WorkflowStateAuthorizer,
        checkpoint_repository: (
            WorkflowCheckpointRepository | None
        ) = None,
        approval_verifier: (
            WorkflowApprovalVerifier | None
        ) = None,
        audit_sink: WorkflowStateAuditSink | None = None,
    ) -> None:

        self.repository = repository
        self.authorizer = authorizer

        self.checkpoint_repository = (
            checkpoint_repository
        )

        self.approval_verifier = (
            approval_verifier
        )

        self.audit_sink = audit_sink

    # --------------------------------------------------------
    # Authorization
    # --------------------------------------------------------

    async def _authorize(
        self,
        *,
        scope: WorkflowScope,
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
        scope: WorkflowScope,
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
    # Event Recording
    # --------------------------------------------------------

    @staticmethod
    def _append_event(
        state: WorkflowState,
        *,
        actor_id: str,
        event_type: WorkflowEventType,
        task_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:

        event = WorkflowEvent(
            event_id=generate_id(),
            workflow_id=state.workflow_id,
            tenant_id=state.tenant_id,
            event_type=event_type,
            actor_id=actor_id,
            task_id=task_id,
            details=copy.deepcopy(
                dict(details or {})
            ),
        )

        state.events.append(event)

        if len(state.events) > MAX_EVENT_HISTORY:
            state.events = state.events[
                -MAX_EVENT_HISTORY:
            ]

    # --------------------------------------------------------
    # Load Workflow
    # --------------------------------------------------------

    async def get_workflow(
        self,
        *,
        scope: WorkflowScope,
        workflow_id: str,
    ) -> WorkflowState:

        validate_identifier(workflow_id)

        await self._authorize(
            scope=scope,
            permission="workflows:read",
        )

        state = await self.repository.get(
            tenant_id=scope.tenant_id,
            workflow_id=workflow_id,
        )

        if state is None:
            raise WorkflowNotFoundError(
                "Workflow not found"
            )

        if state.tenant_id != scope.tenant_id:
            raise WorkflowPermissionError(
                "Workflow tenant mismatch"
            )

        return state

    # --------------------------------------------------------
    # Save Workflow
    # --------------------------------------------------------

    async def _save(
        self,
        *,
        scope: WorkflowScope,
        state: WorkflowState,
    ) -> WorkflowState:

        if state.tenant_id != scope.tenant_id:
            raise WorkflowPermissionError(
                "Workflow tenant mismatch"
            )

        return await self.repository.update(
            state=state,
            expected_version=state.version,
        )

    # --------------------------------------------------------
    # Create Workflow
    # --------------------------------------------------------

    async def create_workflow(
        self,
        *,
        scope: WorkflowScope,
        name: str,
        tasks: Sequence[WorkflowTaskState],
        execution_mode: WorkflowExecutionMode = (
            WorkflowExecutionMode.DAG
        ),
        failure_policy: WorkflowFailurePolicy = (
            WorkflowFailurePolicy.FAIL_FAST
        ),
        metadata: Mapping[str, Any] | None = None,
    ) -> WorkflowState:

        await self._authorize(
            scope=scope,
            permission="workflows:create",
        )

        task_map = {
            task.task_id: copy.deepcopy(task)
            for task in tasks
        }

        if len(task_map) != len(tasks):
            raise WorkflowValidationError(
                "Duplicate task identifiers"
            )

        WorkflowDependencyGraph(task_map)

        state = WorkflowState(
            workflow_id=generate_id(),
            tenant_id=scope.tenant_id,
            created_by=scope.user_id,
            name=name,
            tasks=task_map,
            execution_mode=execution_mode,
            failure_policy=failure_policy,
            metadata=dict(metadata or {}),
        )

        self._append_event(
            state,
            actor_id=scope.user_id,
            event_type=(
                WorkflowEventType.WORKFLOW_CREATED
            ),
        )

        await self.repository.create(
            state=state
        )

        await self._audit(
            scope=scope,
            action="workflow.created",
            resource_id=state.workflow_id,
            metadata={
                "task_count": len(state.tasks),
            },
        )

        return state

    # --------------------------------------------------------
    # Validate Workflow
    # --------------------------------------------------------

    async def validate_workflow(
        self,
        *,
        scope: WorkflowScope,
        workflow_id: str,
    ) -> WorkflowState:

        await self._authorize(
            scope=scope,
            permission="workflows:update",
        )

        state = await self.get_workflow(
            scope=scope,
            workflow_id=workflow_id,
        )

        if state.status != WorkflowStatus.CREATED:
            raise WorkflowTransitionError(
                "Only created workflows can be validated"
            )

        WorkflowDependencyGraph(
            state.tasks
        )

        state.status = WorkflowStatus.VALIDATED

        self._append_event(
            state,
            actor_id=scope.user_id,
            event_type=(
                WorkflowEventType.WORKFLOW_VALIDATED
            ),
        )

        return await self._save(
            scope=scope,
            state=state,
        )

    # --------------------------------------------------------
    # Start Workflow
    # --------------------------------------------------------

    async def start_workflow(
        self,
        *,
        scope: WorkflowScope,
        workflow_id: str,
    ) -> WorkflowState:

        await self._authorize(
            scope=scope,
            permission="workflows:execute",
        )

        state = await self.get_workflow(
            scope=scope,
            workflow_id=workflow_id,
        )

        if state.status not in (
            WorkflowStatus.VALIDATED,
            WorkflowStatus.READY,
        ):
            raise WorkflowTransitionError(
                "Workflow cannot be started "
                "from its current state"
            )

        state.status = WorkflowStatus.RUNNING

        if state.started_at is None:
            state.started_at = utc_now()

        self._append_event(
            state,
            actor_id=scope.user_id,
            event_type=(
                WorkflowEventType.WORKFLOW_STARTED
            ),
        )

        return await self._save(
            scope=scope,
            state=state,
        )

    # --------------------------------------------------------
    # Pause Workflow
    # --------------------------------------------------------

    async def pause_workflow(
        self,
        *,
        scope: WorkflowScope,
        workflow_id: str,
    ) -> WorkflowState:

        await self._authorize(
            scope=scope,
            permission="workflows:update",
        )

        state = await self.get_workflow(
            scope=scope,
            workflow_id=workflow_id,
        )

        if state.status not in (
            WorkflowStatus.RUNNING,
            WorkflowStatus.WAITING_APPROVAL,
        ):
            raise WorkflowTransitionError(
                "Workflow cannot be paused"
            )

        state.status = WorkflowStatus.PAUSED

        self._append_event(
            state,
            actor_id=scope.user_id,
            event_type=(
                WorkflowEventType.WORKFLOW_PAUSED
            ),
        )

        return await self._save(
            scope=scope,
            state=state,
        )

    # --------------------------------------------------------
    # Resume Workflow
    # --------------------------------------------------------

    async def resume_workflow(
        self,
        *,
        scope: WorkflowScope,
        workflow_id: str,
    ) -> WorkflowState:

        await self._authorize(
            scope=scope,
            permission="workflows:execute",
        )

        state = await self.get_workflow(
            scope=scope,
            workflow_id=workflow_id,
        )

        if state.status not in (
            WorkflowStatus.PAUSED,
            WorkflowStatus.WAITING_APPROVAL,
        ):
            raise WorkflowTransitionError(
                "Workflow cannot be resumed"
            )

        state.status = WorkflowStatus.RUNNING

        self._append_event(
            state,
            actor_id=scope.user_id,
            event_type=(
                WorkflowEventType.WORKFLOW_RESUMED
            ),
        )

        return await self._save(
            scope=scope,
            state=state,
        )

    # --------------------------------------------------------
    # Ready Tasks
    # --------------------------------------------------------

    async def get_ready_tasks(
        self,
        *,
        scope: WorkflowScope,
        workflow_id: str,
    ) -> list[WorkflowTaskState]:

        state = await self.get_workflow(
            scope=scope,
            workflow_id=workflow_id,
        )

        if state.status != WorkflowStatus.RUNNING:
            return []

        graph = WorkflowDependencyGraph(
            state.tasks
        )

        return [
            copy.deepcopy(task)
            for task in graph.ready_tasks()
        ]

    # --------------------------------------------------------
    # Start Task
    # --------------------------------------------------------

    async def start_task(
        self,
        *,
        scope: WorkflowScope,
        workflow_id: str,
        task_id: str,
        agent_id: str,
    ) -> WorkflowState:

        validate_identifier(task_id)
        validate_identifier(agent_id)

        await self._authorize(
            scope=scope,
            permission="workflows:execute",
        )

        state = await self.get_workflow(
            scope=scope,
            workflow_id=workflow_id,
        )

        if state.status != WorkflowStatus.RUNNING:
            raise WorkflowTransitionError(
                "Workflow is not running"
            )

        task = state.tasks.get(task_id)

        if task is None:
            raise WorkflowNotFoundError(
                "Task not found"
            )

        if task.status not in (
            WorkflowTaskStatus.PENDING,
            WorkflowTaskStatus.READY,
        ):
            raise WorkflowTransitionError(
                "Task cannot be started"
            )

        if not all(
            state.tasks[dependency].status
            == WorkflowTaskStatus.COMPLETED
            for dependency in task.dependencies
        ):
            raise WorkflowDependencyError(
                "Task dependencies are incomplete"
            )

        if task.requires_human_approval:
            if task.approval_status != (
                WorkflowApprovalStatus.APPROVED
            ):
                raise WorkflowApprovalError(
                    "Human approval is required"
                )

            if task.approval_id is None:
                raise WorkflowApprovalError(
                    "Approval ID is missing"
                )

            if self.approval_verifier is None:
                raise WorkflowApprovalError(
                    "Approval verifier unavailable"
                )

            valid = await self.approval_verifier.verify_approval(
                tenant_id=scope.tenant_id,
                approval_id=task.approval_id,
                workflow_id=workflow_id,
                task_id=task_id,
            )

            if not valid:
                raise WorkflowApprovalError(
                    "Approval verification failed"
                )

        task.status = WorkflowTaskStatus.RUNNING
        task.assigned_agent_id = agent_id

        task.attempts += 1
        task.started_at = utc_now()

        task.error_message = None

        self._append_event(
            state,
            actor_id=scope.user_id,
            event_type=WorkflowEventType.TASK_STARTED,
            task_id=task_id,
            details={
                "agent_id": agent_id,
                "attempt": task.attempts,
            },
        )

        return await self._save(
            scope=scope,
            state=state,
        )

    # --------------------------------------------------------
    # Complete Task
    # --------------------------------------------------------

    async def complete_task(
        self,
        *,
        scope: WorkflowScope,
        workflow_id: str,
        task_id: str,
        output_data: Mapping[str, Any],
    ) -> WorkflowState:

        await self._authorize(
            scope=scope,
            permission="workflows:execute",
        )

        state = await self.get_workflow(
            scope=scope,
            workflow_id=workflow_id,
        )

        task = state.tasks.get(task_id)

        if task is None:
            raise WorkflowNotFoundError(
                "Task not found"
            )

        if task.status != WorkflowTaskStatus.RUNNING:
            raise WorkflowTransitionError(
                "Only running tasks can be completed"
            )

        if output_data.get(
            "physical_execution_authorized"
        ) is True:
            raise WorkflowSafetyError(
                "Task output cannot authorize "
                "physical execution"
            )

        task.output_data = copy.deepcopy(
            dict(output_data)
        )

        task.status = WorkflowTaskStatus.COMPLETED
        task.progress = 1.0
        task.completed_at = utc_now()

        self._append_event(
            state,
            actor_id=scope.user_id,
            event_type=WorkflowEventType.TASK_COMPLETED,
            task_id=task_id,
        )

        self._refresh_workflow_status(
            state
        )

        return await self._save(
            scope=scope,
            state=state,
        )

    # --------------------------------------------------------
    # Fail Task
    # --------------------------------------------------------

    async def fail_task(
        self,
        *,
        scope: WorkflowScope,
        workflow_id: str,
        task_id: str,
        error_message: str,
    ) -> WorkflowState:

        await self._authorize(
            scope=scope,
            permission="workflows:execute",
        )

        state = await self.get_workflow(
            scope=scope,
            workflow_id=workflow_id,
        )

        task = state.tasks.get(task_id)

        if task is None:
            raise WorkflowNotFoundError(
                "Task not found"
            )

        if task.status != WorkflowTaskStatus.RUNNING:
            raise WorkflowTransitionError(
                "Only running tasks can fail"
            )

        task.status = WorkflowTaskStatus.FAILED
        task.error_message = error_message[:2000]
        task.completed_at = utc_now()

        self._append_event(
            state,
            actor_id=scope.user_id,
            event_type=WorkflowEventType.TASK_FAILED,
            task_id=task_id,
            details={
                "error": task.error_message,
            },
        )

        if state.failure_policy == (
            WorkflowFailurePolicy.FAIL_FAST
        ):
            state.status = WorkflowStatus.FAILED
            state.completed_at = utc_now()

        else:
            self._refresh_workflow_status(
                state
            )

        return await self._save(
            scope=scope,
            state=state,
        )

    # --------------------------------------------------------
    # Retry Task
    # --------------------------------------------------------

    async def retry_task(
        self,
        *,
        scope: WorkflowScope,
        workflow_id: str,
        task_id: str,
    ) -> WorkflowState:

        await self._authorize(
            scope=scope,
            permission="workflows:execute",
        )

        state = await self.get_workflow(
            scope=scope,
            workflow_id=workflow_id,
        )

        task = state.tasks.get(task_id)

        if task is None:
            raise WorkflowNotFoundError(
                "Task not found"
            )

        if task.status != WorkflowTaskStatus.FAILED:
            raise WorkflowTransitionError(
                "Only failed tasks can be retried"
            )

        if task.attempts > task.max_retries:
            raise WorkflowTransitionError(
                "Task retry limit exceeded"
            )

        if state.status in (
            WorkflowStatus.COMPLETED,
            WorkflowStatus.CANCELLED,
        ):
            raise WorkflowTransitionError(
                "Terminal workflow cannot retry tasks"
            )

        task.status = WorkflowTaskStatus.PENDING
        task.error_message = None
        task.completed_at = None
        task.assigned_agent_id = None

        state.status = WorkflowStatus.RUNNING
        state.completed_at = None

        self._append_event(
            state,
            actor_id=scope.user_id,
            event_type=WorkflowEventType.TASK_RETRIED,
            task_id=task_id,
        )

        return await self._save(
            scope=scope,
            state=state,
        )

    # --------------------------------------------------------
    # Request Human Approval
    # --------------------------------------------------------

    async def request_task_approval(
        self,
        *,
        scope: WorkflowScope,
        workflow_id: str,
        task_id: str,
        approval_id: str,
    ) -> WorkflowState:

        validate_identifier(approval_id)

        await self._authorize(
            scope=scope,
            permission="workflows:update",
        )

        state = await self.get_workflow(
            scope=scope,
            workflow_id=workflow_id,
        )

        task = state.tasks.get(task_id)

        if task is None:
            raise WorkflowNotFoundError(
                "Task not found"
            )

        if not task.requires_human_approval:
            raise WorkflowApprovalError(
                "Task does not require human approval"
            )

        if task.status not in (
            WorkflowTaskStatus.PENDING,
            WorkflowTaskStatus.READY,
        ):
            raise WorkflowTransitionError(
                "Approval cannot be requested "
                "for this task state"
            )

        task.approval_id = approval_id

        task.approval_status = (
            WorkflowApprovalStatus.PENDING
        )

        task.status = (
            WorkflowTaskStatus.WAITING_APPROVAL
        )

        state.status = (
            WorkflowStatus.WAITING_APPROVAL
        )

        self._append_event(
            state,
            actor_id=scope.user_id,
            event_type=(
                WorkflowEventType.APPROVAL_REQUESTED
            ),
            task_id=task_id,
            details={
                "approval_id": approval_id,
            },
        )

        return await self._save(
            scope=scope,
            state=state,
        )

    # --------------------------------------------------------
    # Apply Approval Decision
    # --------------------------------------------------------

    async def apply_approval_decision(
        self,
        *,
        scope: WorkflowScope,
        workflow_id: str,
        task_id: str,
        approval_status: WorkflowApprovalStatus,
    ) -> WorkflowState:

        await self._authorize(
            scope=scope,
            permission="workflows:update",
        )

        if approval_status not in (
            WorkflowApprovalStatus.APPROVED,
            WorkflowApprovalStatus.REJECTED,
            WorkflowApprovalStatus.EXPIRED,
        ):
            raise WorkflowApprovalError(
                "Invalid approval decision"
            )

        state = await self.get_workflow(
            scope=scope,
            workflow_id=workflow_id,
        )

        task = state.tasks.get(task_id)

        if task is None:
            raise WorkflowNotFoundError(
                "Task not found"
            )

        if task.status != (
            WorkflowTaskStatus.WAITING_APPROVAL
        ):
            raise WorkflowTransitionError(
                "Task is not awaiting approval"
            )

        if task.approval_id is None:
            raise WorkflowApprovalError(
                "Approval ID is missing"
            )

        if approval_status == (
            WorkflowApprovalStatus.APPROVED
        ):
            if self.approval_verifier is None:
                raise WorkflowApprovalError(
                    "Approval verifier unavailable"
                )

            valid = await self.approval_verifier.verify_approval(
                tenant_id=scope.tenant_id,
                approval_id=task.approval_id,
                workflow_id=workflow_id,
                task_id=task_id,
            )

            if not valid:
                raise WorkflowApprovalError(
                    "Approval verification failed"
                )

            task.approval_status = approval_status
            task.status = WorkflowTaskStatus.READY

            event_type = (
                WorkflowEventType.APPROVAL_APPROVED
            )

        else:
            task.approval_status = approval_status
            task.status = WorkflowTaskStatus.FAILED
            task.error_message = (
                "Human approval rejected or expired"
            )
            task.completed_at = utc_now()

            event_type = (
                WorkflowEventType.APPROVAL_REJECTED
            )

        self._append_event(
            state,
            actor_id=scope.user_id,
            event_type=event_type,
            task_id=task_id,
        )

        self._refresh_workflow_status(
            state
        )

        return await self._save(
            scope=scope,
            state=state,
        )

    # --------------------------------------------------------
    # Cancel Workflow
    # --------------------------------------------------------

    async def cancel_workflow(
        self,
        *,
        scope: WorkflowScope,
        workflow_id: str,
    ) -> WorkflowState:

        await self._authorize(
            scope=scope,
            permission="workflows:cancel",
        )

        state = await self.get_workflow(
            scope=scope,
            workflow_id=workflow_id,
        )

        if state.status in (
            WorkflowStatus.COMPLETED,
            WorkflowStatus.CANCELLED,
        ):
            raise WorkflowTransitionError(
                "Workflow cannot be cancelled"
            )

        state.status = WorkflowStatus.CANCELLED
        state.completed_at = utc_now()

        for task in state.tasks.values():
            if task.status not in self.TERMINAL_TASK_STATUSES:
                task.status = (
                    WorkflowTaskStatus.CANCELLED
                )
                task.completed_at = utc_now()

        self._append_event(
            state,
            actor_id=scope.user_id,
            event_type=(
                WorkflowEventType.WORKFLOW_CANCELLED
            ),
        )

        return await self._save(
            scope=scope,
            state=state,
        )

    # --------------------------------------------------------
    # Refresh Workflow Status
    # --------------------------------------------------------

    @staticmethod
    def _refresh_workflow_status(
        state: WorkflowState,
    ) -> None:

        statuses = [
            task.status
            for task in state.tasks.values()
        ]

        if all(
            status == WorkflowTaskStatus.COMPLETED
            for status in statuses
        ):
            state.status = WorkflowStatus.COMPLETED
            state.completed_at = utc_now()
            return

        if any(
            status == WorkflowTaskStatus.RUNNING
            for status in statuses
        ):
            state.status = WorkflowStatus.RUNNING
            return

        if any(
            status == WorkflowTaskStatus.WAITING_APPROVAL
            for status in statuses
        ):
            state.status = (
                WorkflowStatus.WAITING_APPROVAL
            )
            return

        terminal = all(
            status in (
                WorkflowTaskStatus.COMPLETED,
                WorkflowTaskStatus.FAILED,
                WorkflowTaskStatus.SKIPPED,
                WorkflowTaskStatus.CANCELLED,
            )
            for status in statuses
        )

        if terminal:
            completed = any(
                status == WorkflowTaskStatus.COMPLETED
                for status in statuses
            )

            state.status = (
                WorkflowStatus.PARTIALLY_COMPLETED
                if completed
                else WorkflowStatus.FAILED
            )

            state.completed_at = utc_now()

        else:
            state.status = WorkflowStatus.RUNNING

    # --------------------------------------------------------
    # Workflow Progress
    # --------------------------------------------------------

    @staticmethod
    def calculate_progress(
        state: WorkflowState,
    ) -> WorkflowProgress:

        tasks = list(state.tasks.values())
        total = len(tasks)

        def count(
            status: WorkflowTaskStatus,
        ) -> int:
            return sum(
                task.status == status
                for task in tasks
            )

        completed = count(
            WorkflowTaskStatus.COMPLETED
        )

        average_progress = (
            sum(task.progress for task in tasks)
            / total
            if total
            else 0.0
        )

        return WorkflowProgress(
            workflow_id=state.workflow_id,
            tenant_id=state.tenant_id,
            total_tasks=total,
            completed_tasks=completed,
            failed_tasks=count(
                WorkflowTaskStatus.FAILED
            ),
            skipped_tasks=count(
                WorkflowTaskStatus.SKIPPED
            ),
            running_tasks=count(
                WorkflowTaskStatus.RUNNING
            ),
            pending_tasks=(
                count(WorkflowTaskStatus.PENDING)
                + count(WorkflowTaskStatus.READY)
            ),
            waiting_approval_tasks=count(
                WorkflowTaskStatus.WAITING_APPROVAL
            ),
            completion_percentage=round(
                average_progress * 100,
                2,
            ),
            workflow_status=state.status,
        )

    # --------------------------------------------------------
    # Create Checkpoint
    # --------------------------------------------------------

    async def create_checkpoint(
        self,
        *,
        scope: WorkflowScope,
        workflow_id: str,
    ) -> WorkflowCheckpoint:

        await self._authorize(
            scope=scope,
            permission="workflows:checkpoint",
        )

        if self.checkpoint_repository is None:
            raise WorkflowStateError(
                "Checkpoint repository unavailable"
            )

        state = await self.get_workflow(
            scope=scope,
            workflow_id=workflow_id,
        )

        checkpoint = WorkflowCheckpoint(
            checkpoint_id=generate_id(),
            workflow_id=workflow_id,
            tenant_id=scope.tenant_id,
            version=state.version,
            state=copy.deepcopy(state),
            created_by=scope.user_id,
        )

        await self.checkpoint_repository.save_checkpoint(
            checkpoint=checkpoint
        )

        await self._audit(
            scope=scope,
            action="workflow.checkpoint_created",
            resource_id=checkpoint.checkpoint_id,
            metadata={
                "workflow_id": workflow_id,
                "version": state.version,
            },
        )

        return checkpoint

    # --------------------------------------------------------
    # Restore Checkpoint
    # --------------------------------------------------------

    async def restore_checkpoint(
        self,
        *,
        scope: WorkflowScope,
        checkpoint_id: str,
    ) -> WorkflowState:

        await self._authorize(
            scope=scope,
            permission="workflows:restore",
        )

        if self.checkpoint_repository is None:
            raise WorkflowStateError(
                "Checkpoint repository unavailable"
            )

        checkpoint = (
            await self.checkpoint_repository.get_checkpoint(
                tenant_id=scope.tenant_id,
                checkpoint_id=checkpoint_id,
            )
        )

        if checkpoint is None:
            raise WorkflowNotFoundError(
                "Checkpoint not found"
            )

        current = await self.get_workflow(
            scope=scope,
            workflow_id=checkpoint.workflow_id,
        )

        if current.status == WorkflowStatus.RUNNING:
            raise WorkflowTransitionError(
                "Pause workflow before restoring"
            )

        restored = copy.deepcopy(
            checkpoint.state
        )

        restored.version = current.version

        restored.physical_execution_authorized = False

        for task in restored.tasks.values():
            if task.status == WorkflowTaskStatus.RUNNING:
                task.status = WorkflowTaskStatus.PENDING
                task.assigned_agent_id = None

        restored.status = WorkflowStatus.PAUSED
        restored.completed_at = None

        self._append_event(
            restored,
            actor_id=scope.user_id,
            event_type=(
                WorkflowEventType.STATE_RESTORED
            ),
            details={
                "checkpoint_id": checkpoint_id,
            },
        )

        return await self._save(
            scope=scope,
            state=restored,
        )

    # --------------------------------------------------------
    # Workflow Report
    # --------------------------------------------------------

    @staticmethod
    def generate_state_report(
        state: WorkflowState,
    ) -> dict[str, Any]:

        progress = (
            WorkflowStateManager.calculate_progress(
                state
            )
        )

        return {
            "workflow_id": state.workflow_id,
            "tenant_id": state.tenant_id,
            "name": state.name,
            "status": state.status.value,
            "version": state.version,
            "execution_mode": (
                state.execution_mode.value
            ),
            "failure_policy": (
                state.failure_policy.value
            ),
            "progress": {
                "total_tasks": progress.total_tasks,
                "completed_tasks": (
                    progress.completed_tasks
                ),
                "failed_tasks": progress.failed_tasks,
                "skipped_tasks": progress.skipped_tasks,
                "running_tasks": progress.running_tasks,
                "pending_tasks": progress.pending_tasks,
                "waiting_approval_tasks": (
                    progress.waiting_approval_tasks
                ),
                "completion_percentage": (
                    progress.completion_percentage
                ),
            },
            "tasks": [
                {
                    "task_id": task.task_id,
                    "name": task.name,
                    "agent_role": task.agent_role,
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
                    "progress": task.progress,
                    "error": task.error_message,
                }
                for task in state.tasks.values()
            ],
            "created_at": (
                state.created_at.isoformat()
            ),
            "updated_at": (
                state.updated_at.isoformat()
            ),
            "started_at": (
                state.started_at.isoformat()
                if state.started_at
                else None
            ),
            "completed_at": (
                state.completed_at.isoformat()
                if state.completed_at
                else None
            ),
            "event_count": len(state.events),
            "safety": {
                "physical_execution_authorized": False,
                "emergency_stop_reset_allowed": False,
                "safety_interlock_bypass_allowed": False,
                "human_review_required_for_critical_tasks": True,
            },
        }


# ============================================================
# Factory
# ============================================================

def create_workflow_state_manager(
    *,
    authorizer: WorkflowStateAuthorizer,
    repository: WorkflowStateRepository | None = None,
    checkpoint_repository: (
        WorkflowCheckpointRepository | None
    ) = None,
    approval_verifier: (
        WorkflowApprovalVerifier | None
    ) = None,
    audit_sink: WorkflowStateAuditSink | None = None,
) -> WorkflowStateManager:

    return WorkflowStateManager(
        repository=(
            repository
            or InMemoryWorkflowStateRepository()
        ),
        authorizer=authorizer,
        checkpoint_repository=(
            checkpoint_repository
            or InMemoryWorkflowCheckpointRepository()
        ),
        approval_verifier=approval_verifier,
        audit_sink=audit_sink,
    )


# ============================================================
# Exports
# ============================================================

__all__ = [
    "WorkflowStateError",
    "WorkflowValidationError",
    "WorkflowNotFoundError",
    "WorkflowPermissionError",
    "WorkflowConflictError",
    "WorkflowTransitionError",
    "WorkflowDependencyError",
    "WorkflowApprovalError",
    "WorkflowSafetyError",
    "WorkflowStatus",
    "WorkflowTaskStatus",
    "WorkflowTaskPriority",
    "WorkflowExecutionMode",
    "WorkflowFailurePolicy",
    "WorkflowSafetyClass",
    "WorkflowApprovalStatus",
    "WorkflowEventType",
    "WorkflowScope",
    "WorkflowTaskState",
    "WorkflowEvent",
    "WorkflowState",
    "WorkflowProgress",
    "WorkflowCheckpoint",
    "WorkflowStateAuthorizer",
    "WorkflowStateRepository",
    "WorkflowCheckpointRepository",
    "WorkflowStateAuditSink",
    "WorkflowApprovalVerifier",
    "InMemoryWorkflowStateRepository",
    "InMemoryWorkflowCheckpointRepository",
    "WorkflowDependencyGraph",
    "WorkflowStateManager",
    "create_workflow_state_manager",
]
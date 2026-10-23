"""Action Scheduler module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/planning/action_scheduler.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Action Scheduler — Safety-Gated Robotics Planning

Python 3.11+
Dependencies: Python standard library only.

Features:
    - Robot action scheduling
    - Multi-robot task queues
    - Priority-based scheduling
    - Dependency-aware execution planning
    - Task deadlines and execution windows
    - Robot capability matching
    - Resource reservation and conflict detection
    - Action lifecycle management
    - Retry and exponential backoff
    - Task cancellation and suspension
    - Emergency-stop scheduling interlock
    - Human approval requirements
    - Robot safety-state validation
    - Simulation and dry-run support
    - Action timeouts
    - Audit logging
    - Schedule statistics
    - JSON export
    - Thread-safe operations
    - Unit tests

IMPORTANT:
    This module schedules and evaluates proposed actions.

    It does NOT send actuator commands, move robots,
    control motors, or bypass physical safety systems.

    Physical execution requires a separate certified
    robot controller, hardware safety interlocks,
    fresh authorization, and independent validation.
"""

from __future__ import annotations

import heapq
import json
import math
import threading
import time
import unittest
import uuid

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class ActionSchedulerError(Exception):
    pass


class ActionValidationError(ActionSchedulerError):
    pass


class ActionNotFoundError(ActionSchedulerError):
    pass


class ActionAuthorizationError(ActionSchedulerError):
    pass


class ActionDependencyError(ActionSchedulerError):
    pass


class ActionConflictError(ActionSchedulerError):
    pass


class ActionSafetyError(ActionSchedulerError):
    pass


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ActionValidationError(
            "Datetime must include timezone information"
        )

    return value.astimezone(timezone.utc)


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return json_safe(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set, frozenset, deque)):
        return [json_safe(item) for item in value]

    return value


# ============================================================
# ENUMS
# ============================================================

class ActionPriority(int, Enum):
    LOW = 10
    NORMAL = 20
    HIGH = 30
    CRITICAL = 40
    EMERGENCY = 50


class ActionStatus(str, Enum):
    PENDING = "pending"
    WAITING_DEPENDENCY = "waiting_dependency"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_RESOURCE = "waiting_resource"
    WAITING_ROBOT = "waiting_robot"
    SCHEDULED = "scheduled"
    READY = "ready"
    DISPATCHED_SIMULATION = "dispatched_simulation"
    SIMULATING = "simulating"
    RETRY_WAIT = "retry_wait"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    BLOCKED_SAFETY = "blocked_safety"


class ActionType(str, Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PICK = "pick"
    PLACE = "place"
    INSPECTION = "inspection"
    IMAGE_CAPTURE = "image_capture"
    VIDEO_CAPTURE = "video_capture"
    SENSOR_SCAN = "sensor_scan"
    DIAGNOSTICS = "diagnostics"
    MAINTENANCE = "maintenance"
    CALIBRATION = "calibration"
    CHARGING = "charging"
    DATA_COLLECTION = "data_collection"
    SIMULATION = "simulation"
    SAFETY_INSPECTION = "safety_inspection"
    OTHER = "other"


class RobotSafetyState(str, Enum):
    SAFE = "safe"
    CAUTION = "caution"
    UNSAFE = "unsafe"
    EMERGENCY_STOP = "emergency_stop"
    UNKNOWN = "unknown"


class RobotOperationalState(str, Enum):
    IDLE = "idle"
    AVAILABLE = "available"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    FAULTED = "faulted"


class SchedulerMode(str, Enum):
    SIMULATION = "simulation"
    DRY_RUN = "dry_run"


class DependencyCondition(str, Enum):
    ALL_COMPLETED = "all_completed"
    ANY_COMPLETED = "any_completed"


class ApprovalStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ActionSchedulerConfig:
    mode: SchedulerMode = SchedulerMode.SIMULATION

    maximum_actions: int = 100_000
    maximum_queue_size: int = 50_000

    maximum_concurrent_per_robot: int = 1
    maximum_concurrent_global: int = 20

    default_action_timeout_seconds: float = 120.0
    maximum_action_timeout_seconds: float = 3600.0

    default_retry_limit: int = 2
    maximum_retry_limit: int = 5

    retry_base_delay_seconds: float = 2.0
    retry_max_delay_seconds: float = 120.0

    robot_state_max_age_seconds: float = 10.0

    require_human_approval_for_critical: bool = True
    require_human_approval_for_motion: bool = True

    allow_caution_state_for_nonmotion: bool = False

    maximum_audit_events: int = 20_000

    def __post_init__(self) -> None:
        positive = (
            "maximum_actions",
            "maximum_queue_size",
            "maximum_concurrent_per_robot",
            "maximum_concurrent_global",
            "maximum_retry_limit",
            "maximum_audit_events",
        )

        for name in positive:
            value = getattr(self, name)

            if not isinstance(value, int) or value <= 0:
                raise ActionValidationError(
                    f"{name} must be a positive integer"
                )

        if not 0 <= self.default_retry_limit <= self.maximum_retry_limit:
            raise ActionValidationError(
                "Invalid default retry limit"
            )

        if (
            self.default_action_timeout_seconds <= 0
            or self.maximum_action_timeout_seconds
            < self.default_action_timeout_seconds
        ):
            raise ActionValidationError(
                "Invalid action timeout configuration"
            )

        if (
            self.retry_base_delay_seconds <= 0
            or self.retry_max_delay_seconds
            < self.retry_base_delay_seconds
        ):
            raise ActionValidationError(
                "Invalid retry delay configuration"
            )

        if self.robot_state_max_age_seconds <= 0:
            raise ActionValidationError(
                "Robot state maximum age must be positive"
            )


# ============================================================
# SECURITY CONTEXT
# ============================================================

@dataclass(frozen=True)
class SchedulerAccessContext:
    tenant_id: str
    user_id: str

    allowed_robot_ids: frozenset[str] = frozenset()

    roles: frozenset[str] = frozenset()

    can_schedule: bool = False
    can_cancel: bool = False
    can_approve: bool = False
    can_manage_robots: bool = False


# ============================================================
# ROBOT MODELS
# ============================================================

@dataclass(frozen=True)
class RobotSchedulerState:
    robot_id: str
    tenant_id: str

    operational_state: RobotOperationalState = (
        RobotOperationalState.IDLE
    )

    safety_state: RobotSafetyState = (
        RobotSafetyState.UNKNOWN
    )

    capabilities: frozenset[str] = frozenset()

    available_resources: frozenset[str] = frozenset()

    battery_percentage: float | None = None

    active_faults: tuple[str, ...] = ()

    emergency_stop_active: bool = False

    last_updated: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "last_updated",
            ensure_utc(self.last_updated),
        )

        if self.battery_percentage is not None:
            if not (
                0 <= self.battery_percentage <= 100
            ):
                raise ActionValidationError(
                    "Battery percentage must be 0–100"
                )


# ============================================================
# ACTION SPECIFICATION
# ============================================================

@dataclass(frozen=True)
class ScheduledAction:
    action_id: str
    tenant_id: str
    robot_id: str

    action_type: ActionType
    name: str

    parameters: dict[str, Any] = field(
        default_factory=dict
    )

    priority: ActionPriority = ActionPriority.NORMAL

    required_capabilities: frozenset[str] = frozenset()

    required_resources: frozenset[str] = frozenset()

    dependencies: tuple[str, ...] = ()

    dependency_condition: DependencyCondition = (
        DependencyCondition.ALL_COMPLETED
    )

    earliest_start: datetime | None = None
    deadline: datetime | None = None

    estimated_duration_seconds: float = 10.0
    timeout_seconds: float = 120.0

    maximum_retries: int = 2

    requires_human_approval: bool = False

    created_by: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.action_id.strip():
            raise ActionValidationError(
                "Action ID cannot be empty"
            )

        if not self.tenant_id.strip():
            raise ActionValidationError(
                "Tenant ID cannot be empty"
            )

        if not self.robot_id.strip():
            raise ActionValidationError(
                "Robot ID cannot be empty"
            )

        if not self.name.strip():
            raise ActionValidationError(
                "Action name cannot be empty"
            )

        if (
            not math.isfinite(self.estimated_duration_seconds)
            or self.estimated_duration_seconds <= 0
        ):
            raise ActionValidationError(
                "Estimated duration must be positive"
            )

        if (
            not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ActionValidationError(
                "Timeout must be positive"
            )

        if self.maximum_retries < 0:
            raise ActionValidationError(
                "Maximum retries cannot be negative"
            )

        if self.action_id in self.dependencies:
            raise ActionDependencyError(
                "Action cannot depend on itself"
            )

        if len(set(self.dependencies)) != len(self.dependencies):
            raise ActionDependencyError(
                "Duplicate action dependencies"
            )

        object.__setattr__(
            self,
            "created_at",
            ensure_utc(self.created_at),
        )

        if self.earliest_start is not None:
            object.__setattr__(
                self,
                "earliest_start",
                ensure_utc(self.earliest_start),
            )

        if self.deadline is not None:
            object.__setattr__(
                self,
                "deadline",
                ensure_utc(self.deadline),
            )

        if (
            self.earliest_start is not None
            and self.deadline is not None
            and self.earliest_start >= self.deadline
        ):
            raise ActionValidationError(
                "Earliest start must precede deadline"
            )


# ============================================================
# ACTION RUNTIME
# ============================================================

@dataclass
class ActionRuntime:
    action: ScheduledAction

    status: ActionStatus = ActionStatus.PENDING

    approval_status: ApprovalStatus = (
        ApprovalStatus.NOT_REQUIRED
    )

    approval_user_id: str | None = None
    approved_at: datetime | None = None

    attempts: int = 0

    scheduled_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    next_retry_at: datetime | None = None

    simulation_result: dict[str, Any] | None = None

    last_error: str | None = None

    status_history: list[dict[str, Any]] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self)


# ============================================================
# SCHEDULING DECISION
# ============================================================

@dataclass(frozen=True)
class SchedulingDecision:
    action_id: str
    robot_id: str

    allowed: bool

    status: ActionStatus

    reasons: tuple[str, ...]

    evaluated_at: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# SIMULATION RESULT
# ============================================================

@dataclass(frozen=True)
class ActionSimulationResult:
    action_id: str

    success: bool

    simulated_duration_seconds: float

    output: dict[str, Any] = field(
        default_factory=dict
    )

    error: str | None = None

    completed_at: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# AUDIT LOGGER
# ============================================================

class ActionSchedulerAuditLogger:

    def __init__(
        self,
        maximum_events: int = 20_000,
    ) -> None:
        self._events: deque[dict[str, Any]] = deque(
            maxlen=maximum_events
        )

        self._lock = threading.RLock()

    def record(
        self,
        *,
        tenant_id: str,
        actor: str,
        event_type: str,
        action_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        event = {
            "event_id": new_id(),
            "timestamp": utc_iso(),
            "tenant_id": tenant_id,
            "actor": actor,
            "event_type": event_type,
            "action_id": action_id,
            "details": json_safe(
                details or {}
            ),
        }

        with self._lock:
            self._events.append(event)

    def history(
        self,
        tenant_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            events = [
                dict(event)
                for event in self._events
                if event["tenant_id"] == tenant_id
            ]

        return events[-limit:]


# ============================================================
# RESOURCE MANAGER
# ============================================================

class SchedulerResourceManager:
    """
    Resources are reserved only for active simulation
    tasks and released after completion or cancellation.
    """

    def __init__(self) -> None:
        self._owners: dict[
            tuple[str, str],
            str,
        ] = {}

        self._lock = threading.RLock()

    def available(
        self,
        tenant_id: str,
        resources: Sequence[str],
        action_id: str,
    ) -> bool:
        with self._lock:
            return all(
                self._owners.get(
                    (tenant_id, resource)
                ) in (None, action_id)
                for resource in resources
            )

    def acquire(
        self,
        tenant_id: str,
        resources: Sequence[str],
        action_id: str,
    ) -> bool:
        with self._lock:
            if not self.available(
                tenant_id,
                resources,
                action_id,
            ):
                return False

            for resource in resources:
                self._owners[
                    (tenant_id, resource)
                ] = action_id

            return True

    def release(
        self,
        tenant_id: str,
        action_id: str,
    ) -> int:
        with self._lock:
            keys = [
                key
                for key, owner in self._owners.items()
                if (
                    key[0] == tenant_id
                    and owner == action_id
                )
            ]

            for key in keys:
                del self._owners[key]

            return len(keys)

    def snapshot(
        self,
        tenant_id: str,
    ) -> dict[str, str]:
        with self._lock:
            return {
                resource: owner
                for (owner_tenant, resource), owner
                in self._owners.items()
                if owner_tenant == tenant_id
            }


# ============================================================
# SAFETY GATE
# ============================================================

class SchedulerSafetyGate:

    MOTION_ACTIONS = frozenset({
        ActionType.NAVIGATION,
        ActionType.MANIPULATION,
        ActionType.PICK,
        ActionType.PLACE,
    })

    def __init__(
        self,
        config: ActionSchedulerConfig,
    ) -> None:
        self.config = config

    def requires_approval(
        self,
        action: ScheduledAction,
    ) -> bool:
        if action.requires_human_approval:
            return True

        if (
            self.config.require_human_approval_for_critical
            and action.priority.value
            >= ActionPriority.CRITICAL.value
        ):
            return True

        if (
            self.config.require_human_approval_for_motion
            and action.action_type in self.MOTION_ACTIONS
        ):
            return True

        return False

    def evaluate(
        self,
        action: ScheduledAction,
        robot: RobotSchedulerState | None,
        *,
        now: datetime,
    ) -> tuple[bool, tuple[str, ...]]:
        reasons = []

        if robot is None:
            return False, (
                "Robot state unavailable",
            )

        if robot.tenant_id != action.tenant_id:
            return False, (
                "Robot tenant mismatch",
            )

        age_seconds = (
            now - robot.last_updated
        ).total_seconds()

        if (
            age_seconds < -1.0
            or age_seconds
            > self.config.robot_state_max_age_seconds
        ):
            reasons.append(
                "Robot state is stale or has a future timestamp"
            )

        if robot.emergency_stop_active:
            reasons.append(
                "Emergency stop is active"
            )

        if robot.safety_state in (
            RobotSafetyState.UNSAFE,
            RobotSafetyState.EMERGENCY_STOP,
            RobotSafetyState.UNKNOWN,
        ):
            reasons.append(
                f"Robot safety state: {robot.safety_state.value}"
            )

        if robot.safety_state == RobotSafetyState.CAUTION:
            if (
                action.action_type in self.MOTION_ACTIONS
                or not self.config.allow_caution_state_for_nonmotion
            ):
                reasons.append(
                    "Caution state does not permit this action"
                )

        if robot.active_faults:
            reasons.append(
                "Robot reports active faults"
            )

        if robot.operational_state in (
            RobotOperationalState.OFFLINE,
            RobotOperationalState.FAULTED,
            RobotOperationalState.MAINTENANCE,
        ):
            reasons.append(
                "Robot is not operationally available"
            )

        return (
            len(reasons) == 0,
            tuple(reasons),
        )


# ============================================================
# DEPENDENCY MANAGER
# ============================================================

class ActionDependencyManager:

    @staticmethod
    def validate_new_action(
        action: ScheduledAction,
        actions: Mapping[str, ActionRuntime],
    ) -> None:
        for dependency_id in action.dependencies:
            dependency = actions.get(
                dependency_id
            )

            if dependency is None:
                raise ActionDependencyError(
                    f"Missing dependency: {dependency_id}"
                )

            if dependency.action.tenant_id != action.tenant_id:
                raise ActionDependencyError(
                    "Cross-tenant dependencies are forbidden"
                )

        graph = {
            action_id: set(runtime.action.dependencies)
            for action_id, runtime in actions.items()
        }

        graph[action.action_id] = set(
            action.dependencies
        )

        visiting = set()
        visited = set()

        def visit(node: str) -> None:
            if node in visiting:
                raise ActionDependencyError(
                    "Circular action dependency detected"
                )

            if node in visited:
                return

            visiting.add(node)

            for dependency in graph.get(node, set()):
                visit(dependency)

            visiting.remove(node)
            visited.add(node)

        for node in graph:
            visit(node)

    @staticmethod
    def satisfied(
        action: ScheduledAction,
        actions: Mapping[str, ActionRuntime],
    ) -> bool:
        if not action.dependencies:
            return True

        completed = [
            actions[dependency].status
            == ActionStatus.COMPLETED
            for dependency in action.dependencies
        ]

        if (
            action.dependency_condition
            == DependencyCondition.ALL_COMPLETED
        ):
            return all(completed)

        return any(completed)

    @staticmethod
    def impossible(
        action: ScheduledAction,
        actions: Mapping[str, ActionRuntime],
    ) -> bool:
        if not action.dependencies:
            return False

        terminal_failure = {
            ActionStatus.FAILED,
            ActionStatus.CANCELLED,
            ActionStatus.EXPIRED,
        }

        statuses = [
            actions[dependency].status
            for dependency in action.dependencies
        ]

        if (
            action.dependency_condition
            == DependencyCondition.ALL_COMPLETED
        ):
            return any(
                status in terminal_failure
                for status in statuses
            )

        return all(
            status in terminal_failure
            for status in statuses
        )


# ============================================================
# SIMULATION EXECUTOR
# ============================================================

SimulationHandler = Callable[
    [ScheduledAction],
    ActionSimulationResult,
]


class ActionSimulationExecutor:
    """
    Only executes registered in-process simulation
    handlers. No hardware or network command transport.
    """

    def __init__(self) -> None:
        self._handlers: dict[
            ActionType,
            SimulationHandler,
        ] = {}

        self._lock = threading.RLock()

    def register(
        self,
        action_type: ActionType,
        handler: SimulationHandler,
    ) -> None:
        if not callable(handler):
            raise ActionValidationError(
                "Simulation handler must be callable"
            )

        with self._lock:
            self._handlers[action_type] = handler

    def simulate(
        self,
        action: ScheduledAction,
    ) -> ActionSimulationResult:
        with self._lock:
            handler = self._handlers.get(
                action.action_type
            )

        if handler is None:
            return ActionSimulationResult(
                action_id=action.action_id,
                success=True,
                simulated_duration_seconds=(
                    action.estimated_duration_seconds
                ),
                output={
                    "simulation_only": True,
                    "action_type": action.action_type.value,
                    "message": (
                        "No custom simulation handler; "
                        "action evaluated without hardware execution"
                    ),
                },
            )

        result = handler(action)

        if not isinstance(
            result,
            ActionSimulationResult,
        ):
            raise ActionValidationError(
                "Simulation handler returned invalid result"
            )

        if result.action_id != action.action_id:
            raise ActionValidationError(
                "Simulation result action ID mismatch"
            )

        return result


# ============================================================
# MAIN ACTION SCHEDULER
# ============================================================

class RoboticsActionScheduler:

    TERMINAL_STATUSES = frozenset({
        ActionStatus.COMPLETED,
        ActionStatus.FAILED,
        ActionStatus.CANCELLED,
        ActionStatus.EXPIRED,
    })

    ACTIVE_STATUSES = frozenset({
        ActionStatus.DISPATCHED_SIMULATION,
        ActionStatus.SIMULATING,
    })

    def __init__(
        self,
        config: ActionSchedulerConfig | None = None,
    ) -> None:
        self.config = (
            config
            or ActionSchedulerConfig()
        )

        self._actions: dict[
            str,
            ActionRuntime,
        ] = {}

        self._robots: dict[
            tuple[str, str],
            RobotSchedulerState,
        ] = {}

        self._emergency_locks: set[
            tuple[str, str],
        ] = set()

        self._queue: list[
            tuple[int, float, str],
        ] = []

        self._resources = SchedulerResourceManager()

        self._safety = SchedulerSafetyGate(
            self.config
        )

        self._dependencies = ActionDependencyManager()

        self._simulator = ActionSimulationExecutor()

        self._audit = ActionSchedulerAuditLogger(
            self.config.maximum_audit_events
        )

        self._metrics = Counter()

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # AUTHORIZATION
    # --------------------------------------------------------

    @staticmethod
    def _require_robot_access(
        access: SchedulerAccessContext,
        tenant_id: str,
        robot_id: str,
    ) -> None:
        if access.tenant_id != tenant_id:
            raise ActionAuthorizationError(
                "Cross-tenant access denied"
            )

        if robot_id not in access.allowed_robot_ids:
            raise ActionAuthorizationError(
                "Robot access denied"
            )

    def _get_authorized_runtime(
        self,
        action_id: str,
        access: SchedulerAccessContext,
    ) -> ActionRuntime:
        runtime = self._actions.get(
            action_id
        )

        if runtime is None:
            raise ActionNotFoundError(
                f"Action not found: {action_id}"
            )

        self._require_robot_access(
            access,
            runtime.action.tenant_id,
            runtime.action.robot_id,
        )

        return runtime

    # --------------------------------------------------------
    # STATUS TRANSITIONS
    # --------------------------------------------------------

    @staticmethod
    def _set_status(
        runtime: ActionRuntime,
        status: ActionStatus,
        reason: str = "",
    ) -> None:
        if runtime.status == status:
            return

        runtime.status = status

        runtime.status_history.append({
            "timestamp": utc_iso(),
            "status": status.value,
            "reason": reason,
        })

    # --------------------------------------------------------
    # ROBOT REGISTRATION
    # --------------------------------------------------------

    def register_robot(
        self,
        robot: RobotSchedulerState,
        *,
        access: SchedulerAccessContext,
    ) -> None:
        if not access.can_manage_robots:
            raise ActionAuthorizationError(
                "Robot management permission required"
            )

        self._require_robot_access(
            access,
            robot.tenant_id,
            robot.robot_id,
        )

        with self._lock:
            key = (
                robot.tenant_id,
                robot.robot_id,
            )

            self._robots[key] = robot

            if (
                robot.emergency_stop_active
                or robot.safety_state
                == RobotSafetyState.EMERGENCY_STOP
            ):
                self._emergency_locks.add(
                    key
                )

            self._audit.record(
                tenant_id=robot.tenant_id,
                actor=access.user_id,
                event_type="robot_state_updated",
                details={
                    "robot_id": robot.robot_id,
                    "safety_state": robot.safety_state.value,
                    "operational_state": (
                        robot.operational_state.value
                    ),
                },
            )

    # --------------------------------------------------------
    # EMERGENCY STOP INTERLOCK
    # --------------------------------------------------------

    def engage_emergency_lock(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        access: SchedulerAccessContext,
        reason: str,
    ) -> None:
        self._require_robot_access(
            access,
            tenant_id,
            robot_id,
        )

        with self._lock:
            key = (
                tenant_id,
                robot_id,
            )

            self._emergency_locks.add(
                key
            )

            for runtime in self._actions.values():
                action = runtime.action

                if (
                    action.tenant_id != tenant_id
                    or action.robot_id != robot_id
                    or runtime.status in self.TERMINAL_STATUSES
                ):
                    continue

                self._resources.release(
                    tenant_id,
                    action.action_id,
                )

                self._set_status(
                    runtime,
                    ActionStatus.BLOCKED_SAFETY,
                    reason,
                )

            self._audit.record(
                tenant_id=tenant_id,
                actor=access.user_id,
                event_type="emergency_lock_engaged",
                details={
                    "robot_id": robot_id,
                    "reason": reason,
                },
            )

    def clear_emergency_lock(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        access: SchedulerAccessContext,
    ) -> None:
        """
        Clears only the software scheduling lock.

        Does not reset or override a physical emergency
        stop. Requires a fresh SAFE robot state and
        explicit management authorization.
        """

        if not access.can_manage_robots:
            raise ActionAuthorizationError(
                "Robot management permission required"
            )

        self._require_robot_access(
            access,
            tenant_id,
            robot_id,
        )

        with self._lock:
            key = (
                tenant_id,
                robot_id,
            )

            robot = self._robots.get(
                key
            )

            if robot is None:
                raise ActionSafetyError(
                    "Robot state unavailable"
                )

            if (
                robot.emergency_stop_active
                or robot.safety_state
                != RobotSafetyState.SAFE
            ):
                raise ActionSafetyError(
                    "Robot must independently report SAFE "
                    "with emergency stop inactive"
                )

            age = (
                utc_now() - robot.last_updated
            ).total_seconds()

            if not (
                0 <= age
                <= self.config.robot_state_max_age_seconds
            ):
                raise ActionSafetyError(
                    "Fresh robot safety state required"
                )

            self._emergency_locks.discard(
                key
            )

            self._audit.record(
                tenant_id=tenant_id,
                actor=access.user_id,
                event_type="scheduling_lock_cleared",
                details={
                    "robot_id": robot_id,
                    "physical_estop_reset": False,
                },
            )

    # --------------------------------------------------------
    # SUBMIT ACTION
    # --------------------------------------------------------

    def submit(
        self,
        action: ScheduledAction,
        *,
        access: SchedulerAccessContext,
    ) -> ActionRuntime:
        if not access.can_schedule:
            raise ActionAuthorizationError(
                "Scheduling permission required"
            )

        self._require_robot_access(
            access,
            action.tenant_id,
            action.robot_id,
        )

        if (
            action.timeout_seconds
            > self.config.maximum_action_timeout_seconds
        ):
            raise ActionValidationError(
                "Action timeout exceeds configured limit"
            )

        if (
            action.maximum_retries
            > self.config.maximum_retry_limit
        ):
            raise ActionValidationError(
                "Retry limit exceeds configured maximum"
            )

        with self._lock:
            if action.action_id in self._actions:
                raise ActionConflictError(
                    "Action ID already exists"
                )

            if (
                len(self._actions)
                >= self.config.maximum_actions
            ):
                raise ActionSchedulerError(
                    "Maximum action capacity reached"
                )

            self._dependencies.validate_new_action(
                action,
                self._actions,
            )

            approval_required = (
                self._safety.requires_approval(
                    action
                )
            )

            runtime = ActionRuntime(
                action=action,
                approval_status=(
                    ApprovalStatus.PENDING
                    if approval_required
                    else ApprovalStatus.NOT_REQUIRED
                ),
            )

            self._actions[
                action.action_id
            ] = runtime

            self._set_status(
                runtime,
                ActionStatus.PENDING,
                "Action submitted",
            )

            self._metrics["submitted"] += 1

            self._audit.record(
                tenant_id=action.tenant_id,
                actor=access.user_id,
                event_type="action_submitted",
                action_id=action.action_id,
                details={
                    "robot_id": action.robot_id,
                    "action_type": action.action_type.value,
                    "priority": action.priority.value,
                    "approval_required": approval_required,
                },
            )

            return runtime

    # --------------------------------------------------------
    # HUMAN APPROVAL
    # --------------------------------------------------------

    def approve(
        self,
        action_id: str,
        *,
        access: SchedulerAccessContext,
    ) -> None:
        if not access.can_approve:
            raise ActionAuthorizationError(
                "Approval permission required"
            )

        with self._lock:
            runtime = self._get_authorized_runtime(
                action_id,
                access,
            )

            if runtime.status in self.TERMINAL_STATUSES:
                raise ActionConflictError(
                    "Cannot approve terminal action"
                )

            if (
                runtime.approval_status
                == ApprovalStatus.REJECTED
            ):
                raise ActionConflictError(
                    "Rejected action cannot be approved"
                )

            runtime.approval_status = (
                ApprovalStatus.APPROVED
            )

            runtime.approval_user_id = (
                access.user_id
            )

            runtime.approved_at = utc_now()

            self._audit.record(
                tenant_id=access.tenant_id,
                actor=access.user_id,
                event_type="action_approved",
                action_id=action_id,
                details={
                    "approval_is_execution_authorization": False,
                },
            )

    def reject(
        self,
        action_id: str,
        *,
        access: SchedulerAccessContext,
        reason: str,
    ) -> None:
        if not access.can_approve:
            raise ActionAuthorizationError(
                "Approval permission required"
            )

        with self._lock:
            runtime = self._get_authorized_runtime(
                action_id,
                access,
            )

            if runtime.status in self.TERMINAL_STATUSES:
                raise ActionConflictError(
                    "Cannot reject terminal action"
                )

            runtime.approval_status = (
                ApprovalStatus.REJECTED
            )

            self._resources.release(
                access.tenant_id,
                action_id,
            )

            self._set_status(
                runtime,
                ActionStatus.CANCELLED,
                reason,
            )

            self._audit.record(
                tenant_id=access.tenant_id,
                actor=access.user_id,
                event_type="action_rejected",
                action_id=action_id,
                details={
                    "reason": reason,
                },
            )

    # --------------------------------------------------------
    # EVALUATE ACTION
    # --------------------------------------------------------

    def evaluate(
        self,
        action_id: str,
        *,
        access: SchedulerAccessContext,
        now: datetime | None = None,
    ) -> SchedulingDecision:
        current = ensure_utc(
            now or utc_now()
        )

        with self._lock:
            runtime = self._get_authorized_runtime(
                action_id,
                access,
            )

            return self._evaluate_locked(
                runtime,
                current,
            )

    def _evaluate_locked(
        self,
        runtime: ActionRuntime,
        now: datetime,
    ) -> SchedulingDecision:
        action = runtime.action
        reasons = []

        if runtime.status in self.TERMINAL_STATUSES:
            return SchedulingDecision(
                action_id=action.action_id,
                robot_id=action.robot_id,
                allowed=False,
                status=runtime.status,
                reasons=("Action is terminal",),
            )

        if runtime.status in self.ACTIVE_STATUSES:
            return SchedulingDecision(
                action_id=action.action_id,
                robot_id=action.robot_id,
                allowed=False,
                status=runtime.status,
                reasons=("Action is already active",),
            )

        if runtime.status == ActionStatus.SUSPENDED:
            return SchedulingDecision(
                action_id=action.action_id,
                robot_id=action.robot_id,
                allowed=False,
                status=runtime.status,
                reasons=("Action is suspended",),
            )

        if (
            action.deadline is not None
            and now >= action.deadline
        ):
            return SchedulingDecision(
                action_id=action.action_id,
                robot_id=action.robot_id,
                allowed=False,
                status=ActionStatus.EXPIRED,
                reasons=("Action deadline exceeded",),
            )

        if (
            action.earliest_start is not None
            and now < action.earliest_start
        ):
            return SchedulingDecision(
                action_id=action.action_id,
                robot_id=action.robot_id,
                allowed=False,
                status=ActionStatus.SCHEDULED,
                reasons=("Earliest start not reached",),
            )

        if (
            runtime.next_retry_at is not None
            and now < runtime.next_retry_at
        ):
            return SchedulingDecision(
                action_id=action.action_id,
                robot_id=action.robot_id,
                allowed=False,
                status=ActionStatus.RETRY_WAIT,
                reasons=("Retry delay not elapsed",),
            )

        if self._dependencies.impossible(
            action,
            self._actions,
        ):
            return SchedulingDecision(
                action_id=action.action_id,
                robot_id=action.robot_id,
                allowed=False,
                status=ActionStatus.FAILED,
                reasons=("Required dependency failed",),
            )

        if not self._dependencies.satisfied(
            action,
            self._actions,
        ):
            return SchedulingDecision(
                action_id=action.action_id,
                robot_id=action.robot_id,
                allowed=False,
                status=ActionStatus.WAITING_DEPENDENCY,
                reasons=("Dependencies incomplete",),
            )

        if (
            runtime.approval_status
            == ApprovalStatus.PENDING
        ):
            return SchedulingDecision(
                action_id=action.action_id,
                robot_id=action.robot_id,
                allowed=False,
                status=ActionStatus.WAITING_APPROVAL,
                reasons=("Human approval required",),
            )

        if (
            runtime.approval_status
            == ApprovalStatus.REJECTED
        ):
            return SchedulingDecision(
                action_id=action.action_id,
                robot_id=action.robot_id,
                allowed=False,
                status=ActionStatus.CANCELLED,
                reasons=("Action approval rejected",),
            )

        robot_key = (
            action.tenant_id,
            action.robot_id,
        )

        if robot_key in self._emergency_locks:
            return SchedulingDecision(
                action_id=action.action_id,
                robot_id=action.robot_id,
                allowed=False,
                status=ActionStatus.BLOCKED_SAFETY,
                reasons=("Emergency scheduling lock active",),
            )

        robot = self._robots.get(
            robot_key
        )

        safe, safety_reasons = (
            self._safety.evaluate(
                action,
                robot,
                now=now,
            )
        )

        if not safe:
            return SchedulingDecision(
                action_id=action.action_id,
                robot_id=action.robot_id,
                allowed=False,
                status=ActionStatus.BLOCKED_SAFETY,
                reasons=safety_reasons,
            )

        assert robot is not None

        if not action.required_capabilities.issubset(
            robot.capabilities
        ):
            reasons.append(
                "Robot lacks required capabilities"
            )

        if not action.required_resources.issubset(
            robot.available_resources
        ):
            reasons.append(
                "Required resources are not available "
                "on the robot"
            )

        if reasons:
            return SchedulingDecision(
                action_id=action.action_id,
                robot_id=action.robot_id,
                allowed=False,
                status=ActionStatus.WAITING_ROBOT,
                reasons=tuple(reasons),
            )

        if not self._resources.available(
            action.tenant_id,
            action.required_resources,
            action.action_id,
        ):
            return SchedulingDecision(
                action_id=action.action_id,
                robot_id=action.robot_id,
                allowed=False,
                status=ActionStatus.WAITING_RESOURCE,
                reasons=("Resource reservation conflict",),
            )

        active_global = sum(
            runtime.status in self.ACTIVE_STATUSES
            for runtime in self._actions.values()
        )

        active_robot = sum(
            (
                candidate.status in self.ACTIVE_STATUSES
                and candidate.action.tenant_id
                == action.tenant_id
                and candidate.action.robot_id
                == action.robot_id
            )
            for candidate in self._actions.values()
        )

        if (
            active_global
            >= self.config.maximum_concurrent_global
            or active_robot
            >= self.config.maximum_concurrent_per_robot
        ):
            return SchedulingDecision(
                action_id=action.action_id,
                robot_id=action.robot_id,
                allowed=False,
                status=ActionStatus.WAITING_ROBOT,
                reasons=("Concurrency limit reached",),
            )

        return SchedulingDecision(
            action_id=action.action_id,
            robot_id=action.robot_id,
            allowed=True,
            status=ActionStatus.READY,
            reasons=(),
        )

    # --------------------------------------------------------
    # REFRESH QUEUE
    # --------------------------------------------------------

    def refresh_queue(
        self,
        *,
        now: datetime | None = None,
    ) -> int:
        current = ensure_utc(
            now or utc_now()
        )

        with self._lock:
            self._queue.clear()

            for runtime in self._actions.values():
                if runtime.status in self.TERMINAL_STATUSES:
                    continue

                if runtime.status in self.ACTIVE_STATUSES:
                    continue

                if runtime.status == ActionStatus.SUSPENDED:
                    continue

                decision = self._evaluate_locked(
                    runtime,
                    current,
                )

                self._set_status(
                    runtime,
                    decision.status,
                    "; ".join(decision.reasons),
                )

                if decision.allowed:
                    action = runtime.action

                    heapq.heappush(
                        self._queue,
                        (
                            -action.priority.value,
                            action.created_at.timestamp(),
                            action.action_id,
                        ),
                    )

            if (
                len(self._queue)
                > self.config.maximum_queue_size
            ):
                raise ActionSchedulerError(
                    "Scheduler queue capacity exceeded"
                )

            return len(self._queue)

    # --------------------------------------------------------
    # NEXT ACTION
    # --------------------------------------------------------

    def next_action(
        self,
        *,
        access: SchedulerAccessContext,
        now: datetime | None = None,
    ) -> ScheduledAction | None:
        if not access.can_schedule:
            raise ActionAuthorizationError(
                "Scheduling permission required"
            )

        current = ensure_utc(
            now or utc_now()
        )

        with self._lock:
            self.refresh_queue(
                now=current
            )

            for _, _, action_id in sorted(self._queue):
                runtime = self._actions[action_id]

                action = runtime.action

                if (
                    action.tenant_id == access.tenant_id
                    and action.robot_id
                    in access.allowed_robot_ids
                ):
                    return action

            return None

    # --------------------------------------------------------
    # SIMULATION DISPATCH
    # --------------------------------------------------------

    def dispatch_simulation(
        self,
        action_id: str,
        *,
        access: SchedulerAccessContext,
        now: datetime | None = None,
    ) -> ActionRuntime:
        if not access.can_schedule:
            raise ActionAuthorizationError(
                "Scheduling permission required"
            )

        current = ensure_utc(
            now or utc_now()
        )

        with self._lock:
            runtime = self._get_authorized_runtime(
                action_id,
                access,
            )

            decision = self._evaluate_locked(
                runtime,
                current,
            )

            if not decision.allowed:
                self._set_status(
                    runtime,
                    decision.status,
                    "; ".join(decision.reasons),
                )

                raise ActionSafetyError(
                    "Action not dispatchable: "
                    + "; ".join(decision.reasons)
                )

            action = runtime.action

            acquired = self._resources.acquire(
                action.tenant_id,
                action.required_resources,
                action.action_id,
            )

            if not acquired:
                raise ActionConflictError(
                    "Unable to reserve action resources"
                )

            runtime.started_at = current
            runtime.attempts += 1

            self._set_status(
                runtime,
                ActionStatus.SIMULATING,
                "Simulation started",
            )

            self._audit.record(
                tenant_id=action.tenant_id,
                actor=access.user_id,
                event_type="simulation_dispatched",
                action_id=action.action_id,
                details={
                    "attempt": runtime.attempts,
                    "hardware_execution": False,
                },
            )

        # Execute simulation handler outside scheduler lock.
        try:
            if self.config.mode == SchedulerMode.DRY_RUN:
                result = ActionSimulationResult(
                    action_id=action.action_id,
                    success=True,
                    simulated_duration_seconds=0.0,
                    output={
                        "dry_run": True,
                        "hardware_execution": False,
                    },
                )

            else:
                result = self._simulator.simulate(
                    action
                )

        except Exception as exc:
            result = ActionSimulationResult(
                action_id=action.action_id,
                success=False,
                simulated_duration_seconds=0.0,
                error=str(exc),
            )

        with self._lock:
            # A concurrent emergency lock or cancellation
            # must never be overwritten by a late result.
            if runtime.status != ActionStatus.SIMULATING:
                self._resources.release(
                    action.tenant_id,
                    action.action_id,
                )

                return runtime

            self._resources.release(
                action.tenant_id,
                action.action_id,
            )

            elapsed = (
                utc_now() - current
            ).total_seconds()

            timed_out = (
                elapsed > action.timeout_seconds
                or result.simulated_duration_seconds
                > action.timeout_seconds
            )

            if timed_out:
                result = ActionSimulationResult(
                    action_id=action.action_id,
                    success=False,
                    simulated_duration_seconds=(
                        result.simulated_duration_seconds
                    ),
                    error="Simulation timeout exceeded",
                )

            runtime.simulation_result = json_safe(
                result
            )

            if result.success:
                runtime.completed_at = utc_now()

                self._set_status(
                    runtime,
                    ActionStatus.COMPLETED,
                    "Simulation completed",
                )

                self._metrics["completed"] += 1

            else:
                runtime.last_error = (
                    result.error
                    or "Simulation failed"
                )

                if runtime.attempts <= action.maximum_retries:
                    delay = min(
                        self.config.retry_base_delay_seconds
                        * (
                            2 ** (
                                runtime.attempts - 1
                            )
                        ),
                        self.config.retry_max_delay_seconds,
                    )

                    runtime.next_retry_at = (
                        utc_now()
                        + timedelta(seconds=delay)
                    )

                    self._set_status(
                        runtime,
                        ActionStatus.RETRY_WAIT,
                        runtime.last_error,
                    )

                else:
                    runtime.completed_at = utc_now()

                    self._set_status(
                        runtime,
                        ActionStatus.FAILED,
                        runtime.last_error,
                    )

                    self._metrics["failed"] += 1

            self._audit.record(
                tenant_id=action.tenant_id,
                actor=access.user_id,
                event_type="simulation_finished",
                action_id=action.action_id,
                details={
                    "status": runtime.status.value,
                    "success": result.success,
                    "hardware_execution": False,
                },
            )

            return runtime

    # --------------------------------------------------------
    # REGISTER SIMULATION HANDLER
    # --------------------------------------------------------

    def register_simulation_handler(
        self,
        action_type: ActionType,
        handler: SimulationHandler,
    ) -> None:
        self._simulator.register(
            action_type,
            handler,
        )

    # --------------------------------------------------------
    # CANCEL
    # --------------------------------------------------------

    def cancel(
        self,
        action_id: str,
        *,
        access: SchedulerAccessContext,
        reason: str = "Cancelled by user",
    ) -> None:
        if not access.can_cancel:
            raise ActionAuthorizationError(
                "Cancellation permission required"
            )

        with self._lock:
            runtime = self._get_authorized_runtime(
                action_id,
                access,
            )

            if runtime.status in self.TERMINAL_STATUSES:
                return

            self._resources.release(
                runtime.action.tenant_id,
                action_id,
            )

            runtime.completed_at = utc_now()

            self._set_status(
                runtime,
                ActionStatus.CANCELLED,
                reason,
            )

            self._metrics["cancelled"] += 1

            self._audit.record(
                tenant_id=access.tenant_id,
                actor=access.user_id,
                event_type="action_cancelled",
                action_id=action_id,
                details={
                    "reason": reason,
                },
            )

    # --------------------------------------------------------
    # SUSPEND / RESUME
    # --------------------------------------------------------

    def suspend(
        self,
        action_id: str,
        *,
        access: SchedulerAccessContext,
    ) -> None:
        if not access.can_cancel:
            raise ActionAuthorizationError(
                "Action management permission required"
            )

        with self._lock:
            runtime = self._get_authorized_runtime(
                action_id,
                access,
            )

            if runtime.status in self.TERMINAL_STATUSES:
                raise ActionConflictError(
                    "Cannot suspend terminal action"
                )

            self._resources.release(
                runtime.action.tenant_id,
                action_id,
            )

            self._set_status(
                runtime,
                ActionStatus.SUSPENDED,
                "Suspended by user",
            )

    def resume(
        self,
        action_id: str,
        *,
        access: SchedulerAccessContext,
    ) -> None:
        if not access.can_schedule:
            raise ActionAuthorizationError(
                "Scheduling permission required"
            )

        with self._lock:
            runtime = self._get_authorized_runtime(
                action_id,
                access,
            )

            if runtime.status != ActionStatus.SUSPENDED:
                raise ActionConflictError(
                    "Action is not suspended"
                )

            self._set_status(
                runtime,
                ActionStatus.PENDING,
                "Action resumed; safety revalidation required",
            )

    # --------------------------------------------------------
    # GET ACTION
    # --------------------------------------------------------

    def get_action(
        self,
        action_id: str,
        *,
        access: SchedulerAccessContext,
    ) -> dict[str, Any]:
        with self._lock:
            runtime = self._get_authorized_runtime(
                action_id,
                access,
            )

            return runtime.to_dict()

    # --------------------------------------------------------
    # LIST ACTIONS
    # --------------------------------------------------------

    def list_actions(
        self,
        *,
        access: SchedulerAccessContext,
        status: ActionStatus | None = None,
        robot_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if (
            robot_id is not None
            and robot_id not in access.allowed_robot_ids
        ):
            raise ActionAuthorizationError(
                "Robot access denied"
            )

        with self._lock:
            results = []

            for runtime in self._actions.values():
                action = runtime.action

                if action.tenant_id != access.tenant_id:
                    continue

                if action.robot_id not in access.allowed_robot_ids:
                    continue

                if (
                    robot_id is not None
                    and action.robot_id != robot_id
                ):
                    continue

                if status is not None and runtime.status != status:
                    continue

                results.append(
                    runtime.to_dict()
                )

            return results

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        access: SchedulerAccessContext,
    ) -> dict[str, Any]:
        with self._lock:
            runtimes = [
                runtime
                for runtime in self._actions.values()
                if (
                    runtime.action.tenant_id
                    == access.tenant_id
                    and runtime.action.robot_id
                    in access.allowed_robot_ids
                )
            ]

            return {
                "tenant_id": access.tenant_id,
                "total_actions": len(runtimes),
                "statuses": dict(
                    Counter(
                        runtime.status.value
                        for runtime in runtimes
                    )
                ),
                "priorities": dict(
                    Counter(
                        runtime.action.priority.name
                        for runtime in runtimes
                    )
                ),
                "action_types": dict(
                    Counter(
                        runtime.action.action_type.value
                        for runtime in runtimes
                    )
                ),
                "registered_robots": sum(
                    tenant_id == access.tenant_id
                    and robot_id in access.allowed_robot_ids
                    for tenant_id, robot_id in self._robots
                ),
                "mode": self.config.mode.value,
                "hardware_execution": False,
                "timestamp": utc_iso(),
            }

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    def audit_history(
        self,
        *,
        access: SchedulerAccessContext,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ActionValidationError(
                "Limit must be positive"
            )

        return self._audit.history(
            access.tenant_id,
            limit=limit,
        )

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        *,
        access: SchedulerAccessContext,
    ) -> str:
        payload = {
            "platform": (
                "Physical AI & Autonomous Robotics "
                "Intelligence Platform"
            ),
            "module": "action_scheduler",
            "mode": self.config.mode.value,
            "tenant_id": access.tenant_id,
            "actions": self.list_actions(
                access=access
            ),
            "statistics": self.statistics(
                access=access
            ),
            "hardware_execution": False,
            "exported_at": utc_iso(),
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

class TestRoboticsActionScheduler(unittest.TestCase):

    def setUp(self) -> None:
        self.scheduler = RoboticsActionScheduler(
            ActionSchedulerConfig(
                robot_state_max_age_seconds=60.0,
                retry_base_delay_seconds=0.01,
            )
        )

        self.access = SchedulerAccessContext(
            tenant_id="tenant_alpha",
            user_id="engineer_001",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
            }),
            roles=frozenset({
                "engineer",
            }),
            can_schedule=True,
            can_cancel=True,
            can_approve=True,
            can_manage_robots=True,
        )

        self.scheduler.register_robot(
            RobotSchedulerState(
                robot_id="ROBOT-001",
                tenant_id="tenant_alpha",
                operational_state=(
                    RobotOperationalState.AVAILABLE
                ),
                safety_state=RobotSafetyState.SAFE,
                capabilities=frozenset({
                    "inspection",
                    "diagnostics",
                    "navigation",
                }),
                available_resources=frozenset({
                    "camera",
                    "diagnostic_bus",
                }),
                battery_percentage=85.0,
            ),
            access=self.access,
        )

    def make_action(
        self,
        action_id: str,
        *,
        action_type: ActionType = ActionType.INSPECTION,
        dependencies: tuple[str, ...] = (),
        resources: frozenset[str] = frozenset(),
    ) -> ScheduledAction:
        return ScheduledAction(
            action_id=action_id,
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            action_type=action_type,
            name=action_id,
            dependencies=dependencies,
            required_resources=resources,
            timeout_seconds=30.0,
        )

    def test_submit(self) -> None:
        runtime = self.scheduler.submit(
            self.make_action("action_001"),
            access=self.access,
        )

        self.assertEqual(
            runtime.action.action_id,
            "action_001",
        )

    def test_ready_action(self) -> None:
        self.scheduler.submit(
            self.make_action("action_002"),
            access=self.access,
        )

        decision = self.scheduler.evaluate(
            "action_002",
            access=self.access,
        )

        self.assertTrue(
            decision.allowed
        )

    def test_simulation_completion(self) -> None:
        self.scheduler.submit(
            self.make_action("action_003"),
            access=self.access,
        )

        runtime = self.scheduler.dispatch_simulation(
            "action_003",
            access=self.access,
        )

        self.assertEqual(
            runtime.status,
            ActionStatus.COMPLETED,
        )

    def test_dependency(self) -> None:
        self.scheduler.submit(
            self.make_action("parent"),
            access=self.access,
        )

        self.scheduler.submit(
            self.make_action(
                "child",
                dependencies=("parent",),
            ),
            access=self.access,
        )

        decision = self.scheduler.evaluate(
            "child",
            access=self.access,
        )

        self.assertEqual(
            decision.status,
            ActionStatus.WAITING_DEPENDENCY,
        )

        self.scheduler.dispatch_simulation(
            "parent",
            access=self.access,
        )

        decision = self.scheduler.evaluate(
            "child",
            access=self.access,
        )

        self.assertTrue(
            decision.allowed
        )

    def test_motion_approval(self) -> None:
        self.scheduler.submit(
            self.make_action(
                "navigate_001",
                action_type=ActionType.NAVIGATION,
            ),
            access=self.access,
        )

        decision = self.scheduler.evaluate(
            "navigate_001",
            access=self.access,
        )

        self.assertEqual(
            decision.status,
            ActionStatus.WAITING_APPROVAL,
        )

        self.scheduler.approve(
            "navigate_001",
            access=self.access,
        )

        decision = self.scheduler.evaluate(
            "navigate_001",
            access=self.access,
        )

        self.assertTrue(
            decision.allowed
        )

    def test_emergency_lock(self) -> None:
        self.scheduler.submit(
            self.make_action("action_004"),
            access=self.access,
        )

        self.scheduler.engage_emergency_lock(
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            access=self.access,
            reason="Emergency stop",
        )

        decision = self.scheduler.evaluate(
            "action_004",
            access=self.access,
        )

        self.assertFalse(
            decision.allowed
        )

    def test_cancel(self) -> None:
        self.scheduler.submit(
            self.make_action("action_005"),
            access=self.access,
        )

        self.scheduler.cancel(
            "action_005",
            access=self.access,
        )

        runtime = self.scheduler.get_action(
            "action_005",
            access=self.access,
        )

        self.assertEqual(
            runtime["status"],
            "cancelled",
        )

    def test_tenant_isolation(self) -> None:
        self.scheduler.submit(
            self.make_action("action_006"),
            access=self.access,
        )

        unauthorized = SchedulerAccessContext(
            tenant_id="tenant_beta",
            user_id="other_user",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
            }),
            can_schedule=True,
        )

        with self.assertRaises(
            ActionAuthorizationError
        ):
            self.scheduler.get_action(
                "action_006",
                access=unauthorized,
            )

    def test_priority_order(self) -> None:
        self.scheduler.submit(
            replace(
                self.make_action("low_action"),
                priority=ActionPriority.LOW,
            ),
            access=self.access,
        )

        self.scheduler.submit(
            replace(
                self.make_action("high_action"),
                priority=ActionPriority.HIGH,
            ),
            access=self.access,
        )

        next_action = self.scheduler.next_action(
            access=self.access
        )

        self.assertIsNotNone(
            next_action
        )

        self.assertEqual(
            next_action.action_id,
            "high_action",
        )

    def test_json_export(self) -> None:
        self.scheduler.submit(
            self.make_action("action_007"),
            access=self.access,
        )

        exported = self.scheduler.export_json(
            access=self.access
        )

        self.assertIn(
            "action_007",
            exported,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    scheduler = RoboticsActionScheduler(
        ActionSchedulerConfig(
            mode=SchedulerMode.SIMULATION,
            maximum_concurrent_per_robot=1,
            require_human_approval_for_motion=True,
            require_human_approval_for_critical=True,
            robot_state_max_age_seconds=60.0,
        )
    )

    access = SchedulerAccessContext(
        tenant_id="tenant_alpha",
        user_id="robotics_engineer",
        roles=frozenset({
            "robotics_engineer",
        }),
        allowed_robot_ids=frozenset({
            "ROBOT-001",
        }),
        can_schedule=True,
        can_cancel=True,
        can_approve=True,
        can_manage_robots=True,
    )

    scheduler.register_robot(
        RobotSchedulerState(
            robot_id="ROBOT-001",
            tenant_id="tenant_alpha",
            operational_state=(
                RobotOperationalState.AVAILABLE
            ),
            safety_state=RobotSafetyState.SAFE,
            capabilities=frozenset({
                "inspection",
                "diagnostics",
                "camera",
                "navigation",
            }),
            available_resources=frozenset({
                "camera",
                "diagnostic_bus",
            }),
            battery_percentage=88.0,
        ),
        access=access,
    )

    actions = [
        ScheduledAction(
            action_id="inspect_motor",
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            action_type=ActionType.INSPECTION,
            name="Inspect Robot Motor",
            priority=ActionPriority.HIGH,
            required_capabilities=frozenset({
                "inspection",
            }),
            required_resources=frozenset({
                "camera",
            }),
            estimated_duration_seconds=8.0,
        ),
        ScheduledAction(
            action_id="diagnose_motor",
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            action_type=ActionType.DIAGNOSTICS,
            name="Diagnose Motor Condition",
            priority=ActionPriority.NORMAL,
            dependencies=("inspect_motor",),
            required_capabilities=frozenset({
                "diagnostics",
            }),
            required_resources=frozenset({
                "diagnostic_bus",
            }),
            estimated_duration_seconds=12.0,
        ),
        ScheduledAction(
            action_id="navigate_to_station",
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            action_type=ActionType.NAVIGATION,
            name="Simulate Navigation to Service Station",
            priority=ActionPriority.HIGH,
            dependencies=("diagnose_motor",),
            required_capabilities=frozenset({
                "navigation",
            }),
            requires_human_approval=True,
            estimated_duration_seconds=15.0,
        ),
    ]

    for action in actions:
        scheduler.submit(
            action,
            access=access,
        )

    def inspection_simulation(
        action: ScheduledAction,
    ) -> ActionSimulationResult:
        return ActionSimulationResult(
            action_id=action.action_id,
            success=True,
            simulated_duration_seconds=8.0,
            output={
                "inspection_result": "completed",
                "motor_temperature_c": 68.5,
                "vibration_mm_s": 2.1,
                "simulated": True,
            },
        )

    scheduler.register_simulation_handler(
        ActionType.INSPECTION,
        inspection_simulation,
    )

    scheduler.approve(
        "navigate_to_station",
        access=access,
    )

    for action_id in (
        "inspect_motor",
        "diagnose_motor",
        "navigate_to_station",
    ):
        decision = scheduler.evaluate(
            action_id,
            access=access,
        )

        print(
            f"\nAction: {action_id}"
        )

        print(
            f"Allowed: {decision.allowed}"
        )

        print(
            f"Status: {decision.status.value}"
        )

        if decision.allowed:
            result = scheduler.dispatch_simulation(
                action_id,
                access=access,
            )

            print(
                f"Simulation result: {result.status.value}"
            )

    print("\nSCHEDULER STATISTICS")

    print(
        json.dumps(
            scheduler.statistics(
                access=access
            ),
            indent=2,
        )
    )

    print("\nSCHEDULER EXPORT")

    print(
        scheduler.export_json(
            access=access
        )
    )

    print("\nUNIT TESTS")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestRoboticsActionScheduler
    )

    unittest.TextTestRunner(
        verbosity=2
    ).run(suite)


if __name__ == "__main__":
    main()
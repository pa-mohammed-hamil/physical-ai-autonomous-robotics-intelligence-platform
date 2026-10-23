"""Behavior Tree module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/planning/behavior_tree.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================

Module: behavior_tree.py

Production-oriented behavior tree orchestration for robotics
planning, inspection, diagnostics, maintenance, and simulation.

Python: 3.11+
Dependencies: Python standard library

Features
--------
- Behavior tree lifecycle management
- Integration with behavior_tree_nodes.py
- Sequence, selector, reactive, and parallel trees
- Safety-gated execution
- Tenant and robot isolation
- Tree registration and validation
- Robot state management
- Human approval management
- Simulation-only execution
- Tick-based orchestration
- Background simulation workers
- Tree pause, resume, cancel, and reset
- Execution limits and timeouts
- Tree snapshots and status reporting
- Blackboard inspection
- Execution history and auditing
- Tree visualization
- JSON export
- Thread-safe registry
- Unit tests and demonstration

Safety
------
This module NEVER issues real robot actuator commands.

All action execution is simulation-only. Physical execution
requires an independent robot controller, safety validation,
hardware interlocks, and trusted authorization.

Do not expose simulation callbacks as unrestricted user-supplied
code or interpret behavior tree definitions with eval/exec.
"""

from __future__ import annotations

import copy
import json
import math
import threading
import time
import unittest
import uuid

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

try:
    from .behavior_tree_nodes import (
        ActionNode,
        BehaviorTreeAccessContext,
        BehaviorTreeConfig,
        BehaviorTreeNode,
        BehaviorTreeTickResult,
        Blackboard,
        NodeStatus,
        RobotBehaviorState,
        RoboticsBehaviorTreeEngine,
    )
except ImportError:
    try:
        from behavior_tree_nodes import (
            ActionNode,
            BehaviorTreeAccessContext,
            BehaviorTreeConfig,
            BehaviorTreeNode,
            BehaviorTreeTickResult,
            Blackboard,
            NodeStatus,
            RobotBehaviorState,
            RoboticsBehaviorTreeEngine,
        )
    except ImportError as exc:
        raise ImportError(
            "behavior_tree.py requires behavior_tree_nodes.py "
            "in the same package or Python import path."
        ) from exc


# ============================================================
# EXCEPTIONS
# ============================================================

class BehaviorTreeManagerError(Exception):
    """Base exception for behavior tree management."""


class TreeValidationError(BehaviorTreeManagerError):
    """Invalid tree configuration or operation."""


class TreeNotFoundError(BehaviorTreeManagerError):
    """Requested behavior tree does not exist."""


class TreeAlreadyExistsError(BehaviorTreeManagerError):
    """Tree identifier is already registered."""


class TreePermissionError(BehaviorTreeManagerError):
    """Unauthorized tree operation."""


class TreeStateError(BehaviorTreeManagerError):
    """Invalid tree lifecycle transition."""


class TreeSafetyError(BehaviorTreeManagerError):
    """Robot safety validation failed."""


class TreeExecutionError(BehaviorTreeManagerError):
    """Behavior tree execution failed."""


class TreeCapacityError(BehaviorTreeManagerError):
    """Tree registry capacity has been reached."""


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


def generate_id() -> str:
    return uuid.uuid4().hex


def validate_identifier(
    value: str,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise TreeValidationError(
            f"{field_name} must be a string"
        )

    value = value.strip()

    if not value:
        raise TreeValidationError(
            f"{field_name} cannot be empty"
        )

    if len(value) > 128:
        raise TreeValidationError(
            f"{field_name} exceeds 128 characters"
        )

    return value


def validate_positive_number(
    value: int | float,
    field_name: str,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise TreeValidationError(
            f"{field_name} must be positive and finite"
        )


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

    if isinstance(value, (list, tuple, set, frozenset, deque)):
        return [
            json_safe(item)
            for item in value
        ]

    if hasattr(value, "__dataclass_fields__"):
        return {
            name: json_safe(getattr(value, name))
            for name in value.__dataclass_fields__
        }

    if isinstance(value, (str, int, bool)) or value is None:
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    return repr(value)


# ============================================================
# ENUMS
# ============================================================

class TreeLifecycleStatus(str, Enum):
    REGISTERED = "registered"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED_SAFETY = "blocked_safety"
    TIMEOUT = "timeout"


class TreeExecutionMode(str, Enum):
    SIMULATION = "simulation"
    DRY_RUN = "dry_run"


class TreeEventType(str, Enum):
    REGISTERED = "registered"
    VALIDATED = "validated"
    STARTED = "started"
    TICKED = "ticked"
    PAUSED = "paused"
    RESUMED = "resumed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RESET = "reset"
    TIMEOUT = "timeout"
    SAFETY_BLOCKED = "safety_blocked"
    ROBOT_STATE_UPDATED = "robot_state_updated"
    APPROVAL_ADDED = "approval_added"
    APPROVAL_REVOKED = "approval_revoked"
    DELETED = "deleted"


class TreeSafetyDecision(str, Enum):
    ALLOWED = "allowed"
    BLOCKED = "blocked"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class BehaviorTreeManagerConfig:
    maximum_registered_trees: int = 1000

    maximum_concurrent_workers: int = 16

    maximum_ticks_per_tree: int = 100_000

    maximum_run_ticks: int = 1000

    default_tick_interval_seconds: float = 0.1

    maximum_tick_interval_seconds: float = 10.0

    default_execution_timeout_seconds: float = 300.0

    maximum_execution_timeout_seconds: float = 3600.0

    robot_state_max_age_seconds: float = 10.0

    require_safe_robot_state: bool = True

    require_no_active_faults: bool = True

    require_emergency_stop_inactive: bool = True

    minimum_battery_percentage: float = 5.0

    maximum_events_per_tree: int = 5000

    allow_background_simulation: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_registered_trees",
            "maximum_concurrent_workers",
            "maximum_ticks_per_tree",
            "maximum_run_ticks",
            "maximum_events_per_tree",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise TreeValidationError(
                    f"{name} must be a positive integer"
                )

        for name in (
            "default_tick_interval_seconds",
            "maximum_tick_interval_seconds",
            "default_execution_timeout_seconds",
            "maximum_execution_timeout_seconds",
            "robot_state_max_age_seconds",
        ):
            validate_positive_number(
                getattr(self, name),
                name,
            )

        if (
            self.default_tick_interval_seconds
            > self.maximum_tick_interval_seconds
        ):
            raise TreeValidationError(
                "Default tick interval exceeds maximum"
            )

        if (
            self.default_execution_timeout_seconds
            > self.maximum_execution_timeout_seconds
        ):
            raise TreeValidationError(
                "Default timeout exceeds maximum"
            )

        if not 0 <= self.minimum_battery_percentage <= 100:
            raise TreeValidationError(
                "Minimum battery percentage must be 0–100"
            )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class TreeAccessContext:
    tenant_id: str
    user_id: str

    allowed_robot_ids: frozenset[str] = frozenset()

    roles: frozenset[str] = frozenset()

    can_register: bool = False
    can_execute: bool = False
    can_modify: bool = False
    can_cancel: bool = False
    can_approve: bool = False
    can_manage_robot_state: bool = False
    can_delete: bool = False

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
# TREE SPECIFICATION
# ============================================================

@dataclass(frozen=True)
class BehaviorTreeSpecification:
    tree_id: str

    tenant_id: str
    robot_id: str

    name: str

    description: str = ""

    version: str = "1.0.0"

    tags: tuple[str, ...] = ()

    created_by: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "tree_id",
            "tenant_id",
            "robot_id",
            "name",
        ):
            validate_identifier(
                getattr(self, name),
                name,
            )

        if self.created_at.tzinfo is None:
            raise TreeValidationError(
                "created_at must be timezone-aware"
            )


# ============================================================
# TREE EVENTS
# ============================================================

@dataclass(frozen=True)
class BehaviorTreeEvent:
    event_id: str

    tree_id: str
    tenant_id: str
    robot_id: str

    event_type: TreeEventType

    actor: str

    timestamp: datetime = field(
        default_factory=utc_now
    )

    details: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# TREE RUNTIME
# ============================================================

@dataclass
class BehaviorTreeRuntime:
    specification: BehaviorTreeSpecification

    engine: RoboticsBehaviorTreeEngine

    status: TreeLifecycleStatus = (
        TreeLifecycleStatus.REGISTERED
    )

    last_node_status: NodeStatus = NodeStatus.IDLE

    registered_at: datetime = field(
        default_factory=utc_now
    )

    started_at: datetime | None = None

    completed_at: datetime | None = None

    last_tick_at: datetime | None = None

    total_ticks: int = 0

    execution_count: int = 0

    last_error: str | None = None

    paused: bool = False

    cancellation_requested: bool = False

    worker_running: bool = False

    worker: threading.Thread | None = field(
        default=None,
        repr=False,
    )

    stop_event: threading.Event = field(
        default_factory=threading.Event,
        repr=False,
    )

    events: deque[BehaviorTreeEvent] = field(
        default_factory=deque
    )

    lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )

    def snapshot(self) -> dict[str, Any]:
        return {
            "specification": json_safe(
                self.specification
            ),
            "status": self.status.value,
            "last_node_status": (
                self.last_node_status.value
            ),
            "registered_at": (
                self.registered_at.isoformat()
            ),
            "started_at": (
                self.started_at.isoformat()
                if self.started_at
                else None
            ),
            "completed_at": (
                self.completed_at.isoformat()
                if self.completed_at
                else None
            ),
            "last_tick_at": (
                self.last_tick_at.isoformat()
                if self.last_tick_at
                else None
            ),
            "total_ticks": self.total_ticks,
            "execution_count": self.execution_count,
            "last_error": self.last_error,
            "paused": self.paused,
            "cancellation_requested": (
                self.cancellation_requested
            ),
            "worker_running": self.worker_running,
            "hardware_execution": False,
        }


# ============================================================
# SAFETY VALIDATION
# ============================================================

@dataclass(frozen=True)
class TreeSafetyReport:
    tree_id: str
    robot_id: str

    decision: TreeSafetyDecision

    reasons: tuple[str, ...]

    evaluated_at: datetime = field(
        default_factory=utc_now
    )

    @property
    def allowed(self) -> bool:
        return (
            self.decision
            == TreeSafetyDecision.ALLOWED
        )


class BehaviorTreeSafetyValidator:

    def __init__(
        self,
        config: BehaviorTreeManagerConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        *,
        tree_id: str,
        robot: RobotBehaviorState,
    ) -> TreeSafetyReport:
        reasons: list[str] = []

        if self.config.require_emergency_stop_inactive:
            if robot.emergency_stop_active:
                reasons.append(
                    "Emergency stop is active"
                )

        if self.config.require_safe_robot_state:
            if robot.safety_state != RobotSafetyState.SAFE:
                reasons.append(
                    "Robot safety state is not SAFE"
                )

        if self.config.require_no_active_faults:
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
                "Robot operational state is unavailable"
            )

        if robot.battery_percentage is not None:
            if (
                robot.battery_percentage
                < self.config.minimum_battery_percentage
            ):
                reasons.append(
                    "Robot battery below minimum threshold"
                )

        age = (
            utc_now() - robot.last_updated
        ).total_seconds()

        if (
            age < 0
            or age > self.config.robot_state_max_age_seconds
        ):
            reasons.append(
                "Robot state is stale or future-dated"
            )

        return TreeSafetyReport(
            tree_id=tree_id,
            robot_id=robot.robot_id,
            decision=(
                TreeSafetyDecision.ALLOWED
                if not reasons
                else TreeSafetyDecision.BLOCKED
            ),
            reasons=tuple(reasons),
        )


# ============================================================
# EXECUTION REPORT
# ============================================================

@dataclass(frozen=True)
class BehaviorTreeExecutionReport:
    tree_id: str
    tenant_id: str
    robot_id: str

    lifecycle_status: TreeLifecycleStatus

    node_status: NodeStatus

    ticks_executed: int

    execution_time_seconds: float

    simulation_only: bool = True

    error: str | None = None

    timestamp: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# MAIN BEHAVIOR TREE MANAGER
# ============================================================

class RoboticsBehaviorTreeManager:
    """
    Thread-safe behavior tree registry and simulation manager.

    Security boundaries:
        - Tenant isolation
        - Explicit robot permissions
        - Explicit operation permissions
        - Safety validation before every tick
        - No physical actuation
        - Human approvals managed separately

    Registered trees contain trusted Python node objects.
    Untrusted JSON must not be converted into executable
    callback nodes without an explicit allowlist.
    """

    TERMINAL_STATUSES = frozenset({
        TreeLifecycleStatus.COMPLETED,
        TreeLifecycleStatus.FAILED,
        TreeLifecycleStatus.CANCELLED,
        TreeLifecycleStatus.TIMEOUT,
    })

    def __init__(
        self,
        config: BehaviorTreeManagerConfig | None = None,
    ) -> None:
        self.config = (
            config
            or BehaviorTreeManagerConfig()
        )

        self._trees: dict[
            str,
            BehaviorTreeRuntime,
        ] = {}

        self._safety_validator = (
            BehaviorTreeSafetyValidator(
                self.config
            )
        )

        self._lock = threading.RLock()

        self._metrics = Counter()

    # --------------------------------------------------------
    # AUTHORIZATION
    # --------------------------------------------------------

    @staticmethod
    def _authorize_robot(
        access: TreeAccessContext,
        tenant_id: str,
        robot_id: str,
    ) -> None:
        if access.tenant_id != tenant_id:
            raise TreePermissionError(
                "Cross-tenant access denied"
            )

        if robot_id not in access.allowed_robot_ids:
            raise TreePermissionError(
                "Robot access denied"
            )

    @staticmethod
    def _require_permission(
        access: TreeAccessContext,
        permission: str,
    ) -> None:
        if not getattr(access, permission, False):
            raise TreePermissionError(
                f"Permission required: {permission}"
            )

    def _get_runtime(
        self,
        tree_id: str,
        access: TreeAccessContext,
    ) -> BehaviorTreeRuntime:
        with self._lock:
            runtime = self._trees.get(
                tree_id
            )

        if runtime is None:
            raise TreeNotFoundError(
                f"Tree not found: {tree_id}"
            )

        self._authorize_robot(
            access,
            runtime.specification.tenant_id,
            runtime.specification.robot_id,
        )

        return runtime

    # --------------------------------------------------------
    # EVENT RECORDING
    # --------------------------------------------------------

    def _record_event(
        self,
        runtime: BehaviorTreeRuntime,
        event_type: TreeEventType,
        *,
        actor: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        event = BehaviorTreeEvent(
            event_id=generate_id(),
            tree_id=runtime.specification.tree_id,
            tenant_id=runtime.specification.tenant_id,
            robot_id=runtime.specification.robot_id,
            event_type=event_type,
            actor=actor,
            details=json_safe(
                details or {}
            ),
        )

        with runtime.lock:
            if (
                len(runtime.events)
                >= self.config.maximum_events_per_tree
            ):
                runtime.events.popleft()

            runtime.events.append(
                event
            )

    # --------------------------------------------------------
    # REGISTER TREE
    # --------------------------------------------------------

    def register_tree(
        self,
        *,
        specification: BehaviorTreeSpecification,
        root: BehaviorTreeNode,
        robot: RobotBehaviorState,
        access: TreeAccessContext,
        blackboard: Blackboard | None = None,
        node_config: BehaviorTreeConfig | None = None,
    ) -> dict[str, Any]:
        self._require_permission(
            access,
            "can_register",
        )

        self._authorize_robot(
            access,
            specification.tenant_id,
            specification.robot_id,
        )

        if (
            robot.tenant_id != specification.tenant_id
            or robot.robot_id != specification.robot_id
        ):
            raise TreeValidationError(
                "Robot identity does not match tree specification"
            )

        with self._lock:
            if specification.tree_id in self._trees:
                raise TreeAlreadyExistsError(
                    "Tree ID already registered"
                )

            if (
                len(self._trees)
                >= self.config.maximum_registered_trees
            ):
                raise TreeCapacityError(
                    "Maximum registered trees reached"
                )

            engine_access = BehaviorTreeAccessContext(
                tenant_id=access.tenant_id,
                user_id=access.user_id,
                allowed_robot_ids=access.allowed_robot_ids,
                roles=access.roles,
                can_execute_simulation=True,
                can_modify_tree=access.can_modify,
                can_approve_actions=access.can_approve,
            )

            engine = RoboticsBehaviorTreeEngine(
                root=root,
                access=engine_access,
                robot=robot,
                config=(
                    node_config
                    or BehaviorTreeConfig(
                        robot_state_max_age_seconds=(
                            self.config.robot_state_max_age_seconds
                        ),
                        maximum_ticks=(
                            self.config.maximum_ticks_per_tree
                        ),
                    )
                ),
                blackboard=blackboard,
                tree_id=specification.tree_id,
            )

            runtime = BehaviorTreeRuntime(
                specification=specification,
                engine=engine,
                status=TreeLifecycleStatus.READY,
                events=deque(
                    maxlen=self.config.maximum_events_per_tree
                ),
            )

            self._trees[
                specification.tree_id
            ] = runtime

            self._metrics["registered"] += 1

            self._record_event(
                runtime,
                TreeEventType.REGISTERED,
                actor=access.user_id,
                details={
                    "name": specification.name,
                    "node_count": (
                        engine.validation_report["node_count"]
                    ),
                    "hardware_execution": False,
                },
            )

            return runtime.snapshot()

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    def check_safety(
        self,
        tree_id: str,
        *,
        access: TreeAccessContext,
    ) -> TreeSafetyReport:
        runtime = self._get_runtime(
            tree_id,
            access,
        )

        with runtime.lock:
            return self._safety_validator.evaluate(
                tree_id=tree_id,
                robot=runtime.engine.context.robot,
            )

    # --------------------------------------------------------
    # START TREE
    # --------------------------------------------------------

    def start_tree(
        self,
        tree_id: str,
        *,
        access: TreeAccessContext,
    ) -> dict[str, Any]:
        self._require_permission(
            access,
            "can_execute",
        )

        runtime = self._get_runtime(
            tree_id,
            access,
        )

        with runtime.lock:
            if runtime.worker_running:
                raise TreeStateError(
                    "Tree already has an active worker"
                )

            if runtime.status not in (
                TreeLifecycleStatus.READY,
                TreeLifecycleStatus.PAUSED,
            ):
                raise TreeStateError(
                    f"Cannot start tree in status "
                    f"{runtime.status.value}"
                )

            safety = self._safety_validator.evaluate(
                tree_id=tree_id,
                robot=runtime.engine.context.robot,
            )

            if not safety.allowed:
                runtime.status = (
                    TreeLifecycleStatus.BLOCKED_SAFETY
                )

                runtime.last_error = "; ".join(
                    safety.reasons
                )

                self._record_event(
                    runtime,
                    TreeEventType.SAFETY_BLOCKED,
                    actor=access.user_id,
                    details={
                        "reasons": safety.reasons,
                    },
                )

                raise TreeSafetyError(
                    runtime.last_error
                )

            if runtime.status == TreeLifecycleStatus.READY:
                runtime.engine.reset()

                runtime.execution_count += 1
                runtime.started_at = utc_now()
                runtime.completed_at = None

            runtime.status = TreeLifecycleStatus.RUNNING

            runtime.paused = False
            runtime.cancellation_requested = False

            runtime.stop_event.clear()

            self._record_event(
                runtime,
                TreeEventType.STARTED,
                actor=access.user_id,
                details={
                    "execution_count": (
                        runtime.execution_count
                    ),
                },
            )

            return runtime.snapshot()

    # --------------------------------------------------------
    # SINGLE TICK
    # --------------------------------------------------------

    def tick_tree(
        self,
        tree_id: str,
        *,
        access: TreeAccessContext,
    ) -> BehaviorTreeExecutionReport:
        self._require_permission(
            access,
            "can_execute",
        )

        runtime = self._get_runtime(
            tree_id,
            access,
        )

        with runtime.lock:
            if runtime.status != TreeLifecycleStatus.RUNNING:
                raise TreeStateError(
                    "Tree must be RUNNING before ticking"
                )

            if runtime.cancellation_requested:
                raise TreeStateError(
                    "Tree cancellation requested"
                )

            safety = self._safety_validator.evaluate(
                tree_id=tree_id,
                robot=runtime.engine.context.robot,
            )

            if not safety.allowed:
                runtime.engine.root.halt(
                    runtime.engine.context
                )

                runtime.status = (
                    TreeLifecycleStatus.BLOCKED_SAFETY
                )

                runtime.last_error = "; ".join(
                    safety.reasons
                )

                self._record_event(
                    runtime,
                    TreeEventType.SAFETY_BLOCKED,
                    actor=access.user_id,
                    details={
                        "reasons": safety.reasons,
                    },
                )

                return BehaviorTreeExecutionReport(
                    tree_id=tree_id,
                    tenant_id=runtime.specification.tenant_id,
                    robot_id=runtime.specification.robot_id,
                    lifecycle_status=runtime.status,
                    node_status=runtime.last_node_status,
                    ticks_executed=0,
                    execution_time_seconds=0.0,
                    error=runtime.last_error,
                )

            started = time.perf_counter()

            try:
                result = runtime.engine.tick()

            except Exception as exc:
                runtime.status = TreeLifecycleStatus.FAILED
                runtime.last_error = str(exc)
                runtime.completed_at = utc_now()

                self._metrics["failed"] += 1

                self._record_event(
                    runtime,
                    TreeEventType.FAILED,
                    actor=access.user_id,
                    details={
                        "error": str(exc),
                    },
                )

                raise TreeExecutionError(
                    f"Behavior tree tick failed: {exc}"
                ) from exc

            elapsed = (
                time.perf_counter() - started
            )

            runtime.total_ticks += 1
            runtime.last_tick_at = utc_now()
            runtime.last_node_status = result.status

            self._metrics["ticks"] += 1

            if result.status == NodeStatus.SUCCESS:
                runtime.status = (
                    TreeLifecycleStatus.COMPLETED
                )

                runtime.completed_at = utc_now()

                self._metrics["completed"] += 1

                self._record_event(
                    runtime,
                    TreeEventType.COMPLETED,
                    actor=access.user_id,
                )

            elif result.status == NodeStatus.FAILURE:
                runtime.status = (
                    TreeLifecycleStatus.FAILED
                )

                runtime.completed_at = utc_now()

                self._metrics["failed"] += 1

                self._record_event(
                    runtime,
                    TreeEventType.FAILED,
                    actor=access.user_id,
                    details={
                        "reason": "Root node returned FAILURE",
                    },
                )

            else:
                self._record_event(
                    runtime,
                    TreeEventType.TICKED,
                    actor=access.user_id,
                    details={
                        "node_status": result.status.value,
                    },
                )

            return BehaviorTreeExecutionReport(
                tree_id=tree_id,
                tenant_id=runtime.specification.tenant_id,
                robot_id=runtime.specification.robot_id,
                lifecycle_status=runtime.status,
                node_status=result.status,
                ticks_executed=1,
                execution_time_seconds=elapsed,
            )

    # --------------------------------------------------------
    # RUN TREE SYNCHRONOUSLY
    # --------------------------------------------------------

    def run_tree(
        self,
        tree_id: str,
        *,
        access: TreeAccessContext,
        maximum_ticks: int | None = None,
        timeout_seconds: float | None = None,
        tick_interval_seconds: float | None = None,
    ) -> BehaviorTreeExecutionReport:
        self._require_permission(
            access,
            "can_execute",
        )

        tick_limit = (
            maximum_ticks
            if maximum_ticks is not None
            else self.config.maximum_run_ticks
        )

        timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self.config.default_execution_timeout_seconds
        )

        interval = (
            tick_interval_seconds
            if tick_interval_seconds is not None
            else self.config.default_tick_interval_seconds
        )

        if not 1 <= tick_limit <= self.config.maximum_run_ticks:
            raise TreeValidationError(
                "Invalid maximum tick count"
            )

        validate_positive_number(
            timeout,
            "timeout_seconds",
        )

        if timeout > self.config.maximum_execution_timeout_seconds:
            raise TreeValidationError(
                "Execution timeout exceeds maximum"
            )

        if not (
            0 <= interval
            <= self.config.maximum_tick_interval_seconds
        ):
            raise TreeValidationError(
                "Invalid tick interval"
            )

        runtime = self._get_runtime(
            tree_id,
            access,
        )

        if runtime.status == TreeLifecycleStatus.READY:
            self.start_tree(
                tree_id,
                access=access,
            )

        started = time.monotonic()
        executed = 0
        last_status = runtime.last_node_status

        while executed < tick_limit:
            with runtime.lock:
                status = runtime.status

                if status != TreeLifecycleStatus.RUNNING:
                    break

                if runtime.stop_event.is_set():
                    break

            elapsed = time.monotonic() - started

            if elapsed >= timeout:
                with runtime.lock:
                    runtime.engine.root.halt(
                        runtime.engine.context
                    )

                    runtime.status = TreeLifecycleStatus.TIMEOUT
                    runtime.completed_at = utc_now()
                    runtime.last_error = (
                        "Tree execution timeout exceeded"
                    )

                    self._metrics["timeouts"] += 1

                    self._record_event(
                        runtime,
                        TreeEventType.TIMEOUT,
                        actor=access.user_id,
                    )

                break

            report = self.tick_tree(
                tree_id,
                access=access,
            )

            executed += 1
            last_status = report.node_status

            if report.lifecycle_status != TreeLifecycleStatus.RUNNING:
                break

            if interval > 0:
                runtime.stop_event.wait(
                    interval
                )

        with runtime.lock:
            return BehaviorTreeExecutionReport(
                tree_id=tree_id,
                tenant_id=runtime.specification.tenant_id,
                robot_id=runtime.specification.robot_id,
                lifecycle_status=runtime.status,
                node_status=last_status,
                ticks_executed=executed,
                execution_time_seconds=(
                    time.monotonic() - started
                ),
                error=runtime.last_error,
            )

    # --------------------------------------------------------
    # BACKGROUND SIMULATION
    # --------------------------------------------------------

    def run_background(
        self,
        tree_id: str,
        *,
        access: TreeAccessContext,
        maximum_ticks: int | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self._require_permission(
            access,
            "can_execute",
        )

        if not self.config.allow_background_simulation:
            raise TreeStateError(
                "Background simulation is disabled"
            )

        runtime = self._get_runtime(
            tree_id,
            access,
        )

        with self._lock:
            active_workers = sum(
                candidate.worker_running
                for candidate in self._trees.values()
            )

            if (
                active_workers
                >= self.config.maximum_concurrent_workers
            ):
                raise TreeCapacityError(
                    "Maximum concurrent workers reached"
                )

            with runtime.lock:
                if runtime.worker_running:
                    raise TreeStateError(
                        "Tree already has a running worker"
                    )

                if runtime.status == TreeLifecycleStatus.READY:
                    self.start_tree(
                        tree_id,
                        access=access,
                    )

                if runtime.status != TreeLifecycleStatus.RUNNING:
                    raise TreeStateError(
                        "Tree is not in RUNNING state"
                    )

                runtime.worker_running = True

                def worker_main() -> None:
                    try:
                        self.run_tree(
                            tree_id,
                            access=access,
                            maximum_ticks=maximum_ticks,
                            timeout_seconds=timeout_seconds,
                        )

                    except Exception as exc:
                        with runtime.lock:
                            if runtime.status == TreeLifecycleStatus.RUNNING:
                                runtime.status = TreeLifecycleStatus.FAILED
                                runtime.last_error = str(exc)
                                runtime.completed_at = utc_now()

                    finally:
                        with runtime.lock:
                            runtime.worker_running = False
                            runtime.worker = None

                worker = threading.Thread(
                    target=worker_main,
                    name=f"behavior-tree-{tree_id}",
                    daemon=True,
                )

                runtime.worker = worker

                worker.start()

    # --------------------------------------------------------
    # PAUSE TREE
    # --------------------------------------------------------

    def pause_tree(
        self,
        tree_id: str,
        *,
        access: TreeAccessContext,
    ) -> dict[str, Any]:
        self._require_permission(
            access,
            "can_modify",
        )

        runtime = self._get_runtime(
            tree_id,
            access,
        )

        with runtime.lock:
            if runtime.status != TreeLifecycleStatus.RUNNING:
                raise TreeStateError(
                    "Only RUNNING trees can be paused"
                )

            runtime.status = TreeLifecycleStatus.PAUSED
            runtime.paused = True

            runtime.stop_event.set()

            self._record_event(
                runtime,
                TreeEventType.PAUSED,
                actor=access.user_id,
            )

            return runtime.snapshot()

    # --------------------------------------------------------
    # RESUME TREE
    # --------------------------------------------------------

    def resume_tree(
        self,
        tree_id: str,
        *,
        access: TreeAccessContext,
    ) -> dict[str, Any]:
        self._require_permission(
            access,
            "can_modify",
        )

        runtime = self._get_runtime(
            tree_id,
            access,
        )

        with runtime.lock:
            if runtime.worker_running:
                raise TreeStateError(
                    "Wait for existing worker to stop"
                )

            if runtime.status != TreeLifecycleStatus.PAUSED:
                raise TreeStateError(
                    "Only PAUSED trees can be resumed"
                )

            safety = self._safety_validator.evaluate(
                tree_id=tree_id,
                robot=runtime.engine.context.robot,
            )

            if not safety.allowed:
                runtime.status = (
                    TreeLifecycleStatus.BLOCKED_SAFETY
                )

                runtime.last_error = "; ".join(
                    safety.reasons
                )

                raise TreeSafetyError(
                    runtime.last_error
                )

            runtime.status = TreeLifecycleStatus.RUNNING
            runtime.paused = False

            runtime.stop_event.clear()

            self._record_event(
                runtime,
                TreeEventType.RESUMED,
                actor=access.user_id,
            )

            return runtime.snapshot()

    # --------------------------------------------------------
    # CANCEL TREE
    # --------------------------------------------------------

    def cancel_tree(
        self,
        tree_id: str,
        *,
        access: TreeAccessContext,
        reason: str = "Cancelled by user",
    ) -> dict[str, Any]:
        self._require_permission(
            access,
            "can_cancel",
        )

        runtime = self._get_runtime(
            tree_id,
            access,
        )

        with runtime.lock:
            if runtime.status in self.TERMINAL_STATUSES:
                return runtime.snapshot()

            runtime.cancellation_requested = True

            runtime.stop_event.set()

            runtime.engine.cancel()

            runtime.status = TreeLifecycleStatus.CANCELLED
            runtime.completed_at = utc_now()
            runtime.last_error = reason

            self._metrics["cancelled"] += 1

            self._record_event(
                runtime,
                TreeEventType.CANCELLED,
                actor=access.user_id,
                details={
                    "reason": reason,
                },
            )

            return runtime.snapshot()

    # --------------------------------------------------------
    # RESET TREE
    # --------------------------------------------------------

    def reset_tree(
        self,
        tree_id: str,
        *,
        access: TreeAccessContext,
        clear_blackboard: bool = False,
    ) -> dict[str, Any]:
        self._require_permission(
            access,
            "can_modify",
        )

        runtime = self._get_runtime(
            tree_id,
            access,
        )

        with runtime.lock:
            if runtime.worker_running:
                raise TreeStateError(
                    "Cannot reset tree while worker is running"
                )

            runtime.engine.reset()

            if clear_blackboard:
                runtime.engine.context.blackboard.clear()

            runtime.status = TreeLifecycleStatus.READY

            runtime.last_node_status = NodeStatus.IDLE
            runtime.started_at = None
            runtime.completed_at = None
            runtime.last_tick_at = None

            runtime.last_error = None

            runtime.paused = False
            runtime.cancellation_requested = False

            runtime.stop_event.clear()

            self._record_event(
                runtime,
                TreeEventType.RESET,
                actor=access.user_id,
                details={
                    "clear_blackboard": clear_blackboard,
                },
            )

            return runtime.snapshot()

    # --------------------------------------------------------
    # UPDATE ROBOT STATE
    # --------------------------------------------------------

    def update_robot_state(
        self,
        tree_id: str,
        *,
        robot: RobotBehaviorState,
        access: TreeAccessContext,
    ) -> None:
        self._require_permission(
            access,
            "can_manage_robot_state",
        )

        runtime = self._get_runtime(
            tree_id,
            access,
        )

        self._authorize_robot(
            access,
            robot.tenant_id,
            robot.robot_id,
        )

        with runtime.lock:
            runtime.engine.update_robot_state(
                robot
            )

            self._record_event(
                runtime,
                TreeEventType.ROBOT_STATE_UPDATED,
                actor=access.user_id,
                details={
                    "safety_state": robot.safety_state.value,
                    "operational_state": (
                        robot.operational_state.value
                    ),
                    "emergency_stop_active": (
                        robot.emergency_stop_active
                    ),
                },
            )

            safety = self._safety_validator.evaluate(
                tree_id=tree_id,
                robot=robot,
            )

            if not safety.allowed:
                if runtime.status == TreeLifecycleStatus.RUNNING:
                    runtime.engine.root.halt(
                        runtime.engine.context
                    )

                    runtime.status = (
                        TreeLifecycleStatus.BLOCKED_SAFETY
                    )

                    runtime.stop_event.set()

                    runtime.last_error = "; ".join(
                        safety.reasons
                    )

                    self._record_event(
                        runtime,
                        TreeEventType.SAFETY_BLOCKED,
                        actor=access.user_id,
                        details={
                            "reasons": safety.reasons,
                        },
                    )

    # --------------------------------------------------------
    # HUMAN APPROVAL
    # --------------------------------------------------------

    def approve_action(
        self,
        tree_id: str,
        approval_id: str,
        *,
        access: TreeAccessContext,
    ) -> None:
        self._require_permission(
            access,
            "can_approve",
        )

        validate_identifier(
            approval_id,
            "approval_id",
        )

        runtime = self._get_runtime(
            tree_id,
            access,
        )

        engine_access = BehaviorTreeAccessContext(
            tenant_id=access.tenant_id,
            user_id=access.user_id,
            allowed_robot_ids=access.allowed_robot_ids,
            roles=access.roles,
            can_approve_actions=True,
        )

        with runtime.lock:
            runtime.engine.add_approval(
                approval_id,
                access=engine_access,
            )

            self._record_event(
                runtime,
                TreeEventType.APPROVAL_ADDED,
                actor=access.user_id,
                details={
                    "approval_id": approval_id,
                    "physical_execution_authorized": False,
                },
            )

    def revoke_approval(
        self,
        tree_id: str,
        approval_id: str,
        *,
        access: TreeAccessContext,
    ) -> None:
        self._require_permission(
            access,
            "can_approve",
        )

        runtime = self._get_runtime(
            tree_id,
            access,
        )

        with runtime.lock:
            context = runtime.engine.context

            context.approvals = frozenset(
                approval
                for approval in context.approvals
                if approval != approval_id
            )

            self._record_event(
                runtime,
                TreeEventType.APPROVAL_REVOKED,
                actor=access.user_id,
                details={
                    "approval_id": approval_id,
                },
            )

    # --------------------------------------------------------
    # GET TREE
    # --------------------------------------------------------

    def get_tree(
        self,
        tree_id: str,
        *,
        access: TreeAccessContext,
    ) -> dict[str, Any]:
        runtime = self._get_runtime(
            tree_id,
            access,
        )

        with runtime.lock:
            return runtime.snapshot()

    # --------------------------------------------------------
    # LIST TREES
    # --------------------------------------------------------

    def list_trees(
        self,
        *,
        access: TreeAccessContext,
        status: TreeLifecycleStatus | None = None,
        robot_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if (
            robot_id is not None
            and robot_id not in access.allowed_robot_ids
        ):
            raise TreePermissionError(
                "Robot access denied"
            )

        with self._lock:
            runtimes = list(
                self._trees.values()
            )

        results = []

        for runtime in runtimes:
            specification = runtime.specification

            if specification.tenant_id != access.tenant_id:
                continue

            if specification.robot_id not in access.allowed_robot_ids:
                continue

            if (
                robot_id is not None
                and specification.robot_id != robot_id
            ):
                continue

            with runtime.lock:
                if status is not None and runtime.status != status:
                    continue

                results.append(
                    runtime.snapshot()
                )

        return results

    # --------------------------------------------------------
    # BLACKBOARD
    # --------------------------------------------------------

    def get_blackboard(
        self,
        tree_id: str,
        *,
        access: TreeAccessContext,
    ) -> dict[str, Any]:
        runtime = self._get_runtime(
            tree_id,
            access,
        )

        with runtime.lock:
            return runtime.engine.context.blackboard.snapshot()

    def set_blackboard_value(
        self,
        tree_id: str,
        *,
        key: str,
        value: Any,
        access: TreeAccessContext,
    ) -> None:
        self._require_permission(
            access,
            "can_modify",
        )

        runtime = self._get_runtime(
            tree_id,
            access,
        )

        with runtime.lock:
            if runtime.status == TreeLifecycleStatus.RUNNING:
                raise TreeStateError(
                    "Blackboard modification is blocked while running"
                )

            runtime.engine.context.blackboard.set(
                key,
                value,
            )

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    def visualize_tree(
        self,
        tree_id: str,
        *,
        access: TreeAccessContext,
    ) -> str:
        runtime = self._get_runtime(
            tree_id,
            access,
        )

        with runtime.lock:
            return runtime.engine.visualize()

    # --------------------------------------------------------
    # NODE TRACE
    # --------------------------------------------------------

    def node_trace(
        self,
        tree_id: str,
        *,
        access: TreeAccessContext,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        runtime = self._get_runtime(
            tree_id,
            access,
        )

        with runtime.lock:
            return runtime.engine.trace(
                limit=limit
            )

    # --------------------------------------------------------
    # EVENT HISTORY
    # --------------------------------------------------------

    def event_history(
        self,
        tree_id: str,
        *,
        access: TreeAccessContext,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        runtime = self._get_runtime(
            tree_id,
            access,
        )

        if limit < 1:
            raise TreeValidationError(
                "Limit must be positive"
            )

        with runtime.lock:
            return [
                json_safe(event)
                for event in list(runtime.events)[-limit:]
            ]

    # --------------------------------------------------------
    # DELETE TREE
    # --------------------------------------------------------

    def delete_tree(
        self,
        tree_id: str,
        *,
        access: TreeAccessContext,
    ) -> None:
        self._require_permission(
            access,
            "can_delete",
        )

        runtime = self._get_runtime(
            tree_id,
            access,
        )

        with self._lock:
            with runtime.lock:
                if runtime.worker_running:
                    raise TreeStateError(
                        "Cannot delete tree with active worker"
                    )

                if runtime.status == TreeLifecycleStatus.RUNNING:
                    raise TreeStateError(
                        "Cancel or pause tree before deletion"
                    )

                self._record_event(
                    runtime,
                    TreeEventType.DELETED,
                    actor=access.user_id,
                )

                del self._trees[
                    tree_id
                ]

                self._metrics["deleted"] += 1

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        access: TreeAccessContext,
    ) -> dict[str, Any]:
        trees = self.list_trees(
            access=access
        )

        return {
            "tenant_id": access.tenant_id,
            "total_trees": len(trees),
            "statuses": dict(
                Counter(
                    tree["status"]
                    for tree in trees
                )
            ),
            "total_ticks": sum(
                tree["total_ticks"]
                for tree in trees
            ),
            "total_executions": sum(
                tree["execution_count"]
                for tree in trees
            ),
            "active_workers": sum(
                tree["worker_running"]
                for tree in trees
            ),
            "robots": sorted({
                tree["specification"]["robot_id"]
                for tree in trees
            }),
            "hardware_execution": False,
            "timestamp": utc_iso(),
        }

    # --------------------------------------------------------
    # JSON EXPORT
    # --------------------------------------------------------

    def export_json(
        self,
        *,
        access: TreeAccessContext,
        include_blackboards: bool = True,
        include_traces: bool = True,
    ) -> str:
        trees = self.list_trees(
            access=access
        )

        detailed_trees = []

        for tree in trees:
            tree_id = tree["specification"]["tree_id"]

            entry = {
                "runtime": tree,
                "visualization": self.visualize_tree(
                    tree_id,
                    access=access,
                ),
            }

            if include_blackboards:
                entry["blackboard"] = self.get_blackboard(
                    tree_id,
                    access=access,
                )

            if include_traces:
                entry["node_trace"] = self.node_trace(
                    tree_id,
                    access=access,
                )

            detailed_trees.append(
                entry
            )

        payload = {
            "platform": (
                "Physical AI & Autonomous Robotics "
                "Intelligence Platform"
            ),
            "module": "behavior_tree",
            "tenant_id": access.tenant_id,
            "trees": detailed_trees,
            "statistics": self.statistics(
                access=access
            ),
            "simulation_only": True,
            "hardware_execution": False,
            "exported_at": utc_iso(),
        }

        return json.dumps(
            json_safe(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


# ============================================================
# TREE FACTORY
# ============================================================

class BehaviorTreeFactory:
    """
    Helper for constructing commonly used robotics trees.

    Factory output is composed of trusted node classes.
    """

    @staticmethod
    def inspection_workflow(
        *,
        minimum_battery_percentage: float = 20.0,
    ) -> BehaviorTreeNode:
        try:
            from .behavior_tree_nodes import (
                SequenceNode,
                RobotSafetyConditionNode,
                RobotBatteryConditionNode,
                RobotCapabilityConditionNode,
                RobotInspectionNode,
                RobotDiagnosticsNode,
                BlackboardSetNode,
            )
        except ImportError:
            from behavior_tree_nodes import (
                SequenceNode,
                RobotSafetyConditionNode,
                RobotBatteryConditionNode,
                RobotCapabilityConditionNode,
                RobotInspectionNode,
                RobotDiagnosticsNode,
                BlackboardSetNode,
            )

        return SequenceNode(
            "Robot Inspection Workflow",
            [
                RobotSafetyConditionNode(
                    "Validate Robot Safety"
                ),
                RobotBatteryConditionNode(
                    "Validate Battery",
                    minimum_percentage=(
                        minimum_battery_percentage
                    ),
                ),
                RobotCapabilityConditionNode(
                    "Validate Inspection Capabilities",
                    required_capabilities=(
                        "inspection",
                        "diagnostics",
                    ),
                ),
                RobotInspectionNode(
                    "Inspect Robot Components",
                    inspection_type="visual",
                    target_component="robot",
                    result_key="inspection_result",
                ),
                RobotDiagnosticsNode(
                    "Run Robot Diagnostics",
                    subsystem="general",
                    result_key="diagnostics_result",
                ),
                BlackboardSetNode(
                    "Mark Workflow Complete",
                    key="workflow_completed",
                    value=True,
                ),
            ],
        )

    @staticmethod
    def navigation_simulation_workflow(
        *,
        destination: str,
        approval_id: str,
    ) -> BehaviorTreeNode:
        try:
            from .behavior_tree_nodes import (
                SequenceNode,
                RobotSafetyConditionNode,
                RobotBatteryConditionNode,
                RobotCapabilityConditionNode,
                HumanApprovalConditionNode,
                RobotNavigationNode,
            )
        except ImportError:
            from behavior_tree_nodes import (
                SequenceNode,
                RobotSafetyConditionNode,
                RobotBatteryConditionNode,
                RobotCapabilityConditionNode,
                HumanApprovalConditionNode,
                RobotNavigationNode,
            )

        return SequenceNode(
            "Navigation Simulation Workflow",
            [
                RobotSafetyConditionNode(
                    "Validate Safety"
                ),
                RobotBatteryConditionNode(
                    "Validate Battery",
                    minimum_percentage=25.0,
                ),
                RobotCapabilityConditionNode(
                    "Validate Navigation Capability",
                    required_capabilities=(
                        "navigation",
                    ),
                ),
                HumanApprovalConditionNode(
                    "Validate Human Approval",
                    approval_id=approval_id,
                ),
                RobotNavigationNode(
                    "Simulate Navigation",
                    destination=destination,
                    approval_id=approval_id,
                    result_key="navigation_result",
                ),
            ],
        )


# ============================================================
# UNIT TESTS
# ============================================================

class TestRoboticsBehaviorTreeManager(unittest.TestCase):

    def setUp(self) -> None:
        self.manager = RoboticsBehaviorTreeManager(
            BehaviorTreeManagerConfig(
                robot_state_max_age_seconds=60.0,
                default_tick_interval_seconds=0.001,
            )
        )

        self.access = TreeAccessContext(
            tenant_id="tenant_alpha",
            user_id="engineer_001",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
            }),
            roles=frozenset({
                "engineer",
            }),
            can_register=True,
            can_execute=True,
            can_modify=True,
            can_cancel=True,
            can_approve=True,
            can_manage_robot_state=True,
            can_delete=True,
        )

        self.robot = RobotBehaviorState(
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            safety_state=RobotSafetyState.SAFE,
            operational_state=(
                RobotOperationalState.AVAILABLE
            ),
            capabilities=frozenset({
                "inspection",
                "diagnostics",
                "navigation",
            }),
            battery_percentage=85.0,
        )

    def register(
        self,
        tree_id: str,
        root: BehaviorTreeNode | None = None,
    ) -> None:
        if root is None:
            root = BehaviorTreeFactory.inspection_workflow()

        self.manager.register_tree(
            specification=BehaviorTreeSpecification(
                tree_id=tree_id,
                tenant_id="tenant_alpha",
                robot_id="ROBOT-001",
                name="Test Tree",
            ),
            root=root,
            robot=self.robot,
            access=self.access,
        )

    def test_register_tree(self) -> None:
        self.register(
            "tree_001"
        )

        tree = self.manager.get_tree(
            "tree_001",
            access=self.access,
        )

        self.assertEqual(
            tree["status"],
            "ready",
        )

    def test_duplicate_tree(self) -> None:
        self.register(
            "tree_002"
        )

        with self.assertRaises(
            TreeAlreadyExistsError
        ):
            self.register(
                "tree_002"
            )

    def test_safety_check(self) -> None:
        self.register(
            "tree_003"
        )

        report = self.manager.check_safety(
            "tree_003",
            access=self.access,
        )

        self.assertTrue(
            report.allowed
        )

    def test_run_tree(self) -> None:
        self.register(
            "tree_004"
        )

        report = self.manager.run_tree(
            "tree_004",
            access=self.access,
            maximum_ticks=10,
            tick_interval_seconds=0,
        )

        self.assertEqual(
            report.lifecycle_status,
            TreeLifecycleStatus.COMPLETED,
        )

    def test_cancel_tree(self) -> None:
        self.register(
            "tree_005"
        )

        self.manager.start_tree(
            "tree_005",
            access=self.access,
        )

        result = self.manager.cancel_tree(
            "tree_005",
            access=self.access,
        )

        self.assertEqual(
            result["status"],
            "cancelled",
        )

    def test_reset_tree(self) -> None:
        self.register(
            "tree_006"
        )

        self.manager.run_tree(
            "tree_006",
            access=self.access,
            tick_interval_seconds=0,
        )

        result = self.manager.reset_tree(
            "tree_006",
            access=self.access,
        )

        self.assertEqual(
            result["status"],
            "ready",
        )

    def test_blackboard(self) -> None:
        self.register(
            "tree_007"
        )

        self.manager.run_tree(
            "tree_007",
            access=self.access,
            tick_interval_seconds=0,
        )

        blackboard = self.manager.get_blackboard(
            "tree_007",
            access=self.access,
        )

        self.assertTrue(
            blackboard["workflow_completed"]
        )

    def test_tenant_isolation(self) -> None:
        self.register(
            "tree_008"
        )

        unauthorized = TreeAccessContext(
            tenant_id="tenant_beta",
            user_id="other_user",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
            }),
            can_execute=True,
        )

        with self.assertRaises(
            TreePermissionError
        ):
            self.manager.get_tree(
                "tree_008",
                access=unauthorized,
            )

    def test_unsafe_robot_blocks_execution(self) -> None:
        self.register(
            "tree_009"
        )

        unsafe_robot = RobotBehaviorState(
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            safety_state=RobotSafetyState.UNSAFE,
            operational_state=(
                RobotOperationalState.FAULTED
            ),
            capabilities=frozenset({
                "inspection",
                "diagnostics",
            }),
            battery_percentage=80.0,
            active_faults=("MOTOR_FAULT",),
        )

        self.manager.update_robot_state(
            "tree_009",
            robot=unsafe_robot,
            access=self.access,
        )

        with self.assertRaises(
            TreeSafetyError
        ):
            self.manager.start_tree(
                "tree_009",
                access=self.access,
            )

    def test_export_json(self) -> None:
        self.register(
            "tree_010"
        )

        exported = self.manager.export_json(
            access=self.access,
        )

        self.assertIn(
            "tree_010",
            exported,
        )

        self.assertIn(
            "hardware_execution",
            exported,
        )

    def test_delete_tree(self) -> None:
        self.register(
            "tree_011"
        )

        self.manager.delete_tree(
            "tree_011",
            access=self.access,
        )

        with self.assertRaises(
            TreeNotFoundError
        ):
            self.manager.get_tree(
                "tree_011",
                access=self.access,
            )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print(
        "\nPHYSICAL AI & AUTONOMOUS ROBOTICS PLATFORM"
    )

    print(
        "Behavior Tree Orchestration — Simulation Mode\n"
    )

    manager = RoboticsBehaviorTreeManager(
        BehaviorTreeManagerConfig(
            maximum_registered_trees=100,
            robot_state_max_age_seconds=60.0,
            default_tick_interval_seconds=0.01,
            default_execution_timeout_seconds=30.0,
        )
    )

    access = TreeAccessContext(
        tenant_id="tenant_alpha",
        user_id="robotics_engineer",
        roles=frozenset({
            "robotics_engineer",
        }),
        allowed_robot_ids=frozenset({
            "ROBOT-001",
        }),
        can_register=True,
        can_execute=True,
        can_modify=True,
        can_cancel=True,
        can_approve=True,
        can_manage_robot_state=True,
        can_delete=True,
    )

    robot = RobotBehaviorState(
        tenant_id="tenant_alpha",
        robot_id="ROBOT-001",
        safety_state=RobotSafetyState.SAFE,
        operational_state=(
            RobotOperationalState.AVAILABLE
        ),
        capabilities=frozenset({
            "inspection",
            "diagnostics",
            "navigation",
        }),
        battery_percentage=91.0,
    )

    inspection_tree = (
        BehaviorTreeFactory.inspection_workflow(
            minimum_battery_percentage=20.0
        )
    )

    specification = BehaviorTreeSpecification(
        tree_id="robot_inspection_workflow_001",
        tenant_id="tenant_alpha",
        robot_id="ROBOT-001",
        name="Autonomous Robot Inspection Workflow",
        description=(
            "Simulation-only robot safety, inspection "
            "and diagnostic workflow"
        ),
        tags=(
            "inspection",
            "diagnostics",
            "simulation",
        ),
        created_by=access.user_id,
    )

    manager.register_tree(
        specification=specification,
        root=inspection_tree,
        robot=robot,
        access=access,
    )

    print("TREE STRUCTURE\n")

    print(
        manager.visualize_tree(
            specification.tree_id,
            access=access,
        )
    )

    print("\nSAFETY VALIDATION\n")

    safety = manager.check_safety(
        specification.tree_id,
        access=access,
    )

    print(
        json.dumps(
            json_safe(safety),
            indent=2,
        )
    )

    print("\nEXECUTING SIMULATION\n")

    report = manager.run_tree(
        specification.tree_id,
        access=access,
        maximum_ticks=20,
        timeout_seconds=10.0,
        tick_interval_seconds=0,
    )

    print(
        json.dumps(
            json_safe(report),
            indent=2,
        )
    )

    print("\nBLACKBOARD\n")

    print(
        json.dumps(
            manager.get_blackboard(
                specification.tree_id,
                access=access,
            ),
            indent=2,
        )
    )

    print("\nTREE STATISTICS\n")

    print(
        json.dumps(
            manager.statistics(
                access=access,
            ),
            indent=2,
        )
    )

    print("\nEVENT HISTORY\n")

    print(
        json.dumps(
            manager.event_history(
                specification.tree_id,
                access=access,
            ),
            indent=2,
        )
    )

    print("\nUNIT TESTS\n")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestRoboticsBehaviorTreeManager
    )

    unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)


if __name__ == "__main__":
    main()
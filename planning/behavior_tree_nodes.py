"""Behavior Tree Nodes module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/planning/behavior_tree_nodes.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Behavior Tree Nodes — Autonomous Robotics Planning

Python 3.11+
Dependencies: Python standard library only.

Features:
    - Behavior tree execution framework
    - Sequence and selector nodes
    - Parallel execution policies
    - Reactive sequence and selector nodes
    - Condition and action nodes
    - Inverter, retry, repeat and timeout decorators
    - Blackboard shared memory
    - Robot capability checks
    - Robot health and battery conditions
    - Safety-state validation
    - Human approval checks
    - Action scheduler integration
    - Dependency-aware task planning
    - Simulation-only action execution
    - Tick-based execution
    - Node lifecycle management
    - Cancellation and reset
    - Execution tracing
    - Audit logging
    - JSON serialization
    - Tree visualization
    - Unit tests

SAFETY:
    Behavior tree actions are advisory or simulation-only.

    This module does not directly control robot motors,
    actuators, navigation hardware or safety systems.

    Hardware execution must be handled by an independently
    authorized controller with verified safety interlocks.
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
from typing import (
    Any,
    Callable,
    Mapping,
    Protocol,
    Sequence as TypingSequence,
)


# ============================================================
# EXCEPTIONS
# ============================================================

class BehaviorTreeError(Exception):
    """Base behavior tree exception."""


class BehaviorTreeValidationError(BehaviorTreeError):
    """Invalid node or tree configuration."""


class BehaviorTreeExecutionError(BehaviorTreeError):
    """Behavior tree execution failed."""


class BehaviorTreeAuthorizationError(BehaviorTreeError):
    """Unauthorized behavior tree operation."""


class BlackboardError(BehaviorTreeError):
    """Blackboard operation failed."""


class NodeNotFoundError(BehaviorTreeError):
    """Requested behavior tree node does not exist."""


class BehaviorTreeSafetyError(BehaviorTreeError):
    """Safety requirements are not satisfied."""


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


def json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict
        return json_safe(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set, frozenset, deque)):
        return [json_safe(item) for item in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return repr(value)


def validate_identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BehaviorTreeValidationError(
            f"{label} must be a nonempty string"
        )

    if len(value) > 128:
        raise BehaviorTreeValidationError(
            f"{label} exceeds maximum length"
        )

    return value


# ============================================================
# ENUMS
# ============================================================

class NodeStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


class NodeCategory(str, Enum):
    COMPOSITE = "composite"
    DECORATOR = "decorator"
    CONDITION = "condition"
    ACTION = "action"


class ParallelPolicy(str, Enum):
    REQUIRE_ALL = "require_all"
    REQUIRE_ONE = "require_one"


class RobotSafetyState(str, Enum):
    SAFE = "safe"
    CAUTION = "caution"
    UNSAFE = "unsafe"
    EMERGENCY_STOP = "emergency_stop"
    UNKNOWN = "unknown"


class RobotOperationalState(str, Enum):
    AVAILABLE = "available"
    IDLE = "idle"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    FAULTED = "faulted"


class ExecutionMode(str, Enum):
    SIMULATION = "simulation"
    DRY_RUN = "dry_run"


class ActionExecutionState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class BehaviorTreeConfig:
    execution_mode: ExecutionMode = ExecutionMode.SIMULATION

    maximum_nodes: int = 1000
    maximum_depth: int = 50

    maximum_ticks: int = 100_000

    default_tick_interval_seconds: float = 0.1

    maximum_node_execution_seconds: float = 60.0

    maximum_retry_count: int = 10

    maximum_repeat_count: int = 1000

    robot_state_max_age_seconds: float = 10.0

    require_safe_robot_state: bool = True

    require_human_approval_for_motion: bool = True

    enable_execution_trace: bool = True

    maximum_trace_events: int = 20_000

    def __post_init__(self) -> None:
        for name in (
            "maximum_nodes",
            "maximum_depth",
            "maximum_ticks",
            "maximum_retry_count",
            "maximum_repeat_count",
            "maximum_trace_events",
        ):
            value = getattr(self, name)

            if not isinstance(value, int) or value <= 0:
                raise BehaviorTreeValidationError(
                    f"{name} must be a positive integer"
                )

        for name in (
            "default_tick_interval_seconds",
            "maximum_node_execution_seconds",
            "robot_state_max_age_seconds",
        ):
            value = getattr(self, name)

            if not math.isfinite(value) or value <= 0:
                raise BehaviorTreeValidationError(
                    f"{name} must be positive and finite"
                )


# ============================================================
# AUTHORIZATION
# ============================================================

@dataclass(frozen=True)
class BehaviorTreeAccessContext:
    tenant_id: str
    user_id: str

    allowed_robot_ids: frozenset[str] = frozenset()

    roles: frozenset[str] = frozenset()

    can_execute_simulation: bool = False
    can_modify_tree: bool = False
    can_approve_actions: bool = False

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id, "tenant_id")
        validate_identifier(self.user_id, "user_id")


# ============================================================
# ROBOT CONTEXT
# ============================================================

@dataclass(frozen=True)
class RobotBehaviorState:
    tenant_id: str
    robot_id: str

    safety_state: RobotSafetyState = RobotSafetyState.UNKNOWN

    operational_state: RobotOperationalState = (
        RobotOperationalState.OFFLINE
    )

    capabilities: frozenset[str] = frozenset()

    battery_percentage: float | None = None

    active_faults: tuple[str, ...] = ()

    emergency_stop_active: bool = False

    last_updated: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if self.last_updated.tzinfo is None:
            raise BehaviorTreeValidationError(
                "Robot timestamp must be timezone-aware"
            )

        if self.battery_percentage is not None:
            if not 0 <= self.battery_percentage <= 100:
                raise BehaviorTreeValidationError(
                    "Battery percentage must be between 0 and 100"
                )


# ============================================================
# BLACKBOARD
# ============================================================

class Blackboard:
    """
    Thread-safe in-memory blackboard.

    All values are isolated to a single behavior tree
    execution context.

    Do not store authentication credentials or actuator
    authorization tokens in the blackboard.
    """

    def __init__(
        self,
        initial_values: Mapping[str, Any] | None = None,
    ) -> None:
        self._data: dict[str, Any] = {}

        self._lock = threading.RLock()

        if initial_values:
            for key, value in initial_values.items():
                self.set(key, value)

    def set(self, key: str, value: Any) -> None:
        validate_identifier(key, "blackboard key")

        with self._lock:
            self._data[key] = copy.deepcopy(value)

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        with self._lock:
            return copy.deepcopy(
                self._data.get(key, default)
            )

    def require(self, key: str) -> Any:
        with self._lock:
            if key not in self._data:
                raise BlackboardError(
                    f"Blackboard key not found: {key}"
                )

            return copy.deepcopy(self._data[key])

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def delete(self, key: str) -> bool:
        with self._lock:
            if key not in self._data:
                return False

            del self._data[key]
            return True

    def update(
        self,
        values: Mapping[str, Any],
    ) -> None:
        for key, value in values.items():
            self.set(key, value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


# ============================================================
# EXECUTION TRACE
# ============================================================

@dataclass(frozen=True)
class NodeTraceEvent:
    event_id: str
    timestamp: str

    tree_id: str
    node_id: str
    node_name: str

    previous_status: NodeStatus
    current_status: NodeStatus

    tick_number: int

    message: str = ""


class BehaviorTreeTraceLogger:

    def __init__(
        self,
        maximum_events: int = 20_000,
    ) -> None:
        self._events: deque[NodeTraceEvent] = deque(
            maxlen=maximum_events
        )

        self._lock = threading.RLock()

    def record(
        self,
        event: NodeTraceEvent,
    ) -> None:
        with self._lock:
            self._events.append(event)

    def history(
        self,
        tree_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            events = [
                json_safe(event)
                for event in self._events
                if event.tree_id == tree_id
            ]

        return events[-limit:]


# ============================================================
# EXECUTION CONTEXT
# ============================================================

@dataclass
class BehaviorTreeContext:
    tree_id: str

    access: BehaviorTreeAccessContext

    robot: RobotBehaviorState

    blackboard: Blackboard

    config: BehaviorTreeConfig

    trace_logger: BehaviorTreeTraceLogger

    tick_number: int = 0

    simulation_results: dict[
        str, dict[str, Any]
    ] = field(default_factory=dict)

    approvals: frozenset[str] = frozenset()

    cancelled: bool = False

    monotonic_clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        if self.access.tenant_id != self.robot.tenant_id:
            raise BehaviorTreeAuthorizationError(
                "Tenant mismatch between access and robot"
            )

        if self.robot.robot_id not in self.access.allowed_robot_ids:
            raise BehaviorTreeAuthorizationError(
                "Robot is not authorized"
            )

    def record_transition(
        self,
        node: "BehaviorTreeNode",
        previous: NodeStatus,
        current: NodeStatus,
        message: str = "",
    ) -> None:
        if not self.config.enable_execution_trace:
            return

        self.trace_logger.record(
            NodeTraceEvent(
                event_id=new_id(),
                timestamp=utc_iso(),
                tree_id=self.tree_id,
                node_id=node.node_id,
                node_name=node.name,
                previous_status=previous,
                current_status=current,
                tick_number=self.tick_number,
                message=message,
            )
        )


# ============================================================
# BASE NODE
# ============================================================

class BehaviorTreeNode:
    """
    Base class for all behavior tree nodes.
    """

    category: NodeCategory = NodeCategory.ACTION

    def __init__(
        self,
        name: str,
        *,
        node_id: str | None = None,
    ) -> None:
        self.name = validate_identifier(
            name,
            "node name",
        )

        self.node_id = node_id or new_id()

        validate_identifier(
            self.node_id,
            "node_id",
        )

        self.status = NodeStatus.IDLE

        self.tick_count = 0

        self.last_error: str | None = None

        self._lock = threading.RLock()

    def tick(
        self,
        context: BehaviorTreeContext,
    ) -> NodeStatus:
        with self._lock:
            previous = self.status

            if context.cancelled:
                self.halt(context)
                return NodeStatus.FAILURE

            self.tick_count += 1

            try:
                result = self._tick(context)

                if not isinstance(result, NodeStatus):
                    raise BehaviorTreeExecutionError(
                        f"Node {self.name} returned invalid status"
                    )

            except Exception as exc:
                self.last_error = str(exc)
                result = NodeStatus.FAILURE

            self.status = result

            if previous != result:
                context.record_transition(
                    self,
                    previous,
                    result,
                    self.last_error or "",
                )

            return result

    def _tick(
        self,
        context: BehaviorTreeContext,
    ) -> NodeStatus:
        raise NotImplementedError

    def halt(
        self,
        context: BehaviorTreeContext | None = None,
    ) -> None:
        with self._lock:
            self.status = NodeStatus.IDLE

    def reset(self) -> None:
        with self._lock:
            self.status = NodeStatus.IDLE
            self.tick_count = 0
            self.last_error = None

    def children(self) -> tuple["BehaviorTreeNode", ...]:
        return ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "category": self.category.value,
            "node_type": type(self).__name__,
            "status": self.status.value,
            "tick_count": self.tick_count,
            "last_error": self.last_error,
            "children": [
                child.to_dict()
                for child in self.children()
            ],
        }


# ============================================================
# COMPOSITE NODE
# ============================================================

class CompositeNode(BehaviorTreeNode):
    category = NodeCategory.COMPOSITE

    def __init__(
        self,
        name: str,
        children: TypingSequence[BehaviorTreeNode],
        *,
        node_id: str | None = None,
    ) -> None:
        super().__init__(
            name,
            node_id=node_id,
        )

        if not children:
            raise BehaviorTreeValidationError(
                "Composite node requires children"
            )

        self._children = tuple(children)

    def children(self) -> tuple[BehaviorTreeNode, ...]:
        return self._children

    def halt(
        self,
        context: BehaviorTreeContext | None = None,
    ) -> None:
        for child in self._children:
            child.halt(context)

        super().halt(context)

    def reset(self) -> None:
        for child in self._children:
            child.reset()

        super().reset()


# ============================================================
# SEQUENCE NODE
# ============================================================

class SequenceNode(CompositeNode):
    """
    Runs children in order.

    SUCCESS:
        Every child succeeds.

    FAILURE:
        Any child fails.

    RUNNING:
        A child is still running.
    """

    def __init__(
        self,
        name: str,
        children: TypingSequence[BehaviorTreeNode],
        *,
        node_id: str | None = None,
    ) -> None:
        super().__init__(
            name,
            children,
            node_id=node_id,
        )

        self._current_index = 0

    def _tick(
        self,
        context: BehaviorTreeContext,
    ) -> NodeStatus:
        while self._current_index < len(self._children):
            child = self._children[
                self._current_index
            ]

            result = child.tick(context)

            if result == NodeStatus.RUNNING:
                return NodeStatus.RUNNING

            if result == NodeStatus.FAILURE:
                self._current_index = 0
                return NodeStatus.FAILURE

            self._current_index += 1

        self._current_index = 0
        return NodeStatus.SUCCESS

    def halt(
        self,
        context: BehaviorTreeContext | None = None,
    ) -> None:
        self._current_index = 0
        super().halt(context)

    def reset(self) -> None:
        self._current_index = 0
        super().reset()


# ============================================================
# SELECTOR NODE
# ============================================================

class SelectorNode(CompositeNode):
    """
    Attempts children in order until one succeeds.
    """

    def __init__(
        self,
        name: str,
        children: TypingSequence[BehaviorTreeNode],
        *,
        node_id: str | None = None,
    ) -> None:
        super().__init__(
            name,
            children,
            node_id=node_id,
        )

        self._current_index = 0

    def _tick(
        self,
        context: BehaviorTreeContext,
    ) -> NodeStatus:
        while self._current_index < len(self._children):
            child = self._children[
                self._current_index
            ]

            result = child.tick(context)

            if result == NodeStatus.RUNNING:
                return NodeStatus.RUNNING

            if result == NodeStatus.SUCCESS:
                self._current_index = 0
                return NodeStatus.SUCCESS

            self._current_index += 1

        self._current_index = 0
        return NodeStatus.FAILURE

    def halt(
        self,
        context: BehaviorTreeContext | None = None,
    ) -> None:
        self._current_index = 0
        super().halt(context)

    def reset(self) -> None:
        self._current_index = 0
        super().reset()


# ============================================================
# REACTIVE SEQUENCE
# ============================================================

class ReactiveSequenceNode(CompositeNode):
    """
    Rechecks conditions on every tick.

    Suitable for continuous safety validation.
    """

    def _tick(
        self,
        context: BehaviorTreeContext,
    ) -> NodeStatus:
        for index, child in enumerate(self._children):
            result = child.tick(context)

            if result != NodeStatus.SUCCESS:
                for later in self._children[index + 1:]:
                    if later.status == NodeStatus.RUNNING:
                        later.halt(context)

                return result

        return NodeStatus.SUCCESS


# ============================================================
# REACTIVE SELECTOR
# ============================================================

class ReactiveSelectorNode(CompositeNode):
    """
    Rechecks higher-priority branches on every tick.
    """

    def _tick(
        self,
        context: BehaviorTreeContext,
    ) -> NodeStatus:
        for index, child in enumerate(self._children):
            result = child.tick(context)

            if result != NodeStatus.FAILURE:
                for later in self._children[index + 1:]:
                    if later.status == NodeStatus.RUNNING:
                        later.halt(context)

                return result

        return NodeStatus.FAILURE


# ============================================================
# PARALLEL NODE
# ============================================================

class ParallelNode(CompositeNode):
    """
    Logically ticks multiple child nodes sequentially.

    This is not multithreaded hardware execution.
    """

    def __init__(
        self,
        name: str,
        children: TypingSequence[BehaviorTreeNode],
        *,
        policy: ParallelPolicy = ParallelPolicy.REQUIRE_ALL,
        node_id: str | None = None,
    ) -> None:
        super().__init__(
            name,
            children,
            node_id=node_id,
        )

        self.policy = policy

    def _tick(
        self,
        context: BehaviorTreeContext,
    ) -> NodeStatus:
        results = [
            child.tick(context)
            for child in self._children
        ]

        if self.policy == ParallelPolicy.REQUIRE_ALL:
            if NodeStatus.FAILURE in results:
                return NodeStatus.FAILURE

            if all(
                result == NodeStatus.SUCCESS
                for result in results
            ):
                return NodeStatus.SUCCESS

            return NodeStatus.RUNNING

        if NodeStatus.SUCCESS in results:
            return NodeStatus.SUCCESS

        if all(
            result == NodeStatus.FAILURE
            for result in results
        ):
            return NodeStatus.FAILURE

        return NodeStatus.RUNNING


# ============================================================
# DECORATOR BASE
# ============================================================

class DecoratorNode(BehaviorTreeNode):
    category = NodeCategory.DECORATOR

    def __init__(
        self,
        name: str,
        child: BehaviorTreeNode,
        *,
        node_id: str | None = None,
    ) -> None:
        super().__init__(
            name,
            node_id=node_id,
        )

        self.child = child

    def children(self) -> tuple[BehaviorTreeNode, ...]:
        return (self.child,)

    def halt(
        self,
        context: BehaviorTreeContext | None = None,
    ) -> None:
        self.child.halt(context)
        super().halt(context)

    def reset(self) -> None:
        self.child.reset()
        super().reset()


# ============================================================
# INVERTER
# ============================================================

class InverterNode(DecoratorNode):

    def _tick(
        self,
        context: BehaviorTreeContext,
    ) -> NodeStatus:
        result = self.child.tick(context)

        if result == NodeStatus.SUCCESS:
            return NodeStatus.FAILURE

        if result == NodeStatus.FAILURE:
            return NodeStatus.SUCCESS

        return result


# ============================================================
# FORCE SUCCESS / FAILURE
# ============================================================

class ForceSuccessNode(DecoratorNode):

    def _tick(
        self,
        context: BehaviorTreeContext,
    ) -> NodeStatus:
        result = self.child.tick(context)

        if result == NodeStatus.RUNNING:
            return NodeStatus.RUNNING

        return NodeStatus.SUCCESS


class ForceFailureNode(DecoratorNode):

    def _tick(
        self,
        context: BehaviorTreeContext,
    ) -> NodeStatus:
        result = self.child.tick(context)

        if result == NodeStatus.RUNNING:
            return NodeStatus.RUNNING

        return NodeStatus.FAILURE


# ============================================================
# RETRY DECORATOR
# ============================================================

class RetryNode(DecoratorNode):

    def __init__(
        self,
        name: str,
        child: BehaviorTreeNode,
        *,
        maximum_attempts: int = 3,
        node_id: str | None = None,
    ) -> None:
        super().__init__(
            name,
            child,
            node_id=node_id,
        )

        if maximum_attempts < 1:
            raise BehaviorTreeValidationError(
                "Maximum attempts must be positive"
            )

        self.maximum_attempts = maximum_attempts
        self._attempts = 0

    def _tick(
        self,
        context: BehaviorTreeContext,
    ) -> NodeStatus:
        if (
            self.maximum_attempts
            > context.config.maximum_retry_count
        ):
            return NodeStatus.FAILURE

        result = self.child.tick(context)

        if result == NodeStatus.SUCCESS:
            self._attempts = 0
            return NodeStatus.SUCCESS

        if result == NodeStatus.RUNNING:
            return NodeStatus.RUNNING

        self._attempts += 1

        if self._attempts >= self.maximum_attempts:
            self._attempts = 0
            return NodeStatus.FAILURE

        self.child.reset()
        return NodeStatus.RUNNING

    def halt(
        self,
        context: BehaviorTreeContext | None = None,
    ) -> None:
        self._attempts = 0
        super().halt(context)

    def reset(self) -> None:
        self._attempts = 0
        super().reset()


# ============================================================
# REPEAT DECORATOR
# ============================================================

class RepeatNode(DecoratorNode):

    def __init__(
        self,
        name: str,
        child: BehaviorTreeNode,
        *,
        repetitions: int,
        node_id: str | None = None,
    ) -> None:
        super().__init__(
            name,
            child,
            node_id=node_id,
        )

        if repetitions < 1:
            raise BehaviorTreeValidationError(
                "Repetitions must be positive"
            )

        self.repetitions = repetitions
        self._completed = 0

    def _tick(
        self,
        context: BehaviorTreeContext,
    ) -> NodeStatus:
        if (
            self.repetitions
            > context.config.maximum_repeat_count
        ):
            return NodeStatus.FAILURE

        result = self.child.tick(context)

        if result == NodeStatus.FAILURE:
            self._completed = 0
            return NodeStatus.FAILURE

        if result == NodeStatus.RUNNING:
            return NodeStatus.RUNNING

        self._completed += 1

        if self._completed >= self.repetitions:
            self._completed = 0
            return NodeStatus.SUCCESS

        self.child.reset()
        return NodeStatus.RUNNING

    def halt(
        self,
        context: BehaviorTreeContext | None = None,
    ) -> None:
        self._completed = 0
        super().halt(context)

    def reset(self) -> None:
        self._completed = 0
        super().reset()


# ============================================================
# TIMEOUT DECORATOR
# ============================================================

class TimeoutNode(DecoratorNode):

    def __init__(
        self,
        name: str,
        child: BehaviorTreeNode,
        *,
        timeout_seconds: float,
        node_id: str | None = None,
    ) -> None:
        super().__init__(
            name,
            child,
            node_id=node_id,
        )

        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise BehaviorTreeValidationError(
                "Timeout must be positive and finite"
            )

        self.timeout_seconds = timeout_seconds
        self._started_at: float | None = None

    def _tick(
        self,
        context: BehaviorTreeContext,
    ) -> NodeStatus:
        now = context.monotonic_clock()

        if self._started_at is None:
            self._started_at = now

        if (
            now - self._started_at
            >= self.timeout_seconds
        ):
            self.child.halt(context)
            self._started_at = None
            return NodeStatus.FAILURE

        result = self.child.tick(context)

        if result != NodeStatus.RUNNING:
            self._started_at = None

        return result

    def halt(
        self,
        context: BehaviorTreeContext | None = None,
    ) -> None:
        self._started_at = None
        super().halt(context)

    def reset(self) -> None:
        self._started_at = None
        super().reset()


# ============================================================
# CONDITION NODE
# ============================================================

ConditionCallback = Callable[
    [BehaviorTreeContext],
    bool,
]


class ConditionNode(BehaviorTreeNode):
    category = NodeCategory.CONDITION

    def __init__(
        self,
        name: str,
        predicate: ConditionCallback,
        *,
        node_id: str | None = None,
    ) -> None:
        super().__init__(
            name,
            node_id=node_id,
        )

        if not callable(predicate):
            raise BehaviorTreeValidationError(
                "Condition predicate must be callable"
            )

        self.predicate = predicate

    def _tick(
        self,
        context: BehaviorTreeContext,
    ) -> NodeStatus:
        return (
            NodeStatus.SUCCESS
            if self.predicate(context)
            else NodeStatus.FAILURE
        )


# ============================================================
# BLACKBOARD CONDITION NODES
# ============================================================

class BlackboardExistsNode(BehaviorTreeNode):
    category = NodeCategory.CONDITION

    def __init__(
        self,
        name: str,
        key: str,
    ) -> None:
        super().__init__(name)
        self.key = key

    def _tick(
        self,
        context: BehaviorTreeContext,
    ) -> NodeStatus:
        return (
            NodeStatus.SUCCESS
            if context.blackboard.contains(self.key)
            else NodeStatus.FAILURE
        )


class BlackboardEqualsNode(BehaviorTreeNode):
    category = NodeCategory.CONDITION

    def __init__(
        self,
        name: str,
        key: str,
        expected: Any,
    ) -> None:
        super().__init__(name)

        self.key = key
        self.expected = expected

    def _tick(
        self,
        context: BehaviorTreeContext,
    ) -> NodeStatus:
        actual = context.blackboard.get(
            self.key
        )

        return (
            NodeStatus.SUCCESS
            if actual == self.expected
            else NodeStatus.FAILURE
        )


class BlackboardThresholdNode(BehaviorTreeNode):
    category = NodeCategory.CONDITION

    def __init__(
        self,
        name: str,
        key: str,
        threshold: float,
        *,
        comparison: str = ">=",
    ) -> None:
        super().__init__(name)

        if comparison not in {
            ">", ">=", "<", "<=", "==", "!=",
        }:
            raise BehaviorTreeValidationError(
                "Unsupported comparison operator"
            )

        self.key = key
        self.threshold = threshold
        self.comparison = comparison

    def _tick(
        self,
        context: BehaviorTreeContext,
    ) -> NodeStatus:
        value = context.blackboard.get(
            self.key
        )

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            return NodeStatus.FAILURE

        operations = {
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }

        result = operations[self.comparison](
            value,
            self.threshold,
        )

        return (
            NodeStatus.SUCCESS
            if result
            else NodeStatus.FAILURE
        )


# ============================================================
# ROBOT SAFETY CONDITION
# ============================================================

class RobotSafetyConditionNode(BehaviorTreeNode):
    category = NodeCategory.CONDITION

    def __init__(
        self,
        name: str = "Robot Safety Check",
        *,
        require_safe: bool = True,
    ) -> None:
        super().__init__(name)

        self.require_safe = require_safe

    def _tick(
        self,
        context: BehaviorTreeContext,
    ) -> NodeStatus:
        robot = context.robot

        if robot.emergency_stop_active:
            return NodeStatus.FAILURE

        if robot.active_faults:
            return NodeStatus.FAILURE

        if robot.safety_state in {
            RobotSafetyState.UNSAFE,
            RobotSafetyState.EMERGENCY_STOP,
            RobotSafetyState.UNKNOWN,
        }:
            return NodeStatus.FAILURE

        if (
            self.require_safe
            and robot.safety_state != RobotSafetyState.SAFE
        ):
            return NodeStatus.FAILURE

        age = (
            utc_now() - robot.last_updated
        ).total_seconds()

        if not (
            0 <= age
            <= context.config.robot_state_max_age_seconds
        ):
            return NodeStatus.FAILURE

        if robot.operational_state in {
            RobotOperationalState.OFFLINE,
            RobotOperationalState.FAULTED,
            RobotOperationalState.MAINTENANCE,
        }:
            return NodeStatus.FAILURE

        return NodeStatus.SUCCESS


# ============================================================
# ROBOT BATTERY CONDITION
# ============================================================

class RobotBatteryConditionNode(BehaviorTreeNode):
    category = NodeCategory.CONDITION

    def __init__(
        self,
        name: str,
        *,
        minimum_percentage: float = 20.0,
    ) -> None:
        super().__init__(name)

        if not 0 <= minimum_percentage <= 100:
            raise BehaviorTreeValidationError(
                "Battery threshold must be between 0 and 100"
            )

        self.minimum_percentage = minimum_percentage

    def _tick(
        self,
        context: BehaviorTreeContext,
    ) -> NodeStatus:
        battery = context.robot.battery_percentage

        if battery is None:
            return NodeStatus.FAILURE

        return (
            NodeStatus.SUCCESS
            if battery >= self.minimum_percentage
            else NodeStatus.FAILURE
        )


# ============================================================
# ROBOT CAPABILITY CONDITION
# ============================================================

class RobotCapabilityConditionNode(BehaviorTreeNode):
    category = NodeCategory.CONDITION

    def __init__(
        self,
        name: str,
        *,
        required_capabilities: TypingSequence[str],
    ) -> None:
        super().__init__(name)

        self.required_capabilities = frozenset(
            required_capabilities
        )

    def _tick(
        self,
        context: BehaviorTreeContext,
    ) -> NodeStatus:
        supported = (
            self.required_capabilities
            .issubset(context.robot.capabilities)
        )

        return (
            NodeStatus.SUCCESS
            if supported
            else NodeStatus.FAILURE
        )


# ============================================================
# HUMAN APPROVAL CONDITION
# ============================================================

class HumanApprovalConditionNode(BehaviorTreeNode):
    category = NodeCategory.CONDITION

    def __init__(
        self,
        name: str,
        *,
        approval_id: str,
    ) -> None:
        super().__init__(name)

        self.approval_id = validate_identifier(
            approval_id,
            "approval_id",
        )

    def _tick(
        self,
        context: BehaviorTreeContext,
    ) -> NodeStatus:
        approved = (
            self.approval_id
            in context.approvals
        )

        return (
            NodeStatus.SUCCESS
            if approved
            else NodeStatus.FAILURE
        )


# ============================================================
# ACTION NODE
# ============================================================

ActionCallback = Callable[
    [BehaviorTreeContext],
    NodeStatus | bool,
]


class ActionNode(BehaviorTreeNode):
    """
    Executes a local simulation callback.

    Callbacks must not perform real robot actuation.
    """

    category = NodeCategory.ACTION

    def __init__(
        self,
        name: str,
        callback: ActionCallback,
        *,
        node_id: str | None = None,
    ) -> None:
        super().__init__(
            name,
            node_id=node_id,
        )

        if not callable(callback):
            raise BehaviorTreeValidationError(
                "Action callback must be callable"
            )

        self.callback = callback

    def _tick(
        self,
        context: BehaviorTreeContext,
    ) -> NodeStatus:
        if not context.access.can_execute_simulation:
            raise BehaviorTreeAuthorizationError(
                "Simulation permission required"
            )

        result = self.callback(context)

        if isinstance(result, NodeStatus):
            return result

        if isinstance(result, bool):
            return (
                NodeStatus.SUCCESS
                if result
                else NodeStatus.FAILURE
            )

        raise BehaviorTreeExecutionError(
            "Action callback must return bool or NodeStatus"
        )


# ============================================================
# BLACKBOARD WRITE ACTION
# ============================================================

class BlackboardSetNode(BehaviorTreeNode):
    category = NodeCategory.ACTION

    def __init__(
        self,
        name: str,
        *,
        key: str,
        value: Any,
    ) -> None:
        super().__init__(name)

        self.key = key
        self.value = value

    def _tick(
        self,
        context: BehaviorTreeContext,
    ) -> NodeStatus:
        context.blackboard.set(
            self.key,
            self.value,
        )

        return NodeStatus.SUCCESS


# ============================================================
# WAIT NODE
# ============================================================

class WaitNode(BehaviorTreeNode):
    category = NodeCategory.ACTION

    def __init__(
        self,
        name: str,
        *,
        duration_seconds: float,
    ) -> None:
        super().__init__(name)

        if not math.isfinite(duration_seconds) or duration_seconds < 0:
            raise BehaviorTreeValidationError(
                "Wait duration must be finite and nonnegative"
            )

        self.duration_seconds = duration_seconds
        self._started_at: float | None = None

    def _tick(
        self,
        context: BehaviorTreeContext,
    ) -> NodeStatus:
        now = context.monotonic_clock()

        if self._started_at is None:
            self._started_at = now

        if (
            now - self._started_at
            >= self.duration_seconds
        ):
            self._started_at = None
            return NodeStatus.SUCCESS

        return NodeStatus.RUNNING

    def halt(
        self,
        context: BehaviorTreeContext | None = None,
    ) -> None:
        self._started_at = None
        super().halt(context)

    def reset(self) -> None:
        self._started_at = None
        super().reset()


# ============================================================
# SIMULATED ROBOT ACTION
# ============================================================

class SimulatedRobotActionNode(BehaviorTreeNode):
    """
    Represents a robot action without controlling hardware.

    The simulation result is stored on the blackboard.
    """

    category = NodeCategory.ACTION

    MOTION_ACTIONS = frozenset({
        "navigate",
        "move",
        "pick",
        "place",
        "manipulate",
        "rotate",
        "drive",
    })

    def __init__(
        self,
        name: str,
        *,
        action_name: str,
        parameters: Mapping[str, Any] | None = None,
        required_capabilities: TypingSequence[str] = (),
        approval_id: str | None = None,
        result_key: str | None = None,
    ) -> None:
        super().__init__(name)

        self.action_name = action_name

        self.parameters = dict(
            parameters or {}
        )

        self.required_capabilities = frozenset(
            required_capabilities
        )

        self.approval_id = approval_id

        self.result_key = (
            result_key
            or f"simulation_{self.node_id}"
        )

    def _tick(
        self,
        context: BehaviorTreeContext,
    ) -> NodeStatus:
        if not context.access.can_execute_simulation:
            return NodeStatus.FAILURE

        robot = context.robot

        if robot.emergency_stop_active:
            return NodeStatus.FAILURE

        if robot.safety_state != RobotSafetyState.SAFE:
            return NodeStatus.FAILURE

        if robot.active_faults:
            return NodeStatus.FAILURE

        age = (
            utc_now() - robot.last_updated
        ).total_seconds()

        if not (
            0 <= age
            <= context.config.robot_state_max_age_seconds
        ):
            return NodeStatus.FAILURE

        if not self.required_capabilities.issubset(
            robot.capabilities
        ):
            return NodeStatus.FAILURE

        is_motion = (
            self.action_name.lower()
            in self.MOTION_ACTIONS
        )

        if (
            is_motion
            and context.config.require_human_approval_for_motion
        ):
            if (
                not self.approval_id
                or self.approval_id not in context.approvals
            ):
                return NodeStatus.FAILURE

        result = {
            "action_name": self.action_name,
            "robot_id": robot.robot_id,
            "parameters": copy.deepcopy(
                self.parameters
            ),
            "mode": context.config.execution_mode.value,
            "simulated": True,
            "hardware_execution": False,
            "timestamp": utc_iso(),
            "status": "simulated_success",
        }

        context.blackboard.set(
            self.result_key,
            result,
        )

        context.simulation_results[
            self.node_id
        ] = result

        return NodeStatus.SUCCESS


# ============================================================
# SIMULATED INSPECTION NODE
# ============================================================

class RobotInspectionNode(SimulatedRobotActionNode):

    def __init__(
        self,
        name: str = "Robot Inspection",
        *,
        inspection_type: str = "visual",
        target_component: str = "robot",
        result_key: str = "inspection_result",
    ) -> None:
        super().__init__(
            name,
            action_name="inspection",
            parameters={
                "inspection_type": inspection_type,
                "target_component": target_component,
            },
            required_capabilities=("inspection",),
            result_key=result_key,
        )


# ============================================================
# SIMULATED DIAGNOSTICS NODE
# ============================================================

class RobotDiagnosticsNode(SimulatedRobotActionNode):

    def __init__(
        self,
        name: str = "Robot Diagnostics",
        *,
        subsystem: str = "general",
        result_key: str = "diagnostics_result",
    ) -> None:
        super().__init__(
            name,
            action_name="diagnostics",
            parameters={
                "subsystem": subsystem,
            },
            required_capabilities=("diagnostics",),
            result_key=result_key,
        )


# ============================================================
# SIMULATED NAVIGATION NODE
# ============================================================

class RobotNavigationNode(SimulatedRobotActionNode):

    def __init__(
        self,
        name: str = "Robot Navigation",
        *,
        destination: str,
        approval_id: str,
        result_key: str = "navigation_result",
    ) -> None:
        super().__init__(
            name,
            action_name="navigate",
            parameters={
                "destination": destination,
            },
            required_capabilities=("navigation",),
            approval_id=approval_id,
            result_key=result_key,
        )


# ============================================================
# TREE VALIDATOR
# ============================================================

class BehaviorTreeValidator:

    def __init__(
        self,
        config: BehaviorTreeConfig,
    ) -> None:
        self.config = config

    def validate(
        self,
        root: BehaviorTreeNode,
    ) -> dict[str, int]:
        visited_objects: set[int] = set()
        node_ids: set[str] = set()

        maximum_depth = 0
        node_count = 0

        def walk(
            node: BehaviorTreeNode,
            depth: int,
        ) -> None:
            nonlocal maximum_depth
            nonlocal node_count

            if depth > self.config.maximum_depth:
                raise BehaviorTreeValidationError(
                    "Behavior tree depth exceeded"
                )

            identity = id(node)

            if identity in visited_objects:
                raise BehaviorTreeValidationError(
                    "Shared or circular node reference detected"
                )

            if node.node_id in node_ids:
                raise BehaviorTreeValidationError(
                    "Duplicate node ID detected"
                )

            visited_objects.add(identity)
            node_ids.add(node.node_id)

            node_count += 1

            maximum_depth = max(
                maximum_depth,
                depth,
            )

            if node_count > self.config.maximum_nodes:
                raise BehaviorTreeValidationError(
                    "Maximum node count exceeded"
                )

            for child in node.children():
                walk(
                    child,
                    depth + 1,
                )

        walk(root, 1)

        return {
            "node_count": node_count,
            "maximum_depth": maximum_depth,
        }


# ============================================================
# TREE VISUALIZER
# ============================================================

class BehaviorTreeVisualizer:

    @staticmethod
    def to_text(
        root: BehaviorTreeNode,
    ) -> str:
        lines = []

        def walk(
            node: BehaviorTreeNode,
            prefix: str,
            is_last: bool,
        ) -> None:
            connector = (
                "└── "
                if is_last
                else "├── "
            )

            lines.append(
                f"{prefix}{connector}"
                f"{node.name} "
                f"[{type(node).__name__}] "
                f"({node.status.value})"
            )

            children = node.children()

            next_prefix = (
                prefix
                + ("    " if is_last else "│   ")
            )

            for index, child in enumerate(children):
                walk(
                    child,
                    next_prefix,
                    index == len(children) - 1,
                )

        walk(root, "", True)

        return "\n".join(lines)


# ============================================================
# BEHAVIOR TREE ENGINE
# ============================================================

@dataclass(frozen=True)
class BehaviorTreeTickResult:
    tree_id: str
    status: NodeStatus
    tick_number: int
    execution_time_ms: float
    timestamp: str


class RoboticsBehaviorTreeEngine:

    def __init__(
        self,
        *,
        root: BehaviorTreeNode,
        access: BehaviorTreeAccessContext,
        robot: RobotBehaviorState,
        config: BehaviorTreeConfig | None = None,
        blackboard: Blackboard | None = None,
        tree_id: str | None = None,
    ) -> None:
        self.config = (
            config
            or BehaviorTreeConfig()
        )

        self.tree_id = (
            tree_id
            or new_id()
        )

        self.root = root

        self.validator = BehaviorTreeValidator(
            self.config
        )

        self.validation_report = self.validator.validate(
            root
        )

        self.trace_logger = BehaviorTreeTraceLogger(
            self.config.maximum_trace_events
        )

        self.context = BehaviorTreeContext(
            tree_id=self.tree_id,
            access=access,
            robot=robot,
            blackboard=(
                blackboard
                or Blackboard()
            ),
            config=self.config,
            trace_logger=self.trace_logger,
        )

        self._lock = threading.RLock()

        self._completed = False

    # --------------------------------------------------------
    # TICK
    # --------------------------------------------------------

    def tick(self) -> BehaviorTreeTickResult:
        with self._lock:
            if not self.context.access.can_execute_simulation:
                raise BehaviorTreeAuthorizationError(
                    "Simulation permission required"
                )

            if (
                self.context.tick_number
                >= self.config.maximum_ticks
            ):
                raise BehaviorTreeExecutionError(
                    "Maximum tick count exceeded"
                )

            started = time.perf_counter()

            self.context.tick_number += 1

            status = self.root.tick(
                self.context
            )

            if status in (
                NodeStatus.SUCCESS,
                NodeStatus.FAILURE,
            ):
                self._completed = True

            return BehaviorTreeTickResult(
                tree_id=self.tree_id,
                status=status,
                tick_number=self.context.tick_number,
                execution_time_ms=round(
                    (
                        time.perf_counter()
                        - started
                    ) * 1000,
                    3,
                ),
                timestamp=utc_iso(),
            )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    def run(
        self,
        *,
        maximum_ticks: int | None = None,
        sleep_between_ticks: bool = False,
    ) -> BehaviorTreeTickResult:
        limit = (
            maximum_ticks
            or self.config.maximum_ticks
        )

        if limit < 1:
            raise BehaviorTreeValidationError(
                "Maximum ticks must be positive"
            )

        result: BehaviorTreeTickResult | None = None

        for _ in range(limit):
            result = self.tick()

            if result.status in (
                NodeStatus.SUCCESS,
                NodeStatus.FAILURE,
            ):
                return result

            if sleep_between_ticks:
                time.sleep(
                    self.config.default_tick_interval_seconds
                )

        assert result is not None
        return result

    # --------------------------------------------------------
    # CANCEL
    # --------------------------------------------------------

    def cancel(self) -> None:
        with self._lock:
            self.context.cancelled = True
            self.root.halt(self.context)

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(self) -> None:
        with self._lock:
            self.root.reset()

            self.context.cancelled = False
            self.context.tick_number = 0
            self.context.simulation_results.clear()

            self._completed = False

    # --------------------------------------------------------
    # UPDATE ROBOT STATE
    # --------------------------------------------------------

    def update_robot_state(
        self,
        robot: RobotBehaviorState,
    ) -> None:
        with self._lock:
            if (
                robot.tenant_id
                != self.context.access.tenant_id
                or robot.robot_id
                != self.context.robot.robot_id
            ):
                raise BehaviorTreeAuthorizationError(
                    "Robot identity mismatch"
                )

            self.context.robot = robot

    # --------------------------------------------------------
    # ADD HUMAN APPROVAL
    # --------------------------------------------------------

    def add_approval(
        self,
        approval_id: str,
        *,
        access: BehaviorTreeAccessContext,
    ) -> None:
        if not access.can_approve_actions:
            raise BehaviorTreeAuthorizationError(
                "Approval permission required"
            )

        if (
            access.tenant_id
            != self.context.access.tenant_id
        ):
            raise BehaviorTreeAuthorizationError(
                "Cross-tenant approval denied"
            )

        if (
            self.context.robot.robot_id
            not in access.allowed_robot_ids
        ):
            raise BehaviorTreeAuthorizationError(
                "Robot approval access denied"
            )

        with self._lock:
            self.context.approvals = (
                self.context.approvals
                | frozenset({
                    approval_id,
                })
            )

    # --------------------------------------------------------
    # FIND NODE
    # --------------------------------------------------------

    def find_node(
        self,
        node_id: str,
    ) -> BehaviorTreeNode:
        def search(
            node: BehaviorTreeNode,
        ) -> BehaviorTreeNode | None:
            if node.node_id == node_id:
                return node

            for child in node.children():
                found = search(child)

                if found is not None:
                    return found

            return None

        found = search(
            self.root
        )

        if found is None:
            raise NodeNotFoundError(
                f"Node not found: {node_id}"
            )

        return found

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(self) -> dict[str, Any]:
        nodes = []

        def collect(
            node: BehaviorTreeNode,
        ) -> None:
            nodes.append(node)

            for child in node.children():
                collect(child)

        collect(
            self.root
        )

        return {
            "tree_id": self.tree_id,
            "tenant_id": self.context.access.tenant_id,
            "robot_id": self.context.robot.robot_id,
            "root_status": self.root.status.value,
            "tick_number": self.context.tick_number,
            "completed": self._completed,
            "node_count": len(nodes),
            "maximum_depth": (
                self.validation_report["maximum_depth"]
            ),
            "node_categories": dict(
                Counter(
                    node.category.value
                    for node in nodes
                )
            ),
            "node_statuses": dict(
                Counter(
                    node.status.value
                    for node in nodes
                )
            ),
            "simulation_results": len(
                self.context.simulation_results
            ),
            "hardware_execution": False,
            "timestamp": utc_iso(),
        }

    # --------------------------------------------------------
    # TRACE
    # --------------------------------------------------------

    def trace(
        self,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self.trace_logger.history(
            self.tree_id,
            limit,
        )

    # --------------------------------------------------------
    # VISUALIZE
    # --------------------------------------------------------

    def visualize(self) -> str:
        return BehaviorTreeVisualizer.to_text(
            self.root
        )

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------

    def export_json(self) -> str:
        payload = {
            "platform": (
                "Physical AI & Autonomous Robotics "
                "Intelligence Platform"
            ),
            "module": "behavior_tree_nodes",
            "tree_id": self.tree_id,
            "tree": self.root.to_dict(),
            "statistics": self.statistics(),
            "blackboard": (
                self.context.blackboard.snapshot()
            ),
            "simulation_results": (
                self.context.simulation_results
            ),
            "trace": self.trace(
                limit=self.config.maximum_trace_events
            ),
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
# UNIT TESTS
# ============================================================

class TestBehaviorTreeNodes(unittest.TestCase):

    def setUp(self) -> None:
        self.access = BehaviorTreeAccessContext(
            tenant_id="tenant_alpha",
            user_id="engineer_001",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
            }),
            can_execute_simulation=True,
            can_modify_tree=True,
            can_approve_actions=True,
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

        self.config = BehaviorTreeConfig(
            robot_state_max_age_seconds=60.0,
        )

    def engine(
        self,
        root: BehaviorTreeNode,
    ) -> RoboticsBehaviorTreeEngine:
        return RoboticsBehaviorTreeEngine(
            root=root,
            access=self.access,
            robot=self.robot,
            config=self.config,
        )

    def test_sequence_success(self) -> None:
        root = SequenceNode(
            "Sequence",
            [
                ConditionNode(
                    "True Condition",
                    lambda ctx: True,
                ),
                BlackboardSetNode(
                    "Set Result",
                    key="result",
                    value="success",
                ),
            ],
        )

        engine = self.engine(root)

        result = engine.tick()

        self.assertEqual(
            result.status,
            NodeStatus.SUCCESS,
        )

    def test_sequence_failure(self) -> None:
        root = SequenceNode(
            "Sequence",
            [
                ConditionNode(
                    "False Condition",
                    lambda ctx: False,
                ),
                BlackboardSetNode(
                    "Should Not Execute",
                    key="executed",
                    value=True,
                ),
            ],
        )

        engine = self.engine(root)

        result = engine.tick()

        self.assertEqual(
            result.status,
            NodeStatus.FAILURE,
        )

        self.assertFalse(
            engine.context.blackboard.contains(
                "executed"
            )
        )

    def test_selector(self) -> None:
        root = SelectorNode(
            "Selector",
            [
                ConditionNode(
                    "Failure",
                    lambda ctx: False,
                ),
                ConditionNode(
                    "Success",
                    lambda ctx: True,
                ),
            ],
        )

        result = self.engine(root).tick()

        self.assertEqual(
            result.status,
            NodeStatus.SUCCESS,
        )

    def test_inverter(self) -> None:
        root = InverterNode(
            "Invert",
            ConditionNode(
                "False",
                lambda ctx: False,
            ),
        )

        result = self.engine(root).tick()

        self.assertEqual(
            result.status,
            NodeStatus.SUCCESS,
        )

    def test_blackboard(self) -> None:
        root = SequenceNode(
            "Blackboard Test",
            [
                BlackboardSetNode(
                    "Set Value",
                    key="temperature",
                    value=42,
                ),
                BlackboardThresholdNode(
                    "Check Temperature",
                    key="temperature",
                    threshold=40,
                ),
            ],
        )

        result = self.engine(root).tick()

        self.assertEqual(
            result.status,
            NodeStatus.SUCCESS,
        )

    def test_robot_safety(self) -> None:
        root = RobotSafetyConditionNode()

        result = self.engine(root).tick()

        self.assertEqual(
            result.status,
            NodeStatus.SUCCESS,
        )

    def test_battery(self) -> None:
        root = RobotBatteryConditionNode(
            "Battery Check",
            minimum_percentage=30,
        )

        result = self.engine(root).tick()

        self.assertEqual(
            result.status,
            NodeStatus.SUCCESS,
        )

    def test_capability(self) -> None:
        root = RobotCapabilityConditionNode(
            "Capability Check",
            required_capabilities=(
                "inspection",
                "diagnostics",
            ),
        )

        result = self.engine(root).tick()

        self.assertEqual(
            result.status,
            NodeStatus.SUCCESS,
        )

    def test_simulated_inspection(self) -> None:
        root = RobotInspectionNode()

        engine = self.engine(root)

        result = engine.tick()

        self.assertEqual(
            result.status,
            NodeStatus.SUCCESS,
        )

        self.assertTrue(
            engine.context.blackboard.contains(
                "inspection_result"
            )
        )

    def test_navigation_requires_approval(self) -> None:
        root = RobotNavigationNode(
            destination="service_station",
            approval_id="NAV-APPROVAL-001",
        )

        engine = self.engine(root)

        first = engine.tick()

        self.assertEqual(
            first.status,
            NodeStatus.FAILURE,
        )

        engine.add_approval(
            "NAV-APPROVAL-001",
            access=self.access,
        )

        second = engine.tick()

        self.assertEqual(
            second.status,
            NodeStatus.SUCCESS,
        )

    def test_duplicate_node_rejected(self) -> None:
        shared = ConditionNode(
            "Shared",
            lambda ctx: True,
        )

        root = SequenceNode(
            "Invalid Tree",
            [shared, shared],
        )

        with self.assertRaises(
            BehaviorTreeValidationError
        ):
            self.engine(root)

    def test_json_export(self) -> None:
        engine = self.engine(
            ConditionNode(
                "Condition",
                lambda ctx: True,
            )
        )

        engine.tick()

        exported = engine.export_json()

        self.assertIn(
            "behavior_tree_nodes",
            exported,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    access = BehaviorTreeAccessContext(
        tenant_id="tenant_alpha",
        user_id="robotics_engineer",
        roles=frozenset({
            "engineer",
        }),
        allowed_robot_ids=frozenset({
            "ROBOT-001",
        }),
        can_execute_simulation=True,
        can_modify_tree=True,
        can_approve_actions=True,
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
        battery_percentage=87.0,
    )

    root = ReactiveSequenceNode(
        "Robot Maintenance Workflow",
        [
            RobotSafetyConditionNode(
                "Verify Robot Safety"
            ),
            RobotBatteryConditionNode(
                "Verify Battery Level",
                minimum_percentage=25.0,
            ),
            RobotCapabilityConditionNode(
                "Verify Inspection Capabilities",
                required_capabilities=(
                    "inspection",
                    "diagnostics",
                ),
            ),
            SequenceNode(
                "Maintenance Tasks",
                [
                    RobotInspectionNode(
                        "Inspect Robot Motor",
                        inspection_type="visual",
                        target_component="motor",
                        result_key="motor_inspection",
                    ),
                    RobotDiagnosticsNode(
                        "Run Motor Diagnostics",
                        subsystem="motor",
                        result_key="motor_diagnostics",
                    ),
                    HumanApprovalConditionNode(
                        "Verify Navigation Approval",
                        approval_id="NAV-001",
                    ),
                    RobotNavigationNode(
                        "Simulate Service Station Navigation",
                        destination="service_station",
                        approval_id="NAV-001",
                        result_key="navigation_simulation",
                    ),
                    BlackboardSetNode(
                        "Complete Maintenance Workflow",
                        key="maintenance_completed",
                        value=True,
                    ),
                ],
            ),
        ],
    )

    engine = RoboticsBehaviorTreeEngine(
        root=root,
        access=access,
        robot=robot,
        config=BehaviorTreeConfig(
            execution_mode=ExecutionMode.SIMULATION,
            robot_state_max_age_seconds=60.0,
        ),
    )

    engine.add_approval(
        "NAV-001",
        access=access,
    )

    print("\nBEHAVIOR TREE STRUCTURE\n")

    print(
        engine.visualize()
    )

    print("\nBEHAVIOR TREE EXECUTION\n")

    result = engine.run(
        maximum_ticks=10
    )

    print(
        json.dumps(
            json_safe(result),
            indent=2,
        )
    )

    print("\nBLACKBOARD RESULTS\n")

    print(
        json.dumps(
            engine.context.blackboard.snapshot(),
            indent=2,
        )
    )

    print("\nBEHAVIOR TREE STATISTICS\n")

    print(
        json.dumps(
            engine.statistics(),
            indent=2,
        )
    )

    print("\nEXECUTION TRACE\n")

    print(
        json.dumps(
            engine.trace(),
            indent=2,
        )
    )

    print("\nUNIT TESTS\n")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestBehaviorTreeNodes
    )

    unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)


if __name__ == "__main__":
    main()
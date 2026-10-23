"""Execution Monitor module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/planning/execution_monitor.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
========================================================

Module: execution_monitor.py

Real-time execution monitoring for robotic task plans,
behavior trees, autonomous agents, industrial inspection,
maintenance workflows, and simulation.

Python: 3.11+
Dependencies: Python standard library only.

Features
--------
- Multi-robot execution monitoring
- Task and plan lifecycle tracking
- Behavior tree execution monitoring
- Heartbeat and watchdog monitoring
- Execution timeout detection
- Task progress tracking
- Robot health and safety monitoring
- Battery and fault monitoring
- Human approval verification
- Task dependency monitoring
- Constraint violation detection
- Execution anomaly detection
- Resource and schedule monitoring
- Simulation telemetry ingestion
- Safety escalation recommendations
- Event history and audit logging
- Tenant and robot access control
- Execution statistics
- JSON export
- Unit tests

Safety
------
Monitoring is advisory and fail-closed.

This module does not directly operate robot hardware,
disable safety mechanisms, issue actuator commands, or
reset emergency-stop equipment.

Safety-critical stop actions must be implemented through
independent, hardware-enforced control systems.

Telemetry is considered untrusted until authenticated
and validated by the surrounding application.
"""

from __future__ import annotations

import json
import math
import statistics
import threading
import time
import unittest
import uuid

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class ExecutionMonitorError(Exception):
    """Base execution monitoring error."""


class ExecutionValidationError(ExecutionMonitorError):
    """Invalid execution monitoring input."""


class ExecutionAuthorizationError(ExecutionMonitorError):
    """Unauthorized execution monitoring operation."""


class ExecutionNotFoundError(ExecutionMonitorError):
    """Execution or task was not found."""


class ExecutionStateError(ExecutionMonitorError):
    """Invalid execution state transition."""


class ExecutionSafetyError(ExecutionMonitorError):
    """Safety requirement was not satisfied."""


class ExecutionCapacityError(ExecutionMonitorError):
    """Monitoring capacity limit exceeded."""


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


def validate_id(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise ExecutionValidationError(
            f"{name} must be a string"
        )

    value = value.strip()

    if not value or len(value) > 128:
        raise ExecutionValidationError(
            f"{name} must contain 1–128 characters"
        )

    return value


def validate_number(
    value: Any,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ExecutionValidationError(
            f"{name} must be a finite number"
        )

    number = float(value)

    if minimum is not None and number < minimum:
        raise ExecutionValidationError(
            f"{name} must be >= {minimum}"
        )

    if maximum is not None and number > maximum:
        raise ExecutionValidationError(
            f"{name} must be <= {maximum}"
        )

    return number


def validate_timestamp(
    value: datetime,
    name: str,
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ExecutionValidationError(
            f"{name} must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


def serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dataclass_fields__"):
        return {
            key: serialize(getattr(value, key))
            for key in value.__dataclass_fields__
        }

    if isinstance(value, Mapping):
        return {
            str(key): serialize(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set, frozenset, deque)):
        return [serialize(item) for item in value]

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, (str, int, bool)) or value is None:
        return value

    return repr(value)


# ============================================================
# ENUMS
# ============================================================

class ExecutionStatus(str, Enum):
    REGISTERED = "registered"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    BLOCKED_SAFETY = "blocked_safety"


class TaskExecutionStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class ExecutionEventType(str, Enum):
    REGISTERED = "registered"
    STARTED = "started"
    HEARTBEAT = "heartbeat"
    TELEMETRY = "telemetry"
    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    PAUSED = "paused"
    RESUMED = "resumed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    SAFETY_ALERT = "safety_alert"
    ANOMALY = "anomaly"
    CONSTRAINT_VIOLATION = "constraint_violation"
    WATCHDOG_TIMEOUT = "watchdog_timeout"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_REVOKED = "approval_revoked"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertCategory(str, Enum):
    SAFETY = "safety"
    HEARTBEAT = "heartbeat"
    BATTERY = "battery"
    TEMPERATURE = "temperature"
    EXECUTION = "execution"
    TASK = "task"
    DEPENDENCY = "dependency"
    CONSTRAINT = "constraint"
    APPROVAL = "approval"
    ANOMALY = "anomaly"
    ROBOT_HEALTH = "robot_health"


class RobotSafetyState(str, Enum):
    SAFE = "safe"
    CAUTION = "caution"
    UNSAFE = "unsafe"
    EMERGENCY_STOP = "emergency_stop"
    UNKNOWN = "unknown"


class MonitoringDecision(str, Enum):
    CONTINUE = "continue"
    REVIEW_REQUIRED = "review_required"
    SUSPEND_RECOMMENDED = "suspend_recommended"
    EMERGENCY_STOP_RECOMMENDED = (
        "emergency_stop_recommended"
    )


class ExecutionMode(str, Enum):
    SIMULATION = "simulation"
    DRY_RUN = "dry_run"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ExecutionMonitorConfig:
    maximum_executions: int = 1000
    maximum_tasks_per_execution: int = 500

    maximum_events_per_execution: int = 5000
    maximum_alerts_per_execution: int = 1000
    maximum_telemetry_samples: int = 1000

    heartbeat_timeout_seconds: float = 10.0
    execution_timeout_seconds: float = 3600.0
    task_timeout_seconds: float = 600.0

    robot_state_max_age_seconds: float = 10.0

    minimum_battery_percentage: float = 10.0
    critical_battery_percentage: float = 5.0

    warning_temperature_c: float = 65.0
    critical_temperature_c: float = 85.0

    maximum_progress_regression: float = 0.0

    anomaly_window_size: int = 20
    anomaly_z_score_threshold: float = 3.5
    anomaly_minimum_samples: int = 6

    enable_heartbeat_monitoring: bool = True
    enable_safety_monitoring: bool = True
    enable_anomaly_detection: bool = True
    enable_constraint_monitoring: bool = True

    require_approval_for_gated_tasks: bool = True

    def __post_init__(self) -> None:
        integer_fields = (
            "maximum_executions",
            "maximum_tasks_per_execution",
            "maximum_events_per_execution",
            "maximum_alerts_per_execution",
            "maximum_telemetry_samples",
            "anomaly_window_size",
            "anomaly_minimum_samples",
        )

        for name in integer_fields:
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ExecutionValidationError(
                    f"{name} must be a positive integer"
                )

        positive_fields = (
            "heartbeat_timeout_seconds",
            "execution_timeout_seconds",
            "task_timeout_seconds",
            "robot_state_max_age_seconds",
            "anomaly_z_score_threshold",
        )

        for name in positive_fields:
            if getattr(self, name) <= 0:
                raise ExecutionValidationError(
                    f"{name} must be positive"
                )

            validate_number(
                getattr(self, name),
                name,
            )

        for name in (
            "minimum_battery_percentage",
            "critical_battery_percentage",
        ):
            validate_number(
                getattr(self, name),
                name,
                0,
                100,
            )

        if (
            self.critical_battery_percentage
            > self.minimum_battery_percentage
        ):
            raise ExecutionValidationError(
                "Critical battery threshold must not "
                "exceed minimum battery threshold"
            )

        if (
            self.warning_temperature_c
            >= self.critical_temperature_c
        ):
            raise ExecutionValidationError(
                "Warning temperature must be below "
                "critical temperature"
            )

        if (
            self.anomaly_minimum_samples
            > self.anomaly_window_size
        ):
            raise ExecutionValidationError(
                "Minimum anomaly samples exceed window size"
            )


# ============================================================
# ACCESS CONTROL
# ============================================================

@dataclass(frozen=True)
class ExecutionAccessContext:
    tenant_id: str
    user_id: str

    allowed_robot_ids: frozenset[str] = frozenset()

    roles: frozenset[str] = frozenset()

    can_register: bool = False
    can_monitor: bool = False
    can_update_execution: bool = False
    can_update_telemetry: bool = False
    can_manage_approvals: bool = False
    can_cancel: bool = False
    can_delete: bool = False

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.user_id, "user_id")


# ============================================================
# TASK DEFINITION
# ============================================================

@dataclass(frozen=True)
class MonitoredTask:
    task_id: str
    name: str

    robot_id: str

    estimated_duration_seconds: float

    dependencies: tuple[str, ...] = ()

    required_capabilities: frozenset[str] = frozenset()

    required_zone: str | None = None

    requires_human_approval: bool = False
    approval_id: str | None = None

    maximum_energy_wh: float | None = None

    priority: int = 1

    def __post_init__(self) -> None:
        validate_id(self.task_id, "task_id")
        validate_id(self.name, "name")
        validate_id(self.robot_id, "robot_id")

        validate_number(
            self.estimated_duration_seconds,
            "estimated_duration_seconds",
            minimum=0.001,
        )

        if self.maximum_energy_wh is not None:
            validate_number(
                self.maximum_energy_wh,
                "maximum_energy_wh",
                minimum=0,
            )

        if self.priority < 1:
            raise ExecutionValidationError(
                "Task priority must be positive"
            )

        if (
            self.requires_human_approval
            and not self.approval_id
        ):
            raise ExecutionValidationError(
                "Approval ID required for gated task"
            )


# ============================================================
# EXECUTION SPECIFICATION
# ============================================================

@dataclass(frozen=True)
class ExecutionSpecification:
    execution_id: str

    tenant_id: str

    name: str

    tasks: tuple[MonitoredTask, ...]

    mode: ExecutionMode = ExecutionMode.SIMULATION

    plan_id: str | None = None

    behavior_tree_id: str | None = None

    created_by: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_id(
            self.execution_id,
            "execution_id",
        )

        validate_id(
            self.tenant_id,
            "tenant_id",
        )

        validate_id(
            self.name,
            "name",
        )

        if not self.tasks:
            raise ExecutionValidationError(
                "Execution must contain tasks"
            )

        if len({
            task.task_id
            for task in self.tasks
        }) != len(self.tasks):
            raise ExecutionValidationError(
                "Duplicate task IDs"
            )

        validate_timestamp(
            self.created_at,
            "created_at",
        )


# ============================================================
# ROBOT TELEMETRY
# ============================================================

@dataclass(frozen=True)
class RobotExecutionTelemetry:
    tenant_id: str
    robot_id: str

    timestamp: datetime

    battery_percentage: float

    safety_state: RobotSafetyState

    emergency_stop_active: bool = False

    temperature_c: float | None = None

    cpu_usage_percentage: float | None = None

    memory_usage_percentage: float | None = None

    velocity_mps: float | None = None

    position_x: float | None = None
    position_y: float | None = None
    position_z: float | None = None

    current_zone: str | None = None

    active_faults: tuple[str, ...] = ()

    capabilities: frozenset[str] = frozenset()

    energy_consumed_wh: float = 0.0

    custom_metrics: dict[str, float] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")

        validate_timestamp(
            self.timestamp,
            "timestamp",
        )

        validate_number(
            self.battery_percentage,
            "battery_percentage",
            0,
            100,
        )

        for name in (
            "cpu_usage_percentage",
            "memory_usage_percentage",
        ):
            value = getattr(self, name)

            if value is not None:
                validate_number(
                    value,
                    name,
                    0,
                    100,
                )

        for name in (
            "temperature_c",
            "velocity_mps",
            "position_x",
            "position_y",
            "position_z",
        ):
            value = getattr(self, name)

            if value is not None:
                validate_number(
                    value,
                    name,
                )

        validate_number(
            self.energy_consumed_wh,
            "energy_consumed_wh",
            minimum=0,
        )

        for key, value in self.custom_metrics.items():
            validate_id(key, "custom metric name")
            validate_number(
                value,
                f"custom metric {key}",
            )


# ============================================================
# EXECUTION ALERT
# ============================================================

@dataclass(frozen=True)
class ExecutionAlert:
    alert_id: str

    execution_id: str
    tenant_id: str

    robot_id: str | None

    task_id: str | None

    category: AlertCategory
    severity: AlertSeverity

    code: str
    message: str

    recommended_action: str

    timestamp: datetime = field(
        default_factory=utc_now
    )

    details: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# EXECUTION EVENT
# ============================================================

@dataclass(frozen=True)
class ExecutionEvent:
    event_id: str

    execution_id: str
    tenant_id: str

    event_type: ExecutionEventType

    actor: str

    timestamp: datetime = field(
        default_factory=utc_now
    )

    task_id: str | None = None
    robot_id: str | None = None

    details: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# TASK RUNTIME
# ============================================================

@dataclass
class TaskRuntime:
    specification: MonitoredTask

    status: TaskExecutionStatus = (
        TaskExecutionStatus.PENDING
    )

    progress_percentage: float = 0.0

    started_at: datetime | None = None
    completed_at: datetime | None = None

    last_progress_at: datetime | None = None

    energy_consumed_wh: float = 0.0

    failure_reason: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return serialize(self)


# ============================================================
# EXECUTION RUNTIME
# ============================================================

@dataclass
class ExecutionRuntime:
    specification: ExecutionSpecification

    status: ExecutionStatus = ExecutionStatus.REGISTERED

    tasks: dict[str, TaskRuntime] = field(
        default_factory=dict
    )

    telemetry: dict[
        str, RobotExecutionTelemetry
    ] = field(default_factory=dict)

    telemetry_history: dict[
        str, deque[RobotExecutionTelemetry]
    ] = field(default_factory=dict)

    heartbeats: dict[
        str, datetime
    ] = field(default_factory=dict)

    approvals: set[str] = field(
        default_factory=set
    )

    alerts: deque[ExecutionAlert] = field(
        default_factory=deque
    )

    events: deque[ExecutionEvent] = field(
        default_factory=deque
    )

    started_at: datetime | None = None
    completed_at: datetime | None = None

    last_updated: datetime = field(
        default_factory=utc_now
    )

    last_error: str | None = None

    lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )

    def progress_percentage(self) -> float:
        if not self.tasks:
            return 0.0

        return sum(
            task.progress_percentage
            for task in self.tasks.values()
        ) / len(self.tasks)

    def snapshot(self) -> dict[str, Any]:
        return {
            "specification": serialize(
                self.specification
            ),
            "status": self.status.value,
            "progress_percentage": (
                self.progress_percentage()
            ),
            "tasks": {
                task_id: task.snapshot()
                for task_id, task in self.tasks.items()
            },
            "started_at": serialize(
                self.started_at
            ),
            "completed_at": serialize(
                self.completed_at
            ),
            "last_updated": serialize(
                self.last_updated
            ),
            "last_error": self.last_error,
            "alert_count": len(self.alerts),
            "event_count": len(self.events),
            "simulation_only": True,
            "hardware_execution": False,
        }


# ============================================================
# SAFETY EVALUATOR
# ============================================================

@dataclass(frozen=True)
class SafetyEvaluation:
    allowed: bool

    decision: MonitoringDecision

    reasons: tuple[str, ...]

    severity: AlertSeverity | None = None


class ExecutionSafetyEvaluator:

    def __init__(
        self,
        config: ExecutionMonitorConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        telemetry: RobotExecutionTelemetry,
    ) -> SafetyEvaluation:
        reasons = []

        severity = None

        if telemetry.emergency_stop_active:
            reasons.append(
                "Emergency stop is active"
            )

            severity = AlertSeverity.EMERGENCY

        if telemetry.safety_state in (
            RobotSafetyState.UNSAFE,
            RobotSafetyState.EMERGENCY_STOP,
            RobotSafetyState.UNKNOWN,
        ):
            reasons.append(
                "Robot safety state is not verified safe"
            )

            if severity != AlertSeverity.EMERGENCY:
                severity = AlertSeverity.CRITICAL

        if telemetry.active_faults:
            reasons.append(
                "Robot reports active faults"
            )

            if severity is None:
                severity = AlertSeverity.CRITICAL

        if (
            telemetry.battery_percentage
            <= self.config.critical_battery_percentage
        ):
            reasons.append(
                "Critical battery level"
            )

            if severity is None:
                severity = AlertSeverity.CRITICAL

        elif (
            telemetry.battery_percentage
            < self.config.minimum_battery_percentage
        ):
            reasons.append(
                "Battery below operational threshold"
            )

            if severity is None:
                severity = AlertSeverity.WARNING

        if telemetry.temperature_c is not None:
            if (
                telemetry.temperature_c
                >= self.config.critical_temperature_c
            ):
                reasons.append(
                    "Critical robot temperature"
                )

                if severity is None:
                    severity = AlertSeverity.CRITICAL

            elif (
                telemetry.temperature_c
                >= self.config.warning_temperature_c
            ):
                reasons.append(
                    "Robot temperature warning"
                )

                if severity is None:
                    severity = AlertSeverity.WARNING

        age = (
            utc_now() - telemetry.timestamp
        ).total_seconds()

        if (
            age < 0
            or age > self.config.robot_state_max_age_seconds
        ):
            reasons.append(
                "Robot telemetry is stale or future-dated"
            )

            if severity is None:
                severity = AlertSeverity.CRITICAL

        if telemetry.safety_state == RobotSafetyState.CAUTION:
            reasons.append(
                "Robot reports caution safety state"
            )

            if severity is None:
                severity = AlertSeverity.WARNING

        if not reasons:
            return SafetyEvaluation(
                allowed=True,
                decision=MonitoringDecision.CONTINUE,
                reasons=(),
            )

        if severity == AlertSeverity.EMERGENCY:
            decision = (
                MonitoringDecision.EMERGENCY_STOP_RECOMMENDED
            )

        elif severity == AlertSeverity.CRITICAL:
            decision = (
                MonitoringDecision.SUSPEND_RECOMMENDED
            )

        else:
            decision = MonitoringDecision.REVIEW_REQUIRED

        return SafetyEvaluation(
            allowed=False,
            decision=decision,
            reasons=tuple(reasons),
            severity=severity,
        )


# ============================================================
# ANOMALY DETECTOR
# ============================================================

@dataclass(frozen=True)
class ExecutionAnomaly:
    metric_name: str

    observed_value: float

    expected_mean: float

    standard_deviation: float

    z_score: float

    severity: AlertSeverity

    timestamp: datetime = field(
        default_factory=utc_now
    )


class ExecutionAnomalyDetector:
    """
    Simple statistical anomaly detector.

    Uses historical telemetry values and z-scores.

    This is a heuristic monitoring component, not a
    validated safety-critical fault detection model.
    """

    DEFAULT_METRICS = (
        "temperature_c",
        "cpu_usage_percentage",
        "memory_usage_percentage",
        "velocity_mps",
    )

    def __init__(
        self,
        config: ExecutionMonitorConfig,
    ) -> None:
        self.config = config

    def detect(
        self,
        current: RobotExecutionTelemetry,
        history: Sequence[RobotExecutionTelemetry],
    ) -> tuple[ExecutionAnomaly, ...]:
        if not self.config.enable_anomaly_detection:
            return ()

        anomalies = []

        metric_names = list(
            self.DEFAULT_METRICS
        )

        metric_names.extend(
            current.custom_metrics.keys()
        )

        for metric_name in metric_names:
            if metric_name in current.custom_metrics:
                current_value = (
                    current.custom_metrics[metric_name]
                )

                values = [
                    sample.custom_metrics[metric_name]
                    for sample in history
                    if metric_name in sample.custom_metrics
                ]

            else:
                current_value = getattr(
                    current,
                    metric_name,
                    None,
                )

                values = [
                    getattr(sample, metric_name)
                    for sample in history
                    if getattr(
                        sample,
                        metric_name,
                        None,
                    ) is not None
                ]

            if current_value is None:
                continue

            values = values[
                -self.config.anomaly_window_size:
            ]

            if (
                len(values)
                < self.config.anomaly_minimum_samples
            ):
                continue

            mean = statistics.mean(
                values
            )

            standard_deviation = (
                statistics.pstdev(values)
            )

            if standard_deviation <= 1e-9:
                continue

            z_score = abs(
                (current_value - mean)
                / standard_deviation
            )

            if (
                z_score
                < self.config.anomaly_z_score_threshold
            ):
                continue

            severity = (
                AlertSeverity.CRITICAL
                if z_score >= (
                    self.config.anomaly_z_score_threshold
                    * 2
                )
                else AlertSeverity.WARNING
            )

            anomalies.append(
                ExecutionAnomaly(
                    metric_name=metric_name,
                    observed_value=float(current_value),
                    expected_mean=float(mean),
                    standard_deviation=float(
                        standard_deviation
                    ),
                    z_score=float(z_score),
                    severity=severity,
                )
            )

        return tuple(anomalies)


# ============================================================
# EXECUTION CONSTRAINT CHECKER
# ============================================================

@dataclass(frozen=True)
class ExecutionConstraintViolation:
    code: str
    task_id: str
    robot_id: str

    description: str

    severity: AlertSeverity


class ExecutionConstraintChecker:

    @staticmethod
    def validate_task_start(
        task: TaskRuntime,
        tasks: Mapping[str, TaskRuntime],
        telemetry: RobotExecutionTelemetry,
        approvals: set[str],
    ) -> tuple[ExecutionConstraintViolation, ...]:
        violations = []

        specification = task.specification

        for dependency_id in specification.dependencies:
            dependency = tasks.get(
                dependency_id
            )

            if (
                dependency is None
                or dependency.status
                != TaskExecutionStatus.COMPLETED
            ):
                violations.append(
                    ExecutionConstraintViolation(
                        code="DEPENDENCY_INCOMPLETE",
                        task_id=specification.task_id,
                        robot_id=specification.robot_id,
                        description=(
                            f"Dependency {dependency_id} "
                            "is not completed"
                        ),
                        severity=AlertSeverity.WARNING,
                    )
                )

        if (
            specification.requires_human_approval
            and specification.approval_id not in approvals
        ):
            violations.append(
                ExecutionConstraintViolation(
                    code="APPROVAL_MISSING",
                    task_id=specification.task_id,
                    robot_id=specification.robot_id,
                    description=(
                        "Required human approval is missing"
                    ),
                    severity=AlertSeverity.CRITICAL,
                )
            )

        if not specification.required_capabilities.issubset(
            telemetry.capabilities
        ):
            violations.append(
                ExecutionConstraintViolation(
                    code="CAPABILITY_MISMATCH",
                    task_id=specification.task_id,
                    robot_id=specification.robot_id,
                    description=(
                        "Robot lacks required capabilities"
                    ),
                    severity=AlertSeverity.CRITICAL,
                )
            )

        if (
            specification.required_zone is not None
            and telemetry.current_zone
            != specification.required_zone
        ):
            violations.append(
                ExecutionConstraintViolation(
                    code="ZONE_MISMATCH",
                    task_id=specification.task_id,
                    robot_id=specification.robot_id,
                    description=(
                        "Robot is outside required task zone"
                    ),
                    severity=AlertSeverity.CRITICAL,
                )
            )

        return tuple(violations)


# ============================================================
# MAIN EXECUTION MONITOR
# ============================================================

class RoboticsExecutionMonitor:
    """
    Thread-safe execution monitoring engine.

    All execution state transitions are logical simulation
    states, not hardware command acknowledgements.
    """

    TERMINAL_STATUSES = frozenset({
        ExecutionStatus.COMPLETED,
        ExecutionStatus.FAILED,
        ExecutionStatus.CANCELLED,
        ExecutionStatus.TIMED_OUT,
    })

    def __init__(
        self,
        config: ExecutionMonitorConfig | None = None,
    ) -> None:
        self.config = (
            config
            or ExecutionMonitorConfig()
        )

        self._executions: dict[
            str, ExecutionRuntime
        ] = {}

        self._lock = threading.RLock()

        self._safety = ExecutionSafetyEvaluator(
            self.config
        )

        self._anomaly_detector = (
            ExecutionAnomalyDetector(
                self.config
            )
        )

        self._metrics = Counter()

    # --------------------------------------------------------
    # AUTHORIZATION
    # --------------------------------------------------------

    @staticmethod
    def _require_permission(
        access: ExecutionAccessContext,
        permission: str,
    ) -> None:
        if not getattr(access, permission, False):
            raise ExecutionAuthorizationError(
                f"Permission required: {permission}"
            )

    @staticmethod
    def _authorize_robot(
        access: ExecutionAccessContext,
        tenant_id: str,
        robot_id: str,
    ) -> None:
        if access.tenant_id != tenant_id:
            raise ExecutionAuthorizationError(
                "Cross-tenant access denied"
            )

        if robot_id not in access.allowed_robot_ids:
            raise ExecutionAuthorizationError(
                "Robot access denied"
            )

    def _get_runtime(
        self,
        execution_id: str,
        access: ExecutionAccessContext,
    ) -> ExecutionRuntime:
        with self._lock:
            runtime = self._executions.get(
                execution_id
            )

        if runtime is None:
            raise ExecutionNotFoundError(
                f"Execution not found: {execution_id}"
            )

        if (
            runtime.specification.tenant_id
            != access.tenant_id
        ):
            raise ExecutionAuthorizationError(
                "Cross-tenant execution access denied"
            )

        for task in runtime.specification.tasks:
            if task.robot_id not in access.allowed_robot_ids:
                raise ExecutionAuthorizationError(
                    "Robot access denied"
                )

        return runtime

    # --------------------------------------------------------
    # EVENT RECORDING
    # --------------------------------------------------------

    def _event(
        self,
        runtime: ExecutionRuntime,
        event_type: ExecutionEventType,
        *,
        actor: str,
        task_id: str | None = None,
        robot_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        runtime.events.append(
            ExecutionEvent(
                event_id=new_id(),
                execution_id=(
                    runtime.specification.execution_id
                ),
                tenant_id=(
                    runtime.specification.tenant_id
                ),
                event_type=event_type,
                actor=actor,
                task_id=task_id,
                robot_id=robot_id,
                details=serialize(
                    details or {}
                ),
            )
        )

        runtime.last_updated = utc_now()

    # --------------------------------------------------------
    # ALERT RECORDING
    # --------------------------------------------------------

    def _alert(
        self,
        runtime: ExecutionRuntime,
        *,
        category: AlertCategory,
        severity: AlertSeverity,
        code: str,
        message: str,
        recommended_action: str,
        robot_id: str | None = None,
        task_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> ExecutionAlert:
        alert = ExecutionAlert(
            alert_id=new_id(),
            execution_id=(
                runtime.specification.execution_id
            ),
            tenant_id=(
                runtime.specification.tenant_id
            ),
            robot_id=robot_id,
            task_id=task_id,
            category=category,
            severity=severity,
            code=code,
            message=message,
            recommended_action=recommended_action,
            details=serialize(
                details or {}
            ),
        )

        runtime.alerts.append(
            alert
        )

        self._metrics[
            f"alert_{severity.value}"
        ] += 1

        return alert

    # --------------------------------------------------------
    # REGISTER EXECUTION
    # --------------------------------------------------------

    def register_execution(
        self,
        specification: ExecutionSpecification,
        *,
        access: ExecutionAccessContext,
    ) -> dict[str, Any]:
        self._require_permission(
            access,
            "can_register",
        )

        if (
            specification.tenant_id
            != access.tenant_id
        ):
            raise ExecutionAuthorizationError(
                "Execution tenant mismatch"
            )

        if (
            len(specification.tasks)
            > self.config.maximum_tasks_per_execution
        ):
            raise ExecutionCapacityError(
                "Too many tasks in execution"
            )

        for task in specification.tasks:
            self._authorize_robot(
                access,
                specification.tenant_id,
                task.robot_id,
            )

        self._validate_dependencies(
            specification.tasks
        )

        with self._lock:
            if (
                specification.execution_id
                in self._executions
            ):
                raise ExecutionValidationError(
                    "Execution ID already registered"
                )

            if (
                len(self._executions)
                >= self.config.maximum_executions
            ):
                raise ExecutionCapacityError(
                    "Execution registry capacity exceeded"
                )

            runtime = ExecutionRuntime(
                specification=specification,
                status=ExecutionStatus.READY,
                tasks={
                    task.task_id: TaskRuntime(
                        specification=task
                    )
                    for task in specification.tasks
                },
                alerts=deque(
                    maxlen=(
                        self.config.maximum_alerts_per_execution
                    )
                ),
                events=deque(
                    maxlen=(
                        self.config.maximum_events_per_execution
                    )
                ),
            )

            self._executions[
                specification.execution_id
            ] = runtime

            self._metrics["registered"] += 1

            self._event(
                runtime,
                ExecutionEventType.REGISTERED,
                actor=access.user_id,
                details={
                    "task_count": len(specification.tasks),
                    "mode": specification.mode.value,
                },
            )

            return runtime.snapshot()

    # --------------------------------------------------------
    # DEPENDENCY VALIDATION
    # --------------------------------------------------------

    @staticmethod
    def _validate_dependencies(
        tasks: Sequence[MonitoredTask],
    ) -> None:
        task_map = {
            task.task_id: task
            for task in tasks
        }

        visiting = set()
        visited = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ExecutionValidationError(
                    "Circular task dependency detected"
                )

            if task_id in visited:
                return

            visiting.add(task_id)

            for dependency in task_map[task_id].dependencies:
                if dependency not in task_map:
                    raise ExecutionValidationError(
                        f"Unknown task dependency: {dependency}"
                    )

                visit(dependency)

            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in task_map:
            visit(task_id)

    # --------------------------------------------------------
    # INGEST TELEMETRY
    # --------------------------------------------------------

    def ingest_telemetry(
        self,
        execution_id: str,
        telemetry: RobotExecutionTelemetry,
        *,
        access: ExecutionAccessContext,
    ) -> dict[str, Any]:
        self._require_permission(
            access,
            "can_update_telemetry",
        )

        runtime = self._get_runtime(
            execution_id,
            access,
        )

        self._authorize_robot(
            access,
            telemetry.tenant_id,
            telemetry.robot_id,
        )

        with runtime.lock:
            robot_ids = {
                task.robot_id
                for task in runtime.specification.tasks
            }

            if telemetry.robot_id not in robot_ids:
                raise ExecutionValidationError(
                    "Robot does not belong to execution"
                )

            previous = runtime.telemetry.get(
                telemetry.robot_id
            )

            if (
                previous is not None
                and telemetry.timestamp <= previous.timestamp
            ):
                raise ExecutionValidationError(
                    "Telemetry timestamp must increase"
                )

            history = runtime.telemetry_history.setdefault(
                telemetry.robot_id,
                deque(
                    maxlen=(
                        self.config.maximum_telemetry_samples
                    )
                ),
            )

            anomalies = self._anomaly_detector.detect(
                telemetry,
                list(history),
            )

            history.append(
                telemetry
            )

            runtime.telemetry[
                telemetry.robot_id
            ] = telemetry

            runtime.heartbeats[
                telemetry.robot_id
            ] = utc_now()

            self._event(
                runtime,
                ExecutionEventType.TELEMETRY,
                actor=access.user_id,
                robot_id=telemetry.robot_id,
                details={
                    "battery_percentage": (
                        telemetry.battery_percentage
                    ),
                    "safety_state": (
                        telemetry.safety_state.value
                    ),
                    "temperature_c": (
                        telemetry.temperature_c
                    ),
                },
            )

            safety = self._safety.evaluate(
                telemetry
            )

            if not safety.allowed:
                self._alert(
                    runtime,
                    category=AlertCategory.SAFETY,
                    severity=(
                        safety.severity
                        or AlertSeverity.WARNING
                    ),
                    code="ROBOT_SAFETY_WARNING",
                    message="; ".join(safety.reasons),
                    recommended_action=(
                        safety.decision.value
                    ),
                    robot_id=telemetry.robot_id,
                )

                self._event(
                    runtime,
                    ExecutionEventType.SAFETY_ALERT,
                    actor=access.user_id,
                    robot_id=telemetry.robot_id,
                    details={
                        "reasons": safety.reasons,
                        "decision": safety.decision.value,
                    },
                )

                if safety.severity in (
                    AlertSeverity.CRITICAL,
                    AlertSeverity.EMERGENCY,
                ):
                    if runtime.status == ExecutionStatus.RUNNING:
                        runtime.status = (
                            ExecutionStatus.BLOCKED_SAFETY
                        )

                        runtime.last_error = (
                            "; ".join(safety.reasons)
                        )

            for anomaly in anomalies:
                self._alert(
                    runtime,
                    category=AlertCategory.ANOMALY,
                    severity=anomaly.severity,
                    code="TELEMETRY_ANOMALY",
                    message=(
                        f"Anomaly detected in "
                        f"{anomaly.metric_name}"
                    ),
                    recommended_action=(
                        "Inspect telemetry and verify "
                        "robot operating conditions"
                    ),
                    robot_id=telemetry.robot_id,
                    details=serialize(anomaly),
                )

            return {
                "execution_id": execution_id,
                "robot_id": telemetry.robot_id,
                "safety": serialize(safety),
                "anomalies": serialize(anomalies),
                "execution_status": runtime.status.value,
            }

    # --------------------------------------------------------
    # HEARTBEAT
    # --------------------------------------------------------

    def record_heartbeat(
        self,
        execution_id: str,
        robot_id: str,
        *,
        access: ExecutionAccessContext,
    ) -> None:
        self._require_permission(
            access,
            "can_update_telemetry",
        )

        runtime = self._get_runtime(
            execution_id,
            access,
        )

        with runtime.lock:
            if robot_id not in {
                task.robot_id
                for task in runtime.specification.tasks
            }:
                raise ExecutionValidationError(
                    "Robot not part of execution"
                )

            runtime.heartbeats[
                robot_id
            ] = utc_now()

            self._event(
                runtime,
                ExecutionEventType.HEARTBEAT,
                actor=access.user_id,
                robot_id=robot_id,
            )

    # --------------------------------------------------------
    # START EXECUTION
    # --------------------------------------------------------

    def start_execution(
        self,
        execution_id: str,
        *,
        access: ExecutionAccessContext,
    ) -> dict[str, Any]:
        self._require_permission(
            access,
            "can_update_execution",
        )

        runtime = self._get_runtime(
            execution_id,
            access,
        )

        with runtime.lock:
            if runtime.status != ExecutionStatus.READY:
                raise ExecutionStateError(
                    "Execution must be READY"
                )

            robot_ids = {
                task.robot_id
                for task in runtime.specification.tasks
            }

            for robot_id in robot_ids:
                telemetry = runtime.telemetry.get(
                    robot_id
                )

                if telemetry is None:
                    raise ExecutionSafetyError(
                        f"Missing telemetry for {robot_id}"
                    )

                safety = self._safety.evaluate(
                    telemetry
                )

                if not safety.allowed:
                    raise ExecutionSafetyError(
                        "; ".join(safety.reasons)
                    )

            runtime.status = ExecutionStatus.RUNNING

            runtime.started_at = utc_now()

            self._event(
                runtime,
                ExecutionEventType.STARTED,
                actor=access.user_id,
            )

            self._metrics["started"] += 1

            return runtime.snapshot()

    # --------------------------------------------------------
    # START TASK
    # --------------------------------------------------------

    def start_task(
        self,
        execution_id: str,
        task_id: str,
        *,
        access: ExecutionAccessContext,
    ) -> dict[str, Any]:
        self._require_permission(
            access,
            "can_update_execution",
        )

        runtime = self._get_runtime(
            execution_id,
            access,
        )

        with runtime.lock:
            if runtime.status != ExecutionStatus.RUNNING:
                raise ExecutionStateError(
                    "Execution must be RUNNING"
                )

            task = runtime.tasks.get(
                task_id
            )

            if task is None:
                raise ExecutionNotFoundError(
                    "Task not found"
                )

            if task.status not in (
                TaskExecutionStatus.PENDING,
                TaskExecutionStatus.READY,
            ):
                raise ExecutionStateError(
                    "Task cannot be started"
                )

            telemetry = runtime.telemetry.get(
                task.specification.robot_id
            )

            if telemetry is None:
                raise ExecutionSafetyError(
                    "Robot telemetry unavailable"
                )

            safety = self._safety.evaluate(
                telemetry
            )

            if not safety.allowed:
                raise ExecutionSafetyError(
                    "; ".join(safety.reasons)
                )

            violations = (
                ExecutionConstraintChecker.validate_task_start(
                    task,
                    runtime.tasks,
                    telemetry,
                    runtime.approvals,
                )
            )

            if violations:
                for violation in violations:
                    self._alert(
                        runtime,
                        category=AlertCategory.CONSTRAINT,
                        severity=violation.severity,
                        code=violation.code,
                        message=violation.description,
                        recommended_action=(
                            "Resolve constraint before "
                            "starting task"
                        ),
                        robot_id=violation.robot_id,
                        task_id=violation.task_id,
                    )

                raise ExecutionSafetyError(
                    "Task constraints not satisfied"
                )

            for other in runtime.tasks.values():
                if (
                    other.specification.robot_id
                    == task.specification.robot_id
                    and other.status
                    == TaskExecutionStatus.RUNNING
                ):
                    raise ExecutionStateError(
                        "Robot already has a running task"
                    )

            task.status = TaskExecutionStatus.RUNNING
            task.started_at = utc_now()

            self._event(
                runtime,
                ExecutionEventType.TASK_STARTED,
                actor=access.user_id,
                task_id=task_id,
                robot_id=task.specification.robot_id,
            )

            return task.snapshot()

    # --------------------------------------------------------
    # UPDATE TASK PROGRESS
    # --------------------------------------------------------

    def update_task_progress(
        self,
        execution_id: str,
        task_id: str,
        progress_percentage: float,
        *,
        access: ExecutionAccessContext,
        energy_consumed_wh: float | None = None,
    ) -> dict[str, Any]:
        self._require_permission(
            access,
            "can_update_execution",
        )

        progress = validate_number(
            progress_percentage,
            "progress_percentage",
            0,
            100,
        )

        runtime = self._get_runtime(
            execution_id,
            access,
        )

        with runtime.lock:
            if runtime.status != ExecutionStatus.RUNNING:
                raise ExecutionStateError(
                    "Execution must be RUNNING"
                )

            task = runtime.tasks.get(
                task_id
            )

            if task is None:
                raise ExecutionNotFoundError(
                    "Task not found"
                )

            if task.status != TaskExecutionStatus.RUNNING:
                raise ExecutionStateError(
                    "Task must be RUNNING"
                )

            if (
                progress
                < task.progress_percentage
                - self.config.maximum_progress_regression
            ):
                raise ExecutionValidationError(
                    "Task progress regression exceeds limit"
                )

            task.progress_percentage = progress
            task.last_progress_at = utc_now()

            if energy_consumed_wh is not None:
                energy = validate_number(
                    energy_consumed_wh,
                    "energy_consumed_wh",
                    minimum=0,
                )

                if energy < task.energy_consumed_wh:
                    raise ExecutionValidationError(
                        "Energy consumption cannot decrease"
                    )

                task.energy_consumed_wh = energy

                maximum_energy = (
                    task.specification.maximum_energy_wh
                )

                if (
                    maximum_energy is not None
                    and energy > maximum_energy
                ):
                    self._alert(
                        runtime,
                        category=AlertCategory.CONSTRAINT,
                        severity=AlertSeverity.WARNING,
                        code="TASK_ENERGY_BUDGET_EXCEEDED",
                        message=(
                            "Task energy consumption "
                            "exceeds configured budget"
                        ),
                        recommended_action=(
                            "Review task energy consumption"
                        ),
                        robot_id=task.specification.robot_id,
                        task_id=task_id,
                    )

            self._event(
                runtime,
                ExecutionEventType.TASK_PROGRESS,
                actor=access.user_id,
                task_id=task_id,
                robot_id=task.specification.robot_id,
                details={
                    "progress_percentage": progress,
                    "energy_consumed_wh": (
                        task.energy_consumed_wh
                    ),
                },
            )

            return task.snapshot()

    # --------------------------------------------------------
    # COMPLETE TASK
    # --------------------------------------------------------

    def complete_task(
        self,
        execution_id: str,
        task_id: str,
        *,
        access: ExecutionAccessContext,
    ) -> dict[str, Any]:
        self._require_permission(
            access,
            "can_update_execution",
        )

        runtime = self._get_runtime(
            execution_id,
            access,
        )

        with runtime.lock:
            if runtime.status != ExecutionStatus.RUNNING:
                raise ExecutionStateError(
                    "Execution must be RUNNING"
                )

            task = runtime.tasks.get(
                task_id
            )

            if task is None:
                raise ExecutionNotFoundError(
                    "Task not found"
                )

            if task.status != TaskExecutionStatus.RUNNING:
                raise ExecutionStateError(
                    "Only RUNNING tasks can complete"
                )

            task.status = TaskExecutionStatus.COMPLETED
            task.progress_percentage = 100.0
            task.completed_at = utc_now()

            self._event(
                runtime,
                ExecutionEventType.TASK_COMPLETED,
                actor=access.user_id,
                task_id=task_id,
                robot_id=task.specification.robot_id,
            )

            if all(
                item.status == TaskExecutionStatus.COMPLETED
                for item in runtime.tasks.values()
            ):
                runtime.status = ExecutionStatus.COMPLETED
                runtime.completed_at = utc_now()

                self._event(
                    runtime,
                    ExecutionEventType.COMPLETED,
                    actor=access.user_id,
                )

                self._metrics["completed"] += 1

            return runtime.snapshot()

    # --------------------------------------------------------
    # FAIL TASK
    # --------------------------------------------------------

    def fail_task(
        self,
        execution_id: str,
        task_id: str,
        *,
        reason: str,
        access: ExecutionAccessContext,
    ) -> dict[str, Any]:
        self._require_permission(
            access,
            "can_update_execution",
        )

        runtime = self._get_runtime(
            execution_id,
            access,
        )

        with runtime.lock:
            task = runtime.tasks.get(
                task_id
            )

            if task is None:
                raise ExecutionNotFoundError(
                    "Task not found"
                )

            if task.status in (
                TaskExecutionStatus.COMPLETED,
                TaskExecutionStatus.CANCELLED,
            ):
                raise ExecutionStateError(
                    "Cannot fail a terminal task"
                )

            task.status = TaskExecutionStatus.FAILED
            task.failure_reason = reason
            task.completed_at = utc_now()

            runtime.status = ExecutionStatus.FAILED
            runtime.completed_at = utc_now()
            runtime.last_error = reason

            self._alert(
                runtime,
                category=AlertCategory.TASK,
                severity=AlertSeverity.CRITICAL,
                code="TASK_EXECUTION_FAILED",
                message=reason,
                recommended_action=(
                    "Investigate failure and review "
                    "the execution plan"
                ),
                task_id=task_id,
                robot_id=task.specification.robot_id,
            )

            self._event(
                runtime,
                ExecutionEventType.TASK_FAILED,
                actor=access.user_id,
                task_id=task_id,
                robot_id=task.specification.robot_id,
                details={
                    "reason": reason,
                },
            )

            self._metrics["failed"] += 1

            return runtime.snapshot()

    # --------------------------------------------------------
    # HUMAN APPROVAL
    # --------------------------------------------------------

    def grant_approval(
        self,
        execution_id: str,
        approval_id: str,
        *,
        access: ExecutionAccessContext,
    ) -> None:
        self._require_permission(
            access,
            "can_manage_approvals",
        )

        validate_id(
            approval_id,
            "approval_id",
        )

        runtime = self._get_runtime(
            execution_id,
            access,
        )

        with runtime.lock:
            valid_approvals = {
                task.approval_id
                for task in runtime.specification.tasks
                if task.requires_human_approval
            }

            if approval_id not in valid_approvals:
                raise ExecutionValidationError(
                    "Approval ID is not registered "
                    "for this execution"
                )

            runtime.approvals.add(
                approval_id
            )

            self._event(
                runtime,
                ExecutionEventType.APPROVAL_GRANTED,
                actor=access.user_id,
                details={
                    "approval_id": approval_id,
                    "hardware_authorization": False,
                },
            )

    def revoke_approval(
        self,
        execution_id: str,
        approval_id: str,
        *,
        access: ExecutionAccessContext,
    ) -> None:
        self._require_permission(
            access,
            "can_manage_approvals",
        )

        runtime = self._get_runtime(
            execution_id,
            access,
        )

        with runtime.lock:
            runtime.approvals.discard(
                approval_id
            )

            self._event(
                runtime,
                ExecutionEventType.APPROVAL_REVOKED,
                actor=access.user_id,
                details={
                    "approval_id": approval_id,
                },
            )

            affected_running_tasks = [
                task
                for task in runtime.tasks.values()
                if (
                    task.status == TaskExecutionStatus.RUNNING
                    and task.specification.approval_id
                    == approval_id
                )
            ]

            if affected_running_tasks:
                runtime.status = (
                    ExecutionStatus.BLOCKED_SAFETY
                )

                runtime.last_error = (
                    "Approval revoked during execution"
                )

                self._alert(
                    runtime,
                    category=AlertCategory.APPROVAL,
                    severity=AlertSeverity.CRITICAL,
                    code="ACTIVE_APPROVAL_REVOKED",
                    message=runtime.last_error,
                    recommended_action=(
                        "Suspend simulation and obtain "
                        "new approval before continuation"
                    ),
                )

    # --------------------------------------------------------
    # PAUSE EXECUTION
    # --------------------------------------------------------

    def pause_execution(
        self,
        execution_id: str,
        *,
        access: ExecutionAccessContext,
    ) -> dict[str, Any]:
        self._require_permission(
            access,
            "can_update_execution",
        )

        runtime = self._get_runtime(
            execution_id,
            access,
        )

        with runtime.lock:
            if runtime.status != ExecutionStatus.RUNNING:
                raise ExecutionStateError(
                    "Execution must be RUNNING"
                )

            runtime.status = ExecutionStatus.PAUSED

            self._event(
                runtime,
                ExecutionEventType.PAUSED,
                actor=access.user_id,
            )

            return runtime.snapshot()

    # --------------------------------------------------------
    # RESUME EXECUTION
    # --------------------------------------------------------

    def resume_execution(
        self,
        execution_id: str,
        *,
        access: ExecutionAccessContext,
    ) -> dict[str, Any]:
        self._require_permission(
            access,
            "can_update_execution",
        )

        runtime = self._get_runtime(
            execution_id,
            access,
        )

        with runtime.lock:
            if runtime.status != ExecutionStatus.PAUSED:
                raise ExecutionStateError(
                    "Execution must be PAUSED"
                )

            for robot_id in {
                task.robot_id
                for task in runtime.specification.tasks
            }:
                telemetry = runtime.telemetry.get(
                    robot_id
                )

                if telemetry is None:
                    raise ExecutionSafetyError(
                        "Robot telemetry missing"
                    )

                safety = self._safety.evaluate(
                    telemetry
                )

                if not safety.allowed:
                    raise ExecutionSafetyError(
                        "; ".join(safety.reasons)
                    )

            runtime.status = ExecutionStatus.RUNNING

            self._event(
                runtime,
                ExecutionEventType.RESUMED,
                actor=access.user_id,
            )

            return runtime.snapshot()

    # --------------------------------------------------------
    # CANCEL EXECUTION
    # --------------------------------------------------------

    def cancel_execution(
        self,
        execution_id: str,
        *,
        access: ExecutionAccessContext,
        reason: str = "Cancelled by operator",
    ) -> dict[str, Any]:
        self._require_permission(
            access,
            "can_cancel",
        )

        runtime = self._get_runtime(
            execution_id,
            access,
        )

        with runtime.lock:
            if runtime.status in self.TERMINAL_STATUSES:
                return runtime.snapshot()

            runtime.status = ExecutionStatus.CANCELLED
            runtime.completed_at = utc_now()
            runtime.last_error = reason

            for task in runtime.tasks.values():
                if task.status not in (
                    TaskExecutionStatus.COMPLETED,
                    TaskExecutionStatus.FAILED,
                ):
                    task.status = TaskExecutionStatus.CANCELLED
                    task.completed_at = utc_now()

            self._event(
                runtime,
                ExecutionEventType.CANCELLED,
                actor=access.user_id,
                details={
                    "reason": reason,
                },
            )

            self._metrics["cancelled"] += 1

            return runtime.snapshot()

    # --------------------------------------------------------
    # WATCHDOG / MONITORING CYCLE
    # --------------------------------------------------------

    def monitor_execution(
        self,
        execution_id: str,
        *,
        access: ExecutionAccessContext,
    ) -> dict[str, Any]:
        self._require_permission(
            access,
            "can_monitor",
        )

        runtime = self._get_runtime(
            execution_id,
            access,
        )

        with runtime.lock:
            if runtime.status != ExecutionStatus.RUNNING:
                return {
                    "execution_id": execution_id,
                    "status": runtime.status.value,
                    "alerts_created": 0,
                    "timestamp": utc_iso(),
                }

            now = utc_now()
            alert_count_before = len(runtime.alerts)

            if runtime.started_at is not None:
                elapsed = (
                    now - runtime.started_at
                ).total_seconds()

                if (
                    elapsed
                    > self.config.execution_timeout_seconds
                ):
                    runtime.status = ExecutionStatus.TIMED_OUT
                    runtime.completed_at = now
                    runtime.last_error = (
                        "Execution timeout exceeded"
                    )

                    self._alert(
                        runtime,
                        category=AlertCategory.EXECUTION,
                        severity=AlertSeverity.CRITICAL,
                        code="EXECUTION_TIMEOUT",
                        message=runtime.last_error,
                        recommended_action=(
                            "Investigate execution timeout"
                        ),
                    )

                    self._event(
                        runtime,
                        ExecutionEventType.TIMEOUT,
                        actor=access.user_id,
                    )

            for task in runtime.tasks.values():
                if task.status != TaskExecutionStatus.RUNNING:
                    continue

                if task.started_at is None:
                    continue

                elapsed = (
                    now - task.started_at
                ).total_seconds()

                task_limit = min(
                    self.config.task_timeout_seconds,
                    max(
                        task.specification.estimated_duration_seconds
                        * 2,
                        task.specification.estimated_duration_seconds
                        + 30,
                    ),
                )

                if elapsed > task_limit:
                    self._alert(
                        runtime,
                        category=AlertCategory.TASK,
                        severity=AlertSeverity.CRITICAL,
                        code="TASK_TIMEOUT",
                        message=(
                            f"Task {task.specification.task_id} "
                            "exceeded its execution timeout"
                        ),
                        recommended_action=(
                            "Review task state and "
                            "suspend execution if necessary"
                        ),
                        task_id=task.specification.task_id,
                        robot_id=task.specification.robot_id,
                    )

                    runtime.status = ExecutionStatus.TIMED_OUT
                    runtime.completed_at = now
                    runtime.last_error = (
                        "Task execution timeout exceeded"
                    )

            if self.config.enable_heartbeat_monitoring:
                robot_ids = {
                    task.robot_id
                    for task in runtime.specification.tasks
                }

                for robot_id in robot_ids:
                    last_heartbeat = runtime.heartbeats.get(
                        robot_id
                    )

                    if last_heartbeat is None:
                        overdue = True
                    else:
                        overdue = (
                            now - last_heartbeat
                        ).total_seconds() > (
                            self.config.heartbeat_timeout_seconds
                        )

                    if overdue:
                        self._alert(
                            runtime,
                            category=AlertCategory.HEARTBEAT,
                            severity=AlertSeverity.CRITICAL,
                            code="ROBOT_HEARTBEAT_TIMEOUT",
                            message=(
                                f"Robot {robot_id} heartbeat "
                                "missing or timed out"
                            ),
                            recommended_action=(
                                "Treat robot state as unknown; "
                                "verify communication and "
                                "independent safety systems"
                            ),
                            robot_id=robot_id,
                        )

                        runtime.status = (
                            ExecutionStatus.BLOCKED_SAFETY
                        )

                        runtime.last_error = (
                            "Robot heartbeat timeout"
                        )

                        self._event(
                            runtime,
                            ExecutionEventType.WATCHDOG_TIMEOUT,
                            actor=access.user_id,
                            robot_id=robot_id,
                        )

            return {
                "execution_id": execution_id,
                "status": runtime.status.value,
                "alerts_created": (
                    len(runtime.alerts)
                    - alert_count_before
                ),
                "progress_percentage": (
                    runtime.progress_percentage()
                ),
                "timestamp": utc_iso(),
            }

    # --------------------------------------------------------
    # GET EXECUTION
    # --------------------------------------------------------

    def get_execution(
        self,
        execution_id: str,
        *,
        access: ExecutionAccessContext,
    ) -> dict[str, Any]:
        self._require_permission(
            access,
            "can_monitor",
        )

        runtime = self._get_runtime(
            execution_id,
            access,
        )

        with runtime.lock:
            return runtime.snapshot()

    # --------------------------------------------------------
    # LIST EXECUTIONS
    # --------------------------------------------------------

    def list_executions(
        self,
        *,
        access: ExecutionAccessContext,
        status: ExecutionStatus | None = None,
    ) -> list[dict[str, Any]]:
        self._require_permission(
            access,
            "can_monitor",
        )

        with self._lock:
            runtimes = list(
                self._executions.values()
            )

        results = []

        for runtime in runtimes:
            if (
                runtime.specification.tenant_id
                != access.tenant_id
            ):
                continue

            if any(
                task.robot_id
                not in access.allowed_robot_ids
                for task in runtime.specification.tasks
            ):
                continue

            with runtime.lock:
                if (
                    status is not None
                    and runtime.status != status
                ):
                    continue

                results.append(
                    runtime.snapshot()
                )

        return results

    # --------------------------------------------------------
    # GET ALERTS
    # --------------------------------------------------------

    def get_alerts(
        self,
        execution_id: str,
        *,
        access: ExecutionAccessContext,
        severity: AlertSeverity | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        self._require_permission(
            access,
            "can_monitor",
        )

        runtime = self._get_runtime(
            execution_id,
            access,
        )

        if limit < 1:
            raise ExecutionValidationError(
                "Limit must be positive"
            )

        with runtime.lock:
            alerts = [
                alert
                for alert in runtime.alerts
                if (
                    severity is None
                    or alert.severity == severity
                )
            ]

            return serialize(
                alerts[-limit:]
            )

    # --------------------------------------------------------
    # EVENT HISTORY
    # --------------------------------------------------------

    def event_history(
        self,
        execution_id: str,
        *,
        access: ExecutionAccessContext,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        self._require_permission(
            access,
            "can_monitor",
        )

        runtime = self._get_runtime(
            execution_id,
            access,
        )

        if limit < 1:
            raise ExecutionValidationError(
                "Limit must be positive"
            )

        with runtime.lock:
            return serialize(
                list(runtime.events)[-limit:]
            )

    # --------------------------------------------------------
    # ROBOT TELEMETRY HISTORY
    # --------------------------------------------------------

    def telemetry_history(
        self,
        execution_id: str,
        robot_id: str,
        *,
        access: ExecutionAccessContext,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        self._require_permission(
            access,
            "can_monitor",
        )

        runtime = self._get_runtime(
            execution_id,
            access,
        )

        self._authorize_robot(
            access,
            runtime.specification.tenant_id,
            robot_id,
        )

        with runtime.lock:
            samples = runtime.telemetry_history.get(
                robot_id,
                (),
            )

            return serialize(
                list(samples)[-limit:]
            )

    # --------------------------------------------------------
    # EXECUTION HEALTH SUMMARY
    # --------------------------------------------------------

    def health_summary(
        self,
        execution_id: str,
        *,
        access: ExecutionAccessContext,
    ) -> dict[str, Any]:
        self._require_permission(
            access,
            "can_monitor",
        )

        runtime = self._get_runtime(
            execution_id,
            access,
        )

        with runtime.lock:
            severity_counts = Counter(
                alert.severity.value
                for alert in runtime.alerts
            )

            task_status_counts = Counter(
                task.status.value
                for task in runtime.tasks.values()
            )

            robot_health = {}

            for robot_id, telemetry in runtime.telemetry.items():
                safety = self._safety.evaluate(
                    telemetry
                )

                robot_health[robot_id] = {
                    "battery_percentage": (
                        telemetry.battery_percentage
                    ),
                    "temperature_c": (
                        telemetry.temperature_c
                    ),
                    "safety_state": (
                        telemetry.safety_state.value
                    ),
                    "active_faults": (
                        list(telemetry.active_faults)
                    ),
                    "safety_decision": (
                        safety.decision.value
                    ),
                    "telemetry_timestamp": (
                        telemetry.timestamp.isoformat()
                    ),
                }

            return {
                "execution_id": execution_id,
                "tenant_id": (
                    runtime.specification.tenant_id
                ),
                "status": runtime.status.value,
                "progress_percentage": (
                    runtime.progress_percentage()
                ),
                "task_statuses": dict(
                    task_status_counts
                ),
                "alert_severities": dict(
                    severity_counts
                ),
                "robot_health": robot_health,
                "hardware_execution": False,
                "timestamp": utc_iso(),
            }

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        access: ExecutionAccessContext,
    ) -> dict[str, Any]:
        executions = self.list_executions(
            access=access
        )

        return {
            "tenant_id": access.tenant_id,
            "total_executions": len(executions),
            "statuses": dict(
                Counter(
                    execution["status"]
                    for execution in executions
                )
            ),
            "total_tasks": sum(
                len(execution["tasks"])
                for execution in executions
            ),
            "average_progress_percentage": (
                statistics.mean(
                    execution["progress_percentage"]
                    for execution in executions
                )
                if executions
                else 0.0
            ),
            "simulation_only": True,
            "hardware_execution": False,
            "timestamp": utc_iso(),
        }

    # --------------------------------------------------------
    # JSON EXPORT
    # --------------------------------------------------------

    def export_execution_json(
        self,
        execution_id: str,
        *,
        access: ExecutionAccessContext,
    ) -> str:
        execution = self.get_execution(
            execution_id,
            access=access,
        )

        payload = {
            "platform": (
                "Physical AI & Autonomous Robotics "
                "Intelligence Platform"
            ),
            "module": "execution_monitor",
            "execution": execution,
            "health": self.health_summary(
                execution_id,
                access=access,
            ),
            "alerts": self.get_alerts(
                execution_id,
                access=access,
            ),
            "events": self.event_history(
                execution_id,
                access=access,
            ),
            "simulation_only": True,
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

class TestRoboticsExecutionMonitor(unittest.TestCase):

    def setUp(self) -> None:
        self.monitor = RoboticsExecutionMonitor(
            ExecutionMonitorConfig(
                heartbeat_timeout_seconds=10,
                robot_state_max_age_seconds=60,
            )
        )

        self.access = ExecutionAccessContext(
            tenant_id="tenant_alpha",
            user_id="robotics_engineer",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
            }),
            can_register=True,
            can_monitor=True,
            can_update_execution=True,
            can_update_telemetry=True,
            can_manage_approvals=True,
            can_cancel=True,
            can_delete=True,
        )

        self.task = MonitoredTask(
            task_id="TASK-001",
            name="Robot Inspection",
            robot_id="ROBOT-001",
            estimated_duration_seconds=120,
            required_capabilities=frozenset({
                "inspection",
            }),
            required_zone="ZONE-A",
        )

        self.specification = ExecutionSpecification(
            execution_id="EXEC-001",
            tenant_id="tenant_alpha",
            name="Inspection Execution",
            tasks=(self.task,),
        )

        self.telemetry = RobotExecutionTelemetry(
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            timestamp=utc_now(),
            battery_percentage=90,
            safety_state=RobotSafetyState.SAFE,
            temperature_c=35,
            capabilities=frozenset({
                "inspection",
            }),
            current_zone="ZONE-A",
        )

    def register_and_prepare(self) -> None:
        self.monitor.register_execution(
            self.specification,
            access=self.access,
        )

        self.monitor.ingest_telemetry(
            "EXEC-001",
            self.telemetry,
            access=self.access,
        )

    def test_register_execution(self) -> None:
        self.monitor.register_execution(
            self.specification,
            access=self.access,
        )

        result = self.monitor.get_execution(
            "EXEC-001",
            access=self.access,
        )

        self.assertEqual(
            result["status"],
            "ready",
        )

    def test_start_execution(self) -> None:
        self.register_and_prepare()

        result = self.monitor.start_execution(
            "EXEC-001",
            access=self.access,
        )

        self.assertEqual(
            result["status"],
            "running",
        )

    def test_complete_task(self) -> None:
        self.register_and_prepare()

        self.monitor.start_execution(
            "EXEC-001",
            access=self.access,
        )

        self.monitor.start_task(
            "EXEC-001",
            "TASK-001",
            access=self.access,
        )

        self.monitor.update_task_progress(
            "EXEC-001",
            "TASK-001",
            50,
            access=self.access,
        )

        result = self.monitor.complete_task(
            "EXEC-001",
            "TASK-001",
            access=self.access,
        )

        self.assertEqual(
            result["status"],
            "completed",
        )

    def test_unsafe_telemetry(self) -> None:
        self.monitor.register_execution(
            self.specification,
            access=self.access,
        )

        unsafe = RobotExecutionTelemetry(
            tenant_id="tenant_alpha",
            robot_id="ROBOT-001",
            timestamp=utc_now(),
            battery_percentage=90,
            safety_state=RobotSafetyState.UNSAFE,
            emergency_stop_active=True,
        )

        result = self.monitor.ingest_telemetry(
            "EXEC-001",
            unsafe,
            access=self.access,
        )

        self.assertFalse(
            result["safety"]["allowed"]
        )

    def test_cancel_execution(self) -> None:
        self.register_and_prepare()

        result = self.monitor.cancel_execution(
            "EXEC-001",
            access=self.access,
        )

        self.assertEqual(
            result["status"],
            "cancelled",
        )

    def test_tenant_isolation(self) -> None:
        self.register_and_prepare()

        unauthorized = ExecutionAccessContext(
            tenant_id="tenant_beta",
            user_id="other",
            allowed_robot_ids=frozenset({
                "ROBOT-001",
            }),
            can_monitor=True,
        )

        with self.assertRaises(
            ExecutionAuthorizationError
        ):
            self.monitor.get_execution(
                "EXEC-001",
                access=unauthorized,
            )

    def test_export_json(self) -> None:
        self.register_and_prepare()

        exported = self.monitor.export_execution_json(
            "EXEC-001",
            access=self.access,
        )

        self.assertIn(
            "EXEC-001",
            exported,
        )

        self.assertIn(
            "hardware_execution",
            exported,
        )

    def test_missing_approval(self) -> None:
        task = MonitoredTask(
            task_id="TASK-APPROVAL",
            name="Approved Inspection",
            robot_id="ROBOT-001",
            estimated_duration_seconds=60,
            required_capabilities=frozenset({
                "inspection",
            }),
            required_zone="ZONE-A",
            requires_human_approval=True,
            approval_id="APP-001",
        )

        specification = ExecutionSpecification(
            execution_id="EXEC-APPROVAL",
            tenant_id="tenant_alpha",
            name="Approval Test",
            tasks=(task,),
        )

        self.monitor.register_execution(
            specification,
            access=self.access,
        )

        self.monitor.ingest_telemetry(
            "EXEC-APPROVAL",
            self.telemetry,
            access=self.access,
        )

        self.monitor.start_execution(
            "EXEC-APPROVAL",
            access=self.access,
        )

        with self.assertRaises(
            ExecutionSafetyError
        ):
            self.monitor.start_task(
                "EXEC-APPROVAL",
                "TASK-APPROVAL",
                access=self.access,
            )

        self.monitor.grant_approval(
            "EXEC-APPROVAL",
            "APP-001",
            access=self.access,
        )

        result = self.monitor.start_task(
            "EXEC-APPROVAL",
            "TASK-APPROVAL",
            access=self.access,
        )

        self.assertEqual(
            result["status"],
            "running",
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    print(
        "\nPHYSICAL AI & AUTONOMOUS ROBOTICS PLATFORM"
    )

    print(
        "Execution Monitor — Simulation Mode\n"
    )

    monitor = RoboticsExecutionMonitor()

    access = ExecutionAccessContext(
        tenant_id="tenant_alpha",
        user_id="robotics_engineer",
        allowed_robot_ids=frozenset({
            "ROBOT-001",
            "ROBOT-002",
        }),
        roles=frozenset({
            "robotics_engineer",
        }),
        can_register=True,
        can_monitor=True,
        can_update_execution=True,
        can_update_telemetry=True,
        can_manage_approvals=True,
        can_cancel=True,
        can_delete=True,
    )

    tasks = (
        MonitoredTask(
            task_id="TASK-001",
            name="Visual Robot Inspection",
            robot_id="ROBOT-001",
            estimated_duration_seconds=120,
            required_capabilities=frozenset({
                "inspection",
            }),
            required_zone="ZONE-A",
            maximum_energy_wh=100,
        ),
        MonitoredTask(
            task_id="TASK-002",
            name="Robot Diagnostics",
            robot_id="ROBOT-002",
            estimated_duration_seconds=180,
            required_capabilities=frozenset({
                "diagnostics",
            }),
            required_zone="ZONE-B",
            dependencies=("TASK-001",),
            requires_human_approval=True,
            approval_id="APPROVAL-002",
            maximum_energy_wh=150,
        ),
    )

    specification = ExecutionSpecification(
        execution_id="EXECUTION-001",
        tenant_id="tenant_alpha",
        name="Autonomous Robot Inspection Workflow",
        tasks=tasks,
        plan_id="PLAN-001",
        behavior_tree_id="TREE-001",
        created_by=access.user_id,
    )

    monitor.register_execution(
        specification,
        access=access,
    )

    telemetry_1 = RobotExecutionTelemetry(
        tenant_id="tenant_alpha",
        robot_id="ROBOT-001",
        timestamp=utc_now(),
        battery_percentage=92,
        safety_state=RobotSafetyState.SAFE,
        temperature_c=38,
        capabilities=frozenset({
            "inspection",
        }),
        current_zone="ZONE-A",
    )

    telemetry_2 = RobotExecutionTelemetry(
        tenant_id="tenant_alpha",
        robot_id="ROBOT-002",
        timestamp=utc_now(),
        battery_percentage=88,
        safety_state=RobotSafetyState.SAFE,
        temperature_c=40,
        capabilities=frozenset({
            "diagnostics",
        }),
        current_zone="ZONE-B",
    )

    monitor.ingest_telemetry(
        specification.execution_id,
        telemetry_1,
        access=access,
    )

    monitor.ingest_telemetry(
        specification.execution_id,
        telemetry_2,
        access=access,
    )

    monitor.grant_approval(
        specification.execution_id,
        "APPROVAL-002",
        access=access,
    )

    monitor.start_execution(
        specification.execution_id,
        access=access,
    )

    monitor.start_task(
        specification.execution_id,
        "TASK-001",
        access=access,
    )

    monitor.update_task_progress(
        specification.execution_id,
        "TASK-001",
        50,
        energy_consumed_wh=30,
        access=access,
    )

    monitor.complete_task(
        specification.execution_id,
        "TASK-001",
        access=access,
    )

    monitor.start_task(
        specification.execution_id,
        "TASK-002",
        access=access,
    )

    monitor.update_task_progress(
        specification.execution_id,
        "TASK-002",
        100,
        energy_consumed_wh=75,
        access=access,
    )

    monitor.complete_task(
        specification.execution_id,
        "TASK-002",
        access=access,
    )

    print("EXECUTION STATUS\n")

    print(
        json.dumps(
            monitor.get_execution(
                specification.execution_id,
                access=access,
            ),
            indent=2,
        )
    )

    print("\nHEALTH SUMMARY\n")

    print(
        json.dumps(
            monitor.health_summary(
                specification.execution_id,
                access=access,
            ),
            indent=2,
        )
    )

    print("\nEXECUTION STATISTICS\n")

    print(
        json.dumps(
            monitor.statistics(
                access=access,
            ),
            indent=2,
        )
    )

    print("\nEVENT HISTORY\n")

    print(
        json.dumps(
            monitor.event_history(
                specification.execution_id,
                access=access,
            ),
            indent=2,
        )
    )

    print("\nUNIT TESTS\n")

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        TestRoboticsExecutionMonitor
    )

    unittest.TextTestRunner(
        verbosity=2,
    ).run(suite)


if __name__ == "__main__":
    main()
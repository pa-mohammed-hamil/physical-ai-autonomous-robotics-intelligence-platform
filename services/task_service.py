"""Task Service module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
    backend/app/services/task_service.py

Robotics Task Management and Orchestration Service

Python 3.11+
Standard library only.

Features:
    - Task creation and validation
    - Priority-based scheduling
    - Capability-aware robot assignment
    - Robot availability and health checks
    - Task dependencies
    - Task lifecycle management
    - Task progress updates
    - Task timeout detection
    - Retry scheduling
    - Task cancellation
    - Audit events
    - Fleet task statistics
    - JSON and CSV export

Safety:
    - Simulation/task coordination only
    - No direct hardware control
    - No motor or actuator commands
"""

from __future__ import annotations

import csv
import json
import math
import threading
import uuid

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================

class TaskServiceError(Exception):
    pass


class TaskValidationError(TaskServiceError):
    pass


class TaskNotFoundError(TaskServiceError):
    pass


class TaskStateError(TaskServiceError):
    pass


class RobotUnavailableError(TaskServiceError):
    pass


# ============================================================
# ENUMS
# ============================================================

class TaskType(str, Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PICK_AND_PLACE = "pick_and_place"
    INSPECTION = "inspection"
    TRANSPORT = "transport"
    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    CHARGING = "charging"
    PERCEPTION = "perception"
    DATA_COLLECTION = "data_collection"
    SIMULATION = "simulation"
    OTHER = "other"


class TaskStatus(str, Enum):
    PENDING = "pending"
    BLOCKED = "blocked"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class TaskPriority(int, Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskEventType(str, Enum):
    CREATED = "created"
    ASSIGNED = "assigned"
    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    RETRIED = "retried"
    BLOCKED = "blocked"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_time(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except (ValueError, AttributeError) as exc:
        raise TaskValidationError(
            "Invalid timestamp"
        ) from exc

    if result.tzinfo is None:
        raise TaskValidationError(
            "Timestamp must include timezone"
        )

    return result.astimezone(timezone.utc)


def require_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskValidationError(
            f"{name} is required"
        )
    return value.strip()


def finite(
    value: float,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise TaskValidationError(
            f"{name} must be finite"
        )

    result = float(value)

    if minimum is not None and result < minimum:
        raise TaskValidationError(
            f"{name} must be >= {minimum}"
        )

    if maximum is not None and result > maximum:
        raise TaskValidationError(
            f"{name} must be <= {maximum}"
        )

    return result


def json_copy(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(value, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise TaskValidationError(
            "Data must be JSON serializable"
        ) from exc


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class TaskServiceConfig:
    maximum_tasks: int = 100000
    maximum_events: int = 20000

    minimum_robot_battery_soc: float = 20.0
    minimum_robot_health_score: float = 70.0

    default_timeout_seconds: float = 1800.0
    default_max_retries: int = 2

    allow_hardware_control: bool = False

    def __post_init__(self) -> None:
        for name in ("maximum_tasks", "maximum_events"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise TaskValidationError(
                    f"{name} must be positive"
                )

        for name in (
            "minimum_robot_battery_soc",
            "minimum_robot_health_score",
        ):
            finite(
                getattr(self, name),
                name,
                minimum=0,
                maximum=100,
            )

        finite(
            self.default_timeout_seconds,
            "default_timeout_seconds",
            minimum=0.001,
        )

        if (
            isinstance(self.default_max_retries, bool)
            or not isinstance(self.default_max_retries, int)
            or self.default_max_retries < 0
        ):
            raise TaskValidationError(
                "default_max_retries must be nonnegative"
            )

        if self.allow_hardware_control:
            raise TaskValidationError(
                "Hardware control is disabled"
            )


# ============================================================
# ROBOT AVAILABILITY SNAPSHOT
# ============================================================

@dataclass(frozen=True)
class TaskRobot:
    robot_id: str
    capabilities: frozenset[str]

    status: str = "idle"

    battery_soc_percent: float = 100.0
    health_score: float = 100.0

    maintenance_required: bool = False
    repair_required: bool = False

    sensor_available: bool = True
    network_connected: bool = True

    def __post_init__(self) -> None:
        require_text(self.robot_id, "robot_id")

        finite(
            self.battery_soc_percent,
            "battery_soc_percent",
            minimum=0,
            maximum=100,
        )

        finite(
            self.health_score,
            "health_score",
            minimum=0,
            maximum=100,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "capabilities": sorted(self.capabilities),
            "status": self.status,
            "battery_soc_percent": (
                self.battery_soc_percent
            ),
            "health_score": self.health_score,
            "maintenance_required": (
                self.maintenance_required
            ),
            "repair_required": self.repair_required,
            "sensor_available": self.sensor_available,
            "network_connected": self.network_connected,
        }


# ============================================================
# TASK ENTITY
# ============================================================

@dataclass
class RoboticsTask:
    task_id: str
    name: str
    task_type: TaskType

    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING

    robot_id: str | None = None

    required_capabilities: set[str] = field(
        default_factory=set
    )

    dependencies: set[str] = field(
        default_factory=set
    )

    payload: dict[str, Any] = field(
        default_factory=dict
    )

    result: dict[str, Any] | None = None

    progress_percent: float = 0.0

    timeout_seconds: float = 1800.0

    max_retries: int = 2
    retry_count: int = 0

    created_at: str = field(default_factory=utc_now)
    assigned_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    updated_at: str = field(default_factory=utc_now)

    error: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "task_type": self.task_type.value,
            "priority": self.priority.name.lower(),
            "status": self.status.value,
            "robot_id": self.robot_id,
            "required_capabilities": sorted(
                self.required_capabilities
            ),
            "dependencies": sorted(
                self.dependencies
            ),
            "payload": self.payload,
            "result": self.result,
            "progress_percent": self.progress_percent,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "created_at": self.created_at,
            "assigned_at": self.assigned_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "metadata": self.metadata,
        }


# ============================================================
# TASK EVENT
# ============================================================

@dataclass(frozen=True)
class TaskEvent:
    event_id: str
    task_id: str
    event_type: TaskEventType
    timestamp: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "task_id": self.task_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "details": self.details,
        }


# ============================================================
# TASK SERVICE
# ============================================================

class RoboticsTaskService:

    TERMINAL_STATES = {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.TIMED_OUT,
    }

    def __init__(
        self,
        config: TaskServiceConfig | None = None,
    ) -> None:
        self.config = config or TaskServiceConfig()

        self.tasks: dict[str, RoboticsTask] = {}
        self.robots: dict[str, TaskRobot] = {}

        self.events: deque[TaskEvent] = deque(
            maxlen=self.config.maximum_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # EVENT RECORDING
    # --------------------------------------------------------

    def _event(
        self,
        task_id: str,
        event_type: TaskEventType,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.events.append(
            TaskEvent(
                event_id=str(uuid.uuid4()),
                task_id=task_id,
                event_type=event_type,
                timestamp=utc_now(),
                details=json_copy(dict(details or {})),
            )
        )

    # --------------------------------------------------------
    # REGISTER ROBOT SNAPSHOT
    # --------------------------------------------------------

    def register_robot(
        self,
        robot: TaskRobot,
    ) -> None:
        with self._lock:
            self.robots[robot.robot_id] = robot

    # --------------------------------------------------------
    # UPDATE ROBOT SNAPSHOT
    # --------------------------------------------------------

    def update_robot(
        self,
        robot: TaskRobot,
    ) -> None:
        self.register_robot(robot)

    # --------------------------------------------------------
    # CREATE TASK
    # --------------------------------------------------------

    def create_task(
        self,
        *,
        name: str,
        task_type: TaskType,
        priority: TaskPriority = TaskPriority.NORMAL,
        required_capabilities: set[str] | None = None,
        dependencies: set[str] | None = None,
        payload: Mapping[str, Any] | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RoboticsTask:
        name = require_text(name, "name")

        timeout = (
            self.config.default_timeout_seconds
            if timeout_seconds is None
            else finite(
                timeout_seconds,
                "timeout_seconds",
                minimum=0.001,
            )
        )

        retries = (
            self.config.default_max_retries
            if max_retries is None
            else max_retries
        )

        if (
            isinstance(retries, bool)
            or not isinstance(retries, int)
            or retries < 0
        ):
            raise TaskValidationError(
                "max_retries must be nonnegative"
            )

        with self._lock:
            if len(self.tasks) >= self.config.maximum_tasks:
                raise TaskStateError(
                    "Maximum task capacity reached"
                )

            deps = set(dependencies or ())

            missing = deps - self.tasks.keys()

            if missing:
                raise TaskValidationError(
                    f"Unknown dependencies: {sorted(missing)}"
                )

            task = RoboticsTask(
                task_id=str(uuid.uuid4()),
                name=name,
                task_type=TaskType(task_type),
                priority=TaskPriority(priority),
                required_capabilities=set(
                    required_capabilities or ()
                ),
                dependencies=deps,
                payload=json_copy(dict(payload or {})),
                timeout_seconds=timeout,
                max_retries=retries,
                metadata=json_copy(dict(metadata or {})),
            )

            if not self._dependencies_complete(task):
                task.status = TaskStatus.BLOCKED

            self.tasks[task.task_id] = task

            self._event(
                task.task_id,
                TaskEventType.CREATED,
                {"name": name},
            )

            return task

    # --------------------------------------------------------
    # GET TASK
    # --------------------------------------------------------

    def get_task(
        self,
        task_id: str,
    ) -> RoboticsTask:
        task = self.tasks.get(task_id)

        if task is None:
            raise TaskNotFoundError(
                f"Task not found: {task_id}"
            )

        return task

    # --------------------------------------------------------
    # DEPENDENCY CHECK
    # --------------------------------------------------------

    def _dependencies_complete(
        self,
        task: RoboticsTask,
    ) -> bool:
        return all(
            self.tasks[dependency].status
            == TaskStatus.COMPLETED
            for dependency in task.dependencies
        )

    # --------------------------------------------------------
    # REFRESH BLOCKED TASKS
    # --------------------------------------------------------

    def refresh_dependencies(self) -> int:
        changed = 0

        with self._lock:
            for task in self.tasks.values():
                if task.status != TaskStatus.BLOCKED:
                    continue

                if self._dependencies_complete(task):
                    task.status = TaskStatus.PENDING
                    task.updated_at = utc_now()
                    changed += 1

        return changed

    # --------------------------------------------------------
    # ROBOT AVAILABILITY
    # --------------------------------------------------------

    def robot_available(
        self,
        robot_id: str,
        task: RoboticsTask,
    ) -> tuple[bool, str]:
        robot = self.robots.get(robot_id)

        if robot is None:
            return False, "Robot not registered"

        if robot.status.lower() != "idle":
            return False, "Robot not idle"

        if robot.maintenance_required:
            return False, "Maintenance required"

        if robot.repair_required:
            return False, "Repair required"

        if not robot.sensor_available:
            return False, "Sensors unavailable"

        if not robot.network_connected:
            return False, "Network unavailable"

        if (
            robot.battery_soc_percent
            < self.config.minimum_robot_battery_soc
        ):
            return False, "Battery below threshold"

        if (
            robot.health_score
            < self.config.minimum_robot_health_score
        ):
            return False, "Health below threshold"

        if not task.required_capabilities.issubset(
            robot.capabilities
        ):
            return False, "Missing capabilities"

        for other in self.tasks.values():
            if (
                other.robot_id == robot_id
                and other.status in (
                    TaskStatus.ASSIGNED,
                    TaskStatus.RUNNING,
                )
            ):
                return False, "Robot already has active task"

        return True, "Available"

    # --------------------------------------------------------
    # ASSIGN TASK
    # --------------------------------------------------------

    def assign_task(
        self,
        task_id: str,
        robot_id: str,
    ) -> RoboticsTask:
        with self._lock:
            task = self.get_task(task_id)

            if task.status != TaskStatus.PENDING:
                raise TaskStateError(
                    "Task is not pending"
                )

            if not self._dependencies_complete(task):
                task.status = TaskStatus.BLOCKED
                raise TaskStateError(
                    "Dependencies incomplete"
                )

            available, reason = self.robot_available(
                robot_id,
                task,
            )

            if not available:
                raise RobotUnavailableError(reason)

            task.robot_id = robot_id
            task.status = TaskStatus.ASSIGNED
            task.assigned_at = utc_now()
            task.updated_at = utc_now()

            self._event(
                task_id,
                TaskEventType.ASSIGNED,
                {"robot_id": robot_id},
            )

            return task

    # --------------------------------------------------------
    # PRIORITY SCHEDULER
    # --------------------------------------------------------

    def schedule_pending(self) -> list[str]:
        assigned: list[str] = []

        with self._lock:
            self.refresh_dependencies()

            pending = [
                task
                for task in self.tasks.values()
                if task.status == TaskStatus.PENDING
            ]

            pending.sort(
                key=lambda task: (
                    -task.priority.value,
                    task.created_at,
                    task.task_id,
                )
            )

            for task in pending:
                for robot_id in sorted(self.robots):
                    available, _ = self.robot_available(
                        robot_id,
                        task,
                    )

                    if available:
                        self.assign_task(
                            task.task_id,
                            robot_id,
                        )

                        assigned.append(task.task_id)
                        break

        return assigned

    # --------------------------------------------------------
    # START TASK
    # --------------------------------------------------------

    def start_task(
        self,
        task_id: str,
    ) -> RoboticsTask:
        with self._lock:
            task = self.get_task(task_id)

            if task.status != TaskStatus.ASSIGNED:
                raise TaskStateError(
                    "Task must be assigned"
                )

            if task.robot_id is None:
                raise TaskStateError(
                    "No robot assigned"
                )

            task.status = TaskStatus.RUNNING
            task.started_at = utc_now()
            task.updated_at = utc_now()

            self._event(
                task_id,
                TaskEventType.STARTED,
                {"robot_id": task.robot_id},
            )

            return task

    # --------------------------------------------------------
    # UPDATE PROGRESS
    # --------------------------------------------------------

    def update_progress(
        self,
        task_id: str,
        progress_percent: float,
    ) -> RoboticsTask:
        progress = finite(
            progress_percent,
            "progress_percent",
            minimum=0,
            maximum=100,
        )

        with self._lock:
            task = self.get_task(task_id)

            if task.status != TaskStatus.RUNNING:
                raise TaskStateError(
                    "Task is not running"
                )

            if progress < task.progress_percent:
                raise TaskValidationError(
                    "Progress cannot decrease"
                )

            task.progress_percent = progress
            task.updated_at = utc_now()

            self._event(
                task_id,
                TaskEventType.PROGRESS,
                {"progress_percent": progress},
            )

            return task

    # --------------------------------------------------------
    # COMPLETE TASK
    # --------------------------------------------------------

    def complete_task(
        self,
        task_id: str,
        *,
        result: Mapping[str, Any] | None = None,
    ) -> RoboticsTask:
        with self._lock:
            task = self.get_task(task_id)

            if task.status != TaskStatus.RUNNING:
                raise TaskStateError(
                    "Task is not running"
                )

            task.status = TaskStatus.COMPLETED
            task.progress_percent = 100.0
            task.result = json_copy(
                dict(result or {})
            )

            task.completed_at = utc_now()
            task.updated_at = utc_now()

            self._event(
                task_id,
                TaskEventType.COMPLETED,
            )

            self.refresh_dependencies()

            return task

    # --------------------------------------------------------
    # FAIL TASK
    # --------------------------------------------------------

    def fail_task(
        self,
        task_id: str,
        *,
        error: str,
    ) -> RoboticsTask:
        error = require_text(error, "error")

        with self._lock:
            task = self.get_task(task_id)

            if task.status not in (
                TaskStatus.ASSIGNED,
                TaskStatus.RUNNING,
            ):
                raise TaskStateError(
                    "Task cannot fail from current state"
                )

            task.status = TaskStatus.FAILED
            task.error = error
            task.completed_at = utc_now()
            task.updated_at = utc_now()

            self._event(
                task_id,
                TaskEventType.FAILED,
                {"error": error},
            )

            return task

    # --------------------------------------------------------
    # RETRY TASK
    # --------------------------------------------------------

    def retry_task(
        self,
        task_id: str,
    ) -> RoboticsTask:
        with self._lock:
            task = self.get_task(task_id)

            if task.status not in (
                TaskStatus.FAILED,
                TaskStatus.TIMED_OUT,
            ):
                raise TaskStateError(
                    "Only failed or timed-out tasks can retry"
                )

            if task.retry_count >= task.max_retries:
                raise TaskStateError(
                    "Maximum retries reached"
                )

            task.retry_count += 1
            task.status = (
                TaskStatus.PENDING
                if self._dependencies_complete(task)
                else TaskStatus.BLOCKED
            )

            task.robot_id = None
            task.progress_percent = 0.0
            task.assigned_at = None
            task.started_at = None
            task.completed_at = None
            task.error = None
            task.result = None
            task.updated_at = utc_now()

            self._event(
                task_id,
                TaskEventType.RETRIED,
                {"retry_count": task.retry_count},
            )

            return task

    # --------------------------------------------------------
    # CANCEL TASK
    # --------------------------------------------------------

    def cancel_task(
        self,
        task_id: str,
        *,
        reason: str,
    ) -> RoboticsTask:
        reason = require_text(reason, "reason")

        with self._lock:
            task = self.get_task(task_id)

            if task.status in self.TERMINAL_STATES:
                raise TaskStateError(
                    "Task is already terminal"
                )

            task.status = TaskStatus.CANCELLED
            task.error = reason
            task.completed_at = utc_now()
            task.updated_at = utc_now()

            self._event(
                task_id,
                TaskEventType.CANCELLED,
                {"reason": reason},
            )

            return task

    # --------------------------------------------------------
    # CHECK TIMEOUTS
    # --------------------------------------------------------

    def check_timeouts(
        self,
        *,
        now: datetime | None = None,
    ) -> list[str]:
        current = now or datetime.now(
            timezone.utc
        )

        if current.tzinfo is None:
            raise TaskValidationError(
                "now must be timezone-aware"
            )

        timed_out = []

        with self._lock:
            for task in self.tasks.values():
                if (
                    task.status != TaskStatus.RUNNING
                    or task.started_at is None
                ):
                    continue

                elapsed = (
                    current - parse_time(task.started_at)
                ).total_seconds()

                if elapsed >= task.timeout_seconds:
                    task.status = TaskStatus.TIMED_OUT
                    task.error = "Execution timeout"
                    task.completed_at = current.isoformat()
                    task.updated_at = current.isoformat()

                    timed_out.append(task.task_id)

                    self._event(
                        task.task_id,
                        TaskEventType.TIMED_OUT,
                        {"elapsed_seconds": elapsed},
                    )

        return timed_out

    # --------------------------------------------------------
    # LIST TASKS
    # --------------------------------------------------------

    def list_tasks(
        self,
        *,
        status: TaskStatus | None = None,
        robot_id: str | None = None,
        task_type: TaskType | None = None,
    ) -> list[RoboticsTask]:
        with self._lock:
            tasks = list(self.tasks.values())

            if status is not None:
                tasks = [
                    task
                    for task in tasks
                    if task.status == TaskStatus(status)
                ]

            if robot_id is not None:
                tasks = [
                    task
                    for task in tasks
                    if task.robot_id == robot_id
                ]

            if task_type is not None:
                tasks = [
                    task
                    for task in tasks
                    if task.task_type == TaskType(task_type)
                ]

            return sorted(
                tasks,
                key=lambda task: (
                    -task.priority.value,
                    task.created_at,
                ),
            )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(self) -> dict[str, Any]:
        with self._lock:
            tasks = list(self.tasks.values())

            completed = sum(
                task.status == TaskStatus.COMPLETED
                for task in tasks
            )

            failed = sum(
                task.status == TaskStatus.FAILED
                for task in tasks
            )

            finished = sum(
                task.status in self.TERMINAL_STATES
                for task in tasks
            )

            durations = []

            for task in tasks:
                if (
                    task.status == TaskStatus.COMPLETED
                    and task.started_at
                    and task.completed_at
                ):
                    duration = (
                        parse_time(task.completed_at)
                        - parse_time(task.started_at)
                    ).total_seconds()

                    durations.append(duration)

            return {
                "total_tasks": len(tasks),
                "registered_robots": len(self.robots),
                "status_distribution": dict(
                    Counter(
                        task.status.value
                        for task in tasks
                    )
                ),
                "type_distribution": dict(
                    Counter(
                        task.task_type.value
                        for task in tasks
                    )
                ),
                "priority_distribution": dict(
                    Counter(
                        task.priority.name.lower()
                        for task in tasks
                    )
                ),
                "completed_tasks": completed,
                "failed_tasks": failed,
                "terminal_tasks": finished,
                "completion_rate_percent": (
                    round(
                        completed / finished * 100,
                        2,
                    )
                    if finished
                    else None
                ),
                "average_completed_duration_seconds": (
                    round(
                        sum(durations) / len(durations),
                        2,
                    )
                    if durations
                    else None
                ),
                "total_retries": sum(
                    task.retry_count
                    for task in tasks
                ),
                "simulation_only": True,
            }

    # --------------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------------

    def dashboard(self) -> dict[str, Any]:
        with self._lock:
            return {
                "generated_at": utc_now(),
                "statistics": self.statistics(),
                "tasks": [
                    task.to_dict()
                    for task in self.list_tasks()
                ],
                "recent_events": [
                    event.to_dict()
                    for event in list(self.events)[-100:]
                ],
            }

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        path: str | Path,
    ) -> Path:
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = self.dashboard()

        output.write_text(
            json.dumps(
                payload,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        return output

    # --------------------------------------------------------
    # EXPORT CSV
    # --------------------------------------------------------

    def export_csv(
        self,
        path: str | Path,
    ) -> Path:
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = [
            "task_id",
            "name",
            "task_type",
            "priority",
            "status",
            "robot_id",
            "progress_percent",
            "timeout_seconds",
            "retry_count",
            "created_at",
            "assigned_at",
            "started_at",
            "completed_at",
            "error",
        ]

        with self._lock:
            rows = [
                task.to_dict()
                for task in self.list_tasks()
            ]

        with output.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )

            writer.writeheader()
            writer.writerows(rows)

        return output


# ============================================================
# ROBOT SERVICE ADAPTER
# ============================================================

class RobotTaskServiceAdapter:
    """
    Convert robot_service.py Robot objects into
    task scheduler availability snapshots.
    """

    @staticmethod
    def from_robot(
        robot: Any,
    ) -> TaskRobot:
        status = getattr(
            robot.status,
            "value",
            robot.status,
        )

        return TaskRobot(
            robot_id=robot.robot_id,
            capabilities=frozenset(
                robot.capabilities
            ),
            status=str(status),
            battery_soc_percent=(
                robot.battery_soc_percent
            ),
            health_score=robot.health_score,
            maintenance_required=(
                robot.maintenance_required
            ),
            repair_required=robot.repair_required,
            sensor_available=robot.sensor_available,
            network_connected=robot.network_connected,
        )

    @staticmethod
    def synchronize(
        task_service: RoboticsTaskService,
        robot_service: Any,
    ) -> int:
        count = 0

        for robot in robot_service.list_robots():
            task_service.update_robot(
                RobotTaskServiceAdapter.from_robot(
                    robot
                )
            )
            count += 1

        return count


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    service = RoboticsTaskService()

    service.register_robot(
        TaskRobot(
            robot_id="robot_001",
            capabilities=frozenset({
                "navigation",
                "transport",
                "inspection",
            }),
            status="idle",
            battery_soc_percent=92.0,
            health_score=95.0,
        )
    )

    service.register_robot(
        TaskRobot(
            robot_id="robot_002",
            capabilities=frozenset({
                "navigation",
                "manipulation",
                "inspection",
            }),
            status="idle",
            battery_soc_percent=85.0,
            health_score=90.0,
        )
    )

    navigation_task = service.create_task(
        name="Navigate to warehouse shelf",
        task_type=TaskType.NAVIGATION,
        priority=TaskPriority.HIGH,
        required_capabilities={"navigation"},
        payload={
            "goal_x": 12.0,
            "goal_y": 8.0,
        },
    )

    inspection_task = service.create_task(
        name="Inspect warehouse shelf",
        task_type=TaskType.INSPECTION,
        priority=TaskPriority.NORMAL,
        required_capabilities={"inspection"},
        dependencies={navigation_task.task_id},
    )

    service.schedule_pending()

    service.start_task(
        navigation_task.task_id
    )

    service.update_progress(
        navigation_task.task_id,
        50.0,
    )

    service.complete_task(
        navigation_task.task_id,
        result={
            "goal_reached": True,
            "distance_m": 14.2,
        },
    )

    service.schedule_pending()

    service.start_task(
        inspection_task.task_id
    )

    service.complete_task(
        inspection_task.task_id,
        result={
            "inspection_completed": True,
            "defects_detected": 0,
        },
    )

    json_path = service.export_json(
        "data/tasks/task_report.json"
    )

    csv_path = service.export_csv(
        "data/tasks/task_inventory.csv"
    )

    print(
        json.dumps(
            {
                "statistics": service.statistics(),
                "files": [
                    str(json_path),
                    str(csv_path),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

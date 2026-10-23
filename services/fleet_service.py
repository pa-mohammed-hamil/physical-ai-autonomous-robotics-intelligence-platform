"""Fleet Service module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
    backend/app/services/fleet_service.py

Robotics Fleet Management Service

Python 3.11+
Standard library only.

Features:
    - Register and manage multiple robots
    - Fleet-wide robot inventory
    - Robot status and availability
    - Battery and health monitoring
    - Real-time telemetry updates
    - Task assignment and lifecycle
    - Maintenance scheduling
    - Robot capability matching
    - Fleet utilization analytics
    - Fleet health scoring
    - Fleet alerts
    - Robot location tracking
    - Dashboard-ready summaries
    - Fleet state JSON persistence
    - Simulation integration

Safety:
    Advisory fleet management only.
    Does not issue physical robot commands.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import uuid

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class FleetServiceError(Exception):
    pass


class FleetValidationError(FleetServiceError):
    pass


class RobotNotFoundError(FleetServiceError):
    pass


class RobotAlreadyExistsError(FleetServiceError):
    pass


class FleetTaskNotFoundError(FleetServiceError):
    pass


class FleetStateError(FleetServiceError):
    pass


# ============================================================
# ENUMS
# ============================================================

class RobotStatus(str, Enum):
    IDLE = "idle"
    WORKING = "working"
    NAVIGATING = "navigating"
    CHARGING = "charging"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    FAULT = "fault"
    EMERGENCY = "emergency"


class RobotHealthStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


class FleetTaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FleetPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def validate_number(
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
        raise FleetValidationError(
            f"{name} must be finite"
        )

    result = float(value)

    if minimum is not None and result < minimum:
        raise FleetValidationError(
            f"{name} must be >= {minimum}"
        )

    if maximum is not None and result > maximum:
        raise FleetValidationError(
            f"{name} must be <= {maximum}"
        )

    return result


def calculate_health_status(
    score: float,
) -> RobotHealthStatus:
    if score < 40:
        return RobotHealthStatus.CRITICAL

    if score < 70:
        return RobotHealthStatus.WARNING

    return RobotHealthStatus.HEALTHY


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class FleetServiceConfig:
    maximum_robots: int = 10000

    minimum_dispatch_battery_soc: float = 20.0

    minimum_dispatch_health_score: float = 70.0

    low_battery_threshold: float = 20.0

    critical_battery_threshold: float = 10.0

    maximum_event_history: int = 10000

    allow_hardware_control: bool = False

    def __post_init__(self) -> None:
        if self.maximum_robots < 1:
            raise FleetValidationError(
                "maximum_robots must be positive"
            )

        if self.maximum_event_history < 1:
            raise FleetValidationError(
                "maximum_event_history must be positive"
            )

        for name in (
            "minimum_dispatch_battery_soc",
            "minimum_dispatch_health_score",
            "low_battery_threshold",
            "critical_battery_threshold",
        ):
            validate_number(
                getattr(self, name),
                name,
                minimum=0.0,
                maximum=100.0,
            )

        if (
            self.critical_battery_threshold
            >= self.low_battery_threshold
        ):
            raise FleetValidationError(
                "Critical battery threshold must "
                "be below low battery threshold"
            )

        if self.allow_hardware_control:
            raise FleetValidationError(
                "Physical robot control is prohibited"
            )


# ============================================================
# ROBOT POSITION
# ============================================================

@dataclass
class RobotPosition:
    x_m: float = 0.0
    y_m: float = 0.0
    heading_rad: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "x_m": self.x_m,
            "y_m": self.y_m,
            "heading_rad": self.heading_rad,
        }


# ============================================================
# ROBOT MODEL
# ============================================================

@dataclass
class FleetRobot:
    robot_id: str
    name: str
    robot_type: str

    status: RobotStatus = RobotStatus.IDLE

    health_score: float = 100.0

    battery_soc_percent: float = 100.0

    motor_temperature_c: float = 25.0

    vibration_mm_s: float = 0.0

    position: RobotPosition = field(
        default_factory=RobotPosition
    )

    capabilities: set[str] = field(
        default_factory=set
    )

    completed_tasks: int = 0
    failed_tasks: int = 0

    operating_hours: float = 0.0
    downtime_hours: float = 0.0

    active_task_id: str | None = None

    maintenance_required: bool = False

    registered_at: str = field(
        default_factory=utc_now
    )

    updated_at: str = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def health_status(self) -> RobotHealthStatus:
        return calculate_health_status(
            self.health_score
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "name": self.name,
            "robot_type": self.robot_type,
            "status": self.status.value,
            "health_score": self.health_score,
            "health_status": self.health_status.value,
            "battery_soc_percent": (
                self.battery_soc_percent
            ),
            "motor_temperature_c": (
                self.motor_temperature_c
            ),
            "vibration_mm_s": self.vibration_mm_s,
            "position": self.position.to_dict(),
            "capabilities": sorted(
                self.capabilities
            ),
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "operating_hours": self.operating_hours,
            "downtime_hours": self.downtime_hours,
            "active_task_id": self.active_task_id,
            "maintenance_required": (
                self.maintenance_required
            ),
            "registered_at": self.registered_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


# ============================================================
# FLEET TASK MODEL
# ============================================================

@dataclass
class FleetTask:
    task_id: str
    name: str

    required_capabilities: set[str]

    priority: FleetPriority = (
        FleetPriority.NORMAL
    )

    status: FleetTaskStatus = (
        FleetTaskStatus.PENDING
    )

    assigned_robot_id: str | None = None

    created_at: str = field(
        default_factory=utc_now
    )

    started_at: str | None = None
    completed_at: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "required_capabilities": sorted(
                self.required_capabilities
            ),
            "priority": self.priority.value,
            "status": self.status.value,
            "assigned_robot_id": (
                self.assigned_robot_id
            ),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }


# ============================================================
# TELEMETRY MODEL
# ============================================================

@dataclass(frozen=True)
class FleetRobotTelemetry:
    robot_id: str

    battery_soc_percent: float
    motor_temperature_c: float
    vibration_mm_s: float

    health_score: float

    x_m: float
    y_m: float
    heading_rad: float

    operating_hours: float | None = None
    downtime_hours: float | None = None

    def __post_init__(self) -> None:
        if not self.robot_id.strip():
            raise FleetValidationError(
                "robot_id is required"
            )

        for name in (
            "battery_soc_percent",
            "health_score",
        ):
            validate_number(
                getattr(self, name),
                name,
                minimum=0.0,
                maximum=100.0,
            )

        for name in (
            "motor_temperature_c",
            "vibration_mm_s",
            "x_m",
            "y_m",
            "heading_rad",
        ):
            validate_number(
                getattr(self, name),
                name,
            )

        if self.vibration_mm_s < 0:
            raise FleetValidationError(
                "Vibration cannot be negative"
            )

        for name in (
            "operating_hours",
            "downtime_hours",
        ):
            value = getattr(self, name)

            if value is not None:
                validate_number(
                    value,
                    name,
                    minimum=0.0,
                )


# ============================================================
# ROBOT DISPATCH POLICY
# ============================================================

class FleetDispatchPolicy:

    def __init__(
        self,
        config: FleetServiceConfig,
    ) -> None:
        self.config = config

    def is_available(
        self,
        robot: FleetRobot,
    ) -> bool:
        return (
            robot.status == RobotStatus.IDLE
            and robot.active_task_id is None
            and not robot.maintenance_required
            and robot.battery_soc_percent
            >= self.config.minimum_dispatch_battery_soc
            and robot.health_score
            >= self.config.minimum_dispatch_health_score
        )

    def matches(
        self,
        robot: FleetRobot,
        task: FleetTask,
    ) -> bool:
        return (
            self.is_available(robot)
            and task.required_capabilities.issubset(
                robot.capabilities
            )
        )

    def rank(
        self,
        robot: FleetRobot,
    ) -> float:
        return (
            robot.health_score * 0.5
            + robot.battery_soc_percent * 0.4
            - robot.operating_hours * 0.001
        )


# ============================================================
# FLEET SERVICE
# ============================================================

class RoboticsFleetService:

    def __init__(
        self,
        config: FleetServiceConfig | None = None,
    ) -> None:
        self.config = (
            config or FleetServiceConfig()
        )

        self.robots: dict[
            str,
            FleetRobot
        ] = {}

        self.tasks: dict[
            str,
            FleetTask
        ] = {}

        self.dispatch_policy = (
            FleetDispatchPolicy(
                self.config
            )
        )

        self.events: deque[
            dict[str, Any]
        ] = deque(
            maxlen=self.config.maximum_event_history
        )

        self.callbacks: list[
            Callable[[dict[str, Any]], Any]
        ] = []

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # EVENTS
    # --------------------------------------------------------

    def register_callback(
        self,
        callback: Callable[
            [dict[str, Any]],
            Any
        ],
    ) -> None:
        self.callbacks.append(
            callback
        )

    def _emit(
        self,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "timestamp": utc_now(),
            "payload": dict(payload),
        }

        self.events.append(
            event
        )

        for callback in self.callbacks:
            try:
                callback(event)
            except Exception:
                pass

    # --------------------------------------------------------
    # ROBOT REGISTRATION
    # --------------------------------------------------------

    def register_robot(
        self,
        *,
        robot_id: str,
        name: str,
        robot_type: str,
        capabilities: Sequence[str] = (),
        position: RobotPosition | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> FleetRobot:
        with self._lock:
            if not robot_id.strip():
                raise FleetValidationError(
                    "robot_id is required"
                )

            if robot_id in self.robots:
                raise RobotAlreadyExistsError(
                    f"Robot already exists: {robot_id}"
                )

            if (
                len(self.robots)
                >= self.config.maximum_robots
            ):
                raise FleetServiceError(
                    "Maximum fleet capacity reached"
                )

            robot = FleetRobot(
                robot_id=robot_id,
                name=name,
                robot_type=robot_type,
                capabilities=set(
                    capabilities
                ),
                position=(
                    position or RobotPosition()
                ),
                metadata=dict(
                    metadata or {}
                ),
            )

            self.robots[
                robot_id
            ] = robot

            self._emit(
                "robot_registered",
                {
                    "robot_id": robot_id,
                    "robot_type": robot_type,
                },
            )

            return robot

    # --------------------------------------------------------
    # GET ROBOT
    # --------------------------------------------------------

    def get_robot(
        self,
        robot_id: str,
    ) -> FleetRobot:
        robot = self.robots.get(
            robot_id
        )

        if robot is None:
            raise RobotNotFoundError(
                f"Robot not found: {robot_id}"
            )

        return robot

    # --------------------------------------------------------
    # LIST ROBOTS
    # --------------------------------------------------------

    def list_robots(
        self,
        *,
        status: RobotStatus | None = None,
        robot_type: str | None = None,
        capability: str | None = None,
    ) -> list[FleetRobot]:
        robots = list(
            self.robots.values()
        )

        if status is not None:
            robots = [
                robot
                for robot in robots
                if robot.status == status
            ]

        if robot_type is not None:
            robots = [
                robot
                for robot in robots
                if robot.robot_type == robot_type
            ]

        if capability is not None:
            robots = [
                robot
                for robot in robots
                if capability in robot.capabilities
            ]

        return robots

    # --------------------------------------------------------
    # UPDATE STATUS
    # --------------------------------------------------------

    def update_robot_status(
        self,
        robot_id: str,
        status: RobotStatus,
    ) -> FleetRobot:
        with self._lock:
            robot = self.get_robot(
                robot_id
            )

            robot.status = RobotStatus(
                status
            )

            robot.updated_at = utc_now()

            self._emit(
                "robot_status_updated",
                {
                    "robot_id": robot_id,
                    "status": robot.status.value,
                },
            )

            return robot

    # --------------------------------------------------------
    # UPDATE TELEMETRY
    # --------------------------------------------------------

    def update_telemetry(
        self,
        telemetry: FleetRobotTelemetry,
    ) -> FleetRobot:
        with self._lock:
            robot = self.get_robot(
                telemetry.robot_id
            )

            robot.battery_soc_percent = (
                telemetry.battery_soc_percent
            )

            robot.motor_temperature_c = (
                telemetry.motor_temperature_c
            )

            robot.vibration_mm_s = (
                telemetry.vibration_mm_s
            )

            robot.health_score = (
                telemetry.health_score
            )

            robot.position = RobotPosition(
                x_m=telemetry.x_m,
                y_m=telemetry.y_m,
                heading_rad=telemetry.heading_rad,
            )

            if telemetry.operating_hours is not None:
                robot.operating_hours = (
                    telemetry.operating_hours
                )

            if telemetry.downtime_hours is not None:
                robot.downtime_hours = (
                    telemetry.downtime_hours
                )

            robot.updated_at = utc_now()

            if robot.health_score < 40:
                robot.maintenance_required = True

            if (
                robot.battery_soc_percent
                <= self.config.critical_battery_threshold
            ):
                self._emit(
                    "robot_battery_critical",
                    {
                        "robot_id": robot.robot_id,
                        "battery_soc_percent": (
                            robot.battery_soc_percent
                        ),
                    },
                )

            elif (
                robot.battery_soc_percent
                <= self.config.low_battery_threshold
            ):
                self._emit(
                    "robot_battery_low",
                    {
                        "robot_id": robot.robot_id,
                        "battery_soc_percent": (
                            robot.battery_soc_percent
                        ),
                    },
                )

            if robot.health_status == RobotHealthStatus.CRITICAL:
                self._emit(
                    "robot_health_critical",
                    {
                        "robot_id": robot.robot_id,
                        "health_score": robot.health_score,
                    },
                )

            self._emit(
                "robot_telemetry_updated",
                {
                    "robot_id": robot.robot_id,
                    "health_score": robot.health_score,
                },
            )

            return robot

    # --------------------------------------------------------
    # MAINTENANCE
    # --------------------------------------------------------

    def schedule_maintenance(
        self,
        robot_id: str,
        *,
        reason: str,
    ) -> FleetRobot:
        with self._lock:
            robot = self.get_robot(
                robot_id
            )

            robot.maintenance_required = True

            if robot.active_task_id is None:
                robot.status = RobotStatus.MAINTENANCE

            robot.updated_at = utc_now()

            self._emit(
                "maintenance_scheduled",
                {
                    "robot_id": robot_id,
                    "reason": reason,
                },
            )

            return robot

    def complete_maintenance(
        self,
        robot_id: str,
    ) -> FleetRobot:
        with self._lock:
            robot = self.get_robot(
                robot_id
            )

            if robot.active_task_id is not None:
                raise FleetStateError(
                    "Cannot complete maintenance "
                    "while robot has an active task"
                )

            robot.maintenance_required = False
            robot.status = RobotStatus.IDLE
            robot.updated_at = utc_now()

            self._emit(
                "maintenance_completed",
                {
                    "robot_id": robot_id,
                },
            )

            return robot

    # --------------------------------------------------------
    # CREATE TASK
    # --------------------------------------------------------

    def create_task(
        self,
        *,
        name: str,
        required_capabilities: Sequence[str] = (),
        priority: FleetPriority = (
            FleetPriority.NORMAL
        ),
        metadata: Mapping[str, Any] | None = None,
    ) -> FleetTask:
        with self._lock:
            task = FleetTask(
                task_id=str(uuid.uuid4()),
                name=name,
                required_capabilities=set(
                    required_capabilities
                ),
                priority=FleetPriority(
                    priority
                ),
                metadata=dict(
                    metadata or {}
                ),
            )

            self.tasks[
                task.task_id
            ] = task

            self._emit(
                "fleet_task_created",
                {
                    "task_id": task.task_id,
                    "name": name,
                },
            )

            return task

    # --------------------------------------------------------
    # GET TASK
    # --------------------------------------------------------

    def get_task(
        self,
        task_id: str,
    ) -> FleetTask:
        task = self.tasks.get(
            task_id
        )

        if task is None:
            raise FleetTaskNotFoundError(
                f"Task not found: {task_id}"
            )

        return task

    # --------------------------------------------------------
    # ASSIGN TASK
    # --------------------------------------------------------

    def assign_task(
        self,
        task_id: str,
        *,
        robot_id: str | None = None,
    ) -> FleetTask:
        with self._lock:
            task = self.get_task(
                task_id
            )

            if task.status != FleetTaskStatus.PENDING:
                raise FleetStateError(
                    "Only pending tasks can be assigned"
                )

            if robot_id is not None:
                robot = self.get_robot(
                    robot_id
                )

                if not self.dispatch_policy.matches(
                    robot,
                    task,
                ):
                    raise FleetStateError(
                        "Robot is not available or capable"
                    )

            else:
                candidates = [
                    robot
                    for robot in self.robots.values()
                    if self.dispatch_policy.matches(
                        robot,
                        task,
                    )
                ]

                if not candidates:
                    raise FleetStateError(
                        "No suitable robot available"
                    )

                robot = max(
                    candidates,
                    key=self.dispatch_policy.rank,
                )

            task.assigned_robot_id = (
                robot.robot_id
            )

            task.status = (
                FleetTaskStatus.ASSIGNED
            )

            robot.active_task_id = task.task_id
            robot.status = RobotStatus.WORKING

            robot.updated_at = utc_now()

            self._emit(
                "fleet_task_assigned",
                {
                    "task_id": task.task_id,
                    "robot_id": robot.robot_id,
                },
            )

            return task

    # --------------------------------------------------------
    # START TASK
    # --------------------------------------------------------

    def start_task(
        self,
        task_id: str,
    ) -> FleetTask:
        with self._lock:
            task = self.get_task(
                task_id
            )

            if task.status != FleetTaskStatus.ASSIGNED:
                raise FleetStateError(
                    "Task must be assigned first"
                )

            task.status = FleetTaskStatus.RUNNING
            task.started_at = utc_now()

            self._emit(
                "fleet_task_started",
                {
                    "task_id": task.task_id,
                    "robot_id": task.assigned_robot_id,
                },
            )

            return task

    # --------------------------------------------------------
    # COMPLETE TASK
    # --------------------------------------------------------

    def complete_task(
        self,
        task_id: str,
        *,
        successful: bool = True,
    ) -> FleetTask:
        with self._lock:
            task = self.get_task(
                task_id
            )

            if task.status not in (
                FleetTaskStatus.ASSIGNED,
                FleetTaskStatus.RUNNING,
            ):
                raise FleetStateError(
                    "Task is not active"
                )

            task.status = (
                FleetTaskStatus.COMPLETED
                if successful
                else FleetTaskStatus.FAILED
            )

            task.completed_at = utc_now()

            if task.assigned_robot_id:
                robot = self.get_robot(
                    task.assigned_robot_id
                )

                if successful:
                    robot.completed_tasks += 1
                else:
                    robot.failed_tasks += 1

                robot.active_task_id = None

                robot.status = (
                    RobotStatus.MAINTENANCE
                    if robot.maintenance_required
                    else RobotStatus.IDLE
                )

                robot.updated_at = utc_now()

            self._emit(
                "fleet_task_finished",
                {
                    "task_id": task.task_id,
                    "status": task.status.value,
                },
            )

            return task

    # --------------------------------------------------------
    # CANCEL TASK
    # --------------------------------------------------------

    def cancel_task(
        self,
        task_id: str,
    ) -> FleetTask:
        with self._lock:
            task = self.get_task(
                task_id
            )

            if task.status in (
                FleetTaskStatus.COMPLETED,
                FleetTaskStatus.FAILED,
                FleetTaskStatus.CANCELLED,
            ):
                raise FleetStateError(
                    "Task is already terminal"
                )

            task.status = (
                FleetTaskStatus.CANCELLED
            )

            task.completed_at = utc_now()

            if task.assigned_robot_id:
                robot = self.get_robot(
                    task.assigned_robot_id
                )

                robot.active_task_id = None

                robot.status = (
                    RobotStatus.MAINTENANCE
                    if robot.maintenance_required
                    else RobotStatus.IDLE
                )

            self._emit(
                "fleet_task_cancelled",
                {
                    "task_id": task.task_id,
                },
            )

            return task

    # --------------------------------------------------------
    # AVAILABLE ROBOTS
    # --------------------------------------------------------

    def available_robots(
        self,
        required_capabilities: Sequence[str] = (),
    ) -> list[FleetRobot]:
        required = set(
            required_capabilities
        )

        robots = [
            robot
            for robot in self.robots.values()
            if (
                self.dispatch_policy.is_available(
                    robot
                )
                and required.issubset(
                    robot.capabilities
                )
            )
        ]

        return sorted(
            robots,
            key=self.dispatch_policy.rank,
            reverse=True,
        )

    # --------------------------------------------------------
    # FLEET STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
    ) -> dict[str, Any]:
        robots = list(
            self.robots.values()
        )

        tasks = list(
            self.tasks.values()
        )

        total_robots = len(
            robots
        )

        active_robots = sum(
            robot.status in (
                RobotStatus.WORKING,
                RobotStatus.NAVIGATING,
            )
            for robot in robots
        )

        available = sum(
            self.dispatch_policy.is_available(
                robot
            )
            for robot in robots
        )

        average_health = (
            sum(
                robot.health_score
                for robot in robots
            )
            / total_robots
            if total_robots
            else 0.0
        )

        average_battery = (
            sum(
                robot.battery_soc_percent
                for robot in robots
            )
            / total_robots
            if total_robots
            else 0.0
        )

        return {
            "total_robots": total_robots,
            "active_robots": active_robots,
            "available_robots": available,
            "fleet_utilization_percent": (
                active_robots
                / total_robots
                * 100.0
                if total_robots
                else 0.0
            ),
            "average_health_score": (
                average_health
            ),
            "average_battery_soc_percent": (
                average_battery
            ),
            "maintenance_required_count": sum(
                robot.maintenance_required
                for robot in robots
            ),
            "robot_status_distribution": dict(
                Counter(
                    robot.status.value
                    for robot in robots
                )
            ),
            "health_distribution": dict(
                Counter(
                    robot.health_status.value
                    for robot in robots
                )
            ),
            "total_tasks": len(tasks),
            "task_status_distribution": dict(
                Counter(
                    task.status.value
                    for task in tasks
                )
            ),
            "total_completed_tasks": sum(
                robot.completed_tasks
                for robot in robots
            ),
            "total_failed_tasks": sum(
                robot.failed_tasks
                for robot in robots
            ),
            "simulation_only": True,
        }

    # --------------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------------

    def dashboard(
        self,
    ) -> dict[str, Any]:
        return {
            "statistics": self.statistics(),
            "robots": [
                robot.to_dict()
                for robot in self.robots.values()
            ],
            "tasks": [
                task.to_dict()
                for task in self.tasks.values()
            ],
            "recent_events": list(
                self.events
            )[-100:],
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

        output.write_text(
            json.dumps(
                self.dashboard(),
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        return output


# ============================================================
# PHYSICS SIMULATION ADAPTER
# ============================================================

class PhysicsFleetAdapter:

    @staticmethod
    def register_robots(
        service: RoboticsFleetService,
        physics: Any,
    ) -> None:
        for robot_id, robot in physics.robots.items():
            if robot_id in service.robots:
                continue

            service.register_robot(
                robot_id=robot_id,
                name=robot_id,
                robot_type="simulation_robot",
                capabilities=[
                    "navigation",
                    "telemetry",
                    "simulation",
                ],
                position=RobotPosition(
                    x_m=robot.pose.x,
                    y_m=robot.pose.y,
                    heading_rad=robot.pose.heading_rad,
                ),
            )

    @staticmethod
    def synchronize(
        service: RoboticsFleetService,
        physics: Any,
        *,
        health_scores: Mapping[
            str,
            float
        ] | None = None,
    ) -> None:
        health_scores = (
            health_scores or {}
        )

        for robot_id, robot in physics.robots.items():
            if robot_id not in service.robots:
                PhysicsFleetAdapter.register_robots(
                    service,
                    physics,
                )

            service.update_telemetry(
                FleetRobotTelemetry(
                    robot_id=robot_id,
                    battery_soc_percent=(
                        robot.battery_soc_percent()
                    ),
                    motor_temperature_c=(
                        robot.motor_temperature_c
                    ),
                    vibration_mm_s=0.0,
                    health_score=(
                        health_scores.get(
                            robot_id,
                            100.0,
                        )
                    ),
                    x_m=robot.pose.x,
                    y_m=robot.pose.y,
                    heading_rad=robot.pose.heading_rad,
                    operating_hours=(
                        physics.simulation_time_seconds
                        / 3600.0
                    ),
                )
            )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    fleet = RoboticsFleetService()

    fleet.register_robot(
        robot_id="robot_001",
        name="Warehouse AMR 001",
        robot_type="mobile_robot",
        capabilities=[
            "navigation",
            "transport",
            "inspection",
        ],
    )

    fleet.register_robot(
        robot_id="robot_002",
        name="Inspection Robot 002",
        robot_type="inspection_robot",
        capabilities=[
            "navigation",
            "inspection",
            "camera",
        ],
    )

    fleet.register_robot(
        robot_id="robot_003",
        name="Maintenance Robot 003",
        robot_type="service_robot",
        capabilities=[
            "navigation",
            "maintenance",
            "repair",
        ],
    )

    fleet.update_telemetry(
        FleetRobotTelemetry(
            robot_id="robot_001",
            battery_soc_percent=85.0,
            motor_temperature_c=42.0,
            vibration_mm_s=2.0,
            health_score=95.0,
            x_m=5.0,
            y_m=8.0,
            heading_rad=0.2,
            operating_hours=120.0,
        )
    )

    task = fleet.create_task(
        name="Inspect Warehouse Zone A",
        required_capabilities=[
            "inspection",
            "navigation",
        ],
        priority=FleetPriority.HIGH,
    )

    fleet.assign_task(
        task.task_id
    )

    fleet.start_task(
        task.task_id
    )

    fleet.complete_task(
        task.task_id,
        successful=True,
    )

    output = fleet.export_json(
        "data/fleet/fleet_dashboard.json"
    )

    print(
        json.dumps(
            {
                "statistics": fleet.statistics(),
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

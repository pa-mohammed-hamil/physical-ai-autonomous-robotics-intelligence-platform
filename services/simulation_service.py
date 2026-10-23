"""Simulation Service module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File: backend/app/services/robot_service.py

Central Robot Management Service

Python 3.11+
Standard library only.

Features:
- Robot registration and inventory
- Robot configuration and capabilities
- Robot status management
- Robot lifecycle tracking
- Robot telemetry management
- Battery and health monitoring
- Robot position tracking
- Task assignment and completion
- Maintenance and repair restrictions
- Robot metadata
- Fleet dashboard and statistics
- Audit event history
- JSON and CSV export

Safety:
- No direct robot hardware control
- No actuator commands
- No automatic restart
- Maintenance clearance is administrative only
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

class RobotServiceError(Exception):
    pass


class RobotValidationError(RobotServiceError):
    pass


class RobotNotFoundError(RobotServiceError):
    pass


class RobotAlreadyExistsError(RobotServiceError):
    pass


class RobotStateError(RobotServiceError):
    pass


# ============================================================
# ENUMS
# ============================================================

class RobotType(str, Enum):
    MOBILE = "mobile"
    MANIPULATOR = "manipulator"
    HUMANOID = "humanoid"
    INDUSTRIAL = "industrial"
    WAREHOUSE = "warehouse"
    SERVICE = "service"
    MEDICAL = "medical"
    AGRICULTURAL = "agricultural"
    INSPECTION = "inspection"
    DRONE = "drone"
    OTHER = "other"


class RobotStatus(str, Enum):
    REGISTERED = "registered"
    IDLE = "idle"
    WORKING = "working"
    NAVIGATING = "navigating"
    CHARGING = "charging"
    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    FAULT = "fault"
    EMERGENCY = "emergency"
    OFFLINE = "offline"
    RETIRED = "retired"


class RobotHealthStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class RobotTaskStatus(str, Enum):
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RobotValidationError(
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
        raise RobotValidationError(
            f"{name} must be a finite number"
        )

    result = float(value)

    if minimum is not None and result < minimum:
        raise RobotValidationError(
            f"{name} must be >= {minimum}"
        )

    if maximum is not None and result > maximum:
        raise RobotValidationError(
            f"{name} must be <= {maximum}"
        )

    return result


def json_copy(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(value, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise RobotValidationError(
            "Value must be JSON serializable"
        ) from exc


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RobotServiceConfig:
    maximum_robots: int = 10000
    maximum_events: int = 20000

    minimum_task_battery_soc: float = 20.0
    minimum_task_health_score: float = 70.0

    low_battery_soc: float = 20.0
    critical_battery_soc: float = 10.0

    allow_hardware_control: bool = False

    def __post_init__(self) -> None:
        if (
            self.maximum_robots < 1
            or self.maximum_events < 1
        ):
            raise RobotValidationError(
                "Capacity limits must be positive"
            )

        for name in (
            "minimum_task_battery_soc",
            "minimum_task_health_score",
            "low_battery_soc",
            "critical_battery_soc",
        ):
            finite(
                getattr(self, name),
                name,
                minimum=0.0,
                maximum=100.0,
            )

        if self.critical_battery_soc >= self.low_battery_soc:
            raise RobotValidationError(
                "Critical battery threshold must be lower"
            )

        if self.allow_hardware_control:
            raise RobotValidationError(
                "Hardware control is prohibited"
            )


# ============================================================
# ROBOT POSITION
# ============================================================

@dataclass
class RobotPosition:
    x_m: float = 0.0
    y_m: float = 0.0
    heading_rad: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "x_m",
            "y_m",
            "heading_rad",
        ):
            finite(getattr(self, name), name)

    def to_dict(self) -> dict[str, float]:
        return {
            "x_m": self.x_m,
            "y_m": self.y_m,
            "heading_rad": self.heading_rad,
        }


# ============================================================
# ROBOT TELEMETRY
# ============================================================

@dataclass(frozen=True)
class RobotTelemetry:
    robot_id: str

    battery_soc_percent: float
    health_score: float

    motor_temperature_c: float
    vibration_mm_s: float

    x_m: float
    y_m: float
    heading_rad: float

    operating_hours: float = 0.0

    sensor_available: bool = True
    network_connected: bool = True

    timestamp: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        require_text(self.robot_id, "robot_id")

        for name in (
            "battery_soc_percent",
            "health_score",
        ):
            finite(
                getattr(self, name),
                name,
                minimum=0.0,
                maximum=100.0,
            )

        for name in (
            "motor_temperature_c",
            "x_m",
            "y_m",
            "heading_rad",
        ):
            finite(getattr(self, name), name)

        for name in (
            "vibration_mm_s",
            "operating_hours",
        ):
            finite(
                getattr(self, name),
                name,
                minimum=0.0,
            )

        datetime.fromisoformat(
            self.timestamp.replace("Z", "+00:00")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "battery_soc_percent": self.battery_soc_percent,
            "health_score": self.health_score,
            "motor_temperature_c": self.motor_temperature_c,
            "vibration_mm_s": self.vibration_mm_s,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "heading_rad": self.heading_rad,
            "operating_hours": self.operating_hours,
            "sensor_available": self.sensor_available,
            "network_connected": self.network_connected,
            "timestamp": self.timestamp,
        }


# ============================================================
# ROBOT TASK
# ============================================================

@dataclass
class RobotTask:
    task_id: str
    robot_id: str
    name: str

    status: RobotTaskStatus = RobotTaskStatus.ASSIGNED

    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    completed_at: str | None = None

    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "robot_id": self.robot_id,
            "name": self.name,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "failure_reason": self.failure_reason,
        }


# ============================================================
# ROBOT ENTITY
# ============================================================

@dataclass
class Robot:
    robot_id: str
    name: str
    robot_type: RobotType

    manufacturer: str = "Unknown"
    model: str = "Unknown"
    serial_number: str | None = None

    status: RobotStatus = RobotStatus.REGISTERED

    health_status: RobotHealthStatus = (
        RobotHealthStatus.UNKNOWN
    )

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

    operating_hours: float = 0.0

    completed_tasks: int = 0
    failed_tasks: int = 0

    active_task_id: str | None = None

    maintenance_required: bool = False
    repair_required: bool = False

    sensor_available: bool = True
    network_connected: bool = True

    registered_at: str = field(
        default_factory=utc_now
    )

    updated_at: str = field(
        default_factory=utc_now
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "name": self.name,
            "robot_type": self.robot_type.value,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "serial_number": self.serial_number,
            "status": self.status.value,
            "health_status": self.health_status.value,
            "health_score": self.health_score,
            "battery_soc_percent": (
                self.battery_soc_percent
            ),
            "motor_temperature_c": (
                self.motor_temperature_c
            ),
            "vibration_mm_s": self.vibration_mm_s,
            "position": self.position.to_dict(),
            "capabilities": sorted(self.capabilities),
            "operating_hours": self.operating_hours,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "active_task_id": self.active_task_id,
            "maintenance_required": (
                self.maintenance_required
            ),
            "repair_required": self.repair_required,
            "sensor_available": self.sensor_available,
            "network_connected": self.network_connected,
            "registered_at": self.registered_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


# ============================================================
# ROBOT SERVICE
# ============================================================

class RoboticsRobotService:

    def __init__(
        self,
        config: RobotServiceConfig | None = None,
    ) -> None:
        self.config = (
            config or RobotServiceConfig()
        )

        self.robots: dict[str, Robot] = {}
        self.tasks: dict[str, RobotTask] = {}

        self.events: deque[dict[str, Any]] = deque(
            maxlen=self.config.maximum_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # AUDIT EVENT
    # --------------------------------------------------------

    def _event(
        self,
        event_type: str,
        robot_id: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.events.append({
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "robot_id": robot_id,
            "timestamp": utc_now(),
            "details": json_copy(
                dict(details or {})
            ),
        })

    # --------------------------------------------------------
    # REGISTER ROBOT
    # --------------------------------------------------------

    def register_robot(
        self,
        *,
        robot_id: str,
        name: str,
        robot_type: RobotType,
        manufacturer: str = "Unknown",
        model: str = "Unknown",
        serial_number: str | None = None,
        capabilities: set[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Robot:
        robot_id = require_text(
            robot_id,
            "robot_id",
        )

        name = require_text(
            name,
            "name",
        )

        with self._lock:
            if robot_id in self.robots:
                raise RobotAlreadyExistsError(
                    f"Robot already exists: {robot_id}"
                )

            if len(self.robots) >= self.config.maximum_robots:
                raise RobotStateError(
                    "Robot capacity reached"
                )

            robot = Robot(
                robot_id=robot_id,
                name=name,
                robot_type=RobotType(robot_type),
                manufacturer=manufacturer,
                model=model,
                serial_number=serial_number,
                capabilities=set(
                    capabilities or ()
                ),
                metadata=json_copy(
                    dict(metadata or {})
                ),
            )

            self.robots[robot_id] = robot

            self._event(
                "robot_registered",
                robot_id,
                {
                    "robot_type": robot.robot_type.value,
                    "name": robot.name,
                },
            )

            return robot

    # --------------------------------------------------------
    # GET ROBOT
    # --------------------------------------------------------

    def get_robot(
        self,
        robot_id: str,
    ) -> Robot:
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
        robot_type: RobotType | None = None,
        capability: str | None = None,
    ) -> list[Robot]:
        with self._lock:
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

            return sorted(
                robots,
                key=lambda robot: robot.robot_id,
            )

    # --------------------------------------------------------
    # UPDATE STATUS
    # --------------------------------------------------------

    def update_status(
        self,
        robot_id: str,
        status: RobotStatus,
    ) -> Robot:
        status = RobotStatus(status)

        with self._lock:
            robot = self.get_robot(
                robot_id
            )

            if robot.status == RobotStatus.RETIRED:
                raise RobotStateError(
                    "Retired robots cannot change status"
                )

            if (
                status == RobotStatus.WORKING
                and (
                    robot.maintenance_required
                    or robot.repair_required
                    or robot.health_status
                    == RobotHealthStatus.CRITICAL
                )
            ):
                raise RobotStateError(
                    "Robot is not cleared for work"
                )

            if (
                robot.active_task_id is not None
                and status in (
                    RobotStatus.MAINTENANCE,
                    RobotStatus.REPAIR,
                    RobotStatus.RETIRED,
                )
            ):
                raise RobotStateError(
                    "Resolve active task first"
                )

            old_status = robot.status

            robot.status = status
            robot.updated_at = utc_now()

            self._event(
                "robot_status_updated",
                robot_id,
                {
                    "old_status": old_status.value,
                    "new_status": status.value,
                },
            )

            return robot

    # --------------------------------------------------------
    # UPDATE TELEMETRY
    # --------------------------------------------------------

    def update_telemetry(
        self,
        telemetry: RobotTelemetry,
    ) -> Robot:
        with self._lock:
            robot = self.get_robot(
                telemetry.robot_id
            )

            if robot.status == RobotStatus.RETIRED:
                raise RobotStateError(
                    "Retired robot cannot accept telemetry"
                )

            robot.battery_soc_percent = (
                telemetry.battery_soc_percent
            )

            robot.health_score = (
                telemetry.health_score
            )

            robot.motor_temperature_c = (
                telemetry.motor_temperature_c
            )

            robot.vibration_mm_s = (
                telemetry.vibration_mm_s
            )

            robot.position = RobotPosition(
                x_m=telemetry.x_m,
                y_m=telemetry.y_m,
                heading_rad=telemetry.heading_rad,
            )

            robot.operating_hours = (
                telemetry.operating_hours
            )

            robot.sensor_available = (
                telemetry.sensor_available
            )

            robot.network_connected = (
                telemetry.network_connected
            )

            if robot.health_score <= 40:
                robot.health_status = (
                    RobotHealthStatus.CRITICAL
                )

            elif robot.health_score <= 70:
                robot.health_status = (
                    RobotHealthStatus.WARNING
                )

            else:
                robot.health_status = (
                    RobotHealthStatus.HEALTHY
                )

            if (
                robot.battery_soc_percent
                <= self.config.critical_battery_soc
            ):
                self._event(
                    "critical_battery",
                    robot.robot_id,
                    {
                        "battery_soc_percent": (
                            robot.battery_soc_percent
                        )
                    },
                )

            if (
                robot.health_status
                == RobotHealthStatus.CRITICAL
            ):
                robot.maintenance_required = True

                self._event(
                    "critical_robot_health",
                    robot.robot_id,
                    {
                        "health_score": robot.health_score
                    },
                )

            robot.updated_at = utc_now()

            self._event(
                "robot_telemetry_updated",
                robot.robot_id,
                {
                    "battery_soc_percent": (
                        robot.battery_soc_percent
                    ),
                    "health_score": robot.health_score,
                },
            )

            return robot

    # --------------------------------------------------------
    # ADD CAPABILITY
    # --------------------------------------------------------

    def add_capability(
        self,
        robot_id: str,
        capability: str,
    ) -> Robot:
        capability = require_text(
            capability,
            "capability",
        )

        with self._lock:
            robot = self.get_robot(
                robot_id
            )

            robot.capabilities.add(
                capability
            )

            robot.updated_at = utc_now()

            self._event(
                "robot_capability_added",
                robot_id,
                {
                    "capability": capability
                },
            )

            return robot

    # --------------------------------------------------------
    # REMOVE CAPABILITY
    # --------------------------------------------------------

    def remove_capability(
        self,
        robot_id: str,
        capability: str,
    ) -> Robot:
        with self._lock:
            robot = self.get_robot(
                robot_id
            )

            robot.capabilities.discard(
                capability
            )

            robot.updated_at = utc_now()

            return robot

    # --------------------------------------------------------
    # REQUIRE MAINTENANCE
    # --------------------------------------------------------

    def require_maintenance(
        self,
        robot_id: str,
        *,
        reason: str,
    ) -> Robot:
        require_text(reason, "reason")

        with self._lock:
            robot = self.get_robot(
                robot_id
            )

            robot.maintenance_required = True
            robot.updated_at = utc_now()

            self._event(
                "maintenance_required",
                robot_id,
                {
                    "reason": reason
                },
            )

            return robot

    # --------------------------------------------------------
    # CLEAR MAINTENANCE
    # --------------------------------------------------------

    def clear_maintenance(
        self,
        robot_id: str,
        *,
        approved_by: str,
    ) -> Robot:
        require_text(
            approved_by,
            "approved_by",
        )

        with self._lock:
            robot = self.get_robot(
                robot_id
            )

            if (
                robot.health_status
                == RobotHealthStatus.CRITICAL
            ):
                raise RobotStateError(
                    "Critical robot health blocks clearance"
                )

            robot.maintenance_required = False
            robot.updated_at = utc_now()

            self._event(
                "maintenance_cleared",
                robot_id,
                {
                    "approved_by": approved_by
                },
            )

            return robot

    # --------------------------------------------------------
    # REQUIRE REPAIR
    # --------------------------------------------------------

    def require_repair(
        self,
        robot_id: str,
        *,
        reason: str,
    ) -> Robot:
        require_text(reason, "reason")

        with self._lock:
            robot = self.get_robot(
                robot_id
            )

            robot.repair_required = True
            robot.updated_at = utc_now()

            self._event(
                "repair_required",
                robot_id,
                {
                    "reason": reason
                },
            )

            return robot

    # --------------------------------------------------------
    # CLEAR REPAIR
    # --------------------------------------------------------

    def clear_repair(
        self,
        robot_id: str,
        *,
        approved_by: str,
    ) -> Robot:
        require_text(
            approved_by,
            "approved_by",
        )

        with self._lock:
            robot = self.get_robot(
                robot_id
            )

            if (
                robot.health_status
                == RobotHealthStatus.CRITICAL
            ):
                raise RobotStateError(
                    "Critical health blocks repair clearance"
                )

            robot.repair_required = False
            robot.updated_at = utc_now()

            self._event(
                "repair_cleared",
                robot_id,
                {
                    "approved_by": approved_by
                },
            )

            return robot

    # --------------------------------------------------------
    # CAN ACCEPT TASK
    # --------------------------------------------------------

    def can_accept_task(
        self,
        robot_id: str,
        *,
        required_capabilities: set[str] | None = None,
    ) -> tuple[bool, str]:
        robot = self.get_robot(
            robot_id
        )

        if robot.status != RobotStatus.IDLE:
            return False, "Robot is not idle"

        if robot.active_task_id is not None:
            return False, "Robot already has an active task"

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
            < self.config.minimum_task_battery_soc
        ):
            return False, "Battery too low"

        if (
            robot.health_score
            < self.config.minimum_task_health_score
        ):
            return False, "Health score too low"

        required = set(
            required_capabilities or ()
        )

        if not required.issubset(
            robot.capabilities
        ):
            return False, "Missing capabilities"

        return True, "Robot available"

    # --------------------------------------------------------
    # ASSIGN TASK
    # --------------------------------------------------------

    def assign_task(
        self,
        *,
        robot_id: str,
        name: str,
        required_capabilities: set[str] | None = None,
    ) -> RobotTask:
        name = require_text(
            name,
            "name",
        )

        with self._lock:
            available, reason = self.can_accept_task(
                robot_id,
                required_capabilities=required_capabilities,
            )

            if not available:
                raise RobotStateError(
                    reason
                )

            robot = self.get_robot(
                robot_id
            )

            task = RobotTask(
                task_id=str(uuid.uuid4()),
                robot_id=robot_id,
                name=name,
            )

            self.tasks[
                task.task_id
            ] = task

            robot.active_task_id = task.task_id
            robot.updated_at = utc_now()

            self._event(
                "robot_task_assigned",
                robot_id,
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
    ) -> RobotTask:
        task = self.tasks.get(
            task_id
        )

        if task is None:
            raise RobotNotFoundError(
                f"Task not found: {task_id}"
            )

        return task

    # --------------------------------------------------------
    # START TASK
    # --------------------------------------------------------

    def start_task(
        self,
        task_id: str,
    ) -> RobotTask:
        with self._lock:
            task = self.get_task(
                task_id
            )

            if task.status != RobotTaskStatus.ASSIGNED:
                raise RobotStateError(
                    "Task is not assigned"
                )

            robot = self.get_robot(
                task.robot_id
            )

            if robot.active_task_id != task_id:
                raise RobotStateError(
                    "Task is not active on robot"
                )

            task.status = RobotTaskStatus.RUNNING
            task.started_at = utc_now()

            robot.status = RobotStatus.WORKING
            robot.updated_at = utc_now()

            self._event(
                "robot_task_started",
                robot.robot_id,
                {
                    "task_id": task_id
                },
            )

            return task

    # --------------------------------------------------------
    # FINISH TASK
    # --------------------------------------------------------

    def finish_task(
        self,
        task_id: str,
        *,
        success: bool,
        failure_reason: str | None = None,
    ) -> RobotTask:
        with self._lock:
            task = self.get_task(
                task_id
            )

            if task.status not in (
                RobotTaskStatus.ASSIGNED,
                RobotTaskStatus.RUNNING,
            ):
                raise RobotStateError(
                    "Task already finished"
                )

            robot = self.get_robot(
                task.robot_id
            )

            task.status = (
                RobotTaskStatus.COMPLETED
                if success
                else RobotTaskStatus.FAILED
            )

            task.completed_at = utc_now()

            if success:
                robot.completed_tasks += 1
            else:
                robot.failed_tasks += 1
                task.failure_reason = (
                    failure_reason or "Unknown failure"
                )

            robot.active_task_id = None

            if robot.status in (
                RobotStatus.WORKING,
                RobotStatus.NAVIGATING,
                RobotStatus.IDLE,
            ):
                robot.status = RobotStatus.IDLE

            robot.updated_at = utc_now()

            self._event(
                "robot_task_finished",
                robot.robot_id,
                {
                    "task_id": task_id,
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
        *,
        reason: str,
    ) -> RobotTask:
        require_text(
            reason,
            "reason",
        )

        with self._lock:
            task = self.get_task(
                task_id
            )

            if task.status not in (
                RobotTaskStatus.ASSIGNED,
                RobotTaskStatus.RUNNING,
            ):
                raise RobotStateError(
                    "Task cannot be cancelled"
                )

            task.status = RobotTaskStatus.CANCELLED
            task.failure_reason = reason
            task.completed_at = utc_now()

            robot = self.get_robot(
                task.robot_id
            )

            robot.active_task_id = None

            if robot.status in (
                RobotStatus.WORKING,
                RobotStatus.NAVIGATING,
            ):
                robot.status = RobotStatus.IDLE

            robot.updated_at = utc_now()

            self._event(
                "robot_task_cancelled",
                robot.robot_id,
                {
                    "task_id": task_id,
                    "reason": reason,
                },
            )

            return task

    # --------------------------------------------------------
    # RETIRE ROBOT
    # --------------------------------------------------------

    def retire_robot(
        self,
        robot_id: str,
        *,
        approved_by: str,
    ) -> Robot:
        require_text(
            approved_by,
            "approved_by",
        )

        with self._lock:
            robot = self.get_robot(
                robot_id
            )

            if robot.active_task_id is not None:
                raise RobotStateError(
                    "Resolve active task before retirement"
                )

            robot.status = RobotStatus.RETIRED
            robot.updated_at = utc_now()

            self._event(
                "robot_retired",
                robot_id,
                {
                    "approved_by": approved_by
                },
            )

            return robot

    # --------------------------------------------------------
    # FLEET STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
    ) -> dict[str, Any]:
        with self._lock:
            robots = list(
                self.robots.values()
            )

            tasks = list(
                self.tasks.values()
            )

            return {
                "total_robots": len(robots),
                "status_distribution": dict(
                    Counter(
                        robot.status.value
                        for robot in robots
                    )
                ),
                "robot_type_distribution": dict(
                    Counter(
                        robot.robot_type.value
                        for robot in robots
                    )
                ),
                "health_distribution": dict(
                    Counter(
                        robot.health_status.value
                        for robot in robots
                    )
                ),
                "maintenance_required": sum(
                    robot.maintenance_required
                    for robot in robots
                ),
                "repair_required": sum(
                    robot.repair_required
                    for robot in robots
                ),
                "average_battery_soc": (
                    round(
                        sum(
                            robot.battery_soc_percent
                            for robot in robots
                        ) / len(robots),
                        2,
                    )
                    if robots
                    else None
                ),
                "average_health_score": (
                    round(
                        sum(
                            robot.health_score
                            for robot in robots
                        ) / len(robots),
                        2,
                    )
                    if robots
                    else None
                ),
                "total_tasks": len(tasks),
                "completed_tasks": sum(
                    task.status == RobotTaskStatus.COMPLETED
                    for task in tasks
                ),
                "failed_tasks": sum(
                    task.status == RobotTaskStatus.FAILED
                    for task in tasks
                ),
                "simulation_only": True,
            }

    # --------------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------------

    def dashboard(
        self,
    ) -> dict[str, Any]:
        with self._lock:
            return {
                "generated_at": utc_now(),
                "statistics": self.statistics(),
                "robots": [
                    robot.to_dict()
                    for robot in self.robots.values()
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

        with self._lock:
            payload = {
                "generated_at": utc_now(),
                "statistics": self.statistics(),
                "robots": [
                    robot.to_dict()
                    for robot in self.robots.values()
                ],
                "tasks": [
                    task.to_dict()
                    for task in self.tasks.values()
                ],
                "events": list(self.events),
            }

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
            "robot_id",
            "name",
            "robot_type",
            "manufacturer",
            "model",
            "status",
            "health_status",
            "health_score",
            "battery_soc_percent",
            "motor_temperature_c",
            "vibration_mm_s",
            "operating_hours",
            "completed_tasks",
            "failed_tasks",
            "maintenance_required",
            "repair_required",
            "active_task_id",
            "updated_at",
        ]

        with self._lock:
            rows = [
                robot.to_dict()
                for robot in self.robots.values()
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
# PHYSICS SIMULATION ADAPTER
# ============================================================

class PhysicsRobotServiceAdapter:
    """
    Synchronize physics simulation state
    with the central robot registry.

    This adapter expects the existing
    RoboticsPhysicsAdapter interface.
    """

    @staticmethod
    def synchronize(
        service: RoboticsRobotService,
        physics: Any,
        *,
        health_scores: Mapping[str, float] | None = None,
    ) -> int:
        updated = 0

        for robot_id, state in physics.robots.items():
            if robot_id not in service.robots:
                continue

            telemetry = RobotTelemetry(
                robot_id=robot_id,
                battery_soc_percent=(
                    state.battery_soc_percent()
                ),
                health_score=(
                    health_scores.get(robot_id, 100.0)
                    if health_scores
                    else 100.0
                ),
                motor_temperature_c=(
                    state.motor_temperature_c
                ),
                vibration_mm_s=0.0,
                x_m=state.pose.x,
                y_m=state.pose.y,
                heading_rad=state.pose.heading_rad,
                operating_hours=(
                    physics.simulation_time_seconds
                    / 3600.0
                ),
            )

            service.update_telemetry(
                telemetry
            )

            updated += 1

        return updated


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    service = RoboticsRobotService()

    service.register_robot(
        robot_id="robot_001",
        name="Warehouse Autonomous Robot",
        robot_type=RobotType.WAREHOUSE,
        manufacturer="Demo Robotics",
        model="AMR-X1",
        capabilities={
            "navigation",
            "obstacle_avoidance",
            "object_detection",
            "warehouse_transport",
        },
    )

    service.update_status(
        "robot_001",
        RobotStatus.IDLE,
    )

    service.update_telemetry(
        RobotTelemetry(
            robot_id="robot_001",
            battery_soc_percent=87.0,
            health_score=94.0,
            motor_temperature_c=43.0,
            vibration_mm_s=2.1,
            x_m=4.0,
            y_m=2.0,
            heading_rad=0.5,
            operating_hours=320.0,
        )
    )

    task = service.assign_task(
        robot_id="robot_001",
        name="Transport warehouse package",
        required_capabilities={
            "navigation",
            "warehouse_transport",
        },
    )

    service.start_task(
        task.task_id
    )

    service.finish_task(
        task.task_id,
        success=True,
    )

    service.export_json(
        "data/robots/robot_registry.json"
    )

    service.export_csv(
        "data/robots/robot_inventory.csv"
    )

    print(
        json.dumps(
            service.dashboard(),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

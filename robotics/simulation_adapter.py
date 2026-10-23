"""Simulation Adapter module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
backend/app/robotics/control/simulation_adapter.py

Python 3.11+
Standard library only.

SIMULATION ADAPTER

Features:
- Multi-robot simulation registry
- Differential-drive robot simulation
- Robot pose and velocity updates
- Battery consumption
- Temperature simulation
- Collision and workspace boundaries
- Simulated obstacle detection
- Robot command validation
- Sensor telemetry generation
- Emergency-stop state latching
- Pause/resume and reset
- Simulation stepping
- Event history
- JSON export
- Tenant isolation

SIMULATION ONLY:
This module never controls physical hardware.

The physics model is simplified and not suitable
for validating real-world robot safety.
"""

from __future__ import annotations

import json
import math
import re
import threading
import uuid

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


# ============================================================
# EXCEPTIONS
# ============================================================

class SimulationAdapterError(Exception):
    pass


class SimulationValidationError(SimulationAdapterError):
    pass


class SimulationRobotNotFound(SimulationAdapterError):
    pass


class SimulationConflict(SimulationAdapterError):
    pass


class SimulationSafetyError(SimulationAdapterError):
    pass


# ============================================================
# ENUMS
# ============================================================

class SimulationRobotStatus(str, Enum):
    IDLE = "idle"
    MOVING = "moving"
    PAUSED = "paused"
    COLLISION = "collision"
    FAULTED = "faulted"
    EMERGENCY_STOP = "emergency_stop"


class SimulationCommandType(str, Enum):
    SET_VELOCITY = "set_velocity"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    RESET = "reset"


class SimulationEventType(str, Enum):
    REGISTERED = "registered"
    COMMAND_ACCEPTED = "command_accepted"
    STEP_COMPLETED = "step_completed"
    COLLISION_DETECTED = "collision_detected"
    EMERGENCY_STOP = "emergency_stop"
    EMERGENCY_RESET = "emergency_reset"
    PAUSED = "paused"
    RESUMED = "resumed"
    RESET = "reset"


# ============================================================
# HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_id(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(
            r"[A-Za-z0-9_.-]{1,128}",
            value,
        )
    ):
        raise SimulationValidationError(
            f"Invalid {name}"
        )
    return value


def finite(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise SimulationValidationError(
            f"{name} must be finite"
        )
    return float(value)


def positive(value: float, name: str) -> float:
    result = finite(value, name)
    if result <= 0:
        raise SimulationValidationError(
            f"{name} must be positive"
        )
    return result


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(minimum, min(maximum, value))


def normalize_angle(angle: float) -> float:
    return (
        (angle + math.pi) % (2 * math.pi)
    ) - math.pi


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class SimulationAdapterConfig:
    maximum_robots: int = 1000
    maximum_events: int = 20000

    maximum_step_seconds: float = 0.2
    maximum_linear_speed_mps: float = 1.0
    maximum_angular_speed_radps: float = 1.5

    default_battery_percentage: float = 100.0
    default_temperature_c: float = 25.0

    idle_battery_drain_per_second: float = 0.0005
    moving_battery_drain_per_second: float = 0.005

    temperature_increase_per_second: float = 0.03
    temperature_cooling_per_second: float = 0.02

    maximum_temperature_c: float = 80.0
    critical_battery_percentage: float = 10.0

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_robots",
            "maximum_events",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise SimulationValidationError(
                    f"{name} must be positive"
                )

        for name in (
            "maximum_step_seconds",
            "maximum_linear_speed_mps",
            "maximum_angular_speed_radps",
            "maximum_temperature_c",
        ):
            positive(
                getattr(self, name),
                name,
            )

        for name in (
            "idle_battery_drain_per_second",
            "moving_battery_drain_per_second",
            "temperature_increase_per_second",
            "temperature_cooling_per_second",
        ):
            value = finite(
                getattr(self, name),
                name,
            )
            if value < 0:
                raise SimulationValidationError(
                    f"{name} cannot be negative"
                )

        battery = finite(
            self.default_battery_percentage,
            "default_battery_percentage",
        )
        critical = finite(
            self.critical_battery_percentage,
            "critical_battery_percentage",
        )

        if not (
            0 <= battery <= 100
            and 0 <= critical <= 100
        ):
            raise SimulationValidationError(
                "Battery percentage outside 0-100"
            )

        finite(
            self.default_temperature_c,
            "default_temperature_c",
        )

        if not self.simulation_only:
            raise SimulationValidationError(
                "Physical hardware simulation adapter unsupported"
            )


# ============================================================
# GEOMETRY
# ============================================================

@dataclass(frozen=True)
class SimulationPose:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def __post_init__(self) -> None:
        finite(self.x, "x")
        finite(self.y, "y")
        finite(self.theta, "theta")

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "theta": self.theta,
        }


@dataclass(frozen=True)
class SimulationObstacle:
    obstacle_id: str
    x: float
    y: float
    radius: float

    def __post_init__(self) -> None:
        validate_id(
            self.obstacle_id,
            "obstacle_id",
        )
        finite(self.x, "obstacle_x")
        finite(self.y, "obstacle_y")
        positive(self.radius, "obstacle_radius")

    def to_dict(self) -> dict[str, Any]:
        return {
            "obstacle_id": self.obstacle_id,
            "x": self.x,
            "y": self.y,
            "radius": self.radius,
        }


@dataclass(frozen=True)
class SimulationWorkspace:
    minimum_x: float = -10.0
    maximum_x: float = 10.0
    minimum_y: float = -10.0
    maximum_y: float = 10.0

    obstacles: tuple[SimulationObstacle, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "minimum_x",
            "maximum_x",
            "minimum_y",
            "maximum_y",
        ):
            finite(
                getattr(self, name),
                name,
            )

        if (
            self.minimum_x >= self.maximum_x
            or self.minimum_y >= self.maximum_y
        ):
            raise SimulationValidationError(
                "Invalid workspace bounds"
            )

        if not isinstance(self.obstacles, tuple):
            raise SimulationValidationError(
                "Obstacles must be a tuple"
            )

        ids = set()

        for obstacle in self.obstacles:
            if not isinstance(
                obstacle,
                SimulationObstacle,
            ):
                raise SimulationValidationError(
                    "Invalid obstacle"
                )

            if obstacle.obstacle_id in ids:
                raise SimulationValidationError(
                    "Duplicate obstacle"
                )

            ids.add(obstacle.obstacle_id)


# ============================================================
# ROBOT DEFINITION
# ============================================================

@dataclass(frozen=True)
class SimulationRobotDefinition:
    tenant_id: str
    robot_id: str

    name: str
    radius_m: float = 0.3

    initial_pose: SimulationPose = field(
        default_factory=SimulationPose
    )

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")

        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or len(self.name) > 256
        ):
            raise SimulationValidationError(
                "Invalid robot name"
            )

        positive(self.radius_m, "radius_m")

        if not isinstance(
            self.initial_pose,
            SimulationPose,
        ):
            raise SimulationValidationError(
                "Invalid initial pose"
            )


# ============================================================
# COMMAND
# ============================================================

@dataclass(frozen=True)
class SimulationCommand:
    tenant_id: str
    robot_id: str

    command_type: SimulationCommandType

    linear_velocity_mps: float = 0.0
    angular_velocity_radps: float = 0.0

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")

        if not isinstance(
            self.command_type,
            SimulationCommandType,
        ):
            raise SimulationValidationError(
                "Invalid simulation command"
            )

        finite(
            self.linear_velocity_mps,
            "linear_velocity_mps",
        )

        finite(
            self.angular_velocity_radps,
            "angular_velocity_radps",
        )


# ============================================================
# ROBOT STATE
# ============================================================

@dataclass
class SimulationRobotRecord:
    definition: SimulationRobotDefinition

    pose: SimulationPose

    status: SimulationRobotStatus = (
        SimulationRobotStatus.IDLE
    )

    linear_velocity_mps: float = 0.0
    angular_velocity_radps: float = 0.0

    battery_percentage: float = 100.0
    temperature_c: float = 25.0

    emergency_latched: bool = False
    collision_latched: bool = False
    fault_latched: bool = False

    sequence_number: int = 0
    simulation_time_seconds: float = 0.0

    registered_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.definition.tenant_id,
            "robot_id": self.definition.robot_id,
            "name": self.definition.name,
            "pose": self.pose.to_dict(),
            "radius_m": self.definition.radius_m,
            "status": self.status.value,
            "linear_velocity_mps": (
                self.linear_velocity_mps
            ),
            "angular_velocity_radps": (
                self.angular_velocity_radps
            ),
            "battery_percentage": (
                self.battery_percentage
            ),
            "temperature_c": self.temperature_c,
            "emergency_latched": (
                self.emergency_latched
            ),
            "collision_latched": (
                self.collision_latched
            ),
            "fault_latched": self.fault_latched,
            "sequence_number": (
                self.sequence_number
            ),
            "simulation_time_seconds": (
                self.simulation_time_seconds
            ),
            "registered_at": (
                self.registered_at.isoformat()
            ),
        }


# ============================================================
# EVENT
# ============================================================

@dataclass(frozen=True)
class SimulationEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    robot_id: str

    event_type: SimulationEventType
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "event_type": self.event_type.value,
            "details": self.details,
        }


# ============================================================
# SIMULATION ADAPTER
# ============================================================

class RoboticsSimulationAdapter:

    """
    Thread-safe simplified 2D mobile robot simulator.

    Motion:
        x += v * cos(theta) * dt
        y += v * sin(theta) * dt
        theta += omega * dt

    Collision:
        Circle vs circle obstacles
        Circle vs workspace boundaries

    This is a discrete-time educational model.
    It is not a safety-rated physics engine.
    """

    def __init__(
        self,
        workspace: SimulationWorkspace | None = None,
        config: SimulationAdapterConfig | None = None,
    ) -> None:
        self.workspace = (
            workspace or SimulationWorkspace()
        )

        self.config = (
            config or SimulationAdapterConfig()
        )

        self._robots: dict[
            tuple[str, str],
            SimulationRobotRecord
        ] = {}

        self._events: deque[
            SimulationEvent
        ] = deque(
            maxlen=self.config.maximum_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # INTERNAL
    # --------------------------------------------------------

    def _key(
        self,
        tenant_id: str,
        robot_id: str,
    ) -> tuple[str, str]:
        return (
            validate_id(tenant_id, "tenant_id"),
            validate_id(robot_id, "robot_id"),
        )

    def _get(
        self,
        tenant_id: str,
        robot_id: str,
    ) -> SimulationRobotRecord:
        record = self._robots.get(
            self._key(
                tenant_id,
                robot_id,
            )
        )

        if record is None:
            raise SimulationRobotNotFound(
                "Robot not registered"
            )

        return record

    def _event(
        self,
        record: SimulationRobotRecord,
        event_type: SimulationEventType,
        details: str,
    ) -> None:
        self._events.append(
            SimulationEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                tenant_id=record.definition.tenant_id,
                robot_id=record.definition.robot_id,
                event_type=event_type,
                details=details,
            )
        )

    # --------------------------------------------------------
    # COLLISION DETECTION
    # --------------------------------------------------------

    def _collision_reason(
        self,
        pose: SimulationPose,
        radius: float,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> str | None:
        w = self.workspace

        if (
            pose.x - radius < w.minimum_x
            or pose.x + radius > w.maximum_x
            or pose.y - radius < w.minimum_y
            or pose.y + radius > w.maximum_y
        ):
            return "workspace_boundary"

        for obstacle in w.obstacles:
            distance = math.hypot(
                pose.x - obstacle.x,
                pose.y - obstacle.y,
            )

            if distance <= (
                radius + obstacle.radius
            ):
                return (
                    f"obstacle:{obstacle.obstacle_id}"
                )

        for other in self._robots.values():
            if (
                other.definition.tenant_id != tenant_id
                or other.definition.robot_id == robot_id
            ):
                continue

            distance = math.hypot(
                pose.x - other.pose.x,
                pose.y - other.pose.y,
            )

            if distance <= (
                radius + other.definition.radius_m
            ):
                return (
                    f"robot:{other.definition.robot_id}"
                )

        return None

    # --------------------------------------------------------
    # REGISTER ROBOT
    # --------------------------------------------------------

    def register_robot(
        self,
        definition: SimulationRobotDefinition,
    ) -> dict[str, Any]:
        if not isinstance(
            definition,
            SimulationRobotDefinition,
        ):
            raise SimulationValidationError(
                "Invalid robot definition"
            )

        with self._lock:
            key = self._key(
                definition.tenant_id,
                definition.robot_id,
            )

            if key in self._robots:
                raise SimulationConflict(
                    "Robot already registered"
                )

            if (
                len(self._robots)
                >= self.config.maximum_robots
            ):
                raise SimulationAdapterError(
                    "Maximum robot count reached"
                )

            reason = self._collision_reason(
                definition.initial_pose,
                definition.radius_m,
                tenant_id=definition.tenant_id,
                robot_id=definition.robot_id,
            )

            if reason is not None:
                raise SimulationSafetyError(
                    f"Invalid initial pose: {reason}"
                )

            record = SimulationRobotRecord(
                definition=definition,
                pose=definition.initial_pose,
                battery_percentage=(
                    self.config.default_battery_percentage
                ),
                temperature_c=(
                    self.config.default_temperature_c
                ),
            )

            self._robots[key] = record

            self._event(
                record,
                SimulationEventType.REGISTERED,
                "Robot registered in simulation",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # APPLY COMMAND
    # --------------------------------------------------------

    def apply_command(
        self,
        command: SimulationCommand,
    ) -> dict[str, Any]:
        if not isinstance(
            command,
            SimulationCommand,
        ):
            raise SimulationValidationError(
                "Invalid command"
            )

        with self._lock:
            record = self._get(
                command.tenant_id,
                command.robot_id,
            )

            kind = command.command_type

            if kind == SimulationCommandType.STOP:
                record.linear_velocity_mps = 0.0
                record.angular_velocity_radps = 0.0

                if not (
                    record.emergency_latched
                    or record.collision_latched
                    or record.fault_latched
                ):
                    record.status = (
                        SimulationRobotStatus.IDLE
                    )

            elif kind == SimulationCommandType.PAUSE:
                if record.status in (
                    SimulationRobotStatus.COLLISION,
                    SimulationRobotStatus.FAULTED,
                    SimulationRobotStatus.EMERGENCY_STOP,
                ):
                    raise SimulationConflict(
                        "Cannot pause faulted robot"
                    )

                record.linear_velocity_mps = 0.0
                record.angular_velocity_radps = 0.0
                record.status = (
                    SimulationRobotStatus.PAUSED
                )

                self._event(
                    record,
                    SimulationEventType.PAUSED,
                    "Simulation paused",
                )

            elif kind == SimulationCommandType.RESUME:
                if record.status != (
                    SimulationRobotStatus.PAUSED
                ):
                    raise SimulationConflict(
                        "Robot is not paused"
                    )

                record.status = (
                    SimulationRobotStatus.IDLE
                )

                self._event(
                    record,
                    SimulationEventType.RESUMED,
                    "Simulation resumed",
                )

            elif kind == SimulationCommandType.RESET:
                raise SimulationConflict(
                    "Use reset_robot() with verification"
                )

            elif kind == SimulationCommandType.SET_VELOCITY:
                if (
                    record.emergency_latched
                    or record.collision_latched
                    or record.fault_latched
                ):
                    raise SimulationSafetyError(
                        "Robot has latched safety condition"
                    )

                if record.status == (
                    SimulationRobotStatus.PAUSED
                ):
                    raise SimulationConflict(
                        "Robot is paused"
                    )

                if abs(
                    command.linear_velocity_mps
                ) > self.config.maximum_linear_speed_mps:
                    raise SimulationValidationError(
                        "Linear velocity exceeds limit"
                    )

                if abs(
                    command.angular_velocity_radps
                ) > self.config.maximum_angular_speed_radps:
                    raise SimulationValidationError(
                        "Angular velocity exceeds limit"
                    )

                record.linear_velocity_mps = (
                    command.linear_velocity_mps
                )

                record.angular_velocity_radps = (
                    command.angular_velocity_radps
                )

                record.status = (
                    SimulationRobotStatus.MOVING
                    if (
                        command.linear_velocity_mps != 0
                        or command.angular_velocity_radps != 0
                    )
                    else SimulationRobotStatus.IDLE
                )

            self._event(
                record,
                SimulationEventType.COMMAND_ACCEPTED,
                f"Command: {kind.value}",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # EMERGENCY STOP
    # --------------------------------------------------------

    def emergency_stop(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
            )

            record.emergency_latched = True
            record.linear_velocity_mps = 0.0
            record.angular_velocity_radps = 0.0
            record.status = (
                SimulationRobotStatus.EMERGENCY_STOP
            )

            self._event(
                record,
                SimulationEventType.EMERGENCY_STOP,
                "Simulation emergency stop latched",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # RESET ROBOT
    # --------------------------------------------------------

    def reset_robot(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        authorized: bool,
        independently_verified: bool,
    ) -> dict[str, Any]:
        if (
            authorized is not True
            or independently_verified is not True
        ):
            raise SimulationSafetyError(
                "Verified reset authorization required"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
            )

            reason = self._collision_reason(
                record.pose,
                record.definition.radius_m,
                tenant_id=tenant_id,
                robot_id=robot_id,
            )

            if reason is not None:
                raise SimulationSafetyError(
                    f"Collision condition remains: {reason}"
                )

            if (
                record.battery_percentage
                <= self.config.critical_battery_percentage
            ):
                raise SimulationSafetyError(
                    "Critical battery condition remains"
                )

            if (
                record.temperature_c
                >= self.config.maximum_temperature_c
            ):
                raise SimulationSafetyError(
                    "Overtemperature condition remains"
                )

            record.emergency_latched = False
            record.collision_latched = False
            record.fault_latched = False

            record.linear_velocity_mps = 0.0
            record.angular_velocity_radps = 0.0

            record.status = SimulationRobotStatus.IDLE

            self._event(
                record,
                SimulationEventType.RESET,
                "Verified simulation reset",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # SIMULATION STEP
    # --------------------------------------------------------

    def step(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        dt_seconds: float,
    ) -> dict[str, Any]:
        dt = positive(
            dt_seconds,
            "dt_seconds",
        )

        if dt > self.config.maximum_step_seconds:
            raise SimulationValidationError(
                "Simulation step exceeds configured limit"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
            )

            if record.status in (
                SimulationRobotStatus.PAUSED,
                SimulationRobotStatus.EMERGENCY_STOP,
                SimulationRobotStatus.COLLISION,
                SimulationRobotStatus.FAULTED,
            ):
                return record.to_dict()

            old_pose = record.pose

            velocity = record.linear_velocity_mps
            angular = record.angular_velocity_radps

            # Midpoint integration for differential-drive
            # unicycle kinematics.
            heading = (
                old_pose.theta
                + angular * dt / 2.0
            )

            candidate = SimulationPose(
                x=(
                    old_pose.x
                    + velocity * math.cos(heading) * dt
                ),
                y=(
                    old_pose.y
                    + velocity * math.sin(heading) * dt
                ),
                theta=normalize_angle(
                    old_pose.theta + angular * dt
                ),
            )

            reason = self._collision_reason(
                candidate,
                record.definition.radius_m,
                tenant_id=tenant_id,
                robot_id=robot_id,
            )

            record.simulation_time_seconds += dt
            record.sequence_number += 1

            if reason is not None:
                record.collision_latched = True
                record.linear_velocity_mps = 0.0
                record.angular_velocity_radps = 0.0
                record.status = (
                    SimulationRobotStatus.COLLISION
                )

                self._event(
                    record,
                    SimulationEventType.COLLISION_DETECTED,
                    reason,
                )

                return record.to_dict()

            record.pose = candidate

            moving = (
                abs(velocity) > 1e-9
                or abs(angular) > 1e-9
            )

            drain = (
                self.config.moving_battery_drain_per_second
                if moving
                else self.config.idle_battery_drain_per_second
            )

            record.battery_percentage = clamp(
                record.battery_percentage - drain * dt,
                0.0,
                100.0,
            )

            if moving:
                record.temperature_c += (
                    self.config.temperature_increase_per_second
                    * dt
                )
            else:
                record.temperature_c -= (
                    self.config.temperature_cooling_per_second
                    * dt
                )

                record.temperature_c = max(
                    self.config.default_temperature_c,
                    record.temperature_c,
                )

            if (
                record.battery_percentage
                <= self.config.critical_battery_percentage
                or record.temperature_c
                >= self.config.maximum_temperature_c
            ):
                record.fault_latched = True
                record.linear_velocity_mps = 0.0
                record.angular_velocity_radps = 0.0
                record.status = (
                    SimulationRobotStatus.FAULTED
                )

            else:
                record.status = (
                    SimulationRobotStatus.MOVING
                    if moving
                    else SimulationRobotStatus.IDLE
                )

            self._event(
                record,
                SimulationEventType.STEP_COMPLETED,
                f"dt={dt}",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # SENSOR TELEMETRY
    # --------------------------------------------------------

    def generate_sensor_telemetry(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
            )

            nearest_obstacle_distance = None

            for obstacle in self.workspace.obstacles:
                distance = max(
                    0.0,
                    math.hypot(
                        record.pose.x - obstacle.x,
                        record.pose.y - obstacle.y,
                    )
                    - obstacle.radius
                    - record.definition.radius_m,
                )

                if (
                    nearest_obstacle_distance is None
                    or distance < nearest_obstacle_distance
                ):
                    nearest_obstacle_distance = distance

            return {
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "sequence_number": (
                    record.sequence_number
                ),
                "timestamp": utc_now().isoformat(),
                "pose": record.pose.to_dict(),
                "odometry": {
                    "linear_velocity_mps": (
                        record.linear_velocity_mps
                    ),
                    "angular_velocity_radps": (
                        record.angular_velocity_radps
                    ),
                },
                "battery": {
                    "percentage": (
                        record.battery_percentage
                    ),
                },
                "temperature": {
                    "celsius": record.temperature_c,
                },
                "proximity": {
                    "nearest_static_obstacle_m": (
                        nearest_obstacle_distance
                    ),
                },
                "status": record.status.value,
                "collision_latched": (
                    record.collision_latched
                ),
                "emergency_latched": (
                    record.emergency_latched
                ),
                "fault_latched": (
                    record.fault_latched
                ),
                "simulation_only": True,
            }

    # --------------------------------------------------------
    # GET ROBOT
    # --------------------------------------------------------

    def get_robot(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            return self._get(
                tenant_id,
                robot_id,
            ).to_dict()

    # --------------------------------------------------------
    # LIST ROBOTS
    # --------------------------------------------------------

    def list_robots(
        self,
        *,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        with self._lock:
            return sorted(
                [
                    record.to_dict()
                    for record in self._robots.values()
                    if (
                        record.definition.tenant_id
                        == tenant_id
                    )
                ],
                key=lambda item: item["robot_id"],
            )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        tenant_id: str,
    ) -> dict[str, Any]:
        robots = self.list_robots(
            tenant_id=tenant_id
        )

        counts = {
            status.value: 0
            for status in SimulationRobotStatus
        }

        for robot in robots:
            counts[
                robot["status"]
            ] += 1

        return {
            "tenant_id": tenant_id,
            "total_robots": len(robots),
            "status_counts": counts,
            "simulation_only": True,
        }

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        path: str | Path,
        *,
        tenant_id: str,
    ) -> Path:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            payload = {
                "generated_at": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "statistics": self.statistics(
                    tenant_id=tenant_id
                ),
                "robots": self.list_robots(
                    tenant_id=tenant_id
                ),
                "events": [
                    event.to_dict()
                    for event in self._events
                    if event.tenant_id == tenant_id
                ],
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


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    workspace = SimulationWorkspace(
        minimum_x=-10.0,
        maximum_x=10.0,
        minimum_y=-10.0,
        maximum_y=10.0,
        obstacles=(
            SimulationObstacle(
                obstacle_id="obstacle_001",
                x=5.0,
                y=5.0,
                radius=0.5,
            ),
        ),
    )

    simulator = RoboticsSimulationAdapter(
        workspace=workspace
    )

    tenant_id = "tenant_alpha"
    robot_id = "robot_001"

    simulator.register_robot(
        SimulationRobotDefinition(
            tenant_id=tenant_id,
            robot_id=robot_id,
            name="Autonomous Mobile Robot",
            radius_m=0.3,
            initial_pose=SimulationPose(
                x=0.0,
                y=0.0,
                theta=0.0,
            ),
        )
    )

    simulator.apply_command(
        SimulationCommand(
            tenant_id=tenant_id,
            robot_id=robot_id,
            command_type=(
                SimulationCommandType.SET_VELOCITY
            ),
            linear_velocity_mps=0.5,
            angular_velocity_radps=0.1,
        )
    )

    for _ in range(10):
        simulator.step(
            tenant_id=tenant_id,
            robot_id=robot_id,
            dt_seconds=0.1,
        )

    print(
        "Robot state:",
        json.dumps(
            simulator.get_robot(
                tenant_id=tenant_id,
                robot_id=robot_id,
            ),
            indent=2,
        ),
    )

    print(
        "\nSensor telemetry:",
        json.dumps(
            simulator.generate_sensor_telemetry(
                tenant_id=tenant_id,
                robot_id=robot_id,
            ),
            indent=2,
        ),
    )

    report = simulator.export_json(
        "data/robotics/simulation_adapter_report.json",
        tenant_id=tenant_id,
    )

    print("\nReport:", report)


if __name__ == "__main__":
    main()

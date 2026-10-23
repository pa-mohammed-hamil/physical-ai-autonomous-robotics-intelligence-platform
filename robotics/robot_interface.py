"""Robot Interface module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
backend/app/robotics/control/robot_interface.py

Python 3.11+
Standard library only.

UNIFIED ROBOT INTERFACE — SIMULATION ONLY

Features:
    - Abstract robot interface
    - Simulation robot adapter
    - Multi-robot registration
    - Tenant isolation
    - Robot connection lifecycle
    - Robot telemetry
    - Robot pose and coordinate frame
    - Capability declarations
    - Command submission
    - Command execution
    - Command cancellation
    - Emergency-stop simulation latch
    - Fail-closed command authorization
    - Capability and safety callbacks
    - Command deadline checks
    - Audit events
    - JSON export

IMPORTANT:
    No physical hardware commands are generated.

    Connection means attachment to an in-memory
    simulator, not a real robot or ROS 2 network.

    Emergency stop is a simulation state flag,
    not a physical safety-rated stop.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
import uuid

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================

class RobotInterfaceError(Exception):
    pass


class RobotInterfaceValidationError(RobotInterfaceError):
    pass


class RobotInterfaceNotFound(RobotInterfaceError):
    pass


class RobotInterfaceConflict(RobotInterfaceError):
    pass


class RobotInterfaceSafetyError(RobotInterfaceError):
    pass


class RobotInterfacePermissionDenied(RobotInterfaceError):
    pass


# ============================================================
# ENUMS
# ============================================================

class RobotConnectionStatus(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    BUSY = "busy"
    FAULTED = "faulted"
    EMERGENCY_STOP = "emergency_stop"


class RobotPlatformType(str, Enum):
    MOBILE_ROBOT = "mobile_robot"
    ROBOT_ARM = "robot_arm"
    MOBILE_MANIPULATOR = "mobile_manipulator"
    INSPECTION_ROBOT = "inspection_robot"
    HUMANOID = "humanoid"
    DRONE_SIMULATION = "drone_simulation"
    CUSTOM = "custom"


class RobotInterfaceCommandType(str, Enum):
    NAVIGATE = "navigate"
    MOVE_BASE = "move_base"
    MOVE_ARM = "move_arm"
    GRIPPER = "gripper"
    PICK = "pick"
    PLACE = "place"
    INSPECT = "inspect"
    SCAN = "scan"
    DOCK = "dock"
    WAIT = "wait"
    DIAGNOSTICS = "diagnostics"


class InterfaceCommandStatus(str, Enum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class RobotInterfaceEventType(str, Enum):
    REGISTERED = "registered"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    STATE_UPDATED = "state_updated"
    COMMAND_ACCEPTED = "command_accepted"
    COMMAND_REJECTED = "command_rejected"
    COMMAND_STARTED = "command_started"
    COMMAND_COMPLETED = "command_completed"
    COMMAND_FAILED = "command_failed"
    COMMAND_CANCELLED = "command_cancelled"
    EMERGENCY_STOP = "emergency_stop"
    EMERGENCY_RESET = "emergency_reset"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_id(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(
            r"[A-Za-z0-9_.-]{1,128}",
            value,
        )
    ):
        raise RobotInterfaceValidationError(
            f"Invalid {label}"
        )

    return value


def finite_number(
    value: Any,
    label: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise RobotInterfaceValidationError(
            f"{label} must be finite"
        )

    return float(value)


def positive_number(
    value: Any,
    label: str,
) -> float:
    result = finite_number(value, label)

    if result <= 0:
        raise RobotInterfaceValidationError(
            f"{label} must be positive"
        )

    return result


def valid_timestamp(
    value: datetime,
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise RobotInterfaceValidationError(
            "Timezone-aware timestamp required"
        )

    return value.astimezone(timezone.utc)


def json_object(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RobotInterfaceValidationError(
            "JSON object required"
        )

    try:
        normalized = json.loads(
            json.dumps(
                dict(value),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise RobotInterfaceValidationError(
            "Invalid JSON payload"
        ) from exc

    if not isinstance(normalized, dict):
        raise RobotInterfaceValidationError(
            "JSON object required"
        )

    return normalized


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RobotInterfaceConfig:
    maximum_robots: int = 10000
    maximum_commands: int = 100000
    maximum_audit_events: int = 20000

    maximum_payload_bytes: int = 65536
    maximum_command_ttl_seconds: float = 300.0

    telemetry_timeout_seconds: float = 5.0

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_robots",
            "maximum_commands",
            "maximum_audit_events",
            "maximum_payload_bytes",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise RobotInterfaceValidationError(
                    f"{name} must be positive"
                )

        positive_number(
            self.maximum_command_ttl_seconds,
            "maximum_command_ttl_seconds",
        )

        positive_number(
            self.telemetry_timeout_seconds,
            "telemetry_timeout_seconds",
        )

        if not self.simulation_only:
            raise RobotInterfaceValidationError(
                "Hardware control is not implemented"
            )


# ============================================================
# ROBOT POSE
# ============================================================

@dataclass(frozen=True)
class RobotPose:
    x: float
    y: float
    z: float = 0.0

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    frame_id: str = "map"

    def __post_init__(self) -> None:
        for axis in (
            "x",
            "y",
            "z",
            "roll",
            "pitch",
            "yaw",
        ):
            finite_number(
                getattr(self, axis),
                axis,
            )

        validate_id(
            self.frame_id,
            "frame_id",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "roll": self.roll,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "frame_id": self.frame_id,
        }


# ============================================================
# ROBOT SPECIFICATION
# ============================================================

@dataclass(frozen=True)
class RobotInterfaceDefinition:
    tenant_id: str
    robot_id: str

    name: str
    platform_type: RobotPlatformType

    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_id(
            self.tenant_id,
            "tenant_id",
        )

        validate_id(
            self.robot_id,
            "robot_id",
        )

        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or len(self.name) > 256
        ):
            raise RobotInterfaceValidationError(
                "Invalid robot name"
            )

        if not isinstance(
            self.platform_type,
            RobotPlatformType,
        ):
            raise RobotInterfaceValidationError(
                "Invalid robot platform"
            )

        if not isinstance(
            self.capabilities,
            tuple,
        ):
            raise RobotInterfaceValidationError(
                "Capabilities must be tuple"
            )

        for capability in self.capabilities:
            validate_id(
                capability,
                "capability",
            )

        if (
            len(self.capabilities)
            != len(set(self.capabilities))
        ):
            raise RobotInterfaceValidationError(
                "Duplicate capabilities"
            )


# ============================================================
# ROBOT STATE
# ============================================================

@dataclass(frozen=True)
class RobotInterfaceState:
    tenant_id: str
    robot_id: str

    sequence_number: int

    pose: RobotPose

    battery_percentage: float
    temperature_c: float

    operational: bool = True

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_id(
            self.tenant_id,
            "tenant_id",
        )

        validate_id(
            self.robot_id,
            "robot_id",
        )

        if (
            isinstance(self.sequence_number, bool)
            or not isinstance(self.sequence_number, int)
            or self.sequence_number < 0
        ):
            raise RobotInterfaceValidationError(
                "Invalid state sequence"
            )

        if not isinstance(self.pose, RobotPose):
            raise RobotInterfaceValidationError(
                "Invalid robot pose"
            )

        battery = finite_number(
            self.battery_percentage,
            "battery_percentage",
        )

        if not 0 <= battery <= 100:
            raise RobotInterfaceValidationError(
                "Battery percentage out of range"
            )

        finite_number(
            self.temperature_c,
            "temperature_c",
        )

        if not isinstance(self.operational, bool):
            raise RobotInterfaceValidationError(
                "operational must be boolean"
            )

        valid_timestamp(
            self.timestamp
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "sequence_number": self.sequence_number,
            "pose": self.pose.to_dict(),
            "battery_percentage": (
                self.battery_percentage
            ),
            "temperature_c": self.temperature_c,
            "operational": self.operational,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================
# COMMAND
# ============================================================

@dataclass(frozen=True)
class RobotInterfaceCommand:
    tenant_id: str
    robot_id: str
    actor_id: str

    command_type: RobotInterfaceCommandType

    required_capability: str

    payload: Mapping[str, Any]

    command_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    ttl_seconds: float = 30.0

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for name in (
            "tenant_id",
            "robot_id",
            "actor_id",
            "required_capability",
            "command_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        if not isinstance(
            self.command_type,
            RobotInterfaceCommandType,
        ):
            raise RobotInterfaceValidationError(
                "Invalid command type"
            )

        positive_number(
            self.ttl_seconds,
            "ttl_seconds",
        )

        valid_timestamp(
            self.timestamp
        )

        object.__setattr__(
            self,
            "payload",
            json_object(self.payload),
        )


# ============================================================
# COMMAND RECORD
# ============================================================

@dataclass
class RobotInterfaceCommandRecord:
    command: RobotInterfaceCommand

    status: InterfaceCommandStatus = (
        InterfaceCommandStatus.ACCEPTED
    )

    result: dict[str, Any] | None = None
    error: str | None = None

    expires_monotonic: float = 0.0

    cancellation_event: threading.Event = field(
        default_factory=threading.Event
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    finished_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command.command_id,
            "tenant_id": self.command.tenant_id,
            "robot_id": self.command.robot_id,
            "actor_id": self.command.actor_id,
            "command_type": (
                self.command.command_type.value
            ),
            "required_capability": (
                self.command.required_capability
            ),
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "created_at": (
                self.created_at.isoformat()
            ),
            "finished_at": (
                self.finished_at.isoformat()
                if self.finished_at
                else None
            ),
        }


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class RobotInterfaceAuditEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    robot_id: str
    resource_id: str

    event_type: RobotInterfaceEventType
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "resource_id": self.resource_id,
            "event_type": self.event_type.value,
            "details": self.details,
        }


# ============================================================
# ABSTRACT ROBOT INTERFACE
# ============================================================

class BaseRobotInterface(ABC):

    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_capabilities(self) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def execute(
        self,
        command: RobotInterfaceCommand,
        cancellation_event: threading.Event,
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def emergency_stop(self) -> None:
        raise NotImplementedError


# ============================================================
# SIMULATION ROBOT ADAPTER
# ============================================================

class SimulationRobotAdapter(BaseRobotInterface):

    """
    In-memory simulation adapter.

    Simulated navigation changes an in-memory pose.
    Other commands return simulated results.

    No physics engine, motion planning, collision
    avoidance or hardware execution is included.
    """

    def __init__(
        self,
        definition: RobotInterfaceDefinition,
    ) -> None:
        self.definition = definition

        self._connected = False
        self._emergency_latched = False

        self._state: RobotInterfaceState | None = None

        self._lock = threading.RLock()

    def connect(self) -> None:
        with self._lock:
            if self._emergency_latched:
                raise RobotInterfaceSafetyError(
                    "Emergency stop latched"
                )

            self._connected = True

    def disconnect(self) -> None:
        with self._lock:
            self._connected = False

    def set_state(
        self,
        state: RobotInterfaceState,
    ) -> None:
        if (
            state.tenant_id
            != self.definition.tenant_id
            or state.robot_id
            != self.definition.robot_id
        ):
            raise RobotInterfaceValidationError(
                "State identity mismatch"
            )

        with self._lock:
            if (
                self._state is not None
                and state.sequence_number
                <= self._state.sequence_number
            ):
                raise RobotInterfaceValidationError(
                    "Out-of-order state"
                )

            self._state = state

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "connected": self._connected,
                "emergency_latched": (
                    self._emergency_latched
                ),
                "state": (
                    self._state.to_dict()
                    if self._state
                    else None
                ),
            }

    def get_capabilities(self) -> tuple[str, ...]:
        return self.definition.capabilities

    def execute(
        self,
        command: RobotInterfaceCommand,
        cancellation_event: threading.Event,
    ) -> Mapping[str, Any]:
        with self._lock:
            if not self._connected:
                raise RobotInterfaceError(
                    "Simulation robot disconnected"
                )

            if self._emergency_latched:
                raise RobotInterfaceSafetyError(
                    "Emergency stop latched"
                )

            if cancellation_event.is_set():
                raise RobotInterfaceError(
                    "Command cancelled"
                )

            if (
                command.required_capability
                not in self.definition.capabilities
            ):
                raise RobotInterfaceSafetyError(
                    "Required capability unavailable"
                )

            if (
                command.command_type
                in (
                    RobotInterfaceCommandType.NAVIGATE,
                    RobotInterfaceCommandType.MOVE_BASE,
                )
            ):
                target = command.payload.get("target")

                if not isinstance(target, dict):
                    raise RobotInterfaceValidationError(
                        "Navigation target required"
                    )

                x = finite_number(
                    target.get("x"),
                    "target.x",
                )

                y = finite_number(
                    target.get("y"),
                    "target.y",
                )

                yaw = finite_number(
                    target.get("yaw", 0.0),
                    "target.yaw",
                )

                if self._state is None:
                    raise RobotInterfaceError(
                        "Robot state unavailable"
                    )

                current = self._state

                self._state = RobotInterfaceState(
                    tenant_id=current.tenant_id,
                    robot_id=current.robot_id,
                    sequence_number=(
                        current.sequence_number + 1
                    ),
                    pose=RobotPose(
                        x=x,
                        y=y,
                        z=current.pose.z,
                        yaw=yaw,
                        frame_id=current.pose.frame_id,
                    ),
                    battery_percentage=(
                        current.battery_percentage
                    ),
                    temperature_c=(
                        current.temperature_c
                    ),
                    operational=current.operational,
                )

                return {
                    "simulation_only": True,
                    "command_type": (
                        command.command_type.value
                    ),
                    "target_reached_in_simulation": True,
                    "pose": self._state.pose.to_dict(),
                }

            return {
                "simulation_only": True,
                "command_type": (
                    command.command_type.value
                ),
                "simulated_success": True,
                "payload": dict(command.payload),
            }

    def emergency_stop(self) -> None:
        with self._lock:
            self._emergency_latched = True

    def reset_emergency_stop(
        self,
        *,
        authorized: bool,
        independently_verified: bool,
    ) -> None:
        if (
            authorized is not True
            or independently_verified is not True
        ):
            raise RobotInterfacePermissionDenied(
                "Verified reset approval required"
            )

        with self._lock:
            self._emergency_latched = False


# ============================================================
# CALLBACK TYPES
# ============================================================

RobotCommandCheck = Callable[
    [RobotInterfaceCommand],
    bool,
]


# ============================================================
# UNIFIED ROBOT INTERFACE MANAGER
# ============================================================

class RoboticsRobotInterfaceManager:

    TERMINAL_STATUSES = {
        InterfaceCommandStatus.SUCCEEDED,
        InterfaceCommandStatus.FAILED,
        InterfaceCommandStatus.REJECTED,
        InterfaceCommandStatus.CANCELLED,
        InterfaceCommandStatus.EXPIRED,
    }

    def __init__(
        self,
        config: RobotInterfaceConfig | None = None,
        *,
        authorization_check: RobotCommandCheck | None = None,
        safety_check: RobotCommandCheck | None = None,
        capability_check: RobotCommandCheck | None = None,
        calibration_check: RobotCommandCheck | None = None,
    ) -> None:
        self.config = (
            config or RobotInterfaceConfig()
        )

        self.authorization_check = authorization_check
        self.safety_check = safety_check
        self.capability_check = capability_check
        self.calibration_check = calibration_check

        self._robots: dict[
            tuple[str, str],
            SimulationRobotAdapter
        ] = {}

        self._commands: dict[
            tuple[str, str, str],
            RobotInterfaceCommandRecord
        ] = {}

        self._active_commands: dict[
            tuple[str, str],
            str
        ] = {}

        self._audit_events: deque[
            RobotInterfaceAuditEvent
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # INTERNAL KEYS
    # --------------------------------------------------------

    def _robot_key(
        self,
        tenant_id: str,
        robot_id: str,
    ) -> tuple[str, str]:
        return (
            validate_id(tenant_id, "tenant_id"),
            validate_id(robot_id, "robot_id"),
        )

    def _command_key(
        self,
        tenant_id: str,
        robot_id: str,
        command_id: str,
    ) -> tuple[str, str, str]:
        return (
            validate_id(tenant_id, "tenant_id"),
            validate_id(robot_id, "robot_id"),
            validate_id(command_id, "command_id"),
        )

    def _get_robot(
        self,
        tenant_id: str,
        robot_id: str,
    ) -> SimulationRobotAdapter:
        adapter = self._robots.get(
            self._robot_key(
                tenant_id,
                robot_id,
            )
        )

        if adapter is None:
            raise RobotInterfaceNotFound(
                "Robot not registered"
            )

        return adapter

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        resource_id: str,
        event_type: RobotInterfaceEventType,
        details: str,
    ) -> None:
        self._audit_events.append(
            RobotInterfaceAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                tenant_id=tenant_id,
                robot_id=robot_id,
                resource_id=resource_id,
                event_type=event_type,
                details=details,
            )
        )

    # --------------------------------------------------------
    # REGISTER ROBOT
    # --------------------------------------------------------

    def register_robot(
        self,
        definition: RobotInterfaceDefinition,
    ) -> dict[str, Any]:
        if not isinstance(
            definition,
            RobotInterfaceDefinition,
        ):
            raise RobotInterfaceValidationError(
                "Invalid robot definition"
            )

        with self._lock:
            key = self._robot_key(
                definition.tenant_id,
                definition.robot_id,
            )

            if key in self._robots:
                raise RobotInterfaceConflict(
                    "Robot already registered"
                )

            if (
                len(self._robots)
                >= self.config.maximum_robots
            ):
                raise RobotInterfaceError(
                    "Maximum robot capacity reached"
                )

            adapter = SimulationRobotAdapter(
                definition
            )

            self._robots[key] = adapter

            self._audit(
                tenant_id=definition.tenant_id,
                robot_id=definition.robot_id,
                resource_id=definition.robot_id,
                event_type=(
                    RobotInterfaceEventType.REGISTERED
                ),
                details="Simulation robot registered",
            )

            return {
                "tenant_id": definition.tenant_id,
                "robot_id": definition.robot_id,
                "platform_type": (
                    definition.platform_type.value
                ),
                "capabilities": list(
                    definition.capabilities
                ),
                "simulation_only": True,
            }

    # --------------------------------------------------------
    # CONNECT
    # --------------------------------------------------------

    def connect_robot(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            adapter = self._get_robot(
                tenant_id,
                robot_id,
            )

            adapter.connect()

            self._audit(
                tenant_id=tenant_id,
                robot_id=robot_id,
                resource_id=robot_id,
                event_type=(
                    RobotInterfaceEventType.CONNECTED
                ),
                details="Simulation adapter connected",
            )

            return adapter.get_state()

    # --------------------------------------------------------
    # DISCONNECT
    # --------------------------------------------------------

    def disconnect_robot(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            key = self._robot_key(
                tenant_id,
                robot_id,
            )

            if key in self._active_commands:
                raise RobotInterfaceConflict(
                    "Cannot disconnect during execution"
                )

            adapter = self._get_robot(
                tenant_id,
                robot_id,
            )

            adapter.disconnect()

            self._audit(
                tenant_id=tenant_id,
                robot_id=robot_id,
                resource_id=robot_id,
                event_type=(
                    RobotInterfaceEventType.DISCONNECTED
                ),
                details="Simulation adapter disconnected",
            )

            return adapter.get_state()

    # --------------------------------------------------------
    # UPDATE STATE
    # --------------------------------------------------------

    def update_robot_state(
        self,
        state: RobotInterfaceState,
    ) -> dict[str, Any]:
        if not isinstance(
            state,
            RobotInterfaceState,
        ):
            raise RobotInterfaceValidationError(
                "Invalid robot state"
            )

        with self._lock:
            adapter = self._get_robot(
                state.tenant_id,
                state.robot_id,
            )

            adapter.set_state(state)

            self._audit(
                tenant_id=state.tenant_id,
                robot_id=state.robot_id,
                resource_id=state.robot_id,
                event_type=(
                    RobotInterfaceEventType.STATE_UPDATED
                ),
                details="Robot simulation state updated",
            )

            return adapter.get_state()

    # --------------------------------------------------------
    # CHECK CALLBACK
    # --------------------------------------------------------

    @staticmethod
    def _check(
        callback: RobotCommandCheck | None,
        command: RobotInterfaceCommand,
    ) -> bool:
        if callback is None:
            return False

        try:
            return callback(command) is True
        except Exception:
            return False

    # --------------------------------------------------------
    # COMMAND PREFLIGHT
    # --------------------------------------------------------

    def _preflight(
        self,
        command: RobotInterfaceCommand,
        adapter: SimulationRobotAdapter,
    ) -> str | None:
        state = adapter.get_state()

        if not state["connected"]:
            return "robot_disconnected"

        if state["emergency_latched"]:
            return "emergency_stop_latched"

        telemetry = state["state"]

        if telemetry is None:
            return "robot_state_missing"

        if not telemetry["operational"]:
            return "robot_not_operational"

        if (
            command.required_capability
            not in adapter.get_capabilities()
        ):
            return "capability_not_declared"

        if not all((
            self._check(
                self.authorization_check,
                command,
            ),
            self._check(
                self.safety_check,
                command,
            ),
            self._check(
                self.capability_check,
                command,
            ),
            self._check(
                self.calibration_check,
                command,
            ),
        )):
            return "external_precondition_denied"

        return None

    # --------------------------------------------------------
    # SUBMIT COMMAND
    # --------------------------------------------------------

    def submit_command(
        self,
        command: RobotInterfaceCommand,
    ) -> dict[str, Any]:
        if not isinstance(
            command,
            RobotInterfaceCommand,
        ):
            raise RobotInterfaceValidationError(
                "Invalid command"
            )

        if (
            command.ttl_seconds
            > self.config.maximum_command_ttl_seconds
        ):
            raise RobotInterfaceValidationError(
                "Command TTL exceeds maximum"
            )

        payload_size = len(
            json.dumps(
                dict(command.payload),
                allow_nan=False,
            ).encode("utf-8")
        )

        if (
            payload_size
            > self.config.maximum_payload_bytes
        ):
            raise RobotInterfaceValidationError(
                "Payload exceeds size limit"
            )

        with self._lock:
            adapter = self._get_robot(
                command.tenant_id,
                command.robot_id,
            )

            key = self._command_key(
                command.tenant_id,
                command.robot_id,
                command.command_id,
            )

            existing = self._commands.get(key)

            if existing is not None:
                if existing.command != command:
                    raise RobotInterfaceConflict(
                        "Command ID conflict"
                    )

                return existing.to_dict()

            if (
                len(self._commands)
                >= self.config.maximum_commands
            ):
                raise RobotInterfaceError(
                    "Maximum command capacity reached"
                )

            age = (
                utc_now()
                - command.timestamp
            ).total_seconds()

            record = RobotInterfaceCommandRecord(
                command=command
            )

            self._commands[key] = record

            if (
                age < 0
                or age >= command.ttl_seconds
            ):
                record.status = (
                    InterfaceCommandStatus.EXPIRED
                )

                record.error = (
                    "invalid_or_expired_timestamp"
                )

                record.finished_at = utc_now()

                return record.to_dict()

            record.expires_monotonic = (
                time.monotonic()
                + command.ttl_seconds
                - age
            )

            rejection = self._preflight(
                command,
                adapter,
            )

            if rejection:
                record.status = (
                    InterfaceCommandStatus.REJECTED
                )

                record.error = rejection
                record.finished_at = utc_now()

                self._audit(
                    tenant_id=command.tenant_id,
                    robot_id=command.robot_id,
                    resource_id=command.command_id,
                    event_type=(
                        RobotInterfaceEventType.COMMAND_REJECTED
                    ),
                    details=rejection,
                )

                return record.to_dict()

            self._audit(
                tenant_id=command.tenant_id,
                robot_id=command.robot_id,
                resource_id=command.command_id,
                event_type=(
                    RobotInterfaceEventType.COMMAND_ACCEPTED
                ),
                details="Simulation command accepted",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # EXECUTE COMMAND
    # --------------------------------------------------------

    def execute_command(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        command_id: str,
    ) -> dict[str, Any]:
        key = self._command_key(
            tenant_id,
            robot_id,
            command_id,
        )

        robot_key = self._robot_key(
            tenant_id,
            robot_id,
        )

        with self._lock:
            record = self._commands.get(key)

            if record is None:
                raise RobotInterfaceNotFound(
                    "Command not found"
                )

            if (
                record.status
                != InterfaceCommandStatus.ACCEPTED
            ):
                raise RobotInterfaceConflict(
                    "Command not executable"
                )

            if robot_key in self._active_commands:
                raise RobotInterfaceConflict(
                    "Robot already executing a command"
                )

            adapter = self._get_robot(
                tenant_id,
                robot_id,
            )

            if (
                time.monotonic()
                >= record.expires_monotonic
            ):
                record.status = (
                    InterfaceCommandStatus.EXPIRED
                )

                record.finished_at = utc_now()

                return record.to_dict()

            rejection = self._preflight(
                record.command,
                adapter,
            )

            if rejection:
                record.status = (
                    InterfaceCommandStatus.REJECTED
                )

                record.error = rejection
                record.finished_at = utc_now()

                return record.to_dict()

            self._active_commands[
                robot_key
            ] = command_id

            record.status = (
                InterfaceCommandStatus.RUNNING
            )

            self._audit(
                tenant_id=tenant_id,
                robot_id=robot_id,
                resource_id=command_id,
                event_type=(
                    RobotInterfaceEventType.COMMAND_STARTED
                ),
                details="Simulation execution started",
            )

        try:
            if record.cancellation_event.is_set():
                raise RobotInterfaceError(
                    "Command cancelled"
                )

            result = adapter.execute(
                record.command,
                record.cancellation_event,
            )

            normalized = json_object(result)
            error = None

        except Exception as exc:
            normalized = None

            error = (
                f"{type(exc).__name__}: "
                "simulation execution failed"
            )

        with self._lock:
            self._active_commands.pop(
                robot_key,
                None,
            )

            if record.cancellation_event.is_set():
                record.status = (
                    InterfaceCommandStatus.CANCELLED
                )

                record.error = (
                    "simulation_execution_cancelled"
                )

                event = (
                    RobotInterfaceEventType.COMMAND_CANCELLED
                )

            elif (
                time.monotonic()
                >= record.expires_monotonic
            ):
                record.status = (
                    InterfaceCommandStatus.EXPIRED
                )

                record.error = (
                    "simulation_deadline_exceeded"
                )

                event = (
                    RobotInterfaceEventType.COMMAND_FAILED
                )

            elif error is None:
                record.status = (
                    InterfaceCommandStatus.SUCCEEDED
                )

                record.result = normalized

                event = (
                    RobotInterfaceEventType.COMMAND_COMPLETED
                )

            else:
                record.status = (
                    InterfaceCommandStatus.FAILED
                )

                record.error = error

                event = (
                    RobotInterfaceEventType.COMMAND_FAILED
                )

            record.finished_at = utc_now()

            self._audit(
                tenant_id=tenant_id,
                robot_id=robot_id,
                resource_id=command_id,
                event_type=event,
                details=record.status.value,
            )

            return record.to_dict()

    # --------------------------------------------------------
    # CANCEL COMMAND
    # --------------------------------------------------------

    def cancel_command(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        command_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            key = self._command_key(
                tenant_id,
                robot_id,
                command_id,
            )

            record = self._commands.get(key)

            if record is None:
                raise RobotInterfaceNotFound(
                    "Command not found"
                )

            if (
                record.status
                in self.TERMINAL_STATUSES
            ):
                return record.to_dict()

            record.cancellation_event.set()

            if (
                record.status
                != InterfaceCommandStatus.RUNNING
            ):
                record.status = (
                    InterfaceCommandStatus.CANCELLED
                )

                record.finished_at = utc_now()

            self._audit(
                tenant_id=tenant_id,
                robot_id=robot_id,
                resource_id=command_id,
                event_type=(
                    RobotInterfaceEventType.COMMAND_CANCELLED
                ),
                details="Cancellation requested",
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
            adapter = self._get_robot(
                tenant_id,
                robot_id,
            )

            adapter.emergency_stop()

            for record in self._commands.values():
                command = record.command

                if (
                    command.tenant_id != tenant_id
                    or command.robot_id != robot_id
                    or record.status
                    in self.TERMINAL_STATUSES
                ):
                    continue

                record.cancellation_event.set()

                if (
                    record.status
                    != InterfaceCommandStatus.RUNNING
                ):
                    record.status = (
                        InterfaceCommandStatus.CANCELLED
                    )

                    record.finished_at = utc_now()

            self._audit(
                tenant_id=tenant_id,
                robot_id=robot_id,
                resource_id=robot_id,
                event_type=(
                    RobotInterfaceEventType.EMERGENCY_STOP
                ),
                details="Simulation emergency stop latched",
            )

            return adapter.get_state()

    # --------------------------------------------------------
    # RESET EMERGENCY STOP
    # --------------------------------------------------------

    def reset_emergency_stop(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        authorized: bool,
        independently_verified: bool,
    ) -> dict[str, Any]:
        with self._lock:
            robot_key = self._robot_key(
                tenant_id,
                robot_id,
            )

            if robot_key in self._active_commands:
                raise RobotInterfaceSafetyError(
                    "Cannot reset during execution"
                )

            adapter = self._get_robot(
                tenant_id,
                robot_id,
            )

            adapter.reset_emergency_stop(
                authorized=authorized,
                independently_verified=(
                    independently_verified
                ),
            )

            self._audit(
                tenant_id=tenant_id,
                robot_id=robot_id,
                resource_id=robot_id,
                event_type=(
                    RobotInterfaceEventType.EMERGENCY_RESET
                ),
                details="Simulation emergency stop reset",
            )

            return adapter.get_state()

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
            adapter = self._get_robot(
                tenant_id,
                robot_id,
            )

            return {
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "platform_type": (
                    adapter.definition.platform_type.value
                ),
                "capabilities": list(
                    adapter.get_capabilities()
                ),
                "state": adapter.get_state(),
                "active_command_id": (
                    self._active_commands.get(
                        self._robot_key(
                            tenant_id,
                            robot_id,
                        )
                    )
                ),
            }

    # --------------------------------------------------------
    # GET COMMAND
    # --------------------------------------------------------

    def get_command(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        command_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._commands.get(
                self._command_key(
                    tenant_id,
                    robot_id,
                    command_id,
                )
            )

            if record is None:
                raise RobotInterfaceNotFound(
                    "Command not found"
                )

            return record.to_dict()

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        tenant_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            robots = [
                adapter
                for (tenant, _), adapter
                in self._robots.items()
                if tenant == tenant_id
            ]

            commands = [
                record
                for record in self._commands.values()
                if record.command.tenant_id == tenant_id
            ]

            return {
                "total_robots": len(robots),
                "connected_robots": sum(
                    adapter.get_state()["connected"]
                    for adapter in robots
                ),
                "active_commands": sum(
                    key[0] == tenant_id
                    for key in self._active_commands
                ),
                "total_commands": len(commands),
                "successful_commands": sum(
                    record.status
                    == InterfaceCommandStatus.SUCCEEDED
                    for record in commands
                ),
                "failed_commands": sum(
                    record.status
                    == InterfaceCommandStatus.FAILED
                    for record in commands
                ),
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
                "robots": [
                    self.get_robot(
                        tenant_id=tenant_id,
                        robot_id=robot_id,
                    )
                    for (tenant, robot_id)
                    in self._robots
                    if tenant == tenant_id
                ],
                "commands": [
                    record.to_dict()
                    for record in self._commands.values()
                    if record.command.tenant_id == tenant_id
                ],
                "audit_events": [
                    event.to_dict()
                    for event in self._audit_events
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
    manager = RoboticsRobotInterfaceManager(
        authorization_check=lambda command: (
            command.actor_id == "simulation_operator"
        ),
        safety_check=lambda command: (
            command.payload.get("simulation_only")
            is True
        ),
        capability_check=lambda command: (
            command.required_capability == "navigation"
        ),
        calibration_check=lambda command: True,
    )

    tenant_id = "tenant_alpha"
    robot_id = "robot_001"

    manager.register_robot(
        RobotInterfaceDefinition(
            tenant_id=tenant_id,
            robot_id=robot_id,
            name="Autonomous Mobile Robot",
            platform_type=(
                RobotPlatformType.MOBILE_ROBOT
            ),
            capabilities=(
                "navigation",
                "inspection",
            ),
        )
    )

    manager.connect_robot(
        tenant_id=tenant_id,
        robot_id=robot_id,
    )

    manager.update_robot_state(
        RobotInterfaceState(
            tenant_id=tenant_id,
            robot_id=robot_id,
            sequence_number=1,
            pose=RobotPose(
                x=0.0,
                y=0.0,
                yaw=0.0,
            ),
            battery_percentage=90.0,
            temperature_c=32.0,
        )
    )

    command = RobotInterfaceCommand(
        tenant_id=tenant_id,
        robot_id=robot_id,
        actor_id="simulation_operator",
        command_type=(
            RobotInterfaceCommandType.NAVIGATE
        ),
        required_capability="navigation",
        payload={
            "simulation_only": True,
            "target": {
                "x": 5.0,
                "y": 3.0,
                "yaw": 0.0,
            },
        },
    )

    print(
        "Submitted:",
        json.dumps(
            manager.submit_command(command),
            indent=2,
        ),
    )

    print(
        "\nExecuted:",
        json.dumps(
            manager.execute_command(
                tenant_id=tenant_id,
                robot_id=robot_id,
                command_id=command.command_id,
            ),
            indent=2,
        ),
    )

    print(
        "\nRobot state:",
        json.dumps(
            manager.get_robot(
                tenant_id=tenant_id,
                robot_id=robot_id,
            ),
            indent=2,
        ),
    )

    report = manager.export_json(
        "data/robotics/robot_interface_report.json",
        tenant_id=tenant_id,
    )

    print("\nReport:", report)


if __name__ == "__main__":
    main()

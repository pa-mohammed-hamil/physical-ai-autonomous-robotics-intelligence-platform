"""Command Dispatcher module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
backend/app/robotics/control/command_dispatcher.py

Python 3.11+
Standard library only.

ROBOTICS COMMAND DISPATCHER — SIMULATION ONLY

Features:
- Command registration and dispatch
- Priority-based bounded queue
- Agent/VLA/planner command sources
- Tenant and robot isolation
- Idempotent command submission
- Trusted authorization callback
- External safety and capability checks
- Fail-closed dispatch
- Per-robot execution serialization
- Command TTL and deadlines
- Cancellation and emergency-stop latch
- Retry of explicitly idempotent commands
- Command lifecycle tracking
- Audit logging and JSON reporting

IMPORTANT:
    This module does NOT control physical robots.

    No ROS 2, CAN, PWM, EtherCAT, serial,
    motor driver, or actuator transport exists.

    All command handlers are simulation-only.

    Approval from this dispatcher must never
    be interpreted as permission to operate
    real-world hardware.

    Emergency stop is a logical simulation
    latch, NOT a hardware safety-rated E-stop.
"""

from __future__ import annotations

import heapq
import json
import math
import threading
import time
import uuid

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Callable, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================

class CommandDispatcherError(Exception):
    pass


class CommandValidationError(CommandDispatcherError):
    pass


class CommandNotFound(CommandDispatcherError):
    pass


class CommandQueueFull(CommandDispatcherError):
    pass


class CommandConflict(CommandDispatcherError):
    pass


# ============================================================
# ENUMS
# ============================================================

class CommandSource(str, Enum):
    AI_AGENT = "ai_agent"
    VLA_MODEL = "vla_model"
    NAVIGATION_PLANNER = "navigation_planner"
    MANIPULATION_PLANNER = "manipulation_planner"
    TASK_SERVICE = "task_service"
    SIMULATION_ENGINE = "simulation_engine"
    OPERATOR = "operator"


class CommandType(str, Enum):
    NAVIGATE = "navigate"
    MOVE_BASE = "move_base"
    MOVE_ARM = "move_arm"
    GRIPPER = "gripper"
    PICK = "pick"
    PLACE = "place"
    INSPECT = "inspect"
    SCAN = "scan"
    DOCK = "dock"
    UNDOCK = "undock"
    WAIT = "wait"
    SIMULATION_TASK = "simulation_task"


class CommandPriority(IntEnum):
    LOW = 10
    NORMAL = 20
    HIGH = 30
    CRITICAL = 40


class CommandStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    RETRY_PENDING = "retry_pending"


class CommandEventType(str, Enum):
    SUBMITTED = "submitted"
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    RETRY_SCHEDULED = "retry_scheduled"
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
        or not value
        or len(value) > 128
        or not all(
            char.isascii()
            and (
                char.isalnum()
                or char in "_.-"
            )
            for char in value
        )
    ):
        raise CommandValidationError(
            f"Invalid {label}"
        )

    return value


def positive_float(
    value: float,
    label: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise CommandValidationError(
            f"{label} must be finite and positive"
        )

    return float(value)


def json_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise CommandValidationError(
            "Payload must be a mapping"
        )

    try:
        encoded = json.dumps(
            dict(payload),
            allow_nan=False,
            separators=(",", ":"),
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise CommandValidationError(
            "Payload must be JSON serializable"
        ) from exc

    if not isinstance(decoded, dict):
        raise CommandValidationError(
            "Payload must be a JSON object"
        )

    return decoded


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class CommandDispatcherConfig:
    maximum_commands: int = 100000
    maximum_queue_size: int = 10000
    maximum_audit_events: int = 20000

    maximum_payload_bytes: int = 65536
    maximum_ttl_seconds: float = 300.0

    maximum_attempts: int = 3
    retry_delay_seconds: float = 0.2

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_commands",
            "maximum_queue_size",
            "maximum_audit_events",
            "maximum_payload_bytes",
            "maximum_attempts",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise CommandValidationError(
                    f"{name} must be positive"
                )

        positive_float(
            self.maximum_ttl_seconds,
            "maximum_ttl_seconds",
        )

        if (
            isinstance(self.retry_delay_seconds, bool)
            or not isinstance(
                self.retry_delay_seconds,
                (int, float),
            )
            or not math.isfinite(
                self.retry_delay_seconds
            )
            or self.retry_delay_seconds < 0
        ):
            raise CommandValidationError(
                "Invalid retry delay"
            )

        if not self.simulation_only:
            raise CommandValidationError(
                "Physical command dispatch unsupported"
            )


# ============================================================
# COMMAND REQUEST
# ============================================================

@dataclass(frozen=True)
class RobotCommand:
    tenant_id: str
    robot_id: str
    actor_id: str

    command_type: CommandType
    source: CommandSource

    payload: Mapping[str, Any]

    required_capability: str

    priority: CommandPriority = (
        CommandPriority.NORMAL
    )

    command_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    ttl_seconds: float = 30.0

    idempotent: bool = False

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for label in (
            "tenant_id",
            "robot_id",
            "actor_id",
            "command_id",
            "required_capability",
        ):
            validate_id(
                getattr(self, label),
                label,
            )

        if not isinstance(
            self.command_type,
            CommandType,
        ):
            raise CommandValidationError(
                "Invalid command type"
            )

        if not isinstance(
            self.source,
            CommandSource,
        ):
            raise CommandValidationError(
                "Invalid command source"
            )

        if not isinstance(
            self.priority,
            CommandPriority,
        ):
            raise CommandValidationError(
                "Invalid command priority"
            )

        if not isinstance(self.idempotent, bool):
            raise CommandValidationError(
                "idempotent must be boolean"
            )

        positive_float(
            self.ttl_seconds,
            "ttl_seconds",
        )

        if (
            not isinstance(self.timestamp, datetime)
            or self.timestamp.tzinfo is None
            or self.timestamp.utcoffset() is None
        ):
            raise CommandValidationError(
                "Timezone-aware timestamp required"
            )

        object.__setattr__(
            self,
            "payload",
            json_payload(self.payload),
        )


# ============================================================
# EXECUTION CONTEXT
# ============================================================

@dataclass(frozen=True)
class CommandExecutionContext:
    command_id: str
    tenant_id: str
    robot_id: str

    cancellation_event: threading.Event

    deadline_monotonic: float

    def cancelled(self) -> bool:
        return self.cancellation_event.is_set()

    def expired(self) -> bool:
        return (
            time.monotonic()
            >= self.deadline_monotonic
        )

    def ensure_active(self) -> None:
        if self.cancelled():
            raise CommandDispatcherError(
                "Simulation execution cancelled"
            )

        if self.expired():
            raise CommandDispatcherError(
                "Simulation execution deadline exceeded"
            )


# ============================================================
# CALLBACK INTERFACES
# ============================================================

CommandAuthorizer = Callable[
    [RobotCommand],
    bool,
]

CommandSafetyCheck = Callable[
    [RobotCommand],
    bool,
]

CapabilityCheck = Callable[
    [RobotCommand],
    bool,
]

SimulationHandler = Callable[
    [
        RobotCommand,
        CommandExecutionContext,
    ],
    Mapping[str, Any],
]


# ============================================================
# COMMAND RECORD
# ============================================================

@dataclass
class CommandRecord:
    command: RobotCommand

    status: CommandStatus = (
        CommandStatus.QUEUED
    )

    attempts: int = 0

    result: dict[str, Any] | None = None
    last_error: str | None = None

    created_at: datetime = field(
        default_factory=utc_now
    )

    started_at: datetime | None = None
    finished_at: datetime | None = None

    expires_monotonic: float = 0.0

    cancellation_event: threading.Event = field(
        default_factory=threading.Event
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command.command_id,
            "tenant_id": self.command.tenant_id,
            "robot_id": self.command.robot_id,
            "actor_id": self.command.actor_id,
            "command_type": (
                self.command.command_type.value
            ),
            "source": self.command.source.value,
            "required_capability": (
                self.command.required_capability
            ),
            "priority": int(
                self.command.priority
            ),
            "status": self.status.value,
            "attempts": self.attempts,
            "result": self.result,
            "last_error": self.last_error,
            "created_at": (
                self.created_at.isoformat()
            ),
            "started_at": (
                self.started_at.isoformat()
                if self.started_at
                else None
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
class CommandAuditEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    robot_id: str
    command_id: str

    event_type: CommandEventType
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "command_id": self.command_id,
            "event_type": self.event_type.value,
            "details": self.details,
        }


# ============================================================
# COMMAND DISPATCHER
# ============================================================

class RoboticsCommandDispatcher:

    TERMINAL_STATUSES = {
        CommandStatus.SUCCEEDED,
        CommandStatus.FAILED,
        CommandStatus.REJECTED,
        CommandStatus.CANCELLED,
        CommandStatus.EXPIRED,
    }

    def __init__(
        self,
        config: CommandDispatcherConfig | None = None,
        *,
        authorizer: CommandAuthorizer | None = None,
        safety_check: CommandSafetyCheck | None = None,
        capability_check: CapabilityCheck | None = None,
    ) -> None:
        self.config = (
            config or CommandDispatcherConfig()
        )

        self.authorizer = authorizer
        self.safety_check = safety_check
        self.capability_check = capability_check

        self._handlers: dict[
            CommandType,
            SimulationHandler
        ] = {}

        self._commands: dict[
            tuple[str, str, str],
            CommandRecord
        ] = {}

        self._queue: list[
            tuple[int, int, tuple[str, str, str]]
        ] = []

        self._sequence = 0

        self._active_robots: set[
            tuple[str, str]
        ] = set()

        self._emergency_latches: set[
            tuple[str, str]
        ] = set()

        self._audit_events: deque[
            CommandAuditEvent
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------

    def _key(
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

    def _get(
        self,
        tenant_id: str,
        robot_id: str,
        command_id: str,
    ) -> CommandRecord:
        key = self._key(
            tenant_id,
            robot_id,
            command_id,
        )

        record = self._commands.get(key)

        if record is None:
            raise CommandNotFound(
                "Command not found"
            )

        return record

    def _audit(
        self,
        record: CommandRecord,
        event_type: CommandEventType,
        details: str,
    ) -> None:
        command = record.command

        self._audit_events.append(
            CommandAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                tenant_id=command.tenant_id,
                robot_id=command.robot_id,
                command_id=command.command_id,
                event_type=event_type,
                details=details,
            )
        )

    def _enqueue(
        self,
        record: CommandRecord,
    ) -> None:
        self._sequence += 1

        key = self._key(
            record.command.tenant_id,
            record.command.robot_id,
            record.command.command_id,
        )

        heapq.heappush(
            self._queue,
            (
                -int(record.command.priority),
                self._sequence,
                key,
            ),
        )

    def _reject(
        self,
        record: CommandRecord,
        reason: str,
    ) -> None:
        record.status = CommandStatus.REJECTED
        record.last_error = reason
        record.finished_at = utc_now()

        self._audit(
            record,
            CommandEventType.REJECTED,
            reason,
        )

    @staticmethod
    def _check(
        callback: Callable[[RobotCommand], bool] | None,
        command: RobotCommand,
    ) -> bool:
        if callback is None:
            return False

        try:
            return callback(command) is True
        except Exception:
            return False

    # --------------------------------------------------------
    # REGISTER SIMULATION HANDLER
    # --------------------------------------------------------

    def register_handler(
        self,
        command_type: CommandType,
        handler: SimulationHandler,
    ) -> None:
        if not isinstance(
            command_type,
            CommandType,
        ):
            raise CommandValidationError(
                "Invalid command type"
            )

        if not callable(handler):
            raise CommandValidationError(
                "Handler must be callable"
            )

        with self._lock:
            if command_type in self._handlers:
                raise CommandConflict(
                    "Handler already registered"
                )

            self._handlers[command_type] = handler

    # --------------------------------------------------------
    # SUBMIT COMMAND
    # --------------------------------------------------------

    def submit(
        self,
        command: RobotCommand,
    ) -> dict[str, Any]:
        if not isinstance(command, RobotCommand):
            raise CommandValidationError(
                "Invalid command request"
            )

        if (
            command.ttl_seconds
            > self.config.maximum_ttl_seconds
        ):
            raise CommandValidationError(
                "Command TTL exceeds configured limit"
            )

        encoded = json.dumps(
            dict(command.payload),
            allow_nan=False,
        ).encode("utf-8")

        if (
            len(encoded)
            > self.config.maximum_payload_bytes
        ):
            raise CommandValidationError(
                "Command payload too large"
            )

        with self._lock:
            key = self._key(
                command.tenant_id,
                command.robot_id,
                command.command_id,
            )

            existing = self._commands.get(key)

            if existing is not None:
                if existing.command != command:
                    raise CommandConflict(
                        "Command ID reused with different request"
                    )

                return existing.to_dict()

            if (
                len(self._commands)
                >= self.config.maximum_commands
            ):
                raise CommandDispatcherError(
                    "Maximum command capacity reached"
                )

            if (
                len(self._queue)
                >= self.config.maximum_queue_size
            ):
                raise CommandQueueFull(
                    "Command queue full"
                )

            now = utc_now()

            age = (
                now - command.timestamp
            ).total_seconds()

            record = CommandRecord(
                command=command
            )

            self._commands[key] = record

            if age < 0:
                self._reject(
                    record,
                    "future_command_timestamp",
                )
                return record.to_dict()

            if age >= command.ttl_seconds:
                self._reject(
                    record,
                    "command_expired",
                )
                return record.to_dict()

            remaining = (
                command.ttl_seconds - age
            )

            record.expires_monotonic = (
                time.monotonic()
                + remaining
            )

            if (
                command.tenant_id,
                command.robot_id,
            ) in self._emergency_latches:
                self._reject(
                    record,
                    "robot_emergency_stop_latched",
                )
                return record.to_dict()

            if command.command_type not in self._handlers:
                self._reject(
                    record,
                    "simulation_handler_missing",
                )
                return record.to_dict()

            if not self._check(
                self.authorizer,
                command,
            ):
                self._reject(
                    record,
                    "authorization_denied",
                )
                return record.to_dict()

            if not self._check(
                self.capability_check,
                command,
            ):
                self._reject(
                    record,
                    "capability_unavailable",
                )
                return record.to_dict()

            if not self._check(
                self.safety_check,
                command,
            ):
                self._reject(
                    record,
                    "safety_check_denied",
                )
                return record.to_dict()

            self._enqueue(record)

            self._audit(
                record,
                CommandEventType.SUBMITTED,
                "Simulation command queued",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # PROCESS QUEUE
    # --------------------------------------------------------

    def process_queue(
        self,
        *,
        maximum_commands: int = 100,
    ) -> dict[str, int]:
        if (
            isinstance(maximum_commands, bool)
            or not isinstance(maximum_commands, int)
            or maximum_commands <= 0
        ):
            raise CommandValidationError(
                "maximum_commands must be positive"
            )

        processed = 0
        failed = 0
        expired = 0
        skipped = 0

        for _ in range(maximum_commands):
            with self._lock:
                if not self._queue:
                    break

                deferred = []
                selected = None

                while self._queue:
                    item = heapq.heappop(
                        self._queue
                    )

                    _, _, key = item

                    record = self._commands.get(key)

                    if record is None:
                        continue

                    if record.status not in (
                        CommandStatus.QUEUED,
                        CommandStatus.RETRY_PENDING,
                    ):
                        continue

                    robot_key = (
                        record.command.tenant_id,
                        record.command.robot_id,
                    )

                    if robot_key in self._active_robots:
                        deferred.append(item)
                        continue

                    selected = record
                    break

                for item in deferred:
                    heapq.heappush(
                        self._queue,
                        item,
                    )

                if selected is None:
                    break

                record = selected
                command = record.command

                robot_key = (
                    command.tenant_id,
                    command.robot_id,
                )

                if (
                    time.monotonic()
                    >= record.expires_monotonic
                ):
                    record.status = CommandStatus.EXPIRED
                    record.finished_at = utc_now()

                    self._audit(
                        record,
                        CommandEventType.EXPIRED,
                        "Command TTL exceeded",
                    )

                    expired += 1
                    continue

                if robot_key in self._emergency_latches:
                    record.status = CommandStatus.CANCELLED
                    record.finished_at = utc_now()

                    self._audit(
                        record,
                        CommandEventType.CANCELLED,
                        "Emergency stop active",
                    )

                    skipped += 1
                    continue

                if not all((
                    self._check(
                        self.authorizer,
                        command,
                    ),
                    self._check(
                        self.capability_check,
                        command,
                    ),
                    self._check(
                        self.safety_check,
                        command,
                    ),
                )):
                    self._reject(
                        record,
                        "dispatch_precondition_denied",
                    )

                    skipped += 1
                    continue

                handler = self._handlers.get(
                    command.command_type
                )

                if handler is None:
                    self._reject(
                        record,
                        "handler_unavailable",
                    )

                    skipped += 1
                    continue

                record.status = CommandStatus.RUNNING
                record.attempts += 1

                record.started_at = utc_now()

                self._active_robots.add(
                    robot_key
                )

                self._audit(
                    record,
                    CommandEventType.STARTED,
                    "Simulation execution started",
                )

                context = CommandExecutionContext(
                    command_id=command.command_id,
                    tenant_id=command.tenant_id,
                    robot_id=command.robot_id,
                    cancellation_event=(
                        record.cancellation_event
                    ),
                    deadline_monotonic=(
                        record.expires_monotonic
                    ),
                )

            # Run handler outside the registry lock.
            try:
                context.ensure_active()

                result = handler(
                    command,
                    context,
                )

                context.ensure_active()

                normalized_result = (
                    json_payload(result)
                )

                error = None

            except Exception as exc:
                normalized_result = None
                error = (
                    f"{type(exc).__name__}: "
                    "simulation handler failed"
                )

            with self._lock:
                self._active_robots.discard(
                    robot_key
                )

                # Cancellation and emergency stop
                # take precedence over handler result.
                if (
                    record.cancellation_event.is_set()
                    or robot_key in self._emergency_latches
                ):
                    record.status = (
                        CommandStatus.CANCELLED
                    )

                    record.last_error = (
                        "execution_cancelled"
                    )

                    record.finished_at = utc_now()

                    self._audit(
                        record,
                        CommandEventType.CANCELLED,
                        "Simulation execution cancelled",
                    )

                    skipped += 1
                    continue

                if (
                    time.monotonic()
                    >= record.expires_monotonic
                ):
                    record.status = (
                        CommandStatus.EXPIRED
                    )

                    record.finished_at = utc_now()

                    self._audit(
                        record,
                        CommandEventType.EXPIRED,
                        "Execution deadline exceeded",
                    )

                    expired += 1
                    continue

                if error is None:
                    record.status = (
                        CommandStatus.SUCCEEDED
                    )

                    record.result = normalized_result

                    record.finished_at = utc_now()

                    self._audit(
                        record,
                        CommandEventType.SUCCEEDED,
                        "Simulation execution succeeded",
                    )

                    processed += 1

                else:
                    record.last_error = error

                    if (
                        command.idempotent
                        and record.attempts
                        < self.config.maximum_attempts
                    ):
                        record.status = (
                            CommandStatus.RETRY_PENDING
                        )

                        self._enqueue(record)

                        self._audit(
                            record,
                            CommandEventType.RETRY_SCHEDULED,
                            error,
                        )

                    else:
                        record.status = (
                            CommandStatus.FAILED
                        )

                        record.finished_at = utc_now()

                        self._audit(
                            record,
                            CommandEventType.FAILED,
                            error,
                        )

                        failed += 1

        with self._lock:
            remaining = len(self._queue)

        return {
            "succeeded": processed,
            "failed": failed,
            "expired": expired,
            "cancelled_or_rejected": skipped,
            "remaining_queue": remaining,
        }

    # --------------------------------------------------------
    # CANCEL COMMAND
    # --------------------------------------------------------

    def cancel(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        command_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                command_id,
            )

            if record.status in self.TERMINAL_STATUSES:
                return record.to_dict()

            record.cancellation_event.set()

            if record.status != CommandStatus.RUNNING:
                record.status = (
                    CommandStatus.CANCELLED
                )

                record.finished_at = utc_now()

                self._audit(
                    record,
                    CommandEventType.CANCELLED,
                    "Command cancelled",
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
        robot_key = (
            validate_id(tenant_id, "tenant_id"),
            validate_id(robot_id, "robot_id"),
        )

        with self._lock:
            self._emergency_latches.add(
                robot_key
            )

            affected = []

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

                if record.status != CommandStatus.RUNNING:
                    record.status = (
                        CommandStatus.CANCELLED
                    )

                    record.finished_at = utc_now()

                affected.append(
                    command.command_id
                )

                self._audit(
                    record,
                    CommandEventType.EMERGENCY_STOP,
                    "Simulation emergency-stop latch",
                )

            return {
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "simulation_stop_latched": True,
                "affected_commands": affected,
                "timestamp": utc_now().isoformat(),
            }

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
        if (
            authorized is not True
            or independently_verified is not True
        ):
            raise CommandDispatcherError(
                "Verified reset approval required"
            )

        robot_key = (
            validate_id(tenant_id, "tenant_id"),
            validate_id(robot_id, "robot_id"),
        )

        with self._lock:
            if robot_key in self._active_robots:
                raise CommandDispatcherError(
                    "Cannot reset while execution is active"
                )

            self._emergency_latches.discard(
                robot_key
            )

            return {
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "simulation_stop_latched": False,
                "timestamp": utc_now().isoformat(),
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
            return self._get(
                tenant_id,
                robot_id,
                command_id,
            ).to_dict()

    # --------------------------------------------------------
    # LIST COMMANDS
    # --------------------------------------------------------

    def list_commands(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            return [
                record.to_dict()
                for record in self._commands.values()
                if (
                    record.command.tenant_id
                    == tenant_id
                    and (
                        robot_id is None
                        or record.command.robot_id
                        == robot_id
                    )
                )
            ]

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            records = [
                record
                for record in self._commands.values()
                if (
                    tenant_id is None
                    or record.command.tenant_id
                    == tenant_id
                )
            ]

            counts = {
                status.value: 0
                for status in CommandStatus
            }

            for record in records:
                counts[record.status.value] += 1

            return {
                "total_commands": len(records),
                "status_counts": counts,
                "queue_entries": len(self._queue),
                "active_robots": len(
                    self._active_robots
                ),
                "emergency_latches": len(
                    self._emergency_latches
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
                "simulation_only": True,
                "statistics": self.statistics(
                    tenant_id=tenant_id
                ),
                "commands": self.list_commands(
                    tenant_id=tenant_id
                ),
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

def simulation_navigation_handler(
    command: RobotCommand,
    context: CommandExecutionContext,
) -> dict[str, Any]:
    context.ensure_active()

    target = command.payload.get("target")

    if not isinstance(target, dict):
        raise CommandValidationError(
            "Navigation target missing"
        )

    return {
        "robot_id": command.robot_id,
        "command_id": command.command_id,
        "target": target,
        "simulation_completed": True,
    }


def main() -> None:
    dispatcher = RoboticsCommandDispatcher(
        authorizer=lambda command: (
            command.actor_id == "simulation_controller"
        ),
        capability_check=lambda command: (
            command.required_capability == "navigation"
        ),
        safety_check=lambda command: (
            command.payload.get(
                "simulation_only"
            ) is True
        ),
    )

    dispatcher.register_handler(
        CommandType.NAVIGATE,
        simulation_navigation_handler,
    )

    command = RobotCommand(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        actor_id="simulation_controller",
        command_type=CommandType.NAVIGATE,
        source=CommandSource.NAVIGATION_PLANNER,
        required_capability="navigation",
        priority=CommandPriority.NORMAL,
        payload={
            "target": {
                "x": 5.0,
                "y": 3.0,
                "theta": 0.0,
            },
            "simulation_only": True,
        },
        ttl_seconds=30.0,
        idempotent=True,
    )

    print(
        "Submitted:",
        json.dumps(
            dispatcher.submit(command),
            indent=2,
        ),
    )

    print(
        "\nProcessing:",
        json.dumps(
            dispatcher.process_queue(),
            indent=2,
        ),
    )

    print(
        "\nFinal command:",
        json.dumps(
            dispatcher.get_command(
                tenant_id="tenant_alpha",
                robot_id="robot_001",
                command_id=command.command_id,
            ),
            indent=2,
        ),
    )

    report = dispatcher.export_json(
        "data/robotics/command_dispatcher_report.json",
        tenant_id="tenant_alpha",
    )

    print("\nReport:", report)


if __name__ == "__main__":
    main()

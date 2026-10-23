"""Action Client module.

TODO: implement and test; no real robot control is enabled.
"""
"""Message Converter module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File: backend/app/robotics/control/action_client.py

Python 3.11+
Standard library only.

Simulation-only robot action client.

Features:
- Robot action request validation
- Navigation and manipulation action types
- Pluggable safety checks
- Simulation transport abstraction
- Action lifecycle tracking
- Idempotent submission
- Request expiration
- Command timeout handling
- Action cancellation
- Robot-specific concurrency control
- Tenant isolation
- Audit logging
- JSON report export

IMPORTANT:
This module does not communicate with physical
robot hardware or execute real actuator commands.

The in-memory simulation transport is a demo.
Real robot control requires an independent,
validated safety architecture.
"""

from __future__ import annotations

import json
import math
import threading
import time
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol


# ============================================================
# EXCEPTIONS
# ============================================================

class ActionClientError(Exception):
    pass


class ActionValidationError(ActionClientError):
    pass


class ActionNotFound(ActionClientError):
    pass


class ActionRejected(ActionClientError):
    pass


class ActionConflict(ActionClientError):
    pass


# ============================================================
# ENUMS
# ============================================================

class ActionType(str, Enum):
    NAVIGATE = "navigate"
    MOVE_BASE = "move_base"
    MOVE_ARM = "move_arm"
    PICK = "pick"
    PLACE = "place"
    GRIPPER = "gripper"
    INSPECT = "inspect"
    SCAN = "scan"
    WAIT = "wait"
    STOP = "stop"
    CUSTOM_SIMULATION = "custom_simulation"


class ActionStatus(str, Enum):
    PENDING = "pending"
    REJECTED = "rejected"
    DISPATCHED = "dispatched"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class SafetyDecision(str, Enum):
    ALLOW_SIMULATION = "allow_simulation"
    DENY = "deny"


class ActionEventType(str, Enum):
    CREATED = "created"
    REJECTED = "rejected"
    DISPATCHED = "dispatched"
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


TERMINAL_STATUSES = {
    ActionStatus.REJECTED,
    ActionStatus.SUCCEEDED,
    ActionStatus.FAILED,
    ActionStatus.CANCELLED,
    ActionStatus.TIMED_OUT,
}


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_id(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not all(
            character.isalnum() or character in "_-"
            for character in value
        )
    ):
        raise ActionValidationError(
            f"Invalid {name}"
        )

    return value


def positive_float(
    value: float,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ActionValidationError(
            f"{name} must be finite and positive"
        )

    return float(value)


def validate_timestamp(
    value: datetime,
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ActionValidationError(
            "Timezone-aware timestamp required"
        )

    return value.astimezone(timezone.utc)


def safe_json(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        serialized = json.dumps(
            dict(value),
            sort_keys=True,
            allow_nan=False,
        )

        result = json.loads(serialized)

        if not isinstance(result, dict):
            raise ValueError(
                "Expected JSON object"
            )

        return result

    except (TypeError, ValueError) as exc:
        raise ActionValidationError(
            "Parameters must be JSON serializable"
        ) from exc


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ActionClientConfig:
    default_timeout_seconds: float = 30.0
    maximum_timeout_seconds: float = 300.0

    maximum_actions: int = 100000
    maximum_audit_events: int = 20000

    require_safety_checks: bool = True
    one_active_action_per_robot: bool = True

    simulation_only: bool = True

    def __post_init__(self) -> None:
        positive_float(
            self.default_timeout_seconds,
            "default_timeout_seconds",
        )

        positive_float(
            self.maximum_timeout_seconds,
            "maximum_timeout_seconds",
        )

        if (
            self.default_timeout_seconds
            > self.maximum_timeout_seconds
        ):
            raise ActionValidationError(
                "Default timeout exceeds maximum"
            )

        for name in (
            "maximum_actions",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ActionValidationError(
                    f"{name} must be a positive integer"
                )

        if not self.simulation_only:
            raise ActionValidationError(
                "Physical hardware control unsupported"
            )


# ============================================================
# ACTION REQUEST
# ============================================================

@dataclass(frozen=True)
class RobotActionRequest:
    tenant_id: str
    robot_id: str
    action_type: ActionType

    parameters: Mapping[str, Any] = field(
        default_factory=dict
    )

    request_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    timeout_seconds: float = 30.0

    created_at: datetime = field(
        default_factory=utc_now
    )

    simulation_only: bool = True

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")
        validate_id(self.request_id, "request_id")

        if not isinstance(self.action_type, ActionType):
            raise ActionValidationError(
                "Invalid action type"
            )

        positive_float(
            self.timeout_seconds,
            "timeout_seconds",
        )

        validate_timestamp(self.created_at)

        if self.simulation_only is not True:
            raise ActionValidationError(
                "Only simulation actions are supported"
            )

        object.__setattr__(
            self,
            "parameters",
            safe_json(self.parameters),
        )


# ============================================================
# SAFETY CHECK RESULT
# ============================================================

@dataclass(frozen=True)
class ActionSafetyResult:
    decision: SafetyDecision
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.decision,
            SafetyDecision,
        ):
            raise ActionValidationError(
                "Invalid safety decision"
            )

        if not isinstance(self.reason, str):
            raise ActionValidationError(
                "Invalid safety reason"
            )


SafetyCheck = Callable[
    [RobotActionRequest],
    ActionSafetyResult,
]


# ============================================================
# TRANSPORT INTERFACE
# ============================================================

class SimulationActionTransport(Protocol):

    def submit(
        self,
        action_id: str,
        request: RobotActionRequest,
    ) -> str:
        ...

    def get_status(
        self,
        transport_id: str,
    ) -> ActionStatus:
        ...

    def cancel(
        self,
        transport_id: str,
    ) -> bool:
        ...


# ============================================================
# IN-MEMORY SIMULATION TRANSPORT
# ============================================================

class InMemorySimulationTransport:
    """
    Demonstration transport.

    Actions enter RUNNING state immediately.
    They can be completed explicitly by tests
    or simulation workers.

    No physical action is performed.
    """

    def __init__(self) -> None:
        self._actions: dict[
            str,
            ActionStatus
        ] = {}

        self._lock = threading.RLock()

    def submit(
        self,
        action_id: str,
        request: RobotActionRequest,
    ) -> str:
        with self._lock:
            if action_id in self._actions:
                raise ActionConflict(
                    "Action already submitted"
                )

            self._actions[action_id] = (
                ActionStatus.RUNNING
            )

            return action_id

    def get_status(
        self,
        transport_id: str,
    ) -> ActionStatus:
        with self._lock:
            if transport_id not in self._actions:
                raise ActionNotFound(
                    "Simulation action not found"
                )

            return self._actions[transport_id]

    def cancel(
        self,
        transport_id: str,
    ) -> bool:
        with self._lock:
            status = self._actions.get(
                transport_id
            )

            if status is None:
                return False

            if status in TERMINAL_STATUSES:
                return False

            self._actions[transport_id] = (
                ActionStatus.CANCELLED
            )

            return True

    def complete(
        self,
        transport_id: str,
        *,
        success: bool = True,
    ) -> None:
        with self._lock:
            status = self._actions.get(
                transport_id
            )

            if status != ActionStatus.RUNNING:
                raise ActionConflict(
                    "Action is not running"
                )

            self._actions[transport_id] = (
                ActionStatus.SUCCEEDED
                if success
                else ActionStatus.FAILED
            )


# ============================================================
# ACTION RECORD
# ============================================================

@dataclass
class ActionRecord:
    action_id: str
    request: RobotActionRequest

    status: ActionStatus = ActionStatus.PENDING

    transport_id: str | None = None
    safety_results: list[
        ActionSafetyResult
    ] = field(default_factory=list)

    created_at: datetime = field(
        default_factory=utc_now
    )

    started_at: datetime | None = None
    finished_at: datetime | None = None

    deadline_monotonic: float | None = None

    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "request_id": self.request.request_id,
            "tenant_id": self.request.tenant_id,
            "robot_id": self.request.robot_id,
            "action_type": (
                self.request.action_type.value
            ),
            "parameters": dict(
                self.request.parameters
            ),
            "status": self.status.value,
            "transport_id": self.transport_id,
            "safety_results": [
                {
                    "decision": result.decision.value,
                    "reason": result.reason,
                }
                for result in self.safety_results
            ],
            "created_at": self.created_at.isoformat(),
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
            "error": self.error,
        }


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class ActionAuditEvent:
    event_id: str
    timestamp: datetime

    action_id: str
    tenant_id: str
    robot_id: str

    event_type: ActionEventType
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "action_id": self.action_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "event_type": self.event_type.value,
            "details": self.details,
        }


# ============================================================
# ACTION CLIENT
# ============================================================

class RoboticsActionClient:

    def __init__(
        self,
        *,
        config: ActionClientConfig | None = None,
        transport: SimulationActionTransport | None = None,
        safety_checks: list[SafetyCheck] | None = None,
    ) -> None:
        self.config = (
            config or ActionClientConfig()
        )

        self.transport = (
            transport or InMemorySimulationTransport()
        )

        self.safety_checks = list(
            safety_checks or []
        )

        self.actions: dict[
            str,
            ActionRecord
        ] = {}

        self.request_index: dict[
            tuple[str, str],
            str
        ] = {}

        self.active_robot_actions: dict[
            tuple[str, str],
            str
        ] = {}

        self.audit_events: list[
            ActionAuditEvent
        ] = []

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit(
        self,
        record: ActionRecord,
        event_type: ActionEventType,
        details: str,
    ) -> None:
        self.audit_events.append(
            ActionAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                action_id=record.action_id,
                tenant_id=record.request.tenant_id,
                robot_id=record.request.robot_id,
                event_type=event_type,
                details=details,
            )
        )

        if (
            len(self.audit_events)
            > self.config.maximum_audit_events
        ):
            self.audit_events.pop(0)

    # --------------------------------------------------------
    # INTERNAL GET
    # --------------------------------------------------------

    def _get(
        self,
        tenant_id: str,
        action_id: str,
    ) -> ActionRecord:
        validate_id(tenant_id, "tenant_id")

        record = self.actions.get(
            action_id
        )

        if (
            record is None
            or record.request.tenant_id != tenant_id
        ):
            raise ActionNotFound(
                "Action not found"
            )

        return record

    # --------------------------------------------------------
    # SAFETY CHECKS
    # --------------------------------------------------------

    def _check_safety(
        self,
        request: RobotActionRequest,
    ) -> list[ActionSafetyResult]:
        if (
            self.config.require_safety_checks
            and not self.safety_checks
        ):
            return [
                ActionSafetyResult(
                    decision=SafetyDecision.DENY,
                    reason="No safety checks configured",
                )
            ]

        results: list[ActionSafetyResult] = []

        for check in self.safety_checks:
            try:
                result = check(request)

                if not isinstance(
                    result,
                    ActionSafetyResult,
                ):
                    raise TypeError(
                        "Invalid safety result"
                    )

                results.append(result)

            except Exception as exc:
                results.append(
                    ActionSafetyResult(
                        decision=SafetyDecision.DENY,
                        reason=(
                            "Safety check failed: "
                            f"{type(exc).__name__}"
                        ),
                    )
                )

        return results

    # --------------------------------------------------------
    # SUBMIT ACTION
    # --------------------------------------------------------

    def submit(
        self,
        request: RobotActionRequest,
    ) -> dict[str, Any]:
        if (
            request.timeout_seconds
            > self.config.maximum_timeout_seconds
        ):
            raise ActionValidationError(
                "Action timeout exceeds maximum"
            )

        if request.created_at > utc_now():
            raise ActionValidationError(
                "Future action timestamp rejected"
            )

        with self._lock:
            request_key = (
                request.tenant_id,
                request.request_id,
            )

            previous_id = self.request_index.get(
                request_key
            )

            if previous_id is not None:
                previous = self.actions[
                    previous_id
                ]

                if (
                    previous.request.robot_id
                    != request.robot_id
                    or previous.request.action_type
                    != request.action_type
                    or previous.request.parameters
                    != request.parameters
                ):
                    raise ActionConflict(
                        "Request ID reused with different action"
                    )

                return previous.to_dict()

            if (
                len(self.actions)
                >= self.config.maximum_actions
            ):
                raise ActionClientError(
                    "Action capacity reached"
                )

            robot_key = (
                request.tenant_id,
                request.robot_id,
            )

            if self.config.one_active_action_per_robot:
                active_id = (
                    self.active_robot_actions.get(
                        robot_key
                    )
                )

                if active_id is not None:
                    active_record = (
                        self.actions[active_id]
                    )

                    if (
                        active_record.status
                        not in TERMINAL_STATUSES
                    ):
                        raise ActionConflict(
                            "Robot already has active action"
                        )

            record = ActionRecord(
                action_id=str(uuid.uuid4()),
                request=request,
            )

            self.actions[
                record.action_id
            ] = record

            self.request_index[
                request_key
            ] = record.action_id

            self._audit(
                record,
                ActionEventType.CREATED,
                "Action request created",
            )

            results = self._check_safety(
                request
            )

            record.safety_results = results

            if any(
                result.decision
                != SafetyDecision.ALLOW_SIMULATION
                for result in results
            ):
                record.status = ActionStatus.REJECTED
                record.finished_at = utc_now()
                record.error = (
                    "Safety validation denied action"
                )

                self._audit(
                    record,
                    ActionEventType.REJECTED,
                    record.error,
                )

                return record.to_dict()

            try:
                record.transport_id = (
                    self.transport.submit(
                        record.action_id,
                        request,
                    )
                )

                record.status = (
                    ActionStatus.RUNNING
                )

                record.started_at = utc_now()

                record.deadline_monotonic = (
                    time.monotonic()
                    + request.timeout_seconds
                )

                self.active_robot_actions[
                    robot_key
                ] = record.action_id

                self._audit(
                    record,
                    ActionEventType.DISPATCHED,
                    "Action dispatched to simulator",
                )

                self._audit(
                    record,
                    ActionEventType.STARTED,
                    "Simulation action started",
                )

            except Exception as exc:
                record.status = ActionStatus.FAILED
                record.finished_at = utc_now()
                record.error = (
                    "Simulation transport error: "
                    f"{type(exc).__name__}"
                )

                self._audit(
                    record,
                    ActionEventType.FAILED,
                    record.error,
                )

            return record.to_dict()

    # --------------------------------------------------------
    # REFRESH ACTION
    # --------------------------------------------------------

    def _refresh(
        self,
        record: ActionRecord,
    ) -> None:
        if record.status in TERMINAL_STATUSES:
            return

        if (
            record.deadline_monotonic is not None
            and time.monotonic()
            >= record.deadline_monotonic
        ):
            if record.transport_id is not None:
                try:
                    self.transport.cancel(
                        record.transport_id
                    )
                except Exception:
                    pass

            record.status = (
                ActionStatus.TIMED_OUT
            )
            record.finished_at = utc_now()
            record.error = (
                "Simulation action timeout"
            )

            self._audit(
                record,
                ActionEventType.TIMED_OUT,
                record.error,
            )

        elif record.transport_id is not None:
            try:
                transport_status = (
                    self.transport.get_status(
                        record.transport_id
                    )
                )

                if transport_status in TERMINAL_STATUSES:
                    record.status = transport_status
                    record.finished_at = utc_now()

                    event_type = {
                        ActionStatus.SUCCEEDED:
                            ActionEventType.SUCCEEDED,
                        ActionStatus.FAILED:
                            ActionEventType.FAILED,
                        ActionStatus.CANCELLED:
                            ActionEventType.CANCELLED,
                        ActionStatus.TIMED_OUT:
                            ActionEventType.TIMED_OUT,
                        ActionStatus.REJECTED:
                            ActionEventType.REJECTED,
                    }[transport_status]

                    self._audit(
                        record,
                        event_type,
                        "Simulation transport status updated",
                    )

            except Exception as exc:
                record.status = ActionStatus.FAILED
                record.finished_at = utc_now()
                record.error = (
                    "Status query failed: "
                    f"{type(exc).__name__}"
                )

                self._audit(
                    record,
                    ActionEventType.FAILED,
                    record.error,
                )

        if record.status in TERMINAL_STATUSES:
            robot_key = (
                record.request.tenant_id,
                record.request.robot_id,
            )

            if (
                self.active_robot_actions.get(
                    robot_key
                )
                == record.action_id
            ):
                self.active_robot_actions.pop(
                    robot_key,
                    None,
                )

    # --------------------------------------------------------
    # GET ACTION
    # --------------------------------------------------------

    def get_action(
        self,
        *,
        tenant_id: str,
        action_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                action_id,
            )

            self._refresh(record)

            return record.to_dict()

    # --------------------------------------------------------
    # CANCEL ACTION
    # --------------------------------------------------------

    def cancel(
        self,
        *,
        tenant_id: str,
        action_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                action_id,
            )

            self._refresh(record)

            if record.status in TERMINAL_STATUSES:
                return record.to_dict()

            if record.transport_id is None:
                raise ActionClientError(
                    "Action has no simulation transport"
                )

            cancelled = self.transport.cancel(
                record.transport_id
            )

            if not cancelled:
                self._refresh(record)
                return record.to_dict()

            record.status = ActionStatus.CANCELLED
            record.finished_at = utc_now()

            robot_key = (
                record.request.tenant_id,
                record.request.robot_id,
            )

            if (
                self.active_robot_actions.get(
                    robot_key
                )
                == record.action_id
            ):
                self.active_robot_actions.pop(
                    robot_key,
                    None,
                )

            self._audit(
                record,
                ActionEventType.CANCELLED,
                "Simulation action cancelled",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # LIST ACTIONS
    # --------------------------------------------------------

    def list_actions(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
        status: ActionStatus | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            results = []

            for record in self.actions.values():
                if record.request.tenant_id != tenant_id:
                    continue

                if (
                    robot_id is not None
                    and record.request.robot_id
                    != robot_id
                ):
                    continue

                self._refresh(record)

                if (
                    status is not None
                    and record.status != status
                ):
                    continue

                results.append(
                    record.to_dict()
                )

            return results

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
                for record in self.actions.values()
                if (
                    tenant_id is None
                    or record.request.tenant_id
                    == tenant_id
                )
            ]

            for record in records:
                self._refresh(record)

            return {
                "total_actions": len(records),
                "running": sum(
                    record.status == ActionStatus.RUNNING
                    for record in records
                ),
                "succeeded": sum(
                    record.status == ActionStatus.SUCCEEDED
                    for record in records
                ),
                "failed": sum(
                    record.status == ActionStatus.FAILED
                    for record in records
                ),
                "rejected": sum(
                    record.status == ActionStatus.REJECTED
                    for record in records
                ),
                "cancelled": sum(
                    record.status == ActionStatus.CANCELLED
                    for record in records
                ),
                "timed_out": sum(
                    record.status == ActionStatus.TIMED_OUT
                    for record in records
                ),
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
                "actions": self.list_actions(
                    tenant_id=tenant_id
                ),
                "audit_events": [
                    event.to_dict()
                    for event in self.audit_events
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

def demo_safety_check(
    request: RobotActionRequest,
) -> ActionSafetyResult:
    """
    Demo-only allowlist check.

    Replace with adapters for:
    - safety_monitor.py
    - command_interlock.py
    - workspace_limits.py
    - speed_limits.py
    - human_approval.py
    - watchdog.py

    Never treat this demonstration check
    as production safety authorization.
    """
    if request.action_type == ActionType.STOP:
        return ActionSafetyResult(
            decision=SafetyDecision.ALLOW_SIMULATION,
            reason="Simulation stop action",
        )

    if request.action_type in {
        ActionType.NAVIGATE,
        ActionType.INSPECT,
        ActionType.WAIT,
    }:
        return ActionSafetyResult(
            decision=SafetyDecision.ALLOW_SIMULATION,
            reason="Demo simulation action permitted",
        )

    return ActionSafetyResult(
        decision=SafetyDecision.DENY,
        reason="Action not permitted by demo policy",
    )


def main() -> None:
    transport = InMemorySimulationTransport()

    client = RoboticsActionClient(
        transport=transport,
        safety_checks=[demo_safety_check],
    )

    request = RobotActionRequest(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        action_type=ActionType.NAVIGATE,
        parameters={
            "target": {
                "x": 4.0,
                "y": 3.0,
                "z": 0.0,
            },
            "requested_speed_mps": 0.5,
        },
    )

    submitted = client.submit(request)

    print(
        "Submitted action:"
    )

    print(
        json.dumps(
            submitted,
            indent=2,
        )
    )

    if submitted["status"] == "running":
        transport.complete(
            submitted["transport_id"],
            success=True,
        )

        result = client.get_action(
            tenant_id="tenant_alpha",
            action_id=submitted["action_id"],
        )

        print(
            "Final action:"
        )

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

    client.export_json(
        "data/robotics/action_client_report.json",
        tenant_id="tenant_alpha",
    )


if __name__ == "__main__":
    main()

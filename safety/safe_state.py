
"""
Physical AI & Autonomous Robotics Intelligence Platform

File: backend/app/security/safe_state.py

Python 3.11+
Standard library only.

Features:
    - Safe-state lifecycle management
    - Protective-stop and emergency-stop requests
    - Latched faults
    - Fail-closed command gating
    - Robot telemetry freshness validation
    - Hazard and interlock integration signals
    - Recovery approval workflow
    - Independent safety verification requirement
    - Tenant-scoped state isolation
    - Immutable transition records
    - JSON reporting

SAFETY:
    Simulation and supervisory decision support ONLY.

    This module does not:
      - Send motor commands
      - Actuate physical brakes
      - Perform certified emergency stops
      - Reset hardware safety circuits
      - Authorize physical robot restart

    A software STOP_REQUESTED state does not
    prove that a robot has physically stopped.
"""

from __future__ import annotations

import json
import math
import threading
import uuid

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


# ============================================================
# EXCEPTIONS
# ============================================================

class SafeStateError(Exception):
    pass


class SafeStateValidationError(SafeStateError):
    pass


class SafeStateNotFound(SafeStateError):
    pass


class SafeStateTransitionDenied(SafeStateError):
    pass


# ============================================================
# ENUMS
# ============================================================

class RobotSafetyState(str, Enum):
    UNKNOWN = "unknown"
    NORMAL = "normal"
    DEGRADED = "degraded"
    STOP_REQUESTED = "stop_requested"
    STOP_CONFIRMED = "stop_confirmed"
    EMERGENCY_LATCHED = "emergency_latched"
    RECOVERY_PENDING = "recovery_pending"


class SafeStopReason(str, Enum):
    COLLISION_RISK = "collision_risk"
    HUMAN_PROXIMITY = "human_proximity"
    GEOFENCE_BREACH = "geofence_breach"
    HAZARD_DETECTED = "hazard_detected"
    EMERGENCY_STOP = "emergency_stop"
    SENSOR_FAILURE = "sensor_failure"
    COMMUNICATION_FAILURE = "communication_failure"
    ROBOT_HEALTH = "robot_health"
    COMMAND_INTERLOCK = "command_interlock"
    CYBERSECURITY = "cybersecurity"
    MANUAL_REQUEST = "manual_request"
    UNKNOWN = "unknown"


class StopCategory(str, Enum):
    PROTECTIVE = "protective"
    EMERGENCY = "emergency"


class SafetyCommandDecision(str, Enum):
    ALLOW_SIMULATION = "allow_simulation"
    DENY = "deny"


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
            c.isalnum() or c in "_-"
            for c in value
        )
    ):
        raise SafeStateValidationError(
            f"Invalid {name}"
        )
    return value


def validate_time(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise SafeStateValidationError(
            "Timezone-aware timestamp required"
        )
    return value.astimezone(timezone.utc)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class SafeStateConfig:
    maximum_telemetry_age_seconds: float = 2.0
    maximum_audit_events: int = 20000
    simulation_only: bool = True

    def __post_init__(self) -> None:
        age = self.maximum_telemetry_age_seconds

        if (
            isinstance(age, bool)
            or not isinstance(age, (int, float))
            or not math.isfinite(age)
            or age <= 0
        ):
            raise SafeStateValidationError(
                "Invalid telemetry age limit"
            )

        if (
            isinstance(self.maximum_audit_events, bool)
            or not isinstance(self.maximum_audit_events, int)
            or self.maximum_audit_events <= 0
        ):
            raise SafeStateValidationError(
                "Invalid audit capacity"
            )

        if not self.simulation_only:
            raise SafeStateValidationError(
                "Physical hardware control unsupported"
            )


# ============================================================
# TELEMETRY EVIDENCE
# ============================================================

@dataclass(frozen=True)
class SafeStateTelemetry:
    tenant_id: str
    robot_id: str

    velocity_mps: float
    motors_enabled: bool
    brakes_engaged: bool

    sensors_healthy: bool
    communication_healthy: bool
    hazards_clear: bool
    interlocks_clear: bool

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")
        validate_time(self.timestamp)

        if (
            isinstance(self.velocity_mps, bool)
            or not isinstance(
                self.velocity_mps,
                (int, float),
            )
            or not math.isfinite(self.velocity_mps)
            or self.velocity_mps < 0
        ):
            raise SafeStateValidationError(
                "Velocity must be finite and nonnegative"
            )


# ============================================================
# STOP CONFIRMATION
# ============================================================

@dataclass(frozen=True)
class SafeStopConfirmation:
    tenant_id: str
    robot_id: str

    evidence_id: str
    verifier_id: str

    verified_stop: bool

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")
        validate_id(self.evidence_id, "evidence_id")
        validate_id(self.verifier_id, "verifier_id")
        validate_time(self.timestamp)


# ============================================================
# RECOVERY APPROVAL
# ============================================================

@dataclass(frozen=True)
class SafeRecoveryApproval:
    tenant_id: str
    robot_id: str

    approval_id: str
    operator_id: str

    approved: bool
    independent_safety_verified: bool

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")
        validate_id(self.approval_id, "approval_id")
        validate_id(self.operator_id, "operator_id")
        validate_time(self.timestamp)


# ============================================================
# ROBOT SAFETY RECORD
# ============================================================

@dataclass
class RobotSafeStateRecord:
    tenant_id: str
    robot_id: str

    state: RobotSafetyState = RobotSafetyState.UNKNOWN

    stop_reason: SafeStopReason | None = None
    stop_category: StopCategory | None = None

    latched: bool = False

    last_telemetry: SafeStateTelemetry | None = None

    stop_requested_at: datetime | None = None
    stop_confirmed_at: datetime | None = None

    recovery_requested_at: datetime | None = None

    updated_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "state": self.state.value,
            "stop_reason": (
                self.stop_reason.value
                if self.stop_reason
                else None
            ),
            "stop_category": (
                self.stop_category.value
                if self.stop_category
                else None
            ),
            "latched": self.latched,
            "stop_requested_at": (
                self.stop_requested_at.isoformat()
                if self.stop_requested_at
                else None
            ),
            "stop_confirmed_at": (
                self.stop_confirmed_at.isoformat()
                if self.stop_confirmed_at
                else None
            ),
            "recovery_requested_at": (
                self.recovery_requested_at.isoformat()
                if self.recovery_requested_at
                else None
            ),
            "updated_at": self.updated_at.isoformat(),
        }


# ============================================================
# TRANSITION EVENT
# ============================================================

@dataclass(frozen=True)
class SafeStateTransitionEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    robot_id: str

    previous_state: RobotSafetyState
    new_state: RobotSafetyState

    action: str
    actor_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "previous_state": self.previous_state.value,
            "new_state": self.new_state.value,
            "action": self.action,
            "actor_id": self.actor_id,
        }


# ============================================================
# SAFE STATE SERVICE
# ============================================================

class RoboticsSafeStateService:

    def __init__(
        self,
        config: SafeStateConfig | None = None,
    ) -> None:
        self.config = config or SafeStateConfig()

        self.records: dict[
            tuple[str, str],
            RobotSafeStateRecord,
        ] = {}

        self.audit_events: list[
            SafeStateTransitionEvent
        ] = []

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------

    def _key(
        self,
        tenant_id: str,
        robot_id: str,
    ) -> tuple[str, str]:
        validate_id(tenant_id, "tenant_id")
        validate_id(robot_id, "robot_id")
        return tenant_id, robot_id

    def _get(
        self,
        tenant_id: str,
        robot_id: str,
    ) -> RobotSafeStateRecord:
        key = self._key(tenant_id, robot_id)

        if key not in self.records:
            raise SafeStateNotFound(
                "Robot safety record not found"
            )

        return self.records[key]

    def _fresh(
        self,
        timestamp: datetime,
    ) -> bool:
        age = (
            utc_now() - validate_time(timestamp)
        ).total_seconds()

        return (
            -1.0 <= age
            <= self.config.maximum_telemetry_age_seconds
        )

    def _transition(
        self,
        record: RobotSafeStateRecord,
        new_state: RobotSafetyState,
        action: str,
        actor_id: str,
    ) -> None:
        validate_id(actor_id, "actor_id")

        previous = record.state
        record.state = new_state
        record.updated_at = utc_now()

        self.audit_events.append(
            SafeStateTransitionEvent(
                event_id=str(uuid.uuid4()),
                timestamp=record.updated_at,
                tenant_id=record.tenant_id,
                robot_id=record.robot_id,
                previous_state=previous,
                new_state=new_state,
                action=action,
                actor_id=actor_id,
            )
        )

        if (
            len(self.audit_events)
            > self.config.maximum_audit_events
        ):
            self.audit_events.pop(0)

    # --------------------------------------------------------
    # REGISTER ROBOT
    # --------------------------------------------------------

    def register_robot(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> RobotSafeStateRecord:
        with self._lock:
            key = self._key(
                tenant_id,
                robot_id,
            )

            if key in self.records:
                raise SafeStateValidationError(
                    "Robot already registered"
                )

            record = RobotSafeStateRecord(
                tenant_id=tenant_id,
                robot_id=robot_id,
            )

            self.records[key] = record

            return replace(record)

    # --------------------------------------------------------
    # UPDATE TELEMETRY
    # --------------------------------------------------------

    def update_telemetry(
        self,
        telemetry: SafeStateTelemetry,
    ) -> RobotSafeStateRecord:
        with self._lock:
            record = self._get(
                telemetry.tenant_id,
                telemetry.robot_id,
            )

            if (
                record.last_telemetry is not None
                and telemetry.timestamp
                < record.last_telemetry.timestamp
            ):
                raise SafeStateValidationError(
                    "Out-of-order telemetry"
                )

            record.last_telemetry = telemetry

            # Never automatically clear a stop,
            # emergency latch or recovery state.
            if record.state in {
                RobotSafetyState.UNKNOWN,
                RobotSafetyState.NORMAL,
                RobotSafetyState.DEGRADED,
            }:
                if (
                    self._fresh(telemetry.timestamp)
                    and telemetry.sensors_healthy
                    and telemetry.communication_healthy
                    and telemetry.hazards_clear
                    and telemetry.interlocks_clear
                ):
                    target = RobotSafetyState.NORMAL
                else:
                    target = RobotSafetyState.DEGRADED

                if record.state != target:
                    self._transition(
                        record,
                        target,
                        "telemetry_status_update",
                        "system",
                    )

            return replace(record)

    # --------------------------------------------------------
    # REQUEST STOP
    # --------------------------------------------------------

    def request_stop(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        reason: SafeStopReason,
        category: StopCategory,
        actor_id: str,
    ) -> RobotSafeStateRecord:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
            )

            reason = SafeStopReason(reason)
            category = StopCategory(category)

            if category == StopCategory.EMERGENCY:
                record.latched = True

            if record.latched:
                category = StopCategory.EMERGENCY

            record.stop_reason = reason
            record.stop_category = category
            record.stop_requested_at = utc_now()
            record.stop_confirmed_at = None
            record.recovery_requested_at = None

            target = (
                RobotSafetyState.EMERGENCY_LATCHED
                if record.latched
                else RobotSafetyState.STOP_REQUESTED
            )

            self._transition(
                record,
                target,
                "safe_stop_requested",
                actor_id,
            )

            return replace(record)

    # --------------------------------------------------------
    # CONFIRM STOP
    # --------------------------------------------------------

    def confirm_stop(
        self,
        confirmation: SafeStopConfirmation,
    ) -> RobotSafeStateRecord:
        with self._lock:
            record = self._get(
                confirmation.tenant_id,
                confirmation.robot_id,
            )

            if record.state not in {
                RobotSafetyState.STOP_REQUESTED,
                RobotSafetyState.EMERGENCY_LATCHED,
            }:
                raise SafeStateTransitionDenied(
                    "No stop is pending"
                )

            if (
                not confirmation.verified_stop
                or not self._fresh(
                    confirmation.timestamp
                )
            ):
                raise SafeStateTransitionDenied(
                    "Valid stop confirmation required"
                )

            telemetry = record.last_telemetry

            if (
                telemetry is None
                or not self._fresh(telemetry.timestamp)
                or telemetry.velocity_mps > 0.01
                or telemetry.motors_enabled
                or not telemetry.brakes_engaged
            ):
                raise SafeStateTransitionDenied(
                    "Telemetry does not confirm "
                    "simulated safe stop"
                )

            record.stop_confirmed_at = utc_now()

            target = (
                RobotSafetyState.EMERGENCY_LATCHED
                if record.latched
                else RobotSafetyState.STOP_CONFIRMED
            )

            self._transition(
                record,
                target,
                "safe_stop_confirmed",
                confirmation.verifier_id,
            )

            return replace(record)

    # --------------------------------------------------------
    # REQUEST RECOVERY
    # --------------------------------------------------------

    def request_recovery(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        operator_id: str,
    ) -> RobotSafeStateRecord:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
            )

            if (
                record.stop_confirmed_at is None
                or record.state not in {
                    RobotSafetyState.STOP_CONFIRMED,
                    RobotSafetyState.EMERGENCY_LATCHED,
                }
            ):
                raise SafeStateTransitionDenied(
                    "Confirmed simulated stop required"
                )

            record.recovery_requested_at = utc_now()

            self._transition(
                record,
                RobotSafetyState.RECOVERY_PENDING,
                "recovery_requested",
                operator_id,
            )

            return replace(record)

    # --------------------------------------------------------
    # COMPLETE RECOVERY
    # --------------------------------------------------------

    def complete_recovery(
        self,
        approval: SafeRecoveryApproval,
    ) -> RobotSafeStateRecord:
        with self._lock:
            record = self._get(
                approval.tenant_id,
                approval.robot_id,
            )

            if (
                record.state
                != RobotSafetyState.RECOVERY_PENDING
            ):
                raise SafeStateTransitionDenied(
                    "Recovery is not pending"
                )

            if (
                not approval.approved
                or not approval.independent_safety_verified
                or not self._fresh(approval.timestamp)
            ):
                raise SafeStateTransitionDenied(
                    "Fresh verified approval required"
                )

            telemetry = record.last_telemetry

            if (
                telemetry is None
                or not self._fresh(telemetry.timestamp)
                or telemetry.velocity_mps > 0.01
                or telemetry.motors_enabled
                or not telemetry.brakes_engaged
                or not telemetry.sensors_healthy
                or not telemetry.communication_healthy
                or not telemetry.hazards_clear
                or not telemetry.interlocks_clear
            ):
                raise SafeStateTransitionDenied(
                    "Recovery safety conditions not met"
                )

            record.latched = False
            record.stop_reason = None
            record.stop_category = None
            record.stop_requested_at = None
            record.stop_confirmed_at = None
            record.recovery_requested_at = None

            self._transition(
                record,
                RobotSafetyState.NORMAL,
                "simulation_recovery_completed",
                approval.operator_id,
            )

            return replace(record)

    # --------------------------------------------------------
    # COMMAND GATE
    # --------------------------------------------------------

    def evaluate_command(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        simulation_only: bool = True,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
            )

            telemetry = record.last_telemetry

            allowed = (
                simulation_only
                and record.state
                == RobotSafetyState.NORMAL
                and not record.latched
                and telemetry is not None
                and self._fresh(telemetry.timestamp)
                and telemetry.sensors_healthy
                and telemetry.communication_healthy
                and telemetry.hazards_clear
                and telemetry.interlocks_clear
            )

            return {
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "decision": (
                    SafetyCommandDecision.ALLOW_SIMULATION.value
                    if allowed
                    else SafetyCommandDecision.DENY.value
                ),
                "state": record.state.value,
                "latched": record.latched,
                "evaluated_at": utc_now().isoformat(),
            }

    # --------------------------------------------------------
    # GET STATUS
    # --------------------------------------------------------

    def get_status(
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
                for record in self.records.values()
                if (
                    tenant_id is None
                    or record.tenant_id == tenant_id
                )
            ]

            return {
                "total_robots": len(records),
                "normal": sum(
                    r.state == RobotSafetyState.NORMAL
                    for r in records
                ),
                "stop_requested": sum(
                    r.state
                    == RobotSafetyState.STOP_REQUESTED
                    for r in records
                ),
                "emergency_latched": sum(
                    r.latched
                    for r in records
                ),
                "recovery_pending": sum(
                    r.state
                    == RobotSafetyState.RECOVERY_PENDING
                    for r in records
                ),
            }

    # --------------------------------------------------------
    # EXPORT
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
                "robots": [
                    record.to_dict()
                    for record in self.records.values()
                    if record.tenant_id == tenant_id
                ],
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

def main() -> None:
    service = RoboticsSafeStateService()

    service.register_robot(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
    )

    service.update_telemetry(
        SafeStateTelemetry(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            velocity_mps=1.2,
            motors_enabled=True,
            brakes_engaged=False,
            sensors_healthy=True,
            communication_healthy=True,
            hazards_clear=True,
            interlocks_clear=True,
        )
    )

    service.request_stop(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        reason=SafeStopReason.HUMAN_PROXIMITY,
        category=StopCategory.EMERGENCY,
        actor_id="safety_monitor",
    )

    service.update_telemetry(
        SafeStateTelemetry(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            velocity_mps=0.0,
            motors_enabled=False,
            brakes_engaged=True,
            sensors_healthy=True,
            communication_healthy=True,
            hazards_clear=True,
            interlocks_clear=True,
        )
    )

    service.confirm_stop(
        SafeStopConfirmation(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            evidence_id="stop_evidence_001",
            verifier_id="simulation_verifier",
            verified_stop=True,
        )
    )

    print(
        json.dumps(
            service.get_status(
                tenant_id="tenant_alpha",
                robot_id="robot_001",
            ),
            indent=2,
        )
    )

    service.export_json(
        "data/security/safe_state_report.json",
        tenant_id="tenant_alpha",
    )


if __name__ == "__main__":
    main()

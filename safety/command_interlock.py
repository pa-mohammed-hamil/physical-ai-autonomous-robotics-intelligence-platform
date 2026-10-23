"""Emergency Advisory module.

TODO: implement and test; no real robot control is enabled.
"""


"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
    backend/app/security/command_interlock.py

Python 3.11+
Standard library only.

Features:
    - Robot command safety interlocks
    - Emergency-stop status monitoring
    - Collision-risk interlock
    - Geofence interlock
    - Battery and thermal safety checks
    - Sensor and network health checks
    - Maintenance and repair interlocks
    - Command authorization status
    - Telemetry freshness checks
    - Latched safety faults
    - Manual reset workflow
    - Tenant-scoped robot states
    - Audit events and JSON export

IMPORTANT:
    This is a software supervisory gate for
    simulation and testing only.

    It is NOT a safety-rated PLC, certified
    emergency-stop controller, or physical
    robot safety system.

    Real hardware must enforce independent
    fail-safe safety interlocks.
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

class CommandInterlockError(Exception):
    pass


class InterlockValidationError(CommandInterlockError):
    pass


class InterlockAccessDenied(CommandInterlockError):
    pass


class InterlockStateNotFound(CommandInterlockError):
    pass


class InterlockResetDenied(CommandInterlockError):
    pass


# ============================================================
# ENUMS
# ============================================================

class InterlockStatus(str, Enum):
    CLEAR = "clear"
    BLOCKED = "blocked"
    LATCHED = "latched"


class InterlockType(str, Enum):
    EMERGENCY_STOP = "emergency_stop"
    COLLISION_RISK = "collision_risk"
    GEOFENCE = "geofence"
    LOW_BATTERY = "low_battery"
    OVER_TEMPERATURE = "over_temperature"
    SENSOR_FAILURE = "sensor_failure"
    NETWORK_FAILURE = "network_failure"
    MAINTENANCE_REQUIRED = "maintenance_required"
    REPAIR_REQUIRED = "repair_required"
    AUTHORIZATION_FAILURE = "authorization_failure"
    STALE_TELEMETRY = "stale_telemetry"
    HARDWARE_CONTROL_DISABLED = (
        "hardware_control_disabled"
    )


class InterlockSeverity(str, Enum):
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class CommandSafetyDecision(str, Enum):
    ALLOW_SIMULATION = "allow_simulation"
    DENY = "deny"


class RobotCommandKind(str, Enum):
    NAVIGATE = "navigate"
    MOVE = "move"
    MANIPULATE = "manipulate"
    PICK = "pick"
    PLACE = "place"
    START_TASK = "start_task"
    START_SIMULATION = "start_simulation"
    STOP_SIMULATION = "stop_simulation"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


def validate_identifier(
    value: str,
    field_name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not all(
            char.isalnum() or char in "_-"
            for char in value
        )
    ):
        raise InterlockValidationError(
            f"Invalid {field_name}"
        )

    return value


def finite_number(
    value: float,
    field_name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise InterlockValidationError(
            f"{field_name} must be finite"
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
        raise InterlockValidationError(
            "Timestamp must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class CommandInterlockConfig:
    minimum_battery_percent: float = 20.0
    maximum_temperature_c: float = 70.0

    maximum_telemetry_age_seconds: float = 2.0

    maximum_collision_risk_score: float = 40.0

    require_sensor_health: bool = True
    require_network_health: bool = True
    require_geofence_clear: bool = True
    require_authorization: bool = True

    latch_emergency_stop: bool = True
    latch_collision_risk: bool = True
    latch_geofence_breach: bool = True

    allow_hardware_control: bool = False

    maximum_audit_events: int = 20000

    def __post_init__(self) -> None:
        battery = finite_number(
            self.minimum_battery_percent,
            "minimum_battery_percent",
        )

        temperature = finite_number(
            self.maximum_temperature_c,
            "maximum_temperature_c",
        )

        age = finite_number(
            self.maximum_telemetry_age_seconds,
            "maximum_telemetry_age_seconds",
        )

        risk = finite_number(
            self.maximum_collision_risk_score,
            "maximum_collision_risk_score",
        )

        if not (0 <= battery <= 100):
            raise InterlockValidationError(
                "Battery threshold out of range"
            )

        if not (-50 <= temperature <= 250):
            raise InterlockValidationError(
                "Temperature threshold out of range"
            )

        if age <= 0:
            raise InterlockValidationError(
                "Telemetry age limit must be positive"
            )

        if not (0 <= risk <= 100):
            raise InterlockValidationError(
                "Collision risk threshold out of range"
            )

        if self.allow_hardware_control:
            raise InterlockValidationError(
                "Hardware control cannot be enabled "
                "by this simulation module"
            )

        if (
            isinstance(self.maximum_audit_events, bool)
            or self.maximum_audit_events <= 0
        ):
            raise InterlockValidationError(
                "Maximum audit events must be positive"
            )


# ============================================================
# ROBOT SAFETY STATE
# ============================================================

@dataclass(frozen=True)
class RobotInterlockState:
    tenant_id: str
    robot_id: str

    battery_percent: float
    temperature_c: float

    emergency_stop_active: bool = False

    collision_risk_score: float = 0.0
    geofence_clear: bool = True

    sensors_healthy: bool = True
    network_healthy: bool = True

    maintenance_required: bool = False
    repair_required: bool = False

    authorization_valid: bool = False

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_identifier(
            self.tenant_id,
            "tenant_id",
        )

        validate_identifier(
            self.robot_id,
            "robot_id",
        )

        battery = finite_number(
            self.battery_percent,
            "battery_percent",
        )

        temperature = finite_number(
            self.temperature_c,
            "temperature_c",
        )

        risk = finite_number(
            self.collision_risk_score,
            "collision_risk_score",
        )

        if not (0 <= battery <= 100):
            raise InterlockValidationError(
                "Battery must be between 0 and 100"
            )

        if not (-50 <= temperature <= 250):
            raise InterlockValidationError(
                "Temperature outside accepted range"
            )

        if not (0 <= risk <= 100):
            raise InterlockValidationError(
                "Collision risk must be between 0 and 100"
            )

        validate_timestamp(
            self.timestamp
        )


# ============================================================
# INTERLOCK FINDING
# ============================================================

@dataclass(frozen=True)
class InterlockFinding:
    interlock_type: InterlockType
    status: InterlockStatus
    severity: InterlockSeverity

    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "interlock_type": (
                self.interlock_type.value
            ),
            "status": self.status.value,
            "severity": self.severity.value,
            "message": self.message,
        }


# ============================================================
# COMMAND REQUEST
# ============================================================

@dataclass(frozen=True)
class InterlockCommandRequest:
    tenant_id: str
    robot_id: str

    command: RobotCommandKind

    simulation_only: bool = True

    request_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    def __post_init__(self) -> None:
        validate_identifier(
            self.tenant_id,
            "tenant_id",
        )

        validate_identifier(
            self.robot_id,
            "robot_id",
        )


# ============================================================
# COMMAND DECISION
# ============================================================

@dataclass(frozen=True)
class CommandInterlockDecision:
    request_id: str
    tenant_id: str
    robot_id: str

    command: RobotCommandKind

    decision: CommandSafetyDecision

    findings: tuple[
        InterlockFinding,
        ...
    ]

    evaluated_at: str = field(
        default_factory=utc_iso
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "command": self.command.value,
            "decision": self.decision.value,
            "findings": [
                finding.to_dict()
                for finding in self.findings
            ],
            "evaluated_at": self.evaluated_at,
        }


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class InterlockAuditEvent:
    event_id: str
    timestamp: str

    tenant_id: str
    robot_id: str

    action: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "action": self.action,
            "details": self.details,
        }


# ============================================================
# INTERLOCK POLICY ENGINE
# ============================================================

class CommandInterlockPolicyEngine:

    def __init__(
        self,
        config: CommandInterlockConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        state: RobotInterlockState,
    ) -> list[InterlockFinding]:
        findings: list[
            InterlockFinding
        ] = []

        def add(
            kind: InterlockType,
            severity: InterlockSeverity,
            message: str,
        ) -> None:
            findings.append(
                InterlockFinding(
                    interlock_type=kind,
                    status=InterlockStatus.BLOCKED,
                    severity=severity,
                    message=message,
                )
            )

        age = (
            utc_now()
            - validate_timestamp(
                state.timestamp
            )
        ).total_seconds()

        if (
            age < -1
            or age
            > self.config.maximum_telemetry_age_seconds
        ):
            add(
                InterlockType.STALE_TELEMETRY,
                InterlockSeverity.CRITICAL,
                "Robot safety telemetry is stale",
            )

        if state.emergency_stop_active:
            add(
                InterlockType.EMERGENCY_STOP,
                InterlockSeverity.CRITICAL,
                "Emergency stop is active",
            )

        if (
            state.collision_risk_score
            > self.config.maximum_collision_risk_score
        ):
            add(
                InterlockType.COLLISION_RISK,
                InterlockSeverity.CRITICAL,
                "Collision risk exceeds threshold",
            )

        if (
            self.config.require_geofence_clear
            and not state.geofence_clear
        ):
            add(
                InterlockType.GEOFENCE,
                InterlockSeverity.CRITICAL,
                "Geofence restriction detected",
            )

        if (
            state.battery_percent
            < self.config.minimum_battery_percent
        ):
            add(
                InterlockType.LOW_BATTERY,
                InterlockSeverity.HIGH,
                "Battery below minimum threshold",
            )

        if (
            state.temperature_c
            > self.config.maximum_temperature_c
        ):
            add(
                InterlockType.OVER_TEMPERATURE,
                InterlockSeverity.CRITICAL,
                "Robot temperature exceeds threshold",
            )

        if (
            self.config.require_sensor_health
            and not state.sensors_healthy
        ):
            add(
                InterlockType.SENSOR_FAILURE,
                InterlockSeverity.CRITICAL,
                "Required sensors are unhealthy",
            )

        if (
            self.config.require_network_health
            and not state.network_healthy
        ):
            add(
                InterlockType.NETWORK_FAILURE,
                InterlockSeverity.HIGH,
                "Robot communication is unhealthy",
            )

        if state.maintenance_required:
            add(
                InterlockType.MAINTENANCE_REQUIRED,
                InterlockSeverity.HIGH,
                "Robot maintenance is required",
            )

        if state.repair_required:
            add(
                InterlockType.REPAIR_REQUIRED,
                InterlockSeverity.HIGH,
                "Robot repair is required",
            )

        if (
            self.config.require_authorization
            and not state.authorization_valid
        ):
            add(
                InterlockType.AUTHORIZATION_FAILURE,
                InterlockSeverity.CRITICAL,
                "Command authorization is invalid",
            )

        return findings


# ============================================================
# CENTRAL INTERLOCK SERVICE
# ============================================================

class RoboticsCommandInterlockService:

    def __init__(
        self,
        config: CommandInterlockConfig | None = None,
    ) -> None:
        self.config = (
            config or CommandInterlockConfig()
        )

        self.policy = (
            CommandInterlockPolicyEngine(
                self.config
            )
        )

        self.states: dict[
            tuple[str, str],
            RobotInterlockState
        ] = {}

        self.latched_faults: dict[
            tuple[str, str],
            set[InterlockType]
        ] = {}

        self.audit_events: list[
            InterlockAuditEvent
        ] = []

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        action: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.audit_events.append(
            InterlockAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_iso(),
                tenant_id=tenant_id,
                robot_id=robot_id,
                action=action,
                details=dict(details or {}),
            )
        )

        if (
            len(self.audit_events)
            > self.config.maximum_audit_events
        ):
            self.audit_events.pop(0)

    # --------------------------------------------------------
    # UPDATE STATE
    # --------------------------------------------------------

    def update_state(
        self,
        state: RobotInterlockState,
    ) -> None:
        key = (
            state.tenant_id,
            state.robot_id,
        )

        with self._lock:
            existing = self.states.get(
                key
            )

            if (
                existing is not None
                and state.timestamp
                < existing.timestamp
            ):
                raise InterlockValidationError(
                    "Out-of-order safety telemetry"
                )

            self.states[key] = state

            findings = self.policy.evaluate(
                state
            )

            latched = self.latched_faults.setdefault(
                key,
                set(),
            )

            for finding in findings:
                kind = finding.interlock_type

                if (
                    kind == InterlockType.EMERGENCY_STOP
                    and self.config.latch_emergency_stop
                ):
                    latched.add(kind)

                elif (
                    kind == InterlockType.COLLISION_RISK
                    and self.config.latch_collision_risk
                ):
                    latched.add(kind)

                elif (
                    kind == InterlockType.GEOFENCE
                    and self.config.latch_geofence_breach
                ):
                    latched.add(kind)

            self._audit(
                tenant_id=state.tenant_id,
                robot_id=state.robot_id,
                action="state_updated",
                details={
                    "active_findings": len(findings),
                    "latched_faults": [
                        item.value
                        for item in sorted(
                            latched,
                            key=lambda value: value.value,
                        )
                    ],
                },
            )

    # --------------------------------------------------------
    # EVALUATE COMMAND
    # --------------------------------------------------------

    def evaluate_command(
        self,
        request: InterlockCommandRequest,
    ) -> CommandInterlockDecision:
        key = (
            request.tenant_id,
            request.robot_id,
        )

        with self._lock:
            if not request.simulation_only:
                return self._decision(
                    request,
                    CommandSafetyDecision.DENY,
                    [
                        InterlockFinding(
                            interlock_type=(
                                InterlockType.HARDWARE_CONTROL_DISABLED
                            ),
                            status=InterlockStatus.BLOCKED,
                            severity=InterlockSeverity.CRITICAL,
                            message=(
                                "Hardware command execution "
                                "is not supported"
                            ),
                        )
                    ],
                )

            state = self.states.get(
                key
            )

            if state is None:
                return self._decision(
                    request,
                    CommandSafetyDecision.DENY,
                    [
                        InterlockFinding(
                            interlock_type=(
                                InterlockType.STALE_TELEMETRY
                            ),
                            status=InterlockStatus.BLOCKED,
                            severity=InterlockSeverity.CRITICAL,
                            message=(
                                "No robot safety state available"
                            ),
                        )
                    ],
                )

            findings = self.policy.evaluate(
                state
            )

            latched = self.latched_faults.setdefault(
                key,
                set(),
            )

            # Latch newly detected faults even if
            # evaluate_command is called without a
            # preceding update_state().
            for finding in findings:
                kind = finding.interlock_type

                if (
                    kind == InterlockType.EMERGENCY_STOP
                    and self.config.latch_emergency_stop
                ):
                    latched.add(kind)

                if (
                    kind == InterlockType.COLLISION_RISK
                    and self.config.latch_collision_risk
                ):
                    latched.add(kind)

                if (
                    kind == InterlockType.GEOFENCE
                    and self.config.latch_geofence_breach
                ):
                    latched.add(kind)

            active_types = {
                finding.interlock_type
                for finding in findings
            }

            for kind in sorted(
                latched,
                key=lambda value: value.value,
            ):
                if kind not in active_types:
                    findings.append(
                        InterlockFinding(
                            interlock_type=kind,
                            status=InterlockStatus.LATCHED,
                            severity=InterlockSeverity.CRITICAL,
                            message=(
                                "Safety fault remains latched "
                                "until manual reset"
                            ),
                        )
                    )

            decision = (
                CommandSafetyDecision.DENY
                if findings
                else CommandSafetyDecision.ALLOW_SIMULATION
            )

            return self._decision(
                request,
                decision,
                findings,
            )

    def _decision(
        self,
        request: InterlockCommandRequest,
        decision: CommandSafetyDecision,
        findings: list[InterlockFinding],
    ) -> CommandInterlockDecision:
        result = CommandInterlockDecision(
            request_id=request.request_id,
            tenant_id=request.tenant_id,
            robot_id=request.robot_id,
            command=request.command,
            decision=decision,
            findings=tuple(findings),
        )

        self._audit(
            tenant_id=request.tenant_id,
            robot_id=request.robot_id,
            action="command_evaluated",
            details={
                "command": request.command.value,
                "decision": decision.value,
                "findings": [
                    finding.interlock_type.value
                    for finding in findings
                ],
            },
        )

        return result

    # --------------------------------------------------------
    # MANUAL RESET
    # --------------------------------------------------------

    def reset_latched_faults(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        operator_id: str,
        authorized: bool,
    ) -> None:
        validate_identifier(
            operator_id,
            "operator_id",
        )

        if not authorized:
            raise InterlockResetDenied(
                "Verified operator authorization required"
            )

        key = (
            tenant_id,
            robot_id,
        )

        with self._lock:
            state = self.states.get(
                key
            )

            if state is None:
                raise InterlockStateNotFound(
                    "Robot safety state not found"
                )

            findings = self.policy.evaluate(
                state
            )

            if findings:
                raise InterlockResetDenied(
                    "Active safety faults must be cleared "
                    "before reset"
                )

            self.latched_faults.pop(
                key,
                None,
            )

            self._audit(
                tenant_id=tenant_id,
                robot_id=robot_id,
                action="latched_faults_reset",
                details={
                    "operator_id": operator_id,
                },
            )

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
            key = (
                tenant_id,
                robot_id,
            )

            state = self.states.get(
                key
            )

            if state is None:
                raise InterlockStateNotFound(
                    "Robot state not found"
                )

            findings = self.policy.evaluate(
                state
            )

            latched = self.latched_faults.get(
                key,
                set(),
            )

            return {
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "status": (
                    InterlockStatus.LATCHED.value
                    if latched
                    else (
                        InterlockStatus.BLOCKED.value
                        if findings
                        else InterlockStatus.CLEAR.value
                    )
                ),
                "active_findings": [
                    finding.to_dict()
                    for finding in findings
                ],
                "latched_faults": sorted(
                    item.value
                    for item in latched
                ),
                "checked_at": utc_iso(),
            }

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
    ) -> dict[str, Any]:
        with self._lock:
            return {
                "registered_robot_states": len(
                    self.states
                ),
                "robots_with_latched_faults": sum(
                    bool(faults)
                    for faults in self.latched_faults.values()
                ),
                "audit_events": len(
                    self.audit_events
                ),
                "hardware_control_enabled": False,
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
                "generated_at": utc_iso(),
                "statistics": self.statistics(),
                "robots": [
                    self.get_status(
                        tenant_id=tenant_id,
                        robot_id=robot_id,
                    )
                    for tenant_id, robot_id
                    in self.states
                ],
                "audit_events": [
                    event.to_dict()
                    for event in self.audit_events
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
    service = RoboticsCommandInterlockService()

    service.update_state(
        RobotInterlockState(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            battery_percent=85.0,
            temperature_c=35.0,
            emergency_stop_active=False,
            collision_risk_score=10.0,
            geofence_clear=True,
            sensors_healthy=True,
            network_healthy=True,
            maintenance_required=False,
            repair_required=False,
            authorization_valid=True,
        )
    )

    request = InterlockCommandRequest(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        command=RobotCommandKind.NAVIGATE,
        simulation_only=True,
    )

    result = service.evaluate_command(
        request
    )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
        )
    )

    service.update_state(
        RobotInterlockState(
            tenant_id="tenant_alpha",
            robot_id="robot_001",
            battery_percent=85.0,
            temperature_c=35.0,
            emergency_stop_active=True,
            authorization_valid=True,
        )
    )

    blocked = service.evaluate_command(
        request
    )

    print(
        json.dumps(
            blocked.to_dict(),
            indent=2,
        )
    )

    service.export_json(
        "data/security/command_interlock_report.json"
    )


if __name__ == "__main__":
    main()

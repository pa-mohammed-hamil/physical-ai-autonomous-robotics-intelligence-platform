"""Watchdog module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File: backend/app/security/watchdog.py

Python 3.11+
Standard library only.

Features:
    - Robot and service heartbeat monitoring
    - Component timeout detection
    - Sensor and communication monitoring
    - Agent, navigation and perception watchdogs
    - Configurable heartbeat intervals
    - Monotonic-clock timeout checks
    - Missing heartbeat detection
    - Out-of-order heartbeat rejection
    - Critical failure latching
    - Explicit fault recovery workflow
    - Fail-closed simulation command gate
    - Tenant isolation
    - Thread-safe operations
    - Audit logging
    - JSON report export

SAFETY:
    Simulation and supervisory monitoring ONLY.

    This is not a hardware safety watchdog.
    It does not actuate emergency stops,
    brakes, motors or physical relays.

    A watchdog timeout is an advisory input
    to independent safety systems.
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
from typing import Any


# ============================================================
# EXCEPTIONS
# ============================================================

class WatchdogError(Exception):
    pass


class WatchdogValidationError(WatchdogError):
    pass


class WatchdogNotFound(WatchdogError):
    pass


class WatchdogRecoveryDenied(WatchdogError):
    pass


# ============================================================
# ENUMS
# ============================================================

class WatchdogComponentType(str, Enum):
    ROBOT = "robot"
    PERCEPTION = "perception"
    NAVIGATION = "navigation"
    PLANNING = "planning"
    CONTROL = "control"
    SENSOR = "sensor"
    COMMUNICATION = "communication"
    TELEMETRY = "telemetry"
    SAFETY_MONITOR = "safety_monitor"
    COMMAND_INTERLOCK = "command_interlock"
    AI_AGENT = "ai_agent"
    SIMULATOR = "simulator"
    OTHER = "other"


class WatchdogStatus(str, Enum):
    WAITING = "waiting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    TIMED_OUT = "timed_out"
    FAULT_LATCHED = "fault_latched"


class WatchdogSeverity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class WatchdogDecision(str, Enum):
    ALLOW_SIMULATION = "allow_simulation"
    DENY = "deny"


class WatchdogEventType(str, Enum):
    COMPONENT_REGISTERED = "component_registered"
    HEARTBEAT_RECEIVED = "heartbeat_received"
    HEARTBEAT_DEGRADED = "heartbeat_degraded"
    HEARTBEAT_TIMEOUT = "heartbeat_timeout"
    FAULT_LATCHED = "fault_latched"
    RECOVERY_COMPLETED = "recovery_completed"


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
        raise WatchdogValidationError(
            f"Invalid {name}"
        )

    return value


def positive_number(
    value: float,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise WatchdogValidationError(
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
        raise WatchdogValidationError(
            "Timezone-aware timestamp required"
        )

    return value.astimezone(timezone.utc)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class WatchdogConfig:
    default_timeout_seconds: float = 3.0
    default_warning_seconds: float = 1.5

    maximum_components: int = 10000
    maximum_audit_events: int = 20000

    latch_critical_timeouts: bool = True
    deny_on_degraded: bool = True

    simulation_only: bool = True

    def __post_init__(self) -> None:
        positive_number(
            self.default_timeout_seconds,
            "default_timeout_seconds",
        )

        positive_number(
            self.default_warning_seconds,
            "default_warning_seconds",
        )

        if (
            self.default_warning_seconds
            >= self.default_timeout_seconds
        ):
            raise WatchdogValidationError(
                "Warning threshold must be below timeout"
            )

        for name in (
            "maximum_components",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise WatchdogValidationError(
                    f"{name} must be a positive integer"
                )

        if not self.simulation_only:
            raise WatchdogValidationError(
                "Physical hardware control unsupported"
            )


# ============================================================
# COMPONENT REGISTRATION
# ============================================================

@dataclass(frozen=True)
class WatchdogComponent:
    tenant_id: str
    robot_id: str
    component_id: str

    component_type: WatchdogComponentType

    severity: WatchdogSeverity = (
        WatchdogSeverity.CRITICAL
    )

    timeout_seconds: float = 3.0
    warning_seconds: float = 1.5

    required_for_operation: bool = True

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")
        validate_id(self.component_id, "component_id")

        positive_number(
            self.timeout_seconds,
            "timeout_seconds",
        )

        positive_number(
            self.warning_seconds,
            "warning_seconds",
        )

        if (
            self.warning_seconds
            >= self.timeout_seconds
        ):
            raise WatchdogValidationError(
                "Warning threshold must be below timeout"
            )


# ============================================================
# HEARTBEAT
# ============================================================

@dataclass(frozen=True)
class WatchdogHeartbeat:
    tenant_id: str
    robot_id: str
    component_id: str

    sequence_number: int

    healthy: bool = True

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")
        validate_id(self.component_id, "component_id")

        if (
            isinstance(self.sequence_number, bool)
            or not isinstance(
                self.sequence_number,
                int,
            )
            or self.sequence_number < 0
        ):
            raise WatchdogValidationError(
                "Invalid heartbeat sequence"
            )

        if not isinstance(self.healthy, bool):
            raise WatchdogValidationError(
                "healthy must be boolean"
            )

        validate_timestamp(self.timestamp)


# ============================================================
# WATCHDOG RECORD
# ============================================================

@dataclass
class WatchdogRecord:
    component: WatchdogComponent

    status: WatchdogStatus = (
        WatchdogStatus.WAITING
    )

    last_sequence_number: int = -1

    last_heartbeat_at: datetime | None = None
    last_received_monotonic: float | None = None

    registered_monotonic: float = field(
        default_factory=time.monotonic
    )

    fault_latched: bool = False

    timeout_count: int = 0
    heartbeat_count: int = 0

    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.component.tenant_id,
            "robot_id": self.component.robot_id,
            "component_id": self.component.component_id,
            "component_type": (
                self.component.component_type.value
            ),
            "severity": self.component.severity.value,
            "required_for_operation": (
                self.component.required_for_operation
            ),
            "status": self.status.value,
            "fault_latched": self.fault_latched,
            "last_sequence_number": (
                self.last_sequence_number
            ),
            "last_heartbeat_at": (
                self.last_heartbeat_at.isoformat()
                if self.last_heartbeat_at
                else None
            ),
            "timeout_count": self.timeout_count,
            "heartbeat_count": self.heartbeat_count,
            "last_error": self.last_error,
        }


# ============================================================
# WATCHDOG EVENT
# ============================================================

@dataclass(frozen=True)
class WatchdogEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    robot_id: str
    component_id: str

    event_type: WatchdogEventType
    severity: WatchdogSeverity

    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "component_id": self.component_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "description": self.description,
        }


# ============================================================
# WATCHDOG SERVICE
# ============================================================

class RoboticsWatchdogService:

    def __init__(
        self,
        config: WatchdogConfig | None = None,
    ) -> None:
        self.config = config or WatchdogConfig()

        self.records: dict[
            tuple[str, str, str],
            WatchdogRecord
        ] = {}

        self.events: list[
            WatchdogEvent
        ] = []

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------------

    def _key(
        self,
        tenant_id: str,
        robot_id: str,
        component_id: str,
    ) -> tuple[str, str, str]:
        return (
            validate_id(tenant_id, "tenant_id"),
            validate_id(robot_id, "robot_id"),
            validate_id(component_id, "component_id"),
        )

    def _get(
        self,
        tenant_id: str,
        robot_id: str,
        component_id: str,
    ) -> WatchdogRecord:
        key = self._key(
            tenant_id,
            robot_id,
            component_id,
        )

        if key not in self.records:
            raise WatchdogNotFound(
                "Watchdog component not registered"
            )

        return self.records[key]

    def _event(
        self,
        record: WatchdogRecord,
        event_type: WatchdogEventType,
        description: str,
    ) -> None:
        component = record.component

        self.events.append(
            WatchdogEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                tenant_id=component.tenant_id,
                robot_id=component.robot_id,
                component_id=component.component_id,
                event_type=event_type,
                severity=component.severity,
                description=description,
            )
        )

        if (
            len(self.events)
            > self.config.maximum_audit_events
        ):
            self.events.pop(0)

    def _refresh(
        self,
        record: WatchdogRecord,
        now_monotonic: float,
    ) -> None:
        base = (
            record.last_received_monotonic
            if record.last_received_monotonic is not None
            else record.registered_monotonic
        )

        age = max(
            0.0,
            now_monotonic - base,
        )

        if record.fault_latched:
            record.status = (
                WatchdogStatus.FAULT_LATCHED
            )
            return

        if age >= record.component.timeout_seconds:
            if record.status != WatchdogStatus.TIMED_OUT:
                record.timeout_count += 1

                record.last_error = (
                    "Heartbeat timeout"
                )

                self._event(
                    record,
                    WatchdogEventType.HEARTBEAT_TIMEOUT,
                    "Component heartbeat timed out",
                )

                if (
                    self.config.latch_critical_timeouts
                    and record.component.severity
                    == WatchdogSeverity.CRITICAL
                ):
                    record.fault_latched = True

                    record.status = (
                        WatchdogStatus.FAULT_LATCHED
                    )

                    self._event(
                        record,
                        WatchdogEventType.FAULT_LATCHED,
                        "Critical watchdog fault latched",
                    )
                    return

            record.status = WatchdogStatus.TIMED_OUT

        elif age >= record.component.warning_seconds:
            if record.status == WatchdogStatus.HEALTHY:
                self._event(
                    record,
                    WatchdogEventType.HEARTBEAT_DEGRADED,
                    "Heartbeat delay exceeded warning",
                )

            record.status = WatchdogStatus.DEGRADED

        elif record.last_received_monotonic is None:
            record.status = WatchdogStatus.WAITING

        elif record.status != WatchdogStatus.DEGRADED:
            record.status = WatchdogStatus.HEALTHY

    # --------------------------------------------------------
    # REGISTER COMPONENT
    # --------------------------------------------------------

    def register_component(
        self,
        component: WatchdogComponent,
    ) -> dict[str, Any]:
        with self._lock:
            key = self._key(
                component.tenant_id,
                component.robot_id,
                component.component_id,
            )

            if key in self.records:
                raise WatchdogValidationError(
                    "Component already registered"
                )

            if (
                len(self.records)
                >= self.config.maximum_components
            ):
                raise WatchdogError(
                    "Watchdog component capacity reached"
                )

            record = WatchdogRecord(
                component=component
            )

            self.records[key] = record

            self._event(
                record,
                WatchdogEventType.COMPONENT_REGISTERED,
                "Watchdog component registered",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # RECEIVE HEARTBEAT
    # --------------------------------------------------------

    def heartbeat(
        self,
        heartbeat: WatchdogHeartbeat,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                heartbeat.tenant_id,
                heartbeat.robot_id,
                heartbeat.component_id,
            )

            if (
                heartbeat.sequence_number
                <= record.last_sequence_number
            ):
                raise WatchdogValidationError(
                    "Duplicate or out-of-order heartbeat"
                )

            # The sender timestamp is validated but
            # is not trusted for timeout measurement.
            # Timeouts use the local monotonic clock.

            record.last_sequence_number = (
                heartbeat.sequence_number
            )

            record.last_heartbeat_at = (
                heartbeat.timestamp
            )

            record.last_received_monotonic = (
                time.monotonic()
            )

            record.heartbeat_count += 1

            if record.fault_latched:
                record.status = (
                    WatchdogStatus.FAULT_LATCHED
                )

            elif heartbeat.healthy:
                record.status = (
                    WatchdogStatus.HEALTHY
                )
                record.last_error = None

            else:
                record.status = (
                    WatchdogStatus.DEGRADED
                )
                record.last_error = (
                    "Component reported unhealthy"
                )

            self._event(
                record,
                WatchdogEventType.HEARTBEAT_RECEIVED,
                "Heartbeat received",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # SCAN TIMEOUTS
    # --------------------------------------------------------

    def scan(
        self,
        *,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            now = time.monotonic()

            results: list[dict[str, Any]] = []

            for record in self.records.values():
                if (
                    tenant_id is not None
                    and record.component.tenant_id
                    != tenant_id
                ):
                    continue

                self._refresh(
                    record,
                    now,
                )

                results.append(
                    record.to_dict()
                )

            return results

    # --------------------------------------------------------
    # SIMULATION COMMAND GATE
    # --------------------------------------------------------

    def evaluate_robot(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        simulation_only: bool = True,
    ) -> dict[str, Any]:
        with self._lock:
            key_prefix = (
                validate_id(tenant_id, "tenant_id"),
                validate_id(robot_id, "robot_id"),
            )

            records = [
                record
                for key, record
                in self.records.items()
                if key[:2] == key_prefix
                and record.component.required_for_operation
            ]

            now = time.monotonic()

            for record in records:
                self._refresh(record, now)

            allowed = (
                simulation_only
                and bool(records)
                and all(
                    record.status
                    == WatchdogStatus.HEALTHY
                    for record in records
                )
            )

            if not self.config.deny_on_degraded:
                allowed = (
                    simulation_only
                    and bool(records)
                    and all(
                        record.status in {
                            WatchdogStatus.HEALTHY,
                            WatchdogStatus.DEGRADED,
                        }
                        and not record.fault_latched
                        for record in records
                    )
                )

            return {
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "decision": (
                    WatchdogDecision.ALLOW_SIMULATION.value
                    if allowed
                    else WatchdogDecision.DENY.value
                ),
                "required_components": len(records),
                "unhealthy_components": [
                    record.component.component_id
                    for record in records
                    if record.status
                    != WatchdogStatus.HEALTHY
                ],
                "evaluated_at": utc_now().isoformat(),
            }

    # --------------------------------------------------------
    # RESET FAULT
    # --------------------------------------------------------

    def reset_fault(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        component_id: str,
        authorized: bool,
        independently_verified: bool,
    ) -> dict[str, Any]:
        """
        Simulation-only fault reset.

        The boolean inputs are illustrative.
        Production recovery must use authenticated
        approval and independent safety evidence.
        """
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                component_id,
            )

            if (
                not authorized
                or not independently_verified
            ):
                raise WatchdogRecoveryDenied(
                    "Verified authorization required"
                )

            if not record.fault_latched:
                raise WatchdogRecoveryDenied(
                    "No latched fault"
                )

            if record.last_received_monotonic is None:
                raise WatchdogRecoveryDenied(
                    "No heartbeat received"
                )

            age = (
                time.monotonic()
                - record.last_received_monotonic
            )

            if age >= record.component.warning_seconds:
                raise WatchdogRecoveryDenied(
                    "Fresh healthy heartbeat required"
                )

            if record.last_error not in {
                None,
                "Heartbeat timeout",
            }:
                raise WatchdogRecoveryDenied(
                    "Component remains unhealthy"
                )

            record.fault_latched = False
            record.status = WatchdogStatus.HEALTHY
            record.last_error = None

            self._event(
                record,
                WatchdogEventType.RECOVERY_COMPLETED,
                "Simulation watchdog fault reset",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # GET COMPONENT
    # --------------------------------------------------------

    def get_component(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        component_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                robot_id,
                component_id,
            )

            self._refresh(
                record,
                time.monotonic(),
            )

            return record.to_dict()

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        records = self.scan(
            tenant_id=tenant_id
        )

        return {
            "total_components": len(records),
            "healthy": sum(
                record["status"]
                == WatchdogStatus.HEALTHY.value
                for record in records
            ),
            "degraded": sum(
                record["status"]
                == WatchdogStatus.DEGRADED.value
                for record in records
            ),
            "timed_out": sum(
                record["status"]
                == WatchdogStatus.TIMED_OUT.value
                for record in records
            ),
            "fault_latched": sum(
                record["fault_latched"]
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
            components = self.scan(
                tenant_id=tenant_id
            )

            payload = {
                "generated_at": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "statistics": self.statistics(
                    tenant_id=tenant_id
                ),
                "components": components,
                "events": [
                    event.to_dict()
                    for event in self.events
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
    service = RoboticsWatchdogService()

    components = [
        ("perception_001", WatchdogComponentType.PERCEPTION),
        ("navigation_001", WatchdogComponentType.NAVIGATION),
        ("safety_monitor_001", WatchdogComponentType.SAFETY_MONITOR),
    ]

    for component_id, component_type in components:
        service.register_component(
            WatchdogComponent(
                tenant_id="tenant_alpha",
                robot_id="robot_001",
                component_id=component_id,
                component_type=component_type,
                severity=WatchdogSeverity.CRITICAL,
                timeout_seconds=3.0,
                warning_seconds=1.5,
            )
        )

        service.heartbeat(
            WatchdogHeartbeat(
                tenant_id="tenant_alpha",
                robot_id="robot_001",
                component_id=component_id,
                sequence_number=1,
                healthy=True,
            )
        )

    print(
        json.dumps(
            service.evaluate_robot(
                tenant_id="tenant_alpha",
                robot_id="robot_001",
            ),
            indent=2,
        )
    )

    service.export_json(
        "data/security/watchdog_report.json",
        tenant_id="tenant_alpha",
    )


if __name__ == "__main__":
    main()




"""
Physical AI & Autonomous Robotics Intelligence Platform

File: backend/app/security/safety_events.py

Python 3.11+
Standard library only.

Features:
    - Centralized robotics safety events
    - Multiple safety event categories
    - Event severity and lifecycle
    - Real-time event ingestion
    - Event deduplication
    - Tenant and robot isolation
    - Operator acknowledgment
    - Escalation and resolution
    - Safety event filtering
    - Audit logging
    - JSON reporting
    - Simulation-only advisory integration

IMPORTANT:
    This module records and manages safety events.

    It does not execute physical robot commands,
    guarantee physical stopping, or replace
    certified safety controllers.
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
from typing import Any, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================

class SafetyEventError(Exception):
    pass


class SafetyEventValidationError(SafetyEventError):
    pass


class SafetyEventNotFound(SafetyEventError):
    pass


class SafetyEventTransitionError(SafetyEventError):
    pass


# ============================================================
# ENUMS
# ============================================================

class SafetyEventType(str, Enum):
    COLLISION_RISK = "collision_risk"
    COLLISION_DETECTED = "collision_detected"

    HUMAN_PROXIMITY = "human_proximity"
    HUMAN_DETECTED = "human_detected"

    GEOFENCE_BREACH = "geofence_breach"

    FIRE = "fire"
    SMOKE = "smoke"
    GAS_LEAK = "gas_leak"
    OVERHEATING = "overheating"
    BATTERY_THERMAL = "battery_thermal"

    ELECTRICAL_FAULT = "electrical_fault"
    MECHANICAL_FAULT = "mechanical_fault"

    SENSOR_FAILURE = "sensor_failure"
    COMMUNICATION_FAILURE = "communication_failure"

    EMERGENCY_STOP = "emergency_stop"
    SAFE_STOP_REQUESTED = "safe_stop_requested"
    SAFE_STOP_CONFIRMED = "safe_stop_confirmed"

    COMMAND_DENIED = "command_denied"
    INTERLOCK_TRIGGERED = "interlock_triggered"

    SECURITY_VIOLATION = "security_violation"

    ROBOT_HEALTH = "robot_health"
    MAINTENANCE_REQUIRED = "maintenance_required"

    UNKNOWN = "unknown"


class SafetyEventSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class SafetyEventStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"


class SafetyEventSource(str, Enum):
    COLLISION_RISK = "collision_risk"
    HUMAN_PROXIMITY = "human_proximity"
    GEOFENCING = "geofencing"
    HAZARD_DETECTION = "hazard_detection"
    RISK_ASSESSMENT = "risk_assessment"
    SAFE_STATE = "safe_state"
    COMMAND_INTERLOCK = "command_interlock"
    ROBOT_HEALTH = "robot_health"
    EMERGENCY_ADVISORY = "emergency_advisory"
    CYBERSECURITY = "cybersecurity"
    MANUAL = "manual"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


def validate_identifier(
    value: str,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not all(
            character.isalnum()
            or character in "_-"
            for character in value
        )
    ):
        raise SafetyEventValidationError(
            f"Invalid {name}"
        )

    return value


def validate_timestamp(
    value: datetime,
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise SafetyEventValidationError(
            "Timezone-aware timestamp required"
        )

    return value.astimezone(timezone.utc)


def validate_risk_score(
    value: float,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not (0 <= value <= 100)
    ):
        raise SafetyEventValidationError(
            "Risk score must be between 0 and 100"
        )

    return float(value)


def json_safe_mapping(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        serialized = json.dumps(
            dict(value),
            allow_nan=False,
            sort_keys=True,
        )
        return json.loads(serialized)
    except (TypeError, ValueError) as exc:
        raise SafetyEventValidationError(
            "Metadata must be JSON serializable"
        ) from exc


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class SafetyEventConfig:
    maximum_events: int = 100000
    maximum_audit_events: int = 20000

    deduplicate_open_events: bool = True

    auto_escalate_critical: bool = True

    require_acknowledgment_before_resolution: bool = True

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_events",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise SafetyEventValidationError(
                    f"{name} must be positive"
                )

        if not self.simulation_only:
            raise SafetyEventValidationError(
                "Physical hardware control is unsupported"
            )


# ============================================================
# EVENT INPUT
# ============================================================

@dataclass(frozen=True)
class SafetyEventInput:
    tenant_id: str
    robot_id: str

    event_type: SafetyEventType
    severity: SafetyEventSeverity

    source: SafetyEventSource

    title: str
    description: str

    risk_score: float = 0.0

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

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

        validate_risk_score(
            self.risk_score
        )

        validate_timestamp(
            self.timestamp
        )

        for name in (
            "title",
            "description",
        ):
            value = getattr(self, name)

            if (
                not isinstance(value, str)
                or not value.strip()
                or len(value) > 2000
            ):
                raise SafetyEventValidationError(
                    f"Invalid {name}"
                )

        object.__setattr__(
            self,
            "metadata",
            json_safe_mapping(self.metadata),
        )


# ============================================================
# EVENT RECORD
# ============================================================

@dataclass
class SafetyEventRecord:
    event_id: str

    tenant_id: str
    robot_id: str

    event_type: SafetyEventType
    severity: SafetyEventSeverity
    source: SafetyEventSource

    title: str
    description: str

    risk_score: float
    metadata: dict[str, Any]

    status: SafetyEventStatus = (
        SafetyEventStatus.OPEN
    )

    occurrence_count: int = 1

    created_at: datetime = field(
        default_factory=utc_now
    )

    last_seen_at: datetime = field(
        default_factory=utc_now
    )

    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None

    escalated_by: str | None = None
    escalated_at: datetime | None = None

    resolved_by: str | None = None
    resolved_at: datetime | None = None

    resolution_notes: str | None = None

    closed_by: str | None = None
    closed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "source": self.source.value,
            "title": self.title,
            "description": self.description,
            "risk_score": self.risk_score,
            "metadata": self.metadata,
            "status": self.status.value,
            "occurrence_count": self.occurrence_count,
            "created_at": self.created_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat(),
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": (
                self.acknowledged_at.isoformat()
                if self.acknowledged_at
                else None
            ),
            "escalated_by": self.escalated_by,
            "escalated_at": (
                self.escalated_at.isoformat()
                if self.escalated_at
                else None
            ),
            "resolved_by": self.resolved_by,
            "resolved_at": (
                self.resolved_at.isoformat()
                if self.resolved_at
                else None
            ),
            "resolution_notes": self.resolution_notes,
            "closed_by": self.closed_by,
            "closed_at": (
                self.closed_at.isoformat()
                if self.closed_at
                else None
            ),
        }


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class SafetyAuditEvent:
    audit_id: str
    timestamp: datetime

    tenant_id: str
    robot_id: str
    event_id: str

    action: str
    actor_id: str

    details: Mapping[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "event_id": self.event_id,
            "action": self.action,
            "actor_id": self.actor_id,
            "details": dict(self.details),
        }


# ============================================================
# SAFETY EVENT SERVICE
# ============================================================

class RoboticsSafetyEventService:

    def __init__(
        self,
        config: SafetyEventConfig | None = None,
    ) -> None:
        self.config = (
            config or SafetyEventConfig()
        )

        self.events: dict[
            str,
            SafetyEventRecord
        ] = {}

        self.audit_events: list[
            SafetyAuditEvent
        ] = []

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit(
        self,
        record: SafetyEventRecord,
        action: str,
        actor_id: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.audit_events.append(
            SafetyAuditEvent(
                audit_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                tenant_id=record.tenant_id,
                robot_id=record.robot_id,
                event_id=record.event_id,
                action=action,
                actor_id=actor_id,
                details=dict(details or {}),
            )
        )

        if (
            len(self.audit_events)
            > self.config.maximum_audit_events
        ):
            self.audit_events.pop(0)

    # --------------------------------------------------------
    # GET EVENT
    # --------------------------------------------------------

    def _get(
        self,
        tenant_id: str,
        event_id: str,
    ) -> SafetyEventRecord:
        record = self.events.get(
            event_id
        )

        if (
            record is None
            or record.tenant_id != tenant_id
        ):
            raise SafetyEventNotFound(
                "Safety event not found"
            )

        return record

    def get_event(
        self,
        *,
        tenant_id: str,
        event_id: str,
    ) -> SafetyEventRecord:
        with self._lock:
            return replace(
                self._get(
                    tenant_id,
                    event_id,
                ),
                metadata=dict(
                    self._get(
                        tenant_id,
                        event_id,
                    ).metadata
                ),
            )

    # --------------------------------------------------------
    # INGEST EVENT
    # --------------------------------------------------------

    def ingest(
        self,
        event: SafetyEventInput,
    ) -> SafetyEventRecord:
        with self._lock:
            if self.config.deduplicate_open_events:
                for existing in self.events.values():
                    if (
                        existing.tenant_id
                        == event.tenant_id
                        and existing.robot_id
                        == event.robot_id
                        and existing.event_type
                        == event.event_type
                        and existing.source
                        == event.source
                        and existing.status in {
                            SafetyEventStatus.OPEN,
                            SafetyEventStatus.ACKNOWLEDGED,
                            SafetyEventStatus.ESCALATED,
                        }
                    ):
                        existing.occurrence_count += 1
                        existing.last_seen_at = utc_now()

                        severity_rank = {
                            SafetyEventSeverity.INFO: 0,
                            SafetyEventSeverity.LOW: 1,
                            SafetyEventSeverity.MODERATE: 2,
                            SafetyEventSeverity.HIGH: 3,
                            SafetyEventSeverity.CRITICAL: 4,
                        }

                        if (
                            severity_rank[event.severity]
                            > severity_rank[existing.severity]
                        ):
                            existing.severity = event.severity

                        existing.risk_score = max(
                            existing.risk_score,
                            event.risk_score,
                        )

                        if (
                            self.config.auto_escalate_critical
                            and existing.severity
                            == SafetyEventSeverity.CRITICAL
                        ):
                            existing.status = (
                                SafetyEventStatus.ESCALATED
                            )
                            existing.escalated_by = "system"
                            existing.escalated_at = utc_now()

                        self._audit(
                            existing,
                            "event_repeated",
                            "system",
                            {
                                "occurrence_count": (
                                    existing.occurrence_count
                                )
                            },
                        )

                        return replace(
                            existing,
                            metadata=dict(
                                existing.metadata
                            ),
                        )

            if (
                len(self.events)
                >= self.config.maximum_events
            ):
                raise SafetyEventError(
                    "Safety event capacity reached"
                )

            record = SafetyEventRecord(
                event_id=str(uuid.uuid4()),
                tenant_id=event.tenant_id,
                robot_id=event.robot_id,
                event_type=event.event_type,
                severity=event.severity,
                source=event.source,
                title=event.title,
                description=event.description,
                risk_score=event.risk_score,
                metadata=dict(event.metadata),
            )

            if (
                self.config.auto_escalate_critical
                and record.severity
                == SafetyEventSeverity.CRITICAL
            ):
                record.status = (
                    SafetyEventStatus.ESCALATED
                )
                record.escalated_by = "system"
                record.escalated_at = utc_now()

            self.events[
                record.event_id
            ] = record

            self._audit(
                record,
                "event_created",
                "system",
            )

            return replace(
                record,
                metadata=dict(record.metadata),
            )

    # --------------------------------------------------------
    # ACKNOWLEDGE
    # --------------------------------------------------------

    def acknowledge(
        self,
        *,
        tenant_id: str,
        event_id: str,
        operator_id: str,
    ) -> SafetyEventRecord:
        validate_identifier(
            operator_id,
            "operator_id",
        )

        with self._lock:
            record = self._get(
                tenant_id,
                event_id,
            )

            if record.status not in {
                SafetyEventStatus.OPEN,
                SafetyEventStatus.ESCALATED,
            }:
                raise SafetyEventTransitionError(
                    "Event cannot be acknowledged"
                )

            was_escalated = (
                record.status
                == SafetyEventStatus.ESCALATED
            )

            record.acknowledged_by = operator_id
            record.acknowledged_at = utc_now()

            if not was_escalated:
                record.status = (
                    SafetyEventStatus.ACKNOWLEDGED
                )

            self._audit(
                record,
                "event_acknowledged",
                operator_id,
            )

            return replace(
                record,
                metadata=dict(record.metadata),
            )

    # --------------------------------------------------------
    # ESCALATE
    # --------------------------------------------------------

    def escalate(
        self,
        *,
        tenant_id: str,
        event_id: str,
        actor_id: str,
    ) -> SafetyEventRecord:
        validate_identifier(
            actor_id,
            "actor_id",
        )

        with self._lock:
            record = self._get(
                tenant_id,
                event_id,
            )

            if record.status not in {
                SafetyEventStatus.OPEN,
                SafetyEventStatus.ACKNOWLEDGED,
            }:
                raise SafetyEventTransitionError(
                    "Event cannot be escalated"
                )

            record.status = (
                SafetyEventStatus.ESCALATED
            )

            record.escalated_by = actor_id
            record.escalated_at = utc_now()

            self._audit(
                record,
                "event_escalated",
                actor_id,
            )

            return replace(
                record,
                metadata=dict(record.metadata),
            )

    # --------------------------------------------------------
    # RESOLVE
    # --------------------------------------------------------

    def resolve(
        self,
        *,
        tenant_id: str,
        event_id: str,
        reviewer_id: str,
        resolution_notes: str,
        independently_verified: bool,
    ) -> SafetyEventRecord:
        validate_identifier(
            reviewer_id,
            "reviewer_id",
        )

        if (
            not isinstance(resolution_notes, str)
            or not resolution_notes.strip()
            or len(resolution_notes) > 2000
        ):
            raise SafetyEventValidationError(
                "Resolution notes required"
            )

        if not independently_verified:
            raise SafetyEventTransitionError(
                "Independent verification required"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                event_id,
            )

            if record.status not in {
                SafetyEventStatus.ACKNOWLEDGED,
                SafetyEventStatus.ESCALATED,
            }:
                raise SafetyEventTransitionError(
                    "Event cannot be resolved"
                )

            if (
                self.config.require_acknowledgment_before_resolution
                and record.acknowledged_at is None
            ):
                raise SafetyEventTransitionError(
                    "Event must be acknowledged first"
                )

            record.status = (
                SafetyEventStatus.RESOLVED
            )

            record.resolved_by = reviewer_id
            record.resolved_at = utc_now()
            record.resolution_notes = (
                resolution_notes
            )

            self._audit(
                record,
                "event_resolved",
                reviewer_id,
            )

            return replace(
                record,
                metadata=dict(record.metadata),
            )

    # --------------------------------------------------------
    # CLOSE
    # --------------------------------------------------------

    def close(
        self,
        *,
        tenant_id: str,
        event_id: str,
        actor_id: str,
    ) -> SafetyEventRecord:
        validate_identifier(
            actor_id,
            "actor_id",
        )

        with self._lock:
            record = self._get(
                tenant_id,
                event_id,
            )

            if (
                record.status
                != SafetyEventStatus.RESOLVED
            ):
                raise SafetyEventTransitionError(
                    "Only resolved events can be closed"
                )

            record.status = (
                SafetyEventStatus.CLOSED
            )

            record.closed_by = actor_id
            record.closed_at = utc_now()

            self._audit(
                record,
                "event_closed",
                actor_id,
            )

            return replace(
                record,
                metadata=dict(record.metadata),
            )

    # --------------------------------------------------------
    # LIST EVENTS
    # --------------------------------------------------------

    def list_events(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
        severity: SafetyEventSeverity | None = None,
        status: SafetyEventStatus | None = None,
        event_type: SafetyEventType | None = None,
    ) -> list[SafetyEventRecord]:
        with self._lock:
            records = [
                record
                for record in self.events.values()
                if record.tenant_id == tenant_id
            ]

            if robot_id is not None:
                records = [
                    record
                    for record in records
                    if record.robot_id == robot_id
                ]

            if severity is not None:
                records = [
                    record
                    for record in records
                    if record.severity == severity
                ]

            if status is not None:
                records = [
                    record
                    for record in records
                    if record.status == status
                ]

            if event_type is not None:
                records = [
                    record
                    for record in records
                    if record.event_type == event_type
                ]

            return [
                replace(
                    record,
                    metadata=dict(record.metadata),
                )
                for record in records
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
                for record in self.events.values()
                if (
                    tenant_id is None
                    or record.tenant_id == tenant_id
                )
            ]

            return {
                "total_events": len(records),
                "open_events": sum(
                    record.status
                    == SafetyEventStatus.OPEN
                    for record in records
                ),
                "escalated_events": sum(
                    record.status
                    == SafetyEventStatus.ESCALATED
                    for record in records
                ),
                "critical_events": sum(
                    record.severity
                    == SafetyEventSeverity.CRITICAL
                    for record in records
                ),
                "resolved_events": sum(
                    record.status
                    == SafetyEventStatus.RESOLVED
                    for record in records
                ),
                "closed_events": sum(
                    record.status
                    == SafetyEventStatus.CLOSED
                    for record in records
                ),
                "total_occurrences": sum(
                    record.occurrence_count
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
                "generated_at": utc_iso(),
                "tenant_id": tenant_id,
                "statistics": self.statistics(
                    tenant_id=tenant_id
                ),
                "events": [
                    record.to_dict()
                    for record in self.events.values()
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
    service = RoboticsSafetyEventService()

    event = SafetyEventInput(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        event_type=SafetyEventType.HUMAN_PROXIMITY,
        severity=SafetyEventSeverity.CRITICAL,
        source=SafetyEventSource.HUMAN_PROXIMITY,
        title="Unsafe human proximity",
        description=(
            "A person was detected inside the "
            "configured critical proximity zone."
        ),
        risk_score=100.0,
        metadata={
            "distance_meters": 0.55,
            "simulation_only": True,
        },
    )

    record = service.ingest(
        event
    )

    service.acknowledge(
        tenant_id="tenant_alpha",
        event_id=record.event_id,
        operator_id="operator_001",
    )

    service.resolve(
        tenant_id="tenant_alpha",
        event_id=record.event_id,
        reviewer_id="safety_officer_001",
        resolution_notes=(
            "Simulation review completed and "
            "clearance independently checked."
        ),
        independently_verified=True,
    )

    service.close(
        tenant_id="tenant_alpha",
        event_id=record.event_id,
        actor_id="safety_officer_001",
    )

    print(
        json.dumps(
            service.get_event(
                tenant_id="tenant_alpha",
                event_id=record.event_id,
            ).to_dict(),
            indent=2,
        )
    )

    service.export_json(
        "data/security/safety_events_report.json",
        tenant_id="tenant_alpha",
    )


if __name__ == "__main__":
    main()

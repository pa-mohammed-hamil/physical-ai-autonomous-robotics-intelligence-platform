
"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
    backend/app/security/emergency_advisory.py

Python 3.11+
Standard library only.

Features:
    - Emergency incident classification
    - Collision and human-safety advisories
    - Fire, smoke and overheating alerts
    - Battery emergency recommendations
    - Robot communication failure advisories
    - Geofence violation escalation
    - Incident lifecycle management
    - Human acknowledgement and review
    - Deterministic advisory generation
    - Emergency audit logging
    - JSON report export

IMPORTANT:
    This is a simulation and decision-support module.

    It does NOT:
        - Control physical robots
        - Trigger emergency-stop hardware
        - Replace trained emergency responders
        - Override command interlocks
        - Authorize restarting equipment

    Physical emergency responses must be
    enforced by independent safety systems.
"""

from __future__ import annotations

import json
import math
import threading
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


# ============================================================
# EXCEPTIONS
# ============================================================

class EmergencyAdvisoryError(Exception):
    pass


class EmergencyValidationError(
    EmergencyAdvisoryError
):
    pass


class EmergencyIncidentNotFound(
    EmergencyAdvisoryError
):
    pass


class EmergencyTransitionError(
    EmergencyAdvisoryError
):
    pass


class EmergencyAccessDenied(
    EmergencyAdvisoryError
):
    pass


# ============================================================
# ENUMS
# ============================================================

class EmergencyType(str, Enum):
    COLLISION_RISK = "collision_risk"
    COLLISION_DETECTED = "collision_detected"

    HUMAN_PROXIMITY = "human_proximity"

    FIRE = "fire"
    SMOKE = "smoke"
    OVERHEATING = "overheating"

    BATTERY_THERMAL = "battery_thermal"
    BATTERY_LOW = "battery_low"

    ELECTRICAL_FAULT = "electrical_fault"

    GEOFENCE_BREACH = "geofence_breach"

    COMMUNICATION_LOSS = "communication_loss"
    SENSOR_FAILURE = "sensor_failure"

    ROBOT_MALFUNCTION = "robot_malfunction"

    EMERGENCY_STOP = "emergency_stop"

    UNKNOWN = "unknown"


class EmergencySeverity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class EmergencyIncidentStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    UNDER_REVIEW = "under_review"
    RESOLVED = "resolved"
    CLOSED = "closed"


class EmergencyAdvisoryPriority(str, Enum):
    INFORMATION = "information"
    CAUTION = "caution"
    URGENT = "urgent"
    IMMEDIATE = "immediate"


class EmergencyActionType(str, Enum):
    NOTIFY_OPERATOR = "notify_operator"
    NOTIFY_SAFETY_TEAM = "notify_safety_team"
    CONTACT_EMERGENCY_SERVICES = (
        "contact_emergency_services"
    )

    KEEP_CLEAR = "keep_clear"
    EVACUATE_AREA = "evacuate_area"

    REQUEST_SAFE_STOP = "request_safe_stop"
    AVOID_ROBOT_APPROACH = "avoid_robot_approach"

    ISOLATE_AREA = "isolate_area"
    REQUEST_TECHNICIAN = "request_technician"

    INSPECT_SENSORS = "inspect_sensors"
    CHECK_COMMUNICATION = "check_communication"

    MONITOR_CONDITION = "monitor_condition"
    REVIEW_GEOFENCE = "review_geofence"

    REQUIRE_HUMAN_REVIEW = "require_human_review"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


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
            character.isalnum()
            or character in "_-"
            for character in value
        )
    ):
        raise EmergencyValidationError(
            f"Invalid {field_name}"
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
        raise EmergencyValidationError(
            "Timestamp must be timezone-aware"
        )

    return value.astimezone(
        timezone.utc
    )


def validate_score(
    value: float,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not (0 <= value <= 100)
    ):
        raise EmergencyValidationError(
            f"{name} must be between 0 and 100"
        )

    return float(value)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class EmergencyAdvisoryConfig:
    high_risk_threshold: float = 70.0
    critical_risk_threshold: float = 90.0

    maximum_incidents: int = 100000
    maximum_audit_events: int = 20000

    require_human_review_for_high: bool = True

    simulation_only: bool = True

    def __post_init__(self) -> None:
        high = validate_score(
            self.high_risk_threshold,
            "high_risk_threshold",
        )

        critical = validate_score(
            self.critical_risk_threshold,
            "critical_risk_threshold",
        )

        if high >= critical:
            raise EmergencyValidationError(
                "Critical threshold must exceed high"
            )

        if not self.simulation_only:
            raise EmergencyValidationError(
                "Hardware emergency actions "
                "are not supported"
            )

        for name in (
            "maximum_incidents",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise EmergencyValidationError(
                    f"{name} must be positive"
                )


# ============================================================
# EMERGENCY OBSERVATION
# ============================================================

@dataclass(frozen=True)
class EmergencyObservation:
    tenant_id: str
    robot_id: str

    emergency_type: EmergencyType

    risk_score: float

    description: str

    location: str | None = None

    source: str = "robot_telemetry"

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

        validate_score(
            self.risk_score,
            "risk_score",
        )

        validate_timestamp(
            self.timestamp
        )

        if (
            not isinstance(self.description, str)
            or not self.description.strip()
            or len(self.description) > 2000
        ):
            raise EmergencyValidationError(
                "Invalid emergency description"
            )

        if (
            self.location is not None
            and len(self.location) > 256
        ):
            raise EmergencyValidationError(
                "Location description too long"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "emergency_type": (
                self.emergency_type.value
            ),
            "risk_score": self.risk_score,
            "description": self.description,
            "location": self.location,
            "source": self.source,
            "timestamp": (
                self.timestamp.isoformat()
            ),
        }


# ============================================================
# EMERGENCY ADVISORY
# ============================================================

@dataclass(frozen=True)
class EmergencyAdvisory:
    advisory_id: str

    emergency_type: EmergencyType
    severity: EmergencySeverity
    priority: EmergencyAdvisoryPriority

    title: str
    summary: str

    recommended_actions: tuple[
        EmergencyActionType,
        ...
    ]

    requires_human_review: bool

    generated_at: str = field(
        default_factory=utc_iso
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "advisory_id": self.advisory_id,
            "emergency_type": (
                self.emergency_type.value
            ),
            "severity": self.severity.value,
            "priority": self.priority.value,
            "title": self.title,
            "summary": self.summary,
            "recommended_actions": [
                action.value
                for action in self.recommended_actions
            ],
            "requires_human_review": (
                self.requires_human_review
            ),
            "generated_at": self.generated_at,
        }


# ============================================================
# INCIDENT RECORD
# ============================================================

@dataclass
class EmergencyIncident:
    incident_id: str

    tenant_id: str
    robot_id: str

    observation: EmergencyObservation
    advisory: EmergencyAdvisory

    status: EmergencyIncidentStatus = (
        EmergencyIncidentStatus.OPEN
    )

    created_at: str = field(
        default_factory=utc_iso
    )

    acknowledged_by: str | None = None
    acknowledged_at: str | None = None

    reviewed_by: str | None = None
    reviewed_at: str | None = None

    resolved_by: str | None = None
    resolved_at: str | None = None

    closed_by: str | None = None
    closed_at: str | None = None

    resolution_notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "observation": (
                self.observation.to_dict()
            ),
            "advisory": (
                self.advisory.to_dict()
            ),
            "status": self.status.value,
            "created_at": self.created_at,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at,
            "closed_by": self.closed_by,
            "closed_at": self.closed_at,
            "resolution_notes": (
                self.resolution_notes
            ),
        }


# ============================================================
# ADVISORY POLICY
# ============================================================

ADVISORY_ACTIONS: dict[
    EmergencyType,
    tuple[EmergencyActionType, ...]
] = {
    EmergencyType.COLLISION_RISK: (
        EmergencyActionType.REQUEST_SAFE_STOP,
        EmergencyActionType.KEEP_CLEAR,
        EmergencyActionType.NOTIFY_OPERATOR,
    ),

    EmergencyType.COLLISION_DETECTED: (
        EmergencyActionType.REQUEST_SAFE_STOP,
        EmergencyActionType.KEEP_CLEAR,
        EmergencyActionType.NOTIFY_SAFETY_TEAM,
        EmergencyActionType.REQUIRE_HUMAN_REVIEW,
    ),

    EmergencyType.HUMAN_PROXIMITY: (
        EmergencyActionType.REQUEST_SAFE_STOP,
        EmergencyActionType.KEEP_CLEAR,
        EmergencyActionType.NOTIFY_OPERATOR,
    ),

    EmergencyType.FIRE: (
        EmergencyActionType.EVACUATE_AREA,
        EmergencyActionType.CONTACT_EMERGENCY_SERVICES,
        EmergencyActionType.NOTIFY_SAFETY_TEAM,
        EmergencyActionType.AVOID_ROBOT_APPROACH,
    ),

    EmergencyType.SMOKE: (
        EmergencyActionType.KEEP_CLEAR,
        EmergencyActionType.NOTIFY_SAFETY_TEAM,
        EmergencyActionType.REQUIRE_HUMAN_REVIEW,
    ),

    EmergencyType.OVERHEATING: (
        EmergencyActionType.REQUEST_SAFE_STOP,
        EmergencyActionType.ISOLATE_AREA,
        EmergencyActionType.REQUEST_TECHNICIAN,
    ),

    EmergencyType.BATTERY_THERMAL: (
        EmergencyActionType.EVACUATE_AREA,
        EmergencyActionType.NOTIFY_SAFETY_TEAM,
        EmergencyActionType.CONTACT_EMERGENCY_SERVICES,
        EmergencyActionType.AVOID_ROBOT_APPROACH,
    ),

    EmergencyType.BATTERY_LOW: (
        EmergencyActionType.NOTIFY_OPERATOR,
        EmergencyActionType.MONITOR_CONDITION,
    ),

    EmergencyType.ELECTRICAL_FAULT: (
        EmergencyActionType.REQUEST_SAFE_STOP,
        EmergencyActionType.KEEP_CLEAR,
        EmergencyActionType.NOTIFY_SAFETY_TEAM,
        EmergencyActionType.REQUEST_TECHNICIAN,
    ),

    EmergencyType.GEOFENCE_BREACH: (
        EmergencyActionType.REQUEST_SAFE_STOP,
        EmergencyActionType.NOTIFY_OPERATOR,
        EmergencyActionType.REVIEW_GEOFENCE,
    ),

    EmergencyType.COMMUNICATION_LOSS: (
        EmergencyActionType.NOTIFY_OPERATOR,
        EmergencyActionType.CHECK_COMMUNICATION,
        EmergencyActionType.REQUIRE_HUMAN_REVIEW,
    ),

    EmergencyType.SENSOR_FAILURE: (
        EmergencyActionType.REQUEST_SAFE_STOP,
        EmergencyActionType.INSPECT_SENSORS,
        EmergencyActionType.REQUEST_TECHNICIAN,
    ),

    EmergencyType.ROBOT_MALFUNCTION: (
        EmergencyActionType.REQUEST_SAFE_STOP,
        EmergencyActionType.REQUEST_TECHNICIAN,
        EmergencyActionType.NOTIFY_OPERATOR,
    ),

    EmergencyType.EMERGENCY_STOP: (
        EmergencyActionType.KEEP_CLEAR,
        EmergencyActionType.NOTIFY_OPERATOR,
        EmergencyActionType.REQUIRE_HUMAN_REVIEW,
    ),

    EmergencyType.UNKNOWN: (
        EmergencyActionType.NOTIFY_OPERATOR,
        EmergencyActionType.REQUIRE_HUMAN_REVIEW,
        EmergencyActionType.KEEP_CLEAR,
    ),
}


# ============================================================
# ADVISORY ENGINE
# ============================================================

class EmergencyAdvisoryEngine:

    def __init__(
        self,
        config: EmergencyAdvisoryConfig,
    ) -> None:
        self.config = config

    def classify_severity(
        self,
        observation: EmergencyObservation,
    ) -> EmergencySeverity:
        emergency = observation.emergency_type
        score = observation.risk_score

        critical_types = {
            EmergencyType.FIRE,
            EmergencyType.COLLISION_DETECTED,
            EmergencyType.BATTERY_THERMAL,
            EmergencyType.EMERGENCY_STOP,
        }

        if emergency in critical_types:
            return EmergencySeverity.CRITICAL

        if score >= self.config.critical_risk_threshold:
            return EmergencySeverity.CRITICAL

        if score >= self.config.high_risk_threshold:
            return EmergencySeverity.HIGH

        if score >= 40:
            return EmergencySeverity.MODERATE

        return EmergencySeverity.LOW

    def generate(
        self,
        observation: EmergencyObservation,
    ) -> EmergencyAdvisory:
        severity = self.classify_severity(
            observation
        )

        priority_map = {
            EmergencySeverity.LOW:
                EmergencyAdvisoryPriority.INFORMATION,

            EmergencySeverity.MODERATE:
                EmergencyAdvisoryPriority.CAUTION,

            EmergencySeverity.HIGH:
                EmergencyAdvisoryPriority.URGENT,

            EmergencySeverity.CRITICAL:
                EmergencyAdvisoryPriority.IMMEDIATE,
        }

        actions = list(
            ADVISORY_ACTIONS.get(
                observation.emergency_type,
                ADVISORY_ACTIONS[
                    EmergencyType.UNKNOWN
                ],
            )
        )

        requires_review = (
            severity in {
                EmergencySeverity.HIGH,
                EmergencySeverity.CRITICAL,
            }
            and self.config.require_human_review_for_high
        )

        if (
            requires_review
            and EmergencyActionType.REQUIRE_HUMAN_REVIEW
            not in actions
        ):
            actions.append(
                EmergencyActionType.REQUIRE_HUMAN_REVIEW
            )

        title = (
            observation.emergency_type.value
            .replace("_", " ")
            .title()
        )

        summary = (
            f"{title} detected for robot "
            f"{observation.robot_id}. "
            f"Risk score: {observation.risk_score:.1f}/100. "
            "Recommendations require validation by "
            "authorized safety personnel."
        )

        return EmergencyAdvisory(
            advisory_id=str(uuid.uuid4()),
            emergency_type=observation.emergency_type,
            severity=severity,
            priority=priority_map[severity],
            title=title,
            summary=summary,
            recommended_actions=tuple(actions),
            requires_human_review=requires_review,
        )


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class EmergencyAuditEvent:
    event_id: str
    timestamp: str

    tenant_id: str
    robot_id: str
    incident_id: str

    action: str
    actor_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "incident_id": self.incident_id,
            "action": self.action,
            "actor_id": self.actor_id,
        }


# ============================================================
# CENTRAL EMERGENCY ADVISORY SERVICE
# ============================================================

class RoboticsEmergencyAdvisoryService:

    def __init__(
        self,
        config: EmergencyAdvisoryConfig | None = None,
    ) -> None:
        self.config = (
            config or EmergencyAdvisoryConfig()
        )

        self.engine = EmergencyAdvisoryEngine(
            self.config
        )

        self.incidents: dict[
            str,
            EmergencyIncident
        ] = {}

        self.audit_events: list[
            EmergencyAuditEvent
        ] = []

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit(
        self,
        incident: EmergencyIncident,
        action: str,
        actor_id: str | None = None,
    ) -> None:
        self.audit_events.append(
            EmergencyAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_iso(),
                tenant_id=incident.tenant_id,
                robot_id=incident.robot_id,
                incident_id=incident.incident_id,
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
    # CREATE INCIDENT
    # --------------------------------------------------------

    def create_incident(
        self,
        observation: EmergencyObservation,
    ) -> EmergencyIncident:
        advisory = self.engine.generate(
            observation
        )

        incident = EmergencyIncident(
            incident_id=str(uuid.uuid4()),
            tenant_id=observation.tenant_id,
            robot_id=observation.robot_id,
            observation=observation,
            advisory=advisory,
        )

        with self._lock:
            if (
                len(self.incidents)
                >= self.config.maximum_incidents
            ):
                raise EmergencyAdvisoryError(
                    "Incident capacity reached"
                )

            self.incidents[
                incident.incident_id
            ] = incident

            self._audit(
                incident,
                "incident_created",
            )

        return incident

    # --------------------------------------------------------
    # GET INCIDENT
    # --------------------------------------------------------

    def get_incident(
        self,
        *,
        tenant_id: str,
        incident_id: str,
    ) -> EmergencyIncident:
        with self._lock:
            incident = self.incidents.get(
                incident_id
            )

            if (
                incident is None
                or incident.tenant_id != tenant_id
            ):
                raise EmergencyIncidentNotFound(
                    "Incident not found"
                )

            return incident

    # --------------------------------------------------------
    # ACKNOWLEDGE INCIDENT
    # --------------------------------------------------------

    def acknowledge(
        self,
        *,
        tenant_id: str,
        incident_id: str,
        operator_id: str,
    ) -> EmergencyIncident:
        validate_identifier(
            operator_id,
            "operator_id",
        )

        with self._lock:
            incident = self.get_incident(
                tenant_id=tenant_id,
                incident_id=incident_id,
            )

            if incident.status != (
                EmergencyIncidentStatus.OPEN
            ):
                raise EmergencyTransitionError(
                    "Only open incidents can be acknowledged"
                )

            incident.status = (
                EmergencyIncidentStatus.ACKNOWLEDGED
            )

            incident.acknowledged_by = operator_id
            incident.acknowledged_at = utc_iso()

            self._audit(
                incident,
                "incident_acknowledged",
                operator_id,
            )

            return incident

    # --------------------------------------------------------
    # START REVIEW
    # --------------------------------------------------------

    def start_review(
        self,
        *,
        tenant_id: str,
        incident_id: str,
        reviewer_id: str,
    ) -> EmergencyIncident:
        validate_identifier(
            reviewer_id,
            "reviewer_id",
        )

        with self._lock:
            incident = self.get_incident(
                tenant_id=tenant_id,
                incident_id=incident_id,
            )

            if incident.status not in {
                EmergencyIncidentStatus.OPEN,
                EmergencyIncidentStatus.ACKNOWLEDGED,
            }:
                raise EmergencyTransitionError(
                    "Incident cannot enter review"
                )

            incident.status = (
                EmergencyIncidentStatus.UNDER_REVIEW
            )

            incident.reviewed_by = reviewer_id
            incident.reviewed_at = utc_iso()

            self._audit(
                incident,
                "incident_review_started",
                reviewer_id,
            )

            return incident

    # --------------------------------------------------------
    # RESOLVE INCIDENT
    # --------------------------------------------------------

    def resolve(
        self,
        *,
        tenant_id: str,
        incident_id: str,
        reviewer_id: str,
        resolution_notes: str,
    ) -> EmergencyIncident:
        validate_identifier(
            reviewer_id,
            "reviewer_id",
        )

        if (
            not isinstance(resolution_notes, str)
            or not resolution_notes.strip()
            or len(resolution_notes) > 2000
        ):
            raise EmergencyValidationError(
                "Valid resolution notes are required"
            )

        with self._lock:
            incident = self.get_incident(
                tenant_id=tenant_id,
                incident_id=incident_id,
            )

            if incident.status != (
                EmergencyIncidentStatus.UNDER_REVIEW
            ):
                raise EmergencyTransitionError(
                    "Incident must be under review"
                )

            incident.status = (
                EmergencyIncidentStatus.RESOLVED
            )

            incident.resolved_by = reviewer_id
            incident.resolved_at = utc_iso()
            incident.resolution_notes = resolution_notes

            self._audit(
                incident,
                "incident_resolved",
                reviewer_id,
            )

            return incident

    # --------------------------------------------------------
    # CLOSE INCIDENT
    # --------------------------------------------------------

    def close(
        self,
        *,
        tenant_id: str,
        incident_id: str,
        operator_id: str,
    ) -> EmergencyIncident:
        validate_identifier(
            operator_id,
            "operator_id",
        )

        with self._lock:
            incident = self.get_incident(
                tenant_id=tenant_id,
                incident_id=incident_id,
            )

            if incident.status != (
                EmergencyIncidentStatus.RESOLVED
            ):
                raise EmergencyTransitionError(
                    "Only resolved incidents can be closed"
                )

            incident.status = (
                EmergencyIncidentStatus.CLOSED
            )

            incident.closed_by = operator_id
            incident.closed_at = utc_iso()

            self._audit(
                incident,
                "incident_closed",
                operator_id,
            )

            return incident

    # --------------------------------------------------------
    # LIST INCIDENTS
    # --------------------------------------------------------

    def list_incidents(
        self,
        *,
        tenant_id: str,
        status: EmergencyIncidentStatus | None = None,
    ) -> list[EmergencyIncident]:
        with self._lock:
            results = [
                incident
                for incident in self.incidents.values()
                if incident.tenant_id == tenant_id
            ]

            if status is not None:
                results = [
                    incident
                    for incident in results
                    if incident.status == status
                ]

            return results

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
    ) -> dict[str, Any]:
        with self._lock:
            return {
                "total_incidents": len(
                    self.incidents
                ),
                "open_incidents": sum(
                    incident.status
                    == EmergencyIncidentStatus.OPEN
                    for incident in self.incidents.values()
                ),
                "critical_incidents": sum(
                    incident.advisory.severity
                    == EmergencySeverity.CRITICAL
                    for incident in self.incidents.values()
                ),
                "resolved_incidents": sum(
                    incident.status
                    == EmergencyIncidentStatus.RESOLVED
                    for incident in self.incidents.values()
                ),
                "closed_incidents": sum(
                    incident.status
                    == EmergencyIncidentStatus.CLOSED
                    for incident in self.incidents.values()
                ),
                "audit_events": len(
                    self.audit_events
                ),
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
                "incidents": [
                    incident.to_dict()
                    for incident in self.incidents.values()
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
    service = RoboticsEmergencyAdvisoryService()

    observation = EmergencyObservation(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        emergency_type=EmergencyType.BATTERY_THERMAL,
        risk_score=96.0,
        description=(
            "Battery temperature rising rapidly "
            "during simulated robot operation"
        ),
        location="Warehouse Zone A",
    )

    incident = service.create_incident(
        observation
    )

    print(
        json.dumps(
            incident.to_dict(),
            indent=2,
        )
    )

    service.acknowledge(
        tenant_id="tenant_alpha",
        incident_id=incident.incident_id,
        operator_id="operator_001",
    )

    service.start_review(
        tenant_id="tenant_alpha",
        incident_id=incident.incident_id,
        reviewer_id="safety_reviewer_001",
    )

    service.resolve(
        tenant_id="tenant_alpha",
        incident_id=incident.incident_id,
        reviewer_id="safety_reviewer_001",
        resolution_notes=(
            "Simulation reviewed; battery "
            "thermal fault condition cleared"
        ),
    )

    service.close(
        tenant_id="tenant_alpha",
        incident_id=incident.incident_id,
        operator_id="operator_001",
    )

    service.export_json(
        "data/security/emergency_advisory_report.json"
    )

    print(
        json.dumps(
            service.statistics(),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

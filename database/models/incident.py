"""
backend/app/db/models/incident.py

Physical AI & Autonomous Robotics Intelligence Platform

SQLAlchemy 2.x incident management models.

Features:
- Multi-tenant incident tracking
- Robot and fleet incidents
- Safety and emergency incidents
- Hardware and software failures
- AI model and agent incidents
- Collision and navigation incidents
- Cybersecurity incidents
- Incident severity and priority
- Investigation and root cause analysis
- Human review and resolution
- Incident timeline and event history
- Corrective and preventive actions
- Incident evidence and attachments
- Audit-friendly lifecycle tracking

Dependencies:
    SQLAlchemy >= 2.0
"""

from __future__ import annotations

import math
import uuid

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKeyConstraint,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
    validates,
)

from app.db.base import Base


# ============================================================
# Helpers
# ============================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_uuid() -> str:
    return str(uuid.uuid4())


def serialize_datetime(
    value: Optional[datetime],
) -> Optional[str]:
    return value.isoformat() if value else None


def validate_finite(
    value: Optional[float],
    field_name: str,
) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, bool):
        raise ValueError(
            f"{field_name} must be numeric"
        )

    number = float(value)

    if not math.isfinite(number):
        raise ValueError(
            f"{field_name} must be finite"
        )

    return number


# ============================================================
# Enumerations
# ============================================================


class IncidentType(str, Enum):
    ROBOT_FAILURE = "robot_failure"
    ROBOT_OFFLINE = "robot_offline"
    HARDWARE_FAILURE = "hardware_failure"
    SOFTWARE_FAILURE = "software_failure"
    FIRMWARE_FAILURE = "firmware_failure"
    SENSOR_FAILURE = "sensor_failure"
    ACTUATOR_FAILURE = "actuator_failure"
    MOTOR_FAILURE = "motor_failure"
    BATTERY_FAILURE = "battery_failure"
    POWER_FAILURE = "power_failure"
    THERMAL_FAILURE = "thermal_failure"
    NETWORK_FAILURE = "network_failure"
    COMMUNICATION_FAILURE = "communication_failure"

    NAVIGATION_FAILURE = "navigation_failure"
    LOCALIZATION_FAILURE = "localization_failure"
    PATH_PLANNING_FAILURE = "path_planning_failure"
    MANIPULATION_FAILURE = "manipulation_failure"
    GRIPPER_FAILURE = "gripper_failure"
    PERCEPTION_FAILURE = "perception_failure"
    CONTROL_FAILURE = "control_failure"

    COLLISION = "collision"
    NEAR_MISS = "near_miss"
    OBSTACLE_COLLISION = "obstacle_collision"
    HUMAN_COLLISION = "human_collision"
    PROPERTY_DAMAGE = "property_damage"
    EQUIPMENT_DAMAGE = "equipment_damage"

    HUMAN_SAFETY = "human_safety"
    WORKER_INJURY = "worker_injury"
    SAFETY_VIOLATION = "safety_violation"
    EMERGENCY_STOP = "emergency_stop"
    GEOFENCE_VIOLATION = "geofence_violation"
    UNAUTHORIZED_OPERATION = "unauthorized_operation"

    AI_MODEL_FAILURE = "ai_model_failure"
    MODEL_DRIFT = "model_drift"
    INFERENCE_FAILURE = "inference_failure"
    AI_AGENT_FAILURE = "ai_agent_failure"
    AGENT_TOOL_FAILURE = "agent_tool_failure"
    VLA_FAILURE = "vla_failure"
    WORLD_MODEL_FAILURE = "world_model_failure"
    POLICY_FAILURE = "policy_failure"
    HALLUCINATION = "hallucination"

    TASK_FAILURE = "task_failure"
    MISSION_FAILURE = "mission_failure"
    FLEET_FAILURE = "fleet_failure"
    SIMULATION_FAILURE = "simulation_failure"

    CYBERSECURITY = "cybersecurity"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH = "data_breach"
    PROMPT_INJECTION = "prompt_injection"
    MALWARE_DETECTED = "malware_detected"

    FIRE = "fire"
    SMOKE = "smoke"
    OVERHEATING = "overheating"
    ENVIRONMENTAL_HAZARD = "environmental_hazard"
    NATURAL_DISASTER = "natural_disaster"

    MAINTENANCE = "maintenance"
    PREDICTIVE_FAILURE = "predictive_failure"
    OPERATIONAL = "operational"
    OTHER = "other"


class IncidentSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class IncidentPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class IncidentStatus(str, Enum):
    REPORTED = "reported"
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    MITIGATING = "mitigating"
    MONITORING = "monitoring"
    RESOLVED = "resolved"
    CLOSED = "closed"
    REOPENED = "reopened"
    FALSE_POSITIVE = "false_positive"


class IncidentSource(str, Enum):
    SYSTEM = "system"
    ROBOT = "robot"
    SENSOR = "sensor"
    TELEMETRY = "telemetry"
    AI_MODEL = "ai_model"
    AI_AGENT = "ai_agent"
    SAFETY_ENGINE = "safety_engine"
    SECURITY_ENGINE = "security_engine"
    MONITORING = "monitoring"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    HUMAN_OPERATOR = "human_operator"
    EXTERNAL_API = "external_api"
    SIMULATION = "simulation"
    UNKNOWN = "unknown"


class IncidentImpact(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    SEVERE = "severe"
    CATASTROPHIC = "catastrophic"


class IncidentEventType(str, Enum):
    CREATED = "created"
    UPDATED = "updated"
    ACKNOWLEDGED = "acknowledged"
    ASSIGNED = "assigned"
    STATUS_CHANGED = "status_changed"
    SEVERITY_CHANGED = "severity_changed"
    PRIORITY_CHANGED = "priority_changed"
    INVESTIGATION_STARTED = "investigation_started"
    ROOT_CAUSE_IDENTIFIED = "root_cause_identified"
    CONTAINMENT_STARTED = "containment_started"
    CONTAINED = "contained"
    MITIGATION_STARTED = "mitigation_started"
    MITIGATION_COMPLETED = "mitigation_completed"
    ESCALATED = "escalated"
    EVIDENCE_ADDED = "evidence_added"
    CORRECTIVE_ACTION_ADDED = "corrective_action_added"
    CORRECTIVE_ACTION_COMPLETED = "corrective_action_completed"
    RESOLVED = "resolved"
    CLOSED = "closed"
    REOPENED = "reopened"
    FALSE_POSITIVE = "false_positive"
    NOTE_ADDED = "note_added"
    DELETED = "deleted"
    RESTORED = "restored"


class IncidentActionType(str, Enum):
    CONTAINMENT = "containment"
    CORRECTIVE = "corrective"
    PREVENTIVE = "preventive"
    INVESTIGATION = "investigation"
    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    REPLACEMENT = "replacement"
    SOFTWARE_PATCH = "software_patch"
    MODEL_ROLLBACK = "model_rollback"
    MODEL_RETRAINING = "model_retraining"
    SAFETY_REVIEW = "safety_review"
    SECURITY_REVIEW = "security_review"
    HUMAN_TRAINING = "human_training"
    OTHER = "other"


class IncidentActionStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class IncidentEvidenceType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    SENSOR_DATA = "sensor_data"
    TELEMETRY = "telemetry"
    LOG = "log"
    TRACE = "trace"
    DOCUMENT = "document"
    REPORT = "report"
    SIMULATION = "simulation"
    MODEL_OUTPUT = "model_output"
    OTHER = "other"


# ============================================================
# Incident Model
# ============================================================


class Incident(Base):
    """
    Main incident record.

    All incident operations must be tenant-scoped.
    Physical robot commands must be handled by a
    separate authorized safety-control subsystem.
    """

    __tablename__ = "incidents"

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "id",
            name="uq_incident_tenant_id",
        ),
        UniqueConstraint(
            "tenant_id",
            "incident_number",
            name="uq_incident_tenant_number",
        ),
        Index(
            "ix_incident_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_incident_tenant_severity",
            "tenant_id",
            "severity",
        ),
        Index(
            "ix_incident_tenant_priority",
            "tenant_id",
            "priority",
        ),
        Index(
            "ix_incident_tenant_type",
            "tenant_id",
            "incident_type",
        ),
        Index(
            "ix_incident_tenant_robot",
            "tenant_id",
            "robot_id",
        ),
        Index(
            "ix_incident_tenant_fleet",
            "tenant_id",
            "fleet_id",
        ),
        Index(
            "ix_incident_tenant_assigned",
            "tenant_id",
            "assigned_to",
        ),
        Index(
            "ix_incident_tenant_occurred",
            "tenant_id",
            "occurred_at",
        ),
        Index(
            "ix_incident_tenant_created",
            "tenant_id",
            "created_at",
        ),
        CheckConstraint(
            "occurrence_count >= 1",
            name="ck_incident_occurrence_positive",
        ),
        CheckConstraint(
            "escalation_level >= 0",
            name="ck_incident_escalation_nonnegative",
        ),
        CheckConstraint(
            "estimated_downtime_minutes IS NULL OR "
            "estimated_downtime_minutes >= 0",
            name="ck_incident_estimated_downtime",
        ),
        CheckConstraint(
            "actual_downtime_minutes IS NULL OR "
            "actual_downtime_minutes >= 0",
            name="ck_incident_actual_downtime",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    tenant_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    organization_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    incident_number: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    robot_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    fleet_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    component_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    sensor_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    model_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    agent_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    task_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    mission_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    alert_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    failure_event_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    maintenance_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    incident_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default=IncidentType.OPERATIONAL.value,
    )

    severity: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=IncidentSeverity.MEDIUM.value,
    )

    priority: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=IncidentPriority.NORMAL.value,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=IncidentStatus.REPORTED.value,
    )

    source: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default=IncidentSource.SYSTEM.value,
    )

    impact: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=IncidentImpact.MODERATE.value,
    )

    source_identifier: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    correlation_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    deduplication_key: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    location: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )

    latitude: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    longitude: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    altitude_m: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    assigned_to: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    reported_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    acknowledged_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    investigated_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    resolved_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    closed_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    escalation_level: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    occurrence_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    affected_robot_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    affected_person_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    estimated_downtime_minutes: Mapped[
        Optional[float]
    ] = mapped_column(
        Float,
        nullable=True,
    )

    actual_downtime_minutes: Mapped[
        Optional[float]
    ] = mapped_column(
        Float,
        nullable=True,
    )

    estimated_cost: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    actual_cost: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    root_cause: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    investigation_summary: Mapped[Optional[str]] = (
        mapped_column(
            Text,
            nullable=True,
        )
    )

    containment_action: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    mitigation_action: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    corrective_action: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    preventive_action: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    resolution: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    lessons_learned: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    requires_immediate_action: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    requires_human_review: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    is_safety_related: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    is_security_related: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    is_reportable: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    impact_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    investigation_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    safety_assessment_json: Mapped[dict[str, Any]] = (
        mapped_column(
            JSON,
            nullable=False,
            default=dict,
        )
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    detected_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    acknowledged_at: Mapped[
        Optional[datetime]
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    investigation_started_at: Mapped[
        Optional[datetime]
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    contained_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    escalated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    resolved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    closed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    events: Mapped[list["IncidentEvent"]] = relationship(
        "IncidentEvent",
        back_populates="incident",
        cascade="save-update, merge",
        lazy="selectin",
    )

    actions: Mapped[list["IncidentAction"]] = relationship(
        "IncidentAction",
        back_populates="incident",
        cascade="save-update, merge",
        lazy="selectin",
    )

    evidence: Mapped[list["IncidentEvidence"]] = relationship(
        "IncidentEvidence",
        back_populates="incident",
        cascade="save-update, merge",
        lazy="selectin",
    )

    @validates(
        "latitude",
        "longitude",
        "altitude_m",
        "estimated_downtime_minutes",
        "actual_downtime_minutes",
        "estimated_cost",
        "actual_cost",
    )
    def validate_numeric(
        self,
        key: str,
        value: Optional[float],
    ) -> Optional[float]:
        number = validate_finite(value, key)

        if number is None:
            return None

        if key == "latitude" and not -90 <= number <= 90:
            raise ValueError(
                "latitude must be between -90 and 90"
            )

        if key == "longitude" and not -180 <= number <= 180:
            raise ValueError(
                "longitude must be between -180 and 180"
            )

        if key in {
            "estimated_downtime_minutes",
            "actual_downtime_minutes",
            "estimated_cost",
            "actual_cost",
        } and number < 0:
            raise ValueError(
                f"{key} cannot be negative"
            )

        return number

    @property
    def is_open(self) -> bool:
        return (
            not self.is_deleted
            and self.status not in {
                IncidentStatus.RESOLVED.value,
                IncidentStatus.CLOSED.value,
                IncidentStatus.FALSE_POSITIVE.value,
            }
        )

    @property
    def is_critical(self) -> bool:
        return self.severity in {
            IncidentSeverity.CRITICAL.value,
            IncidentSeverity.EMERGENCY.value,
        }

    @property
    def is_resolved(self) -> bool:
        return self.status in {
            IncidentStatus.RESOLVED.value,
            IncidentStatus.CLOSED.value,
        }

    def acknowledge(self, user_id: str) -> None:
        if self.status not in {
            IncidentStatus.REPORTED.value,
            IncidentStatus.OPEN.value,
            IncidentStatus.REOPENED.value,
        }:
            raise ValueError(
                "Incident cannot be acknowledged"
            )

        now = utc_now()

        self.acknowledged_by = user_id
        self.acknowledged_at = now
        self.status = IncidentStatus.ACKNOWLEDGED.value
        self.updated_at = now

    def start_investigation(
        self,
        user_id: str,
    ) -> None:
        if not self.is_open:
            raise ValueError(
                "Cannot investigate a closed incident"
            )

        now = utc_now()

        self.investigated_by = user_id
        self.investigation_started_at = now
        self.status = IncidentStatus.INVESTIGATING.value
        self.updated_at = now

    def mark_contained(
        self,
        action: str,
    ) -> None:
        if not self.is_open:
            raise ValueError(
                "Cannot contain a closed incident"
            )

        if not action.strip():
            raise ValueError(
                "Containment action is required"
            )

        now = utc_now()

        self.containment_action = action.strip()
        self.contained_at = now
        self.status = IncidentStatus.CONTAINED.value
        self.updated_at = now

    def escalate(self) -> None:
        if not self.is_open:
            raise ValueError(
                "Cannot escalate a closed incident"
            )

        self.escalation_level += 1
        self.escalated_at = utc_now()
        self.updated_at = utc_now()

    def resolve(
        self,
        user_id: str,
        resolution: str,
    ) -> None:
        if not self.is_open:
            raise ValueError(
                "Incident is already resolved or closed"
            )

        if not resolution.strip():
            raise ValueError(
                "Resolution cannot be empty"
            )

        now = utc_now()

        self.resolved_by = user_id
        self.resolution = resolution.strip()
        self.resolved_at = now
        self.status = IncidentStatus.RESOLVED.value
        self.updated_at = now

    def close(self, user_id: str) -> None:
        if self.status != IncidentStatus.RESOLVED.value:
            raise ValueError(
                "Incident must be resolved before closing"
            )

        now = utc_now()

        self.closed_by = user_id
        self.closed_at = now
        self.status = IncidentStatus.CLOSED.value
        self.updated_at = now

    def reopen(self) -> None:
        if self.status not in {
            IncidentStatus.RESOLVED.value,
            IncidentStatus.CLOSED.value,
        }:
            raise ValueError(
                "Only resolved or closed incidents can reopen"
            )

        self.status = IncidentStatus.REOPENED.value
        self.resolved_at = None
        self.closed_at = None
        self.updated_at = utc_now()

    def increment_occurrence(self) -> None:
        self.occurrence_count += 1
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "organization_id": self.organization_id,
            "incident_number": self.incident_number,
            "robot_id": self.robot_id,
            "fleet_id": self.fleet_id,
            "component_id": self.component_id,
            "sensor_id": self.sensor_id,
            "model_id": self.model_id,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "mission_id": self.mission_id,
            "alert_id": self.alert_id,
            "failure_event_id": self.failure_event_id,
            "maintenance_id": self.maintenance_id,
            "title": self.title,
            "description": self.description,
            "incident_type": self.incident_type,
            "severity": self.severity,
            "priority": self.priority,
            "status": self.status,
            "source": self.source,
            "impact": self.impact,
            "location": self.location,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "assigned_to": self.assigned_to,
            "reported_by": self.reported_by,
            "escalation_level": self.escalation_level,
            "occurrence_count": self.occurrence_count,
            "affected_robot_count": self.affected_robot_count,
            "affected_person_count": self.affected_person_count,
            "estimated_downtime_minutes": (
                self.estimated_downtime_minutes
            ),
            "actual_downtime_minutes": (
                self.actual_downtime_minutes
            ),
            "estimated_cost": self.estimated_cost,
            "actual_cost": self.actual_cost,
            "root_cause": self.root_cause,
            "investigation_summary": (
                self.investigation_summary
            ),
            "containment_action": self.containment_action,
            "mitigation_action": self.mitigation_action,
            "corrective_action": self.corrective_action,
            "preventive_action": self.preventive_action,
            "resolution": self.resolution,
            "lessons_learned": self.lessons_learned,
            "requires_immediate_action": (
                self.requires_immediate_action
            ),
            "requires_human_review": (
                self.requires_human_review
            ),
            "is_safety_related": self.is_safety_related,
            "is_security_related": self.is_security_related,
            "is_reportable": self.is_reportable,
            "is_deleted": self.is_deleted,
            "impact_details": self.impact_json,
            "investigation_details": self.investigation_json,
            "safety_assessment": self.safety_assessment_json,
            "metadata": self.metadata_json,
            "occurred_at": serialize_datetime(
                self.occurred_at
            ),
            "detected_at": serialize_datetime(
                self.detected_at
            ),
            "acknowledged_at": serialize_datetime(
                self.acknowledged_at
            ),
            "resolved_at": serialize_datetime(
                self.resolved_at
            ),
            "closed_at": serialize_datetime(
                self.closed_at
            ),
            "created_at": serialize_datetime(
                self.created_at
            ),
            "updated_at": serialize_datetime(
                self.updated_at
            ),
        }

    def __repr__(self) -> str:
        return (
            f"<Incident("
            f"id={self.id!r}, "
            f"incident_number={self.incident_number!r}, "
            f"severity={self.severity!r}, "
            f"status={self.status!r}"
            f")>"
        )


# ============================================================
# Incident Event
# ============================================================


class IncidentEvent(Base):
    """
    Append-only incident timeline.

    Events should be inserted in the same transaction
    as their associated incident state changes.
    """

    __tablename__ = "incident_events"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "incident_id"],
            ["incidents.tenant_id", "incidents.id"],
            name="fk_incident_event_tenant_incident",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_incident_event_tenant_incident",
            "tenant_id",
            "incident_id",
        ),
        Index(
            "ix_incident_event_tenant_type",
            "tenant_id",
            "event_type",
        ),
        Index(
            "ix_incident_event_tenant_created",
            "tenant_id",
            "created_at",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    tenant_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    incident_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    event_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    actor_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    previous_status: Mapped[Optional[str]] = mapped_column(
        String(32),
        nullable=True,
    )

    new_status: Mapped[Optional[str]] = mapped_column(
        String(32),
        nullable=True,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    incident: Mapped["Incident"] = relationship(
        "Incident",
        back_populates="events",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "incident_id": self.incident_id,
            "event_type": self.event_type,
            "message": self.message,
            "actor_id": self.actor_id,
            "previous_status": self.previous_status,
            "new_status": self.new_status,
            "metadata": self.metadata_json,
            "created_at": serialize_datetime(
                self.created_at
            ),
        }


# ============================================================
# Incident Corrective / Preventive Action
# ============================================================


class IncidentAction(Base):
    """
    Tracks investigation, containment, corrective
    and preventive actions.

    Actions are records of authorized work.
    They do not execute physical robot commands.
    """

    __tablename__ = "incident_actions"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "incident_id"],
            ["incidents.tenant_id", "incidents.id"],
            name="fk_incident_action_tenant_incident",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_incident_action_tenant_incident",
            "tenant_id",
            "incident_id",
        ),
        Index(
            "ix_incident_action_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_incident_action_tenant_assigned",
            "tenant_id",
            "assigned_to",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    tenant_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    incident_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    action_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default=IncidentActionType.CORRECTIVE.value,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=IncidentActionStatus.PENDING.value,
    )

    assigned_to: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    created_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    completed_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    due_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completion_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    incident: Mapped["Incident"] = relationship(
        "Incident",
        back_populates="actions",
    )

    def start(self) -> None:
        if self.status != IncidentActionStatus.PENDING.value:
            raise ValueError(
                "Only pending actions can start"
            )

        self.status = IncidentActionStatus.IN_PROGRESS.value
        self.started_at = utc_now()
        self.updated_at = utc_now()

    def complete(
        self,
        user_id: str,
        *,
        notes: Optional[str] = None,
    ) -> None:
        if self.status != IncidentActionStatus.IN_PROGRESS.value:
            raise ValueError(
                "Only in-progress actions can complete"
            )

        now = utc_now()

        self.status = IncidentActionStatus.COMPLETED.value
        self.completed_by = user_id
        self.completed_at = now
        self.completion_notes = notes
        self.updated_at = now

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "incident_id": self.incident_id,
            "title": self.title,
            "description": self.description,
            "action_type": self.action_type,
            "status": self.status,
            "assigned_to": self.assigned_to,
            "created_by": self.created_by,
            "completed_by": self.completed_by,
            "due_at": serialize_datetime(self.due_at),
            "started_at": serialize_datetime(self.started_at),
            "completed_at": serialize_datetime(
                self.completed_at
            ),
            "completion_notes": self.completion_notes,
            "metadata": self.metadata_json,
        }


# ============================================================
# Incident Evidence
# ============================================================


class IncidentEvidence(Base):
    """
    Evidence metadata for incident investigation.

    Store large files in protected object storage.
    Evidence URIs must not contain credentials or
    long-lived signed access tokens.
    """

    __tablename__ = "incident_evidence"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "incident_id"],
            ["incidents.tenant_id", "incidents.id"],
            name="fk_incident_evidence_tenant_incident",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_incident_evidence_tenant_incident",
            "tenant_id",
            "incident_id",
        ),
        Index(
            "ix_incident_evidence_tenant_type",
            "tenant_id",
            "evidence_type",
        ),
        CheckConstraint(
            "file_size_bytes IS NULL OR file_size_bytes >= 0",
            name="ck_incident_evidence_file_size",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
    )

    tenant_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    incident_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    evidence_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    evidence_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default=IncidentEvidenceType.OTHER.value,
    )

    evidence_uri: Mapped[str] = mapped_column(
        String(2048),
        nullable=False,
    )

    content_type: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    file_size_bytes: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    checksum_sha256: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    collected_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    collected_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    incident: Mapped["Incident"] = relationship(
        "Incident",
        back_populates="evidence",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "incident_id": self.incident_id,
            "evidence_name": self.evidence_name,
            "evidence_type": self.evidence_type,
            "evidence_uri": self.evidence_uri,
            "content_type": self.content_type,
            "file_size_bytes": self.file_size_bytes,
            "checksum_sha256": self.checksum_sha256,
            "description": self.description,
            "collected_by": self.collected_by,
            "collected_at": serialize_datetime(
                self.collected_at
            ),
            "metadata": self.metadata_json,
            "created_at": serialize_datetime(
                self.created_at
            ),
        }


# ============================================================
# Exports
# ============================================================


__all__ = [
    "Incident",
    "IncidentEvent",
    "IncidentAction",
    "IncidentEvidence",
    "IncidentType",
    "IncidentSeverity",
    "IncidentPriority",
    "IncidentStatus",
    "IncidentSource",
    "IncidentImpact",
    "IncidentEventType",
    "IncidentActionType",
    "IncidentActionStatus",
    "IncidentEvidenceType",
    "utc_now",
    "generate_uuid",
    "serialize_datetime",
    "validate_finite",
]
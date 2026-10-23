"""Failure Event module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/db/models/failure_event.py

Physical AI & Autonomous Robotics Intelligence Platform

SQLAlchemy 2.x ORM models for:
- Robot and component failure events
- Sensor, actuator, motor and battery failures
- AI model and agent failures
- Navigation and manipulation failures
- Predictive maintenance failure analysis
- Failure detection and diagnosis
- Root cause analysis
- Failure severity and impact assessment
- Remaining Useful Life (RUL) tracking
- Repair and replacement recommendations
- Failure event history
- Human review and verification

Database: PostgreSQL recommended.
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


def serialize_datetime(
    value: Optional[datetime],
) -> Optional[str]:
    return value.isoformat() if value else None


# ============================================================
# Enumerations
# ============================================================


class FailureType(str, Enum):
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"
    ELECTRONIC = "electronic"
    SOFTWARE = "software"
    FIRMWARE = "firmware"
    HARDWARE = "hardware"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    MOTOR = "motor"
    GEARBOX = "gearbox"
    BEARING = "bearing"
    JOINT = "joint"
    GRIPPER = "gripper"
    BATTERY = "battery"
    POWER_SYSTEM = "power_system"
    CHARGING_SYSTEM = "charging_system"
    THERMAL = "thermal"
    VIBRATION = "vibration"
    HYDRAULIC = "hydraulic"
    PNEUMATIC = "pneumatic"
    CAMERA = "camera"
    LIDAR = "lidar"
    RADAR = "radar"
    IMU = "imu"
    GPS = "gps"
    NETWORK = "network"
    COMMUNICATION = "communication"
    COMPUTE = "compute"
    MEMORY = "memory"
    STORAGE = "storage"
    NAVIGATION = "navigation"
    LOCALIZATION = "localization"
    PATH_PLANNING = "path_planning"
    MANIPULATION = "manipulation"
    PERCEPTION = "perception"
    CONTROL_SYSTEM = "control_system"
    AI_MODEL = "ai_model"
    AI_AGENT = "ai_agent"
    VLA = "vla"
    WORLD_MODEL = "world_model"
    POLICY = "policy"
    TASK_EXECUTION = "task_execution"
    SAFETY_SYSTEM = "safety_system"
    COLLISION = "collision"
    ENVIRONMENTAL = "environmental"
    CYBERSECURITY = "cybersecurity"
    UNKNOWN = "unknown"
    OTHER = "other"


class FailureSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


class FailureStatus(str, Enum):
    DETECTED = "detected"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    DIAGNOSED = "diagnosed"
    REPAIR_SCHEDULED = "repair_scheduled"
    REPAIRING = "repairing"
    MONITORING = "monitoring"
    RESOLVED = "resolved"
    CLOSED = "closed"
    FALSE_POSITIVE = "false_positive"


class FailureDetectionMethod(str, Enum):
    SENSOR_THRESHOLD = "sensor_threshold"
    ANOMALY_DETECTION = "anomaly_detection"
    PREDICTIVE_MODEL = "predictive_model"
    COMPUTER_VISION = "computer_vision"
    VIBRATION_ANALYSIS = "vibration_analysis"
    THERMAL_ANALYSIS = "thermal_analysis"
    CURRENT_ANALYSIS = "current_analysis"
    VOLTAGE_ANALYSIS = "voltage_analysis"
    RUL_PREDICTION = "rul_prediction"
    HEALTH_MONITORING = "health_monitoring"
    DIAGNOSTIC_CODE = "diagnostic_code"
    RULE_ENGINE = "rule_engine"
    AI_AGENT = "ai_agent"
    HUMAN_INSPECTION = "human_inspection"
    MAINTENANCE_INSPECTION = "maintenance_inspection"
    SYSTEM_MONITORING = "system_monitoring"
    SIMULATION = "simulation"
    EXTERNAL_SYSTEM = "external_system"
    UNKNOWN = "unknown"


class FailureImpact(str, Enum):
    NONE = "none"
    MINOR = "minor"
    DEGRADED_PERFORMANCE = "degraded_performance"
    PARTIAL_OUTAGE = "partial_outage"
    FULL_OUTAGE = "full_outage"
    SAFETY_RISK = "safety_risk"
    EMERGENCY = "emergency"


class FailureEventType(str, Enum):
    CREATED = "created"
    UPDATED = "updated"
    DETECTED = "detected"
    ACKNOWLEDGED = "acknowledged"
    ASSIGNED = "assigned"
    STATUS_CHANGED = "status_changed"
    SEVERITY_CHANGED = "severity_changed"
    DIAGNOSIS_ADDED = "diagnosis_added"
    ROOT_CAUSE_IDENTIFIED = "root_cause_identified"
    RUL_UPDATED = "rul_updated"
    REPAIR_RECOMMENDED = "repair_recommended"
    REPLACEMENT_RECOMMENDED = "replacement_recommended"
    MAINTENANCE_LINKED = "maintenance_linked"
    REPAIR_STARTED = "repair_started"
    REPAIR_COMPLETED = "repair_completed"
    RESOLVED = "resolved"
    CLOSED = "closed"
    REOPENED = "reopened"
    FALSE_POSITIVE = "false_positive"
    NOTE_ADDED = "note_added"
    DELETED = "deleted"
    RESTORED = "restored"


class FailureReviewStatus(str, Enum):
    NOT_REVIEWED = "not_reviewed"
    PENDING = "pending"
    IN_REVIEW = "in_review"
    VERIFIED = "verified"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"


# ============================================================
# Main Failure Event Model
# ============================================================


class FailureEvent(Base):
    """
    Represents a detected or predicted robot failure.

    All records are tenant-scoped.

    This model stores failure intelligence only.
    It does not directly control robot hardware.
    """

    __tablename__ = "failure_events"

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "id",
            name="uq_failure_event_tenant_id",
        ),
        UniqueConstraint(
            "tenant_id",
            "failure_number",
            name="uq_failure_event_number",
        ),
        Index(
            "ix_failure_tenant_robot",
            "tenant_id",
            "robot_id",
        ),
        Index(
            "ix_failure_tenant_component",
            "tenant_id",
            "component_id",
        ),
        Index(
            "ix_failure_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_failure_tenant_severity",
            "tenant_id",
            "severity",
        ),
        Index(
            "ix_failure_tenant_type",
            "tenant_id",
            "failure_type",
        ),
        Index(
            "ix_failure_tenant_detected",
            "tenant_id",
            "detected_at",
        ),
        Index(
            "ix_failure_tenant_created",
            "tenant_id",
            "created_at",
        ),
        Index(
            "ix_failure_tenant_model",
            "tenant_id",
            "model_id",
        ),
        CheckConstraint(
            "confidence_score IS NULL OR "
            "(confidence_score >= 0 AND confidence_score <= 1)",
            name="ck_failure_confidence_range",
        ),
        CheckConstraint(
            "health_score IS NULL OR "
            "(health_score >= 0 AND health_score <= 100)",
            name="ck_failure_health_score_range",
        ),
        CheckConstraint(
            "rul_hours IS NULL OR rul_hours >= 0",
            name="ck_failure_rul_nonnegative",
        ),
        CheckConstraint(
            "estimated_downtime_minutes IS NULL OR "
            "estimated_downtime_minutes >= 0",
            name="ck_failure_downtime_nonnegative",
        ),
        CheckConstraint(
            "occurrence_count >= 1",
            name="ck_failure_occurrence_positive",
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

    failure_number: Mapped[str] = mapped_column(
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

    part_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    model_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    model_version_id: Mapped[Optional[str]] = mapped_column(
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

    incident_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    alert_id: Mapped[Optional[str]] = mapped_column(
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

    failure_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default=FailureType.UNKNOWN.value,
    )

    severity: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=FailureSeverity.MEDIUM.value,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=FailureStatus.DETECTED.value,
    )

    impact: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default=FailureImpact.DEGRADED_PERFORMANCE.value,
    )

    detection_method: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default=FailureDetectionMethod.UNKNOWN.value,
    )

    failure_code: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    diagnostic_code: Mapped[Optional[str]] = mapped_column(
        String(100),
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

    source_identifier: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    confidence_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    health_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    rul_hours: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    failure_probability: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    anomaly_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    estimated_repair_cost: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    estimated_replacement_cost: Mapped[Optional[float]] = (
        mapped_column(
            Float,
            nullable=True,
        )
    )

    actual_repair_cost: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
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

    metric_name: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    metric_value: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    threshold_value: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    metric_unit: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )

    root_cause: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    diagnosis: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    recommended_action: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    repair_recommendation: Mapped[Optional[str]] = (
        mapped_column(
            Text,
            nullable=True,
        )
    )

    replacement_recommendation: Mapped[Optional[str]] = (
        mapped_column(
            Text,
            nullable=True,
        )
    )

    resolution: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    assigned_to: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    acknowledged_by: Mapped[Optional[str]] = mapped_column(
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

    occurrence_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    is_predicted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    is_safety_related: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
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

    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    review_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=FailureReviewStatus.NOT_REVIEWED.value,
    )

    reviewed_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    review_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    sensor_readings_json: Mapped[dict[str, Any]] = (
        mapped_column(
            JSON,
            nullable=False,
            default=dict,
        )
    )

    prediction_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    diagnosis_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    cost_breakdown_json: Mapped[dict[str, Any]] = (
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

    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    predicted_failure_at: Mapped[
        Optional[datetime]
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    first_occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    last_occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    acknowledged_at: Mapped[
        Optional[datetime]
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    diagnosed_at: Mapped[
        Optional[datetime]
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    repair_started_at: Mapped[
        Optional[datetime]
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    repair_completed_at: Mapped[
        Optional[datetime]
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    resolved_at: Mapped[
        Optional[datetime]
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    closed_at: Mapped[
        Optional[datetime]
    ] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    reviewed_at: Mapped[
        Optional[datetime]
    ] = mapped_column(
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

    history: Mapped[list["FailureEventHistory"]] = (
        relationship(
            "FailureEventHistory",
            back_populates="failure",
            cascade="save-update, merge",
            lazy="selectin",
        )
    )

    @validates(
        "confidence_score",
        "health_score",
        "rul_hours",
        "failure_probability",
        "anomaly_score",
        "estimated_repair_cost",
        "estimated_replacement_cost",
        "actual_repair_cost",
        "estimated_downtime_minutes",
        "actual_downtime_minutes",
        "metric_value",
        "threshold_value",
    )
    def validate_numeric(
        self,
        key: str,
        value: Optional[float],
    ) -> Optional[float]:
        number = validate_finite(
            value,
            key,
        )

        if number is None:
            return None

        if key in {
            "confidence_score",
            "failure_probability",
        } and not 0 <= number <= 1:
            raise ValueError(
                f"{key} must be between 0 and 1"
            )

        if (
            key == "health_score"
            and not 0 <= number <= 100
        ):
            raise ValueError(
                "health_score must be between 0 and 100"
            )

        if key in {
            "rul_hours",
            "estimated_repair_cost",
            "estimated_replacement_cost",
            "actual_repair_cost",
            "estimated_downtime_minutes",
            "actual_downtime_minutes",
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
                FailureStatus.RESOLVED.value,
                FailureStatus.CLOSED.value,
                FailureStatus.FALSE_POSITIVE.value,
            }
        )

    @property
    def is_critical(self) -> bool:
        return self.severity in {
            FailureSeverity.CRITICAL.value,
            FailureSeverity.CATASTROPHIC.value,
        }

    @property
    def is_resolved(self) -> bool:
        return self.status in {
            FailureStatus.RESOLVED.value,
            FailureStatus.CLOSED.value,
        }

    @property
    def is_verified(self) -> bool:
        return (
            self.review_status
            == FailureReviewStatus.VERIFIED.value
        )

    def acknowledge(
        self,
        user_id: str,
    ) -> None:
        if not self.is_open:
            raise ValueError(
                "Cannot acknowledge a closed failure"
            )

        now = utc_now()

        self.acknowledged_by = user_id
        self.acknowledged_at = now
        self.status = FailureStatus.ACKNOWLEDGED.value
        self.updated_at = now

    def add_diagnosis(
        self,
        diagnosis: str,
        *,
        root_cause: Optional[str] = None,
    ) -> None:
        if not self.is_open:
            raise ValueError(
                "Cannot diagnose a closed failure"
            )

        if not diagnosis.strip():
            raise ValueError(
                "Diagnosis cannot be empty"
            )

        now = utc_now()

        self.diagnosis = diagnosis.strip()

        if root_cause is not None:
            self.root_cause = root_cause.strip()

        self.diagnosed_at = now
        self.status = FailureStatus.DIAGNOSED.value
        self.updated_at = now

    def start_repair(self) -> None:
        if self.status not in {
            FailureStatus.DIAGNOSED.value,
            FailureStatus.REPAIR_SCHEDULED.value,
        }:
            raise ValueError(
                "Failure must be diagnosed before repair"
            )

        now = utc_now()

        self.status = FailureStatus.REPAIRING.value
        self.repair_started_at = now
        self.updated_at = now

    def complete_repair(
        self,
        *,
        actual_cost: Optional[float] = None,
    ) -> None:
        if self.status != FailureStatus.REPAIRING.value:
            raise ValueError(
                "Failure is not currently under repair"
            )

        now = utc_now()

        if actual_cost is not None:
            self.actual_repair_cost = actual_cost

        self.repair_completed_at = now
        self.status = FailureStatus.MONITORING.value
        self.updated_at = now

    def resolve(
        self,
        user_id: str,
        resolution: str,
    ) -> None:
        if not self.is_open:
            raise ValueError(
                "Failure is already closed"
            )

        if not resolution.strip():
            raise ValueError(
                "Resolution cannot be empty"
            )

        now = utc_now()

        self.resolved_by = user_id
        self.resolution = resolution.strip()
        self.resolved_at = now
        self.status = FailureStatus.RESOLVED.value
        self.updated_at = now

    def close(
        self,
        user_id: str,
    ) -> None:
        if self.status != FailureStatus.RESOLVED.value:
            raise ValueError(
                "Failure must be resolved before closing"
            )

        now = utc_now()

        self.closed_by = user_id
        self.closed_at = now
        self.status = FailureStatus.CLOSED.value
        self.updated_at = now

    def mark_false_positive(
        self,
        reason: str,
    ) -> None:
        if self.status in {
            FailureStatus.RESOLVED.value,
            FailureStatus.CLOSED.value,
        }:
            raise ValueError(
                "Cannot mark a resolved failure as false positive"
            )

        if not reason.strip():
            raise ValueError(
                "False-positive reason is required"
            )

        now = utc_now()

        self.status = FailureStatus.FALSE_POSITIVE.value
        self.resolution = reason.strip()
        self.resolved_at = now
        self.updated_at = now

    def increment_occurrence(self) -> None:
        now = utc_now()

        self.occurrence_count += 1
        self.last_occurred_at = now
        self.updated_at = now

    def update_rul(
        self,
        rul_hours: float,
        *,
        failure_probability: Optional[float] = None,
    ) -> None:
        self.rul_hours = rul_hours

        if failure_probability is not None:
            self.failure_probability = failure_probability

        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "organization_id": self.organization_id,
            "failure_number": self.failure_number,
            "robot_id": self.robot_id,
            "fleet_id": self.fleet_id,
            "component_id": self.component_id,
            "sensor_id": self.sensor_id,
            "part_id": self.part_id,
            "model_id": self.model_id,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "incident_id": self.incident_id,
            "alert_id": self.alert_id,
            "maintenance_id": self.maintenance_id,
            "title": self.title,
            "description": self.description,
            "failure_type": self.failure_type,
            "severity": self.severity,
            "status": self.status,
            "impact": self.impact,
            "detection_method": self.detection_method,
            "failure_code": self.failure_code,
            "diagnostic_code": self.diagnostic_code,
            "confidence_score": self.confidence_score,
            "health_score": self.health_score,
            "rul_hours": self.rul_hours,
            "failure_probability": self.failure_probability,
            "anomaly_score": self.anomaly_score,
            "estimated_repair_cost": self.estimated_repair_cost,
            "estimated_replacement_cost": (
                self.estimated_replacement_cost
            ),
            "actual_repair_cost": self.actual_repair_cost,
            "estimated_downtime_minutes": (
                self.estimated_downtime_minutes
            ),
            "actual_downtime_minutes": (
                self.actual_downtime_minutes
            ),
            "root_cause": self.root_cause,
            "diagnosis": self.diagnosis,
            "recommended_action": self.recommended_action,
            "repair_recommendation": (
                self.repair_recommendation
            ),
            "replacement_recommendation": (
                self.replacement_recommendation
            ),
            "resolution": self.resolution,
            "assigned_to": self.assigned_to,
            "occurrence_count": self.occurrence_count,
            "is_predicted": self.is_predicted,
            "is_safety_related": self.is_safety_related,
            "requires_immediate_action": (
                self.requires_immediate_action
            ),
            "requires_human_review": (
                self.requires_human_review
            ),
            "review_status": self.review_status,
            "is_deleted": self.is_deleted,
            "sensor_readings": self.sensor_readings_json,
            "prediction": self.prediction_json,
            "diagnosis_details": self.diagnosis_json,
            "cost_breakdown": self.cost_breakdown_json,
            "metadata": self.metadata_json,
            "detected_at": serialize_datetime(
                self.detected_at
            ),
            "predicted_failure_at": serialize_datetime(
                self.predicted_failure_at
            ),
            "acknowledged_at": serialize_datetime(
                self.acknowledged_at
            ),
            "diagnosed_at": serialize_datetime(
                self.diagnosed_at
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
            f"<FailureEvent("
            f"id={self.id!r}, "
            f"robot_id={self.robot_id!r}, "
            f"failure_type={self.failure_type!r}, "
            f"severity={self.severity!r}, "
            f"status={self.status!r}"
            f")>"
        )


# ============================================================
# Failure Event History
# ============================================================


class FailureEventHistory(Base):
    """
    Append-only failure lifecycle event history.

    Record lifecycle events in the same transaction
    as the corresponding FailureEvent changes.
    """

    __tablename__ = "failure_event_history"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "failure_event_id"],
            [
                "failure_events.tenant_id",
                "failure_events.id",
            ],
            name="fk_failure_history_tenant_failure",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_failure_history_tenant_failure",
            "tenant_id",
            "failure_event_id",
        ),
        Index(
            "ix_failure_history_tenant_type",
            "tenant_id",
            "event_type",
        ),
        Index(
            "ix_failure_history_tenant_created",
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

    failure_event_id: Mapped[str] = mapped_column(
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

    failure: Mapped["FailureEvent"] = relationship(
        "FailureEvent",
        back_populates="history",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "failure_event_id": self.failure_event_id,
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

    def __repr__(self) -> str:
        return (
            f"<FailureEventHistory("
            f"id={self.id!r}, "
            f"failure_event_id={self.failure_event_id!r}, "
            f"event_type={self.event_type!r}"
            f")>"
        )


# ============================================================
# Exports
# ============================================================


__all__ = [
    "FailureEvent",
    "FailureEventHistory",
    "FailureType",
    "FailureSeverity",
    "FailureStatus",
    "FailureDetectionMethod",
    "FailureImpact",
    "FailureEventType",
    "FailureReviewStatus",
    "utc_now",
    "generate_uuid",
    "validate_finite",
    "serialize_datetime",
]
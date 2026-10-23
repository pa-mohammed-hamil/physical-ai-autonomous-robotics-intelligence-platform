"""Alert module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/db/models/alert.py

Physical AI & Autonomous Robotics Intelligence Platform

SQLAlchemy 2.x ORM models for:
- Robot alerts
- Safety alerts
- Predictive maintenance alerts
- Sensor and telemetry alerts
- AI model alerts
- Alert acknowledgements
- Alert escalation
- Alert resolution
- Alert event history

Compatible with PostgreSQL and SQLAlchemy 2.x.
"""

from __future__ import annotations

import uuid

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
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
)

from app.db.base import Base


# ============================================================
# Helpers
# ============================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_uuid() -> str:
    return str(uuid.uuid4())


# ============================================================
# Enumerations
# ============================================================


class AlertType(str, Enum):
    SYSTEM = "system"
    ROBOT_HEALTH = "robot_health"
    ROBOT_OFFLINE = "robot_offline"
    BATTERY_LOW = "battery_low"
    BATTERY_CRITICAL = "battery_critical"
    OVERHEATING = "overheating"
    MOTOR_FAILURE = "motor_failure"
    ACTUATOR_FAILURE = "actuator_failure"
    SENSOR_FAILURE = "sensor_failure"
    CAMERA_FAILURE = "camera_failure"
    LIDAR_FAILURE = "lidar_failure"
    GPS_FAILURE = "gps_failure"
    NETWORK_FAILURE = "network_failure"
    POWER_FAILURE = "power_failure"
    HARDWARE_FAILURE = "hardware_failure"
    SOFTWARE_FAILURE = "software_failure"
    FIRMWARE_FAILURE = "firmware_failure"
    NAVIGATION_FAILURE = "navigation_failure"
    LOCALIZATION_FAILURE = "localization_failure"
    PATH_BLOCKED = "path_blocked"
    OBSTACLE_DETECTED = "obstacle_detected"
    COLLISION_RISK = "collision_risk"
    COLLISION_DETECTED = "collision_detected"
    HUMAN_DETECTED = "human_detected"
    HUMAN_SAFETY = "human_safety"
    EMERGENCY_STOP = "emergency_stop"
    SAFETY_VIOLATION = "safety_violation"
    GEOFENCE_VIOLATION = "geofence_violation"
    MANIPULATION_FAILURE = "manipulation_failure"
    GRIPPER_FAILURE = "gripper_failure"
    TASK_FAILURE = "task_failure"
    MISSION_FAILURE = "mission_failure"
    AI_MODEL_FAILURE = "ai_model_failure"
    MODEL_DRIFT = "model_drift"
    INFERENCE_FAILURE = "inference_failure"
    HIGH_LATENCY = "high_latency"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    REMAINING_USEFUL_LIFE = "remaining_useful_life"
    ANOMALY_DETECTED = "anomaly_detected"
    VIBRATION_ANOMALY = "vibration_anomaly"
    TEMPERATURE_ANOMALY = "temperature_anomaly"
    CYBERSECURITY = "cybersecurity"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    ENVIRONMENTAL_HAZARD = "environmental_hazard"
    FIRE_DETECTED = "fire_detected"
    SMOKE_DETECTED = "smoke_detected"
    EMERGENCY = "emergency"
    MAINTENANCE_DUE = "maintenance_due"
    COMPONENT_REPLACEMENT = "component_replacement"
    INVENTORY_LOW = "inventory_low"
    OTHER = "other"


class AlertSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"
    SUPPRESSED = "suppressed"


class AlertSource(str, Enum):
    SYSTEM = "system"
    ROBOT = "robot"
    SENSOR = "sensor"
    TELEMETRY = "telemetry"
    AI_MODEL = "ai_model"
    AGENT = "agent"
    SAFETY_ENGINE = "safety_engine"
    MAINTENANCE_ENGINE = "maintenance_engine"
    PREDICTIVE_MODEL = "predictive_model"
    USER = "user"
    EXTERNAL_API = "external_api"
    MONITORING = "monitoring"
    SECURITY = "security"
    SIMULATION = "simulation"


class AlertEventType(str, Enum):
    CREATED = "created"
    UPDATED = "updated"
    ACKNOWLEDGED = "acknowledged"
    ASSIGNED = "assigned"
    ESCALATED = "escalated"
    STATUS_CHANGED = "status_changed"
    SEVERITY_CHANGED = "severity_changed"
    PRIORITY_CHANGED = "priority_changed"
    SUPPRESSED = "suppressed"
    UNSUPPRESSED = "unsuppressed"
    RESOLVED = "resolved"
    REOPENED = "reopened"
    CLOSED = "closed"
    NOTIFICATION_SENT = "notification_sent"
    NOTIFICATION_FAILED = "notification_failed"
    NOTE_ADDED = "note_added"
    DELETED = "deleted"
    RESTORED = "restored"


class AlertNotificationChannel(str, Enum):
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"
    SLACK = "slack"
    TEAMS = "teams"
    VOICE_CALL = "voice_call"


class AlertNotificationStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================
# Alert Model
# ============================================================


class Alert(Base):
    """
    Main robot and AI platform alert record.

    Multi-tenant isolation:
        Every alert belongs to one tenant.

    Safety:
        Alert records are informational and do not
        directly actuate robot hardware.
    """

    __tablename__ = "alerts"

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "alert_number",
            name="uq_alert_tenant_number",
        ),
        UniqueConstraint(
            "tenant_id",
            "id",
            name="uq_alert_tenant_id",
        ),
        Index(
            "ix_alert_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_alert_tenant_severity",
            "tenant_id",
            "severity",
        ),
        Index(
            "ix_alert_tenant_robot",
            "tenant_id",
            "robot_id",
        ),
        Index(
            "ix_alert_tenant_fleet",
            "tenant_id",
            "fleet_id",
        ),
        Index(
            "ix_alert_tenant_type",
            "tenant_id",
            "alert_type",
        ),
        Index(
            "ix_alert_tenant_detected",
            "tenant_id",
            "detected_at",
        ),
        Index(
            "ix_alert_tenant_created",
            "tenant_id",
            "created_at",
        ),
        Index(
            "ix_alert_deduplication",
            "tenant_id",
            "deduplication_key",
        ),
        CheckConstraint(
            "occurrence_count >= 1",
            name="ck_alert_occurrence_positive",
        ),
        CheckConstraint(
            "escalation_level >= 0",
            name="ck_alert_escalation_nonnegative",
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

    alert_number: Mapped[str] = mapped_column(
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

    task_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    incident_id: Mapped[Optional[str]] = mapped_column(
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

    alert_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default=AlertType.SYSTEM.value,
    )

    severity: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=AlertSeverity.MEDIUM.value,
    )

    priority: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=AlertPriority.NORMAL.value,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=AlertStatus.OPEN.value,
    )

    source: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default=AlertSource.SYSTEM.value,
    )

    source_identifier: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    deduplication_key: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    correlation_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    metric_name: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    metric_value: Mapped[Optional[float]] = mapped_column(
        nullable=True,
    )

    threshold_value: Mapped[Optional[float]] = mapped_column(
        nullable=True,
    )

    metric_unit: Mapped[Optional[str]] = mapped_column(
        String(50),
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

    is_suppressed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    requires_immediate_action: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    auto_generated: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    resolution: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    root_cause: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    recommended_action: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
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

    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(
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

    suppressed_until: Mapped[Optional[datetime]] = mapped_column(
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

    events: Mapped[list["AlertEvent"]] = relationship(
        "AlertEvent",
        back_populates="alert",
        cascade="save-update, merge",
        lazy="selectin",
    )

    notifications: Mapped[list["AlertNotification"]] = relationship(
        "AlertNotification",
        back_populates="alert",
        cascade="save-update, merge",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return (
            f"<Alert("
            f"id={self.id!r}, "
            f"robot_id={self.robot_id!r}, "
            f"severity={self.severity!r}, "
            f"status={self.status!r}"
            f")>"
        )

    @property
    def is_open(self) -> bool:
        return (
            not self.is_deleted
            and self.status
            not in {
                AlertStatus.RESOLVED.value,
                AlertStatus.CLOSED.value,
            }
        )

    @property
    def is_critical(self) -> bool:
        return self.severity in {
            AlertSeverity.CRITICAL.value,
            AlertSeverity.EMERGENCY.value,
        }

    @property
    def is_resolved(self) -> bool:
        return self.status in {
            AlertStatus.RESOLVED.value,
            AlertStatus.CLOSED.value,
        }

    @property
    def is_acknowledged(self) -> bool:
        return self.acknowledged_at is not None

    def acknowledge(
        self,
        user_id: str,
    ) -> None:
        if self.is_resolved:
            raise ValueError(
                "Cannot acknowledge a resolved alert"
            )

        now = utc_now()

        self.acknowledged_by = user_id
        self.acknowledged_at = now
        self.status = AlertStatus.ACKNOWLEDGED.value
        self.updated_at = now

    def resolve(
        self,
        user_id: str,
        resolution: str,
    ) -> None:
        if not resolution.strip():
            raise ValueError(
                "Resolution is required"
            )

        now = utc_now()

        self.resolved_by = user_id
        self.resolution = resolution.strip()
        self.resolved_at = now
        self.status = AlertStatus.RESOLVED.value
        self.updated_at = now

    def close(
        self,
        user_id: str,
    ) -> None:
        if self.status != AlertStatus.RESOLVED.value:
            raise ValueError(
                "Alert must be resolved before closing"
            )

        now = utc_now()

        self.closed_by = user_id
        self.closed_at = now
        self.status = AlertStatus.CLOSED.value
        self.updated_at = now

    def escalate(self) -> None:
        if self.is_resolved:
            raise ValueError(
                "Cannot escalate a resolved alert"
            )

        now = utc_now()

        self.escalation_level += 1
        self.escalated_at = now
        self.status = AlertStatus.ESCALATED.value
        self.updated_at = now

    def increment_occurrence(self) -> None:
        now = utc_now()

        self.occurrence_count += 1
        self.last_occurred_at = now
        self.updated_at = now

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "organization_id": self.organization_id,
            "alert_number": self.alert_number,
            "robot_id": self.robot_id,
            "fleet_id": self.fleet_id,
            "component_id": self.component_id,
            "sensor_id": self.sensor_id,
            "model_id": self.model_id,
            "task_id": self.task_id,
            "incident_id": self.incident_id,
            "maintenance_id": self.maintenance_id,
            "title": self.title,
            "description": self.description,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "priority": self.priority,
            "status": self.status,
            "source": self.source,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold_value": self.threshold_value,
            "metric_unit": self.metric_unit,
            "assigned_to": self.assigned_to,
            "acknowledged_by": self.acknowledged_by,
            "resolved_by": self.resolved_by,
            "escalation_level": self.escalation_level,
            "occurrence_count": self.occurrence_count,
            "is_suppressed": self.is_suppressed,
            "is_deleted": self.is_deleted,
            "requires_immediate_action": (
                self.requires_immediate_action
            ),
            "recommended_action": self.recommended_action,
            "resolution": self.resolution,
            "root_cause": self.root_cause,
            "metadata": self.metadata_json,
            "detected_at": (
                self.detected_at.isoformat()
                if self.detected_at
                else None
            ),
            "acknowledged_at": (
                self.acknowledged_at.isoformat()
                if self.acknowledged_at
                else None
            ),
            "resolved_at": (
                self.resolved_at.isoformat()
                if self.resolved_at
                else None
            ),
            "created_at": (
                self.created_at.isoformat()
                if self.created_at
                else None
            ),
            "updated_at": (
                self.updated_at.isoformat()
                if self.updated_at
                else None
            ),
        }


# ============================================================
# Alert Event Model
# ============================================================


class AlertEvent(Base):
    """
    Append-only alert lifecycle event.

    Events should be created by the repository or
    service layer within the same database transaction
    as the corresponding alert update.
    """

    __tablename__ = "alert_events"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "alert_id"],
            ["alerts.tenant_id", "alerts.id"],
            name="fk_alert_event_tenant_alert",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_alert_event_tenant_alert",
            "tenant_id",
            "alert_id",
        ),
        Index(
            "ix_alert_event_tenant_created",
            "tenant_id",
            "created_at",
        ),
        Index(
            "ix_alert_event_type",
            "tenant_id",
            "event_type",
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

    alert_id: Mapped[str] = mapped_column(
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

    alert: Mapped["Alert"] = relationship(
        "Alert",
        back_populates="events",
    )

    def __repr__(self) -> str:
        return (
            f"<AlertEvent("
            f"id={self.id!r}, "
            f"alert_id={self.alert_id!r}, "
            f"event_type={self.event_type!r}"
            f")>"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "alert_id": self.alert_id,
            "event_type": self.event_type,
            "message": self.message,
            "actor_id": self.actor_id,
            "previous_status": self.previous_status,
            "new_status": self.new_status,
            "metadata": self.metadata_json,
            "created_at": (
                self.created_at.isoformat()
                if self.created_at
                else None
            ),
        }


# ============================================================
# Alert Notification Model
# ============================================================


class AlertNotification(Base):
    """
    Delivery tracking for alert notifications.

    Notification delivery must be performed by a
    background worker or notification service.

    Never send external notifications directly from
    ORM lifecycle methods.
    """

    __tablename__ = "alert_notifications"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "alert_id"],
            ["alerts.tenant_id", "alerts.id"],
            name="fk_alert_notification_tenant_alert",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_alert_notification_tenant_alert",
            "tenant_id",
            "alert_id",
        ),
        Index(
            "ix_alert_notification_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_alert_notification_created",
            "tenant_id",
            "created_at",
        ),
        UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_alert_notification_idempotency",
        ),
        CheckConstraint(
            "attempt_count >= 0",
            name="ck_alert_notification_attempts",
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

    alert_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    channel: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    recipient_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    recipient_address: Mapped[Optional[str]] = mapped_column(
        String(512),
        nullable=True,
    )

    subject: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=AlertNotificationStatus.PENDING.value,
    )

    idempotency_key: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    provider_message_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    max_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=3,
    )

    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    scheduled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    sent_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    delivered_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    failed_at: Mapped[Optional[datetime]] = mapped_column(
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

    alert: Mapped["Alert"] = relationship(
        "Alert",
        back_populates="notifications",
    )

    def __repr__(self) -> str:
        return (
            f"<AlertNotification("
            f"id={self.id!r}, "
            f"alert_id={self.alert_id!r}, "
            f"channel={self.channel!r}, "
            f"status={self.status!r}"
            f")>"
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            AlertNotificationStatus.DELIVERED.value,
            AlertNotificationStatus.CANCELLED.value,
        }

    @property
    def can_retry(self) -> bool:
        return (
            self.status
            == AlertNotificationStatus.FAILED.value
            and self.attempt_count < self.max_attempts
        )

    def mark_processing(self) -> None:
        if self.is_terminal:
            raise ValueError(
                "Terminal notifications cannot be processed"
            )

        if self.attempt_count >= self.max_attempts:
            raise ValueError(
                "Maximum notification attempts reached"
            )

        self.status = (
            AlertNotificationStatus.PROCESSING.value
        )

        self.attempt_count += 1
        self.updated_at = utc_now()

    def mark_sent(
        self,
        provider_message_id: Optional[str] = None,
    ) -> None:
        now = utc_now()

        self.status = (
            AlertNotificationStatus.SENT.value
        )

        self.provider_message_id = provider_message_id
        self.sent_at = now
        self.error_message = None
        self.updated_at = now

    def mark_delivered(self) -> None:
        now = utc_now()

        self.status = (
            AlertNotificationStatus.DELIVERED.value
        )

        self.delivered_at = now
        self.updated_at = now

    def mark_failed(
        self,
        error_message: str,
    ) -> None:
        now = utc_now()

        self.status = (
            AlertNotificationStatus.FAILED.value
        )

        self.error_message = error_message[:4000]
        self.failed_at = now
        self.updated_at = now

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "alert_id": self.alert_id,
            "channel": self.channel,
            "recipient_id": self.recipient_id,
            "subject": self.subject,
            "status": self.status,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "provider_message_id": self.provider_message_id,
            "error_message": self.error_message,
            "metadata": self.metadata_json,
            "scheduled_at": (
                self.scheduled_at.isoformat()
                if self.scheduled_at
                else None
            ),
            "sent_at": (
                self.sent_at.isoformat()
                if self.sent_at
                else None
            ),
            "delivered_at": (
                self.delivered_at.isoformat()
                if self.delivered_at
                else None
            ),
            "created_at": (
                self.created_at.isoformat()
                if self.created_at
                else None
            ),
        }


# ============================================================
# Exports
# ============================================================


__all__ = [
    "Alert",
    "AlertEvent",
    "AlertNotification",
    "AlertType",
    "AlertSeverity",
    "AlertPriority",
    "AlertStatus",
    "AlertSource",
    "AlertEventType",
    "AlertNotificationChannel",
    "AlertNotificationStatus",
    "utc_now",
    "generate_uuid",
]
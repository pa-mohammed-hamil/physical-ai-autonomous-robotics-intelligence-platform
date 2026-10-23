"""Audit Log module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/db/models/audit_log.py

Physical AI & Autonomous Robotics Intelligence Platform

SQLAlchemy 2.x audit logging models.

Features:
- Multi-tenant audit logging
- User, robot, agent, system and API activity tracking
- Authentication and authorization events
- Robot commands and safety events
- AI model and agent execution auditing
- Data access and configuration change tracking
- Security incident auditing
- Request and correlation identifiers
- Before/after change tracking
- Tamper-evident hash chaining
- Sensitive data redaction
- Audit event serialization
- Audit retention metadata

Security notes:
- Audit logs should be append-only.
- Restrict UPDATE and DELETE permissions at database level.
- Use trusted server-side actor and tenant context.
- Do not log passwords, tokens, secrets or raw credentials.
- Hash chaining is tamper-evident only when chain heads are
  independently anchored and protected.
- This module defines ORM models and helpers; it does not
  independently enforce database immutability.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional

from sqlalchemy import (
    BigInteger,
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


SENSITIVE_KEYS = frozenset(
    {
        "password",
        "password_hash",
        "passwd",
        "secret",
        "client_secret",
        "api_key",
        "access_token",
        "refresh_token",
        "id_token",
        "authorization",
        "cookie",
        "session_token",
        "private_key",
        "credential",
        "credentials",
        "otp",
        "pin",
        "reset_token",
        "webhook_secret",
    }
)


def redact_sensitive_data(
    value: Any,
    *,
    max_depth: int = 10,
    _depth: int = 0,
) -> Any:
    """
    Recursively redact commonly sensitive fields.

    Redaction is defense-in-depth, not a substitute for
    explicit allowlisting of audit payload fields.
    """
    if _depth >= max_depth:
        return "[MAX_DEPTH_REACHED]"

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}

        for key, item in value.items():
            normalized_key = str(key).lower()

            if any(
                sensitive in normalized_key
                for sensitive in SENSITIVE_KEYS
            ):
                result[str(key)] = "[REDACTED]"
            else:
                result[str(key)] = redact_sensitive_data(
                    item,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                )

        return result

    if isinstance(value, (list, tuple)):
        return [
            redact_sensitive_data(
                item,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            for item in value
        ]

    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError(
                "Audit datetimes must be timezone-aware"
            )

        return value.astimezone(
            timezone.utc
        ).isoformat()

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, (str, int, float, bool)):
        return value

    if value is None:
        return None

    return str(value)


def canonical_json(
    value: Mapping[str, Any],
) -> str:
    return json.dumps(
        redact_sensitive_data(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def calculate_audit_hash(
    payload: Mapping[str, Any],
    *,
    previous_hash: Optional[str] = None,
) -> str:
    """
    Calculate a SHA-256 hash over canonical audit data.

    For a reliable chain:
    - Allocate sequence numbers transactionally.
    - Serialize concurrent writers per tenant.
    - Store previous_hash from the preceding event.
    - Anchor chain heads outside the primary database.
    """
    body = {
        "previous_hash": previous_hash,
        "payload": dict(payload),
    }

    return hashlib.sha256(
        canonical_json(body).encode("utf-8")
    ).hexdigest()


def verify_audit_hash(
    payload: Mapping[str, Any],
    expected_hash: str,
    *,
    previous_hash: Optional[str] = None,
) -> bool:
    actual_hash = calculate_audit_hash(
        payload,
        previous_hash=previous_hash,
    )

    return hmac.compare_digest(
        actual_hash,
        expected_hash,
    )


# ============================================================
# Enumerations
# ============================================================


class AuditActorType(str, Enum):
    USER = "user"
    ADMIN = "admin"
    SYSTEM = "system"
    SERVICE = "service"
    ROBOT = "robot"
    AI_AGENT = "ai_agent"
    AI_MODEL = "ai_model"
    API_CLIENT = "api_client"
    BACKGROUND_WORKER = "background_worker"
    EXTERNAL_INTEGRATION = "external_integration"
    UNKNOWN = "unknown"


class AuditAction(str, Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    RESTORE = "restore"
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    PASSWORD_CHANGED = "password_changed"
    PASSWORD_RESET = "password_reset"
    EMAIL_VERIFIED = "email_verified"
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REVOKED = "role_revoked"
    PERMISSION_CHANGED = "permission_changed"
    TOKEN_CREATED = "token_created"
    TOKEN_REVOKED = "token_revoked"
    CONFIGURATION_CHANGED = "configuration_changed"
    FILE_UPLOADED = "file_uploaded"
    FILE_DOWNLOADED = "file_downloaded"
    FILE_DELETED = "file_deleted"
    DATA_EXPORTED = "data_exported"
    DATA_IMPORTED = "data_imported"
    ROBOT_REGISTERED = "robot_registered"
    ROBOT_UPDATED = "robot_updated"
    ROBOT_COMMAND = "robot_command"
    ROBOT_STARTED = "robot_started"
    ROBOT_STOPPED = "robot_stopped"
    ROBOT_SHUTDOWN = "robot_shutdown"
    ROBOT_EMERGENCY_STOP = "robot_emergency_stop"
    ROBOT_CONTROL_TRANSFERRED = "robot_control_transferred"
    ROBOT_MAINTENANCE = "robot_maintenance"
    FLEET_CREATED = "fleet_created"
    FLEET_UPDATED = "fleet_updated"
    TASK_CREATED = "task_created"
    TASK_ASSIGNED = "task_assigned"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    MISSION_STARTED = "mission_started"
    MISSION_COMPLETED = "mission_completed"
    MISSION_FAILED = "mission_failed"
    MODEL_REGISTERED = "model_registered"
    MODEL_TRAINED = "model_trained"
    MODEL_EVALUATED = "model_evaluated"
    MODEL_APPROVED = "model_approved"
    MODEL_DEPLOYED = "model_deployed"
    MODEL_ROLLED_BACK = "model_rolled_back"
    AGENT_EXECUTED = "agent_executed"
    AGENT_TOOL_CALLED = "agent_tool_called"
    AGENT_ACTION_APPROVED = "agent_action_approved"
    AGENT_ACTION_REJECTED = "agent_action_rejected"
    POLICY_EVALUATED = "policy_evaluated"
    SAFETY_CHECK = "safety_check"
    SAFETY_VIOLATION = "safety_violation"
    ALERT_CREATED = "alert_created"
    ALERT_ACKNOWLEDGED = "alert_acknowledged"
    ALERT_ESCALATED = "alert_escalated"
    ALERT_RESOLVED = "alert_resolved"
    INCIDENT_CREATED = "incident_created"
    INCIDENT_UPDATED = "incident_updated"
    INCIDENT_RESOLVED = "incident_resolved"
    SECURITY_EVENT = "security_event"
    SECURITY_INCIDENT = "security_incident"
    PROMPT_INJECTION_DETECTED = "prompt_injection_detected"
    PII_DETECTED = "pii_detected"
    MALWARE_DETECTED = "malware_detected"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    WEBHOOK_RECEIVED = "webhook_received"
    WEBHOOK_SENT = "webhook_sent"
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"
    OTHER = "other"


class AuditCategory(str, Enum):
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    USER_MANAGEMENT = "user_management"
    ORGANIZATION = "organization"
    ROBOTICS = "robotics"
    FLEET = "fleet"
    TELEMETRY = "telemetry"
    SENSOR = "sensor"
    TASK = "task"
    MISSION = "mission"
    MAINTENANCE = "maintenance"
    AI_MODEL = "ai_model"
    AI_AGENT = "ai_agent"
    RAG = "rag"
    SIMULATION = "simulation"
    EVALUATION = "evaluation"
    SAFETY = "safety"
    SECURITY = "security"
    INCIDENT = "incident"
    ALERT = "alert"
    DATA_ACCESS = "data_access"
    CONFIGURATION = "configuration"
    INFRASTRUCTURE = "infrastructure"
    INTEGRATION = "integration"
    SYSTEM = "system"
    OTHER = "other"


class AuditOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    ERROR = "error"
    PARTIAL = "partial"
    PENDING = "pending"


class AuditSeverity(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    NOTICE = "notice"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class AuditResourceType(str, Enum):
    USER = "user"
    ORGANIZATION = "organization"
    ROBOT = "robot"
    FLEET = "fleet"
    ROBOT_COMPONENT = "robot_component"
    SENSOR = "sensor"
    TELEMETRY = "telemetry"
    TASK = "task"
    MISSION = "mission"
    MAINTENANCE = "maintenance"
    PART = "part"
    AI_MODEL = "ai_model"
    MODEL_VERSION = "model_version"
    MODEL_DEPLOYMENT = "model_deployment"
    AI_AGENT = "ai_agent"
    EVALUATION = "evaluation"
    EXPERIMENT = "experiment"
    SIMULATION = "simulation"
    ALERT = "alert"
    INCIDENT = "incident"
    FILE = "file"
    DATASET = "dataset"
    API_KEY = "api_key"
    SESSION = "session"
    POLICY = "policy"
    CONFIGURATION = "configuration"
    INTEGRATION = "integration"
    SYSTEM = "system"
    OTHER = "other"


# ============================================================
# Audit Log Model
# ============================================================


class AuditLog(Base):
    """
    Primary append-only audit event.

    This model records who performed an action,
    what was affected, when it happened and
    whether the operation succeeded.

    Avoid storing raw secrets or unnecessary PII.
    """

    __tablename__ = "audit_logs"

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "id",
            name="uq_audit_log_tenant_id",
        ),
        UniqueConstraint(
            "tenant_id",
            "sequence_number",
            name="uq_audit_log_tenant_sequence",
        ),
        Index(
            "ix_audit_tenant_created",
            "tenant_id",
            "created_at",
        ),
        Index(
            "ix_audit_tenant_actor",
            "tenant_id",
            "actor_id",
        ),
        Index(
            "ix_audit_tenant_action",
            "tenant_id",
            "action",
        ),
        Index(
            "ix_audit_tenant_category",
            "tenant_id",
            "category",
        ),
        Index(
            "ix_audit_tenant_outcome",
            "tenant_id",
            "outcome",
        ),
        Index(
            "ix_audit_tenant_resource",
            "tenant_id",
            "resource_type",
            "resource_id",
        ),
        Index(
            "ix_audit_tenant_request",
            "tenant_id",
            "request_id",
        ),
        Index(
            "ix_audit_tenant_correlation",
            "tenant_id",
            "correlation_id",
        ),
        Index(
            "ix_audit_tenant_robot",
            "tenant_id",
            "robot_id",
        ),
        Index(
            "ix_audit_tenant_security",
            "tenant_id",
            "is_security_event",
        ),
        CheckConstraint(
            "sequence_number > 0",
            name="ck_audit_sequence_positive",
        ),
        CheckConstraint(
            "duration_ms IS NULL OR duration_ms >= 0",
            name="ck_audit_duration_nonnegative",
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

    organization_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    sequence_number: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    actor_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    actor_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default=AuditActorType.UNKNOWN.value,
    )

    actor_display_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    action: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    category: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default=AuditCategory.SYSTEM.value,
    )

    outcome: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=AuditOutcome.SUCCESS.value,
    )

    severity: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=AuditSeverity.INFO.value,
    )

    resource_type: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    resource_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    resource_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    robot_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    fleet_id: Mapped[Optional[str]] = mapped_column(
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

    request_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    correlation_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    trace_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    span_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    session_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    ip_address: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )

    user_agent: Mapped[Optional[str]] = mapped_column(
        String(1024),
        nullable=True,
    )

    http_method: Mapped[Optional[str]] = mapped_column(
        String(16),
        nullable=True,
    )

    endpoint: Mapped[Optional[str]] = mapped_column(
        String(2048),
        nullable=True,
    )

    http_status_code: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    error_code: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    before_state_json: Mapped[Optional[dict[str, Any]]] = (
        mapped_column(
            JSON,
            nullable=True,
        )
    )

    after_state_json: Mapped[Optional[dict[str, Any]]] = (
        mapped_column(
            JSON,
            nullable=True,
        )
    )

    changes_json: Mapped[Optional[dict[str, Any]]] = (
        mapped_column(
            JSON,
            nullable=True,
        )
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    duration_ms: Mapped[Optional[float]] = mapped_column(
        nullable=True,
    )

    is_security_event: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    is_sensitive: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    requires_review: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    previous_hash: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )

    event_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    hash_algorithm: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="sha256",
    )

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    retention_until: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    def __repr__(self) -> str:
        return (
            f"<AuditLog("
            f"id={self.id!r}, "
            f"tenant_id={self.tenant_id!r}, "
            f"action={self.action!r}, "
            f"outcome={self.outcome!r}"
            f")>"
        )

    @property
    def is_failure(self) -> bool:
        return self.outcome in {
            AuditOutcome.FAILURE.value,
            AuditOutcome.ERROR.value,
            AuditOutcome.DENIED.value,
        }

    @property
    def is_critical(self) -> bool:
        return self.severity == AuditSeverity.CRITICAL.value

    def hash_payload(self) -> dict[str, Any]:
        """
        Return immutable event content used for hashing.

        Do not change the payload schema after records
        have been written without versioning the hash
        algorithm and canonicalization format.
        """
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "organization_id": self.organization_id,
            "sequence_number": self.sequence_number,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "action": self.action,
            "category": self.category,
            "outcome": self.outcome,
            "severity": self.severity,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "robot_id": self.robot_id,
            "fleet_id": self.fleet_id,
            "model_id": self.model_id,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "message": self.message,
            "error_code": self.error_code,
            "before_state": self.before_state_json,
            "after_state": self.after_state_json,
            "changes": self.changes_json,
            "metadata": self.metadata_json,
            "is_security_event": self.is_security_event,
            "occurred_at": self.occurred_at,
        }

    def calculate_hash(self) -> str:
        return calculate_audit_hash(
            self.hash_payload(),
            previous_hash=self.previous_hash,
        )

    def verify_integrity(self) -> bool:
        if not self.event_hash:
            return False

        return verify_audit_hash(
            self.hash_payload(),
            self.event_hash,
            previous_hash=self.previous_hash,
        )

    def to_dict(
        self,
        *,
        include_changes: bool = False,
        include_integrity: bool = False,
    ) -> dict[str, Any]:
        result = {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "organization_id": self.organization_id,
            "sequence_number": self.sequence_number,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "action": self.action,
            "category": self.category,
            "outcome": self.outcome,
            "severity": self.severity,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "resource_name": self.resource_name,
            "robot_id": self.robot_id,
            "fleet_id": self.fleet_id,
            "model_id": self.model_id,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
            "trace_id": self.trace_id,
            "message": self.message,
            "error_code": self.error_code,
            "duration_ms": self.duration_ms,
            "is_security_event": self.is_security_event,
            "requires_review": self.requires_review,
            "occurred_at": (
                self.occurred_at.isoformat()
                if self.occurred_at
                else None
            ),
            "created_at": (
                self.created_at.isoformat()
                if self.created_at
                else None
            ),
        }

        if include_changes:
            result["before_state"] = (
                redact_sensitive_data(
                    self.before_state_json
                )
            )

            result["after_state"] = (
                redact_sensitive_data(
                    self.after_state_json
                )
            )

            result["changes"] = (
                redact_sensitive_data(
                    self.changes_json
                )
            )

        if include_integrity:
            result["previous_hash"] = self.previous_hash
            result["event_hash"] = self.event_hash
            result["hash_algorithm"] = self.hash_algorithm
            result["integrity_valid"] = (
                self.verify_integrity()
            )

        return result


# ============================================================
# Audit Chain State
# ============================================================


class AuditChainState(Base):
    """
    Per-tenant chain-head state.

    A writer should:
        1. Lock this row with SELECT FOR UPDATE.
        2. Increment last_sequence_number.
        3. Create AuditLog with previous_hash.
        4. Calculate and store event_hash.
        5. Update last_event_hash.
        6. Commit both changes atomically.

    Independently anchor the chain head periodically.
    """

    __tablename__ = "audit_chain_states"

    tenant_id: Mapped[str] = mapped_column(
        String(255),
        primary_key=True,
    )

    last_sequence_number: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
    )

    last_event_hash: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )

    last_event_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        nullable=True,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    __table_args__ = (
        CheckConstraint(
            "last_sequence_number >= 0",
            name="ck_audit_chain_sequence_nonnegative",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<AuditChainState("
            f"tenant_id={self.tenant_id!r}, "
            f"last_sequence_number="
            f"{self.last_sequence_number}"
            f")>"
        )


# ============================================================
# Audit Review Model
# ============================================================


class AuditReviewStatus(str, Enum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    FLAGGED = "flagged"
    ESCALATED = "escalated"
    CLOSED = "closed"


class AuditReview(Base):
    """
    Human review and investigation workflow.

    Reviews are separate from immutable audit events.
    """

    __tablename__ = "audit_reviews"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "audit_log_id"],
            ["audit_logs.tenant_id", "audit_logs.id"],
            name="fk_audit_review_tenant_log",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_audit_review_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_audit_review_tenant_log",
            "tenant_id",
            "audit_log_id",
        ),
        Index(
            "ix_audit_review_assigned",
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

    audit_log_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=AuditReviewStatus.PENDING.value,
    )

    assigned_to: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    reviewed_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    reason: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    findings: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    resolution: Mapped[Optional[str]] = mapped_column(
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

    reviewed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    closed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    audit_log: Mapped["AuditLog"] = relationship(
        "AuditLog",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return (
            f"<AuditReview("
            f"id={self.id!r}, "
            f"audit_log_id={self.audit_log_id!r}, "
            f"status={self.status!r}"
            f")>"
        )


# ============================================================
# Exports
# ============================================================


__all__ = [
    "AuditLog",
    "AuditChainState",
    "AuditReview",
    "AuditActorType",
    "AuditAction",
    "AuditCategory",
    "AuditOutcome",
    "AuditSeverity",
    "AuditResourceType",
    "AuditReviewStatus",
    "utc_now",
    "generate_uuid",
    "redact_sensitive_data",
    "canonical_json",
    "calculate_audit_hash",
    "verify_audit_hash",
]
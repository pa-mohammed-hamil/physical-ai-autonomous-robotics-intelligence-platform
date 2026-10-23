"""
backend/app/db/models/fleet.py

Physical AI & Autonomous Robotics Intelligence Platform

SQLAlchemy 2.x ORM models for:
- Robot fleet management
- Fleet registration and configuration
- Multi-tenant fleet isolation
- Robot-to-fleet assignments
- Fleet operational status
- Fleet health monitoring
- Fleet location and capacity
- Fleet event history
- Fleet statistics and metadata

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


def serialize_datetime(
    value: Optional[datetime],
) -> Optional[str]:
    return value.isoformat() if value else None


# ============================================================
# Enumerations
# ============================================================


class FleetStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"
    OFFLINE = "offline"
    SUSPENDED = "suspended"
    DECOMMISSIONED = "decommissioned"


class FleetType(str, Enum):
    GENERAL = "general"
    INDUSTRIAL = "industrial"
    WAREHOUSE = "warehouse"
    MANUFACTURING = "manufacturing"
    LOGISTICS = "logistics"
    DELIVERY = "delivery"
    AUTONOMOUS_MOBILE = "autonomous_mobile"
    HUMANOID = "humanoid"
    DRONE = "drone"
    AGRICULTURAL = "agricultural"
    MEDICAL = "medical"
    SERVICE = "service"
    INSPECTION = "inspection"
    SECURITY = "security"
    RESCUE = "rescue"
    RESEARCH = "research"
    MIXED = "mixed"
    OTHER = "other"


class FleetHealthStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class FleetEventType(str, Enum):
    CREATED = "created"
    UPDATED = "updated"
    STATUS_CHANGED = "status_changed"
    HEALTH_CHANGED = "health_changed"
    ROBOT_ASSIGNED = "robot_assigned"
    ROBOT_REMOVED = "robot_removed"
    CONFIGURATION_UPDATED = "configuration_updated"
    CAPACITY_UPDATED = "capacity_updated"
    LOCATION_UPDATED = "location_updated"
    MAINTENANCE_STARTED = "maintenance_started"
    MAINTENANCE_COMPLETED = "maintenance_completed"
    EMERGENCY_DECLARED = "emergency_declared"
    EMERGENCY_CLEARED = "emergency_cleared"
    DECOMMISSIONED = "decommissioned"
    NOTE_ADDED = "note_added"
    DELETED = "deleted"
    RESTORED = "restored"


class FleetEventSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ============================================================
# Fleet Model
# ============================================================


class Fleet(Base):
    """
    Represents a collection of robots managed together.

    Fleet capacity and assignment uniqueness must be
    enforced transactionally by the repository/service.
    """

    __tablename__ = "fleets"

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "id",
            name="uq_fleet_tenant_id",
        ),
        UniqueConstraint(
            "tenant_id",
            "name",
            name="uq_fleet_tenant_name",
        ),
        Index(
            "ix_fleet_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_fleet_tenant_organization",
            "tenant_id",
            "organization_id",
        ),
        Index(
            "ix_fleet_tenant_created",
            "tenant_id",
            "created_at",
        ),
        CheckConstraint(
            "max_robots IS NULL OR max_robots > 0",
            name="ck_fleet_max_robots_positive",
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
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    fleet_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default=FleetType.GENERAL.value,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=FleetStatus.ACTIVE.value,
    )

    health_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=FleetHealthStatus.UNKNOWN.value,
    )

    location: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )

    max_robots: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    manager_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    created_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    configuration_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
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

    robot_assignments: Mapped[list["FleetRobot"]] = relationship(
        "FleetRobot",
        back_populates="fleet",
        cascade="save-update, merge",
        lazy="selectin",
    )

    events: Mapped[list["FleetEvent"]] = relationship(
        "FleetEvent",
        back_populates="fleet",
        cascade="save-update, merge",
        lazy="selectin",
    )

    @property
    def is_operational(self) -> bool:
        return (
            not self.is_deleted
            and self.status in {
                FleetStatus.ACTIVE.value,
                FleetStatus.OPERATIONAL.value,
            }
        )

    @property
    def is_critical(self) -> bool:
        return (
            self.health_status
            == FleetHealthStatus.CRITICAL.value
            or self.status == FleetStatus.EMERGENCY.value
        )

    @property
    def active_robot_count(self) -> int:
        return sum(
            1
            for assignment in self.robot_assignments
            if assignment.is_active
        )

    @property
    def available_capacity(self) -> Optional[int]:
        if self.max_robots is None:
            return None

        return max(
            0,
            self.max_robots - self.active_robot_count,
        )

    def set_status(
        self,
        status: FleetStatus | str,
    ) -> None:
        value = (
            status.value
            if isinstance(status, FleetStatus)
            else status
        )

        if value not in {item.value for item in FleetStatus}:
            raise ValueError("Invalid fleet status")

        self.status = value
        self.updated_at = utc_now()

    def set_health_status(
        self,
        health_status: FleetHealthStatus | str,
    ) -> None:
        value = (
            health_status.value
            if isinstance(health_status, FleetHealthStatus)
            else health_status
        )

        if value not in {
            item.value for item in FleetHealthStatus
        }:
            raise ValueError("Invalid fleet health status")

        self.health_status = value
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "organization_id": self.organization_id,
            "name": self.name,
            "description": self.description,
            "fleet_type": self.fleet_type,
            "status": self.status,
            "health_status": self.health_status,
            "location": self.location,
            "max_robots": self.max_robots,
            "manager_id": self.manager_id,
            "configuration": self.configuration_json,
            "metadata": self.metadata_json,
            "is_deleted": self.is_deleted,
            "created_at": serialize_datetime(self.created_at),
            "updated_at": serialize_datetime(self.updated_at),
        }

    def __repr__(self) -> str:
        return (
            f"<Fleet("
            f"id={self.id!r}, "
            f"name={self.name!r}, "
            f"status={self.status!r}"
            f")>"
        )


# ============================================================
# Fleet Robot Assignment
# ============================================================


class FleetRobot(Base):
    """
    Tracks robot membership in a fleet.

    Historical assignments are retained by marking
    is_active=False and setting removed_at.

    A database-level partial unique index may be added
    in PostgreSQL if robots must belong to only one
    active fleet at a time.
    """

    __tablename__ = "fleet_robots"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "fleet_id"],
            ["fleets.tenant_id", "fleets.id"],
            name="fk_fleet_robot_tenant_fleet",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_fleet_robot_tenant_fleet",
            "tenant_id",
            "fleet_id",
        ),
        Index(
            "ix_fleet_robot_tenant_robot",
            "tenant_id",
            "robot_id",
        ),
        Index(
            "ix_fleet_robot_tenant_active",
            "tenant_id",
            "is_active",
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

    fleet_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    robot_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    assigned_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    removed_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    removed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    fleet: Mapped["Fleet"] = relationship(
        "Fleet",
        back_populates="robot_assignments",
    )

    def deactivate(
        self,
        *,
        actor_id: Optional[str] = None,
    ) -> None:
        if not self.is_active:
            raise ValueError(
                "Robot assignment is already inactive"
            )

        self.is_active = False
        self.removed_by = actor_id
        self.removed_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "fleet_id": self.fleet_id,
            "robot_id": self.robot_id,
            "assigned_by": self.assigned_by,
            "removed_by": self.removed_by,
            "is_active": self.is_active,
            "assigned_at": serialize_datetime(self.assigned_at),
            "removed_at": serialize_datetime(self.removed_at),
            "metadata": self.metadata_json,
        }

    def __repr__(self) -> str:
        return (
            f"<FleetRobot("
            f"fleet_id={self.fleet_id!r}, "
            f"robot_id={self.robot_id!r}, "
            f"is_active={self.is_active}"
            f")>"
        )


# ============================================================
# Fleet Event
# ============================================================


class FleetEvent(Base):
    """
    Append-only history of fleet lifecycle events.

    Event creation should occur in the same transaction
    as the associated fleet change.
    """

    __tablename__ = "fleet_events"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "fleet_id"],
            ["fleets.tenant_id", "fleets.id"],
            name="fk_fleet_event_tenant_fleet",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_fleet_event_tenant_fleet",
            "tenant_id",
            "fleet_id",
        ),
        Index(
            "ix_fleet_event_tenant_type",
            "tenant_id",
            "event_type",
        ),
        Index(
            "ix_fleet_event_tenant_created",
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

    fleet_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    robot_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    event_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    severity: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=FleetEventSeverity.INFO.value,
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

    fleet: Mapped["Fleet"] = relationship(
        "Fleet",
        back_populates="events",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "fleet_id": self.fleet_id,
            "robot_id": self.robot_id,
            "event_type": self.event_type,
            "severity": self.severity,
            "message": self.message,
            "actor_id": self.actor_id,
            "previous_status": self.previous_status,
            "new_status": self.new_status,
            "metadata": self.metadata_json,
            "created_at": serialize_datetime(self.created_at),
        }

    def __repr__(self) -> str:
        return (
            f"<FleetEvent("
            f"id={self.id!r}, "
            f"fleet_id={self.fleet_id!r}, "
            f"event_type={self.event_type!r}"
            f")>"
        )


# ============================================================
# Exports
# ============================================================


__all__ = [
    "Fleet",
    "FleetRobot",
    "FleetEvent",
    "FleetStatus",
    "FleetType",
    "FleetHealthStatus",
    "FleetEventType",
    "FleetEventSeverity",
    "utc_now",
    "generate_uuid",
    "serialize_datetime",
]
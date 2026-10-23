"""Organization module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/db/models/organization.py

Physical AI & Autonomous Robotics Intelligence Platform

SQLAlchemy 2.x ORM models for:
- Organization management
- Multi-tenant isolation
- Organization membership
- Role-based access management
- Organization settings
- Organization lifecycle events
- Subscription and resource limits
- Audit-friendly organization history
"""

from __future__ import annotations

import re
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


def normalize_slug(value: str) -> str:
    slug = re.sub(
        r"[^a-z0-9]+",
        "-",
        value.strip().lower(),
    ).strip("-")

    if not slug:
        raise ValueError("Organization slug cannot be empty")

    if len(slug) > 150:
        raise ValueError("Organization slug is too long")

    return slug


# ============================================================
# Enumerations
# ============================================================


class OrganizationStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    RESTRICTED = "restricted"
    DEACTIVATED = "deactivated"
    ARCHIVED = "archived"


class OrganizationType(str, Enum):
    INDIVIDUAL = "individual"
    STARTUP = "startup"
    ENTERPRISE = "enterprise"
    RESEARCH = "research"
    UNIVERSITY = "university"
    GOVERNMENT = "government"
    INDUSTRIAL = "industrial"
    MANUFACTURING = "manufacturing"
    LOGISTICS = "logistics"
    HEALTHCARE = "healthcare"
    ROBOTICS_COMPANY = "robotics_company"
    OTHER = "other"


class OrganizationPlan(str, Enum):
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class OrganizationMemberRole(str, Enum):
    OWNER = "owner"
    ADMIN = "admin"
    FLEET_MANAGER = "fleet_manager"
    ROBOT_OPERATOR = "robot_operator"
    ROBOT_ENGINEER = "robot_engineer"
    AI_ENGINEER = "ai_engineer"
    ML_ENGINEER = "ml_engineer"
    MAINTENANCE_ENGINEER = "maintenance_engineer"
    SAFETY_OFFICER = "safety_officer"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    AUDITOR = "auditor"
    VIEWER = "viewer"


class OrganizationMemberStatus(str, Enum):
    INVITED = "invited"
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    REMOVED = "removed"


class OrganizationEventType(str, Enum):
    CREATED = "created"
    UPDATED = "updated"
    ACTIVATED = "activated"
    DEACTIVATED = "deactivated"
    SUSPENDED = "suspended"
    UNSUSPENDED = "unsuspended"
    RESTRICTED = "restricted"
    ARCHIVED = "archived"
    RESTORED = "restored"

    OWNER_CHANGED = "owner_changed"
    MEMBER_INVITED = "member_invited"
    MEMBER_JOINED = "member_joined"
    MEMBER_UPDATED = "member_updated"
    MEMBER_SUSPENDED = "member_suspended"
    MEMBER_REMOVED = "member_removed"
    MEMBER_RESTORED = "member_restored"
    ROLE_CHANGED = "role_changed"

    SETTINGS_UPDATED = "settings_updated"
    SECURITY_UPDATED = "security_updated"
    PLAN_CHANGED = "plan_changed"
    LIMITS_UPDATED = "limits_updated"
    BILLING_UPDATED = "billing_updated"

    ROBOT_REGISTERED = "robot_registered"
    ROBOT_REMOVED = "robot_removed"
    FLEET_CREATED = "fleet_created"
    FLEET_REMOVED = "fleet_removed"

    NOTE_ADDED = "note_added"
    DELETED = "deleted"


class OrganizationEventSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ============================================================
# Organization
# ============================================================


class Organization(Base):
    """
    Represents an organization within a tenant.

    Authentication, authorization, billing enforcement,
    resource quotas and physical robot permissions must
    be validated by the application service layer.
    """

    __tablename__ = "organizations"

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "id",
            name="uq_organization_tenant_id",
        ),
        UniqueConstraint(
            "tenant_id",
            "slug",
            name="uq_organization_tenant_slug",
        ),
        Index(
            "ix_organization_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_organization_tenant_owner",
            "tenant_id",
            "owner_id",
        ),
        Index(
            "ix_organization_tenant_plan",
            "tenant_id",
            "plan",
        ),
        Index(
            "ix_organization_tenant_created",
            "tenant_id",
            "created_at",
        ),
        CheckConstraint(
            "max_users IS NULL OR max_users >= 0",
            name="ck_organization_max_users",
        ),
        CheckConstraint(
            "max_robots IS NULL OR max_robots >= 0",
            name="ck_organization_max_robots",
        ),
        CheckConstraint(
            "max_fleets IS NULL OR max_fleets >= 0",
            name="ck_organization_max_fleets",
        ),
        CheckConstraint(
            "max_models IS NULL OR max_models >= 0",
            name="ck_organization_max_models",
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

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    slug: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    organization_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default=OrganizationType.ENTERPRISE.value,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=OrganizationStatus.PENDING.value,
    )

    plan: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=OrganizationPlan.FREE.value,
    )

    owner_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    created_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    contact_email: Mapped[Optional[str]] = mapped_column(
        String(320),
        nullable=True,
    )

    contact_phone: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )

    website_url: Mapped[Optional[str]] = mapped_column(
        String(2048),
        nullable=True,
    )

    logo_url: Mapped[Optional[str]] = mapped_column(
        String(2048),
        nullable=True,
    )

    country: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    state: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    city: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    address: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    timezone_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="UTC",
    )

    max_users: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    max_robots: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    max_fleets: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    max_models: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    settings_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    security_settings_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    notification_settings_json: Mapped[
        dict[str, Any]
    ] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    resource_limits_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    activated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    suspended_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    deleted_at: Mapped[Optional[datetime]] = mapped_column(
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

    members: Mapped[list["OrganizationMember"]] = relationship(
        "OrganizationMember",
        back_populates="organization",
        cascade="save-update, merge",
        lazy="selectin",
    )

    events: Mapped[list["OrganizationEvent"]] = relationship(
        "OrganizationEvent",
        back_populates="organization",
        cascade="save-update, merge",
        lazy="selectin",
    )

    @validates("slug")
    def validate_slug(
        self,
        key: str,
        value: str,
    ) -> str:
        return normalize_slug(value)

    @validates("name")
    def validate_name(
        self,
        key: str,
        value: str,
    ) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Organization name is required")

        return value.strip()

    @validates(
        "max_users",
        "max_robots",
        "max_fleets",
        "max_models",
    )
    def validate_limit(
        self,
        key: str,
        value: Optional[int],
    ) -> Optional[int]:
        if value is None:
            return None

        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")

        if value < 0:
            raise ValueError(f"{key} cannot be negative")

        return value

    @property
    def is_active(self) -> bool:
        return (
            not self.is_deleted
            and self.status == OrganizationStatus.ACTIVE.value
        )

    @property
    def active_member_count(self) -> int:
        """
        Counts currently loaded organization members.

        For authoritative counts and quota enforcement,
        use a transactionally consistent database query.
        """
        return sum(
            1
            for member in self.members
            if member.status == OrganizationMemberStatus.ACTIVE.value
        )

    def activate(self) -> None:
        if self.is_deleted:
            raise ValueError(
                "Deleted organizations cannot be activated"
            )

        now = utc_now()

        self.status = OrganizationStatus.ACTIVE.value
        self.activated_at = now
        self.suspended_at = None
        self.updated_at = now

    def suspend(self) -> None:
        if self.is_deleted:
            raise ValueError(
                "Deleted organizations cannot be suspended"
            )

        now = utc_now()

        self.status = OrganizationStatus.SUSPENDED.value
        self.suspended_at = now
        self.updated_at = now

    def deactivate(self) -> None:
        if self.is_deleted:
            raise ValueError(
                "Deleted organizations cannot be deactivated"
            )

        self.status = OrganizationStatus.DEACTIVATED.value
        self.updated_at = utc_now()

    def archive(self) -> None:
        if self.is_deleted:
            raise ValueError(
                "Deleted organizations cannot be archived"
            )

        self.status = OrganizationStatus.ARCHIVED.value
        self.updated_at = utc_now()

    def soft_delete(self) -> None:
        if self.is_deleted:
            raise ValueError(
                "Organization is already deleted"
            )

        now = utc_now()

        self.is_deleted = True
        self.deleted_at = now
        self.status = OrganizationStatus.DEACTIVATED.value
        self.updated_at = now

    def restore(self) -> None:
        if not self.is_deleted:
            raise ValueError(
                "Organization is not deleted"
            )

        self.is_deleted = False
        self.deleted_at = None
        self.status = OrganizationStatus.INACTIVE.value
        self.updated_at = utc_now()

    def update_plan(
        self,
        plan: OrganizationPlan | str,
    ) -> None:
        value = (
            plan.value
            if isinstance(plan, OrganizationPlan)
            else plan
        )

        if value not in {item.value for item in OrganizationPlan}:
            raise ValueError("Invalid organization plan")

        self.plan = value
        self.updated_at = utc_now()

    def to_dict(
        self,
        *,
        include_settings: bool = False,
        include_limits: bool = True,
    ) -> dict[str, Any]:
        result = {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "organization_type": self.organization_type,
            "status": self.status,
            "plan": self.plan,
            "owner_id": self.owner_id,
            "contact_email": self.contact_email,
            "contact_phone": self.contact_phone,
            "website_url": self.website_url,
            "logo_url": self.logo_url,
            "country": self.country,
            "state": self.state,
            "city": self.city,
            "timezone": self.timezone_name,
            "is_verified": self.is_verified,
            "is_deleted": self.is_deleted,
            "activated_at": serialize_datetime(
                self.activated_at
            ),
            "suspended_at": serialize_datetime(
                self.suspended_at
            ),
            "created_at": serialize_datetime(
                self.created_at
            ),
            "updated_at": serialize_datetime(
                self.updated_at
            ),
        }

        if include_limits:
            result["resource_limits"] = {
                "max_users": self.max_users,
                "max_robots": self.max_robots,
                "max_fleets": self.max_fleets,
                "max_models": self.max_models,
            }

        if include_settings:
            result["settings"] = self.settings_json
            result["notification_settings"] = (
                self.notification_settings_json
            )
            result["metadata"] = self.metadata_json

        return result

    def __repr__(self) -> str:
        return (
            f"<Organization("
            f"id={self.id!r}, "
            f"name={self.name!r}, "
            f"status={self.status!r}"
            f")>"
        )


# ============================================================
# Organization Member
# ============================================================


class OrganizationMember(Base):
    """
    Represents a user's membership in an organization.

    The organization role is descriptive here.
    API authorization must verify tenant membership,
    account state, role permissions and safety policy.
    """

    __tablename__ = "organization_members"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "organization_id"],
            ["organizations.tenant_id", "organizations.id"],
            name="fk_org_member_tenant_organization",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "tenant_id",
            "organization_id",
            "user_id",
            name="uq_org_member_tenant_org_user",
        ),
        Index(
            "ix_org_member_tenant_user",
            "tenant_id",
            "user_id",
        ),
        Index(
            "ix_org_member_tenant_role",
            "tenant_id",
            "role",
        ),
        Index(
            "ix_org_member_tenant_status",
            "tenant_id",
            "status",
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

    organization_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    user_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    role: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default=OrganizationMemberRole.VIEWER.value,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=OrganizationMemberStatus.INVITED.value,
    )

    invited_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    removed_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    permissions_json: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    invited_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    joined_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    suspended_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    removed_at: Mapped[Optional[datetime]] = mapped_column(
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

    organization: Mapped["Organization"] = relationship(
        "Organization",
        back_populates="members",
    )

    @validates("role")
    def validate_role(
        self,
        key: str,
        value: str,
    ) -> str:
        if value not in {
            role.value for role in OrganizationMemberRole
        }:
            raise ValueError("Invalid organization member role")

        return value

    @property
    def is_active(self) -> bool:
        return self.status == OrganizationMemberStatus.ACTIVE.value

    @property
    def is_admin(self) -> bool:
        return (
            self.is_active
            and self.role in {
                OrganizationMemberRole.OWNER.value,
                OrganizationMemberRole.ADMIN.value,
            }
        )

    def activate(self) -> None:
        if self.status not in {
            OrganizationMemberStatus.INVITED.value,
            OrganizationMemberStatus.PENDING.value,
            OrganizationMemberStatus.SUSPENDED.value,
        }:
            raise ValueError(
                "Membership cannot be activated"
            )

        now = utc_now()

        self.status = OrganizationMemberStatus.ACTIVE.value
        self.joined_at = self.joined_at or now
        self.suspended_at = None
        self.removed_at = None
        self.updated_at = now

    def suspend(self) -> None:
        if self.status != OrganizationMemberStatus.ACTIVE.value:
            raise ValueError(
                "Only active members can be suspended"
            )

        now = utc_now()

        self.status = OrganizationMemberStatus.SUSPENDED.value
        self.suspended_at = now
        self.updated_at = now

    def remove(
        self,
        *,
        actor_id: Optional[str] = None,
    ) -> None:
        if self.status == OrganizationMemberStatus.REMOVED.value:
            raise ValueError(
                "Membership is already removed"
            )

        now = utc_now()

        self.status = OrganizationMemberStatus.REMOVED.value
        self.removed_by = actor_id
        self.removed_at = now
        self.updated_at = now

    def change_role(
        self,
        role: OrganizationMemberRole | str,
    ) -> None:
        if not self.is_active:
            raise ValueError(
                "Only active members can change roles"
            )

        value = (
            role.value
            if isinstance(role, OrganizationMemberRole)
            else role
        )

        self.role = value
        self.updated_at = utc_now()

    def to_dict(
        self,
        *,
        include_permissions: bool = False,
    ) -> dict[str, Any]:
        result = {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "organization_id": self.organization_id,
            "user_id": self.user_id,
            "role": self.role,
            "status": self.status,
            "invited_by": self.invited_by,
            "invited_at": serialize_datetime(
                self.invited_at
            ),
            "joined_at": serialize_datetime(
                self.joined_at
            ),
            "suspended_at": serialize_datetime(
                self.suspended_at
            ),
            "removed_at": serialize_datetime(
                self.removed_at
            ),
            "created_at": serialize_datetime(
                self.created_at
            ),
            "updated_at": serialize_datetime(
                self.updated_at
            ),
        }

        if include_permissions:
            result["permissions"] = self.permissions_json

        return result

    def __repr__(self) -> str:
        return (
            f"<OrganizationMember("
            f"id={self.id!r}, "
            f"organization_id={self.organization_id!r}, "
            f"user_id={self.user_id!r}, "
            f"role={self.role!r}"
            f")>"
        )


# ============================================================
# Organization Event
# ============================================================


class OrganizationEvent(Base):
    """
    Organization lifecycle and membership event history.

    Insert events transactionally alongside state changes.
    For immutable audit requirements, use database-level
    append-only enforcement and the central audit system.
    """

    __tablename__ = "organization_events"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "organization_id"],
            ["organizations.tenant_id", "organizations.id"],
            name="fk_org_event_tenant_organization",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_org_event_tenant_organization",
            "tenant_id",
            "organization_id",
        ),
        Index(
            "ix_org_event_tenant_type",
            "tenant_id",
            "event_type",
        ),
        Index(
            "ix_org_event_tenant_created",
            "tenant_id",
            "created_at",
        ),
        Index(
            "ix_org_event_tenant_actor",
            "tenant_id",
            "actor_id",
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

    organization_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    event_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    severity: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=OrganizationEventSeverity.INFO.value,
    )

    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    actor_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    target_user_id: Mapped[Optional[str]] = mapped_column(
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

    correlation_id: Mapped[Optional[str]] = mapped_column(
        String(255),
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

    organization: Mapped["Organization"] = relationship(
        "Organization",
        back_populates="events",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "organization_id": self.organization_id,
            "event_type": self.event_type,
            "severity": self.severity,
            "message": self.message,
            "actor_id": self.actor_id,
            "target_user_id": self.target_user_id,
            "previous_status": self.previous_status,
            "new_status": self.new_status,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata_json,
            "created_at": serialize_datetime(
                self.created_at
            ),
        }

    def __repr__(self) -> str:
        return (
            f"<OrganizationEvent("
            f"id={self.id!r}, "
            f"organization_id={self.organization_id!r}, "
            f"event_type={self.event_type!r}"
            f")>"
        )


# ============================================================
# Exports
# ============================================================


__all__ = [
    "Organization",
    "OrganizationMember",
    "OrganizationEvent",
    "OrganizationStatus",
    "OrganizationType",
    "OrganizationPlan",
    "OrganizationMemberRole",
    "OrganizationMemberStatus",
    "OrganizationEventType",
    "OrganizationEventSeverity",
    "utc_now",
    "generate_uuid",
    "serialize_datetime",
    "normalize_slug",
]
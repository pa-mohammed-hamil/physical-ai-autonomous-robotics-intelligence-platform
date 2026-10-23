"""Repair module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/db/models/repair.py

Physical AI & Autonomous Robotics Intelligence Platform

SQLAlchemy 2.x ORM models for:
- Robot repair work orders
- Component and spare-part repairs
- Repair lifecycle tracking
- Technician assignments
- Repair diagnosis and root-cause analysis
- Repair labor and parts tracking
- Repair costs and downtime
- Quality assurance and safety verification
- Repair history and audit events
- Multi-tenant isolation

Physical robot operations require separate authorization,
safety interlocks, and human-supervised control systems.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
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
    Numeric,
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


def validate_decimal(
    value: Any,
    field_name: str,
    *,
    allow_zero: bool = True,
) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")

    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(
            f"{field_name} must be a valid number"
        ) from exc

    if not result.is_finite():
        raise ValueError(f"{field_name} must be finite")

    if result < 0 or (not allow_zero and result == 0):
        raise ValueError(
            f"{field_name} must be "
            f"{'nonnegative' if allow_zero else 'positive'}"
        )

    return result


# ============================================================
# Enumerations
# ============================================================


class RepairStatus(str, Enum):
    DRAFT = "draft"
    REQUESTED = "requested"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    DIAGNOSING = "diagnosing"
    WAITING_FOR_PARTS = "waiting_for_parts"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    REPAIR_COMPLETED = "repair_completed"
    TESTING = "testing"
    QUALITY_REVIEW = "quality_review"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RepairType(str, Enum):
    CORRECTIVE = "corrective"
    PREVENTIVE = "preventive"
    EMERGENCY = "emergency"
    PREDICTIVE = "predictive"
    WARRANTY = "warranty"
    HARDWARE = "hardware"
    SOFTWARE = "software"
    FIRMWARE = "firmware"
    ELECTRICAL = "electrical"
    MECHANICAL = "mechanical"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    MOTOR = "motor"
    BATTERY = "battery"
    NETWORK = "network"
    CALIBRATION = "calibration"
    COMPONENT_REPLACEMENT = "component_replacement"
    REFURBISHMENT = "refurbishment"
    OTHER = "other"


class RepairPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class RepairTaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class RepairPartAction(str, Enum):
    INSTALL = "install"
    REPLACE = "replace"
    REMOVE = "remove"
    REPAIR = "repair"
    REFURBISH = "refurbish"
    INSPECT = "inspect"
    RETURN = "return"


class RepairQualityStatus(str, Enum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    PASSED = "passed"
    FAILED = "failed"
    REWORK_REQUIRED = "rework_required"


class RepairEventType(str, Enum):
    CREATED = "created"
    UPDATED = "updated"
    REQUESTED = "requested"
    APPROVED = "approved"
    REJECTED = "rejected"
    ASSIGNED = "assigned"
    SCHEDULED = "scheduled"
    DIAGNOSIS_STARTED = "diagnosis_started"
    DIAGNOSIS_COMPLETED = "diagnosis_completed"
    REPAIR_STARTED = "repair_started"
    REPAIR_PAUSED = "repair_paused"
    REPAIR_RESUMED = "repair_resumed"
    PART_REQUESTED = "part_requested"
    PART_INSTALLED = "part_installed"
    PART_REPLACED = "part_replaced"
    PART_REMOVED = "part_removed"
    TASK_COMPLETED = "task_completed"
    REPAIR_COMPLETED = "repair_completed"
    TESTING_STARTED = "testing_started"
    QUALITY_PASSED = "quality_passed"
    QUALITY_FAILED = "quality_failed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    COST_UPDATED = "cost_updated"
    NOTE_ADDED = "note_added"


# ============================================================
# Repair Work Order
# ============================================================


class Repair(Base):
    """
    Robot repair work order.

    Tracks repair workflow and associated records.
    Does not issue commands to physical robots.
    """

    __tablename__ = "repairs"

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "id",
            name="uq_repair_tenant_id",
        ),
        UniqueConstraint(
            "tenant_id",
            "repair_number",
            name="uq_repair_tenant_number",
        ),
        Index(
            "ix_repair_tenant_robot",
            "tenant_id",
            "robot_id",
        ),
        Index(
            "ix_repair_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_repair_tenant_priority",
            "tenant_id",
            "priority",
        ),
        Index(
            "ix_repair_tenant_technician",
            "tenant_id",
            "assigned_to",
        ),
        Index(
            "ix_repair_tenant_created",
            "tenant_id",
            "created_at",
        ),
        CheckConstraint(
            "estimated_cost IS NULL OR estimated_cost >= 0",
            name="ck_repair_estimated_cost",
        ),
        CheckConstraint(
            "actual_cost >= 0",
            name="ck_repair_actual_cost",
        ),
        CheckConstraint(
            "parts_cost >= 0",
            name="ck_repair_parts_cost",
        ),
        CheckConstraint(
            "labor_cost >= 0",
            name="ck_repair_labor_cost",
        ),
        CheckConstraint(
            "other_cost >= 0",
            name="ck_repair_other_cost",
        ),
        CheckConstraint(
            "downtime_minutes >= 0",
            name="ck_repair_downtime",
        ),
        CheckConstraint(
            "estimated_labor_hours IS NULL OR "
            "estimated_labor_hours >= 0",
            name="ck_repair_estimated_labor",
        ),
        CheckConstraint(
            "actual_labor_hours >= 0",
            name="ck_repair_actual_labor",
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
    )

    repair_number: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    robot_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    fleet_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    component_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    part_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        nullable=True,
    )

    incident_id: Mapped[Optional[str]] = mapped_column(
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

    repair_estimate_id: Mapped[Optional[str]] = mapped_column(
        String(36),
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

    repair_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default=RepairType.CORRECTIVE.value,
    )

    priority: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=RepairPriority.NORMAL.value,
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=RepairStatus.DRAFT.value,
    )

    diagnosis: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    root_cause: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    repair_procedure: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    resolution: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    failure_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    assigned_to: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    created_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    approved_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    completed_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    quality_reviewed_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    quality_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=RepairQualityStatus.PENDING.value,
    )

    currency: Mapped[str] = mapped_column(
        String(3),
        nullable=False,
        default="INR",
    )

    estimated_cost: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(16, 2),
        nullable=True,
    )

    actual_cost: Mapped[Decimal] = mapped_column(
        Numeric(16, 2),
        nullable=False,
        default=Decimal("0"),
    )

    parts_cost: Mapped[Decimal] = mapped_column(
        Numeric(16, 2),
        nullable=False,
        default=Decimal("0"),
    )

    labor_cost: Mapped[Decimal] = mapped_column(
        Numeric(16, 2),
        nullable=False,
        default=Decimal("0"),
    )

    other_cost: Mapped[Decimal] = mapped_column(
        Numeric(16, 2),
        nullable=False,
        default=Decimal("0"),
    )

    estimated_labor_hours: Mapped[Optional[Decimal]] = (
        mapped_column(
            Numeric(10, 2),
            nullable=True,
        )
    )

    actual_labor_hours: Mapped[Decimal] = mapped_column(
        Numeric(10, 2),
        nullable=False,
        default=Decimal("0"),
    )

    downtime_minutes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    requires_human_approval: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    requires_safety_verification: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    safety_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    is_warranty_repair: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    diagnosis_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    repair_plan_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    test_results_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    safety_verification_json: Mapped[
        dict[str, Any]
    ] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
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

    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    repair_completed_at: Mapped[Optional[datetime]] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
    )

    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    approved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    quality_reviewed_at: Mapped[Optional[datetime]] = (
        mapped_column(
            DateTime(timezone=True),
            nullable=True,
        )
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

    tasks: Mapped[list["RepairTask"]] = relationship(
        "RepairTask",
        back_populates="repair",
        cascade="save-update, merge",
        lazy="selectin",
    )

    parts: Mapped[list["RepairPart"]] = relationship(
        "RepairPart",
        back_populates="repair",
        cascade="save-update, merge",
        lazy="selectin",
    )

    events: Mapped[list["RepairEvent"]] = relationship(
        "RepairEvent",
        back_populates="repair",
        cascade="save-update, merge",
        lazy="selectin",
    )

    @validates(
        "estimated_cost",
        "actual_cost",
        "parts_cost",
        "labor_cost",
        "other_cost",
        "estimated_labor_hours",
        "actual_labor_hours",
    )
    def validate_amount(
        self,
        key: str,
        value: Any,
    ) -> Optional[Decimal]:
        if value is None and key in {
            "estimated_cost",
            "estimated_labor_hours",
        }:
            return None

        return validate_decimal(value, key)

    @property
    def is_active(self) -> bool:
        return self.status in {
            RepairStatus.DIAGNOSING.value,
            RepairStatus.WAITING_FOR_PARTS.value,
            RepairStatus.IN_PROGRESS.value,
            RepairStatus.PAUSED.value,
            RepairStatus.REPAIR_COMPLETED.value,
            RepairStatus.TESTING.value,
            RepairStatus.QUALITY_REVIEW.value,
        }

    @property
    def is_completed(self) -> bool:
        return self.status == RepairStatus.COMPLETED.value

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            RepairStatus.COMPLETED.value,
            RepairStatus.FAILED.value,
            RepairStatus.CANCELLED.value,
        }

    def approve(self, user_id: str) -> None:
        if self.status != RepairStatus.PENDING_APPROVAL.value:
            raise ValueError(
                "Repair is not pending approval"
            )

        if not user_id.strip():
            raise ValueError("Approver ID is required")

        now = utc_now()

        self.status = RepairStatus.APPROVED.value
        self.approved_by = user_id
        self.approved_at = now
        self.updated_at = now

    def start(self) -> None:
        if self.status not in {
            RepairStatus.APPROVED.value,
            RepairStatus.SCHEDULED.value,
        }:
            raise ValueError(
                "Repair must be approved or scheduled"
            )

        now = utc_now()

        self.status = RepairStatus.IN_PROGRESS.value
        self.started_at = self.started_at or now
        self.updated_at = now

    def pause(self) -> None:
        if self.status != RepairStatus.IN_PROGRESS.value:
            raise ValueError(
                "Only in-progress repairs can be paused"
            )

        self.status = RepairStatus.PAUSED.value
        self.updated_at = utc_now()

    def resume(self) -> None:
        if self.status != RepairStatus.PAUSED.value:
            raise ValueError(
                "Only paused repairs can resume"
            )

        self.status = RepairStatus.IN_PROGRESS.value
        self.updated_at = utc_now()

    def mark_repair_completed(self) -> None:
        if self.status != RepairStatus.IN_PROGRESS.value:
            raise ValueError(
                "Repair must be in progress"
            )

        now = utc_now()

        self.status = RepairStatus.REPAIR_COMPLETED.value
        self.repair_completed_at = now
        self.updated_at = now

    def start_testing(self) -> None:
        if self.status != RepairStatus.REPAIR_COMPLETED.value:
            raise ValueError(
                "Repair work must be completed before testing"
            )

        self.status = RepairStatus.TESTING.value
        self.updated_at = utc_now()

    def complete(
        self,
        user_id: str,
        resolution: str,
    ) -> None:
        if self.status not in {
            RepairStatus.TESTING.value,
            RepairStatus.QUALITY_REVIEW.value,
        }:
            raise ValueError(
                "Repair must pass testing and review"
            )

        if self.quality_status != RepairQualityStatus.PASSED.value:
            raise ValueError(
                "Quality review must pass"
            )

        if (
            self.requires_safety_verification
            and not self.safety_verified
        ):
            raise ValueError(
                "Safety verification is required"
            )

        if not user_id.strip() or not resolution.strip():
            raise ValueError(
                "Completer and resolution are required"
            )

        now = utc_now()

        self.status = RepairStatus.COMPLETED.value
        self.completed_by = user_id
        self.resolution = resolution.strip()
        self.completed_at = now
        self.updated_at = now

    def fail(self, reason: str) -> None:
        if self.is_terminal:
            raise ValueError(
                "Repair is already in a terminal state"
            )

        if not reason.strip():
            raise ValueError(
                "Failure reason is required"
            )

        self.status = RepairStatus.FAILED.value
        self.failure_reason = reason.strip()
        self.updated_at = utc_now()

    def recalculate_costs(self) -> Decimal:
        if self.is_terminal:
            raise ValueError(
                "Cannot recalculate finalized repair"
            )

        self.parts_cost = sum(
            (
                part.total_cost
                for part in self.parts
                if not part.is_deleted
            ),
            Decimal("0"),
        )

        self.actual_cost = (
            self.parts_cost
            + self.labor_cost
            + self.other_cost
        )

        self.updated_at = utc_now()
        return self.actual_cost

    def to_dict(
        self,
        *,
        include_tasks: bool = False,
        include_parts: bool = False,
    ) -> dict[str, Any]:
        result = {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "organization_id": self.organization_id,
            "repair_number": self.repair_number,
            "robot_id": self.robot_id,
            "fleet_id": self.fleet_id,
            "component_id": self.component_id,
            "part_id": self.part_id,
            "incident_id": self.incident_id,
            "failure_event_id": self.failure_event_id,
            "maintenance_id": self.maintenance_id,
            "repair_estimate_id": self.repair_estimate_id,
            "title": self.title,
            "description": self.description,
            "repair_type": self.repair_type,
            "priority": self.priority,
            "status": self.status,
            "diagnosis": self.diagnosis,
            "root_cause": self.root_cause,
            "repair_procedure": self.repair_procedure,
            "resolution": self.resolution,
            "failure_reason": self.failure_reason,
            "assigned_to": self.assigned_to,
            "approved_by": self.approved_by,
            "completed_by": self.completed_by,
            "quality_status": self.quality_status,
            "currency": self.currency,
            "estimated_cost": (
                str(self.estimated_cost)
                if self.estimated_cost is not None
                else None
            ),
            "actual_cost": str(self.actual_cost),
            "parts_cost": str(self.parts_cost),
            "labor_cost": str(self.labor_cost),
            "other_cost": str(self.other_cost),
            "estimated_labor_hours": (
                str(self.estimated_labor_hours)
                if self.estimated_labor_hours is not None
                else None
            ),
            "actual_labor_hours": str(
                self.actual_labor_hours
            ),
            "downtime_minutes": self.downtime_minutes,
            "requires_human_approval": (
                self.requires_human_approval
            ),
            "requires_safety_verification": (
                self.requires_safety_verification
            ),
            "safety_verified": self.safety_verified,
            "is_warranty_repair": self.is_warranty_repair,
            "is_deleted": self.is_deleted,
            "scheduled_at": serialize_datetime(
                self.scheduled_at
            ),
            "started_at": serialize_datetime(
                self.started_at
            ),
            "repair_completed_at": serialize_datetime(
                self.repair_completed_at
            ),
            "completed_at": serialize_datetime(
                self.completed_at
            ),
            "created_at": serialize_datetime(
                self.created_at
            ),
            "updated_at": serialize_datetime(
                self.updated_at
            ),
        }

        if include_tasks:
            result["tasks"] = [
                task.to_dict()
                for task in self.tasks
            ]

        if include_parts:
            result["parts"] = [
                part.to_dict()
                for part in self.parts
                if not part.is_deleted
            ]

        return result

    def __repr__(self) -> str:
        return (
            f"<Repair("
            f"id={self.id!r}, "
            f"repair_number={self.repair_number!r}, "
            f"status={self.status!r}"
            f")>"
        )


# ============================================================
# Repair Task
# ============================================================


class RepairTask(Base):
    __tablename__ = "repair_tasks"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "repair_id"],
            ["repairs.tenant_id", "repairs.id"],
            name="fk_repair_task_tenant_repair",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "tenant_id",
            "repair_id",
            "sequence_number",
            name="uq_repair_task_sequence",
        ),
        Index(
            "ix_repair_task_tenant_status",
            "tenant_id",
            "status",
        ),
        CheckConstraint(
            "sequence_number >= 1",
            name="ck_repair_task_sequence",
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

    repair_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    sequence_number: Mapped[int] = mapped_column(
        Integer,
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

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=RepairTaskStatus.PENDING.value,
    )

    assigned_to: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    completed_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    instructions_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    results_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[Optional[datetime]] = mapped_column(
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

    repair: Mapped["Repair"] = relationship(
        "Repair",
        back_populates="tasks",
    )

    def start(self) -> None:
        if self.status != RepairTaskStatus.PENDING.value:
            raise ValueError(
                "Only pending tasks can start"
            )

        self.status = RepairTaskStatus.IN_PROGRESS.value
        self.started_at = utc_now()
        self.updated_at = utc_now()

    def complete(
        self,
        user_id: str,
        results: Optional[dict[str, Any]] = None,
    ) -> None:
        if self.status != RepairTaskStatus.IN_PROGRESS.value:
            raise ValueError(
                "Only in-progress tasks can complete"
            )

        now = utc_now()

        self.status = RepairTaskStatus.COMPLETED.value
        self.completed_by = user_id
        self.completed_at = now
        self.results_json = results or {}
        self.updated_at = now

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "repair_id": self.repair_id,
            "sequence_number": self.sequence_number,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "assigned_to": self.assigned_to,
            "completed_by": self.completed_by,
            "instructions": self.instructions_json,
            "results": self.results_json,
            "started_at": serialize_datetime(
                self.started_at
            ),
            "completed_at": serialize_datetime(
                self.completed_at
            ),
        }


# ============================================================
# Repair Parts
# ============================================================


class RepairPart(Base):
    __tablename__ = "repair_parts"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "repair_id"],
            ["repairs.tenant_id", "repairs.id"],
            name="fk_repair_part_tenant_repair",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_repair_part_tenant_repair",
            "tenant_id",
            "repair_id",
        ),
        CheckConstraint(
            "quantity > 0",
            name="ck_repair_part_quantity",
        ),
        CheckConstraint(
            "unit_cost >= 0",
            name="ck_repair_part_unit_cost",
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

    repair_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    part_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    part_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    action: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=RepairPartAction.REPLACE.value,
    )

    quantity: Mapped[Decimal] = mapped_column(
        Numeric(12, 3),
        nullable=False,
        default=Decimal("1"),
    )

    unit_cost: Mapped[Decimal] = mapped_column(
        Numeric(16, 2),
        nullable=False,
        default=Decimal("0"),
    )

    supplier_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    inventory_transaction_id: Mapped[
        Optional[str]
    ] = mapped_column(
        String(36),
        nullable=True,
    )

    notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
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

    repair: Mapped["Repair"] = relationship(
        "Repair",
        back_populates="parts",
    )

    @validates("quantity", "unit_cost")
    def validate_amount(
        self,
        key: str,
        value: Any,
    ) -> Decimal:
        return validate_decimal(
            value,
            key,
            allow_zero=(key != "quantity"),
        )

    @property
    def total_cost(self) -> Decimal:
        return self.quantity * self.unit_cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "repair_id": self.repair_id,
            "part_id": self.part_id,
            "part_name": self.part_name,
            "action": self.action,
            "quantity": str(self.quantity),
            "unit_cost": str(self.unit_cost),
            "total_cost": str(self.total_cost),
            "supplier_id": self.supplier_id,
            "inventory_transaction_id": (
                self.inventory_transaction_id
            ),
            "notes": self.notes,
            "is_deleted": self.is_deleted,
            "created_at": serialize_datetime(
                self.created_at
            ),
        }


# ============================================================
# Repair Event History
# ============================================================


class RepairEvent(Base):
    """
    Append-only repair lifecycle event.

    Enforce immutability using database permissions
    or triggers in production.
    """

    __tablename__ = "repair_events"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "repair_id"],
            ["repairs.tenant_id", "repairs.id"],
            name="fk_repair_event_tenant_repair",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_repair_event_tenant_repair",
            "tenant_id",
            "repair_id",
        ),
        Index(
            "ix_repair_event_tenant_created",
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

    repair_id: Mapped[str] = mapped_column(
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

    repair: Mapped["Repair"] = relationship(
        "Repair",
        back_populates="events",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "repair_id": self.repair_id,
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


__all__ = [
    "Repair",
    "RepairTask",
    "RepairPart",
    "RepairEvent",
    "RepairStatus",
    "RepairType",
    "RepairPriority",
    "RepairTaskStatus",
    "RepairPartAction",
    "RepairQualityStatus",
    "RepairEventType",
    "utc_now",
    "generate_uuid",
    "serialize_datetime",
    "validate_decimal",
]
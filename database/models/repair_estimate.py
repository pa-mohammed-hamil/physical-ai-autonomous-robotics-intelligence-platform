"""Repair Estimate module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/db/models/repair_estimate.py

Physical AI & Autonomous Robotics Intelligence Platform

SQLAlchemy 2.x models for robot repair and replacement cost estimates.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

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
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.db.base import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_uuid() -> str:
    return str(uuid.uuid4())


def money(value: Any, field: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {field}") from exc
    if not result.is_finite() or result < 0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return result


def as_iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


class RepairEstimateStatus(str, Enum):
    DRAFT = "draft"
    PENDING_ANALYSIS = "pending_analysis"
    ANALYZING = "analyzing"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class RepairRecommendation(str, Enum):
    INSPECT = "inspect"
    REPAIR = "repair"
    REPLACE = "replace"
    REFURBISH = "refurbish"
    MONITOR = "monitor"
    RETIRE = "retire"
    UNDETERMINED = "undetermined"


class RepairPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class EstimateItemType(str, Enum):
    PART = "part"
    LABOR = "labor"
    DIAGNOSTIC = "diagnostic"
    SHIPPING = "shipping"
    SERVICE = "service"
    SOFTWARE = "software"
    CALIBRATION = "calibration"
    TESTING = "testing"
    OTHER = "other"


class RepairEstimateEventType(str, Enum):
    CREATED = "created"
    UPDATED = "updated"
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_COMPLETED = "analysis_completed"
    ITEM_ADDED = "item_added"
    ITEM_UPDATED = "item_updated"
    ITEM_REMOVED = "item_removed"
    COST_RECALCULATED = "cost_recalculated"
    REVIEW_REQUESTED = "review_requested"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    NOTE_ADDED = "note_added"


class RepairEstimate(Base):
    __tablename__ = "repair_estimates"

    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "id",
            name="uq_repair_estimate_tenant_id",
        ),
        UniqueConstraint(
            "tenant_id", "estimate_number",
            name="uq_repair_estimate_number",
        ),
        Index(
            "ix_repair_estimate_robot",
            "tenant_id", "robot_id",
        ),
        Index(
            "ix_repair_estimate_status",
            "tenant_id", "status",
        ),
        Index(
            "ix_repair_estimate_created",
            "tenant_id", "created_at",
        ),
        CheckConstraint(
            "estimated_labor_hours IS NULL OR estimated_labor_hours >= 0",
            name="ck_repair_estimate_labor_hours",
        ),
        CheckConstraint(
            "estimated_downtime_hours IS NULL OR estimated_downtime_hours >= 0",
            name="ck_repair_estimate_downtime_hours",
        ),
        CheckConstraint(
            "confidence_score IS NULL OR "
            "(confidence_score >= 0 AND confidence_score <= 1)",
            name="ck_repair_estimate_confidence",
        ),
        CheckConstraint(
            "parts_cost >= 0 AND labor_cost >= 0 AND "
            "diagnostic_cost >= 0 AND shipping_cost >= 0 AND "
            "other_cost >= 0 AND tax_amount >= 0 AND "
            "discount_amount >= 0 AND total_cost >= 0",
            name="ck_repair_estimate_costs",
        ),
        CheckConstraint(
            "replacement_cost IS NULL OR replacement_cost >= 0",
            name="ck_repair_estimate_replacement_cost",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid,
    )
    tenant_id: Mapped[str] = mapped_column(
        String(255), nullable=False,
    )
    organization_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True,
    )
    estimate_number: Mapped[str] = mapped_column(
        String(100), nullable=False,
    )

    robot_id: Mapped[str] = mapped_column(
        String(255), nullable=False,
    )
    fleet_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True,
    )
    part_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True,
    )
    component_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True,
    )
    incident_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True,
    )
    maintenance_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True,
    )
    inspection_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True,
    )

    title: Mapped[str] = mapped_column(
        String(255), nullable=False,
    )
    description: Mapped[str | None] = mapped_column(
        Text, nullable=True,
    )
    diagnosis: Mapped[str | None] = mapped_column(
        Text, nullable=True,
    )
    detected_issue: Mapped[str | None] = mapped_column(
        Text, nullable=True,
    )
    root_cause: Mapped[str | None] = mapped_column(
        Text, nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(32), nullable=False,
        default=RepairEstimateStatus.DRAFT.value,
    )
    priority: Mapped[str] = mapped_column(
        String(32), nullable=False,
        default=RepairPriority.NORMAL.value,
    )
    recommendation: Mapped[str] = mapped_column(
        String(32), nullable=False,
        default=RepairRecommendation.UNDETERMINED.value,
    )

    currency: Mapped[str] = mapped_column(
        String(3), nullable=False, default="INR",
    )

    parts_cost: Mapped[Decimal] = mapped_column(
        Numeric(16, 2), nullable=False, default=Decimal("0"),
    )
    labor_cost: Mapped[Decimal] = mapped_column(
        Numeric(16, 2), nullable=False, default=Decimal("0"),
    )
    diagnostic_cost: Mapped[Decimal] = mapped_column(
        Numeric(16, 2), nullable=False, default=Decimal("0"),
    )
    shipping_cost: Mapped[Decimal] = mapped_column(
        Numeric(16, 2), nullable=False, default=Decimal("0"),
    )
    other_cost: Mapped[Decimal] = mapped_column(
        Numeric(16, 2), nullable=False, default=Decimal("0"),
    )
    tax_amount: Mapped[Decimal] = mapped_column(
        Numeric(16, 2), nullable=False, default=Decimal("0"),
    )
    discount_amount: Mapped[Decimal] = mapped_column(
        Numeric(16, 2), nullable=False, default=Decimal("0"),
    )
    total_cost: Mapped[Decimal] = mapped_column(
        Numeric(16, 2), nullable=False, default=Decimal("0"),
    )
    replacement_cost: Mapped[Decimal | None] = mapped_column(
        Numeric(16, 2), nullable=True,
    )

    estimated_labor_hours: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 2), nullable=True,
    )
    estimated_downtime_hours: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 2), nullable=True,
    )
    confidence_score: Mapped[Decimal | None] = mapped_column(
        Numeric(5, 4), nullable=True,
    )

    ai_generated: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
    )
    requires_human_review: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True,
    )
    is_deleted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
    )

    model_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True,
    )
    model_version: Mapped[str | None] = mapped_column(
        String(100), nullable=True,
    )
    created_by: Mapped[str | None] = mapped_column(
        String(255), nullable=True,
    )
    reviewed_by: Mapped[str | None] = mapped_column(
        String(255), nullable=True,
    )
    review_notes: Mapped[str | None] = mapped_column(
        Text, nullable=True,
    )

    image_analysis_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict,
    )
    sensor_analysis_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict,
    )
    pricing_sources_json: Mapped[list[Any]] = mapped_column(
        JSON, nullable=False, default=list,
    )
    assumptions_json: Mapped[list[Any]] = mapped_column(
        JSON, nullable=False, default=list,
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict,
    )

    valid_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=utc_now, onupdate=utc_now,
    )

    items: Mapped[list["RepairEstimateItem"]] = relationship(
        "RepairEstimateItem",
        back_populates="estimate",
        cascade="save-update, merge",
        lazy="selectin",
        order_by="RepairEstimateItem.line_number",
    )
    events: Mapped[list["RepairEstimateEvent"]] = relationship(
        "RepairEstimateEvent",
        back_populates="estimate",
        cascade="save-update, merge",
        lazy="selectin",
    )

    @validates(
        "parts_cost", "labor_cost", "diagnostic_cost",
        "shipping_cost", "other_cost", "tax_amount",
        "discount_amount", "total_cost", "replacement_cost",
        "estimated_labor_hours", "estimated_downtime_hours",
        "confidence_score",
    )
    def validate_decimal(
        self, key: str, value: Any,
    ) -> Decimal | None:
        if value is None:
            if key in {
                "replacement_cost", "estimated_labor_hours",
                "estimated_downtime_hours", "confidence_score",
            }:
                return None
            raise ValueError(f"{key} cannot be null")

        result = money(value, key)
        if key == "confidence_score" and result > 1:
            raise ValueError("confidence_score must be between 0 and 1")
        return result

    @property
    def is_approved(self) -> bool:
        return self.status == RepairEstimateStatus.APPROVED.value

    @property
    def is_expired(self) -> bool:
        if self.status == RepairEstimateStatus.EXPIRED.value:
            return True
        if self.valid_until is None:
            return False
        expiry = self.valid_until
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        return utc_now() > expiry

    @property
    def repair_vs_replace(self) -> str:
        if self.replacement_cost is None:
            return RepairRecommendation.UNDETERMINED.value
        if self.total_cost < self.replacement_cost:
            return RepairRecommendation.REPAIR.value
        return RepairRecommendation.REPLACE.value

    def recalculate_totals(self) -> Decimal:
        if self.status in {
            RepairEstimateStatus.APPROVED.value,
            RepairEstimateStatus.CANCELLED.value,
            RepairEstimateStatus.REJECTED.value,
            RepairEstimateStatus.EXPIRED.value,
        }:
            raise ValueError("Cannot modify a finalized estimate")

        categories = {
            EstimateItemType.PART.value: Decimal("0"),
            EstimateItemType.LABOR.value: Decimal("0"),
            EstimateItemType.DIAGNOSTIC.value: Decimal("0"),
            EstimateItemType.SHIPPING.value: Decimal("0"),
        }
        other = Decimal("0")

        for item in self.items:
            if item.is_deleted:
                continue
            subtotal = money(item.quantity, "quantity") * money(
                item.unit_price, "unit_price"
            )
            if item.item_type in categories:
                categories[item.item_type] += subtotal
            else:
                other += subtotal

        self.parts_cost = categories[EstimateItemType.PART.value]
        self.labor_cost = categories[EstimateItemType.LABOR.value]
        self.diagnostic_cost = categories[
            EstimateItemType.DIAGNOSTIC.value
        ]
        self.shipping_cost = categories[
            EstimateItemType.SHIPPING.value
        ]
        self.other_cost = other

        gross = (
            self.parts_cost
            + self.labor_cost
            + self.diagnostic_cost
            + self.shipping_cost
            + self.other_cost
            + self.tax_amount
        )

        if self.discount_amount > gross:
            raise ValueError("Discount exceeds estimate amount")

        self.total_cost = gross - self.discount_amount
        self.updated_at = utc_now()
        return self.total_cost

    def request_review(self) -> None:
        if self.status not in {
            RepairEstimateStatus.DRAFT.value,
            RepairEstimateStatus.PENDING_ANALYSIS.value,
            RepairEstimateStatus.ANALYZING.value,
        }:
            raise ValueError("Estimate cannot enter review")
        self.status = RepairEstimateStatus.PENDING_REVIEW.value
        self.updated_at = utc_now()

    def approve(self, reviewer_id: str) -> None:
        if self.status != RepairEstimateStatus.PENDING_REVIEW.value:
            raise ValueError("Estimate is not pending review")
        if self.is_expired:
            raise ValueError("Cannot approve an expired estimate")
        if not reviewer_id.strip():
            raise ValueError("Reviewer ID is required")

        self.status = RepairEstimateStatus.APPROVED.value
        self.reviewed_by = reviewer_id
        self.reviewed_at = utc_now()
        self.updated_at = utc_now()

    def reject(self, reviewer_id: str, reason: str) -> None:
        if self.status != RepairEstimateStatus.PENDING_REVIEW.value:
            raise ValueError("Estimate is not pending review")
        if not reviewer_id.strip() or not reason.strip():
            raise ValueError("Reviewer and reason are required")

        self.status = RepairEstimateStatus.REJECTED.value
        self.reviewed_by = reviewer_id
        self.review_notes = reason
        self.reviewed_at = utc_now()
        self.updated_at = utc_now()

    def to_dict(self, include_items: bool = False) -> dict[str, Any]:
        result = {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "organization_id": self.organization_id,
            "estimate_number": self.estimate_number,
            "robot_id": self.robot_id,
            "fleet_id": self.fleet_id,
            "part_id": self.part_id,
            "component_id": self.component_id,
            "incident_id": self.incident_id,
            "maintenance_id": self.maintenance_id,
            "inspection_id": self.inspection_id,
            "title": self.title,
            "description": self.description,
            "diagnosis": self.diagnosis,
            "detected_issue": self.detected_issue,
            "root_cause": self.root_cause,
            "status": self.status,
            "priority": self.priority,
            "recommendation": self.recommendation,
            "currency": self.currency,
            "parts_cost": str(self.parts_cost),
            "labor_cost": str(self.labor_cost),
            "diagnostic_cost": str(self.diagnostic_cost),
            "shipping_cost": str(self.shipping_cost),
            "other_cost": str(self.other_cost),
            "tax_amount": str(self.tax_amount),
            "discount_amount": str(self.discount_amount),
            "total_cost": str(self.total_cost),
            "replacement_cost": (
                str(self.replacement_cost)
                if self.replacement_cost is not None else None
            ),
            "estimated_labor_hours": (
                str(self.estimated_labor_hours)
                if self.estimated_labor_hours is not None else None
            ),
            "estimated_downtime_hours": (
                str(self.estimated_downtime_hours)
                if self.estimated_downtime_hours is not None else None
            ),
            "confidence_score": (
                str(self.confidence_score)
                if self.confidence_score is not None else None
            ),
            "ai_generated": self.ai_generated,
            "requires_human_review": self.requires_human_review,
            "reviewed_by": self.reviewed_by,
            "review_notes": self.review_notes,
            "is_expired": self.is_expired,
            "is_deleted": self.is_deleted,
            "valid_until": as_iso(self.valid_until),
            "reviewed_at": as_iso(self.reviewed_at),
            "created_at": as_iso(self.created_at),
            "updated_at": as_iso(self.updated_at),
        }
        if include_items:
            result["items"] = [
                item.to_dict()
                for item in self.items
                if not item.is_deleted
            ]
        return result

    def __repr__(self) -> str:
        return (
            f"<RepairEstimate(id={self.id!r}, "
            f"estimate_number={self.estimate_number!r}, "
            f"status={self.status!r})>"
        )


class RepairEstimateItem(Base):
    __tablename__ = "repair_estimate_items"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "estimate_id"],
            ["repair_estimates.tenant_id", "repair_estimates.id"],
            name="fk_repair_estimate_item_parent",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "tenant_id", "estimate_id", "line_number",
            name="uq_repair_estimate_item_line",
        ),
        CheckConstraint(
            "line_number >= 1",
            name="ck_repair_estimate_item_line",
        ),
        CheckConstraint(
            "quantity > 0",
            name="ck_repair_estimate_item_quantity",
        ),
        CheckConstraint(
            "unit_price >= 0",
            name="ck_repair_estimate_item_price",
        ),
        Index(
            "ix_repair_estimate_item_parent",
            "tenant_id", "estimate_id",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid,
    )
    tenant_id: Mapped[str] = mapped_column(
        String(255), nullable=False,
    )
    estimate_id: Mapped[str] = mapped_column(
        String(36), nullable=False,
    )
    line_number: Mapped[int] = mapped_column(
        Integer, nullable=False,
    )
    item_type: Mapped[str] = mapped_column(
        String(32), nullable=False,
        default=EstimateItemType.PART.value,
    )
    part_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True,
    )
    description: Mapped[str] = mapped_column(
        Text, nullable=False,
    )
    quantity: Mapped[Decimal] = mapped_column(
        Numeric(12, 3), nullable=False, default=Decimal("1"),
    )
    unit: Mapped[str] = mapped_column(
        String(32), nullable=False, default="piece",
    )
    unit_price: Mapped[Decimal] = mapped_column(
        Numeric(16, 2), nullable=False, default=Decimal("0"),
    )
    supplier_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True,
    )
    supplier_quote_reference: Mapped[str | None] = mapped_column(
        String(255), nullable=True,
    )
    is_optional: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
    )
    is_deleted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=utc_now, onupdate=utc_now,
    )

    estimate: Mapped["RepairEstimate"] = relationship(
        "RepairEstimate",
        back_populates="items",
    )

    @validates("quantity", "unit_price")
    def validate_amount(self, key: str, value: Any) -> Decimal:
        result = money(value, key)
        if key == "quantity" and result <= 0:
            raise ValueError("Quantity must be positive")
        return result

    @property
    def subtotal(self) -> Decimal:
        return self.quantity * self.unit_price

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "estimate_id": self.estimate_id,
            "line_number": self.line_number,
            "item_type": self.item_type,
            "part_id": self.part_id,
            "description": self.description,
            "quantity": str(self.quantity),
            "unit": self.unit,
            "unit_price": str(self.unit_price),
            "subtotal": str(self.subtotal),
            "supplier_id": self.supplier_id,
            "supplier_quote_reference": self.supplier_quote_reference,
            "is_optional": self.is_optional,
            "is_deleted": self.is_deleted,
            "metadata": self.metadata_json,
            "created_at": as_iso(self.created_at),
        }


class RepairEstimateEvent(Base):
    __tablename__ = "repair_estimate_events"

    __table_args__ = (
        ForeignKeyConstraint(
            ["tenant_id", "estimate_id"],
            ["repair_estimates.tenant_id", "repair_estimates.id"],
            name="fk_repair_estimate_event_parent",
            ondelete="RESTRICT",
        ),
        Index(
            "ix_repair_estimate_event_parent_time",
            "tenant_id", "estimate_id", "created_at",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=generate_uuid,
    )
    tenant_id: Mapped[str] = mapped_column(
        String(255), nullable=False,
    )
    estimate_id: Mapped[str] = mapped_column(
        String(36), nullable=False,
    )
    event_type: Mapped[str] = mapped_column(
        String(100), nullable=False,
    )
    message: Mapped[str] = mapped_column(
        Text, nullable=False,
    )
    actor_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True,
    )
    previous_status: Mapped[str | None] = mapped_column(
        String(32), nullable=True,
    )
    new_status: Mapped[str | None] = mapped_column(
        String(32), nullable=True,
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now,
    )

    estimate: Mapped["RepairEstimate"] = relationship(
        "RepairEstimate",
        back_populates="events",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "estimate_id": self.estimate_id,
            "event_type": self.event_type,
            "message": self.message,
            "actor_id": self.actor_id,
            "previous_status": self.previous_status,
            "new_status": self.new_status,
            "metadata": self.metadata_json,
            "created_at": as_iso(self.created_at),
        }


__all__ = [
    "RepairEstimate",
    "RepairEstimateItem",
    "RepairEstimateEvent",
    "RepairEstimateStatus",
    "RepairRecommendation",
    "RepairPriority",
    "EstimateItemType",
    "RepairEstimateEventType",
]
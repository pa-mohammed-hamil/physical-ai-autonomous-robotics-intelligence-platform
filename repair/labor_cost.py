"""Labor Cost module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/labor_cost.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Robotics Labor Cost Estimation Engine

Python 3.11+
Standard library only.

Features:
    - Robot inspection labor costs
    - Maintenance and repair labor estimation
    - Installation and replacement labor
    - Technician hourly rates
    - Skill-level and labor-role support
    - Regular, overtime and emergency labor
    - Travel and on-site service costs
    - Multiple technicians per work order
    - Labor cost breakdown
    - Estimated versus actual labor costs
    - Labor budget variance
    - Multi-tenant isolation
    - Thread-safe work-order management
    - Audit logging
    - JSON report export
    - Runnable demonstration

Safety:
    - Financial planning and simulation only.
    - No robot commands or physical actuation.
    - Rates are supplied by the application.
    - No automatic wage or market-price retrieval.
    - Authorization flags are illustrative, not authentication.
"""

from __future__ import annotations

import json
import re
import threading
import uuid

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from typing import Any, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class LaborCostError(Exception):
    pass


class LaborValidationError(LaborCostError):
    pass


class LaborWorkOrderNotFound(LaborCostError):
    pass


class LaborConflict(LaborCostError):
    pass


class LaborPermissionDenied(LaborCostError):
    pass


# ============================================================
# ENUMS
# ============================================================

class LaborRole(str, Enum):
    ROBOTICS_TECHNICIAN = "robotics_technician"
    MECHANICAL_TECHNICIAN = "mechanical_technician"
    ELECTRICAL_TECHNICIAN = "electrical_technician"
    MAINTENANCE_ENGINEER = "maintenance_engineer"
    ROBOTICS_ENGINEER = "robotics_engineer"
    SOFTWARE_ENGINEER = "software_engineer"
    FIELD_SERVICE_ENGINEER = "field_service_engineer"
    SAFETY_INSPECTOR = "safety_inspector"
    QUALITY_INSPECTOR = "quality_inspector"
    SUPERVISOR = "supervisor"
    OTHER = "other"


class SkillLevel(str, Enum):
    JUNIOR = "junior"
    INTERMEDIATE = "intermediate"
    SENIOR = "senior"
    SPECIALIST = "specialist"


class LaborTaskType(str, Enum):
    INSPECTION = "inspection"
    DIAGNOSIS = "diagnosis"
    PREVENTIVE_MAINTENANCE = "preventive_maintenance"
    CORRECTIVE_MAINTENANCE = "corrective_maintenance"
    REPAIR = "repair"
    REPLACEMENT = "replacement"
    INSTALLATION = "installation"
    CALIBRATION = "calibration"
    TESTING = "testing"
    EMERGENCY_RESPONSE = "emergency_response"
    OTHER = "other"


class WorkOrderStatus(str, Enum):
    DRAFT = "draft"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class LaborEventType(str, Enum):
    WORK_ORDER_CREATED = "work_order_created"
    WORK_ORDER_APPROVED = "work_order_approved"
    WORK_ORDER_STARTED = "work_order_started"
    ACTUAL_HOURS_RECORDED = "actual_hours_recorded"
    WORK_ORDER_COMPLETED = "work_order_completed"
    WORK_ORDER_CANCELLED = "work_order_cancelled"


# ============================================================
# UTILITIES
# ============================================================

MONEY_QUANTUM = Decimal("0.01")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_id(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(
            r"[A-Za-z0-9_.-]{1,128}",
            value,
        ) is None
    ):
        raise LaborValidationError(
            f"Invalid {name}"
        )

    return value


def validate_text(
    value: str,
    name: str,
    maximum_length: int = 1024,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum_length
    ):
        raise LaborValidationError(
            f"Invalid {name}"
        )

    return value.strip()


def validate_currency(value: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Z]{3}", value) is None
    ):
        raise LaborValidationError(
            "Currency must contain three uppercase letters"
        )

    return value


def decimal_value(
    value: Any,
    name: str,
    *,
    minimum: Decimal = Decimal("0"),
) -> Decimal:
    if isinstance(value, bool):
        raise LaborValidationError(
            f"{name} must be numeric"
        )

    try:
        result = Decimal(str(value))
    except (
        InvalidOperation,
        ValueError,
        TypeError,
    ) as exc:
        raise LaborValidationError(
            f"Invalid {name}"
        ) from exc

    if (
        not result.is_finite()
        or result < minimum
        or result.adjusted() > 15
    ):
        raise LaborValidationError(
            f"{name} is outside the allowed range"
        )

    return result


def money(value: Decimal) -> Decimal:
    return value.quantize(
        MONEY_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def money_string(value: Decimal) -> str:
    return str(money(value))


def validate_hours(
    value: Any,
    name: str,
) -> Decimal:
    result = decimal_value(value, name)

    if result > Decimal("100000"):
        raise LaborValidationError(
            f"{name} exceeds the maximum"
        )

    return result


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class LaborCostConfig:
    maximum_work_orders: int = 100000
    maximum_labor_items: int = 500
    maximum_audit_events: int = 20000

    default_currency: str = "INR"

    default_overtime_multiplier: Decimal = Decimal("1.5")
    default_emergency_multiplier: Decimal = Decimal("2.0")

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_work_orders",
            "maximum_labor_items",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise LaborValidationError(
                    f"{name} must be a positive integer"
                )

        validate_currency(self.default_currency)

        for name in (
            "default_overtime_multiplier",
            "default_emergency_multiplier",
        ):
            value = decimal_value(
                getattr(self, name),
                name,
                minimum=Decimal("1"),
            )

            object.__setattr__(
                self,
                name,
                value,
            )

        if self.simulation_only is not True:
            raise LaborValidationError(
                "Only planning/simulation mode is supported"
            )


# ============================================================
# TECHNICIAN
# ============================================================

@dataclass(frozen=True)
class LaborTechnician:
    technician_id: str
    technician_name: str

    role: LaborRole
    skill_level: SkillLevel

    hourly_rate: Decimal
    currency: str = "INR"

    def __post_init__(self) -> None:
        validate_id(
            self.technician_id,
            "technician_id",
        )

        validate_text(
            self.technician_name,
            "technician_name",
            256,
        )

        if not isinstance(self.role, LaborRole):
            raise LaborValidationError(
                "Invalid labor role"
            )

        if not isinstance(
            self.skill_level,
            SkillLevel,
        ):
            raise LaborValidationError(
                "Invalid skill level"
            )

        validate_currency(self.currency)

        object.__setattr__(
            self,
            "hourly_rate",
            decimal_value(
                self.hourly_rate,
                "hourly_rate",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "technician_id": self.technician_id,
            "technician_name": self.technician_name,
            "role": self.role.value,
            "skill_level": self.skill_level.value,
            "hourly_rate": money_string(
                self.hourly_rate
            ),
            "currency": self.currency,
        }


# ============================================================
# LABOR HOURS
# ============================================================

@dataclass(frozen=True)
class LaborHours:
    regular_hours: Decimal = Decimal("0")
    overtime_hours: Decimal = Decimal("0")
    emergency_hours: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        for name in (
            "regular_hours",
            "overtime_hours",
            "emergency_hours",
        ):
            object.__setattr__(
                self,
                name,
                validate_hours(
                    getattr(self, name),
                    name,
                ),
            )

    @property
    def total_hours(self) -> Decimal:
        return (
            self.regular_hours
            + self.overtime_hours
            + self.emergency_hours
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "regular_hours": str(self.regular_hours),
            "overtime_hours": str(self.overtime_hours),
            "emergency_hours": str(self.emergency_hours),
            "total_hours": str(self.total_hours),
        }


# ============================================================
# LABOR ITEM
# ============================================================

@dataclass(frozen=True)
class LaborCostItem:
    item_id: str

    technician: LaborTechnician
    task_type: LaborTaskType

    estimated_hours: LaborHours

    overtime_multiplier: Decimal = Decimal("1.5")
    emergency_multiplier: Decimal = Decimal("2.0")

    travel_cost: Decimal = Decimal("0")
    onsite_service_fee: Decimal = Decimal("0")
    other_cost: Decimal = Decimal("0")

    description: str = ""

    def __post_init__(self) -> None:
        validate_id(self.item_id, "item_id")

        if not isinstance(
            self.technician,
            LaborTechnician,
        ):
            raise LaborValidationError(
                "Invalid technician"
            )

        if not isinstance(
            self.task_type,
            LaborTaskType,
        ):
            raise LaborValidationError(
                "Invalid task type"
            )

        if not isinstance(
            self.estimated_hours,
            LaborHours,
        ):
            raise LaborValidationError(
                "Invalid estimated hours"
            )

        for name in (
            "overtime_multiplier",
            "emergency_multiplier",
        ):
            object.__setattr__(
                self,
                name,
                decimal_value(
                    getattr(self, name),
                    name,
                    minimum=Decimal("1"),
                ),
            )

        for name in (
            "travel_cost",
            "onsite_service_fee",
            "other_cost",
        ):
            object.__setattr__(
                self,
                name,
                decimal_value(
                    getattr(self, name),
                    name,
                ),
            )

        if self.description:
            validate_text(
                self.description,
                "description",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "technician": self.technician.to_dict(),
            "task_type": self.task_type.value,
            "estimated_hours": (
                self.estimated_hours.to_dict()
            ),
            "overtime_multiplier": str(
                self.overtime_multiplier
            ),
            "emergency_multiplier": str(
                self.emergency_multiplier
            ),
            "travel_cost": money_string(
                self.travel_cost
            ),
            "onsite_service_fee": money_string(
                self.onsite_service_fee
            ),
            "other_cost": money_string(
                self.other_cost
            ),
            "description": self.description,
        }


# ============================================================
# WORK ORDER REQUEST
# ============================================================

@dataclass(frozen=True)
class LaborWorkOrderRequest:
    tenant_id: str
    robot_id: str
    component_id: str

    title: str
    labor_items: tuple[LaborCostItem, ...]

    currency: str = "INR"

    issue_reference: str = ""
    maintenance_reference: str = ""

    def __post_init__(self) -> None:
        for name in (
            "tenant_id",
            "robot_id",
            "component_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        validate_text(
            self.title,
            "title",
            256,
        )

        validate_currency(self.currency)

        if (
            not isinstance(self.labor_items, tuple)
            or not self.labor_items
        ):
            raise LaborValidationError(
                "At least one labor item is required"
            )

        seen = set()

        for item in self.labor_items:
            if not isinstance(
                item,
                LaborCostItem,
            ):
                raise LaborValidationError(
                    "Invalid labor item"
                )

            if item.item_id in seen:
                raise LaborValidationError(
                    "Duplicate labor item ID"
                )

            seen.add(item.item_id)

            if item.technician.currency != self.currency:
                raise LaborValidationError(
                    "Technician and work-order currencies differ"
                )

        for name in (
            "issue_reference",
            "maintenance_reference",
        ):
            value = getattr(self, name)

            if value:
                validate_id(value, name)


# ============================================================
# COST BREAKDOWN
# ============================================================

@dataclass(frozen=True)
class LaborCostBreakdown:
    currency: str

    regular_hours: Decimal
    overtime_hours: Decimal
    emergency_hours: Decimal

    regular_cost: Decimal
    overtime_cost: Decimal
    emergency_cost: Decimal

    travel_cost: Decimal
    onsite_service_cost: Decimal
    other_cost: Decimal

    total_hours: Decimal
    total_cost: Decimal

    def to_dict(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "regular_hours": str(self.regular_hours),
            "overtime_hours": str(self.overtime_hours),
            "emergency_hours": str(self.emergency_hours),
            "total_hours": str(self.total_hours),
            "regular_cost": money_string(
                self.regular_cost
            ),
            "overtime_cost": money_string(
                self.overtime_cost
            ),
            "emergency_cost": money_string(
                self.emergency_cost
            ),
            "travel_cost": money_string(
                self.travel_cost
            ),
            "onsite_service_cost": money_string(
                self.onsite_service_cost
            ),
            "other_cost": money_string(
                self.other_cost
            ),
            "total_cost": money_string(
                self.total_cost
            ),
        }


# ============================================================
# WORK ORDER RECORD
# ============================================================

@dataclass
class LaborWorkOrderRecord:
    work_order_id: str
    request: LaborWorkOrderRequest

    estimated_cost: LaborCostBreakdown

    status: WorkOrderStatus = WorkOrderStatus.DRAFT

    actual_hours: dict[
        str,
        LaborHours,
    ] = field(default_factory=dict)

    approved_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_order_id": self.work_order_id,
            "tenant_id": self.request.tenant_id,
            "robot_id": self.request.robot_id,
            "component_id": self.request.component_id,
            "title": self.request.title,
            "currency": self.request.currency,
            "status": self.status.value,
            "issue_reference": (
                self.request.issue_reference
            ),
            "maintenance_reference": (
                self.request.maintenance_reference
            ),
            "labor_items": [
                item.to_dict()
                for item in self.request.labor_items
            ],
            "estimated_cost": (
                self.estimated_cost.to_dict()
            ),
            "actual_hours": {
                item_id: hours.to_dict()
                for item_id, hours
                in self.actual_hours.items()
            },
            "approved_at": (
                self.approved_at.isoformat()
                if self.approved_at
                else None
            ),
            "started_at": (
                self.started_at.isoformat()
                if self.started_at
                else None
            ),
            "completed_at": (
                self.completed_at.isoformat()
                if self.completed_at
                else None
            ),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "notes": self.notes,
        }


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class LaborAuditEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    robot_id: str
    work_order_id: str

    event_type: LaborEventType
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "work_order_id": self.work_order_id,
            "event_type": self.event_type.value,
            "description": self.description,
        }


# ============================================================
# LABOR COST ENGINE
# ============================================================

class RoboticsLaborCostEngine:
    """
    Labor cost formula:

    Regular:
        regular_hours * hourly_rate

    Overtime:
        overtime_hours
        * hourly_rate
        * overtime_multiplier

    Emergency:
        emergency_hours
        * hourly_rate
        * emergency_multiplier

    Additional:
        travel_cost
        + onsite_service_fee
        + other_cost

    Total:
        regular + overtime + emergency + additional

    Each labor item represents one technician assignment.

    Overtime and emergency hours must be mutually
    exclusive to avoid double counting.
    """

    def __init__(
        self,
        config: LaborCostConfig | None = None,
    ) -> None:
        self.config = (
            config or LaborCostConfig()
        )

        self._work_orders: dict[
            str,
            LaborWorkOrderRecord,
        ] = {}

        self._audit_events: deque[
            LaborAuditEvent
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit(
        self,
        record: LaborWorkOrderRecord,
        event_type: LaborEventType,
        description: str,
    ) -> None:
        self._audit_events.append(
            LaborAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                tenant_id=record.request.tenant_id,
                robot_id=record.request.robot_id,
                work_order_id=record.work_order_id,
                event_type=event_type,
                description=description,
            )
        )

    # --------------------------------------------------------
    # INTERNAL LOOKUP
    # --------------------------------------------------------

    def _get(
        self,
        *,
        tenant_id: str,
        work_order_id: str,
    ) -> LaborWorkOrderRecord:
        validate_id(tenant_id, "tenant_id")
        validate_id(work_order_id, "work_order_id")

        record = self._work_orders.get(
            work_order_id
        )

        if (
            record is None
            or record.request.tenant_id != tenant_id
        ):
            raise LaborWorkOrderNotFound(
                "Labor work order not found"
            )

        return record

    # --------------------------------------------------------
    # CALCULATE LABOR COST
    # --------------------------------------------------------

    @staticmethod
    def calculate_labor_cost(
        items: Sequence[LaborCostItem],
        *,
        currency: str,
        actual_hours: dict[
            str,
            LaborHours,
        ] | None = None,
    ) -> LaborCostBreakdown:
        validate_currency(currency)

        regular_hours = Decimal("0")
        overtime_hours = Decimal("0")
        emergency_hours = Decimal("0")

        regular_cost = Decimal("0")
        overtime_cost = Decimal("0")
        emergency_cost = Decimal("0")

        travel_cost = Decimal("0")
        onsite_service_cost = Decimal("0")
        other_cost = Decimal("0")

        actual_hours = (
            actual_hours
            if actual_hours is not None
            else {}
        )

        for item in items:
            if not isinstance(
                item,
                LaborCostItem,
            ):
                raise LaborValidationError(
                    "Invalid labor cost item"
                )

            if item.technician.currency != currency:
                raise LaborValidationError(
                    "Currency mismatch"
                )

            hours = actual_hours.get(
                item.item_id,
                item.estimated_hours,
            )

            if not isinstance(hours, LaborHours):
                raise LaborValidationError(
                    "Invalid labor hours"
                )

            rate = item.technician.hourly_rate

            regular_hours += hours.regular_hours
            overtime_hours += hours.overtime_hours
            emergency_hours += hours.emergency_hours

            regular_cost += (
                hours.regular_hours * rate
            )

            overtime_cost += (
                hours.overtime_hours
                * rate
                * item.overtime_multiplier
            )

            emergency_cost += (
                hours.emergency_hours
                * rate
                * item.emergency_multiplier
            )

            travel_cost += item.travel_cost
            onsite_service_cost += (
                item.onsite_service_fee
            )
            other_cost += item.other_cost

        total_hours = (
            regular_hours
            + overtime_hours
            + emergency_hours
        )

        total_cost = sum(
            (
                regular_cost,
                overtime_cost,
                emergency_cost,
                travel_cost,
                onsite_service_cost,
                other_cost,
            ),
            Decimal("0"),
        )

        return LaborCostBreakdown(
            currency=currency,
            regular_hours=regular_hours,
            overtime_hours=overtime_hours,
            emergency_hours=emergency_hours,
            regular_cost=regular_cost,
            overtime_cost=overtime_cost,
            emergency_cost=emergency_cost,
            travel_cost=travel_cost,
            onsite_service_cost=onsite_service_cost,
            other_cost=other_cost,
            total_hours=total_hours,
            total_cost=total_cost,
        )

    # --------------------------------------------------------
    # CREATE WORK ORDER
    # --------------------------------------------------------

    def create_work_order(
        self,
        request: LaborWorkOrderRequest,
    ) -> dict[str, Any]:
        if not isinstance(
            request,
            LaborWorkOrderRequest,
        ):
            raise LaborValidationError(
                "Invalid work-order request"
            )

        if (
            len(request.labor_items)
            > self.config.maximum_labor_items
        ):
            raise LaborValidationError(
                "Too many labor items"
            )

        estimate = self.calculate_labor_cost(
            request.labor_items,
            currency=request.currency,
        )

        with self._lock:
            if (
                len(self._work_orders)
                >= self.config.maximum_work_orders
            ):
                raise LaborCostError(
                    "Work-order capacity reached"
                )

            record = LaborWorkOrderRecord(
                work_order_id=str(uuid.uuid4()),
                request=request,
                estimated_cost=estimate,
            )

            self._work_orders[
                record.work_order_id
            ] = record

            self._audit(
                record,
                LaborEventType.WORK_ORDER_CREATED,
                "Labor work order created",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # APPROVE WORK ORDER
    # --------------------------------------------------------

    def approve_work_order(
        self,
        *,
        tenant_id: str,
        work_order_id: str,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise LaborPermissionDenied(
                "Approval authorization required"
            )

        with self._lock:
            record = self._get(
                tenant_id=tenant_id,
                work_order_id=work_order_id,
            )

            if record.status != WorkOrderStatus.DRAFT:
                raise LaborConflict(
                    "Only draft work orders can be approved"
                )

            record.status = WorkOrderStatus.APPROVED
            record.approved_at = utc_now()
            record.updated_at = utc_now()

            self._audit(
                record,
                LaborEventType.WORK_ORDER_APPROVED,
                "Work order approved",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # START WORK ORDER
    # --------------------------------------------------------

    def start_work_order(
        self,
        *,
        tenant_id: str,
        work_order_id: str,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise LaborPermissionDenied(
                "Start authorization required"
            )

        with self._lock:
            record = self._get(
                tenant_id=tenant_id,
                work_order_id=work_order_id,
            )

            if record.status != WorkOrderStatus.APPROVED:
                raise LaborConflict(
                    "Work order must be approved first"
                )

            record.status = WorkOrderStatus.IN_PROGRESS
            record.started_at = utc_now()
            record.updated_at = utc_now()

            self._audit(
                record,
                LaborEventType.WORK_ORDER_STARTED,
                "Work order started",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # RECORD ACTUAL HOURS
    # --------------------------------------------------------

    def record_actual_hours(
        self,
        *,
        tenant_id: str,
        work_order_id: str,
        item_id: str,
        hours: LaborHours,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise LaborPermissionDenied(
                "Time-entry authorization required"
            )

        validate_id(item_id, "item_id")

        if not isinstance(hours, LaborHours):
            raise LaborValidationError(
                "Invalid actual hours"
            )

        with self._lock:
            record = self._get(
                tenant_id=tenant_id,
                work_order_id=work_order_id,
            )

            if (
                record.status
                != WorkOrderStatus.IN_PROGRESS
            ):
                raise LaborConflict(
                    "Work order is not in progress"
                )

            valid_ids = {
                item.item_id
                for item in record.request.labor_items
            }

            if item_id not in valid_ids:
                raise LaborValidationError(
                    "Labor item not found"
                )

            # Replaces the cumulative total for this item.
            # Repeated submissions are not added together.
            record.actual_hours[item_id] = hours
            record.updated_at = utc_now()

            self._audit(
                record,
                LaborEventType.ACTUAL_HOURS_RECORDED,
                f"Actual hours updated for {item_id}",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # COMPLETE WORK ORDER
    # --------------------------------------------------------

    def complete_work_order(
        self,
        *,
        tenant_id: str,
        work_order_id: str,
        notes: str,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise LaborPermissionDenied(
                "Completion authorization required"
            )

        notes = validate_text(
            notes,
            "notes",
            2048,
        )

        with self._lock:
            record = self._get(
                tenant_id=tenant_id,
                work_order_id=work_order_id,
            )

            if (
                record.status
                != WorkOrderStatus.IN_PROGRESS
            ):
                raise LaborConflict(
                    "Work order is not in progress"
                )

            item_ids = {
                item.item_id
                for item in record.request.labor_items
            }

            if set(record.actual_hours) != item_ids:
                raise LaborConflict(
                    "Actual hours must be recorded "
                    "for every labor item"
                )

            record.status = WorkOrderStatus.COMPLETED
            record.completed_at = utc_now()
            record.updated_at = utc_now()
            record.notes = notes

            self._audit(
                record,
                LaborEventType.WORK_ORDER_COMPLETED,
                "Labor work order completed",
            )

            return record.to_dict()

    # --------------------------------------------------------
    # CANCEL WORK ORDER
    # --------------------------------------------------------

    def cancel_work_order(
        self,
        *,
        tenant_id: str,
        work_order_id: str,
        reason: str,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise LaborPermissionDenied(
                "Cancellation authorization required"
            )

        reason = validate_text(
            reason,
            "reason",
        )

        with self._lock:
            record = self._get(
                tenant_id=tenant_id,
                work_order_id=work_order_id,
            )

            if record.status in (
                WorkOrderStatus.COMPLETED,
                WorkOrderStatus.CANCELLED,
            ):
                raise LaborConflict(
                    "Cannot cancel finalized work order"
                )

            record.status = WorkOrderStatus.CANCELLED
            record.updated_at = utc_now()
            record.notes = reason

            self._audit(
                record,
                LaborEventType.WORK_ORDER_CANCELLED,
                reason,
            )

            return record.to_dict()

    # --------------------------------------------------------
    # COST COMPARISON
    # --------------------------------------------------------

    def compare_estimated_actual(
        self,
        *,
        tenant_id: str,
        work_order_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id=tenant_id,
                work_order_id=work_order_id,
            )

            if record.status != WorkOrderStatus.COMPLETED:
                raise LaborConflict(
                    "Actual cost comparison requires "
                    "a completed work order"
                )

            actual = self.calculate_labor_cost(
                record.request.labor_items,
                currency=record.request.currency,
                actual_hours=record.actual_hours,
            )

            estimated = record.estimated_cost

            variance = (
                actual.total_cost
                - estimated.total_cost
            )

            variance_percentage = (
                variance
                / estimated.total_cost
                * Decimal("100")
                if estimated.total_cost > 0
                else None
            )

            return {
                "work_order_id": work_order_id,
                "tenant_id": tenant_id,
                "robot_id": record.request.robot_id,
                "currency": record.request.currency,
                "estimated": estimated.to_dict(),
                "actual": actual.to_dict(),
                "cost_variance": money_string(
                    variance
                ),
                "variance_percentage": (
                    str(
                        variance_percentage.quantize(
                            Decimal("0.01"),
                            rounding=ROUND_HALF_UP,
                        )
                    )
                    if variance_percentage is not None
                    else None
                ),
                "over_budget": variance > 0,
            }

    # --------------------------------------------------------
    # GET WORK ORDER
    # --------------------------------------------------------

    def get_work_order(
        self,
        *,
        tenant_id: str,
        work_order_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            return self._get(
                tenant_id=tenant_id,
                work_order_id=work_order_id,
            ).to_dict()

    # --------------------------------------------------------
    # LIST WORK ORDERS
    # --------------------------------------------------------

    def list_work_orders(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
        status: WorkOrderStatus | None = None,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        if robot_id is not None:
            validate_id(robot_id, "robot_id")

        if (
            status is not None
            and not isinstance(status, WorkOrderStatus)
        ):
            raise LaborValidationError(
                "Invalid status"
            )

        with self._lock:
            return [
                record.to_dict()
                for record in self._work_orders.values()
                if (
                    record.request.tenant_id == tenant_id
                    and (
                        robot_id is None
                        or record.request.robot_id == robot_id
                    )
                    and (
                        status is None
                        or record.status == status
                    )
                )
            ]

    # --------------------------------------------------------
    # ROBOT LABOR SUMMARY
    # --------------------------------------------------------

    def robot_labor_summary(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        validate_id(tenant_id, "tenant_id")
        validate_id(robot_id, "robot_id")

        with self._lock:
            records = [
                record
                for record in self._work_orders.values()
                if (
                    record.request.tenant_id == tenant_id
                    and record.request.robot_id == robot_id
                )
            ]

            totals: dict[
                str,
                dict[str, Decimal],
            ] = defaultdict(
                lambda: defaultdict(
                    lambda: Decimal("0")
                )
            )

            status_counts: dict[str, int] = defaultdict(int)

            for record in records:
                status_counts[
                    record.status.value
                ] += 1

                currency = record.request.currency

                totals[currency]["estimated_cost"] += (
                    record.estimated_cost.total_cost
                )

                totals[currency]["estimated_hours"] += (
                    record.estimated_cost.total_hours
                )

                if record.status == WorkOrderStatus.COMPLETED:
                    actual = self.calculate_labor_cost(
                        record.request.labor_items,
                        currency=currency,
                        actual_hours=record.actual_hours,
                    )

                    totals[currency]["actual_cost"] += (
                        actual.total_cost
                    )

                    totals[currency]["actual_hours"] += (
                        actual.total_hours
                    )

            return {
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "work_order_count": len(records),
                "status_counts": dict(status_counts),
                "totals_by_currency": {
                    currency: {
                        "estimated_cost": money_string(
                            values["estimated_cost"]
                        ),
                        "actual_cost_completed": money_string(
                            values["actual_cost"]
                        ),
                        "estimated_hours": str(
                            values["estimated_hours"]
                        ),
                        "actual_hours_completed": str(
                            values["actual_hours"]
                        ),
                    }
                    for currency, values in totals.items()
                },
                "simulation_only": True,
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
        validate_id(tenant_id, "tenant_id")

        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            payload = {
                "generated_at": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "work_orders": self.list_work_orders(
                    tenant_id=tenant_id
                ),
                "audit_events": [
                    event.to_dict()
                    for event in self._audit_events
                    if event.tenant_id == tenant_id
                ],
                "simulation_only": True,
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
    engine = RoboticsLaborCostEngine()

    technician_1 = LaborTechnician(
        technician_id="tech_001",
        technician_name="Mechanical Technician",
        role=LaborRole.MECHANICAL_TECHNICIAN,
        skill_level=SkillLevel.SENIOR,
        hourly_rate=Decimal("750"),
        currency="INR",
    )

    technician_2 = LaborTechnician(
        technician_id="tech_002",
        technician_name="Robotics Engineer",
        role=LaborRole.ROBOTICS_ENGINEER,
        skill_level=SkillLevel.SPECIALIST,
        hourly_rate=Decimal("1200"),
        currency="INR",
    )

    request = LaborWorkOrderRequest(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        component_id="joint_001",
        title="Robot Arm Joint Repair",
        currency="INR",
        issue_reference="issue_001",
        labor_items=(
            LaborCostItem(
                item_id="labor_001",
                technician=technician_1,
                task_type=LaborTaskType.REPAIR,
                estimated_hours=LaborHours(
                    regular_hours=Decimal("4"),
                    overtime_hours=Decimal("1"),
                ),
                travel_cost=Decimal("500"),
                onsite_service_fee=Decimal("300"),
                description="Joint repair and mechanical inspection",
            ),
            LaborCostItem(
                item_id="labor_002",
                technician=technician_2,
                task_type=LaborTaskType.CALIBRATION,
                estimated_hours=LaborHours(
                    regular_hours=Decimal("2"),
                ),
                description="Robot joint calibration and testing",
            ),
        ),
    )

    work_order = engine.create_work_order(
        request
    )

    work_order_id = work_order["work_order_id"]

    engine.approve_work_order(
        tenant_id="tenant_alpha",
        work_order_id=work_order_id,
        authorized=True,
    )

    engine.start_work_order(
        tenant_id="tenant_alpha",
        work_order_id=work_order_id,
        authorized=True,
    )

    engine.record_actual_hours(
        tenant_id="tenant_alpha",
        work_order_id=work_order_id,
        item_id="labor_001",
        hours=LaborHours(
            regular_hours=Decimal("5"),
            overtime_hours=Decimal("1"),
        ),
        authorized=True,
    )

    engine.record_actual_hours(
        tenant_id="tenant_alpha",
        work_order_id=work_order_id,
        item_id="labor_002",
        hours=LaborHours(
            regular_hours=Decimal("2.5"),
        ),
        authorized=True,
    )

    engine.complete_work_order(
        tenant_id="tenant_alpha",
        work_order_id=work_order_id,
        notes="Simulated repair and calibration completed",
        authorized=True,
    )

    comparison = engine.compare_estimated_actual(
        tenant_id="tenant_alpha",
        work_order_id=work_order_id,
    )

    print(
        json.dumps(
            comparison,
            indent=2,
        )
    )

    summary = engine.robot_labor_summary(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
    )

    print(
        "\nRobot labor summary:"
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    report = engine.export_json(
        "data/robotics/labor_cost_report.json",
        tenant_id="tenant_alpha",
    )

    print("\nReport:", report)


if __name__ == "__main__":
    main()
"""Repair Report module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/repair_report.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Robot Repair Report & Maintenance Intelligence Engine

Python 3.11+
Dependencies: Standard library only

Features:
    - Robot repair inspection reports
    - Component fault and damage reporting
    - Diagnostic evidence and confidence tracking
    - Image inspection reference integration
    - Repair work-order summaries
    - Parts replacement records
    - Labor, parts and downtime cost breakdowns
    - Repair vs replacement decision integration
    - Technician recommendations
    - Repair progress tracking
    - Safety-critical review workflow
    - Before/after repair condition records
    - Maintenance recommendations
    - Repair quality verification
    - Report approval and finalization
    - Multi-tenant isolation
    - Thread-safe in-memory storage
    - Audit logging
    - JSON, Markdown and HTML report export

Integrations:
    - issue_diagnosis.py
    - damage_analysis.py
    - damage_severity.py
    - image_inspection.py
    - part_detection.py
    - part_identification.py
    - parts_catalog.py
    - parts_pricing.py
    - labor_cost.py
    - downtime_cost.py
    - repair_vs_replace.py
    - maintenance_service.py
    - repair_service.py

Safety:
    Reports are advisory records.

    Final approval must be performed by an authorized
    reviewer. A finalized report is not permission to
    operate a robot or execute a repair.

    Diagnostic findings, component compatibility,
    repairs and safety tests must be verified by
    qualified personnel.
"""

from __future__ import annotations

import html
import json
import threading
import uuid

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class RepairReportError(Exception):
    pass


class ReportValidationError(RepairReportError):
    pass


class ReportNotFound(RepairReportError):
    pass


class ReportConflict(RepairReportError):
    pass


class ReportPermissionDenied(RepairReportError):
    pass


# ============================================================
# ENUMS
# ============================================================

class ReportStatus(str, Enum):
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    FINALIZED = "finalized"


class RepairPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class DamageSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class RepairAction(str, Enum):
    INSPECT = "inspect"
    CLEAN = "clean"
    ADJUST = "adjust"
    CALIBRATE = "calibrate"
    REPAIR = "repair"
    REPLACE = "replace"
    TEST = "test"
    MONITOR = "monitor"
    OTHER = "other"


class WorkStatus(str, Enum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class VerificationStatus(str, Enum):
    NOT_TESTED = "not_tested"
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class CostCategory(str, Enum):
    PARTS = "parts"
    LABOR = "labor"
    DOWNTIME = "downtime"
    SHIPPING = "shipping"
    INSPECTION = "inspection"
    TAX = "tax"
    OTHER = "other"


class RecommendationType(str, Enum):
    REPAIR = "repair"
    REPLACE = "replace"
    INSPECT = "inspect"
    MONITOR = "monitor"
    ESCALATE = "escalate"
    NO_ACTION = "no_action"


# ============================================================
# UTILITIES
# ============================================================

MONEY_QUANTUM = Decimal("0.01")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


def validate_id(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 128
        or not all(
            character.isalnum() or character in "._-"
            for character in value
        )
    ):
        raise ReportValidationError(f"Invalid {name}")
    return value


def validate_text(
    value: str,
    name: str,
    *,
    allow_empty: bool = False,
    maximum: int = 4000,
) -> str:
    if not isinstance(value, str):
        raise ReportValidationError(
            f"{name} must be a string"
        )

    result = value.strip()

    if len(result) > maximum or (
        not allow_empty and not result
    ):
        raise ReportValidationError(f"Invalid {name}")

    return result


def decimal_value(
    value: Any,
    name: str,
    *,
    maximum: Decimal = Decimal("1000000000000"),
) -> Decimal:
    if isinstance(value, bool):
        raise ReportValidationError(f"Invalid {name}")

    try:
        result = Decimal(str(value))
    except (
        InvalidOperation,
        TypeError,
        ValueError,
    ) as exc:
        raise ReportValidationError(
            f"Invalid {name}"
        ) from exc

    if (
        not result.is_finite()
        or result < 0
        or result > maximum
    ):
        raise ReportValidationError(
            f"{name} must be a finite nonnegative number"
        )

    return result


def money(value: Any, name: str = "amount") -> Decimal:
    return decimal_value(value, name).quantize(
        MONEY_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def money_str(value: Decimal) -> str:
    return str(
        value.quantize(
            MONEY_QUANTUM,
            rounding=ROUND_HALF_UP,
        )
    )


def validate_confidence(value: Any) -> Decimal:
    return decimal_value(
        value,
        "confidence",
        maximum=Decimal("1"),
    )


def validate_currency(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 3
        or not value.isascii()
        or not value.isalpha()
        or value != value.upper()
    ):
        raise ReportValidationError(
            "Currency must be a three-letter uppercase code"
        )
    return value


def validate_timestamp(
    value: datetime,
    name: str,
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ReportValidationError(
            f"{name} must be timezone-aware"
        )
    return value.astimezone(timezone.utc)


def validate_positive_integer(
    value: int,
    name: str,
    *,
    allow_zero: bool = False,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or (value == 0 and not allow_zero)
        or value > 1_000_000
    ):
        raise ReportValidationError(f"Invalid {name}")
    return value


def enum_value(value: Enum | str) -> str:
    return value.value if isinstance(value, Enum) else str(value)


def iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def clean_filename(value: str) -> str:
    return "".join(
        character
        if character.isalnum() or character in "._-"
        else "_"
        for character in value
    )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RepairReportConfig:
    default_currency: str = "INR"

    maximum_reports: int = 100000
    maximum_audit_events: int = 50000

    maximum_findings_per_report: int = 500
    maximum_tasks_per_report: int = 500
    maximum_parts_per_report: int = 500
    maximum_cost_items_per_report: int = 500
    maximum_evidence_per_report: int = 500

    require_verification_for_approval: bool = True
    require_reviewer_for_finalization: bool = True

    simulation_only: bool = True

    def __post_init__(self) -> None:
        validate_currency(self.default_currency)

        for name in (
            "maximum_reports",
            "maximum_audit_events",
            "maximum_findings_per_report",
            "maximum_tasks_per_report",
            "maximum_parts_per_report",
            "maximum_cost_items_per_report",
            "maximum_evidence_per_report",
        ):
            validate_positive_integer(
                getattr(self, name),
                name,
            )

        if self.simulation_only is not True:
            raise ReportValidationError(
                "Only advisory report mode is supported"
            )


# ============================================================
# ROBOT INFORMATION
# ============================================================

@dataclass(frozen=True)
class RepairRobotProfile:
    tenant_id: str
    robot_id: str
    robot_name: str

    manufacturer: str = ""
    model: str = ""
    serial_number: str = ""
    location: str = ""

    safety_critical: bool = False

    def __post_init__(self) -> None:
        validate_id(self.tenant_id, "tenant_id")
        validate_id(self.robot_id, "robot_id")
        validate_text(self.robot_name, "robot_name")

        for name in (
            "manufacturer",
            "model",
            "serial_number",
            "location",
        ):
            validate_text(
                getattr(self, name),
                name,
                allow_empty=True,
            )

        if not isinstance(self.safety_critical, bool):
            raise ReportValidationError(
                "safety_critical must be boolean"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "robot_name": self.robot_name,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "serial_number": self.serial_number,
            "location": self.location,
            "safety_critical": self.safety_critical,
        }


# ============================================================
# DIAGNOSTIC FINDING
# ============================================================

@dataclass(frozen=True)
class RepairFinding:
    finding_id: str
    component_id: str
    component_name: str

    issue_type: str
    description: str

    severity: DamageSeverity = DamageSeverity.UNKNOWN
    confidence: Decimal = Decimal("0")

    possible_cause: str = ""
    recommended_action: RecommendationType = (
        RecommendationType.INSPECT
    )

    diagnosis_reference: str = ""
    verified_by_human: bool = False

    def __post_init__(self) -> None:
        validate_id(self.finding_id, "finding_id")
        validate_id(self.component_id, "component_id")

        validate_text(
            self.component_name,
            "component_name",
        )
        validate_text(self.issue_type, "issue_type")
        validate_text(self.description, "description")

        validate_text(
            self.possible_cause,
            "possible_cause",
            allow_empty=True,
        )

        validate_text(
            self.diagnosis_reference,
            "diagnosis_reference",
            allow_empty=True,
        )

        object.__setattr__(
            self,
            "confidence",
            validate_confidence(self.confidence),
        )

        if not isinstance(self.severity, DamageSeverity):
            raise ReportValidationError(
                "Invalid damage severity"
            )

        if not isinstance(
            self.recommended_action,
            RecommendationType,
        ):
            raise ReportValidationError(
                "Invalid recommendation"
            )

        if not isinstance(
            self.verified_by_human,
            bool,
        ):
            raise ReportValidationError(
                "verified_by_human must be boolean"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "component_id": self.component_id,
            "component_name": self.component_name,
            "issue_type": self.issue_type,
            "description": self.description,
            "severity": self.severity.value,
            "confidence": str(self.confidence),
            "possible_cause": self.possible_cause,
            "recommended_action": (
                self.recommended_action.value
            ),
            "diagnosis_reference": (
                self.diagnosis_reference
            ),
            "verified_by_human": (
                self.verified_by_human
            ),
        }


# ============================================================
# INSPECTION EVIDENCE
# ============================================================

@dataclass(frozen=True)
class RepairEvidence:
    evidence_id: str
    evidence_type: str
    reference: str

    description: str = ""
    component_id: str = ""

    captured_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_id(self.evidence_id, "evidence_id")
        validate_text(self.evidence_type, "evidence_type")
        validate_text(self.reference, "reference")

        validate_text(
            self.description,
            "description",
            allow_empty=True,
        )

        if self.component_id:
            validate_id(self.component_id, "component_id")

        validate_timestamp(
            self.captured_at,
            "captured_at",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "reference": self.reference,
            "description": self.description,
            "component_id": self.component_id,
            "captured_at": iso(self.captured_at),
        }


# ============================================================
# REPAIR TASK
# ============================================================

@dataclass(frozen=True)
class RepairTask:
    task_id: str
    component_id: str

    action: RepairAction
    description: str

    status: WorkStatus = WorkStatus.PLANNED

    technician_id: str = ""
    estimated_hours: Decimal = Decimal("0")
    actual_hours: Decimal = Decimal("0")

    notes: str = ""

    def __post_init__(self) -> None:
        validate_id(self.task_id, "task_id")
        validate_id(self.component_id, "component_id")

        if not isinstance(self.action, RepairAction):
            raise ReportValidationError(
                "Invalid repair action"
            )

        if not isinstance(self.status, WorkStatus):
            raise ReportValidationError(
                "Invalid work status"
            )

        validate_text(self.description, "description")

        if self.technician_id:
            validate_id(self.technician_id, "technician_id")

        decimal_value(
            self.estimated_hours,
            "estimated_hours",
            maximum=Decimal("100000"),
        )

        decimal_value(
            self.actual_hours,
            "actual_hours",
            maximum=Decimal("100000"),
        )

        validate_text(
            self.notes,
            "notes",
            allow_empty=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "component_id": self.component_id,
            "action": self.action.value,
            "description": self.description,
            "status": self.status.value,
            "technician_id": self.technician_id,
            "estimated_hours": str(self.estimated_hours),
            "actual_hours": str(self.actual_hours),
            "notes": self.notes,
        }


# ============================================================
# PART RECORD
# ============================================================

@dataclass(frozen=True)
class RepairPart:
    record_id: str
    part_id: str
    part_name: str

    quantity: int
    unit_price: Decimal
    currency: str

    supplier_id: str = ""
    catalog_reference: str = ""
    pricing_reference: str = ""

    installed: bool = False
    compatibility_verified: bool = False

    def __post_init__(self) -> None:
        validate_id(self.record_id, "record_id")
        validate_id(self.part_id, "part_id")
        validate_text(self.part_name, "part_name")

        validate_positive_integer(
            self.quantity,
            "quantity",
        )

        object.__setattr__(
            self,
            "unit_price",
            money(self.unit_price, "unit_price"),
        )

        validate_currency(self.currency)

        if self.supplier_id:
            validate_id(self.supplier_id, "supplier_id")

        for name in (
            "catalog_reference",
            "pricing_reference",
        ):
            validate_text(
                getattr(self, name),
                name,
                allow_empty=True,
            )

        if not isinstance(self.installed, bool):
            raise ReportValidationError(
                "installed must be boolean"
            )

        if not isinstance(
            self.compatibility_verified,
            bool,
        ):
            raise ReportValidationError(
                "compatibility_verified must be boolean"
            )

    @property
    def total_price(self) -> Decimal:
        return money(
            self.unit_price * self.quantity,
            "total_price",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "part_id": self.part_id,
            "part_name": self.part_name,
            "quantity": self.quantity,
            "unit_price": money_str(self.unit_price),
            "total_price": money_str(self.total_price),
            "currency": self.currency,
            "supplier_id": self.supplier_id,
            "catalog_reference": self.catalog_reference,
            "pricing_reference": self.pricing_reference,
            "installed": self.installed,
            "compatibility_verified": (
                self.compatibility_verified
            ),
        }


# ============================================================
# COST ITEM
# ============================================================

@dataclass(frozen=True)
class RepairCostItem:
    cost_id: str
    category: CostCategory
    description: str
    amount: Decimal
    currency: str

    source_reference: str = ""

    def __post_init__(self) -> None:
        validate_id(self.cost_id, "cost_id")

        if not isinstance(self.category, CostCategory):
            raise ReportValidationError(
                "Invalid cost category"
            )

        validate_text(self.description, "description")

        object.__setattr__(
            self,
            "amount",
            money(self.amount, "amount"),
        )

        validate_currency(self.currency)

        validate_text(
            self.source_reference,
            "source_reference",
            allow_empty=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cost_id": self.cost_id,
            "category": self.category.value,
            "description": self.description,
            "amount": money_str(self.amount),
            "currency": self.currency,
            "source_reference": self.source_reference,
        }


# ============================================================
# REPAIR VERIFICATION
# ============================================================

@dataclass(frozen=True)
class RepairVerification:
    verification_id: str
    test_name: str

    status: VerificationStatus

    technician_id: str
    notes: str = ""

    verified_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        validate_id(
            self.verification_id,
            "verification_id",
        )

        validate_text(self.test_name, "test_name")
        validate_id(self.technician_id, "technician_id")

        if not isinstance(
            self.status,
            VerificationStatus,
        ):
            raise ReportValidationError(
                "Invalid verification status"
            )

        validate_text(
            self.notes,
            "notes",
            allow_empty=True,
        )

        validate_timestamp(
            self.verified_at,
            "verified_at",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "verification_id": self.verification_id,
            "test_name": self.test_name,
            "status": self.status.value,
            "technician_id": self.technician_id,
            "notes": self.notes,
            "verified_at": iso(self.verified_at),
        }


# ============================================================
# REPORT REQUEST
# ============================================================

@dataclass(frozen=True)
class RepairReportRequest:
    robot: RepairRobotProfile

    title: str
    summary: str

    priority: RepairPriority = RepairPriority.NORMAL

    currency: str = "INR"

    work_order_id: str = ""
    diagnosis_id: str = ""
    inspection_id: str = ""
    repair_vs_replace_id: str = ""

    created_by: str = "system"

    def __post_init__(self) -> None:
        if not isinstance(self.robot, RepairRobotProfile):
            raise ReportValidationError(
                "Invalid robot profile"
            )

        validate_text(self.title, "title")
        validate_text(self.summary, "summary")

        if not isinstance(self.priority, RepairPriority):
            raise ReportValidationError(
                "Invalid repair priority"
            )

        validate_currency(self.currency)

        for name in (
            "work_order_id",
            "diagnosis_id",
            "inspection_id",
            "repair_vs_replace_id",
        ):
            value = getattr(self, name)

            if value:
                validate_id(value, name)

        validate_id(self.created_by, "created_by")


# ============================================================
# REPORT RECORD
# ============================================================

@dataclass
class RepairReportRecord:
    report_id: str
    request: RepairReportRequest

    status: ReportStatus = ReportStatus.DRAFT

    findings: list[RepairFinding] = field(
        default_factory=list
    )

    evidence: list[RepairEvidence] = field(
        default_factory=list
    )

    tasks: list[RepairTask] = field(
        default_factory=list
    )

    parts: list[RepairPart] = field(
        default_factory=list
    )

    cost_items: list[RepairCostItem] = field(
        default_factory=list
    )

    verifications: list[RepairVerification] = field(
        default_factory=list
    )

    before_condition: str = ""
    after_condition: str = ""

    technician_recommendations: list[str] = field(
        default_factory=list
    )

    maintenance_recommendations: list[str] = field(
        default_factory=list
    )

    approved_by: str | None = None
    approval_notes: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    approved_at: datetime | None = None
    finalized_at: datetime | None = None

    version: int = 1

    @property
    def tenant_id(self) -> str:
        return self.request.robot.tenant_id

    @property
    def robot_id(self) -> str:
        return self.request.robot.robot_id

    def calculate_costs(self) -> dict[str, Any]:
        """
        Registered part prices are counted in PARTS.

        Separate PARTS cost items are assumed to be
        additional costs not already represented by
        registered part records.

        Callers must avoid duplicate accounting.
        """
        totals = {
            category.value: Decimal("0")
            for category in CostCategory
        }

        registered_parts_total = sum(
            (
                part.total_price
                for part in self.parts
            ),
            Decimal("0"),
        )

        totals[CostCategory.PARTS.value] += (
            registered_parts_total
        )

        for item in self.cost_items:
            totals[item.category.value] += (
                item.amount
            )

        grand_total = sum(
            totals.values(),
            Decimal("0"),
        )

        return {
            "currency": self.request.currency,
            "registered_parts_total": money_str(
                registered_parts_total
            ),
            "categories": {
                key: money_str(value)
                for key, value in totals.items()
            },
            "grand_total": money_str(grand_total),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "robot": self.request.robot.to_dict(),
            "title": self.request.title,
            "summary": self.request.summary,
            "priority": self.request.priority.value,
            "status": self.status.value,
            "currency": self.request.currency,
            "work_order_id": self.request.work_order_id,
            "diagnosis_id": self.request.diagnosis_id,
            "inspection_id": self.request.inspection_id,
            "repair_vs_replace_id": (
                self.request.repair_vs_replace_id
            ),
            "created_by": self.request.created_by,
            "findings": [
                item.to_dict()
                for item in self.findings
            ],
            "evidence": [
                item.to_dict()
                for item in self.evidence
            ],
            "tasks": [
                item.to_dict()
                for item in self.tasks
            ],
            "parts": [
                item.to_dict()
                for item in self.parts
            ],
            "cost_items": [
                item.to_dict()
                for item in self.cost_items
            ],
            "verifications": [
                item.to_dict()
                for item in self.verifications
            ],
            "before_condition": self.before_condition,
            "after_condition": self.after_condition,
            "technician_recommendations": list(
                self.technician_recommendations
            ),
            "maintenance_recommendations": list(
                self.maintenance_recommendations
            ),
            "cost_summary": self.calculate_costs(),
            "approved_by": self.approved_by,
            "approval_notes": self.approval_notes,
            "created_at": iso(self.created_at),
            "updated_at": iso(self.updated_at),
            "approved_at": iso(self.approved_at),
            "finalized_at": iso(self.finalized_at),
            "version": self.version,
            "advisory_only": True,
            "robot_operation_authorized": False,
        }


# ============================================================
# REPORT ENGINE
# ============================================================

class RoboticsRepairReportEngine:

    def __init__(
        self,
        config: RepairReportConfig | None = None,
    ) -> None:
        self.config = config or RepairReportConfig()

        self._reports: dict[
            tuple[str, str],
            RepairReportRecord,
        ] = {}

        self._audit: deque[
            dict[str, Any]
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------------

    def _get(
        self,
        tenant_id: str,
        report_id: str,
    ) -> RepairReportRecord:
        validate_id(tenant_id, "tenant_id")
        validate_id(report_id, "report_id")

        record = self._reports.get(
            (tenant_id, report_id)
        )

        if record is None:
            raise ReportNotFound(
                "Repair report not found"
            )

        return record

    @staticmethod
    def _ensure_editable(
        record: RepairReportRecord,
    ) -> None:
        if record.status in (
            ReportStatus.PENDING_REVIEW,
            ReportStatus.APPROVED,
            ReportStatus.FINALIZED,
        ):
            raise ReportConflict(
                "Report cannot be modified in its current status"
            )

    @staticmethod
    def _touch(record: RepairReportRecord) -> None:
        record.updated_at = utc_now()
        record.version += 1

    def _log(
        self,
        *,
        tenant_id: str,
        report_id: str,
        action: str,
        actor_id: str = "system",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._audit.append(
            {
                "event_id": new_id(),
                "timestamp": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "report_id": report_id,
                "action": action,
                "actor_id": actor_id,
                "details": dict(details or {}),
            }
        )

    def _add_item(
        self,
        *,
        tenant_id: str,
        report_id: str,
        collection_name: str,
        item: Any,
        identifier_field: str,
        maximum: int,
        actor_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                report_id,
            )

            self._ensure_editable(record)

            collection = getattr(
                record,
                collection_name,
            )

            if len(collection) >= maximum:
                raise ReportConflict(
                    f"{collection_name} capacity reached"
                )

            item_id = getattr(
                item,
                identifier_field,
            )

            if any(
                getattr(existing, identifier_field)
                == item_id
                for existing in collection
            ):
                raise ReportConflict(
                    f"Duplicate {identifier_field}"
                )

            collection.append(item)

            if record.status == ReportStatus.DRAFT:
                record.status = ReportStatus.IN_PROGRESS

            self._touch(record)

            self._log(
                tenant_id=tenant_id,
                report_id=report_id,
                action=f"{collection_name}_item_added",
                actor_id=actor_id,
                details={
                    identifier_field: item_id,
                },
            )

            return record.to_dict()

    # --------------------------------------------------------
    # CREATE REPORT
    # --------------------------------------------------------

    def create_report(
        self,
        request: RepairReportRequest,
    ) -> dict[str, Any]:
        if not isinstance(
            request,
            RepairReportRequest,
        ):
            raise ReportValidationError(
                "Invalid repair report request"
            )

        record = RepairReportRecord(
            report_id=new_id(),
            request=request,
        )

        with self._lock:
            if (
                len(self._reports)
                >= self.config.maximum_reports
            ):
                raise RepairReportError(
                    "Report capacity reached"
                )

            self._reports[
                (
                    record.tenant_id,
                    record.report_id,
                )
            ] = record

            self._log(
                tenant_id=record.tenant_id,
                report_id=record.report_id,
                action="report_created",
                actor_id=request.created_by,
            )

            return record.to_dict()

    # --------------------------------------------------------
    # ADD FINDING
    # --------------------------------------------------------

    def add_finding(
        self,
        *,
        tenant_id: str,
        report_id: str,
        finding: RepairFinding,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if not isinstance(finding, RepairFinding):
            raise ReportValidationError(
                "Invalid repair finding"
            )

        return self._add_item(
            tenant_id=tenant_id,
            report_id=report_id,
            collection_name="findings",
            item=finding,
            identifier_field="finding_id",
            maximum=(
                self.config.maximum_findings_per_report
            ),
            actor_id=actor_id,
        )

    # --------------------------------------------------------
    # ADD EVIDENCE
    # --------------------------------------------------------

    def add_evidence(
        self,
        *,
        tenant_id: str,
        report_id: str,
        evidence: RepairEvidence,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if not isinstance(evidence, RepairEvidence):
            raise ReportValidationError(
                "Invalid repair evidence"
            )

        return self._add_item(
            tenant_id=tenant_id,
            report_id=report_id,
            collection_name="evidence",
            item=evidence,
            identifier_field="evidence_id",
            maximum=(
                self.config.maximum_evidence_per_report
            ),
            actor_id=actor_id,
        )

    # --------------------------------------------------------
    # ADD REPAIR TASK
    # --------------------------------------------------------

    def add_task(
        self,
        *,
        tenant_id: str,
        report_id: str,
        task: RepairTask,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if not isinstance(task, RepairTask):
            raise ReportValidationError(
                "Invalid repair task"
            )

        return self._add_item(
            tenant_id=tenant_id,
            report_id=report_id,
            collection_name="tasks",
            item=task,
            identifier_field="task_id",
            maximum=self.config.maximum_tasks_per_report,
            actor_id=actor_id,
        )

    # --------------------------------------------------------
    # ADD PART
    # --------------------------------------------------------

    def add_part(
        self,
        *,
        tenant_id: str,
        report_id: str,
        part: RepairPart,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if not isinstance(part, RepairPart):
            raise ReportValidationError(
                "Invalid repair part"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                report_id,
            )

            if part.currency != record.request.currency:
                raise ReportValidationError(
                    "Part currency does not match report"
                )

        return self._add_item(
            tenant_id=tenant_id,
            report_id=report_id,
            collection_name="parts",
            item=part,
            identifier_field="record_id",
            maximum=self.config.maximum_parts_per_report,
            actor_id=actor_id,
        )

    # --------------------------------------------------------
    # ADD COST
    # --------------------------------------------------------

    def add_cost(
        self,
        *,
        tenant_id: str,
        report_id: str,
        cost: RepairCostItem,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if not isinstance(cost, RepairCostItem):
            raise ReportValidationError(
                "Invalid repair cost item"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                report_id,
            )

            if cost.currency != record.request.currency:
                raise ReportValidationError(
                    "Cost currency does not match report"
                )

        return self._add_item(
            tenant_id=tenant_id,
            report_id=report_id,
            collection_name="cost_items",
            item=cost,
            identifier_field="cost_id",
            maximum=(
                self.config.maximum_cost_items_per_report
            ),
            actor_id=actor_id,
        )

    # --------------------------------------------------------
    # ADD VERIFICATION
    # --------------------------------------------------------

    def add_verification(
        self,
        *,
        tenant_id: str,
        report_id: str,
        verification: RepairVerification,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if not isinstance(
            verification,
            RepairVerification,
        ):
            raise ReportValidationError(
                "Invalid repair verification"
            )

        return self._add_item(
            tenant_id=tenant_id,
            report_id=report_id,
            collection_name="verifications",
            item=verification,
            identifier_field="verification_id",
            maximum=self.config.maximum_tasks_per_report,
            actor_id=actor_id,
        )

    # --------------------------------------------------------
    # UPDATE CONDITIONS
    # --------------------------------------------------------

    def update_conditions(
        self,
        *,
        tenant_id: str,
        report_id: str,
        before_condition: str,
        after_condition: str,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        validate_text(
            before_condition,
            "before_condition",
            allow_empty=True,
        )

        validate_text(
            after_condition,
            "after_condition",
            allow_empty=True,
        )

        with self._lock:
            record = self._get(
                tenant_id,
                report_id,
            )

            self._ensure_editable(record)

            record.before_condition = before_condition
            record.after_condition = after_condition

            self._touch(record)

            self._log(
                tenant_id=tenant_id,
                report_id=report_id,
                action="conditions_updated",
                actor_id=actor_id,
            )

            return record.to_dict()

    # --------------------------------------------------------
    # ADD RECOMMENDATION
    # --------------------------------------------------------

    def add_recommendation(
        self,
        *,
        tenant_id: str,
        report_id: str,
        recommendation: str,
        maintenance: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        validate_text(
            recommendation,
            "recommendation",
        )

        if not isinstance(maintenance, bool):
            raise ReportValidationError(
                "maintenance must be boolean"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                report_id,
            )

            self._ensure_editable(record)

            target = (
                record.maintenance_recommendations
                if maintenance
                else record.technician_recommendations
            )

            if len(target) >= 500:
                raise ReportConflict(
                    "Recommendation capacity reached"
                )

            target.append(recommendation)

            self._touch(record)

            self._log(
                tenant_id=tenant_id,
                report_id=report_id,
                action="recommendation_added",
                actor_id=actor_id,
            )

            return record.to_dict()

    # --------------------------------------------------------
    # UPDATE TASK STATUS
    # --------------------------------------------------------

    def update_task_status(
        self,
        *,
        tenant_id: str,
        report_id: str,
        task_id: str,
        status: WorkStatus,
        actual_hours: Any | None = None,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        validate_id(task_id, "task_id")

        if not isinstance(status, WorkStatus):
            raise ReportValidationError(
                "Invalid task status"
            )

        with self._lock:
            record = self._get(
                tenant_id,
                report_id,
            )

            self._ensure_editable(record)

            for index, task in enumerate(record.tasks):
                if task.task_id != task_id:
                    continue

                from dataclasses import replace

                changes: dict[str, Any] = {
                    "status": status,
                }

                if actual_hours is not None:
                    changes["actual_hours"] = decimal_value(
                        actual_hours,
                        "actual_hours",
                        maximum=Decimal("100000"),
                    )

                record.tasks[index] = replace(
                    task,
                    **changes,
                )

                self._touch(record)

                self._log(
                    tenant_id=tenant_id,
                    report_id=report_id,
                    action="task_status_updated",
                    actor_id=actor_id,
                    details={
                        "task_id": task_id,
                        "status": status.value,
                    },
                )

                return record.to_dict()

            raise ReportNotFound(
                "Repair task not found"
            )

    # --------------------------------------------------------
    # COST SUMMARY
    # --------------------------------------------------------

    def calculate_report_cost(
        self,
        *,
        tenant_id: str,
        report_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                report_id,
            )

            return record.calculate_costs()

    # --------------------------------------------------------
    # VALIDATE REPORT
    # --------------------------------------------------------

    def validate_report(
        self,
        *,
        tenant_id: str,
        report_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(
                tenant_id,
                report_id,
            )

            blockers = []
            warnings = []

            if not record.findings:
                blockers.append(
                    "At least one diagnostic finding is required."
                )

            if not record.tasks:
                blockers.append(
                    "At least one repair task is required."
                )

            incomplete_tasks = [
                task.task_id
                for task in record.tasks
                if task.status not in (
                    WorkStatus.COMPLETED,
                    WorkStatus.CANCELLED,
                )
            ]

            if incomplete_tasks:
                blockers.append(
                    "Repair tasks remain incomplete or blocked."
                )

            failed_tests = [
                verification.test_name
                for verification in record.verifications
                if verification.status
                == VerificationStatus.FAILED
            ]

            if failed_tests:
                blockers.append(
                    "One or more verification tests failed."
                )

            if (
                self.config.require_verification_for_approval
                and (
                    not record.verifications
                    or not any(
                        verification.status
                        == VerificationStatus.PASSED
                        for verification in record.verifications
                    )
                )
            ):
                blockers.append(
                    "At least one passed verification is required."
                )

            unverified_parts = [
                part.part_id
                for part in record.parts
                if (
                    part.installed
                    and not part.compatibility_verified
                )
            ]

            if unverified_parts:
                blockers.append(
                    "Installed parts have unverified compatibility."
                )

            unverified_critical_findings = [
                finding.finding_id
                for finding in record.findings
                if (
                    finding.severity
                    in (
                        DamageSeverity.HIGH,
                        DamageSeverity.CRITICAL,
                    )
                    and not finding.verified_by_human
                )
            ]

            if unverified_critical_findings:
                blockers.append(
                    "High or critical findings require human verification."
                )

            if (
                record.request.robot.safety_critical
                and not record.verifications
            ):
                blockers.append(
                    "Safety-critical robot requires verification evidence."
                )

            if not record.evidence:
                warnings.append(
                    "No inspection evidence has been attached."
                )

            if not record.after_condition:
                warnings.append(
                    "After-repair condition is not documented."
                )

            return {
                "report_id": report_id,
                "valid_for_review": not blockers,
                "blockers": blockers,
                "warnings": warnings,
                "incomplete_task_ids": incomplete_tasks,
                "failed_tests": failed_tests,
                "unverified_part_ids": unverified_parts,
                "unverified_critical_findings": (
                    unverified_critical_findings
                ),
            }

    # --------------------------------------------------------
    # SUBMIT FOR REVIEW
    # --------------------------------------------------------

    def submit_for_review(
        self,
        *,
        tenant_id: str,
        report_id: str,
        actor_id: str,
    ) -> dict[str, Any]:
        validate_id(actor_id, "actor_id")

        with self._lock:
            record = self._get(
                tenant_id,
                report_id,
            )

            self._ensure_editable(record)

            validation = self.validate_report(
                tenant_id=tenant_id,
                report_id=report_id,
            )

            if not validation["valid_for_review"]:
                raise ReportConflict(
                    "Report has validation blockers: "
                    + "; ".join(validation["blockers"])
                )

            record.status = ReportStatus.PENDING_REVIEW

            self._touch(record)

            self._log(
                tenant_id=tenant_id,
                report_id=report_id,
                action="submitted_for_review",
                actor_id=actor_id,
            )

            return record.to_dict()

    # --------------------------------------------------------
    # REVIEW REPORT
    # --------------------------------------------------------

    def review_report(
        self,
        *,
        tenant_id: str,
        report_id: str,
        reviewer_id: str,
        approve: bool,
        notes: str,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise ReportPermissionDenied(
                "Authorized reviewer required"
            )

        validate_id(reviewer_id, "reviewer_id")

        if not isinstance(approve, bool):
            raise ReportValidationError(
                "approve must be boolean"
            )

        validate_text(
            notes,
            "notes",
            allow_empty=True,
        )

        with self._lock:
            record = self._get(
                tenant_id,
                report_id,
            )

            if record.status != ReportStatus.PENDING_REVIEW:
                raise ReportConflict(
                    "Report is not pending review"
                )

            if approve:
                validation = self.validate_report(
                    tenant_id=tenant_id,
                    report_id=report_id,
                )

                if not validation["valid_for_review"]:
                    raise ReportConflict(
                        "Report no longer meets approval requirements"
                    )

            record.status = (
                ReportStatus.APPROVED
                if approve
                else ReportStatus.REJECTED
            )

            record.approved_by = (
                reviewer_id if approve else None
            )

            record.approval_notes = notes

            record.approved_at = (
                utc_now() if approve else None
            )

            self._touch(record)

            self._log(
                tenant_id=tenant_id,
                report_id=report_id,
                action="report_reviewed",
                actor_id=reviewer_id,
                details={"approved": approve},
            )

            return record.to_dict()

    # --------------------------------------------------------
    # FINALIZE REPORT
    # --------------------------------------------------------

    def finalize_report(
        self,
        *,
        tenant_id: str,
        report_id: str,
        actor_id: str,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise ReportPermissionDenied(
                "Finalization authorization required"
            )

        validate_id(actor_id, "actor_id")

        with self._lock:
            record = self._get(
                tenant_id,
                report_id,
            )

            if record.status != ReportStatus.APPROVED:
                raise ReportConflict(
                    "Only approved reports can be finalized"
                )

            if (
                self.config.require_reviewer_for_finalization
                and not record.approved_by
            ):
                raise ReportConflict(
                    "Reviewer approval is missing"
                )

            record.status = ReportStatus.FINALIZED
            record.finalized_at = utc_now()

            self._touch(record)

            self._log(
                tenant_id=tenant_id,
                report_id=report_id,
                action="report_finalized",
                actor_id=actor_id,
            )

            return record.to_dict()

    # --------------------------------------------------------
    # GET REPORT
    # --------------------------------------------------------

    def get_report(
        self,
        *,
        tenant_id: str,
        report_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            return self._get(
                tenant_id,
                report_id,
            ).to_dict()

    # --------------------------------------------------------
    # LIST REPORTS
    # --------------------------------------------------------

    def list_reports(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
        status: ReportStatus | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        if robot_id is not None:
            validate_id(robot_id, "robot_id")

        if status is not None and not isinstance(
            status,
            ReportStatus,
        ):
            raise ReportValidationError(
                "Invalid report status"
            )

        validate_positive_integer(limit, "limit")

        with self._lock:
            reports = [
                record
                for (tenant, _), record
                in self._reports.items()
                if (
                    tenant == tenant_id
                    and (
                        robot_id is None
                        or record.robot_id == robot_id
                    )
                    and (
                        status is None
                        or record.status == status
                    )
                )
            ]

            reports.sort(
                key=lambda item: item.created_at,
                reverse=True,
            )

            return [
                record.to_dict()
                for record in reports[:limit]
            ]

    # --------------------------------------------------------
    # ROBOT SUMMARY
    # --------------------------------------------------------

    def robot_repair_summary(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        reports = self.list_reports(
            tenant_id=tenant_id,
            robot_id=robot_id,
            limit=1000000,
        )

        statuses = Counter(
            report["status"]
            for report in reports
        )

        total_cost = sum(
            (
                Decimal(
                    report["cost_summary"]["grand_total"]
                )
                for report in reports
            ),
            Decimal("0"),
        )

        return {
            "tenant_id": tenant_id,
            "robot_id": robot_id,
            "report_count": len(reports),
            "statuses": dict(statuses),
            "total_recorded_cost": money_str(
                total_cost
            ),
            "currency": (
                reports[0]["currency"]
                if reports
                else self.config.default_currency
            ),
            "note": (
                "Total is meaningful only when all "
                "reports use the same currency and "
                "do not duplicate the same costs."
            ),
        }

    # --------------------------------------------------------
    # DIAGNOSIS INTEGRATION
    # --------------------------------------------------------

    def import_diagnosis(
        self,
        *,
        tenant_id: str,
        report_id: str,
        diagnosis: Mapping[str, Any],
        actor_id: str = "system",
    ) -> dict[str, Any]:
        """
        Import candidate faults from issue_diagnosis.py.

        Imported candidates are not considered
        human-verified diagnoses.
        """
        if not isinstance(diagnosis, Mapping):
            raise ReportValidationError(
                "Invalid diagnosis payload"
            )

        if diagnosis.get("tenant_id") != tenant_id:
            raise ReportValidationError(
                "Diagnosis tenant mismatch"
            )

        report = self.get_report(
            tenant_id=tenant_id,
            report_id=report_id,
        )

        if diagnosis.get("robot_id") != report["robot_id"]:
            raise ReportValidationError(
                "Diagnosis robot mismatch"
            )

        candidates = diagnosis.get("candidates", [])

        if not isinstance(candidates, list):
            raise ReportValidationError(
                "Diagnosis candidates must be a list"
            )

        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise ReportValidationError(
                    "Invalid diagnosis candidate"
                )

            fault_type = str(
                candidate.get("fault_type", "unknown")
            )

            severity_value = str(
                candidate.get("severity", "unknown")
            )

            severity = (
                DamageSeverity(severity_value)
                if severity_value in DamageSeverity._value2member_map_
                else DamageSeverity.UNKNOWN
            )

            confidence = candidate.get(
                "diagnostic_score",
                0,
            )

            finding = RepairFinding(
                finding_id=new_id(),
                component_id=str(
                    candidate.get(
                        "component_id",
                        "unknown_component",
                    )
                ),
                component_name=str(
                    candidate.get(
                        "component_name",
                        "Unknown Component",
                    )
                ),
                issue_type=fault_type,
                description=(
                    f"Imported diagnostic candidate "
                    f"{index + 1}: {fault_type}"
                ),
                severity=severity,
                confidence=confidence,
                possible_cause=", ".join(
                    str(item)
                    for item in candidate.get(
                        "possible_causes",
                        [],
                    )
                ),
                recommended_action=(
                    RecommendationType.INSPECT
                ),
                diagnosis_reference=str(
                    diagnosis.get(
                        "diagnosis_id",
                        "",
                    )
                ),
                verified_by_human=False,
            )

            self.add_finding(
                tenant_id=tenant_id,
                report_id=report_id,
                finding=finding,
                actor_id=actor_id,
            )

        return self.get_report(
            tenant_id=tenant_id,
            report_id=report_id,
        )

    # --------------------------------------------------------
    # REPAIR VS REPLACE INTEGRATION
    # --------------------------------------------------------

    def import_repair_vs_replace(
        self,
        *,
        tenant_id: str,
        report_id: str,
        decision: Mapping[str, Any],
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if not isinstance(decision, Mapping):
            raise ReportValidationError(
                "Invalid decision payload"
            )

        if decision.get("tenant_id") != tenant_id:
            raise ReportValidationError(
                "Decision tenant mismatch"
            )

        report = self.get_report(
            tenant_id=tenant_id,
            report_id=report_id,
        )

        if decision.get("robot_id") != report["robot_id"]:
            raise ReportValidationError(
                "Decision robot mismatch"
            )

        recommendation = str(
            decision.get(
                "recommendation",
                "insufficient_data",
            )
        )

        text = (
            "Repair vs replacement advisory: "
            f"{recommendation}. "
            "This recommendation requires engineering review."
        )

        return self.add_recommendation(
            tenant_id=tenant_id,
            report_id=report_id,
            recommendation=text,
            maintenance=False,
            actor_id=actor_id,
        )

    # --------------------------------------------------------
    # MARKDOWN REPORT
    # --------------------------------------------------------

    @staticmethod
    def render_markdown(
        report: Mapping[str, Any],
    ) -> str:
        lines = [
            f"# {report['title']}",
            "",
            f"**Report ID:** {report['report_id']}",
            f"**Robot:** {report['robot']['robot_name']}",
            f"**Robot ID:** {report['robot_id']}",
            f"**Status:** {report['status']}",
            f"**Priority:** {report['priority']}",
            f"**Created:** {report['created_at']}",
            "",
            "## Executive Summary",
            "",
            report["summary"],
            "",
            "## Diagnostic Findings",
            "",
        ]

        if not report["findings"]:
            lines.append("No findings recorded.")

        for finding in report["findings"]:
            lines.extend(
                [
                    f"### {finding['component_name']}",
                    "",
                    f"- Issue: {finding['issue_type']}",
                    f"- Severity: {finding['severity']}",
                    f"- Confidence: {finding['confidence']}",
                    f"- Description: {finding['description']}",
                    f"- Human verified: {finding['verified_by_human']}",
                    "",
                ]
            )

        lines.extend(
            [
                "## Repair Tasks",
                "",
            ]
        )

        if not report["tasks"]:
            lines.append("No repair tasks recorded.")

        for task in report["tasks"]:
            lines.append(
                f"- **{task['action']}** — "
                f"{task['description']} "
                f"({task['status']})"
            )

        lines.extend(
            [
                "",
                "## Parts",
                "",
            ]
        )

        if not report["parts"]:
            lines.append("No parts recorded.")

        for part in report["parts"]:
            lines.append(
                f"- {part['part_name']} × {part['quantity']}: "
                f"{part['currency']} {part['total_price']}"
            )

        lines.extend(
            [
                "",
                "## Cost Summary",
                "",
            ]
        )

        costs = report["cost_summary"]

        for category, value in costs["categories"].items():
            lines.append(
                f"- {category.title()}: "
                f"{costs['currency']} {value}"
            )

        lines.extend(
            [
                "",
                f"**Total: {costs['currency']} "
                f"{costs['grand_total']}**",
                "",
                "## Repair Verification",
                "",
            ]
        )

        if not report["verifications"]:
            lines.append("No verification tests recorded.")

        for verification in report["verifications"]:
            lines.append(
                f"- {verification['test_name']}: "
                f"{verification['status']}"
            )

        lines.extend(
            [
                "",
                "## Before Repair",
                "",
                report["before_condition"] or "Not recorded.",
                "",
                "## After Repair",
                "",
                report["after_condition"] or "Not recorded.",
                "",
                "## Technician Recommendations",
                "",
            ]
        )

        for item in report["technician_recommendations"]:
            lines.append(f"- {item}")

        lines.extend(
            [
                "",
                "## Maintenance Recommendations",
                "",
            ]
        )

        for item in report["maintenance_recommendations"]:
            lines.append(f"- {item}")

        lines.extend(
            [
                "",
                "## Approval",
                "",
                f"- Status: {report['status']}",
                f"- Reviewer: {report['approved_by'] or 'None'}",
                f"- Approval time: {report['approved_at'] or 'None'}",
                "",
                "---",
                "",
                "**Safety Notice:** This report is advisory. "
                "It does not authorize robot operation, "
                "repair execution, or safety-critical actuation.",
            ]
        )

        return "\n".join(lines) + "\n"

    # --------------------------------------------------------
    # HTML REPORT
    # --------------------------------------------------------

    @staticmethod
    def render_html(
        report: Mapping[str, Any],
    ) -> str:
        escape = lambda value: html.escape(
            str(value),
            quote=True,
        )

        findings_html = "".join(
            (
                "<tr>"
                f"<td>{escape(item['component_name'])}</td>"
                f"<td>{escape(item['issue_type'])}</td>"
                f"<td>{escape(item['severity'])}</td>"
                f"<td>{escape(item['description'])}</td>"
                "</tr>"
            )
            for item in report["findings"]
        )

        tasks_html = "".join(
            (
                "<tr>"
                f"<td>{escape(item['action'])}</td>"
                f"<td>{escape(item['description'])}</td>"
                f"<td>{escape(item['status'])}</td>"
                "</tr>"
            )
            for item in report["tasks"]
        )

        parts_html = "".join(
            (
                "<tr>"
                f"<td>{escape(item['part_name'])}</td>"
                f"<td>{escape(item['quantity'])}</td>"
                f"<td>{escape(item['total_price'])}</td>"
                "</tr>"
            )
            for item in report["parts"]
        )

        cost_rows = "".join(
            (
                "<tr>"
                f"<td>{escape(category.title())}</td>"
                f"<td>{escape(value)}</td>"
                "</tr>"
            )
            for category, value
            in report["cost_summary"]["categories"].items()
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(report['title'])}</title>
<style>
:root {{
    color-scheme: light;
    font-family: Arial, Helvetica, sans-serif;
}}
body {{
    margin: 0;
    background: #f3f6fa;
    color: #172334;
}}
main {{
    max-width: 1100px;
    margin: 32px auto;
    padding: 32px;
    background: white;
    border-radius: 14px;
    box-shadow: 0 4px 22px rgba(0,0,0,.06);
}}
h1 {{
    margin-bottom: 8px;
}}
h2 {{
    margin-top: 32px;
    border-bottom: 1px solid #dce3eb;
    padding-bottom: 8px;
}}
.meta {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin: 24px 0;
}}
.meta div {{
    background: #f3f6fa;
    padding: 12px;
    border-radius: 8px;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
}}
th, td {{
    border-bottom: 1px solid #e2e8f0;
    text-align: left;
    padding: 12px;
    vertical-align: top;
}}
th {{
    background: #f3f6fa;
}}
.total {{
    font-size: 22px;
    font-weight: 700;
    margin-top: 18px;
}}
.notice {{
    margin-top: 36px;
    padding: 16px;
    border-left: 4px solid #c18a2a;
    background: #fff8eb;
}}
@media print {{
    body {{ background: white; }}
    main {{
        margin: 0;
        padding: 0;
        box-shadow: none;
    }}
}}
</style>
</head>
<body>
<main>
<h1>{escape(report['title'])}</h1>
<p>{escape(report['summary'])}</p>

<div class="meta">
<div><strong>Report ID</strong><p>{escape(report['report_id'])}</p></div>
<div><strong>Robot</strong><p>{escape(report['robot']['robot_name'])}</p></div>
<div><strong>Status</strong><p>{escape(report['status'])}</p></div>
<div><strong>Priority</strong><p>{escape(report['priority'])}</p></div>
<div><strong>Created</strong><p>{escape(report['created_at'])}</p></div>
<div><strong>Currency</strong><p>{escape(report['currency'])}</p></div>
</div>

<h2>Diagnostic Findings</h2>
<table>
<thead><tr>
<th>Component</th><th>Issue</th><th>Severity</th><th>Description</th>
</tr></thead>
<tbody>{findings_html}</tbody>
</table>

<h2>Repair Tasks</h2>
<table>
<thead><tr>
<th>Action</th><th>Description</th><th>Status</th>
</tr></thead>
<tbody>{tasks_html}</tbody>
</table>

<h2>Replacement Parts</h2>
<table>
<thead><tr>
<th>Part</th><th>Quantity</th><th>Total Price</th>
</tr></thead>
<tbody>{parts_html}</tbody>
</table>

<h2>Repair Costs</h2>
<table>
<thead><tr><th>Category</th><th>Amount</th></tr></thead>
<tbody>{cost_rows}</tbody>
</table>
<div class="total">
Total: {escape(report['currency'])}
{escape(report['cost_summary']['grand_total'])}
</div>

<h2>Before Repair</h2>
<p>{escape(report['before_condition'] or 'Not recorded')}</p>

<h2>After Repair</h2>
<p>{escape(report['after_condition'] or 'Not recorded')}</p>

<h2>Approval</h2>
<p>Status: {escape(report['status'])}</p>
<p>Reviewer: {escape(report['approved_by'] or 'None')}</p>

<div class="notice">
<strong>Safety Notice</strong>
<p>
This report is advisory and does not authorize
robot operation or physical actuation.
</p>
</div>
</main>
</body>
</html>
"""

    # --------------------------------------------------------
    # EXPORT REPORT
    # --------------------------------------------------------

    def export_report(
        self,
        *,
        tenant_id: str,
        report_id: str,
        output_dir: str | Path,
        formats: Sequence[str] = (
            "json",
            "md",
            "html",
        ),
    ) -> dict[str, str]:
        report = self.get_report(
            tenant_id=tenant_id,
            report_id=report_id,
        )

        supported = {"json", "md", "html"}

        if not formats:
            raise ReportValidationError(
                "At least one export format is required"
            )

        if any(
            item not in supported
            for item in formats
        ):
            raise ReportValidationError(
                "Unsupported export format"
            )

        directory = Path(output_dir)

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        base = clean_filename(report_id)

        exported: dict[str, str] = {}

        for format_name in dict.fromkeys(formats):
            path = directory / (
                f"repair_report_{base}.{format_name}"
            )

            if format_name == "json":
                content = json.dumps(
                    report,
                    indent=2,
                    allow_nan=False,
                )

            elif format_name == "md":
                content = self.render_markdown(report)

            else:
                content = self.render_html(report)

            path.write_text(
                content,
                encoding="utf-8",
            )

            exported[format_name] = str(path)

        return exported

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    def audit_history(
        self,
        *,
        tenant_id: str,
        report_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        if report_id is not None:
            validate_id(report_id, "report_id")

        validate_positive_integer(limit, "limit")

        with self._lock:
            events = [
                dict(event)
                for event in self._audit
                if (
                    event["tenant_id"] == tenant_id
                    and (
                        report_id is None
                        or event["report_id"] == report_id
                    )
                )
            ]

            return events[-limit:]


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    engine = RoboticsRepairReportEngine()

    tenant_id = "tenant_alpha"

    robot = RepairRobotProfile(
        tenant_id=tenant_id,
        robot_id="robot_001",
        robot_name="Industrial Robot Arm 01",
        manufacturer="Demo Robotics",
        model="ARM-X100",
        serial_number="DEMO-ARM-001",
        location="Factory Zone A",
        safety_critical=True,
    )

    request = RepairReportRequest(
        robot=robot,
        title="Robot Joint Motor Repair Report",
        summary=(
            "Inspection and repair assessment for "
            "abnormal joint motor vibration and overheating."
        ),
        priority=RepairPriority.HIGH,
        currency="INR",
        work_order_id="work_order_001",
        diagnosis_id="diagnosis_001",
        inspection_id="inspection_001",
        repair_vs_replace_id="decision_001",
        created_by="technician_001",
    )

    report = engine.create_report(request)

    report_id = report["report_id"]

    engine.add_finding(
        tenant_id=tenant_id,
        report_id=report_id,
        finding=RepairFinding(
            finding_id=new_id(),
            component_id="joint_motor_001",
            component_name="Joint Motor",
            issue_type="motor_overheating",
            description=(
                "Motor temperature exceeded the "
                "configured inspection threshold."
            ),
            severity=DamageSeverity.HIGH,
            confidence=Decimal("0.92"),
            possible_cause=(
                "Bearing friction or motor overload"
            ),
            recommended_action=(
                RecommendationType.REPAIR
            ),
            diagnosis_reference="diagnosis_001",
            verified_by_human=True,
        ),
        actor_id="technician_001",
    )

    engine.add_evidence(
        tenant_id=tenant_id,
        report_id=report_id,
        evidence=RepairEvidence(
            evidence_id=new_id(),
            evidence_type="inspection_image",
            reference="inspection_001/image_01",
            description="Joint motor inspection image",
            component_id="joint_motor_001",
        ),
    )

    engine.add_task(
        tenant_id=tenant_id,
        report_id=report_id,
        task=RepairTask(
            task_id="task_001",
            component_id="joint_motor_001",
            action=RepairAction.REPAIR,
            description=(
                "Inspect and repair the joint motor "
                "bearing assembly."
            ),
            status=WorkStatus.COMPLETED,
            technician_id="technician_001",
            estimated_hours=Decimal("3"),
            actual_hours=Decimal("3.5"),
        ),
    )

    engine.add_part(
        tenant_id=tenant_id,
        report_id=report_id,
        part=RepairPart(
            record_id=new_id(),
            part_id="bearing_001",
            part_name="Industrial Motor Bearing",
            quantity=2,
            unit_price=Decimal("850"),
            currency="INR",
            supplier_id="supplier_001",
            catalog_reference="catalog_001",
            pricing_reference="offer_001",
            installed=True,
            compatibility_verified=True,
        ),
    )

    engine.add_cost(
        tenant_id=tenant_id,
        report_id=report_id,
        cost=RepairCostItem(
            cost_id=new_id(),
            category=CostCategory.LABOR,
            description="Motor repair labor",
            amount=Decimal("2500"),
            currency="INR",
            source_reference="labor_quote_001",
        ),
    )

    engine.add_cost(
        tenant_id=tenant_id,
        report_id=report_id,
        cost=RepairCostItem(
            cost_id=new_id(),
            category=CostCategory.DOWNTIME,
            description="Production downtime",
            amount=Decimal("4000"),
            currency="INR",
            source_reference="downtime_001",
        ),
    )

    engine.add_verification(
        tenant_id=tenant_id,
        report_id=report_id,
        verification=RepairVerification(
            verification_id=new_id(),
            test_name="Motor Temperature Verification",
            status=VerificationStatus.PASSED,
            technician_id="technician_001",
            notes=(
                "Synthetic demonstration: measured "
                "temperature within configured limits."
            ),
        ),
    )

    engine.update_conditions(
        tenant_id=tenant_id,
        report_id=report_id,
        before_condition=(
            "Motor overheating and abnormal vibration."
        ),
        after_condition=(
            "Demonstration repair completed; "
            "temperature verification passed."
        ),
        actor_id="technician_001",
    )

    engine.add_recommendation(
        tenant_id=tenant_id,
        report_id=report_id,
        recommendation=(
            "Schedule bearing inspection during "
            "the next preventive maintenance window."
        ),
        maintenance=True,
    )

    print("\nREPORT VALIDATION")

    print(
        json.dumps(
            engine.validate_report(
                tenant_id=tenant_id,
                report_id=report_id,
            ),
            indent=2,
        )
    )

    engine.submit_for_review(
        tenant_id=tenant_id,
        report_id=report_id,
        actor_id="technician_001",
    )

    engine.review_report(
        tenant_id=tenant_id,
        report_id=report_id,
        reviewer_id="supervisor_001",
        approve=True,
        notes="Demo review completed.",
        authorized=True,
    )

    final_report = engine.finalize_report(
        tenant_id=tenant_id,
        report_id=report_id,
        actor_id="supervisor_001",
        authorized=True,
    )

    print("\nFINAL REPAIR REPORT")

    print(
        json.dumps(
            final_report,
            indent=2,
        )
    )

    exported = engine.export_report(
        tenant_id=tenant_id,
        report_id=report_id,
        output_dir="data/robotics/repair_reports",
        formats=("json", "md", "html"),
    )

    print("\nEXPORTED FILES")

    print(
        json.dumps(
            exported,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
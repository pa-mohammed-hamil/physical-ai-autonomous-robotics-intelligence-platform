"""Warranty Checker module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/warranty_checker.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Robotics Warranty Checker and Coverage Intelligence Engine

Python 3.11+
Dependencies: Standard library only

Features:
    - Robot and component warranty registration
    - Warranty validity and expiration checking
    - Manufacturer, supplier and extended warranties
    - Parts, labor, repair and replacement coverage
    - Warranty exclusions and eligibility assessment
    - Warranty claim creation and tracking
    - Warranty document references
    - Warranty expiration alerts
    - Remaining warranty duration
    - Serial number and part number verification
    - Maintenance compliance checks
    - Service and replacement advisory integration
    - Warranty cost and potential savings estimation
    - Multi-tenant data isolation
    - Thread-safe in-memory storage
    - Audit logging and JSON export
    - Human claim review and approval workflow

Integrations:
    parts_catalog.py
    parts_pricing.py
    repair_vs_replace.py
    repair_report.py
    replacement_advisor.py
    service_recommendations.py
    maintenance_service.py
    robot_health.py

Safety and legal limitations:
    Warranty eligibility is advisory and depends on
    actual contractual terms and manufacturer decisions.

    This module does not:
        - Submit claims to manufacturers
        - Guarantee warranty coverage
        - Authorize repairs or replacement
        - Authorize robot operation
        - Perform financial transactions

    Application authentication and role-based authorization
    must be enforced outside this module.
"""

from __future__ import annotations

import json
import threading
import uuid

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class WarrantyError(Exception):
    pass


class WarrantyValidationError(WarrantyError):
    pass


class WarrantyNotFound(WarrantyError):
    pass


class WarrantyConflict(WarrantyError):
    pass


class WarrantyPermissionDenied(WarrantyError):
    pass


# ============================================================
# ENUMS
# ============================================================

class WarrantyType(str, Enum):
    MANUFACTURER = "manufacturer"
    SUPPLIER = "supplier"
    EXTENDED = "extended"
    SERVICE_CONTRACT = "service_contract"
    PARTS = "parts"
    LABOR = "labor"
    REPLACEMENT = "replacement"
    OTHER = "other"


class WarrantyStatus(str, Enum):
    NOT_STARTED = "not_started"
    ACTIVE = "active"
    EXPIRED = "expired"
    VOID_REPORTED = "void_reported"
    CANCELLED = "cancelled"


class CoverageType(str, Enum):
    PARTS = "parts"
    LABOR = "labor"
    REPAIR = "repair"
    REPLACEMENT = "replacement"
    DIAGNOSTICS = "diagnostics"
    TRANSPORT = "transport"
    ONSITE_SERVICE = "onsite_service"
    SOFTWARE = "software"
    BATTERY = "battery"
    OTHER = "other"


class FailureCause(str, Enum):
    MANUFACTURING_DEFECT = "manufacturing_defect"
    MATERIAL_DEFECT = "material_defect"
    NORMAL_WEAR = "normal_wear"
    ACCIDENTAL_DAMAGE = "accidental_damage"
    MISUSE = "misuse"
    UNAUTHORIZED_MODIFICATION = "unauthorized_modification"
    ENVIRONMENTAL_DAMAGE = "environmental_damage"
    ELECTRICAL_DAMAGE = "electrical_damage"
    SOFTWARE_FAILURE = "software_failure"
    UNKNOWN = "unknown"


class EligibilityStatus(str, Enum):
    POTENTIALLY_ELIGIBLE = "potentially_eligible"
    NOT_ELIGIBLE = "not_eligible"
    NEEDS_REVIEW = "needs_review"


class ClaimStatus(str, Enum):
    DRAFT = "draft"
    INTERNAL_REVIEW = "internal_review"
    READY_FOR_SUBMISSION = "ready_for_submission"
    SUBMITTED_EXTERNALLY = "submitted_externally"
    APPROVED_EXTERNALLY = "approved_externally"
    REJECTED_EXTERNALLY = "rejected_externally"
    CLOSED = "closed"


class ClaimDecision(str, Enum):
    APPROVE_INTERNAL = "approve_internal"
    REJECT_INTERNAL = "reject_internal"


# ============================================================
# UTILITIES
# ============================================================

CENT = Decimal("0.01")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


def validate_id(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or not all(
            char.isalnum() or char in "._-"
            for char in value
        )
    ):
        raise WarrantyValidationError(
            f"Invalid {name}"
        )

    return value


def validate_text(
    value: str,
    name: str,
    *,
    allow_empty: bool = False,
    maximum: int = 4000,
) -> str:
    if not isinstance(value, str):
        raise WarrantyValidationError(
            f"{name} must be a string"
        )

    result = value.strip()

    if len(result) > maximum:
        raise WarrantyValidationError(
            f"{name} exceeds maximum length"
        )

    if not allow_empty and not result:
        raise WarrantyValidationError(
            f"{name} cannot be empty"
        )

    return result


def validate_integer(
    value: int,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = 10000000,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise WarrantyValidationError(
            f"Invalid {name}"
        )

    return value


def decimal_value(
    value: Any,
    name: str,
    *,
    minimum: Decimal = Decimal("0"),
    maximum: Decimal = Decimal("1000000000000"),
) -> Decimal:
    if isinstance(value, bool):
        raise WarrantyValidationError(
            f"Invalid {name}"
        )

    try:
        result = Decimal(str(value))
    except (
        InvalidOperation,
        ValueError,
        TypeError,
    ) as exc:
        raise WarrantyValidationError(
            f"Invalid {name}"
        ) from exc

    if (
        not result.is_finite()
        or result < minimum
        or result > maximum
    ):
        raise WarrantyValidationError(
            f"Invalid {name}"
        )

    return result


def money(value: Any, name: str = "amount") -> Decimal:
    return decimal_value(
        value,
        name,
    ).quantize(
        CENT,
        rounding=ROUND_HALF_UP,
    )


def money_string(value: Decimal) -> str:
    return str(
        value.quantize(
            CENT,
            rounding=ROUND_HALF_UP,
        )
    )


def validate_currency(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 3
        or not value.isascii()
        or not value.isalpha()
        or value != value.upper()
    ):
        raise WarrantyValidationError(
            "Invalid currency"
        )

    return value


def validate_date(value: date, name: str) -> date:
    if (
        not isinstance(value, date)
        or isinstance(value, datetime)
    ):
        raise WarrantyValidationError(
            f"{name} must be a date"
        )

    return value


def validate_datetime(
    value: datetime,
    name: str,
) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise WarrantyValidationError(
            f"{name} must be timezone-aware"
        )

    return value.astimezone(timezone.utc)


def enum_value(value: Enum) -> str:
    return str(value.value)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class WarrantyCheckerConfig:
    default_currency: str = "INR"

    expiration_alert_days: tuple[int, ...] = (
        90,
        60,
        30,
        7,
    )

    maximum_warranties: int = 100000
    maximum_claims: int = 100000
    maximum_audit_events: int = 20000

    advisory_only: bool = True

    def __post_init__(self) -> None:
        validate_currency(self.default_currency)

        if not isinstance(
            self.expiration_alert_days,
            tuple,
        ):
            raise WarrantyValidationError(
                "expiration_alert_days must be a tuple"
            )

        for days in self.expiration_alert_days:
            validate_integer(
                days,
                "expiration_alert_days",
                maximum=36500,
            )

        for name in (
            "maximum_warranties",
            "maximum_claims",
            "maximum_audit_events",
        ):
            validate_integer(
                getattr(self, name),
                name,
                minimum=1,
            )

        if self.advisory_only is not True:
            raise WarrantyValidationError(
                "Only advisory mode is supported"
            )


# ============================================================
# WARRANTY COVERAGE
# ============================================================

@dataclass(frozen=True)
class WarrantyCoverage:
    coverage_type: CoverageType

    covered: bool = True

    coverage_percentage: Decimal = Decimal("100")

    maximum_covered_amount: Decimal | None = None

    deductible: Decimal = Decimal("0")

    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(
            self.coverage_type,
            CoverageType,
        ):
            raise WarrantyValidationError(
                "Invalid coverage type"
            )

        if not isinstance(self.covered, bool):
            raise WarrantyValidationError(
                "covered must be boolean"
            )

        decimal_value(
            self.coverage_percentage,
            "coverage_percentage",
            maximum=Decimal("100"),
        )

        if self.maximum_covered_amount is not None:
            object.__setattr__(
                self,
                "maximum_covered_amount",
                money(
                    self.maximum_covered_amount,
                    "maximum_covered_amount",
                ),
            )

        object.__setattr__(
            self,
            "deductible",
            money(
                self.deductible,
                "deductible",
            ),
        )

        validate_text(
            self.notes,
            "notes",
            allow_empty=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage_type": self.coverage_type.value,
            "covered": self.covered,
            "coverage_percentage": str(
                self.coverage_percentage
            ),
            "maximum_covered_amount": (
                money_string(
                    self.maximum_covered_amount
                )
                if self.maximum_covered_amount is not None
                else None
            ),
            "deductible": money_string(
                self.deductible
            ),
            "notes": self.notes,
        }


# ============================================================
# WARRANTY RECORD
# ============================================================

@dataclass(frozen=True)
class RobotWarranty:
    warranty_id: str

    tenant_id: str
    robot_id: str
    component_id: str

    warranty_name: str
    warranty_type: WarrantyType

    manufacturer: str
    provider_name: str

    start_date: date
    end_date: date

    serial_number: str = ""
    part_number: str = ""
    model_number: str = ""

    policy_number: str = ""

    coverage: tuple[WarrantyCoverage, ...] = ()

    excluded_failure_causes: tuple[FailureCause, ...] = ()

    maintenance_compliance_required: bool = False

    maximum_operating_hours: Decimal | None = None

    currency: str = "INR"

    document_references: tuple[str, ...] = ()

    void_reported: bool = False
    cancelled: bool = False

    notes: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for name in (
            "warranty_id",
            "tenant_id",
            "robot_id",
            "component_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        for name in (
            "warranty_name",
            "manufacturer",
            "provider_name",
        ):
            validate_text(
                getattr(self, name),
                name,
            )

        if not isinstance(
            self.warranty_type,
            WarrantyType,
        ):
            raise WarrantyValidationError(
                "Invalid warranty type"
            )

        validate_date(
            self.start_date,
            "start_date",
        )

        validate_date(
            self.end_date,
            "end_date",
        )

        if self.end_date < self.start_date:
            raise WarrantyValidationError(
                "Warranty end date cannot precede start date"
            )

        for name in (
            "serial_number",
            "part_number",
            "model_number",
            "policy_number",
        ):
            validate_text(
                getattr(self, name),
                name,
                allow_empty=True,
                maximum=256,
            )

        if not isinstance(self.coverage, tuple):
            raise WarrantyValidationError(
                "coverage must be a tuple"
            )

        seen_coverage = set()

        for item in self.coverage:
            if not isinstance(item, WarrantyCoverage):
                raise WarrantyValidationError(
                    "Invalid warranty coverage"
                )

            if item.coverage_type in seen_coverage:
                raise WarrantyValidationError(
                    "Duplicate coverage type"
                )

            seen_coverage.add(item.coverage_type)

        if not isinstance(
            self.excluded_failure_causes,
            tuple,
        ):
            raise WarrantyValidationError(
                "excluded_failure_causes must be a tuple"
            )

        for cause in self.excluded_failure_causes:
            if not isinstance(cause, FailureCause):
                raise WarrantyValidationError(
                    "Invalid excluded failure cause"
                )

        if not isinstance(
            self.maintenance_compliance_required,
            bool,
        ):
            raise WarrantyValidationError(
                "maintenance_compliance_required "
                "must be boolean"
            )

        if self.maximum_operating_hours is not None:
            decimal_value(
                self.maximum_operating_hours,
                "maximum_operating_hours",
                maximum=Decimal("100000000"),
            )

        validate_currency(self.currency)

        if not isinstance(
            self.document_references,
            tuple,
        ):
            raise WarrantyValidationError(
                "document_references must be a tuple"
            )

        for reference in self.document_references:
            validate_text(
                reference,
                "document_reference",
                maximum=512,
            )

        if not isinstance(self.void_reported, bool):
            raise WarrantyValidationError(
                "void_reported must be boolean"
            )

        if not isinstance(self.cancelled, bool):
            raise WarrantyValidationError(
                "cancelled must be boolean"
            )

        validate_text(
            self.notes,
            "notes",
            allow_empty=True,
        )

        validate_datetime(
            self.created_at,
            "created_at",
        )

    def status(
        self,
        as_of: date | None = None,
    ) -> WarrantyStatus:
        today = as_of or utc_now().date()

        validate_date(today, "as_of")

        if self.cancelled:
            return WarrantyStatus.CANCELLED

        if self.void_reported:
            return WarrantyStatus.VOID_REPORTED

        if today < self.start_date:
            return WarrantyStatus.NOT_STARTED

        if today > self.end_date:
            return WarrantyStatus.EXPIRED

        return WarrantyStatus.ACTIVE

    def remaining_days(
        self,
        as_of: date | None = None,
    ) -> int:
        today = as_of or utc_now().date()

        validate_date(today, "as_of")

        if self.status(today) != WarrantyStatus.ACTIVE:
            return 0

        return max(
            0,
            (self.end_date - today).days,
        )

    def coverage_for(
        self,
        coverage_type: CoverageType,
    ) -> WarrantyCoverage | None:
        for item in self.coverage:
            if item.coverage_type == coverage_type:
                return item

        return None

    def to_dict(
        self,
        as_of: date | None = None,
    ) -> dict[str, Any]:
        return {
            "warranty_id": self.warranty_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "component_id": self.component_id,
            "warranty_name": self.warranty_name,
            "warranty_type": self.warranty_type.value,
            "manufacturer": self.manufacturer,
            "provider_name": self.provider_name,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "status": self.status(as_of).value,
            "remaining_days": self.remaining_days(as_of),
            "serial_number": self.serial_number,
            "part_number": self.part_number,
            "model_number": self.model_number,
            "policy_number": self.policy_number,
            "coverage": [
                item.to_dict()
                for item in self.coverage
            ],
            "excluded_failure_causes": [
                item.value
                for item in self.excluded_failure_causes
            ],
            "maintenance_compliance_required": (
                self.maintenance_compliance_required
            ),
            "maximum_operating_hours": (
                str(self.maximum_operating_hours)
                if self.maximum_operating_hours is not None
                else None
            ),
            "currency": self.currency,
            "document_references": list(
                self.document_references
            ),
            "void_reported": self.void_reported,
            "cancelled": self.cancelled,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
        }


# ============================================================
# WARRANTY CHECK REQUEST
# ============================================================

@dataclass(frozen=True)
class WarrantyCheckRequest:
    tenant_id: str
    robot_id: str
    component_id: str

    coverage_type: CoverageType

    failure_cause: FailureCause = FailureCause.UNKNOWN

    serial_number: str = ""
    part_number: str = ""

    operating_hours: Decimal | None = None

    maintenance_compliant: bool | None = None

    estimated_cost: Decimal = Decimal("0")

    currency: str = "INR"

    incident_date: date | None = None

    notes: str = ""

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

        if not isinstance(
            self.coverage_type,
            CoverageType,
        ):
            raise WarrantyValidationError(
                "Invalid coverage type"
            )

        if not isinstance(
            self.failure_cause,
            FailureCause,
        ):
            raise WarrantyValidationError(
                "Invalid failure cause"
            )

        for name in (
            "serial_number",
            "part_number",
        ):
            validate_text(
                getattr(self, name),
                name,
                allow_empty=True,
                maximum=256,
            )

        if self.operating_hours is not None:
            decimal_value(
                self.operating_hours,
                "operating_hours",
                maximum=Decimal("100000000"),
            )

        if (
            self.maintenance_compliant is not None
            and not isinstance(
                self.maintenance_compliant,
                bool,
            )
        ):
            raise WarrantyValidationError(
                "maintenance_compliant must be "
                "boolean or None"
            )

        object.__setattr__(
            self,
            "estimated_cost",
            money(
                self.estimated_cost,
                "estimated_cost",
            ),
        )

        validate_currency(self.currency)

        if self.incident_date is not None:
            validate_date(
                self.incident_date,
                "incident_date",
            )

        validate_text(
            self.notes,
            "notes",
            allow_empty=True,
        )


# ============================================================
# WARRANTY CHECK RESULT
# ============================================================

@dataclass(frozen=True)
class WarrantyCheckResult:
    check_id: str

    tenant_id: str
    robot_id: str
    component_id: str

    warranty_id: str

    warranty_status: WarrantyStatus
    eligibility: EligibilityStatus

    coverage_type: CoverageType

    estimated_cost: Decimal
    potential_covered_amount: Decimal
    potential_customer_cost: Decimal

    currency: str

    reasons: tuple[str, ...]
    warnings: tuple[str, ...]

    checked_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "component_id": self.component_id,
            "warranty_id": self.warranty_id,
            "warranty_status": (
                self.warranty_status.value
            ),
            "eligibility": self.eligibility.value,
            "coverage_type": self.coverage_type.value,
            "estimated_cost": money_string(
                self.estimated_cost
            ),
            "potential_covered_amount": money_string(
                self.potential_covered_amount
            ),
            "potential_customer_cost": money_string(
                self.potential_customer_cost
            ),
            "currency": self.currency,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "checked_at": self.checked_at.isoformat(),
            "coverage_guaranteed": False,
            "manufacturer_approval_required": True,
            "advisory_only": True,
        }


# ============================================================
# WARRANTY CLAIM
# ============================================================

@dataclass
class WarrantyClaim:
    claim_id: str

    tenant_id: str
    robot_id: str
    component_id: str

    warranty_id: str
    check_id: str

    claim_title: str
    description: str

    requested_amount: Decimal
    currency: str

    status: ClaimStatus = ClaimStatus.DRAFT

    external_reference: str = ""

    reviewer_id: str | None = None
    review_notes: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "component_id": self.component_id,
            "warranty_id": self.warranty_id,
            "check_id": self.check_id,
            "claim_title": self.claim_title,
            "description": self.description,
            "requested_amount": money_string(
                self.requested_amount
            ),
            "currency": self.currency,
            "status": self.status.value,
            "external_reference": self.external_reference,
            "reviewer_id": self.reviewer_id,
            "review_notes": self.review_notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "submitted_by_this_module": False,
            "advisory_only": True,
        }


# ============================================================
# WARRANTY CHECKER ENGINE
# ============================================================

class RoboticsWarrantyChecker:

    def __init__(
        self,
        config: WarrantyCheckerConfig | None = None,
    ) -> None:
        self.config = config or WarrantyCheckerConfig()

        self._warranties: dict[
            tuple[str, str],
            RobotWarranty,
        ] = {}

        self._checks: dict[
            tuple[str, str],
            WarrantyCheckResult,
        ] = {}

        self._claims: dict[
            tuple[str, str],
            WarrantyClaim,
        ] = {}

        self._audit: deque[
            dict[str, Any]
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _log(
        self,
        *,
        tenant_id: str,
        action: str,
        entity_id: str,
        actor_id: str = "system",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self._audit.append(
            {
                "event_id": new_id(),
                "timestamp": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "action": action,
                "entity_id": entity_id,
                "actor_id": actor_id,
                "details": dict(details or {}),
            }
        )

    # --------------------------------------------------------
    # REGISTER WARRANTY
    # --------------------------------------------------------

    def register_warranty(
        self,
        warranty: RobotWarranty,
        *,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise WarrantyPermissionDenied(
                "Authorized warranty registration required"
            )

        if not isinstance(warranty, RobotWarranty):
            raise WarrantyValidationError(
                "Invalid warranty"
            )

        validate_id(actor_id, "actor_id")

        key = (
            warranty.tenant_id,
            warranty.warranty_id,
        )

        with self._lock:
            if key in self._warranties:
                raise WarrantyConflict(
                    "Warranty already exists"
                )

            if (
                len(self._warranties)
                >= self.config.maximum_warranties
            ):
                raise WarrantyConflict(
                    "Warranty storage capacity reached"
                )

            self._warranties[key] = warranty

            self._log(
                tenant_id=warranty.tenant_id,
                action="warranty_registered",
                entity_id=warranty.warranty_id,
                actor_id=actor_id,
            )

            return warranty.to_dict()

    # --------------------------------------------------------
    # GET WARRANTY
    # --------------------------------------------------------

    def get_warranty(
        self,
        *,
        tenant_id: str,
        warranty_id: str,
        as_of: date | None = None,
    ) -> dict[str, Any]:
        validate_id(tenant_id, "tenant_id")
        validate_id(warranty_id, "warranty_id")

        with self._lock:
            warranty = self._warranties.get(
                (tenant_id, warranty_id)
            )

            if warranty is None:
                raise WarrantyNotFound(
                    "Warranty not found"
                )

            return warranty.to_dict(as_of)

    # --------------------------------------------------------
    # LIST WARRANTIES
    # --------------------------------------------------------

    def list_warranties(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
        component_id: str | None = None,
        status: WarrantyStatus | None = None,
        as_of: date | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        if robot_id is not None:
            validate_id(robot_id, "robot_id")

        if component_id is not None:
            validate_id(component_id, "component_id")

        if status is not None and not isinstance(
            status,
            WarrantyStatus,
        ):
            raise WarrantyValidationError(
                "Invalid warranty status"
            )

        if as_of is not None:
            validate_date(as_of, "as_of")

        validate_integer(
            limit,
            "limit",
            minimum=1,
            maximum=100000,
        )

        with self._lock:
            warranties = [
                warranty
                for (tenant, _), warranty
                in self._warranties.items()
                if (
                    tenant == tenant_id
                    and (
                        robot_id is None
                        or warranty.robot_id == robot_id
                    )
                    and (
                        component_id is None
                        or warranty.component_id == component_id
                    )
                    and (
                        status is None
                        or warranty.status(as_of) == status
                    )
                )
            ]

            warranties.sort(
                key=lambda item: item.end_date
            )

            return [
                item.to_dict(as_of)
                for item in warranties[:limit]
            ]

    # --------------------------------------------------------
    # CHECK ONE WARRANTY
    # --------------------------------------------------------

    def check_warranty(
        self,
        *,
        warranty_id: str,
        request: WarrantyCheckRequest,
    ) -> dict[str, Any]:
        if not isinstance(
            request,
            WarrantyCheckRequest,
        ):
            raise WarrantyValidationError(
                "Invalid warranty check request"
            )

        validate_id(warranty_id, "warranty_id")

        with self._lock:
            warranty = self._warranties.get(
                (
                    request.tenant_id,
                    warranty_id,
                )
            )

            if warranty is None:
                raise WarrantyNotFound(
                    "Warranty not found"
                )

            if (
                warranty.robot_id != request.robot_id
                or warranty.component_id
                != request.component_id
            ):
                raise WarrantyValidationError(
                    "Warranty does not belong to "
                    "the requested robot component"
                )

            result = self._evaluate(
                warranty,
                request,
            )

            self._checks[
                (
                    request.tenant_id,
                    result.check_id,
                )
            ] = result

            self._log(
                tenant_id=request.tenant_id,
                action="warranty_checked",
                entity_id=result.check_id,
                details={
                    "warranty_id": warranty_id,
                    "eligibility": (
                        result.eligibility.value
                    ),
                },
            )

            return result.to_dict()

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    def _evaluate(
        self,
        warranty: RobotWarranty,
        request: WarrantyCheckRequest,
    ) -> WarrantyCheckResult:
        reasons = []
        warnings = []

        incident_date = (
            request.incident_date
            or utc_now().date()
        )

        status = warranty.status(incident_date)

        eligibility = EligibilityStatus.POTENTIALLY_ELIGIBLE

        if status != WarrantyStatus.ACTIVE:
            eligibility = EligibilityStatus.NOT_ELIGIBLE

            reasons.append(
                f"Warranty status on incident date: "
                f"{status.value}."
            )

        coverage = warranty.coverage_for(
            request.coverage_type
        )

        if coverage is None or not coverage.covered:
            eligibility = EligibilityStatus.NOT_ELIGIBLE

            reasons.append(
                "Requested service is not listed "
                "as covered."
            )

        if (
            request.failure_cause
            in warranty.excluded_failure_causes
        ):
            eligibility = EligibilityStatus.NOT_ELIGIBLE

            reasons.append(
                "Reported failure cause is excluded "
                "under the recorded warranty terms."
            )

        if (
            warranty.serial_number
            and request.serial_number
            and warranty.serial_number
            != request.serial_number
        ):
            eligibility = EligibilityStatus.NOT_ELIGIBLE

            reasons.append(
                "Component serial number mismatch."
            )

        if (
            warranty.part_number
            and request.part_number
            and warranty.part_number
            != request.part_number
        ):
            eligibility = EligibilityStatus.NOT_ELIGIBLE

            reasons.append(
                "Component part number mismatch."
            )

        if (
            warranty.maximum_operating_hours is not None
            and request.operating_hours is not None
            and request.operating_hours
            > warranty.maximum_operating_hours
        ):
            eligibility = EligibilityStatus.NOT_ELIGIBLE

            reasons.append(
                "Recorded operating hours exceed "
                "the warranty limit."
            )

        if (
            warranty.maintenance_compliance_required
            and request.maintenance_compliant is False
        ):
            # Maintenance noncompliance may require
            # contractual and legal interpretation.
            if eligibility != EligibilityStatus.NOT_ELIGIBLE:
                eligibility = EligibilityStatus.NEEDS_REVIEW

            warnings.append(
                "Maintenance noncompliance requires "
                "provider review; exclusion is not assumed."
            )

        missing_evidence = []

        if (
            warranty.serial_number
            and not request.serial_number
        ):
            missing_evidence.append(
                "serial number"
            )

        if (
            warranty.part_number
            and not request.part_number
        ):
            missing_evidence.append(
                "part number"
            )

        if (
            warranty.maximum_operating_hours is not None
            and request.operating_hours is None
        ):
            missing_evidence.append(
                "operating hours"
            )

        if (
            warranty.maintenance_compliance_required
            and request.maintenance_compliant is None
        ):
            missing_evidence.append(
                "maintenance compliance"
            )

        if request.failure_cause == FailureCause.UNKNOWN:
            missing_evidence.append(
                "failure cause"
            )

        if missing_evidence:
            warnings.append(
                "Additional verification needed: "
                + ", ".join(missing_evidence)
            )

            if eligibility == EligibilityStatus.POTENTIALLY_ELIGIBLE:
                eligibility = EligibilityStatus.NEEDS_REVIEW

        if warranty.currency != request.currency:
            warnings.append(
                "Currency mismatch: no automatic "
                "currency conversion is performed."
            )

            if eligibility == EligibilityStatus.POTENTIALLY_ELIGIBLE:
                eligibility = EligibilityStatus.NEEDS_REVIEW

        potential_covered = Decimal("0")

        if (
            eligibility != EligibilityStatus.NOT_ELIGIBLE
            and coverage is not None
            and coverage.covered
            and warranty.currency == request.currency
        ):
            potential_covered = (
                request.estimated_cost
                * coverage.coverage_percentage
                / Decimal("100")
            )

            if coverage.maximum_covered_amount is not None:
                potential_covered = min(
                    potential_covered,
                    coverage.maximum_covered_amount,
                )

            potential_covered = max(
                Decimal("0"),
                potential_covered - coverage.deductible,
            )

            potential_covered = min(
                request.estimated_cost,
                potential_covered,
            )

        potential_covered = money(
            potential_covered
        )

        potential_customer_cost = money(
            request.estimated_cost - potential_covered
        )

        warnings.append(
            "Coverage estimate is illustrative and "
            "subject to provider verification and approval."
        )

        return WarrantyCheckResult(
            check_id=new_id(),
            tenant_id=request.tenant_id,
            robot_id=request.robot_id,
            component_id=request.component_id,
            warranty_id=warranty.warranty_id,
            warranty_status=status,
            eligibility=eligibility,
            coverage_type=request.coverage_type,
            estimated_cost=request.estimated_cost,
            potential_covered_amount=potential_covered,
            potential_customer_cost=potential_customer_cost,
            currency=request.currency,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
        )

    # --------------------------------------------------------
    # CHECK ALL MATCHING WARRANTIES
    # --------------------------------------------------------

    def check_component_warranties(
        self,
        request: WarrantyCheckRequest,
    ) -> dict[str, Any]:
        if not isinstance(
            request,
            WarrantyCheckRequest,
        ):
            raise WarrantyValidationError(
                "Invalid warranty check request"
            )

        with self._lock:
            matching = [
                warranty
                for (tenant, _), warranty
                in self._warranties.items()
                if (
                    tenant == request.tenant_id
                    and warranty.robot_id == request.robot_id
                    and warranty.component_id
                    == request.component_id
                )
            ]

            results = []

            for warranty in matching:
                result = self._evaluate(
                    warranty,
                    request,
                )

                self._checks[
                    (
                        request.tenant_id,
                        result.check_id,
                    )
                ] = result

                results.append(result)

            results.sort(
                key=lambda item: (
                    item.eligibility
                    == EligibilityStatus.NOT_ELIGIBLE,
                    item.eligibility
                    == EligibilityStatus.NEEDS_REVIEW,
                    -item.potential_covered_amount,
                )
            )

            best = (
                results[0]
                if results
                else None
            )

            return {
                "tenant_id": request.tenant_id,
                "robot_id": request.robot_id,
                "component_id": request.component_id,
                "warranties_checked": len(results),
                "results": [
                    item.to_dict()
                    for item in results
                ],
                "best_potential_warranty_id": (
                    best.warranty_id
                    if (
                        best is not None
                        and best.eligibility
                        != EligibilityStatus.NOT_ELIGIBLE
                    )
                    else None
                ),
                "best_potential_covered_amount": (
                    money_string(
                        best.potential_covered_amount
                    )
                    if best is not None
                    else "0.00"
                ),
                "currency": request.currency,
                "coverage_not_additive": True,
                "advisory_only": True,
            }

    # --------------------------------------------------------
    # GET CHECK RESULT
    # --------------------------------------------------------

    def get_check(
        self,
        *,
        tenant_id: str,
        check_id: str,
    ) -> dict[str, Any]:
        validate_id(tenant_id, "tenant_id")
        validate_id(check_id, "check_id")

        with self._lock:
            result = self._checks.get(
                (tenant_id, check_id)
            )

            if result is None:
                raise WarrantyNotFound(
                    "Warranty check not found"
                )

            return result.to_dict()

    # --------------------------------------------------------
    # EXPIRATION ALERTS
    # --------------------------------------------------------

    def expiration_alerts(
        self,
        *,
        tenant_id: str,
        as_of: date | None = None,
        within_days: int = 90,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        validate_integer(
            within_days,
            "within_days",
            maximum=36500,
        )

        today = as_of or utc_now().date()

        validate_date(today, "as_of")

        with self._lock:
            alerts = []

            for (tenant, _), warranty in self._warranties.items():
                if tenant != tenant_id:
                    continue

                if warranty.status(today) != WarrantyStatus.ACTIVE:
                    continue

                remaining = warranty.remaining_days(today)

                if remaining > within_days:
                    continue

                if remaining <= 7:
                    severity = "critical"
                elif remaining <= 30:
                    severity = "high"
                elif remaining <= 60:
                    severity = "moderate"
                else:
                    severity = "low"

                alerts.append(
                    {
                        "warranty_id": warranty.warranty_id,
                        "robot_id": warranty.robot_id,
                        "component_id": warranty.component_id,
                        "warranty_name": warranty.warranty_name,
                        "provider_name": warranty.provider_name,
                        "end_date": warranty.end_date.isoformat(),
                        "remaining_days": remaining,
                        "severity": severity,
                        "message": (
                            "Warranty is approaching expiration."
                        ),
                    }
                )

            alerts.sort(
                key=lambda item: (
                    item["remaining_days"],
                    item["warranty_id"],
                )
            )

            return alerts

    # --------------------------------------------------------
    # CREATE CLAIM
    # --------------------------------------------------------

    def create_claim(
        self,
        *,
        tenant_id: str,
        check_id: str,
        claim_title: str,
        description: str,
        requested_amount: Any,
        authorized: bool = False,
        actor_id: str = "system",
    ) -> dict[str, Any]:
        if authorized is not True:
            raise WarrantyPermissionDenied(
                "Authorized claim preparation required"
            )

        validate_id(tenant_id, "tenant_id")
        validate_id(check_id, "check_id")
        validate_id(actor_id, "actor_id")

        validate_text(
            claim_title,
            "claim_title",
        )

        validate_text(
            description,
            "description",
        )

        amount = money(
            requested_amount,
            "requested_amount",
        )

        with self._lock:
            result = self._checks.get(
                (tenant_id, check_id)
            )

            if result is None:
                raise WarrantyNotFound(
                    "Warranty check not found"
                )

            if (
                result.eligibility
                == EligibilityStatus.NOT_ELIGIBLE
            ):
                raise WarrantyConflict(
                    "Cannot prepare a claim from "
                    "an ineligible warranty check"
                )

            if len(self._claims) >= self.config.maximum_claims:
                raise WarrantyConflict(
                    "Claim storage capacity reached"
                )

            claim = WarrantyClaim(
                claim_id=new_id(),
                tenant_id=tenant_id,
                robot_id=result.robot_id,
                component_id=result.component_id,
                warranty_id=result.warranty_id,
                check_id=result.check_id,
                claim_title=claim_title,
                description=description,
                requested_amount=amount,
                currency=result.currency,
            )

            self._claims[
                (
                    tenant_id,
                    claim.claim_id,
                )
            ] = claim

            self._log(
                tenant_id=tenant_id,
                action="warranty_claim_created",
                entity_id=claim.claim_id,
                actor_id=actor_id,
                details={
                    "warranty_id": claim.warranty_id,
                    "check_id": check_id,
                },
            )

            return claim.to_dict()

    # --------------------------------------------------------
    # INTERNAL CLAIM REVIEW
    # --------------------------------------------------------

    def review_claim(
        self,
        *,
        tenant_id: str,
        claim_id: str,
        reviewer_id: str,
        decision: ClaimDecision,
        notes: str = "",
        authorized: bool = False,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise WarrantyPermissionDenied(
                "Authorized claim reviewer required"
            )

        validate_id(tenant_id, "tenant_id")
        validate_id(claim_id, "claim_id")
        validate_id(reviewer_id, "reviewer_id")

        if not isinstance(decision, ClaimDecision):
            raise WarrantyValidationError(
                "Invalid claim decision"
            )

        validate_text(
            notes,
            "notes",
            allow_empty=True,
        )

        with self._lock:
            claim = self._claims.get(
                (tenant_id, claim_id)
            )

            if claim is None:
                raise WarrantyNotFound(
                    "Warranty claim not found"
                )

            if claim.status not in (
                ClaimStatus.DRAFT,
                ClaimStatus.INTERNAL_REVIEW,
            ):
                raise WarrantyConflict(
                    "Claim cannot be reviewed in "
                    "its current status"
                )

            if decision == ClaimDecision.APPROVE_INTERNAL:
                claim.status = (
                    ClaimStatus.READY_FOR_SUBMISSION
                )
            else:
                claim.status = ClaimStatus.CLOSED

            claim.reviewer_id = reviewer_id
            claim.review_notes = notes
            claim.updated_at = utc_now()

            self._log(
                tenant_id=tenant_id,
                action="warranty_claim_reviewed",
                entity_id=claim_id,
                actor_id=reviewer_id,
                details={
                    "decision": decision.value,
                    "status": claim.status.value,
                },
            )

            return claim.to_dict()

    # --------------------------------------------------------
    # RECORD EXTERNAL CLAIM STATUS
    # --------------------------------------------------------

    def record_external_claim_status(
        self,
        *,
        tenant_id: str,
        claim_id: str,
        new_status: ClaimStatus,
        external_reference: str,
        actor_id: str,
        authorized: bool = False,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise WarrantyPermissionDenied(
                "Authorized claim status update required"
            )

        validate_id(tenant_id, "tenant_id")
        validate_id(claim_id, "claim_id")
        validate_id(actor_id, "actor_id")

        validate_text(
            external_reference,
            "external_reference",
            maximum=256,
        )

        if not isinstance(new_status, ClaimStatus):
            raise WarrantyValidationError(
                "Invalid claim status"
            )

        transitions = {
            ClaimStatus.READY_FOR_SUBMISSION: {
                ClaimStatus.SUBMITTED_EXTERNALLY,
            },
            ClaimStatus.SUBMITTED_EXTERNALLY: {
                ClaimStatus.APPROVED_EXTERNALLY,
                ClaimStatus.REJECTED_EXTERNALLY,
            },
            ClaimStatus.APPROVED_EXTERNALLY: {
                ClaimStatus.CLOSED,
            },
            ClaimStatus.REJECTED_EXTERNALLY: {
                ClaimStatus.CLOSED,
            },
        }

        with self._lock:
            claim = self._claims.get(
                (tenant_id, claim_id)
            )

            if claim is None:
                raise WarrantyNotFound(
                    "Warranty claim not found"
                )

            if new_status not in transitions.get(
                claim.status,
                set(),
            ):
                raise WarrantyConflict(
                    "Invalid claim status transition"
                )

            claim.status = new_status
            claim.external_reference = external_reference
            claim.updated_at = utc_now()

            self._log(
                tenant_id=tenant_id,
                action="external_claim_status_recorded",
                entity_id=claim_id,
                actor_id=actor_id,
                details={
                    "status": new_status.value,
                    "external_reference": external_reference,
                },
            )

            return claim.to_dict()

    # --------------------------------------------------------
    # GET CLAIM
    # --------------------------------------------------------

    def get_claim(
        self,
        *,
        tenant_id: str,
        claim_id: str,
    ) -> dict[str, Any]:
        validate_id(tenant_id, "tenant_id")
        validate_id(claim_id, "claim_id")

        with self._lock:
            claim = self._claims.get(
                (tenant_id, claim_id)
            )

            if claim is None:
                raise WarrantyNotFound(
                    "Warranty claim not found"
                )

            return claim.to_dict()

    # --------------------------------------------------------
    # LIST CLAIMS
    # --------------------------------------------------------

    def list_claims(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
        status: ClaimStatus | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        if robot_id is not None:
            validate_id(robot_id, "robot_id")

        if status is not None and not isinstance(
            status,
            ClaimStatus,
        ):
            raise WarrantyValidationError(
                "Invalid claim status"
            )

        validate_integer(
            limit,
            "limit",
            minimum=1,
            maximum=100000,
        )

        with self._lock:
            claims = [
                claim
                for (tenant, _), claim
                in self._claims.items()
                if (
                    tenant == tenant_id
                    and (
                        robot_id is None
                        or claim.robot_id == robot_id
                    )
                    and (
                        status is None
                        or claim.status == status
                    )
                )
            ]

            claims.sort(
                key=lambda item: item.created_at,
                reverse=True,
            )

            return [
                item.to_dict()
                for item in claims[:limit]
            ]

    # --------------------------------------------------------
    # ROBOT WARRANTY SUMMARY
    # --------------------------------------------------------

    def robot_warranty_summary(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        as_of: date | None = None,
    ) -> dict[str, Any]:
        warranties = self.list_warranties(
            tenant_id=tenant_id,
            robot_id=robot_id,
            as_of=as_of,
            limit=100000,
        )

        claims = self.list_claims(
            tenant_id=tenant_id,
            robot_id=robot_id,
            limit=100000,
        )

        statuses = Counter(
            item["status"]
            for item in warranties
        )

        claim_statuses = Counter(
            item["status"]
            for item in claims
        )

        return {
            "tenant_id": tenant_id,
            "robot_id": robot_id,
            "total_warranties": len(warranties),
            "warranty_statuses": dict(statuses),
            "active_warranties": statuses.get(
                WarrantyStatus.ACTIVE.value,
                0,
            ),
            "expired_warranties": statuses.get(
                WarrantyStatus.EXPIRED.value,
                0,
            ),
            "total_claims": len(claims),
            "claim_statuses": dict(claim_statuses),
            "advisory_only": True,
        }

    # --------------------------------------------------------
    # AUDIT HISTORY
    # --------------------------------------------------------

    def audit_history(
        self,
        *,
        tenant_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        validate_integer(
            limit,
            "limit",
            minimum=1,
            maximum=100000,
        )

        with self._lock:
            events = [
                dict(event)
                for event in self._audit
                if event["tenant_id"] == tenant_id
            ]

            return events[-limit:]

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
            warranties = [
                warranty.to_dict()
                for (tenant, _), warranty
                in self._warranties.items()
                if tenant == tenant_id
            ]

            checks = [
                result.to_dict()
                for (tenant, _), result
                in self._checks.items()
                if tenant == tenant_id
            ]

            claims = [
                claim.to_dict()
                for (tenant, _), claim
                in self._claims.items()
                if tenant == tenant_id
            ]

            audit = [
                dict(event)
                for event in self._audit
                if event["tenant_id"] == tenant_id
            ]

        payload = {
            "generated_at": utc_now().isoformat(),
            "tenant_id": tenant_id,
            "warranties": warranties,
            "warranty_checks": checks,
            "claims": claims,
            "audit_events": audit,
            "advisory_only": True,
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
    checker = RoboticsWarrantyChecker()

    today = utc_now().date()

    warranty = RobotWarranty(
        warranty_id="warranty_motor_001",
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        component_id="joint_motor_001",
        warranty_name="Robot Joint Motor Warranty",
        warranty_type=WarrantyType.MANUFACTURER,
        manufacturer="Demo Robotics",
        provider_name="Demo Robotics Service",
        start_date=today - timedelta(days=300),
        end_date=today + timedelta(days=65),
        serial_number="MTR100-SN-001",
        part_number="DR-MTR-100",
        model_number="MTR-100",
        policy_number="POL-MTR-001",
        coverage=(
            WarrantyCoverage(
                coverage_type=CoverageType.PARTS,
                covered=True,
                coverage_percentage=Decimal("100"),
                maximum_covered_amount=Decimal("20000"),
                deductible=Decimal("0"),
            ),
            WarrantyCoverage(
                coverage_type=CoverageType.LABOR,
                covered=True,
                coverage_percentage=Decimal("80"),
                maximum_covered_amount=Decimal("5000"),
                deductible=Decimal("250"),
            ),
            WarrantyCoverage(
                coverage_type=CoverageType.REPLACEMENT,
                covered=True,
                coverage_percentage=Decimal("90"),
                maximum_covered_amount=Decimal("18000"),
                deductible=Decimal("500"),
            ),
        ),
        excluded_failure_causes=(
            FailureCause.MISUSE,
            FailureCause.UNAUTHORIZED_MODIFICATION,
            FailureCause.ACCIDENTAL_DAMAGE,
        ),
        maintenance_compliance_required=True,
        maximum_operating_hours=Decimal("10000"),
        currency="INR",
        document_references=(
            "document_warranty_motor_001",
        ),
        notes="Synthetic demonstration warranty.",
    )

    checker.register_warranty(
        warranty,
        authorized=True,
        actor_id="admin_001",
    )

    request = WarrantyCheckRequest(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        component_id="joint_motor_001",
        coverage_type=CoverageType.REPLACEMENT,
        failure_cause=FailureCause.MANUFACTURING_DEFECT,
        serial_number="MTR100-SN-001",
        part_number="DR-MTR-100",
        operating_hours=Decimal("6200"),
        maintenance_compliant=True,
        estimated_cost=Decimal("15000"),
        currency="INR",
        incident_date=today,
        notes="Demo motor replacement warranty check",
    )

    result = checker.check_warranty(
        warranty_id="warranty_motor_001",
        request=request,
    )

    print("\nWARRANTY CHECK RESULT")

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    print("\nEXPIRATION ALERTS")

    print(
        json.dumps(
            checker.expiration_alerts(
                tenant_id="tenant_alpha",
                within_days=90,
            ),
            indent=2,
        )
    )

    claim = checker.create_claim(
        tenant_id="tenant_alpha",
        check_id=result["check_id"],
        claim_title="Robot Joint Motor Replacement",
        description=(
            "Replacement request for a motor "
            "with a reported manufacturing defect."
        ),
        requested_amount=Decimal("13000"),
        authorized=True,
        actor_id="service_manager_001",
    )

    print("\nWARRANTY CLAIM")

    print(
        json.dumps(
            claim,
            indent=2,
        )
    )

    reviewed = checker.review_claim(
        tenant_id="tenant_alpha",
        claim_id=claim["claim_id"],
        reviewer_id="engineer_001",
        decision=ClaimDecision.APPROVE_INTERNAL,
        notes="Documentation ready for provider submission.",
        authorized=True,
    )

    print("\nINTERNAL CLAIM REVIEW")

    print(
        json.dumps(
            reviewed,
            indent=2,
        )
    )

    print("\nROBOT WARRANTY SUMMARY")

    print(
        json.dumps(
            checker.robot_warranty_summary(
                tenant_id="tenant_alpha",
                robot_id="robot_001",
            ),
            indent=2,
        )
    )

    output = checker.export_json(
        "data/robotics/warranty_checker_report.json",
        tenant_id="tenant_alpha",
    )

    print("\nExported:", output)


if __name__ == "__main__":
    main()
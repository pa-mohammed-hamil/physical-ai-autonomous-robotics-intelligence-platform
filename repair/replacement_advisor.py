"""Replacement Advisor module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/replacement_advisor.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Robotics Component Replacement Advisor

Python 3.11+
Dependencies: Standard library only

Features:
    - Robot component replacement recommendations
    - Component health, age and remaining useful life analysis
    - Fault severity and reliability assessment
    - Replacement-part compatibility verification
    - OEM and aftermarket replacement comparison
    - Replacement pricing and supplier evaluation
    - Stock availability and procurement lead times
    - Warranty and manufacturer support assessment
    - Repair vs replacement decision integration
    - Replacement urgency classification
    - Downtime and lifecycle cost estimation
    - Budget-aware replacement selection
    - Human safety approval requirements
    - Multi-tenant data isolation
    - Thread-safe advisory history
    - Audit logging and JSON export

Integrations:
    - issue_diagnosis.py
    - damage_analysis.py
    - damage_severity.py
    - part_detection.py
    - part_identification.py
    - parts_catalog.py
    - parts_pricing.py
    - repair_vs_replace.py
    - repair_report.py
    - maintenance_service.py

Safety:
    This module only generates recommendations.

    It does not:
        - Order parts
        - Execute repairs
        - Authorize robot movement
        - Install replacement components
        - Release equipment back into service

    Replacement compatibility and safety must be
    verified by qualified engineering personnel.
"""

from __future__ import annotations

import json
import threading
import uuid

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================

class ReplacementAdvisorError(Exception):
    pass


class ReplacementValidationError(ReplacementAdvisorError):
    pass


class ReplacementNotFound(ReplacementAdvisorError):
    pass


class ReplacementConflict(ReplacementAdvisorError):
    pass


class ReplacementPermissionDenied(ReplacementAdvisorError):
    pass


# ============================================================
# ENUMS
# ============================================================

class ComponentCondition(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ComponentCriticality(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SAFETY_CRITICAL = "safety_critical"


class FaultSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ReplacementUrgency(str, Enum):
    ROUTINE = "routine"
    PLANNED = "planned"
    PRIORITY = "priority"
    URGENT = "urgent"
    CRITICAL = "critical"


class ReplacementRecommendation(str, Enum):
    CONTINUE_MONITORING = "continue_monitoring"
    INSPECT_COMPONENT = "inspect_component"
    SCHEDULE_REPLACEMENT = "schedule_replacement"
    PRIORITIZE_REPLACEMENT = "prioritize_replacement"
    ENGINEERING_REVIEW = "engineering_review"
    INSUFFICIENT_DATA = "insufficient_data"


class PartType(str, Enum):
    OEM = "oem"
    AFTERMARKET = "aftermarket"
    REFURBISHED = "refurbished"


class StockStatus(str, Enum):
    IN_STOCK = "in_stock"
    LIMITED_STOCK = "limited_stock"
    OUT_OF_STOCK = "out_of_stock"
    BACKORDER = "backorder"
    UNKNOWN = "unknown"


class CompatibilityStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    INCOMPATIBLE = "incompatible"


class AdviceStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"


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
        raise ReplacementValidationError(
            f"Invalid {name}"
        )
    return value


def validate_text(
    value: str,
    name: str,
    *,
    allow_empty: bool = False,
    maximum: int = 2048,
) -> str:
    if not isinstance(value, str):
        raise ReplacementValidationError(
            f"{name} must be a string"
        )

    result = value.strip()

    if len(result) > maximum or (
        not allow_empty and not result
    ):
        raise ReplacementValidationError(
            f"Invalid {name}"
        )

    return result


def decimal_value(
    value: Any,
    name: str,
    *,
    minimum: Decimal = Decimal("0"),
    maximum: Decimal = Decimal("1000000000000"),
) -> Decimal:
    if isinstance(value, bool):
        raise ReplacementValidationError(
            f"Invalid {name}"
        )

    try:
        result = Decimal(str(value))
    except (
        InvalidOperation,
        TypeError,
        ValueError,
    ) as exc:
        raise ReplacementValidationError(
            f"Invalid {name}"
        ) from exc

    if (
        not result.is_finite()
        or result < minimum
        or result > maximum
    ):
        raise ReplacementValidationError(
            f"Invalid {name}"
        )

    return result


def money(value: Any, name: str = "amount") -> Decimal:
    return decimal_value(value, name).quantize(
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


def probability(value: Any, name: str) -> Decimal:
    return decimal_value(
        value,
        name,
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
        raise ReplacementValidationError(
            "Invalid currency"
        )
    return value


def validate_integer(
    value: int,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = 1000000,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ReplacementValidationError(
            f"Invalid {name}"
        )
    return value


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class ReplacementAdvisorConfig:
    default_currency: str = "INR"

    low_health_threshold: Decimal = Decimal("40")
    critical_health_threshold: Decimal = Decimal("20")

    low_rul_months: Decimal = Decimal("12")
    critical_rul_months: Decimal = Decimal("3")

    high_failure_probability: Decimal = Decimal("0.25")
    critical_failure_probability: Decimal = Decimal("0.50")

    maximum_advice_records: int = 100000
    maximum_audit_events: int = 20000
    maximum_candidates: int = 100

    simulation_only: bool = True

    def __post_init__(self) -> None:
        validate_currency(self.default_currency)

        for name in (
            "low_health_threshold",
            "critical_health_threshold",
        ):
            decimal_value(
                getattr(self, name),
                name,
                maximum=Decimal("100"),
            )

        if (
            self.critical_health_threshold
            > self.low_health_threshold
        ):
            raise ReplacementValidationError(
                "Critical health threshold must not "
                "exceed low health threshold"
            )

        for name in (
            "low_rul_months",
            "critical_rul_months",
        ):
            decimal_value(
                getattr(self, name),
                name,
                maximum=Decimal("1200"),
            )

        if self.critical_rul_months > self.low_rul_months:
            raise ReplacementValidationError(
                "Critical RUL threshold must not "
                "exceed low RUL threshold"
            )

        for name in (
            "high_failure_probability",
            "critical_failure_probability",
        ):
            probability(
                getattr(self, name),
                name,
            )

        if (
            self.critical_failure_probability
            < self.high_failure_probability
        ):
            raise ReplacementValidationError(
                "Critical failure threshold must be "
                "at least the high failure threshold"
            )

        for name in (
            "maximum_advice_records",
            "maximum_audit_events",
            "maximum_candidates",
        ):
            validate_integer(
                getattr(self, name),
                name,
                minimum=1,
            )

        if self.simulation_only is not True:
            raise ReplacementValidationError(
                "Only advisory mode is supported"
            )


# ============================================================
# COMPONENT HEALTH
# ============================================================

@dataclass(frozen=True)
class ComponentHealthProfile:
    tenant_id: str
    robot_id: str
    component_id: str
    component_name: str

    manufacturer: str = ""
    model_number: str = ""
    original_part_number: str = ""

    condition: ComponentCondition = (
        ComponentCondition.UNKNOWN
    )

    criticality: ComponentCriticality = (
        ComponentCriticality.MODERATE
    )

    fault_severity: FaultSeverity = FaultSeverity.UNKNOWN

    health_score: Decimal | None = None
    remaining_useful_life_months: Decimal | None = None
    annual_failure_probability: Decimal | None = None

    age_months: Decimal = Decimal("0")
    design_life_months: Decimal = Decimal("60")

    recurring_failures: int = 0

    obsolete: bool = False
    manufacturer_support_available: bool = True

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
            self.component_name,
            "component_name",
        )

        for name in (
            "manufacturer",
            "model_number",
            "original_part_number",
        ):
            validate_text(
                getattr(self, name),
                name,
                allow_empty=True,
            )

        for name, enum_type in (
            ("condition", ComponentCondition),
            ("criticality", ComponentCriticality),
            ("fault_severity", FaultSeverity),
        ):
            if not isinstance(
                getattr(self, name),
                enum_type,
            ):
                raise ReplacementValidationError(
                    f"Invalid {name}"
                )

        if self.health_score is not None:
            decimal_value(
                self.health_score,
                "health_score",
                maximum=Decimal("100"),
            )

        if self.remaining_useful_life_months is not None:
            decimal_value(
                self.remaining_useful_life_months,
                "remaining_useful_life_months",
                maximum=Decimal("1200"),
            )

        if self.annual_failure_probability is not None:
            probability(
                self.annual_failure_probability,
                "annual_failure_probability",
            )

        decimal_value(
            self.age_months,
            "age_months",
            maximum=Decimal("1200"),
        )

        decimal_value(
            self.design_life_months,
            "design_life_months",
            minimum=Decimal("0.01"),
            maximum=Decimal("1200"),
        )

        validate_integer(
            self.recurring_failures,
            "recurring_failures",
        )

        if not isinstance(self.obsolete, bool):
            raise ReplacementValidationError(
                "obsolete must be boolean"
            )

        if not isinstance(
            self.manufacturer_support_available,
            bool,
        ):
            raise ReplacementValidationError(
                "manufacturer_support_available "
                "must be boolean"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "component_id": self.component_id,
            "component_name": self.component_name,
            "manufacturer": self.manufacturer,
            "model_number": self.model_number,
            "original_part_number": (
                self.original_part_number
            ),
            "condition": self.condition.value,
            "criticality": self.criticality.value,
            "fault_severity": self.fault_severity.value,
            "health_score": (
                str(self.health_score)
                if self.health_score is not None
                else None
            ),
            "remaining_useful_life_months": (
                str(self.remaining_useful_life_months)
                if self.remaining_useful_life_months
                is not None
                else None
            ),
            "annual_failure_probability": (
                str(self.annual_failure_probability)
                if self.annual_failure_probability
                is not None
                else None
            ),
            "age_months": str(self.age_months),
            "design_life_months": str(
                self.design_life_months
            ),
            "recurring_failures": (
                self.recurring_failures
            ),
            "obsolete": self.obsolete,
            "manufacturer_support_available": (
                self.manufacturer_support_available
            ),
        }


# ============================================================
# REPLACEMENT PART CANDIDATE
# ============================================================

@dataclass(frozen=True)
class ReplacementPartCandidate:
    candidate_id: str
    part_id: str
    part_name: str
    manufacturer: str
    part_number: str

    part_type: PartType

    unit_price: Decimal
    currency: str

    supplier_id: str = ""

    stock_status: StockStatus = StockStatus.UNKNOWN
    available_quantity: int = 0
    lead_time_days: int = 0

    compatibility: CompatibilityStatus = (
        CompatibilityStatus.UNVERIFIED
    )

    expected_life_months: Decimal = Decimal("60")
    expected_reliability: Decimal = Decimal("0.90")

    warranty_months: Decimal = Decimal("0")

    installation_cost: Decimal = Decimal("0")
    shipping_cost: Decimal = Decimal("0")
    downtime_cost: Decimal = Decimal("0")

    annual_maintenance_cost: Decimal = Decimal("0")

    catalog_reference: str = ""
    pricing_reference: str = ""

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "part_id",
        ):
            validate_id(
                getattr(self, name),
                name,
            )

        for name in (
            "part_name",
            "manufacturer",
            "part_number",
        ):
            validate_text(
                getattr(self, name),
                name,
            )

        if self.supplier_id:
            validate_id(
                self.supplier_id,
                "supplier_id",
            )

        if not isinstance(self.part_type, PartType):
            raise ReplacementValidationError(
                "Invalid part type"
            )

        if not isinstance(
            self.stock_status,
            StockStatus,
        ):
            raise ReplacementValidationError(
                "Invalid stock status"
            )

        if not isinstance(
            self.compatibility,
            CompatibilityStatus,
        ):
            raise ReplacementValidationError(
                "Invalid compatibility status"
            )

        validate_currency(self.currency)

        for name in (
            "unit_price",
            "installation_cost",
            "shipping_cost",
            "downtime_cost",
            "annual_maintenance_cost",
        ):
            object.__setattr__(
                self,
                name,
                money(
                    getattr(self, name),
                    name,
                ),
            )

        for name in (
            "available_quantity",
            "lead_time_days",
        ):
            validate_integer(
                getattr(self, name),
                name,
            )

        for name in (
            "expected_life_months",
            "warranty_months",
        ):
            decimal_value(
                getattr(self, name),
                name,
                maximum=Decimal("1200"),
            )

        probability(
            self.expected_reliability,
            "expected_reliability",
        )

        for name in (
            "catalog_reference",
            "pricing_reference",
        ):
            validate_text(
                getattr(self, name),
                name,
                allow_empty=True,
            )

    @property
    def initial_cost(self) -> Decimal:
        return (
            self.unit_price
            + self.installation_cost
            + self.shipping_cost
            + self.downtime_cost
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "part_id": self.part_id,
            "part_name": self.part_name,
            "manufacturer": self.manufacturer,
            "part_number": self.part_number,
            "part_type": self.part_type.value,
            "unit_price": money_string(
                self.unit_price
            ),
            "currency": self.currency,
            "supplier_id": self.supplier_id,
            "stock_status": self.stock_status.value,
            "available_quantity": (
                self.available_quantity
            ),
            "lead_time_days": self.lead_time_days,
            "compatibility": self.compatibility.value,
            "expected_life_months": str(
                self.expected_life_months
            ),
            "expected_reliability": str(
                self.expected_reliability
            ),
            "warranty_months": str(
                self.warranty_months
            ),
            "installation_cost": money_string(
                self.installation_cost
            ),
            "shipping_cost": money_string(
                self.shipping_cost
            ),
            "downtime_cost": money_string(
                self.downtime_cost
            ),
            "annual_maintenance_cost": money_string(
                self.annual_maintenance_cost
            ),
            "initial_cost": money_string(
                self.initial_cost
            ),
            "catalog_reference": self.catalog_reference,
            "pricing_reference": self.pricing_reference,
        }


# ============================================================
# ADVISORY REQUEST
# ============================================================

@dataclass(frozen=True)
class ReplacementAdviceRequest:
    component: ComponentHealthProfile

    candidates: tuple[
        ReplacementPartCandidate,
        ...
    ] = ()

    currency: str = "INR"

    budget: Decimal | None = None
    maximum_lead_time_days: int | None = None

    repair_cost_estimate: Decimal | None = None
    repair_vs_replace_recommendation: str = ""

    required_quantity: int = 1

    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(
            self.component,
            ComponentHealthProfile,
        ):
            raise ReplacementValidationError(
                "Invalid component profile"
            )

        if not isinstance(self.candidates, tuple):
            raise ReplacementValidationError(
                "candidates must be a tuple"
            )

        identifiers = set()

        for candidate in self.candidates:
            if not isinstance(
                candidate,
                ReplacementPartCandidate,
            ):
                raise ReplacementValidationError(
                    "Invalid replacement candidate"
                )

            if candidate.candidate_id in identifiers:
                raise ReplacementValidationError(
                    "Duplicate replacement candidate ID"
                )

            identifiers.add(candidate.candidate_id)

        validate_currency(self.currency)

        if self.budget is not None:
            object.__setattr__(
                self,
                "budget",
                money(self.budget, "budget"),
            )

        if self.maximum_lead_time_days is not None:
            validate_integer(
                self.maximum_lead_time_days,
                "maximum_lead_time_days",
            )

        if self.repair_cost_estimate is not None:
            object.__setattr__(
                self,
                "repair_cost_estimate",
                money(
                    self.repair_cost_estimate,
                    "repair_cost_estimate",
                ),
            )

        validate_integer(
            self.required_quantity,
            "required_quantity",
            minimum=1,
        )

        validate_text(
            self.repair_vs_replace_recommendation,
            "repair_vs_replace_recommendation",
            allow_empty=True,
        )

        validate_text(
            self.notes,
            "notes",
            allow_empty=True,
        )


# ============================================================
# RANKED CANDIDATE
# ============================================================

@dataclass(frozen=True)
class RankedReplacementCandidate:
    candidate: ReplacementPartCandidate

    feasible: bool
    score: Decimal

    total_initial_cost: Decimal
    projected_annual_maintenance_cost: Decimal

    reasons: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.candidate.to_dict(),
            "feasible": self.feasible,
            "score": str(self.score),
            "total_initial_cost": money_string(
                self.total_initial_cost
            ),
            "projected_annual_maintenance_cost": (
                money_string(
                    self.projected_annual_maintenance_cost
                )
            ),
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
        }


# ============================================================
# ADVISORY RECORD
# ============================================================

@dataclass
class ReplacementAdviceRecord:
    advice_id: str

    tenant_id: str
    robot_id: str
    component_id: str

    recommendation: ReplacementRecommendation
    urgency: ReplacementUrgency

    health_risk_score: Decimal

    ranked_candidates: list[
        RankedReplacementCandidate
    ]

    selected_candidate_id: str | None

    reasons: list[str]
    warnings: list[str]

    status: AdviceStatus = AdviceStatus.PENDING_REVIEW

    reviewer_id: str | None = None
    review_notes: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    reviewed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "advice_id": self.advice_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "component_id": self.component_id,
            "recommendation": (
                self.recommendation.value
            ),
            "urgency": self.urgency.value,
            "health_risk_score": str(
                self.health_risk_score
            ),
            "ranked_candidates": [
                item.to_dict()
                for item in self.ranked_candidates
            ],
            "selected_candidate_id": (
                self.selected_candidate_id
            ),
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "status": self.status.value,
            "reviewer_id": self.reviewer_id,
            "review_notes": self.review_notes,
            "created_at": self.created_at.isoformat(),
            "reviewed_at": (
                self.reviewed_at.isoformat()
                if self.reviewed_at
                else None
            ),
            "human_review_required": True,
            "advisory_only": True,
            "procurement_authorized": False,
            "robot_operation_authorized": False,
        }


# ============================================================
# REPLACEMENT ADVISOR ENGINE
# ============================================================

class RoboticsReplacementAdvisor:

    def __init__(
        self,
        config: ReplacementAdvisorConfig | None = None,
    ) -> None:
        self.config = (
            config or ReplacementAdvisorConfig()
        )

        self._records: dict[
            tuple[str, str],
            ReplacementAdviceRecord,
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
        advice_id: str,
        action: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._audit.append(
            {
                "event_id": new_id(),
                "timestamp": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "advice_id": advice_id,
                "action": action,
                "details": details or {},
            }
        )

    # --------------------------------------------------------
    # HEALTH RISK
    # --------------------------------------------------------

    def calculate_health_risk(
        self,
        component: ComponentHealthProfile,
    ) -> tuple[
        Decimal,
        list[str],
        int,
    ]:
        """
        Heuristic risk score from 0 to 100.

        This is not a calibrated probability of
        failure or a certified safety assessment.
        """
        score = Decimal("0")
        reasons = []
        observations = 0

        if component.health_score is not None:
            observations += 1

            score += (
                Decimal("100")
                - component.health_score
            ) * Decimal("0.30")

            if (
                component.health_score
                <= self.config.critical_health_threshold
            ):
                reasons.append(
                    "Component health is critically low."
                )

            elif (
                component.health_score
                <= self.config.low_health_threshold
            ):
                reasons.append(
                    "Component health is below threshold."
                )

        if (
            component.remaining_useful_life_months
            is not None
        ):
            observations += 1

            rul = (
                component.remaining_useful_life_months
            )

            if rul <= self.config.critical_rul_months:
                score += Decimal("25")
                reasons.append(
                    "Remaining useful life is critically low."
                )

            elif rul <= self.config.low_rul_months:
                score += Decimal("15")
                reasons.append(
                    "Remaining useful life is low."
                )

        if component.annual_failure_probability is not None:
            observations += 1

            failure_probability = (
                component.annual_failure_probability
            )

            score += (
                failure_probability
                * Decimal("25")
            )

            if (
                failure_probability
                >= self.config.critical_failure_probability
            ):
                reasons.append(
                    "Estimated failure probability is critical."
                )

            elif (
                failure_probability
                >= self.config.high_failure_probability
            ):
                reasons.append(
                    "Estimated failure probability is elevated."
                )

        severity_points = {
            FaultSeverity.NONE: Decimal("0"),
            FaultSeverity.LOW: Decimal("3"),
            FaultSeverity.MODERATE: Decimal("8"),
            FaultSeverity.HIGH: Decimal("15"),
            FaultSeverity.CRITICAL: Decimal("25"),
            FaultSeverity.UNKNOWN: Decimal("0"),
        }

        if component.fault_severity != FaultSeverity.UNKNOWN:
            observations += 1

        score += severity_points[
            component.fault_severity
        ]

        if component.condition != ComponentCondition.UNKNOWN:
            observations += 1

            condition_points = {
                ComponentCondition.EXCELLENT: Decimal("0"),
                ComponentCondition.GOOD: Decimal("2"),
                ComponentCondition.FAIR: Decimal("6"),
                ComponentCondition.POOR: Decimal("12"),
                ComponentCondition.FAILED: Decimal("25"),
                ComponentCondition.UNKNOWN: Decimal("0"),
            }

            score += condition_points[
                component.condition
            ]

        if component.recurring_failures:
            observations += 1

            score += min(
                Decimal("15"),
                Decimal(component.recurring_failures)
                * Decimal("3"),
            )

            reasons.append(
                "Component has recurring failure history."
            )

        if (
            component.age_months
            >= component.design_life_months
        ):
            score += Decimal("8")

            reasons.append(
                "Component has reached its nominal design life."
            )

        if component.obsolete:
            score += Decimal("5")

            reasons.append(
                "Component is marked obsolete."
            )

        if not component.manufacturer_support_available:
            score += Decimal("5")

            reasons.append(
                "Manufacturer support is unavailable."
            )

        return (
            min(
                Decimal("100"),
                score,
            ).quantize(
                Decimal("0.01"),
                rounding=ROUND_HALF_UP,
            ),
            reasons,
            observations,
        )

    # --------------------------------------------------------
    # URGENCY
    # --------------------------------------------------------

    def classify_urgency(
        self,
        component: ComponentHealthProfile,
        risk_score: Decimal,
    ) -> ReplacementUrgency:
        if (
            component.condition == ComponentCondition.FAILED
            or component.fault_severity
            == FaultSeverity.CRITICAL
        ):
            return ReplacementUrgency.CRITICAL

        if (
            component.criticality
            == ComponentCriticality.SAFETY_CRITICAL
            and (
                component.fault_severity
                in (
                    FaultSeverity.HIGH,
                    FaultSeverity.CRITICAL,
                )
                or risk_score >= Decimal("60")
            )
        ):
            return ReplacementUrgency.CRITICAL

        if risk_score >= Decimal("70"):
            return ReplacementUrgency.URGENT

        if risk_score >= Decimal("45"):
            return ReplacementUrgency.PRIORITY

        if risk_score >= Decimal("20"):
            return ReplacementUrgency.PLANNED

        return ReplacementUrgency.ROUTINE

    # --------------------------------------------------------
    # CANDIDATE EVALUATION
    # --------------------------------------------------------

    def evaluate_candidate(
        self,
        candidate: ReplacementPartCandidate,
        request: ReplacementAdviceRequest,
    ) -> RankedReplacementCandidate:
        reasons = []
        warnings = []

        feasible = True

        if candidate.currency != request.currency:
            feasible = False
            warnings.append(
                "Candidate currency does not match "
                "the analysis currency."
            )

        if (
            candidate.compatibility
            != CompatibilityStatus.VERIFIED
        ):
            feasible = False

            warnings.append(
                "Replacement compatibility is not verified."
            )

        if candidate.stock_status in (
            StockStatus.OUT_OF_STOCK,
            StockStatus.BACKORDER,
            StockStatus.UNKNOWN,
        ):
            feasible = False

            warnings.append(
                "Part is not confirmed available."
            )

        if (
            candidate.available_quantity
            < request.required_quantity
        ):
            feasible = False

            warnings.append(
                "Insufficient confirmed stock."
            )

        total_cost = (
            candidate.unit_price
            * request.required_quantity
            + candidate.installation_cost
            + candidate.shipping_cost
            + candidate.downtime_cost
        ).quantize(
            CENT,
            rounding=ROUND_HALF_UP,
        )

        if (
            request.budget is not None
            and total_cost > request.budget
        ):
            feasible = False

            warnings.append(
                "Candidate exceeds the replacement budget."
            )

        if (
            request.maximum_lead_time_days is not None
            and candidate.lead_time_days
            > request.maximum_lead_time_days
        ):
            feasible = False

            warnings.append(
                "Candidate exceeds the maximum lead time."
            )

        # Heuristic quality score, not a certification.
        reliability_score = (
            candidate.expected_reliability
            * Decimal("40")
        )

        life_score = min(
            Decimal("20"),
            candidate.expected_life_months
            / Decimal("120")
            * Decimal("20"),
        )

        warranty_score = min(
            Decimal("10"),
            candidate.warranty_months
            / Decimal("36")
            * Decimal("10"),
        )

        lead_time_score = max(
            Decimal("0"),
            Decimal("10")
            - Decimal(candidate.lead_time_days)
            / Decimal("10"),
        )

        stock_score = (
            Decimal("10")
            if candidate.stock_status == StockStatus.IN_STOCK
            else Decimal("5")
            if candidate.stock_status == StockStatus.LIMITED_STOCK
            else Decimal("0")
        )

        if request.budget is not None and request.budget > 0:
            cost_score = max(
                Decimal("0"),
                Decimal("20")
                * (
                    Decimal("1")
                    - total_cost / request.budget
                ),
            )
        else:
            cost_score = Decimal("10")

        score = (
            reliability_score
            + life_score
            + warranty_score
            + lead_time_score
            + stock_score
            + cost_score
        )

        if candidate.part_type == PartType.OEM:
            reasons.append(
                "OEM replacement candidate."
            )

        if candidate.warranty_months >= Decimal("24"):
            reasons.append(
                "Candidate includes at least 24 months "
                "of stated warranty coverage."
            )

        if candidate.expected_reliability >= Decimal("0.95"):
            reasons.append(
                "Candidate has high stated reliability."
            )

        return RankedReplacementCandidate(
            candidate=candidate,
            feasible=feasible,
            score=score.quantize(
                Decimal("0.01"),
                rounding=ROUND_HALF_UP,
            ),
            total_initial_cost=total_cost,
            projected_annual_maintenance_cost=(
                candidate.annual_maintenance_cost
            ),
            reasons=tuple(reasons),
            warnings=tuple(warnings),
        )

    # --------------------------------------------------------
    # GENERATE ADVICE
    # --------------------------------------------------------

    def advise(
        self,
        request: ReplacementAdviceRequest,
    ) -> dict[str, Any]:
        if not isinstance(
            request,
            ReplacementAdviceRequest,
        ):
            raise ReplacementValidationError(
                "Invalid replacement advice request"
            )

        if (
            len(request.candidates)
            > self.config.maximum_candidates
        ):
            raise ReplacementValidationError(
                "Too many replacement candidates"
            )

        component = request.component

        risk_score, reasons, observations = (
            self.calculate_health_risk(component)
        )

        urgency = self.classify_urgency(
            component,
            risk_score,
        )

        warnings = []

        if observations == 0:
            recommendation = (
                ReplacementRecommendation.INSUFFICIENT_DATA
            )

            warnings.append(
                "Insufficient health and diagnostic evidence."
            )

        elif urgency == ReplacementUrgency.CRITICAL:
            recommendation = (
                ReplacementRecommendation.ENGINEERING_REVIEW
            )

            reasons.append(
                "Critical component condition requires "
                "immediate qualified engineering assessment."
            )

        elif urgency == ReplacementUrgency.URGENT:
            recommendation = (
                ReplacementRecommendation.PRIORITIZE_REPLACEMENT
            )

        elif urgency in (
            ReplacementUrgency.PRIORITY,
            ReplacementUrgency.PLANNED,
        ):
            recommendation = (
                ReplacementRecommendation.SCHEDULE_REPLACEMENT
            )

        elif (
            component.condition
            == ComponentCondition.UNKNOWN
        ):
            recommendation = (
                ReplacementRecommendation.INSPECT_COMPONENT
            )

        else:
            recommendation = (
                ReplacementRecommendation.CONTINUE_MONITORING
            )

        ranked = [
            self.evaluate_candidate(
                candidate,
                request,
            )
            for candidate in request.candidates
        ]

        ranked.sort(
            key=lambda item: (
                not item.feasible,
                -item.score,
                item.total_initial_cost,
                item.candidate.candidate_id,
            )
        )

        selected_candidate_id = None

        feasible_candidates = [
            item
            for item in ranked
            if item.feasible
        ]

        if feasible_candidates:
            selected_candidate_id = (
                feasible_candidates[0]
                .candidate.candidate_id
            )

        elif request.candidates:
            warnings.append(
                "No replacement candidate satisfies "
                "all configured constraints."
            )

        if component.criticality == (
            ComponentCriticality.SAFETY_CRITICAL
        ):
            warnings.append(
                "Safety-critical component requires "
                "qualified engineering approval."
            )

        if request.repair_vs_replace_recommendation:
            reasons.append(
                "External repair-vs-replace advisory: "
                + request.repair_vs_replace_recommendation
            )

        if (
            request.repair_cost_estimate is not None
            and feasible_candidates
        ):
            lowest_replacement_cost = min(
                item.total_initial_cost
                for item in feasible_candidates
            )

            if (
                request.repair_cost_estimate
                >= lowest_replacement_cost
            ):
                reasons.append(
                    "Supplied repair estimate is at least "
                    "as high as the lowest feasible "
                    "replacement initial cost."
                )

        if (
            recommendation
            in (
                ReplacementRecommendation.SCHEDULE_REPLACEMENT,
                ReplacementRecommendation.PRIORITIZE_REPLACEMENT,
            )
            and not feasible_candidates
        ):
            recommendation = (
                ReplacementRecommendation.ENGINEERING_REVIEW
            )

            warnings.append(
                "Replacement is indicated, but no "
                "verified feasible part is available."
            )

        record = ReplacementAdviceRecord(
            advice_id=new_id(),
            tenant_id=component.tenant_id,
            robot_id=component.robot_id,
            component_id=component.component_id,
            recommendation=recommendation,
            urgency=urgency,
            health_risk_score=risk_score,
            ranked_candidates=ranked,
            selected_candidate_id=selected_candidate_id,
            reasons=reasons,
            warnings=warnings,
        )

        with self._lock:
            if (
                len(self._records)
                >= self.config.maximum_advice_records
            ):
                raise ReplacementAdvisorError(
                    "Advisory storage capacity reached"
                )

            self._records[
                (
                    record.tenant_id,
                    record.advice_id,
                )
            ] = record

            self._log(
                tenant_id=record.tenant_id,
                advice_id=record.advice_id,
                action="replacement_advice_created",
                details={
                    "recommendation": (
                        recommendation.value
                    ),
                    "urgency": urgency.value,
                },
            )

            return record.to_dict()

    # --------------------------------------------------------
    # GET ADVICE
    # --------------------------------------------------------

    def get_advice(
        self,
        *,
        tenant_id: str,
        advice_id: str,
    ) -> dict[str, Any]:
        validate_id(tenant_id, "tenant_id")
        validate_id(advice_id, "advice_id")

        with self._lock:
            record = self._records.get(
                (tenant_id, advice_id)
            )

            if record is None:
                raise ReplacementNotFound(
                    "Replacement advice not found"
                )

            return record.to_dict()

    # --------------------------------------------------------
    # REVIEW ADVICE
    # --------------------------------------------------------

    def review_advice(
        self,
        *,
        tenant_id: str,
        advice_id: str,
        reviewer_id: str,
        approve: bool,
        notes: str,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise ReplacementPermissionDenied(
                "Authorized engineering reviewer required"
            )

        validate_id(reviewer_id, "reviewer_id")

        if not isinstance(approve, bool):
            raise ReplacementValidationError(
                "approve must be boolean"
            )

        validate_text(
            notes,
            "notes",
            allow_empty=True,
        )

        with self._lock:
            record = self._records.get(
                (tenant_id, advice_id)
            )

            if record is None:
                raise ReplacementNotFound(
                    "Replacement advice not found"
                )

            if record.status != AdviceStatus.PENDING_REVIEW:
                raise ReplacementConflict(
                    "Advice has already been reviewed"
                )

            record.status = (
                AdviceStatus.APPROVED
                if approve
                else AdviceStatus.REJECTED
            )

            record.reviewer_id = reviewer_id
            record.review_notes = notes
            record.reviewed_at = utc_now()

            self._log(
                tenant_id=tenant_id,
                advice_id=advice_id,
                action="replacement_advice_reviewed",
                details={
                    "reviewer_id": reviewer_id,
                    "approved": approve,
                },
            )

            return record.to_dict()

    # --------------------------------------------------------
    # LIST ADVICE
    # --------------------------------------------------------

    def list_advice(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_id(tenant_id, "tenant_id")

        if robot_id is not None:
            validate_id(robot_id, "robot_id")

        validate_integer(
            limit,
            "limit",
            minimum=1,
            maximum=100000,
        )

        with self._lock:
            records = [
                record
                for (tenant, _), record
                in self._records.items()
                if (
                    tenant == tenant_id
                    and (
                        robot_id is None
                        or record.robot_id == robot_id
                    )
                )
            ]

            records.sort(
                key=lambda item: item.created_at,
                reverse=True,
            )

            return [
                record.to_dict()
                for record in records[:limit]
            ]

    # --------------------------------------------------------
    # ROBOT SUMMARY
    # --------------------------------------------------------

    def robot_replacement_summary(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        records = self.list_advice(
            tenant_id=tenant_id,
            robot_id=robot_id,
            limit=100000,
        )

        recommendations = Counter(
            item["recommendation"]
            for item in records
        )

        urgencies = Counter(
            item["urgency"]
            for item in records
        )

        return {
            "tenant_id": tenant_id,
            "robot_id": robot_id,
            "total_advice_records": len(records),
            "recommendations": dict(recommendations),
            "urgencies": dict(urgencies),
            "pending_reviews": sum(
                item["status"]
                == AdviceStatus.PENDING_REVIEW.value
                for item in records
            ),
            "advisory_only": True,
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
            records = [
                record.to_dict()
                for (tenant, _), record
                in self._records.items()
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
            "replacement_advice": records,
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
    advisor = RoboticsReplacementAdvisor()

    component = ComponentHealthProfile(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        component_id="joint_motor_001",
        component_name="Robot Joint Motor",
        manufacturer="Demo Robotics",
        model_number="MTR-100",
        original_part_number="DR-MTR-100",
        condition=ComponentCondition.POOR,
        criticality=ComponentCriticality.HIGH,
        fault_severity=FaultSeverity.HIGH,
        health_score=Decimal("28"),
        remaining_useful_life_months=Decimal("5"),
        annual_failure_probability=Decimal("0.35"),
        age_months=Decimal("54"),
        design_life_months=Decimal("60"),
        recurring_failures=3,
        obsolete=False,
        manufacturer_support_available=True,
    )

    candidates = (
        ReplacementPartCandidate(
            candidate_id="candidate_oem_001",
            part_id="motor_oem_001",
            part_name="OEM Joint Motor",
            manufacturer="Demo Robotics",
            part_number="DR-MTR-100-R2",
            part_type=PartType.OEM,
            unit_price=Decimal("14500"),
            currency="INR",
            supplier_id="supplier_001",
            stock_status=StockStatus.IN_STOCK,
            available_quantity=4,
            lead_time_days=3,
            compatibility=CompatibilityStatus.VERIFIED,
            expected_life_months=Decimal("96"),
            expected_reliability=Decimal("0.97"),
            warranty_months=Decimal("24"),
            installation_cost=Decimal("2500"),
            shipping_cost=Decimal("500"),
            downtime_cost=Decimal("3000"),
            annual_maintenance_cost=Decimal("700"),
            catalog_reference="catalog_oem_001",
            pricing_reference="price_oem_001",
        ),
        ReplacementPartCandidate(
            candidate_id="candidate_aftermarket_001",
            part_id="motor_alt_001",
            part_name="Aftermarket Joint Motor",
            manufacturer="Demo Components",
            part_number="ALT-MTR-100",
            part_type=PartType.AFTERMARKET,
            unit_price=Decimal("10500"),
            currency="INR",
            supplier_id="supplier_002",
            stock_status=StockStatus.IN_STOCK,
            available_quantity=10,
            lead_time_days=5,
            compatibility=CompatibilityStatus.UNVERIFIED,
            expected_life_months=Decimal("72"),
            expected_reliability=Decimal("0.91"),
            warranty_months=Decimal("12"),
            installation_cost=Decimal("2500"),
            shipping_cost=Decimal("400"),
            downtime_cost=Decimal("3000"),
            annual_maintenance_cost=Decimal("1100"),
            catalog_reference="catalog_alt_001",
            pricing_reference="price_alt_001",
        ),
    )

    request = ReplacementAdviceRequest(
        component=component,
        candidates=candidates,
        currency="INR",
        budget=Decimal("25000"),
        maximum_lead_time_days=7,
        repair_cost_estimate=Decimal("12000"),
        repair_vs_replace_recommendation="replace",
        required_quantity=1,
        notes="Demo motor replacement assessment",
    )

    result = advisor.advise(request)

    print("\nREPLACEMENT ADVICE")

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    print("\nROBOT REPLACEMENT SUMMARY")

    print(
        json.dumps(
            advisor.robot_replacement_summary(
                tenant_id="tenant_alpha",
                robot_id="robot_001",
            ),
            indent=2,
        )
    )

    report_path = advisor.export_json(
        "data/robotics/replacement_advice_report.json",
        tenant_id="tenant_alpha",
    )

    print("\nExported:", report_path)


if __name__ == "__main__":
    main()
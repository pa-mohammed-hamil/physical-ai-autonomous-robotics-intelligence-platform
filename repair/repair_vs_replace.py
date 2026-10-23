"""Repair Vs Replace module.

TODO: implement and test; no real robot control is enabled.
"""
# backend/app/robotics/control/repair_vs_replace.py
"""
Physical AI & Autonomous Robotics Intelligence Platform
Repair vs Replace Decision Intelligence Engine

Python 3.11+
Dependencies: Standard library only

Features:
    - Robot component repair vs replacement analysis
    - Repair and replacement cost comparison
    - Parts, labor, downtime, logistics and tax costs
    - Future maintenance and operating cost analysis
    - Expected failure-risk cost
    - Remaining useful life (RUL) comparison
    - Reliability and safety evaluation
    - Warranty and component obsolescence checks
    - Budget and downtime constraints
    - Economic break-even analysis
    - Multi-year cost projection
    - Discounted lifecycle cost (NPV)
    - Cost savings and ROI calculations
    - Weighted decision scoring
    - Safety-critical human approval
    - Decision review workflow
    - Multi-tenant isolation
    - Thread-safe in-memory storage
    - Audit logging
    - JSON report export

Integrations:
    - damage_analysis.py
    - damage_severity.py
    - issue_diagnosis.py
    - cost_estimation.py
    - labor_cost.py
    - downtime_cost.py
    - parts_catalog.py
    - parts_pricing.py
    - maintenance_service.py
    - repair_service.py

Safety:
    This engine provides advisory decisions only.

    It does not authorize installation, repair,
    replacement, robot motion, or physical actuation.

    Safety-critical decisions require qualified
    human review and manufacturer verification.

    Financial inputs must be supplied by trusted
    pricing, maintenance, and engineering sources.
"""

from __future__ import annotations

import json
import threading
import uuid

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import (
    Decimal,
    InvalidOperation,
    ROUND_HALF_UP,
)
from enum import Enum
from pathlib import Path
from typing import Any


# ============================================================
# EXCEPTIONS
# ============================================================

class RepairReplaceError(Exception):
    pass


class DecisionValidationError(RepairReplaceError):
    pass


class DecisionNotFound(RepairReplaceError):
    pass


class DecisionConflict(RepairReplaceError):
    pass


class DecisionPermissionDenied(RepairReplaceError):
    pass


# ============================================================
# ENUMS
# ============================================================

class DecisionType(str, Enum):
    REPAIR = "repair"
    REPLACE = "replace"
    FURTHER_INSPECTION = "further_inspection"
    INSUFFICIENT_DATA = "insufficient_data"


class ComponentCriticality(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SAFETY_CRITICAL = "safety_critical"


class ComponentCondition(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    FAILED = "failed"
    UNKNOWN = "unknown"


class DecisionStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class DecisionReason(str, Enum):
    LOWER_LIFECYCLE_COST = "lower_lifecycle_cost"
    LOWER_INITIAL_COST = "lower_initial_cost"
    HIGHER_RELIABILITY = "higher_reliability"
    LONGER_USEFUL_LIFE = "longer_useful_life"
    LOWER_DOWNTIME = "lower_downtime"
    SAFETY_CONCERN = "safety_concern"
    COMPONENT_OBSOLETE = "component_obsolete"
    BUDGET_CONSTRAINT = "budget_constraint"
    ENGINEERING_REVIEW = "engineering_review"


# ============================================================
# UTILITIES
# ============================================================

CENT = Decimal("0.01")
PERCENT = Decimal("100")


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
            character.isalnum()
            or character in "._-"
            for character in value
        )
    ):
        raise DecisionValidationError(
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
        raise DecisionValidationError(
            f"{name} must be text"
        )

    value = value.strip()

    if (
        len(value) > maximum
        or (not allow_empty and not value)
    ):
        raise DecisionValidationError(
            f"Invalid {name}"
        )

    return value


def number(
    value: Any,
    name: str,
    *,
    minimum: Decimal = Decimal("0"),
    maximum: Decimal = Decimal("1000000000000"),
) -> Decimal:
    if isinstance(value, bool):
        raise DecisionValidationError(
            f"Invalid {name}"
        )

    try:
        result = Decimal(str(value))
    except (
        InvalidOperation,
        TypeError,
        ValueError,
    ) as exc:
        raise DecisionValidationError(
            f"Invalid {name}"
        ) from exc

    if (
        not result.is_finite()
        or result < minimum
        or result > maximum
    ):
        raise DecisionValidationError(
            f"Invalid {name}"
        )

    return result


def amount(value: Any, name: str) -> Decimal:
    return number(value, name).quantize(
        CENT,
        rounding=ROUND_HALF_UP,
    )


def money(value: Decimal) -> str:
    return str(
        value.quantize(
            CENT,
            rounding=ROUND_HALF_UP,
        )
    )


def validate_probability(
    value: Any,
    name: str,
) -> Decimal:
    return number(
        value,
        name,
        minimum=Decimal("0"),
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
        raise DecisionValidationError(
            "Currency must be a three-letter uppercase code"
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
        raise DecisionValidationError(
            f"Invalid {name}"
        )
    return value


def decimal_to_float(value: Decimal) -> float:
    return float(
        value.quantize(
            Decimal("0.0001"),
            rounding=ROUND_HALF_UP,
        )
    )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RepairReplaceConfig:
    default_currency: str = "INR"

    maximum_decisions: int = 100000
    maximum_audit_events: int = 20000

    default_analysis_years: int = 5

    default_discount_rate: Decimal = Decimal("0.08")

    minimum_acceptable_reliability: Decimal = (
        Decimal("0.80")
    )

    maximum_acceptable_failure_probability: Decimal = (
        Decimal("0.20")
    )

    repair_cost_replacement_threshold: Decimal = (
        Decimal("0.70")
    )

    simulation_only: bool = True

    def __post_init__(self) -> None:
        validate_currency(
            self.default_currency
        )

        validate_integer(
            self.maximum_decisions,
            "maximum_decisions",
            minimum=1,
        )

        validate_integer(
            self.maximum_audit_events,
            "maximum_audit_events",
            minimum=1,
        )

        validate_integer(
            self.default_analysis_years,
            "default_analysis_years",
            minimum=1,
            maximum=30,
        )

        number(
            self.default_discount_rate,
            "default_discount_rate",
            maximum=Decimal("1"),
        )

        validate_probability(
            self.minimum_acceptable_reliability,
            "minimum_acceptable_reliability",
        )

        validate_probability(
            self.maximum_acceptable_failure_probability,
            "maximum_acceptable_failure_probability",
        )

        validate_probability(
            self.repair_cost_replacement_threshold,
            "repair_cost_replacement_threshold",
        )

        if self.simulation_only is not True:
            raise DecisionValidationError(
                "Only advisory mode is supported"
            )


# ============================================================
# COMPONENT PROFILE
# ============================================================

@dataclass(frozen=True)
class RobotComponentProfile:
    tenant_id: str
    robot_id: str
    component_id: str
    component_name: str

    manufacturer: str = ""
    model_number: str = ""
    part_number: str = ""

    condition: ComponentCondition = (
        ComponentCondition.UNKNOWN
    )

    criticality: ComponentCriticality = (
        ComponentCriticality.MODERATE
    )

    age_years: Decimal = Decimal("0")

    expected_design_life_years: Decimal = (
        Decimal("5")
    )

    current_reliability: Decimal = (
        Decimal("0.85")
    )

    obsolete: bool = False
    repairable: bool = True

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
            "part_number",
        ):
            validate_text(
                getattr(self, name),
                name,
                allow_empty=True,
            )

        if not isinstance(
            self.condition,
            ComponentCondition,
        ):
            raise DecisionValidationError(
                "Invalid component condition"
            )

        if not isinstance(
            self.criticality,
            ComponentCriticality,
        ):
            raise DecisionValidationError(
                "Invalid component criticality"
            )

        number(
            self.age_years,
            "age_years",
            maximum=Decimal("100"),
        )

        number(
            self.expected_design_life_years,
            "expected_design_life_years",
            minimum=Decimal("0.01"),
            maximum=Decimal("100"),
        )

        validate_probability(
            self.current_reliability,
            "current_reliability",
        )

        if not isinstance(self.obsolete, bool):
            raise DecisionValidationError(
                "obsolete must be boolean"
            )

        if not isinstance(self.repairable, bool):
            raise DecisionValidationError(
                "repairable must be boolean"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "component_id": self.component_id,
            "component_name": self.component_name,
            "manufacturer": self.manufacturer,
            "model_number": self.model_number,
            "part_number": self.part_number,
            "condition": self.condition.value,
            "criticality": self.criticality.value,
            "age_years": str(self.age_years),
            "expected_design_life_years": str(
                self.expected_design_life_years
            ),
            "current_reliability": str(
                self.current_reliability
            ),
            "obsolete": self.obsolete,
            "repairable": self.repairable,
        }


# ============================================================
# COST INPUT
# ============================================================

@dataclass(frozen=True)
class RepairReplaceCostInput:
    parts_cost: Decimal = Decimal("0")
    labor_cost: Decimal = Decimal("0")
    downtime_cost: Decimal = Decimal("0")
    shipping_cost: Decimal = Decimal("0")
    installation_cost: Decimal = Decimal("0")
    inspection_cost: Decimal = Decimal("0")
    disposal_cost: Decimal = Decimal("0")
    other_cost: Decimal = Decimal("0")

    tax_percent: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        for name in (
            "parts_cost",
            "labor_cost",
            "downtime_cost",
            "shipping_cost",
            "installation_cost",
            "inspection_cost",
            "disposal_cost",
            "other_cost",
        ):
            object.__setattr__(
                self,
                name,
                amount(
                    getattr(self, name),
                    name,
                ),
            )

        number(
            self.tax_percent,
            "tax_percent",
            maximum=Decimal("100"),
        )

    def calculate(self) -> dict[str, Decimal]:
        # Taxes apply only to supplied parts,
        # labor, shipping and installation.
        # Downtime is an economic loss and is
        # excluded from the taxable base.
        taxable = (
            self.parts_cost
            + self.labor_cost
            + self.shipping_cost
            + self.installation_cost
        )

        tax = (
            taxable
            * self.tax_percent
            / PERCENT
        ).quantize(
            CENT,
            rounding=ROUND_HALF_UP,
        )

        subtotal = sum(
            (
                self.parts_cost,
                self.labor_cost,
                self.downtime_cost,
                self.shipping_cost,
                self.installation_cost,
                self.inspection_cost,
                self.disposal_cost,
                self.other_cost,
            ),
            Decimal("0"),
        )

        return {
            "parts_cost": self.parts_cost,
            "labor_cost": self.labor_cost,
            "downtime_cost": self.downtime_cost,
            "shipping_cost": self.shipping_cost,
            "installation_cost": self.installation_cost,
            "inspection_cost": self.inspection_cost,
            "disposal_cost": self.disposal_cost,
            "other_cost": self.other_cost,
            "tax_amount": tax,
            "initial_total": subtotal + tax,
        }

    def to_dict(self) -> dict[str, Any]:
        values = self.calculate()

        return {
            **{
                key: money(value)
                for key, value in values.items()
            },
            "tax_percent": str(self.tax_percent),
        }


# ============================================================
# LIFECYCLE INPUT
# ============================================================

@dataclass(frozen=True)
class LifecycleEstimate:
    remaining_useful_life_years: Decimal

    annual_maintenance_cost: Decimal = (
        Decimal("0")
    )

    annual_operating_cost: Decimal = (
        Decimal("0")
    )

    annual_failure_probability: Decimal = (
        Decimal("0")
    )

    failure_event_cost: Decimal = (
        Decimal("0")
    )

    expected_reliability: Decimal = (
        Decimal("0.90")
    )

    warranty_years: Decimal = Decimal("0")

    downtime_hours: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        number(
            self.remaining_useful_life_years,
            "remaining_useful_life_years",
            maximum=Decimal("100"),
        )

        for name in (
            "annual_maintenance_cost",
            "annual_operating_cost",
            "failure_event_cost",
        ):
            object.__setattr__(
                self,
                name,
                amount(
                    getattr(self, name),
                    name,
                ),
            )

        validate_probability(
            self.annual_failure_probability,
            "annual_failure_probability",
        )

        validate_probability(
            self.expected_reliability,
            "expected_reliability",
        )

        number(
            self.warranty_years,
            "warranty_years",
            maximum=Decimal("100"),
        )

        number(
            self.downtime_hours,
            "downtime_hours",
            maximum=Decimal("100000"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "remaining_useful_life_years": str(
                self.remaining_useful_life_years
            ),
            "annual_maintenance_cost": money(
                self.annual_maintenance_cost
            ),
            "annual_operating_cost": money(
                self.annual_operating_cost
            ),
            "annual_failure_probability": str(
                self.annual_failure_probability
            ),
            "failure_event_cost": money(
                self.failure_event_cost
            ),
            "expected_reliability": str(
                self.expected_reliability
            ),
            "warranty_years": str(
                self.warranty_years
            ),
            "downtime_hours": str(
                self.downtime_hours
            ),
        }


# ============================================================
# ANALYSIS REQUEST
# ============================================================

@dataclass(frozen=True)
class RepairReplaceRequest:
    component: RobotComponentProfile

    repair_cost: RepairReplaceCostInput
    replacement_cost: RepairReplaceCostInput

    repair_lifecycle: LifecycleEstimate
    replacement_lifecycle: LifecycleEstimate

    currency: str = "INR"

    analysis_years: int = 5

    discount_rate: Decimal = Decimal("0.08")

    budget: Decimal | None = None

    maximum_downtime_hours: Decimal | None = None

    replacement_compatible: bool | None = None

    repair_safety_verified: bool = False
    replacement_safety_verified: bool = False

    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(
            self.component,
            RobotComponentProfile,
        ):
            raise DecisionValidationError(
                "Invalid component"
            )

        for name in (
            "repair_cost",
            "replacement_cost",
        ):
            if not isinstance(
                getattr(self, name),
                RepairReplaceCostInput,
            ):
                raise DecisionValidationError(
                    f"Invalid {name}"
                )

        for name in (
            "repair_lifecycle",
            "replacement_lifecycle",
        ):
            if not isinstance(
                getattr(self, name),
                LifecycleEstimate,
            ):
                raise DecisionValidationError(
                    f"Invalid {name}"
                )

        validate_currency(
            self.currency
        )

        validate_integer(
            self.analysis_years,
            "analysis_years",
            minimum=1,
            maximum=30,
        )

        number(
            self.discount_rate,
            "discount_rate",
            maximum=Decimal("1"),
        )

        if self.budget is not None:
            object.__setattr__(
                self,
                "budget",
                amount(self.budget, "budget"),
            )

        if self.maximum_downtime_hours is not None:
            number(
                self.maximum_downtime_hours,
                "maximum_downtime_hours",
                maximum=Decimal("100000"),
            )

        if (
            self.replacement_compatible is not None
            and not isinstance(
                self.replacement_compatible,
                bool,
            )
        ):
            raise DecisionValidationError(
                "replacement_compatible must be bool or None"
            )

        for name in (
            "repair_safety_verified",
            "replacement_safety_verified",
        ):
            if not isinstance(
                getattr(self, name),
                bool,
            ):
                raise DecisionValidationError(
                    f"{name} must be boolean"
                )

        validate_text(
            self.notes,
            "notes",
            allow_empty=True,
        )


# ============================================================
# DECISION RECORD
# ============================================================

@dataclass
class RepairReplaceDecision:
    decision_id: str
    tenant_id: str
    robot_id: str
    component_id: str

    recommendation: DecisionType

    repair_initial_cost: Decimal
    replacement_initial_cost: Decimal

    repair_lifecycle_npv: Decimal
    replacement_lifecycle_npv: Decimal

    potential_savings: Decimal

    repair_score: Decimal
    replacement_score: Decimal

    repair_feasible: bool
    replacement_feasible: bool

    risk_level: RiskLevel

    reasons: list[str]
    warnings: list[str]

    repair_projection: list[dict[str, Any]]
    replacement_projection: list[dict[str, Any]]

    analysis_years: int
    currency: str

    human_review_required: bool = True

    status: DecisionStatus = (
        DecisionStatus.PENDING_REVIEW
    )

    reviewed_by: str | None = None
    review_notes: str = ""

    created_at: datetime = field(
        default_factory=utc_now
    )

    reviewed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "component_id": self.component_id,
            "recommendation": self.recommendation.value,
            "repair_initial_cost": money(
                self.repair_initial_cost
            ),
            "replacement_initial_cost": money(
                self.replacement_initial_cost
            ),
            "repair_lifecycle_npv": money(
                self.repair_lifecycle_npv
            ),
            "replacement_lifecycle_npv": money(
                self.replacement_lifecycle_npv
            ),
            "potential_savings": money(
                self.potential_savings
            ),
            "repair_score": str(
                self.repair_score
            ),
            "replacement_score": str(
                self.replacement_score
            ),
            "repair_feasible": self.repair_feasible,
            "replacement_feasible": (
                self.replacement_feasible
            ),
            "risk_level": self.risk_level.value,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "repair_projection": (
                self.repair_projection
            ),
            "replacement_projection": (
                self.replacement_projection
            ),
            "analysis_years": self.analysis_years,
            "currency": self.currency,
            "human_review_required": (
                self.human_review_required
            ),
            "status": self.status.value,
            "reviewed_by": self.reviewed_by,
            "review_notes": self.review_notes,
            "created_at": (
                self.created_at.isoformat()
            ),
            "reviewed_at": (
                self.reviewed_at.isoformat()
                if self.reviewed_at
                else None
            ),
            "advisory_only": True,
        }


# ============================================================
# ENGINE
# ============================================================

class RoboticsRepairVsReplaceEngine:

    def __init__(
        self,
        config: RepairReplaceConfig | None = None,
    ) -> None:
        self.config = (
            config or RepairReplaceConfig()
        )

        self._decisions: dict[
            tuple[str, str],
            RepairReplaceDecision,
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
        tenant_id: str,
        decision_id: str,
        action: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._audit.append(
            {
                "event_id": new_id(),
                "timestamp": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "decision_id": decision_id,
                "action": action,
                "details": details or {},
            }
        )

    # --------------------------------------------------------
    # ANNUAL COST
    # --------------------------------------------------------

    @staticmethod
    def annual_expected_cost(
        lifecycle: LifecycleEstimate,
    ) -> Decimal:
        failure_risk_cost = (
            lifecycle.annual_failure_probability
            * lifecycle.failure_event_cost
        )

        return (
            lifecycle.annual_maintenance_cost
            + lifecycle.annual_operating_cost
            + failure_risk_cost
        )

    # --------------------------------------------------------
    # LIFECYCLE PROJECTION
    # --------------------------------------------------------

    @staticmethod
    def lifecycle_projection(
        *,
        initial_cost: Decimal,
        lifecycle: LifecycleEstimate,
        years: int,
        discount_rate: Decimal,
    ) -> tuple[
        Decimal,
        list[dict[str, Any]],
    ]:
        """
        Discounted cash-flow approximation.

        Annual costs are assumed to occur at
        year-end and remain constant.

        Failure probability is treated as the
        expected annual event rate for costing.

        If useful life is shorter than the analysis
        horizon, the estimate is not extrapolated:
        the result is flagged by the caller as
        requiring a renewal/replacement model.
        """
        annual = (
            RoboticsRepairVsReplaceEngine
            .annual_expected_cost(lifecycle)
        )

        total_npv = initial_cost

        projection = [
            {
                "year": 0,
                "annual_cost": money(
                    initial_cost
                ),
                "discounted_cost": money(
                    initial_cost
                ),
                "cumulative_npv": money(
                    initial_cost
                ),
            }
        ]

        for year in range(1, years + 1):
            discount_factor = (
                Decimal("1") + discount_rate
            ) ** year

            discounted = (
                annual / discount_factor
            )

            total_npv += discounted

            projection.append(
                {
                    "year": year,
                    "annual_cost": money(
                        annual
                    ),
                    "discounted_cost": money(
                        discounted
                    ),
                    "cumulative_npv": money(
                        total_npv
                    ),
                }
            )

        return (
            total_npv.quantize(
                CENT,
                rounding=ROUND_HALF_UP,
            ),
            projection,
        )

    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    @staticmethod
    def classify_risk(
        component: RobotComponentProfile,
        repair_lifecycle: LifecycleEstimate,
    ) -> RiskLevel:
        if (
            component.criticality
            == ComponentCriticality.SAFETY_CRITICAL
            and (
                repair_lifecycle.expected_reliability
                < Decimal("0.90")
                or repair_lifecycle.annual_failure_probability
                > Decimal("0.10")
            )
        ):
            return RiskLevel.CRITICAL

        if (
            repair_lifecycle.annual_failure_probability
            >= Decimal("0.30")
            or repair_lifecycle.expected_reliability
            < Decimal("0.70")
        ):
            return RiskLevel.HIGH

        if (
            repair_lifecycle.annual_failure_probability
            >= Decimal("0.15")
            or repair_lifecycle.expected_reliability
            < Decimal("0.85")
        ):
            return RiskLevel.MODERATE

        return RiskLevel.LOW

    # --------------------------------------------------------
    # FEASIBILITY
    # --------------------------------------------------------

    def assess_feasibility(
        self,
        request: RepairReplaceRequest,
        *,
        repair_initial: Decimal,
        replacement_initial: Decimal,
    ) -> tuple[
        bool,
        bool,
        list[str],
    ]:
        warnings = []

        repair_feasible = (
            request.component.repairable
        )

        replacement_feasible = True

        if not request.component.repairable:
            warnings.append(
                "Component is marked non-repairable."
            )

        if request.replacement_compatible is False:
            replacement_feasible = False

            warnings.append(
                "Replacement is marked incompatible."
            )

        if request.replacement_compatible is None:
            replacement_feasible = False

            warnings.append(
                "Replacement compatibility has not "
                "been verified."
            )

        if request.budget is not None:
            if repair_initial > request.budget:
                repair_feasible = False

                warnings.append(
                    "Repair exceeds the supplied budget."
                )

            if replacement_initial > request.budget:
                replacement_feasible = False

                warnings.append(
                    "Replacement exceeds the supplied budget."
                )

        if request.maximum_downtime_hours is not None:
            if (
                request.repair_lifecycle.downtime_hours
                > request.maximum_downtime_hours
            ):
                repair_feasible = False

                warnings.append(
                    "Repair exceeds maximum downtime."
                )

            if (
                request.replacement_lifecycle.downtime_hours
                > request.maximum_downtime_hours
            ):
                replacement_feasible = False

                warnings.append(
                    "Replacement exceeds maximum downtime."
                )

        if (
            request.component.criticality
            == ComponentCriticality.SAFETY_CRITICAL
        ):
            if not request.repair_safety_verified:
                repair_feasible = False

                warnings.append(
                    "Repair safety verification is missing."
                )

            if not request.replacement_safety_verified:
                replacement_feasible = False

                warnings.append(
                    "Replacement safety verification is missing."
                )

        return (
            repair_feasible,
            replacement_feasible,
            warnings,
        )

    # --------------------------------------------------------
    # DECISION SCORE
    # --------------------------------------------------------

    @staticmethod
    def score_option(
        *,
        lifecycle_cost: Decimal,
        lowest_lifecycle_cost: Decimal,
        reliability: Decimal,
        remaining_life: Decimal,
        maximum_remaining_life: Decimal,
        downtime_hours: Decimal,
        maximum_downtime_hours: Decimal,
    ) -> Decimal:
        """
        Heuristic comparison score, not a
        calibrated probability or safety rating.

        Weights:
            Lifecycle cost: 40%
            Reliability:    30%
            Useful life:    20%
            Downtime:       10%
        """
        if lifecycle_cost <= 0:
            cost_score = Decimal("1")
        else:
            cost_score = min(
                Decimal("1"),
                lowest_lifecycle_cost
                / lifecycle_cost,
            )

        reliability_score = reliability

        life_score = (
            remaining_life
            / maximum_remaining_life
            if maximum_remaining_life > 0
            else Decimal("0")
        )

        downtime_score = (
            Decimal("1")
            - downtime_hours
            / maximum_downtime_hours
            if maximum_downtime_hours > 0
            else Decimal("1")
        )

        downtime_score = max(
            Decimal("0"),
            min(
                Decimal("1"),
                downtime_score,
            ),
        )

        score = (
            Decimal("0.40") * cost_score
            + Decimal("0.30") * reliability_score
            + Decimal("0.20") * life_score
            + Decimal("0.10") * downtime_score
        )

        return score.quantize(
            Decimal("0.0001"),
            rounding=ROUND_HALF_UP,
        )

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    def analyze(
        self,
        request: RepairReplaceRequest,
    ) -> dict[str, Any]:
        if not isinstance(
            request,
            RepairReplaceRequest,
        ):
            raise DecisionValidationError(
                "Invalid analysis request"
            )

        component = request.component

        repair_costs = (
            request.repair_cost.calculate()
        )

        replacement_costs = (
            request.replacement_cost.calculate()
        )

        repair_initial = (
            repair_costs["initial_total"]
        )

        replacement_initial = (
            replacement_costs["initial_total"]
        )

        repair_npv, repair_projection = (
            self.lifecycle_projection(
                initial_cost=repair_initial,
                lifecycle=request.repair_lifecycle,
                years=request.analysis_years,
                discount_rate=request.discount_rate,
            )
        )

        replacement_npv, replacement_projection = (
            self.lifecycle_projection(
                initial_cost=replacement_initial,
                lifecycle=request.replacement_lifecycle,
                years=request.analysis_years,
                discount_rate=request.discount_rate,
            )
        )

        (
            repair_feasible,
            replacement_feasible,
            warnings,
        ) = self.assess_feasibility(
            request,
            repair_initial=repair_initial,
            replacement_initial=replacement_initial,
        )

        # Do not compare incomplete lifecycle models
        # as though they cover the full horizon.
        repair_life_covers_horizon = (
            request.repair_lifecycle
            .remaining_useful_life_years
            >= request.analysis_years
        )

        replacement_life_covers_horizon = (
            request.replacement_lifecycle
            .remaining_useful_life_years
            >= request.analysis_years
        )

        if not repair_life_covers_horizon:
            warnings.append(
                "Repaired component useful life is "
                "shorter than the analysis horizon; "
                "additional renewal costs are unmodeled."
            )

        if not replacement_life_covers_horizon:
            warnings.append(
                "Replacement component useful life is "
                "shorter than the analysis horizon; "
                "additional renewal costs are unmodeled."
            )

        if (
            request.repair_lifecycle.expected_reliability
            < self.config.minimum_acceptable_reliability
        ):
            repair_feasible = False

            warnings.append(
                "Expected repair reliability is below "
                "the configured acceptance threshold."
            )

        if (
            request.replacement_lifecycle.expected_reliability
            < self.config.minimum_acceptable_reliability
        ):
            replacement_feasible = False

            warnings.append(
                "Expected replacement reliability is below "
                "the configured acceptance threshold."
            )

        if (
            request.repair_lifecycle
            .annual_failure_probability
            > self.config
            .maximum_acceptable_failure_probability
        ):
            repair_feasible = False

            warnings.append(
                "Repair failure probability exceeds "
                "the configured acceptance threshold."
            )

        if (
            request.replacement_lifecycle
            .annual_failure_probability
            > self.config
            .maximum_acceptable_failure_probability
        ):
            replacement_feasible = False

            warnings.append(
                "Replacement failure probability exceeds "
                "the configured acceptance threshold."
            )

        lowest_cost = min(
            repair_npv,
            replacement_npv,
        )

        max_life = max(
            request.repair_lifecycle
            .remaining_useful_life_years,
            request.replacement_lifecycle
            .remaining_useful_life_years,
        )

        max_downtime = max(
            request.repair_lifecycle.downtime_hours,
            request.replacement_lifecycle.downtime_hours,
        )

        repair_score = self.score_option(
            lifecycle_cost=repair_npv,
            lowest_lifecycle_cost=lowest_cost,
            reliability=(
                request.repair_lifecycle
                .expected_reliability
            ),
            remaining_life=(
                request.repair_lifecycle
                .remaining_useful_life_years
            ),
            maximum_remaining_life=max_life,
            downtime_hours=(
                request.repair_lifecycle.downtime_hours
            ),
            maximum_downtime_hours=max_downtime,
        )

        replacement_score = self.score_option(
            lifecycle_cost=replacement_npv,
            lowest_lifecycle_cost=lowest_cost,
            reliability=(
                request.replacement_lifecycle
                .expected_reliability
            ),
            remaining_life=(
                request.replacement_lifecycle
                .remaining_useful_life_years
            ),
            maximum_remaining_life=max_life,
            downtime_hours=(
                request.replacement_lifecycle
                .downtime_hours
            ),
            maximum_downtime_hours=max_downtime,
        )

        reasons: list[str] = []

        complete_horizon = (
            repair_life_covers_horizon
            and replacement_life_covers_horizon
        )

        if not complete_horizon:
            recommendation = (
                DecisionType.INSUFFICIENT_DATA
            )

            reasons.append(
                "The lifecycle model does not include "
                "all required renewals within the "
                "analysis horizon."
            )

        elif repair_feasible and not replacement_feasible:
            recommendation = DecisionType.REPAIR

            reasons.append(
                "Repair meets the configured constraints "
                "while replacement does not."
            )

        elif replacement_feasible and not repair_feasible:
            recommendation = DecisionType.REPLACE

            reasons.append(
                "Replacement meets the configured "
                "constraints while repair does not."
            )

        elif not repair_feasible and not replacement_feasible:
            recommendation = (
                DecisionType.FURTHER_INSPECTION
            )

            reasons.append(
                "Neither option currently satisfies "
                "the required constraints."
            )

        elif repair_score > replacement_score:
            recommendation = DecisionType.REPAIR

            reasons.append(
                "Repair has the higher weighted "
                "economic and reliability score."
            )

        elif replacement_score > repair_score:
            recommendation = DecisionType.REPLACE

            reasons.append(
                "Replacement has the higher weighted "
                "economic and reliability score."
            )

        else:
            recommendation = (
                DecisionType.FURTHER_INSPECTION
            )

            reasons.append(
                "Repair and replacement scores are equal."
            )

        if repair_npv < replacement_npv:
            reasons.append(
                "Repair has lower modeled lifecycle cost."
            )

        elif replacement_npv < repair_npv:
            reasons.append(
                "Replacement has lower modeled lifecycle cost."
            )

        if component.obsolete:
            warnings.append(
                "Component is obsolete; verify parts "
                "availability and manufacturer support."
            )

        if (
            component.condition
            == ComponentCondition.UNKNOWN
        ):
            warnings.append(
                "Component condition is unknown."
            )

        if (
            component.criticality
            == ComponentCriticality.SAFETY_CRITICAL
        ):
            warnings.append(
                "Safety-critical component: qualified "
                "engineering approval is mandatory."
            )

        if (
            replacement_initial > 0
            and repair_initial
            / replacement_initial
            >= self.config
            .repair_cost_replacement_threshold
        ):
            reasons.append(
                "Repair initial cost exceeds the configured "
                "fraction of replacement initial cost."
            )

        risk = self.classify_risk(
            component,
            request.repair_lifecycle,
        )

        if recommendation == DecisionType.REPAIR:
            potential_savings = max(
                Decimal("0"),
                replacement_npv - repair_npv,
            )

        elif recommendation == DecisionType.REPLACE:
            potential_savings = max(
                Decimal("0"),
                repair_npv - replacement_npv,
            )

        else:
            potential_savings = Decimal("0")

        decision = RepairReplaceDecision(
            decision_id=new_id(),
            tenant_id=component.tenant_id,
            robot_id=component.robot_id,
            component_id=component.component_id,
            recommendation=recommendation,
            repair_initial_cost=repair_initial,
            replacement_initial_cost=(
                replacement_initial
            ),
            repair_lifecycle_npv=repair_npv,
            replacement_lifecycle_npv=(
                replacement_npv
            ),
            potential_savings=potential_savings,
            repair_score=repair_score,
            replacement_score=replacement_score,
            repair_feasible=repair_feasible,
            replacement_feasible=(
                replacement_feasible
            ),
            risk_level=risk,
            reasons=reasons,
            warnings=warnings,
            repair_projection=repair_projection,
            replacement_projection=(
                replacement_projection
            ),
            analysis_years=request.analysis_years,
            currency=request.currency,
        )

        with self._lock:
            if (
                len(self._decisions)
                >= self.config.maximum_decisions
            ):
                raise RepairReplaceError(
                    "Decision storage capacity reached"
                )

            key = (
                component.tenant_id,
                decision.decision_id,
            )

            self._decisions[key] = decision

            self._log(
                component.tenant_id,
                decision.decision_id,
                "decision_created",
                {
                    "recommendation": (
                        recommendation.value
                    ),
                    "component_id": (
                        component.component_id
                    ),
                },
            )

        return decision.to_dict()

    # --------------------------------------------------------
    # GET DECISION
    # --------------------------------------------------------

    def get_decision(
        self,
        *,
        tenant_id: str,
        decision_id: str,
    ) -> dict[str, Any]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        validate_id(
            decision_id,
            "decision_id",
        )

        with self._lock:
            decision = self._decisions.get(
                (tenant_id, decision_id)
            )

            if decision is None:
                raise DecisionNotFound(
                    "Decision not found"
                )

            return decision.to_dict()

    # --------------------------------------------------------
    # REVIEW DECISION
    # --------------------------------------------------------

    def review_decision(
        self,
        *,
        tenant_id: str,
        decision_id: str,
        reviewer_id: str,
        approve: bool,
        notes: str,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise DecisionPermissionDenied(
                "Review authorization required"
            )

        validate_id(
            tenant_id,
            "tenant_id",
        )

        validate_id(
            decision_id,
            "decision_id",
        )

        validate_id(
            reviewer_id,
            "reviewer_id",
        )

        validate_text(
            notes,
            "notes",
            allow_empty=True,
        )

        if not isinstance(approve, bool):
            raise DecisionValidationError(
                "approve must be boolean"
            )

        with self._lock:
            decision = self._decisions.get(
                (tenant_id, decision_id)
            )

            if decision is None:
                raise DecisionNotFound(
                    "Decision not found"
                )

            if (
                decision.status
                != DecisionStatus.PENDING_REVIEW
            ):
                raise DecisionConflict(
                    "Decision already reviewed"
                )

            if (
                approve
                and decision.recommendation
                not in (
                    DecisionType.REPAIR,
                    DecisionType.REPLACE,
                )
            ):
                raise DecisionConflict(
                    "Cannot approve an unresolved decision"
                )

            decision.status = (
                DecisionStatus.APPROVED
                if approve
                else DecisionStatus.REJECTED
            )

            decision.reviewed_by = reviewer_id
            decision.review_notes = notes
            decision.reviewed_at = utc_now()

            self._log(
                tenant_id,
                decision_id,
                "decision_reviewed",
                {
                    "reviewer_id": reviewer_id,
                    "approved": approve,
                },
            )

            return decision.to_dict()

    # --------------------------------------------------------
    # LIST DECISIONS
    # --------------------------------------------------------

    def list_decisions(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        if robot_id is not None:
            validate_id(
                robot_id,
                "robot_id",
            )

        validate_integer(
            limit,
            "limit",
            minimum=1,
            maximum=10000,
        )

        with self._lock:
            decisions = [
                decision
                for (tenant, _), decision
                in self._decisions.items()
                if (
                    tenant == tenant_id
                    and (
                        robot_id is None
                        or decision.robot_id == robot_id
                    )
                )
            ]

        decisions.sort(
            key=lambda item: item.created_at,
            reverse=True,
        )

        return [
            decision.to_dict()
            for decision in decisions[:limit]
        ]

    # --------------------------------------------------------
    # ROBOT DECISION SUMMARY
    # --------------------------------------------------------

    def robot_decision_summary(
        self,
        *,
        tenant_id: str,
        robot_id: str,
    ) -> dict[str, Any]:
        decisions = self.list_decisions(
            tenant_id=tenant_id,
            robot_id=robot_id,
            limit=10000,
        )

        counts = {
            decision.value: 0
            for decision in DecisionType
        }

        pending = 0

        for decision in decisions:
            counts[
                decision["recommendation"]
            ] += 1

            if (
                decision["status"]
                == DecisionStatus.PENDING_REVIEW.value
            ):
                pending += 1

        return {
            "tenant_id": tenant_id,
            "robot_id": robot_id,
            "total_decisions": len(decisions),
            "recommendations": counts,
            "pending_reviews": pending,
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
        validate_id(
            tenant_id,
            "tenant_id",
        )

        validate_integer(
            limit,
            "limit",
            minimum=1,
            maximum=10000,
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
        validate_id(
            tenant_id,
            "tenant_id",
        )

        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            decisions = [
                decision.to_dict()
                for (tenant, _), decision
                in self._decisions.items()
                if tenant == tenant_id
            ]

            audit_events = [
                dict(event)
                for event in self._audit
                if event["tenant_id"] == tenant_id
            ]

        payload = {
            "generated_at": utc_now().isoformat(),
            "tenant_id": tenant_id,
            "decisions": decisions,
            "audit_events": audit_events,
            "simulation_only": True,
            "automatic_actuation": False,
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
    engine = RoboticsRepairVsReplaceEngine()

    component = RobotComponentProfile(
        tenant_id="tenant_alpha",
        robot_id="robot_001",
        component_id="joint_motor_001",
        component_name="Industrial Robot Joint Motor",
        manufacturer="Demo Robotics",
        model_number="MTR-100",
        part_number="DR-MTR-100",
        condition=ComponentCondition.POOR,
        criticality=ComponentCriticality.HIGH,
        age_years=Decimal("4"),
        expected_design_life_years=Decimal("6"),
        current_reliability=Decimal("0.72"),
        obsolete=False,
        repairable=True,
    )

    repair_cost = RepairReplaceCostInput(
        parts_cost=Decimal("3500"),
        labor_cost=Decimal("2500"),
        downtime_cost=Decimal("4000"),
        inspection_cost=Decimal("500"),
        shipping_cost=Decimal("200"),
        tax_percent=Decimal("18"),
    )

    replacement_cost = RepairReplaceCostInput(
        parts_cost=Decimal("12500"),
        labor_cost=Decimal("3000"),
        downtime_cost=Decimal("2500"),
        shipping_cost=Decimal("500"),
        installation_cost=Decimal("1000"),
        disposal_cost=Decimal("300"),
        tax_percent=Decimal("18"),
    )

    repair_lifecycle = LifecycleEstimate(
        remaining_useful_life_years=Decimal("5"),
        annual_maintenance_cost=Decimal("1800"),
        annual_operating_cost=Decimal("2500"),
        annual_failure_probability=Decimal("0.12"),
        failure_event_cost=Decimal("10000"),
        expected_reliability=Decimal("0.86"),
        warranty_years=Decimal("0.5"),
        downtime_hours=Decimal("10"),
    )

    replacement_lifecycle = LifecycleEstimate(
        remaining_useful_life_years=Decimal("8"),
        annual_maintenance_cost=Decimal("600"),
        annual_operating_cost=Decimal("2100"),
        annual_failure_probability=Decimal("0.04"),
        failure_event_cost=Decimal("10000"),
        expected_reliability=Decimal("0.96"),
        warranty_years=Decimal("2"),
        downtime_hours=Decimal("6"),
    )

    request = RepairReplaceRequest(
        component=component,
        repair_cost=repair_cost,
        replacement_cost=replacement_cost,
        repair_lifecycle=repair_lifecycle,
        replacement_lifecycle=replacement_lifecycle,
        currency="INR",
        analysis_years=5,
        discount_rate=Decimal("0.08"),
        budget=Decimal("30000"),
        maximum_downtime_hours=Decimal("12"),
        replacement_compatible=True,
        repair_safety_verified=True,
        replacement_safety_verified=True,
        notes="Demo robot joint motor assessment",
    )

    decision = engine.analyze(request)

    print("\nREPAIR VS REPLACE ANALYSIS")

    print(
        json.dumps(
            decision,
            indent=2,
        )
    )

    print("\nROBOT DECISION SUMMARY")

    print(
        json.dumps(
            engine.robot_decision_summary(
                tenant_id="tenant_alpha",
                robot_id="robot_001",
            ),
            indent=2,
        )
    )

    report_path = engine.export_json(
        "data/robotics/repair_vs_replace_report.json",
        tenant_id="tenant_alpha",
    )

    print("\nReport:", report_path)


if __name__ == "__main__":
    main()
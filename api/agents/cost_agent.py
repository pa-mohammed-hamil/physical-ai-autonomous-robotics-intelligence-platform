"""Cost Agent module.

TODO: implement and test; no real robot control is enabled.
"""
"""
backend/app/ai/agents/cost_agent.py

Physical AI & Autonomous Robotics Intelligence Platform
Cost Intelligence Agent

Features:
- Robot maintenance cost estimation
- Robot component repair and replacement analysis
- Spare parts and labor cost calculations
- Downtime and production-loss estimation
- Energy consumption cost analysis
- Preventive versus corrective maintenance comparison
- Failure risk and expected financial loss
- Repair-versus-replace recommendations
- Return on investment (ROI)
- Total cost of ownership (TCO)
- Multi-currency calculations
- Agent decision support and explainable reports
- Tenant isolation
- Input validation
- Deterministic financial calculations

All calculations use Decimal for monetary precision.

This agent provides financial decision support only.
It does not approve purchases, dispatch technicians,
or operate physical robots.
"""

from __future__ import annotations

import logging
import re
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


logger = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

MONEY_QUANTUM = Decimal("0.01")
ZERO = Decimal("0")
ONE = Decimal("1")
HUNDRED = Decimal("100")

MAX_PARTS = 200
MAX_COST_ITEMS = 300
MAX_ANALYSIS_YEARS = 30

IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z0-9_.:-]{1,128}$"
)


# ============================================================
# Utilities
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id() -> str:
    return str(uuid.uuid4())


def money(value: Decimal | int | str) -> Decimal:
    if isinstance(value, float):
        raise TypeError(
            "Use Decimal or a string for monetary values"
        )

    result = Decimal(value)

    if not result.is_finite():
        raise ValueError(
            "Monetary value must be finite"
        )

    return result.quantize(
        MONEY_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def nonnegative_money(
    value: Decimal | int | str,
) -> Decimal:
    result = money(value)

    if result < ZERO:
        raise ValueError(
            "Monetary value cannot be negative"
        )

    return result


def decimal_value(
    value: Decimal | int | str,
) -> Decimal:
    if isinstance(value, float):
        raise TypeError(
            "Use Decimal or a string for numeric inputs"
        )

    result = Decimal(value)

    if not result.is_finite():
        raise ValueError(
            "Numeric value must be finite"
        )

    return result


def nonnegative_decimal(
    value: Decimal | int | str,
) -> Decimal:
    result = decimal_value(value)

    if result < ZERO:
        raise ValueError(
            "Value cannot be negative"
        )

    return result


def probability(
    value: Decimal | int | str,
) -> Decimal:
    result = decimal_value(value)

    if not ZERO <= result <= ONE:
        raise ValueError(
            "Probability must be between 0 and 1"
        )

    return result


def validate_identifier(value: str) -> str:
    if not IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            "Invalid identifier"
        )

    return value


def safe_percentage(
    numerator: Decimal,
    denominator: Decimal,
) -> Decimal | None:
    if denominator == ZERO:
        return None

    return (
        numerator / denominator * HUNDRED
    ).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )


# ============================================================
# Enums
# ============================================================

class Currency(str, Enum):
    INR = "INR"
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    AED = "AED"
    CAD = "CAD"
    AUD = "AUD"
    SGD = "SGD"


class CostCategory(str, Enum):
    PARTS = "parts"
    LABOR = "labor"
    DIAGNOSTICS = "diagnostics"
    TRANSPORT = "transport"
    DOWNTIME = "downtime"
    PRODUCTION_LOSS = "production_loss"
    ENERGY = "energy"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"
    SOFTWARE = "software"
    HARDWARE = "hardware"
    TRAINING = "training"
    INSPECTION = "inspection"
    OTHER = "other"


class MaintenanceType(str, Enum):
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    EMERGENCY = "emergency"


class CostDecision(str, Enum):
    REPAIR = "repair"
    REPLACE = "replace"
    INSPECT = "inspect"
    DEFER = "defer"
    HUMAN_REVIEW = "human_review"


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class EstimateStatus(str, Enum):
    DRAFT = "draft"
    COMPLETED = "completed"
    REQUIRES_REVIEW = "requires_review"


# ============================================================
# Exceptions
# ============================================================

class CostAgentError(Exception):
    pass


class CostValidationError(CostAgentError):
    pass


class CostCalculationError(CostAgentError):
    pass


class CostDataUnavailableError(CostAgentError):
    pass


class CostPermissionError(CostAgentError):
    pass


# ============================================================
# Scope
# ============================================================

@dataclass(frozen=True, slots=True)
class CostAgentScope:
    tenant_id: str
    user_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.tenant_id)
        validate_identifier(self.user_id)


# ============================================================
# Cost Models
# ============================================================

@dataclass(slots=True)
class CostLineItem:
    item_id: str
    description: str
    category: CostCategory

    quantity: Decimal
    unit_cost: Decimal

    total_cost: Decimal = field(
        init=False
    )

    def __post_init__(self) -> None:
        validate_identifier(self.item_id)

        if not self.description.strip():
            raise CostValidationError(
                "Cost item description is required"
            )

        self.quantity = nonnegative_decimal(
            self.quantity
        )

        self.unit_cost = nonnegative_money(
            self.unit_cost
        )

        self.total_cost = money(
            self.quantity * self.unit_cost
        )


@dataclass(slots=True)
class RobotPart:
    part_id: str
    part_name: str

    quantity: int
    unit_price: Decimal

    installation_hours: Decimal = ZERO

    failure_probability: Decimal = ZERO

    def __post_init__(self) -> None:
        validate_identifier(self.part_id)

        if not self.part_name.strip():
            raise CostValidationError(
                "Part name is required"
            )

        if self.quantity < 1:
            raise CostValidationError(
                "Part quantity must be positive"
            )

        self.unit_price = nonnegative_money(
            self.unit_price
        )

        self.installation_hours = (
            nonnegative_decimal(
                self.installation_hours
            )
        )

        self.failure_probability = probability(
            self.failure_probability
        )

    @property
    def total_price(self) -> Decimal:
        return money(
            self.unit_price * self.quantity
        )


@dataclass(slots=True)
class LaborCost:
    technician_hours: Decimal
    hourly_rate: Decimal

    specialist_hours: Decimal = ZERO
    specialist_hourly_rate: Decimal = ZERO

    def __post_init__(self) -> None:
        self.technician_hours = nonnegative_decimal(
            self.technician_hours
        )

        self.hourly_rate = nonnegative_money(
            self.hourly_rate
        )

        self.specialist_hours = nonnegative_decimal(
            self.specialist_hours
        )

        self.specialist_hourly_rate = (
            nonnegative_money(
                self.specialist_hourly_rate
            )
        )

    @property
    def total_cost(self) -> Decimal:
        return money(
            self.technician_hours * self.hourly_rate
            + self.specialist_hours
            * self.specialist_hourly_rate
        )


@dataclass(slots=True)
class DowntimeCost:
    downtime_hours: Decimal
    cost_per_hour: Decimal

    lost_production_units: Decimal = ZERO
    profit_per_unit: Decimal = ZERO

    def __post_init__(self) -> None:
        self.downtime_hours = nonnegative_decimal(
            self.downtime_hours
        )

        self.cost_per_hour = nonnegative_money(
            self.cost_per_hour
        )

        self.lost_production_units = (
            nonnegative_decimal(
                self.lost_production_units
            )
        )

        self.profit_per_unit = nonnegative_money(
            self.profit_per_unit
        )

    @property
    def operational_downtime_cost(self) -> Decimal:
        return money(
            self.downtime_hours
            * self.cost_per_hour
        )

    @property
    def production_loss(self) -> Decimal:
        return money(
            self.lost_production_units
            * self.profit_per_unit
        )

    @property
    def total_cost(self) -> Decimal:
        return money(
            self.operational_downtime_cost
            + self.production_loss
        )


@dataclass(slots=True)
class EnergyCost:
    power_kw: Decimal
    operating_hours: Decimal
    electricity_price_per_kwh: Decimal

    def __post_init__(self) -> None:
        self.power_kw = nonnegative_decimal(
            self.power_kw
        )

        self.operating_hours = nonnegative_decimal(
            self.operating_hours
        )

        self.electricity_price_per_kwh = (
            nonnegative_money(
                self.electricity_price_per_kwh
            )
        )

    @property
    def energy_kwh(self) -> Decimal:
        return (
            self.power_kw
            * self.operating_hours
        )

    @property
    def total_cost(self) -> Decimal:
        return money(
            self.energy_kwh
            * self.electricity_price_per_kwh
        )


# ============================================================
# Repair Estimate
# ============================================================

@dataclass(slots=True)
class RepairEstimateRequest:
    robot_id: str
    currency: Currency

    parts: list[RobotPart]

    labor: LaborCost
    downtime: DowntimeCost

    diagnostic_cost: Decimal = ZERO
    transportation_cost: Decimal = ZERO
    emergency_cost: Decimal = ZERO
    miscellaneous_cost: Decimal = ZERO

    contingency_rate: Decimal = Decimal("0.10")
    tax_rate: Decimal = ZERO

    def __post_init__(self) -> None:
        validate_identifier(self.robot_id)

        if len(self.parts) > MAX_PARTS:
            raise CostValidationError(
                "Too many repair parts"
            )

        for name in (
            "diagnostic_cost",
            "transportation_cost",
            "emergency_cost",
            "miscellaneous_cost",
        ):
            setattr(
                self,
                name,
                nonnegative_money(
                    getattr(self, name)
                ),
            )

        self.contingency_rate = probability(
            self.contingency_rate
        )

        self.tax_rate = probability(
            self.tax_rate
        )


@dataclass(slots=True)
class RepairEstimate:
    estimate_id: str
    tenant_id: str
    robot_id: str
    currency: Currency

    parts_cost: Decimal
    labor_cost: Decimal
    downtime_cost: Decimal
    production_loss_cost: Decimal

    diagnostic_cost: Decimal
    transportation_cost: Decimal
    emergency_cost: Decimal
    miscellaneous_cost: Decimal

    subtotal: Decimal
    contingency_cost: Decimal
    tax_cost: Decimal
    total_estimated_cost: Decimal

    status: EstimateStatus

    created_at: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# Repair Versus Replace
# ============================================================

@dataclass(slots=True)
class RepairReplaceRequest:
    robot_id: str
    currency: Currency

    repair_cost: Decimal
    replacement_cost: Decimal

    repair_expected_remaining_years: Decimal
    replacement_expected_life_years: Decimal

    annual_repair_maintenance_cost: Decimal
    annual_replacement_maintenance_cost: Decimal

    repair_downtime_cost: Decimal = ZERO
    replacement_downtime_cost: Decimal = ZERO

    old_robot_resale_value: Decimal = ZERO

    discount_rate: Decimal = Decimal("0.08")

    safety_critical_issue: bool = False
    repair_feasible: bool = True

    def __post_init__(self) -> None:
        validate_identifier(self.robot_id)

        for name in (
            "repair_cost",
            "replacement_cost",
            "annual_repair_maintenance_cost",
            "annual_replacement_maintenance_cost",
            "repair_downtime_cost",
            "replacement_downtime_cost",
            "old_robot_resale_value",
        ):
            setattr(
                self,
                name,
                nonnegative_money(
                    getattr(self, name)
                ),
            )

        self.repair_expected_remaining_years = (
            nonnegative_decimal(
                self.repair_expected_remaining_years
            )
        )

        self.replacement_expected_life_years = (
            nonnegative_decimal(
                self.replacement_expected_life_years
            )
        )

        if (
            self.repair_expected_remaining_years <= ZERO
            or self.replacement_expected_life_years <= ZERO
        ):
            raise CostValidationError(
                "Expected life must be positive"
            )

        if (
            self.repair_expected_remaining_years
            > MAX_ANALYSIS_YEARS
            or self.replacement_expected_life_years
            > MAX_ANALYSIS_YEARS
        ):
            raise CostValidationError(
                "Expected life exceeds supported horizon"
            )

        self.discount_rate = probability(
            self.discount_rate
        )


@dataclass(slots=True)
class RepairReplaceAnalysis:
    analysis_id: str
    tenant_id: str
    robot_id: str
    currency: Currency

    repair_lifecycle_cost: Decimal
    replacement_lifecycle_cost: Decimal

    repair_annualized_cost: Decimal
    replacement_annualized_cost: Decimal

    estimated_annual_savings: Decimal

    recommendation: CostDecision
    requires_human_review: bool

    reasoning: list[str]

    generated_at: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# Failure Risk
# ============================================================

@dataclass(slots=True)
class FailureRiskRequest:
    robot_id: str
    currency: Currency

    failure_probability: Decimal

    repair_cost_if_failure: Decimal
    downtime_cost_if_failure: Decimal
    emergency_cost_if_failure: Decimal = ZERO

    preventive_maintenance_cost: Decimal = ZERO

    residual_failure_probability: Decimal | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.robot_id)

        self.failure_probability = probability(
            self.failure_probability
        )

        if self.residual_failure_probability is not None:
            self.residual_failure_probability = probability(
                self.residual_failure_probability
            )

        for name in (
            "repair_cost_if_failure",
            "downtime_cost_if_failure",
            "emergency_cost_if_failure",
            "preventive_maintenance_cost",
        ):
            setattr(
                self,
                name,
                nonnegative_money(
                    getattr(self, name)
                ),
            )


@dataclass(slots=True)
class FailureRiskAnalysis:
    analysis_id: str
    tenant_id: str
    robot_id: str
    currency: Currency

    failure_probability: Decimal
    total_failure_impact: Decimal
    expected_failure_loss: Decimal

    preventive_maintenance_cost: Decimal
    expected_cost_after_prevention: Decimal | None

    estimated_prevention_savings: Decimal | None

    risk_level: RiskLevel

    recommendation: CostDecision

    generated_at: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# ROI and TCO
# ============================================================

@dataclass(slots=True)
class ROIRequest:
    currency: Currency

    initial_investment: Decimal
    annual_savings: Decimal
    annual_operating_cost: Decimal

    years: int

    discount_rate: Decimal = Decimal("0.08")

    def __post_init__(self) -> None:
        self.initial_investment = nonnegative_money(
            self.initial_investment
        )

        self.annual_savings = nonnegative_money(
            self.annual_savings
        )

        self.annual_operating_cost = nonnegative_money(
            self.annual_operating_cost
        )

        if not 1 <= self.years <= MAX_ANALYSIS_YEARS:
            raise CostValidationError(
                "Invalid analysis horizon"
            )

        self.discount_rate = probability(
            self.discount_rate
        )


@dataclass(slots=True)
class ROIAnalysis:
    analysis_id: str
    currency: Currency

    initial_investment: Decimal
    total_savings: Decimal
    total_operating_cost: Decimal

    net_benefit: Decimal

    roi_percentage: Decimal | None
    net_present_value: Decimal

    payback_period_years: Decimal | None

    generated_at: datetime = field(
        default_factory=utc_now
    )


@dataclass(slots=True)
class TCORequest:
    robot_id: str
    currency: Currency

    purchase_cost: Decimal
    installation_cost: Decimal

    annual_maintenance_cost: Decimal
    annual_energy_cost: Decimal
    annual_software_cost: Decimal
    annual_operating_cost: Decimal

    expected_residual_value: Decimal = ZERO

    years: int = 5

    discount_rate: Decimal = Decimal("0.08")

    def __post_init__(self) -> None:
        validate_identifier(self.robot_id)

        for name in (
            "purchase_cost",
            "installation_cost",
            "annual_maintenance_cost",
            "annual_energy_cost",
            "annual_software_cost",
            "annual_operating_cost",
            "expected_residual_value",
        ):
            setattr(
                self,
                name,
                nonnegative_money(
                    getattr(self, name)
                ),
            )

        if not 1 <= self.years <= MAX_ANALYSIS_YEARS:
            raise CostValidationError(
                "Invalid TCO horizon"
            )

        self.discount_rate = probability(
            self.discount_rate
        )


@dataclass(slots=True)
class TCOAnalysis:
    analysis_id: str
    tenant_id: str
    robot_id: str
    currency: Currency

    acquisition_cost: Decimal
    total_operational_cost: Decimal

    residual_value: Decimal

    undiscounted_tco: Decimal
    discounted_tco: Decimal

    annualized_tco: Decimal

    generated_at: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# Cost Repository
# ============================================================

class CostAnalysisRepository(Protocol):

    async def save_analysis(
        self,
        *,
        tenant_id: str,
        analysis_id: str,
        analysis_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        ...


class CostAuditSink(Protocol):

    async def record(
        self,
        *,
        tenant_id: str,
        user_id: str,
        action: str,
        resource_id: str,
        metadata: Mapping[str, Any],
    ) -> None:
        ...


# ============================================================
# Cost Agent
# ============================================================

class CostAgent:

    def __init__(
        self,
        *,
        repository: CostAnalysisRepository | None = None,
        audit_sink: CostAuditSink | None = None,
    ) -> None:
        self.repository = repository
        self.audit_sink = audit_sink

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    async def _save(
        self,
        *,
        scope: CostAgentScope,
        analysis_id: str,
        analysis_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        if self.repository is None:
            return

        await self.repository.save_analysis(
            tenant_id=scope.tenant_id,
            analysis_id=analysis_id,
            analysis_type=analysis_type,
            payload=payload,
        )

    async def _audit(
        self,
        *,
        scope: CostAgentScope,
        action: str,
        resource_id: str,
        metadata: Mapping[str, Any],
    ) -> None:
        if self.audit_sink is None:
            return

        await self.audit_sink.record(
            tenant_id=scope.tenant_id,
            user_id=scope.user_id,
            action=action,
            resource_id=resource_id,
            metadata=metadata,
        )

    # --------------------------------------------------------
    # Repair Cost
    # --------------------------------------------------------

    async def estimate_repair_cost(
        self,
        *,
        scope: CostAgentScope,
        request: RepairEstimateRequest,
    ) -> RepairEstimate:

        parts_cost = money(
            sum(
                (
                    part.total_price
                    for part in request.parts
                ),
                ZERO,
            )
        )

        labor_cost = request.labor.total_cost

        downtime_cost = (
            request.downtime.operational_downtime_cost
        )

        production_loss = (
            request.downtime.production_loss
        )

        subtotal = money(
            parts_cost
            + labor_cost
            + downtime_cost
            + production_loss
            + request.diagnostic_cost
            + request.transportation_cost
            + request.emergency_cost
            + request.miscellaneous_cost
        )

        contingency = money(
            subtotal * request.contingency_rate
        )

        taxable_base = subtotal + contingency

        tax_cost = money(
            taxable_base * request.tax_rate
        )

        total_cost = money(
            taxable_base + tax_cost
        )

        estimate = RepairEstimate(
            estimate_id=generate_id(),
            tenant_id=scope.tenant_id,
            robot_id=request.robot_id,
            currency=request.currency,
            parts_cost=parts_cost,
            labor_cost=labor_cost,
            downtime_cost=downtime_cost,
            production_loss_cost=production_loss,
            diagnostic_cost=request.diagnostic_cost,
            transportation_cost=request.transportation_cost,
            emergency_cost=request.emergency_cost,
            miscellaneous_cost=request.miscellaneous_cost,
            subtotal=subtotal,
            contingency_cost=contingency,
            tax_cost=tax_cost,
            total_estimated_cost=total_cost,
            status=EstimateStatus.COMPLETED,
        )

        await self._save(
            scope=scope,
            analysis_id=estimate.estimate_id,
            analysis_type="repair_estimate",
            payload={
                "robot_id": request.robot_id,
                "currency": request.currency.value,
                "total_cost": str(total_cost),
            },
        )

        await self._audit(
            scope=scope,
            action="cost_agent.repair_estimate",
            resource_id=estimate.estimate_id,
            metadata={
                "robot_id": request.robot_id,
            },
        )

        return estimate

    # --------------------------------------------------------
    # Discounted Cash Flow
    # --------------------------------------------------------

    @staticmethod
    def present_value_annuity(
        *,
        annual_cost: Decimal,
        years: Decimal,
        discount_rate: Decimal,
    ) -> Decimal:

        if years <= ZERO:
            raise CostValidationError(
                "Years must be positive"
            )

        if discount_rate == ZERO:
            return money(
                annual_cost * years
            )

        factor = (
            ONE
            - (
                ONE
                / (ONE + discount_rate) ** years
            )
        ) / discount_rate

        return money(
            annual_cost * factor
        )

    # --------------------------------------------------------
    # Repair Versus Replace
    # --------------------------------------------------------

    async def analyze_repair_vs_replace(
        self,
        *,
        scope: CostAgentScope,
        request: RepairReplaceRequest,
    ) -> RepairReplaceAnalysis:

        repair_annual_cost = (
            request.annual_repair_maintenance_cost
        )

        replacement_annual_cost = (
            request.annual_replacement_maintenance_cost
        )

        repair_maintenance_pv = (
            self.present_value_annuity(
                annual_cost=repair_annual_cost,
                years=(
                    request.repair_expected_remaining_years
                ),
                discount_rate=request.discount_rate,
            )
        )

        replacement_maintenance_pv = (
            self.present_value_annuity(
                annual_cost=replacement_annual_cost,
                years=(
                    request.replacement_expected_life_years
                ),
                discount_rate=request.discount_rate,
            )
        )

        repair_lifecycle_cost = money(
            request.repair_cost
            + request.repair_downtime_cost
            + repair_maintenance_pv
        )

        replacement_lifecycle_cost = money(
            request.replacement_cost
            + request.replacement_downtime_cost
            + replacement_maintenance_pv
            - request.old_robot_resale_value
        )

        repair_annualized = money(
            repair_lifecycle_cost
            / request.repair_expected_remaining_years
        )

        replacement_annualized = money(
            replacement_lifecycle_cost
            / request.replacement_expected_life_years
        )

        estimated_savings = money(
            repair_annualized
            - replacement_annualized
        )

        reasoning: list[str] = []

        requires_review = (
            request.safety_critical_issue
            or not request.repair_feasible
        )

        if request.safety_critical_issue:
            recommendation = CostDecision.HUMAN_REVIEW

            reasoning.append(
                "Safety-critical issue requires engineering "
                "inspection and human authorization."
            )

        elif not request.repair_feasible:
            recommendation = CostDecision.REPLACE

            reasoning.append(
                "Repair has been marked technically infeasible."
            )

        elif repair_annualized < replacement_annualized:
            recommendation = CostDecision.REPAIR

            reasoning.append(
                "Repair has a lower estimated annualized cost."
            )

        elif replacement_annualized < repair_annualized:
            recommendation = CostDecision.REPLACE

            reasoning.append(
                "Replacement has a lower estimated annualized cost."
            )

        else:
            recommendation = CostDecision.HUMAN_REVIEW
            requires_review = True

            reasoning.append(
                "Annualized financial costs are equivalent."
            )

        reasoning.append(
            "Comparison uses estimated service life, "
            "discounted maintenance costs, downtime, "
            "and replacement resale value."
        )

        reasoning.append(
            "Different service-life assumptions can affect "
            "the comparison. Confirm operational and "
            "technical suitability before deciding."
        )

        result = RepairReplaceAnalysis(
            analysis_id=generate_id(),
            tenant_id=scope.tenant_id,
            robot_id=request.robot_id,
            currency=request.currency,
            repair_lifecycle_cost=repair_lifecycle_cost,
            replacement_lifecycle_cost=(
                replacement_lifecycle_cost
            ),
            repair_annualized_cost=repair_annualized,
            replacement_annualized_cost=(
                replacement_annualized
            ),
            estimated_annual_savings=estimated_savings,
            recommendation=recommendation,
            requires_human_review=requires_review,
            reasoning=reasoning,
        )

        await self._save(
            scope=scope,
            analysis_id=result.analysis_id,
            analysis_type="repair_replace",
            payload={
                "robot_id": request.robot_id,
                "recommendation": recommendation.value,
                "repair_annualized_cost": str(
                    repair_annualized
                ),
                "replacement_annualized_cost": str(
                    replacement_annualized
                ),
            },
        )

        await self._audit(
            scope=scope,
            action="cost_agent.repair_replace",
            resource_id=result.analysis_id,
            metadata={
                "robot_id": request.robot_id,
                "recommendation": recommendation.value,
            },
        )

        return result

    # --------------------------------------------------------
    # Failure Risk
    # --------------------------------------------------------

    async def analyze_failure_risk(
        self,
        *,
        scope: CostAgentScope,
        request: FailureRiskRequest,
    ) -> FailureRiskAnalysis:

        total_failure_impact = money(
            request.repair_cost_if_failure
            + request.downtime_cost_if_failure
            + request.emergency_cost_if_failure
        )

        expected_failure_loss = money(
            request.failure_probability
            * total_failure_impact
        )

        expected_cost_after_prevention: Decimal | None = None
        prevention_savings: Decimal | None = None

        if request.residual_failure_probability is not None:
            expected_cost_after_prevention = money(
                request.preventive_maintenance_cost
                + (
                    request.residual_failure_probability
                    * total_failure_impact
                )
            )

            prevention_savings = money(
                expected_failure_loss
                - expected_cost_after_prevention
            )

        if request.failure_probability >= Decimal("0.75"):
            risk_level = RiskLevel.CRITICAL

        elif request.failure_probability >= Decimal("0.50"):
            risk_level = RiskLevel.HIGH

        elif request.failure_probability >= Decimal("0.20"):
            risk_level = RiskLevel.MODERATE

        else:
            risk_level = RiskLevel.LOW

        if risk_level == RiskLevel.CRITICAL:
            recommendation = CostDecision.HUMAN_REVIEW

        elif (
            prevention_savings is not None
            and prevention_savings > ZERO
        ):
            recommendation = CostDecision.INSPECT

        elif risk_level == RiskLevel.HIGH:
            recommendation = CostDecision.INSPECT

        else:
            recommendation = CostDecision.DEFER

        result = FailureRiskAnalysis(
            analysis_id=generate_id(),
            tenant_id=scope.tenant_id,
            robot_id=request.robot_id,
            currency=request.currency,
            failure_probability=request.failure_probability,
            total_failure_impact=total_failure_impact,
            expected_failure_loss=expected_failure_loss,
            preventive_maintenance_cost=(
                request.preventive_maintenance_cost
            ),
            expected_cost_after_prevention=(
                expected_cost_after_prevention
            ),
            estimated_prevention_savings=(
                prevention_savings
            ),
            risk_level=risk_level,
            recommendation=recommendation,
        )

        await self._save(
            scope=scope,
            analysis_id=result.analysis_id,
            analysis_type="failure_risk",
            payload={
                "robot_id": request.robot_id,
                "risk_level": risk_level.value,
                "expected_failure_loss": str(
                    expected_failure_loss
                ),
            },
        )

        await self._audit(
            scope=scope,
            action="cost_agent.failure_risk",
            resource_id=result.analysis_id,
            metadata={
                "robot_id": request.robot_id,
                "risk_level": risk_level.value,
            },
        )

        return result

    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    async def calculate_roi(
        self,
        *,
        request: ROIRequest,
    ) -> ROIAnalysis:

        total_savings = money(
            request.annual_savings * request.years
        )

        total_operating_cost = money(
            request.annual_operating_cost
            * request.years
        )

        net_benefit = money(
            total_savings
            - total_operating_cost
            - request.initial_investment
        )

        roi_percentage = safe_percentage(
            net_benefit,
            request.initial_investment,
        )

        annual_net_cashflow = (
            request.annual_savings
            - request.annual_operating_cost
        )

        npv = -request.initial_investment

        for year in range(1, request.years + 1):
            npv += (
                annual_net_cashflow
                / (ONE + request.discount_rate) ** year
            )

        npv = money(npv)

        payback: Decimal | None = None

        if annual_net_cashflow > ZERO:
            payback = (
                request.initial_investment
                / annual_net_cashflow
            ).quantize(
                Decimal("0.01"),
                rounding=ROUND_HALF_UP,
            )

        return ROIAnalysis(
            analysis_id=generate_id(),
            currency=request.currency,
            initial_investment=request.initial_investment,
            total_savings=total_savings,
            total_operating_cost=total_operating_cost,
            net_benefit=net_benefit,
            roi_percentage=roi_percentage,
            net_present_value=npv,
            payback_period_years=payback,
        )

    # --------------------------------------------------------
    # Total Cost of Ownership
    # --------------------------------------------------------

    async def calculate_tco(
        self,
        *,
        scope: CostAgentScope,
        request: TCORequest,
    ) -> TCOAnalysis:

        acquisition_cost = money(
            request.purchase_cost
            + request.installation_cost
        )

        annual_operational_cost = money(
            request.annual_maintenance_cost
            + request.annual_energy_cost
            + request.annual_software_cost
            + request.annual_operating_cost
        )

        total_operational_cost = money(
            annual_operational_cost
            * request.years
        )

        undiscounted_tco = money(
            acquisition_cost
            + total_operational_cost
            - request.expected_residual_value
        )

        operational_pv = (
            self.present_value_annuity(
                annual_cost=annual_operational_cost,
                years=Decimal(request.years),
                discount_rate=request.discount_rate,
            )
        )

        residual_pv = (
            request.expected_residual_value
            / (
                ONE + request.discount_rate
            ) ** request.years
        )

        discounted_tco = money(
            acquisition_cost
            + operational_pv
            - residual_pv
        )

        annualized_tco = money(
            undiscounted_tco
            / request.years
        )

        result = TCOAnalysis(
            analysis_id=generate_id(),
            tenant_id=scope.tenant_id,
            robot_id=request.robot_id,
            currency=request.currency,
            acquisition_cost=acquisition_cost,
            total_operational_cost=total_operational_cost,
            residual_value=request.expected_residual_value,
            undiscounted_tco=undiscounted_tco,
            discounted_tco=discounted_tco,
            annualized_tco=annualized_tco,
        )

        await self._save(
            scope=scope,
            analysis_id=result.analysis_id,
            analysis_type="total_cost_of_ownership",
            payload={
                "robot_id": request.robot_id,
                "discounted_tco": str(
                    discounted_tco
                ),
                "currency": request.currency.value,
            },
        )

        await self._audit(
            scope=scope,
            action="cost_agent.tco",
            resource_id=result.analysis_id,
            metadata={
                "robot_id": request.robot_id,
            },
        )

        return result

    # --------------------------------------------------------
    # Energy Cost
    # --------------------------------------------------------

    async def calculate_energy_cost(
        self,
        *,
        energy: EnergyCost,
    ) -> Decimal:
        return energy.total_cost

    # --------------------------------------------------------
    # Parts Cost
    # --------------------------------------------------------

    async def calculate_parts_cost(
        self,
        *,
        parts: Sequence[RobotPart],
    ) -> Decimal:

        if len(parts) > MAX_PARTS:
            raise CostValidationError(
                "Too many parts"
            )

        return money(
            sum(
                (
                    part.total_price
                    for part in parts
                ),
                ZERO,
            )
        )

    # --------------------------------------------------------
    # Cost Breakdown
    # --------------------------------------------------------

    async def calculate_cost_breakdown(
        self,
        *,
        items: Sequence[CostLineItem],
    ) -> dict[CostCategory, Decimal]:

        if len(items) > MAX_COST_ITEMS:
            raise CostValidationError(
                "Too many cost line items"
            )

        totals: dict[CostCategory, Decimal] = {
            category: ZERO
            for category in CostCategory
        }

        for item in items:
            totals[item.category] = money(
                totals[item.category]
                + item.total_cost
            )

        return totals


# ============================================================
# Factory
# ============================================================

def create_cost_agent(
    *,
    repository: CostAnalysisRepository | None = None,
    audit_sink: CostAuditSink | None = None,
) -> CostAgent:

    return CostAgent(
        repository=repository,
        audit_sink=audit_sink,
    )


# ============================================================
# Exports
# ============================================================

__all__ = [
    "Currency",
    "CostCategory",
    "MaintenanceType",
    "CostDecision",
    "RiskLevel",
    "EstimateStatus",
    "CostAgentScope",
    "CostLineItem",
    "RobotPart",
    "LaborCost",
    "DowntimeCost",
    "EnergyCost",
    "RepairEstimateRequest",
    "RepairEstimate",
    "RepairReplaceRequest",
    "RepairReplaceAnalysis",
    "FailureRiskRequest",
    "FailureRiskAnalysis",
    "ROIRequest",
    "ROIAnalysis",
    "TCORequest",
    "TCOAnalysis",
    "CostAnalysisRepository",
    "CostAuditSink",
    "CostAgent",
    "CostAgentError",
    "CostValidationError",
    "CostCalculationError",
    "CostDataUnavailableError",
    "CostPermissionError",
    "create_cost_agent",
]
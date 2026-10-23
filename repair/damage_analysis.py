"""Damage Analysis module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
backend/app/robotics/control/cost_estimation.py

Python 3.11+
Standard library only.

ROBOTICS COST ESTIMATION ENGINE

Features:
    - Multi-tenant robot cost estimates
    - Inspection and diagnostic costs
    - Spare-part and component costs
    - Technician labor costs
    - Repair and replacement estimates
    - Downtime and production loss
    - Energy consumption cost
    - Emergency service costs
    - Expected failure loss
    - Predictive maintenance costs
    - Repair-versus-replace comparison
    - Cost breakdown and ROI
    - Estimate history
    - Audit logging
    - JSON report export

All estimates are planning calculations.

They are not live vendor quotations, certified
engineering estimates, or guarantees of cost.
"""

from __future__ import annotations

import json
import math
import re
import threading
import uuid

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from typing import Any


# ============================================================
# EXCEPTIONS
# ============================================================

class CostEstimationError(Exception):
    pass


class CostValidationError(CostEstimationError):
    pass


class CostEstimateNotFound(CostEstimationError):
    pass


class CostConflict(CostEstimationError):
    pass


# ============================================================
# ENUMS
# ============================================================

class CostCategory(str, Enum):
    INSPECTION = "inspection"
    DIAGNOSTICS = "diagnostics"
    PARTS = "parts"
    LABOR = "labor"
    DOWNTIME = "downtime"
    ENERGY = "energy"
    EMERGENCY = "emergency"
    TRANSPORT = "transport"
    SOFTWARE = "software"
    CALIBRATION = "calibration"
    PREVENTIVE_MAINTENANCE = "preventive_maintenance"
    OTHER = "other"


class MaintenanceStrategy(str, Enum):
    CORRECTIVE = "corrective"
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    REPLACEMENT = "replacement"


class EstimateStatus(str, Enum):
    DRAFT = "draft"
    FINALIZED = "finalized"


class CostEventType(str, Enum):
    ESTIMATE_CREATED = "estimate_created"
    ESTIMATE_FINALIZED = "estimate_finalized"
    COMPARISON_CREATED = "comparison_created"


# ============================================================
# VALIDATION
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def validate_id(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(
            r"[A-Za-z0-9_.-]{1,128}",
            value,
        )
    ):
        raise CostValidationError(
            f"Invalid {name}"
        )

    return value


def validate_currency(value: str) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(
            r"[A-Z]{3}",
            value,
        )
    ):
        raise CostValidationError(
            "Currency must be a three-letter code"
        )

    return value


def number(
    value: Any,
    name: str,
    *,
    minimum: float = 0.0,
) -> Decimal:
    if isinstance(value, bool):
        raise CostValidationError(
            f"{name} cannot be boolean"
        )

    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise CostValidationError(
            f"Invalid {name}"
        ) from exc

    if not result.is_finite():
        raise CostValidationError(
            f"{name} must be finite"
        )

    if result < Decimal(str(minimum)):
        raise CostValidationError(
            f"{name} below minimum"
        )

    return result


def money(value: Decimal) -> Decimal:
    return value.quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )


def money_string(value: Decimal) -> str:
    return str(money(value))


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class CostEstimationConfig:
    maximum_estimates: int = 100000
    maximum_items_per_estimate: int = 500
    maximum_audit_events: int = 20000

    default_currency: str = "INR"

    simulation_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "maximum_estimates",
            "maximum_items_per_estimate",
            "maximum_audit_events",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise CostValidationError(
                    f"{name} must be positive"
                )

        validate_currency(
            self.default_currency
        )

        if not self.simulation_only:
            raise CostValidationError(
                "This estimator is planning-only"
            )


# ============================================================
# COST ITEM
# ============================================================

@dataclass(frozen=True)
class CostItem:
    item_id: str
    category: CostCategory

    description: str

    quantity: Decimal
    unit_cost: Decimal

    def __post_init__(self) -> None:
        validate_id(
            self.item_id,
            "item_id",
        )

        if not isinstance(
            self.category,
            CostCategory,
        ):
            raise CostValidationError(
                "Invalid cost category"
            )

        if (
            not isinstance(self.description, str)
            or not self.description.strip()
            or len(self.description) > 512
        ):
            raise CostValidationError(
                "Invalid item description"
            )

        object.__setattr__(
            self,
            "quantity",
            number(
                self.quantity,
                "quantity",
            ),
        )

        object.__setattr__(
            self,
            "unit_cost",
            number(
                self.unit_cost,
                "unit_cost",
            ),
        )

    @property
    def total(self) -> Decimal:
        return self.quantity * self.unit_cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "category": self.category.value,
            "description": self.description,
            "quantity": str(self.quantity),
            "unit_cost": money_string(
                self.unit_cost
            ),
            "total": money_string(
                self.total
            ),
        }


# ============================================================
# FAILURE RISK
# ============================================================

@dataclass(frozen=True)
class FailureRiskEstimate:
    failure_probability: Decimal
    failure_consequence_cost: Decimal

    def __post_init__(self) -> None:
        probability = number(
            self.failure_probability,
            "failure_probability",
        )

        if probability > 1:
            raise CostValidationError(
                "Failure probability must be 0-1"
            )

        object.__setattr__(
            self,
            "failure_probability",
            probability,
        )

        object.__setattr__(
            self,
            "failure_consequence_cost",
            number(
                self.failure_consequence_cost,
                "failure_consequence_cost",
            ),
        )

    @property
    def expected_loss(self) -> Decimal:
        return (
            self.failure_probability
            * self.failure_consequence_cost
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_probability": str(
                self.failure_probability
            ),
            "failure_consequence_cost": (
                money_string(
                    self.failure_consequence_cost
                )
            ),
            "expected_loss": money_string(
                self.expected_loss
            ),
        }


# ============================================================
# OPERATING COST
# ============================================================

@dataclass(frozen=True)
class RobotOperatingCost:
    operating_hours: Decimal

    energy_consumption_kwh_per_hour: Decimal
    electricity_price_per_kwh: Decimal

    downtime_hours: Decimal = Decimal("0")
    downtime_loss_per_hour: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        for name in (
            "operating_hours",
            "energy_consumption_kwh_per_hour",
            "electricity_price_per_kwh",
            "downtime_hours",
            "downtime_loss_per_hour",
        ):
            object.__setattr__(
                self,
                name,
                number(
                    getattr(self, name),
                    name,
                ),
            )

    @property
    def energy_cost(self) -> Decimal:
        return (
            self.operating_hours
            * self.energy_consumption_kwh_per_hour
            * self.electricity_price_per_kwh
        )

    @property
    def downtime_cost(self) -> Decimal:
        return (
            self.downtime_hours
            * self.downtime_loss_per_hour
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "operating_hours": str(
                self.operating_hours
            ),
            "energy_cost": money_string(
                self.energy_cost
            ),
            "downtime_hours": str(
                self.downtime_hours
            ),
            "downtime_cost": money_string(
                self.downtime_cost
            ),
        }


# ============================================================
# ESTIMATE REQUEST
# ============================================================

@dataclass(frozen=True)
class RobotCostEstimateRequest:
    tenant_id: str
    robot_id: str
    estimate_name: str

    strategy: MaintenanceStrategy

    currency: str = "INR"

    items: tuple[CostItem, ...] = ()

    operating_cost: RobotOperatingCost | None = None
    failure_risk: FailureRiskEstimate | None = None

    tax_rate: Decimal = Decimal("0")
    contingency_rate: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        validate_id(
            self.tenant_id,
            "tenant_id",
        )

        validate_id(
            self.robot_id,
            "robot_id",
        )

        if (
            not isinstance(self.estimate_name, str)
            or not self.estimate_name.strip()
            or len(self.estimate_name) > 256
        ):
            raise CostValidationError(
                "Invalid estimate name"
            )

        if not isinstance(
            self.strategy,
            MaintenanceStrategy,
        ):
            raise CostValidationError(
                "Invalid maintenance strategy"
            )

        validate_currency(
            self.currency
        )

        if not isinstance(self.items, tuple):
            raise CostValidationError(
                "Items must be a tuple"
            )

        item_ids = set()

        for item in self.items:
            if not isinstance(item, CostItem):
                raise CostValidationError(
                    "Invalid cost item"
                )

            if item.item_id in item_ids:
                raise CostValidationError(
                    "Duplicate cost item"
                )

            item_ids.add(item.item_id)

        if (
            self.operating_cost is not None
            and not isinstance(
                self.operating_cost,
                RobotOperatingCost,
            )
        ):
            raise CostValidationError(
                "Invalid operating cost"
            )

        if (
            self.failure_risk is not None
            and not isinstance(
                self.failure_risk,
                FailureRiskEstimate,
            )
        ):
            raise CostValidationError(
                "Invalid failure risk"
            )

        for name in (
            "tax_rate",
            "contingency_rate",
        ):
            rate = number(
                getattr(self, name),
                name,
            )

            if rate > 1:
                raise CostValidationError(
                    f"{name} must be 0-1"
                )

            object.__setattr__(
                self,
                name,
                rate,
            )


# ============================================================
# ESTIMATE RECORD
# ============================================================

@dataclass
class RobotCostEstimateRecord:
    estimate_id: str
    request: RobotCostEstimateRequest

    status: EstimateStatus = EstimateStatus.DRAFT

    category_totals: dict[
        CostCategory,
        Decimal,
    ] = field(default_factory=dict)

    item_subtotal: Decimal = Decimal("0")
    energy_cost: Decimal = Decimal("0")
    downtime_cost: Decimal = Decimal("0")

    expected_failure_loss: Decimal = Decimal("0")

    contingency_cost: Decimal = Decimal("0")
    tax_cost: Decimal = Decimal("0")

    total_cost: Decimal = Decimal("0")

    created_at: datetime = field(
        default_factory=utc_now
    )

    finalized_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate_id": self.estimate_id,
            "tenant_id": self.request.tenant_id,
            "robot_id": self.request.robot_id,
            "estimate_name": (
                self.request.estimate_name
            ),
            "strategy": (
                self.request.strategy.value
            ),
            "currency": self.request.currency,
            "status": self.status.value,
            "items": [
                item.to_dict()
                for item in self.request.items
            ],
            "category_totals": {
                category.value: money_string(value)
                for category, value
                in self.category_totals.items()
            },
            "item_subtotal": money_string(
                self.item_subtotal
            ),
            "energy_cost": money_string(
                self.energy_cost
            ),
            "downtime_cost": money_string(
                self.downtime_cost
            ),
            "expected_failure_loss": money_string(
                self.expected_failure_loss
            ),
            "contingency_cost": money_string(
                self.contingency_cost
            ),
            "tax_cost": money_string(
                self.tax_cost
            ),
            "total_cost": money_string(
                self.total_cost
            ),
            "created_at": (
                self.created_at.isoformat()
            ),
            "finalized_at": (
                self.finalized_at.isoformat()
                if self.finalized_at
                else None
            ),
        }


# ============================================================
# AUDIT EVENT
# ============================================================

@dataclass(frozen=True)
class CostAuditEvent:
    event_id: str
    timestamp: datetime

    tenant_id: str
    robot_id: str

    event_type: CostEventType
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": self.tenant_id,
            "robot_id": self.robot_id,
            "event_type": self.event_type.value,
            "details": self.details,
        }


# ============================================================
# COST ESTIMATION ENGINE
# ============================================================

class RoboticsCostEstimationEngine:

    """
    Thread-safe robot cost estimation engine.

    Formula:

        Item subtotal = sum(quantity * unit_cost)

        Energy = operating_hours
                 * kWh_per_hour
                 * electricity_rate

        Downtime = downtime_hours
                   * downtime_loss_per_hour

        Expected failure loss =
            failure_probability
            * failure_consequence_cost

        Base =
            items + energy + downtime
            + expected failure loss

        Contingency = base * contingency_rate

        Tax = (base + contingency) * tax_rate

        Total = base + contingency + tax

    The failure consequence cost should exclude
    losses already counted elsewhere to avoid
    double counting.
    """

    def __init__(
        self,
        config: CostEstimationConfig | None = None,
    ) -> None:
        self.config = (
            config or CostEstimationConfig()
        )

        self._estimates: dict[
            str,
            RobotCostEstimateRecord,
        ] = {}

        self._audit_events: deque[
            CostAuditEvent
        ] = deque(
            maxlen=self.config.maximum_audit_events
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    def _audit(
        self,
        tenant_id: str,
        robot_id: str,
        event_type: CostEventType,
        details: str,
    ) -> None:
        self._audit_events.append(
            CostAuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=utc_now(),
                tenant_id=tenant_id,
                robot_id=robot_id,
                event_type=event_type,
                details=details,
            )
        )

    # --------------------------------------------------------
    # GET INTERNAL
    # --------------------------------------------------------

    def _get(
        self,
        estimate_id: str,
        tenant_id: str,
    ) -> RobotCostEstimateRecord:
        validate_id(
            tenant_id,
            "tenant_id",
        )

        record = self._estimates.get(
            estimate_id
        )

        if (
            record is None
            or record.request.tenant_id != tenant_id
        ):
            raise CostEstimateNotFound(
                "Estimate not found"
            )

        return record

    # --------------------------------------------------------
    # CALCULATE ESTIMATE
    # --------------------------------------------------------

    def create_estimate(
        self,
        request: RobotCostEstimateRequest,
    ) -> dict[str, Any]:
        if not isinstance(
            request,
            RobotCostEstimateRequest,
        ):
            raise CostValidationError(
                "Invalid estimate request"
            )

        if (
            len(request.items)
            > self.config.maximum_items_per_estimate
        ):
            raise CostValidationError(
                "Too many cost items"
            )

        category_totals = {
            category: Decimal("0")
            for category in CostCategory
        }

        for item in request.items:
            category_totals[
                item.category
            ] += item.total

        item_subtotal = sum(
            category_totals.values(),
            Decimal("0"),
        )

        energy_cost = Decimal("0")
        downtime_cost = Decimal("0")

        if request.operating_cost is not None:
            energy_cost = (
                request.operating_cost.energy_cost
            )

            downtime_cost = (
                request.operating_cost.downtime_cost
            )

        expected_failure_loss = Decimal("0")

        if request.failure_risk is not None:
            expected_failure_loss = (
                request.failure_risk.expected_loss
            )

        base = (
            item_subtotal
            + energy_cost
            + downtime_cost
            + expected_failure_loss
        )

        contingency = (
            base * request.contingency_rate
        )

        tax = (
            (base + contingency)
            * request.tax_rate
        )

        total = (
            base + contingency + tax
        )

        record = RobotCostEstimateRecord(
            estimate_id=str(uuid.uuid4()),
            request=request,
            category_totals=category_totals,
            item_subtotal=item_subtotal,
            energy_cost=energy_cost,
            downtime_cost=downtime_cost,
            expected_failure_loss=(
                expected_failure_loss
            ),
            contingency_cost=contingency,
            tax_cost=tax,
            total_cost=total,
        )

        with self._lock:
            if (
                len(self._estimates)
                >= self.config.maximum_estimates
            ):
                raise CostEstimationError(
                    "Estimate capacity reached"
                )

            self._estimates[
                record.estimate_id
            ] = record

            self._audit(
                request.tenant_id,
                request.robot_id,
                CostEventType.ESTIMATE_CREATED,
                (
                    f"Estimate {record.estimate_id} "
                    f"created"
                ),
            )

            return record.to_dict()

    # --------------------------------------------------------
    # FINALIZE
    # --------------------------------------------------------

    def finalize_estimate(
        self,
        *,
        estimate_id: str,
        tenant_id: str,
        authorized: bool,
    ) -> dict[str, Any]:
        if authorized is not True:
            raise CostConflict(
                "Finalization authorization required"
            )

        with self._lock:
            record = self._get(
                estimate_id,
                tenant_id,
            )

            if record.status == EstimateStatus.FINALIZED:
                raise CostConflict(
                    "Estimate already finalized"
                )

            record.status = (
                EstimateStatus.FINALIZED
            )

            record.finalized_at = utc_now()

            self._audit(
                record.request.tenant_id,
                record.request.robot_id,
                CostEventType.ESTIMATE_FINALIZED,
                (
                    f"Estimate {estimate_id} finalized"
                ),
            )

            return record.to_dict()

    # --------------------------------------------------------
    # GET ESTIMATE
    # --------------------------------------------------------

    def get_estimate(
        self,
        *,
        estimate_id: str,
        tenant_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            return self._get(
                estimate_id,
                tenant_id,
            ).to_dict()

    # --------------------------------------------------------
    # LIST ESTIMATES
    # --------------------------------------------------------

    def list_estimates(
        self,
        *,
        tenant_id: str,
        robot_id: str | None = None,
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

        with self._lock:
            records = [
                record
                for record in self._estimates.values()
                if (
                    record.request.tenant_id
                    == tenant_id
                    and (
                        robot_id is None
                        or record.request.robot_id
                        == robot_id
                    )
                )
            ]

            return [
                record.to_dict()
                for record in sorted(
                    records,
                    key=lambda item: item.created_at,
                )
            ]

    # --------------------------------------------------------
    # REPAIR VS REPLACEMENT
    # --------------------------------------------------------

    def compare_repair_replacement(
        self,
        *,
        tenant_id: str,
        robot_id: str,
        repair_estimate_id: str,
        replacement_estimate_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            repair = self._get(
                repair_estimate_id,
                tenant_id,
            )

            replacement = self._get(
                replacement_estimate_id,
                tenant_id,
            )

            if (
                repair.request.robot_id != robot_id
                or replacement.request.robot_id != robot_id
            ):
                raise CostValidationError(
                    "Robot identity mismatch"
                )

            if (
                repair.request.currency
                != replacement.request.currency
            ):
                raise CostValidationError(
                    "Currencies must match"
                )

            if (
                replacement.request.strategy
                != MaintenanceStrategy.REPLACEMENT
            ):
                raise CostValidationError(
                    "Replacement estimate must use "
                    "replacement strategy"
                )

            repair_cost = repair.total_cost
            replacement_cost = (
                replacement.total_cost
            )

            savings = (
                replacement_cost - repair_cost
            )

            if repair_cost < replacement_cost:
                recommendation = "repair"
            elif replacement_cost < repair_cost:
                recommendation = "replace"
            else:
                recommendation = "equal_cost"

            result = {
                "tenant_id": tenant_id,
                "robot_id": robot_id,
                "currency": (
                    repair.request.currency
                ),
                "repair_estimate_id": (
                    repair_estimate_id
                ),
                "replacement_estimate_id": (
                    replacement_estimate_id
                ),
                "repair_cost": money_string(
                    repair_cost
                ),
                "replacement_cost": money_string(
                    replacement_cost
                ),
                "repair_savings_vs_replacement": (
                    money_string(savings)
                ),
                "lower_estimated_cost": recommendation,
                "note": (
                    "Cost-only comparison; remaining "
                    "useful life, reliability, warranty "
                    "and performance require review."
                ),
            }

            self._audit(
                tenant_id,
                robot_id,
                CostEventType.COMPARISON_CREATED,
                "Repair/replacement comparison created",
            )

            return result

    # --------------------------------------------------------
    # MAINTENANCE ROI
    # --------------------------------------------------------

    def calculate_maintenance_roi(
        self,
        *,
        tenant_id: str,
        baseline_estimate_id: str,
        proposed_estimate_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            baseline = self._get(
                baseline_estimate_id,
                tenant_id,
            )

            proposed = self._get(
                proposed_estimate_id,
                tenant_id,
            )

            if (
                baseline.request.robot_id
                != proposed.request.robot_id
            ):
                raise CostValidationError(
                    "Robot identity mismatch"
                )

            if (
                baseline.request.currency
                != proposed.request.currency
            ):
                raise CostValidationError(
                    "Currencies must match"
                )

            savings = (
                baseline.total_cost
                - proposed.total_cost
            )

            if proposed.total_cost == 0:
                roi = None
            else:
                roi = (
                    savings
                    / proposed.total_cost
                ) * Decimal("100")

            return {
                "tenant_id": tenant_id,
                "robot_id": (
                    baseline.request.robot_id
                ),
                "currency": (
                    baseline.request.currency
                ),
                "baseline_cost": money_string(
                    baseline.total_cost
                ),
                "proposed_cost": money_string(
                    proposed.total_cost
                ),
                "estimated_savings": money_string(
                    savings
                ),
                "cost_reduction_percentage": (
                    str(
                        money(
                            savings
                            / baseline.total_cost
                            * Decimal("100")
                        )
                    )
                    if baseline.total_cost > 0
                    else None
                ),
                "savings_to_proposed_cost_percent": (
                    str(money(roi))
                    if roi is not None
                    else None
                ),
                "note": (
                    "Planning metric, not investment ROI. "
                    "Compare matching time horizons and "
                    "include all relevant costs."
                ),
            }

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(
        self,
        *,
        tenant_id: str,
    ) -> dict[str, Any]:
        estimates = self.list_estimates(
            tenant_id=tenant_id
        )

        currencies = sorted({
            item["currency"]
            for item in estimates
        })

        total_by_currency: dict[
            str,
            Decimal,
        ] = {}

        for item in estimates:
            currency = item["currency"]

            total_by_currency.setdefault(
                currency,
                Decimal("0"),
            )

            total_by_currency[
                currency
            ] += Decimal(
                item["total_cost"]
            )

        return {
            "tenant_id": tenant_id,
            "total_estimates": len(estimates),
            "finalized_estimates": sum(
                item["status"]
                == EstimateStatus.FINALIZED.value
                for item in estimates
            ),
            "currencies": currencies,
            "total_by_currency": {
                currency: money_string(total)
                for currency, total
                in total_by_currency.items()
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
            payload = {
                "generated_at": utc_now().isoformat(),
                "tenant_id": tenant_id,
                "statistics": self.statistics(
                    tenant_id=tenant_id
                ),
                "estimates": self.list_estimates(
                    tenant_id=tenant_id
                ),
                "audit_events": [
                    event.to_dict()
                    for event in self._audit_events
                    if event.tenant_id == tenant_id
                ],
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
    engine = RoboticsCostEstimationEngine()

    tenant_id = "tenant_alpha"
    robot_id = "robot_001"

    repair = engine.create_estimate(
        RobotCostEstimateRequest(
            tenant_id=tenant_id,
            robot_id=robot_id,
            estimate_name="Motor Repair Estimate",
            strategy=MaintenanceStrategy.CORRECTIVE,
            currency="INR",
            items=(
                CostItem(
                    item_id="motor_bearing",
                    category=CostCategory.PARTS,
                    description="Motor bearing replacement",
                    quantity=Decimal("2"),
                    unit_cost=Decimal("2500"),
                ),
                CostItem(
                    item_id="technician_labor",
                    category=CostCategory.LABOR,
                    description="Technician labor",
                    quantity=Decimal("4"),
                    unit_cost=Decimal("800"),
                ),
                CostItem(
                    item_id="inspection",
                    category=CostCategory.INSPECTION,
                    description="Robot inspection",
                    quantity=Decimal("1"),
                    unit_cost=Decimal("1500"),
                ),
            ),
            operating_cost=RobotOperatingCost(
                operating_hours=Decimal("100"),
                energy_consumption_kwh_per_hour=(
                    Decimal("0.8")
                ),
                electricity_price_per_kwh=(
                    Decimal("8")
                ),
                downtime_hours=Decimal("5"),
                downtime_loss_per_hour=(
                    Decimal("2000")
                ),
            ),
            failure_risk=FailureRiskEstimate(
                failure_probability=Decimal("0.10"),
                failure_consequence_cost=(
                    Decimal("15000")
                ),
            ),
            contingency_rate=Decimal("0.10"),
            tax_rate=Decimal("0.18"),
        )
    )

    replacement = engine.create_estimate(
        RobotCostEstimateRequest(
            tenant_id=tenant_id,
            robot_id=robot_id,
            estimate_name="Motor Replacement Estimate",
            strategy=MaintenanceStrategy.REPLACEMENT,
            currency="INR",
            items=(
                CostItem(
                    item_id="new_motor",
                    category=CostCategory.PARTS,
                    description="New robot motor",
                    quantity=Decimal("1"),
                    unit_cost=Decimal("25000"),
                ),
                CostItem(
                    item_id="installation",
                    category=CostCategory.LABOR,
                    description="Installation labor",
                    quantity=Decimal("3"),
                    unit_cost=Decimal("800"),
                ),
            ),
            contingency_rate=Decimal("0.10"),
            tax_rate=Decimal("0.18"),
        )
    )

    comparison = (
        engine.compare_repair_replacement(
            tenant_id=tenant_id,
            robot_id=robot_id,
            repair_estimate_id=repair["estimate_id"],
            replacement_estimate_id=(
                replacement["estimate_id"]
            ),
        )
    )

    print(
        "Repair estimate:",
        json.dumps(
            repair,
            indent=2,
        ),
    )

    print(
        "\nRepair vs replacement:",
        json.dumps(
            comparison,
            indent=2,
        ),
    )

    report = engine.export_json(
        "data/robotics/cost_estimation_report.json",
        tenant_id=tenant_id,
    )

    print("\nReport:", report)


if __name__ == "__main__":
    main()

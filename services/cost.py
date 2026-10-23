
"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
    backend/app/services/cost.py

Robotics Cost Intelligence Service

Features:
    - Robot operating cost calculation
    - Electricity and charging cost
    - Maintenance cost estimation
    - Repair and replacement cost
    - Downtime and production-loss cost
    - Labor cost
    - Spare parts cost
    - Fleet cost aggregation
    - Cost per operating hour
    - Cost per completed task
    - ROI and payback analysis
    - Monthly and annual cost forecasting
    - Simulation cost integration
    - JSON and CSV exports

Python:
    3.11+

Dependencies:
    Standard library only

All monetary values are expressed in the
configured currency.

Default currency:
    INR

Safety:
    Financial estimation only.
    No payments or hardware operations.
"""

from __future__ import annotations

import csv
import json
import math
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class CostServiceError(Exception):
    pass


class CostValidationError(CostServiceError):
    pass


# ============================================================
# ENUMS
# ============================================================

class CostCategory(str, Enum):
    ENERGY = "energy"
    MAINTENANCE = "maintenance"
    REPAIR = "repair"
    REPLACEMENT = "replacement"
    LABOR = "labor"
    SPARE_PARTS = "spare_parts"
    DOWNTIME = "downtime"
    SOFTWARE = "software"
    INFRASTRUCTURE = "infrastructure"
    OTHER = "other"


class CostPeriod(str, Enum):
    DAILY = "daily"
    MONTHLY = "monthly"
    ANNUAL = "annual"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def number(
    value: float,
    name: str,
    *,
    minimum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise CostValidationError(
            f"{name} must be finite"
        )

    result = float(value)

    if (
        minimum is not None
        and result < minimum
    ):
        raise CostValidationError(
            f"{name} must be >= {minimum}"
        )

    return result


def money(
    value: float | Decimal,
) -> Decimal:
    amount = Decimal(str(value))

    if not amount.is_finite():
        raise CostValidationError(
            "Monetary amount must be finite"
        )

    return amount.quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class CostServiceConfig:
    currency: str = "INR"

    electricity_rate_per_kwh: float = 8.0

    labor_rate_per_hour: float = 350.0

    downtime_loss_per_hour: float = 1500.0

    maintenance_cost_per_hour: float = 20.0

    software_cost_per_robot_month: float = 500.0

    infrastructure_cost_per_robot_month: float = 300.0

    working_days_per_month: int = 26

    operating_hours_per_day: float = 8.0

    annual_discount_rate: float = 0.10

    allow_hardware_control: bool = False

    def __post_init__(self) -> None:
        if not self.currency.strip():
            raise CostValidationError(
                "Currency is required"
            )

        for name in (
            "electricity_rate_per_kwh",
            "labor_rate_per_hour",
            "downtime_loss_per_hour",
            "maintenance_cost_per_hour",
            "software_cost_per_robot_month",
            "infrastructure_cost_per_robot_month",
            "annual_discount_rate",
        ):
            number(
                getattr(self, name),
                name,
                minimum=0.0,
            )

        if self.working_days_per_month < 1:
            raise CostValidationError(
                "working_days_per_month must be positive"
            )

        number(
            self.operating_hours_per_day,
            "operating_hours_per_day",
            minimum=0.0,
        )

        if self.allow_hardware_control:
            raise CostValidationError(
                "Hardware control is prohibited"
            )


# ============================================================
# COST ITEM
# ============================================================

@dataclass(frozen=True)
class CostItem:
    category: CostCategory
    description: str
    amount: Decimal

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "description": self.description,
            "amount": str(
                money(self.amount)
            ),
        }


# ============================================================
# ROBOT COST INPUT
# ============================================================

@dataclass(frozen=True)
class RobotCostInput:
    robot_id: str

    operating_hours: float

    energy_consumed_kwh: float

    completed_tasks: int = 0

    maintenance_hours: float = 0.0
    repair_hours: float = 0.0

    downtime_hours: float = 0.0

    spare_parts_cost: float = 0.0
    replacement_cost: float = 0.0

    software_months: float = 0.0
    infrastructure_months: float = 0.0

    additional_cost: float = 0.0

    def __post_init__(self) -> None:
        if not self.robot_id.strip():
            raise CostValidationError(
                "robot_id is required"
            )

        for name in (
            "operating_hours",
            "energy_consumed_kwh",
            "maintenance_hours",
            "repair_hours",
            "downtime_hours",
            "spare_parts_cost",
            "replacement_cost",
            "software_months",
            "infrastructure_months",
            "additional_cost",
        ):
            number(
                getattr(self, name),
                name,
                minimum=0.0,
            )

        if self.completed_tasks < 0:
            raise CostValidationError(
                "completed_tasks cannot be negative"
            )


# ============================================================
# COST RESULT
# ============================================================

@dataclass
class RobotCostResult:
    robot_id: str
    currency: str

    total_cost: Decimal

    cost_per_operating_hour: Decimal | None
    cost_per_completed_task: Decimal | None

    cost_items: list[CostItem]

    calculated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "currency": self.currency,
            "total_cost": str(
                money(self.total_cost)
            ),
            "cost_per_operating_hour": (
                str(self.cost_per_operating_hour)
                if self.cost_per_operating_hour is not None
                else None
            ),
            "cost_per_completed_task": (
                str(self.cost_per_completed_task)
                if self.cost_per_completed_task is not None
                else None
            ),
            "cost_items": [
                item.to_dict()
                for item in self.cost_items
            ],
            "calculated_at": self.calculated_at,
        }


# ============================================================
# COST CALCULATOR
# ============================================================

class RoboticsCostCalculator:

    def __init__(
        self,
        config: CostServiceConfig,
    ) -> None:
        self.config = config

    def calculate(
        self,
        data: RobotCostInput,
    ) -> RobotCostResult:
        items = []

        def add(
            category: CostCategory,
            description: str,
            amount: float,
        ) -> None:
            items.append(
                CostItem(
                    category=category,
                    description=description,
                    amount=money(amount),
                )
            )

        add(
            CostCategory.ENERGY,
            "Electricity consumption",
            (
                data.energy_consumed_kwh
                * self.config.electricity_rate_per_kwh
            ),
        )

        add(
            CostCategory.MAINTENANCE,
            "Scheduled maintenance",
            (
                data.maintenance_hours
                * self.config.maintenance_cost_per_hour
            ),
        )

        add(
            CostCategory.LABOR,
            "Maintenance and repair labor",
            (
                (
                    data.maintenance_hours
                    + data.repair_hours
                )
                * self.config.labor_rate_per_hour
            ),
        )

        add(
            CostCategory.DOWNTIME,
            "Robot downtime loss",
            (
                data.downtime_hours
                * self.config.downtime_loss_per_hour
            ),
        )

        add(
            CostCategory.SPARE_PARTS,
            "Spare parts",
            data.spare_parts_cost,
        )

        add(
            CostCategory.REPLACEMENT,
            "Replacement components",
            data.replacement_cost,
        )

        add(
            CostCategory.SOFTWARE,
            "Software subscription",
            (
                data.software_months
                * self.config.software_cost_per_robot_month
            ),
        )

        add(
            CostCategory.INFRASTRUCTURE,
            "Infrastructure allocation",
            (
                data.infrastructure_months
                * self.config.infrastructure_cost_per_robot_month
            ),
        )

        add(
            CostCategory.OTHER,
            "Additional expenses",
            data.additional_cost,
        )

        total = sum(
            (
                item.amount
                for item in items
            ),
            Decimal("0.00"),
        )

        cost_per_hour = (
            money(
                total
                / Decimal(
                    str(data.operating_hours)
                )
            )
            if data.operating_hours > 0
            else None
        )

        cost_per_task = (
            money(
                total
                / Decimal(
                    data.completed_tasks
                )
            )
            if data.completed_tasks > 0
            else None
        )

        return RobotCostResult(
            robot_id=data.robot_id,
            currency=self.config.currency,
            total_cost=money(total),
            cost_per_operating_hour=cost_per_hour,
            cost_per_completed_task=cost_per_task,
            cost_items=items,
            calculated_at=utc_now(),
        )


# ============================================================
# FLEET COST RESULT
# ============================================================

@dataclass
class FleetCostResult:
    report_id: str
    currency: str

    robot_count: int
    total_cost: Decimal

    category_totals: dict[str, Decimal]

    robot_results: list[RobotCostResult]

    calculated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "currency": self.currency,
            "robot_count": self.robot_count,
            "total_cost": str(
                money(self.total_cost)
            ),
            "category_totals": {
                category: str(
                    money(amount)
                )
                for category, amount
                in self.category_totals.items()
            },
            "robot_results": [
                result.to_dict()
                for result in self.robot_results
            ],
            "calculated_at": self.calculated_at,
        }


# ============================================================
# FLEET COST CALCULATOR
# ============================================================

class RoboticsFleetCostCalculator:

    def __init__(
        self,
        config: CostServiceConfig,
    ) -> None:
        self.config = config

        self.robot_calculator = (
            RoboticsCostCalculator(
                config
            )
        )

    def calculate(
        self,
        robots: Sequence[RobotCostInput],
    ) -> FleetCostResult:
        results = [
            self.robot_calculator.calculate(
                robot
            )
            for robot in robots
        ]

        category_totals: dict[
            str,
            Decimal
        ] = {}

        for result in results:
            for item in result.cost_items:
                key = item.category.value

                category_totals[key] = (
                    category_totals.get(
                        key,
                        Decimal("0.00"),
                    )
                    + item.amount
                )

        total_cost = sum(
            (
                result.total_cost
                for result in results
            ),
            Decimal("0.00"),
        )

        return FleetCostResult(
            report_id=str(uuid.uuid4()),
            currency=self.config.currency,
            robot_count=len(results),
            total_cost=money(total_cost),
            category_totals=category_totals,
            robot_results=results,
            calculated_at=utc_now(),
        )


# ============================================================
# COST FORECAST
# ============================================================

@dataclass(frozen=True)
class CostForecastResult:
    robot_id: str
    currency: str

    daily_cost: Decimal
    monthly_cost: Decimal
    annual_cost: Decimal

    assumptions: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "currency": self.currency,
            "daily_cost": str(self.daily_cost),
            "monthly_cost": str(self.monthly_cost),
            "annual_cost": str(self.annual_cost),
            "assumptions": self.assumptions,
        }


class RoboticsCostForecaster:

    def __init__(
        self,
        config: CostServiceConfig,
    ) -> None:
        self.config = config

    def forecast(
        self,
        result: RobotCostResult,
        *,
        observed_operating_hours: float,
    ) -> CostForecastResult:
        hours = number(
            observed_operating_hours,
            "observed_operating_hours",
            minimum=0.0,
        )

        if hours <= 0:
            raise CostValidationError(
                "Observed operating hours must be positive"
            )

        cost_per_hour = (
            result.total_cost
            / Decimal(str(hours))
        )

        daily = money(
            cost_per_hour
            * Decimal(
                str(
                    self.config.operating_hours_per_day
                )
            )
        )

        monthly = money(
            daily
            * Decimal(
                self.config.working_days_per_month
            )
        )

        annual = money(
            monthly * Decimal("12")
        )

        return CostForecastResult(
            robot_id=result.robot_id,
            currency=result.currency,
            daily_cost=daily,
            monthly_cost=monthly,
            annual_cost=annual,
            assumptions={
                "observed_operating_hours": hours,
                "working_days_per_month": (
                    self.config.working_days_per_month
                ),
                "operating_hours_per_day": (
                    self.config.operating_hours_per_day
                ),
                "annual_months": 12,
                "method": (
                    "Linear extrapolation from observed cost"
                ),
            },
        )


# ============================================================
# ROI CALCULATOR
# ============================================================

@dataclass(frozen=True)
class RoboticsROIInput:
    initial_investment: float
    annual_savings: float
    annual_operating_cost: float
    project_lifetime_years: int = 5


@dataclass(frozen=True)
class RoboticsROIResult:
    currency: str
    roi_percent: float
    payback_years: float | None
    net_present_value: Decimal
    total_net_benefit: Decimal

    def to_dict(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "roi_percent": self.roi_percent,
            "payback_years": self.payback_years,
            "net_present_value": str(
                self.net_present_value
            ),
            "total_net_benefit": str(
                self.total_net_benefit
            ),
        }


class RoboticsROICalculator:

    def __init__(
        self,
        config: CostServiceConfig,
    ) -> None:
        self.config = config

    def calculate(
        self,
        data: RoboticsROIInput,
    ) -> RoboticsROIResult:
        for name in (
            "initial_investment",
            "annual_savings",
            "annual_operating_cost",
        ):
            number(
                getattr(data, name),
                name,
                minimum=0.0,
            )

        if data.project_lifetime_years < 1:
            raise CostValidationError(
                "Project lifetime must be positive"
            )

        annual_net_savings = (
            data.annual_savings
            - data.annual_operating_cost
        )

        total_net_benefit = (
            annual_net_savings
            * data.project_lifetime_years
            - data.initial_investment
        )

        if data.initial_investment > 0:
            roi_percent = (
                total_net_benefit
                / data.initial_investment
                * 100.0
            )
        else:
            roi_percent = 0.0

        payback_years = (
            data.initial_investment
            / annual_net_savings
            if annual_net_savings > 0
            else None
        )

        npv = -data.initial_investment

        for year in range(
            1,
            data.project_lifetime_years + 1
        ):
            npv += (
                annual_net_savings
                / (
                    1.0
                    + self.config.annual_discount_rate
                ) ** year
            )

        return RoboticsROIResult(
            currency=self.config.currency,
            roi_percent=roi_percent,
            payback_years=payback_years,
            net_present_value=money(npv),
            total_net_benefit=money(
                total_net_benefit
            ),
        )


# ============================================================
# SIMULATION COST ADAPTER
# ============================================================

class SimulationCostAdapter:

    @staticmethod
    def extract(
        physics: Any,
        *,
        completed_tasks: Mapping[
            str,
            int
        ] | None = None,
        maintenance_hours: Mapping[
            str,
            float
        ] | None = None,
        downtime_hours: Mapping[
            str,
            float
        ] | None = None,
    ) -> list[RobotCostInput]:
        completed_tasks = (
            completed_tasks or {}
        )

        maintenance_hours = (
            maintenance_hours or {}
        )

        downtime_hours = (
            downtime_hours or {}
        )

        operating_hours = (
            physics.simulation_time_seconds
            / 3600.0
        )

        results = []

        for robot_id, robot in physics.robots.items():
            results.append(
                RobotCostInput(
                    robot_id=robot_id,
                    operating_hours=operating_hours,
                    energy_consumed_kwh=(
                        robot.energy_consumed_wh
                        / 1000.0
                    ),
                    completed_tasks=(
                        completed_tasks.get(
                            robot_id,
                            0,
                        )
                    ),
                    maintenance_hours=(
                        maintenance_hours.get(
                            robot_id,
                            0.0,
                        )
                    ),
                    downtime_hours=(
                        downtime_hours.get(
                            robot_id,
                            0.0,
                        )
                    ),
                )
            )

        return results


# ============================================================
# REPORT EXPORTER
# ============================================================

class RoboticsCostReportExporter:

    @staticmethod
    def export_json(
        report: FleetCostResult,
        path: str | Path,
    ) -> Path:
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.write_text(
            json.dumps(
                report.to_dict(),
                indent=2,
            ),
            encoding="utf-8",
        )

        return output

    @staticmethod
    def export_csv(
        report: FleetCostResult,
        path: str | Path,
    ) -> Path:
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "robot_id",
                    "currency",
                    "category",
                    "description",
                    "amount",
                ],
            )

            writer.writeheader()

            for result in report.robot_results:
                for item in result.cost_items:
                    writer.writerow({
                        "robot_id": result.robot_id,
                        "currency": result.currency,
                        "category": item.category.value,
                        "description": item.description,
                        "amount": str(item.amount),
                    })

        return output


# ============================================================
# UNIFIED COST SERVICE
# ============================================================

class RoboticsCostService:

    def __init__(
        self,
        config: CostServiceConfig | None = None,
    ) -> None:
        self.config = (
            config or CostServiceConfig()
        )

        self.robot_calculator = (
            RoboticsCostCalculator(
                self.config
            )
        )

        self.fleet_calculator = (
            RoboticsFleetCostCalculator(
                self.config
            )
        )

        self.forecaster = RoboticsCostForecaster(
            self.config
        )

        self.roi_calculator = RoboticsROICalculator(
            self.config
        )

    def robot_cost(
        self,
        data: RobotCostInput,
    ) -> RobotCostResult:
        return self.robot_calculator.calculate(
            data
        )

    def fleet_cost(
        self,
        robots: Sequence[RobotCostInput],
    ) -> FleetCostResult:
        return self.fleet_calculator.calculate(
            robots
        )

    def forecast(
        self,
        result: RobotCostResult,
        *,
        observed_operating_hours: float,
    ) -> CostForecastResult:
        return self.forecaster.forecast(
            result,
            observed_operating_hours=(
                observed_operating_hours
            ),
        )

    def roi(
        self,
        data: RoboticsROIInput,
    ) -> RoboticsROIResult:
        return self.roi_calculator.calculate(
            data
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    service = RoboticsCostService()

    robots = [
        RobotCostInput(
            robot_id="robot_001",
            operating_hours=200.0,
            energy_consumed_kwh=120.0,
            completed_tasks=500,
            maintenance_hours=4.0,
            repair_hours=2.0,
            downtime_hours=3.0,
            spare_parts_cost=2500.0,
            software_months=1.0,
            infrastructure_months=1.0,
        ),
        RobotCostInput(
            robot_id="robot_002",
            operating_hours=180.0,
            energy_consumed_kwh=110.0,
            completed_tasks=420,
            maintenance_hours=3.0,
            repair_hours=1.0,
            downtime_hours=2.0,
            spare_parts_cost=1500.0,
            software_months=1.0,
            infrastructure_months=1.0,
        ),
    ]

    fleet_report = service.fleet_cost(
        robots
    )

    forecast = service.forecast(
        fleet_report.robot_results[0],
        observed_operating_hours=200.0,
    )

    roi = service.roi(
        RoboticsROIInput(
            initial_investment=1_500_000.0,
            annual_savings=800_000.0,
            annual_operating_cost=250_000.0,
            project_lifetime_years=5,
        )
    )

    RoboticsCostReportExporter.export_json(
        fleet_report,
        "data/cost/fleet_cost_report.json",
    )

    RoboticsCostReportExporter.export_csv(
        fleet_report,
        "data/cost/fleet_cost_breakdown.csv",
    )

    print(
        json.dumps(
            {
                "fleet": fleet_report.to_dict(),
                "forecast": forecast.to_dict(),
                "roi": roi.to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

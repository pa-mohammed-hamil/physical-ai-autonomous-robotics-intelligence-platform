
#!/usr/bin/env python3
"""
Physical AI & Autonomous Robotics Intelligence Platform
File: scripts/seed_prices.py

Synthetic robot spare-parts pricing database seeder.

Compatible with:
    scripts/seed_parts.py
    scripts/seed_database.py

Features:
- Historical part prices
- Supplier price quotations
- Price trend simulation
- Repair and replacement cost estimates
- Price forecasting scenarios
- Low-stock price impact
- Cost comparison and alerts
- SQLite / PostgreSQL
- Deterministic and idempotent seeding
- Transactional writes
- Dry-run and reset options
- JSON reporting

Dependencies:
    pip install sqlalchemy

Usage:
    python scripts/seed_prices.py

    python scripts/seed_prices.py --days 180

    python scripts/seed_prices.py --seed 42

    python scripts/seed_prices.py --dry-run

    python scripts/seed_prices.py --reset --confirm-reset

    python scripts/seed_prices.py \
        --output reports/pricing_seed.json

Prerequisites:
    python scripts/seed_database.py
    python scripts/seed_parts.py

Safety:
- Synthetic data only.
- No real supplier quotations.
- No purchases or financial transactions.
- No physical robot actuation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import uuid

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    func,
    inspect,
    select,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)

try:
    from seed_parts import (
        PartCatalog,
        PartSupplier,
        SparePartInventory,
    )
except ImportError:
    from scripts.seed_parts import (
        PartCatalog,
        PartSupplier,
        SparePartInventory,
    )


LOGGER = logging.getLogger("seed_prices")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

SCHEMA_VERSION = "1.0.0"

DEFAULT_DATABASE_URL = (
    "sqlite:///storage/robotics_demo.db"
)

NAMESPACE = uuid.UUID(
    "bd1f74b4-7081-4f6b-91a4-6c31267a2091"
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class PriceSeedConfig:
    database_url: str = DEFAULT_DATABASE_URL

    days: int = 180
    seed: int = 42

    supplier_quotes_per_part: int = 3
    forecast_days: int = 30

    volatility: float = 0.012
    annual_inflation: float = 0.045

    currency: str = "INR"

    dry_run: bool = False
    reset: bool = False
    confirm_reset: bool = False

    output: str | None = None

    def validate(self) -> None:
        if self.days < 1 or self.days > 3650:
            raise ValueError(
                "--days must be between 1 and 3650"
            )

        if self.forecast_days < 1 or self.forecast_days > 365:
            raise ValueError(
                "--forecast-days must be between 1 and 365"
            )

        if self.supplier_quotes_per_part < 1:
            raise ValueError(
                "--quotes-per-part must be positive"
            )

        if not math.isfinite(self.volatility) or not (
            0 <= self.volatility <= 0.5
        ):
            raise ValueError(
                "--volatility must be between 0 and 0.5"
            )

        if not math.isfinite(self.annual_inflation) or not (
            -0.5 <= self.annual_inflation <= 1.0
        ):
            raise ValueError(
                "--annual-inflation must be between -0.5 and 1"
            )

        if not self.currency.strip():
            raise ValueError("Currency is required")

        if self.reset and not self.confirm_reset:
            raise ValueError(
                "--reset requires --confirm-reset"
            )

        if self.dry_run and self.reset:
            raise ValueError(
                "--dry-run cannot be combined with --reset"
            )


# ============================================================
# SQLAlchemy Models
# ============================================================

class PriceBase(DeclarativeBase):
    pass


class PartPriceHistory(PriceBase):
    __tablename__ = "demo_part_price_history"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    catalog_part_id: Mapped[str] = mapped_column(
        ForeignKey("demo_part_catalog.id"),
        nullable=False,
        index=True,
    )

    price_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    unit_price: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    currency: Mapped[str] = mapped_column(
        String(10),
        default="INR",
    )

    daily_change_percent: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    market_condition: Mapped[str] = mapped_column(
        String(50),
        default="stable",
    )

    synthetic: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
    )

    __table_args__ = (
        UniqueConstraint(
            "catalog_part_id",
            "price_date",
            name="uq_demo_part_price_date",
        ),
    )


class SupplierPriceQuote(PriceBase):
    __tablename__ = "demo_supplier_price_quotes"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    catalog_part_id: Mapped[str] = mapped_column(
        ForeignKey("demo_part_catalog.id"),
        nullable=False,
    )

    supplier_id: Mapped[str] = mapped_column(
        ForeignKey("demo_part_suppliers.id"),
        nullable=False,
    )

    quote_reference: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
    )

    quoted_unit_price: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    shipping_cost: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    tax_estimate: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    total_landed_cost: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    lead_time_days: Mapped[int] = mapped_column(
        Integer,
        default=7,
    )

    minimum_order_quantity: Mapped[int] = mapped_column(
        Integer,
        default=1,
    )

    currency: Mapped[str] = mapped_column(
        String(10),
        default="INR",
    )

    quote_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    valid_until: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    synthetic: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
    )


class PartPriceForecast(PriceBase):
    __tablename__ = "demo_part_price_forecasts"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    catalog_part_id: Mapped[str] = mapped_column(
        ForeignKey("demo_part_catalog.id"),
        nullable=False,
    )

    forecast_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    predicted_price: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    lower_bound: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    upper_bound: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    model_name: Mapped[str] = mapped_column(
        String(100),
        default="synthetic_trend_simulator",
    )

    currency: Mapped[str] = mapped_column(
        String(10),
        default="INR",
    )

    synthetic: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
    )

    __table_args__ = (
        UniqueConstraint(
            "catalog_part_id",
            "forecast_date",
            name="uq_demo_part_forecast_date",
        ),
    )


class PartPriceAlert(PriceBase):
    __tablename__ = "demo_part_price_alerts"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    catalog_part_id: Mapped[str] = mapped_column(
        ForeignKey("demo_part_catalog.id"),
        nullable=False,
    )

    alert_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    severity: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )

    title: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    current_price: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    reference_price: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    change_percent: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    acknowledged: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    synthetic: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
    )

    __table_args__ = (
        UniqueConstraint(
            "catalog_part_id",
            "alert_type",
            name="uq_demo_part_price_alert_type",
        ),
    )


class PartCostScenario(PriceBase):
    __tablename__ = "demo_part_cost_scenarios"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    catalog_part_id: Mapped[str] = mapped_column(
        ForeignKey("demo_part_catalog.id"),
        nullable=False,
    )

    scenario_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    repair_cost: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    replacement_cost: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    labor_cost: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    downtime_cost: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    total_repair_cost: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    total_replacement_cost: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    recommended_option: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    currency: Mapped[str] = mapped_column(
        String(10),
        default="INR",
    )

    human_approval_required: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
    )

    __table_args__ = (
        UniqueConstraint(
            "catalog_part_id",
            "scenario_name",
            name="uq_demo_part_cost_scenario",
        ),
    )


# ============================================================
# Helpers
# ============================================================

def stable_id(namespace: str, key: str) -> str:
    return str(
        uuid.uuid5(
            NAMESPACE,
            f"{namespace}:{key}",
        )
    )


def money(value: float) -> float:
    return round(max(0.0, value), 2)


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(minimum, min(maximum, value))


# ============================================================
# Seeder
# ============================================================

class PriceSeeder:
    def __init__(
        self,
        session: Session,
        config: PriceSeedConfig,
    ) -> None:
        self.session = session
        self.config = config
        self.rng = random.Random(config.seed)
        self.counts = Counter()

        # Fixed simulation reference date ensures
        # reproducible results across runs.
        self.reference_date = datetime(
            2026, 10, 16,
            tzinfo=timezone.utc,
        )

    def upsert(
        self,
        model_class: type,
        identifier: str,
        values: dict[str, Any],
    ) -> Any:
        existing = self.session.get(
            model_class,
            identifier,
        )

        if existing is None:
            existing = model_class(
                id=identifier,
                **values,
            )
            self.session.add(existing)
            self.counts["inserted"] += 1
        else:
            for key, value in values.items():
                setattr(existing, key, value)
            self.counts["updated"] += 1

        self.counts[model_class.__tablename__] += 1
        return existing

    def load_catalog(self) -> list[PartCatalog]:
        return list(
            self.session.scalars(
                select(PartCatalog).order_by(
                    PartCatalog.part_number
                )
            ).all()
        )

    def load_suppliers(self) -> list[PartSupplier]:
        return list(
            self.session.scalars(
                select(PartSupplier).order_by(
                    PartSupplier.supplier_code
                )
            ).all()
        )

    def load_inventory(self) -> list[SparePartInventory]:
        return list(
            self.session.scalars(
                select(SparePartInventory)
            ).all()
        )

    # --------------------------------------------------------
    # Historical Prices
    # --------------------------------------------------------

    def seed_history(
        self,
        catalog: list[PartCatalog],
    ) -> dict[str, float]:
        latest_prices = {}

        start_date = (
            self.reference_date
            - timedelta(days=self.config.days - 1)
        )

        for part in catalog:
            base_price = float(part.unit_cost)
            price = base_price

            for day_index in range(self.config.days):
                price_date = (
                    start_date
                    + timedelta(days=day_index)
                )

                daily_drift = (
                    self.config.annual_inflation / 365.0
                )

                daily_noise = self.rng.gauss(
                    0,
                    self.config.volatility,
                )

                previous_price = price

                price = max(
                    base_price * 0.35,
                    price * (
                        1 + daily_drift + daily_noise
                    ),
                )

                change_percent = (
                    (price - previous_price)
                    / previous_price
                    * 100.0
                )

                if change_percent >= 2:
                    condition = "rising"
                elif change_percent <= -2:
                    condition = "falling"
                else:
                    condition = "stable"

                date_key = price_date.date().isoformat()

                self.upsert(
                    PartPriceHistory,
                    stable_id(
                        "history",
                        f"{part.id}:{date_key}",
                    ),
                    {
                        "catalog_part_id": part.id,
                        "price_date": price_date,
                        "unit_price": money(price),
                        "currency": self.config.currency,
                        "daily_change_percent": round(
                            change_percent, 4
                        ),
                        "market_condition": condition,
                        "synthetic": True,
                    },
                )

            latest_prices[part.id] = money(price)

        self.session.flush()
        return latest_prices

    # --------------------------------------------------------
    # Supplier Quotations
    # --------------------------------------------------------

    def seed_quotes(
        self,
        catalog: list[PartCatalog],
        suppliers: list[PartSupplier],
        latest_prices: dict[str, float],
    ) -> None:
        for part in catalog:
            for index in range(
                self.config.supplier_quotes_per_part
            ):
                supplier = suppliers[
                    index % len(suppliers)
                ]

                quoted_price = money(
                    latest_prices[part.id]
                    * self.rng.uniform(0.92, 1.15)
                )

                shipping = money(
                    self.rng.uniform(250, 2500)
                )

                # Illustrative tax only; not a tax calculation.
                tax = money(
                    quoted_price
                    * self.rng.uniform(0.05, 0.18)
                )

                landed_cost = money(
                    quoted_price + shipping + tax
                )

                quote_reference = (
                    f"DEMO-QUOTE-"
                    f"{part.part_number}-"
                    f"{supplier.supplier_code}-"
                    f"{index:02d}"
                )

                self.upsert(
                    SupplierPriceQuote,
                    stable_id(
                        "quote",
                        quote_reference,
                    ),
                    {
                        "catalog_part_id": part.id,
                        "supplier_id": supplier.id,
                        "quote_reference": quote_reference,
                        "quoted_unit_price": quoted_price,
                        "shipping_cost": shipping,
                        "tax_estimate": tax,
                        "total_landed_cost": landed_cost,
                        "lead_time_days": (
                            supplier.lead_time_days
                        ),
                        "minimum_order_quantity": (
                            self.rng.randint(1, 5)
                        ),
                        "currency": self.config.currency,
                        "quote_date": self.reference_date,
                        "valid_until": (
                            self.reference_date
                            + timedelta(days=30)
                        ),
                        "synthetic": True,
                    },
                )

        self.session.flush()

    # --------------------------------------------------------
    # Price Forecast
    # --------------------------------------------------------

    def seed_forecasts(
        self,
        catalog: list[PartCatalog],
        latest_prices: dict[str, float],
    ) -> None:
        for part in catalog:
            starting_price = latest_prices[part.id]

            for day in range(
                1,
                self.config.forecast_days + 1,
            ):
                forecast_date = (
                    self.reference_date
                    + timedelta(days=day)
                )

                trend = (
                    self.config.annual_inflation
                    * day / 365.0
                )

                predicted_price = money(
                    starting_price * (1 + trend)
                )

                # These are illustrative scenario bands,
                # not statistically calibrated intervals.
                uncertainty = (
                    self.config.volatility
                    * math.sqrt(day)
                    * 1.96
                )

                lower = money(
                    predicted_price * max(
                        0.0,
                        1 - uncertainty,
                    )
                )

                upper = money(
                    predicted_price * (1 + uncertainty)
                )

                self.upsert(
                    PartPriceForecast,
                    stable_id(
                        "forecast",
                        f"{part.id}:{forecast_date.date()}",
                    ),
                    {
                        "catalog_part_id": part.id,
                        "forecast_date": forecast_date,
                        "predicted_price": predicted_price,
                        "lower_bound": lower,
                        "upper_bound": upper,
                        "model_name": (
                            "synthetic_trend_simulator"
                        ),
                        "currency": self.config.currency,
                        "synthetic": True,
                    },
                )

        self.session.flush()

    # --------------------------------------------------------
    # Repair vs Replacement Scenarios
    # --------------------------------------------------------

    def seed_cost_scenarios(
        self,
        catalog: list[PartCatalog],
        latest_prices: dict[str, float],
    ) -> None:
        scenarios = {
            "normal": 1.0,
            "high_labor": 1.5,
            "urgent_repair": 1.8,
            "supply_disruption": 2.0,
        }

        for part in catalog:
            current_price = latest_prices[part.id]

            for scenario, multiplier in scenarios.items():
                repair_cost = money(
                    current_price
                    * self.rng.uniform(0.2, 0.65)
                )

                replacement_cost = money(
                    current_price
                    * (
                        multiplier
                        if scenario == "supply_disruption"
                        else 1.0
                    )
                )

                labor_cost = money(
                    self.rng.uniform(1000, 6500)
                    * (
                        multiplier
                        if scenario != "supply_disruption"
                        else 1.0
                    )
                )

                downtime_cost = money(
                    self.rng.uniform(1500, 12000)
                    * multiplier
                )

                total_repair = money(
                    repair_cost
                    + labor_cost
                    + downtime_cost
                )

                total_replacement = money(
                    replacement_cost
                    + labor_cost
                    + downtime_cost
                )

                recommendation = (
                    "repair"
                    if total_repair < total_replacement
                    else "replace"
                )

                self.upsert(
                    PartCostScenario,
                    stable_id(
                        "cost_scenario",
                        f"{part.id}:{scenario}",
                    ),
                    {
                        "catalog_part_id": part.id,
                        "scenario_name": scenario,
                        "repair_cost": repair_cost,
                        "replacement_cost": replacement_cost,
                        "labor_cost": labor_cost,
                        "downtime_cost": downtime_cost,
                        "total_repair_cost": total_repair,
                        "total_replacement_cost": (
                            total_replacement
                        ),
                        "recommended_option": recommendation,
                        "currency": self.config.currency,
                        "human_approval_required": True,
                    },
                )

        self.session.flush()

    # --------------------------------------------------------
    # Price Alerts
    # --------------------------------------------------------

    def seed_alerts(
        self,
        catalog: list[PartCatalog],
        latest_prices: dict[str, float],
        inventory: list[SparePartInventory],
    ) -> None:
        low_stock_ids = {
            item.catalog_part_id
            for item in inventory
            if item.low_stock
        }

        for part in catalog:
            current_price = latest_prices[part.id]
            reference_price = float(part.unit_cost)

            change_percent = (
                (current_price - reference_price)
                / reference_price * 100.0
                if reference_price > 0
                else 0.0
            )

            alerts = []

            if change_percent >= 10:
                alerts.append((
                    "price_increase",
                    "high",
                    "Synthetic Price Increase",
                    "Demonstration part price increased "
                    "relative to its catalog baseline.",
                ))

            if change_percent <= -10:
                alerts.append((
                    "price_decrease",
                    "info",
                    "Synthetic Price Decrease",
                    "Demonstration part price decreased "
                    "relative to its catalog baseline.",
                ))

            if part.id in low_stock_ids:
                alerts.append((
                    "low_stock_price_risk",
                    "warning",
                    "Low Stock Pricing Risk",
                    "Synthetic inventory is low; review "
                    "potential replacement-cost exposure.",
                ))

            active_types = {
                item[0] for item in alerts
            }

            # Remove stale alerts owned by this seeder.
            for alert_type in (
                "price_increase",
                "price_decrease",
                "low_stock_price_risk",
            ):
                if alert_type not in active_types:
                    existing = self.session.get(
                        PartPriceAlert,
                        stable_id(
                            "alert",
                            f"{part.id}:{alert_type}",
                        ),
                    )
                    if existing is not None:
                        self.session.delete(existing)

            for (
                alert_type,
                severity,
                title,
                message,
            ) in alerts:
                self.upsert(
                    PartPriceAlert,
                    stable_id(
                        "alert",
                        f"{part.id}:{alert_type}",
                    ),
                    {
                        "catalog_part_id": part.id,
                        "alert_type": alert_type,
                        "severity": severity,
                        "title": title,
                        "message": message,
                        "current_price": current_price,
                        "reference_price": reference_price,
                        "change_percent": round(
                            change_percent, 4
                        ),
                        "acknowledged": False,
                        "created_at": self.reference_date,
                        "synthetic": True,
                    },
                )

        self.session.flush()

    # --------------------------------------------------------
    # Seed All
    # --------------------------------------------------------

    def seed_all(self) -> dict[str, Any]:
        catalog = self.load_catalog()
        suppliers = self.load_suppliers()
        inventory = self.load_inventory()

        if not catalog:
            raise RuntimeError(
                "No demo parts found. "
                "Run scripts/seed_parts.py first."
            )

        if not suppliers:
            raise RuntimeError(
                "No demo suppliers found. "
                "Run scripts/seed_parts.py first."
            )

        latest_prices = self.seed_history(catalog)

        self.seed_quotes(
            catalog,
            suppliers,
            latest_prices,
        )

        self.seed_forecasts(
            catalog,
            latest_prices,
        )

        self.seed_cost_scenarios(
            catalog,
            latest_prices,
        )

        self.seed_alerts(
            catalog,
            latest_prices,
            inventory,
        )

        return {
            "catalog_parts": len(catalog),
            "suppliers": len(suppliers),
            "inventory_records": len(inventory),
            "history_days": self.config.days,
            "forecast_days": self.config.forecast_days,
            "supplier_quotes_per_part": (
                self.config.supplier_quotes_per_part
            ),
            "statistics": dict(self.counts),
            "latest_synthetic_prices": {
                part.part_number: latest_prices[part.id]
                for part in catalog
            },
        }


# ============================================================
# Reset Demo Pricing Tables
# ============================================================

def reset_price_tables(session: Session) -> None:
    for model_class in (
        PartPriceAlert,
        PartCostScenario,
        PartPriceForecast,
        SupplierPriceQuote,
        PartPriceHistory,
    ):
        session.execute(delete(model_class))

    session.flush()


# ============================================================
# Statistics
# ============================================================

def collect_statistics(
    session: Session,
) -> dict[str, int]:
    models = [
        PartPriceHistory,
        SupplierPriceQuote,
        PartPriceForecast,
        PartPriceAlert,
        PartCostScenario,
    ]

    return {
        model.__tablename__: int(
            session.scalar(
                select(func.count()).select_from(model)
            ) or 0
        )
        for model in models
    }


# ============================================================
# JSON Report
# ============================================================

def save_report(
    path: str | Path,
    report: dict[str, Any],
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )

    try:
        temporary.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        os.replace(temporary, path)

    finally:
        if temporary.exists():
            temporary.unlink()


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Seed synthetic robot spare-part pricing data"
        )
    )

    parser.add_argument(
        "--database-url",
        default=os.getenv(
            "DATABASE_URL",
            DEFAULT_DATABASE_URL,
        ),
    )

    parser.add_argument(
        "--days",
        type=int,
        default=180,
    )

    parser.add_argument(
        "--forecast-days",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--quotes-per-part",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--volatility",
        type=float,
        default=0.012,
    )

    parser.add_argument(
        "--annual-inflation",
        type=float,
        default=0.045,
    )

    parser.add_argument(
        "--currency",
        default="INR",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
    )

    parser.add_argument(
        "--confirm-reset",
        action="store_true",
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    return parser


# ============================================================
# Main
# ============================================================

def main() -> int:
    args = build_parser().parse_args()

    config = PriceSeedConfig(
        database_url=args.database_url,
        days=args.days,
        forecast_days=args.forecast_days,
        supplier_quotes_per_part=args.quotes_per_part,
        seed=args.seed,
        volatility=args.volatility,
        annual_inflation=args.annual_inflation,
        currency=args.currency,
        dry_run=args.dry_run,
        reset=args.reset,
        confirm_reset=args.confirm_reset,
        output=args.output,
    )

    config.validate()

    if config.dry_run:
        report = {
            "status": "dry_run",
            "schema_version": SCHEMA_VERSION,
            "days": config.days,
            "forecast_days": config.forecast_days,
            "database_modified": False,
            "simulation_only": True,
        }

        print(json.dumps(report, indent=2))
        return 0

    if config.database_url.startswith("sqlite:///"):
        database_path = config.database_url.removeprefix(
            "sqlite:///"
        )

        if database_path != ":memory:":
            path = Path(database_path)

            if not path.is_file():
                raise FileNotFoundError(
                    "Demo database not found. "
                    "Run seed_database.py and seed_parts.py first."
                )

    engine = create_engine(
        config.database_url,
        pool_pre_ping=True,
        hide_parameters=True,
    )

    try:
        # Check prerequisites before creating pricing tables.
        existing_tables = set(
            inspect(engine).get_table_names()
        )

        required_tables = {
            "demo_part_catalog",
            "demo_part_suppliers",
            "demo_spare_part_inventory",
        }

        missing = required_tables - existing_tables

        if missing:
            raise RuntimeError(
                "Missing prerequisite tables: "
                + ", ".join(sorted(missing))
            )

        # Only pricing tables are created here.
        PriceBase.metadata.create_all(engine)

        SessionLocal = sessionmaker(
            bind=engine,
            expire_on_commit=False,
            autoflush=False,
        )

        with SessionLocal() as session:
            with session.begin():
                if config.reset:
                    reset_price_tables(session)

                seeder = PriceSeeder(
                    session,
                    config,
                )

                result = seeder.seed_all()

                statistics = collect_statistics(
                    session
                )

        report = {
            "status": "completed",
            "schema_version": SCHEMA_VERSION,
            "seed": config.seed,
            "currency": config.currency,
            "reset_performed": config.reset,
            "database_modified": True,
            "simulation_only": True,
            "real_market_data_used": False,
            "real_supplier_quotes_used": False,
            "procurement_executed": False,
            "human_approval_required": True,
            "result": result,
            "database_statistics": statistics,
            "limitations": [
                "Prices are synthetic demonstration values.",
                "Forecasts are not trained ML predictions.",
                "Forecast bounds are not calibrated confidence intervals.",
                "Supplier quotations are fictional.",
                "Taxes and shipping are illustrative estimates.",
                "Repair/replacement recommendations require human review.",
            ],
        }

        if config.output:
            save_report(
                config.output,
                report,
            )

        print(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
        )

        LOGGER.info(
            "Synthetic price seeding completed successfully"
        )

        return 0

    finally:
        engine.dispose()


if __name__ == "__main__":
    try:
        sys.exit(main())

    except (
        ValueError,
        RuntimeError,
        FileNotFoundError,
        OSError,
        SQLAlchemyError,
    ) as exc:
        LOGGER.error(
            "Price seeding failed: %s",
            exc,
        )
        sys.exit(1)
```

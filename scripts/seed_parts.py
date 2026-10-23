
#!/usr/bin/env python3
"""
Physical AI & Autonomous Robotics Intelligence Platform
File: scripts/seed_parts.py

Synthetic robot parts and spare inventory seeder.

Features:
- SQLite and PostgreSQL
- Idempotent database seeding
- Robot component catalog
- Installed robot parts
- Spare-parts inventory
- Supplier information
- Synthetic inspection records
- Component health scores
- Failure probability and RUL estimates
- Replacement and repair cost estimates
- Maintenance recommendations
- Low-stock alerts
- Human approval requirements
- Dry-run and reset options
- Transactional writes
- Deterministic random generation
- JSON reporting

Dependencies:
    pip install sqlalchemy

PostgreSQL:
    pip install "psycopg[binary]"

Usage:
    python scripts/seed_parts.py

    python scripts/seed_parts.py --parts-per-robot 8

    python scripts/seed_parts.py --seed 42

    python scripts/seed_parts.py --dry-run

    python scripts/seed_parts.py --reset --confirm-reset

    python scripts/seed_parts.py \
        --output reports/parts_seed.json

Run scripts/seed_database.py first.

All records are synthetic and intended for simulation.
No hardware control or real procurement is performed.
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
    from seed_database import Robot
except ImportError:
    from scripts.seed_database import Robot


LOGGER = logging.getLogger("seed_parts")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

SCHEMA_VERSION = "1.0.0"

DEFAULT_DATABASE_URL = (
    "sqlite:///storage/robotics_demo.db"
)

NAMESPACE = uuid.UUID(
    "d4d7e5a1-8912-4a50-9d2b-702c67d8a012"
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class PartsSeedConfig:
    database_url: str = DEFAULT_DATABASE_URL

    parts_per_robot: int = 8
    inspections_per_part: int = 3

    seed: int = 42

    low_stock_probability: float = 0.20
    degraded_probability: float = 0.18
    critical_probability: float = 0.05

    currency: str = "INR"

    dry_run: bool = False
    reset: bool = False
    confirm_reset: bool = False

    output: str | None = None

    def validate(self) -> None:
        if self.parts_per_robot < 1:
            raise ValueError(
                "parts_per_robot must be positive"
            )

        if self.inspections_per_part < 0:
            raise ValueError(
                "inspections_per_part cannot be negative"
            )

        for name in (
            "low_stock_probability",
            "degraded_probability",
            "critical_probability",
        ):
            value = getattr(self, name)

            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(
                    f"{name} must be between 0 and 1"
                )

        if (
            self.degraded_probability
            + self.critical_probability
            > 1
        ):
            raise ValueError(
                "Combined degraded and critical probabilities "
                "cannot exceed 1"
            )

        if self.reset and not self.confirm_reset:
            raise ValueError(
                "--reset requires --confirm-reset"
            )

        if self.dry_run and self.reset:
            raise ValueError(
                "--dry-run cannot be combined with --reset"
            )


# ============================================================
# Database Models
# ============================================================

class PartsBase(DeclarativeBase):
    pass


class PartCatalog(PartsBase):
    __tablename__ = "demo_part_catalog"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    part_number: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    category: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    manufacturer: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    description: Mapped[str] = mapped_column(
        Text,
        default="",
    )

    unit_cost: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    currency: Mapped[str] = mapped_column(
        String(10),
        default="INR",
    )

    expected_life_hours: Mapped[float] = mapped_column(
        Float,
        default=1000.0,
    )

    specifications: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
    )

    simulation_only: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )


class PartSupplier(PartsBase):
    __tablename__ = "demo_part_suppliers"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    supplier_code: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    region: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    lead_time_days: Mapped[int] = mapped_column(
        Integer,
        default=7,
    )

    reliability_score: Mapped[float] = mapped_column(
        Float,
        default=0.9,
    )

    active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
    )

    synthetic: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
    )


class InstalledRobotPart(PartsBase):
    __tablename__ = "demo_installed_robot_parts"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    robot_id: Mapped[str] = mapped_column(
        ForeignKey("demo_robots.id"),
        nullable=False,
        index=True,
    )

    catalog_part_id: Mapped[str] = mapped_column(
        ForeignKey("demo_part_catalog.id"),
        nullable=False,
    )

    serial_number: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
    )

    installation_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    operating_hours: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    health_score: Mapped[float] = mapped_column(
        Float,
        default=1.0,
    )

    failure_probability: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    remaining_useful_life_hours: Mapped[float] = mapped_column(
        Float,
        default=1000.0,
    )

    condition_status: Mapped[str] = mapped_column(
        String(50),
        default="healthy",
    )

    replacement_recommended: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )

    human_approval_required: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
    )

    simulation_only: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
    )


class SparePartInventory(PartsBase):
    __tablename__ = "demo_spare_part_inventory"

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

    warehouse_code: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    quantity_available: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )

    quantity_reserved: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )

    reorder_point: Mapped[int] = mapped_column(
        Integer,
        default=5,
    )

    reorder_quantity: Mapped[int] = mapped_column(
        Integer,
        default=10,
    )

    low_stock: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )

    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "catalog_part_id",
            "warehouse_code",
            name="uq_demo_inventory_part_warehouse",
        ),
    )


class PartInspection(PartsBase):
    __tablename__ = "demo_part_inspections"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    installed_part_id: Mapped[str] = mapped_column(
        ForeignKey("demo_installed_robot_parts.id"),
        nullable=False,
        index=True,
    )

    inspection_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    inspection_type: Mapped[str] = mapped_column(
        String(100),
        default="synthetic_visual",
    )

    temperature_c: Mapped[float] = mapped_column(
        Float,
        default=40.0,
    )

    vibration_mm_s: Mapped[float] = mapped_column(
        Float,
        default=1.0,
    )

    wear_percent: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    anomaly_score: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    defect_detected: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )

    defect_type: Mapped[str] = mapped_column(
        String(100),
        default="none",
    )

    severity: Mapped[str] = mapped_column(
        String(50),
        default="normal",
    )

    notes: Mapped[str] = mapped_column(
        Text,
        default="",
    )

    human_review_required: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
    )


class PartCostEstimate(PartsBase):
    __tablename__ = "demo_part_cost_estimates"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    installed_part_id: Mapped[str] = mapped_column(
        ForeignKey("demo_installed_robot_parts.id"),
        nullable=False,
        unique=True,
    )

    repair_cost: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    replacement_cost: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    labor_cost: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    downtime_cost: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    total_estimated_cost: Mapped[float] = mapped_column(
        Float,
        default=0.0,
    )

    currency: Mapped[str] = mapped_column(
        String(10),
        default="INR",
    )

    recommendation: Mapped[str] = mapped_column(
        String(50),
        default="inspect",
    )

    approval_required: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
    )


class PartStockAlert(PartsBase):
    __tablename__ = "demo_part_stock_alerts"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
    )

    inventory_id: Mapped[str] = mapped_column(
        ForeignKey("demo_spare_part_inventory.id"),
        nullable=False,
        unique=True,
    )

    severity: Mapped[str] = mapped_column(
        String(30),
        default="warning",
    )

    title: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    acknowledged: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )

    procurement_approved: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )


# ============================================================
# Synthetic Part Catalog
# ============================================================

PART_DEFINITIONS = [
    ("MOTOR-001", "Servo Motor", "motor", 18500, 9000),
    ("MOTOR-002", "BLDC Drive Motor", "motor", 14500, 8500),
    ("GEAR-001", "Precision Gearbox", "gearbox", 22000, 12000),
    ("GEAR-002", "Planetary Gearbox", "gearbox", 17500, 11000),
    ("ACT-001", "Linear Actuator", "actuator", 13500, 8000),
    ("ACT-002", "Rotary Actuator", "actuator", 16500, 8500),
    ("BAT-001", "Battery Module", "battery", 32000, 6000),
    ("BAT-002", "Battery Management Unit", "battery", 8500, 9000),
    ("SEN-001", "LiDAR Sensor", "sensor", 45000, 15000),
    ("SEN-002", "IMU Sensor", "sensor", 6500, 12000),
    ("SEN-003", "Depth Camera", "sensor", 19500, 10000),
    ("SEN-004", "Force Torque Sensor", "sensor", 26000, 12000),
    ("ENC-001", "Rotary Encoder", "encoder", 4200, 10000),
    ("WHEEL-001", "Drive Wheel Assembly", "wheel", 9500, 6500),
    ("JOINT-001", "Robot Joint Assembly", "joint", 28500, 9500),
    ("COOL-001", "Cooling Fan Module", "cooling", 3500, 5500),
    ("CTRL-001", "Motor Controller", "controller", 12500, 11000),
    ("PWR-001", "Power Distribution Board", "power", 7800, 11000),
]

SUPPLIER_DEFINITIONS = [
    ("SUP-001", "Demo Robotics Components", "India", 5, 0.95),
    ("SUP-002", "Synthetic Motion Systems", "Germany", 14, 0.93),
    ("SUP-003", "Simulation Sensor Supply", "Japan", 12, 0.96),
    ("SUP-004", "Virtual Industrial Parts", "India", 4, 0.90),
]

ROBOT_PART_CATEGORIES = {
    "mobile_robot": {
        "motor", "gearbox", "battery", "sensor",
        "encoder", "wheel", "controller", "power",
    },
    "robotic_arm": {
        "motor", "gearbox", "actuator", "sensor",
        "encoder", "joint", "controller", "cooling",
    },
    "humanoid": {
        "motor", "actuator", "battery", "sensor",
        "encoder", "joint", "cooling", "power",
    },
    "quadruped": {
        "motor", "gearbox", "actuator", "battery",
        "sensor", "encoder", "joint", "controller",
    },
}


# ============================================================
# Utilities
# ============================================================

def stable_id(namespace: str, value: str) -> str:
    return str(
        uuid.uuid5(
            NAMESPACE,
            f"{namespace}:{value}",
        )
    )


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def calculate_condition(
    health_score: float,
) -> str:
    if health_score < 0.35:
        return "critical"
    if health_score < 0.65:
        return "degraded"
    if health_score < 0.85:
        return "warning"
    return "healthy"


def calculate_priority(
    failure_probability: float,
    rul_hours: float,
) -> str:
    if failure_probability >= 0.8 or rul_hours <= 24:
        return "critical"
    if failure_probability >= 0.6 or rul_hours <= 72:
        return "high"
    if failure_probability >= 0.3 or rul_hours <= 240:
        return "medium"
    return "low"


# ============================================================
# Parts Seeder
# ============================================================

class PartsSeeder:
    def __init__(
        self,
        session: Session,
        config: PartsSeedConfig,
    ) -> None:
        self.session = session
        self.config = config
        self.rng = random.Random(config.seed)
        self.counts = Counter()

        self.base_time = datetime(
            2026, 1, 1, tzinfo=timezone.utc
        )

    def upsert(
        self,
        model_class: type,
        identifier: str,
        values: dict[str, Any],
    ) -> Any:
        instance = self.session.get(
            model_class,
            identifier,
        )

        if instance is None:
            instance = model_class(
                id=identifier,
                **values,
            )
            self.session.add(instance)
            self.counts["inserted"] += 1
        else:
            for key, value in values.items():
                setattr(instance, key, value)
            self.counts["updated"] += 1

        self.counts[model_class.__tablename__] += 1
        return instance

    def load_robots(self) -> list[Robot]:
        return list(
            self.session.scalars(
                select(Robot).order_by(Robot.robot_code)
            ).all()
        )

    def seed_catalog(self) -> list[PartCatalog]:
        catalog = []

        for (
            part_number,
            name,
            category,
            cost,
            life_hours,
        ) in PART_DEFINITIONS:

            part = self.upsert(
                PartCatalog,
                stable_id("catalog", part_number),
                {
                    "part_number": part_number,
                    "name": name,
                    "category": category,
                    "manufacturer": "Synthetic Robotics Lab",
                    "description": (
                        f"Demonstration {name.lower()} "
                        "for robotics simulation."
                    ),
                    "unit_cost": float(cost),
                    "currency": self.config.currency,
                    "expected_life_hours": float(life_hours),
                    "specifications": {
                        "simulation_only": True,
                        "part_category": category,
                        "validated_for_hardware": False,
                    },
                    "simulation_only": True,
                },
            )

            catalog.append(part)

        self.session.flush()
        return catalog

    def seed_suppliers(self) -> list[PartSupplier]:
        suppliers = []

        for (
            code,
            name,
            region,
            lead_time,
            reliability,
        ) in SUPPLIER_DEFINITIONS:

            supplier = self.upsert(
                PartSupplier,
                stable_id("supplier", code),
                {
                    "supplier_code": code,
                    "name": name,
                    "region": region,
                    "lead_time_days": lead_time,
                    "reliability_score": reliability,
                    "active": True,
                    "synthetic": True,
                },
            )

            suppliers.append(supplier)

        self.session.flush()
        return suppliers

    def generate_health(self) -> float:
        probability = self.rng.random()

        if probability < self.config.critical_probability:
            return round(self.rng.uniform(0.1, 0.34), 4)

        if probability < (
            self.config.critical_probability
            + self.config.degraded_probability
        ):
            return round(self.rng.uniform(0.35, 0.64), 4)

        return round(self.rng.uniform(0.75, 1.0), 4)

    def seed_installed_parts(
        self,
        robots: list[Robot],
        catalog: list[PartCatalog],
    ) -> list[InstalledRobotPart]:

        installed_parts = []

        for robot in robots:
            allowed = ROBOT_PART_CATEGORIES.get(
                robot.robot_type,
                set(ROBOT_PART_CATEGORIES["mobile_robot"]),
            )

            compatible_parts = [
                part
                for part in catalog
                if part.category in allowed
            ]

            for index in range(self.config.parts_per_robot):
                catalog_part = compatible_parts[
                    index % len(compatible_parts)
                ]

                serial = (
                    f"SIM-{robot.robot_code}-"
                    f"{catalog_part.part_number}-{index:03d}"
                )

                health = self.generate_health()
                condition = calculate_condition(health)

                operating_hours = self.rng.uniform(
                    50,
                    2500,
                )

                failure_probability = clamp(
                    (1.0 - health) * 0.9
                    + self.rng.uniform(0.0, 0.08),
                    0.0,
                    1.0,
                )

                rul = max(
                    1.0,
                    catalog_part.expected_life_hours
                    * health
                    - operating_hours * 0.05,
                )

                replacement_recommended = (
                    condition in {"critical", "degraded"}
                    or failure_probability >= 0.65
                )

                part = self.upsert(
                    InstalledRobotPart,
                    stable_id("installed", serial),
                    {
                        "robot_id": robot.id,
                        "catalog_part_id": catalog_part.id,
                        "serial_number": serial,
                        "installation_date": (
                            self.base_time
                            - timedelta(
                                days=index * 12 + 30
                            )
                        ),
                        "operating_hours": round(
                            operating_hours, 3
                        ),
                        "health_score": health,
                        "failure_probability": round(
                            failure_probability, 4
                        ),
                        "remaining_useful_life_hours": round(
                            rul, 3
                        ),
                        "condition_status": condition,
                        "replacement_recommended": (
                            replacement_recommended
                        ),
                        "human_approval_required": True,
                        "simulation_only": True,
                    },
                )

                installed_parts.append(part)
                self.counts[f"condition_{condition}"] += 1

        self.session.flush()
        return installed_parts

    def seed_inventory(
        self,
        catalog: list[PartCatalog],
        suppliers: list[PartSupplier],
    ) -> list[SparePartInventory]:

        inventory_records = []

        for index, part in enumerate(catalog):
            supplier = suppliers[index % len(suppliers)]

            low_stock = (
                self.rng.random()
                < self.config.low_stock_probability
            )

            reorder_point = self.rng.randint(3, 8)

            if low_stock:
                quantity = self.rng.randint(
                    0, reorder_point
                )
            else:
                quantity = self.rng.randint(
                    reorder_point + 1, 30
                )

            reserved = self.rng.randint(
                0, min(3, quantity)
            )

            available = quantity - reserved
            low_stock = available <= reorder_point

            inventory = self.upsert(
                SparePartInventory,
                stable_id(
                    "inventory",
                    f"{part.part_number}:DEMO-WH-01",
                ),
                {
                    "catalog_part_id": part.id,
                    "supplier_id": supplier.id,
                    "warehouse_code": "DEMO-WH-01",
                    "quantity_available": available,
                    "quantity_reserved": reserved,
                    "reorder_point": reorder_point,
                    "reorder_quantity": self.rng.randint(
                        10, 25
                    ),
                    "low_stock": low_stock,
                    "last_updated": self.base_time,
                },
            )

            inventory_records.append(inventory)

            if low_stock:
                self.counts["low_stock_items"] += 1

        self.session.flush()
        return inventory_records

    def seed_inspections(
        self,
        installed_parts: list[InstalledRobotPart],
    ) -> None:

        defect_types = [
            "wear",
            "corrosion",
            "crack",
            "overheating",
            "vibration",
            "misalignment",
        ]

        for part in installed_parts:
            for index in range(
                self.config.inspections_per_part
            ):
                defect_detected = (
                    part.condition_status
                    in {"degraded", "critical"}
                )

                if not defect_detected:
                    defect_detected = (
                        self.rng.random() < 0.05
                    )

                defect_type = (
                    self.rng.choice(defect_types)
                    if defect_detected
                    else "none"
                )

                wear = clamp(
                    (1.0 - part.health_score) * 100
                    + self.rng.uniform(-5, 5),
                    0.0,
                    100.0,
                )

                temperature = (
                    35.0
                    + 40.0 * (1.0 - part.health_score)
                    + self.rng.uniform(-3, 3)
                )

                vibration = (
                    0.7
                    + 5.0 * (1.0 - part.health_score)
                    + self.rng.uniform(0, 0.5)
                )

                anomaly_score = clamp(
                    1.0 - part.health_score
                    + self.rng.uniform(-0.05, 0.05),
                    0.0,
                    1.0,
                )

                self.upsert(
                    PartInspection,
                    stable_id(
                        "inspection",
                        f"{part.id}:{index}",
                    ),
                    {
                        "installed_part_id": part.id,
                        "inspection_date": (
                            self.base_time
                            + timedelta(days=index * 7)
                        ),
                        "inspection_type": (
                            "synthetic_visual_and_sensor"
                        ),
                        "temperature_c": round(
                            temperature, 3
                        ),
                        "vibration_mm_s": round(
                            vibration, 3
                        ),
                        "wear_percent": round(wear, 3),
                        "anomaly_score": round(
                            anomaly_score, 4
                        ),
                        "defect_detected": defect_detected,
                        "defect_type": defect_type,
                        "severity": part.condition_status,
                        "notes": (
                            "Synthetic inspection only. "
                            "No physical component was inspected."
                        ),
                        "human_review_required": True,
                    },
                )

    def seed_cost_estimates(
        self,
        installed_parts: list[InstalledRobotPart],
        catalog: list[PartCatalog],
    ) -> None:

        catalog_by_id = {
            part.id: part
            for part in catalog
        }

        for installed in installed_parts:
            catalog_part = catalog_by_id[
                installed.catalog_part_id
            ]

            replacement_cost = catalog_part.unit_cost

            repair_cost = (
                replacement_cost
                * self.rng.uniform(0.15, 0.65)
            )

            labor_cost = self.rng.uniform(
                800, 6500
            )

            downtime_cost = self.rng.uniform(
                1000, 12000
            )

            repair_total = (
                repair_cost
                + labor_cost
                + downtime_cost
            )

            replacement_total = (
                replacement_cost
                + labor_cost
                + downtime_cost
            )

            if installed.condition_status == "critical":
                recommendation = "replace"
                total = replacement_total

            elif (
                installed.condition_status == "degraded"
                and repair_total < replacement_total
            ):
                recommendation = "repair"
                total = repair_total

            elif installed.condition_status == "degraded":
                recommendation = "replace"
                total = replacement_total

            elif installed.condition_status == "warning":
                recommendation = "inspect"
                total = labor_cost

            else:
                recommendation = "monitor"
                total = 0.0

            self.upsert(
                PartCostEstimate,
                stable_id("cost", installed.id),
                {
                    "installed_part_id": installed.id,
                    "repair_cost": round(
                        repair_cost, 2
                    ),
                    "replacement_cost": round(
                        replacement_cost, 2
                    ),
                    "labor_cost": round(
                        labor_cost, 2
                    ),
                    "downtime_cost": round(
                        downtime_cost, 2
                    ),
                    "total_estimated_cost": round(
                        total, 2
                    ),
                    "currency": self.config.currency,
                    "recommendation": recommendation,
                    "approval_required": True,
                },
            )

            self.counts[
                f"recommendation_{recommendation}"
            ] += 1

    def seed_stock_alerts(
        self,
        inventory_records: list[SparePartInventory],
    ) -> None:

        for inventory in inventory_records:
            alert_id = stable_id(
                "stock_alert",
                inventory.id,
            )

            if not inventory.low_stock:
                # Clear stale alerts created by this seeder
                # when deterministic inventory is refreshed.
                existing = self.session.get(
                    PartStockAlert,
                    alert_id,
                )
                if existing is not None:
                    self.session.delete(existing)
                continue

            severity = (
                "critical"
                if inventory.quantity_available == 0
                else "warning"
            )

            self.upsert(
                PartStockAlert,
                alert_id,
                {
                    "inventory_id": inventory.id,
                    "severity": severity,
                    "title": "Synthetic Spare Part Stock Alert",
                    "message": (
                        "Demo warehouse inventory is at or "
                        "below the reorder threshold. "
                        "Human procurement approval required."
                    ),
                    "acknowledged": False,
                    "procurement_approved": False,
                    "created_at": self.base_time,
                },
            )

    def seed_all(self) -> dict[str, Any]:
        robots = self.load_robots()

        if not robots:
            raise RuntimeError(
                "No demonstration robots found. "
                "Run scripts/seed_database.py first."
            )

        catalog = self.seed_catalog()
        suppliers = self.seed_suppliers()

        installed_parts = self.seed_installed_parts(
            robots, catalog
        )

        inventory = self.seed_inventory(
            catalog, suppliers
        )

        self.seed_inspections(installed_parts)

        self.seed_cost_estimates(
            installed_parts, catalog
        )

        self.seed_stock_alerts(inventory)

        self.session.flush()

        return {
            "robots": len(robots),
            "catalog_parts": len(catalog),
            "suppliers": len(suppliers),
            "installed_parts": len(installed_parts),
            "inventory_records": len(inventory),
            "inspection_records_requested": (
                len(installed_parts)
                * self.config.inspections_per_part
            ),
            "statistics": dict(self.counts),
        }


# ============================================================
# Reset
# ============================================================

def reset_parts_tables(session: Session) -> None:
    """
    Delete only demo parts records owned by this script.
    Existing application and robot tables remain untouched.
    """

    deletion_order = [
        PartStockAlert,
        PartCostEstimate,
        PartInspection,
        SparePartInventory,
        InstalledRobotPart,
        PartSupplier,
        PartCatalog,
    ]

    for model_class in deletion_order:
        session.execute(
            delete(model_class)
        )

    session.flush()


# ============================================================
# Statistics
# ============================================================

def database_statistics(
    session: Session,
) -> dict[str, int]:

    models = [
        PartCatalog,
        PartSupplier,
        InstalledRobotPart,
        SparePartInventory,
        PartInspection,
        PartCostEstimate,
        PartStockAlert,
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
            "Seed synthetic robot parts and inventory"
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
        "--parts-per-robot",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--inspections-per-part",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--low-stock-probability",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--degraded-probability",
        type=float,
        default=0.18,
    )

    parser.add_argument(
        "--critical-probability",
        type=float,
        default=0.05,
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

    config = PartsSeedConfig(
        database_url=args.database_url,
        parts_per_robot=args.parts_per_robot,
        inspections_per_part=args.inspections_per_part,
        seed=args.seed,
        low_stock_probability=(
            args.low_stock_probability
        ),
        degraded_probability=(
            args.degraded_probability
        ),
        critical_probability=(
            args.critical_probability
        ),
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
            "parts_per_robot": config.parts_per_robot,
            "inspections_per_part": (
                config.inspections_per_part
            ),
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
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if not path.is_file():
                raise FileNotFoundError(
                    "Demo database does not exist. "
                    "Run scripts/seed_database.py first."
                )

    engine = create_engine(
        config.database_url,
        pool_pre_ping=True,
        hide_parameters=True,
    )

    try:
        # Create only the demo parts tables.
        PartsBase.metadata.create_all(engine)

        SessionLocal = sessionmaker(
            bind=engine,
            expire_on_commit=False,
            autoflush=False,
        )

        with SessionLocal() as session:
            with session.begin():
                if config.reset:
                    reset_parts_tables(session)

                seeder = PartsSeeder(
                    session, config
                )

                result = seeder.seed_all()

                statistics = database_statistics(
                    session
                )

        report = {
            "status": "completed",
            "schema_version": SCHEMA_VERSION,
            "seed": config.seed,
            "parts_per_robot": config.parts_per_robot,
            "inspections_per_part": (
                config.inspections_per_part
            ),
            "reset_performed": config.reset,
            "database_modified": True,
            "simulation_only": True,
            "physical_execution_authorized": False,
            "procurement_executed": False,
            "human_approval_required": True,
            "result": result,
            "database_statistics": statistics,
            "limitations": [
                "Part health scores are synthetic.",
                "Failure probability and RUL are illustrative.",
                "Costs are demonstration estimates, not quotations.",
                "Supplier identities are fictional.",
                "No physical parts were inspected.",
                "No procurement or hardware commands were executed.",
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
            "Robot parts seeding completed"
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
            "Parts seeding failed: %s",
            exc,
        )
        sys.exit(1)
```


#!/usr/bin/env python3
"""
Physical AI & Autonomous Robotics Intelligence Platform
File: scripts/seed_maintenance.py

Synthetic predictive-maintenance database seeder.

Compatible with:
    scripts/seed_database.py

Uses the existing SQLAlchemy models:
    Robot
    MaintenanceRecord
    AuditLog

Features:
    - Idempotent maintenance seeding
    - SQLite and PostgreSQL support
    - Preventive, predictive, corrective maintenance
    - Motor, gearbox, actuator, battery and sensor tasks
    - Synthetic failure probability and RUL
    - Risk-based maintenance priorities
    - Scheduled maintenance dates
    - Human approval requirements
    - Simulation-only safety restrictions
    - Transactional database operations
    - Dry-run mode
    - Optional replacement of seeded maintenance records
    - JSON reporting
    - Deterministic random generation

Dependencies:
    pip install sqlalchemy

For PostgreSQL:
    pip install "psycopg[binary]"

Examples:
    python scripts/seed_maintenance.py

    python scripts/seed_maintenance.py --tasks-per-robot 5

    python scripts/seed_maintenance.py --seed 42

    python scripts/seed_maintenance.py --dry-run

    python scripts/seed_maintenance.py --replace

    python scripts/seed_maintenance.py \
        --output reports/maintenance_seed.json

Important:
    Run scripts/seed_database.py first to create
    demonstration robots and database tables.

    This script does not issue physical robot commands.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import uuid

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

try:
    from seed_database import (
        AuditLog,
        MaintenanceRecord,
        Robot,
    )
except ImportError:
    from scripts.seed_database import (
        AuditLog,
        MaintenanceRecord,
        Robot,
    )


LOGGER = logging.getLogger("seed_maintenance")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

DEFAULT_DATABASE_URL = "sqlite:///storage/robotics_demo.db"

MAINTENANCE_NAMESPACE = uuid.UUID(
    "d0a118c2-4c3d-4a50-b47b-8ad891f5b001"
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class MaintenanceSeedConfig:
    database_url: str = DEFAULT_DATABASE_URL

    tasks_per_robot: int = 4
    seed: int = 42

    schedule_days: int = 30

    predictive_ratio: float = 0.6
    preventive_ratio: float = 0.3
    corrective_ratio: float = 0.1

    dry_run: bool = False
    replace: bool = False

    output: str | None = None

    def validate(self) -> None:
        if self.tasks_per_robot < 1:
            raise ValueError(
                "tasks_per_robot must be positive"
            )

        if self.schedule_days < 1:
            raise ValueError(
                "schedule_days must be positive"
            )

        ratios = [
            self.predictive_ratio,
            self.preventive_ratio,
            self.corrective_ratio,
        ]

        if any(
            not 0 <= ratio <= 1
            for ratio in ratios
        ):
            raise ValueError(
                "Maintenance ratios must be between 0 and 1"
            )

        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(
                "Maintenance ratios must sum to 1"
            )


# ============================================================
# Maintenance Components
# ============================================================

COMPONENT_CATALOG = {
    "motor": {
        "failure_modes": [
            "bearing_wear",
            "overheating",
            "winding_degradation",
            "excessive_current",
        ],
        "actions": [
            "Inspect motor bearings",
            "Review motor temperature history",
            "Inspect electrical connections",
            "Review motor current trends",
        ],
    },
    "gearbox": {
        "failure_modes": [
            "gear_wear",
            "lubrication_degradation",
            "excessive_vibration",
            "backlash_increase",
        ],
        "actions": [
            "Inspect gearbox condition",
            "Review lubrication records",
            "Analyze vibration trends",
            "Measure gearbox backlash",
        ],
    },
    "battery": {
        "failure_modes": [
            "capacity_degradation",
            "thermal_stress",
            "voltage_imbalance",
            "charging_fault",
        ],
        "actions": [
            "Review battery health metrics",
            "Inspect charging history",
            "Analyze battery temperature",
            "Check battery management diagnostics",
        ],
    },
    "actuator": {
        "failure_modes": [
            "torque_overload",
            "position_error",
            "thermal_degradation",
            "mechanical_wear",
        ],
        "actions": [
            "Review actuator torque history",
            "Inspect actuator health",
            "Check positioning accuracy",
            "Review actuator temperature trends",
        ],
    },
    "sensor_array": {
        "failure_modes": [
            "calibration_drift",
            "signal_loss",
            "noise_increase",
            "sensor_degradation",
        ],
        "actions": [
            "Review sensor diagnostics",
            "Check calibration records",
            "Inspect sensor data quality",
            "Review sensor communication errors",
        ],
    },
    "wheel_assembly": {
        "failure_modes": [
            "wheel_wear",
            "bearing_failure",
            "traction_loss",
        ],
        "actions": [
            "Inspect wheel assembly",
            "Review traction measurements",
            "Inspect wheel bearings",
        ],
    },
    "joint_assembly": {
        "failure_modes": [
            "joint_wear",
            "encoder_drift",
            "joint_friction",
        ],
        "actions": [
            "Inspect joint assembly",
            "Review encoder diagnostics",
            "Analyze joint friction trends",
        ],
    },
    "cooling_system": {
        "failure_modes": [
            "cooling_degradation",
            "thermal_overload",
            "airflow_restriction",
        ],
        "actions": [
            "Inspect cooling system",
            "Review thermal performance",
            "Inspect ventilation paths",
        ],
    },
}


ROBOT_COMPONENTS = {
    "mobile_robot": [
        "motor",
        "gearbox",
        "battery",
        "sensor_array",
        "wheel_assembly",
    ],
    "robotic_arm": [
        "motor",
        "gearbox",
        "actuator",
        "joint_assembly",
        "sensor_array",
    ],
    "humanoid": [
        "motor",
        "battery",
        "actuator",
        "joint_assembly",
        "cooling_system",
    ],
    "quadruped": [
        "motor",
        "gearbox",
        "battery",
        "joint_assembly",
        "sensor_array",
    ],
}


# ============================================================
# Helpers
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stable_id(
    robot_id: str,
    task_index: int,
) -> str:
    return str(
        uuid.uuid5(
            MAINTENANCE_NAMESPACE,
            f"{robot_id}:maintenance:{task_index}",
        )
    )


def stable_audit_id(
    maintenance_id: str,
) -> str:
    return str(
        uuid.uuid5(
            MAINTENANCE_NAMESPACE,
            f"audit:{maintenance_id}",
        )
    )


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(
        minimum,
        min(maximum, value),
    )


def classify_priority(
    failure_probability: float,
    rul_hours: float,
) -> str:
    if failure_probability >= 0.85 or rul_hours <= 12:
        return "critical"

    if failure_probability >= 0.65 or rul_hours <= 48:
        return "high"

    if failure_probability >= 0.35 or rul_hours <= 168:
        return "medium"

    return "low"


def classify_risk(
    failure_probability: float,
    rul_hours: float,
) -> str:
    if failure_probability >= 0.8 or rul_hours < 24:
        return "critical"

    if failure_probability >= 0.6 or rul_hours < 72:
        return "high"

    if failure_probability >= 0.3 or rul_hours < 240:
        return "moderate"

    return "low"


def schedule_offset_days(
    priority: str,
    rng: random.Random,
    max_days: int,
) -> int:
    if priority == "critical":
        return 0

    if priority == "high":
        return rng.randint(
            0,
            min(2, max_days),
        )

    if priority == "medium":
        return rng.randint(
            1,
            min(7, max_days),
        )

    return rng.randint(
        1,
        max_days,
    )


# ============================================================
# Synthetic Maintenance Generator
# ============================================================

class MaintenanceDataGenerator:
    def __init__(
        self,
        config: MaintenanceSeedConfig,
    ) -> None:
        self.config = config

        self.rng = random.Random(
            config.seed
        )

        self.base_time = datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        )

    def select_maintenance_type(self) -> str:
        value = self.rng.random()

        if value < self.config.predictive_ratio:
            return "predictive"

        if value < (
            self.config.predictive_ratio
            + self.config.preventive_ratio
        ):
            return "preventive"

        return "corrective"

    def generate_failure_probability(
        self,
        maintenance_type: str,
    ) -> float:
        if maintenance_type == "corrective":
            return round(
                self.rng.uniform(
                    0.65,
                    0.98,
                ),
                4,
            )

        if maintenance_type == "predictive":
            return round(
                self.rng.uniform(
                    0.2,
                    0.9,
                ),
                4,
            )

        return round(
            self.rng.uniform(
                0.02,
                0.4,
            ),
            4,
        )

    def generate_rul(
        self,
        failure_probability: float,
    ) -> float:
        base_rul = (
            1200.0
            * (1.0 - failure_probability)
        )

        noise = self.rng.uniform(
            -50,
            50,
        )

        return round(
            max(
                1.0,
                base_rul + noise,
            ),
            3,
        )

    def generate_task(
        self,
        robot: Robot,
        task_index: int,
    ) -> dict[str, Any]:
        components = ROBOT_COMPONENTS.get(
            robot.robot_type,
            [
                "motor",
                "battery",
                "sensor_array",
            ],
        )

        component = components[
            task_index % len(components)
        ]

        component_info = COMPONENT_CATALOG[
            component
        ]

        maintenance_type = (
            self.select_maintenance_type()
        )

        failure_probability = (
            self.generate_failure_probability(
                maintenance_type
            )
        )

        rul_hours = self.generate_rul(
            failure_probability
        )

        priority = classify_priority(
            failure_probability,
            rul_hours,
        )

        risk = classify_risk(
            failure_probability,
            rul_hours,
        )

        failure_mode = self.rng.choice(
            component_info["failure_modes"]
        )

        recommended_action = self.rng.choice(
            component_info["actions"]
        )

        days_until_maintenance = (
            schedule_offset_days(
                priority,
                self.rng,
                self.config.schedule_days,
            )
        )

        scheduled_at = (
            self.base_time
            + timedelta(
                days=days_until_maintenance,
                hours=task_index,
            )
        )

        if maintenance_type == "corrective":
            status = "pending_review"

        else:
            status = "scheduled"

        description = (
            f"SYNTHETIC SIMULATION DATA. "
            f"Robot: {robot.robot_code}. "
            f"Component: {component}. "
            f"Maintenance type: {maintenance_type}. "
            f"Suspected failure mode: {failure_mode}. "
            f"Estimated risk: {risk}. "
            f"Predicted failure probability: "
            f"{failure_probability:.2%}. "
            f"Estimated RUL: {rul_hours:.1f} hours. "
            f"Recommended action: {recommended_action}. "
            f"Human engineering review required. "
            f"No automatic physical maintenance or "
            f"robot actuation is authorized."
        )

        return {
            "id": stable_id(
                robot.id,
                task_index,
            ),
            "robot_id": robot.id,
            "maintenance_type": maintenance_type,
            "component_name": component,
            "priority": priority,
            "status": status,
            "predicted_failure_probability": (
                failure_probability
            ),
            "estimated_rul_hours": rul_hours,
            "scheduled_at": scheduled_at,
            "description": description,
            "human_approval_required": True,
            "risk_level": risk,
            "failure_mode": failure_mode,
            "recommended_action": recommended_action,
        }


# ============================================================
# Database Seeder
# ============================================================

class MaintenanceSeeder:
    def __init__(
        self,
        session: Session,
        config: MaintenanceSeedConfig,
    ) -> None:
        self.session = session
        self.config = config

        self.generator = MaintenanceDataGenerator(
            config
        )

        self.statistics = Counter()

    def load_robots(self) -> list[Robot]:
        return list(
            self.session.scalars(
                select(Robot).order_by(
                    Robot.robot_code
                )
            ).all()
        )

    def replace_seeded_records(
        self,
        robots: list[Robot],
    ) -> int:
        """
        Remove only records with deterministic IDs
        belonging to this maintenance seeder.
        """
        identifiers = [
            stable_id(
                robot.id,
                task_index,
            )
            for robot in robots
            for task_index in range(
                self.config.tasks_per_robot
            )
        ]

        if not identifiers:
            return 0

        audit_ids = [
            stable_audit_id(
                maintenance_id
            )
            for maintenance_id in identifiers
        ]

        self.session.execute(
            delete(AuditLog).where(
                AuditLog.id.in_(audit_ids)
            )
        )

        result = self.session.execute(
            delete(MaintenanceRecord).where(
                MaintenanceRecord.id.in_(
                    identifiers
                )
            )
        )

        return int(
            result.rowcount or 0
        )

    def upsert_maintenance(
        self,
        data: dict[str, Any],
    ) -> str:
        maintenance_id = data["id"]

        existing = self.session.get(
            MaintenanceRecord,
            maintenance_id,
        )

        values = {
            "robot_id": data["robot_id"],
            "maintenance_type": data[
                "maintenance_type"
            ],
            "component_name": data[
                "component_name"
            ],
            "priority": data["priority"],
            "status": data["status"],
            "predicted_failure_probability": data[
                "predicted_failure_probability"
            ],
            "estimated_rul_hours": data[
                "estimated_rul_hours"
            ],
            "scheduled_at": data["scheduled_at"],
            "description": data["description"],
            "human_approval_required": True,
        }

        if existing is None:
            record = MaintenanceRecord(
                id=maintenance_id,
                **values,
            )

            self.session.add(
                record
            )

            return "inserted"

        for key, value in values.items():
            setattr(
                existing,
                key,
                value,
            )

        return "updated"

    def create_audit_log(
        self,
        data: dict[str, Any],
    ) -> None:
        audit_id = stable_audit_id(
            data["id"]
        )

        existing = self.session.get(
            AuditLog,
            audit_id,
        )

        values = {
            "actor": "maintenance-demo-seeder",
            "action": "synthetic_maintenance_seed",
            "resource_type": "maintenance_record",
            "resource_id": data["id"],
            "details": {
                "simulation_only": True,
                "maintenance_type": data[
                    "maintenance_type"
                ],
                "component_name": data[
                    "component_name"
                ],
                "priority": data["priority"],
                "risk_level": data["risk_level"],
                "failure_mode": data["failure_mode"],
                "recommended_action": data[
                    "recommended_action"
                ],
                "human_approval_required": True,
                "physical_actuation_allowed": False,
            },
            "timestamp": self.generator.base_time,
        }

        if existing is None:
            self.session.add(
                AuditLog(
                    id=audit_id,
                    **values,
                )
            )

        else:
            for key, value in values.items():
                setattr(
                    existing,
                    key,
                    value,
                )

    def seed(self) -> dict[str, Any]:
        robots = self.load_robots()

        if not robots:
            raise RuntimeError(
                "No demo robots found. "
                "Run scripts/seed_database.py first."
            )

        LOGGER.info(
            "Found %d demo robots",
            len(robots),
        )

        if self.config.replace:
            removed = self.replace_seeded_records(
                robots
            )

            self.statistics["replaced"] = removed

        records = []

        for robot in robots:
            for task_index in range(
                self.config.tasks_per_robot
            ):
                data = (
                    self.generator.generate_task(
                        robot,
                        task_index,
                    )
                )

                records.append(
                    data
                )

                self.statistics[
                    f"priority_{data['priority']}"
                ] += 1

                self.statistics[
                    f"type_{data['maintenance_type']}"
                ] += 1

                self.statistics[
                    f"component_{data['component_name']}"
                ] += 1

                if self.config.dry_run:
                    continue

                operation = self.upsert_maintenance(
                    data
                )

                self.statistics[
                    operation
                ] += 1

                self.create_audit_log(
                    data
                )

        self.session.flush()

        return {
            "robot_count": len(robots),
            "maintenance_record_count": len(records),
            "statistics": dict(
                self.statistics
            ),
            "sample_records": [
                {
                    "id": record["id"],
                    "robot_id": record["robot_id"],
                    "maintenance_type": record[
                        "maintenance_type"
                    ],
                    "component_name": record[
                        "component_name"
                    ],
                    "priority": record["priority"],
                    "risk_level": record["risk_level"],
                    "failure_probability": record[
                        "predicted_failure_probability"
                    ],
                    "rul_hours": record[
                        "estimated_rul_hours"
                    ],
                    "scheduled_at": record[
                        "scheduled_at"
                    ].isoformat(),
                    "human_approval_required": True,
                }
                for record in records[:10]
            ],
        }


# ============================================================
# Report Export
# ============================================================

def export_json(
    path: str | Path,
    payload: dict[str, Any],
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )

    try:
        temporary_path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        os.replace(
            temporary_path,
            path,
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Seed synthetic robotics maintenance records"
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
        "--tasks-per-robot",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--schedule-days",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--predictive-ratio",
        type=float,
        default=0.6,
    )

    parser.add_argument(
        "--preventive-ratio",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--corrective-ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--replace",
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

    config = MaintenanceSeedConfig(
        database_url=args.database_url,
        tasks_per_robot=args.tasks_per_robot,
        seed=args.seed,
        schedule_days=args.schedule_days,
        predictive_ratio=args.predictive_ratio,
        preventive_ratio=args.preventive_ratio,
        corrective_ratio=args.corrective_ratio,
        dry_run=args.dry_run,
        replace=args.replace,
        output=args.output,
    )

    config.validate()

    if config.dry_run and config.replace:
        raise ValueError(
            "--dry-run and --replace cannot be combined"
        )

    if (
        config.database_url.startswith(
            "sqlite:///"
        )
    ):
        database_path = config.database_url.removeprefix(
            "sqlite:///"
        )

        if database_path != ":memory:":
            Path(database_path).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if not Path(database_path).is_file():
                raise FileNotFoundError(
                    "Demo database does not exist. "
                    "Run scripts/seed_database.py first."
                )

    engine = create_engine(
        config.database_url,
        pool_pre_ping=True,
        hide_parameters=True,
    )

    SessionLocal = sessionmaker(
        bind=engine,
        expire_on_commit=False,
        autoflush=False,
    )

    try:
        with SessionLocal() as session:
            with session.begin():
                seeder = MaintenanceSeeder(
                    session,
                    config,
                )

                result = seeder.seed()

        report = {
            "status": (
                "dry_run"
                if config.dry_run
                else "completed"
            ),
            "schema_version": "1.0.0",
            "seed": config.seed,
            "tasks_per_robot": (
                config.tasks_per_robot
            ),
            "database_modified": (
                not config.dry_run
            ),
            "replace": config.replace,
            "simulation_only": True,
            "physical_execution_authorized": False,
            "human_approval_required": True,
            "result": result,
        }

        if config.output:
            export_json(
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

        return 0

    finally:
        engine.dispose()


if __name__ == "__main__":
    try:
        sys.exit(
            main()
        )

    except (
        ValueError,
        RuntimeError,
        FileNotFoundError,
        OSError,
        SQLAlchemyError,
    ) as exc:
        LOGGER.error(
            "Maintenance seeding failed: %s",
            exc,
        )

        sys.exit(1)
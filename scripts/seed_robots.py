
#!/usr/bin/env python3
"""
Physical AI & Autonomous Robotics Intelligence Platform
File: scripts/seed_robots.py

Synthetic robot fleet database seeder.

Features:
- Mobile robots, robotic arms, humanoids, quadrupeds
- Robot specifications and configuration profiles
- Sensor registration and health
- Actuator registration and status
- Battery health and operating hours
- Fleet operating states
- Simulation-only safety settings
- Deterministic robot identities
- Idempotent database updates
- SQLite and PostgreSQL support
- Dry-run and JSON reporting
- Transactional writes
- No physical hardware actuation

Dependencies:
    pip install sqlalchemy

PostgreSQL:
    pip install "psycopg[binary]"

Examples:
    python scripts/seed_robots.py

    python scripts/seed_robots.py --robots 20

    python scripts/seed_robots.py --robots 12 --seed 42

    python scripts/seed_robots.py --robot-type humanoid

    python scripts/seed_robots.py --robot-type mixed

    python scripts/seed_robots.py --dry-run

    python scripts/seed_robots.py \
        --output reports/robot_seed.json

Prerequisite:
    python scripts/seed_database.py

Safety:
    All robots are synthetic demonstration records.
    Physical robot control is disabled.
    No real hardware commands are issued.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import (
    create_engine,
    inspect,
    select,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import (
    Session,
    sessionmaker,
)

try:
    from seed_database import (
        Robot,
        RobotSensor,
        RobotActuator,
        AuditLog,
    )
except ImportError:
    from scripts.seed_database import (
        Robot,
        RobotSensor,
        RobotActuator,
        AuditLog,
    )


LOGGER = logging.getLogger("seed_robots")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

SCHEMA_VERSION = "1.0.0"

DEFAULT_DATABASE_URL = (
    "sqlite:///storage/robotics_demo.db"
)

ROBOT_NAMESPACE = uuid.UUID(
    "2c3f470e-70e1-44c2-8e75-5c1378c02a11"
)

BASE_TIME = datetime(
    2026, 1, 1, tzinfo=timezone.utc
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class RobotSeedConfig:
    database_url: str = DEFAULT_DATABASE_URL

    robots: int = 10
    seed: int = 42

    robot_type: str = "mixed"

    sensors_per_robot: int = 8
    actuators_per_robot: int = 6

    dry_run: bool = False

    output: str | None = None

    def validate(self) -> None:
        if self.robots < 1 or self.robots > 10000:
            raise ValueError(
                "--robots must be between 1 and 10000"
            )

        if self.sensors_per_robot < 0:
            raise ValueError(
                "--sensors-per-robot cannot be negative"
            )

        if self.actuators_per_robot < 0:
            raise ValueError(
                "--actuators-per-robot cannot be negative"
            )

        allowed = {
            "mixed",
            "mobile_robot",
            "robotic_arm",
            "humanoid",
            "quadruped",
        }

        if self.robot_type not in allowed:
            raise ValueError(
                f"Invalid robot type: {self.robot_type}"
            )


# ============================================================
# Robot Specifications
# ============================================================

ROBOT_PROFILES = {
    "mobile_robot": {
        "model_name": "SIM-AMR-100",
        "manufacturer": "Simulation Robotics Lab",
        "weight_kg": 85.0,
        "payload_capacity_kg": 120.0,
        "degrees_of_freedom": 3,
        "max_linear_speed_m_s": 1.5,
        "max_angular_speed_rad_s": 1.2,
        "battery_capacity_kwh": 4.8,
        "operating_environment": "warehouse",
        "navigation_supported": True,
        "manipulation_supported": False,
        "sensor_types": [
            "camera",
            "lidar",
            "imu",
            "encoder",
            "proximity",
            "temperature",
            "vibration",
            "battery_monitor",
        ],
        "actuator_types": [
            "left_wheel_motor",
            "right_wheel_motor",
            "steering_motor",
            "brake_actuator",
        ],
    },

    "robotic_arm": {
        "model_name": "SIM-ARM-600",
        "manufacturer": "Simulation Robotics Lab",
        "weight_kg": 65.0,
        "payload_capacity_kg": 12.0,
        "degrees_of_freedom": 6,
        "max_linear_speed_m_s": 0.5,
        "max_angular_speed_rad_s": 2.0,
        "battery_capacity_kwh": 0.0,
        "operating_environment": "factory",
        "navigation_supported": False,
        "manipulation_supported": True,
        "sensor_types": [
            "camera",
            "depth_camera",
            "force_torque",
            "joint_encoder",
            "temperature",
            "vibration",
            "proximity",
            "current",
        ],
        "actuator_types": [
            "joint_1_motor",
            "joint_2_motor",
            "joint_3_motor",
            "joint_4_motor",
            "joint_5_motor",
            "joint_6_motor",
            "gripper",
        ],
    },

    "humanoid": {
        "model_name": "SIM-HUM-200",
        "manufacturer": "Simulation Robotics Lab",
        "weight_kg": 72.0,
        "payload_capacity_kg": 8.0,
        "degrees_of_freedom": 28,
        "max_linear_speed_m_s": 1.0,
        "max_angular_speed_rad_s": 1.5,
        "battery_capacity_kwh": 3.2,
        "operating_environment": "research_lab",
        "navigation_supported": True,
        "manipulation_supported": True,
        "sensor_types": [
            "stereo_camera",
            "depth_camera",
            "imu",
            "joint_encoder",
            "force_torque",
            "tactile",
            "temperature",
            "battery_monitor",
        ],
        "actuator_types": [
            "left_leg_joint",
            "right_leg_joint",
            "left_arm_joint",
            "right_arm_joint",
            "torso_joint",
            "neck_joint",
            "left_gripper",
            "right_gripper",
        ],
    },

    "quadruped": {
        "model_name": "SIM-QUAD-400",
        "manufacturer": "Simulation Robotics Lab",
        "weight_kg": 35.0,
        "payload_capacity_kg": 10.0,
        "degrees_of_freedom": 12,
        "max_linear_speed_m_s": 2.0,
        "max_angular_speed_rad_s": 1.8,
        "battery_capacity_kwh": 1.8,
        "operating_environment": "inspection_site",
        "navigation_supported": True,
        "manipulation_supported": False,
        "sensor_types": [
            "camera",
            "depth_camera",
            "lidar",
            "imu",
            "joint_encoder",
            "temperature",
            "vibration",
            "battery_monitor",
        ],
        "actuator_types": [
            "front_left_hip",
            "front_left_knee",
            "front_right_hip",
            "front_right_knee",
            "rear_left_hip",
            "rear_left_knee",
            "rear_right_hip",
            "rear_right_knee",
        ],
    },
}


# ============================================================
# Helpers
# ============================================================

def stable_id(
    namespace: str,
    value: str,
) -> str:
    return str(
        uuid.uuid5(
            ROBOT_NAMESPACE,
            f"{namespace}:{value}",
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


def round_float(
    value: float,
    digits: int = 4,
) -> float:
    return round(float(value), digits)


def classify_health(
    health_score: float,
) -> str:
    if health_score < 0.4:
        return "critical"

    if health_score < 0.65:
        return "degraded"

    if health_score < 0.85:
        return "warning"

    return "healthy"


def robot_operating_status(
    health_score: float,
    battery_soc: float,
) -> str:
    if health_score < 0.4:
        return "maintenance_required"

    if battery_soc < 15:
        return "charging_required"

    if health_score < 0.65:
        return "degraded"

    return "idle"


# ============================================================
# Robot Seeder
# ============================================================

class RobotSeeder:
    ROBOT_TYPES = list(ROBOT_PROFILES.keys())

    def __init__(
        self,
        session: Session | None,
        config: RobotSeedConfig,
    ) -> None:
        self.session = session
        self.config = config

        self.rng = random.Random(
            config.seed
        )

        self.counts = Counter()

    def upsert(
        self,
        model_class: type,
        identifier: str,
        values: dict[str, Any],
    ) -> Any:
        if self.session is None:
            raise RuntimeError(
                "Database session is unavailable"
            )

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

        self.counts[
            model_class.__tablename__
        ] += 1

        return instance

    def choose_robot_type(
        self,
        index: int,
    ) -> str:
        if self.config.robot_type != "mixed":
            return self.config.robot_type

        return self.ROBOT_TYPES[
            index % len(self.ROBOT_TYPES)
        ]

    def robot_code(
        self,
        index: int,
    ) -> str:
        # Shared code format with seed_database.py.
        return f"DEMO-ROBOT-{index + 1:03d}"

    def build_robot_data(
        self,
        index: int,
    ) -> dict[str, Any]:
        robot_type = self.choose_robot_type(index)
        profile = ROBOT_PROFILES[robot_type]

        robot_code = self.robot_code(index)

        health = round_float(
            self.rng.uniform(0.55, 1.0)
        )

        if profile["battery_capacity_kwh"] > 0:
            battery_soc = round_float(
                self.rng.uniform(20, 100)
            )
        else:
            battery_soc = 100.0

        operating_hours = round_float(
            self.rng.uniform(25, 2500)
        )

        status = robot_operating_status(
            health,
            battery_soc,
        )

        configuration = {
            "simulation_only": True,
            "physical_actuation_enabled": False,
            "hardware_connection_enabled": False,
            "remote_control_enabled": False,

            "robot_type": robot_type,
            "model_name": profile["model_name"],

            "physical_properties": {
                "weight_kg": profile["weight_kg"],
                "payload_capacity_kg": (
                    profile["payload_capacity_kg"]
                ),
                "degrees_of_freedom": (
                    profile["degrees_of_freedom"]
                ),
            },

            "motion_limits": {
                "max_linear_speed_m_s": (
                    profile["max_linear_speed_m_s"]
                ),
                "max_angular_speed_rad_s": (
                    profile["max_angular_speed_rad_s"]
                ),
                "max_joint_velocity_rad_s": 2.0,
                "max_joint_torque_nm": 100.0,
            },

            "battery": {
                "capacity_kwh": (
                    profile["battery_capacity_kwh"]
                ),
                "soc_percent": battery_soc,
                "thermal_monitoring_enabled": True,
            },

            "capabilities": {
                "navigation": (
                    profile["navigation_supported"]
                ),
                "manipulation": (
                    profile["manipulation_supported"]
                ),
                "perception": True,
                "predictive_maintenance": True,
                "digital_twin": True,
                "ai_copilot": True,
            },

            "safety": {
                "emergency_stop_required": True,
                "safety_interlocks_required": True,
                "collision_monitoring_enabled": True,
                "human_proximity_monitoring_enabled": True,
                "human_approval_required": True,
                "automatic_safety_override": False,
                "safety_certified": False,
            },

            "environment": (
                profile["operating_environment"]
            ),
        }

        return {
            "id": stable_id(
                "robot",
                robot_code,
            ),
            "robot_code": robot_code,
            "name": (
                f"Demo {robot_type.replace('_', ' ').title()} "
                f"{index + 1}"
            ),
            "robot_type": robot_type,
            "manufacturer": profile["manufacturer"],
            "model_name": profile["model_name"],
            "firmware_version": "sim-1.0.0",
            "operating_mode": "simulation",
            "status": status,
            "battery_soc_percent": battery_soc,
            "health_score": health,
            "operating_hours": operating_hours,
            "location_name": (
                profile["operating_environment"]
            ),
            "physical_control_enabled": False,
            "configuration": configuration,
            "created_at": BASE_TIME,
        }

    def seed_robots(self) -> list[Robot]:
        robots = []

        for index in range(self.config.robots):
            data = self.build_robot_data(index)

            identifier = data.pop("id")

            robot = self.upsert(
                Robot,
                identifier,
                data,
            )

            robots.append(robot)

            self.counts[
                f"type_{robot.robot_type}"
            ] += 1

            self.counts[
                f"status_{robot.status}"
            ] += 1

        self.session.flush()

        return robots

    def sensor_health(self) -> float:
        return round_float(
            self.rng.uniform(0.75, 1.0)
        )

    def seed_sensors(
        self,
        robots: list[Robot],
    ) -> None:
        for robot in robots:
            profile = ROBOT_PROFILES[
                robot.robot_type
            ]

            sensor_types = profile[
                "sensor_types"
            ]

            for index in range(
                self.config.sensors_per_robot
            ):
                sensor_type = sensor_types[
                    index % len(sensor_types)
                ]

                sensor_code = (
                    f"{robot.robot_code}-"
                    f"{sensor_type}-{index:02d}"
                )

                health = self.sensor_health()

                self.upsert(
                    RobotSensor,
                    stable_id(
                        "sensor",
                        sensor_code,
                    ),
                    {
                        "robot_id": robot.id,
                        "sensor_code": sensor_code,
                        "sensor_type": sensor_type,
                        "name": (
                            sensor_type.replace(
                                "_", " "
                            ).title()
                        ),
                        "status": classify_health(
                            health
                        ),
                        "health_score": health,
                        "sample_rate_hz": (
                            30.0
                            if "camera" in sensor_type
                            else 10.0
                        ),
                        "calibration_required": (
                            health < 0.85
                        ),
                        "metadata_json": {
                            "synthetic": True,
                            "connected_to_hardware": False,
                            "sensor_index": index,
                            "simulation_only": True,
                        },
                    },
                )

        self.session.flush()

    def seed_actuators(
        self,
        robots: list[Robot],
    ) -> None:
        for robot in robots:
            profile = ROBOT_PROFILES[
                robot.robot_type
            ]

            actuator_types = profile[
                "actuator_types"
            ]

            for index in range(
                self.config.actuators_per_robot
            ):
                actuator_type = actuator_types[
                    index % len(actuator_types)
                ]

                actuator_code = (
                    f"{robot.robot_code}-"
                    f"{actuator_type}-{index:02d}"
                )

                temperature = round_float(
                    self.rng.uniform(28, 65)
                )

                status = (
                    "warning"
                    if temperature >= 55
                    else "healthy"
                )

                self.upsert(
                    RobotActuator,
                    stable_id(
                        "actuator",
                        actuator_code,
                    ),
                    {
                        "robot_id": robot.id,
                        "actuator_code": actuator_code,
                        "actuator_type": actuator_type,
                        "name": (
                            actuator_type.replace(
                                "_", " "
                            ).title()
                        ),
                        "status": status,
                        "max_torque_nm": 100.0,
                        "max_velocity_rad_s": 2.0,
                        "temperature_c": temperature,
                        "control_enabled": False,
                    },
                )

        self.session.flush()

    def seed_audit_logs(
        self,
        robots: list[Robot],
    ) -> None:
        for robot in robots:
            audit_id = stable_id(
                "audit",
                robot.robot_code,
            )

            self.upsert(
                AuditLog,
                audit_id,
                {
                    "actor": "robot-demo-seeder",
                    "action": "synthetic_robot_seed",
                    "resource_type": "robot",
                    "resource_id": robot.id,
                    "details": {
                        "robot_code": robot.robot_code,
                        "robot_type": robot.robot_type,
                        "simulation_only": True,
                        "physical_actuation_allowed": False,
                        "human_approval_required": True,
                    },
                    "timestamp": BASE_TIME,
                },
            )

        self.session.flush()

    def seed_all(self) -> dict[str, Any]:
        robots = self.seed_robots()

        self.seed_sensors(robots)
        self.seed_actuators(robots)
        self.seed_audit_logs(robots)

        return {
            "robots": len(robots),
            "sensors_requested": (
                len(robots)
                * self.config.sensors_per_robot
            ),
            "actuators_requested": (
                len(robots)
                * self.config.actuators_per_robot
            ),
            "statistics": dict(self.counts),
            "fleet": [
                {
                    "robot_code": robot.robot_code,
                    "robot_type": robot.robot_type,
                    "model": robot.model_name,
                    "status": robot.status,
                    "health_score": robot.health_score,
                    "battery_soc_percent": (
                        robot.battery_soc_percent
                    ),
                    "physical_control_enabled": False,
                }
                for robot in robots
            ],
        }


# ============================================================
# Dry Run
# ============================================================

def dry_run_preview(
    config: RobotSeedConfig,
) -> dict[str, Any]:
    rng = random.Random(config.seed)

    robots = []

    robot_types = list(ROBOT_PROFILES)

    for index in range(config.robots):
        robot_type = (
            config.robot_type
            if config.robot_type != "mixed"
            else robot_types[index % len(robot_types)]
        )

        robots.append({
            "robot_code": (
                f"DEMO-ROBOT-{index + 1:03d}"
            ),
            "robot_type": robot_type,
            "model_name": (
                ROBOT_PROFILES[robot_type]["model_name"]
            ),
            "synthetic_health_score": round(
                rng.uniform(0.55, 1.0), 4
            ),
        })

    return {
        "status": "dry_run",
        "schema_version": SCHEMA_VERSION,
        "robots": config.robots,
        "sensors_requested": (
            config.robots
            * config.sensors_per_robot
        ),
        "actuators_requested": (
            config.robots
            * config.actuators_per_robot
        ),
        "database_modified": False,
        "simulation_only": True,
        "preview": robots[:20],
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

        os.replace(
            temporary,
            path,
        )

    finally:
        if temporary.exists():
            temporary.unlink()


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Seed synthetic Physical AI robot fleet"
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
        "--robots",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--robot-type",
        choices=[
            "mixed",
            "mobile_robot",
            "robotic_arm",
            "humanoid",
            "quadruped",
        ],
        default="mixed",
    )

    parser.add_argument(
        "--sensors-per-robot",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--actuators-per-robot",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--dry-run",
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

    config = RobotSeedConfig(
        database_url=args.database_url,
        robots=args.robots,
        seed=args.seed,
        robot_type=args.robot_type,
        sensors_per_robot=args.sensors_per_robot,
        actuators_per_robot=args.actuators_per_robot,
        dry_run=args.dry_run,
        output=args.output,
    )

    config.validate()

    if config.dry_run:
        report = dry_run_preview(config)

        if config.output:
            save_report(config.output, report)

        print(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            )
        )

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
                    "Run scripts/seed_database.py first."
                )

    engine = create_engine(
        config.database_url,
        pool_pre_ping=True,
        hide_parameters=True,
    )

    try:
        existing_tables = set(
            inspect(engine).get_table_names()
        )

        required_tables = {
            "demo_robots",
            "demo_robot_sensors",
            "demo_robot_actuators",
            "demo_audit_logs",
        }

        missing = required_tables - existing_tables

        if missing:
            raise RuntimeError(
                "Missing prerequisite tables: "
                + ", ".join(sorted(missing))
            )

        SessionLocal = sessionmaker(
            bind=engine,
            expire_on_commit=False,
            autoflush=False,
        )

        with SessionLocal() as session:
            with session.begin():
                seeder = RobotSeeder(
                    session,
                    config,
                )

                result = seeder.seed_all()

        report = {
            "status": "completed",
            "schema_version": SCHEMA_VERSION,
            "seed": config.seed,
            "database_modified": True,
            "simulation_only": True,
            "physical_execution_authorized": False,
            "result": result,
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
            )
        )

        LOGGER.info(
            "Synthetic robot fleet seeding completed"
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
            "Robot seeding failed: %s",
            exc,
        )

        sys.exit(1)
```


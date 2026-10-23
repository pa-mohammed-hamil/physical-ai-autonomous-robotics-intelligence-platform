#!/usr/bin/env python3
"""
Physical AI & Autonomous Robotics Intelligence Platform
File: scripts/generate_demo_telemetry.py

Generate synthetic robot telemetry for:
    - Robot health monitoring
    - Predictive maintenance
    - Anomaly detection
    - Failure prediction
    - Remaining Useful Life (RUL)
    - Robot safety evaluation
    - Sensor fusion demonstrations
    - AI agent dashboards
    - Simulation and offline model testing

Features:
    - Mobile robot, robotic arm, humanoid and quadruped profiles
    - Normal, warning, degraded and critical operating states
    - Joint position, velocity and torque
    - Motor, actuator and gearbox temperatures
    - Battery voltage, temperature and charge
    - Robot linear/angular motion and acceleration
    - Obstacle and human proximity
    - Collision and workspace events
    - Emergency-stop and interlock states
    - Sensor health and communication latency
    - Vibration, current, pressure and fault signals
    - Failure probability and synthetic RUL
    - Correlated degradation and anomaly injection
    - Deterministic random seeds
    - Multiple robots and time-series records
    - CSV, JSON, JSONL and Parquet output
    - Summary statistics and dataset manifest
    - Compatibility with evaluate_robot_safety.py

Install:
    pip install numpy pandas

Optional:
    pip install pyarrow

Examples:
    python scripts/generate_demo_telemetry.py

    python scripts/generate_demo_telemetry.py \
        --samples 10000 \
        --robots 5 \
        --output data/demo/robot_telemetry.csv

    python scripts/generate_demo_telemetry.py \
        --robot-type robotic_arm \
        --samples 5000 \
        --anomaly-rate 0.08 \
        --failure-rate 0.03

    python scripts/generate_demo_telemetry.py \
        --robot-type mixed \
        --samples 20000 \
        --format parquet \
        --seed 42

Safety:
    All generated telemetry is synthetic.
    It does not represent validated robot behavior.
    No hardware, actuators or robot controllers are accessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("generate_demo_telemetry")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

SCHEMA_VERSION = "1.0.0"


# ============================================================
# Enumerations
# ============================================================

class RobotType(str, Enum):
    MOBILE_ROBOT = "mobile_robot"
    ROBOTIC_ARM = "robotic_arm"
    HUMANOID = "humanoid"
    QUADRUPED = "quadruped"
    MIXED = "mixed"


class OperatingState(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class FaultType(str, Enum):
    NONE = "none"
    MOTOR_OVERHEATING = "motor_overheating"
    EXCESSIVE_VIBRATION = "excessive_vibration"
    BATTERY_FAULT = "battery_fault"
    SENSOR_FAILURE = "sensor_failure"
    COMMUNICATION_FAULT = "communication_fault"
    ACTUATOR_OVERLOAD = "actuator_overload"
    COLLISION_EVENT = "collision_event"
    JOINT_LIMIT_VIOLATION = "joint_limit_violation"
    GEARBOX_DEGRADATION = "gearbox_degradation"


# ============================================================
# Configuration
# ============================================================

@dataclass
class RobotProfile:
    robot_type: str

    nominal_battery_voltage_v: float
    nominal_motor_temperature_c: float
    nominal_motor_current_a: float

    nominal_joint_torque_nm: float
    nominal_joint_velocity_rad_s: float

    nominal_linear_speed_m_s: float
    nominal_angular_speed_rad_s: float

    nominal_vibration_mm_s: float

    initial_rul_hours: float

    joint_count: int


ROBOT_PROFILES = {
    RobotType.MOBILE_ROBOT.value: RobotProfile(
        robot_type="mobile_robot",
        nominal_battery_voltage_v=48.0,
        nominal_motor_temperature_c=42.0,
        nominal_motor_current_a=8.0,
        nominal_joint_torque_nm=25.0,
        nominal_joint_velocity_rad_s=0.8,
        nominal_linear_speed_m_s=0.6,
        nominal_angular_speed_rad_s=0.3,
        nominal_vibration_mm_s=1.2,
        initial_rul_hours=1200.0,
        joint_count=4,
    ),
    RobotType.ROBOTIC_ARM.value: RobotProfile(
        robot_type="robotic_arm",
        nominal_battery_voltage_v=48.0,
        nominal_motor_temperature_c=45.0,
        nominal_motor_current_a=12.0,
        nominal_joint_torque_nm=45.0,
        nominal_joint_velocity_rad_s=0.7,
        nominal_linear_speed_m_s=0.0,
        nominal_angular_speed_rad_s=0.0,
        nominal_vibration_mm_s=1.5,
        initial_rul_hours=1600.0,
        joint_count=6,
    ),
    RobotType.HUMANOID.value: RobotProfile(
        robot_type="humanoid",
        nominal_battery_voltage_v=48.0,
        nominal_motor_temperature_c=47.0,
        nominal_motor_current_a=15.0,
        nominal_joint_torque_nm=55.0,
        nominal_joint_velocity_rad_s=0.9,
        nominal_linear_speed_m_s=0.5,
        nominal_angular_speed_rad_s=0.4,
        nominal_vibration_mm_s=1.8,
        initial_rul_hours=1000.0,
        joint_count=24,
    ),
    RobotType.QUADRUPED.value: RobotProfile(
        robot_type="quadruped",
        nominal_battery_voltage_v=48.0,
        nominal_motor_temperature_c=44.0,
        nominal_motor_current_a=13.0,
        nominal_joint_torque_nm=40.0,
        nominal_joint_velocity_rad_s=1.0,
        nominal_linear_speed_m_s=0.8,
        nominal_angular_speed_rad_s=0.5,
        nominal_vibration_mm_s=1.7,
        initial_rul_hours=1100.0,
        joint_count=12,
    ),
}


@dataclass
class TelemetryGeneratorConfig:
    samples: int = 1000
    robots: int = 3

    robot_type: str = RobotType.MIXED.value

    sample_interval_seconds: float = 1.0

    anomaly_rate: float = 0.05
    failure_rate: float = 0.02

    degradation_rate: float = 0.0002

    noise_level: float = 0.05

    seed: int = 42

    output_path: str = (
        "data/demo/robot_telemetry.csv"
    )

    output_format: str = "csv"

    start_time: str = "2026-01-01T00:00:00+00:00"

    include_fault_events: bool = True
    include_safety_events: bool = True

    def validate(self) -> None:
        if self.samples < 1:
            raise ValueError(
                "samples must be positive"
            )

        if self.robots < 1:
            raise ValueError(
                "robots must be positive"
            )

        if self.sample_interval_seconds <= 0:
            raise ValueError(
                "sample interval must be positive"
            )

        for name in (
            "anomaly_rate",
            "failure_rate",
        ):
            value = getattr(self, name)

            if not 0 <= value <= 1:
                raise ValueError(
                    f"{name} must be in [0, 1]"
                )

        if (
            self.anomaly_rate
            + self.failure_rate
            > 1
        ):
            raise ValueError(
                "Combined anomaly and failure rates "
                "cannot exceed 1"
            )

        if self.degradation_rate < 0:
            raise ValueError(
                "degradation_rate cannot be negative"
            )

        if self.noise_level < 0:
            raise ValueError(
                "noise_level cannot be negative"
            )

        if self.robot_type not in {
            item.value
            for item in RobotType
        }:
            raise ValueError(
                "Unsupported robot type"
            )

        if self.output_format not in {
            "csv",
            "json",
            "jsonl",
            "parquet",
        }:
            raise ValueError(
                "Unsupported output format"
            )

        datetime.fromisoformat(
            self.start_time.replace(
                "Z",
                "+00:00",
            )
        )


# ============================================================
# Utilities
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(
        minimum,
        min(maximum, value),
    )


def sha256_file(
    path: str | Path,
) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def atomic_json_write(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

            stream.flush()
            os.fsync(stream.fileno())

            temporary = Path(
                stream.name
            )

        os.replace(
            temporary,
            path,
        )

    finally:
        if temporary and temporary.exists():
            temporary.unlink()


# ============================================================
# Robot State
# ============================================================

@dataclass
class RobotSimulationState:
    robot_id: str
    robot_type: str

    operating_hours: float = 0.0
    degradation: float = 0.0

    battery_soc_percent: float = 100.0

    motor_temperature_c: float = 40.0

    sensor_health_score: float = 1.0

    remaining_useful_life_hours: float = 1000.0

    failure_probability: float = 0.0

    fault_active: bool = False
    fault_type: str = FaultType.NONE.value

    last_operating_state: str = (
        OperatingState.NORMAL.value
    )


# ============================================================
# Synthetic Telemetry Generator
# ============================================================

class RobotTelemetryGenerator:
    FAULT_TYPES = [
        FaultType.MOTOR_OVERHEATING,
        FaultType.EXCESSIVE_VIBRATION,
        FaultType.BATTERY_FAULT,
        FaultType.SENSOR_FAILURE,
        FaultType.COMMUNICATION_FAULT,
        FaultType.ACTUATOR_OVERLOAD,
        FaultType.COLLISION_EVENT,
        FaultType.JOINT_LIMIT_VIOLATION,
        FaultType.GEARBOX_DEGRADATION,
    ]

    def __init__(
        self,
        config: TelemetryGeneratorConfig,
    ) -> None:
        config.validate()

        self.config = config

        self.rng = np.random.default_rng(
            config.seed
        )

        self.start_time = datetime.fromisoformat(
            config.start_time.replace(
                "Z",
                "+00:00",
            )
        )

        if self.start_time.tzinfo is None:
            self.start_time = (
                self.start_time.replace(
                    tzinfo=timezone.utc
                )
            )

        self.robot_states = (
            self._initialize_robots()
        )

    def _initialize_robots(
        self,
    ) -> list[RobotSimulationState]:
        states = []

        available_types = list(
            ROBOT_PROFILES.keys()
        )

        for index in range(
            self.config.robots
        ):
            if (
                self.config.robot_type
                == RobotType.MIXED.value
            ):
                robot_type = available_types[
                    index % len(available_types)
                ]

            else:
                robot_type = (
                    self.config.robot_type
                )

            profile = ROBOT_PROFILES[
                robot_type
            ]

            states.append(
                RobotSimulationState(
                    robot_id=(
                        f"{robot_type}-{index + 1:03d}"
                    ),
                    robot_type=robot_type,
                    battery_soc_percent=float(
                        self.rng.uniform(
                            75,
                            100,
                        )
                    ),
                    motor_temperature_c=(
                        profile.nominal_motor_temperature_c
                    ),
                    remaining_useful_life_hours=(
                        profile.initial_rul_hours
                    ),
                )
            )

        return states

    def _noise(
        self,
        scale: float = 1.0,
    ) -> float:
        return float(
            self.rng.normal(
                0.0,
                self.config.noise_level * scale,
            )
        )

    def _positive_noise(
        self,
        scale: float = 1.0,
    ) -> float:
        return abs(
            self._noise(scale)
        )

    def _update_degradation(
        self,
        state: RobotSimulationState,
        profile: RobotProfile,
    ) -> None:
        interval_hours = (
            self.config.sample_interval_seconds
            / 3600.0
        )

        state.operating_hours += (
            interval_hours
        )

        state.degradation = clamp(
            state.degradation
            + self.config.degradation_rate
            * interval_hours,
            0.0,
            1.0,
        )

        state.remaining_useful_life_hours = max(
            0.0,
            profile.initial_rul_hours
            - state.operating_hours
            - state.degradation
            * profile.initial_rul_hours,
        )

        state.battery_soc_percent = clamp(
            state.battery_soc_percent
            - interval_hours * 1.5,
            0.0,
            100.0,
        )

        # Synthetic battery recharge event.
        if state.battery_soc_percent < 12.0:
            state.battery_soc_percent = float(
                self.rng.uniform(
                    85,
                    100,
                )
            )

    def _choose_fault(
        self,
    ) -> FaultType:
        return self.FAULT_TYPES[
            int(
                self.rng.integers(
                    0,
                    len(self.FAULT_TYPES),
                )
            )
        ]

    def _determine_event(
        self,
    ) -> tuple[OperatingState, FaultType]:
        probability = float(
            self.rng.random()
        )

        if (
            probability
            < self.config.failure_rate
        ):
            return (
                OperatingState.CRITICAL,
                self._choose_fault(),
            )

        if (
            probability
            < self.config.failure_rate
            + self.config.anomaly_rate
        ):
            state = (
                OperatingState.WARNING
                if self.rng.random() < 0.5
                else OperatingState.DEGRADED
            )

            return (
                state,
                self._choose_fault(),
            )

        return (
            OperatingState.NORMAL,
            FaultType.NONE,
        )

    def _generate_base_telemetry(
        self,
        state: RobotSimulationState,
        profile: RobotProfile,
        timestamp: datetime,
        sample_index: int,
    ) -> dict[str, Any]:
        degradation = state.degradation

        load_factor = float(
            self.rng.uniform(
                0.5,
                1.3,
            )
        )

        motor_temperature = (
            profile.nominal_motor_temperature_c
            + 15.0 * degradation
            + 5.0 * (load_factor - 1.0)
            + self._noise(10.0)
        )

        motor_current = (
            profile.nominal_motor_current_a
            * load_factor
            * (1.0 + degradation)
            + self._noise(2.0)
        )

        vibration = (
            profile.nominal_vibration_mm_s
            * (1.0 + 2.0 * degradation)
            + self._noise(1.0)
        )

        torque = (
            profile.nominal_joint_torque_nm
            * load_factor
            * (1.0 + 0.2 * degradation)
            + self._noise(3.0)
        )

        velocity = (
            profile.nominal_joint_velocity_rad_s
            * float(
                self.rng.uniform(
                    0.6,
                    1.3,
                )
            )
        )

        joint_position = float(
            self.rng.uniform(
                -2.5,
                2.5,
            )
        )

        linear_speed = max(
            0.0,
            profile.nominal_linear_speed_m_s
            + self._noise(1.0),
        )

        angular_speed = max(
            0.0,
            profile.nominal_angular_speed_rad_s
            + self._noise(1.0),
        )

        linear_acceleration = float(
            self.rng.uniform(
                0.05,
                1.0,
            )
        )

        battery_voltage = (
            profile.nominal_battery_voltage_v
            * (
                0.85
                + 0.15
                * state.battery_soc_percent
                / 100.0
            )
            + self._noise(2.0)
        )

        battery_temperature = (
            30.0
            + 8.0 * load_factor
            + self._noise(5.0)
        )

        sensor_health = clamp(
            1.0
            - 0.15 * degradation
            - self._positive_noise(0.3),
            0.0,
            1.0,
        )

        telemetry_age_ms = max(
            0.0,
            30.0
            + self._noise(50.0),
        )

        communication_latency_ms = max(
            0.0,
            20.0
            + self._noise(40.0),
        )

        model_confidence = clamp(
            0.95
            - 0.2 * degradation
            - self._positive_noise(0.2),
            0.0,
            1.0,
        )

        failure_probability = clamp(
            0.02
            + 0.65 * degradation
            + 0.05 * load_factor
            + self._noise(0.2),
            0.0,
            1.0,
        )

        obstacle_distance = float(
            self.rng.uniform(
                1.0,
                10.0,
            )
        )

        human_distance = float(
            self.rng.uniform(
                2.0,
                12.0,
            )
        )

        telemetry = {
            "timestamp": timestamp.isoformat(),
            "sample_index": sample_index,
            "robot_id": state.robot_id,
            "robot_type": state.robot_type,
            "simulation_only": True,

            "operating_hours": round(
                state.operating_hours,
                6,
            ),
            "degradation_score": round(
                degradation,
                6,
            ),
            "load_factor": round(
                load_factor,
                4,
            ),

            "joint_position_rad": round(
                joint_position,
                5,
            ),
            "joint_velocity_rad_s": round(
                velocity,
                5,
            ),
            "joint_torque_nm": round(
                max(0.0, torque),
                5,
            ),
            "joint_count": profile.joint_count,

            "motor_temperature_c": round(
                motor_temperature,
                4,
            ),
            "motor_current_a": round(
                max(0.0, motor_current),
                4,
            ),

            "vibration_mm_s": round(
                max(0.0, vibration),
                5,
            ),

            "gearbox_temperature_c": round(
                motor_temperature - 5.0,
                4,
            ),

            "actuator_temperature_c": round(
                motor_temperature - 2.0,
                4,
            ),

            "battery_voltage_v": round(
                battery_voltage,
                4,
            ),
            "battery_temperature_c": round(
                battery_temperature,
                4,
            ),
            "battery_soc_percent": round(
                state.battery_soc_percent,
                4,
            ),

            "linear_speed_m_s": round(
                linear_speed,
                5,
            ),
            "angular_speed_rad_s": round(
                angular_speed,
                5,
            ),
            "linear_acceleration_m_s2": round(
                linear_acceleration,
                5,
            ),

            "obstacle_distance_m": round(
                obstacle_distance,
                4,
            ),
            "human_distance_m": round(
                human_distance,
                4,
            ),

            "collision_force_n": 0.0,
            "collision_detected": False,
            "workspace_violation": False,

            "emergency_stop_active": False,
            "safety_interlock_ok": True,

            "sensor_health_score": round(
                sensor_health,
                5,
            ),
            "telemetry_age_ms": round(
                telemetry_age_ms,
                4,
            ),
            "communication_latency_ms": round(
                communication_latency_ms,
                4,
            ),

            "model_confidence": round(
                model_confidence,
                5,
            ),

            "failure_probability": round(
                failure_probability,
                5,
            ),

            "remaining_useful_life_hours": round(
                state.remaining_useful_life_hours,
                5,
            ),

            "fault_active": False,
            "fault_type": FaultType.NONE.value,

            "operating_state": (
                OperatingState.NORMAL.value
            ),

            "anomaly_label": 0,
            "failure_label": 0,
            "safety_violation_label": 0,
        }

        return telemetry

    def _inject_fault(
        self,
        telemetry: dict[str, Any],
        fault: FaultType,
        severity: OperatingState,
    ) -> None:
        if fault == FaultType.NONE:
            return

        multiplier = (
            1.5
            if severity == OperatingState.WARNING
            else 2.0
            if severity == OperatingState.DEGRADED
            else 3.0
        )

        telemetry["fault_active"] = True
        telemetry["fault_type"] = fault.value
        telemetry["operating_state"] = severity.value
        telemetry["anomaly_label"] = 1

        if severity == OperatingState.CRITICAL:
            telemetry["failure_label"] = 1

        if fault == FaultType.MOTOR_OVERHEATING:
            telemetry["motor_temperature_c"] = round(
                55.0 + 15.0 * multiplier,
                4,
            )

            telemetry["actuator_temperature_c"] = round(
                50.0 + 12.0 * multiplier,
                4,
            )

        elif fault == FaultType.EXCESSIVE_VIBRATION:
            telemetry["vibration_mm_s"] = round(
                3.0 * multiplier,
                4,
            )

        elif fault == FaultType.BATTERY_FAULT:
            telemetry["battery_temperature_c"] = round(
                35.0 + 12.0 * multiplier,
                4,
            )

            telemetry["battery_voltage_v"] = round(
                max(
                    10.0,
                    35.0 - 5.0 * multiplier,
                ),
                4,
            )

        elif fault == FaultType.SENSOR_FAILURE:
            telemetry["sensor_health_score"] = round(
                max(
                    0.0,
                    0.8 - 0.25 * multiplier,
                ),
                5,
            )

            telemetry["model_confidence"] = round(
                max(
                    0.0,
                    0.75 - 0.2 * multiplier,
                ),
                5,
            )

        elif fault == FaultType.COMMUNICATION_FAULT:
            telemetry["telemetry_age_ms"] = round(
                250.0 * multiplier,
                4,
            )

            telemetry["communication_latency_ms"] = round(
                180.0 * multiplier,
                4,
            )

        elif fault == FaultType.ACTUATOR_OVERLOAD:
            telemetry["joint_torque_nm"] = round(
                45.0 * multiplier,
                4,
            )

            telemetry["motor_current_a"] = round(
                15.0 * multiplier,
                4,
            )

        elif fault == FaultType.COLLISION_EVENT:
            telemetry["collision_detected"] = True

            telemetry["collision_force_n"] = round(
                25.0 * multiplier,
                4,
            )

            telemetry["obstacle_distance_m"] = round(
                max(
                    0.0,
                    0.5 - 0.1 * multiplier,
                ),
                4,
            )

        elif fault == FaultType.JOINT_LIMIT_VIOLATION:
            telemetry["joint_position_rad"] = round(
                2.5 + 0.5 * multiplier,
                5,
            )

            telemetry["joint_velocity_rad_s"] = round(
                0.9 * multiplier,
                5,
            )

        elif fault == FaultType.GEARBOX_DEGRADATION:
            telemetry["gearbox_temperature_c"] = round(
                45.0 + 12.0 * multiplier,
                4,
            )

            telemetry["vibration_mm_s"] = round(
                2.0 * multiplier,
                4,
            )

        telemetry["failure_probability"] = round(
            clamp(
                float(
                    telemetry["failure_probability"]
                )
                + 0.15 * multiplier,
                0.0,
                1.0,
            ),
            5,
        )

        if severity == OperatingState.CRITICAL:
            telemetry["remaining_useful_life_hours"] = round(
                min(
                    float(
                        telemetry[
                            "remaining_useful_life_hours"
                        ]
                    ),
                    12.0,
                ),
                5,
            )

    def _inject_safety_event(
        self,
        telemetry: dict[str, Any],
        severity: OperatingState,
    ) -> None:
        if not self.config.include_safety_events:
            return

        if severity != OperatingState.CRITICAL:
            return

        if self.rng.random() < 0.3:
            telemetry["human_distance_m"] = round(
                float(
                    self.rng.uniform(
                        0.2,
                        1.2,
                    )
                ),
                4,
            )

        if self.rng.random() < 0.2:
            telemetry["workspace_violation"] = True

        if self.rng.random() < 0.15:
            telemetry["emergency_stop_active"] = True

        if self.rng.random() < 0.1:
            telemetry["safety_interlock_ok"] = False

    @staticmethod
    def _label_safety_violation(
        telemetry: dict[str, Any],
    ) -> None:
        # These are illustrative demo limits, not
        # certified safety thresholds.
        violations = [
            telemetry["joint_position_rad"] < -3.14,
            telemetry["joint_position_rad"] > 3.14,
            telemetry["joint_velocity_rad_s"] > 2.0,
            telemetry["joint_torque_nm"] > 100.0,
            telemetry["motor_temperature_c"] > 80.0,
            telemetry["battery_temperature_c"] > 60.0,
            telemetry["battery_voltage_v"] < 20.0,
            telemetry["battery_voltage_v"] > 60.0,
            telemetry["battery_soc_percent"] < 10.0,
            telemetry["linear_speed_m_s"] > 1.5,
            telemetry["angular_speed_rad_s"] > 2.0,
            telemetry["linear_acceleration_m_s2"] > 2.0,
            telemetry["obstacle_distance_m"] < 0.5,
            telemetry["human_distance_m"] < 1.5,
            telemetry["collision_force_n"] > 50.0,
            telemetry["collision_detected"],
            telemetry["workspace_violation"],
            telemetry["emergency_stop_active"],
            not telemetry["safety_interlock_ok"],
            telemetry["sensor_health_score"] < 0.7,
            telemetry["telemetry_age_ms"] > 500.0,
            telemetry["communication_latency_ms"] > 300.0,
            telemetry["failure_probability"] > 0.7,
            telemetry["remaining_useful_life_hours"] < 24.0,
        ]

        telemetry["safety_violation_label"] = int(
            any(violations)
        )

    def generate(
        self,
    ) -> pd.DataFrame:
        records = []

        LOGGER.info(
            "Generating %d samples for %d robots",
            self.config.samples,
            self.config.robots,
        )

        for sample_index in range(
            self.config.samples
        ):
            robot_index = (
                sample_index
                % len(self.robot_states)
            )

            state = self.robot_states[
                robot_index
            ]

            profile = ROBOT_PROFILES[
                state.robot_type
            ]

            robot_step = (
                sample_index
                // len(self.robot_states)
            )

            timestamp = (
                self.start_time
                + timedelta(
                    seconds=(
                        robot_step
                        * self.config.sample_interval_seconds
                    )
                )
            )

            self._update_degradation(
                state,
                profile,
            )

            telemetry = self._generate_base_telemetry(
                state,
                profile,
                timestamp,
                sample_index,
            )

            severity, fault = (
                self._determine_event()
            )

            if self.config.include_fault_events:
                self._inject_fault(
                    telemetry,
                    fault,
                    severity,
                )

            self._inject_safety_event(
                telemetry,
                severity,
            )

            self._label_safety_violation(
                telemetry
            )

            state.fault_active = bool(
                telemetry["fault_active"]
            )

            state.fault_type = str(
                telemetry["fault_type"]
            )

            state.last_operating_state = str(
                telemetry["operating_state"]
            )

            state.sensor_health_score = float(
                telemetry["sensor_health_score"]
            )

            state.failure_probability = float(
                telemetry["failure_probability"]
            )

            records.append(
                telemetry
            )

        dataframe = pd.DataFrame(
            records
        )

        LOGGER.info(
            "Generated telemetry shape: %s",
            dataframe.shape,
        )

        return dataframe


# ============================================================
# Dataset Exporter
# ============================================================

class TelemetryDatasetExporter:
    @staticmethod
    def export(
        dataframe: pd.DataFrame,
        path: str | Path,
        output_format: str,
    ) -> Path:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if output_format == "csv":
            dataframe.to_csv(
                path,
                index=False,
            )

        elif output_format == "json":
            dataframe.to_json(
                path,
                orient="records",
                indent=2,
                force_ascii=False,
            )

        elif output_format == "jsonl":
            dataframe.to_json(
                path,
                orient="records",
                lines=True,
                force_ascii=False,
            )

        elif output_format == "parquet":
            dataframe.to_parquet(
                path,
                index=False,
            )

        else:
            raise ValueError(
                f"Unsupported format: {output_format}"
            )

        return path


# ============================================================
# Summary Statistics
# ============================================================

class TelemetryStatistics:
    @staticmethod
    def summarize(
        dataframe: pd.DataFrame,
    ) -> dict[str, Any]:
        if dataframe.empty:
            raise ValueError(
                "Cannot summarize an empty dataset"
            )

        numeric_columns = [
            "motor_temperature_c",
            "motor_current_a",
            "vibration_mm_s",
            "battery_soc_percent",
            "joint_torque_nm",
            "linear_speed_m_s",
            "sensor_health_score",
            "failure_probability",
            "remaining_useful_life_hours",
        ]

        statistics = {}

        for column in numeric_columns:
            if column not in dataframe.columns:
                continue

            values = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            ).dropna()

            if values.empty:
                continue

            statistics[column] = {
                "mean": round(
                    float(values.mean()),
                    5,
                ),
                "minimum": round(
                    float(values.min()),
                    5,
                ),
                "maximum": round(
                    float(values.max()),
                    5,
                ),
                "std": round(
                    float(
                        values.std(ddof=0)
                    ),
                    5,
                ),
            }

        return {
            "sample_count": int(
                len(dataframe)
            ),
            "robot_count": int(
                dataframe["robot_id"].nunique()
            ),
            "robot_types": (
                dataframe["robot_type"]
                .value_counts()
                .to_dict()
            ),
            "operating_states": (
                dataframe["operating_state"]
                .value_counts()
                .to_dict()
            ),
            "fault_types": (
                dataframe["fault_type"]
                .value_counts()
                .to_dict()
            ),
            "anomaly_count": int(
                dataframe["anomaly_label"].sum()
            ),
            "failure_count": int(
                dataframe["failure_label"].sum()
            ),
            "safety_violation_count": int(
                dataframe["safety_violation_label"].sum()
            ),
            "anomaly_rate": round(
                float(
                    dataframe["anomaly_label"].mean()
                ),
                6,
            ),
            "failure_rate": round(
                float(
                    dataframe["failure_label"].mean()
                ),
                6,
            ),
            "safety_violation_rate": round(
                float(
                    dataframe[
                        "safety_violation_label"
                    ].mean()
                ),
                6,
            ),
            "numeric_statistics": statistics,
        }


# ============================================================
# Dataset Manifest
# ============================================================

def create_manifest(
    dataframe: pd.DataFrame,
    config: TelemetryGeneratorConfig,
    dataset_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": uuid.uuid4().hex,
        "dataset_name": "synthetic_robot_telemetry",
        "created_at": utc_now(),
        "simulation_only": True,
        "real_world_validated": False,
        "physical_execution_authorized": False,
        "dataset_path": str(
            dataset_path.resolve()
        ),
        "dataset_sha256": sha256_file(
            dataset_path
        ),
        "generator_config": asdict(
            config
        ),
        "columns": dataframe.columns.tolist(),
        "dtypes": {
            column: str(dtype)
            for column, dtype in dataframe.dtypes.items()
        },
        "statistics": TelemetryStatistics.summarize(
            dataframe
        ),
        "limitations": [
            "Synthetic telemetry does not reproduce all physical dynamics.",
            "Injected faults are simplified approximations.",
            "Failure probabilities are illustrative labels.",
            "RUL values are synthetic and not validated prognostics.",
            "Safety thresholds are illustrative and robot-specific validation is required.",
            "Not suitable for certifying physical robot safety.",
        ],
    }


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic Physical AI robot telemetry"
        )
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--robots",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--robot-type",
        choices=[
            item.value
            for item in RobotType
        ],
        default="mixed",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sample interval per robot in seconds",
    )

    parser.add_argument(
        "--anomaly-rate",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--failure-rate",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--degradation-rate",
        type=float,
        default=0.0002,
    )

    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        default="data/demo/robot_telemetry.csv",
    )

    parser.add_argument(
        "--format",
        choices=[
            "csv",
            "json",
            "jsonl",
            "parquet",
        ],
        default="csv",
    )

    parser.add_argument(
        "--start-time",
        default="2026-01-01T00:00:00+00:00",
    )

    parser.add_argument(
        "--no-fault-events",
        action="store_true",
    )

    parser.add_argument(
        "--no-safety-events",
        action="store_true",
    )

    parser.add_argument(
        "--summary-only",
        action="store_true",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    config = TelemetryGeneratorConfig(
        samples=args.samples,
        robots=args.robots,
        robot_type=args.robot_type,
        sample_interval_seconds=args.interval,
        anomaly_rate=args.anomaly_rate,
        failure_rate=args.failure_rate,
        degradation_rate=args.degradation_rate,
        noise_level=args.noise_level,
        seed=args.seed,
        output_path=args.output,
        output_format=args.format,
        start_time=args.start_time,
        include_fault_events=not args.no_fault_events,
        include_safety_events=not args.no_safety_events,
    )

    config.validate()

    generator = RobotTelemetryGenerator(
        config
    )

    dataframe = generator.generate()

    dataset_path = (
        TelemetryDatasetExporter.export(
            dataframe,
            config.output_path,
            config.output_format,
        )
    )

    manifest = create_manifest(
        dataframe,
        config,
        dataset_path,
    )

    manifest_path = dataset_path.with_name(
        dataset_path.stem + "_manifest.json"
    )

    atomic_json_write(
        manifest_path,
        manifest,
    )

    result = {
        "status": "completed",
        "dataset_path": str(
            dataset_path
        ),
        "manifest_path": str(
            manifest_path
        ),
        "sample_count": len(dataframe),
        "robot_count": config.robots,
        "robot_type": config.robot_type,
        "seed": config.seed,
        "simulation_only": True,
        "physical_execution_authorized": False,
    }

    if not args.summary_only:
        result["statistics"] = manifest[
            "statistics"
        ]

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(
            main()
        )

    except (
        ValueError,
        OSError,
        ImportError,
        KeyError,
    ) as exc:
        LOGGER.error(
            "Telemetry generation failed: %s",
            exc,
        )

        sys.exit(1)
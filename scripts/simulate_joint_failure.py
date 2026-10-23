
#!/usr/bin/env python3
"""
Physical AI & Autonomous Robotics Intelligence Platform
File: scripts/simulate_joint_failure.py

Offline robotic joint failure and degradation simulator.

Features:
    - Robotic arms, humanoids, quadrupeds
    - Multiple robots and joints
    - Joint position / velocity / torque simulation
    - Motor and bearing temperature monitoring
    - Encoder fault simulation
    - Joint overheating
    - Motor overload
    - Bearing degradation
    - Excessive vibration
    - Gearbox backlash
    - Actuator jam detection
    - Remaining Useful Life (RUL) estimation
    - Synthetic failure probability
    - Predictive maintenance recommendations
    - Simulated emergency-stop latch
    - Human approval requirements
    - JSON / CSV exports
    - Deterministic random seed

Dependencies:
    Python 3.10+
    Standard library only

Usage:
    python scripts/simulate_joint_failure.py

    python scripts/simulate_joint_failure.py \
        --robots 5 --steps 200

    python scripts/simulate_joint_failure.py \
        --scenario overheating

    python scripts/simulate_joint_failure.py \
        --scenario encoder_failure

    python scripts/simulate_joint_failure.py \
        --scenario all \
        --robots 8 \
        --steps 500 \
        --output reports/joint_failure.json \
        --csv reports/joint_failure.csv

Safety:
    Simulation-only data.
    No hardware actuation.
    No physical emergency-stop reset.
    No automatic physical robot restart.
    Not a safety-certified diagnostic system.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
import uuid

from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("joint_failure_simulation")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

SCHEMA_VERSION = "1.0.0"

NAMESPACE = uuid.UUID(
    "b2b394a2-7d3f-45d4-9a9b-cc5f38939d23"
)


# ============================================================
# Enumerations
# ============================================================

class JointFailureType(str, Enum):
    NONE = "none"
    OVERHEATING = "overheating"
    MOTOR_OVERLOAD = "motor_overload"
    ENCODER_FAILURE = "encoder_failure"
    BEARING_WEAR = "bearing_wear"
    EXCESSIVE_VIBRATION = "excessive_vibration"
    GEARBOX_BACKLASH = "gearbox_backlash"
    ACTUATOR_JAM = "actuator_jam"


class JointCondition(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


SCENARIOS = [
    "normal",
    "overheating",
    "motor_overload",
    "encoder_failure",
    "bearing_wear",
    "excessive_vibration",
    "gearbox_backlash",
    "actuator_jam",
    "all",
]


# ============================================================
# Configuration
# ============================================================

@dataclass
class JointThresholds:
    temperature_warning_c: float = 75.0
    temperature_critical_c: float = 95.0

    torque_warning_nm: float = 75.0
    torque_critical_nm: float = 110.0

    vibration_warning_mm_s: float = 5.0
    vibration_critical_mm_s: float = 12.0

    position_error_warning_deg: float = 4.0
    position_error_critical_deg: float = 12.0

    backlash_warning_deg: float = 2.0
    backlash_critical_deg: float = 6.0

    bearing_wear_warning_percent: float = 65.0
    bearing_wear_critical_percent: float = 90.0

    jam_velocity_threshold_rad_s: float = 0.01
    jam_torque_threshold_nm: float = 80.0

    def validate(self) -> None:
        for key, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{key} must be positive and finite"
                )

        pairs = [
            (
                self.temperature_warning_c,
                self.temperature_critical_c,
            ),
            (
                self.torque_warning_nm,
                self.torque_critical_nm,
            ),
            (
                self.vibration_warning_mm_s,
                self.vibration_critical_mm_s,
            ),
            (
                self.position_error_warning_deg,
                self.position_error_critical_deg,
            ),
            (
                self.backlash_warning_deg,
                self.backlash_critical_deg,
            ),
            (
                self.bearing_wear_warning_percent,
                self.bearing_wear_critical_percent,
            ),
        ]

        if any(low >= high for low, high in pairs):
            raise ValueError(
                "Warning thresholds must be below critical thresholds"
            )


@dataclass
class SimulationConfig:
    robots: int = 4
    joints_per_robot: int = 6

    steps: int = 200
    delta_time_s: float = 0.1

    seed: int = 42

    scenario: str = "all"
    event_probability: float = 0.02

    thresholds: JointThresholds = field(
        default_factory=JointThresholds
    )

    def validate(self) -> None:
        if not 1 <= self.robots <= 100:
            raise ValueError(
                "robots must be between 1 and 100"
            )

        if not 1 <= self.joints_per_robot <= 64:
            raise ValueError(
                "joints_per_robot must be between 1 and 64"
            )

        if not 1 <= self.steps <= 100000:
            raise ValueError(
                "steps must be between 1 and 100000"
            )

        if (
            not math.isfinite(self.delta_time_s)
            or self.delta_time_s <= 0
        ):
            raise ValueError(
                "delta_time_s must be positive"
            )

        if not (
            math.isfinite(self.event_probability)
            and 0 <= self.event_probability <= 1
        ):
            raise ValueError(
                "event_probability must be between 0 and 1"
            )

        if self.scenario not in SCENARIOS:
            raise ValueError(
                f"Unsupported scenario: {self.scenario}"
            )

        self.thresholds.validate()


# ============================================================
# Joint Telemetry
# ============================================================

@dataclass
class JointTelemetry:
    position_rad: float = 0.0
    target_position_rad: float = 0.0

    velocity_rad_s: float = 0.0
    target_velocity_rad_s: float = 0.0

    torque_nm: float = 12.0
    motor_current_a: float = 4.0

    motor_temperature_c: float = 40.0
    bearing_temperature_c: float = 35.0

    vibration_mm_s: float = 1.0

    encoder_position_error_deg: float = 0.2
    gearbox_backlash_deg: float = 0.3

    bearing_wear_percent: float = 10.0

    sensor_health_score: float = 1.0
    actuator_response_score: float = 1.0

    operating_hours: float = 100.0

    def validate(self) -> None:
        for key, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(
                    f"Invalid joint telemetry: {key}"
                )


@dataclass
class SimulatedJoint:
    joint_id: str
    robot_id: str

    joint_index: int

    telemetry: JointTelemetry = field(
        default_factory=JointTelemetry
    )

    health_score: float = 1.0
    failure_probability: float = 0.0

    remaining_useful_life_hours: float = 10000.0

    condition: JointCondition = JointCondition.HEALTHY

    failure_type: JointFailureType = JointFailureType.NONE

    simulated_motion_inhibited: bool = False
    human_approval_required: bool = False


@dataclass
class SimulatedRobot:
    robot_id: str
    robot_type: str

    joints: list[SimulatedJoint] = field(
        default_factory=list
    )

    simulated_estop_latched: bool = False
    human_approval_required: bool = False


# ============================================================
# Joint Failure Event
# ============================================================

@dataclass
class JointFailureEvent:
    event_id: str

    robot_id: str
    joint_id: str

    timestamp_s: float

    failure_type: str
    severity: str

    description: str

    health_score: float
    failure_probability: float
    estimated_rul_hours: float

    simulated_motion_inhibited: bool
    simulated_estop_latched: bool

    maintenance_recommendation: str

    human_approval_required: bool = True
    simulation_only: bool = True


# ============================================================
# Telemetry Generator
# ============================================================

class JointTelemetryGenerator:
    def __init__(
        self,
        config: SimulationConfig,
    ) -> None:
        self.config = config
        self.rng = random.Random(config.seed)

    def generate_robots(self) -> list[SimulatedRobot]:
        robot_types = [
            "robotic_arm",
            "humanoid",
            "quadruped",
        ]

        robots = []

        for robot_index in range(self.config.robots):
            robot_id = (
                f"SIM-ROBOT-{robot_index + 1:03d}"
            )

            robot = SimulatedRobot(
                robot_id=robot_id,
                robot_type=robot_types[
                    robot_index % len(robot_types)
                ],
            )

            for joint_index in range(
                self.config.joints_per_robot
            ):
                joint_id = (
                    f"{robot_id}-JOINT-{joint_index + 1:02d}"
                )

                telemetry = JointTelemetry(
                    position_rad=self.rng.uniform(-1, 1),
                    target_position_rad=0.0,
                    velocity_rad_s=self.rng.uniform(
                        -0.5, 0.5
                    ),
                    target_velocity_rad_s=0.0,
                    torque_nm=self.rng.uniform(5, 25),
                    motor_current_a=self.rng.uniform(2, 8),
                    motor_temperature_c=self.rng.uniform(
                        30, 50
                    ),
                    bearing_temperature_c=self.rng.uniform(
                        28, 45
                    ),
                    vibration_mm_s=self.rng.uniform(
                        0.5, 2.0
                    ),
                    encoder_position_error_deg=(
                        self.rng.uniform(0.1, 0.8)
                    ),
                    gearbox_backlash_deg=(
                        self.rng.uniform(0.1, 0.8)
                    ),
                    bearing_wear_percent=(
                        self.rng.uniform(5, 25)
                    ),
                    operating_hours=self.rng.uniform(
                        100, 2000
                    ),
                )

                robot.joints.append(
                    SimulatedJoint(
                        joint_id=joint_id,
                        robot_id=robot_id,
                        joint_index=joint_index,
                        telemetry=telemetry,
                    )
                )

            robots.append(robot)

        return robots

    def baseline_update(
        self,
        telemetry: JointTelemetry,
    ) -> None:
        dt = self.config.delta_time_s

        telemetry.target_velocity_rad_s = (
            self.rng.uniform(-0.8, 0.8)
        )

        telemetry.target_position_rad += (
            telemetry.target_velocity_rad_s * dt
        )

        telemetry.velocity_rad_s += (
            telemetry.target_velocity_rad_s
            - telemetry.velocity_rad_s
        ) * 0.25

        telemetry.position_rad += (
            telemetry.velocity_rad_s * dt
        )

        telemetry.torque_nm = max(
            0.0,
            telemetry.torque_nm
            + self.rng.uniform(-1.0, 1.0),
        )

        telemetry.motor_current_a = max(
            0.0,
            telemetry.motor_current_a
            + self.rng.uniform(-0.2, 0.2),
        )

        telemetry.motor_temperature_c = max(
            20.0,
            telemetry.motor_temperature_c
            + self.rng.uniform(-0.2, 0.3),
        )

        telemetry.bearing_temperature_c = max(
            20.0,
            telemetry.bearing_temperature_c
            + self.rng.uniform(-0.1, 0.2),
        )

        telemetry.vibration_mm_s = max(
            0.0,
            telemetry.vibration_mm_s
            + self.rng.uniform(-0.1, 0.1),
        )

        telemetry.bearing_wear_percent = min(
            100.0,
            telemetry.bearing_wear_percent
            + self.rng.uniform(0.0, 0.015),
        )

        telemetry.operating_hours += dt / 3600.0

    def select_scenario(self) -> str:
        if self.config.scenario != "all":
            return self.config.scenario

        return self.rng.choice(
            SCENARIOS[1:-1]
        )

    def inject_failure(
        self,
        telemetry: JointTelemetry,
        scenario: str,
    ) -> None:
        if scenario == "overheating":
            telemetry.motor_temperature_c = (
                self.rng.uniform(96, 125)
            )

        elif scenario == "motor_overload":
            telemetry.torque_nm = (
                self.rng.uniform(115, 160)
            )
            telemetry.motor_current_a = (
                self.rng.uniform(20, 40)
            )

        elif scenario == "encoder_failure":
            telemetry.encoder_position_error_deg = (
                self.rng.uniform(13, 30)
            )
            telemetry.sensor_health_score = (
                self.rng.uniform(0.0, 0.2)
            )

        elif scenario == "bearing_wear":
            telemetry.bearing_wear_percent = (
                self.rng.uniform(91, 100)
            )
            telemetry.bearing_temperature_c = (
                self.rng.uniform(75, 100)
            )

        elif scenario == "excessive_vibration":
            telemetry.vibration_mm_s = (
                self.rng.uniform(13, 25)
            )

        elif scenario == "gearbox_backlash":
            telemetry.gearbox_backlash_deg = (
                self.rng.uniform(7, 15)
            )

        elif scenario == "actuator_jam":
            telemetry.velocity_rad_s = 0.0
            telemetry.target_velocity_rad_s = 0.8
            telemetry.torque_nm = (
                self.rng.uniform(85, 120)
            )
            telemetry.actuator_response_score = 0.0

    def update_joint(
        self,
        joint: SimulatedJoint,
        step: int,
    ) -> None:
        if joint.simulated_motion_inhibited:
            return

        self.baseline_update(
            joint.telemetry
        )

        inject = (
            step == 1
            or self.rng.random()
            < self.config.event_probability
        )

        if inject:
            scenario = self.select_scenario()

            if scenario != "normal":
                self.inject_failure(
                    joint.telemetry,
                    scenario,
                )

        joint.telemetry.validate()


# ============================================================
# Failure Detection
# ============================================================

class JointFailureDetector:
    def __init__(
        self,
        thresholds: JointThresholds,
    ) -> None:
        self.t = thresholds

    def detect(
        self,
        telemetry: JointTelemetry,
    ) -> list[tuple[JointFailureType, Severity, str]]:
        findings = []

        def check(
            value: float,
            warning: float,
            critical: float,
            failure_type: JointFailureType,
            description: str,
        ) -> None:
            if value >= critical:
                findings.append((
                    failure_type,
                    Severity.CRITICAL,
                    f"Critical {description}: {value:.2f}",
                ))

            elif value >= warning:
                findings.append((
                    failure_type,
                    Severity.WARNING,
                    f"Elevated {description}: {value:.2f}",
                ))

        check(
            telemetry.motor_temperature_c,
            self.t.temperature_warning_c,
            self.t.temperature_critical_c,
            JointFailureType.OVERHEATING,
            "motor temperature",
        )

        check(
            telemetry.torque_nm,
            self.t.torque_warning_nm,
            self.t.torque_critical_nm,
            JointFailureType.MOTOR_OVERLOAD,
            "motor torque",
        )

        check(
            telemetry.vibration_mm_s,
            self.t.vibration_warning_mm_s,
            self.t.vibration_critical_mm_s,
            JointFailureType.EXCESSIVE_VIBRATION,
            "vibration",
        )

        check(
            telemetry.encoder_position_error_deg,
            self.t.position_error_warning_deg,
            self.t.position_error_critical_deg,
            JointFailureType.ENCODER_FAILURE,
            "encoder position error",
        )

        check(
            telemetry.gearbox_backlash_deg,
            self.t.backlash_warning_deg,
            self.t.backlash_critical_deg,
            JointFailureType.GEARBOX_BACKLASH,
            "gearbox backlash",
        )

        check(
            telemetry.bearing_wear_percent,
            self.t.bearing_wear_warning_percent,
            self.t.bearing_wear_critical_percent,
            JointFailureType.BEARING_WEAR,
            "bearing wear",
        )

        jam_detected = (
            abs(telemetry.velocity_rad_s)
            <= self.t.jam_velocity_threshold_rad_s
            and abs(telemetry.target_velocity_rad_s) > 0.2
            and telemetry.torque_nm
            >= self.t.jam_torque_threshold_nm
        )

        if jam_detected:
            findings.append((
                JointFailureType.ACTUATOR_JAM,
                Severity.CRITICAL,
                "Joint appears jammed under high torque",
            ))

        return findings


# ============================================================
# Joint Health Estimation
# ============================================================

class JointHealthEstimator:
    def __init__(
        self,
        thresholds: JointThresholds,
    ) -> None:
        self.t = thresholds

    @staticmethod
    def clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> float:
        return max(
            minimum,
            min(maximum, value),
        )

    def estimate(
        self,
        telemetry: JointTelemetry,
    ) -> tuple[float, float, float, JointCondition]:
        temperature_risk = self.clamp(
            (
                telemetry.motor_temperature_c - 40
            ) / (
                self.t.temperature_critical_c - 40
            )
        )

        torque_risk = self.clamp(
            telemetry.torque_nm
            / self.t.torque_critical_nm
        )

        vibration_risk = self.clamp(
            telemetry.vibration_mm_s
            / self.t.vibration_critical_mm_s
        )

        wear_risk = self.clamp(
            telemetry.bearing_wear_percent / 100.0
        )

        encoder_risk = self.clamp(
            telemetry.encoder_position_error_deg
            / self.t.position_error_critical_deg
        )

        backlash_risk = self.clamp(
            telemetry.gearbox_backlash_deg
            / self.t.backlash_critical_deg
        )

        risk = (
            0.22 * temperature_risk
            + 0.18 * torque_risk
            + 0.18 * vibration_risk
            + 0.18 * wear_risk
            + 0.14 * encoder_risk
            + 0.10 * backlash_risk
        )

        health = self.clamp(
            1.0 - risk
        )

        failure_probability = self.clamp(
            risk ** 1.5
        )

        # Illustrative RUL heuristic, not a trained model.
        estimated_rul = max(
            1.0,
            10000.0
            * (health ** 2)
            - telemetry.operating_hours * 0.05,
        )

        if health < 0.35:
            condition = JointCondition.CRITICAL
        elif health < 0.6:
            condition = JointCondition.DEGRADED
        elif health < 0.8:
            condition = JointCondition.WARNING
        else:
            condition = JointCondition.HEALTHY

        return (
            round(health, 4),
            round(failure_probability, 4),
            round(estimated_rul, 3),
            condition,
        )


# ============================================================
# Maintenance Recommendations
# ============================================================

class JointMaintenanceAdvisor:
    RECOMMENDATIONS = {
        JointFailureType.OVERHEATING: (
            "Inspect motor cooling, thermal load, "
            "and temperature sensor calibration"
        ),
        JointFailureType.MOTOR_OVERLOAD: (
            "Inspect mechanical load, motor current, "
            "gearbox resistance, and torque limits"
        ),
        JointFailureType.ENCODER_FAILURE: (
            "Inspect encoder feedback, wiring, "
            "calibration, and signal integrity"
        ),
        JointFailureType.BEARING_WEAR: (
            "Inspect bearing condition, lubrication, "
            "shaft alignment, and replacement need"
        ),
        JointFailureType.EXCESSIVE_VIBRATION: (
            "Inspect imbalance, alignment, "
            "bearings, and mounting hardware"
        ),
        JointFailureType.GEARBOX_BACKLASH: (
            "Inspect gearbox wear, gear engagement, "
            "and mechanical tolerances"
        ),
        JointFailureType.ACTUATOR_JAM: (
            "Inspect for obstruction, mechanical binding, "
            "and actuator damage"
        ),
        JointFailureType.NONE: (
            "Continue synthetic condition monitoring"
        ),
    }

    def recommend(
        self,
        failure_type: JointFailureType,
        condition: JointCondition,
    ) -> str:
        recommendation = (
            self.RECOMMENDATIONS[failure_type]
        )

        if condition == JointCondition.CRITICAL:
            return (
                "SIMULATION: inhibit motion; request "
                "qualified human inspection. "
                + recommendation
            )

        if condition == JointCondition.DEGRADED:
            return (
                "SIMULATION: prioritize maintenance review. "
                + recommendation
            )

        return recommendation


# ============================================================
# Simulation Engine
# ============================================================

class JointFailureSimulation:
    def __init__(
        self,
        config: SimulationConfig,
    ) -> None:
        self.config = config
        self.config.validate()

        self.generator = JointTelemetryGenerator(
            config
        )

        self.detector = JointFailureDetector(
            config.thresholds
        )

        self.health_estimator = JointHealthEstimator(
            config.thresholds
        )

        self.maintenance_advisor = (
            JointMaintenanceAdvisor()
        )

        self.robots = (
            self.generator.generate_robots()
        )

        self.events: list[JointFailureEvent] = []
        self.telemetry_rows: list[dict[str, Any]] = []

        self.recorded_signatures: set[
            tuple[str, str]
        ] = set()

    def create_event(
        self,
        robot: SimulatedRobot,
        joint: SimulatedJoint,
        failure_type: JointFailureType,
        severity: Severity,
        description: str,
        timestamp_s: float,
    ) -> JointFailureEvent | None:
        signature = (
            joint.joint_id,
            failure_type.value,
        )

        if signature in self.recorded_signatures:
            return None

        self.recorded_signatures.add(
            signature
        )

        critical = (
            severity == Severity.CRITICAL
        )

        if critical:
            robot.simulated_estop_latched = True
            robot.human_approval_required = True

            for robot_joint in robot.joints:
                robot_joint.simulated_motion_inhibited = True
                robot_joint.human_approval_required = True

        recommendation = (
            self.maintenance_advisor.recommend(
                failure_type,
                joint.condition,
            )
        )

        event_id = str(
            uuid.uuid5(
                NAMESPACE,
                f"{joint.joint_id}:"
                f"{failure_type.value}:"
                f"{timestamp_s:.6f}",
            )
        )

        event = JointFailureEvent(
            event_id=event_id,
            robot_id=robot.robot_id,
            joint_id=joint.joint_id,
            timestamp_s=round(timestamp_s, 4),
            failure_type=failure_type.value,
            severity=severity.value,
            description=description,
            health_score=joint.health_score,
            failure_probability=(
                joint.failure_probability
            ),
            estimated_rul_hours=(
                joint.remaining_useful_life_hours
            ),
            simulated_motion_inhibited=(
                joint.simulated_motion_inhibited
            ),
            simulated_estop_latched=(
                robot.simulated_estop_latched
            ),
            maintenance_recommendation=(
                recommendation
            ),
            human_approval_required=True,
            simulation_only=True,
        )

        self.events.append(event)
        return event

    def record_telemetry(
        self,
        robot: SimulatedRobot,
        joint: SimulatedJoint,
        timestamp_s: float,
    ) -> None:
        self.telemetry_rows.append({
            "timestamp_s": round(timestamp_s, 4),
            "robot_id": robot.robot_id,
            "robot_type": robot.robot_type,
            "joint_id": joint.joint_id,
            "position_rad": (
                joint.telemetry.position_rad
            ),
            "target_position_rad": (
                joint.telemetry.target_position_rad
            ),
            "velocity_rad_s": (
                joint.telemetry.velocity_rad_s
            ),
            "target_velocity_rad_s": (
                joint.telemetry.target_velocity_rad_s
            ),
            "torque_nm": (
                joint.telemetry.torque_nm
            ),
            "motor_current_a": (
                joint.telemetry.motor_current_a
            ),
            "motor_temperature_c": (
                joint.telemetry.motor_temperature_c
            ),
            "bearing_temperature_c": (
                joint.telemetry.bearing_temperature_c
            ),
            "vibration_mm_s": (
                joint.telemetry.vibration_mm_s
            ),
            "encoder_position_error_deg": (
                joint.telemetry.encoder_position_error_deg
            ),
            "gearbox_backlash_deg": (
                joint.telemetry.gearbox_backlash_deg
            ),
            "bearing_wear_percent": (
                joint.telemetry.bearing_wear_percent
            ),
            "health_score": joint.health_score,
            "failure_probability": (
                joint.failure_probability
            ),
            "remaining_useful_life_hours": (
                joint.remaining_useful_life_hours
            ),
            "condition": joint.condition.value,
            "failure_type": joint.failure_type.value,
            "simulated_motion_inhibited": (
                joint.simulated_motion_inhibited
            ),
            "simulated_estop_latched": (
                robot.simulated_estop_latched
            ),
            "simulation_only": True,
        })

    def run_step(
        self,
        step: int,
    ) -> None:
        timestamp_s = (
            step * self.config.delta_time_s
        )

        for robot in self.robots:
            for joint in robot.joints:
                self.generator.update_joint(
                    joint,
                    step,
                )

                (
                    joint.health_score,
                    joint.failure_probability,
                    joint.remaining_useful_life_hours,
                    joint.condition,
                ) = self.health_estimator.estimate(
                    joint.telemetry
                )

                findings = self.detector.detect(
                    joint.telemetry
                )

                for (
                    failure_type,
                    severity,
                    description,
                ) in findings:
                    joint.failure_type = failure_type

                    self.create_event(
                        robot,
                        joint,
                        failure_type,
                        severity,
                        description,
                        timestamp_s,
                    )

                self.record_telemetry(
                    robot,
                    joint,
                    timestamp_s,
                )

    def run(self) -> dict[str, Any]:
        LOGGER.info(
            "Starting joint failure simulation"
        )

        for step in range(self.config.steps):
            self.run_step(step)

        return self.build_report()

    def build_report(self) -> dict[str, Any]:
        failure_counts = Counter(
            event.failure_type
            for event in self.events
        )

        severity_counts = Counter(
            event.severity
            for event in self.events
        )

        stopped_robots = [
            robot.robot_id
            for robot in self.robots
            if robot.simulated_estop_latched
        ]

        all_joints = [
            joint
            for robot in self.robots
            for joint in robot.joints
        ]

        mean_health = (
            sum(
                joint.health_score
                for joint in all_joints
            ) / len(all_joints)
            if all_joints
            else 0.0
        )

        return {
            "status": "completed",
            "schema_version": SCHEMA_VERSION,
            "simulation_only": True,
            "physical_robot_control_executed": False,
            "physical_estop_reset_executed": False,
            "safety_certified": False,
            "configuration": {
                "robots": self.config.robots,
                "joints_per_robot": (
                    self.config.joints_per_robot
                ),
                "steps": self.config.steps,
                "delta_time_s": self.config.delta_time_s,
                "scenario": self.config.scenario,
                "seed": self.config.seed,
                "event_probability": (
                    self.config.event_probability
                ),
            },
            "summary": {
                "total_robots": len(self.robots),
                "total_joints": len(all_joints),
                "total_telemetry_records": (
                    len(self.telemetry_rows)
                ),
                "total_failure_events": len(self.events),
                "failure_counts": dict(failure_counts),
                "severity_counts": dict(severity_counts),
                "simulated_estop_count": (
                    len(stopped_robots)
                ),
                "mean_joint_health_score": round(
                    mean_health, 4
                ),
                "critical_joints": sum(
                    joint.condition
                    == JointCondition.CRITICAL
                    for joint in all_joints
                ),
                "degraded_joints": sum(
                    joint.condition
                    == JointCondition.DEGRADED
                    for joint in all_joints
                ),
            },
            "simulated_estop_robots": stopped_robots,
            "robots": [
                {
                    "robot_id": robot.robot_id,
                    "robot_type": robot.robot_type,
                    "simulated_estop_latched": (
                        robot.simulated_estop_latched
                    ),
                    "human_approval_required": (
                        robot.human_approval_required
                    ),
                    "joints": [
                        {
                            "joint_id": joint.joint_id,
                            "condition": (
                                joint.condition.value
                            ),
                            "health_score": (
                                joint.health_score
                            ),
                            "failure_probability": (
                                joint.failure_probability
                            ),
                            "remaining_useful_life_hours": (
                                joint.remaining_useful_life_hours
                            ),
                            "failure_type": (
                                joint.failure_type.value
                            ),
                            "simulated_motion_inhibited": (
                                joint.simulated_motion_inhibited
                            ),
                        }
                        for joint in robot.joints
                    ],
                }
                for robot in self.robots
            ],
            "events": [
                asdict(event)
                for event in self.events
            ],
            "limitations": [
                "Joint telemetry is synthetic.",
                "Health scores are heuristic estimates.",
                "Failure probabilities are not calibrated.",
                "RUL is not predicted by a trained model.",
                "Joint dynamics are simplified.",
                "No physical hardware is controlled.",
                "Safety thresholds are not certified.",
                "Human review is required for real maintenance decisions.",
            ],
        }


# ============================================================
# Report Export
# ============================================================

def save_json(
    path: str | Path,
    report: dict[str, Any],
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def save_csv(
    path: str | Path,
    rows: list[dict[str, Any]],
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate robotic joint failures offline"
        )
    )

    parser.add_argument(
        "--robots",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--joints-per-robot",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default="all",
    )

    parser.add_argument(
        "--event-probability",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    parser.add_argument(
        "--csv",
        default=None,
    )

    return parser


# ============================================================
# Main
# ============================================================

def main() -> int:
    args = build_parser().parse_args()

    config = SimulationConfig(
        robots=args.robots,
        joints_per_robot=args.joints_per_robot,
        steps=args.steps,
        delta_time_s=args.dt,
        seed=args.seed,
        scenario=args.scenario,
        event_probability=args.event_probability,
    )

    config.validate()

    simulation = JointFailureSimulation(
        config
    )

    report = simulation.run()

    if args.output:
        save_json(
            args.output,
            report,
        )

    if args.csv:
        save_csv(
            args.csv,
            simulation.telemetry_rows,
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
        "Joint failure simulation completed"
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())

    except (
        ValueError,
        RuntimeError,
        OSError,
        TypeError,
    ) as exc:
        LOGGER.error(
            "Joint failure simulation failed: %s",
            exc,
        )

        sys.exit(1)


#!/usr/bin/env python3
"""
Physical AI & Autonomous Robotics Intelligence Platform
File: scripts/simulate_motor_failure.py

Offline synthetic motor-failure simulator.

Python: 3.10+
Dependencies: standard library only

Examples:
    python scripts/simulate_motor_failure.py
    python scripts/simulate_motor_failure.py --scenario overheating
    python scripts/simulate_motor_failure.py --scenario rotor_stall
    python scripts/simulate_motor_failure.py --scenario all --robots 8
    python scripts/simulate_motor_failure.py --steps 500 --motors-per-robot 6
    python scripts/simulate_motor_failure.py \
        --output reports/motor_failure.json \
        --csv reports/motor_telemetry.csv \
        --events-csv reports/motor_events.csv

All robot motion inhibition and emergency-stop states are
simulated in memory. No hardware is contacted.
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


LOG = logging.getLogger("motor_failure_simulator")

SCHEMA_VERSION = "1.0.0"
NAMESPACE = uuid.UUID(
    "46a66f57-5d46-4c58-b99c-c71a419f67a0"
)


# ============================================================
# Enumerations
# ============================================================

class FailureType(str, Enum):
    NORMAL = "normal"
    OVERHEATING = "overheating"
    OVERCURRENT = "overcurrent"
    BEARING_WEAR = "bearing_wear"
    EXCESSIVE_VIBRATION = "excessive_vibration"
    ENCODER_FAILURE = "encoder_failure"
    ROTOR_STALL = "rotor_stall"
    WINDING_FAULT = "winding_fault"
    MECHANICAL_OVERLOAD = "mechanical_overload"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class HealthState(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"


SCENARIOS = tuple(item.value for item in FailureType) + (
    "all",
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class MotorThresholds:
    temperature_warning_c: float = 75.0
    temperature_critical_c: float = 95.0

    current_warning_a: float = 18.0
    current_critical_a: float = 30.0

    vibration_warning_mm_s: float = 5.0
    vibration_critical_mm_s: float = 12.0

    torque_warning_nm: float = 65.0
    torque_critical_nm: float = 100.0

    encoder_warning_error_deg: float = 4.0
    encoder_critical_error_deg: float = 12.0

    bearing_warning_wear_percent: float = 65.0
    bearing_critical_wear_percent: float = 90.0

    winding_warning_resistance_ohm: float = 2.0
    winding_critical_resistance_ohm: float = 3.5

    stall_speed_threshold_rpm: float = 5.0
    stall_command_threshold_rpm: float = 150.0
    stall_current_threshold_a: float = 20.0

    def validate(self) -> None:
        values = asdict(self)

        for name, value in values.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} must be finite and positive"
                )

        pairs = (
            (
                self.temperature_warning_c,
                self.temperature_critical_c,
            ),
            (
                self.current_warning_a,
                self.current_critical_a,
            ),
            (
                self.vibration_warning_mm_s,
                self.vibration_critical_mm_s,
            ),
            (
                self.torque_warning_nm,
                self.torque_critical_nm,
            ),
            (
                self.encoder_warning_error_deg,
                self.encoder_critical_error_deg,
            ),
            (
                self.bearing_warning_wear_percent,
                self.bearing_critical_wear_percent,
            ),
            (
                self.winding_warning_resistance_ohm,
                self.winding_critical_resistance_ohm,
            ),
        )

        if any(warning >= critical for warning, critical in pairs):
            raise ValueError(
                "Warning thresholds must be below critical thresholds"
            )


@dataclass
class SimulationConfig:
    robots: int = 4
    motors_per_robot: int = 4

    steps: int = 200
    delta_time_s: float = 0.1

    seed: int = 42
    scenario: str = "all"

    event_probability: float = 0.02
    degradation_multiplier: float = 1.0

    thresholds: MotorThresholds = field(
        default_factory=MotorThresholds
    )

    def validate(self) -> None:
        if not 1 <= self.robots <= 100:
            raise ValueError("robots must be between 1 and 100")

        if not 1 <= self.motors_per_robot <= 32:
            raise ValueError(
                "motors_per_robot must be between 1 and 32"
            )

        if not 1 <= self.steps <= 100000:
            raise ValueError(
                "steps must be between 1 and 100000"
            )

        if not math.isfinite(self.delta_time_s) or (
            self.delta_time_s <= 0
        ):
            raise ValueError(
                "delta_time_s must be finite and positive"
            )

        if not (
            math.isfinite(self.event_probability)
            and 0 <= self.event_probability <= 1
        ):
            raise ValueError(
                "event_probability must be between 0 and 1"
            )

        if not (
            math.isfinite(self.degradation_multiplier)
            and self.degradation_multiplier >= 0
        ):
            raise ValueError(
                "degradation_multiplier must be nonnegative"
            )

        if self.scenario not in SCENARIOS:
            raise ValueError(
                f"Unsupported scenario: {self.scenario}"
            )

        self.thresholds.validate()


# ============================================================
# Telemetry Models
# ============================================================

@dataclass
class MotorTelemetry:
    timestamp_s: float = 0.0

    actual_speed_rpm: float = 500.0
    commanded_speed_rpm: float = 500.0

    torque_nm: float = 15.0
    current_a: float = 6.0
    voltage_v: float = 48.0

    motor_temperature_c: float = 40.0
    bearing_temperature_c: float = 35.0

    vibration_mm_s: float = 1.2

    encoder_error_deg: float = 0.3
    winding_resistance_ohm: float = 1.0

    bearing_wear_percent: float = 10.0
    operating_hours: float = 500.0

    efficiency_percent: float = 92.0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite telemetry field: {name}"
                )


@dataclass
class SimulatedMotor:
    motor_id: str
    robot_id: str
    motor_index: int

    telemetry: MotorTelemetry = field(
        default_factory=MotorTelemetry
    )

    health_score: float = 1.0
    failure_probability: float = 0.0
    estimated_rul_hours: float = 10000.0

    health_state: HealthState = HealthState.HEALTHY

    simulated_motion_inhibited: bool = False
    human_review_required: bool = False


@dataclass
class SimulatedRobot:
    robot_id: str
    robot_type: str

    motors: list[SimulatedMotor] = field(
        default_factory=list
    )

    simulated_estop_latched: bool = False
    human_review_required: bool = False


@dataclass
class MotorFailureEvent:
    event_id: str
    robot_id: str
    motor_id: str

    timestamp_s: float

    failure_type: str
    severity: str
    description: str

    health_score: float
    failure_probability: float
    estimated_rul_hours: float

    maintenance_recommendation: str

    simulated_motion_inhibited: bool
    simulated_estop_latched: bool

    human_review_required: bool = True
    simulation_only: bool = True


# ============================================================
# Utilities
# ============================================================

def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(minimum, min(maximum, value))


def stable_event_id(
    motor_id: str,
    failure_type: FailureType,
    timestamp_s: float,
) -> str:
    return str(
        uuid.uuid5(
            NAMESPACE,
            f"{motor_id}:{failure_type.value}:{timestamp_s:.6f}",
        )
    )


# ============================================================
# Synthetic Telemetry Generator
# ============================================================

class MotorTelemetryGenerator:
    ROBOT_TYPES = (
        "mobile_robot",
        "robotic_arm",
        "humanoid",
        "quadruped",
    )

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)

    def generate_robots(self) -> list[SimulatedRobot]:
        robots: list[SimulatedRobot] = []

        for robot_index in range(self.config.robots):
            robot_id = f"SIM-ROBOT-{robot_index + 1:03d}"

            robot = SimulatedRobot(
                robot_id=robot_id,
                robot_type=self.ROBOT_TYPES[
                    robot_index % len(self.ROBOT_TYPES)
                ],
            )

            for motor_index in range(
                self.config.motors_per_robot
            ):
                telemetry = MotorTelemetry(
                    actual_speed_rpm=self.rng.uniform(
                        300, 800
                    ),
                    commanded_speed_rpm=500.0,
                    torque_nm=self.rng.uniform(8, 22),
                    current_a=self.rng.uniform(3, 9),
                    motor_temperature_c=self.rng.uniform(
                        30, 48
                    ),
                    bearing_temperature_c=self.rng.uniform(
                        28, 42
                    ),
                    vibration_mm_s=self.rng.uniform(
                        0.5, 2.0
                    ),
                    bearing_wear_percent=self.rng.uniform(
                        5, 25
                    ),
                    operating_hours=self.rng.uniform(
                        100, 2000
                    ),
                )

                motor = SimulatedMotor(
                    motor_id=(
                        f"{robot_id}-MOTOR-{motor_index + 1:02d}"
                    ),
                    robot_id=robot_id,
                    motor_index=motor_index,
                    telemetry=telemetry,
                )

                robot.motors.append(motor)

            robots.append(robot)

        return robots

    def update_baseline(
        self,
        telemetry: MotorTelemetry,
        timestamp_s: float,
    ) -> None:
        rng = self.rng
        dt = self.config.delta_time_s
        degradation = self.config.degradation_multiplier

        telemetry.timestamp_s = timestamp_s

        telemetry.commanded_speed_rpm = rng.uniform(
            250, 900
        )

        telemetry.actual_speed_rpm += (
            telemetry.commanded_speed_rpm
            - telemetry.actual_speed_rpm
        ) * 0.3

        telemetry.actual_speed_rpm = max(
            0.0,
            telemetry.actual_speed_rpm
            + rng.uniform(-10, 10),
        )

        telemetry.torque_nm = max(
            0.0,
            telemetry.torque_nm
            + rng.uniform(-1.0, 1.0),
        )

        telemetry.current_a = max(
            0.0,
            telemetry.current_a
            + rng.uniform(-0.3, 0.3),
        )

        telemetry.motor_temperature_c = max(
            20.0,
            telemetry.motor_temperature_c
            + rng.uniform(-0.15, 0.3),
        )

        telemetry.bearing_temperature_c = max(
            20.0,
            telemetry.bearing_temperature_c
            + rng.uniform(-0.1, 0.2),
        )

        telemetry.vibration_mm_s = max(
            0.0,
            telemetry.vibration_mm_s
            + rng.uniform(-0.08, 0.1),
        )

        telemetry.encoder_error_deg = max(
            0.0,
            telemetry.encoder_error_deg
            + rng.uniform(-0.03, 0.04),
        )

        telemetry.bearing_wear_percent = min(
            100.0,
            telemetry.bearing_wear_percent
            + rng.uniform(0.0, 0.01) * degradation,
        )

        telemetry.winding_resistance_ohm = max(
            0.01,
            telemetry.winding_resistance_ohm
            + rng.uniform(-0.002, 0.003),
        )

        telemetry.operating_hours += dt / 3600.0

        telemetry.efficiency_percent = clamp(
            0.95
            - telemetry.bearing_wear_percent / 500.0
            - telemetry.vibration_mm_s / 150.0,
        ) * 100.0

    def select_scenario(self) -> str:
        if self.config.scenario != "all":
            return self.config.scenario

        return self.rng.choice(
            [
                item.value
                for item in FailureType
                if item != FailureType.NORMAL
            ]
        )

    def inject_failure(
        self,
        telemetry: MotorTelemetry,
        scenario: str,
    ) -> None:
        rng = self.rng

        if scenario == "overheating":
            telemetry.motor_temperature_c = rng.uniform(
                96, 125
            )

        elif scenario == "overcurrent":
            telemetry.current_a = rng.uniform(32, 48)

        elif scenario == "bearing_wear":
            telemetry.bearing_wear_percent = rng.uniform(
                91, 100
            )
            telemetry.bearing_temperature_c = rng.uniform(
                75, 95
            )

        elif scenario == "excessive_vibration":
            telemetry.vibration_mm_s = rng.uniform(
                13, 25
            )

        elif scenario == "encoder_failure":
            telemetry.encoder_error_deg = rng.uniform(
                13, 30
            )

        elif scenario == "rotor_stall":
            telemetry.commanded_speed_rpm = rng.uniform(
                300, 800
            )
            telemetry.actual_speed_rpm = rng.uniform(
                0, 3
            )
            telemetry.current_a = rng.uniform(
                22, 29
            )

        elif scenario == "winding_fault":
            telemetry.winding_resistance_ohm = rng.uniform(
                3.6, 5.0
            )

        elif scenario == "mechanical_overload":
            telemetry.torque_nm = rng.uniform(
                105, 150
            )

    def update_motor(
        self,
        motor: SimulatedMotor,
        step: int,
    ) -> None:
        if motor.simulated_motion_inhibited:
            motor.telemetry.timestamp_s = (
                step * self.config.delta_time_s
            )
            return

        timestamp_s = step * self.config.delta_time_s

        self.update_baseline(
            motor.telemetry,
            timestamp_s,
        )

        should_inject = (
            step == 1
            or self.rng.random()
            < self.config.event_probability
        )

        if should_inject:
            scenario = self.select_scenario()

            if scenario != "normal":
                self.inject_failure(
                    motor.telemetry,
                    scenario,
                )

        motor.telemetry.validate()


# ============================================================
# Motor Failure Detector
# ============================================================

class MotorFailureDetector:
    def __init__(self, thresholds: MotorThresholds):
        self.t = thresholds

    def detect(
        self,
        telemetry: MotorTelemetry,
    ) -> list[tuple[FailureType, Severity, str]]:
        findings: list[
            tuple[FailureType, Severity, str]
        ] = []

        def check(
            value: float,
            warning: float,
            critical: float,
            failure_type: FailureType,
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
            FailureType.OVERHEATING,
            "motor temperature (C)",
        )

        check(
            telemetry.current_a,
            self.t.current_warning_a,
            self.t.current_critical_a,
            FailureType.OVERCURRENT,
            "motor current (A)",
        )

        check(
            telemetry.vibration_mm_s,
            self.t.vibration_warning_mm_s,
            self.t.vibration_critical_mm_s,
            FailureType.EXCESSIVE_VIBRATION,
            "vibration (mm/s)",
        )

        check(
            telemetry.torque_nm,
            self.t.torque_warning_nm,
            self.t.torque_critical_nm,
            FailureType.MECHANICAL_OVERLOAD,
            "motor torque (Nm)",
        )

        check(
            telemetry.encoder_error_deg,
            self.t.encoder_warning_error_deg,
            self.t.encoder_critical_error_deg,
            FailureType.ENCODER_FAILURE,
            "encoder error (deg)",
        )

        check(
            telemetry.bearing_wear_percent,
            self.t.bearing_warning_wear_percent,
            self.t.bearing_critical_wear_percent,
            FailureType.BEARING_WEAR,
            "bearing wear (%)",
        )

        check(
            telemetry.winding_resistance_ohm,
            self.t.winding_warning_resistance_ohm,
            self.t.winding_critical_resistance_ohm,
            FailureType.WINDING_FAULT,
            "winding resistance (ohm)",
        )

        stalled = (
            telemetry.actual_speed_rpm
            <= self.t.stall_speed_threshold_rpm
            and telemetry.commanded_speed_rpm
            >= self.t.stall_command_threshold_rpm
            and telemetry.current_a
            >= self.t.stall_current_threshold_a
        )

        if stalled:
            findings.append((
                FailureType.ROTOR_STALL,
                Severity.CRITICAL,
                "Motor rotor stalled despite simulated speed demand",
            ))

        return findings


# ============================================================
# Health / Failure / RUL Estimator
# ============================================================

class MotorHealthEstimator:
    """
    Illustrative heuristic estimator.

    These scores are not trained, calibrated,
    or suitable for production maintenance decisions.
    """

    def __init__(self, thresholds: MotorThresholds):
        self.t = thresholds

    def estimate(
        self,
        telemetry: MotorTelemetry,
    ) -> tuple[float, float, float, HealthState]:
        thermal_risk = clamp(
            (
                telemetry.motor_temperature_c - 35.0
            ) / (
                self.t.temperature_critical_c - 35.0
            )
        )

        current_risk = clamp(
            telemetry.current_a
            / self.t.current_critical_a
        )

        vibration_risk = clamp(
            telemetry.vibration_mm_s
            / self.t.vibration_critical_mm_s
        )

        wear_risk = clamp(
            telemetry.bearing_wear_percent / 100.0
        )

        encoder_risk = clamp(
            telemetry.encoder_error_deg
            / self.t.encoder_critical_error_deg
        )

        winding_risk = clamp(
            telemetry.winding_resistance_ohm
            / self.t.winding_critical_resistance_ohm
        )

        torque_risk = clamp(
            telemetry.torque_nm
            / self.t.torque_critical_nm
        )

        risk = (
            0.20 * thermal_risk
            + 0.16 * current_risk
            + 0.16 * vibration_risk
            + 0.16 * wear_risk
            + 0.12 * encoder_risk
            + 0.10 * winding_risk
            + 0.10 * torque_risk
        )

        health_score = clamp(1.0 - risk)

        failure_probability = clamp(
            risk ** 1.4
        )

        estimated_rul_hours = max(
            1.0,
            12000.0 * health_score ** 2
            - telemetry.operating_hours * 0.05,
        )

        if health_score < 0.35:
            state = HealthState.CRITICAL
        elif health_score < 0.6:
            state = HealthState.DEGRADED
        elif health_score < 0.8:
            state = HealthState.WARNING
        else:
            state = HealthState.HEALTHY

        return (
            round(health_score, 4),
            round(failure_probability, 4),
            round(estimated_rul_hours, 2),
            state,
        )


# ============================================================
# Maintenance Advisor
# ============================================================

class MotorMaintenanceAdvisor:
    RECOMMENDATIONS = {
        FailureType.NORMAL: (
            "Continue synthetic condition monitoring"
        ),
        FailureType.OVERHEATING: (
            "Inspect cooling, ventilation, motor load, "
            "and thermal sensors"
        ),
        FailureType.OVERCURRENT: (
            "Inspect electrical load, drive electronics, "
            "wiring, and current measurement"
        ),
        FailureType.BEARING_WEAR: (
            "Inspect bearing condition, lubrication, "
            "shaft alignment, and replacement needs"
        ),
        FailureType.EXCESSIVE_VIBRATION: (
            "Inspect balance, alignment, mounting, "
            "bearings, and coupling condition"
        ),
        FailureType.ENCODER_FAILURE: (
            "Inspect encoder wiring, calibration, "
            "signal quality, and feedback integrity"
        ),
        FailureType.ROTOR_STALL: (
            "Inspect mechanical obstruction, drive faults, "
            "and rotor condition"
        ),
        FailureType.WINDING_FAULT: (
            "Request qualified electrical inspection of "
            "motor windings and insulation"
        ),
        FailureType.MECHANICAL_OVERLOAD: (
            "Inspect drivetrain loading, gearbox resistance, "
            "and torque demand"
        ),
    }

    def recommend(
        self,
        failure_type: FailureType,
        severity: Severity,
    ) -> str:
        base = self.RECOMMENDATIONS[failure_type]

        if severity == Severity.CRITICAL:
            return (
                "SIMULATION: latch safe-state recommendation; "
                "require qualified human inspection. "
                + base
            )

        if severity in (
            Severity.WARNING,
            Severity.HIGH,
        ):
            return (
                "SIMULATION: schedule maintenance review. "
                + base
            )

        return base


# ============================================================
# Simulation Engine
# ============================================================

class MotorFailureSimulation:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.config.validate()

        self.generator = MotorTelemetryGenerator(
            config
        )

        self.detector = MotorFailureDetector(
            config.thresholds
        )

        self.health_estimator = MotorHealthEstimator(
            config.thresholds
        )

        self.maintenance_advisor = MotorMaintenanceAdvisor()

        self.robots = self.generator.generate_robots()

        self.events: list[MotorFailureEvent] = []
        self.telemetry_rows: list[dict[str, Any]] = []

        self.recorded_signatures: set[
            tuple[str, str]
        ] = set()

    def create_event(
        self,
        robot: SimulatedRobot,
        motor: SimulatedMotor,
        failure_type: FailureType,
        severity: Severity,
        description: str,
        timestamp_s: float,
    ) -> None:
        signature = (
            motor.motor_id,
            failure_type.value,
        )

        if signature in self.recorded_signatures:
            return

        self.recorded_signatures.add(signature)

        if severity == Severity.CRITICAL:
            robot.simulated_estop_latched = True
            robot.human_review_required = True

            for robot_motor in robot.motors:
                robot_motor.simulated_motion_inhibited = True
                robot_motor.human_review_required = True

        event = MotorFailureEvent(
            event_id=stable_event_id(
                motor.motor_id,
                failure_type,
                timestamp_s,
            ),
            robot_id=robot.robot_id,
            motor_id=motor.motor_id,
            timestamp_s=round(timestamp_s, 4),
            failure_type=failure_type.value,
            severity=severity.value,
            description=description,
            health_score=motor.health_score,
            failure_probability=(
                motor.failure_probability
            ),
            estimated_rul_hours=(
                motor.estimated_rul_hours
            ),
            maintenance_recommendation=(
                self.maintenance_advisor.recommend(
                    failure_type,
                    severity,
                )
            ),
            simulated_motion_inhibited=(
                motor.simulated_motion_inhibited
            ),
            simulated_estop_latched=(
                robot.simulated_estop_latched
            ),
            human_review_required=True,
            simulation_only=True,
        )

        self.events.append(event)

    def record_telemetry(
        self,
        robot: SimulatedRobot,
        motor: SimulatedMotor,
        timestamp_s: float,
    ) -> None:
        telemetry = asdict(motor.telemetry)

        self.telemetry_rows.append({
            "timestamp_s": round(timestamp_s, 4),
            "robot_id": robot.robot_id,
            "robot_type": robot.robot_type,
            "motor_id": motor.motor_id,
            **telemetry,
            "health_score": motor.health_score,
            "failure_probability": (
                motor.failure_probability
            ),
            "estimated_rul_hours": (
                motor.estimated_rul_hours
            ),
            "health_state": motor.health_state.value,
            "simulated_motion_inhibited": (
                motor.simulated_motion_inhibited
            ),
            "simulated_estop_latched": (
                robot.simulated_estop_latched
            ),
            "simulation_only": True,
        })

    def run_step(self, step: int) -> None:
        timestamp_s = (
            step * self.config.delta_time_s
        )

        for robot in self.robots:
            for motor in robot.motors:
                self.generator.update_motor(
                    motor,
                    step,
                )

                (
                    motor.health_score,
                    motor.failure_probability,
                    motor.estimated_rul_hours,
                    motor.health_state,
                ) = self.health_estimator.estimate(
                    motor.telemetry
                )

                findings = self.detector.detect(
                    motor.telemetry
                )

                for (
                    failure_type,
                    severity,
                    description,
                ) in findings:
                    self.create_event(
                        robot,
                        motor,
                        failure_type,
                        severity,
                        description,
                        timestamp_s,
                    )

                self.record_telemetry(
                    robot,
                    motor,
                    timestamp_s,
                )

    def run(self) -> dict[str, Any]:
        LOG.info(
            "Starting synthetic motor-failure simulation"
        )

        for step in range(self.config.steps):
            self.run_step(step)

        return self.build_report()

    def build_report(self) -> dict[str, Any]:
        motors = [
            motor
            for robot in self.robots
            for motor in robot.motors
        ]

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

        mean_health = (
            sum(
                motor.health_score
                for motor in motors
            ) / len(motors)
            if motors
            else 0.0
        )

        return {
            "status": "completed",
            "schema_version": SCHEMA_VERSION,
            "simulation_only": True,
            "safety_certified": False,
            "physical_control_executed": False,
            "physical_estop_reset_executed": False,
            "configuration": {
                "robots": self.config.robots,
                "motors_per_robot": (
                    self.config.motors_per_robot
                ),
                "steps": self.config.steps,
                "delta_time_s": (
                    self.config.delta_time_s
                ),
                "seed": self.config.seed,
                "scenario": self.config.scenario,
                "event_probability": (
                    self.config.event_probability
                ),
                "degradation_multiplier": (
                    self.config.degradation_multiplier
                ),
            },
            "summary": {
                "total_robots": len(self.robots),
                "total_motors": len(motors),
                "telemetry_records": (
                    len(self.telemetry_rows)
                ),
                "failure_events": len(self.events),
                "failure_counts": dict(failure_counts),
                "severity_counts": dict(severity_counts),
                "simulated_estop_count": (
                    len(stopped_robots)
                ),
                "mean_motor_health_score": round(
                    mean_health, 4
                ),
                "critical_motors": sum(
                    motor.health_state
                    == HealthState.CRITICAL
                    for motor in motors
                ),
                "degraded_motors": sum(
                    motor.health_state
                    == HealthState.DEGRADED
                    for motor in motors
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
                    "human_review_required": (
                        robot.human_review_required
                    ),
                    "motors": [
                        {
                            "motor_id": motor.motor_id,
                            "health_state": (
                                motor.health_state.value
                            ),
                            "health_score": (
                                motor.health_score
                            ),
                            "failure_probability": (
                                motor.failure_probability
                            ),
                            "estimated_rul_hours": (
                                motor.estimated_rul_hours
                            ),
                            "simulated_motion_inhibited": (
                                motor.simulated_motion_inhibited
                            ),
                        }
                        for motor in robot.motors
                    ],
                }
                for robot in self.robots
            ],
            "events": [
                asdict(event)
                for event in self.events
            ],
            "limitations": [
                "All telemetry and failures are synthetic.",
                "Failure probabilities are heuristic, not calibrated.",
                "RUL estimates are illustrative, not model-validated.",
                "Motor physics and electrical behavior are simplified.",
                "Safety thresholds are illustrative, not certified.",
                "No real hardware or emergency-stop system is controlled.",
            ],
        }


# ============================================================
# Export
# ============================================================

def export_json(
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


def export_csv(
    path: str | Path,
    rows: list[dict[str, Any]],
    columns: list[str] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if columns is None:
        if not rows:
            return
        columns = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate robotic motor faults and degradation"
        )
    )

    parser.add_argument(
        "--robots",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--motors-per-robot",
        type=int,
        default=4,
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
        "--degradation-multiplier",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    parser.add_argument(
        "--csv",
        default=None,
    )

    parser.add_argument(
        "--events-csv",
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
        motors_per_robot=args.motors_per_robot,
        steps=args.steps,
        delta_time_s=args.dt,
        seed=args.seed,
        scenario=args.scenario,
        event_probability=args.event_probability,
        degradation_multiplier=(
            args.degradation_multiplier
        ),
    )

    config.validate()

    simulation = MotorFailureSimulation(config)
    report = simulation.run()

    if args.output:
        export_json(args.output, report)

    if args.csv:
        export_csv(
            args.csv,
            simulation.telemetry_rows,
        )

    if args.events_csv:
        export_csv(
            args.events_csv,
            [asdict(event) for event in simulation.events],
            columns=list(
                MotorFailureEvent.__dataclass_fields__.keys()
            ),
        )

    print(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )

    LOG.info("Motor-failure simulation completed")
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
        LOG.error("Simulation failed: %s", exc)
        sys.exit(1)

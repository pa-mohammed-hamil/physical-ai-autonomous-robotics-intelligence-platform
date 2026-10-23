
#!/usr/bin/env python3
"""
Physical AI & Autonomous Robotics Intelligence Platform
File: scripts/simulate_overheating.py

Offline thermal-fault and overheating simulator.

Features:
- Mobile robot, robotic arm, humanoid, quadruped
- Motor, battery, actuator, controller, bearing temperatures
- Ambient temperature and thermal load
- Simplified thermal dynamics
- Cooling system failure
- Motor overload
- Battery overheating
- Controller overheating
- Actuator overheating
- Bearing overheating
- Thermal sensor fault
- Progressive heating and cooling
- Thermal risk classification
- Synthetic anomaly scoring
- Heuristic failure probability and RUL
- Simulated emergency-stop latch
- Human review and maintenance recommendations
- JSON and CSV reports
- Deterministic random seed
- No external dependencies

Examples:
    python scripts/simulate_overheating.py

    python scripts/simulate_overheating.py \
        --scenario cooling_failure --robots 5

    python scripts/simulate_overheating.py \
        --scenario battery_overheating --steps 300

    python scripts/simulate_overheating.py \
        --scenario all --robots 10 --steps 500 \
        --output reports/overheating.json \
        --csv reports/thermal_telemetry.csv \
        --events-csv reports/thermal_events.csv

Safety:
- Synthetic simulation only.
- No hardware actuation.
- No real thermal shutdown command.
- No real emergency-stop reset.
- Not a certified thermal safety system.
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


LOGGER = logging.getLogger("overheating_simulator")

SCHEMA_VERSION = "1.0.0"

NAMESPACE = uuid.UUID(
    "e84e8e23-9f23-4a42-9fb1-87522e71531c"
)


# ============================================================
# Enumerations
# ============================================================

class ThermalScenario(str, Enum):
    NORMAL = "normal"
    COOLING_FAILURE = "cooling_failure"
    MOTOR_OVERLOAD = "motor_overload"
    BATTERY_OVERHEATING = "battery_overheating"
    CONTROLLER_OVERHEATING = "controller_overheating"
    ACTUATOR_OVERHEATING = "actuator_overheating"
    BEARING_OVERHEATING = "bearing_overheating"
    SENSOR_FAULT = "sensor_fault"
    HIGH_AMBIENT = "high_ambient"
    ALL = "all"


class ThermalSeverity(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class ThermalState(str, Enum):
    HEALTHY = "healthy"
    WARM = "warm"
    OVERHEATING = "overheating"
    EMERGENCY_STOPPED = "emergency_stopped"


SCENARIOS = tuple(
    scenario.value
    for scenario in ThermalScenario
)


# ============================================================
# Thresholds
# ============================================================

@dataclass
class ThermalThresholds:
    motor_warning_c: float = 75.0
    motor_critical_c: float = 95.0

    battery_warning_c: float = 48.0
    battery_critical_c: float = 65.0

    actuator_warning_c: float = 75.0
    actuator_critical_c: float = 95.0

    controller_warning_c: float = 70.0
    controller_critical_c: float = 90.0

    bearing_warning_c: float = 70.0
    bearing_critical_c: float = 90.0

    heating_rate_warning_c_s: float = 1.5
    heating_rate_critical_c_s: float = 4.0

    sensor_disagreement_warning_c: float = 8.0
    sensor_disagreement_critical_c: float = 20.0

    def validate(self) -> None:
        values = asdict(self)

        for name, value in values.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} must be finite and positive"
                )

        for prefix in (
            "motor",
            "battery",
            "actuator",
            "controller",
            "bearing",
        ):
            warning = getattr(
                self, f"{prefix}_warning_c"
            )
            critical = getattr(
                self, f"{prefix}_critical_c"
            )

            if warning >= critical:
                raise ValueError(
                    f"{prefix} warning must be below critical"
                )

        if (
            self.heating_rate_warning_c_s
            >= self.heating_rate_critical_c_s
        ):
            raise ValueError(
                "Heating-rate warning must be below critical"
            )

        if (
            self.sensor_disagreement_warning_c
            >= self.sensor_disagreement_critical_c
        ):
            raise ValueError(
                "Sensor disagreement warning must be below critical"
            )


# ============================================================
# Configuration
# ============================================================

@dataclass
class SimulationConfig:
    robots: int = 5
    steps: int = 300
    delta_time_s: float = 0.2

    seed: int = 42
    scenario: str = "all"

    event_probability: float = 0.015

    ambient_temperature_c: float = 28.0

    thresholds: ThermalThresholds = field(
        default_factory=ThermalThresholds
    )

    def validate(self) -> None:
        if not 1 <= self.robots <= 100:
            raise ValueError(
                "robots must be between 1 and 100"
            )

        if not 1 <= self.steps <= 100000:
            raise ValueError(
                "steps must be between 1 and 100000"
            )

        if not (
            math.isfinite(self.delta_time_s)
            and 0 < self.delta_time_s <= 10
        ):
            raise ValueError(
                "delta_time_s must be between 0 and 10"
            )

        if not (
            math.isfinite(self.event_probability)
            and 0 <= self.event_probability <= 1
        ):
            raise ValueError(
                "event_probability must be between 0 and 1"
            )

        if not (
            math.isfinite(self.ambient_temperature_c)
            and -30 <= self.ambient_temperature_c <= 80
        ):
            raise ValueError(
                "Invalid ambient temperature"
            )

        if self.scenario not in SCENARIOS:
            raise ValueError(
                f"Unsupported scenario: {self.scenario}"
            )

        self.thresholds.validate()


# ============================================================
# Telemetry
# ============================================================

@dataclass
class ThermalTelemetry:
    timestamp_s: float = 0.0

    ambient_temperature_c: float = 28.0

    motor_temperature_c: float = 38.0
    battery_temperature_c: float = 32.0
    actuator_temperature_c: float = 36.0
    controller_temperature_c: float = 35.0
    bearing_temperature_c: float = 34.0

    motor_temperature_sensor_c: float = 38.0

    motor_heating_rate_c_s: float = 0.0
    battery_heating_rate_c_s: float = 0.0

    motor_load_percent: float = 35.0
    actuator_load_percent: float = 30.0
    battery_current_a: float = 8.0

    cooling_efficiency: float = 1.0
    fan_operational: bool = True

    sensor_health_score: float = 1.0

    battery_soc_percent: float = 85.0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool):
                continue

            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite telemetry field: {name}"
                )


@dataclass
class SimulatedRobot:
    robot_id: str
    robot_type: str

    telemetry: ThermalTelemetry = field(
        default_factory=ThermalTelemetry
    )

    thermal_state: ThermalState = ThermalState.HEALTHY

    risk_score: float = 0.0
    anomaly_score: float = 0.0
    failure_probability: float = 0.0
    estimated_rul_hours: float = 10000.0

    simulated_estop_latched: bool = False
    simulated_motion_inhibited: bool = False
    human_review_required: bool = False

    active_faults: set[str] = field(
        default_factory=set
    )


@dataclass
class ThermalEvent:
    event_id: str
    robot_id: str

    timestamp_s: float

    component: str
    event_type: str
    severity: str

    measured_value: float
    threshold_value: float

    risk_score: float
    failure_probability: float

    description: str
    recommendation: str

    simulated_estop_latched: bool
    human_review_required: bool = True
    simulation_only: bool = True


# ============================================================
# Helpers
# ============================================================

def clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    return max(low, min(high, value))


def stable_event_id(
    robot_id: str,
    event_type: str,
    timestamp_s: float,
) -> str:
    return str(
        uuid.uuid5(
            NAMESPACE,
            f"{robot_id}:{event_type}:{timestamp_s:.6f}",
        )
    )


# ============================================================
# Thermal Dynamics
# ============================================================

class ThermalDynamics:
    """
    Simplified first-order thermal model.

    dT/dt = heat_input - cooling_coefficient*(T-ambient)

    This is an illustrative model, not a calibrated
    electrothermal or battery safety simulation.
    """

    @staticmethod
    def step_temperature(
        current_c: float,
        ambient_c: float,
        heat_input_c_s: float,
        cooling_coefficient_s: float,
        dt: float,
    ) -> float:
        cooling = (
            cooling_coefficient_s
            * max(0.0, current_c - ambient_c)
        )

        derivative = (
            heat_input_c_s - cooling
        )

        return current_c + derivative * dt


# ============================================================
# Telemetry Generator
# ============================================================

class ThermalTelemetryGenerator:
    ROBOT_TYPES = (
        "mobile_robot",
        "robotic_arm",
        "humanoid",
        "quadruped",
    )

    def __init__(
        self,
        config: SimulationConfig,
    ) -> None:
        self.config = config
        self.rng = random.Random(config.seed)

    def generate_robots(self) -> list[SimulatedRobot]:
        robots = []

        for index in range(self.config.robots):
            robot_type = self.ROBOT_TYPES[
                index % len(self.ROBOT_TYPES)
            ]

            ambient = (
                self.config.ambient_temperature_c
            )

            telemetry = ThermalTelemetry(
                ambient_temperature_c=ambient,
                motor_temperature_c=ambient + 10,
                battery_temperature_c=ambient + 4,
                actuator_temperature_c=ambient + 8,
                controller_temperature_c=ambient + 7,
                bearing_temperature_c=ambient + 6,
                motor_temperature_sensor_c=ambient + 10,
            )

            robots.append(
                SimulatedRobot(
                    robot_id=f"SIM-ROBOT-{index + 1:03d}",
                    robot_type=robot_type,
                    telemetry=telemetry,
                )
            )

        return robots

    def choose_scenario(self) -> str:
        if self.config.scenario != "all":
            return self.config.scenario

        return self.rng.choice([
            scenario.value
            for scenario in ThermalScenario
            if scenario not in (
                ThermalScenario.NORMAL,
                ThermalScenario.ALL,
            )
        ])

    def inject_fault(
        self,
        robot: SimulatedRobot,
        scenario: str,
    ) -> None:
        t = robot.telemetry

        if scenario == "cooling_failure":
            t.cooling_efficiency = 0.05
            t.fan_operational = False

        elif scenario == "motor_overload":
            t.motor_load_percent = 145.0

        elif scenario == "battery_overheating":
            t.battery_current_a = 85.0
            t.battery_temperature_c = max(
                t.battery_temperature_c,
                58.0,
            )

        elif scenario == "controller_overheating":
            t.controller_temperature_c = max(
                t.controller_temperature_c,
                82.0,
            )

        elif scenario == "actuator_overheating":
            t.actuator_load_percent = 150.0
            t.actuator_temperature_c = max(
                t.actuator_temperature_c,
                84.0,
            )

        elif scenario == "bearing_overheating":
            t.bearing_temperature_c = max(
                t.bearing_temperature_c,
                83.0,
            )

        elif scenario == "sensor_fault":
            t.motor_temperature_sensor_c += 30.0
            t.sensor_health_score = 0.15

        elif scenario == "high_ambient":
            t.ambient_temperature_c = 60.0

        robot.active_faults.add(scenario)

    def update(
        self,
        robot: SimulatedRobot,
        step: int,
    ) -> None:
        t = robot.telemetry
        dt = self.config.delta_time_s

        t.timestamp_s = step * dt

        # Deterministic fault injection for named scenarios.
        # A fault remains active once injected.
        should_inject = (
            step == 1
            or (
                self.config.scenario == "all"
                and self.rng.random()
                < self.config.event_probability
            )
        )

        if should_inject:
            scenario = self.choose_scenario()

            if scenario != "normal":
                self.inject_fault(
                    robot,
                    scenario,
                )

        if robot.simulated_estop_latched:
            # Thermal evolution continues after simulated
            # motion inhibition, but load is reduced.
            t.motor_load_percent = 0.0
            t.actuator_load_percent = 0.0
            t.battery_current_a = 0.0

        else:
            if "motor_overload" not in robot.active_faults:
                t.motor_load_percent = self.rng.uniform(
                    20, 70
                )

            if "actuator_overheating" not in robot.active_faults:
                t.actuator_load_percent = self.rng.uniform(
                    15, 65
                )

            if "battery_overheating" not in robot.active_faults:
                t.battery_current_a = self.rng.uniform(
                    5, 20
                )

        cooling_efficiency = clamp(
            t.cooling_efficiency
        )

        cooling_factor = (
            cooling_efficiency
            if t.fan_operational
            else cooling_efficiency * 0.5
        )

        previous_motor = t.motor_temperature_c
        previous_battery = t.battery_temperature_c

        # Thermal dynamics are deliberately simplified.
        t.motor_temperature_c = (
            ThermalDynamics.step_temperature(
                t.motor_temperature_c,
                t.ambient_temperature_c,
                heat_input_c_s=(
                    0.012 * t.motor_load_percent
                ),
                cooling_coefficient_s=(
                    0.03 * cooling_factor
                ),
                dt=dt,
            )
        )

        t.battery_temperature_c = (
            ThermalDynamics.step_temperature(
                t.battery_temperature_c,
                t.ambient_temperature_c,
                heat_input_c_s=(
                    0.015 * t.battery_current_a
                ),
                cooling_coefficient_s=(
                    0.025 * cooling_factor
                ),
                dt=dt,
            )
        )

        t.actuator_temperature_c = (
            ThermalDynamics.step_temperature(
                t.actuator_temperature_c,
                t.ambient_temperature_c,
                heat_input_c_s=(
                    0.010 * t.actuator_load_percent
                ),
                cooling_coefficient_s=(
                    0.03 * cooling_factor
                ),
                dt=dt,
            )
        )

        t.controller_temperature_c = (
            ThermalDynamics.step_temperature(
                t.controller_temperature_c,
                t.ambient_temperature_c,
                heat_input_c_s=0.35,
                cooling_coefficient_s=(
                    0.025 * cooling_factor
                ),
                dt=dt,
            )
        )

        t.bearing_temperature_c = (
            ThermalDynamics.step_temperature(
                t.bearing_temperature_c,
                t.ambient_temperature_c,
                heat_input_c_s=(
                    0.004 * t.motor_load_percent
                ),
                cooling_coefficient_s=(
                    0.02 * cooling_factor
                ),
                dt=dt,
            )
        )

        t.motor_heating_rate_c_s = (
            t.motor_temperature_c - previous_motor
        ) / dt

        t.battery_heating_rate_c_s = (
            t.battery_temperature_c - previous_battery
        ) / dt

        if "sensor_fault" not in robot.active_faults:
            t.motor_temperature_sensor_c = (
                t.motor_temperature_c
                + self.rng.uniform(-0.3, 0.3)
            )

        t.battery_soc_percent = max(
            0.0,
            t.battery_soc_percent
            - t.battery_current_a * dt / 360000.0,
        )

        t.validate()


# ============================================================
# Thermal Fault Detector
# ============================================================

class ThermalFaultDetector:
    def __init__(
        self,
        thresholds: ThermalThresholds,
    ) -> None:
        self.t = thresholds

    def detect(
        self,
        telemetry: ThermalTelemetry,
    ) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []

        components = (
            (
                "motor",
                telemetry.motor_temperature_c,
                self.t.motor_warning_c,
                self.t.motor_critical_c,
            ),
            (
                "battery",
                telemetry.battery_temperature_c,
                self.t.battery_warning_c,
                self.t.battery_critical_c,
            ),
            (
                "actuator",
                telemetry.actuator_temperature_c,
                self.t.actuator_warning_c,
                self.t.actuator_critical_c,
            ),
            (
                "controller",
                telemetry.controller_temperature_c,
                self.t.controller_warning_c,
                self.t.controller_critical_c,
            ),
            (
                "bearing",
                telemetry.bearing_temperature_c,
                self.t.bearing_warning_c,
                self.t.bearing_critical_c,
            ),
        )

        for component, value, warning, critical in components:
            if value >= critical:
                findings.append({
                    "component": component,
                    "event_type": f"{component}_overheating",
                    "severity": ThermalSeverity.CRITICAL,
                    "measured_value": value,
                    "threshold_value": critical,
                    "description": (
                        f"{component.title()} temperature "
                        f"critically high: {value:.2f} C"
                    ),
                })

            elif value >= warning:
                findings.append({
                    "component": component,
                    "event_type": f"{component}_overheating",
                    "severity": ThermalSeverity.WARNING,
                    "measured_value": value,
                    "threshold_value": warning,
                    "description": (
                        f"{component.title()} temperature "
                        f"above warning threshold: {value:.2f} C"
                    ),
                })

        heating_rates = (
            (
                "motor",
                telemetry.motor_heating_rate_c_s,
            ),
            (
                "battery",
                telemetry.battery_heating_rate_c_s,
            ),
        )

        for component, rate in heating_rates:
            if rate >= self.t.heating_rate_critical_c_s:
                findings.append({
                    "component": component,
                    "event_type": "rapid_heating",
                    "severity": ThermalSeverity.CRITICAL,
                    "measured_value": rate,
                    "threshold_value": (
                        self.t.heating_rate_critical_c_s
                    ),
                    "description": (
                        f"Critical {component} heating rate: "
                        f"{rate:.2f} C/s"
                    ),
                })

            elif rate >= self.t.heating_rate_warning_c_s:
                findings.append({
                    "component": component,
                    "event_type": "rapid_heating",
                    "severity": ThermalSeverity.WARNING,
                    "measured_value": rate,
                    "threshold_value": (
                        self.t.heating_rate_warning_c_s
                    ),
                    "description": (
                        f"Elevated {component} heating rate: "
                        f"{rate:.2f} C/s"
                    ),
                })

        sensor_disagreement = abs(
            telemetry.motor_temperature_sensor_c
            - telemetry.motor_temperature_c
        )

        if sensor_disagreement >= (
            self.t.sensor_disagreement_critical_c
        ):
            findings.append({
                "component": "motor_sensor",
                "event_type": "thermal_sensor_fault",
                "severity": ThermalSeverity.CRITICAL,
                "measured_value": sensor_disagreement,
                "threshold_value": (
                    self.t.sensor_disagreement_critical_c
                ),
                "description": (
                    "Critical synthetic temperature "
                    "sensor disagreement"
                ),
            })

        elif sensor_disagreement >= (
            self.t.sensor_disagreement_warning_c
        ):
            findings.append({
                "component": "motor_sensor",
                "event_type": "thermal_sensor_fault",
                "severity": ThermalSeverity.WARNING,
                "measured_value": sensor_disagreement,
                "threshold_value": (
                    self.t.sensor_disagreement_warning_c
                ),
                "description": (
                    "Elevated synthetic temperature "
                    "sensor disagreement"
                ),
            })

        if not telemetry.fan_operational:
            findings.append({
                "component": "cooling",
                "event_type": "cooling_failure",
                "severity": ThermalSeverity.HIGH,
                "measured_value": (
                    telemetry.cooling_efficiency
                ),
                "threshold_value": 0.2,
                "description": (
                    "Synthetic cooling fan is unavailable"
                ),
            })

        return findings


# ============================================================
# Thermal Risk Estimator
# ============================================================

class ThermalRiskEstimator:
    """
    Heuristic risk estimates.

    Failure probability is not calibrated against
    real-world failure data.
    """

    def __init__(
        self,
        thresholds: ThermalThresholds,
    ) -> None:
        self.t = thresholds

    def normalized_temperature_risk(
        self,
        temperature: float,
        warning: float,
        critical: float,
    ) -> float:
        return clamp(
            (temperature - warning)
            / max(critical - warning, 1e-9)
        )

    def estimate(
        self,
        telemetry: ThermalTelemetry,
    ) -> tuple[float, float, float, float]:
        risks = [
            self.normalized_temperature_risk(
                telemetry.motor_temperature_c,
                self.t.motor_warning_c,
                self.t.motor_critical_c,
            ),
            self.normalized_temperature_risk(
                telemetry.battery_temperature_c,
                self.t.battery_warning_c,
                self.t.battery_critical_c,
            ),
            self.normalized_temperature_risk(
                telemetry.actuator_temperature_c,
                self.t.actuator_warning_c,
                self.t.actuator_critical_c,
            ),
            self.normalized_temperature_risk(
                telemetry.controller_temperature_c,
                self.t.controller_warning_c,
                self.t.controller_critical_c,
            ),
            self.normalized_temperature_risk(
                telemetry.bearing_temperature_c,
                self.t.bearing_warning_c,
                self.t.bearing_critical_c,
            ),
        ]

        thermal_risk = max(risks)

        cooling_risk = (
            1.0 - clamp(telemetry.cooling_efficiency)
        )

        sensor_risk = (
            1.0 - clamp(telemetry.sensor_health_score)
        )

        heating_rate_risk = clamp(
            max(
                telemetry.motor_heating_rate_c_s,
                telemetry.battery_heating_rate_c_s,
            ) / self.t.heating_rate_critical_c_s
        )

        risk_score = clamp(
            0.55 * thermal_risk
            + 0.20 * cooling_risk
            + 0.15 * heating_rate_risk
            + 0.10 * sensor_risk
        )

        anomaly_score = clamp(
            max(
                thermal_risk,
                cooling_risk,
                sensor_risk,
                heating_rate_risk,
            )
        )

        failure_probability = clamp(
            risk_score ** 1.5
        )

        estimated_rul_hours = max(
            1.0,
            10000.0 * (1.0 - risk_score) ** 2,
        )

        return (
            round(risk_score, 4),
            round(anomaly_score, 4),
            round(failure_probability, 4),
            round(estimated_rul_hours, 2),
        )


# ============================================================
# Maintenance Advisor
# ============================================================

class ThermalMaintenanceAdvisor:
    RECOMMENDATIONS = {
        "motor": (
            "Inspect motor load, ventilation, cooling, "
            "and thermal sensing"
        ),
        "battery": (
            "Request qualified battery safety inspection; "
            "review thermal history and electrical loading"
        ),
        "actuator": (
            "Inspect actuator duty cycle, mechanical "
            "resistance, and thermal condition"
        ),
        "controller": (
            "Inspect controller ventilation, electronic "
            "loading, and cooling"
        ),
        "bearing": (
            "Inspect lubrication, bearing wear, "
            "alignment, and friction"
        ),
        "motor_sensor": (
            "Inspect temperature sensor calibration "
            "and measurement integrity"
        ),
        "cooling": (
            "Inspect cooling fan, airflow, and "
            "thermal management components"
        ),
    }

    def recommend(
        self,
        component: str,
        severity: ThermalSeverity,
    ) -> str:
        recommendation = self.RECOMMENDATIONS.get(
            component,
            "Request qualified maintenance review",
        )

        if severity == ThermalSeverity.CRITICAL:
            return (
                "SIMULATION: latch emergency state and "
                "request human safety review. "
                + recommendation
            )

        if severity == ThermalSeverity.HIGH:
            return (
                "SIMULATION: prioritize maintenance. "
                + recommendation
            )

        return (
            "SIMULATION: monitor thermal condition. "
            + recommendation
        )


# ============================================================
# Simulation Engine
# ============================================================

class OverheatingSimulation:
    def __init__(
        self,
        config: SimulationConfig,
    ) -> None:
        self.config = config
        self.config.validate()

        self.generator = ThermalTelemetryGenerator(
            config
        )

        self.detector = ThermalFaultDetector(
            config.thresholds
        )

        self.risk_estimator = ThermalRiskEstimator(
            config.thresholds
        )

        self.advisor = ThermalMaintenanceAdvisor()

        self.robots = self.generator.generate_robots()

        self.events: list[ThermalEvent] = []
        self.telemetry_rows: list[dict[str, Any]] = []

        self.recorded_signatures: set[
            tuple[str, str, str]
        ] = set()

    def create_event(
        self,
        robot: SimulatedRobot,
        finding: dict[str, Any],
        timestamp_s: float,
    ) -> None:
        severity = finding["severity"]

        signature = (
            robot.robot_id,
            finding["component"],
            finding["event_type"],
        )

        if signature in self.recorded_signatures:
            return

        self.recorded_signatures.add(signature)

        if severity == ThermalSeverity.CRITICAL:
            robot.simulated_estop_latched = True
            robot.simulated_motion_inhibited = True
            robot.human_review_required = True
            robot.thermal_state = (
                ThermalState.EMERGENCY_STOPPED
            )

        elif severity == ThermalSeverity.HIGH:
            if not robot.simulated_estop_latched:
                robot.thermal_state = (
                    ThermalState.OVERHEATING
                )

        elif severity == ThermalSeverity.WARNING:
            if not robot.simulated_estop_latched:
                robot.thermal_state = ThermalState.WARM

        event = ThermalEvent(
            event_id=stable_event_id(
                robot.robot_id,
                (
                    finding["component"]
                    + ":"
                    + finding["event_type"]
                ),
                timestamp_s,
            ),
            robot_id=robot.robot_id,
            timestamp_s=round(timestamp_s, 4),
            component=finding["component"],
            event_type=finding["event_type"],
            severity=severity.value,
            measured_value=round(
                finding["measured_value"], 4
            ),
            threshold_value=round(
                finding["threshold_value"], 4
            ),
            risk_score=robot.risk_score,
            failure_probability=(
                robot.failure_probability
            ),
            description=finding["description"],
            recommendation=self.advisor.recommend(
                finding["component"],
                severity,
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
        timestamp_s: float,
    ) -> None:
        self.telemetry_rows.append({
            "timestamp_s": round(timestamp_s, 4),
            "robot_id": robot.robot_id,
            "robot_type": robot.robot_type,
            **asdict(robot.telemetry),
            "thermal_state": robot.thermal_state.value,
            "risk_score": robot.risk_score,
            "anomaly_score": robot.anomaly_score,
            "failure_probability": (
                robot.failure_probability
            ),
            "estimated_rul_hours": (
                robot.estimated_rul_hours
            ),
            "simulated_estop_latched": (
                robot.simulated_estop_latched
            ),
            "simulated_motion_inhibited": (
                robot.simulated_motion_inhibited
            ),
            "human_review_required": (
                robot.human_review_required
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
            self.generator.update(
                robot,
                step,
            )

            (
                robot.risk_score,
                robot.anomaly_score,
                robot.failure_probability,
                robot.estimated_rul_hours,
            ) = self.risk_estimator.estimate(
                robot.telemetry
            )

            findings = self.detector.detect(
                robot.telemetry
            )

            for finding in findings:
                self.create_event(
                    robot,
                    finding,
                    timestamp_s,
                )

            self.record_telemetry(
                robot,
                timestamp_s,
            )

    def run(self) -> dict[str, Any]:
        LOGGER.info(
            "Starting synthetic overheating simulation"
        )

        for step in range(self.config.steps):
            self.run_step(step)

        return self.build_report()

    def build_report(self) -> dict[str, Any]:
        severity_counts = Counter(
            event.severity
            for event in self.events
        )

        component_counts = Counter(
            event.component
            for event in self.events
        )

        stopped_robots = [
            robot.robot_id
            for robot in self.robots
            if robot.simulated_estop_latched
        ]

        maximum_risk = max(
            (
                robot.risk_score
                for robot in self.robots
            ),
            default=0.0,
        )

        mean_risk = (
            sum(
                robot.risk_score
                for robot in self.robots
            ) / len(self.robots)
            if self.robots
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
                "steps": self.config.steps,
                "delta_time_s": (
                    self.config.delta_time_s
                ),
                "seed": self.config.seed,
                "scenario": self.config.scenario,
                "event_probability": (
                    self.config.event_probability
                ),
                "ambient_temperature_c": (
                    self.config.ambient_temperature_c
                ),
            },
            "summary": {
                "total_robots": len(self.robots),
                "telemetry_records": (
                    len(self.telemetry_rows)
                ),
                "total_thermal_events": len(self.events),
                "severity_counts": dict(
                    severity_counts
                ),
                "component_counts": dict(
                    component_counts
                ),
                "simulated_estop_count": (
                    len(stopped_robots)
                ),
                "maximum_final_risk_score": round(
                    maximum_risk, 4
                ),
                "mean_final_risk_score": round(
                    mean_risk, 4
                ),
            },
            "simulated_estop_robots": stopped_robots,
            "robots": [
                {
                    "robot_id": robot.robot_id,
                    "robot_type": robot.robot_type,
                    "thermal_state": (
                        robot.thermal_state.value
                    ),
                    "risk_score": robot.risk_score,
                    "anomaly_score": robot.anomaly_score,
                    "failure_probability": (
                        robot.failure_probability
                    ),
                    "estimated_rul_hours": (
                        robot.estimated_rul_hours
                    ),
                    "simulated_estop_latched": (
                        robot.simulated_estop_latched
                    ),
                    "human_review_required": (
                        robot.human_review_required
                    ),
                    "active_faults": sorted(
                        robot.active_faults
                    ),
                    "telemetry": asdict(
                        robot.telemetry
                    ),
                }
                for robot in self.robots
            ],
            "events": [
                asdict(event)
                for event in self.events
            ],
            "limitations": [
                "Thermal dynamics are simplified and synthetic.",
                "No calibrated battery electrothermal model is used.",
                "Risk scores are illustrative heuristics.",
                "Failure probabilities are not calibrated.",
                "RUL estimates are not model-validated.",
                "Thresholds are not safety-certified.",
                "No physical robot or cooling system is controlled.",
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
        columns = list(rows[0])

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
            "Simulate robotic thermal faults and overheating"
        )
    )

    parser.add_argument(
        "--robots",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=0.2,
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
        default=0.015,
    )

    parser.add_argument(
        "--ambient",
        type=float,
        default=28.0,
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
        steps=args.steps,
        delta_time_s=args.dt,
        seed=args.seed,
        scenario=args.scenario,
        event_probability=args.event_probability,
        ambient_temperature_c=args.ambient,
    )

    config.validate()

    simulation = OverheatingSimulation(config)
    report = simulation.run()

    if args.output:
        export_json(
            args.output,
            report,
        )

    if args.csv:
        export_csv(
            args.csv,
            simulation.telemetry_rows,
        )

    if args.events_csv:
        export_csv(
            args.events_csv,
            [
                asdict(event)
                for event in simulation.events
            ],
            columns=list(
                ThermalEvent.__dataclass_fields__
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

    LOGGER.info(
        "Overheating simulation completed"
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
            "Overheating simulation failed: %s",
            exc,
        )

        sys.exit(1)

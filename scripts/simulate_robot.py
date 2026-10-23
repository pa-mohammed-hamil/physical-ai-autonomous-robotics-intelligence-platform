
#!/usr/bin/env python3
"""
Physical AI & Autonomous Robotics Intelligence Platform
File: scripts/simulate_robot.py

Offline, multi-robot simulation engine.

Features:
- Mobile robot, robotic arm, humanoid, quadruped
- Multi-robot fleet simulation
- 2D position, velocity, heading
- Synthetic joint telemetry
- Battery SOC, voltage and temperature
- Motor temperature, current, vibration
- Sensor and communication health
- Obstacle and human proximity
- Simplified collision risk
- Synthetic fault injection
- Motor failure, overheating, sensor failure
- Communication loss and battery emergency
- Simulated emergency-stop latch
- Heuristic health and RUL estimation
- JSON / CSV export
- Reproducible random seed

Safety:
- Simulation only
- No ROS2 publishers
- No motor drivers
- No hardware commands
- No real emergency-stop reset
- No physical robot control
- Not safety certified
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


LOG = logging.getLogger("simulate_robot")

SCHEMA_VERSION = "1.0.0"

EVENT_NAMESPACE = uuid.UUID(
    "e7a5a824-4909-43b2-9c44-8e62b03c2f60"
)


# ============================================================
# Enumerations
# ============================================================

class RobotType(str, Enum):
    MOBILE = "mobile_robot"
    ARM = "robotic_arm"
    HUMANOID = "humanoid"
    QUADRUPED = "quadruped"


class RobotState(str, Enum):
    IDLE = "idle"
    OPERATING = "operating"
    DEGRADED = "degraded"
    EMERGENCY_STOPPED = "emergency_stopped"


class FaultType(str, Enum):
    NONE = "none"
    MOTOR_FAILURE = "motor_failure"
    OVERHEATING = "overheating"
    SENSOR_FAILURE = "sensor_failure"
    COMMUNICATION_LOSS = "communication_loss"
    BATTERY_EMERGENCY = "battery_emergency"
    COLLISION_RISK = "collision_risk"
    JOINT_FAILURE = "joint_failure"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


ROBOT_TYPES = tuple(item.value for item in RobotType)
FAULT_TYPES = tuple(item.value for item in FaultType)

SCENARIOS = FAULT_TYPES + ("all",)


# ============================================================
# Helpers
# ============================================================

def clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    return max(low, min(high, value))


def deterministic_id(
    *parts: object,
) -> str:
    return str(
        uuid.uuid5(
            EVENT_NAMESPACE,
            ":".join(str(part) for part in parts),
        )
    )


def round_dict(
    data: dict[str, Any],
    digits: int = 4,
) -> dict[str, Any]:
    return {
        key: (
            round(value, digits)
            if isinstance(value, float)
            else value
        )
        for key, value in data.items()
    }


# ============================================================
# Configuration
# ============================================================

@dataclass
class SafetyThresholds:
    motor_warning_temp_c: float = 75.0
    motor_critical_temp_c: float = 95.0

    battery_warning_temp_c: float = 48.0
    battery_critical_temp_c: float = 65.0

    battery_warning_soc_percent: float = 15.0
    battery_critical_soc_percent: float = 5.0

    vibration_warning_mm_s: float = 6.0
    vibration_critical_mm_s: float = 12.0

    sensor_warning_health: float = 0.5
    sensor_critical_health: float = 0.2

    communication_warning_ms: float = 500.0
    communication_critical_ms: float = 2000.0

    human_warning_distance_m: float = 2.0
    human_critical_distance_m: float = 0.7

    obstacle_warning_distance_m: float = 1.5
    obstacle_critical_distance_m: float = 0.35

    joint_warning_error_deg: float = 5.0
    joint_critical_error_deg: float = 15.0


@dataclass
class SimulationConfig:
    robots: int = 5
    steps: int = 200
    delta_time_s: float = 0.1

    seed: int = 42
    scenario: str = "all"

    fault_probability: float = 0.01

    world_width_m: float = 30.0
    world_height_m: float = 30.0

    joints_per_robot: int = 6

    thresholds: SafetyThresholds = field(
        default_factory=SafetyThresholds
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

        if not 1 <= self.joints_per_robot <= 64:
            raise ValueError(
                "joints_per_robot must be between 1 and 64"
            )

        if (
            not math.isfinite(self.delta_time_s)
            or self.delta_time_s <= 0
        ):
            raise ValueError(
                "delta_time_s must be positive"
            )

        if not (
            math.isfinite(self.fault_probability)
            and 0 <= self.fault_probability <= 1
        ):
            raise ValueError(
                "fault_probability must be between 0 and 1"
            )

        if (
            self.world_width_m <= 0
            or self.world_height_m <= 0
        ):
            raise ValueError(
                "World dimensions must be positive"
            )

        if self.scenario not in SCENARIOS:
            raise ValueError(
                f"Unsupported scenario: {self.scenario}"
            )


# ============================================================
# Robot Models
# ============================================================

@dataclass
class JointTelemetry:
    joint_id: str

    position_rad: float = 0.0
    target_position_rad: float = 0.0

    velocity_rad_s: float = 0.0
    torque_nm: float = 5.0

    temperature_c: float = 35.0
    position_error_deg: float = 0.0


@dataclass
class RobotTelemetry:
    timestamp_s: float = 0.0

    position_x_m: float = 0.0
    position_y_m: float = 0.0
    heading_rad: float = 0.0

    linear_speed_m_s: float = 0.0
    angular_speed_rad_s: float = 0.0

    battery_soc_percent: float = 85.0
    battery_voltage_v: float = 48.0
    battery_temperature_c: float = 32.0

    motor_temperature_c: float = 40.0
    motor_current_a: float = 8.0
    vibration_mm_s: float = 1.5

    sensor_health_score: float = 1.0
    communication_latency_ms: float = 40.0
    communication_connected: bool = True

    obstacle_distance_m: float = 10.0
    human_distance_m: float = 10.0

    operating_hours: float = 100.0


@dataclass
class SimulatedRobot:
    robot_id: str
    robot_type: RobotType

    telemetry: RobotTelemetry

    joints: list[JointTelemetry] = field(
        default_factory=list
    )

    state: RobotState = RobotState.OPERATING

    health_score: float = 1.0
    failure_probability: float = 0.0
    estimated_rul_hours: float = 10000.0

    simulated_estop_latched: bool = False
    human_review_required: bool = False

    active_faults: set[str] = field(
        default_factory=set
    )


@dataclass
class SimulationEvent:
    event_id: str
    robot_id: str
    timestamp_s: float

    event_type: str
    severity: str
    description: str

    risk_score: float

    simulated_estop_latched: bool
    human_review_required: bool

    recommended_action: str
    simulation_only: bool = True


# ============================================================
# Robot Factory
# ============================================================

class RobotFactory:
    ROBOT_SEQUENCE = (
        RobotType.MOBILE,
        RobotType.ARM,
        RobotType.HUMANOID,
        RobotType.QUADRUPED,
    )

    def __init__(
        self,
        config: SimulationConfig,
        rng: random.Random,
    ) -> None:
        self.config = config
        self.rng = rng

    def create(self) -> list[SimulatedRobot]:
        robots = []

        for index in range(self.config.robots):
            robot_id = f"SIM-ROBOT-{index + 1:03d}"

            robot_type = self.ROBOT_SEQUENCE[
                index % len(self.ROBOT_SEQUENCE)
            ]

            telemetry = RobotTelemetry(
                position_x_m=self.rng.uniform(
                    1, self.config.world_width_m - 1
                ),
                position_y_m=self.rng.uniform(
                    1, self.config.world_height_m - 1
                ),
                heading_rad=self.rng.uniform(
                    -math.pi, math.pi
                ),
                linear_speed_m_s=(
                    0.0
                    if robot_type == RobotType.ARM
                    else self.rng.uniform(0.2, 1.0)
                ),
                battery_soc_percent=self.rng.uniform(
                    65, 100
                ),
                operating_hours=self.rng.uniform(
                    100, 1500
                ),
            )

            joints = [
                JointTelemetry(
                    joint_id=(
                        f"{robot_id}-JOINT-{j + 1:02d}"
                    ),
                    position_rad=self.rng.uniform(
                        -1.0, 1.0
                    ),
                    target_position_rad=0.0,
                )
                for j in range(
                    self.config.joints_per_robot
                )
            ]

            robots.append(
                SimulatedRobot(
                    robot_id=robot_id,
                    robot_type=robot_type,
                    telemetry=telemetry,
                    joints=joints,
                )
            )

        return robots


# ============================================================
# Robot Physics
# ============================================================

class RobotPhysics:
    """
    Simplified kinematic and telemetry model.

    No real robot dynamics or hardware integration.
    """

    def __init__(
        self,
        config: SimulationConfig,
        rng: random.Random,
    ) -> None:
        self.config = config
        self.rng = rng

    def update_motion(
        self,
        robot: SimulatedRobot,
    ) -> None:
        t = robot.telemetry
        dt = self.config.delta_time_s

        if robot.simulated_estop_latched:
            t.linear_speed_m_s = 0.0
            t.angular_speed_rad_s = 0.0
            return

        if robot.robot_type == RobotType.ARM:
            t.linear_speed_m_s = 0.0
            t.angular_speed_rad_s = 0.0
            return

        t.angular_speed_rad_s = self.rng.uniform(
            -0.3, 0.3
        )

        t.heading_rad += (
            t.angular_speed_rad_s * dt
        )

        t.linear_speed_m_s = clamp(
            t.linear_speed_m_s
            + self.rng.uniform(-0.08, 0.08),
            0.0,
            1.5,
        )

        t.position_x_m += (
            math.cos(t.heading_rad)
            * t.linear_speed_m_s
            * dt
        )

        t.position_y_m += (
            math.sin(t.heading_rad)
            * t.linear_speed_m_s
            * dt
        )

        # Reflect heading at simulated boundaries.
        if t.position_x_m < 0:
            t.position_x_m = 0
            t.heading_rad = math.pi - t.heading_rad

        elif t.position_x_m > self.config.world_width_m:
            t.position_x_m = self.config.world_width_m
            t.heading_rad = math.pi - t.heading_rad

        if t.position_y_m < 0:
            t.position_y_m = 0
            t.heading_rad = -t.heading_rad

        elif t.position_y_m > self.config.world_height_m:
            t.position_y_m = self.config.world_height_m
            t.heading_rad = -t.heading_rad

    def update_joints(
        self,
        robot: SimulatedRobot,
    ) -> None:
        dt = self.config.delta_time_s

        for joint in robot.joints:
            if robot.simulated_estop_latched:
                joint.velocity_rad_s = 0.0
                continue

            joint.target_position_rad += (
                self.rng.uniform(-0.5, 0.5) * dt
            )

            position_error = (
                joint.target_position_rad
                - joint.position_rad
            )

            joint.velocity_rad_s = clamp(
                position_error * 2.0,
                -1.0,
                1.0,
            )

            joint.position_rad += (
                joint.velocity_rad_s * dt
            )

            joint.torque_nm = max(
                0.0,
                joint.torque_nm
                + self.rng.uniform(-0.5, 0.5),
            )

            joint.temperature_c = max(
                20.0,
                joint.temperature_c
                + self.rng.uniform(-0.1, 0.2),
            )

            joint.position_error_deg = abs(
                math.degrees(
                    joint.target_position_rad
                    - joint.position_rad
                )
            )

    def update_health_telemetry(
        self,
        robot: SimulatedRobot,
    ) -> None:
        t = robot.telemetry
        dt = self.config.delta_time_s

        if robot.simulated_estop_latched:
            # Synthetic passive cooling only.
            t.motor_temperature_c = max(
                25.0,
                t.motor_temperature_c - 0.1 * dt,
            )
            t.motor_current_a = 0.0
            return

        t.battery_soc_percent = max(
            0.0,
            t.battery_soc_percent
            - 0.003 * dt,
        )

        t.battery_voltage_v = (
            42.0 + 0.08 * t.battery_soc_percent
        )

        t.motor_temperature_c = max(
            20.0,
            t.motor_temperature_c
            + self.rng.uniform(-0.1, 0.2),
        )

        t.battery_temperature_c = max(
            15.0,
            t.battery_temperature_c
            + self.rng.uniform(-0.05, 0.1),
        )

        t.motor_current_a = max(
            0.0,
            t.motor_current_a
            + self.rng.uniform(-0.4, 0.4),
        )

        t.vibration_mm_s = max(
            0.0,
            t.vibration_mm_s
            + self.rng.uniform(-0.1, 0.1),
        )

        t.communication_latency_ms = max(
            1.0,
            t.communication_latency_ms
            + self.rng.uniform(-5.0, 5.0),
        )

        t.operating_hours += dt / 3600.0

    def update_proximity(
        self,
        robot: SimulatedRobot,
    ) -> None:
        t = robot.telemetry

        t.obstacle_distance_m = max(
            0.0,
            t.obstacle_distance_m
            + self.rng.uniform(-0.5, 0.5),
        )

        t.human_distance_m = max(
            0.0,
            t.human_distance_m
            + self.rng.uniform(-0.5, 0.5),
        )

    def update(
        self,
        robot: SimulatedRobot,
        timestamp_s: float,
    ) -> None:
        robot.telemetry.timestamp_s = timestamp_s

        self.update_motion(robot)
        self.update_joints(robot)
        self.update_health_telemetry(robot)
        self.update_proximity(robot)


# ============================================================
# Fault Injection
# ============================================================

class FaultInjector:
    def __init__(
        self,
        config: SimulationConfig,
        rng: random.Random,
    ) -> None:
        self.config = config
        self.rng = rng

    def choose_fault(self) -> str:
        if self.config.scenario != "all":
            return self.config.scenario

        return self.rng.choice([
            item.value
            for item in FaultType
            if item != FaultType.NONE
        ])

    def inject(
        self,
        robot: SimulatedRobot,
        fault: str,
    ) -> None:
        t = robot.telemetry

        if fault == FaultType.MOTOR_FAILURE.value:
            t.vibration_mm_s = self.rng.uniform(
                13, 20
            )
            t.motor_current_a = self.rng.uniform(
                25, 40
            )

        elif fault == FaultType.OVERHEATING.value:
            t.motor_temperature_c = self.rng.uniform(
                96, 115
            )

        elif fault == FaultType.SENSOR_FAILURE.value:
            t.sensor_health_score = self.rng.uniform(
                0.0, 0.15
            )

        elif fault == FaultType.COMMUNICATION_LOSS.value:
            t.communication_connected = False
            t.communication_latency_ms = 3000.0

        elif fault == FaultType.BATTERY_EMERGENCY.value:
            t.battery_temperature_c = self.rng.uniform(
                66, 85
            )

        elif fault == FaultType.COLLISION_RISK.value:
            t.obstacle_distance_m = self.rng.uniform(
                0.05, 0.25
            )

        elif fault == FaultType.JOINT_FAILURE.value:
            if robot.joints:
                joint = self.rng.choice(robot.joints)
                joint.position_error_deg = (
                    self.rng.uniform(16, 30)
                )

        robot.active_faults.add(fault)

    def update(
        self,
        robot: SimulatedRobot,
        step: int,
    ) -> None:
        if robot.simulated_estop_latched:
            return

        should_inject = (
            step == 1
            or self.rng.random()
            < self.config.fault_probability
        )

        if not should_inject:
            return

        fault = self.choose_fault()

        if fault != FaultType.NONE.value:
            self.inject(robot, fault)


# ============================================================
# Safety Monitor
# ============================================================

class RobotSafetyMonitor:
    def __init__(
        self,
        thresholds: SafetyThresholds,
    ) -> None:
        self.t = thresholds

    def evaluate(
        self,
        robot: SimulatedRobot,
    ) -> list[tuple[str, Severity, str]]:
        t = robot.telemetry
        findings = []

        def check_high(
            name: str,
            value: float,
            warning: float,
            critical: float,
        ) -> None:
            if value >= critical:
                findings.append((
                    name,
                    Severity.CRITICAL,
                    f"Critical {name}: {value:.2f}",
                ))
            elif value >= warning:
                findings.append((
                    name,
                    Severity.WARNING,
                    f"Warning {name}: {value:.2f}",
                ))

        check_high(
            "motor_overheating",
            t.motor_temperature_c,
            self.t.motor_warning_temp_c,
            self.t.motor_critical_temp_c,
        )

        check_high(
            "battery_overheating",
            t.battery_temperature_c,
            self.t.battery_warning_temp_c,
            self.t.battery_critical_temp_c,
        )

        check_high(
            "motor_vibration",
            t.vibration_mm_s,
            self.t.vibration_warning_mm_s,
            self.t.vibration_critical_mm_s,
        )

        check_high(
            "communication_latency",
            t.communication_latency_ms,
            self.t.communication_warning_ms,
            self.t.communication_critical_ms,
        )

        if not t.communication_connected:
            findings.append((
                "communication_loss",
                Severity.CRITICAL,
                "Synthetic communication link lost",
            ))

        if t.sensor_health_score <= (
            self.t.sensor_critical_health
        ):
            findings.append((
                "sensor_failure",
                Severity.CRITICAL,
                "Sensor health critically degraded",
            ))

        elif t.sensor_health_score <= (
            self.t.sensor_warning_health
        ):
            findings.append((
                "sensor_failure",
                Severity.WARNING,
                "Sensor health degraded",
            ))

        if t.battery_soc_percent <= (
            self.t.battery_critical_soc_percent
        ):
            findings.append((
                "low_battery",
                Severity.CRITICAL,
                "Battery charge critically low",
            ))

        elif t.battery_soc_percent <= (
            self.t.battery_warning_soc_percent
        ):
            findings.append((
                "low_battery",
                Severity.WARNING,
                "Battery charge below warning threshold",
            ))

        if t.obstacle_distance_m <= (
            self.t.obstacle_critical_distance_m
        ):
            findings.append((
                "collision_risk",
                Severity.CRITICAL,
                "Obstacle critically close",
            ))

        elif t.obstacle_distance_m <= (
            self.t.obstacle_warning_distance_m
        ):
            findings.append((
                "collision_risk",
                Severity.WARNING,
                "Obstacle within warning distance",
            ))

        if t.human_distance_m <= (
            self.t.human_critical_distance_m
        ):
            findings.append((
                "human_proximity",
                Severity.CRITICAL,
                "Human critically close",
            ))

        elif t.human_distance_m <= (
            self.t.human_warning_distance_m
        ):
            findings.append((
                "human_proximity",
                Severity.WARNING,
                "Human within warning distance",
            ))

        for joint in robot.joints:
            if joint.position_error_deg >= (
                self.t.joint_critical_error_deg
            ):
                findings.append((
                    "joint_failure",
                    Severity.CRITICAL,
                    f"{joint.joint_id} tracking error critical",
                ))

            elif joint.position_error_deg >= (
                self.t.joint_warning_error_deg
            ):
                findings.append((
                    "joint_failure",
                    Severity.WARNING,
                    f"{joint.joint_id} tracking error elevated",
                ))

        return findings


# ============================================================
# Robot Health Estimator
# ============================================================

class RobotHealthEstimator:
    """
    Heuristic health and RUL estimator.

    Not a trained or calibrated predictive model.
    """

    def estimate(
        self,
        robot: SimulatedRobot,
    ) -> tuple[float, float, float]:
        t = robot.telemetry

        thermal_risk = clamp(
            (t.motor_temperature_c - 40) / 55
        )

        battery_risk = clamp(
            (t.battery_temperature_c - 35) / 30
        )

        vibration_risk = clamp(
            t.vibration_mm_s / 12.0
        )

        sensor_risk = clamp(
            1.0 - t.sensor_health_score
        )

        communication_risk = clamp(
            t.communication_latency_ms / 2000.0
        )

        joint_risk = max(
            (
                clamp(
                    joint.position_error_deg / 15.0
                )
                for joint in robot.joints
            ),
            default=0.0,
        )

        risk = (
            0.25 * thermal_risk
            + 0.15 * battery_risk
            + 0.15 * vibration_risk
            + 0.15 * sensor_risk
            + 0.15 * communication_risk
            + 0.15 * joint_risk
        )

        health = clamp(1.0 - risk)
        failure_probability = clamp(risk ** 1.4)

        rul = max(
            1.0,
            12000.0 * health ** 2
            - t.operating_hours * 0.05,
        )

        return (
            round(health, 4),
            round(failure_probability, 4),
            round(rul, 2),
        )


# ============================================================
# Simulation Engine
# ============================================================

class RobotSimulationEngine:
    def __init__(
        self,
        config: SimulationConfig,
    ) -> None:
        self.config = config
        self.config.validate()

        self.rng = random.Random(config.seed)

        self.robots = RobotFactory(
            config,
            self.rng,
        ).create()

        self.physics = RobotPhysics(
            config,
            self.rng,
        )

        self.injector = FaultInjector(
            config,
            self.rng,
        )

        self.safety = RobotSafetyMonitor(
            config.thresholds
        )

        self.health = RobotHealthEstimator()

        self.events: list[SimulationEvent] = []
        self.telemetry_rows: list[dict[str, Any]] = []

        self.event_signatures: set[
            tuple[str, str]
        ] = set()

    def create_event(
        self,
        robot: SimulatedRobot,
        event_type: str,
        severity: Severity,
        description: str,
        timestamp_s: float,
    ) -> None:
        signature = (
            robot.robot_id,
            event_type,
        )

        if signature in self.event_signatures:
            return

        self.event_signatures.add(signature)

        critical = severity == Severity.CRITICAL

        if critical:
            robot.simulated_estop_latched = True
            robot.human_review_required = True
            robot.state = RobotState.EMERGENCY_STOPPED

            robot.telemetry.linear_speed_m_s = 0.0
            robot.telemetry.angular_speed_rad_s = 0.0

            for joint in robot.joints:
                joint.velocity_rad_s = 0.0

        elif not robot.simulated_estop_latched:
            robot.state = RobotState.DEGRADED

        recommendation = (
            "SIMULATION: inhibit motion and request "
            "qualified human safety review"
            if critical
            else
            "SIMULATION: review diagnostics and "
            "monitor robot condition"
        )

        self.events.append(
            SimulationEvent(
                event_id=deterministic_id(
                    robot.robot_id,
                    event_type,
                    timestamp_s,
                ),
                robot_id=robot.robot_id,
                timestamp_s=round(timestamp_s, 4),
                event_type=event_type,
                severity=severity.value,
                description=description,
                risk_score=(
                    1.0
                    if critical
                    else robot.failure_probability
                ),
                simulated_estop_latched=(
                    robot.simulated_estop_latched
                ),
                human_review_required=True,
                recommended_action=recommendation,
            )
        )

    def record_telemetry(
        self,
        robot: SimulatedRobot,
    ) -> None:
        t = robot.telemetry

        self.telemetry_rows.append(
            round_dict({
                "timestamp_s": t.timestamp_s,
                "robot_id": robot.robot_id,
                "robot_type": robot.robot_type.value,
                "state": robot.state.value,
                **asdict(t),
                "health_score": robot.health_score,
                "failure_probability": (
                    robot.failure_probability
                ),
                "remaining_useful_life_hours": (
                    robot.estimated_rul_hours
                ),
                "simulated_estop_latched": (
                    robot.simulated_estop_latched
                ),
                "human_review_required": (
                    robot.human_review_required
                ),
                "active_faults": ",".join(
                    sorted(robot.active_faults)
                ),
                "simulation_only": True,
            })
        )

    def run_step(
        self,
        step: int,
    ) -> None:
        timestamp_s = (
            step * self.config.delta_time_s
        )

        for robot in self.robots:
            self.physics.update(
                robot,
                timestamp_s,
            )

            self.injector.update(
                robot,
                step,
            )

            (
                robot.health_score,
                robot.failure_probability,
                robot.estimated_rul_hours,
            ) = self.health.estimate(robot)

            findings = self.safety.evaluate(
                robot
            )

            for (
                event_type,
                severity,
                description,
            ) in findings:
                self.create_event(
                    robot,
                    event_type,
                    severity,
                    description,
                    timestamp_s,
                )

            self.record_telemetry(robot)

    def run(self) -> dict[str, Any]:
        LOG.info(
            "Starting offline robot fleet simulation"
        )

        for step in range(self.config.steps):
            self.run_step(step)

        return self.build_report()

    def build_report(self) -> dict[str, Any]:
        severity_counts = Counter(
            event.severity
            for event in self.events
        )

        event_counts = Counter(
            event.event_type
            for event in self.events
        )

        stopped = [
            robot.robot_id
            for robot in self.robots
            if robot.simulated_estop_latched
        ]

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
                "fault_probability": (
                    self.config.fault_probability
                ),
                "joints_per_robot": (
                    self.config.joints_per_robot
                ),
            },
            "summary": {
                "total_robots": len(self.robots),
                "telemetry_records": (
                    len(self.telemetry_rows)
                ),
                "total_events": len(self.events),
                "severity_counts": dict(
                    severity_counts
                ),
                "event_counts": dict(event_counts),
                "simulated_estop_count": len(stopped),
                "mean_robot_health": round(
                    sum(
                        robot.health_score
                        for robot in self.robots
                    ) / len(self.robots),
                    4,
                ),
            },
            "simulated_estop_robots": stopped,
            "robots": [
                {
                    "robot_id": robot.robot_id,
                    "robot_type": robot.robot_type.value,
                    "state": robot.state.value,
                    "health_score": robot.health_score,
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
                    "joints": [
                        asdict(joint)
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
                "Synthetic robot telemetry only",
                "Simplified 2D kinematics",
                "No articulated rigid-body dynamics",
                "No real obstacle geometry or collision physics",
                "Heuristic health and RUL estimates",
                "No physical actuation",
                "Not a safety-certified robotics controller",
            ],
        }


# ============================================================
# File Exports
# ============================================================

def export_json(
    path: str | Path,
    data: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
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
# Command-Line Interface
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline Physical AI robot fleet simulator"
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
        "--fault-probability",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--joints-per-robot",
        type=int,
        default=6,
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
        fault_probability=args.fault_probability,
        joints_per_robot=args.joints_per_robot,
    )

    config.validate()

    simulation = RobotSimulationEngine(config)
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
            [
                asdict(event)
                for event in simulation.events
            ],
            columns=list(
                SimulationEvent.__dataclass_fields__
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

    LOG.info("Robot simulation completed")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

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

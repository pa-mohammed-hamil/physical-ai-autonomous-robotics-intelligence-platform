
#!/usr/bin/env python3
"""
Physical AI & Autonomous Robotics Intelligence Platform
File: scripts/simulate_emergency.py

Offline robotics emergency simulation engine.

Features:
    - Motor failure simulation
    - Battery overheating simulation
    - Thermal runaway warning simulation
    - Sensor failure simulation
    - Communication loss simulation
    - Collision-risk simulation
    - Human proximity emergency simulation
    - Power failure simulation
    - Actuator malfunction simulation
    - Emergency detection and risk scoring
    - Simulated emergency-stop state
    - Incident escalation
    - Human approval workflow
    - Recovery verification
    - Multi-robot scenarios
    - JSON scenario loading
    - JSON and CSV reports
    - Deterministic random simulation

Dependencies:
    Python 3.10+
    Standard library only

Examples:
    python scripts/simulate_emergency.py

    python scripts/simulate_emergency.py --robots 5

    python scripts/simulate_emergency.py \
        --scenario motor_failure

    python scripts/simulate_emergency.py \
        --scenario overheating \
        --steps 100

    python scripts/simulate_emergency.py \
        --scenario all \
        --robots 10 \
        --steps 200

    python scripts/simulate_emergency.py \
        --config simulation/scenarios/emergency.json

    python scripts/simulate_emergency.py \
        --output reports/emergency.json \
        --csv reports/emergency_events.csv

Safety:
    Simulation only.
    No real robot commands.
    No real emergency dispatch.
    No physical emergency-stop reset.
    No automatic physical restart.
    No certified safety assessment.
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
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("simulate_emergency")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

SCHEMA_VERSION = "1.0.0"

SCENARIOS = (
    "normal",
    "motor_failure",
    "overheating",
    "battery_emergency",
    "sensor_failure",
    "communication_loss",
    "collision",
    "human_proximity",
    "power_failure",
    "actuator_failure",
    "all",
)


# ============================================================
# Enumerations
# ============================================================

class EmergencyType(str, Enum):
    MOTOR_FAILURE = "motor_failure"
    OVERHEATING = "overheating"
    BATTERY_EMERGENCY = "battery_emergency"
    SENSOR_FAILURE = "sensor_failure"
    COMMUNICATION_LOSS = "communication_loss"
    COLLISION = "collision"
    HUMAN_PROXIMITY = "human_proximity"
    POWER_FAILURE = "power_failure"
    ACTUATOR_FAILURE = "actuator_failure"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class RobotState(str, Enum):
    IDLE = "idle"
    OPERATING = "operating"
    DEGRADED = "degraded"
    EMERGENCY_STOPPED = "emergency_stopped"
    AWAITING_REVIEW = "awaiting_review"
    RECOVERY_PENDING = "recovery_pending"


class IncidentStatus(str, Enum):
    OPEN = "open"
    ESCALATED = "escalated"
    AWAITING_APPROVAL = "awaiting_approval"
    REVIEWED = "reviewed"
    CLOSED = "closed"


# ============================================================
# Configuration
# ============================================================

@dataclass
class EmergencyThresholds:
    motor_warning_temp_c: float = 75.0
    motor_critical_temp_c: float = 95.0

    battery_warning_temp_c: float = 50.0
    battery_critical_temp_c: float = 65.0

    battery_low_soc_percent: float = 15.0
    battery_critical_soc_percent: float = 5.0

    vibration_warning_mm_s: float = 6.0
    vibration_critical_mm_s: float = 12.0

    communication_warning_ms: float = 500.0
    communication_timeout_ms: float = 2000.0

    human_warning_distance_m: float = 2.0
    human_critical_distance_m: float = 0.7

    collision_warning_ttc_s: float = 3.0
    collision_critical_ttc_s: float = 1.0

    actuator_error_warning_deg: float = 5.0
    actuator_error_critical_deg: float = 15.0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"Invalid threshold {name}: {value}"
                )


@dataclass
class SimulationConfig:
    robots: int = 5
    steps: int = 100
    delta_time_s: float = 0.1
    seed: int = 42

    scenario: str = "all"

    event_probability: float = 0.025

    thresholds: EmergencyThresholds = field(
        default_factory=EmergencyThresholds
    )

    def validate(self) -> None:
        if not 1 <= self.robots <= 1000:
            raise ValueError(
                "robots must be between 1 and 1000"
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

        if (
            not math.isfinite(self.event_probability)
            or not 0 <= self.event_probability <= 1
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
# Robot Telemetry
# ============================================================

@dataclass
class RobotTelemetry:
    motor_temperature_c: float = 42.0
    battery_temperature_c: float = 32.0

    battery_soc_percent: float = 85.0

    vibration_mm_s: float = 1.5
    motor_current_a: float = 8.0

    sensor_health: float = 1.0
    actuator_position_error_deg: float = 0.5

    communication_latency_ms: float = 40.0
    communication_connected: bool = True

    human_distance_m: float = 10.0
    collision_ttc_s: float = 999.0

    collision_detected: bool = False
    power_available: bool = True

    def validate(self) -> None:
        for key, value in asdict(self).items():
            if isinstance(value, bool):
                continue

            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite telemetry: {key}"
                )


@dataclass
class SimulatedRobot:
    robot_id: str
    robot_type: str

    state: RobotState = RobotState.OPERATING

    telemetry: RobotTelemetry = field(
        default_factory=RobotTelemetry
    )

    simulated_estop_latched: bool = False
    human_approval_required: bool = False

    active_incidents: list[str] = field(
        default_factory=list
    )

    def latch_simulated_estop(self) -> None:
        self.simulated_estop_latched = True
        self.human_approval_required = True
        self.state = RobotState.EMERGENCY_STOPPED


# ============================================================
# Emergency Incident
# ============================================================

@dataclass
class EmergencyIncident:
    incident_id: str
    robot_id: str

    emergency_type: str
    severity: str

    timestamp_s: float

    title: str
    description: str

    risk_score: float

    status: str = IncidentStatus.OPEN.value

    simulated_estop_required: bool = False
    human_approval_required: bool = True

    response_actions: list[str] = field(
        default_factory=list
    )

    verification_checks: list[str] = field(
        default_factory=list
    )

    simulation_only: bool = True


# ============================================================
# Emergency Detection
# ============================================================

class EmergencyDetector:
    def __init__(
        self,
        thresholds: EmergencyThresholds,
    ) -> None:
        self.t = thresholds

    def detect(
        self,
        telemetry: RobotTelemetry,
    ) -> list[tuple[EmergencyType, Severity, str]]:
        findings = []

        if telemetry.motor_temperature_c >= (
            self.t.motor_critical_temp_c
        ):
            findings.append((
                EmergencyType.OVERHEATING,
                Severity.CRITICAL,
                "Motor temperature exceeds critical threshold",
            ))

        elif telemetry.motor_temperature_c >= (
            self.t.motor_warning_temp_c
        ):
            findings.append((
                EmergencyType.OVERHEATING,
                Severity.WARNING,
                "Motor temperature exceeds warning threshold",
            ))

        if telemetry.battery_temperature_c >= (
            self.t.battery_critical_temp_c
        ):
            findings.append((
                EmergencyType.BATTERY_EMERGENCY,
                Severity.CRITICAL,
                "Battery temperature exceeds critical threshold",
            ))

        elif telemetry.battery_temperature_c >= (
            self.t.battery_warning_temp_c
        ):
            findings.append((
                EmergencyType.BATTERY_EMERGENCY,
                Severity.WARNING,
                "Battery temperature exceeds warning threshold",
            ))

        if telemetry.battery_soc_percent <= (
            self.t.battery_critical_soc_percent
        ):
            findings.append((
                EmergencyType.BATTERY_EMERGENCY,
                Severity.CRITICAL,
                "Battery state of charge is critically low",
            ))

        elif telemetry.battery_soc_percent <= (
            self.t.battery_low_soc_percent
        ):
            findings.append((
                EmergencyType.BATTERY_EMERGENCY,
                Severity.WARNING,
                "Battery state of charge is low",
            ))

        if telemetry.vibration_mm_s >= (
            self.t.vibration_critical_mm_s
        ):
            findings.append((
                EmergencyType.MOTOR_FAILURE,
                Severity.CRITICAL,
                "Critical motor vibration detected",
            ))

        elif telemetry.vibration_mm_s >= (
            self.t.vibration_warning_mm_s
        ):
            findings.append((
                EmergencyType.MOTOR_FAILURE,
                Severity.WARNING,
                "Abnormal motor vibration detected",
            ))

        if telemetry.sensor_health <= 0.2:
            findings.append((
                EmergencyType.SENSOR_FAILURE,
                Severity.CRITICAL,
                "Sensor health critically degraded",
            ))

        elif telemetry.sensor_health <= 0.5:
            findings.append((
                EmergencyType.SENSOR_FAILURE,
                Severity.WARNING,
                "Sensor health degraded",
            ))

        if (
            not telemetry.communication_connected
            or telemetry.communication_latency_ms >=
            self.t.communication_timeout_ms
        ):
            findings.append((
                EmergencyType.COMMUNICATION_LOSS,
                Severity.CRITICAL,
                "Robot communication timeout or disconnection",
            ))

        elif telemetry.communication_latency_ms >= (
            self.t.communication_warning_ms
        ):
            findings.append((
                EmergencyType.COMMUNICATION_LOSS,
                Severity.WARNING,
                "Communication latency elevated",
            ))

        if telemetry.collision_detected:
            findings.append((
                EmergencyType.COLLISION,
                Severity.CRITICAL,
                "Synthetic collision event detected",
            ))

        elif telemetry.collision_ttc_s <= (
            self.t.collision_critical_ttc_s
        ):
            findings.append((
                EmergencyType.COLLISION,
                Severity.CRITICAL,
                "Predicted collision time critically low",
            ))

        elif telemetry.collision_ttc_s <= (
            self.t.collision_warning_ttc_s
        ):
            findings.append((
                EmergencyType.COLLISION,
                Severity.WARNING,
                "Predicted collision time below warning threshold",
            ))

        if telemetry.human_distance_m <= (
            self.t.human_critical_distance_m
        ):
            findings.append((
                EmergencyType.HUMAN_PROXIMITY,
                Severity.CRITICAL,
                "Human proximity below critical separation threshold",
            ))

        elif telemetry.human_distance_m <= (
            self.t.human_warning_distance_m
        ):
            findings.append((
                EmergencyType.HUMAN_PROXIMITY,
                Severity.WARNING,
                "Human detected within warning separation distance",
            ))

        if not telemetry.power_available:
            findings.append((
                EmergencyType.POWER_FAILURE,
                Severity.CRITICAL,
                "Synthetic power supply failure",
            ))

        if telemetry.actuator_position_error_deg >= (
            self.t.actuator_error_critical_deg
        ):
            findings.append((
                EmergencyType.ACTUATOR_FAILURE,
                Severity.CRITICAL,
                "Critical actuator position tracking error",
            ))

        elif telemetry.actuator_position_error_deg >= (
            self.t.actuator_error_warning_deg
        ):
            findings.append((
                EmergencyType.ACTUATOR_FAILURE,
                Severity.WARNING,
                "Actuator position tracking error elevated",
            ))

        return findings


# ============================================================
# Risk Assessment
# ============================================================

class EmergencyRiskAssessor:
    BASE_RISK = {
        EmergencyType.MOTOR_FAILURE: 0.75,
        EmergencyType.OVERHEATING: 0.75,
        EmergencyType.BATTERY_EMERGENCY: 0.85,
        EmergencyType.SENSOR_FAILURE: 0.65,
        EmergencyType.COMMUNICATION_LOSS: 0.8,
        EmergencyType.COLLISION: 0.95,
        EmergencyType.HUMAN_PROXIMITY: 0.95,
        EmergencyType.POWER_FAILURE: 0.85,
        EmergencyType.ACTUATOR_FAILURE: 0.8,
    }

    def calculate(
        self,
        emergency_type: EmergencyType,
        severity: Severity,
    ) -> float:
        base = self.BASE_RISK[emergency_type]

        multiplier = (
            1.0
            if severity == Severity.CRITICAL
            else 0.6
        )

        return round(
            min(1.0, base * multiplier),
            4,
        )


# ============================================================
# Emergency Response Coordinator
# ============================================================

class EmergencyResponseCoordinator:
    """
    Creates simulated response recommendations.

    Never communicates with physical controllers,
    safety PLCs, dispatch systems or emergency services.
    """

    RESPONSE_ACTIONS = {
        EmergencyType.MOTOR_FAILURE: [
            "Flag motor subsystem as unsafe in simulation",
            "Recommend simulated motion inhibition",
            "Review vibration and current trends",
            "Request maintenance engineer inspection",
        ],
        EmergencyType.OVERHEATING: [
            "Flag thermal hazard in simulation",
            "Recommend simulated motion inhibition",
            "Inspect synthetic temperature history",
            "Request thermal safety review",
        ],
        EmergencyType.BATTERY_EMERGENCY: [
            "Flag battery hazard in simulation",
            "Recommend simulated energy isolation",
            "Review battery health and temperature",
            "Request qualified technician assessment",
        ],
        EmergencyType.SENSOR_FAILURE: [
            "Flag degraded perception capability",
            "Recommend simulated motion inhibition",
            "Review sensor diagnostics",
            "Request sensor calibration review",
        ],
        EmergencyType.COMMUNICATION_LOSS: [
            "Flag lost communications",
            "Recommend simulated safe-state transition",
            "Inspect synthetic network diagnostics",
            "Request operator acknowledgment",
        ],
        EmergencyType.COLLISION: [
            "Flag collision emergency",
            "Recommend simulated emergency-stop state",
            "Review collision telemetry",
            "Request human safety review",
        ],
        EmergencyType.HUMAN_PROXIMITY: [
            "Flag human proximity hazard",
            "Recommend simulated emergency-stop state",
            "Review separation-distance telemetry",
            "Request safety officer review",
        ],
        EmergencyType.POWER_FAILURE: [
            "Flag power system failure",
            "Recommend simulated safe-state transition",
            "Inspect synthetic power diagnostics",
            "Request electrical safety review",
        ],
        EmergencyType.ACTUATOR_FAILURE: [
            "Flag actuator malfunction",
            "Recommend simulated motion inhibition",
            "Inspect position tracking error",
            "Request mechanical inspection",
        ],
    }

    def build_response(
        self,
        emergency_type: EmergencyType,
        severity: Severity,
    ) -> tuple[list[str], list[str], bool]:
        actions = list(
            self.RESPONSE_ACTIONS[emergency_type]
        )

        checks = [
            "Confirm simulated hazard is no longer present",
            "Verify sensor and telemetry integrity",
            "Review incident evidence",
            "Obtain human approval before closing incident",
            "Do not interpret approval as hardware restart authorization",
        ]

        simulated_estop_required = (
            severity == Severity.CRITICAL
        )

        return (
            actions,
            checks,
            simulated_estop_required,
        )


# ============================================================
# Incident Manager
# ============================================================

class IncidentManager:
    def __init__(self) -> None:
        self.incidents: list[EmergencyIncident] = []
        self.active_signatures: set[tuple[str, str, str]] = set()

        self.assessor = EmergencyRiskAssessor()
        self.coordinator = EmergencyResponseCoordinator()

    def create_incident(
        self,
        robot: SimulatedRobot,
        emergency_type: EmergencyType,
        severity: Severity,
        description: str,
        timestamp_s: float,
    ) -> EmergencyIncident | None:
        signature = (
            robot.robot_id,
            emergency_type.value,
            severity.value,
        )

        if signature in self.active_signatures:
            return None

        self.active_signatures.add(signature)

        risk_score = self.assessor.calculate(
            emergency_type,
            severity,
        )

        actions, checks, estop_required = (
            self.coordinator.build_response(
                emergency_type,
                severity,
            )
        )

        incident = EmergencyIncident(
            incident_id=str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"demo:{robot.robot_id}:"
                f"{emergency_type.value}:"
                f"{severity.value}:"
                f"{timestamp_s:.6f}",
            )),
            robot_id=robot.robot_id,
            emergency_type=emergency_type.value,
            severity=severity.value,
            timestamp_s=round(timestamp_s, 4),
            title=(
                f"{emergency_type.value.replace('_', ' ').title()}"
                f" - {severity.value.upper()}"
            ),
            description=description,
            risk_score=risk_score,
            status=(
                IncidentStatus.ESCALATED.value
                if severity == Severity.CRITICAL
                else IncidentStatus.OPEN.value
            ),
            simulated_estop_required=estop_required,
            human_approval_required=True,
            response_actions=actions,
            verification_checks=checks,
            simulation_only=True,
        )

        self.incidents.append(incident)

        robot.active_incidents.append(
            incident.incident_id
        )

        if estop_required:
            robot.latch_simulated_estop()

        elif not robot.simulated_estop_latched:
            robot.state = RobotState.DEGRADED

        return incident

    def reconcile(
        self,
        active_findings: set[tuple[str, str, str]],
    ) -> None:
        # Signatures may be emitted again after a
        # synthetic condition clears and later recurs.
        self.active_signatures.intersection_update(
            active_findings
        )

    def request_review(
        self,
        incident_id: str,
    ) -> bool:
        for incident in self.incidents:
            if incident.incident_id == incident_id:
                incident.status = (
                    IncidentStatus.AWAITING_APPROVAL.value
                )
                return True

        return False

    def record_human_review(
        self,
        incident_id: str,
        approved: bool,
    ) -> bool:
        """
        Simulated review only.

        Approval does not reset a simulated E-stop,
        close an incident, or authorize physical motion.
        """
        for incident in self.incidents:
            if incident.incident_id == incident_id:
                incident.status = (
                    IncidentStatus.REVIEWED.value
                    if approved
                    else IncidentStatus.ESCALATED.value
                )
                return True

        return False


# ============================================================
# Telemetry Simulator
# ============================================================

class EmergencyTelemetrySimulator:
    ROBOT_TYPES = [
        "mobile_robot",
        "robotic_arm",
        "humanoid",
        "quadruped",
    ]

    def __init__(
        self,
        config: SimulationConfig,
    ) -> None:
        self.config = config
        self.rng = random.Random(
            config.seed
        )

    def generate_robots(
        self,
    ) -> list[SimulatedRobot]:
        return [
            SimulatedRobot(
                robot_id=f"SIM-ROBOT-{index + 1:03d}",
                robot_type=self.ROBOT_TYPES[
                    index % len(self.ROBOT_TYPES)
                ],
            )
            for index in range(self.config.robots)
        ]

    def baseline_update(
        self,
        telemetry: RobotTelemetry,
    ) -> None:
        telemetry.motor_temperature_c = max(
            20.0,
            telemetry.motor_temperature_c
            + self.rng.uniform(-1.0, 1.0),
        )

        telemetry.battery_temperature_c = max(
            15.0,
            telemetry.battery_temperature_c
            + self.rng.uniform(-0.5, 0.5),
        )

        telemetry.battery_soc_percent = max(
            0.0,
            telemetry.battery_soc_percent
            - self.rng.uniform(0.0, 0.025),
        )

        telemetry.vibration_mm_s = max(
            0.0,
            telemetry.vibration_mm_s
            + self.rng.uniform(-0.2, 0.2),
        )

        telemetry.motor_current_a = max(
            0.0,
            telemetry.motor_current_a
            + self.rng.uniform(-0.5, 0.5),
        )

        telemetry.communication_latency_ms = max(
            1.0,
            telemetry.communication_latency_ms
            + self.rng.uniform(-5.0, 5.0),
        )

    def select_scenario(
        self,
    ) -> str:
        if self.config.scenario != "all":
            return self.config.scenario

        return self.rng.choice(
            SCENARIOS[1:-1]
        )

    def inject_emergency(
        self,
        telemetry: RobotTelemetry,
        scenario: str,
    ) -> None:
        if scenario == "motor_failure":
            telemetry.vibration_mm_s = (
                self.rng.uniform(13, 20)
            )
            telemetry.motor_current_a = (
                self.rng.uniform(25, 45)
            )

        elif scenario == "overheating":
            telemetry.motor_temperature_c = (
                self.rng.uniform(96, 120)
            )

        elif scenario == "battery_emergency":
            telemetry.battery_temperature_c = (
                self.rng.uniform(66, 85)
            )

        elif scenario == "sensor_failure":
            telemetry.sensor_health = (
                self.rng.uniform(0.0, 0.15)
            )

        elif scenario == "communication_loss":
            telemetry.communication_connected = False
            telemetry.communication_latency_ms = 3000.0

        elif scenario == "collision":
            telemetry.collision_detected = True
            telemetry.collision_ttc_s = 0.0

        elif scenario == "human_proximity":
            telemetry.human_distance_m = (
                self.rng.uniform(0.1, 0.5)
            )

        elif scenario == "power_failure":
            telemetry.power_available = False

        elif scenario == "actuator_failure":
            telemetry.actuator_position_error_deg = (
                self.rng.uniform(16, 30)
            )

    def update_robot(
        self,
        robot: SimulatedRobot,
        step: int,
    ) -> None:
        if robot.simulated_estop_latched:
            # The simulated E-stop remains latched.
            # No automatic restart or reset.
            return

        self.baseline_update(
            robot.telemetry
        )

        # Inject a requested scenario early enough to
        # make single-scenario runs reproducible.
        should_inject = (
            step == 1
            or self.rng.random()
            < self.config.event_probability
        )

        if should_inject:
            scenario = self.select_scenario()

            if scenario != "normal":
                self.inject_emergency(
                    robot.telemetry,
                    scenario,
                )

        robot.telemetry.validate()


# ============================================================
# Emergency Simulation Engine
# ============================================================

class EmergencySimulationEngine:
    def __init__(
        self,
        config: SimulationConfig,
    ) -> None:
        self.config = config
        self.config.validate()

        self.telemetry_simulator = (
            EmergencyTelemetrySimulator(config)
        )

        self.detector = EmergencyDetector(
            config.thresholds
        )

        self.incident_manager = IncidentManager()

        self.robots = (
            self.telemetry_simulator.generate_robots()
        )

        self.events_by_step: list[dict[str, Any]] = []

    def run_step(
        self,
        step: int,
    ) -> None:
        timestamp_s = (
            step * self.config.delta_time_s
        )

        active_findings = set()

        for robot in self.robots:
            self.telemetry_simulator.update_robot(
                robot,
                step,
            )

            findings = self.detector.detect(
                robot.telemetry
            )

            for (
                emergency_type,
                severity,
                description,
            ) in findings:
                active_findings.add((
                    robot.robot_id,
                    emergency_type.value,
                    severity.value,
                ))

                incident = (
                    self.incident_manager.create_incident(
                        robot,
                        emergency_type,
                        severity,
                        description,
                        timestamp_s,
                    )
                )

                if incident is None:
                    continue

                self.events_by_step.append({
                    "step": step,
                    "timestamp_s": timestamp_s,
                    "robot_id": robot.robot_id,
                    "incident_id": incident.incident_id,
                    "emergency_type": (
                        emergency_type.value
                    ),
                    "severity": severity.value,
                    "risk_score": incident.risk_score,
                    "simulated_estop": (
                        robot.simulated_estop_latched
                    ),
                })

        self.incident_manager.reconcile(
            active_findings
        )

    def run(self) -> dict[str, Any]:
        LOGGER.info(
            "Starting offline emergency simulation"
        )

        for step in range(self.config.steps):
            self.run_step(step)

        return self.build_report()

    def build_report(self) -> dict[str, Any]:
        incidents = self.incident_manager.incidents

        severity_counts = Counter(
            incident.severity
            for incident in incidents
        )

        type_counts = Counter(
            incident.emergency_type
            for incident in incidents
        )

        stopped_robots = [
            robot.robot_id
            for robot in self.robots
            if robot.simulated_estop_latched
        ]

        return {
            "status": "completed",
            "schema_version": SCHEMA_VERSION,
            "simulation_only": True,
            "physical_control_executed": False,
            "real_emergency_services_contacted": False,
            "physical_estop_reset_executed": False,
            "safety_certified": False,
            "configuration": {
                "robots": self.config.robots,
                "steps": self.config.steps,
                "delta_time_s": self.config.delta_time_s,
                "scenario": self.config.scenario,
                "seed": self.config.seed,
                "event_probability": (
                    self.config.event_probability
                ),
            },
            "summary": {
                "total_incidents": len(incidents),
                "severity_counts": dict(
                    severity_counts
                ),
                "emergency_type_counts": dict(
                    type_counts
                ),
                "simulated_estop_count": len(
                    stopped_robots
                ),
                "robots_requiring_human_review": sum(
                    robot.human_approval_required
                    for robot in self.robots
                ),
            },
            "simulated_estop_robots": stopped_robots,
            "robots": [
                {
                    "robot_id": robot.robot_id,
                    "robot_type": robot.robot_type,
                    "state": robot.state.value,
                    "simulated_estop_latched": (
                        robot.simulated_estop_latched
                    ),
                    "human_approval_required": (
                        robot.human_approval_required
                    ),
                    "active_incidents": (
                        robot.active_incidents
                    ),
                    "telemetry": asdict(
                        robot.telemetry
                    ),
                }
                for robot in self.robots
            ],
            "incidents": [
                asdict(incident)
                for incident in incidents
            ],
            "events": self.events_by_step,
            "limitations": [
                "Emergency conditions are synthetically generated.",
                "Thresholds are illustrative, not certified.",
                "No physical robot or safety controller is connected.",
                "No real emergency-stop command is issued.",
                "No automatic physical recovery is supported.",
                "Human review records do not authorize hardware motion.",
            ],
        }


# ============================================================
# Scenario Configuration Loading
# ============================================================

def load_config(
    path: str | Path,
) -> SimulationConfig:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration not found: {path}"
        )

    if path.suffix.lower() == ".json":
        data = json.loads(
            path.read_text(encoding="utf-8")
        )

    elif path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required for YAML configurations"
            ) from exc

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            data = yaml.safe_load(handle)

    else:
        raise ValueError(
            "Configuration must be JSON or YAML"
        )

    if not isinstance(data, dict):
        raise ValueError(
            "Configuration root must be an object"
        )

    simulation_data = data.get(
        "simulation",
        data,
    )

    thresholds_data = data.get(
        "thresholds",
        {},
    )

    if not isinstance(simulation_data, dict):
        raise ValueError(
            "simulation must be an object"
        )

    if not isinstance(thresholds_data, dict):
        raise ValueError(
            "thresholds must be an object"
        )

    simulation_fields = set(
        SimulationConfig.__dataclass_fields__
    )

    threshold_fields = set(
        EmergencyThresholds.__dataclass_fields__
    )

    config = SimulationConfig(
        **{
            key: value
            for key, value in simulation_data.items()
            if key in simulation_fields
            and key != "thresholds"
        },
        thresholds=EmergencyThresholds(
            **{
                key: value
                for key, value in thresholds_data.items()
                if key in threshold_fields
            }
        ),
    )

    config.validate()
    return config


# ============================================================
# Export
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
    events: list[dict[str, Any]],
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "step",
        "timestamp_s",
        "robot_id",
        "incident_id",
        "emergency_type",
        "severity",
        "risk_score",
        "simulated_estop",
    ]

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
        writer.writerows(events)


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate robot emergency scenarios offline"
        )
    )

    parser.add_argument(
        "--config",
        default=None,
    )

    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default=None,
    )

    parser.add_argument(
        "--robots",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--event-probability",
        type=float,
        default=None,
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

    if args.config:
        config = load_config(args.config)
    else:
        config = SimulationConfig()

    overrides = {
        "scenario": args.scenario,
        "robots": args.robots,
        "steps": args.steps,
        "delta_time_s": args.dt,
        "seed": args.seed,
        "event_probability": args.event_probability,
    }

    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)

    config.validate()

    engine = EmergencySimulationEngine(
        config
    )

    report = engine.run()

    if args.output:
        save_json(
            args.output,
            report,
        )

    if args.csv:
        save_csv(
            args.csv,
            engine.events_by_step,
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
        "Emergency simulation completed"
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())

    except (
        ValueError,
        RuntimeError,
        FileNotFoundError,
        OSError,
        TypeError,
        KeyError,
    ) as exc:
        LOGGER.error(
            "Emergency simulation failed: %s",
            exc,
        )

        sys.exit(1)

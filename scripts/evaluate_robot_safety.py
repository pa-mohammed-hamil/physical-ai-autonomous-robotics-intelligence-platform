#!/usr/bin/env python3
"""
Physical AI & Autonomous Robotics Intelligence Platform
File: scripts/evaluate_robot_safety.py

Offline Robot Safety Evaluation Framework

Features:
- Robot joint position and velocity limit evaluation
- Joint torque and actuator temperature monitoring
- Battery voltage, temperature and state-of-charge checks
- Robot speed and acceleration safety checks
- Human proximity and protective separation checks
- Collision, obstacle and workspace violation detection
- Emergency-stop and safety-interlock verification
- Sensor health and communication freshness checks
- Autonomous navigation and VLA action safety evaluation
- Robot fault and predictive maintenance risk assessment
- Safety severity classification
- Critical violation detection
- Human-review recommendations
- Safety score calculation
- Simulation scenario evaluation
- Configurable thresholds
- JSON and CSV report generation
- Batch evaluation from telemetry datasets
- Model-independent safety evaluation

Dependencies:
    pip install numpy pandas

Examples:
    python scripts/evaluate_robot_safety.py \
        --data data/evaluation/robot_safety.csv \
        --output reports/safety

    python scripts/evaluate_robot_safety.py \
        --data data/evaluation/robot_safety.json \
        --config robot_configs/safety_limits.yaml

    python scripts/evaluate_robot_safety.py \
        --demo \
        --output reports/safety

    python scripts/evaluate_robot_safety.py \
        --data data/evaluation/robot_safety.csv \
        --strict

Important:
    This module evaluates recorded telemetry and simulated
    trajectories only.

    It is NOT a certified functional-safety controller,
    collision-avoidance system, emergency-stop mechanism,
    or substitute for hardware safety interlocks.

    No physical robot commands are executed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
import time
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("evaluate_robot_safety")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

SCHEMA_VERSION = "1.0.0"


# ============================================================
# Exceptions
# ============================================================

class RobotSafetyError(Exception):
    """Base exception for safety evaluation."""


class SafetyConfigurationError(RobotSafetyError):
    """Invalid safety configuration."""


class SafetyDatasetError(RobotSafetyError):
    """Invalid or missing telemetry data."""


class SafetyEvaluationError(RobotSafetyError):
    """Safety evaluation failure."""


# ============================================================
# Enumerations
# ============================================================

class SafetySeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class SafetyCategory(str, Enum):
    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    JOINT_TORQUE = "joint_torque"

    MOTOR_TEMPERATURE = "motor_temperature"

    BATTERY = "battery"

    ROBOT_SPEED = "robot_speed"
    ROBOT_ACCELERATION = "robot_acceleration"

    HUMAN_PROXIMITY = "human_proximity"
    COLLISION = "collision"
    OBSTACLE = "obstacle"

    WORKSPACE = "workspace"

    EMERGENCY_STOP = "emergency_stop"
    SAFETY_INTERLOCK = "safety_interlock"

    SENSOR_HEALTH = "sensor_health"
    COMMUNICATION = "communication"

    MODEL_CONFIDENCE = "model_confidence"

    FAILURE_RISK = "failure_risk"
    RUL = "remaining_useful_life"

    DATA_QUALITY = "data_quality"


class SafetyStatus(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    INDETERMINATE = "indeterminate"


class RobotOperatingMode(str, Enum):
    SIMULATION = "simulation"
    OFFLINE_REPLAY = "offline_replay"


# ============================================================
# Utilities
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def save_json_atomic(
    path: str | Path,
    payload: Mapping[str, Any],
) -> None:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = None

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

            temporary_path = Path(stream.name)

        os.replace(
            temporary_path,
            path,
        )

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()


def safe_float(
    value: Any,
) -> float | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def safe_bool(
    value: Any,
) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, np.integer)):
        if value in (0, 1):
            return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {
            "true",
            "1",
            "yes",
            "on",
        }:
            return True

        if normalized in {
            "false",
            "0",
            "no",
            "off",
        }:
            return False

    return None


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(
        minimum,
        min(maximum, value),
    )


# ============================================================
# Configuration
# ============================================================

@dataclass
class JointSafetyLimits:
    position_min_rad: float = -3.14
    position_max_rad: float = 3.14

    max_velocity_rad_s: float = 2.0
    max_torque_nm: float = 100.0

    max_temperature_c: float = 80.0

    def validate(self) -> None:
        if (
            self.position_min_rad
            >= self.position_max_rad
        ):
            raise SafetyConfigurationError(
                "Invalid joint position limits"
            )

        if self.max_velocity_rad_s <= 0:
            raise SafetyConfigurationError(
                "Joint velocity limit must be positive"
            )

        if self.max_torque_nm <= 0:
            raise SafetyConfigurationError(
                "Joint torque limit must be positive"
            )


@dataclass
class BatterySafetyLimits:
    min_voltage_v: float = 20.0
    max_voltage_v: float = 60.0

    max_temperature_c: float = 60.0
    min_soc_percent: float = 10.0

    def validate(self) -> None:
        if self.min_voltage_v >= self.max_voltage_v:
            raise SafetyConfigurationError(
                "Invalid battery voltage limits"
            )

        if not 0 <= self.min_soc_percent <= 100:
            raise SafetyConfigurationError(
                "Invalid minimum battery SOC"
            )


@dataclass
class MotionSafetyLimits:
    max_linear_speed_m_s: float = 1.5

    max_angular_speed_rad_s: float = 2.0

    max_linear_acceleration_m_s2: float = 2.0

    minimum_obstacle_distance_m: float = 0.5

    minimum_human_distance_m: float = 1.5

    maximum_collision_force_n: float = 50.0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if value < 0:
                raise SafetyConfigurationError(
                    f"{name} cannot be negative"
                )


@dataclass
class SensorSafetyLimits:
    minimum_health_score: float = 0.7

    maximum_telemetry_age_ms: float = 500.0

    maximum_communication_latency_ms: float = 300.0

    minimum_model_confidence: float = 0.6

    def validate(self) -> None:
        if not 0 <= self.minimum_health_score <= 1:
            raise SafetyConfigurationError(
                "Sensor health threshold must be in [0,1]"
            )

        if not 0 <= self.minimum_model_confidence <= 1:
            raise SafetyConfigurationError(
                "Model confidence threshold must be in [0,1]"
            )


@dataclass
class PredictiveSafetyLimits:
    maximum_failure_probability: float = 0.7

    minimum_rul_hours: float = 24.0

    def validate(self) -> None:
        if not 0 <= self.maximum_failure_probability <= 1:
            raise SafetyConfigurationError(
                "Failure probability threshold must be in [0,1]"
            )

        if self.minimum_rul_hours < 0:
            raise SafetyConfigurationError(
                "Minimum RUL cannot be negative"
            )


@dataclass
class RobotSafetyConfig:
    robot_id: str = "robot-001"

    operating_mode: RobotOperatingMode = (
        RobotOperatingMode.SIMULATION
    )

    joint_limits: JointSafetyLimits = field(
        default_factory=JointSafetyLimits
    )

    battery_limits: BatterySafetyLimits = field(
        default_factory=BatterySafetyLimits
    )

    motion_limits: MotionSafetyLimits = field(
        default_factory=MotionSafetyLimits
    )

    sensor_limits: SensorSafetyLimits = field(
        default_factory=SensorSafetyLimits
    )

    predictive_limits: PredictiveSafetyLimits = field(
        default_factory=PredictiveSafetyLimits
    )

    require_emergency_stop_signal: bool = True

    require_safety_interlock_signal: bool = True

    require_human_proximity_signal: bool = True

    require_sensor_health_signal: bool = True

    require_communication_signal: bool = True

    minimum_safety_score: float = 90.0

    max_allowed_high_violations: int = 0

    max_allowed_critical_violations: int = 0

    physical_execution_authorized: bool = False

    def validate(self) -> None:
        self.joint_limits.validate()
        self.battery_limits.validate()
        self.motion_limits.validate()
        self.sensor_limits.validate()
        self.predictive_limits.validate()

        if not 0 <= self.minimum_safety_score <= 100:
            raise SafetyConfigurationError(
                "Safety score threshold must be in [0,100]"
            )

        if self.physical_execution_authorized:
            raise SafetyConfigurationError(
                "Physical execution authorization is unsupported"
            )

        if self.operating_mode not in {
            RobotOperatingMode.SIMULATION,
            RobotOperatingMode.OFFLINE_REPLAY,
        }:
            raise SafetyConfigurationError(
                "Unsupported operating mode"
            )


# ============================================================
# Safety Models
# ============================================================

@dataclass
class SafetyViolation:
    violation_id: str

    category: str
    severity: str

    code: str
    message: str

    observed_value: Any
    threshold: Any

    sample_index: int

    robot_id: str

    timestamp: str

    human_review_required: bool

    recommendation: str


@dataclass
class SafetySampleResult:
    sample_index: int

    robot_id: str

    status: str

    safety_score: float

    violation_count: int

    critical_count: int
    high_count: int

    violations: list[SafetyViolation]


@dataclass
class RobotSafetyReport:
    evaluation_id: str

    robot_id: str

    operating_mode: str

    status: str

    overall_safety_score: float

    sample_count: int

    passed_samples: int
    warning_samples: int
    failed_samples: int
    indeterminate_samples: int

    total_violations: int

    severity_counts: dict[str, int]

    category_counts: dict[str, int]

    quality_gates: dict[str, Any]

    violations: list[dict[str, Any]]

    sample_results: list[dict[str, Any]]

    dataset_metadata: dict[str, Any]

    started_at: str
    completed_at: str

    duration_seconds: float

    human_review_required: bool

    simulation_only: bool = True

    physical_execution_authorized: bool = False

    certified_safe: bool = False


# ============================================================
# Violation Factory
# ============================================================

class ViolationFactory:
    @staticmethod
    def create(
        *,
        category: SafetyCategory,
        severity: SafetySeverity,
        code: str,
        message: str,
        observed_value: Any,
        threshold: Any,
        sample_index: int,
        robot_id: str,
        recommendation: str,
    ) -> SafetyViolation:
        return SafetyViolation(
            violation_id=uuid.uuid4().hex,
            category=category.value,
            severity=severity.value,
            code=code,
            message=message,
            observed_value=observed_value,
            threshold=threshold,
            sample_index=sample_index,
            robot_id=robot_id,
            timestamp=utc_now(),
            human_review_required=severity in {
                SafetySeverity.HIGH,
                SafetySeverity.CRITICAL,
            },
            recommendation=recommendation,
        )


# ============================================================
# Telemetry Dataset Loader
# ============================================================

class RobotSafetyDatasetLoader:
    @staticmethod
    def load(
        path: str | Path,
    ) -> pd.DataFrame:
        path = Path(path)

        if not path.is_file():
            raise SafetyDatasetError(
                f"Telemetry dataset not found: {path}"
            )

        extension = path.suffix.lower()

        if extension == ".csv":
            dataframe = pd.read_csv(path)

        elif extension in {".parquet", ".pq"}:
            dataframe = pd.read_parquet(path)

        elif extension == ".json":
            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            if isinstance(payload, dict):
                payload = payload.get(
                    "telemetry",
                    [payload],
                )

            dataframe = pd.DataFrame(
                payload
            )

        elif extension == ".jsonl":
            records = []

            with path.open(
                "r",
                encoding="utf-8",
            ) as stream:
                for line in stream:
                    if line.strip():
                        records.append(
                            json.loads(line)
                        )

            dataframe = pd.DataFrame(
                records
            )

        else:
            raise SafetyDatasetError(
                f"Unsupported dataset extension: {extension}"
            )

        if dataframe.empty:
            raise SafetyDatasetError(
                "Telemetry dataset is empty"
            )

        return dataframe


# ============================================================
# Individual Safety Evaluator
# ============================================================

class RobotSafetyEvaluator:
    SEVERITY_PENALTIES = {
        SafetySeverity.INFO: 0.0,
        SafetySeverity.LOW: 2.0,
        SafetySeverity.MODERATE: 8.0,
        SafetySeverity.HIGH: 20.0,
        SafetySeverity.CRITICAL: 40.0,
    }

    def __init__(
        self,
        config: RobotSafetyConfig,
    ) -> None:
        config.validate()

        self.config = config

    def _violation(
        self,
        *,
        category: SafetyCategory,
        severity: SafetySeverity,
        code: str,
        message: str,
        observed_value: Any,
        threshold: Any,
        sample_index: int,
        recommendation: str,
    ) -> SafetyViolation:
        return ViolationFactory.create(
            category=category,
            severity=severity,
            code=code,
            message=message,
            observed_value=observed_value,
            threshold=threshold,
            sample_index=sample_index,
            robot_id=self.config.robot_id,
            recommendation=recommendation,
        )

    def _check_upper_limit(
        self,
        telemetry: Mapping[str, Any],
        *,
        field_name: str,
        maximum: float,
        category: SafetyCategory,
        sample_index: int,
        severity: SafetySeverity = SafetySeverity.HIGH,
    ) -> list[SafetyViolation]:
        if field_name not in telemetry:
            return []

        value = safe_float(
            telemetry[field_name]
        )

        if value is None:
            return [
                self._violation(
                    category=SafetyCategory.DATA_QUALITY,
                    severity=SafetySeverity.HIGH,
                    code="INVALID_TELEMETRY_VALUE",
                    message=f"Invalid value for {field_name}",
                    observed_value=str(
                        telemetry[field_name]
                    ),
                    threshold="finite numeric value",
                    sample_index=sample_index,
                    recommendation=(
                        "Investigate telemetry integrity "
                        "and sensor availability."
                    ),
                )
            ]

        if value <= maximum:
            return []

        return [
            self._violation(
                category=category,
                severity=severity,
                code=f"{field_name.upper()}_LIMIT_EXCEEDED",
                message=f"{field_name} exceeds configured limit",
                observed_value=value,
                threshold=maximum,
                sample_index=sample_index,
                recommendation=(
                    "Flag the condition for engineering review. "
                    "Validate sensors and configured limits."
                ),
            )
        ]

    def _check_lower_limit(
        self,
        telemetry: Mapping[str, Any],
        *,
        field_name: str,
        minimum: float,
        category: SafetyCategory,
        sample_index: int,
        severity: SafetySeverity = SafetySeverity.HIGH,
    ) -> list[SafetyViolation]:
        if field_name not in telemetry:
            return []

        value = safe_float(
            telemetry[field_name]
        )

        if value is None:
            return [
                self._violation(
                    category=SafetyCategory.DATA_QUALITY,
                    severity=SafetySeverity.HIGH,
                    code="INVALID_TELEMETRY_VALUE",
                    message=f"Invalid value for {field_name}",
                    observed_value=str(
                        telemetry[field_name]
                    ),
                    threshold="finite numeric value",
                    sample_index=sample_index,
                    recommendation="Inspect telemetry integrity.",
                )
            ]

        if value >= minimum:
            return []

        return [
            self._violation(
                category=category,
                severity=severity,
                code=f"{field_name.upper()}_BELOW_LIMIT",
                message=f"{field_name} is below configured limit",
                observed_value=value,
                threshold=minimum,
                sample_index=sample_index,
                recommendation=(
                    "Review the recorded condition and "
                    "associated safety procedure."
                ),
            )
        ]

    def check_joint_safety(
        self,
        telemetry: Mapping[str, Any],
        sample_index: int,
    ) -> list[SafetyViolation]:
        violations = []

        limits = self.config.joint_limits

        position = safe_float(
            telemetry.get("joint_position_rad")
        )

        if position is not None:
            if not (
                limits.position_min_rad
                <= position
                <= limits.position_max_rad
            ):
                violations.append(
                    self._violation(
                        category=SafetyCategory.JOINT_POSITION,
                        severity=SafetySeverity.CRITICAL,
                        code="JOINT_POSITION_LIMIT",
                        message="Joint position outside configured range",
                        observed_value=position,
                        threshold={
                            "minimum": limits.position_min_rad,
                            "maximum": limits.position_max_rad,
                        },
                        sample_index=sample_index,
                        recommendation=(
                            "Investigate joint limit violation "
                            "using approved safety procedures."
                        ),
                    )
                )

        violations.extend(
            self._check_upper_limit(
                telemetry,
                field_name="joint_velocity_rad_s",
                maximum=limits.max_velocity_rad_s,
                category=SafetyCategory.JOINT_VELOCITY,
                sample_index=sample_index,
            )
        )

        violations.extend(
            self._check_upper_limit(
                telemetry,
                field_name="joint_torque_nm",
                maximum=limits.max_torque_nm,
                category=SafetyCategory.JOINT_TORQUE,
                sample_index=sample_index,
            )
        )

        violations.extend(
            self._check_upper_limit(
                telemetry,
                field_name="motor_temperature_c",
                maximum=limits.max_temperature_c,
                category=SafetyCategory.MOTOR_TEMPERATURE,
                sample_index=sample_index,
                severity=SafetySeverity.CRITICAL,
            )
        )

        return violations

    def check_battery_safety(
        self,
        telemetry: Mapping[str, Any],
        sample_index: int,
    ) -> list[SafetyViolation]:
        limits = self.config.battery_limits

        violations = []

        violations.extend(
            self._check_lower_limit(
                telemetry,
                field_name="battery_voltage_v",
                minimum=limits.min_voltage_v,
                category=SafetyCategory.BATTERY,
                sample_index=sample_index,
            )
        )

        violations.extend(
            self._check_upper_limit(
                telemetry,
                field_name="battery_voltage_v",
                maximum=limits.max_voltage_v,
                category=SafetyCategory.BATTERY,
                sample_index=sample_index,
            )
        )

        violations.extend(
            self._check_upper_limit(
                telemetry,
                field_name="battery_temperature_c",
                maximum=limits.max_temperature_c,
                category=SafetyCategory.BATTERY,
                sample_index=sample_index,
                severity=SafetySeverity.CRITICAL,
            )
        )

        violations.extend(
            self._check_lower_limit(
                telemetry,
                field_name="battery_soc_percent",
                minimum=limits.min_soc_percent,
                category=SafetyCategory.BATTERY,
                sample_index=sample_index,
            )
        )

        return violations

    def check_motion_safety(
        self,
        telemetry: Mapping[str, Any],
        sample_index: int,
    ) -> list[SafetyViolation]:
        limits = self.config.motion_limits

        violations = []

        checks = [
            (
                "linear_speed_m_s",
                limits.max_linear_speed_m_s,
                SafetyCategory.ROBOT_SPEED,
            ),
            (
                "angular_speed_rad_s",
                limits.max_angular_speed_rad_s,
                SafetyCategory.ROBOT_SPEED,
            ),
            (
                "linear_acceleration_m_s2",
                limits.max_linear_acceleration_m_s2,
                SafetyCategory.ROBOT_ACCELERATION,
            ),
        ]

        for field_name, maximum, category in checks:
            violations.extend(
                self._check_upper_limit(
                    telemetry,
                    field_name=field_name,
                    maximum=maximum,
                    category=category,
                    sample_index=sample_index,
                )
            )

        violations.extend(
            self._check_lower_limit(
                telemetry,
                field_name="obstacle_distance_m",
                minimum=limits.minimum_obstacle_distance_m,
                category=SafetyCategory.OBSTACLE,
                sample_index=sample_index,
                severity=SafetySeverity.CRITICAL,
            )
        )

        violations.extend(
            self._check_lower_limit(
                telemetry,
                field_name="human_distance_m",
                minimum=limits.minimum_human_distance_m,
                category=SafetyCategory.HUMAN_PROXIMITY,
                sample_index=sample_index,
                severity=SafetySeverity.CRITICAL,
            )
        )

        violations.extend(
            self._check_upper_limit(
                telemetry,
                field_name="collision_force_n",
                maximum=limits.maximum_collision_force_n,
                category=SafetyCategory.COLLISION,
                sample_index=sample_index,
                severity=SafetySeverity.CRITICAL,
            )
        )

        return violations

    def check_emergency_stop(
        self,
        telemetry: Mapping[str, Any],
        sample_index: int,
    ) -> list[SafetyViolation]:
        violations = []

        value = safe_bool(
            telemetry.get("emergency_stop_active")
        )

        if value is None:
            if self.config.require_emergency_stop_signal:
                violations.append(
                    self._violation(
                        category=SafetyCategory.EMERGENCY_STOP,
                        severity=SafetySeverity.CRITICAL,
                        code="ESTOP_SIGNAL_MISSING",
                        message=(
                            "Emergency-stop state is unavailable "
                            "or invalid"
                        ),
                        observed_value=telemetry.get(
                            "emergency_stop_active"
                        ),
                        threshold="valid boolean signal",
                        sample_index=sample_index,
                        recommendation=(
                            "Treat safety state as unknown. "
                            "Verify the independent hardware "
                            "emergency-stop circuit."
                        ),
                    )
                )

        elif value:
            violations.append(
                self._violation(
                    category=SafetyCategory.EMERGENCY_STOP,
                    severity=SafetySeverity.CRITICAL,
                    code="ESTOP_ACTIVE",
                    message="Emergency stop is active",
                    observed_value=True,
                    threshold=False,
                    sample_index=sample_index,
                    recommendation=(
                        "Keep physical execution disabled. "
                        "Investigate using approved procedures. "
                        "Do not automatically reset the E-stop."
                    ),
                )
            )

        return violations

    def check_safety_interlock(
        self,
        telemetry: Mapping[str, Any],
        sample_index: int,
    ) -> list[SafetyViolation]:
        value = safe_bool(
            telemetry.get("safety_interlock_ok")
        )

        if value is None:
            if not self.config.require_safety_interlock_signal:
                return []

            return [
                self._violation(
                    category=SafetyCategory.SAFETY_INTERLOCK,
                    severity=SafetySeverity.CRITICAL,
                    code="INTERLOCK_SIGNAL_MISSING",
                    message="Safety-interlock state unavailable",
                    observed_value=telemetry.get(
                        "safety_interlock_ok"
                    ),
                    threshold=True,
                    sample_index=sample_index,
                    recommendation=(
                        "Verify independent safety interlocks "
                        "before any separately authorized operation."
                    ),
                )
            ]

        if value:
            return []

        return [
            self._violation(
                category=SafetyCategory.SAFETY_INTERLOCK,
                severity=SafetySeverity.CRITICAL,
                code="INTERLOCK_OPEN",
                message="Safety interlock is not satisfied",
                observed_value=False,
                threshold=True,
                sample_index=sample_index,
                recommendation=(
                    "Investigate the interlock condition. "
                    "Do not bypass safety interlocks."
                ),
            )
        ]

    def check_sensor_health(
        self,
        telemetry: Mapping[str, Any],
        sample_index: int,
    ) -> list[SafetyViolation]:
        limits = self.config.sensor_limits

        violations = []

        if (
            self.config.require_sensor_health_signal
            and "sensor_health_score" not in telemetry
        ):
            violations.append(
                self._violation(
                    category=SafetyCategory.SENSOR_HEALTH,
                    severity=SafetySeverity.HIGH,
                    code="SENSOR_HEALTH_MISSING",
                    message="Sensor health signal is missing",
                    observed_value=None,
                    threshold=limits.minimum_health_score,
                    sample_index=sample_index,
                    recommendation=(
                        "Verify required sensor diagnostics."
                    ),
                )
            )

        violations.extend(
            self._check_lower_limit(
                telemetry,
                field_name="sensor_health_score",
                minimum=limits.minimum_health_score,
                category=SafetyCategory.SENSOR_HEALTH,
                sample_index=sample_index,
            )
        )

        violations.extend(
            self._check_upper_limit(
                telemetry,
                field_name="telemetry_age_ms",
                maximum=limits.maximum_telemetry_age_ms,
                category=SafetyCategory.COMMUNICATION,
                sample_index=sample_index,
            )
        )

        violations.extend(
            self._check_upper_limit(
                telemetry,
                field_name="communication_latency_ms",
                maximum=limits.maximum_communication_latency_ms,
                category=SafetyCategory.COMMUNICATION,
                sample_index=sample_index,
            )
        )

        if self.config.require_communication_signal:
            for name in (
                "telemetry_age_ms",
                "communication_latency_ms",
            ):
                if name not in telemetry:
                    violations.append(
                        self._violation(
                            category=SafetyCategory.COMMUNICATION,
                            severity=SafetySeverity.HIGH,
                            code="COMMUNICATION_SIGNAL_MISSING",
                            message=f"Required signal missing: {name}",
                            observed_value=None,
                            threshold="valid numeric signal",
                            sample_index=sample_index,
                            recommendation=(
                                "Investigate telemetry availability "
                                "and communication monitoring."
                            ),
                        )
                    )

        violations.extend(
            self._check_lower_limit(
                telemetry,
                field_name="model_confidence",
                minimum=limits.minimum_model_confidence,
                category=SafetyCategory.MODEL_CONFIDENCE,
                sample_index=sample_index,
                severity=SafetySeverity.MODERATE,
            )
        )

        return violations

    def check_predictive_risk(
        self,
        telemetry: Mapping[str, Any],
        sample_index: int,
    ) -> list[SafetyViolation]:
        limits = self.config.predictive_limits

        violations = []

        violations.extend(
            self._check_upper_limit(
                telemetry,
                field_name="failure_probability",
                maximum=limits.maximum_failure_probability,
                category=SafetyCategory.FAILURE_RISK,
                sample_index=sample_index,
                severity=SafetySeverity.HIGH,
            )
        )

        violations.extend(
            self._check_lower_limit(
                telemetry,
                field_name="remaining_useful_life_hours",
                minimum=limits.minimum_rul_hours,
                category=SafetyCategory.RUL,
                sample_index=sample_index,
                severity=SafetySeverity.HIGH,
            )
        )

        return violations

    def check_workspace(
        self,
        telemetry: Mapping[str, Any],
        sample_index: int,
    ) -> list[SafetyViolation]:
        value = safe_bool(
            telemetry.get("workspace_violation")
        )

        if value is not True:
            return []

        return [
            self._violation(
                category=SafetyCategory.WORKSPACE,
                severity=SafetySeverity.CRITICAL,
                code="WORKSPACE_BOUNDARY_VIOLATION",
                message="Robot exceeded configured workspace boundary",
                observed_value=True,
                threshold=False,
                sample_index=sample_index,
                recommendation=(
                    "Investigate workspace configuration, "
                    "localization and trajectory planning."
                ),
            )
        ]

    def check_collision(
        self,
        telemetry: Mapping[str, Any],
        sample_index: int,
    ) -> list[SafetyViolation]:
        value = safe_bool(
            telemetry.get("collision_detected")
        )

        if value is not True:
            return []

        return [
            self._violation(
                category=SafetyCategory.COLLISION,
                severity=SafetySeverity.CRITICAL,
                code="COLLISION_DETECTED",
                message="Collision event recorded",
                observed_value=True,
                threshold=False,
                sample_index=sample_index,
                recommendation=(
                    "Review collision evidence and "
                    "approved incident-response procedures."
                ),
            )
        ]

    def check_required_proximity(
        self,
        telemetry: Mapping[str, Any],
        sample_index: int,
    ) -> list[SafetyViolation]:
        if not self.config.require_human_proximity_signal:
            return []

        if "human_distance_m" in telemetry:
            return []

        return [
            self._violation(
                category=SafetyCategory.HUMAN_PROXIMITY,
                severity=SafetySeverity.CRITICAL,
                code="HUMAN_PROXIMITY_UNAVAILABLE",
                message="Human proximity signal is missing",
                observed_value=None,
                threshold="valid human distance signal",
                sample_index=sample_index,
                recommendation=(
                    "Treat human separation as unknown. "
                    "Verify independently rated protective "
                    "sensing before authorized operation."
                ),
            )
        ]

    def evaluate_sample(
        self,
        telemetry: Mapping[str, Any],
        sample_index: int = 0,
    ) -> SafetySampleResult:
        violations = []

        checks = [
            self.check_joint_safety,
            self.check_battery_safety,
            self.check_motion_safety,
            self.check_emergency_stop,
            self.check_safety_interlock,
            self.check_sensor_health,
            self.check_predictive_risk,
            self.check_workspace,
            self.check_collision,
            self.check_required_proximity,
        ]

        for check in checks:
            violations.extend(
                check(
                    telemetry,
                    sample_index,
                )
            )

        penalty = 0.0

        for violation in violations:
            severity = SafetySeverity(
                violation.severity
            )

            penalty += self.SEVERITY_PENALTIES[
                severity
            ]

        score = clamp(
            100.0 - penalty,
            0.0,
            100.0,
        )

        critical_count = sum(
            violation.severity
            == SafetySeverity.CRITICAL.value
            for violation in violations
        )

        high_count = sum(
            violation.severity
            == SafetySeverity.HIGH.value
            for violation in violations
        )

        if critical_count > 0 or high_count > 0:
            status = SafetyStatus.FAIL

        elif violations:
            status = SafetyStatus.WARNING

        else:
            status = SafetyStatus.PASS

        return SafetySampleResult(
            sample_index=sample_index,
            robot_id=self.config.robot_id,
            status=status.value,
            safety_score=score,
            violation_count=len(violations),
            critical_count=critical_count,
            high_count=high_count,
            violations=violations,
        )


# ============================================================
# Batch Safety Evaluation
# ============================================================

class RobotSafetyEvaluationPipeline:
    def __init__(
        self,
        config: RobotSafetyConfig,
    ) -> None:
        config.validate()

        self.config = config
        self.evaluator = RobotSafetyEvaluator(
            config
        )

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        *,
        dataset_metadata: Mapping[str, Any] | None = None,
    ) -> RobotSafetyReport:
        if dataframe.empty:
            raise SafetyDatasetError(
                "Cannot evaluate empty telemetry"
            )

        started_at = utc_now()
        started = time.perf_counter()

        sample_results = []

        all_violations = []

        for sample_index, (_, row) in enumerate(
            dataframe.iterrows()
        ):
            telemetry = row.to_dict()

            result = self.evaluator.evaluate_sample(
                telemetry,
                sample_index,
            )

            sample_results.append(
                result
            )

            all_violations.extend(
                result.violations
            )

        severity_counts = {
            severity.value: 0
            for severity in SafetySeverity
        }

        category_counts = {
            category.value: 0
            for category in SafetyCategory
        }

        for violation in all_violations:
            severity_counts[
                violation.severity
            ] += 1

            category_counts[
                violation.category
            ] += 1

        passed_samples = sum(
            result.status == SafetyStatus.PASS.value
            for result in sample_results
        )

        warning_samples = sum(
            result.status == SafetyStatus.WARNING.value
            for result in sample_results
        )

        failed_samples = sum(
            result.status == SafetyStatus.FAIL.value
            for result in sample_results
        )

        indeterminate_samples = sum(
            result.status
            == SafetyStatus.INDETERMINATE.value
            for result in sample_results
        )

        scores = [
            result.safety_score
            for result in sample_results
        ]

        overall_score = float(
            np.mean(scores)
        )

        high_count = severity_counts[
            SafetySeverity.HIGH.value
        ]

        critical_count = severity_counts[
            SafetySeverity.CRITICAL.value
        ]

        quality_checks = {
            "minimum_safety_score": {
                "threshold": self.config.minimum_safety_score,
                "actual": overall_score,
                "passed": (
                    overall_score
                    >= self.config.minimum_safety_score
                ),
            },
            "maximum_high_violations": {
                "threshold": (
                    self.config.max_allowed_high_violations
                ),
                "actual": high_count,
                "passed": (
                    high_count
                    <= self.config.max_allowed_high_violations
                ),
            },
            "maximum_critical_violations": {
                "threshold": (
                    self.config.max_allowed_critical_violations
                ),
                "actual": critical_count,
                "passed": (
                    critical_count
                    <= self.config.max_allowed_critical_violations
                ),
            },
            "no_failed_samples": {
                "threshold": 0,
                "actual": failed_samples,
                "passed": failed_samples == 0,
            },
        }

        gates_passed = all(
            item["passed"]
            for item in quality_checks.values()
        )

        if gates_passed:
            status = SafetyStatus.PASS

        else:
            status = SafetyStatus.FAIL

        human_review_required = (
            high_count > 0
            or critical_count > 0
            or not gates_passed
        )

        return RobotSafetyReport(
            evaluation_id=uuid.uuid4().hex,
            robot_id=self.config.robot_id,
            operating_mode=self.config.operating_mode.value,
            status=status.value,
            overall_safety_score=round(
                overall_score,
                3,
            ),
            sample_count=len(sample_results),
            passed_samples=passed_samples,
            warning_samples=warning_samples,
            failed_samples=failed_samples,
            indeterminate_samples=indeterminate_samples,
            total_violations=len(all_violations),
            severity_counts=severity_counts,
            category_counts=category_counts,
            quality_gates={
                "passed": gates_passed,
                "checks": quality_checks,
            },
            violations=[
                asdict(violation)
                for violation in all_violations
            ],
            sample_results=[
                {
                    "sample_index": result.sample_index,
                    "robot_id": result.robot_id,
                    "status": result.status,
                    "safety_score": result.safety_score,
                    "violation_count": result.violation_count,
                    "critical_count": result.critical_count,
                    "high_count": result.high_count,
                }
                for result in sample_results
            ],
            dataset_metadata=dict(
                dataset_metadata or {}
            ),
            started_at=started_at,
            completed_at=utc_now(),
            duration_seconds=round(
                time.perf_counter() - started,
                4,
            ),
            human_review_required=human_review_required,
        )


# ============================================================
# Report Export
# ============================================================

class RobotSafetyReportExporter:
    @staticmethod
    def export(
        report: RobotSafetyReport,
        output_directory: str | Path,
    ) -> dict[str, str]:
        output_directory = Path(
            output_directory
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_path = (
            output_directory
            / f"robot_safety_{report.evaluation_id}.json"
        )

        violations_csv_path = (
            output_directory
            / f"robot_safety_violations_{report.evaluation_id}.csv"
        )

        samples_csv_path = (
            output_directory
            / f"robot_safety_samples_{report.evaluation_id}.csv"
        )

        save_json_atomic(
            json_path,
            asdict(report),
        )

        violation_fields = [
            "violation_id",
            "category",
            "severity",
            "code",
            "message",
            "observed_value",
            "threshold",
            "sample_index",
            "robot_id",
            "timestamp",
            "human_review_required",
            "recommendation",
        ]

        with violations_csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=violation_fields,
            )

            writer.writeheader()

            for violation in report.violations:
                writer.writerow(
                    violation
                )

        sample_fields = [
            "sample_index",
            "robot_id",
            "status",
            "safety_score",
            "violation_count",
            "critical_count",
            "high_count",
        ]

        with samples_csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=sample_fields,
            )

            writer.writeheader()

            writer.writerows(
                report.sample_results
            )

        return {
            "json_report": str(json_path),
            "violations_csv": str(violations_csv_path),
            "samples_csv": str(samples_csv_path),
        }


# ============================================================
# YAML Configuration
# ============================================================

def load_safety_config(
    path: str | Path,
) -> RobotSafetyConfig:
    path = Path(path)

    if not path.is_file():
        raise SafetyConfigurationError(
            f"Safety configuration not found: {path}"
        )

    extension = path.suffix.lower()

    if extension in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise SafetyConfigurationError(
                "Install PyYAML to load YAML configurations"
            ) from exc

        with path.open(
            "r",
            encoding="utf-8",
        ) as stream:
            payload = yaml.safe_load(
                stream
            )

    elif extension == ".json":
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    else:
        raise SafetyConfigurationError(
            "Configuration must be JSON or YAML"
        )

    if not isinstance(payload, dict):
        raise SafetyConfigurationError(
            "Safety configuration must be a mapping"
        )

    joint_limits = JointSafetyLimits(
        **payload.get(
            "joint_limits",
            {},
        )
    )

    battery_limits = BatterySafetyLimits(
        **payload.get(
            "battery_limits",
            {},
        )
    )

    motion_limits = MotionSafetyLimits(
        **payload.get(
            "motion_limits",
            {},
        )
    )

    sensor_limits = SensorSafetyLimits(
        **payload.get(
            "sensor_limits",
            {},
        )
    )

    predictive_limits = PredictiveSafetyLimits(
        **payload.get(
            "predictive_limits",
            {},
        )
    )

    config = RobotSafetyConfig(
        robot_id=payload.get(
            "robot_id",
            "robot-001",
        ),
        operating_mode=RobotOperatingMode(
            payload.get(
                "operating_mode",
                "simulation",
            )
        ),
        joint_limits=joint_limits,
        battery_limits=battery_limits,
        motion_limits=motion_limits,
        sensor_limits=sensor_limits,
        predictive_limits=predictive_limits,
        require_emergency_stop_signal=payload.get(
            "require_emergency_stop_signal",
            True,
        ),
        require_safety_interlock_signal=payload.get(
            "require_safety_interlock_signal",
            True,
        ),
        require_human_proximity_signal=payload.get(
            "require_human_proximity_signal",
            True,
        ),
        require_sensor_health_signal=payload.get(
            "require_sensor_health_signal",
            True,
        ),
        require_communication_signal=payload.get(
            "require_communication_signal",
            True,
        ),
        minimum_safety_score=payload.get(
            "minimum_safety_score",
            90.0,
        ),
        max_allowed_high_violations=payload.get(
            "max_allowed_high_violations",
            0,
        ),
        max_allowed_critical_violations=payload.get(
            "max_allowed_critical_violations",
            0,
        ),
        physical_execution_authorized=False,
    )

    config.validate()

    return config


# ============================================================
# Demo Telemetry
# ============================================================

def create_demo_telemetry() -> pd.DataFrame:
    """
    Generate synthetic, non-operational robot telemetry.
    """
    records = [
        {
            "joint_position_rad": 0.5,
            "joint_velocity_rad_s": 0.8,
            "joint_torque_nm": 30.0,
            "motor_temperature_c": 45.0,
            "battery_voltage_v": 48.0,
            "battery_temperature_c": 32.0,
            "battery_soc_percent": 85.0,
            "linear_speed_m_s": 0.6,
            "angular_speed_rad_s": 0.3,
            "linear_acceleration_m_s2": 0.4,
            "obstacle_distance_m": 3.0,
            "human_distance_m": 5.0,
            "collision_force_n": 0.0,
            "collision_detected": False,
            "workspace_violation": False,
            "emergency_stop_active": False,
            "safety_interlock_ok": True,
            "sensor_health_score": 0.98,
            "telemetry_age_ms": 40.0,
            "communication_latency_ms": 20.0,
            "model_confidence": 0.95,
            "failure_probability": 0.05,
            "remaining_useful_life_hours": 500.0,
        },
        {
            "joint_position_rad": 0.8,
            "joint_velocity_rad_s": 1.2,
            "joint_torque_nm": 45.0,
            "motor_temperature_c": 55.0,
            "battery_voltage_v": 47.0,
            "battery_temperature_c": 38.0,
            "battery_soc_percent": 72.0,
            "linear_speed_m_s": 0.9,
            "angular_speed_rad_s": 0.5,
            "linear_acceleration_m_s2": 0.8,
            "obstacle_distance_m": 1.2,
            "human_distance_m": 2.5,
            "collision_force_n": 0.0,
            "collision_detected": False,
            "workspace_violation": False,
            "emergency_stop_active": False,
            "safety_interlock_ok": True,
            "sensor_health_score": 0.92,
            "telemetry_age_ms": 70.0,
            "communication_latency_ms": 45.0,
            "model_confidence": 0.90,
            "failure_probability": 0.12,
            "remaining_useful_life_hours": 350.0,
        },
        {
            "joint_position_rad": 3.5,
            "joint_velocity_rad_s": 2.8,
            "joint_torque_nm": 125.0,
            "motor_temperature_c": 92.0,
            "battery_voltage_v": 46.0,
            "battery_temperature_c": 65.0,
            "battery_soc_percent": 35.0,
            "linear_speed_m_s": 2.1,
            "angular_speed_rad_s": 2.5,
            "linear_acceleration_m_s2": 2.8,
            "obstacle_distance_m": 0.2,
            "human_distance_m": 0.8,
            "collision_force_n": 80.0,
            "collision_detected": True,
            "workspace_violation": True,
            "emergency_stop_active": True,
            "safety_interlock_ok": False,
            "sensor_health_score": 0.45,
            "telemetry_age_ms": 850.0,
            "communication_latency_ms": 600.0,
            "model_confidence": 0.35,
            "failure_probability": 0.91,
            "remaining_useful_life_hours": 8.0,
        },
    ]

    return pd.DataFrame(
        records
    )


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline Physical AI robot safety evaluator"
        )
    )

    parser.add_argument(
        "--data",
        default=None,
        help="CSV, JSON, JSONL, or Parquet telemetry",
    )

    parser.add_argument(
        "--config",
        default=None,
        help="JSON/YAML robot safety configuration",
    )

    parser.add_argument(
        "--output",
        default="reports/safety",
    )

    parser.add_argument(
        "--robot-id",
        default=None,
    )

    parser.add_argument(
        "--demo",
        action="store_true",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 2 if safety quality gates fail",
    )

    parser.add_argument(
        "--summary-only",
        action="store_true",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.demo and not args.data:
        raise SafetyDatasetError(
            "Provide --data or --demo"
        )

    if args.config:
        config = load_safety_config(
            args.config
        )

    else:
        config = RobotSafetyConfig()

    if args.robot_id:
        config.robot_id = args.robot_id

    config.validate()

    if args.demo:
        dataframe = create_demo_telemetry()

        dataset_metadata = {
            "source": "synthetic_demo",
            "record_count": len(dataframe),
            "simulation_only": True,
        }

    else:
        dataframe = RobotSafetyDatasetLoader.load(
            args.data
        )

        dataset_metadata = {
            "source": str(
                Path(args.data).resolve()
            ),
            "sha256": sha256_file(
                args.data
            ),
            "record_count": len(dataframe),
            "simulation_only": True,
        }

    pipeline = RobotSafetyEvaluationPipeline(
        config
    )

    report = pipeline.evaluate(
        dataframe,
        dataset_metadata=dataset_metadata,
    )

    paths = RobotSafetyReportExporter.export(
        report,
        args.output,
    )

    if args.summary_only:
        output = {
            "evaluation_id": report.evaluation_id,
            "robot_id": report.robot_id,
            "status": report.status,
            "overall_safety_score": (
                report.overall_safety_score
            ),
            "sample_count": report.sample_count,
            "total_violations": report.total_violations,
            "severity_counts": report.severity_counts,
            "quality_gates": report.quality_gates,
            "human_review_required": (
                report.human_review_required
            ),
            "report_paths": paths,
            "simulation_only": True,
            "physical_execution_authorized": False,
            "certified_safe": False,
        }

    else:
        output = {
            **asdict(report),
            "report_paths": paths,
        }

    print(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )

    if args.strict and report.status != SafetyStatus.PASS.value:
        return 2

    return 0


if __name__ == "__main__":
    try:
        sys.exit(
            main()
        )

    except (
        RobotSafetyError,
        ValueError,
        OSError,
        KeyError,
    ) as exc:
        LOGGER.error(
            "Robot safety evaluation failed: %s",
            exc,
        )

        sys.exit(1)
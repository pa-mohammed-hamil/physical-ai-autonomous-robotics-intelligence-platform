
"""
Physical AI & Autonomous Robotics Intelligence Platform

File: backend/app/services/robot_health.py

Robot Health Monitoring & Intelligence Service

Python 3.11+
Standard library only.

Features:
- Robot health monitoring
- Battery SOC and SOH monitoring
- Motor temperature monitoring
- Vibration monitoring
- Sensor and network health
- Robot health score calculation
- Rule-based anomaly detection
- External ML-based RUL integration
- Health severity classification
- Maintenance recommendations
- Robot fault and health history
- Fleet health dashboard
- JSON and CSV export

Safety:
- Monitoring and recommendations only
- No hardware commands
- No automatic shutdown or restart
- Not a certified safety monitoring system
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import threading
import uuid

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
# EXCEPTIONS
# ============================================================

class RobotHealthError(Exception):
    pass


class RobotHealthValidationError(RobotHealthError):
    pass


class RobotHealthNotFoundError(RobotHealthError):
    pass


# ============================================================
# ENUMS
# ============================================================

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class HealthComponent(str, Enum):
    BATTERY = "battery"
    MOTOR = "motor"
    VIBRATION = "vibration"
    SENSOR = "sensor"
    NETWORK = "network"
    CONTROLLER = "controller"
    OVERALL = "overall"


class HealthAlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class MaintenanceUrgency(str, Enum):
    ROUTINE = "routine"
    SOON = "soon"
    IMMEDIATE_REVIEW = "immediate_review"


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def require_text(
    value: str,
    name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RobotHealthValidationError(
            f"{name} is required"
        )

    return value.strip()


def finite(
    value: float,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise RobotHealthValidationError(
            f"{name} must be a finite number"
        )

    value = float(value)

    if minimum is not None and value < minimum:
        raise RobotHealthValidationError(
            f"{name} must be >= {minimum}"
        )

    if maximum is not None and value > maximum:
        raise RobotHealthValidationError(
            f"{name} must be <= {maximum}"
        )

    return value


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 100.0,
) -> float:
    return max(
        minimum,
        min(maximum, value),
    )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RobotHealthConfig:
    battery_warning_soc: float = 20.0
    battery_critical_soc: float = 10.0

    battery_warning_soh: float = 80.0
    battery_critical_soh: float = 60.0

    motor_warning_temperature_c: float = 70.0
    motor_critical_temperature_c: float = 90.0

    vibration_warning_mm_s: float = 6.0
    vibration_critical_mm_s: float = 12.0

    health_warning_score: float = 70.0
    health_critical_score: float = 40.0

    rul_warning_hours: float = 100.0
    rul_critical_hours: float = 24.0

    anomaly_z_score_threshold: float = 3.0
    minimum_anomaly_samples: int = 10

    battery_weight: float = 0.25
    motor_weight: float = 0.25
    vibration_weight: float = 0.20
    sensor_weight: float = 0.15
    network_weight: float = 0.15

    maximum_history_per_robot: int = 1000
    maximum_alert_history: int = 10000

    allow_hardware_control: bool = False

    def __post_init__(self) -> None:
        percentage_fields = (
            "battery_warning_soc",
            "battery_critical_soc",
            "battery_warning_soh",
            "battery_critical_soh",
            "health_warning_score",
            "health_critical_score",
        )

        for name in percentage_fields:
            finite(
                getattr(self, name),
                name,
                minimum=0.0,
                maximum=100.0,
            )

        if not (
            self.battery_critical_soc
            < self.battery_warning_soc
        ):
            raise RobotHealthValidationError(
                "Invalid battery SOC thresholds"
            )

        if not (
            self.battery_critical_soh
            < self.battery_warning_soh
        ):
            raise RobotHealthValidationError(
                "Invalid battery SOH thresholds"
            )

        if not (
            self.motor_warning_temperature_c
            < self.motor_critical_temperature_c
        ):
            raise RobotHealthValidationError(
                "Invalid motor temperature thresholds"
            )

        if not (
            self.vibration_warning_mm_s
            < self.vibration_critical_mm_s
        ):
            raise RobotHealthValidationError(
                "Invalid vibration thresholds"
            )

        if not (
            self.health_critical_score
            < self.health_warning_score
        ):
            raise RobotHealthValidationError(
                "Invalid health score thresholds"
            )

        if not (
            self.rul_critical_hours
            < self.rul_warning_hours
        ):
            raise RobotHealthValidationError(
                "Invalid RUL thresholds"
            )

        weights = (
            self.battery_weight,
            self.motor_weight,
            self.vibration_weight,
            self.sensor_weight,
            self.network_weight,
        )

        if any(
            finite(weight, "weight") < 0
            for weight in weights
        ):
            raise RobotHealthValidationError(
                "Health weights cannot be negative"
            )

        if not math.isclose(
            sum(weights),
            1.0,
            abs_tol=1e-6,
        ):
            raise RobotHealthValidationError(
                "Health weights must sum to 1.0"
            )

        if self.minimum_anomaly_samples < 3:
            raise RobotHealthValidationError(
                "At least 3 anomaly samples required"
            )

        if (
            self.maximum_history_per_robot < 1
            or self.maximum_alert_history < 1
        ):
            raise RobotHealthValidationError(
                "History limits must be positive"
            )

        if self.allow_hardware_control:
            raise RobotHealthValidationError(
                "Hardware control is disabled"
            )


# ============================================================
# ROBOT TELEMETRY
# ============================================================

@dataclass(frozen=True)
class RobotHealthTelemetry:
    robot_id: str

    timestamp_seconds: float

    battery_soc_percent: float
    battery_soh_percent: float

    motor_temperature_c: float
    vibration_mm_s: float

    sensor_available: bool = True
    network_connected: bool = True
    controller_healthy: bool = True

    operating_hours: float = 0.0

    rul_hours: float | None = None

    fault_count: int = 0

    def __post_init__(self) -> None:
        require_text(
            self.robot_id,
            "robot_id",
        )

        finite(
            self.timestamp_seconds,
            "timestamp_seconds",
        )

        for name in (
            "battery_soc_percent",
            "battery_soh_percent",
        ):
            finite(
                getattr(self, name),
                name,
                minimum=0.0,
                maximum=100.0,
            )

        finite(
            self.motor_temperature_c,
            "motor_temperature_c",
        )

        finite(
            self.vibration_mm_s,
            "vibration_mm_s",
            minimum=0.0,
        )

        finite(
            self.operating_hours,
            "operating_hours",
            minimum=0.0,
        )

        if self.rul_hours is not None:
            finite(
                self.rul_hours,
                "rul_hours",
                minimum=0.0,
            )

        if (
            isinstance(self.fault_count, bool)
            or not isinstance(self.fault_count, int)
            or self.fault_count < 0
        ):
            raise RobotHealthValidationError(
                "fault_count must be non-negative"
            )

        for name in (
            "sensor_available",
            "network_connected",
            "controller_healthy",
        ):
            if not isinstance(
                getattr(self, name),
                bool,
            ):
                raise RobotHealthValidationError(
                    f"{name} must be boolean"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "timestamp_seconds": (
                self.timestamp_seconds
            ),
            "battery_soc_percent": (
                self.battery_soc_percent
            ),
            "battery_soh_percent": (
                self.battery_soh_percent
            ),
            "motor_temperature_c": (
                self.motor_temperature_c
            ),
            "vibration_mm_s": (
                self.vibration_mm_s
            ),
            "sensor_available": (
                self.sensor_available
            ),
            "network_connected": (
                self.network_connected
            ),
            "controller_healthy": (
                self.controller_healthy
            ),
            "operating_hours": (
                self.operating_hours
            ),
            "rul_hours": self.rul_hours,
            "fault_count": self.fault_count,
        }


# ============================================================
# COMPONENT HEALTH RESULT
# ============================================================

@dataclass(frozen=True)
class ComponentHealth:
    component: HealthComponent
    score: float
    status: HealthStatus
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component.value,
            "score": round(self.score, 2),
            "status": self.status.value,
            "explanation": self.explanation,
        }


# ============================================================
# HEALTH ALERT
# ============================================================

@dataclass(frozen=True)
class RobotHealthAlert:
    alert_id: str
    robot_id: str

    component: HealthComponent
    severity: HealthAlertSeverity

    message: str

    timestamp_seconds: float

    created_at: str = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "robot_id": self.robot_id,
            "component": self.component.value,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp_seconds": (
                self.timestamp_seconds
            ),
            "created_at": self.created_at,
        }


# ============================================================
# MAINTENANCE RECOMMENDATION
# ============================================================

@dataclass(frozen=True)
class HealthMaintenanceRecommendation:
    recommendation_id: str
    robot_id: str

    component: HealthComponent
    urgency: MaintenanceUrgency

    action: str
    reason: str

    created_at: str = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation_id": (
                self.recommendation_id
            ),
            "robot_id": self.robot_id,
            "component": self.component.value,
            "urgency": self.urgency.value,
            "action": self.action,
            "reason": self.reason,
            "created_at": self.created_at,
        }


# ============================================================
# HEALTH SCORING ENGINE
# ============================================================

class RobotHealthScoringEngine:

    def __init__(
        self,
        config: RobotHealthConfig,
    ) -> None:
        self.config = config

    def classify(
        self,
        score: float,
    ) -> HealthStatus:
        if score <= self.config.health_critical_score:
            return HealthStatus.CRITICAL

        if score <= self.config.health_warning_score:
            return HealthStatus.WARNING

        return HealthStatus.HEALTHY

    def _low_is_bad(
        self,
        value: float,
        warning: float,
        critical: float,
    ) -> float:
        if value <= critical:
            return 20.0 * (
                value / critical
            ) if critical > 0 else 0.0

        if value <= warning:
            fraction = (
                (value - critical)
                / (warning - critical)
            )

            return 20.0 + 50.0 * fraction

        return clamp(
            70.0
            + 30.0
            * (value - warning)
            / max(100.0 - warning, 1.0)
        )

    def _high_is_bad(
        self,
        value: float,
        warning: float,
        critical: float,
    ) -> float:
        if value >= critical:
            return 0.0

        if value >= warning:
            fraction = (
                (critical - value)
                / (critical - warning)
            )

            return 70.0 * fraction

        if warning <= 0:
            return 100.0

        return clamp(
            70.0
            + 30.0
            * (warning - max(value, 0.0))
            / warning
        )

    def evaluate(
        self,
        telemetry: RobotHealthTelemetry,
    ) -> tuple[
        float,
        HealthStatus,
        list[ComponentHealth],
    ]:
        battery_soc_score = self._low_is_bad(
            telemetry.battery_soc_percent,
            self.config.battery_warning_soc,
            self.config.battery_critical_soc,
        )

        battery_soh_score = self._low_is_bad(
            telemetry.battery_soh_percent,
            self.config.battery_warning_soh,
            self.config.battery_critical_soh,
        )

        battery_score = (
            0.4 * battery_soc_score
            + 0.6 * battery_soh_score
        )

        motor_score = self._high_is_bad(
            telemetry.motor_temperature_c,
            self.config.motor_warning_temperature_c,
            self.config.motor_critical_temperature_c,
        )

        vibration_score = self._high_is_bad(
            telemetry.vibration_mm_s,
            self.config.vibration_warning_mm_s,
            self.config.vibration_critical_mm_s,
        )

        sensor_score = (
            100.0
            if telemetry.sensor_available
            else 0.0
        )

        network_score = (
            100.0
            if telemetry.network_connected
            else 0.0
        )

        components = [
            ComponentHealth(
                component=HealthComponent.BATTERY,
                score=battery_score,
                status=self.classify(battery_score),
                explanation="Battery SOC and SOH assessment",
            ),
            ComponentHealth(
                component=HealthComponent.MOTOR,
                score=motor_score,
                status=self.classify(motor_score),
                explanation="Motor temperature assessment",
            ),
            ComponentHealth(
                component=HealthComponent.VIBRATION,
                score=vibration_score,
                status=self.classify(vibration_score),
                explanation="Mechanical vibration assessment",
            ),
            ComponentHealth(
                component=HealthComponent.SENSOR,
                score=sensor_score,
                status=self.classify(sensor_score),
                explanation="Sensor availability assessment",
            ),
            ComponentHealth(
                component=HealthComponent.NETWORK,
                score=network_score,
                status=self.classify(network_score),
                explanation="Network connectivity assessment",
            ),
        ]

        overall_score = (
            battery_score * self.config.battery_weight
            + motor_score * self.config.motor_weight
            + vibration_score * self.config.vibration_weight
            + sensor_score * self.config.sensor_weight
            + network_score * self.config.network_weight
        )

        if not telemetry.controller_healthy:
            overall_score = min(
                overall_score,
                self.config.health_critical_score,
            )

        if telemetry.fault_count > 0:
            overall_score = max(
                0.0,
                overall_score
                - min(
                    30.0,
                    telemetry.fault_count * 5.0,
                ),
            )

        return (
            round(overall_score, 2),
            self.classify(overall_score),
            components,
        )


# ============================================================
# ANOMALY DETECTION
# ============================================================

class RobotHealthAnomalyDetector:

    def __init__(
        self,
        config: RobotHealthConfig,
    ) -> None:
        self.config = config

    def detect(
        self,
        historical_values: Sequence[float],
        current_value: float,
    ) -> tuple[bool, float | None]:
        if (
            len(historical_values)
            < self.config.minimum_anomaly_samples
        ):
            return False, None

        baseline = [
            finite(value, "historical_value")
            for value in historical_values
        ]

        current_value = finite(
            current_value,
            "current_value",
        )

        mean = statistics.mean(baseline)
        std = statistics.pstdev(baseline)

        if std < 1e-9:
            if abs(current_value - mean) < 1e-9:
                return False, 0.0

            return True, None

        z_score = abs(
            (current_value - mean) / std
        )

        return (
            z_score
            >= self.config.anomaly_z_score_threshold,
            round(z_score, 3),
        )


# ============================================================
# ROBOT HEALTH REPORT
# ============================================================

@dataclass
class RobotHealthReport:
    report_id: str
    robot_id: str

    timestamp_seconds: float

    overall_score: float
    overall_status: HealthStatus

    components: list[ComponentHealth]

    anomalies: dict[str, Any]

    alerts: list[RobotHealthAlert]

    recommendations: list[
        HealthMaintenanceRecommendation
    ]

    rul_hours: float | None

    created_at: str = field(
        default_factory=utc_now
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "robot_id": self.robot_id,
            "timestamp_seconds": (
                self.timestamp_seconds
            ),
            "overall_score": self.overall_score,
            "overall_status": self.overall_status.value,
            "components": [
                component.to_dict()
                for component in self.components
            ],
            "anomalies": self.anomalies,
            "alerts": [
                alert.to_dict()
                for alert in self.alerts
            ],
            "recommendations": [
                recommendation.to_dict()
                for recommendation in self.recommendations
            ],
            "rul_hours": self.rul_hours,
            "created_at": self.created_at,
        }


# ============================================================
# ROBOT HEALTH SERVICE
# ============================================================

class RoboticsRobotHealthService:

    def __init__(
        self,
        config: RobotHealthConfig | None = None,
    ) -> None:
        self.config = (
            config or RobotHealthConfig()
        )

        self.scoring_engine = (
            RobotHealthScoringEngine(
                self.config
            )
        )

        self.anomaly_detector = (
            RobotHealthAnomalyDetector(
                self.config
            )
        )

        self.telemetry_history: dict[
            str,
            deque[RobotHealthTelemetry]
        ] = defaultdict(
            lambda: deque(
                maxlen=self.config.maximum_history_per_robot
            )
        )

        self.report_history: dict[
            str,
            deque[RobotHealthReport]
        ] = defaultdict(
            lambda: deque(
                maxlen=self.config.maximum_history_per_robot
            )
        )

        self.alert_history: deque[
            RobotHealthAlert
        ] = deque(
            maxlen=self.config.maximum_alert_history
        )

        self._lock = threading.RLock()

    # --------------------------------------------------------
    # BUILD ALERT
    # --------------------------------------------------------

    def _alert(
        self,
        telemetry: RobotHealthTelemetry,
        component: HealthComponent,
        severity: HealthAlertSeverity,
        message: str,
    ) -> RobotHealthAlert:
        return RobotHealthAlert(
            alert_id=str(uuid.uuid4()),
            robot_id=telemetry.robot_id,
            component=component,
            severity=severity,
            message=message,
            timestamp_seconds=telemetry.timestamp_seconds,
        )

    # --------------------------------------------------------
    # BUILD RECOMMENDATION
    # --------------------------------------------------------

    def _recommendation(
        self,
        robot_id: str,
        component: HealthComponent,
        urgency: MaintenanceUrgency,
        action: str,
        reason: str,
    ) -> HealthMaintenanceRecommendation:
        return HealthMaintenanceRecommendation(
            recommendation_id=str(uuid.uuid4()),
            robot_id=robot_id,
            component=component,
            urgency=urgency,
            action=action,
            reason=reason,
        )

    # --------------------------------------------------------
    # INGEST TELEMETRY
    # --------------------------------------------------------

    def ingest(
        self,
        telemetry: RobotHealthTelemetry,
    ) -> RobotHealthReport:
        with self._lock:
            history = self.telemetry_history[
                telemetry.robot_id
            ]

            if (
                history
                and telemetry.timestamp_seconds
                <= history[-1].timestamp_seconds
            ):
                raise RobotHealthValidationError(
                    "Telemetry timestamps must increase"
                )

            score, status, components = (
                self.scoring_engine.evaluate(
                    telemetry
                )
            )

            anomalies = {}

            for name in (
                "motor_temperature_c",
                "vibration_mm_s",
            ):
                historical_values = [
                    getattr(item, name)
                    for item in history
                ]

                detected, z_score = (
                    self.anomaly_detector.detect(
                        historical_values,
                        getattr(telemetry, name),
                    )
                )

                anomalies[name] = {
                    "detected": detected,
                    "z_score": z_score,
                }

            alerts = []
            recommendations = []

            if (
                telemetry.battery_soc_percent
                <= self.config.battery_critical_soc
            ):
                alerts.append(
                    self._alert(
                        telemetry,
                        HealthComponent.BATTERY,
                        HealthAlertSeverity.CRITICAL,
                        "Battery SOC critically low",
                    )
                )

            elif (
                telemetry.battery_soc_percent
                <= self.config.battery_warning_soc
            ):
                alerts.append(
                    self._alert(
                        telemetry,
                        HealthComponent.BATTERY,
                        HealthAlertSeverity.WARNING,
                        "Battery SOC low",
                    )
                )

            if (
                telemetry.battery_soh_percent
                <= self.config.battery_critical_soh
            ):
                alerts.append(
                    self._alert(
                        telemetry,
                        HealthComponent.BATTERY,
                        HealthAlertSeverity.CRITICAL,
                        "Battery health critically degraded",
                    )
                )

                recommendations.append(
                    self._recommendation(
                        telemetry.robot_id,
                        HealthComponent.BATTERY,
                        MaintenanceUrgency.IMMEDIATE_REVIEW,
                        "Inspect battery and assess replacement",
                        "Battery SOH below critical threshold",
                    )
                )

            elif (
                telemetry.battery_soh_percent
                <= self.config.battery_warning_soh
            ):
                recommendations.append(
                    self._recommendation(
                        telemetry.robot_id,
                        HealthComponent.BATTERY,
                        MaintenanceUrgency.SOON,
                        "Schedule battery health inspection",
                        "Battery SOH below warning threshold",
                    )
                )

            if (
                telemetry.motor_temperature_c
                >= self.config.motor_critical_temperature_c
            ):
                alerts.append(
                    self._alert(
                        telemetry,
                        HealthComponent.MOTOR,
                        HealthAlertSeverity.CRITICAL,
                        "Motor temperature critically high",
                    )
                )

                recommendations.append(
                    self._recommendation(
                        telemetry.robot_id,
                        HealthComponent.MOTOR,
                        MaintenanceUrgency.IMMEDIATE_REVIEW,
                        "Inspect motor thermal condition",
                        "Critical motor temperature",
                    )
                )

            elif (
                telemetry.motor_temperature_c
                >= self.config.motor_warning_temperature_c
            ):
                alerts.append(
                    self._alert(
                        telemetry,
                        HealthComponent.MOTOR,
                        HealthAlertSeverity.WARNING,
                        "Motor temperature above warning threshold",
                    )
                )

            if (
                telemetry.vibration_mm_s
                >= self.config.vibration_critical_mm_s
            ):
                alerts.append(
                    self._alert(
                        telemetry,
                        HealthComponent.VIBRATION,
                        HealthAlertSeverity.CRITICAL,
                        "Critical mechanical vibration",
                    )
                )

                recommendations.append(
                    self._recommendation(
                        telemetry.robot_id,
                        HealthComponent.VIBRATION,
                        MaintenanceUrgency.IMMEDIATE_REVIEW,
                        "Inspect bearings, joints and drivetrain",
                        "Critical vibration detected",
                    )
                )

            elif (
                telemetry.vibration_mm_s
                >= self.config.vibration_warning_mm_s
            ):
                alerts.append(
                    self._alert(
                        telemetry,
                        HealthComponent.VIBRATION,
                        HealthAlertSeverity.WARNING,
                        "Elevated mechanical vibration",
                    )
                )

            if not telemetry.sensor_available:
                alerts.append(
                    self._alert(
                        telemetry,
                        HealthComponent.SENSOR,
                        HealthAlertSeverity.CRITICAL,
                        "Robot sensor unavailable",
                    )
                )

            if not telemetry.network_connected:
                alerts.append(
                    self._alert(
                        telemetry,
                        HealthComponent.NETWORK,
                        HealthAlertSeverity.WARNING,
                        "Robot network disconnected",
                    )
                )

            if not telemetry.controller_healthy:
                alerts.append(
                    self._alert(
                        telemetry,
                        HealthComponent.CONTROLLER,
                        HealthAlertSeverity.CRITICAL,
                        "Robot controller health failure",
                    )
                )

            if telemetry.rul_hours is not None:
                if (
                    telemetry.rul_hours
                    <= self.config.rul_critical_hours
                ):
                    recommendations.append(
                        self._recommendation(
                            telemetry.robot_id,
                            HealthComponent.OVERALL,
                            MaintenanceUrgency.IMMEDIATE_REVIEW,
                            "Arrange urgent maintenance assessment",
                            "Predicted RUL below critical threshold",
                        )
                    )

                elif (
                    telemetry.rul_hours
                    <= self.config.rul_warning_hours
                ):
                    recommendations.append(
                        self._recommendation(
                            telemetry.robot_id,
                            HealthComponent.OVERALL,
                            MaintenanceUrgency.SOON,
                            "Schedule predictive maintenance",
                            "Predicted RUL below warning threshold",
                        )
                    )

            for metric, anomaly in anomalies.items():
                if anomaly["detected"]:
                    alerts.append(
                        self._alert(
                            telemetry,
                            (
                                HealthComponent.MOTOR
                                if metric == "motor_temperature_c"
                                else HealthComponent.VIBRATION
                            ),
                            HealthAlertSeverity.WARNING,
                            f"Statistical anomaly detected: {metric}",
                        )
                    )

            if status == HealthStatus.CRITICAL:
                alerts.append(
                    self._alert(
                        telemetry,
                        HealthComponent.OVERALL,
                        HealthAlertSeverity.CRITICAL,
                        f"Overall robot health critical: {score}",
                    )
                )

            report = RobotHealthReport(
                report_id=str(uuid.uuid4()),
                robot_id=telemetry.robot_id,
                timestamp_seconds=telemetry.timestamp_seconds,
                overall_score=score,
                overall_status=status,
                components=components,
                anomalies=anomalies,
                alerts=alerts,
                recommendations=recommendations,
                rul_hours=telemetry.rul_hours,
            )

            history.append(telemetry)

            self.report_history[
                telemetry.robot_id
            ].append(report)

            self.alert_history.extend(alerts)

            return report

    # --------------------------------------------------------
    # LATEST HEALTH REPORT
    # --------------------------------------------------------

    def latest(
        self,
        robot_id: str,
    ) -> RobotHealthReport:
        with self._lock:
            reports = self.report_history.get(
                robot_id
            )

            if not reports:
                raise RobotHealthNotFoundError(
                    f"No health data for {robot_id}"
                )

            return reports[-1]

    # --------------------------------------------------------
    # ROBOT HISTORY
    # --------------------------------------------------------

    def robot_history(
        self,
        robot_id: str,
    ) -> list[RobotHealthReport]:
        with self._lock:
            return list(
                self.report_history.get(
                    robot_id,
                    (),
                )
            )

    # --------------------------------------------------------
    # FLEET HEALTH
    # --------------------------------------------------------

    def fleet_health(
        self,
    ) -> dict[str, Any]:
        with self._lock:
            latest_reports = [
                reports[-1]
                for reports in self.report_history.values()
                if reports
            ]

        scores = [
            report.overall_score
            for report in latest_reports
        ]

        return {
            "total_robots": len(latest_reports),
            "average_health_score": (
                round(
                    statistics.mean(scores),
                    2,
                )
                if scores
                else None
            ),
            "minimum_health_score": (
                min(scores)
                if scores
                else None
            ),
            "status_distribution": dict(
                Counter(
                    report.overall_status.value
                    for report in latest_reports
                )
            ),
            "critical_robots": [
                report.robot_id
                for report in latest_reports
                if report.overall_status
                == HealthStatus.CRITICAL
            ],
            "warning_robots": [
                report.robot_id
                for report in latest_reports
                if report.overall_status
                == HealthStatus.WARNING
            ],
            "total_recorded_alerts": len(
                self.alert_history
            ),
            "simulation_only": True,
        }

    # --------------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------------

    def dashboard(
        self,
    ) -> dict[str, Any]:
        with self._lock:
            latest_reports = {
                robot_id: reports[-1].to_dict()
                for robot_id, reports
                in self.report_history.items()
                if reports
            }

            recent_alerts = [
                alert.to_dict()
                for alert in list(
                    self.alert_history
                )[-100:]
            ]

        return {
            "generated_at": utc_now(),
            "fleet_health": self.fleet_health(),
            "robots": latest_reports,
            "recent_alerts": recent_alerts,
        }

    # --------------------------------------------------------
    # EXPORT JSON
    # --------------------------------------------------------

    def export_json(
        self,
        path: str | Path,
    ) -> Path:
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.write_text(
            json.dumps(
                self.dashboard(),
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        return output

    # --------------------------------------------------------
    # EXPORT CSV
    # --------------------------------------------------------

    def export_csv(
        self,
        path: str | Path,
    ) -> Path:
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = [
            "robot_id",
            "timestamp_seconds",
            "overall_score",
            "overall_status",
            "rul_hours",
            "alert_count",
            "recommendation_count",
        ]

        with output.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for robot_id, reports in (
                self.report_history.items()
            ):
                for report in reports:
                    writer.writerow({
                        "robot_id": robot_id,
                        "timestamp_seconds": (
                            report.timestamp_seconds
                        ),
                        "overall_score": (
                            report.overall_score
                        ),
                        "overall_status": (
                            report.overall_status.value
                        ),
                        "rul_hours": (
                            report.rul_hours
                        ),
                        "alert_count": len(
                            report.alerts
                        ),
                        "recommendation_count": len(
                            report.recommendations
                        ),
                    })

        return output


# ============================================================
# EXTERNAL RUL MODEL ADAPTER
# ============================================================

class RobotRULAdapter:
    """
    Integrates externally generated RUL predictions.

    This adapter does not train or run an RUL model.
    """

    @staticmethod
    def apply_prediction(
        telemetry: RobotHealthTelemetry,
        prediction: Mapping[str, Any],
    ) -> RobotHealthTelemetry:
        try:
            rul_hours = finite(
                prediction["rul_hours"],
                "rul_hours",
                minimum=0.0,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RobotHealthValidationError(
                "Invalid RUL prediction"
            ) from exc

        return RobotHealthTelemetry(
            robot_id=telemetry.robot_id,
            timestamp_seconds=telemetry.timestamp_seconds,
            battery_soc_percent=(
                telemetry.battery_soc_percent
            ),
            battery_soh_percent=(
                telemetry.battery_soh_percent
            ),
            motor_temperature_c=(
                telemetry.motor_temperature_c
            ),
            vibration_mm_s=(
                telemetry.vibration_mm_s
            ),
            sensor_available=(
                telemetry.sensor_available
            ),
            network_connected=(
                telemetry.network_connected
            ),
            controller_healthy=(
                telemetry.controller_healthy
            ),
            operating_hours=(
                telemetry.operating_hours
            ),
            rul_hours=rul_hours,
            fault_count=telemetry.fault_count,
        )


# ============================================================
# DEMONSTRATION
# ============================================================

def main() -> None:
    service = RoboticsRobotHealthService()

    for index in range(15):
        telemetry = RobotHealthTelemetry(
            robot_id="robot_001",
            timestamp_seconds=float(index + 1),
            battery_soc_percent=(
                90.0 - index * 2.0
            ),
            battery_soh_percent=86.0,
            motor_temperature_c=(
                45.0 + index * 2.5
            ),
            vibration_mm_s=(
                2.0 + index * 0.5
            ),
            sensor_available=True,
            network_connected=True,
            controller_healthy=True,
            operating_hours=(
                500.0 + index
            ),
            rul_hours=(
                150.0 - index * 8.0
            ),
        )

        report = service.ingest(
            telemetry
        )

    print(
        json.dumps(
            {
                "latest_report": report.to_dict(),
                "fleet_health": service.fleet_health(),
            },
            indent=2,
        )
    )

    service.export_json(
        "data/health/robot_health_report.json"
    )

    service.export_csv(
        "data/health/robot_health_history.csv"
    )


if __name__ == "__main__":
    main()

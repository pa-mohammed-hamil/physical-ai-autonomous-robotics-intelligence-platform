"""Alert Service module.

TODO: implement and test; no real robot control is enabled.
"""

"""
Physical AI & Autonomous Robotics Intelligence Platform

File:
    backend/app/services/alert_service.py

Real-Time Robotics Alert Management Service

Features:
    - Robot health monitoring
    - Battery low / critical alerts
    - Motor overheating detection
    - Excessive vibration detection
    - Sensor failure and dropout alerts
    - Collision and navigation alerts
    - Predictive maintenance alerts
    - Emergency safety alerts
    - Alert deduplication
    - Alert cooldowns
    - Alert acknowledgment
    - Alert resolution
    - Alert escalation
    - Alert history
    - Multi-robot monitoring
    - Async notification callbacks
    - WebSocket-ready event payloads
    - JSON / JSONL export
    - Simulation telemetry integration

Python:
    3.11+

Dependencies:
    Standard library only

Safety:
    Alerts are advisory.
    No automatic physical robot shutdown,
    movement or actuator commands.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import uuid

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping


# ============================================================
# EXCEPTIONS
# ============================================================

class AlertServiceError(Exception):
    pass


class AlertValidationError(AlertServiceError):
    pass


class AlertNotFoundError(AlertServiceError):
    pass


class AlertStateError(AlertServiceError):
    pass


# ============================================================
# ENUMS
# ============================================================

class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(str, Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class AlertCategory(str, Enum):
    BATTERY = "battery"
    MOTOR = "motor"
    VIBRATION = "vibration"
    SENSOR = "sensor"
    NAVIGATION = "navigation"
    COLLISION = "collision"
    HEALTH = "health"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"
    NETWORK = "network"
    SYSTEM = "system"


SEVERITY_RANK = {
    AlertSeverity.INFO: 0,
    AlertSeverity.WARNING: 1,
    AlertSeverity.HIGH: 2,
    AlertSeverity.CRITICAL: 3,
    AlertSeverity.EMERGENCY: 4,
}


# ============================================================
# UTILITIES
# ============================================================

def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def finite(
    value: float,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise AlertValidationError(
            f"{name} must be finite"
        )

    return float(value)


def json_safe(value: Any) -> Any:
    if value is None or isinstance(
        value,
        (str, bool, int),
    ):
        return value

    if isinstance(value, float):
        finite(value, "JSON value")
        return value

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())

    raise AlertValidationError(
        f"Unsupported JSON value: {type(value).__name__}"
    )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class AlertServiceConfig:
    battery_warning_soc_percent: float = 20.0
    battery_critical_soc_percent: float = 10.0

    motor_warning_temperature_c: float = 70.0
    motor_critical_temperature_c: float = 90.0

    vibration_warning_mm_s: float = 6.0
    vibration_critical_mm_s: float = 12.0

    health_warning_score: float = 70.0
    health_critical_score: float = 40.0

    maintenance_warning_rul_hours: float = 100.0
    maintenance_critical_rul_hours: float = 24.0

    alert_cooldown_seconds: float = 30.0

    escalation_after_seconds: float = 300.0

    maximum_history: int = 10000

    maximum_active_alerts: int = 5000

    allow_hardware_control: bool = False

    def __post_init__(self) -> None:
        if not (
            0 <= self.battery_critical_soc_percent
            < self.battery_warning_soc_percent
            <= 100
        ):
            raise AlertValidationError(
                "Invalid battery thresholds"
            )

        if not (
            self.motor_warning_temperature_c
            < self.motor_critical_temperature_c
        ):
            raise AlertValidationError(
                "Invalid motor temperature thresholds"
            )

        if not (
            0 <= self.vibration_warning_mm_s
            < self.vibration_critical_mm_s
        ):
            raise AlertValidationError(
                "Invalid vibration thresholds"
            )

        if not (
            0 <= self.health_critical_score
            < self.health_warning_score
            <= 100
        ):
            raise AlertValidationError(
                "Invalid health thresholds"
            )

        if not (
            0 <= self.maintenance_critical_rul_hours
            < self.maintenance_warning_rul_hours
        ):
            raise AlertValidationError(
                "Invalid RUL thresholds"
            )

        if self.alert_cooldown_seconds < 0:
            raise AlertValidationError(
                "Cooldown cannot be negative"
            )

        if self.escalation_after_seconds < 0:
            raise AlertValidationError(
                "Escalation delay cannot be negative"
            )

        if (
            self.maximum_history < 1
            or self.maximum_active_alerts < 1
        ):
            raise AlertValidationError(
                "Alert limits must be positive"
            )

        if self.allow_hardware_control:
            raise AlertValidationError(
                "Hardware control is prohibited"
            )


# ============================================================
# ALERT MODEL
# ============================================================

@dataclass
class RobotAlert:
    alert_id: str

    robot_id: str

    category: AlertCategory
    severity: AlertSeverity
    status: AlertStatus

    code: str
    title: str
    message: str

    created_at: str
    updated_at: str

    first_seen_seconds: float
    last_seen_seconds: float

    occurrence_count: int = 1

    acknowledged_at: str | None = None
    acknowledged_by: str | None = None

    resolved_at: str | None = None
    resolved_by: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "robot_id": self.robot_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "code": self.code,
            "title": self.title,
            "message": self.message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "first_seen_seconds": (
                self.first_seen_seconds
            ),
            "last_seen_seconds": (
                self.last_seen_seconds
            ),
            "occurrence_count": (
                self.occurrence_count
            ),
            "acknowledged_at": self.acknowledged_at,
            "acknowledged_by": self.acknowledged_by,
            "resolved_at": self.resolved_at,
            "resolved_by": self.resolved_by,
            "metadata": self.metadata,
        }


# ============================================================
# ROBOT TELEMETRY
# ============================================================

@dataclass(frozen=True)
class RobotAlertTelemetry:
    robot_id: str
    timestamp_seconds: float

    battery_soc_percent: float = 100.0
    motor_temperature_c: float = 25.0
    vibration_mm_s: float = 0.0

    health_score: float = 100.0
    rul_hours: float = 1000.0

    sensor_available: bool = True
    network_connected: bool = True

    collision_count: int = 0

    navigation_blocked: bool = False
    emergency_detected: bool = False

    def __post_init__(self) -> None:
        if not self.robot_id.strip():
            raise AlertValidationError(
                "robot_id is required"
            )

        for name in (
            "timestamp_seconds",
            "battery_soc_percent",
            "motor_temperature_c",
            "vibration_mm_s",
            "health_score",
            "rul_hours",
        ):
            finite(
                getattr(self, name),
                name,
            )

        if self.timestamp_seconds < 0:
            raise AlertValidationError(
                "Timestamp cannot be negative"
            )

        if not 0 <= self.battery_soc_percent <= 100:
            raise AlertValidationError(
                "Invalid battery SOC"
            )

        if not 0 <= self.health_score <= 100:
            raise AlertValidationError(
                "Invalid health score"
            )

        if self.vibration_mm_s < 0:
            raise AlertValidationError(
                "Vibration cannot be negative"
            )

        if self.rul_hours < 0:
            raise AlertValidationError(
                "RUL cannot be negative"
            )

        if self.collision_count < 0:
            raise AlertValidationError(
                "Collision count cannot be negative"
            )


# ============================================================
# ALERT RULE ENGINE
# ============================================================

@dataclass(frozen=True)
class AlertRuleResult:
    category: AlertCategory
    severity: AlertSeverity
    code: str
    title: str
    message: str
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class RoboticsAlertRuleEngine:

    def __init__(
        self,
        config: AlertServiceConfig,
    ) -> None:
        self.config = config

    def evaluate(
        self,
        telemetry: RobotAlertTelemetry,
    ) -> list[AlertRuleResult]:
        results = []

        soc = telemetry.battery_soc_percent

        if soc <= self.config.battery_critical_soc_percent:
            results.append(
                AlertRuleResult(
                    category=AlertCategory.BATTERY,
                    severity=AlertSeverity.CRITICAL,
                    code="BATTERY_CRITICAL",
                    title="Critical Battery Level",
                    message=(
                        f"Battery SOC is {soc:.1f}%"
                    ),
                    metadata={
                        "battery_soc_percent": soc
                    },
                )
            )

        elif soc <= self.config.battery_warning_soc_percent:
            results.append(
                AlertRuleResult(
                    category=AlertCategory.BATTERY,
                    severity=AlertSeverity.WARNING,
                    code="BATTERY_LOW",
                    title="Low Battery",
                    message=(
                        f"Battery SOC is {soc:.1f}%"
                    ),
                    metadata={
                        "battery_soc_percent": soc
                    },
                )
            )

        temperature = telemetry.motor_temperature_c

        if (
            temperature
            >= self.config.motor_critical_temperature_c
        ):
            results.append(
                AlertRuleResult(
                    category=AlertCategory.MOTOR,
                    severity=AlertSeverity.CRITICAL,
                    code="MOTOR_OVERHEAT_CRITICAL",
                    title="Critical Motor Overheating",
                    message=(
                        f"Motor temperature is "
                        f"{temperature:.1f} °C"
                    ),
                    metadata={
                        "temperature_c": temperature
                    },
                )
            )

        elif (
            temperature
            >= self.config.motor_warning_temperature_c
        ):
            results.append(
                AlertRuleResult(
                    category=AlertCategory.MOTOR,
                    severity=AlertSeverity.WARNING,
                    code="MOTOR_TEMPERATURE_HIGH",
                    title="Motor Temperature Warning",
                    message=(
                        f"Motor temperature is "
                        f"{temperature:.1f} °C"
                    ),
                )
            )

        vibration = telemetry.vibration_mm_s

        if (
            vibration
            >= self.config.vibration_critical_mm_s
        ):
            results.append(
                AlertRuleResult(
                    category=AlertCategory.VIBRATION,
                    severity=AlertSeverity.CRITICAL,
                    code="VIBRATION_CRITICAL",
                    title="Critical Vibration",
                    message=(
                        f"Vibration is "
                        f"{vibration:.2f} mm/s"
                    ),
                )
            )

        elif (
            vibration
            >= self.config.vibration_warning_mm_s
        ):
            results.append(
                AlertRuleResult(
                    category=AlertCategory.VIBRATION,
                    severity=AlertSeverity.WARNING,
                    code="VIBRATION_HIGH",
                    title="Excessive Vibration",
                    message=(
                        f"Vibration is "
                        f"{vibration:.2f} mm/s"
                    ),
                )
            )

        health = telemetry.health_score

        if health <= self.config.health_critical_score:
            results.append(
                AlertRuleResult(
                    category=AlertCategory.HEALTH,
                    severity=AlertSeverity.CRITICAL,
                    code="ROBOT_HEALTH_CRITICAL",
                    title="Critical Robot Health",
                    message=(
                        f"Robot health score is "
                        f"{health:.1f}"
                    ),
                )
            )

        elif health <= self.config.health_warning_score:
            results.append(
                AlertRuleResult(
                    category=AlertCategory.HEALTH,
                    severity=AlertSeverity.WARNING,
                    code="ROBOT_HEALTH_DEGRADED",
                    title="Robot Health Degraded",
                    message=(
                        f"Robot health score is "
                        f"{health:.1f}"
                    ),
                )
            )

        rul = telemetry.rul_hours

        if (
            rul
            <= self.config.maintenance_critical_rul_hours
        ):
            results.append(
                AlertRuleResult(
                    category=AlertCategory.MAINTENANCE,
                    severity=AlertSeverity.CRITICAL,
                    code="MAINTENANCE_URGENT",
                    title="Urgent Maintenance",
                    message=(
                        f"Estimated RUL is "
                        f"{rul:.1f} hours"
                    ),
                )
            )

        elif (
            rul
            <= self.config.maintenance_warning_rul_hours
        ):
            results.append(
                AlertRuleResult(
                    category=AlertCategory.MAINTENANCE,
                    severity=AlertSeverity.WARNING,
                    code="MAINTENANCE_DUE",
                    title="Maintenance Recommended",
                    message=(
                        f"Estimated RUL is "
                        f"{rul:.1f} hours"
                    ),
                )
            )

        if not telemetry.sensor_available:
            results.append(
                AlertRuleResult(
                    category=AlertCategory.SENSOR,
                    severity=AlertSeverity.HIGH,
                    code="SENSOR_FAILURE",
                    title="Sensor Unavailable",
                    message=(
                        "One or more robot sensors "
                        "are unavailable"
                    ),
                )
            )

        if not telemetry.network_connected:
            results.append(
                AlertRuleResult(
                    category=AlertCategory.NETWORK,
                    severity=AlertSeverity.HIGH,
                    code="NETWORK_DISCONNECTED",
                    title="Robot Network Disconnected",
                    message=(
                        "Robot communication is unavailable"
                    ),
                )
            )

        if telemetry.navigation_blocked:
            results.append(
                AlertRuleResult(
                    category=AlertCategory.NAVIGATION,
                    severity=AlertSeverity.HIGH,
                    code="NAVIGATION_BLOCKED",
                    title="Navigation Blocked",
                    message=(
                        "Robot navigation is blocked"
                    ),
                )
            )

        if telemetry.collision_count > 0:
            results.append(
                AlertRuleResult(
                    category=AlertCategory.COLLISION,
                    severity=AlertSeverity.CRITICAL,
                    code="COLLISION_DETECTED",
                    title="Robot Collision Detected",
                    message=(
                        f"Collision count: "
                        f"{telemetry.collision_count}"
                    ),
                )
            )

        if telemetry.emergency_detected:
            results.append(
                AlertRuleResult(
                    category=AlertCategory.EMERGENCY,
                    severity=AlertSeverity.EMERGENCY,
                    code="ROBOT_EMERGENCY",
                    title="Robot Emergency",
                    message=(
                        "Emergency condition detected"
                    ),
                )
            )

        return results


# ============================================================
# ALERT SERVICE
# ============================================================

class RoboticsAlertService:

    def __init__(
        self,
        config: AlertServiceConfig | None = None,
    ) -> None:
        self.config = (
            config or AlertServiceConfig()
        )

        self.rule_engine = (
            RoboticsAlertRuleEngine(
                self.config
            )
        )

        self.alerts: dict[
            str,
            RobotAlert
        ] = {}

        self.active_index: dict[
            tuple[str, str],
            str
        ] = {}

        self.history: deque[
            dict[str, Any]
        ] = deque(
            maxlen=self.config.maximum_history
        )

        self.callbacks: list[
            Callable[[dict[str, Any]], Any]
        ] = []

        self.metrics = Counter()

    def register_callback(
        self,
        callback: Callable[
            [dict[str, Any]],
            Any
        ],
    ) -> None:
        self.callbacks.append(
            callback
        )

    async def _emit(
        self,
        event_type: str,
        alert: RobotAlert,
    ) -> None:
        event = {
            "event_type": event_type,
            "alert": alert.to_dict(),
            "timestamp": utc_now(),
        }

        self.history.append(event)

        for callback in self.callbacks:
            try:
                result = callback(event)

                if inspect.isawaitable(result):
                    await result

            except Exception:
                self.metrics[
                    "callback_errors"
                ] += 1

    async def create_alert(
        self,
        *,
        robot_id: str,
        category: AlertCategory,
        severity: AlertSeverity,
        code: str,
        title: str,
        message: str,
        simulation_time_seconds: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> RobotAlert:
        if not robot_id.strip():
            raise AlertValidationError(
                "robot_id is required"
            )

        timestamp = finite(
            simulation_time_seconds,
            "simulation_time_seconds",
        )

        if timestamp < 0:
            raise AlertValidationError(
                "Simulation time cannot be negative"
            )

        key = (
            robot_id,
            code,
        )

        existing_id = self.active_index.get(
            key
        )

        if existing_id is not None:
            alert = self.alerts[existing_id]

            if timestamp < alert.last_seen_seconds:
                raise AlertValidationError(
                    "Alert timestamp moved backwards"
                )

            alert.occurrence_count += 1
            alert.last_seen_seconds = timestamp
            alert.updated_at = utc_now()

            previous_severity = alert.severity

            if (
                SEVERITY_RANK[severity]
                > SEVERITY_RANK[alert.severity]
            ):
                alert.severity = severity

            alert.message = message

            alert.metadata.update(
                json_safe(
                    dict(metadata or {})
                )
            )

            self.metrics[
                "deduplicated"
            ] += 1

            if (
                alert.severity != previous_severity
            ):
                await self._emit(
                    "alert_severity_changed",
                    alert,
                )

            return alert

        active_count = len(
            self.active_index
        )

        if (
            active_count
            >= self.config.maximum_active_alerts
        ):
            raise AlertServiceError(
                "Maximum active alerts reached"
            )

        now = utc_now()

        alert = RobotAlert(
            alert_id=str(uuid.uuid4()),
            robot_id=robot_id,
            category=AlertCategory(category),
            severity=AlertSeverity(severity),
            status=AlertStatus.ACTIVE,
            code=code,
            title=title,
            message=message,
            created_at=now,
            updated_at=now,
            first_seen_seconds=timestamp,
            last_seen_seconds=timestamp,
            metadata=json_safe(
                dict(metadata or {})
            ),
        )

        self.alerts[
            alert.alert_id
        ] = alert

        self.active_index[
            key
        ] = alert.alert_id

        self.metrics[
            "created"
        ] += 1

        await self._emit(
            "alert_created",
            alert,
        )

        return alert

    async def process_telemetry(
        self,
        telemetry: RobotAlertTelemetry,
    ) -> list[RobotAlert]:
        rules = self.rule_engine.evaluate(
            telemetry
        )

        generated = []

        for rule in rules:
            alert = await self.create_alert(
                robot_id=telemetry.robot_id,
                category=rule.category,
                severity=rule.severity,
                code=rule.code,
                title=rule.title,
                message=rule.message,
                simulation_time_seconds=(
                    telemetry.timestamp_seconds
                ),
                metadata=rule.metadata,
            )

            generated.append(
                alert
            )

        return generated

    def get_alert(
        self,
        alert_id: str,
    ) -> RobotAlert:
        alert = self.alerts.get(
            alert_id
        )

        if alert is None:
            raise AlertNotFoundError(
                f"Alert not found: {alert_id}"
            )

        return alert

    async def acknowledge(
        self,
        alert_id: str,
        *,
        acknowledged_by: str,
    ) -> RobotAlert:
        alert = self.get_alert(
            alert_id
        )

        if alert.status != AlertStatus.ACTIVE:
            raise AlertStateError(
                "Only active alerts can be acknowledged"
            )

        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = utc_now()
        alert.updated_at = utc_now()

        self.metrics[
            "acknowledged"
        ] += 1

        await self._emit(
            "alert_acknowledged",
            alert,
        )

        return alert

    async def resolve(
        self,
        alert_id: str,
        *,
        resolved_by: str,
    ) -> RobotAlert:
        alert = self.get_alert(
            alert_id
        )

        if alert.status == AlertStatus.RESOLVED:
            raise AlertStateError(
                "Alert is already resolved"
            )

        alert.status = AlertStatus.RESOLVED
        alert.resolved_by = resolved_by
        alert.resolved_at = utc_now()
        alert.updated_at = utc_now()

        self.active_index.pop(
            (
                alert.robot_id,
                alert.code,
            ),
            None,
        )

        self.metrics[
            "resolved"
        ] += 1

        await self._emit(
            "alert_resolved",
            alert,
        )

        return alert

    async def escalate(
        self,
        *,
        current_simulation_time_seconds: float,
    ) -> list[RobotAlert]:
        current_time = finite(
            current_simulation_time_seconds,
            "current_simulation_time_seconds",
        )

        escalated = []

        for alert_id in list(
            self.active_index.values()
        ):
            alert = self.alerts[alert_id]

            age = (
                current_time
                - alert.first_seen_seconds
            )

            if (
                age
                < self.config.escalation_after_seconds
            ):
                continue

            if (
                alert.status
                != AlertStatus.ACTIVE
            ):
                continue

            if alert.severity == AlertSeverity.WARNING:
                alert.severity = AlertSeverity.HIGH

            elif alert.severity == AlertSeverity.HIGH:
                alert.severity = AlertSeverity.CRITICAL

            else:
                continue

            alert.updated_at = utc_now()

            self.metrics[
                "escalated"
            ] += 1

            escalated.append(
                alert
            )

            await self._emit(
                "alert_escalated",
                alert,
            )

        return escalated

    def list_alerts(
        self,
        *,
        robot_id: str | None = None,
        severity: AlertSeverity | None = None,
        status: AlertStatus | None = None,
    ) -> list[RobotAlert]:
        alerts = list(
            self.alerts.values()
        )

        if robot_id is not None:
            alerts = [
                alert
                for alert in alerts
                if alert.robot_id == robot_id
            ]

        if severity is not None:
            alerts = [
                alert
                for alert in alerts
                if alert.severity == severity
            ]

        if status is not None:
            alerts = [
                alert
                for alert in alerts
                if alert.status == status
            ]

        return sorted(
            alerts,
            key=lambda alert: (
                -SEVERITY_RANK[alert.severity],
                alert.created_at,
            ),
        )

    def statistics(
        self,
    ) -> dict[str, Any]:
        alerts = list(
            self.alerts.values()
        )

        return {
            "total_alerts": len(alerts),
            "active_alerts": len(
                self.active_index
            ),
            "severity_distribution": dict(
                Counter(
                    alert.severity.value
                    for alert in alerts
                )
            ),
            "status_distribution": dict(
                Counter(
                    alert.status.value
                    for alert in alerts
                )
            ),
            "category_distribution": dict(
                Counter(
                    alert.category.value
                    for alert in alerts
                )
            ),
            "metrics": dict(
                self.metrics
            ),
            "hardware_control_enabled": False,
        }

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
                {
                    "alerts": [
                        alert.to_dict()
                        for alert in self.alerts.values()
                    ],
                    "statistics": self.statistics(),
                },
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        return output

    def export_events_jsonl(
        self,
        path: str | Path,
    ) -> Path:
        output = Path(path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output.open(
            "w",
            encoding="utf-8",
        ) as handle:
            for event in self.history:
                handle.write(
                    json.dumps(
                        event,
                        allow_nan=False,
                    )
                    + "\n"
                )

        return output


# ============================================================
# SIMULATION INTEGRATION
# ============================================================

class SimulationAlertAdapter:

    @staticmethod
    def from_physics(
        physics: Any,
        *,
        health_scores: Mapping[
            str,
            float
        ] | None = None,
        rul_hours: Mapping[
            str,
            float
        ] | None = None,
    ) -> list[RobotAlertTelemetry]:
        health_scores = (
            health_scores or {}
        )

        rul_hours = (
            rul_hours or {}
        )

        telemetry = []

        for robot_id, robot in physics.robots.items():
            telemetry.append(
                RobotAlertTelemetry(
                    robot_id=robot_id,
                    timestamp_seconds=(
                        physics.simulation_time_seconds
                    ),
                    battery_soc_percent=(
                        robot.battery_soc_percent()
                    ),
                    motor_temperature_c=(
                        robot.motor_temperature_c
                    ),
                    vibration_mm_s=0.0,
                    health_score=(
                        health_scores.get(
                            robot_id,
                            100.0,
                        )
                    ),
                    rul_hours=(
                        rul_hours.get(
                            robot_id,
                            1000.0,
                        )
                    ),
                    collision_count=(
                        robot.collision_count
                    ),
                )
            )

        return telemetry

    @staticmethod
    async def process_physics(
        service: RoboticsAlertService,
        physics: Any,
    ) -> list[RobotAlert]:
        alerts = []

        telemetry = (
            SimulationAlertAdapter.from_physics(
                physics
            )
        )

        for robot in telemetry:
            alerts.extend(
                await service.process_telemetry(
                    robot
                )
            )

        return alerts


# ============================================================
# DEMONSTRATION
# ============================================================

async def main() -> None:
    service = RoboticsAlertService()

    async def notification_callback(
        event: dict[str, Any],
    ) -> None:
        alert = event["alert"]

        print(
            f"[{event['event_type']}] "
            f"{alert['robot_id']} | "
            f"{alert['severity']} | "
            f"{alert['title']}"
        )

    service.register_callback(
        notification_callback
    )

    telemetry_samples = [
        RobotAlertTelemetry(
            robot_id="robot_001",
            timestamp_seconds=10.0,
            battery_soc_percent=15.0,
            motor_temperature_c=75.0,
            vibration_mm_s=3.0,
            health_score=80.0,
            rul_hours=200.0,
        ),
        RobotAlertTelemetry(
            robot_id="robot_002",
            timestamp_seconds=10.0,
            battery_soc_percent=8.0,
            motor_temperature_c=95.0,
            vibration_mm_s=14.0,
            health_score=30.0,
            rul_hours=12.0,
            sensor_available=False,
        ),
        RobotAlertTelemetry(
            robot_id="robot_003",
            timestamp_seconds=10.0,
            battery_soc_percent=80.0,
            motor_temperature_c=40.0,
            vibration_mm_s=2.0,
            collision_count=1,
            navigation_blocked=True,
            emergency_detected=True,
        ),
    ]

    for telemetry in telemetry_samples:
        await service.process_telemetry(
            telemetry
        )

    active = service.list_alerts(
        status=AlertStatus.ACTIVE
    )

    if active:
        await service.acknowledge(
            active[0].alert_id,
            acknowledged_by="simulation_operator",
        )

    await service.escalate(
        current_simulation_time_seconds=400.0
    )

    service.export_json(
        "data/alerts/robot_alerts.json"
    )

    service.export_events_jsonl(
        "data/alerts/alert_events.jsonl"
    )

    print(
        json.dumps(
            service.statistics(),
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())

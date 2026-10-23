"""Health Metrics module.

TODO: implement and test; no real robot control is enabled.
"""

"""
backend/app/monitoring/health_metrics.py

Physical AI & Autonomous Robotics Intelligence Platform

Robot and Fleet Health Metrics Monitoring

Features:
- Robot and component health scores
- Motor, actuator, joint, battery and sensor metrics
- Thermal and vibration metrics
- Failure risk and remaining useful life (RUL)
- Active faults, anomalies and maintenance status
- Fleet-wide health aggregation
- Historical health trends
- Stale telemetry detection
- Health alerts and recommendations
- Prometheus-compatible metric export
- Thread-safe in-memory monitoring
- JSON-ready health reports

Python 3.11+
Dependencies: Standard library only

NOTE:
Health scores and thresholds are monitoring heuristics.
They are not safety-certified control decisions.
"""

from __future__ import annotations

import json
import math
import statistics
import threading
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Sequence


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class ComponentType(str, Enum):
    MOTOR = "motor"
    ACTUATOR = "actuator"
    JOINT = "joint"
    BATTERY = "battery"
    SENSOR = "sensor"
    CONTROLLER = "controller"
    GEARBOX = "gearbox"
    BEARING = "bearing"
    CPU = "cpu"
    GPU = "gpu"
    POWER_SYSTEM = "power_system"
    COMMUNICATION = "communication"
    OTHER = "other"


@dataclass(frozen=True)
class HealthThresholds:
    healthy_score: float = 80.0
    critical_score: float = 40.0
    warning_rul_hours: float = 200.0
    critical_rul_hours: float = 50.0
    stale_after_seconds: float = 30.0
    warning_failure_probability: float = 0.40
    critical_failure_probability: float = 0.75

    def __post_init__(self) -> None:
        values = (
            self.healthy_score,
            self.critical_score,
            self.warning_rul_hours,
            self.critical_rul_hours,
            self.stale_after_seconds,
            self.warning_failure_probability,
            self.critical_failure_probability,
        )

        if not all(math.isfinite(v) for v in values):
            raise ValueError("Thresholds must be finite")

        if not (
            0 <= self.critical_score
            < self.healthy_score <= 100
        ):
            raise ValueError("Invalid health thresholds")

        if not (
            0 <= self.critical_rul_hours
            < self.warning_rul_hours
        ):
            raise ValueError("Invalid RUL thresholds")

        if self.stale_after_seconds <= 0:
            raise ValueError("Invalid stale threshold")

        if not (
            0 <= self.warning_failure_probability
            < self.critical_failure_probability <= 1
        ):
            raise ValueError(
                "Invalid failure probability thresholds"
            )


@dataclass(frozen=True)
class ComponentHealthSnapshot:
    robot_id: str
    component_id: str
    component_type: ComponentType
    health_score: float
    timestamp: datetime

    temperature_c: Optional[float] = None
    vibration_rms: Optional[float] = None
    current_a: Optional[float] = None
    voltage_v: Optional[float] = None
    power_w: Optional[float] = None
    remaining_useful_life_hours: Optional[float] = None
    failure_probability: Optional[float] = None

    active_faults: int = 0
    active_anomalies: int = 0
    maintenance_due: bool = False
    sensor_online: bool = True

    def __post_init__(self) -> None:
        if not self.robot_id or not self.component_id:
            raise ValueError(
                "Robot ID and component ID are required"
            )

        if self.timestamp.tzinfo is None:
            raise ValueError(
                "Timestamp must be timezone-aware"
            )

        if not math.isfinite(self.health_score):
            raise ValueError("Health score must be finite")

        if not 0 <= self.health_score <= 100:
            raise ValueError(
                "Health score must be between 0 and 100"
            )

        optional_values = (
            self.temperature_c,
            self.vibration_rms,
            self.current_a,
            self.voltage_v,
            self.power_w,
            self.remaining_useful_life_hours,
            self.failure_probability,
        )

        if any(
            value is not None and not math.isfinite(value)
            for value in optional_values
        ):
            raise ValueError(
                "Optional measurements must be finite"
            )

        if (
            self.remaining_useful_life_hours is not None
            and self.remaining_useful_life_hours < 0
        ):
            raise ValueError("RUL cannot be negative")

        if (
            self.vibration_rms is not None
            and self.vibration_rms < 0
        ):
            raise ValueError("Vibration RMS cannot be negative")

        if self.failure_probability is not None:
            if not 0 <= self.failure_probability <= 1:
                raise ValueError(
                    "Failure probability must be 0-1"
                )

        if self.active_faults < 0 or self.active_anomalies < 0:
            raise ValueError(
                "Fault and anomaly counts cannot be negative"
            )


@dataclass
class ComponentHealthReport:
    robot_id: str
    component_id: str
    component_type: str
    health_score: float
    status: HealthStatus
    remaining_useful_life_hours: Optional[float]
    failure_probability: Optional[float]
    temperature_c: Optional[float]
    vibration_rms: Optional[float]
    active_faults: int
    active_anomalies: int
    maintenance_due: bool
    data_age_seconds: float
    alerts: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class RobotHealthReport:
    robot_id: str
    overall_health_score: Optional[float]
    status: HealthStatus
    total_components: int
    healthy_components: int
    warning_components: int
    critical_components: int
    offline_components: int
    unknown_components: int
    active_faults: int
    active_anomalies: int
    maintenance_required: bool
    components: list[ComponentHealthReport]
    evaluated_at: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["components"] = [
            component.to_dict()
            for component in self.components
        ]
        return data


class HealthMetricsCollector:
    """
    Thread-safe robot health metrics collector.

    Keeps bounded per-component history.
    Uses latest snapshots for current-state reports.
    """

    def __init__(
        self,
        thresholds: Optional[HealthThresholds] = None,
        max_history_per_component: int = 1000,
    ) -> None:
        if max_history_per_component < 1:
            raise ValueError(
                "History size must be positive"
            )

        self.thresholds = thresholds or HealthThresholds()
        self.max_history = max_history_per_component

        self._history: dict[
            tuple[str, str],
            deque[ComponentHealthSnapshot],
        ] = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )

        self._lock = threading.RLock()

    def record(
        self,
        snapshot: ComponentHealthSnapshot,
    ) -> None:
        key = (
            snapshot.robot_id,
            snapshot.component_id,
        )

        with self._lock:
            history = self._history[key]

            if history and snapshot.timestamp <= history[-1].timestamp:
                raise ValueError(
                    "Snapshots must arrive in increasing "
                    "timestamp order per component"
                )

            history.append(snapshot)

    def get_latest(
        self,
        robot_id: str,
        component_id: str,
    ) -> Optional[ComponentHealthSnapshot]:
        with self._lock:
            history = self._history.get(
                (robot_id, component_id)
            )

            return history[-1] if history else None

    def get_history(
        self,
        robot_id: str,
        component_id: str,
    ) -> list[ComponentHealthSnapshot]:
        with self._lock:
            return list(
                self._history.get(
                    (robot_id, component_id), []
                )
            )

    def _evaluate_component(
        self,
        snapshot: ComponentHealthSnapshot,
        now: datetime,
    ) -> ComponentHealthReport:

        age = (
            now - snapshot.timestamp
        ).total_seconds()

        if age < 0:
            raise ValueError(
                "Snapshot timestamp is in the future"
            )

        alerts: list[str] = []
        recommendations: list[str] = []

        t = self.thresholds

        if (
            not snapshot.sensor_online
            or age > t.stale_after_seconds
        ):
            status = HealthStatus.OFFLINE
            alerts.append("telemetry_unavailable")
            recommendations.append(
                "Verify telemetry, sensor connectivity "
                "and component communication."
            )

        elif snapshot.health_score < t.critical_score:
            status = HealthStatus.CRITICAL

        elif snapshot.health_score < t.healthy_score:
            status = HealthStatus.WARNING

        else:
            status = HealthStatus.HEALTHY

        if snapshot.active_faults > 0:
            alerts.append("active_component_faults")
            recommendations.append(
                "Inspect active diagnostic fault codes."
            )

            if status not in (
                HealthStatus.OFFLINE,
                HealthStatus.CRITICAL,
            ):
                status = HealthStatus.WARNING

        if snapshot.active_anomalies > 0:
            alerts.append("component_anomalies")
            recommendations.append(
                "Investigate abnormal telemetry patterns."
            )

            if status == HealthStatus.HEALTHY:
                status = HealthStatus.WARNING

        rul = snapshot.remaining_useful_life_hours

        if rul is not None:
            if rul <= t.critical_rul_hours:
                alerts.append("critical_remaining_life")
                recommendations.append(
                    "Prioritize component maintenance "
                    "and technician review."
                )

                if status != HealthStatus.OFFLINE:
                    status = HealthStatus.CRITICAL

            elif rul <= t.warning_rul_hours:
                alerts.append("low_remaining_life")
                recommendations.append(
                    "Schedule preventive maintenance."
                )

                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING

        probability = snapshot.failure_probability

        if probability is not None:
            if probability >= t.critical_failure_probability:
                alerts.append("high_failure_probability")
                recommendations.append(
                    "Escalate for immediate diagnostic review."
                )

                if status != HealthStatus.OFFLINE:
                    status = HealthStatus.CRITICAL

            elif probability >= t.warning_failure_probability:
                alerts.append("elevated_failure_probability")

                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING

        if snapshot.maintenance_due:
            alerts.append("maintenance_due")
            recommendations.append(
                "Review and schedule the maintenance task."
            )

            if status == HealthStatus.HEALTHY:
                status = HealthStatus.WARNING

        if status == HealthStatus.CRITICAL:
            recommendations.append(
                "Notify maintenance personnel and "
                "review safe operating conditions."
            )

        if not recommendations:
            recommendations.append(
                "Continue routine health monitoring."
            )

        return ComponentHealthReport(
            robot_id=snapshot.robot_id,
            component_id=snapshot.component_id,
            component_type=snapshot.component_type.value,
            health_score=snapshot.health_score,
            status=status,
            remaining_useful_life_hours=rul,
            failure_probability=probability,
            temperature_c=snapshot.temperature_c,
            vibration_rms=snapshot.vibration_rms,
            active_faults=snapshot.active_faults,
            active_anomalies=snapshot.active_anomalies,
            maintenance_due=snapshot.maintenance_due,
            data_age_seconds=round(age, 3),
            alerts=alerts,
            recommendations=list(
                dict.fromkeys(recommendations)
            ),
        )

    def get_robot_health(
        self,
        robot_id: str,
        now: Optional[datetime] = None,
    ) -> RobotHealthReport:

        now = now or datetime.now(timezone.utc)

        if now.tzinfo is None:
            raise ValueError(
                "now must be timezone-aware"
            )

        with self._lock:
            snapshots = [
                history[-1]
                for (rid, _), history in self._history.items()
                if rid == robot_id and history
            ]

        reports = [
            self._evaluate_component(snapshot, now)
            for snapshot in snapshots
        ]

        counts = {
            status: sum(
                report.status == status
                for report in reports
            )
            for status in HealthStatus
        }

        valid_scores = [
            report.health_score
            for report in reports
            if report.status != HealthStatus.OFFLINE
        ]

        overall_score = (
            round(statistics.fmean(valid_scores), 2)
            if valid_scores else None
        )

        if not reports:
            status = HealthStatus.UNKNOWN
        elif counts[HealthStatus.CRITICAL]:
            status = HealthStatus.CRITICAL
        elif counts[HealthStatus.OFFLINE]:
            status = HealthStatus.OFFLINE
        elif counts[HealthStatus.WARNING]:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY

        return RobotHealthReport(
            robot_id=robot_id,
            overall_health_score=overall_score,
            status=status,
            total_components=len(reports),
            healthy_components=counts[HealthStatus.HEALTHY],
            warning_components=counts[HealthStatus.WARNING],
            critical_components=counts[HealthStatus.CRITICAL],
            offline_components=counts[HealthStatus.OFFLINE],
            unknown_components=counts[HealthStatus.UNKNOWN],
            active_faults=sum(
                r.active_faults for r in reports
            ),
            active_anomalies=sum(
                r.active_anomalies for r in reports
            ),
            maintenance_required=any(
                r.maintenance_due
                or r.status == HealthStatus.CRITICAL
                for r in reports
            ),
            components=reports,
            evaluated_at=now.isoformat(),
        )

    def get_fleet_health(
        self,
        now: Optional[datetime] = None,
    ) -> dict[str, Any]:

        now = now or datetime.now(timezone.utc)

        with self._lock:
            robot_ids = sorted({
                robot_id
                for robot_id, _ in self._history
            })

        reports = [
            self.get_robot_health(robot_id, now)
            for robot_id in robot_ids
        ]

        scores = [
            r.overall_health_score
            for r in reports
            if r.overall_health_score is not None
        ]

        return {
            "total_robots": len(reports),
            "healthy_robots": sum(
                r.status == HealthStatus.HEALTHY
                for r in reports
            ),
            "warning_robots": sum(
                r.status == HealthStatus.WARNING
                for r in reports
            ),
            "critical_robots": sum(
                r.status == HealthStatus.CRITICAL
                for r in reports
            ),
            "offline_robots": sum(
                r.status == HealthStatus.OFFLINE
                for r in reports
            ),
            "average_health_score": (
                round(statistics.fmean(scores), 2)
                if scores else None
            ),
            "total_active_faults": sum(
                r.active_faults for r in reports
            ),
            "total_active_anomalies": sum(
                r.active_anomalies for r in reports
            ),
            "robots_requiring_maintenance": sum(
                r.maintenance_required
                for r in reports
            ),
            "robots": [
                r.to_dict() for r in reports
            ],
            "evaluated_at": now.isoformat(),
        }

    def get_health_trend(
        self,
        robot_id: str,
        component_id: str,
    ) -> dict[str, Any]:

        history = self.get_history(
            robot_id, component_id
        )

        if not history:
            return {
                "robot_id": robot_id,
                "component_id": component_id,
                "trend": "unknown",
                "sample_count": 0,
                "health_change": None,
                "health_change_per_hour": None,
            }

        first = history[0]
        last = history[-1]

        change = (
            last.health_score - first.health_score
        )

        duration_hours = (
            last.timestamp - first.timestamp
        ).total_seconds() / 3600.0

        rate = (
            change / duration_hours
            if duration_hours > 0 else None
        )

        if rate is None:
            trend = "unknown"
        elif rate < -0.01:
            trend = "degrading"
        elif rate > 0.01:
            trend = "improving"
        else:
            trend = "stable"

        return {
            "robot_id": robot_id,
            "component_id": component_id,
            "trend": trend,
            "sample_count": len(history),
            "health_change": round(change, 3),
            "health_change_per_hour": (
                round(rate, 5)
                if rate is not None else None
            ),
        }

    def export_prometheus(
        self,
        now: Optional[datetime] = None,
    ) -> str:
        """
        Export current component metrics as
        Prometheus text exposition.

        Avoid using unbounded robot/component IDs
        as labels in very large deployments.
        """

        now = now or datetime.now(timezone.utc)

        with self._lock:
            snapshots = [
                history[-1]
                for history in self._history.values()
                if history
            ]

        lines = [
            "# HELP robot_component_health_score "
            "Component health score from 0 to 100",
            "# TYPE robot_component_health_score gauge",
            "# HELP robot_component_online "
            "Whether component telemetry is available",
            "# TYPE robot_component_online gauge",
            "# HELP robot_component_failure_probability "
            "Estimated component failure probability",
            "# TYPE robot_component_failure_probability gauge",
            "# HELP robot_component_rul_hours "
            "Estimated remaining useful life in hours",
            "# TYPE robot_component_rul_hours gauge",
        ]

        def escape_label(value: str) -> str:
            return (
                value.replace("\\", "\\\\")
                .replace("\n", "\\n")
                .replace('"', '\\"')
            )

        for snapshot in snapshots:
            report = self._evaluate_component(
                snapshot, now
            )

            labels = (
                '{robot_id="'
                + escape_label(snapshot.robot_id)
                + '",component_id="'
                + escape_label(snapshot.component_id)
                + '",component_type="'
                + escape_label(snapshot.component_type.value)
                + '"}'
            )

            lines.append(
                f"robot_component_health_score{labels} "
                f"{report.health_score}"
            )

            online = (
                0 if report.status == HealthStatus.OFFLINE
                else 1
            )

            lines.append(
                f"robot_component_online{labels} {online}"
            )

            if report.failure_probability is not None:
                lines.append(
                    "robot_component_failure_probability"
                    f"{labels} {report.failure_probability}"
                )

            if report.remaining_useful_life_hours is not None:
                lines.append(
                    "robot_component_rul_hours"
                    f"{labels} "
                    f"{report.remaining_useful_life_hours}"
                )

        return "\n".join(lines) + "\n"

    def clear(self) -> None:
        with self._lock:
            self._history.clear()


health_metrics = HealthMetricsCollector()


def record_component_health(
    robot_id: str,
    component_id: str,
    component_type: str,
    health_score: float,
    temperature_c: Optional[float] = None,
    vibration_rms: Optional[float] = None,
    rul_hours: Optional[float] = None,
    failure_probability: Optional[float] = None,
    active_faults: int = 0,
    active_anomalies: int = 0,
    maintenance_due: bool = False,
) -> None:
    """Convenience function for health workers."""

    snapshot = ComponentHealthSnapshot(
        robot_id=robot_id,
        component_id=component_id,
        component_type=ComponentType(component_type),
        health_score=health_score,
        timestamp=datetime.now(timezone.utc),
        temperature_c=temperature_c,
        vibration_rms=vibration_rms,
        remaining_useful_life_hours=rul_hours,
        failure_probability=failure_probability,
        active_faults=active_faults,
        active_anomalies=active_anomalies,
        maintenance_due=maintenance_due,
    )

    health_metrics.record(snapshot)


if __name__ == "__main__":
    collector = HealthMetricsCollector()

    now = datetime.now(timezone.utc)

    collector.record(
        ComponentHealthSnapshot(
            robot_id="ROBOT-001",
            component_id="MOTOR-001",
            component_type=ComponentType.MOTOR,
            health_score=86.5,
            timestamp=now,
            temperature_c=58.0,
            vibration_rms=2.1,
            remaining_useful_life_hours=450.0,
            failure_probability=0.12,
        )
    )

    collector.record(
        ComponentHealthSnapshot(
            robot_id="ROBOT-001",
            component_id="BATTERY-001",
            component_type=ComponentType.BATTERY,
            health_score=72.0,
            timestamp=now,
            temperature_c=39.0,
            remaining_useful_life_hours=180.0,
            failure_probability=0.35,
            maintenance_due=True,
        )
    )

    collector.record(
        ComponentHealthSnapshot(
            robot_id="ROBOT-002",
            component_id="JOINT-001",
            component_type=ComponentType.JOINT,
            health_score=34.0,
            timestamp=now,
            vibration_rms=7.2,
            remaining_useful_life_hours=25.0,
            failure_probability=0.82,
            active_faults=2,
        )
    )

    print(
        json.dumps(
            collector.get_fleet_health(),
            indent=2,
        )
    )

    print(collector.export_prometheus())
